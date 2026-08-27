"""Training loop for the speaker embedding under one condition and seed.

The front-end is applied once up front and the processed waveforms are held in
memory, because the DSP chain is deterministic per clip and would otherwise be
recomputed every epoch.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import ExperimentConfig
from src.corpus import ClipRecord, load_clips, load_manifest
from src.eval import (
    enrollment_split,
    extract_embeddings,
    nearest_centroid_predict,
    speaker_labels,
)
from src.models import ArcFaceHead, SpeakerCNN
from src.noise import load_noise_pool
from src.pipeline import process_split


@dataclass(frozen=True)
class TrainResult:
    condition: str
    seed: int
    checkpoint_path: str
    best_validation_accuracy: float
    best_epoch: int
    epochs_run: int
    n_train_speakers: int
    n_parameters: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepared_split(
    config: ExperimentConfig,
    split: str,
    condition_name: str,
    seed: int,
    *,
    snr_db: float | None = None,
    workers: int = 8,
) -> tuple[np.ndarray, list[ClipRecord]]:
    """Cached clips for ``split`` with the condition's noise and DSP applied."""
    records = load_manifest(config.cache_root, split)
    clips = load_clips(config.cache_root, split)
    condition = config.condition(condition_name)
    noise_pool = load_noise_pool(config.cache_root) if condition.add_noise else None
    processed = process_split(
        clips,
        records,
        condition,
        config,
        seed=seed,
        noise_pool=noise_pool,
        snr_db=snr_db,
        workers=workers,
    )
    return processed, records


def _validation_accuracy(
    model: nn.Module,
    waveforms: np.ndarray,
    records: list[ClipRecord],
    config: ExperimentConfig,
    device: torch.device,
) -> float:
    """Nearest-centroid identification accuracy on unseen validation speakers."""
    embeddings = extract_embeddings(model, waveforms, device)
    labels, _ = speaker_labels(records)
    enrol, query = enrollment_split(records, config.enrollment_clips)
    truth, prediction, _ = nearest_centroid_predict(embeddings, labels, enrol, query)
    return float(np.mean(truth == prediction))


def train_condition(
    config: ExperimentConfig,
    condition_name: str,
    seed: int,
    *,
    device: torch.device | None = None,
    workers: int = 8,
    overrides: dict[str, float | int] | None = None,
) -> TrainResult:
    device = device or resolve_device()
    seed_everything(seed)

    learning_rate = float((overrides or {}).get("learning_rate", config.learning_rate))
    batch_size = int((overrides or {}).get("batch_size", config.batch_size))
    epochs = int((overrides or {}).get("epochs", config.epochs))

    train_waveforms, train_records = prepared_split(
        config, "train", condition_name, seed, workers=workers
    )
    validation_waveforms, validation_records = prepared_split(
        config, "validation", condition_name, seed, workers=workers
    )

    train_labels, train_names = speaker_labels(train_records)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_waveforms), torch.from_numpy(train_labels)
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model = SpeakerCNN(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        channels=config.cnn_channels,
        embedding_dim=config.embedding_dim,
    ).to(device)
    head = ArcFaceHead(
        config.embedding_dim,
        len(train_names),
        config.arcface_margin,
        config.arcface_scale,
    ).to(device)

    parameters = list(model.parameters()) + list(head.parameters())
    optimiser = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    run_root = config.run_root(seed, condition_name)
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_root / "encoder.pt"

    history: list[dict[str, float]] = []
    best_accuracy, best_epoch = -1.0, -1
    for epoch in range(epochs):
        model.train()
        head.train()
        epoch_loss = 0.0
        for waveform, label in loader:
            waveform, label = waveform.to(device), label.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(head(model(waveform), label), label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimiser.step()
            epoch_loss += float(loss.item())
        scheduler.step()

        accuracy = _validation_accuracy(
            model, validation_waveforms, validation_records, config, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(1, len(loader)),
                "validation_accuracy": accuracy,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "condition": condition_name,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_accuracy": accuracy,
                    "config_version": config.config_version,
                },
                checkpoint_path,
            )

    result = TrainResult(
        condition=condition_name,
        seed=seed,
        checkpoint_path=str(checkpoint_path),
        best_validation_accuracy=best_accuracy,
        best_epoch=best_epoch,
        epochs_run=epochs,
        n_train_speakers=len(train_names),
        n_parameters=sum(p.numel() for p in model.parameters()),
    )
    (run_root / "training-history.json").write_text(
        json.dumps(
            {
                "result": asdict(result),
                "hyperparameters": {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "optimizer": config.optimizer,
                    "weight_decay": config.weight_decay,
                    "arcface_margin": config.arcface_margin,
                    "arcface_scale": config.arcface_scale,
                },
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def load_encoder(
    config: ExperimentConfig, checkpoint_path: Path, device: torch.device
) -> SpeakerCNN:
    model = SpeakerCNN(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        channels=config.cnn_channels,
        embedding_dim=config.embedding_dim,
    ).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model
