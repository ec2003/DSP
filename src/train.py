"""One robust CNN per seed, trained with deterministic on-the-fly MUSAN augmentation."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import ExperimentConfig
from src.corpus import ClipRecord, load_clips, load_manifest
from src.eval import enrollment_split, extract_embeddings, nearest_centroid_predict, speaker_labels
from src.models import ArcFaceHead, SpeakerCNN
from src.noise import load_noise_metadata, load_noise_pool_for_split, make_mixture, training_augmentation_choice


@dataclass(frozen=True)
class TrainResult:
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
    return torch.device(requested if requested != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


class AugmentedClips(Dataset):
    """Noise/SNR is keyed by (seed, epoch, sample_id), not worker order."""

    def __init__(self, clips: np.ndarray, records: list[ClipRecord], labels: np.ndarray, config: ExperimentConfig, seed: int, split: str):
        self.clips, self.records, self.labels, self.config, self.seed, self.split = clips, records, labels, config, seed, split
        self.epoch = 0
        self.pools = {f: load_noise_pool_for_split(config.cache_root, split, f) for f in ("noise", "music", "speech")}
        self.metadata = {f: load_noise_metadata(config.cache_root, split, f) for f in self.pools}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def augmentation_metadata(self, index: int) -> dict[str, object]:
        record = self.records[index]
        noisy, family, snr = training_augmentation_choice(record.sample_id, seed=self.seed, epoch=self.epoch, p_noise=self.config.p_noise, families=self.config.train_noise_families, snr_range_db=self.config.train_snr_range_db)
        data: dict[str, object] = {"epoch": self.epoch, "sample_id": record.sample_id, "seed": self.seed, "noisy": noisy, "family": family, "target_snr_db": snr}
        if noisy:
            data.update(make_mixture(self.clips[index], record.sample_id, seed=self.seed, family=family or "noise", snr_db=float(snr), pools=self.pools, pool_metadata=self.metadata, babble_sources_range=self.config.babble_sources_range, namespace=f"train-epoch-{self.epoch}").metadata)
        return data

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        noisy, family, snr = training_augmentation_choice(record.sample_id, seed=self.seed, epoch=self.epoch, p_noise=self.config.p_noise, families=self.config.train_noise_families, snr_range_db=self.config.train_snr_range_db)
        waveform = self.clips[index]
        if noisy:
            waveform = make_mixture(waveform, record.sample_id, seed=self.seed, family=family or "noise", snr_db=float(snr), pools=self.pools, pool_metadata=self.metadata, babble_sources_range=self.config.babble_sources_range, namespace=f"train-epoch-{self.epoch}").mixture
        return torch.from_numpy(np.asarray(waveform, dtype=np.float32)), torch.tensor(self.labels[index], dtype=torch.long)


def _validation_accuracy(model: nn.Module, dataset: AugmentedClips, records: list[ClipRecord], config: ExperimentConfig, device: torch.device) -> float:
    dataset.set_epoch(-1)  # fixed validation augmentation, distinct from train epochs
    waveforms = np.stack([dataset[i][0].numpy() for i in range(len(dataset))])
    embeddings = extract_embeddings(model, waveforms, device)
    labels, _ = speaker_labels(records)
    enrol, query = enrollment_split(records, config.enrollment_clips)
    truth, prediction, _ = nearest_centroid_predict(embeddings, labels, enrol, query)
    return float(np.mean(truth == prediction))


def train_robust_cnn(config: ExperimentConfig, seed: int, *, device: torch.device | None = None, workers: int = 0) -> TrainResult:
    device = device or resolve_device()
    seed_everything(seed)
    train_records, validation_records = load_manifest(config.cache_root, "train"), load_manifest(config.cache_root, "validation")
    train_clips, validation_clips = load_clips(config.cache_root, "train"), load_clips(config.cache_root, "validation")
    train_labels, names = speaker_labels(train_records)
    validation_labels, _ = speaker_labels(validation_records)
    train_data = AugmentedClips(train_clips, train_records, train_labels, config, seed, "train")
    validation_data = AugmentedClips(validation_clips, validation_records, validation_labels, config, seed, "validation")
    loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=workers, generator=torch.Generator().manual_seed(seed))
    model = SpeakerCNN(sample_rate=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length, n_mels=config.n_mels, channels=config.cnn_channels, embedding_dim=config.embedding_dim).to(device)
    head = ArcFaceHead(config.embedding_dim, len(names), config.arcface_margin, config.arcface_scale).to(device)
    parameters = list(model.parameters()) + list(head.parameters())
    optimiser = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss()
    root = config.seed_dir(seed)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "encoder.pt"
    history, best_accuracy, best_epoch = [], -1.0, -1
    augmentation_path = root / "training-augmentation.jsonl"
    with augmentation_path.open("w", encoding="utf-8") as audit:
        for epoch in range(config.epochs):
            train_data.set_epoch(epoch)
            model.train(); head.train(); total_loss = 0.0
            for waveform, label in loader:
                waveform, label = waveform.to(device), label.to(device)
                optimiser.zero_grad(set_to_none=True)
                loss = criterion(head(model(waveform), label), label)
                loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 5.0); optimiser.step()
                total_loss += float(loss.item())
            for index in range(len(train_data)):
                audit.write(json.dumps(train_data.augmentation_metadata(index)) + "\n")
            accuracy = _validation_accuracy(model, validation_data, validation_records, config, device)
            scheduler.step()
            history.append({"epoch": epoch, "train_loss": total_loss / max(1, len(loader)), "validation_accuracy": accuracy, "learning_rate": scheduler.get_last_lr()[0]})
            if accuracy > best_accuracy:
                best_accuracy, best_epoch = accuracy, epoch
                torch.save({"model_state": model.state_dict(), "recipe": "robust_cnn", "seed": seed, "epoch": epoch, "validation_accuracy": accuracy, "config_hash": config.config_hash}, checkpoint)
    result = TrainResult(seed, str(checkpoint), best_accuracy, best_epoch, config.epochs, len(names), sum(p.numel() for p in model.parameters()))
    (root / "training-history.json").write_text(json.dumps({"result": asdict(result), "augmentation": {"p_noise": config.p_noise, "clean_noisy_ratio": [1 - config.p_noise, config.p_noise], "families": config.train_noise_families, "snr_policy": "Uniform continuous keyed by (seed, epoch, sample_id)"}, "history": history}, indent=2) + "\n", encoding="utf-8")
    return result


def load_encoder(config: ExperimentConfig, checkpoint_path: Path, device: torch.device) -> SpeakerCNN:
    model = SpeakerCNN(sample_rate=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length, n_mels=config.n_mels, channels=config.cnn_channels, embedding_dim=config.embedding_dim).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("recipe") != "robust_cnn":
        raise ValueError(f"checkpoint {checkpoint_path} is not a robust_cnn checkpoint")
    model.load_state_dict(payload["model_state"]); model.eval()
    return model
