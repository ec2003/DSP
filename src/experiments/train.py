from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from pytorch_metric_learning.losses import ArcFaceLoss
from pytorch_metric_learning.samplers import MPerClassSampler
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.audio import (
    CompositeMusanNoiseMixer,
    DspTransformChain,
    HighPassFilter,
    LowPassFilter,
    WienerDenoiser,
)
from src.config.settings import ExperimentConfig
from src.data import ManifestRecord, VCTKWaveformDataset, build_manifests, load_manifest
from src.data.vctk import WaveformTransform
from src.experiments.evaluate import (
    build_verification_pairs,
    clustering_metrics,
    score_pairs,
    verification_metrics,
)
from src.models import EcapaSpeakerEncoder


@dataclass(frozen=True)
class TrainResult:
    checkpoint_path: Path
    best_validation_eer: float
    epochs_completed: int


@dataclass(frozen=True)
class EvaluationResult:
    report_path: Path
    verification: dict[str, float]
    clustering: dict[str, float]


def prepare_manifests(config: ExperimentConfig) -> dict[str, Path]:
    return build_manifests(
        config.vctk_root,
        config.manifest_root,
        seed=config.seed,
        segment_seconds=config.segment_seconds,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
    )


def train_condition(
    config: ExperimentConfig, *, device: torch.device | None = None
) -> TrainResult:
    _seed_everything(config.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_records = load_manifest(config.manifest_root / "train.jsonl")
    validation_records = load_manifest(config.manifest_root / "validation.jsonl")
    train_dataset = _dataset_for_records(config, train_records)
    validation_dataset = _dataset_for_records(config, validation_records)
    labels_by_speaker = {
        speaker: index
        for index, speaker in enumerate(
            sorted({record.speaker_id for record in train_records})
        )
    }
    label_values = [labels_by_speaker[record.speaker_id] for record in train_records]
    if config.batch_size % config.samples_per_speaker != 0:
        raise ValueError("batch_size must be divisible by samples_per_speaker")

    sampler = MPerClassSampler(
        label_values,
        m=config.samples_per_speaker,
        batch_size=config.batch_size,
        length_before_new_iter=len(train_dataset),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, sampler=sampler
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config.batch_size, shuffle=False
    )

    model = EcapaSpeakerEncoder.from_pretrained(
        source=config.ecapa_source,
        cache_dir=config.ecapa_cache,
        device=device,
    )
    model.set_embedding_trainable(config.freeze_encoder_epochs == 0)
    loss_function = ArcFaceLoss(
        num_classes=len(labels_by_speaker),
        embedding_size=config.embedding_dimension,
    ).to(device)
    optimizer = AdamW(
        list(model.parameters()) + list(loss_function.parameters()),
        lr=config.learning_rate,
    )

    checkpoint_path = config.run_root / "best.pt"
    best_validation_eer = float("inf")
    for epoch in range(config.epochs):
        if epoch == config.freeze_encoder_epochs:
            model.set_embedding_trainable(True)
        model.train()
        for batch in train_loader:
            waveforms = _waveforms(batch, device)
            labels = torch.tensor(
                [labels_by_speaker[speaker] for speaker in batch["speaker_id"]],
                device=device,
            )
            optimizer.zero_grad()
            loss = loss_function(model(waveforms), labels)
            loss.backward()
            optimizer.step()

        validation_embeddings = _extract_embeddings(model, validation_loader, device)
        scores, targets = score_pairs(
            build_verification_pairs(validation_records, seed=config.seed),
            validation_embeddings,
        )
        validation_result = verification_metrics(scores, targets)
        if validation_result["eer"] < best_validation_eer:
            best_validation_eer = validation_result["eer"]
            _save_checkpoint(
                checkpoint_path,
                model,
                loss_function,
                optimizer,
                config,
                labels_by_speaker,
                epoch,
                validation_result,
            )

    return TrainResult(checkpoint_path, best_validation_eer, config.epochs)


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: Path,
    *,
    clean_reference: bool = False,
    min_cluster_size: int = 2,
    min_samples: int | None = None,
    device: torch.device | None = None,
) -> EvaluationResult:
    """Evaluate a checkpoint on shared noisy test data or the clean reference protocol."""
    _seed_everything(config.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_records = load_manifest(config.manifest_root / "test.jsonl")
    test_dataset = VCTKWaveformDataset(
        test_records,
        sample_rate=config.sample_rate,
        segment_seconds=config.segment_seconds,
        waveform_transform=build_waveform_transform(
            config, force_noisy=not clean_reference
        ),
    )
    model = EcapaSpeakerEncoder.from_pretrained(
        source=config.ecapa_source,
        cache_dir=config.ecapa_cache,
        device=device,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    embeddings = _extract_embeddings(
        model, DataLoader(test_dataset, batch_size=config.batch_size), device
    )
    scores, targets = score_pairs(
        build_verification_pairs(test_records, seed=config.seed), embeddings
    )
    verification = verification_metrics(scores, targets)
    clustering = clustering_metrics(
        embeddings,
        test_records,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    protocol = "clean_reference" if clean_reference else "shared_noisy_test"
    report_path = config.run_root / f"evaluation-{protocol}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "condition": config.condition,
                "protocol": protocol,
                "checkpoint": str(checkpoint_path),
                "verification": verification,
                "clustering": clustering,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return EvaluationResult(report_path, verification, clustering)


def _dataset_for_records(
    config: ExperimentConfig, records: list[ManifestRecord]
) -> VCTKWaveformDataset:
    return VCTKWaveformDataset(
        records,
        sample_rate=config.sample_rate,
        segment_seconds=config.segment_seconds,
        waveform_transform=build_waveform_transform(config),
    )


def build_waveform_transform(
    config: ExperimentConfig, *, force_noisy: bool = False
) -> WaveformTransform | None:
    if not (config.needs_noise or force_noisy):
        return None
    noise_root = (
        config.musan_root / "noise"
        if (config.musan_root / "noise").is_dir()
        else config.musan_root
    )
    mixer = CompositeMusanNoiseMixer.from_root(
        noise_root,
        seed=config.seed,
        snr_db=config.snr_db,
        low_noise_band_hz=config.low_noise_band_hz,
        high_noise_band_hz=config.high_noise_band_hz,
        filter_order=config.filter_order,
    )
    dsp_chain = (
        DspTransformChain(
            HighPassFilter(config.high_pass_hz, config.filter_order),
            LowPassFilter(config.low_pass_hz, config.filter_order),
            WienerDenoiser(config.wiener_window_size),
        )
        if config.needs_wiener
        else None
    )

    def transform(waveform: Tensor, sample_rate: int, record: ManifestRecord) -> Tensor:
        noisy_waveform = mixer(waveform, sample_rate, record)
        return (
            dsp_chain(noisy_waveform, sample_rate, record)
            if dsp_chain is not None
            else noisy_waveform
        )

    return transform


def _extract_embeddings(
    model: EcapaSpeakerEncoder,
    data_loader: DataLoader[dict[str, str | Tensor]],
    device: torch.device,
) -> dict[str, Tensor]:
    model.eval()
    embeddings: dict[str, Tensor] = {}
    with torch.no_grad():
        for batch in data_loader:
            batch_embeddings = model(_waveforms(batch, device)).cpu()
            embeddings.update(zip(batch["sample_id"], batch_embeddings, strict=True))
    return embeddings


def _waveforms(batch: dict[str, Any], device: torch.device) -> Tensor:
    waveforms = batch["waveform"]
    if not isinstance(waveforms, Tensor):
        raise TypeError("DataLoader batch does not contain waveform tensors")
    return waveforms.to(device)


def _save_checkpoint(
    checkpoint_path: Path,
    model: EcapaSpeakerEncoder,
    loss_function: ArcFaceLoss,
    optimizer: AdamW,
    config: ExperimentConfig,
    labels_by_speaker: dict[str, int],
    epoch: int,
    validation_result: dict[str, float],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "loss_state": loss_function.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": _json_safe_config(config),
            "speaker_labels": labels_by_speaker,
            "epoch": epoch,
            "validation": validation_result,
        },
        checkpoint_path,
    )
    checkpoint_path.with_suffix(".json").write_text(
        json.dumps(
            {"config": _json_safe_config(config), "validation": validation_result},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _json_safe_config(config: ExperimentConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
