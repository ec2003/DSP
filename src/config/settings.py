from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ConditionName = Literal["clean_baseline", "noisy", "noisy_wiener"]


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExperimentConfig:
    """Shared, reproducible settings for one speaker-embedding experiment."""

    condition: ConditionName
    seed: int = 42
    source_sample_rate: int = 48_000
    sample_rate: int = 16_000
    segment_seconds: float = 3.0
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-4
    freeze_encoder_epochs: int = 1
    samples_per_speaker: int = 2
    embedding_dimension: int = 192
    snr_db: tuple[int, ...] = (5, 10, 15, 20)
    noise_components: tuple[str, ...] = (
        "environmental",
        "low_band",
        "high_band",
    )
    low_noise_band_hz: tuple[float, float] = (20.0, 300.0)
    high_noise_band_hz: tuple[float, float] = (3_000.0, 7_500.0)
    high_pass_hz: float = 80.0
    low_pass_hz: float = 7_500.0
    filter_order: int = 4
    wiener_window_size: int = 29
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    vctk_root: Path = PROJECT_ROOT / "dataset" / "VCTK-Corpus" / "wav48"
    musan_root: Path = PROJECT_ROOT / "dataset" / "musan"
    output_root: Path = PROJECT_ROOT / "outputs"
    ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    ecapa_cache: Path = PROJECT_ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"

    def __post_init__(self) -> None:
        nyquist_hz = self.nyquist_hz
        if self.source_sample_rate < self.sample_rate:
            raise ValueError("source_sample_rate must be at least sample_rate")
        if not self.snr_db or any(snr <= 0 for snr in self.snr_db):
            raise ValueError("snr_db must contain positive target SNR values")
        if len(self.noise_components) != 3 or set(self.noise_components) != {
            "environmental",
            "low_band",
            "high_band",
        }:
            raise ValueError(
                "noise_components must contain environmental, low_band, and high_band"
            )
        for name, band in (
            ("low_noise_band_hz", self.low_noise_band_hz),
            ("high_noise_band_hz", self.high_noise_band_hz),
        ):
            if not 0 < band[0] < band[1] < nyquist_hz:
                raise ValueError(
                    f"{name} must be ordered and lie below the model Nyquist frequency"
                )
        if not 0 < self.high_pass_hz < self.low_pass_hz < nyquist_hz:
            raise ValueError(
                "DSP passband must be ordered and lie below the model Nyquist frequency"
            )
        if self.filter_order < 1:
            raise ValueError("filter_order must be positive")

    @property
    def nyquist_hz(self) -> float:
        return self.sample_rate / 2

    @property
    def source_nyquist_hz(self) -> float:
        return self.source_sample_rate / 2

    @property
    def manifest_root(self) -> Path:
        return self.output_root / "manifests"

    @property
    def run_root(self) -> Path:
        return self.output_root / self.condition

    @property
    def needs_noise(self) -> bool:
        return self.condition != "clean_baseline"

    @property
    def needs_wiener(self) -> bool:
        return self.condition == "noisy_wiener"


def get_config(condition: ConditionName) -> ExperimentConfig:
    return ExperimentConfig(condition=condition)
