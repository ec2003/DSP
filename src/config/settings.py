"""Versioned, serialisable configuration for the DSP501 study."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, Literal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ConditionName = Literal["clean_reference", "raw_noisy", "high_pass", "high_pass_low_pass", "full_dsp"]
_LEGACY_CONDITIONS = {"clean_baseline": "clean_reference", "noisy": "raw_noisy", "noisy_wiener": "full_dsp"}


@dataclass(frozen=True)
class ExperimentConfig:
    """All settings that affect the reproducible speaker-verification protocol."""

    study_id: str = "dsp501-speaker-verification-v1"
    config_version: int = 1
    condition: ConditionName | str = "raw_noisy"
    seed: int = 11
    source_sample_rate: int = 48_000
    sample_rate: int = 16_000
    segment_seconds: float = 3.0
    clips_per_speaker: int = 50
    positive_pairs_per_speaker: int = 50
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-4
    freeze_encoder_epochs: int = 1
    samples_per_speaker: int = 2
    embedding_dimension: int = 192
    experimental_seeds: tuple[int, ...] = (11, 22, 33)
    snr_db: tuple[int, ...] = (5, 10, 15, 20)
    noise_components: tuple[str, ...] = ("environmental", "low_band", "high_band")
    low_noise_band_hz: tuple[float, float] = (20.0, 300.0)
    high_noise_band_hz: tuple[float, float] = (3_000.0, 7_500.0)
    high_pass_hz: float = 80.0
    low_pass_hz: float = 7_500.0
    filter_order: int = 4
    wiener_window_size: int = 29
    stft_n_fft: int = 512
    stft_hop_length: int = 160
    mfcc_coefficients: int = 40
    mel_bins: int = 80
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    vctk_root: Path = PROJECT_ROOT / "dataset" / "VCTK-Corpus-0.92" / "wav48_silence_trimmed"
    musan_root: Path = PROJECT_ROOT / "dataset" / "musan"
    output_root: Path = PROJECT_ROOT / "outputs"
    ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    ecapa_revision: str = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
    ecapa_cache: Path = PROJECT_ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"

    def __post_init__(self) -> None:
        condition = _LEGACY_CONDITIONS.get(str(self.condition), self.condition)
        object.__setattr__(self, "condition", condition)
        if condition not in {"clean_reference", "raw_noisy", "high_pass", "high_pass_low_pass", "full_dsp"}:
            raise ValueError(f"Unsupported condition: {condition}")
        if self.source_sample_rate < self.sample_rate:
            raise ValueError("source_sample_rate must be at least sample_rate")
        if self.clips_per_speaker < 2 or self.positive_pairs_per_speaker < 1:
            raise ValueError("clips_per_speaker must be at least 2 and pair count positive")
        if not self.snr_db or any(snr <= 0 for snr in self.snr_db):
            raise ValueError("snr_db must contain positive target SNR values")
        if len(self.noise_components) != 3 or set(self.noise_components) != {"environmental", "low_band", "high_band"}:
            raise ValueError("noise_components must be environmental, low_band, high_band")
        for name, band in (("low_noise_band_hz", self.low_noise_band_hz), ("high_noise_band_hz", self.high_noise_band_hz)):
            if not 0 < band[0] < band[1] < self.nyquist_hz:
                raise ValueError(f"{name} must lie below model Nyquist")
        if not 0 < self.high_pass_hz < self.low_pass_hz < self.nyquist_hz:
            raise ValueError("DSP passband must be ordered and lie below model Nyquist")
        if self.filter_order < 1 or self.wiener_window_size < 3 or self.wiener_window_size % 2 == 0:
            raise ValueError("invalid filter order or Wiener window")

    @property
    def nyquist_hz(self) -> float:
        return self.sample_rate / 2

    @property
    def source_nyquist_hz(self) -> float:
        return self.source_sample_rate / 2

    @property
    def manifest_root(self) -> Path:
        return self.output_root / self.study_id / "manifests" / f"seed-{self.seed}"

    @property
    def run_root(self) -> Path:
        return self.output_root / self.study_id / f"seed-{self.seed}" / str(self.condition)

    @property
    def needs_noise(self) -> bool:
        return self.condition != "clean_reference"

    @property
    def needs_wiener(self) -> bool:
        """Compatibility alias; use ``stages`` for new code."""
        return "wiener" in self.stages

    @property
    def stages(self) -> tuple[str, ...]:
        return {"clean_reference": (), "raw_noisy": (), "high_pass": ("high_pass",), "high_pass_low_pass": ("high_pass", "low_pass"), "full_dsp": ("high_pass", "low_pass", "wiener")}[self.condition]

    def with_seed(self, seed: int) -> "ExperimentConfig":
        return ExperimentConfig(**{**asdict(self), "seed": seed})

    def to_dict(self, *, relative_to: Path = PROJECT_ROOT) -> dict[str, Any]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value.relative_to(relative_to) if value.is_relative_to(relative_to) else value)
        return data


def load_config(path: Path | None = None, **overrides: Any) -> ExperimentConfig:
    """Load a JSON study config and resolve its project-relative paths."""
    data: dict[str, Any] = {}
    if path is not None:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("description", None)
        for key in ("vctk_root", "musan_root", "output_root", "ecapa_cache"):
            if key in data:
                candidate = Path(data[key])
                data[key] = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    data.update({key: value for key, value in overrides.items() if value is not None})
    valid = {field.name for field in fields(ExperimentConfig)}
    return ExperimentConfig(**{key: value for key, value in data.items() if key in valid})


def save_config(config: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_config(condition: ConditionName | str) -> ExperimentConfig:
    return ExperimentConfig(condition=condition)
