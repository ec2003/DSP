"""Versioned experiment configuration.

Every numeric knob that affects a result lives in the JSON config so a run can
be reproduced from the config file plus the seed alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

#: DSP stage names that :func:`src.pipeline.build_chain` knows how to build.
KNOWN_STAGES = frozenset(
    {"highpass", "lowpass", "telephone", "notch", "wiener", "specsub"}
)


@dataclass(frozen=True)
class Condition:
    """One experimental arm: whether noise is added, and which DSP stages run."""

    name: str
    add_noise: bool
    stages: tuple[str, ...]

    def __post_init__(self) -> None:
        unknown = set(self.stages) - KNOWN_STAGES
        if unknown:
            raise ValueError(
                f"condition {self.name!r} has unknown stages: {sorted(unknown)}"
            )


@dataclass(frozen=True)
class ExperimentConfig:
    study_id: str
    config_version: int

    vctk_root: str
    musan_root: str
    output_root: str

    sample_rate: int
    segment_seconds: float
    clips_per_speaker: int
    eval_clips_per_speaker: int
    train_speakers: int
    validation_speakers: int

    seeds: tuple[int, ...]
    conditions: tuple[Condition, ...]
    primary_conditions: tuple[str, ...]
    train_snr_db: tuple[float, ...]
    test_snr_db: tuple[float, ...]

    n_fft: int
    hop_length: int
    n_mels: int
    highpass_hz: float
    lowpass_hz: float
    telephone_band_hz: tuple[float, float]
    notch_max_peaks: int
    notch_prominence_db: float
    notch_quality: float
    wiener_noise_quantile: float
    wiener_alpha: float
    wiener_gain_floor_db: float
    specsub_over_subtraction: float
    specsub_spectral_floor: float

    embedding_dim: int
    cnn_channels: tuple[int, ...]
    arcface_margin: float
    arcface_scale: float
    learning_rate: float
    batch_size: int
    epochs: int
    weight_decay: float
    optimizer: str

    enrollment_clips: int
    closed_set_clips: int
    cluster_min_size: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        nyquist = self.sample_rate / 2
        if not 0 < self.highpass_hz < self.lowpass_hz < nyquist:
            raise ValueError(
                f"require 0 < highpass ({self.highpass_hz}) < lowpass "
                f"({self.lowpass_hz}) < Nyquist ({nyquist})"
            )
        if self.enrollment_clips >= self.eval_clips_per_speaker:
            raise ValueError(
                f"enrollment_clips ({self.enrollment_clips}) must be smaller than "
                f"eval_clips_per_speaker ({self.eval_clips_per_speaker})"
            )
        if self.closed_set_clips and self.enrollment_clips >= self.closed_set_clips:
            raise ValueError(
                f"enrollment_clips ({self.enrollment_clips}) must be smaller than "
                f"closed_set_clips ({self.closed_set_clips})"
            )
        known = {condition.name for condition in self.conditions}
        missing = set(self.primary_conditions) - known
        if missing:
            raise ValueError(f"primary_conditions not defined: {sorted(missing)}")

    # -- derived paths ----------------------------------------------------- #
    @property
    def study_root(self) -> Path:
        return Path(self.output_root) / self.study_id

    @property
    def cache_root(self) -> Path:
        return self.study_root / "cache"

    @property
    def report_root(self) -> Path:
        return self.study_root / "reports"

    @property
    def segment_samples(self) -> int:
        return round(self.sample_rate * self.segment_seconds)

    def run_root(self, seed: int, condition: str) -> Path:
        return self.study_root / f"seed-{seed}" / condition

    def condition(self, name: str) -> Condition:
        for candidate in self.conditions:
            if candidate.name == name:
                return candidate
        raise KeyError(f"unknown condition {name!r}")


def load_config(path: Path) -> ExperimentConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["seeds"] = tuple(payload["seeds"])
    payload["primary_conditions"] = tuple(payload["primary_conditions"])
    payload["train_snr_db"] = tuple(payload["train_snr_db"])
    payload["test_snr_db"] = tuple(payload["test_snr_db"])
    payload["cnn_channels"] = tuple(payload["cnn_channels"])
    payload["telephone_band_hz"] = tuple(payload["telephone_band_hz"])
    payload["conditions"] = tuple(
        Condition(name=name, add_noise=spec["add_noise"], stages=tuple(spec["stages"]))
        for name, spec in payload["conditions"].items()
    )
    return ExperimentConfig(**payload)
