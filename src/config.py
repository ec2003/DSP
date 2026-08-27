"""Configuration for one immutable frozen-CNN DSP factorial experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

KNOWN_STAGES = frozenset({"bandpass", "wiener", "notch"})
FACTORIAL_CELLS = ("raw", "bandpass", "wiener", "bandpass_wiener")
MUSAN_FAMILIES = ("noise", "music", "speech", "babble")


@dataclass(frozen=True)
class Condition:
    """An inference-time front-end cell; it never chooses a training recipe."""

    name: str
    stages: tuple[str, ...]

    def __post_init__(self) -> None:
        unknown = set(self.stages) - KNOWN_STAGES
        if unknown:
            raise ValueError(f"condition {self.name!r} has unknown stages: {sorted(unknown)}")


@dataclass(frozen=True)
class ExperimentConfig:
    study_id: str
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
    train_snr_range_db: tuple[float, float]
    test_snr_db: tuple[float, ...]
    noise_pool_size: int
    noise_split_fractions: tuple[float, ...]
    p_noise: float
    train_noise_families: tuple[str, ...]
    test_noise_families: tuple[str, ...]
    babble_sources_range: tuple[int, int]
    band_speech_energy_margin: float
    bandpass_order: int
    n_fft: int
    hop_length: int
    n_mels: int
    notch_enabled: bool
    notch_max_peaks: int
    notch_prominence_db: float
    notch_quality: float
    wiener_noise_quantile: float
    wiener_alpha: float
    wiener_gain_floor_db: float
    positive_control_snr_db: tuple[float, ...]
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
    bootstrap_replicates: int
    # Bound by the orchestration API; deliberately absent from config.json.
    run_id: int | None = None

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.segment_seconds <= 0:
            raise ValueError("sample_rate and segment_seconds must be positive")
        if not 0 <= self.p_noise <= 1:
            raise ValueError("p_noise must lie in [0, 1]")
        lo, hi = self.train_snr_range_db
        if lo < 0 or hi < lo:
            raise ValueError("train_snr_range_db must be increasing and non-negative")
        names = tuple(condition.name for condition in self.conditions)
        if names != FACTORIAL_CELLS:
            raise ValueError(f"conditions must be exactly {FACTORIAL_CELLS}, got {names}")
        if len(self.noise_split_fractions) != 3 or min(self.noise_split_fractions) <= 0 or abs(sum(self.noise_split_fractions) - 1) > 1e-6:
            raise ValueError("noise_split_fractions must be three positive values summing to 1")
        if not set(self.train_noise_families) <= set(MUSAN_FAMILIES) or not set(self.test_noise_families) <= set(MUSAN_FAMILIES):
            raise ValueError(f"noise families must be drawn from {MUSAN_FAMILIES}")
        if not self.train_noise_families or not self.test_noise_families:
            raise ValueError("at least one train and test noise family is required")
        if not 3 <= self.babble_sources_range[0] <= self.babble_sources_range[1] <= 7:
            raise ValueError("babble_sources_range must lie in [3, 7]")
        if not 0 < self.band_speech_energy_margin < 0.5:
            raise ValueError("band_speech_energy_margin must lie in (0, .5)")
        if self.bandpass_order < 1:
            raise ValueError("bandpass_order must be positive")
        if self.enrollment_clips >= self.eval_clips_per_speaker:
            raise ValueError("enrollment_clips must be smaller than eval_clips_per_speaker")
        if self.closed_set_clips and self.enrollment_clips >= self.closed_set_clips:
            raise ValueError("enrollment_clips must be smaller than closed_set_clips")
        if self.bootstrap_replicates < 100:
            raise ValueError("bootstrap_replicates must be at least 100")

    @property
    def study_root(self) -> Path:
        return Path(self.output_root) / self.study_id

    @property
    def cache_root(self) -> Path:
        return self.study_root / "cache" / self.data_cache_hash

    @property
    def data_cache_hash(self) -> str:
        """Stable identity for artifacts materialised by the prepare stage."""
        payload = asdict(self)
        keys = (
            "vctk_root", "musan_root", "sample_rate",
            "segment_seconds", "clips_per_speaker", "eval_clips_per_speaker",
            "train_speakers", "validation_speakers", "seeds", "noise_pool_size",
            "noise_split_fractions", "closed_set_clips",
        )
        return sha256(json.dumps({key: payload[key] for key in keys}, sort_keys=True, default=list).encode()).hexdigest()[:16]

    @property
    def config_hash(self) -> str:
        """Identity for an experiment design, excluding its destination/run."""
        payload = asdict(self)
        payload.pop("output_root", None)
        payload.pop("run_id", None)
        return sha256(json.dumps(payload, sort_keys=True, default=list).encode()).hexdigest()

    @property
    def run_tag(self) -> str:
        if self.run_id is None:
            raise RuntimeError("configuration is not bound to an experiment run")
        return f"run-{self.run_id}"

    @property
    def run_dir(self) -> Path:
        return self.study_root / "runs" / self.run_tag

    @property
    def report_root(self) -> Path:
        return self.run_dir / "reports"

    @property
    def dsp_design_path(self) -> Path:
        return self.run_dir / "dsp-design.json"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run-manifest.json"

    @property
    def segment_samples(self) -> int:
        return round(self.sample_rate * self.segment_seconds)

    @property
    def config_snapshot_path(self) -> Path:
        return self.run_dir / "config.json"

    def seed_dir(self, seed: int) -> Path:
        return self.run_dir / f"seed-{seed}" / "robust_cnn"

    def condition(self, name: str) -> Condition:
        for condition in self.conditions:
            if condition.name == name:
                return condition
        raise KeyError(f"unknown condition {name!r}")


def load_config(path: Path | str) -> ExperimentConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in (
        "seeds", "train_snr_range_db", "test_snr_db", "noise_split_fractions",
        "train_noise_families", "test_noise_families", "babble_sources_range",
        "positive_control_snr_db", "cnn_channels",
    ):
        payload[key] = tuple(payload[key])
    payload["conditions"] = tuple(
        Condition(name=name, stages=tuple(spec["stages"]))
        for name, spec in payload["conditions"].items()
    )
    return ExperimentConfig(**payload)
