"""Frozen-artifact DSP front-end and matched 2×2 mixture construction."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from src import dsp
from src.analysis import config_hash
from src.config import Condition, ExperimentConfig
from src.corpus import ClipRecord
from src.noise import MixResult, load_noise_metadata_for_split, load_noise_pool_for_split, make_mixture


@dataclass(frozen=True)
class ProcessedSplit:
    waveforms: np.ndarray
    mixtures: np.ndarray
    clean_components: np.ndarray
    noise_components: np.ndarray
    mixing: list[dict[str, object]]


def frozen_band_edges(config: ExperimentConfig) -> tuple[float, float]:
    """Read, validate, and use the sole cutoff source for every consumer."""
    if not config.dsp_design_path.is_file():
        raise FileNotFoundError("Missing frozen DSP design artifact; run the `eda` phase after prepare.")
    artifact = json.loads(config.dsp_design_path.read_text(encoding="utf-8"))
    if artifact.get("artifact") != "frozen-dsp-band-design" or artifact.get("config_hash") != config_hash(config):
        raise ValueError("frozen DSP design artifact does not match this configuration")
    edges = artifact.get("selected_edges_hz", {})
    low, high = float(edges.get("low", 0)), float(edges.get("high", 0))
    if not 0 < low < high < config.sample_rate / 2:
        raise ValueError("frozen DSP design has invalid band-pass edges")
    return low, high


def build_chain(condition: Condition, config: ExperimentConfig) -> dsp.DspChain:
    low, high = frozen_band_edges(config) if "bandpass" in condition.stages else (0.0, 0.0)
    builders = {
        "bandpass": lambda x: dsp.bandpass(x, config.sample_rate, low, high, config.bandpass_order),
        "wiener": lambda x: dsp.stft_wiener(x, config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length, noise_quantile=config.wiener_noise_quantile, alpha=config.wiener_alpha, gain_floor_db=config.wiener_gain_floor_db),
        "notch": lambda x: dsp.adaptive_notch(x, config.sample_rate, config.notch_max_peaks, config.notch_prominence_db, config.notch_quality),
    }
    return dsp.DspChain(names=condition.stages, stages=tuple(builders[name] for name in condition.stages))


def process_split(clips: np.ndarray, records: list[ClipRecord], condition: Condition, config: ExperimentConfig, *, seed: int, family: str | None, snr_db: float | None) -> ProcessedSplit:
    """Apply a cell to clean controls or deterministic, shared pre-DSP mixtures."""
    if len(clips) != len(records):
        raise ValueError("clip/record mismatch")
    if (family is None) != (snr_db is None):
        raise ValueError("clean controls require neither family nor SNR; noisy cells require both")
    chain = build_chain(condition, config)
    if family is not None:
        pools = {name: load_noise_pool_for_split(config.cache_root, records[0].split, name) for name in ("noise", "music", "speech")}
        metadata = {name: load_noise_metadata_for_split(config.cache_root, records[0].split, name) for name in pools}
    outputs: list[np.ndarray] = []
    mixtures: list[np.ndarray] = []
    clean_components: list[np.ndarray] = []
    noise_components: list[np.ndarray] = []
    mixing: list[dict[str, object]] = []
    for clip, record in zip(clips, records, strict=True):
        if family is None:
            result = MixResult(np.asarray(clip, dtype=np.float32), np.asarray(clip, dtype=np.float32), np.zeros_like(clip, dtype=np.float32), {"sample_id": record.sample_id, "family": "clean", "target_snr_db": None, "measured_snr_db": None, "source_recordings": [], "seed": seed})
        else:
            result = make_mixture(clip, record.sample_id, seed=seed, family=family, snr_db=snr_db, pools=pools, pool_metadata=metadata, babble_sources_range=config.babble_sources_range)
        mixtures.append(result.mixture)
        clean_components.append(result.clean_component)
        noise_components.append(result.noise_component)
        mixing.append(result.metadata)
        outputs.append(chain(result.mixture))
    return ProcessedSplit(np.stack(outputs).astype(np.float32), np.stack(mixtures).astype(np.float32), np.stack(clean_components).astype(np.float32), np.stack(noise_components).astype(np.float32), mixing)
