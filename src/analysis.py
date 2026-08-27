"""Frozen band-pass design and signal-level controls for the factorial study."""

from __future__ import annotations

import json

import numpy as np

from src.config import ExperimentConfig
from src.corpus import load_clips, load_manifest
from src.features import band_power, residual_snr_db, welch_psd
from src.noise import gaussian_positive_control, load_noise_pool_for_split, load_noise_metadata, make_mixture


def config_hash(config: ExperimentConfig) -> str:
    """Hash all design-affecting configuration, not a hand-copied cutoff."""
    return config.config_hash


def average_psd(waveforms: np.ndarray, sample_rate: int, limit: int = 800) -> tuple[np.ndarray, np.ndarray]:
    subset = waveforms[: min(limit, len(waveforms))]
    if not len(subset):
        raise ValueError("cannot derive a filter from zero waveforms")
    frequencies, total = welch_psd(subset[0], sample_rate, 512)
    for waveform in subset[1:]:
        _, psd = welch_psd(waveform, sample_rate, 512)
        total += psd
    return frequencies, total / len(subset)


def derive_band_design(config: ExperimentConfig) -> dict[str, object]:
    """Freeze the passband using training speech and training MUSAN *only*."""
    speech = load_clips(config.cache_root, "train")
    noise_pools = [load_noise_pool_for_split(config.cache_root, "train", family) for family in ("noise", "music", "speech")]
    frequencies, speech_psd = average_psd(speech, config.sample_rate)
    _, noise_psd = average_psd(np.concatenate(noise_pools), config.sample_rate)
    speech_psd /= np.trapezoid(speech_psd, frequencies)
    noise_psd /= np.trapezoid(noise_psd, frequencies)
    cumulative = np.cumsum(speech_psd) / np.sum(speech_psd)
    margin = config.band_speech_energy_margin
    low_index = int(np.searchsorted(cumulative, margin))
    high_index = int(np.searchsorted(cumulative, 1 - margin))
    low_index = min(max(1, low_index), len(frequencies) - 2)
    high_index = min(max(low_index + 1, high_index), len(frequencies) - 1)
    low, high = float(frequencies[low_index]), float(frequencies[high_index])
    # An edge may be rejected only if its discarded region is not noise-dominated.
    if band_power(frequencies, speech_psd, 0, low) > band_power(frequencies, noise_psd, 0, low):
        low = float(frequencies[1])
    nyquist = config.sample_rate / 2
    if band_power(frequencies, speech_psd, high, nyquist) > band_power(frequencies, noise_psd, high, nyquist):
        high = float(frequencies[-2])
    if not 0 < low < high < nyquist:
        raise ValueError("data-derived passband is invalid")
    keep = (frequencies >= low) & (frequencies <= high)
    artifact = {
        "artifact": "frozen-dsp-band-design",
        "config_hash": config_hash(config),
        "analysis_inputs": {
            "speech_split": "train",
            "noise_partitions": {family: "train" for family in ("noise", "music", "speech")},
            "n_speech_clips": int(min(800, len(speech))),
            "n_noise_segments": int(sum(min(800, len(pool)) for pool in noise_pools)),
        },
        "selection_rule": {"speech_energy_margin": margin, "require_noise_dominance": True, "butterworth_order": config.bandpass_order},
        "selected_edges_hz": {"low": round(low, 3), "high": round(high, 3)},
        "retained_energy_pct": {"speech": float(100 * np.trapezoid(speech_psd[keep], frequencies[keep])), "noise": float(100 * np.trapezoid(noise_psd[keep], frequencies[keep]))},
        "spectra": {"frequencies_hz": frequencies.tolist(), "speech_psd": speech_psd.tolist(), "noise_psd": noise_psd.tolist()},
    }
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.dsp_design_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def positive_control_wiener(config: ExperimentConfig) -> dict[str, object]:
    """DSP implementation check: Gaussian noise should show positive Wiener SNR gain."""
    from src.pipeline import build_chain

    # A deterministic speech-like envelope creates genuinely quiet STFT frames,
    # so the test exercises the intended stationary-noise estimator rather than
    # relying on a chance pause in a corpus utterance.
    t = np.arange(config.segment_samples) / config.sample_rate
    clean = (
        (0.25 * np.sin(2 * np.pi * 300 * t) + 0.10 * np.sin(2 * np.pi * 1400 * t))
        * np.hanning(config.segment_samples)
    ).astype(np.float32)
    chain = build_chain(config.condition("wiener"), config)
    rows = []
    for snr in config.positive_control_snr_db:
        trial = gaussian_positive_control(clean, snr, seed=config.seeds[0] + int(snr * 10))
        before = residual_snr_db(trial.clean_component, trial.mixture)
        after = residual_snr_db(trial.clean_component, chain(trial.mixture))
        rows.append({"input_snr_db": snr, "before_snr_db": before, "after_snr_db": after, "snr_gain_db": after - before})
    report = {"purpose": "technical DSP-positive control; not recognition evidence", "measurements": rows}
    config.report_root.mkdir(parents=True, exist_ok=True)
    (config.report_root / "wiener-positive-control.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def characterize_frontends(config: ExperimentConfig) -> dict[str, object]:
    """Waveform SNR changes for the actual factorial cells on matched mixtures."""
    from src.pipeline import build_chain

    clips, records = load_clips(config.cache_root, "test"), load_manifest(config.cache_root, "test")
    pools = {f: load_noise_pool_for_split(config.cache_root, "test", f) for f in ("noise", "music", "speech")}
    metadata = {f: load_noise_metadata(config.cache_root, "test", f) for f in pools}
    rows = []
    for family in config.test_noise_families:
        for snr in config.test_snr_db:
            mixtures = [make_mixture(clips[i], records[i].sample_id, seed=config.seeds[0], family=family, snr_db=snr, pools=pools, pool_metadata=metadata, babble_sources_range=config.babble_sources_range) for i in range(min(100, len(records)))]
            raw = float(np.mean([residual_snr_db(x.clean_component, x.mixture) for x in mixtures]))
            for condition in config.conditions:
                output = [build_chain(condition, config)(x.mixture) for x in mixtures]
                value = float(np.mean([residual_snr_db(x.clean_component, y) for x, y in zip(mixtures, output, strict=True)]))
                rows.append({"cell": condition.name, "family": family, "input_snr_db": snr, "output_snr_db": value, "snr_gain_db": value - raw})
    report = {"n_clips": min(100, len(records)), "measurements": rows}
    (config.report_root / "front-end-characterisation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
