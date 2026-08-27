"""Empirical filter design and signal-level characterisation of the front-end.

Cutoff frequencies are not asserted from convention: they are derived from the
measured long-term spectra of the VCTK speech and the MUSAN noise actually used
in the study. The rule sacrifices at most ``SPEECH_ENERGY_MARGIN`` of speech
energy at each band edge, and only cuts where the discarded region is genuinely
noise-dominated.

The second analysis measures what each arm does to real clips: how much noise it
removes, and how much it distorts the speech while doing so.

The third analysis is model-free. Waveform SNR says how close the processed
signal is to the clean one in energy terms, which turns out not to predict
recognition. The Fisher ratio below instead measures how separable the speakers
are in a fixed cepstral feature space, with no trained encoder involved, and so
isolates how much speaker-discriminative information the front-end preserves.
"""

from __future__ import annotations

import json

import numpy as np

from src.config import ExperimentConfig
from src.corpus import ClipRecord, load_clips, load_manifest
from src.features import (
    OCTAVE_BAND_EDGES,
    band_power,
    mfcc_frames,
    mfcc_statistics,
    residual_snr_db,
    welch_psd,
)
from src.noise import load_noise_pool_for_split, mix_at_snr, noise_for_sample
from src.pipeline import build_chain

#: Fraction of total speech energy we accept discarding at each band edge.
#: 0.01 removes 13% of the noise energy for 2.4% of the speech energy on this
#: corpus; larger margins cut into the formant region.
SPEECH_ENERGY_MARGIN = 0.01

#: Clips sampled when characterising the front-end at the signal level.
DSP_EFFECT_CLIPS = 200

#: Clips sampled for the model-free speaker-separability measurement.
DISCRIMINABILITY_CLIPS = 900


def average_psd(
    waveforms: np.ndarray, sample_rate: int, nperseg: int = 512, limit: int = 800
) -> tuple[np.ndarray, np.ndarray]:
    """Long-term average PSD over a sample of clips."""
    subset = waveforms[: min(limit, len(waveforms))]
    freqs, accumulated = welch_psd(subset[0], sample_rate, nperseg)
    for waveform in subset[1:]:
        _, psd = welch_psd(waveform, sample_rate, nperseg)
        accumulated = accumulated + psd
    return freqs, accumulated / len(subset)


def _cutoffs_from_cumulative_energy(
    freqs: np.ndarray, speech_psd: np.ndarray, margin: float
) -> tuple[float, float]:
    cumulative = np.cumsum(speech_psd) / speech_psd.sum()
    low_index = min(int(np.searchsorted(cumulative, margin)), freqs.size - 2)
    high_index = int(np.searchsorted(cumulative, 1.0 - margin))
    high_index = min(max(high_index, low_index + 1), freqs.size - 1)
    return float(freqs[low_index]), float(freqs[high_index])


def analyse_bands(config: ExperimentConfig) -> dict[str, object]:
    """Measure speech vs noise band power and recommend high-pass / low-pass cutoffs."""
    speech = load_clips(config.cache_root, "train")
    # Cutoffs are a design decision, so they may only look at training noise.
    noise = load_noise_pool_for_split(config.cache_root, "train")

    freqs, speech_psd = average_psd(speech, config.sample_rate)
    _, noise_psd = average_psd(noise, config.sample_rate)
    # Match total power so the comparison describes a globally 0 dB SNR mixture.
    speech_psd = speech_psd / np.trapezoid(speech_psd, freqs)
    noise_psd = noise_psd / np.trapezoid(noise_psd, freqs)

    band_rows = [
        {
            "low_hz": low_hz,
            "high_hz": high_hz,
            "speech_power_pct": 100.0 * band_power(freqs, speech_psd, low_hz, high_hz),
            "noise_power_pct": 100.0 * band_power(freqs, noise_psd, low_hz, high_hz),
            "band_snr_db": float(
                10.0
                * np.log10(
                    (band_power(freqs, speech_psd, low_hz, high_hz) + 1e-15)
                    / (band_power(freqs, noise_psd, low_hz, high_hz) + 1e-15)
                )
            ),
        }
        for low_hz, high_hz in OCTAVE_BAND_EDGES
    ]

    nyquist = config.sample_rate / 2.0
    highpass_hz, lowpass_hz = _cutoffs_from_cumulative_energy(
        freqs, speech_psd, SPEECH_ENERGY_MARGIN
    )
    # Only cut a band edge when the discarded region is noise-dominated.
    if band_power(freqs, speech_psd, 0.0, highpass_hz) > band_power(
        freqs, noise_psd, 0.0, highpass_hz
    ):
        highpass_hz = 20.0
    if band_power(freqs, speech_psd, lowpass_hz, nyquist) > band_power(
        freqs, noise_psd, lowpass_hz, nyquist
    ):
        lowpass_hz = nyquist * 0.98

    kept = (freqs >= highpass_hz) & (freqs <= lowpass_hz)
    report = {
        "speech_energy_margin": SPEECH_ENERGY_MARGIN,
        "n_speech_clips": int(min(800, len(speech))),
        "n_noise_segments": int(min(800, len(noise))),
        "bands": band_rows,
        "recommended_cutoffs": {
            "highpass_hz": round(highpass_hz, 1),
            "lowpass_hz": round(lowpass_hz, 1),
            "configured_highpass_hz": config.highpass_hz,
            "configured_lowpass_hz": config.lowpass_hz,
            "speech_energy_retained_pct": round(
                100.0 * float(np.trapezoid(speech_psd[kept], freqs[kept])), 2
            ),
            "noise_energy_retained_pct": round(
                100.0 * float(np.trapezoid(noise_psd[kept], freqs[kept])), 2
            ),
        },
        "spectra": {
            "freqs_hz": freqs.tolist(),
            "speech_psd": speech_psd.tolist(),
            "noise_psd": noise_psd.tolist(),
        },
    }

    config.report_root.mkdir(parents=True, exist_ok=True)
    (config.report_root / "band-analysis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def analyse_dsp_effect(config: ExperimentConfig) -> dict[str, object]:
    """Measure noise suppression against speech distortion for every arm.

    ``output_snr_db`` is measured against the clean reference, so it captures
    residual noise *and* processing artefacts together. ``clean_path_snr_db``
    runs the same chain on clean input, isolating the distortion the front-end
    injects by itself; a transparent chain would score arbitrarily high.
    """
    records: list[ClipRecord] = load_manifest(config.cache_root, "test")
    clips = load_clips(config.cache_root, "test")
    pool = load_noise_pool_for_split(config.cache_root, "test")
    seed = config.seeds[0]
    n_clips = min(DSP_EFFECT_CLIPS, len(records))

    rows = []
    for condition in config.conditions:
        if not condition.add_noise:
            continue
        chain = build_chain(condition, config)
        clean_path = [
            residual_snr_db(clips[i], chain(clips[i])) for i in range(n_clips)
        ]

        for input_snr in config.test_snr_db:
            output_snr = []
            for index in range(n_clips):
                noisy = mix_at_snr(
                    clips[index],
                    noise_for_sample(records[index].sample_id, seed, pool),
                    input_snr,
                )
                output_snr.append(residual_snr_db(clips[index], chain(noisy)))
            rows.append(
                {
                    "condition": condition.name,
                    "stages": list(condition.stages),
                    "input_snr_db": float(input_snr),
                    "output_snr_db": float(np.mean(output_snr)),
                    "clean_path_snr_db": float(np.mean(clean_path)),
                }
            )

    baseline = {
        row["input_snr_db"]: row["output_snr_db"]
        for row in rows
        if row["condition"] == "A_raw_noisy"
    }
    for row in rows:
        row["snr_gain_db"] = row["output_snr_db"] - baseline[row["input_snr_db"]]

    report = {"n_clips": n_clips, "seed": seed, "measurements": rows}
    config.report_root.mkdir(parents=True, exist_ok=True)
    (config.report_root / "dsp-effect.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _fisher_ratio(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Between-speaker scatter over within-speaker scatter, per feature dimension.

    Standardising first makes the ratio comparable across arms that change the
    overall feature scale (band limiting does exactly that).
    """
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    grand_mean = features.mean(axis=0)

    within, between = [], []
    for label in np.unique(labels):
        group = features[labels == label]
        within.append(np.sum(group.var(axis=0)) * len(group))
        between.append(np.sum((group.mean(axis=0) - grand_mean) ** 2) * len(group))

    within_scatter = float(np.sum(within) / len(features))
    between_scatter = float(np.sum(between) / len(features))
    return {
        "within_speaker_scatter": within_scatter,
        "between_speaker_scatter": between_scatter,
        "fisher_ratio": between_scatter / (within_scatter + 1e-12),
    }


def analyse_speaker_discriminability(config: ExperimentConfig) -> dict[str, object]:
    """Measure speaker separability in a fixed cepstral space, with no model.

    Waveform SNR measures energy-domain fidelity; this measures whether the
    speakers are still told apart after the front-end. The two can disagree,
    and where they do the recognition results follow this measure.
    """
    records: list[ClipRecord] = load_manifest(config.cache_root, "test")
    clips = load_clips(config.cache_root, "test")
    pool = load_noise_pool_for_split(config.cache_root, "test")
    seed = config.seeds[0]
    n_clips = min(DISCRIMINABILITY_CLIPS, len(records))

    speakers = sorted({record.speaker_id for record in records[:n_clips]})
    labels = np.array([speakers.index(records[i].speaker_id) for i in range(n_clips)])
    probe_snrs = (min(config.test_snr_db), max(config.test_snr_db))

    rows = []
    for condition in config.conditions:
        chain = build_chain(condition, config)
        for input_snr in probe_snrs:
            features = []
            for index in range(n_clips):
                signal = clips[index]
                if condition.add_noise:
                    signal = mix_at_snr(
                        signal,
                        noise_for_sample(records[index].sample_id, seed, pool),
                        input_snr,
                    )
                features.append(
                    mfcc_statistics(mfcc_frames(chain(signal), config.sample_rate))
                )
            rows.append(
                {
                    "condition": condition.name,
                    "stages": list(condition.stages),
                    "input_snr_db": float(input_snr),
                    **_fisher_ratio(np.stack(features), labels),
                }
            )
            if not condition.add_noise:
                break  # the clean arm has no SNR axis

    report = {"n_clips": n_clips, "n_speakers": len(speakers), "measurements": rows}
    (config.report_root / "discriminability.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
