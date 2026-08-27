"""Offline tests: DSP correctness, mixing, feature shapes, and metric behaviour.

These run without the corpus so they stay fast and independent of the cache.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import dsp
from src.config import Condition, load_config
from src.corpus import trim_silence
from src.eval import (
    classification_metrics,
    clustering_metrics,
    enrollment_split,
    paired_significance,
)
from src.features import handcrafted_features, octave_band_powers, spectral_entropy
from src.noise import deterministic_choice, mix_at_snr
from src.pipeline import build_chain

SAMPLE_RATE = 16000
DURATION = 2.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)
TIME = np.arange(N_SAMPLES) / SAMPLE_RATE


def tone(freq_hz: float, amplitude: float = 1.0) -> np.ndarray:
    return (amplitude * np.sin(2 * np.pi * freq_hz * TIME)).astype(np.float32)


def bin_amplitude(waveform: np.ndarray, freq_hz: float) -> float:
    spectrum = np.fft.rfft(waveform)
    return float(abs(spectrum[int(round(freq_hz * waveform.size / SAMPLE_RATE))]))


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    length = min(reference.size, estimate.size)
    error = estimate[:length] - reference[:length]
    return float(
        10 * np.log10(np.sum(reference[:length] ** 2) / (np.sum(error**2) + 1e-12))
    )


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def test_highpass_attenuates_below_cutoff_and_passes_above():
    mixture = tone(40) + tone(500)
    filtered = dsp.highpass(mixture, SAMPLE_RATE, 80.0)
    assert bin_amplitude(filtered, 40) / bin_amplitude(mixture, 40) < 0.05
    assert bin_amplitude(filtered, 500) / bin_amplitude(mixture, 500) > 0.95


def test_lowpass_attenuates_above_cutoff():
    mixture = tone(500) + tone(6000)
    filtered = dsp.lowpass(mixture, SAMPLE_RATE, 4100.0)
    assert bin_amplitude(filtered, 6000) / bin_amplitude(mixture, 6000) < 0.05
    assert bin_amplitude(filtered, 500) / bin_amplitude(mixture, 500) > 0.95


def test_bandpass_keeps_only_the_passband():
    mixture = tone(100) + tone(1000) + tone(6000)
    filtered = dsp.bandpass(mixture, SAMPLE_RATE, 300.0, 3400.0)
    assert bin_amplitude(filtered, 1000) / bin_amplitude(mixture, 1000) > 0.9
    assert bin_amplitude(filtered, 100) / bin_amplitude(mixture, 100) < 0.1
    assert bin_amplitude(filtered, 6000) / bin_amplitude(mixture, 6000) < 0.1


def test_cutoff_above_nyquist_is_rejected():
    with pytest.raises(ValueError):
        dsp.highpass(tone(500), SAMPLE_RATE, 9000.0)


def test_notch_detection_finds_injected_tone():
    rng = np.random.default_rng(0)
    noisy_tone = (tone(1000, 0.5) + rng.normal(0, 0.05, N_SAMPLES)).astype(np.float32)
    peaks = dsp.detect_tonal_peaks(noisy_tone, SAMPLE_RATE)
    assert peaks and abs(peaks[0] - 1000.0) < 50.0

    notched = dsp.adaptive_notch(noisy_tone, SAMPLE_RATE)
    assert bin_amplitude(notched, 1000) / bin_amplitude(noisy_tone, 1000) < 0.2


def test_adaptive_notch_is_a_no_op_without_tonal_interference():
    rng = np.random.default_rng(1)
    broadband = rng.normal(0, 0.1, N_SAMPLES).astype(np.float32)
    assert dsp.detect_tonal_peaks(broadband, SAMPLE_RATE) == []


# --------------------------------------------------------------------------- #
# Denoising
# --------------------------------------------------------------------------- #
@pytest.fixture
def noisy_pair():
    rng = np.random.default_rng(2)
    clean = (tone(300) * np.hanning(N_SAMPLES)).astype(np.float32)
    return clean, (clean + rng.normal(0, 0.1, N_SAMPLES)).astype(np.float32)


def test_stft_wiener_improves_snr(noisy_pair):
    clean, noisy = noisy_pair
    assert (
        snr_db(clean, dsp.stft_wiener(noisy, SAMPLE_RATE)) > snr_db(clean, noisy) + 5.0
    )


def test_spectral_subtraction_improves_snr(noisy_pair):
    clean, noisy = noisy_pair
    assert (
        snr_db(clean, dsp.spectral_subtraction(noisy, SAMPLE_RATE))
        > snr_db(clean, noisy) + 5.0
    )


def test_wiener_gain_floor_bounds_attenuation():
    rng = np.random.default_rng(3)
    noise_only = rng.normal(0, 0.1, N_SAMPLES).astype(np.float32)
    denoised = dsp.stft_wiener(noise_only, SAMPLE_RATE, gain_floor_db=-6.0)
    ratio = np.sqrt(np.mean(denoised**2) / np.mean(noise_only**2))
    assert ratio > 10 ** (-6.0 / 20.0) * 0.5


def test_denoisers_preserve_length(noisy_pair):
    _, noisy = noisy_pair
    assert dsp.stft_wiener(noisy, SAMPLE_RATE).size == noisy.size
    assert dsp.spectral_subtraction(noisy, SAMPLE_RATE).size == noisy.size


# --------------------------------------------------------------------------- #
# Mixing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target_snr", [0.0, 5.0, 20.0])
def test_mix_at_snr_hits_the_requested_ratio(target_snr):
    rng = np.random.default_rng(4)
    speech = (tone(300) * np.hanning(N_SAMPLES)).astype(np.float32)
    noise = rng.normal(0, 0.3, N_SAMPLES).astype(np.float32)
    mixture = mix_at_snr(speech, noise, target_snr)

    # Anti-clipping applies a single unknown gain to speech and noise alike;
    # recover it by projecting the mixture onto the (uncorrelated) speech.
    gain = float(np.dot(mixture, speech) / np.dot(speech, speech))
    residual = mixture - gain * speech
    measured = 10 * np.log10(np.mean((gain * speech) ** 2) / np.mean(residual**2))
    assert abs(measured - target_snr) < 1.0


def test_deterministic_choice_is_stable_and_in_range():
    assert deterministic_choice("11:p225/clip-000:noise", 2000) == deterministic_choice(
        "11:p225/clip-000:noise", 2000
    )
    assert 0 <= deterministic_choice("11:p225/clip-000:noise", 2000) < 2000
    assert deterministic_choice("a", 100) != deterministic_choice("b", 100)


# --------------------------------------------------------------------------- #
# Corpus helpers
# --------------------------------------------------------------------------- #
def test_trim_silence_removes_leading_and_trailing_silence():
    speech = (tone(300) * np.hanning(N_SAMPLES)).astype(np.float32)
    padded = np.concatenate(
        [np.zeros(8000, np.float32), speech, np.zeros(8000, np.float32)]
    )
    trimmed = trim_silence(padded)
    assert trimmed.size < padded.size
    assert trimmed.size >= speech.size * 0.5


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def test_handcrafted_features_are_finite_and_fixed_width():
    rng = np.random.default_rng(5)
    first = handcrafted_features(tone(300), SAMPLE_RATE)
    second = handcrafted_features(
        rng.normal(0, 0.1, N_SAMPLES).astype(np.float32), SAMPLE_RATE
    )
    assert first.shape == second.shape
    assert np.isfinite(first).all() and np.isfinite(second).all()


def test_spectral_entropy_ranks_noise_above_a_pure_tone():
    rng = np.random.default_rng(6)
    noise = rng.normal(0, 0.1, N_SAMPLES).astype(np.float32)
    assert spectral_entropy(noise, SAMPLE_RATE) > spectral_entropy(
        tone(300), SAMPLE_RATE
    )


def test_octave_band_power_tracks_the_active_band():
    powers = octave_band_powers(tone(300), SAMPLE_RATE)
    assert int(np.argmax(powers)) == 2  # the 250-500 Hz band


# --------------------------------------------------------------------------- #
# Pipeline wiring
# --------------------------------------------------------------------------- #
def test_build_chain_applies_stages_in_order():
    config = load_config("configs/dsp501-v2.json")
    chain = build_chain(config.condition("B_full"), config)
    assert chain.names == ("highpass", "lowpass", "notch", "wiener")

    mixture = tone(100) + tone(6000) + tone(1000)
    processed = chain(mixture)
    assert processed.size == mixture.size
    assert bin_amplitude(processed, 6000) < bin_amplitude(mixture, 6000) * 0.1


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError):
        Condition(name="bad", add_noise=True, stages=("wavelet",))


def test_pipeline_a_has_no_dsp_stages():
    config = load_config("configs/dsp501-v2.json")
    assert config.condition("A_raw_noisy").stages == ()
    assert config.condition("A_raw_noisy").add_noise is True
    assert config.condition("clean").add_noise is False


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_classification_metrics_on_a_perfect_prediction():
    labels = np.array([0, 0, 1, 1, 2, 2])
    metrics = classification_metrics(labels, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert np.trace(np.array(metrics["confusion_matrix"])) == labels.size


def test_clustering_metrics_separate_well_formed_clusters():
    rng = np.random.default_rng(7)
    centres = rng.normal(0, 1, (6, 16))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    embeddings = np.repeat(centres, 20, axis=0) + rng.normal(0, 0.01, (120, 16))
    labels = np.repeat(np.arange(6), 20)
    metrics = clustering_metrics(
        embeddings.astype(np.float32), labels, min_cluster_size=5
    )
    assert metrics["agglomerative_ari"] > 0.95
    assert metrics["agglomerative_purity"] > 0.95


def test_enrollment_split_is_disjoint_and_covers_every_speaker():
    from src.corpus import ClipRecord

    records = [
        ClipRecord(f"p{s}/u/clip-{i:03d}", f"p{s}", "x.wav", "test", 2.0, 0.0)
        for s in range(4)
        for i in range(10)
    ]
    enrol, query = enrollment_split(records, enrollment_clips=3)
    assert set(enrol).isdisjoint(query)
    assert enrol.size == 4 * 3
    assert query.size == 4 * 7


def test_paired_significance_detects_a_consistent_improvement():
    baseline = [0.50, 0.52, 0.48, 0.51]
    proposed = [0.60, 0.61, 0.59, 0.62]
    result = paired_significance(baseline, proposed)
    assert result["mean_difference"] > 0
    assert result["t_p_value"] < 0.05
    assert result["ci_low"] > 0


def test_paired_significance_handles_identical_samples():
    result = paired_significance([0.5, 0.5], [0.5, 0.5])
    assert result["mean_difference"] == 0.0
    assert result["t_p_value"] == 1.0
