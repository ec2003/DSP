"""Tests for demo_live.py Gradio Speaker Verification App."""

import numpy as np
import pytest

from demo_live import (
    add_synthetic_noise,
    build_gradio_app,
    compute_cosine_similarity,
    preprocess_audio,
    process_and_compare,
)


def test_preprocess_audio_numpy_mono():
    # 1 second of 16kHz sine wave
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = (np.sin(2 * np.pi * 440 * t) * 0.8).astype(np.float32)

    processed = preprocess_audio((sr, sine), target_sr=16000, segment_seconds=2.0)
    assert processed is not None
    assert processed.shape == (32000,)
    assert processed.dtype == np.float32
    assert np.max(np.abs(processed)) <= 1.0


def test_preprocess_audio_stereo_int16():
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    left = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
    right = (np.cos(2 * np.pi * 440 * t) * 20000).astype(np.int16)
    stereo = np.column_stack([left, right])

    processed = preprocess_audio((sr, stereo), target_sr=16000, segment_seconds=2.0)
    assert processed is not None
    assert processed.shape == (32000,)
    assert processed.dtype == np.float32


def test_add_synthetic_noise():
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    signal = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    noisy_white = add_synthetic_noise(signal, "White Noise (Gaussian)", snr_db=10.0, sample_rate=sr)
    assert noisy_white.shape == signal.shape
    assert not np.array_equal(noisy_white, signal)

    noisy_pink = add_synthetic_noise(signal, "Pink Noise (1/f)", snr_db=5.0, sample_rate=sr)
    assert noisy_pink.shape == signal.shape

    noisy_none = add_synthetic_noise(signal, "None", snr_db=10.0, sample_rate=sr)
    assert np.array_equal(noisy_none, signal)


def test_compute_cosine_similarity():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])

    assert np.isclose(compute_cosine_similarity(v1, v2), 1.0)
    assert np.isclose(compute_cosine_similarity(v1, v3), 0.0)


def test_process_and_compare_synthetic_inputs():
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    s1 = (np.sin(2 * np.pi * 440 * t) * 0.8).astype(np.float32)
    s2 = (np.sin(2 * np.pi * 880 * t) * 0.8).astype(np.float32)

    raw_html, dsp_html, summary, a1_dsp, a2_dsp, fig = process_and_compare(
        (sr, s1),
        (sr, s2),
        dsp_condition="bandpass_wiener",
        sim_threshold=0.45,
        noise_type="White Noise (Gaussian)",
        snr_db=10.0,
    )

    assert "Model Không có DSP" in raw_html
    assert "Model Có DSP" in dsp_html
    assert "Đánh giá tác động" in summary
    assert a1_dsp is not None
    assert a2_dsp is not None
    assert fig is not None


def test_build_gradio_app():
    app = build_gradio_app()
    assert app is not None
