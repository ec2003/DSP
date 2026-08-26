from __future__ import annotations

import io
from pathlib import Path
import tarfile

import numpy as np
import pytest
import torch

from src.audio import mfcc_features, mfcc_summary_embedding, stft_magnitude, welch_psd
from src.config.settings import ExperimentConfig
from src.data.download import safe_extract_archive
from src.experiments.evaluate import calibrate_threshold, paired_stratified_bootstrap, verification_metrics


def test_dsp_features_have_documented_shapes_and_are_finite() -> None:
    waveform = torch.sin(2 * torch.pi * 440 * torch.arange(16_000) / 16_000)
    stft = stft_magnitude(waveform)
    mfcc = mfcc_features(waveform, sample_rate=16_000)
    embedding = mfcc_summary_embedding(waveform, sample_rate=16_000)
    frequencies, psd = welch_psd(waveform, sample_rate=16_000)
    assert stft.shape[0] == 257 and mfcc.shape[0] == 40 and embedding.shape == (80,)
    assert torch.isfinite(stft).all() and torch.isfinite(mfcc).all() and torch.isclose(embedding.norm(), torch.tensor(1.0))
    assert frequencies.shape == psd.shape and np.isfinite(psd).all()


def test_fixed_threshold_reports_exact_confusion_counts() -> None:
    scores = np.asarray([0.9, 0.6, 0.65, 0.1])
    labels = np.asarray([1, 1, 0, 0])
    calibrated = calibrate_threshold(scores, labels)
    metrics = verification_metrics(scores, labels, threshold=0.6)
    assert calibrated["threshold"] >= 0.6
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 0, 2)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1.0 and metrics["f1"] == pytest.approx(0.8)


def test_paired_bootstrap_is_deterministic() -> None:
    labels = np.asarray([1, 1, 0, 0])
    raw = np.asarray([.7, .4, .6, .2])
    dsp = np.asarray([.9, .8, .3, .1])
    first = paired_stratified_bootstrap(raw, dsp, labels, threshold_raw=.5, threshold_dsp=.5, repetitions=100, seed=9)
    second = paired_stratified_bootstrap(raw, dsp, labels, threshold_raw=.5, threshold_dsp=.5, repetitions=100, seed=9)
    assert first == second and first["ci95_low"] >= 0


def test_safe_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as bundle:
        info = tarfile.TarInfo("../outside.txt"); info.size = 1
        bundle.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract_archive(archive, tmp_path / "out")


def test_new_conditions_declare_required_ablation_stages() -> None:
    assert ExperimentConfig(condition="raw_noisy").stages == ()
    assert ExperimentConfig(condition="high_pass").stages == ("high_pass",)
    assert ExperimentConfig(condition="high_pass_low_pass").stages == ("high_pass", "low_pass")
    assert ExperimentConfig(condition="full_dsp").stages == ("high_pass", "low_pass", "wiener")
