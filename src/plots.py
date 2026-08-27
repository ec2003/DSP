"""Compact figures for frozen-CNN factorial results."""

from __future__ import annotations

import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import ExperimentConfig


def plot_dsp_design(config: ExperimentConfig, design: dict[str, object]) -> str:
    """Persist the EDA spectra and chosen passband beside the frozen artifact."""
    spectra = design["spectra"]
    frequencies = np.asarray(spectra["frequencies_hz"])
    figure, axis = plt.subplots(figsize=(7, 3.5))
    axis.semilogy(frequencies, spectra["speech_psd"], label="training speech PSD")
    axis.semilogy(frequencies, spectra["noise_psd"], label="training MUSAN PSD")
    edges = design["selected_edges_hz"]
    axis.axvspan(edges["low"], edges["high"], alpha=.15, color="tab:green", label="frozen passband")
    axis.set(xlabel="frequency (Hz)", ylabel="normalised PSD", title="EDA: speech/noise spectra and selected DSP band")
    axis.legend(fontsize=8); axis.grid(alpha=.2); figure.tight_layout()
    config.report_root.mkdir(parents=True, exist_ok=True)
    path = config.report_root / "eda-passband.png"
    figure.savefig(path, dpi=160); plt.close(figure)
    return str(path)


def plot_training_histories(config: ExperimentConfig) -> list[str]:
    histories = []
    for seed in config.seeds:
        path = config.seed_dir(seed) / "training-history.json"
        if path.is_file():
            histories.append((seed, json.loads(path.read_text(encoding="utf-8"))["history"]))
    if not histories:
        return []
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for seed, history in histories:
        epochs = [row["epoch"] + 1 for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], label=f"seed {seed}")
        axes[1].plot(epochs, [row["validation_accuracy"] for row in history], label=f"seed {seed}")
    axes[0].set(title="Training loss", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Validation identification accuracy", xlabel="epoch", ylabel="accuracy")
    for axis in axes: axis.grid(alpha=.2); axis.legend(fontsize=7)
    figure.tight_layout(); config.report_root.mkdir(parents=True, exist_ok=True)
    output = config.report_root / "training-histories.png"
    figure.savefig(output, dpi=160); plt.close(figure)
    return [str(output)]


def plot_snr_curves(config: ExperimentConfig, rows: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(1, len(config.test_noise_families), figsize=(4.2 * len(config.test_noise_families), 3.5), sharey=True)
    for axis, family in zip(np.atleast_1d(axes), config.test_noise_families, strict=True):
        for cell in ("raw", "bandpass", "wiener", "bandpass_wiener"):
            points = defaultdict(list)
            for row in rows:
                if row["protocol"] == "unseen" and row["noise_family"] == family and row["cell"] == cell:
                    points[float(row["test_snr_db"])].append(float(row["accuracy"]))
            if points:
                snrs = sorted(points); axis.plot(snrs, [np.mean(points[x]) for x in snrs], "-o", label=cell, markersize=3)
        axis.set_title(family); axis.set_xlabel("input SNR (dB)"); axis.grid(alpha=.25)
    axes[0].set_ylabel("open-set identification accuracy")
    axes[-1].legend(fontsize=7)
    figure.tight_layout(); figure.savefig(config.report_root / "snr-curves.png", dpi=160); plt.close(figure)


def plot_factorial_effects(config: ExperimentConfig) -> None:
    path = config.report_root / "factorial-bootstrap.json"
    if not path.is_file(): return
    report = json.loads(path.read_text(encoding="utf-8")).get("unseen", {}).get("accuracy", {})
    names = list(report)
    if not names: return
    values = [report[name]["mean_delta"] for name in names]
    errors = [[values[i] - report[name]["ci_low"] for i, name in enumerate(names)], [report[name]["ci_high"] - values[i] for i, name in enumerate(names)]]
    figure, axis = plt.subplots(figsize=(8, 3.5)); axis.errorbar(names, values, yerr=errors, fmt="o", capsize=4)
    axis.axhline(0, color="black", linewidth=.8); axis.set_ylabel("paired accuracy delta"); axis.set_title("Frozen-CNN inference-time DSP factorial effects (95% speaker bootstrap CI)")
    axis.tick_params(axis="x", rotation=25); figure.tight_layout(); figure.savefig(config.report_root / "factorial-effects.png", dpi=160); plt.close(figure)


def plot_snr_recognition_tradeoff(config: ExperimentConfig, rows: list[dict[str, object]]) -> None:
    path = config.report_root / "front-end-characterisation.json"
    if not path.is_file(): return
    signal = json.loads(path.read_text(encoding="utf-8"))["measurements"]
    figure, axis = plt.subplots(figsize=(6, 4))
    for cell in ("bandpass", "wiener", "bandpass_wiener"):
        points = []
        for item in signal:
            if item["cell"] != cell:
                continue
            matching = [float(row["accuracy"]) for row in rows if row["protocol"] == "unseen" and row["cell"] == cell and row["noise_family"] == item["family"] and float(row["test_snr_db"]) == float(item["input_snr_db"])]
            raw = [float(row["accuracy"]) for row in rows if row["protocol"] == "unseen" and row["cell"] == "raw" and row["noise_family"] == item["family"] and float(row["test_snr_db"]) == float(item["input_snr_db"])]
            if matching and raw:
                points.append((item["snr_gain_db"], np.mean(matching) - np.mean(raw)))
        if points:
            axis.scatter(*zip(*points), alpha=.65, label=cell)
    axis.axhline(0, color="black", linewidth=.8); axis.axvline(0, color="black", linewidth=.8)
    axis.set_xlabel("waveform SNR gain vs raw (dB)"); axis.set_ylabel("paired-cell recognition delta vs raw")
    axis.set_title("Upper-left quadrant: SNR ↑, recognition ↓")
    figure.tight_layout(); figure.savefig(config.report_root / "snr-recognition-tradeoff.png", dpi=160); plt.close(figure)


def render_all(config: ExperimentConfig, rows: list[dict[str, object]]) -> None:
    config.report_root.mkdir(parents=True, exist_ok=True)
    plot_snr_curves(config, rows); plot_factorial_effects(config); plot_snr_recognition_tradeoff(config, rows)
