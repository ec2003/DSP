"""Compact figures for frozen-CNN factorial results."""

from __future__ import annotations

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal

from src.config import ExperimentConfig


def plot_dsp_design(config: ExperimentConfig, design: dict[str, object]) -> str:
    """Persist the EDA spectra and chosen passband beside the frozen artifact."""
    spectra = design["spectra"]
    frequencies = np.asarray(spectra["frequencies_hz"])
    figure, axis = plt.subplots(figsize=(7, 3.5))
    axis.semilogy(frequencies, spectra["speech_psd"], label="training speech PSD")
    axis.semilogy(frequencies, spectra["noise_psd"], label="training MUSAN PSD")
    edges = design["selected_edges_hz"]
    axis.axvspan(
        edges["low"],
        edges["high"],
        alpha=0.15,
        color="tab:green",
        label="frozen passband",
    )
    axis.set(
        xlabel="frequency (Hz)",
        ylabel="normalised PSD",
        title="EDA: speech/noise spectra and selected DSP band",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    config.report_root.mkdir(parents=True, exist_ok=True)
    path = config.report_root / "eda-passband.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return str(path)


def plot_signal_gallery(config: ExperimentConfig) -> str:
    """Waveform, STFT spectrogram and log-mel input for the clean control and each cell."""
    from src.corpus import load_clips, load_manifest
    from src.features import LogMelSpectrogram
    from src.noise import (
        load_noise_metadata_for_split,
        load_noise_pool_for_split,
        make_mixture,
    )
    from src.pipeline import build_chain

    clips, records = (
        load_clips(config.cache_root, "test"),
        load_manifest(config.cache_root, "test"),
    )
    families = ("noise", "music", "speech")
    pools = {
        name: load_noise_pool_for_split(config.cache_root, "test", name)
        for name in families
    }
    metadata = {
        name: load_noise_metadata_for_split(config.cache_root, "test", name)
        for name in families
    }
    clean = np.asarray(clips[0], dtype=np.float32)
    snr = float(min(config.test_snr_db))
    mixture = make_mixture(
        clean,
        records[0].sample_id,
        seed=config.seeds[0],
        family="noise",
        snr_db=snr,
        pools=pools,
        pool_metadata=metadata,
        babble_sources_range=config.babble_sources_range,
    ).mixture
    panels = [("clean control", clean), (f"raw mixture @ {snr:g} dB", mixture)]
    panels += [
        (f"{condition.name} front-end", build_chain(condition, config)(mixture))
        for condition in config.conditions
        if condition.stages
    ]
    mel = LogMelSpectrogram(
        config.sample_rate, config.n_fft, config.hop_length, config.n_mels
    )
    time = np.arange(config.segment_samples) / config.sample_rate
    figure, axes = plt.subplots(len(panels), 3, figsize=(9.5, 2.05 * len(panels)))
    for row, (title, waveform) in enumerate(panels):
        waveform = np.asarray(waveform, dtype=np.float32)[: config.segment_samples]
        axes[row, 0].plot(time[: len(waveform)], waveform, linewidth=0.4)
        axes[row, 0].set(ylabel=title, xlim=(0, config.segment_seconds))
        _, _, spectrum = signal.stft(
            waveform,
            config.sample_rate,
            nperseg=config.n_fft,
            noverlap=config.n_fft - config.hop_length,
        )
        axes[row, 1].imshow(
            20 * np.log10(np.abs(spectrum) + 1e-10),
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=-120,
            vmax=-20,
            extent=(0, config.segment_seconds, 0, config.sample_rate / 2000),
        )
        with torch.no_grad():
            features = mel(torch.from_numpy(waveform)).numpy()
        axes[row, 2].imshow(
            features,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=-3,
            vmax=3,
            extent=(0, config.segment_seconds, 0, config.n_mels),
        )
        for column in (1, 2):
            axes[row, column].set_xlim(0, config.segment_seconds)
    for column, name in enumerate(
        ("waveform", "STFT magnitude (dB)", "normalised log-mel")
    ):
        axes[0, column].set_title(name, fontsize=9)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    axes[-1, 2].set_xlabel("time (s)")
    axes[0, 1].set_ylabel("kHz")
    axes[0, 2].set_ylabel("mel bin")
    figure.tight_layout()
    config.report_root.mkdir(parents=True, exist_ok=True)
    path = config.report_root / "signal-gallery.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return str(path)


def plot_tuning_grid(config: ExperimentConfig, report: dict[str, object]) -> str:
    """Validation accuracy over the searched learning-rate x ArcFace-margin grid."""
    rates, margins = (
        report["search_space"]["learning_rate"],
        report["search_space"]["arcface_margin"],
    )
    scores = np.full((len(rates), len(margins)), np.nan)
    for trial in report["trials"]:
        scores[
            rates.index(trial["learning_rate"]), margins.index(trial["arcface_margin"])
        ] = trial["best_validation_accuracy"]
    figure, axis = plt.subplots(figsize=(5.2, 3.6))
    image = axis.imshow(scores, cmap="viridis", aspect="auto")
    axis.set(
        xticks=range(len(margins)),
        yticks=range(len(rates)),
        xlabel="ArcFace margin",
        ylabel="learning rate",
        title=f"Validation accuracy over {len(rates) * len(margins)} tuning trials",
    )
    axis.set_xticklabels([f"{x:g}" for x in margins])
    axis.set_yticklabels([f"{x:g}" for x in rates])
    for i in range(len(rates)):
        for j in range(len(margins)):
            axis.text(
                j,
                i,
                f"{scores[i, j]:.3f}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="validation accuracy")
    figure.tight_layout()
    config.report_root.mkdir(parents=True, exist_ok=True)
    path = config.report_root / "tuning-grid.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return str(path)


def plot_confusion_matrices(
    config: ExperimentConfig, *, protocol: str = "unseen", family: str = "noise"
) -> str | None:
    """Pipeline A vs Pipeline B confusion at the hardest configured SNR."""
    snr = min(config.test_snr_db)
    seed = config.seeds[0]
    suffix = "" if protocol == "unseen" else "-seen"
    cells = ("raw", "bandpass_wiener")
    matrices = []
    for cell in cells:
        path = (
            config.seed_dir(seed)
            / f"evaluation-{family}-snr-{snr:g}-{cell}{suffix}.json"
        )
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        matrices.append(
            (
                cell,
                np.asarray(
                    payload["nearest_centroid"]["confusion_matrix"], dtype=float
                ),
                payload["nearest_centroid"]["accuracy"],
            )
        )
    figure, axes = plt.subplots(1, len(matrices), figsize=(4.6 * len(matrices), 4.1))
    for axis, (cell, matrix, accuracy) in zip(
        np.atleast_1d(axes), matrices, strict=True
    ):
        normalised = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
        image = axis.imshow(normalised, cmap="magma", vmin=0, vmax=1)
        axis.set(
            title=f"{cell} — accuracy {accuracy:.3f}",
            xlabel="predicted speaker",
            ylabel="true speaker",
        )
        figure.colorbar(image, ax=axis, fraction=0.046, label="row-normalised rate")
    figure.suptitle(
        f"Confusion, {protocol} protocol, {family} @ {snr:g} dB, seed {seed}",
        fontsize=10,
    )
    figure.tight_layout()
    config.report_root.mkdir(parents=True, exist_ok=True)
    path = config.report_root / "confusion-matrices.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return str(path)


def plot_training_histories(config: ExperimentConfig) -> list[str]:
    histories = []
    for seed in config.seeds:
        path = config.seed_dir(seed) / "training-history.json"
        if path.is_file():
            histories.append(
                (seed, json.loads(path.read_text(encoding="utf-8"))["history"])
            )
    if not histories:
        return []
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for seed, history in histories:
        epochs = [row["epoch"] + 1 for row in history]
        axes[0].plot(
            epochs, [row["train_loss"] for row in history], label=f"seed {seed}"
        )
        axes[1].plot(
            epochs,
            [row["validation_accuracy"] for row in history],
            label=f"seed {seed}",
        )
    axes[0].set(title="Training loss", xlabel="epoch", ylabel="loss")
    axes[1].set(
        title="Validation identification accuracy", xlabel="epoch", ylabel="accuracy"
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.tight_layout()
    config.report_root.mkdir(parents=True, exist_ok=True)
    output = config.report_root / "training-histories.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return [str(output)]


def plot_snr_curves(config: ExperimentConfig, rows: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(
        1,
        len(config.test_noise_families),
        figsize=(4.2 * len(config.test_noise_families), 3.5),
        sharey=True,
    )
    for axis, family in zip(
        np.atleast_1d(axes), config.test_noise_families, strict=True
    ):
        for cell in ("raw", "bandpass", "wiener", "bandpass_wiener"):
            points = defaultdict(list)
            for row in rows:
                if (
                    row["protocol"] == "unseen"
                    and row["noise_family"] == family
                    and row["cell"] == cell
                ):
                    points[float(row["test_snr_db"])].append(float(row["accuracy"]))
            if points:
                snrs = sorted(points)
                axis.plot(
                    snrs,
                    [np.mean(points[x]) for x in snrs],
                    "-o",
                    label=cell,
                    markersize=3,
                )
        axis.set_title(family)
        axis.set_xlabel("input SNR (dB)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("open-set identification accuracy")
    axes[-1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(config.report_root / "snr-curves.png", dpi=160)
    plt.close(figure)


def plot_factorial_effects(config: ExperimentConfig) -> None:
    path = config.report_root / "factorial-bootstrap.json"
    if not path.is_file():
        return
    report = (
        json.loads(path.read_text(encoding="utf-8"))
        .get("unseen", {})
        .get("accuracy", {})
    )
    names = list(report)
    if not names:
        return
    values = [report[name]["mean_delta"] for name in names]
    errors = [
        [values[i] - report[name]["ci_low"] for i, name in enumerate(names)],
        [report[name]["ci_high"] - values[i] for i, name in enumerate(names)],
    ]
    figure, axis = plt.subplots(figsize=(8, 3.5))
    axis.errorbar(names, values, yerr=errors, fmt="o", capsize=4)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("paired accuracy delta")
    axis.set_title(
        "Frozen-CNN inference-time DSP factorial effects (95% speaker bootstrap CI)"
    )
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(config.report_root / "factorial-effects.png", dpi=160)
    plt.close(figure)


def plot_snr_recognition_tradeoff(
    config: ExperimentConfig, rows: list[dict[str, object]]
) -> None:
    path = config.report_root / "front-end-characterisation.json"
    if not path.is_file():
        return
    signal = json.loads(path.read_text(encoding="utf-8"))["measurements"]
    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    for cell in ("bandpass", "wiener", "bandpass_wiener"):
        points = []
        for item in signal:
            if item["cell"] != cell:
                continue
            matching = [
                float(row["accuracy"])
                for row in rows
                if row["protocol"] == "unseen"
                and row["cell"] == cell
                and row["noise_family"] == item["family"]
                and float(row["test_snr_db"]) == float(item["input_snr_db"])
            ]
            raw = [
                float(row["accuracy"])
                for row in rows
                if row["protocol"] == "unseen"
                and row["cell"] == "raw"
                and row["noise_family"] == item["family"]
                and float(row["test_snr_db"]) == float(item["input_snr_db"])
            ]
            if matching and raw:
                points.append((item["snr_gain_db"], np.mean(matching) - np.mean(raw)))
        if points:
            axis.scatter(*zip(*points), alpha=0.65, label=cell)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("waveform reference-SNR gain vs raw (dB)")
    axis.set_ylabel("recognition accuracy delta vs raw")
    axis.set_title("Lower-right quadrant: SNR ↑, recognition ↓")
    axis.legend(frameon=False, ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(config.report_root / "snr-recognition-tradeoff.png", dpi=160)
    plt.close(figure)


def render_all(config: ExperimentConfig, rows: list[dict[str, object]]) -> None:
    config.report_root.mkdir(parents=True, exist_ok=True)
    plot_snr_curves(config, rows)
    plot_factorial_effects(config)
    plot_snr_recognition_tradeoff(config, rows)
    plot_confusion_matrices(config)
