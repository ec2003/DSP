"""Figures for the report: signal diagnostics, filter design, and results."""

from __future__ import annotations

import json
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

from src import dsp
from src.config import ExperimentConfig
from src.corpus import load_clips, load_manifest
from src.noise import load_noise_pool_for_split, mix_at_snr, noise_for_sample
from src.pipeline import build_chain

matplotlib.use("Agg")

FIGSIZE_WIDE = (11, 4.5)
DEMO_SNR_DB = 5.0


def _demo_signals(config: ExperimentConfig) -> dict[str, np.ndarray]:
    """One deterministic held-out clip at each stage of Pipeline B."""
    records = load_manifest(config.cache_root, "test")
    clips = load_clips(config.cache_root, "test")
    pool = load_noise_pool_for_split(config.cache_root, "test")
    seed = config.seeds[0]

    clean = clips[0]
    noisy = mix_at_snr(
        clean, noise_for_sample(records[0].sample_id, seed, pool), DEMO_SNR_DB
    )

    signals = {"clean": clean, "noisy": noisy}
    stages = config.condition("B_full").stages
    for depth in range(1, len(stages) + 1):
        partial = dsp.DspChain(
            names=stages[:depth],
            stages=build_chain(config.condition("B_full"), config).stages[:depth],
        )
        signals["+".join(stages[:depth])] = partial(noisy)
    return signals


def plot_waveforms(config: ExperimentConfig) -> None:
    signals = _demo_signals(config)
    time = np.arange(len(signals["clean"])) / config.sample_rate
    figure, axes = plt.subplots(
        len(signals), 1, figsize=(11, 1.6 * len(signals)), sharex=True
    )
    for axis, (name, waveform) in zip(axes, signals.items()):
        axis.plot(time, waveform, linewidth=0.4)
        axis.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
        axis.set_ylim(-1, 1)
    axes[-1].set_xlabel("time (s)")
    figure.suptitle(
        f"Pipeline B stages on one held-out clip at {DEMO_SNR_DB:.0f} dB SNR"
    )
    figure.tight_layout()
    figure.savefig(config.report_root / "waveforms.png", dpi=150)
    plt.close(figure)


def plot_spectrograms(config: ExperimentConfig) -> None:
    signals = _demo_signals(config)
    keys = ["clean", "noisy", list(signals)[-1]]
    figure, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE, sharey=True)
    for axis, key in zip(axes, keys):
        freqs, times, spectrum = scipy_signal.stft(
            signals[key],
            fs=config.sample_rate,
            nperseg=config.n_fft,
            noverlap=config.n_fft - config.hop_length,
        )
        axis.pcolormesh(
            times,
            freqs,
            20 * np.log10(np.abs(spectrum) + 1e-8),
            shading="auto",
            vmin=-100,
            vmax=-20,
            cmap="magma",
        )
        axis.set_title(key, fontsize=9)
        axis.set_xlabel("time (s)")
    axes[0].set_ylabel("frequency (Hz)")
    figure.suptitle("STFT magnitude before and after the DSP front-end")
    figure.tight_layout()
    figure.savefig(config.report_root / "spectrograms.png", dpi=150)
    plt.close(figure)


def plot_band_analysis(config: ExperimentConfig) -> None:
    report_path = config.report_root / "band-analysis.json"
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    freqs = np.array(report["spectra"]["freqs_hz"])
    speech = np.array(report["spectra"]["speech_psd"])
    noise = np.array(report["spectra"]["noise_psd"])
    cutoffs = report["recommended_cutoffs"]

    figure, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    axes[0].semilogy(freqs, speech, label="VCTK speech")
    axes[0].semilogy(freqs, noise, label="MUSAN noise")
    for cutoff in (cutoffs["highpass_hz"], cutoffs["lowpass_hz"]):
        axes[0].axvline(cutoff, color="k", linestyle="--", linewidth=1)
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("normalised PSD")
    axes[0].set_title("Long-term average spectra (0 dB global SNR)")
    axes[0].legend(fontsize=8)

    bands = report["bands"]
    positions = np.arange(len(bands))
    axes[1].bar(positions, [band["band_snr_db"] for band in bands], color="tab:blue")
    axes[1].axhline(0, color="k", linewidth=0.8)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(
        [f"{int(b['low_hz'])}-{int(b['high_hz'])}" for b in bands],
        rotation=45,
        fontsize=7,
    )
    axes[1].set_ylabel("band SNR (dB)")
    axes[1].set_title("Octave-band speech-to-noise ratio")

    figure.tight_layout()
    figure.savefig(config.report_root / "band-analysis.png", dpi=150)
    plt.close(figure)


def plot_filter_response(config: ExperimentConfig) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    designs = {
        f"high-pass {config.highpass_hz:.0f} Hz": dsp.butter_sos(
            "highpass", config.highpass_hz, config.sample_rate
        ),
        f"low-pass {config.lowpass_hz:.0f} Hz": dsp.butter_sos(
            "lowpass", config.lowpass_hz, config.sample_rate
        ),
        "telephone band 300-3400 Hz": dsp.butter_sos(
            "bandpass", config.telephone_band_hz, config.sample_rate
        ),
    }
    for label, sos in designs.items():
        worN, response = scipy_signal.sosfreqz(sos, worN=4096, fs=config.sample_rate)
        axis.semilogx(
            worN[1:], 20 * np.log10(np.abs(response[1:]) + 1e-12), label=label
        )
    axis.set_ylim(-80, 5)
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel("magnitude (dB)")
    axis.set_title("Butterworth designs (single-pass response)")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(config.report_root / "filter-response.png", dpi=150)
    plt.close(figure)


def _series_by_condition(rows, metric, model="cnn", protocol="unseen"):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["model"] != model or row["test_snr_db"] is None:
            continue
        if row.get("protocol", "unseen") != protocol:
            continue
        grouped[row["condition"]][row["test_snr_db"]].append(row[metric])
    return grouped


def _clean_ceiling(rows, metric, model="cnn", protocol="unseen") -> float | None:
    """Mean score of the clean-trained arm, which has no test SNR axis."""
    values = [
        row[metric]
        for row in rows
        if row["condition"] == "clean"
        and row["model"] == model
        and row["test_snr_db"] is None
        and row.get("protocol", "unseen") == protocol
    ]
    return float(np.mean(values)) if values else None


def plot_protocol_comparison(config: ExperimentConfig, rows) -> None:
    """Seen versus unseen speakers under an identical enrolment protocol.

    The gap is the share of accuracy that comes from having fitted those
    speakers during training rather than from a transferable representation.
    """
    protocols = {row.get("protocol", "unseen") for row in rows}
    if "seen" not in protocols:
        return

    figure, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=True)
    for axis, condition in zip(axes, ("A_raw_noisy", "B_full")):
        for protocol, style in (("seen", "-o"), ("unseen", "--s")):
            grouped = _series_by_condition(rows, "accuracy", protocol=protocol)
            if condition not in grouped:
                continue
            snrs = sorted(grouped[condition])
            axis.plot(
                snrs,
                [float(np.mean(grouped[condition][snr])) for snr in snrs],
                style,
                label=f"{protocol} speakers",
                markersize=4,
            )
        axis.set_title(condition, fontsize=9)
        axis.set_xlabel("test SNR (dB)")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("identification accuracy")
    axes[0].legend(fontsize=8)
    figure.suptitle("Closed set (training speakers) versus open set (unseen speakers)")
    figure.tight_layout()
    figure.savefig(config.report_root / "protocol-comparison.png", dpi=150)
    plt.close(figure)


def plot_snr_curves(config: ExperimentConfig, rows) -> None:
    figure, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharex=True)
    for axis, metric, title in (
        (axes[0], "accuracy", "Identification accuracy"),
        (axes[1], "agglomerative_ari", "Clustering ARI"),
    ):
        for condition, by_snr in sorted(_series_by_condition(rows, metric).items()):
            snrs = sorted(by_snr)
            means = [float(np.mean(by_snr[snr])) for snr in snrs]
            axis.plot(
                snrs, means, marker="o", label=condition, linewidth=1.2, markersize=4
            )
        ceiling = _clean_ceiling(rows, metric)
        if ceiling is not None:
            axis.axhline(
                ceiling, color="k", linestyle=":", linewidth=1.2, label="clean ceiling"
            )
        axis.set_xlabel("test SNR (dB)")
        axis.set_title(title)
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("score")
    axes[1].legend(fontsize=7, loc="lower right")
    figure.suptitle("Speaker embedding quality versus test SNR")
    figure.tight_layout()
    figure.savefig(config.report_root / "snr-curves.png", dpi=150)
    plt.close(figure)


def plot_ablation(config: ExperimentConfig, rows) -> None:
    grouped = _series_by_condition(rows, "f1_macro")
    conditions = [c.name for c in config.conditions if c.name in grouped]
    means = [
        float(np.mean([value for values in grouped[c].values() for value in values]))
        for c in conditions
    ]

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(np.arange(len(conditions)), means, color="tab:green")
    ceiling = _clean_ceiling(rows, "f1_macro")
    if ceiling is not None:
        axis.axhline(
            ceiling, color="k", linestyle=":", linewidth=1.2, label="clean ceiling"
        )
        axis.legend(fontsize=8)
    axis.set_xticks(np.arange(len(conditions)))
    axis.set_xticklabels(conditions, rotation=30, ha="right", fontsize=8)
    axis.set_ylabel("macro F1 (mean over seeds and SNRs)")
    axis.set_title("DSP stage ablation")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(config.report_root / "ablation-f1.png", dpi=150)
    plt.close(figure)


def plot_confusion_matrices(config: ExperimentConfig) -> None:
    seed = config.seeds[0]
    hardest_snr = int(min(config.test_snr_db))
    targets = [("A_raw_noisy", "Pipeline A"), ("B_full", "Pipeline B")]
    figure, axes = plt.subplots(1, len(targets), figsize=(10, 4.5))
    for axis, (condition, title) in zip(np.atleast_1d(axes), targets):
        path = config.run_root(seed, condition) / f"evaluation-snr-{hardest_snr}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        matrix = np.array(payload["nearest_centroid"]["confusion_matrix"], dtype=float)
        matrix /= matrix.sum(axis=1, keepdims=True) + 1e-12
        axis.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
        accuracy = payload["nearest_centroid"]["accuracy"]
        axis.set_title(f"{title} at {hardest_snr} dB (acc {accuracy:.3f})", fontsize=9)
        axis.set_xlabel("predicted speaker")
    np.atleast_1d(axes)[0].set_ylabel("true speaker")
    figure.suptitle("Row-normalised confusion matrices, unseen test speakers")
    figure.tight_layout()
    figure.savefig(config.report_root / "confusion-matrices.png", dpi=150)
    plt.close(figure)


def plot_embedding_projection(config: ExperimentConfig) -> None:
    """PCA projection of test embeddings, coloured by true speaker."""
    from sklearn.decomposition import PCA

    from src.corpus import load_manifest as _load_manifest

    seed = config.seeds[0]
    panels = [
        ("clean", "clean", "Clean"),
        ("A_raw_noisy", f"snr-{int(config.test_snr_db[0])}", "Pipeline A"),
        ("B_full", f"snr-{int(config.test_snr_db[0])}", "Pipeline B"),
    ]
    available = [
        (condition, title, config.run_root(seed, condition) / f"embeddings-{tag}.npy")
        for condition, tag, title in panels
        if (config.run_root(seed, condition) / f"embeddings-{tag}.npy").is_file()
    ]
    if not available:
        return

    records = _load_manifest(config.cache_root, "test")
    speakers = sorted({record.speaker_id for record in records})
    colours = np.array([speakers.index(record.speaker_id) for record in records])

    figure, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 4))
    for axis, (_, title, path) in zip(np.atleast_1d(axes), available):
        projected = PCA(n_components=2, random_state=0).fit_transform(np.load(path))
        axis.scatter(
            projected[:, 0], projected[:, 1], c=colours, cmap="tab20", s=4, alpha=0.7
        )
        axis.set_title(title, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"PCA of test-speaker embeddings at {int(config.test_snr_db[0])} dB, coloured by speaker"
    )
    figure.tight_layout()
    figure.savefig(config.report_root / "embedding-projection.png", dpi=150)
    plt.close(figure)


def plot_dsp_effect(config: ExperimentConfig) -> None:
    """Noise suppression against speech distortion, the mechanism behind the results."""
    report_path = config.report_root / "dsp-effect.json"
    if not report_path.is_file():
        return
    measurements = json.loads(report_path.read_text(encoding="utf-8"))["measurements"]

    by_condition = defaultdict(dict)
    distortion = {}
    for row in measurements:
        by_condition[row["condition"]][row["input_snr_db"]] = row["snr_gain_db"]
        distortion[row["condition"]] = row["clean_path_snr_db"]

    figure, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    for condition, gains in sorted(by_condition.items()):
        snrs = sorted(gains)
        axes[0].plot(
            snrs, [gains[s] for s in snrs], marker="o", markersize=4, label=condition
        )
    axes[0].axhline(0, color="k", linewidth=0.8)
    axes[0].set_xlabel("input SNR (dB)")
    axes[0].set_ylabel("SNR gain over Pipeline A (dB)")
    axes[0].set_title("Noise suppression achieved")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    names = sorted(distortion, key=lambda n: distortion[n])
    axes[1].barh(np.arange(len(names)), [distortion[n] for n in names], color="tab:red")
    axes[1].set_yticks(np.arange(len(names)))
    axes[1].set_yticklabels(names, fontsize=7)
    axes[1].set_xlabel("SNR of clean input after the chain (dB)")
    axes[1].set_title(
        "Distortion injected into clean speech\n(higher is more transparent)"
    )
    axes[1].grid(axis="x", alpha=0.3)

    figure.tight_layout()
    figure.savefig(config.report_root / "dsp-effect.png", dpi=150)
    plt.close(figure)


def plot_generalisation_matrix(config: ExperimentConfig) -> None:
    """Front-end benefit as a function of the train/inference mismatch.

    One panel per encoder. The shaded band marks the SNR range the encoders were
    trained on; outside it the inference condition is out of distribution, which
    is where a front-end is supposed to earn its cost.
    """
    report_path = config.report_root / "cross-eval.json"
    if not report_path.is_file():
        return
    rows = json.loads(report_path.read_text(encoding="utf-8"))
    if not rows:
        return

    encoders = [
        e for e in config.cross_eval_encoders if any(r["encoder"] == e for r in rows)
    ]
    figure, axes = plt.subplots(
        1, len(encoders), figsize=(4.2 * len(encoders), 4.2), sharey=True
    )
    train_low, train_high = min(config.train_snr_db), max(config.train_snr_db)

    for axis, encoder in zip(np.atleast_1d(axes), encoders):
        for frontend in config.cross_eval_frontends:
            grouped = defaultdict(list)
            for row in rows:
                if row["encoder"] == encoder and row["frontend"] == frontend:
                    grouped[row["test_snr_db"]].append(row["accuracy"])
            if not grouped:
                continue
            snrs = sorted(grouped)
            label = "no DSP" if frontend == "A_raw_noisy" else "full DSP"
            axis.plot(
                snrs,
                [float(np.mean(grouped[s])) for s in snrs],
                marker="o",
                markersize=4,
                label=label,
            )
        axis.axvspan(train_low, train_high, color="tab:grey", alpha=0.15)
        trained = "trained on clean" if encoder == "clean" else f"trained on {encoder}"
        axis.set_title(trained, fontsize=9)
        axis.set_xlabel("test SNR (dB)")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)

    np.atleast_1d(axes)[0].set_ylabel("identification accuracy")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    figure.suptitle(
        "Does the DSP front-end help? Shaded band = SNR range seen during training"
    )
    figure.tight_layout()
    figure.savefig(config.report_root / "generalisation-matrix.png", dpi=150)
    plt.close(figure)


def render_all(config: ExperimentConfig, rows) -> None:
    config.report_root.mkdir(parents=True, exist_ok=True)
    plot_band_analysis(config)
    plot_filter_response(config)
    plot_dsp_effect(config)
    plot_waveforms(config)
    plot_spectrograms(config)
    plot_snr_curves(config, rows)
    plot_ablation(config, rows)
    plot_protocol_comparison(config, rows)
    plot_confusion_matrices(config)
    plot_embedding_projection(config)
    plot_generalisation_matrix(config)
