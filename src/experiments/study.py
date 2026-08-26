"""Study-level orchestration used by the canonical run.py command line."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, sosfreqz

from src.audio import mfcc_features, mfcc_summary_embedding, stft_magnitude, welch_psd
from src.config.settings import PROJECT_ROOT, ExperimentConfig, save_config
from src.data import VCTKWaveformDataset, load_manifest
from src.experiments.evaluate import (
    build_verification_pairs,
    paired_stratified_bootstrap,
    score_pairs,
    verification_metrics,
)
from src.experiments.train import (
    build_waveform_transform,
    evaluate_checkpoint,
    prepare_manifests,
    train_condition,
)

CONDITIONS = (
    "clean_reference",
    "raw_noisy",
    "high_pass",
    "high_pass_low_pass",
    "full_dsp",
)


def write_run_metadata(config: ExperimentConfig, command: str) -> Path:
    target = (
        config.output_root
        / config.study_id
        / "metadata"
        / f"seed-{config.seed}-{command}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    packages = {}
    for package in (
        "torch",
        "torchaudio",
        "speechbrain",
        "scipy",
        "scikit-learn",
        "hdbscan",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    payload = {
        "command": command,
        "config": config.to_dict(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "packages": packages,
        "source_revision": _git_revision(),
    }
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    save_config(config, target.with_name(target.stem + "-config.json"))
    return target


def prepare_study(config: ExperimentConfig) -> dict[str, dict[str, Path]]:
    result = {}
    for seed in config.experimental_seeds:
        seeded = config.with_seed(seed)
        result[str(seed)] = prepare_manifests(seeded)
        write_run_metadata(seeded, "prepare")
    return result


def tune_raw_noisy(config: ExperimentConfig) -> dict[str, Any]:
    """One protocol-mandated grid search; later models consume the locked outcome."""
    seed = config.experimental_seeds[0]
    base = config.with_seed(seed)
    if not (base.manifest_root / "train.jsonl").exists():
        prepare_manifests(base)
    trials = []
    for learning_rate in (3e-5, 1e-4):
        for freeze_encoder_epochs in (0, 1):
            trial = replace(
                base,
                condition="raw_noisy",
                learning_rate=learning_rate,
                freeze_encoder_epochs=freeze_encoder_epochs,
            )
            result = train_condition(trial)
            trials.append(
                {
                    "learning_rate": learning_rate,
                    "freeze_encoder_epochs": freeze_encoder_epochs,
                    "validation_eer": result.best_validation_eer,
                    "checkpoint": str(result.checkpoint_path),
                }
            )
    winner = min(trials, key=lambda trial: trial["validation_eer"])
    output = {
        "selection_metric": "minimum validation EER",
        "seed": seed,
        "trials": trials,
        "locked_hyperparameters": {
            "learning_rate": winner["learning_rate"],
            "freeze_encoder_epochs": winner["freeze_encoder_epochs"],
        },
    }
    path = config.output_root / config.study_id / "tuning.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def train_study(config: ExperimentConfig) -> list[Path]:
    locked = _locked_hyperparameters(config)
    checkpoints = []
    for seed in config.experimental_seeds:
        for condition in CONDITIONS:
            trial = replace(config.with_seed(seed), condition=condition, **locked)
            if not (trial.manifest_root / "train.jsonl").exists():
                prepare_manifests(trial)
            checkpoints.append(train_condition(trial).checkpoint_path)
            write_run_metadata(trial, "train")
    return checkpoints


def evaluate_study(config: ExperimentConfig) -> list[Path]:
    reports: list[Path] = []
    for seed in config.experimental_seeds:
        for condition in CONDITIONS:
            trial = replace(
                config.with_seed(seed),
                condition=condition,
                **_locked_hyperparameters(config),
            )
            checkpoint = trial.run_root / "best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Checkpoint missing: {checkpoint}; run train first"
                )
            for snr in trial.snr_db:
                reports.append(
                    evaluate_checkpoint(trial, checkpoint, test_snr_db=snr).report_path
                )
            if condition == "clean_reference":
                reports.append(
                    evaluate_checkpoint(
                        trial, checkpoint, clean_reference=True
                    ).report_path
                )
            reports.append(_evaluate_mfcc_reference(trial))
            write_run_metadata(trial, "evaluate")
    return reports


def analyze_study(config: ExperimentConfig) -> Path:
    root = config.output_root / config.study_id
    reports = sorted(root.glob("seed-*/*/evaluation-*.json"))
    rows = []
    for report in reports:
        payload = json.loads(report.read_text(encoding="utf-8"))
        metrics = payload["verification"]
        rows.append(
            {
                "seed": report.parts[-3],
                "condition": payload["condition"],
                "protocol": payload["protocol"],
                **metrics,
            }
        )
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = report_dir / "metric-summary.json"
    summary.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv = report_dir / "metric-summary.csv"
    if rows:
        columns = list(rows[0])
        csv.write_text(
            "\n".join(
                [
                    ",".join(columns),
                    *[
                        ",".join(str(row.get(column, "")) for column in columns)
                        for row in rows
                    ],
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        _plot_snr_curves(rows, report_dir / "snr-curves.png")
        _plot_ablation(rows, report_dir / "ablation-f1.png")
        _plot_confusion_matrices(rows, report_dir / "confusion-matrices.png")
    _save_signal_figures(config, report_dir)
    _write_bootstrap_intervals(reports, report_dir / "paired-bootstrap-ci.json")
    return summary


def package_study(config: ExperimentConfig, *, dry_run: bool = False) -> Path:
    """Stage a portable ignored release bundle with a SHA-256 manifest; never includes raw data."""
    destination = PROJECT_ROOT / "release" / config.study_id
    required = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "uv.lock",
        PROJECT_ROOT / "pyproject.toml",
        config.output_root / config.study_id,
    ]
    if dry_run:
        return destination
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for source in required:
        if not source.exists():
            continue
        target = destination / source.relative_to(PROJECT_ROOT)
        if source.is_dir():
            shutil.copytree(
                source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
        else:
            shutil.copy2(source, target)
    _export_requirements(destination / "requirements.txt")
    (destination / "DATASETS.md").write_text(
        "Raw VCTK and MUSAN data are intentionally excluded. Recreate them with: uv run python run.py download-data --accept-data-licenses\n",
        encoding="utf-8",
    )
    _copy_if_exists(PROJECT_ROOT / "README.md", destination / "README.md")
    _copy_if_exists(
        config.ecapa_cache, destination / "pretrained_models" / config.ecapa_cache.name
    )
    manifest = {
        str(path.relative_to(destination)): _sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    (destination / "SHA256SUMS.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def _evaluate_mfcc_reference(config: ExperimentConfig) -> Path:
    validation = load_manifest(config.manifest_root / "validation.jsonl")
    test = load_manifest(config.manifest_root / "test.jsonl")

    def embeddings(records, snr):
        dataset = VCTKWaveformDataset(
            records,
            sample_rate=config.sample_rate,
            segment_seconds=config.segment_seconds,
            waveform_transform=build_waveform_transform(config, fixed_snr_db=snr),
            dataset_root=PROJECT_ROOT,
        )
        return {
            item["sample_id"]: mfcc_summary_embedding(
                item["waveform"],
                sample_rate=config.sample_rate,
                n_mfcc=config.mfcc_coefficients,
                n_mels=config.mel_bins,
            )
            for item in dataset
        }

    validation_embeddings = embeddings(validation, None)
    validation_pairs = build_verification_pairs(
        validation,
        seed=config.seed,
        positive_pairs_per_speaker=config.positive_pairs_per_speaker,
    )
    validation_scores, labels = score_pairs(validation_pairs, validation_embeddings)
    threshold = float(verification_metrics(validation_scores, labels)["threshold"])
    output = []
    for snr in config.snr_db:
        test_embeddings = embeddings(test, snr)
        pairs = build_verification_pairs(
            test,
            seed=config.seed,
            positive_pairs_per_speaker=config.positive_pairs_per_speaker,
        )
        scores, labels = score_pairs(pairs, test_embeddings)
        output.append(
            {
                "snr_db": snr,
                "verification": verification_metrics(
                    scores, labels, threshold=threshold
                ),
                "pair_ids": [pair.pair_id for pair in pairs],
            }
        )
    path = config.run_root / "evaluation-mfcc-cosine.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "condition": config.condition,
                "reference": "MFCC mean/std cosine",
                "validation_threshold": threshold,
                "results": output,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _locked_hyperparameters(config: ExperimentConfig) -> dict[str, Any]:
    path = config.output_root / config.study_id / "tuning.json"
    if not path.exists():
        raise FileNotFoundError(
            "Tuning result missing; run `tune` before `train` or `evaluate`."
        )
    return json.loads(path.read_text(encoding="utf-8"))["locked_hyperparameters"]


def _plot_snr_curves(rows: list[dict[str, Any]], destination: Path) -> None:
    selected = [row for row in rows if row["protocol"].startswith("test-snr-")]
    if not selected:
        return
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(row["condition"], []).append(row)
    plt.figure(figsize=(8, 4.5))
    for condition, values in grouped.items():
        by_snr: dict[int, list[float]] = {}
        for value in values:
            by_snr.setdefault(int(value["protocol"].rsplit("-", 1)[-1]), []).append(
                float(value["eer"])
            )
        snrs = sorted(by_snr)
        plt.plot(
            snrs, [np.mean(by_snr[snr]) for snr in snrs], marker="o", label=condition
        )
    plt.xlabel("Test SNR (dB)")
    plt.ylabel("EER")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=160)
    plt.close()


def _plot_ablation(rows: list[dict[str, Any]], destination: Path) -> None:
    values: dict[str, list[float]] = {}
    for row in rows:
        if row["protocol"].startswith("test-snr-"):
            values.setdefault(row["condition"], []).append(float(row["f1"]))
    if not values:
        return
    labels = list(values)
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, [np.mean(values[label]) for label in labels])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean F1 across test SNRs/seeds")
    plt.tight_layout()
    plt.savefig(destination, dpi=160)
    plt.close()


def _plot_confusion_matrices(rows: list[dict[str, Any]], destination: Path) -> None:
    chosen = [row for row in rows if row["protocol"] == "test-snr-5"]
    if not chosen:
        return
    labels = sorted({row["condition"] for row in chosen})
    figure, axes = plt.subplots(
        1, len(labels), figsize=(3 * len(labels), 3), squeeze=False
    )
    for axis, label in zip(axes[0], labels, strict=True):
        subset = [row for row in chosen if row["condition"] == label]
        matrix = np.asarray(
            [
                [
                    sum(int(row["tn"]) for row in subset),
                    sum(int(row["fp"]) for row in subset),
                ],
                [
                    sum(int(row["fn"]) for row in subset),
                    sum(int(row["tp"]) for row in subset),
                ],
            ]
        )
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(label)
        axis.set_xticks([0, 1], ["different", "same"])
        axis.set_yticks([0, 1], ["different", "same"])
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, str(value), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _save_signal_figures(config: ExperimentConfig, report_dir: Path) -> None:
    """Save waveform/PSD/STFT/MFCC/filter-response views for one reproducible test clip."""
    manifest = (
        config.with_seed(config.experimental_seeds[0]).manifest_root / "test.jsonl"
    )
    if not manifest.exists():
        return
    record = load_manifest(manifest)[0]
    clean = VCTKWaveformDataset(
        [record],
        sample_rate=config.sample_rate,
        segment_seconds=config.segment_seconds,
        dataset_root=PROJECT_ROOT,
    )[0]["waveform"]
    if not isinstance(clean, torch.Tensor):
        return
    waveforms = {"clean": clean}
    for condition in ("raw_noisy", "high_pass", "high_pass_low_pass", "full_dsp"):
        variant = replace(
            config.with_seed(config.experimental_seeds[0]), condition=condition
        )
        processed = VCTKWaveformDataset(
            [record],
            sample_rate=variant.sample_rate,
            segment_seconds=variant.segment_seconds,
            waveform_transform=build_waveform_transform(variant, fixed_snr_db=5),
            dataset_root=PROJECT_ROOT,
        )[0]["waveform"]
        if isinstance(processed, torch.Tensor):
            waveforms[condition] = processed
    time = np.arange(clean.numel()) / config.sample_rate
    figure, axis = plt.subplots(figsize=(9, 3))
    for name, waveform in waveforms.items():
        axis.plot(time, waveform.detach().cpu(), label=name, alpha=0.75)
    axis.set(xlabel="Time (s)", ylabel="Amplitude")
    axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(report_dir / "waveforms.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for name, waveform in waveforms.items():
        frequency, power = welch_psd(waveform, sample_rate=config.sample_rate)
        axes[0].semilogy(frequency, power, label=name)
    axes[0].set(xlabel="Hz", ylabel="PSD")
    axes[0].legend(fontsize=7)
    image = axes[1].imshow(
        20 * torch.log10(stft_magnitude(waveforms["full_dsp"]).clamp_min(1e-8)).cpu(),
        origin="lower",
        aspect="auto",
    )
    axes[1].set(title="Full-DSP STFT", xlabel="Frame", ylabel="Bin")
    figure.colorbar(image, ax=axes[1])
    figure.tight_layout()
    figure.savefig(report_dir / "psd-spectrogram.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].imshow(
        mfcc_features(waveforms["raw_noisy"], sample_rate=config.sample_rate).cpu(),
        origin="lower",
        aspect="auto",
    )
    axes[0].set(title="Raw-noisy MFCC", ylabel="Coefficient")
    axes[1].imshow(
        mfcc_features(waveforms["full_dsp"], sample_rate=config.sample_rate).cpu(),
        origin="lower",
        aspect="auto",
    )
    axes[1].set(title="Full-DSP MFCC", ylabel="Coefficient")
    figure.tight_layout()
    figure.savefig(report_dir / "mfcc.png", dpi=160)
    plt.close(figure)
    sos_high = butter(
        config.filter_order,
        config.high_pass_hz / config.nyquist_hz,
        btype="highpass",
        output="sos",
    )
    sos_low = butter(
        config.filter_order,
        config.low_pass_hz / config.nyquist_hz,
        btype="lowpass",
        output="sos",
    )
    figure, axis = plt.subplots(figsize=(8, 3))
    for name, sos in (("80 Hz high-pass", sos_high), ("7.5 kHz low-pass", sos_low)):
        frequency, response = sosfreqz(sos, fs=config.sample_rate)
        axis.plot(
            frequency, 20 * np.log10(np.maximum(np.abs(response), 1e-8)), label=name
        )
    axis.set(xlabel="Hz", ylabel="dB", ylim=(-80, 5))
    axis.legend()
    figure.tight_layout()
    figure.savefig(report_dir / "filter-response.png", dpi=160)
    plt.close(figure)


def _write_bootstrap_intervals(reports: list[Path], destination: Path) -> None:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for report in reports:
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload["condition"] not in {"raw_noisy", "full_dsp"} or not payload[
            "protocol"
        ].startswith("test-snr-"):
            continue
        indexed.setdefault((report.parts[-3], payload["protocol"]), {})[
            payload["condition"]
        ] = payload
    intervals = []
    for (seed, protocol), entry in indexed.items():
        if set(entry) != {"raw_noisy", "full_dsp"}:
            continue
        raw, dsp = entry["raw_noisy"], entry["full_dsp"]
        if raw["pair_ids"] != dsp["pair_ids"] or raw["labels"] != dsp["labels"]:
            raise ValueError("Paired comparison requires identical pair IDs and labels")
        labels = np.asarray(raw["labels"])
        intervals.append(
            {
                "seed": seed,
                "protocol": protocol,
                **paired_stratified_bootstrap(
                    np.asarray(raw["scores"]),
                    np.asarray(dsp["scores"]),
                    labels,
                    metric="f1",
                    threshold_raw=float(raw["verification"]["threshold"]),
                    threshold_dsp=float(dsp["verification"]["threshold"]),
                    seed=int(seed.removeprefix("seed-")),
                ),
            }
        )
    destination.write_text(
        json.dumps(intervals, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _export_requirements(destination: Path) -> None:
    completed = subprocess.run(
        ["uv", "export", "--frozen", "--no-hashes", "--output-file", str(destination)],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if completed.returncode:
        shutil.copy2(PROJECT_ROOT / "pyproject.toml", destination)


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    elif source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
