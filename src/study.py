"""Orchestration and speaker-clustered inference for one frozen robust CNN."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import torch

from src.config import FACTORIAL_CELLS, ExperimentConfig
from src.corpus import load_clips, load_manifest
from src.eval import (
    classification_metrics,
    clustering_metrics,
    confusable_speakers,
    enrollment_split,
    error_cases,
    extract_embeddings,
    nearest_centroid_predict,
    speaker_labels,
)
from src.features import residual_snr_db
from src.pipeline import process_split
from src.train import TrainResult, load_encoder, resolve_device, train_robust_cnn


def train_study(
    config: ExperimentConfig,
    *,
    workers: int = 0,
    only: str | None = None,
    missing_only: bool = False,
    learning_rate: float | None = None,
    arcface_margin: float | None = None,
) -> list[TrainResult]:
    if only is not None and only != "robust_cnn":
        raise ValueError("there is one training recipe only: robust_cnn")
    seeds = [
        seed
        for seed in config.seeds
        if not missing_only or not (config.seed_dir(seed) / "encoder.pt").is_file()
    ]
    results = [
        train_robust_cnn(
            config,
            seed,
            workers=workers,
            learning_rate=learning_rate,
            arcface_margin=arcface_margin,
        )
        for seed in seeds
    ]
    config.run_dir.mkdir(parents=True, exist_ok=True)
    summary = [asdict(result) for result in results]
    (config.run_dir / "training-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return results


def tune_hyperparameters(
    config: ExperimentConfig, *, workers: int = 0
) -> dict[str, object]:
    """Grid-search lr x ArcFace margin on the speaker-disjoint validation split only.

    Trials are shortened to `tune_epochs` and use the first seed; the test split is
    never touched, so selection cannot leak into the reported comparison.
    """
    seed = config.seeds[0]
    trials = []
    for learning_rate in config.tune_learning_rates:
        for margin in config.tune_arcface_margins:
            root = config.run_dir / "tuning" / f"lr-{learning_rate:g}-margin-{margin:g}"
            result = train_robust_cnn(
                config,
                seed,
                workers=workers,
                learning_rate=learning_rate,
                arcface_margin=margin,
                epochs=config.tune_epochs,
                root=root,
                audit=False,
            )
            trials.append(asdict(result) | {"trial_dir": str(root)})
            print(
                f"  tune lr {learning_rate:<8g} margin {margin:<5g} validation {result.best_validation_accuracy:.4f}",
                flush=True,
            )
    best = max(
        trials,
        key=lambda x: (
            x["best_validation_accuracy"],
            -x["learning_rate"],
            -x["arcface_margin"],
        ),
    )
    report = {
        "search_space": {
            "learning_rate": list(config.tune_learning_rates),
            "arcface_margin": list(config.tune_arcface_margins),
        },
        "protocol": {
            "seed": seed,
            "epochs_per_trial": config.tune_epochs,
            "final_epochs": config.epochs,
            "selection_metric": "nearest-centroid identification accuracy on the validation speakers",
            "split": "validation",
        },
        "trials": trials,
        "selected": {
            "learning_rate": best["learning_rate"],
            "arcface_margin": best["arcface_margin"],
            "validation_accuracy": best["best_validation_accuracy"],
        },
    }
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.tuning_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def selected_hyperparameters(config: ExperimentConfig) -> dict[str, float]:
    if not config.tuning_path.is_file():
        raise FileNotFoundError(
            f"run the tune phase first; {config.tuning_path} is missing"
        )
    return json.loads(config.tuning_path.read_text(encoding="utf-8"))["selected"]


def _evaluation_report(
    embeddings: np.ndarray, records, processed, config: ExperimentConfig, seed: int
) -> dict[str, object]:
    labels, names = speaker_labels(records)
    enrol, query = enrollment_split(records, config.enrollment_clips)
    truth, prediction, scores = nearest_centroid_predict(
        embeddings, labels, enrol, query
    )
    base = {
        "n_speakers": len(names),
        "speakers": names,
        "nearest_centroid": classification_metrics(truth, prediction),
        "clustering": clustering_metrics(embeddings, labels, config.cluster_min_size),
        "errors": error_cases(truth, prediction, scores, query, records, names),
        "confusable_speakers": confusable_speakers(truth, prediction, names),
    }
    queries, grouped = [], defaultdict(list)
    for position, (actual, predicted, clip_index) in enumerate(
        zip(truth, prediction, query, strict=True)
    ):
        record, mix = records[int(clip_index)], processed.mixing[int(clip_index)]
        item = {
            "sample_id": record.sample_id,
            "speaker_id": record.speaker_id,
            "true_speaker": names[int(actual)],
            "predicted_speaker": names[int(predicted)],
            "correct": bool(actual == predicted),
            "score": float(scores[position].max()),
            "mix": mix,
            "waveform_snr_db": None
            if mix["family"] == "clean"
            else residual_snr_db(
                processed.clean_components[int(clip_index)],
                processed.waveforms[int(clip_index)],
            ),
        }
        queries.append(item)
        grouped[record.speaker_id].append(item)
    base["query_predictions"] = queries
    base["per_speaker"] = [
        {
            "speaker_id": speaker,
            "n_queries": len(items),
            "accuracy": float(np.mean([x["correct"] for x in items])),
            "mean_waveform_snr_db": None
            if items[0]["waveform_snr_db"] is None
            else float(np.mean([x["waveform_snr_db"] for x in items])),
        }
        for speaker, items in sorted(grouped.items())
    ]
    return base


def evaluate_study(
    config: ExperimentConfig,
    *,
    workers: int = 0,
    only: str | None = None,
    cells: tuple[str, ...] | None = None,
) -> list[str]:
    if only is not None and only not in FACTORIAL_CELLS:
        raise ValueError(f"unknown factorial cell {only!r}")
    if only is not None and cells is not None:
        raise ValueError("pass either `only` or `cells`, not both")
    selected = (only,) if only else (cells if cells is not None else FACTORIAL_CELLS)
    if not selected or set(selected) - set(FACTORIAL_CELLS):
        raise ValueError("evaluation cells must be non-empty factorial cells")
    device = resolve_device()
    conditions = tuple(config.condition(name) for name in selected)
    written: list[str] = []
    for seed in config.seeds:
        root = config.seed_dir(seed)
        checkpoint = root / "encoder.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing robust_cnn checkpoint for seed {seed}: {checkpoint}"
            )
        model = load_encoder(config, checkpoint, device)
        for split, protocol in (("test", "unseen"), ("seen_test", "seen")):
            if split == "seen_test" and not config.closed_set_clips:
                continue
            clips, records = (
                load_clips(config.cache_root, split),
                load_manifest(config.cache_root, split),
            )
            for family, snr in [
                (None, None),
                *(
                    (family, snr)
                    for family in config.test_noise_families
                    for snr in config.test_snr_db
                ),
            ]:
                label = "clean" if family is None else f"{family}-snr-{snr:g}"
                for condition in conditions:
                    processed = process_split(
                        clips,
                        records,
                        condition,
                        config,
                        seed=seed,
                        family=family,
                        snr_db=snr,
                    )
                    embeddings = extract_embeddings(model, processed.waveforms, device)
                    report = {
                        "recipe": "robust_cnn",
                        "seed": seed,
                        "cell": condition.name,
                        "stages": list(condition.stages),
                        "protocol": protocol,
                        "audio_domain": "clean" if family is None else "noisy",
                        "noise_family": family,
                        "test_snr_db": snr,
                        **_evaluation_report(
                            embeddings, records, processed, config, seed
                        ),
                    }
                    suffix = "" if protocol == "unseen" else "-seen"
                    output = root / f"evaluation-{label}-{condition.name}{suffix}.json"
                    output.write_text(
                        json.dumps(report, indent=2) + "\n", encoding="utf-8"
                    )
                    written.append(str(output))
                    print(
                        f"  seed {seed} {protocol:<6} {label:<17} {condition.name:<16} acc {report['nearest_centroid']['accuracy']:.4f}",
                        flush=True,
                    )
        del model
        torch.cuda.empty_cache()
    return written


def collect_metrics(config: ExperimentConfig) -> list[dict[str, object]]:
    rows = []
    for path in sorted(config.run_dir.glob("seed-*/robust_cnn/evaluation-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        score, cluster = payload["nearest_centroid"], payload["clustering"]
        rows.append(
            {
                "seed": payload["seed"],
                "cell": payload["cell"],
                "protocol": payload["protocol"],
                "audio_domain": payload["audio_domain"],
                "noise_family": payload["noise_family"],
                "test_snr_db": payload["test_snr_db"],
                "accuracy": score["accuracy"],
                "f1_macro": score["f1_macro"],
                "agglomerative_ari": cluster["agglomerative_ari"],
                "agglomerative_nmi": cluster["agglomerative_nmi"],
                "path": str(path),
            }
        )
    return rows


def evaluation_summary(
    config: ExperimentConfig, cells: tuple[str, ...], label: str
) -> dict[str, object]:
    """Persist a compact phase-local table plus representative error samples."""
    rows = [row for row in collect_metrics(config) if row["cell"] in cells]
    config.report_root.mkdir(parents=True, exist_ok=True)
    table_path = config.report_root / f"{label}-metric-summary.csv"
    if rows:
        with table_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    errors = []
    for path in sorted(config.run_dir.glob("seed-*/robust_cnn/evaluation-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["cell"] not in cells:
            continue
        errors.extend(
            {"report": str(path), **item}
            for item in payload["query_predictions"]
            if not item["correct"]
        )
    error_path = config.report_root / f"{label}-error-samples.json"
    error_path.write_text(json.dumps(errors[:100], indent=2) + "\n", encoding="utf-8")
    return {
        "rows": rows,
        "table_path": str(table_path),
        "error_samples": errors[:20],
        "error_path": str(error_path),
    }


def paired_cluster_bootstrap(
    observations: list[dict[str, object]], *, replicates: int, seed: int
) -> dict[str, object]:
    """Resample speakers; a draw retains every seed/SNR/family/query for that speaker."""
    by_speaker: dict[str, list[float]] = defaultdict(list)
    for row in observations:
        by_speaker[str(row["speaker_id"])].append(float(row["delta"]))
    speakers = sorted(by_speaker)
    if not speakers:
        raise ValueError("no paired speaker observations")
    point = float(
        np.mean([value for values in by_speaker.values() for value in values])
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for i in range(replicates):
        sampled = rng.choice(speakers, size=len(speakers), replace=True)
        draws[i] = np.mean(
            [value for speaker in sampled for value in by_speaker[speaker]]
        )
    return {
        "n_speakers": len(speakers),
        "n_paired_speaker_strata": len(observations),
        "mean_delta": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "bootstrap_replicates": replicates,
        "resampling_unit": "test speaker; all seed, SNR, family, and query observations retained",
    }


def _factorial_observations(
    config: ExperimentConfig, metric: str, protocol: str
) -> dict[str, list[dict[str, object]]]:
    reports = {}
    for path in config.run_dir.glob("seed-*/robust_cnn/evaluation-*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["protocol"] != protocol:
            continue
        key = (
            payload["seed"],
            payload["audio_domain"],
            payload["noise_family"],
            payload["test_snr_db"],
        )
        reports.setdefault(key, {})[payload["cell"]] = payload
    effects = {
        "bandpass_main": [],
        "wiener_main": [],
        "bandpass_x_wiener": [],
        **{f"{cell}_minus_raw": [] for cell in FACTORIAL_CELLS if cell != "raw"},
    }
    for context, cells in reports.items():
        if set(cells) != set(FACTORIAL_CELLS):
            continue
        speakers = {
            row["speaker_id"] for cell in cells.values() for row in cell["per_speaker"]
        }
        indexed = {
            cell: {row["speaker_id"]: row for row in payload["per_speaker"]}
            for cell, payload in cells.items()
        }
        for speaker in speakers:
            if not all(speaker in indexed[cell] for cell in FACTORIAL_CELLS):
                continue
            values = {
                cell: float(indexed[cell][speaker][metric]) for cell in FACTORIAL_CELLS
            }
            common = {
                "speaker_id": speaker,
                "seed": context[0],
                "audio_domain": context[1],
                "noise_family": context[2],
                "test_snr_db": context[3],
            }
            effects["bandpass_main"].append(
                common
                | {
                    "delta": (
                        (values["bandpass"] - values["raw"])
                        + (values["bandpass_wiener"] - values["wiener"])
                    )
                    / 2
                }
            )
            effects["wiener_main"].append(
                common
                | {
                    "delta": (
                        (values["wiener"] - values["raw"])
                        + (values["bandpass_wiener"] - values["bandpass"])
                    )
                    / 2
                }
            )
            effects["bandpass_x_wiener"].append(
                common
                | {
                    "delta": values["bandpass_wiener"]
                    - values["bandpass"]
                    - values["wiener"]
                    + values["raw"]
                }
            )
            for cell in ("bandpass", "wiener", "bandpass_wiener"):
                effects[f"{cell}_minus_raw"].append(
                    common | {"delta": values[cell] - values["raw"]}
                )
    return effects


def statistical_analysis(config: ExperimentConfig) -> dict[str, object]:
    rows = collect_metrics(config)
    if not rows:
        raise FileNotFoundError(
            "No evaluation reports found; run raw and DSP evaluation phases first."
        )
    config.report_root.mkdir(parents=True, exist_ok=True)
    with (config.report_root / "metric-summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    factorial = {}
    for protocol in sorted({str(row["protocol"]) for row in rows}):
        factorial[protocol] = {}
        for metric in ("accuracy",):
            factorial[protocol][metric] = {
                name: paired_cluster_bootstrap(
                    items, replicates=config.bootstrap_replicates, seed=config.seeds[0]
                )
                for name, items in _factorial_observations(
                    config, metric, protocol
                ).items()
                if items
            }
    (config.report_root / "factorial-bootstrap.json").write_text(
        json.dumps(factorial, indent=2) + "\n", encoding="utf-8"
    )
    return {"rows": rows, "factorial": factorial}


def render_figures(config: ExperimentConfig) -> dict[str, object]:
    from src import plots

    rows = collect_metrics(config)
    if not rows:
        raise FileNotFoundError("No evaluation reports found; cannot render figures.")
    plots.render_all(config, rows)
    return {
        "figures": [str(path) for path in sorted(config.report_root.glob("*.png"))],
        "metric_rows": len(rows),
    }


def analyze_study(config: ExperimentConfig) -> dict[str, object]:
    """Compatibility convenience that runs statistics then rendering."""
    result = statistical_analysis(config)
    result.update(render_figures(config))
    return result
