"""Study orchestration: hyperparameter tuning, training, evaluation, analysis."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import torch

from src.config import ExperimentConfig
from src.eval import (
    classification_metrics,
    clustering_metrics,
    confusable_speakers,
    enrollment_split,
    error_cases,
    extract_embeddings,
    nearest_centroid_predict,
    paired_significance,
    speaker_labels,
    svm_predict,
)
from src.features import handcrafted_features
from src.train import (
    TrainResult,
    load_encoder,
    prepared_split,
    resolve_device,
    train_condition,
)

#: Grid searched once on Pipeline A, then locked for every condition and seed.
TUNING_GRID = (
    {"learning_rate": 1e-3, "batch_size": 64},
    {"learning_rate": 3e-4, "batch_size": 64},
    {"learning_rate": 1e-3, "batch_size": 32},
)
TUNING_EPOCHS = 8


def _tuning_path(config: ExperimentConfig):
    return config.study_root / "tuning.json"


def locked_hyperparameters(config: ExperimentConfig) -> dict[str, float | int]:
    path = _tuning_path(config)
    if not path.is_file():
        return {"learning_rate": config.learning_rate, "batch_size": config.batch_size}
    return json.loads(path.read_text(encoding="utf-8"))["selected"]


def tune_hyperparameters(
    config: ExperimentConfig, *, workers: int = 8
) -> dict[str, object]:
    """Grid-search on the Pipeline A baseline with the first seed only."""
    seed = config.seeds[0]
    trials = []
    for candidate in TUNING_GRID:
        result = train_condition(
            config,
            "A_raw_noisy",
            seed,
            workers=workers,
            overrides={**candidate, "epochs": TUNING_EPOCHS},
        )
        trials.append(
            {**candidate, "validation_accuracy": result.best_validation_accuracy}
        )
        print(
            f"  lr={candidate['learning_rate']} bs={candidate['batch_size']} "
            f"-> val-acc {result.best_validation_accuracy:.4f}"
        )

    best = max(trials, key=lambda trial: trial["validation_accuracy"])
    payload = {
        "tuned_on": {"condition": "A_raw_noisy", "seed": seed, "epochs": TUNING_EPOCHS},
        "grid": list(TUNING_GRID),
        "trials": trials,
        "selected": {
            "learning_rate": best["learning_rate"],
            "batch_size": best["batch_size"],
        },
        "optimizer": config.optimizer,
        "weight_decay": config.weight_decay,
        "epochs": config.epochs,
    }
    _tuning_path(config).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _planned_runs(config: ExperimentConfig, only: str | None) -> list[tuple[str, int]]:
    """Primary arms run on every seed; ablation arms run on the first seed only."""
    if only is not None:
        return [(only, seed) for seed in config.seeds]
    runs = []
    for condition in config.conditions:
        seeds = (
            config.seeds
            if condition.name in config.primary_conditions
            else config.seeds[:1]
        )
        runs.extend((condition.name, seed) for seed in seeds)
    return runs


def train_study(
    config: ExperimentConfig, *, workers: int = 8, only: str | None = None
) -> list[TrainResult]:
    overrides = locked_hyperparameters(config)
    results = []
    for condition_name, seed in _planned_runs(config, only):
        results.append(
            train_condition(
                config, condition_name, seed, workers=workers, overrides=overrides
            )
        )
    (config.study_root / "training-summary.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2) + "\n",
        encoding="utf-8",
    )
    return results


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _evaluate_embeddings(
    embeddings: np.ndarray, records, config: ExperimentConfig, seed: int
) -> dict[str, object]:
    labels, names = speaker_labels(records)
    enrol, query = enrollment_split(records, config.enrollment_clips)
    truth, prediction, scores = nearest_centroid_predict(
        embeddings, labels, enrol, query
    )
    svm_truth, svm_prediction = svm_predict(embeddings, labels, enrol, query, seed)

    return {
        "n_speakers": len(names),
        "speakers": names,
        "nearest_centroid": classification_metrics(truth, prediction),
        "svm": classification_metrics(svm_truth, svm_prediction),
        "clustering": clustering_metrics(embeddings, labels, config.cluster_min_size),
        "errors": error_cases(truth, prediction, scores, query, records, names),
        "confusable_speakers": confusable_speakers(truth, prediction, names),
    }


def _mfcc_reference(
    waveforms: np.ndarray, records, config: ExperimentConfig, seed: int
) -> dict[str, object]:
    """Classical DSP-only baseline: handcrafted features, no learned encoder."""
    features = np.stack(
        [handcrafted_features(w, config.sample_rate) for w in waveforms]
    )
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    normalised = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    return _evaluate_embeddings(normalised.astype(np.float32), records, config, seed)


def evaluate_study(
    config: ExperimentConfig, *, workers: int = 8, only: str | None = None
) -> None:
    device = resolve_device()
    for condition_name, seed in _planned_runs(config, only):
        run_root = config.run_root(seed, condition_name)
        checkpoint = run_root / "encoder.pt"
        if not checkpoint.is_file():
            print(f"  skip {condition_name} seed {seed}: no checkpoint")
            continue
        model = load_encoder(config, checkpoint, device)
        condition = config.condition(condition_name)

        # Clean-trained models are evaluated on clean test audio only; noisy arms
        # sweep the test SNR grid.
        snr_grid = (None,) if not condition.add_noise else config.test_snr_db
        for snr_db in snr_grid:
            tag = "clean" if snr_db is None else f"snr-{int(snr_db)}"

            # `test` holds speakers never seen in training (open set); `seen_test`
            # holds unseen utterances of the training speakers (closed set). The
            # enrolment protocol is identical, so the gap between them isolates
            # how much of the score comes from speaker-specific fitting.
            for split, protocol in (("test", "unseen"), ("seen_test", "seen")):
                if split == "seen_test" and not config.closed_set_clips:
                    continue
                waveforms, records = prepared_split(
                    config, split, condition_name, seed, snr_db=snr_db, workers=workers
                )
                embeddings = extract_embeddings(model, waveforms, device)
                report = {
                    "condition": condition_name,
                    "seed": seed,
                    "test_snr_db": snr_db,
                    "protocol": protocol,
                    "stages": list(condition.stages),
                    **_evaluate_embeddings(embeddings, records, config, seed),
                }
                suffix = "" if protocol == "unseen" else "-seen"
                (run_root / f"evaluation-{tag}{suffix}.json").write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                accuracy = report["nearest_centroid"]["accuracy"]
                print(
                    f"  {condition_name:<18} seed {seed} {tag:<8} {protocol:<6} "
                    f"acc {accuracy:.4f}",
                    flush=True,
                )

                if protocol != "unseen":
                    continue

                # Kept for the embedding-geometry figure; only the arms the report
                # contrasts directly, to bound artifact size.
                if condition_name in config.primary_conditions and snr_db in (
                    None,
                    config.test_snr_db[0],
                ):
                    np.save(run_root / f"embeddings-{tag}.npy", embeddings)

                # The DSP-only reference uses the same audio, so it is only computed
                # for the two pipelines being contrasted.
                if condition_name in ("A_raw_noisy", "B_full"):
                    reference = {
                        "condition": condition_name,
                        "seed": seed,
                        "test_snr_db": snr_db,
                        "protocol": protocol,
                        "stages": list(condition.stages),
                        **_mfcc_reference(waveforms, records, config, seed),
                    }
                    (run_root / f"evaluation-{tag}-dsponly.json").write_text(
                        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
                    )
        del model
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Generalisation matrix
# --------------------------------------------------------------------------- #
def cross_evaluate_study(
    config: ExperimentConfig, *, workers: int = 8
) -> list[dict[str, object]]:
    """Evaluate every encoder against every front-end, over an extended SNR grid.

    The main evaluation is matched: an encoder is only ever shown audio produced
    by its own front-end, and the SNR range is the one it trained on. That hides
    the case a front-end is actually for. Here the encoder and the front-end are
    varied independently, and the grid extends below the training SNR range, so
    the results show when a front-end earns its cost at inference time.
    """
    device = resolve_device()
    rows: list[dict[str, object]] = []

    for encoder_condition in config.cross_eval_encoders:
        trained_on_noise = config.condition(encoder_condition).add_noise
        for seed in config.seeds:
            checkpoint = config.run_root(seed, encoder_condition) / "encoder.pt"
            if not checkpoint.is_file():
                print(f"  skip encoder {encoder_condition} seed {seed}: no checkpoint")
                continue
            model = load_encoder(config, checkpoint, device)

            for frontend in config.cross_eval_frontends:
                for snr_db in config.cross_eval_snr_db:
                    waveforms, records = prepared_split(
                        config, "test", frontend, seed, snr_db=snr_db, workers=workers
                    )
                    embeddings = extract_embeddings(model, waveforms, device)
                    labels, names = speaker_labels(records)
                    enrol, query = enrollment_split(records, config.enrollment_clips)
                    truth, prediction, _ = nearest_centroid_predict(
                        embeddings, labels, enrol, query
                    )
                    metrics = classification_metrics(truth, prediction)
                    rows.append(
                        {
                            "encoder": encoder_condition,
                            "encoder_trained_on_noise": trained_on_noise,
                            "frontend": frontend,
                            "frontend_stages": "+".join(
                                config.condition(frontend).stages
                            )
                            or "none",
                            "seed": seed,
                            "test_snr_db": float(snr_db),
                            "within_training_snr": float(snr_db) in config.train_snr_db,
                            "accuracy": metrics["accuracy"],
                            "f1_macro": metrics["f1_macro"],
                            "n_speakers": len(names),
                        }
                    )
                    print(
                        f"  encoder {encoder_condition:<12} frontend {frontend:<12} "
                        f"seed {seed} {snr_db:>5.0f} dB  acc {metrics['accuracy']:.4f}",
                        flush=True,
                    )
            del model
            torch.cuda.empty_cache()

    config.report_root.mkdir(parents=True, exist_ok=True)
    (config.report_root / "cross-eval.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    if rows:
        fieldnames = list(rows[0].keys())
        lines = [",".join(fieldnames)]
        lines += [",".join(str(row[key]) for key in fieldnames) for row in rows]
        (config.report_root / "cross-eval.csv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    return rows


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def collect_metrics(config: ExperimentConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for report_path in sorted(config.study_root.glob("seed-*/*/evaluation-*.json")):
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if "condition" not in payload:
            continue
        rows.append(
            {
                "condition": payload["condition"],
                "seed": payload["seed"],
                "test_snr_db": payload["test_snr_db"],
                "protocol": payload.get("protocol", "unseen"),
                "model": "dsp_only" if report_path.stem.endswith("dsponly") else "cnn",
                "accuracy": payload["nearest_centroid"]["accuracy"],
                "precision_macro": payload["nearest_centroid"]["precision_macro"],
                "recall_macro": payload["nearest_centroid"]["recall_macro"],
                "f1_macro": payload["nearest_centroid"]["f1_macro"],
                "svm_accuracy": payload["svm"]["accuracy"],
                "svm_f1_macro": payload["svm"]["f1_macro"],
                "agglomerative_ari": payload["clustering"]["agglomerative_ari"],
                "agglomerative_nmi": payload["clustering"]["agglomerative_nmi"],
                "hdbscan_ari": payload["clustering"]["hdbscan_ari"],
                "hdbscan_outlier_rate": payload["clustering"]["hdbscan_outlier_rate"],
            }
        )
    return rows


def _significance_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Paired tests of Pipeline B against Pipeline A, matched on seed and SNR."""
    results = []
    for protocol in sorted({row["protocol"] for row in rows}):
        indexed = {
            (row["condition"], row["seed"], row["test_snr_db"]): row
            for row in rows
            if row["model"] == "cnn" and row["protocol"] == protocol
        }
        for metric in ("accuracy", "f1_macro", "agglomerative_ari"):
            baseline, proposed = [], []
            for (condition, seed, snr), row in indexed.items():
                if condition != "A_raw_noisy":
                    continue
                partner = indexed.get(("B_full", seed, snr))
                if partner is not None:
                    baseline.append(row[metric])
                    proposed.append(partner[metric])
            if baseline:
                results.append(
                    {
                        "metric": metric,
                        "protocol": protocol,
                        "comparison": "B_full - A_raw_noisy",
                        **paired_significance(baseline, proposed),
                    }
                )
    return results


def analyze_study(config: ExperimentConfig) -> dict[str, object]:
    from src import plots

    rows = collect_metrics(config)
    if not rows:
        raise FileNotFoundError(
            "No evaluation reports found; run `run.py evaluate` first."
        )

    config.report_root.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    csv_lines = [",".join(fieldnames)]
    csv_lines += [
        ",".join("" if row[key] is None else str(row[key]) for key in fieldnames)
        for row in rows
    ]
    (config.report_root / "metric-summary.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )

    significance = _significance_table(rows)
    (config.report_root / "significance.json").write_text(
        json.dumps(significance, indent=2) + "\n", encoding="utf-8"
    )

    plots.render_all(config, rows)
    print(f"  wrote {len(rows)} metric rows and {len(significance)} significance tests")
    return {"rows": rows, "significance": significance}
