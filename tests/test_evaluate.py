from __future__ import annotations

import numpy as np
import torch

from src.data import ManifestRecord
from src.experiments.evaluate import (
    build_verification_pairs,
    clustering_metrics,
    score_pairs,
    verification_metrics,
)


def _records() -> list[ManifestRecord]:
    return [
        ManifestRecord(f"p001/{index}", "p001", "unused.wav", "test", 3.0, 0.0)
        for index in range(2)
    ] + [
        ManifestRecord(f"p002/{index}", "p002", "unused.wav", "test", 3.0, 0.0)
        for index in range(2)
    ]


def test_verification_pairs_and_metrics_separate_known_speakers() -> None:
    records = _records()
    embeddings = {
        "p001/0": torch.tensor([1.0, 0.0]),
        "p001/1": torch.tensor([0.98, 0.02]),
        "p002/0": torch.tensor([0.0, 1.0]),
        "p002/1": torch.tensor([0.02, 0.98]),
    }

    pairs = build_verification_pairs(records, seed=42)
    scores, labels = score_pairs(pairs, embeddings)
    metrics = verification_metrics(scores, labels)

    assert len(pairs) == 4
    assert set(labels) == {0, 1}
    assert metrics["roc_auc"] == 1.0
    assert metrics["eer"] == 0.0


def test_hdbscan_metrics_expose_coverage_and_outliers() -> None:
    records = _records()
    embeddings = {
        "p001/0": torch.tensor([1.0, 0.0]),
        "p001/1": torch.tensor([0.99, 0.01]),
        "p002/0": torch.tensor([0.0, 1.0]),
        "p002/1": torch.tensor([0.01, 0.99]),
    }

    metrics = clustering_metrics(embeddings, records, min_cluster_size=2, min_samples=1)

    assert set(metrics) == {
        "ari",
        "nmi",
        "v_measure",
        "clustered_coverage",
        "outlier_rate",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["clustered_coverage"] + metrics["outlier_rate"] == 1.0
