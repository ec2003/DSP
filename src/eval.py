"""Evaluation: enrolment-based identification metrics and clustering metrics.

The embedding is scored two ways on speaker-disjoint held-out speakers:

* an enrolment probe (nearest centroid and a linear SVM) which yields the
  accuracy / precision / recall / F1 / confusion-matrix set, and
* unsupervised clustering (HDBSCAN and agglomerative with the true speaker
  count) which measures whether the embedding geometry separates speakers
  without any labels.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC

from src.corpus import ClipRecord


# --------------------------------------------------------------------------- #
# Embedding extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    waveforms: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(waveforms), batch_size):
        batch = torch.from_numpy(waveforms[start : start + batch_size]).to(device)
        outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def speaker_labels(records: list[ClipRecord]) -> tuple[np.ndarray, list[str]]:
    names = sorted({record.speaker_id for record in records})
    index = {name: position for position, name in enumerate(names)}
    return np.array([index[record.speaker_id] for record in records]), names


def enrollment_split(
    records: list[ClipRecord], enrollment_clips: int
) -> tuple[np.ndarray, np.ndarray]:
    """First ``enrollment_clips`` per speaker enrol; the rest are queries."""
    seen: dict[str, int] = defaultdict(int)
    enrol, query = [], []
    for position, record in enumerate(records):
        if seen[record.speaker_id] < enrollment_clips:
            enrol.append(position)
        else:
            query.append(position)
        seen[record.speaker_id] += 1
    return np.array(enrol), np.array(query)


# --------------------------------------------------------------------------- #
# Identification metrics
# --------------------------------------------------------------------------- #
def _class_centroids(
    embeddings: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(labels)
    centroids = np.stack(
        [embeddings[labels == label].mean(axis=0) for label in classes]
    )
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    return centroids, classes


def nearest_centroid_predict(
    embeddings: np.ndarray,
    labels: np.ndarray,
    enrol_index: np.ndarray,
    query_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cosine nearest-centroid identification. Returns (truth, prediction, scores)."""
    centroids, classes = _class_centroids(embeddings[enrol_index], labels[enrol_index])
    scores = embeddings[query_index] @ centroids.T
    return labels[query_index], classes[scores.argmax(axis=1)], scores


def svm_predict(
    embeddings: np.ndarray,
    labels: np.ndarray,
    enrol_index: np.ndarray,
    query_index: np.ndarray,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    classifier = LinearSVC(C=1.0, random_state=seed, max_iter=5000)
    classifier.fit(embeddings[enrol_index], labels[enrol_index])
    return labels[query_index], classifier.predict(embeddings[query_index])


def classification_metrics(
    truth: np.ndarray, prediction: np.ndarray
) -> dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, prediction, average="macro", zero_division=0
    )
    return {
        "accuracy": float(np.mean(truth == prediction)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "n_queries": int(truth.size),
        "confusion_matrix": confusion_matrix(
            truth, prediction, labels=np.arange(max(truth.max(), prediction.max()) + 1)
        ).tolist(),
    }


# --------------------------------------------------------------------------- #
# Clustering metrics
# --------------------------------------------------------------------------- #
def clustering_metrics(
    embeddings: np.ndarray, labels: np.ndarray, min_cluster_size: int = 5
) -> dict[str, float]:
    """HDBSCAN (speaker count unknown) and agglomerative (speaker count known)."""
    import hdbscan

    n_speakers = int(np.unique(labels).size)

    agglomerative = AgglomerativeClustering(
        n_clusters=n_speakers, metric="cosine", linkage="average"
    ).fit_predict(embeddings)

    density = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    hdbscan_labels = density.fit_predict(embeddings.astype(np.float64))
    assigned = hdbscan_labels >= 0

    metrics = {
        "agglomerative_ari": float(adjusted_rand_score(labels, agglomerative)),
        "agglomerative_nmi": float(normalized_mutual_info_score(labels, agglomerative)),
        "agglomerative_purity": _purity(labels, agglomerative),
        "hdbscan_n_clusters": int(np.unique(hdbscan_labels[assigned]).size),
        "hdbscan_outlier_rate": float(1.0 - assigned.mean()),
        "true_n_speakers": n_speakers,
    }
    if assigned.any():
        metrics["hdbscan_ari"] = float(
            adjusted_rand_score(labels[assigned], hdbscan_labels[assigned])
        )
        metrics["hdbscan_nmi"] = float(
            normalized_mutual_info_score(labels[assigned], hdbscan_labels[assigned])
        )
        metrics["hdbscan_purity"] = _purity(labels[assigned], hdbscan_labels[assigned])
    else:
        metrics |= {"hdbscan_ari": 0.0, "hdbscan_nmi": 0.0, "hdbscan_purity": 0.0}
    return metrics


def _purity(truth: np.ndarray, clusters: np.ndarray) -> float:
    total = 0
    for cluster in np.unique(clusters):
        members = truth[clusters == cluster]
        total += np.bincount(members).max()
    return float(total / truth.size)


# --------------------------------------------------------------------------- #
# Error analysis
# --------------------------------------------------------------------------- #
def error_cases(
    truth: np.ndarray,
    prediction: np.ndarray,
    scores: np.ndarray,
    query_index: np.ndarray,
    records: list[ClipRecord],
    names: list[str],
    limit: int = 100,
) -> list[dict[str, object]]:
    """Misclassified queries with the score margin that caused the error."""
    wrong = np.flatnonzero(truth != prediction)
    ordered = wrong[np.argsort(scores[wrong].max(axis=1))[::-1]][:limit]
    return [
        {
            "sample_id": records[int(query_index[position])].sample_id,
            "true_speaker": names[int(truth[position])],
            "predicted_speaker": names[int(prediction[position])],
            "predicted_score": float(scores[position].max()),
            "true_score": float(scores[position, int(truth[position])]),
            "margin": float(
                scores[position].max() - scores[position, int(truth[position])]
            ),
        }
        for position in ordered
    ]


def confusable_speakers(
    truth: np.ndarray, prediction: np.ndarray, names: list[str], limit: int = 10
) -> list[dict[str, object]]:
    """Speaker pairs that account for the most confusions."""
    matrix = confusion_matrix(truth, prediction, labels=np.arange(len(names)))
    np.fill_diagonal(matrix, 0)
    flat = np.dstack(
        np.unravel_index(np.argsort(matrix, axis=None)[::-1], matrix.shape)
    )[0]
    return [
        {
            "true_speaker": names[int(i)],
            "predicted_speaker": names[int(j)],
            "count": int(matrix[i, j]),
        }
        for i, j in flat[:limit]
        if matrix[i, j] > 0
    ]

