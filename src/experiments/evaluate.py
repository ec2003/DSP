from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Mapping, Sequence

import hdbscan
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    roc_auc_score,
    roc_curve,
    v_measure_score,
)
import torch
from torch import Tensor
from torch.nn import functional as functional

from src.data import ManifestRecord


@dataclass(frozen=True)
class VerificationPair:
    left_sample_id: str
    right_sample_id: str
    is_same_speaker: bool


def build_verification_pairs(
    records: Sequence[ManifestRecord], *, seed: int
) -> list[VerificationPair]:
    """Build balanced deterministic same- and different-speaker verification pairs."""
    by_speaker: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        by_speaker[record.speaker_id].append(record)

    speakers = sorted(
        speaker for speaker, samples in by_speaker.items() if len(samples) >= 2
    )
    if len(speakers) < 2:
        raise ValueError(
            "At least two speakers with two samples each are required for verification"
        )

    random_generator = random.Random(seed)
    shuffled_samples: dict[str, list[ManifestRecord]] = {}
    for speaker in speakers:
        samples = sorted(by_speaker[speaker], key=lambda record: record.sample_id)
        random_generator.shuffle(samples)
        shuffled_samples[speaker] = samples

    pairs: list[VerificationPair] = []
    for speaker_index, speaker in enumerate(speakers):
        samples = shuffled_samples[speaker]
        pairs.append(VerificationPair(samples[0].sample_id, samples[1].sample_id, True))
        other_speaker = speakers[(speaker_index + 1) % len(speakers)]
        pairs.append(
            VerificationPair(
                samples[0].sample_id,
                shuffled_samples[other_speaker][0].sample_id,
                False,
            )
        )
    return pairs


def score_pairs(
    pairs: Sequence[VerificationPair],
    embeddings: Mapping[str, Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    labels: list[int] = []
    for pair in pairs:
        try:
            left_embedding = embeddings[pair.left_sample_id]
            right_embedding = embeddings[pair.right_sample_id]
        except KeyError as error:
            raise KeyError(
                f"Missing embedding for verification pair: {error.args[0]}"
            ) from error
        score = functional.cosine_similarity(
            left_embedding.unsqueeze(0), right_embedding.unsqueeze(0)
        ).item()
        scores.append(score)
        labels.append(int(pair.is_same_speaker))
    return np.asarray(scores), np.asarray(labels)


def verification_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if scores.shape != labels.shape or scores.ndim != 1:
        raise ValueError(
            "scores and labels must be one-dimensional arrays with the same shape"
        )
    if set(labels.tolist()) != {0, 1}:
        raise ValueError(
            "verification labels must include same-speaker and different-speaker pairs"
        )

    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    false_negative_rate = 1 - true_positive_rate
    eer_index = int(np.argmin(np.abs(false_positive_rate - false_negative_rate)))
    threshold = thresholds[eer_index]
    predictions = (scores >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "eer": float(
            (false_positive_rate[eer_index] + false_negative_rate[eer_index]) / 2
        ),
        "threshold": float(threshold),
        "accuracy": float((predictions == labels).mean()),
    }


def clustering_metrics(
    embeddings: Mapping[str, Tensor],
    records: Sequence[ManifestRecord],
    *,
    min_cluster_size: int,
    min_samples: int | None = None,
) -> dict[str, float]:
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")
    ordered_records = sorted(records, key=lambda record: record.sample_id)
    if len(ordered_records) < min_cluster_size:
        raise ValueError("Not enough embeddings for the requested minimum cluster size")

    vectors = torch.stack(
        [embeddings[record.sample_id].detach().cpu() for record in ordered_records]
    )
    vectors = functional.normalize(vectors, p=2, dim=-1).numpy()
    predicted_labels = hdbscan.HDBSCAN(
        metric="euclidean",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    ).fit_predict(vectors)
    true_labels = [record.speaker_id for record in ordered_records]
    return {
        "ari": float(adjusted_rand_score(true_labels, predicted_labels)),
        "nmi": float(normalized_mutual_info_score(true_labels, predicted_labels)),
        "v_measure": float(v_measure_score(true_labels, predicted_labels)),
        "clustered_coverage": float((predicted_labels >= 0).mean()),
        "outlier_rate": float((predicted_labels < 0).mean()),
    }
