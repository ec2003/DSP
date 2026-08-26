from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import random
from typing import Mapping, Sequence

import hdbscan
import numpy as np
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score, precision_recall_fscore_support, roc_auc_score, roc_curve, v_measure_score
import torch
from torch import Tensor
from torch.nn import functional as functional

from src.data import ManifestRecord


@dataclass(frozen=True)
class VerificationPair:
    left_sample_id: str
    right_sample_id: str
    is_same_speaker: bool

    @property
    def pair_id(self) -> str:
        return sha256(f"{self.left_sample_id}|{self.right_sample_id}|{int(self.is_same_speaker)}".encode()).hexdigest()[:16]


def build_verification_pairs(records: Sequence[ManifestRecord], *, seed: int, positive_pairs_per_speaker: int = 50) -> list[VerificationPair]:
    """Balanced deterministic pairs, aiming for 50 positives per speaker in study data."""
    by_speaker: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        by_speaker[record.speaker_id].append(record)
    speakers = sorted(speaker for speaker, samples in by_speaker.items() if len(samples) >= 2)
    if len(speakers) < 2:
        raise ValueError("At least two speakers with two samples each are required")
    rng = random.Random(seed)
    pairs: list[VerificationPair] = []
    positives: list[VerificationPair] = []
    for speaker in speakers:
        samples = sorted(by_speaker[speaker], key=lambda record: record.sample_id)
        combinations = [(samples[left], samples[right]) for left in range(len(samples)) for right in range(left + 1, len(samples))]
        rng.shuffle(combinations)
        for left, right in combinations[:min(positive_pairs_per_speaker, len(combinations))]:
            positives.append(VerificationPair(left.sample_id, right.sample_id, True))
    pairs.extend(positives)
    # One deterministic different-speaker partner per positive; equal class balance.
    for index, positive in enumerate(positives):
        source_speaker = next(s for s in speakers if positive.left_sample_id.startswith(f"{s}/"))
        other_speakers = [speaker for speaker in speakers if speaker != source_speaker]
        other = other_speakers[(index + rng.randrange(len(other_speakers))) % len(other_speakers)]
        candidates = sorted(by_speaker[other], key=lambda record: record.sample_id)
        right = candidates[(index + rng.randrange(len(candidates))) % len(candidates)]
        pairs.append(VerificationPair(positive.left_sample_id, right.sample_id, False))
    return pairs


def score_pairs(pairs: Sequence[VerificationPair], embeddings: Mapping[str, Tensor]) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    labels: list[int] = []
    for pair in pairs:
        try:
            left, right = embeddings[pair.left_sample_id], embeddings[pair.right_sample_id]
        except KeyError as error:
            raise KeyError(f"Missing embedding for verification pair: {error.args[0]}") from error
        scores.append(functional.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())
        labels.append(int(pair.is_same_speaker))
    return np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)


def calibrate_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Choose an EER threshold using validation scores only."""
    _validate_scores(scores, labels)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return {"threshold": float(thresholds[index]), "eer": float((fpr[index] + fnr[index]) / 2)}


def verification_metrics(scores: np.ndarray, labels: np.ndarray, *, threshold: float | None = None) -> dict[str, float | int]:
    """Report EER/ROC plus fixed-threshold binary classification metrics."""
    _validate_scores(scores, labels)
    calibration = calibrate_threshold(scores, labels)
    threshold = calibration["threshold"] if threshold is None else threshold
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(labels, predictions, labels=[0, 1]).ravel())
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)), "eer": calibration["eer"], "eer_threshold": calibration["threshold"],
        "threshold": float(threshold), "accuracy": float((predictions == labels).mean()),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def pair_errors(pairs: Sequence[VerificationPair], scores: np.ndarray, labels: np.ndarray, *, threshold: float, diagnostics: Mapping[str, Mapping[str, object]] | None = None) -> list[dict[str, object]]:
    """Return false positives/negatives with reproducible pair provenance."""
    result: list[dict[str, object]] = []
    predictions = scores >= threshold
    for pair, score, label, prediction in zip(pairs, scores, labels, predictions, strict=True):
        if label == prediction:
            continue
        row: dict[str, object] = {**asdict(pair), "pair_id": pair.pair_id, "error": "false_negative" if label else "false_positive", "score": float(score), "threshold": threshold, "threshold_margin": float(score - threshold)}
        if diagnostics:
            row["left"] = diagnostics.get(pair.left_sample_id, {})
            row["right"] = diagnostics.get(pair.right_sample_id, {})
        result.append(row)
    return result


def paired_stratified_bootstrap(raw_scores: np.ndarray, dsp_scores: np.ndarray, labels: np.ndarray, *, metric: str = "f1", threshold_raw: float = 0.0, threshold_dsp: float = 0.0, seed: int = 11, repetitions: int = 2000) -> dict[str, float]:
    """Paired, class-stratified bootstrap CI for full-DSP minus raw-noisy."""
    _validate_scores(raw_scores, labels); _validate_scores(dsp_scores, labels)
    if raw_scores.shape != dsp_scores.shape:
        raise ValueError("paired scores must have equal shape")
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    deltas = []
    for _ in range(repetitions):
        indices = np.concatenate((rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)))
        raw = verification_metrics(raw_scores[indices], labels[indices], threshold=threshold_raw)[metric]
        dsp = verification_metrics(dsp_scores[indices], labels[indices], threshold=threshold_dsp)[metric]
        deltas.append(float(dsp) - float(raw))
    values = np.asarray(deltas)
    return {"metric": metric, "delta": float(values.mean()), "ci95_low": float(np.quantile(values, .025)), "ci95_high": float(np.quantile(values, .975)), "repetitions": repetitions}


def clustering_metrics(embeddings: Mapping[str, Tensor], records: Sequence[ManifestRecord], *, min_cluster_size: int, min_samples: int | None = None) -> dict[str, float]:
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")
    ordered = sorted(records, key=lambda record: record.sample_id)
    if len(ordered) < min_cluster_size:
        raise ValueError("Not enough embeddings for the requested minimum cluster size")
    vectors = functional.normalize(torch.stack([embeddings[record.sample_id].detach().cpu() for record in ordered]), p=2, dim=-1).numpy()
    predicted = hdbscan.HDBSCAN(metric="euclidean", min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(vectors)
    truth = [record.speaker_id for record in ordered]
    return {"ari": float(adjusted_rand_score(truth, predicted)), "nmi": float(normalized_mutual_info_score(truth, predicted)), "v_measure": float(v_measure_score(truth, predicted)), "clustered_coverage": float((predicted >= 0).mean()), "outlier_rate": float((predicted < 0).mean())}


def _validate_scores(scores: np.ndarray, labels: np.ndarray) -> None:
    if scores.shape != labels.shape or scores.ndim != 1 or set(labels.tolist()) != {0, 1}:
        raise ValueError("scores/labels must be one-dimensional, equal-sized, and contain both classes")
