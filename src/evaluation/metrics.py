"""Ranking metrics for impression-level news recommendation evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def _as_arrays(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)

    if labels_array.ndim != 1 or scores_array.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional sequences")
    if len(labels_array) != len(scores_array):
        raise ValueError("labels and scores must have the same length")
    if len(labels_array) == 0:
        raise ValueError("labels and scores must not be empty")

    return labels_array, scores_array


def ranked_labels(labels: Sequence[int], scores: Sequence[float]) -> np.ndarray:
    labels_array, scores_array = _as_arrays(labels, scores)
    ranking = np.argsort(-scores_array, kind="mergesort")
    return labels_array[ranking]


def precision_at_k(labels: Sequence[int], scores: Sequence[float], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    ranked = ranked_labels(labels, scores)
    cutoff = min(k, len(ranked))
    return float(np.sum(ranked[:cutoff]) / cutoff)


def recall_at_k(labels: Sequence[int], scores: Sequence[float], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    ranked = ranked_labels(labels, scores)
    positives = np.sum(ranked)
    if positives == 0:
        return float("nan")

    cutoff = min(k, len(ranked))
    return float(np.sum(ranked[:cutoff]) / positives)


def ndcg_at_k(labels: Sequence[int], scores: Sequence[float], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    ranked = ranked_labels(labels, scores)
    cutoff = min(k, len(ranked))

    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
    dcg = float(np.sum(ranked[:cutoff] * discounts))

    ideal_hits = min(int(np.sum(ranked)), cutoff)
    if ideal_hits == 0:
        return float("nan")

    ideal_dcg = float(np.sum(discounts[:ideal_hits]))
    return dcg / ideal_dcg


def mrr(labels: Sequence[int], scores: Sequence[float]) -> float:
    ranked = ranked_labels(labels, scores)
    positive_ranks = np.flatnonzero(ranked > 0)
    if len(positive_ranks) == 0:
        return float("nan")

    return float(1.0 / (positive_ranks[0] + 1))


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels_array, scores_array = _as_arrays(labels, scores)
    if len(np.unique(labels_array)) < 2:
        return float("nan")

    return float(roc_auc_score(labels_array, scores_array))


def evaluate_ranking(
    labels: Sequence[int],
    scores: Sequence[float],
    k_values: Sequence[int] = (5, 10),
) -> dict[str, float]:
    metrics = {
        "mrr": mrr(labels, scores),
        "auc": auc(labels, scores),
    }

    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(labels, scores, k)
        metrics[f"recall@{k}"] = recall_at_k(labels, scores, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(labels, scores, k)

    return metrics
