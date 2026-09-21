"""Exact pair-ranking and fixed-threshold readouts for the saved trackers."""

from __future__ import annotations

from typing import Any

import numpy as np


def precision_recall_by_score(
    scores: np.ndarray, labels: np.ndarray, target_precision: float
) -> dict[str, Any]:
    """Include every tied float32 score before reporting a PR point."""
    score = np.asarray(scores, dtype=np.float32).reshape(-1)
    label = np.asarray(labels, dtype=np.bool_).reshape(-1)
    if score.size != label.size or not np.isfinite(score).all():
        raise ValueError("scores and labels must align and scores must be finite")
    if not 0.0 < target_precision <= 1.0:
        raise ValueError("target_precision must be in (0, 1]")
    positives = int(label.sum())
    if score.size == 0 or positives == 0:
        return {
            "threshold": [],
            "precision": [],
            "recall": [],
            "positive_count": positives,
            "pair_count": int(score.size),
            "recall_at_target_precision": None,
            "threshold_at_target_precision": None,
            "target_reachable": False,
        }
    order = np.argsort(-score, kind="stable")
    sorted_scores = score[order]
    cumulative_tp = np.cumsum(label[order], dtype=np.int64)
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), score.size - 1]
    tp = cumulative_tp[ends]
    predicted = ends + 1
    precision = tp / predicted
    recall = tp / positives
    eligible = np.flatnonzero(precision >= target_precision)
    best_index = int(eligible[np.argmax(recall[eligible])]) if eligible.size else None
    return {
        "threshold": sorted_scores[ends].astype(float).tolist(),
        "precision": precision.astype(float).tolist(),
        "recall": recall.astype(float).tolist(),
        "positive_count": positives,
        "pair_count": int(score.size),
        "recall_at_target_precision": (
            float(recall[best_index]) if best_index is not None else None
        ),
        "threshold_at_target_precision": (
            float(sorted_scores[ends[best_index]]) if best_index is not None else None
        ),
        "target_reachable": best_index is not None,
    }


def fixed_threshold_counts(
    scores: np.ndarray,
    labels: np.ndarray,
    active: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, int | float | None]:
    score = np.asarray(scores, dtype=np.float32)
    label = np.asarray(labels, dtype=np.bool_)
    mask = np.asarray(active, dtype=np.bool_)
    if score.shape != label.shape or score.shape != mask.shape:
        raise ValueError("scores, labels and active mask must have identical shape")
    selected = score > threshold
    tp = int(np.count_nonzero(selected & label))
    positives = int(np.count_nonzero(label))
    fp = int(np.count_nonzero(selected & ~label & mask))
    return {
        "positive_edge_count": positives,
        "true_positive_edge_count": tp,
        "false_positive_pair_count": fp,
        "active_pair_count": int(np.count_nonzero(mask)),
        "positive_edge_recall": tp / positives if positives else None,
        "positive_pair_precision": tp / (tp + fp) if tp + fp else None,
    }


def nearest_neighbor_distance_um(coords: np.ndarray) -> np.ndarray:
    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("coords must have shape [N, 3]")
    if len(points) <= 1:
        return np.full(len(points), np.nan, dtype=np.float32)
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(points).query(points, k=2)
    return distances[:, 1].astype(np.float32)


def quantile_boundaries(values: np.ndarray, quantiles: list[float]) -> list[float]:
    numbers = np.asarray(values, dtype=np.float64).reshape(-1)
    numbers = numbers[np.isfinite(numbers)]
    if numbers.size == 0:
        raise ValueError("training-side distribution has no finite values")
    if any(not 0.0 < point < 1.0 for point in quantiles):
        raise ValueError("quantiles must lie in (0, 1)")
    return np.unique(np.quantile(numbers, quantiles)).astype(float).tolist()


def bucket_readout(
    values: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    edges: list[float],
    *,
    threshold: float = 0.5,
    min_positive_edges: int = 30,
) -> list[dict[str, Any]]:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    score = np.asarray(scores, dtype=np.float32).reshape(-1)
    label = np.asarray(labels, dtype=np.bool_).reshape(-1)
    if not (len(value) == len(score) == len(label)):
        raise ValueError("bucket arrays must align")
    bins = np.searchsorted(np.asarray(edges), value, side="right")
    bins[~np.isfinite(value)] = -1
    rows = []
    for bucket in range(-1, len(edges) + 1):
        selected = bins == bucket
        if not selected.any():
            continue
        counts = fixed_threshold_counts(
            score[selected],
            label[selected],
            np.ones(int(selected.sum()), dtype=np.bool_),
            threshold=threshold,
        )
        if counts["positive_edge_count"] < min_positive_edges:
            counts["positive_edge_recall"] = None
            counts["positive_pair_precision"] = None
        rows.append(
            {
                "bucket": bucket,
                "lower": None if bucket <= 0 else edges[bucket - 1],
                "upper": None if bucket < 0 or bucket >= len(edges) else edges[bucket],
                "missing_value": bucket < 0,
                **counts,
            }
        )
    return rows
