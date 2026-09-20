"""Internal-only threshold selection and sparse-label confidence diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def active_scores(probabilities: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    probabilities = np.asarray(probabilities)
    positive = np.asarray(target) > 0.5
    if probabilities.shape != positive.shape or positive.ndim != 2:
        raise ValueError("probability and teacher shapes differ")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("invalid probability")
    active = positive.any(axis=1)[:, None] | positive.any(axis=0)[None, :]
    unique_child = positive.sum(axis=0) == 1
    children = np.flatnonzero(unique_child)
    parents = probabilities[:, children].argmax(axis=0)
    return {
        "positive": probabilities[positive],
        "negative": probabilities[active & ~positive],
        "known_child_confidence": probabilities[parents, children],
        "known_child_correct": positive[parents, children],
    }


def select_negative_budget_threshold(negative: np.ndarray, budget: int) -> float:
    """Smallest threshold in [0,1] with strict-greater predictions <= budget."""
    negative = np.asarray(negative).reshape(-1)
    if budget < 0 or not np.isfinite(negative).all() or np.any((negative < 0) | (negative > 1)):
        raise ValueError("invalid negative scores or budget")
    if len(negative) <= budget:
        return 0.0
    position = len(negative) - budget - 1
    return float(np.partition(negative, position)[position])


def describe_scores(
    scores: dict[str, np.ndarray], threshold: float, quantiles: list[float], bins: list[float]
) -> dict[str, Any]:
    positive, negative = scores["positive"], scores["negative"]
    confidence, correct = scores["known_child_confidence"], scores["known_child_correct"]
    rows = []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:], strict=True)):
        mask = (confidence >= lo) & ((confidence < hi) if i < len(bins) - 2 else (confidence <= hi))
        rows.append(
            {
                "lower": lo,
                "upper": hi,
                "count": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()) if mask.any() else None,
                "accuracy": float(correct[mask].mean()) if mask.any() else None,
            }
        )
    return {
        "threshold": threshold,
        "positive_edge_count": len(positive),
        "true_positive_count": int(np.count_nonzero(positive > threshold)),
        "positive_edge_recall": float(np.mean(positive > threshold)) if len(positive) else None,
        "false_positive_active_pair_count": int(np.count_nonzero(negative > threshold)),
        "teacher_negative_pair_count": len(negative),
        "true_parent_probability_quantiles": np.quantile(positive, quantiles).tolist()
        if len(positive)
        else [],
        "mean_true_parent_probability": float(positive.mean()) if len(positive) else None,
        "known_unique_parent_child_count": len(confidence),
        "known_child_top1_accuracy": float(correct.mean()) if len(correct) else None,
        "known_child_mean_max_probability": float(confidence.mean()) if len(confidence) else None,
        "known_child_confidence_bins": rows,
    }
