"""Counts for annotated parent links before graph optimization and repair.

The caller supplies the exact source-axis probabilities used to create candidate
edges. Targets without one known, candidate-visible parent are excluded: sparse
annotations do not establish that an unmatched target has no parent.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

COUNT_KEYS = (
    "known_parent_target_count",
    "ambiguous_parent_target_count",
    "correct_parent_top1_count",
    "true_edge_count",
    "false_edge_count",
    "missed_true_edge_count",
    "annotated_division_parent_count",
    "both_daughters_linked_count",
)


def count_known_parent_links(
    probabilities: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
) -> dict[str, int]:
    """Count candidate links on targets with exactly one annotated parent.

    ``probabilities`` must be source-by-target and normalized over sources,
    matching the candidate-edge generator. A wrong source linked to a target
    with a known parent is a false edge; targets without a known parent are not
    scored as negatives.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(target)
    if probabilities.ndim != 2 or target.shape != probabilities.shape:
        raise ValueError("probabilities and target must have the same 2D shape")
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be finite and between zero and one")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")

    positive = target > 0.5
    parent_counts = positive.sum(axis=0)
    eligible = parent_counts == 1
    counts = dict.fromkeys(COUNT_KEYS, 0)
    counts["known_parent_target_count"] = int(eligible.sum())
    counts["ambiguous_parent_target_count"] = int((parent_counts > 1).sum())
    if eligible.any():
        selected = probabilities[:, eligible]
        if not np.allclose(selected.sum(axis=0), 1.0, rtol=1e-4, atol=1e-4):
            raise ValueError("probabilities must sum to one over source candidates")
        truth = positive[:, eligible]
        predicted = selected > threshold
        counts["correct_parent_top1_count"] = int(
            (selected.argmax(axis=0) == truth.argmax(axis=0)).sum()
        )
        counts["true_edge_count"] = int((predicted & truth).sum())
        counts["false_edge_count"] = int((predicted & ~truth).sum())
        counts["missed_true_edge_count"] = int((~predicted & truth).sum())

    binary_division_rows = np.flatnonzero(positive.sum(axis=1) == 2)
    for row in binary_division_rows:
        daughters = np.flatnonzero(positive[row])
        if not eligible[daughters].all():
            continue
        counts["annotated_division_parent_count"] += 1
        counts["both_daughters_linked_count"] += int(
            bool((probabilities[row, daughters] > threshold).all())
        )
    return counts


def aggregate_known_parent_links(records: Iterable[Mapping[str, int]]) -> dict[str, int]:
    """Sum counts across windows before computing ratios."""
    totals = dict.fromkeys(COUNT_KEYS, 0)
    for record in records:
        if set(record) != set(COUNT_KEYS) or any(record[key] < 0 for key in COUNT_KEYS):
            raise ValueError("record has missing, extra, or negative counts")
        for key in COUNT_KEYS:
            totals[key] += int(record[key])
    return totals


def summarize_known_parent_links(counts: Mapping[str, int]) -> dict[str, float | None]:
    """Report candidate-edge Jaccard and its components on scorable targets."""
    if set(counts) != set(COUNT_KEYS):
        raise ValueError("counts must contain exactly the known-parent count keys")
    known = counts["known_parent_target_count"]
    true = counts["true_edge_count"]
    false = counts["false_edge_count"]
    missed = counts["missed_true_edge_count"]
    if true + missed != known:
        raise ValueError("known parent count must equal true plus missed edges")
    union = true + false + missed
    divisions = counts["annotated_division_parent_count"]
    return {
        "annotated_target_edge_jaccard": true / union if union else None,
        "correct_parent_top1_rate": counts["correct_parent_top1_count"] / known if known else None,
        "known_parent_edge_recall": true / known if known else None,
        "false_edges_per_known_parent_target": false / known if known else None,
        "annotated_division_parent_recall": counts["both_daughters_linked_count"] / divisions
        if divisions
        else None,
    }
