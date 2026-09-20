from __future__ import annotations

import numpy as np
import pytest

from src.tracker_pair_metrics import (
    aggregate_known_parent_links,
    count_known_parent_links,
    summarize_known_parent_links,
)


def test_known_parent_links_ignore_unannotated_targets_and_count_wrong_edges() -> None:
    probabilities = np.array(
        [
            [0.60, 0.10, 0.47, 0.10],
            [0.20, 0.70, 0.47, 0.80],
            [0.20, 0.20, 0.06, 0.10],
        ]
    )
    target = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ]
    )
    counts = count_known_parent_links(probabilities, target, threshold=0.48)
    assert counts == {
        "known_parent_target_count": 3,
        "ambiguous_parent_target_count": 0,
        "correct_parent_top1_count": 1,
        "true_edge_count": 1,
        "false_edge_count": 1,
        "missed_true_edge_count": 2,
        "annotated_division_parent_count": 1,
        "both_daughters_linked_count": 0,
    }
    assert summarize_known_parent_links(counts) == {
        "annotated_target_edge_jaccard": 0.25,
        "correct_parent_top1_rate": 1 / 3,
        "known_parent_edge_recall": 1 / 3,
        "false_edges_per_known_parent_target": 1 / 3,
        "annotated_division_parent_recall": 0.0,
    }


def test_threshold_below_half_can_create_correct_and_wrong_edge_for_one_target() -> None:
    counts = count_known_parent_links(
        np.array([[0.49], [0.49], [0.02]]),
        np.array([[1], [0], [0]]),
        threshold=0.48,
    )
    assert counts["true_edge_count"] == 1
    assert counts["false_edge_count"] == 1
    assert summarize_known_parent_links(counts)["annotated_target_edge_jaccard"] == 0.5


def test_ambiguous_and_unannotated_targets_do_not_become_known_negatives() -> None:
    counts = count_known_parent_links(
        np.array([[0.6, 0.6], [0.4, 0.4]]),
        np.array([[1, 0], [1, 0]]),
        threshold=0.48,
    )
    assert counts["known_parent_target_count"] == 0
    assert counts["ambiguous_parent_target_count"] == 1
    assert summarize_known_parent_links(counts)["annotated_target_edge_jaccard"] is None


def test_aggregate_counts_before_ratios() -> None:
    first = count_known_parent_links(np.array([[0.9], [0.1]]), np.array([[1], [0]]), threshold=0.48)
    second = count_known_parent_links(
        np.array([[0.1, 0.1], [0.9, 0.9]]),
        np.array([[1, 1], [0, 0]]),
        threshold=0.48,
    )
    totals = aggregate_known_parent_links([first, second])
    assert totals["known_parent_target_count"] == 3
    assert totals["true_edge_count"] == 1
    assert totals["false_edge_count"] == 2
    assert totals["missed_true_edge_count"] == 2
    assert summarize_known_parent_links(totals)["annotated_target_edge_jaccard"] == 0.2


def test_rejects_non_normalized_probabilities() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        count_known_parent_links(np.array([[0.2], [0.2]]), np.array([[1], [0]]), threshold=0.48)
