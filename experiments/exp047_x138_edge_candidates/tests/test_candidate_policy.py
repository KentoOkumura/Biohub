import numpy as np
import pytest

from experiments.exp047_x138_edge_candidates.candidate_policy import select_candidate_edges


def test_union_retains_baseline_and_tie_uses_parent_id():
    coords = np.array([[0, i, 0, 0] for i in range(5)] + [[1, 0, 0, 0]], dtype=np.float32)
    src = np.arange(5, dtype=np.int32)
    tgt = np.full(5, 5, dtype=np.int32)
    prob = np.array([0.11, 0.20, 0.20, 0.12, 0.49], dtype=np.float32)
    admitted = np.array([[4, 5, prob[4], 4.0]], dtype=np.float64)
    edges = select_candidate_edges(coords, src, tgt, prob, admitted)
    assert {(int(row[0]), int(row[1])) for row in edges.expanded} == {(1, 5), (2, 5), (4, 5)}
    assert edges.baseline.tolist() == admitted.tolist()
    assert len(edges.added) == 2


def test_baseline_mismatch_fails():
    coords = np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    with pytest.raises(ValueError, match="baseline_cache_mismatch"):
        select_candidate_edges(
            coords, np.array([0]), np.array([1]), np.array([0.6]), np.empty((0, 4))
        )


def test_strict_threshold_and_independent_daughter_ranks():
    coords = np.array(
        [[0, i, 0, 0] for i in range(7)] + [[1, 0, 0, 0], [1, 1, 0, 0]],
        dtype=np.float32,
    )
    src = np.array([0, 2, 4, 6, 1, 3, 5], dtype=np.int32)
    tgt = np.array([7, 7, 7, 7, 8, 8, 8], dtype=np.int32)
    prob = np.array([0.51, 0.20, 0.15, 0.10, 0.11, 0.09, 0.49], dtype=np.float32)
    admitted = np.array([[0, 7, prob[0], 0.0], [5, 8, prob[6], 0.0]])
    edges = select_candidate_edges(coords, src, tgt, prob, admitted)
    assert {(int(row[0]), int(row[1])) for row in edges.expanded} == {
        (0, 7),
        (2, 7),
        (4, 7),
        (1, 8),
        (5, 8),
    }
    assert len(edges.baseline) == 2
