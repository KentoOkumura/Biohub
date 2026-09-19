from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "exp020_division_triplet_candidates_diagnostic.py"
FUNCTIONS = run_path(str(SOURCE), run_name="triplet_diagnostic_test")


def test_triplet_enumeration_is_daughter_order_invariant():
    enumerate_triplets = FUNCTIONS["triplets_for_window"]
    source_ids = np.array([5])
    source_coords = np.array([[0.0, 0.0, 0.0]])
    target_ids = np.array([10, 11, 12])
    target_coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [12.0, 0.0, 0.0]])
    triplets, maximum = enumerate_triplets(
        source_ids, source_coords, target_ids, target_coords, 2.0, 2.0, 100
    )
    assert triplets == {(5, 10, 11)}
    assert maximum == 1
    reversed_triplets, _ = enumerate_triplets(
        source_ids, source_coords, target_ids[::-1], target_coords[::-1], 2.0, 2.0, 100
    )
    assert reversed_triplets == triplets
    with pytest.raises(ValueError, match="guard exceeded"):
        enumerate_triplets(source_ids, source_coords, target_ids, target_coords, 2.0, 2.0, 0)


def test_only_known_division_mother_provides_wrong_pair_labels():
    evaluate = FUNCTIONS["evaluate_known_division"]
    source_matches = {100: 5}
    target_matches = {200: 10, 201: 11, 202: 12}
    candidates = {(5, 10, 11), (5, 10, 12), (5, 11, 99)}
    assert evaluate((100, 200, 201), source_matches, target_matches, candidates) == (
        True,
        True,
        1,
    )
    assert evaluate((100, 200, 203), source_matches, target_matches, candidates) == (
        False,
        False,
        0,
    )


def test_sparse_single_edge_is_not_a_division():
    divisions = FUNCTIONS["true_divisions"]
    frames = {
        0: {100: np.zeros(3), 101: np.ones(3)},
        1: {200: np.zeros(3), 201: np.ones(3), 202: np.full(3, 2.0)},
    }
    outgoing = {100: (200, 201), 101: (202,)}
    assert divisions(0, frames, outgoing) == [(100, 200, 201)]
    assert divisions(1, frames, outgoing) == []


def test_gt_matching_is_one_to_one_and_radius_limited():
    match = FUNCTIONS["match_frame"]
    gt = {100: np.array([0.0, 0.0, 0.0]), 101: np.array([0.1, 0.0, 0.0])}
    ids = np.array([5, 6])
    coords = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    mapping = match(gt, ids, coords, 7.0)
    assert len(mapping) == 1
    assert set(mapping.values()) == {5}
