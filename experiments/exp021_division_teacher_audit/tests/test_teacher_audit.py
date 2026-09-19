from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "exp021_division_teacher_audit_diagnostic.py"
FUNCTIONS = run_path(str(SOURCE), run_name="teacher_audit_test")


def test_candidate_generation_keeps_daughter_order_invariant():
    generate = FUNCTIONS["triplets_for_window"]
    sources = np.array([5])
    source_coords = np.zeros((1, 3))
    targets = np.array([10, 11, 12])
    target_coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [12.0, 0.0, 0.0]])
    first, maximum = generate(sources, source_coords, targets, target_coords, 2, 2, 100)
    reverse, _ = generate(sources, source_coords, targets[::-1], target_coords[::-1], 2, 2, 100)
    assert first == reverse == {(5, 10, 11)}
    assert maximum == 1


def test_positive_strict_wrong_and_foreign_overlap():
    classify = FUNCTIONS["classify_triplet"]
    source_inverse = {5: 100}
    target_inverse = {10: 200, 11: 201, 12: 202}
    target_match = {200: 10, 201: 11, 202: 12}
    outgoing = {100: (200, 201), 101: (202,)}
    incoming = {200: 100, 201: 100, 202: 101}
    positive = classify(
        (5, 11, 10), source_inverse, target_inverse, target_match, outgoing, incoming
    )
    wrong = classify((5, 10, 12), source_inverse, target_inverse, target_match, outgoing, incoming)
    assert positive["known_positive"]
    assert not positive["strict_wrong"] and not positive["foreign_parent_wrong"]
    assert wrong["strict_wrong"] and wrong["foreign_parent_wrong"]
    assert wrong["foreign_both_daughters_gt"]
    assert int(wrong["strict_wrong"] or wrong["foreign_parent_wrong"]) == 1


def test_missing_division_daughter_does_not_make_strict_negative():
    classify = FUNCTIONS["classify_triplet"]
    classes = classify(
        (5, 10, 12),
        {5: 100},
        {10: 200, 12: 202},
        {200: 10, 202: 12},
        {100: (200, 201), 101: (202,)},
        {200: 100, 201: 100, 202: 101},
    )
    assert not classes["known_positive"]
    assert not classes["strict_wrong"]
    assert classes["foreign_parent_wrong"]


def test_single_edge_parent_is_reported_without_nondivision_label():
    classify = FUNCTIONS["classify_triplet"]
    classes = classify(
        (6, 12, 13),
        {6: 102},
        {12: 202, 13: 203},
        {202: 12, 203: 13},
        {101: (202,), 102: (203,)},
        {202: 101, 203: 102},
    )
    assert classes["single_outgoing_parent"]
    assert classes["single_outgoing_with_known_child"]
    assert classes["foreign_parent_wrong"]
    assert not classes["known_positive"] and not classes["strict_wrong"]


def test_gt_edge_validation_rejects_merges_and_nonadjacent_edges():
    build = FUNCTIONS["build_incoming"]
    frames = {
        0: {100: np.zeros(3), 101: np.zeros(3)},
        1: {200: np.zeros(3)},
        2: {300: np.zeros(3)},
    }
    assert build({100: (200,)}, frames) == {200: 100}
    with pytest.raises(ValueError, match="multiple incoming"):
        build({100: (200,), 101: (200,)}, frames)
    with pytest.raises(ValueError, match="nonadjacent"):
        build({100: (300,)}, frames)
