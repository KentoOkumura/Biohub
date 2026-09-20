from pathlib import Path
from runpy import run_path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
FUNCTIONS = run_path(
    str(EXPERIMENT_DIR / "exp023_synthetic_detector_teacher_audit_diagnostic.py"),
    run_name="detector_teacher_test",
)


def test_one_to_one_gt_match_respects_distance_gate():
    candidates = np.array([[0, 0, 0], [0, 0.2, 0], [50, 0, 0]], dtype=float)
    gt = np.array([[0, 0, 0], [2, 0, 0]], dtype=float)
    matched = FUNCTIONS["match_candidates"](candidates, gt, np.array([7, 8]), 1.0)
    assert np.count_nonzero(matched == 7) == 1
    assert 8 not in matched
    assert matched[-1] == -1


def test_complete_lineage_labels_same_mother_wrong_and_continuation():
    source = np.array([[0, 0, 0], [0, 4.875, 0]], dtype=float)
    target = np.array([[0, -0.8125, 0], [0, 0.8125, 0], [0, 4.875, 0]], dtype=float)
    children = [[2, 3], [4], [], [], []]
    classify = FUNCTIONS["classify_window_triplets"]
    counts, ids, labels = classify(
        source,
        target,
        np.array([0, 1]),
        np.array([2, 3, 4]),
        children,
        9.0,
        14.0,
        100,
        1000,
    )
    assert counts["positive_triplets"] == 1
    assert counts["wrong_division_triplets"] == 2
    assert counts["positive_with_wrong_mother"] == 1
    assert counts["continuation_triplets"] == 3
    assert counts["unmatched_triplets"] == 0
    assert len(ids) == len(labels) == 6
    unknown, _, unknown_labels = classify(
        source,
        target,
        np.array([0, 1]),
        np.array([2, 3, -1]),
        children,
        9.0,
        14.0,
        100,
        1000,
    )
    assert unknown["unmatched_triplets"] == 4
    assert len(unknown_labels) == 2


def test_synthetic_normalization_preserves_model_float32_input():
    volumes = np.arange(16, dtype=np.uint16).reshape(2, 2, 2, 2)
    normalized, q_low, q_high = FUNCTIONS["normalize_synthetic_volumes"](volumes, 0.001, 0.999)
    assert normalized.dtype == np.float32
    assert normalized.shape == volumes.shape
    assert np.isfinite(normalized).all()
    assert q_low < q_high
