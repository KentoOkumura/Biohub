from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from validation_diagnostic import active_scores, select_negative_budget_threshold  # noqa: E402


def test_negative_budget_threshold_handles_strict_comparison_and_ties() -> None:
    scores = np.array([0.1, 0.4, 0.4, 0.7, 0.9], dtype=np.float32)
    for budget in range(7):
        threshold = select_negative_budget_threshold(scores, budget)
        assert np.count_nonzero(scores > threshold) <= budget
        if threshold:
            lower = np.nextafter(np.float32(threshold), np.float32(-np.inf))
            assert np.count_nonzero(scores > lower) > budget
    assert select_negative_budget_threshold(scores, 2) == float(scores[1])
    assert select_negative_budget_threshold(np.array([]), 0) == 0


def test_teacher_scores_exclude_inactive_pairs_and_ambiguous_children() -> None:
    p = np.array([[0.7, 0.1, 0.3], [0.2, 0.7, 0.2], [0.1, 0.2, 0.5]])
    target = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]])
    scores = active_scores(p, target)
    assert sorted(scores["positive"]) == [0.1, 0.7, 0.7]
    assert len(scores["negative"]) == 5
    assert scores["known_child_confidence"].tolist() == [0.7]
    assert scores["known_child_correct"].tolist() == [True]
    # Inactive source 2, child 2 must not become a teacher-negative observation.
    assert 0.5 not in scores["negative"]


def test_internal_notebook_embeds_exact_inference_and_selection_definitions() -> None:
    notebook = ast.parse((EXP / "exp027_multi_frame_tracker_validation_diagnostic.py").read_text())
    compiled = {n.name: n for n in notebook.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for filename, names in {
        "local_tracker_model.py": ["LocalThreeFrameTracker", "LocalAttentionBlock"],
        "frozen_tracker.py": ["filter_nonempty_gt_window_paths", "build_three_frame_example"],
        "validation_diagnostic.py": [
            "select_negative_budget_threshold",
            "active_scores",
            "describe_scores",
        ],
    }.items():
        original = ast.parse((EXP / filename).read_text())
        definitions = {
            n.name: n for n in original.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        for name in names:
            assert ast.dump(compiled[name]) == ast.dump(definitions[name])
    assert "train_one_epoch" not in compiled
    assert "new_optimizer" not in compiled
