from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from context_diagnostic import (  # noqa: E402
    compare_known_edges,
    diagnose_pair,
    paired_video_bootstrap,
)


def test_diagnostic_distinguishes_correct_rank_from_confidence() -> None:
    probabilities = np.array([[0.48, 0.10], [0.30, 0.80], [0.22, 0.10]])
    target = np.array([[1, 0], [0, 1], [0, 0]])
    summary, edges = diagnose_pair(
        np.log(probabilities),
        probabilities,
        target,
        threshold=0.5,
        threshold_grid=[0.4, 0.5],
    )
    assert summary["true_positive_count"] == 1
    assert summary["top1_correct_count"] == 2
    assert summary["threshold_counts"][0]["true_positive_count"] == 2
    assert edges[0]["true_parent_rank"] == 1
    assert not edges[0]["recovered"]


def test_rank_ties_and_source_softmax_contract() -> None:
    probabilities = np.array([[0.5], [0.5]])
    target = np.array([[0], [1]])
    summary, edges = diagnose_pair(
        np.zeros((2, 1)),
        probabilities,
        target,
        threshold=0.5,
        threshold_grid=[0.5],
    )
    assert edges[0]["true_parent_rank"] == 2
    assert edges[0]["tied_parent_count"] == 2
    assert not edges[0]["top1_correct"]
    assert summary["true_positive_count"] == 0
    with pytest.raises(ValueError, match="source parents"):
        diagnose_pair(
            np.zeros((2, 1)), np.ones((2, 1)), target, threshold=0.5, threshold_grid=[0.5]
        )


def edge(sample: str, probability: float) -> dict:
    return {
        "sample": sample,
        "source_frame": 1,
        "target_frame": 2,
        "source_id": 5,
        "target_id": 8,
        "true_parent_rank": 1,
        "top1_correct": True,
        "true_parent_probability": probability,
        "recovered": probability > 0.5,
        "predicted_parent_id": 5,
        "division_parent": False,
    }


def test_paired_confidence_change_and_video_bootstrap() -> None:
    before = [edge("video_a", 0.48), edge("video_b", 0.47)]
    after = [edge("video_a", 0.52), edge("video_b", 0.53)]
    summary, paired = compare_known_edges(before, after)
    assert summary["rescued_threshold"] == 2
    assert summary["rescued_top1"] == 0
    assert summary["threshold_rescues_with_both_top1"] == 2
    interval = paired_video_bootstrap(
        paired, ["video_a", "video_b"], repeats=100, seed=42, confidence=0.95
    )
    assert interval["recall_delta_interval"] == [1.0, 1.0]
    assert interval["top1_delta_interval"] == [0.0, 0.0]
    with pytest.raises(ValueError, match="identities"):
        compare_known_edges(before, after[:1])


def test_selfcontained_diagnostic_uses_the_exact_inference_definitions() -> None:
    notebook = ast.parse((EXP / "exp027_multi_frame_tracker_diagnostic.py").read_text())
    compiled = {n.name: n for n in notebook.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    for filename, names in {
        "local_tracker_model.py": ["LocalAttentionBlock", "LocalThreeFrameTracker"],
        "frozen_tracker.py": [
            "build_three_frame_example",
            "build_window_example",
            "legacy_focal_bce",
            "collate_three_frame_examples",
            "greedy_match_candidates",
        ],
        "context_diagnostic.py": ["diagnose_pair", "compare_known_edges", "paired_video_bootstrap"],
    }.items():
        original = ast.parse((EXP / filename).read_text())
        definitions = {
            n.name: n for n in original.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        }
        for name in names:
            assert ast.dump(compiled[name]) == ast.dump(definitions[name]), (filename, name)
    assert "train_one_epoch" not in compiled
    assert "new_optimizer" not in compiled
