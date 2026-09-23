from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

EXP = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = EXP / "exp036_detection_score_pair_features_train.py"
TRACKER_SOURCE = EXP / "frozen_tracker.py"
sys.path.insert(0, str(EXP))

from frozen_tracker import (  # noqa: E402
    detection_logit,
    evaluate_progression_gate,
    fit_detection_score_normalizer,
)


def load_config() -> dict[str, Any]:
    return yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))


def test_contract_uses_fold_only_score_pair_ablation() -> None:
    config = load_config()
    assert config["experiment"]["notebooks"] == ["train"]
    assert config["lineage"]["parent"] == "exp016_frozen_image_encoder"
    assert config["lineage"]["hypothesis_id"] == "HYP-20260920-03"
    assert config["lineage"]["backlog_candidate"] == "detection_score_features"
    model = config["model"]
    assert model["trainable_components"] == [
        "primary_SimpleNodeTransformer",
        "detection_score_pair_head",
    ]
    assert model["training"]["active_modes"] == [
        "neutral_score_control",
        "detection_score_pair_residual",
    ]
    assert model["feature_ablation"]["active_variants"] == model["training"]["active_modes"]
    assert model["control"]["retrain"] is True
    assert model["control"]["same_structure"] is True
    assert model["output"]["model_count"] == 4
    assert model["inference"]["implemented"] is False
    score = model["detection_score"]
    assert score["standardization_scope"] == "gradient_update_candidates_only"
    assert score["pair_feature_dim"] == 4
    assert score["output_layer_initialization"] == "zeros"


def test_fold_normalizer_uses_only_explicit_paths(tmp_path: Path) -> None:
    train_path = tmp_path / "train.npz"
    held_out_path = tmp_path / "held_out.npz"
    train_source = np.asarray([0.2, 0.8], dtype=np.float32)
    train_target = np.asarray([0.4], dtype=np.float32)
    np.savez(
        train_path,
        detection_scores_src=train_source,
        detection_scores_tgt=train_target,
    )
    np.savez(
        held_out_path,
        detection_scores_src=np.asarray([0.9999], dtype=np.float32),
        detection_scores_tgt=np.asarray([0.9999], dtype=np.float32),
    )
    normalizer = fit_detection_score_normalizer(
        [train_path],
        epsilon=1e-4,
        standard_deviation_floor=1e-6,
    )
    expected = detection_logit(np.concatenate([train_source, train_target]), 1e-4)
    assert normalizer["scope"] == "gradient_update_candidates_only"
    assert normalizer["count"] == 3
    assert normalizer["mean"] == pytest.approx(float(expected.mean()))
    assert normalizer["observed_standard_deviation"] == pytest.approx(float(expected.std()))
    assert normalizer["raw_max"] == pytest.approx(0.8)
    assert held_out_path.exists()


def fold_metrics(value: float, divisions: int = 2) -> dict[str, float | int]:
    return {
        "positive_edge_recall": value,
        "edge_accuracy": value,
        "division_parent_recall": value,
        "positive_edge_count": 5,
        "division_parent_count": divisions,
    }


def gate_config() -> dict[str, Any]:
    return {
        "minimum_positive_edge_recall_delta": 0.0,
        "minimum_edge_accuracy_delta": 0.0,
        "minimum_division_parent_recall_delta": 0.0,
        "require_positive_edge_count": True,
        "require_division_parent_count": True,
        "require_each_outer_fold": True,
        "on_failure": "stop_before_full_graph_inference",
    }


def test_progression_gate_requires_non_decline_in_every_fold() -> None:
    passing = evaluate_progression_gate(
        {
            "neutral_score_control": {0: fold_metrics(0.5), 1: fold_metrics(0.5)},
            "detection_score_pair_residual": {
                0: fold_metrics(0.5),
                1: fold_metrics(0.6),
            },
        },
        gate_config(),
    )
    assert passing["passed"] is True
    assert passing["full_graph_inference_allowed"] is True

    failing = evaluate_progression_gate(
        {
            "neutral_score_control": {0: fold_metrics(0.5), 1: fold_metrics(0.5)},
            "detection_score_pair_residual": {
                0: fold_metrics(0.49),
                1: fold_metrics(0.6),
            },
        },
        gate_config(),
    )
    assert failing["passed"] is False
    assert failing["full_graph_inference_allowed"] is False
    assert failing["on_failure"] == "stop_before_full_graph_inference"


def test_tracker_logits_matches_detection_score_tracker_argument_order() -> None:
    module = ast.parse(TRACKER_SOURCE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "tracker_logits"
    )
    return_node = next(
        node
        for node in function.body
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
    )
    keys = [argument.slice.value for argument in return_node.value.args]
    assert keys == [
        "features_src",
        "features_tgt",
        "coords_src",
        "coords_tgt",
        "source_mask",
        "target_mask",
        "detection_scores_src",
        "detection_scores_tgt",
    ]


def test_train_source_has_four_model_layout_and_no_graph_or_submission() -> None:
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    for marker in (
        "fit_detection_score_normalizer",
        '"gradient_update_samples"',
        "training_jobs = [(record, variant)",
        'OUTPUT_MODELS / variant / f"fold_{fold}"',
        "initial_logits_by_fold",
        "torch.equal(initial_logits, reference_logits)",
        "evaluate_progression_gate",
        '"full_graph_inference_run": False',
        '"model_count": len(model_manifest_records)',
        "No submission was created.",
    ):
        assert marker in source
    assert "kaggle competitions submit" not in source
    assert "from graph_inference import" not in source
