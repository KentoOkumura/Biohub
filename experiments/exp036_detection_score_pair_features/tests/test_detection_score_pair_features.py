from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
EXP = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = EXP / "exp036_detection_score_pair_features_train.py"
sys.path.insert(0, str(EXP))

from detection_score_tracker import (  # noqa: E402
    DetectionScorePairTracker,
    build_detection_score_pair_features,
)
from frozen_tracker import (  # noqa: E402
    detection_logit,
    evaluate_progression_gate,
    evaluate_tracker,
    fit_detection_score_normalizer,
    tracker_logits,
)


def load_config() -> dict[str, Any]:
    return yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))


def test_contract_uses_four_pair_features_and_four_retrained_trackers() -> None:
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
    assert score["pair_features"] == [
        "standardized_source_logit",
        "standardized_target_logit",
        "minimum_standardized_endpoint_logit",
        "absolute_standardized_logit_difference",
    ]
    assert score["pair_feature_dim"] == 4
    assert score["output_layer_initialization"] == "zeros"


def test_fold_normalizer_uses_only_explicit_training_paths(tmp_path: Path) -> None:
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


def test_pair_feature_order_and_zero_control() -> None:
    source = torch.tensor([[1.0, -2.0]])
    target = torch.tensor([[0.5, 3.0]])
    treatment = build_detection_score_pair_features(
        source,
        target,
        use_detection_score_values=True,
    )
    expected = torch.tensor(
        [
            [
                [[1.0, 0.5, 0.5, 0.5], [1.0, 3.0, 1.0, 2.0]],
                [[-2.0, 0.5, -2.0, 2.5], [-2.0, 3.0, -2.0, 5.0]],
            ]
        ]
    )
    assert torch.equal(treatment, expected)
    control = build_detection_score_pair_features(
        source,
        target,
        use_detection_score_values=False,
    )
    assert torch.equal(control, torch.zeros_like(expected))


class ZeroBase(torch.nn.Module):
    def forward(
        self,
        features_src: torch.Tensor,
        features_tgt: torch.Tensor,
        coords_src: torch.Tensor,
        coords_tgt: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        del features_tgt
        assert coords_src.shape[-1] == 3
        assert coords_tgt.shape[-1] == 3
        assert source_mask.dtype == torch.bool
        assert target_mask.dtype == torch.bool
        batch, source_count = features_src.shape[:2]
        return torch.zeros(batch, source_count, 2)


def tracker_inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, 3),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[1.0, -1.0]]),
        torch.tensor([[0.5, 2.0]]),
    )


def test_tracker_logits_routes_coordinates_masks_and_detection_scores() -> None:
    values = tracker_inputs()
    batch = {
        "features_src": values[0],
        "features_tgt": values[1],
        "coords_src": values[2],
        "coords_tgt": values[3],
        "source_mask": values[4],
        "target_mask": values[5],
        "detection_scores_src": values[6],
        "detection_scores_tgt": values[7],
    }
    model = DetectionScorePairTracker(
        ZeroBase(),
        pair_feature_dim=4,
        head_hidden_dim=2,
        use_detection_score_values=True,
    )
    assert tracker_logits(model, batch).shape == (1, 2, 2)


def test_zero_initialized_head_matches_base_and_control_isolates_score_values() -> None:
    treatment = DetectionScorePairTracker(
        ZeroBase(),
        pair_feature_dim=4,
        head_hidden_dim=2,
        use_detection_score_values=True,
    )
    control = DetectionScorePairTracker(
        ZeroBase(),
        pair_feature_dim=4,
        head_hidden_dim=2,
        use_detection_score_values=False,
    )
    assert torch.equal(treatment(*tracker_inputs()), control(*tracker_inputs()))
    for model in (treatment, control):
        first = model.score_pair_head[0]
        final = model.score_pair_head[-1]
        with torch.no_grad():
            first.weight.zero_()
            first.bias.zero_()
            first.weight[0, 0] = 1.0
            final.weight.zero_()
            final.bias.zero_()
            final.weight[0, 0] = 1.0
    assert not torch.equal(treatment(*tracker_inputs()), torch.zeros(1, 2, 2))
    assert torch.equal(control(*tracker_inputs()), torch.zeros(1, 2, 2))


class FixedLogits(torch.nn.Module):
    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        batch = args[0].shape[0]
        logits = torch.tensor([[8.0, 8.0], [-8.0, -8.0]])
        return logits.unsqueeze(0).expand(batch, -1, -1)


def test_score_band_diagnostics_count_edges_negatives_and_division() -> None:
    teacher = {
        "candidate_nodes": 4,
        "matched_candidate_nodes": 4,
        "gt_nodes": 4,
        "positive_edges": 2,
        "active_pairs": 4,
        "active_pairs_with_unknown_endpoint": 0,
        "division_parents": 1,
        "source_match_distance_sum_um": 0.0,
        "source_match_distance_count": 2,
        "target_match_distance_sum_um": 0.0,
        "target_match_distance_count": 2,
    }
    batch = {
        "features_src": torch.zeros(1, 2, 3),
        "features_tgt": torch.zeros(1, 2, 3),
        "coords_src": torch.zeros(1, 2, 3),
        "coords_tgt": torch.zeros(1, 2, 3),
        "source_mask": torch.ones(1, 2, dtype=torch.bool),
        "target_mask": torch.ones(1, 2, dtype=torch.bool),
        "detection_scores_src": torch.zeros(1, 2),
        "detection_scores_tgt": torch.zeros(1, 2),
        "raw_detection_scores_src": torch.tensor([[0.8, 0.99]]),
        "raw_detection_scores_tgt": torch.tensor([[0.85, 0.95]]),
        "target": torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]),
        "metadata": [{"teacher_stats": teacher}],
    }
    metrics = evaluate_tracker(
        FixedLogits(),
        [batch],
        torch.device("cpu"),
        gamma=2.0,
        use_amp=False,
        score_bin_edges=[0.0, 0.9, 1.0],
    )
    low, high = metrics["detection_score_diagnostics"]["bins"]
    assert low["positive_edge_count"] == 2
    assert low["true_positive_edge_count"] == 2
    assert low["division_parent_count"] == 1
    assert low["recovered_division_parent_count"] == 1
    assert low["active_negative_pair_count"] == 1
    assert high["active_negative_pair_count"] == 1
    assert high["false_positive_pair_count"] == 0


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


def fold_metrics(value: float, divisions: int = 2) -> dict[str, float | int]:
    return {
        "positive_edge_recall": value,
        "edge_accuracy": value,
        "division_parent_recall": value,
        "positive_edge_count": 5,
        "division_parent_count": divisions,
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


def test_train_source_preserves_train_only_boundary_and_four_model_layout() -> None:
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    for marker in (
        "fit_detection_score_normalizer",
        "gradient_update_samples",
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
