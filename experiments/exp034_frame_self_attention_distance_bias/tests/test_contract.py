from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]


def load_graph_inference():
    spec = importlib.util.spec_from_file_location(
        "exp034_graph_inference", EXP / "graph_inference.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_uses_final_graph_evaluation_without_pair_gate() -> None:
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    assert config["experiment"]["notebooks"] == ["train", "inference"]
    assert "early_pair_gate" not in config["validation"]
    final = config["validation"]["final_graph_evaluation"]
    assert final["metric"] == "official_adjusted_edge_jaccard_plus_0.1_division_jaccard"
    assert final["fold_model_by_evaluation_embryo"] == {"6bba": 0, "44b6": 1}
    inference = config["model"]["inference"]
    assert inference["implemented"] is True
    assert inference["selected_variant"] == "model_b_spatial_distance_bias"
    assert inference["submission_created"] is False


def test_train_source_has_no_pair_diagnostic_passes() -> None:
    source = (EXP / "exp034_frame_self_attention_distance_bias_train.py").read_text()
    for removed in (
        "evaluate_pair_ranking",
        "fit_pair_boundaries",
        "pair_gate_readout",
        "pair_diagnostic_summary",
        "diagnostic_metrics",
        "baseline_outer",
        "trained_outer",
    ):
        assert removed not in source
    assert "identity_check_windows" in source
    assert 'int(train_cfg["epochs"]) * (len(train_paths) + len(validation_paths))' in source


def test_fold_architecture_resolution_uses_training_side_scales() -> None:
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    architecture = config["model"]["architectures"]["model_b_spatial_distance_bias"]
    graph_inference = load_graph_inference()
    fold_zero = graph_inference.resolve_fold_architecture(architecture, 0)
    fold_one = graph_inference.resolve_fold_architecture(architecture, 1)
    assert "spatial_distance_scale_um_by_fold" not in fold_zero
    assert fold_zero["spatial_distance_scale_um"] == 8.125
    assert fold_one["spatial_distance_scale_um"] == 8.285906791687012
