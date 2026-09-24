from __future__ import annotations

from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]


def test_config_keeps_exp016_contract_and_only_trains_two_new_models() -> None:
    config = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    assert config["experiment"]["notebooks"] == ["train"]
    assert config["lineage"] == {
        "parent": "exp016_frozen_image_encoder",
        "hypothesis_id": "HYP-20260910-10",
        "backlog_candidate": "past_candidate_attention",
        "diff_summary": config["lineage"]["diff_summary"],
    }
    model = config["model"]
    assert model["training"]["active_variants"] == ["past_candidate_attention"]
    assert model["training"]["epochs"] == 3
    assert model["output"]["model_count"] == 2
    assert model["control"]["retrain"] is False
    assert model["past_candidate_attention"]["expected_added_parameter_count"] == 1570
    assert model["past_candidate_attention"]["candidate_scope"].startswith("all_valid")
    assert model["inference"]["implemented"] is False
    assert model["inference"]["active_variants"] == []


def test_pipeline_has_runtime_gate_saved_control_diagnostic_and_no_submission() -> None:
    source = (EXP / "attention_train_pipeline.py").read_text(encoding="utf-8")
    for marker in (
        "load_control_tracker",
        "evaluate_tracker_diagnostic",
        "runtime_gate_seconds",
        "max_candidate_product",
        "graph_progression_gate",
        "request_user_approval_for_full_graph_inference",
        "No submission was created.",
    ):
        assert marker in source
    assert "kaggle competitions submit" not in source
    assert "generate_fixed_first_pass_history" not in source
