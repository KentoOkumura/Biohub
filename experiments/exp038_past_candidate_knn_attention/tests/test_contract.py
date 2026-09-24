from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

EXP = Path(__file__).resolve().parents[1]


def test_config_keeps_exp016_contract_and_only_trains_two_new_models() -> None:
    config = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    assert config["experiment"]["notebooks"] == ["train"]
    assert config["lineage"] == {
        "parent": "exp037_past_candidate_attention",
        "hypothesis_id": "HYP-20260910-10",
        "backlog_candidate": "past_candidate_knn_attention",
        "diff_summary": config["lineage"]["diff_summary"],
    }
    model = config["model"]
    assert model["training"]["active_variants"] == ["past_candidate_knn_attention"]
    assert model["training"]["epochs"] == 3
    assert model["output"]["model_count"] == 2
    assert model["control"]["retrain"] is False
    assert model["past_candidate_attention"]["expected_added_parameter_count"] == 1570
    assert model["past_candidate_attention"]["candidate_scope"].startswith("nearest_8")
    assert model["past_candidate_attention"]["max_past_candidates"] == 8
    assert model["candidate_retention"]["min_retention"] == 0.99
    assert model["inference"]["implemented"] is False
    assert model["inference"]["active_variants"] == []
    assert config["runtime"]["kaggle"]["train"]["kernel_id"] == (
        "kentookumura/exp038-past-candidate-knn-attention-train"
    )


@pytest.mark.parametrize("filename", ["attention_train_pipeline.py", f"{EXP.name}_train.py"])
def test_real_setup_reads_config_and_accepts_declared_variant(filename: str) -> None:
    from past_candidate_attention import PAST_CANDIDATE_FEATURE_NAMES

    tree = ast.parse((EXP / filename).read_text())
    if filename == "attention_train_pipeline.py":
        statements = next(
            node.body
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
    else:
        statements = next(
            node.body
            for node in tree.body
            if isinstance(node, ast.With)
            and isinstance(node.items[0].context_expr, ast.Call)
            and isinstance(node.items[0].context_expr.args[0], ast.Constant)
            and node.items[0].context_expr.args[0].value == "Setup and configuration"
        )
    selected = []
    for node in statements:
        target = node.targets[0] if isinstance(node, ast.Assign) else None
        if isinstance(target, ast.Name) and target.id == "budget":
            break
        if (isinstance(target, ast.Name) and target.id == "config") or selected:
            selected.append(node)
    assert len(selected) >= 10
    namespace = {
        "yaml": yaml,
        "CONFIG_PATH": EXP / "config.yaml",
        "PAST_CANDIDATE_FEATURE_NAMES": PAST_CANDIDATE_FEATURE_NAMES,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), namespace)
    assert namespace["attention_cfg"]["max_past_candidates"] == 8
    assert namespace["train_cfg"]["active_variants"] == ["past_candidate_knn_attention"]


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


def test_notebook_is_statically_inlined_and_checks_retention_before_training() -> None:
    source = (EXP / f"{EXP.name}_train.py").read_text()
    tree = ast.parse(source)
    local_modules = {path.stem for path in EXP.glob("*.py")}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in local_modules
        elif isinstance(node, ast.Import):
            assert not any(name.name in local_modules for name in node.names)
    assert "__file__" not in source
    assert source.index("retention = audit_candidate_retention") < source.index(
        "training_started ="
    )
    assert source.count("# %% [markdown]") >= 12
