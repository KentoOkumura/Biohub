from __future__ import annotations

import ast
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import yaml

EXP_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXP_DIR / "config.yaml"
PARENT_CONFIG_PATH = EXP_DIR.parent / "exp005_embryo_holdout_batch8" / "config.yaml"
EXP003_SOURCE_MANIFEST_PATH = (
    EXP_DIR.parent / "exp003_official_metric_audit" / "assets" / "source_manifest.json"
)
SOURCE_ROOT = EXP_DIR / "official_source"
SOURCE_MANIFEST_PATH = SOURCE_ROOT / "SOURCE.json"
EXPERIMENT = "exp007_graph_checkpoint_selection"
SOURCE_COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"
REFERENCE_SELECTOR = "edge_accuracy_times_node_recall"
TREATMENT_SELECTOR = "official_adjusted_edge_jaccard_plus_0.1_division_jaccard"


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text())
    assert isinstance(value, dict)
    return value


def notebook_source(kind: str) -> str:
    path = EXP_DIR / f"{EXPERIMENT}_{kind}.ipynb"
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extracted_split_builder() -> Any:
    tree = ast.parse((EXP_DIR / f"{EXPERIMENT}_train.py").read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"embryo_id", "build_embryo_splits"}
    ]
    namespace: dict[str, Any] = {"Any": Any, "random": random}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "split_builder", "exec"),
        namespace,
    )
    return namespace["build_embryo_splits"]


def test_lineage_and_selector_contract() -> None:
    config = load_config()
    assert config["lineage"] == {
        "parent": "exp005_embryo_holdout_batch8",
        "hypothesis_id": "HYP-20260910-14",
        "backlog_candidate": "graph_checkpoint",
        "diff_summary": config["lineage"]["diff_summary"],
    }
    validation = config["validation"]
    assert validation["strategy"] == "leave_one_embryo_out"
    assert validation["expected_sample_count"] == 199
    assert validation["expected_embryo_counts"] == {"44b6": 71, "6bba": 128}
    assert validation["internal_split_seed"] == 0
    assert validation["reference_selector"] == REFERENCE_SELECTOR
    assert validation["treatment_selector"] == TREATMENT_SELECTOR
    assert validation["selector_data"] == "internal_checkpoint_selection_videos"
    assert validation["selector_tie_break"] == "latest_epoch"
    assert validation["comparison_mode"] == "same_training_run_two_selectors"
    assert config["model"]["training"]["save_epoch_indices"] == [0, 1, 2]
    assert config["model"]["training"]["selectors"] == [
        REFERENCE_SELECTOR,
        TREATMENT_SELECTOR,
    ]
    assert config["model"]["inference"]["evaluate_unique_selected_checkpoints_only"] is True


def test_exp005_model_loss_split_and_decode_are_fixed() -> None:
    config = load_config()
    parent = yaml.safe_load(PARENT_CONFIG_PATH.read_text())
    assert config["source"] == parent["source"]
    child_data = dict(config["data"])
    assert child_data.pop("split_contract_source") == (
        "../exp005_embryo_holdout_batch8/config.yaml"
    )
    assert child_data == parent["data"]
    for key in (
        "strategy",
        "outer_folds",
        "internal_split_seed",
        "internal_selection_fraction",
        "expected_sample_count",
        "expected_embryo_counts",
        "checkpoint_selection_metric",
        "primary_metric",
        "max_matching_distance_um",
        "diagnostic_only",
        "metric",
    ):
        assert config["validation"][key] == parent["validation"][key]
    assert config["model"]["name"] == parent["model"]["name"]
    assert config["model"]["params"] == parent["model"]["params"]

    child_training = dict(config["model"]["training"])
    assert child_training.pop("save_epoch_indices") == [0, 1, 2]
    assert child_training.pop("selectors") == [REFERENCE_SELECTOR, TREATMENT_SELECTOR]
    assert child_training == parent["model"]["training"]

    child_inference = dict(config["model"]["inference"])
    assert child_inference.pop("evaluate_unique_selected_checkpoints_only") is True
    assert child_inference.pop("smoke_videos_per_unique_checkpoint") == 1
    assert child_inference.pop("runtime_gate_hours") == 11.5
    parent_inference = dict(parent["model"]["inference"])
    assert parent_inference.pop("smoke_videos_per_fold") == 1
    assert parent_inference.pop("runtime_gate_hours") == 12
    assert child_inference == parent_inference

    assert config["runtime"]["batch_size"] == parent["runtime"]["batch_size"] == 8
    for key in (
        "use_amp",
        "num_workers",
        "expected_cuda_device_count",
        "expected_gpu_name_substring",
        "environment",
    ):
        assert config["runtime"][key] == parent["runtime"][key]


def test_split_builder_produces_exact_two_direction_contract() -> None:
    names_44b6 = [f"44b6_{index:08d}" for index in range(71)]
    names_6bba = [f"6bba_{index:08d}" for index in range(128)]
    all_names = names_44b6 + names_6bba
    records = extracted_split_builder()(all_names, load_config()["validation"]["outer_folds"], 0)
    assert [
        (len(item["train"]), len(item["test"]), len(item["outer_evaluation"])) for item in records
    ] == [(64, 7, 128), (116, 12, 71)]
    for record in records:
        training = set(record["train"]) | set(record["test"])
        evaluation = set(record["outer_evaluation"])
        assert not training & evaluation
        assert {name.split("_", 1)[0] for name in training} == {record["outer_train_embryo"]}
        assert {name.split("_", 1)[0] for name in evaluation} == {record["outer_evaluation_embryo"]}
    assert set(records[0]["outer_evaluation"]) | set(records[1]["outer_evaluation"]) == set(
        all_names
    )


def test_pinned_official_source_matches_manifests() -> None:
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text())
    assert manifest["commit"] == SOURCE_COMMIT
    assert manifest["license"] == "BSD-3-Clause"
    for relative, expected in manifest["files"].items():
        path = SOURCE_ROOT / relative
        assert path.is_file(), relative
        assert sha256(path) == expected, relative
    for relative, normalization in manifest["upstream_byte_normalizations"].items():
        assert normalization["change"] == "added_final_newline"
        normalized_to_upstream = (SOURCE_ROOT / relative).read_bytes().removesuffix(b"\n")
        assert (
            hashlib.sha256(normalized_to_upstream).hexdigest() == normalization["upstream_sha256"]
        )

    exp003_manifest = json.loads(EXP003_SOURCE_MANIFEST_PATH.read_text())
    assert exp003_manifest["commit"] == SOURCE_COMMIT
    for relative in ("metrics.py", "division_metrics.py", "LICENSE"):
        expected = exp003_manifest["files"][relative]
        mapped = {
            "metrics.py": SOURCE_ROOT / "src/tracking_cellmot/metrics.py",
            "division_metrics.py": SOURCE_ROOT / "src/tracking_cellmot/division_metrics.py",
            "LICENSE": SOURCE_ROOT / "LICENSE",
        }[relative]
        assert sha256(mapped) == expected


def test_every_pinned_source_file_is_bootstrapped() -> None:
    config = load_config()
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text())
    bootstrap = set(config["runtime"]["kaggle"]["bootstrap_files"])
    expected = {f"official_source/{relative}" for relative in manifest["files"]}
    expected.add("official_source/SOURCE.json")
    assert expected.issubset(bootstrap)
    assert not list(SOURCE_ROOT.rglob("*.pth"))


def test_train_notebook_saves_all_epoch_checkpoints_without_outer_leakage() -> None:
    source = notebook_source("train")
    assert "build_embryo_splits" in source
    assert "outer evaluation embryo leaked into training" in source
    assert "run_training_with_epoch_checkpoints" in source
    assert "train_module.evaluate = evaluate_and_capture" in source
    assert "train_module.evaluate = original_evaluate" in source
    assert 'checkpoint_path = checkpoint_dir / f"epoch_{epoch_index}.pth"' in source
    assert 'train_cfg["save_epoch_indices"]' in source
    assert '"manifest_type": "embryo_holdout_epoch_checkpoint_bundle"' in source
    assert '"model_count": sum(' in source
    assert '"outer_evaluation_used_for_selection": False' in source
    assert "unet_weights=None" in source
    assert "for record in splits:" in source
    assert "del smoke_model" in source
    assert "del trained_model" in source
    assert source.count("torch.cuda.empty_cache()") >= 4
    assert "__file__" not in source
    assert "TODO" not in source


def test_latest_epoch_tie_break_matches_upstream_greater_equal_selection() -> None:
    tree = ast.parse((EXP_DIR / f"{EXPERIMENT}_evaluation.py").read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "select_latest_epoch"
    ]
    namespace: dict[str, Any] = {
        "Any": Any,
        "validation_cfg": {"selector_tie_break": "latest_epoch"},
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "selector", "exec"),
        namespace,
    )
    records = [
        {"epoch_index": 0, "score": 0.4},
        {"epoch_index": 1, "score": 0.5},
        {"epoch_index": 2, "score": 0.5},
    ]
    selected_record = namespace["select_latest_epoch"](records, "score")
    assert selected_record["epoch_index"] == 2


def test_evaluation_scores_same_internal_videos_then_freezes_selection() -> None:
    source = notebook_source("evaluation")
    assert '"manifest_type") == "embryo_holdout_epoch_checkpoint_bundle"' in source
    assert 'get("model_count", -1)) != 6' in source
    assert "for epoch_index in expected_epoch_indices:" in source
    assert 'internal_names = sorted(split["test"])' in source
    assert 'forbidden_outer_names = set(split["outer_evaluation"])' in source
    assert '"outer_evaluation_used_for_selection": False' in source
    assert "select_latest_epoch" in source
    assert 'validation_cfg["selector_tie_break"] != "latest_epoch"' in source
    assert "unique_epochs_by_fold" in source
    selection_write = source.index("atomic_json(CHECKPOINT_SELECTION_PATH, checkpoint_selection)")
    outer_label_read = source.index("outer_rows_by_model:")
    assert selection_write < outer_label_read
    assert 'find_competition_dir("train")' in source
    assert 'find_competition_dir("test")' not in source
    assert "submission.csv" not in source
    assert "competitions submit" not in source
    assert "__file__" not in source
    assert "TODO" not in source


def test_evaluation_reuses_unique_outer_predictions_for_both_selectors() -> None:
    source = notebook_source("evaluation")
    assert 'if item["fold"] == fold and item["epoch_index"] == epoch_index' in source
    assert "outer_rows_by_model[(fold, epoch_index)] = rows" in source
    assert "selector_rows[selector] = rows" in source
    assert source.count("evaluate_pairs(") >= 1
    assert source.count("summarise(") >= 3
    assert '"recomputed_summary_matches": True' in source
    assert '"both_embryos_improved"' in source
    assert '"execution_failure_delta": 0' in source
    assert "does not cover all 199 outer videos" in source


def test_evaluation_saves_full_candidate_masks() -> None:
    source = notebook_source("evaluation")
    required_arrays = {
        "coords_tzyx",
        "detection_probability",
        "edge_source_index",
        "edge_target_index",
        "edge_probability",
        "edge_distance_downsampled",
        "edge_valid_mask",
        "edge_threshold_mask",
        "edge_graph_input_mask",
        "edge_final_selection_mask",
        "edge_pair_index",
    }
    for name in required_arrays:
        assert name in source
    assert "torch.softmax(edge_logits, dim=0)" in source
    assert "flat_probabilities > float(predict_cfg.threshold)" in source
    assert "td.solvers.ILPSolver" in source
    assert "np.savez_compressed" in source
    assert "sha256_arrays" in source


def test_evaluation_kernel_depends_only_on_exp007_train_kernel() -> None:
    kernel_sources = load_config()["runtime"]["kaggle"]["evaluation"]["kernel_sources"]
    assert kernel_sources == ["kentookumura/exp007-graph-checkpoint-selection-train"]


def test_numerical_stack_is_imported_after_offline_install() -> None:
    for kind in ("train", "evaluation"):
        source = notebook_source(kind)
        install_index = source.index("subprocess.run(")
        assert '"--no-deps"' in source
        assert '"imagecodecs==' not in source
        assert '"numpy==' not in source
        assert '"scipy==' not in source
        assert install_index < source.index("import numpy as np")
        assert install_index < source.index("import scipy")
        assert install_index < source.index("import torch")


def test_notebooks_have_readable_role_sections() -> None:
    expected_sections = {
        "train": (
            "# ## 1.",
            "# ## 2.",
            "# ## 3.",
            "# ## 4.",
            "# ## 5.",
            "# ## 6.",
        ),
        "evaluation": (
            "# ## 1.",
            "# ## 2.",
            "# ## 3.",
            "# ## 4.",
            "# ## 5.",
            "# ## 6.",
            "# ## 7.",
            "# ## 8.",
        ),
    }
    for kind, sections in expected_sections.items():
        source = (EXP_DIR / f"{EXPERIMENT}_{kind}.py").read_text()
        assert "# ## Contents" in source
        for section in sections:
            assert section in source
        assert len(source.splitlines()) > 400
