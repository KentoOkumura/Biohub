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
PARENT_CONFIG_PATH = EXP_DIR.parent / "exp004_embryo_holdout_baseline" / "config.yaml"
EXP003_SOURCE_MANIFEST_PATH = (
    EXP_DIR.parent / "exp003_official_metric_audit" / "assets" / "source_manifest.json"
)
SOURCE_ROOT = EXP_DIR / "official_source"
SOURCE_MANIFEST_PATH = SOURCE_ROOT / "SOURCE.json"
EXPERIMENT = "exp005_embryo_holdout_batch8"
SOURCE_COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"


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
    exec(compile(ast.Module(body=selected, type_ignores=[]), "split_builder", "exec"), namespace)
    return namespace["build_embryo_splits"]


def test_lineage_and_embryo_holdout_contract() -> None:
    config = load_config()
    assert config["lineage"]["parent"] == "exp004_embryo_holdout_baseline"
    assert config["lineage"]["hypothesis_id"] == "HYP-20260910-14"
    assert config["lineage"]["backlog_candidate"] == "embryo_holdout_baseline"
    assert config["validation"]["strategy"] == "leave_one_embryo_out"
    assert config["validation"]["expected_sample_count"] == 199
    assert config["validation"]["expected_embryo_counts"] == {"44b6": 71, "6bba": 128}
    assert config["validation"]["internal_split_seed"] == 0
    assert config["validation"]["primary_metric"] == (
        "official_adjusted_edge_jaccard_plus_0.1_division_jaccard"
    )
    assert config["validation"]["max_matching_distance_um"] == 7.0
    assert config["validation"]["outer_folds"] == [
        {
            "fold": 0,
            "train_embryo": "44b6",
            "evaluation_embryo": "6bba",
            "expected_gradient_update_count": 64,
            "expected_checkpoint_selection_count": 7,
            "expected_evaluation_count": 128,
        },
        {
            "fold": 1,
            "train_embryo": "6bba",
            "evaluation_embryo": "44b6",
            "expected_gradient_update_count": 116,
            "expected_checkpoint_selection_count": 12,
            "expected_evaluation_count": 71,
        },
    ]


def test_model_loss_and_decode_match_exp004_except_training_batch() -> None:
    config = load_config()
    parent = yaml.safe_load(PARENT_CONFIG_PATH.read_text())
    assert config["source"] == parent["source"]
    assert config["data"] == parent["data"]
    assert config["model"]["name"] == parent["model"]["name"]
    assert config["model"]["params"] == parent["model"]["params"]
    child_training = dict(config["model"]["training"])
    parent_training = dict(parent["model"]["training"])
    assert child_training.pop("batch_size") == 8
    assert parent_training.pop("batch_size") == 16
    assert child_training == parent_training
    for key, value in parent["model"]["inference"].items():
        assert config["model"]["inference"][key] == value
    for key in (
        "use_amp",
        "num_workers",
        "expected_cuda_device_count",
        "expected_gpu_name_substring",
    ):
        assert config["runtime"][key] == parent["runtime"][key]
    assert config["runtime"]["batch_size"] == 8
    assert parent["runtime"]["batch_size"] == 16
    for key in (
        "enable_gpu",
        "enable_internet",
        "machine_shape",
        "time_limit_hours",
        "dataset_sources",
        "bootstrap_files",
    ):
        assert config["runtime"]["kaggle"][key] == parent["runtime"]["kaggle"][key]
    assert config["runtime"]["environment"] == {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    assert config["reproducibility"]["seed"] == 42


def test_split_builder_produces_exact_two_direction_contract() -> None:
    names_44b6 = [f"44b6_{index:08d}" for index in range(71)]
    names_6bba = [f"6bba_{index:08d}" for index in range(128)]
    all_names = names_44b6 + names_6bba
    records = extracted_split_builder()(all_names, load_config()["validation"]["outer_folds"], 0)
    assert len(records) == 2
    assert (
        len(records[0]["train"]),
        len(records[0]["test"]),
        len(records[0]["outer_evaluation"]),
    ) == (
        64,
        7,
        128,
    )
    assert (
        len(records[1]["train"]),
        len(records[1]["test"]),
        len(records[1]["outer_evaluation"]),
    ) == (
        116,
        12,
        71,
    )
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


def test_train_notebook_keeps_outer_evaluation_out_of_training() -> None:
    source = notebook_source("train")
    assert "build_embryo_splits" in source
    assert "random.Random(split_seed).shuffle(shuffled)" in source
    assert '"outer_evaluation": evaluation' in source
    assert 'record["train"]' in source
    assert 'record["test"]' in source
    assert "outer evaluation embryo leaked into training" in source
    assert 'method="unet_transformer_smoke_seed42"' in source
    assert 'method="unet_transformer"' in source
    assert "unet_weights=None" in source
    assert "for record in splits:" in source
    assert "smoke_model, smoke_elapsed_seconds = run_with_log(" in source
    assert "trained_model, full_elapsed_seconds = run_with_log(" in source
    assert "del smoke_model" in source
    assert "del trained_model" in source
    assert source.count("torch.cuda.empty_cache()") >= 4
    assert '"model_count": len(model_records)' in source
    assert "control_retrained" in source
    assert "__file__" not in source
    assert "TODO" not in source


def test_allocator_is_configured_before_torch_import_and_recorded() -> None:
    source = notebook_source("train")
    config_index = source.index('allocator_variable = "PYTORCH_ALLOC_CONF"')
    assignment_index = source.index("os.environ[allocator_variable] = allocator_conf")
    torch_import_index = source.index("import torch")
    first_cuda_index = source.index("torch.cuda.get_device_name")
    assert config_index < assignment_index < torch_import_index < first_cuda_index
    assert 'if "torch" in sys.modules' in source
    assert 'runtime_cfg["environment"][allocator_variable]' in source
    assert '"peak_allocated_gib_by_device"' in source
    assert '"peak_reserved_gib_by_device"' in source


def test_inference_notebook_saves_full_candidate_masks_without_submission() -> None:
    source = notebook_source("inference")
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
    assert 'find_competition_dir("train")' in source
    assert 'find_competition_dir("test")' not in source
    assert "submission.csv" not in source
    assert "competitions submit" not in source
    assert "__file__" not in source
    assert "TODO" not in source


def test_inference_uses_two_fold_bundle_and_official_aggregate_twice() -> None:
    source = notebook_source("inference")
    assert 'payload.get("manifest_type") == "embryo_holdout_model_bundle"' in source
    assert 'payload.get("evidence", {}).get("artifacts", {}).get("model_count") == 2' in source
    assert "set(fold_models) != {0, 1}" in source
    assert 'fold_manifest["outer_evaluation_used_for_selection"]' in source
    assert source.count("evaluate_pairs(") == 2
    assert source.count("summarise(") >= 3
    assert '"recomputed_summary_matches": True' in source
    assert "set(predicted_names) != set(paired_stems)" in source
    assert (
        "an average of the two embryo scores"
        in (EXP_DIR / f"{EXPERIMENT}_inference.py").read_text()
    )


def test_inference_kernel_depends_only_on_exp005_train_kernel() -> None:
    kernel_sources = load_config()["runtime"]["kaggle"]["inference"]["kernel_sources"]
    assert kernel_sources == ["kentookumura/exp005-embryo-holdout-batch8-train"]


def test_candidate_recording_config_is_explicit() -> None:
    inference = load_config()["model"]["inference"]
    assert inference["det_threshold"] == 0.99
    assert inference["edge_threshold"] == 0.5
    assert inference["edge_activation"] == "softmax"
    assert inference["use_ilp"] is True
    assert inference["unet_batch_size"] == 4
    for key in (
        "save_detection_scores",
        "save_all_edge_scores",
        "save_edge_valid_mask",
        "save_edge_threshold_mask",
        "save_graph_input_mask",
        "save_final_selection_mask",
    ):
        assert inference[key] is True
    assert inference["candidate_cache_format"] == "npz"
    assert inference["runtime_gate_hours"] == 12
    assert inference["smoke_videos_per_fold"] == 1


def test_numerical_stack_is_imported_after_offline_install() -> None:
    for kind in ("train", "inference"):
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
    for kind in ("train", "inference"):
        source = (EXP_DIR / f"{EXPERIMENT}_{kind}.py").read_text()
        assert "# ## Contents" in source
        assert "# ## 1." in source
        assert "# ## 2." in source
        assert "# ## 3." in source
        assert len(source.splitlines()) > 400
