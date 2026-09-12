from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

EXP_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXP_DIR / "config.yaml"
PARENT_CONFIG_PATH = EXP_DIR.parent / "exp006_embryo_holdout_seed314159" / "config.yaml"
SOURCE_ROOT = EXP_DIR / "official_source"
SOURCE_MANIFEST_PATH = SOURCE_ROOT / "SOURCE.json"
EXPERIMENT = "exp009_exp006_twofold_ensemble"
SOURCE_COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text())
    assert isinstance(value, dict)
    return value


def notebook_source() -> str:
    path = EXP_DIR / f"{EXPERIMENT}_inference.ipynb"
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_only_inference_notebook_is_executable() -> None:
    assert (EXP_DIR / f"{EXPERIMENT}_inference.py").is_file()
    assert (EXP_DIR / f"{EXPERIMENT}_inference.ipynb").is_file()
    assert not (EXP_DIR / f"{EXPERIMENT}_train.py").exists()
    train_notebook = json.loads((EXP_DIR / f"{EXPERIMENT}_train.ipynb").read_text())
    train_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in train_notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "exp009 is inference-only" in train_code
    assert "RuntimeError" in train_code


def test_config_reuses_exp006_models_and_changes_only_inference_policy() -> None:
    config = load_config()
    parent = yaml.safe_load(PARENT_CONFIG_PATH.read_text())
    assert config["lineage"]["parent"] == "exp006_embryo_holdout_seed314159"
    assert config["lineage"]["hypothesis_id"] == "N/A"
    assert config["lineage"]["backlog_candidate"] == "N/A"
    assert config["experiment"]["route"] == "temporal_unet3d_twofold_probability_ensemble"
    assert config["source"] == parent["source"]
    assert config["model"]["name"] == parent["model"]["name"]
    assert config["model"]["params"] == parent["model"]["params"]
    assert config["model"]["training"] == parent["model"]["training"]
    for key in (
        "det_threshold",
        "det_tta",
        "pool_kernel_um",
        "edge_activation",
        "edge_threshold",
        "use_ilp",
        "ilp_edge_weight",
        "ilp_appearance_weight",
        "ilp_disappearance_weight",
        "ilp_division_weight",
        "unet_batch_size",
    ):
        assert config["model"]["inference"][key] == parent["model"]["inference"][key]
    assert config["runtime"]["kaggle"]["inference"]["kernel_sources"] == [
        "kentookumura/exp006-embryo-holdout-seed314159-train"
    ]


def test_twofold_probability_mean_and_single_decode_are_fixed() -> None:
    ensemble = load_config()["model"]["inference"]["ensemble"]
    assert ensemble == {
        "folds": [0, 1],
        "detection_probability": "arithmetic_mean",
        "detection_peak_extraction": "after_probability_mean",
        "edge_probability": "arithmetic_mean_on_shared_candidates",
        "decode_count": 1,
        "device_by_fold": {0: "cuda:0", 1: "cuda:1"},
    }
    source = notebook_source()
    assert "detect_cells_from_mean_probability" in source
    assert "mean_edge_probabilities" in source
    assert "torch.stack(probabilities).mean(dim=0)" in source
    assert "graph = solver.solve(graph)" in source
    assert source.count("graph = solver.solve(graph)") == 1
    assert "finished_graph" not in source
    assert "graph_union" not in source


def test_notebook_uses_parent_manifest_and_dynamic_hidden_test() -> None:
    source = notebook_source()
    assert 'payload.get("experiment") == PARENT_EXPERIMENT' in source
    assert 'rglob("model_manifest.json")' in source
    assert 'glob("*.zarr")' in source
    assert "find_sample_submission" in source
    assert "sample_columns != expected_csv_columns" in source
    assert "submission.to_csv(SUBMISSION_PATH)" in source
    assert "No Kaggle competition submission was made." in source
    assert "competitions submit" not in source
    assert "__file__" not in source
    assert "TODO" not in source


def test_fold_models_are_pinned_to_separate_t4_devices() -> None:
    source = notebook_source()
    assert 'torch.device("cuda:0")' in source
    assert 'torch.device("cuda:1")' in source
    assert "expected_cuda_device_count" in source
    assert "expected_gpu_name_substring" in source
    assert "fold window sizes differ" in source
    assert "fold downsample factors differ" in source


def test_pinned_official_source_matches_manifest_and_is_bootstrapped() -> None:
    config = load_config()
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text())
    assert manifest["commit"] == SOURCE_COMMIT
    assert config["source"]["commit"] == SOURCE_COMMIT
    for relative, expected in manifest["files"].items():
        path = SOURCE_ROOT / relative
        assert path.is_file(), relative
        assert sha256(path) == expected, relative
    bootstrap = set(config["runtime"]["kaggle"]["bootstrap_files"])
    expected_bootstrap = {f"official_source/{relative}" for relative in manifest["files"]}
    expected_bootstrap.add("official_source/SOURCE.json")
    assert expected_bootstrap.issubset(bootstrap)
    assert not list(SOURCE_ROOT.rglob("*.pth"))


def test_inference_source_is_valid_and_self_contained() -> None:
    source_path = EXP_DIR / f"{EXPERIMENT}_inference.py"
    source = source_path.read_text()
    ast.parse(source)
    headings = [
        "# ## 1. Configuration, hashing, and input resolution",
        "# ## 2. Runtime, offline dependencies, and pinned source",
        "# ## 3. Resolve and verify both exp006 fold models",
        "# ## 4. Two-fold probability ensemble helpers",
        "# ## 5. Dynamic hidden-test inference",
        "# ## 6. Build and validate the variable-row graph CSV",
        "# ## 7. Metrics and output contract",
    ]
    for heading in headings:
        assert heading in source
    assert "from exp009_exp006_twofold_ensemble" not in source
    assert "__file__" not in source
