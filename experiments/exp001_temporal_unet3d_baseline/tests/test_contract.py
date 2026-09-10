from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

EXP_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXP_DIR / "config.yaml"
SOURCE_ROOT = EXP_DIR / "official_source"
SOURCE_MANIFEST_PATH = SOURCE_ROOT / "SOURCE.json"
EXPERIMENT = "exp001_temporal_unet3d_baseline"
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


def test_training_contract_is_fixed() -> None:
    config = load_config()
    assert config["lineage"] == {
        "parent": "N/A",
        "hypothesis_id": "HYP-20260909-01",
        "backlog_candidate": "official_temporal_unet3d_train_submission_baseline",
        "diff_summary": (
            "First local experiment; adds seed 42 and evidence capture around "
            "the organizer-published three-epoch command."
        ),
    }
    assert config["source"]["commit"] == SOURCE_COMMIT
    assert config["validation"]["strategy"] == "official_seed0_sample_holdout"
    assert config["validation"]["diagnostic_only"] is True
    assert config["model"]["params"] == {
        "in_channels": 1,
        "unet_out_channels": 32,
        "unet_layers": [32, 64, 128],
        "gradient_checkpointing": True,
        "skip_fullres_temporal": True,
        "window_size": 2,
        "downsample": [1, 4, 4],
    }
    training = config["model"]["training"]
    assert training["initialization"] == "random"
    assert training["upstream_checkpoint"] is None
    assert training["epochs"] == 3
    assert training["optimizer"] == "AdamW"
    assert training["learning_rate"] == 1e-4
    assert training["batch_size"] == 16
    assert training["num_workers"] == 8
    assert training["det_loss_weight"] == 1.0
    assert training["det_neg_weight"] == 1e-2
    assert training["augmentations"] == ["brightness_augment", "flip_augment"]
    assert training["data_parallel"] is True
    assert config["model"]["smoke"]["runtime_gate_hours"] == 11
    assert config["runtime"]["expected_cuda_device_count"] == 2
    assert config["runtime"]["kaggle"]["enable_gpu"] is True
    assert config["runtime"]["kaggle"]["enable_internet"] is False
    assert config["runtime"]["kaggle"]["machine_shape"] == "NvidiaTeslaT4"


def test_inference_contract_matches_public_notebook() -> None:
    inference = load_config()["model"]["inference"]
    assert inference == {
        "det_threshold": 0.99,
        "det_tta": True,
        "pool_kernel_um": 3.0,
        "edge_activation": "softmax",
        "edge_threshold": 0.5,
        "use_ilp": True,
        "ilp_edge_weight": -1.0,
        "ilp_appearance_weight": 0.1,
        "ilp_disappearance_weight": 0.1,
        "ilp_division_weight": 1.0,
        "unet_batch_size": 4,
    }


def test_pinned_official_source_matches_manifest() -> None:
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

    train_tree = ast.parse((SOURCE_ROOT / "scripts/train_unet_transformer.py").read_text())
    train_names = {
        node.name
        for node in ast.walk(train_tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert {"UNetNodeTransformer", "train"}.issubset(train_names)

    temporal_tree = ast.parse(
        (SOURCE_ROOT / "src/tracking_cellmot/models/temporal_unet.py").read_text()
    )
    temporal_names = {
        node.name for node in ast.walk(temporal_tree) if isinstance(node, ast.ClassDef)
    }
    assert "TemporalUNet3D" in temporal_names

    transformer_tree = ast.parse(
        (SOURCE_ROOT / "src/tracking_cellmot/models/simple_node_transformer.py").read_text()
    )
    transformer_names = {
        node.name for node in ast.walk(transformer_tree) if isinstance(node, ast.ClassDef)
    }
    assert "SimpleNodeTransformer" in transformer_names


def test_every_pinned_source_file_is_bootstrapped() -> None:
    config = load_config()
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text())
    bootstrap = set(config["runtime"]["kaggle"]["bootstrap_files"])
    expected = {f"official_source/{relative}" for relative in manifest["files"]}
    expected.add("official_source/SOURCE.json")
    assert expected.issubset(bootstrap)
    assert not list(SOURCE_ROOT.rglob("*.pth"))


def test_train_notebook_guards_scratch_training_and_runtime() -> None:
    source = notebook_source("train")
    assert "unet_weights=None" in source
    assert '"epochs"' in source
    assert "runtime_gate_hours" in source
    assert "expected_cuda_device_count" in source
    assert 'random.Random(int(config["validation"]["split_seed"]))' in source
    assert 'method="unet_transformer_smoke_seed42"' in source
    assert 'method="unet_transformer"' in source
    assert "trained_from_scratch" in source
    assert "upstream_checkpoint_loaded" in source
    assert "__file__" not in source
    assert "TODO" not in source


def test_inference_notebook_uses_train_manifest_and_dynamic_test_names() -> None:
    source = notebook_source("inference")
    assert 'rglob("model_manifest.json")' in source
    assert 'glob("*.zarr")' in source
    assert "test_stems" in source
    assert "trained_from_scratch" in source
    assert "upstream_checkpoint_loaded" in source
    assert "submission.to_csv(SUBMISSION_PATH)" in source
    assert "No Kaggle competition submission was made." in source
    assert "competitions submit" not in source
    assert "__file__" not in source
    assert "TODO" not in source


def test_inference_kernel_depends_on_train_kernel() -> None:
    kernel_sources = load_config()["runtime"]["kaggle"]["inference"]["kernel_sources"]
    assert kernel_sources == ["kentookumura/exp001-temporal-unet3d-baseline-train"]


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
