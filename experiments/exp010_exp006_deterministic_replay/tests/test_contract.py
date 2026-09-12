from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import yaml

EXP_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT = "exp010_exp006_deterministic_replay"
CONFIG_PATH = EXP_DIR / "config.yaml"
SOURCE_ROOT = EXP_DIR / "official_source"
SOURCE_MANIFEST_PATH = SOURCE_ROOT / "SOURCE.json"


def load_config() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text())
    assert isinstance(payload, dict)
    return payload


def notebook_source(kind: str) -> str:
    notebook = json.loads((EXP_DIR / f"{EXPERIMENT}_{kind}.ipynb").read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_lineage_and_full_replay_contract() -> None:
    config = load_config()
    assert config["lineage"]["parent"] == "exp006_embryo_holdout_seed314159"
    assert config["lineage"]["hypothesis_id"] == "N/A"
    assert config["lineage"]["backlog_candidate"] == "N/A"
    assert config["validation"]["strategy"] == "leave_one_embryo_out"
    assert config["validation"]["expected_sample_count"] == 199
    assert config["validation"]["n_folds"] == 2
    assert config["model"]["training"]["epochs"] == 3
    assert config["model"]["training"]["batch_size"] == 8
    comparison = config["model"]["comparison"]
    assert comparison["kind"] == "exact_replay"
    assert comparison["required_full_runs"] == 2


def test_deterministic_runtime_and_rng_contract() -> None:
    config = load_config()
    training = config["model"]["training"]
    reproducibility = config["reproducibility"]
    assert training["augmentation_seed_key"] == [
        "global_seed",
        "epoch_index",
        "dataset_item_index",
    ]
    assert training["deterministic_algorithms"] is True
    assert reproducibility["deterministic_anchor"] is False
    assert config["runtime"]["environment"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    source = (SOURCE_ROOT / "scripts/train_unet_transformer.py").read_text()
    assert "class DeterministicEpochSampler" in source
    assert "torch.randperm(len(self.data_source), generator=generator)" in source
    assert "np.random.SeedSequence([self.augmentation_seed, epoch, idx])" in source
    assert "augmentation_seed is required when augmentations are enabled" in source
    assert "np.random.default_rng()" not in source
    assert "shuffle=train_sampler is None" in source
    assert "sampler=train_sampler" in source

    model_source = (SOURCE_ROOT / "src/tracking_cellmot/models/temporal_unet.py").read_text()
    assert "class _DeterministicMaxPool3d" in model_source
    assert "blocks.max(dim=-1).values" in model_source
    assert "self.pool = _DeterministicMaxPool3d()" in model_source
    assert "nn.MaxPool3d" not in model_source


def test_source_manifest_records_only_the_local_deterministic_patch() -> None:
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text())
    for relative, expected in manifest["files"].items():
        observed = hashlib.sha256((SOURCE_ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected, relative
    patch = manifest["local_experiment_patches"]["scripts/train_unet_transformer.py"]
    assert patch["experiment"] == EXPERIMENT
    assert patch["base_sha256"] == (
        "83e3f30a3313ea452b5072b118bfd06539302cf6886e317c542a602deb27d281"
    )
    model_patch = manifest["local_experiment_patches"][
        "src/tracking_cellmot/models/temporal_unet.py"
    ]
    assert model_patch["experiment"] == EXPERIMENT
    assert model_patch["base_sha256"] == (
        "d809c35d42f504161074ddeaaa7aee5b407e5bca7f9b4e1d5f9b2ff345666cac"
    )


def test_train_source_enables_determinism_before_cuda_work() -> None:
    source = notebook_source("train")
    environment_index = source.index("for variable, value in runtime_environment.items()")
    torch_import_index = source.index("import torch")
    first_cuda_index = source.index("torch.cuda.get_device_name")
    assert environment_index < torch_import_index < first_cuda_index
    assert "torch.use_deterministic_algorithms(True)" in source
    assert 'torch.set_deterministic_debug_mode("error")' in source
    assert "torch.backends.cudnn.deterministic = True" in source
    assert "torch.backends.cudnn.benchmark = False" in source
    assert "torch.backends.cuda.matmul.allow_tf32 = False" in source
    assert "checkpoint_content_sha256" in source
    assert '"augmentation_byte_deterministic": True' in source
    assert "__file__" not in source
    assert "TODO" not in source


def test_inference_records_canonical_prediction_and_metric_hashes() -> None:
    source = notebook_source("inference")
    assert "sha256_state_dict(checkpoint_path)" in source
    assert "final_graph_content_sha" in source
    assert '"graph_content_sha256"' in source
    assert '"candidate_content_sha256"' in source
    assert '"oof_prediction_content_sha256"' in source
    assert '"per_sample_metric_content_sha256"' in source
    assert '"official_metric_summary_content_sha256"' in source
    assert source.count("evaluate_pairs(") == 2
    assert 'find_competition_dir("test")' not in source
    assert "submission.csv" not in source
    assert "competitions submit" not in source


def test_inference_depends_only_on_the_matching_train_kernel() -> None:
    kernel_sources = load_config()["runtime"]["kaggle"]["inference"]["kernel_sources"]
    assert kernel_sources == ["kentookumura/exp010-exp006-deterministic-replay-train"]
    source = notebook_source("inference")
    assert "kentookumura/exp005" not in source
    assert "load_unique_reference_bundle" not in source


def test_all_python_sources_parse() -> None:
    paths = [
        EXP_DIR / f"{EXPERIMENT}_train.py",
        EXP_DIR / f"{EXPERIMENT}_inference.py",
        EXP_DIR / "compare_replays.py",
        SOURCE_ROOT / "scripts/train_unet_transformer.py",
    ]
    for path in paths:
        ast.parse(path.read_text(), filename=str(path))


def test_compare_replays_requires_every_canonical_field_to_match() -> None:
    spec = importlib.util.spec_from_file_location("exp010_compare", EXP_DIR / "compare_replays.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    train = {
        "experiment": EXPERIMENT,
        "config_sha256": "config",
        "source_manifest_sha256": "source",
        "split_manifest_sha256": "split",
        "folds": [
            {
                "fold": 0,
                "checkpoint_content_sha256": "model-0",
                "model_config_sha256": "model-config",
            },
            {
                "fold": 1,
                "checkpoint_content_sha256": "model-1",
                "model_config_sha256": "model-config",
            },
        ],
    }
    inference = {
        "experiment": EXPERIMENT,
        "prediction_count": 199,
        "input_contract_sha256": "input",
        "candidate_content_sha256": "candidate",
        "oof_prediction_content_sha256": "oof",
        "per_sample_metric_content_sha256": "rows",
        "official_metric_summary_content_sha256": "summary",
        "cv": {"score": 0.5},
    }
    matching = module.compare_runs(train, inference, copy.deepcopy(train), copy.deepcopy(inference))
    assert matching["passed"] is True

    changed = copy.deepcopy(inference)
    changed["cv"]["score"] = 0.500001
    mismatch = module.compare_runs(train, inference, copy.deepcopy(train), changed)
    assert mismatch["passed"] is False
    assert mismatch["comparison"]["cv"]["equal"] is False
