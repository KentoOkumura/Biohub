# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp002 TemporalUNet3D expandable-segments inference
#
# This notebook accepts only the checkpoint and manifest produced by the exp002
# train kernel. It dynamically enumerates every hidden test dataset, runs the
# organizer prediction path, and validates the variable-row graph CSV.
# It creates submission.csv but does not submit it.

# %% [markdown]
# ## 1. Configuration and integrity helpers

# %%
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT = "exp002_unet3d_expandable_segments"
WORKING_ROOT = Path.cwd()
CONFIG_PATH = WORKING_ROOT / "config.yaml"
SOURCE_ROOT = WORKING_ROOT / "official_source"
SOURCE_MANIFEST_PATH = SOURCE_ROOT / "SOURCE.json"
METRICS_PATH = WORKING_ROOT / "metrics.json"
SUBMISSION_PATH = WORKING_ROOT / "submission.csv"
INFERENCE_SUMMARY_PATH = WORKING_ROOT / "inference_summary.json"

config = yaml.safe_load(CONFIG_PATH.read_text())
inference_cfg = config["model"]["inference"]
runtime_cfg = config["runtime"]
started = time.monotonic()

print("Experiment:", EXPERIMENT)
print("Source commit:", config["source"]["commit"])
print("Inference config:", json.dumps(inference_cfg, indent=2))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def find_competition_dir(child: str) -> Path:
    slug = "biohub-cell-tracking-during-development"
    candidates = [
        Path("/kaggle/input/competitions") / slug / child,
        Path("/kaggle/input") / slug / child,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    matches = sorted(
        path
        for path in Path("/kaggle/input").rglob(child)
        if path.is_dir() and any(path.glob("*.zarr")) and slug in str(path)
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one competition {child} directory, found {matches}")
    return matches[0]


def find_wheels_dir() -> Path:
    candidates = sorted(
        path
        for path in Path("/kaggle/input").rglob("wheels")
        if path.is_dir() and any(path.glob("tracksdata-*.whl"))
    )
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one offline wheels directory, found {candidates}")
    return candidates[0]


def verify_source() -> dict[str, Any]:
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text())
    if manifest["commit"] != config["source"]["commit"]:
        raise RuntimeError("source commit differs from config.yaml")
    mismatches = {}
    for relative, expected in manifest["files"].items():
        observed = sha256_file(SOURCE_ROOT / relative)
        if observed != expected:
            mismatches[relative] = {"expected": expected, "observed": observed}
    if mismatches:
        raise RuntimeError(f"pinned source hash mismatch: {mismatches}")
    print(f"Verified {len(manifest['files'])} pinned source files.")
    return manifest


def load_unique_train_manifest() -> tuple[Path, dict[str, Any]]:
    candidates = []
    for path in Path("/kaggle/input").rglob("model_manifest.json"):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("experiment") == EXPERIMENT:
            candidates.append((path, payload))
    if len(candidates) != 1:
        paths = [str(path) for path, _ in candidates]
        raise FileNotFoundError(f"expected one exp002 training model manifest, found {paths}")
    return candidates[0]


def load_train_metrics() -> dict[str, Any]:
    candidates = []
    for path in Path("/kaggle/input").rglob("metrics.json"):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if (
            payload.get("experiment") == EXPERIMENT
            and payload.get("evidence", {}).get("artifacts", {}).get("model_count") == 1
        ):
            candidates.append(payload)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one exp002 train metrics record, found {len(candidates)}"
        )
    return candidates[0]


# %% [markdown]
# ## 2. Runtime, offline dependencies, and pinned source

# %%
if not Path("/kaggle/input").is_dir():
    raise RuntimeError("The authoritative run must execute on Kaggle.")
wheels_dir = find_wheels_dir()
offline_packages = [
    "bidict==0.23.1",
    "donfig==0.8.1.post1",
    "geff==1.2.0.1.1",
    "geff-spec==1.1.1",
    "ilpy==0.6.0",
    "numcodecs==0.15.1",
    "polars==1.42.0",
    "polars-runtime-32==1.42.0",
    "pyscipopt==6.2.1",
    "rustworkx==0.18.0",
    "tracksdata==0.1.0rc6.dev3+g980c2d30a",
    "zarr==3.2.1",
]
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(wheels_dir),
        "--no-deps",
        *offline_packages,
    ],
    check=True,
)

# Keep the running kernel free of mixed NumPy modules. The exact no-deps list
# omits the bundled imagecodecs release because it would upgrade NumPy in place.
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import torch  # noqa: E402

gpu_names = [
    torch.cuda.get_device_name(device_index) for device_index in range(torch.cuda.device_count())
]
if not gpu_names:
    raise RuntimeError("GPU inference is required.")
expected_name = str(runtime_cfg["expected_gpu_name_substring"])
if expected_name.lower() not in gpu_names[0].lower():
    raise RuntimeError(f"expected a T4 inference GPU, found {gpu_names}")
print("Numerical stack:", {"numpy": np.__version__, "scipy": scipy.__version__})
print("Visible GPUs:", gpu_names)

source_manifest = verify_source()
source_manifest_sha = sha256_file(SOURCE_MANIFEST_PATH)
sys.path.insert(0, str(SOURCE_ROOT / "src"))
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))

import tracksdata as td  # noqa: E402
from predict_unet_transformer import PredictConfig, predict  # noqa: E402

# %% [markdown]
# ## 3. Resolve the scratch-trained checkpoint
#
# The public checkpoint in the dependency dataset is not searched or copied.
# The model manifest from the attached exp002 train kernel is the sole model input.

# %%
train_manifest_path, train_manifest = load_unique_train_manifest()
train_metrics = load_train_metrics()
if not train_manifest.get("trained_from_scratch"):
    raise RuntimeError("training manifest does not attest scratch training")
if train_manifest.get("upstream_checkpoint_loaded"):
    raise RuntimeError("training manifest indicates an upstream checkpoint was loaded")
if train_manifest.get("source_commit") != source_manifest["commit"]:
    raise RuntimeError("training and inference source commits differ")
if int(train_manifest.get("epochs", -1)) != int(config["model"]["training"]["epochs"]):
    raise RuntimeError("training epoch count differs from the experiment contract")

train_checkpoint = train_manifest_path.parent / train_manifest["checkpoint"]
train_model_config = train_manifest_path.parent / train_manifest["model_config"]
if sha256_file(train_checkpoint) != train_manifest["checkpoint_sha256"]:
    raise RuntimeError("training checkpoint SHA-256 mismatch")
if sha256_file(train_model_config) != train_manifest["model_config_sha256"]:
    raise RuntimeError("training model config SHA-256 mismatch")

runtime_weights_dir = SOURCE_ROOT / "weights" / "unet_transformer" / "split_0"
runtime_weights_dir.mkdir(parents=True, exist_ok=True)
runtime_checkpoint = runtime_weights_dir / "edge_predictor_best.pth"
runtime_model_config = runtime_weights_dir / "config.json"
runtime_checkpoint.write_bytes(train_checkpoint.read_bytes())
runtime_model_config.write_bytes(train_model_config.read_bytes())
if sha256_file(runtime_checkpoint) != train_manifest["checkpoint_sha256"]:
    raise RuntimeError("copied checkpoint SHA-256 mismatch")

print("Training manifest:", train_manifest_path)
print("Checkpoint SHA-256:", train_manifest["checkpoint_sha256"])

# %% [markdown]
# ## 4. Dynamic hidden-test discovery and organizer inference
#
# Every test Zarr present at runtime is included in a one-fold test manifest.
# The public inference Notebook settings are fixed: threshold 0.99, UNet batch 4,
# and ILP linking with the published weights.

# %%
test_dir = find_competition_dir("test")
test_stems = sorted(path.name[:-5] for path in test_dir.glob("*.zarr"))
if not test_stems:
    raise RuntimeError(f"no test Zarr datasets found in {test_dir}")
if len(test_stems) != len(set(test_stems)):
    raise RuntimeError("duplicate test dataset names")

test_splits_path = SOURCE_ROOT / "kaggle_test_splits.json"
test_splits_path.write_text(
    json.dumps([{"split": 0, "train": [], "test": test_stems}], indent=2) + "\n"
)
method = "exp002_submission"
predict_config = PredictConfig(
    det_threshold=float(inference_cfg["det_threshold"]),
    det_tta=bool(inference_cfg["det_tta"]),
    pool_kernel_um=float(inference_cfg["pool_kernel_um"]),
    edge_activation=str(inference_cfg["edge_activation"]),
    threshold=float(inference_cfg["edge_threshold"]),
    use_ilp=bool(inference_cfg["use_ilp"]),
    ilp_edge_weight=float(inference_cfg["ilp_edge_weight"]),
    ilp_appearance_weight=float(inference_cfg["ilp_appearance_weight"]),
    ilp_disappearance_weight=float(inference_cfg["ilp_disappearance_weight"]),
    ilp_division_weight=float(inference_cfg["ilp_division_weight"]),
)

predict(
    data_dir=test_dir,
    fold=0,
    splits_file=test_splits_path,
    weights_path=runtime_checkpoint,
    cfg=predict_config,
    method=method,
    debug_video=None,
    unet_batch_size=int(inference_cfg["unet_batch_size"]),
    video_slice=None,
    evaluate=False,
)

prediction_dirs = sorted(
    path for path in (SOURCE_ROOT / "predictions").rglob("split_0") if path.parent.name == method
)
if len(prediction_dirs) != 1:
    raise FileNotFoundError(
        f"expected one prediction directory for {method}, found {prediction_dirs}"
    )
geff_paths = sorted(prediction_dirs[0].glob("*.geff"))
if [path.stem for path in geff_paths] != test_stems:
    raise RuntimeError("prediction graph names do not match the runtime test datasets")
print(f"Generated {len(geff_paths)} GEFF prediction graphs.")

# %% [markdown]
# ## 5. Build and validate the variable-row graph CSV

# %%
columns = [
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]
rows: list[dict[str, Any]] = []
for geff_path in geff_paths:
    dataset_name = geff_path.stem
    graph = td.graph.IndexedRXGraph.from_geff(geff_path)
    graph = graph[0] if isinstance(graph, tuple) else graph
    for record in graph.node_attrs().iter_rows(named=True):
        rows.append(
            {
                "dataset": dataset_name,
                "row_type": "node",
                "node_id": int(record["node_id"]),
                "t": int(record["t"]),
                "z": int(round(record["z"])),
                "y": int(round(record["y"])),
                "x": int(round(record["x"])),
                "source_id": -1,
                "target_id": -1,
            }
        )
    for record in graph.edge_attrs().iter_rows(named=True):
        rows.append(
            {
                "dataset": dataset_name,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": int(record["source_id"]),
                "target_id": int(record["target_id"]),
            }
        )

submission = pd.DataFrame(rows, columns=columns)
if submission.empty:
    raise RuntimeError("submission has no node or edge rows")
if set(submission["dataset"]) != set(test_stems):
    raise RuntimeError("submission datasets do not match hidden test discovery")
if not set(submission["row_type"]).issubset({"node", "edge"}):
    raise RuntimeError("submission contains an invalid row_type")

node_rows = submission[submission["row_type"] == "node"]
edge_rows = submission[submission["row_type"] == "edge"]
if node_rows.duplicated(["dataset", "node_id"]).any():
    raise RuntimeError("duplicate node_id within a dataset")
for dataset_name in test_stems:
    node_ids = set(node_rows.loc[node_rows["dataset"] == dataset_name, "node_id"].astype(int))
    dataset_edges = edge_rows[edge_rows["dataset"] == dataset_name]
    referenced = set(dataset_edges["source_id"].astype(int)) | set(
        dataset_edges["target_id"].astype(int)
    )
    if not referenced.issubset(node_ids):
        raise RuntimeError(f"edge references a missing node in {dataset_name}")

submission.index.name = "id"
submission.to_csv(SUBMISSION_PATH)
reloaded = pd.read_csv(SUBMISSION_PATH)
expected_csv_columns = ["id", *columns]
if list(reloaded.columns) != expected_csv_columns:
    raise RuntimeError(
        f"CSV columns differ: expected {expected_csv_columns}, got {list(reloaded.columns)}"
    )
if reloaded["id"].duplicated().any() or reloaded.isna().any().any():
    raise RuntimeError("submission contains duplicate ids or missing values")
numeric_columns = ["id", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
if not np.isfinite(reloaded[numeric_columns].to_numpy(dtype=float)).all():
    raise RuntimeError("submission contains non-finite numeric values")

sample_candidates = sorted(Path("/kaggle/input").rglob("sample_submission.csv"))
if sample_candidates:
    sample_columns = list(pd.read_csv(sample_candidates[0], nrows=1).columns)
    if sample_columns != expected_csv_columns:
        raise RuntimeError(f"sample submission schema differs: {sample_columns}")

submission_sha = sha256_file(SUBMISSION_PATH)
elapsed_seconds = time.monotonic() - started
inference_summary = {
    "experiment": EXPERIMENT,
    "source_commit": source_manifest["commit"],
    "source_manifest_sha256": source_manifest_sha,
    "checkpoint_sha256": train_manifest["checkpoint_sha256"],
    "test_dataset_count": len(test_stems),
    "test_datasets": test_stems,
    "row_count": len(submission),
    "node_row_count": len(node_rows),
    "edge_row_count": len(edge_rows),
    "submission_sha256": submission_sha,
    "elapsed_seconds": elapsed_seconds,
    "settings": inference_cfg,
}
atomic_json(INFERENCE_SUMMARY_PATH, inference_summary)

metrics = deep_merge(
    train_metrics,
    {
        "experiment": EXPERIMENT,
        "status": "running",
        "updated_at": datetime.now(UTC).isoformat(),
        "public_lb": None,
        "private_lb": None,
        "evidence": {
            "kaggle": {
                "kernel_id": ("kentookumura/exp002-unet3d-expandable-segments-inference"),
                "resource": {
                    "gpu_names": gpu_names,
                    "visible_gpu_count": len(gpu_names),
                },
                "notebook_runtime_seconds": elapsed_seconds,
                "internet_enabled": False,
                "kernel_source_ids": ["kentookumura/exp002-unet3d-expandable-segments-train"],
            },
            "artifacts": {
                "source_manifest_sha": source_manifest_sha,
                "model_manifest_sha": sha256_file(train_manifest_path),
                "model_count": 1,
                "model_shas": {"edge_predictor_best.pth": train_manifest["checkpoint_sha256"]},
                "selected_mode": "scratch_three_epochs",
                "selected_model": "edge_predictor_best.pth",
                "test_prediction_content_sha": submission_sha,
                "submission_sha": submission_sha,
                "row_count": len(submission),
                "group_count": len(test_stems),
            },
            "submission_validation": {
                "passed": True,
                "checked_at": datetime.now(UTC).isoformat(),
                "row_count": len(submission),
                "id_column": "id",
                "target_columns": columns,
                "duplicate_id_count": 0,
                "missing_value_count": 0,
                "infinite_value_count": 0,
                "errors": [],
            },
        },
        "notes": (
            "Inference and CSV validation completed in this notebook. Competition submission "
            "is performed separately after repository submit-check."
        ),
    },
)
atomic_json(METRICS_PATH, metrics)

print(json.dumps(inference_summary, indent=2))
print("Created and validated:", SUBMISSION_PATH)
print("No Kaggle competition submission was made.")

# %% [markdown]
# ## 6. Output contract

# %%
required_outputs = [SUBMISSION_PATH, INFERENCE_SUMMARY_PATH, METRICS_PATH]
missing_outputs = [str(path) for path in required_outputs if not path.is_file()]
if missing_outputs:
    raise FileNotFoundError(f"missing inference outputs: {missing_outputs}")
print("Inference outputs are ready for repository submit-check.")
