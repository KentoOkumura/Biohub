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
# # exp007 all-epoch checkpoint training
#
# This notebook preserves the exp005 embryo-disjoint training contract and saves
# the model state after every epoch. The internal sample holdout remains the only
# checkpoint-selection data; the outer evaluation embryo never enters training or selection.

# %% [markdown]
# ## Contents
# 1. Configuration and evidence helpers
# 2. Runtime, offline dependencies, and pinned source
# 3. Competition input and embryo-disjoint split construction
# 4. Per-fold smoke runs and combined runtime gate
# 5. Two-fold scratch training and six epoch checkpoints
# 6. Metrics and artifact contract

# %% [markdown]
# ## 1. Configuration and evidence helpers

# %%
from __future__ import annotations

import contextlib
import gc
import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT = "exp007_graph_checkpoint_selection"
TRAIN_KERNEL_ID = "kentookumura/exp007-graph-checkpoint-selection-train"
WORKING_ROOT = Path.cwd()
CONFIG_PATH = WORKING_ROOT / "config.yaml"
SOURCE_ROOT = WORKING_ROOT / "official_source"
SOURCE_MANIFEST_PATH = SOURCE_ROOT / "SOURCE.json"
ARTIFACTS_ROOT = WORKING_ROOT / "artifacts"
MODELS_ROOT = ARTIFACTS_ROOT / "models"
METRICS_PATH = WORKING_ROOT / "metrics.json"

config = yaml.safe_load(CONFIG_PATH.read_text())
validation_cfg = config["validation"]
train_cfg = config["model"]["training"]
smoke_cfg = config["model"]["smoke"]
model_cfg = config["model"]["params"]
runtime_cfg = config["runtime"]

allocator_variable = "PYTORCH_ALLOC_CONF"
allocator_conf = str(runtime_cfg["environment"][allocator_variable])
if "torch" in sys.modules:
    raise RuntimeError(f"{allocator_variable} must be configured before importing torch")
os.environ[allocator_variable] = allocator_conf

print("Experiment:", EXPERIMENT)
print("Source commit:", config["source"]["commit"])
print("Outer folds:", json.dumps(validation_cfg["outer_folds"], indent=2))
print("Training config:", json.dumps(train_cfg, indent=2))
print("PyTorch allocator config:", {allocator_variable: os.environ[allocator_variable]})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def update_metrics(update: dict[str, Any]) -> dict[str, Any]:
    current = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    merged = deep_merge(current, update)
    atomic_json(METRICS_PATH, merged)
    return merged


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def embryo_id(sample_name: str) -> str:
    if "_" not in sample_name:
        raise ValueError(f"sample name has no embryo prefix: {sample_name}")
    return sample_name.split("_", 1)[0]


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
    observed = {relative: sha256_file(SOURCE_ROOT / relative) for relative in manifest["files"]}
    mismatches = {
        relative: {"expected": manifest["files"][relative], "observed": digest}
        for relative, digest in observed.items()
        if digest != manifest["files"][relative]
    }
    if mismatches:
        raise RuntimeError(f"pinned source hash mismatch: {mismatches}")
    print(f"Verified {len(observed)} pinned source files.")
    return manifest


class Tee(io.TextIOBase):
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def run_with_log(function: Any, log_path: Path, **kwargs: Any) -> tuple[Any, float]:
    stdout = sys.stdout
    stderr = sys.stderr
    buffer = io.StringIO()
    started = time.monotonic()
    try:
        with (
            contextlib.redirect_stdout(Tee(stdout, buffer)),
            contextlib.redirect_stderr(Tee(stderr, buffer)),
        ):
            result = function(**kwargs)
    finally:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(buffer.getvalue())
    return result, time.monotonic() - started


def build_embryo_splits(
    sample_names: list[str],
    fold_specs: list[dict[str, Any]],
    split_seed: int,
) -> list[dict[str, Any]]:
    sample_set = set(sample_names)
    if len(sample_set) != len(sample_names):
        raise RuntimeError("duplicate sample names")
    records: list[dict[str, Any]] = []
    outer_evaluation_seen: set[str] = set()
    for spec in fold_specs:
        fold = int(spec["fold"])
        train_embryo = str(spec["train_embryo"])
        evaluation_embryo = str(spec["evaluation_embryo"])
        train_pool = sorted(name for name in sample_names if embryo_id(name) == train_embryo)
        evaluation = sorted(name for name in sample_names if embryo_id(name) == evaluation_embryo)
        shuffled = list(train_pool)
        random.Random(split_seed).shuffle(shuffled)
        n_selection = max(1, len(shuffled) // 10)
        checkpoint_selection = shuffled[:n_selection]
        gradient_update = shuffled[n_selection:]
        record = {
            "split": fold,
            "outer_train_embryo": train_embryo,
            "outer_evaluation_embryo": evaluation_embryo,
            "train": gradient_update,
            "test": checkpoint_selection,
            "outer_evaluation": evaluation,
        }
        expected = {
            "train": int(spec["expected_gradient_update_count"]),
            "test": int(spec["expected_checkpoint_selection_count"]),
            "outer_evaluation": int(spec["expected_evaluation_count"]),
        }
        observed = {key: len(record[key]) for key in expected}
        if observed != expected:
            raise RuntimeError(f"fold {fold} count mismatch: expected {expected}, found {observed}")
        if set(gradient_update) & set(checkpoint_selection):
            raise RuntimeError(f"fold {fold} internal split overlaps")
        if (set(gradient_update) | set(checkpoint_selection)) != set(train_pool):
            raise RuntimeError(f"fold {fold} internal split does not cover its training embryo")
        if (set(gradient_update) | set(checkpoint_selection)) & set(evaluation):
            raise RuntimeError(f"fold {fold} outer evaluation embryo leaked into training")
        if any(embryo_id(name) != train_embryo for name in gradient_update + checkpoint_selection):
            raise RuntimeError(f"fold {fold} contains the wrong training embryo")
        if any(embryo_id(name) != evaluation_embryo for name in evaluation):
            raise RuntimeError(f"fold {fold} contains the wrong evaluation embryo")
        if outer_evaluation_seen & set(evaluation):
            raise RuntimeError("a sample appears in more than one outer evaluation fold")
        outer_evaluation_seen.update(evaluation)
        records.append(record)
    if outer_evaluation_seen != sample_set:
        missing = sorted(sample_set - outer_evaluation_seen)
        extra = sorted(outer_evaluation_seen - sample_set)
        raise RuntimeError(
            f"outer evaluation folds do not cover all samples: missing={missing}, extra={extra}"
        )
    return records


def parse_epoch_records(log_path: Path, expected_epochs: int) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+\|\s+edge=([0-9.]+)\s+\|\s+"
        r"det=([0-9.]+)\s+\|\s+test_loss=([0-9.]+)\s+\|\s+"
        r"acc=([0-9.]+)\s+\|\s+recall=([0-9.]+)\s+\|\s+"
        r"best=([0-9.]+).*train=([0-9.]+)s test=([0-9.]+)s"
    )
    records = [
        {
            "epoch_index": int(match.group(1)),
            "n_epochs": int(match.group(2)),
            "edge_loss": float(match.group(3)),
            "detection_loss": float(match.group(4)),
            "validation_edge_loss": float(match.group(5)),
            "validation_edge_accuracy": float(match.group(6)),
            "validation_node_recall": float(match.group(7)),
            "selection_score": float(match.group(8)),
            "train_seconds": float(match.group(9)),
            "validation_seconds": float(match.group(10)),
        }
        for match in pattern.finditer(log_path.read_text())
    ]
    if len(records) != expected_epochs:
        raise RuntimeError(f"expected {expected_epochs} epoch records, found {len(records)}")
    return records


def normalized_model_state(model: Any) -> dict[str, Any]:
    return {
        key.replace("unet.module.", "unet.", 1): value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def run_training_with_epoch_checkpoints(
    log_path: Path,
    checkpoint_dir: Path,
    expected_epoch_indices: list[int],
    **kwargs: Any,
) -> tuple[Any, float, list[dict[str, Any]]]:
    if expected_epoch_indices != list(range(len(expected_epoch_indices))):
        raise RuntimeError("save_epoch_indices must be contiguous and zero-based")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    original_evaluate = train_module.evaluate
    captured: list[dict[str, Any]] = []

    def evaluate_and_capture(
        model: Any,
        loader: Any,
        device: Any,
        pool_kernel_um: float = 5.0,
    ) -> tuple[float, float, float]:
        metrics = original_evaluate(
            model,
            loader,
            device,
            pool_kernel_um=pool_kernel_um,
        )
        epoch_index = len(captured)
        if epoch_index not in expected_epoch_indices:
            raise RuntimeError(f"unexpected epoch evaluation index {epoch_index}")
        checkpoint_path = checkpoint_dir / f"epoch_{epoch_index}.pth"
        state = normalized_model_state(model)
        torch.save(state, checkpoint_path)
        del state
        captured.append(
            {
                "epoch_index": epoch_index,
                "checkpoint": checkpoint_path.name,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "validation_edge_loss": float(metrics[0]),
                "validation_edge_accuracy": float(metrics[1]),
                "validation_node_recall": float(metrics[2]),
                "proxy_selection_score": float(metrics[1] * metrics[2]),
            }
        )
        return metrics

    train_module.evaluate = evaluate_and_capture
    try:
        trained_model, elapsed_seconds = run_with_log(
            train_module.train,
            log_path,
            **kwargs,
        )
    finally:
        train_module.evaluate = original_evaluate

    observed_epoch_indices = [item["epoch_index"] for item in captured]
    if observed_epoch_indices != expected_epoch_indices:
        raise RuntimeError(
            f"expected checkpoint epochs {expected_epoch_indices}, found {observed_epoch_indices}"
        )
    return trained_model, elapsed_seconds, captured


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

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

expected_gpus = int(runtime_cfg["expected_cuda_device_count"])
gpu_names = [
    torch.cuda.get_device_name(device_index) for device_index in range(torch.cuda.device_count())
]
if len(gpu_names) != expected_gpus:
    raise RuntimeError(f"expected {expected_gpus} visible GPUs for T4 x2, found {gpu_names}")
expected_name = str(runtime_cfg["expected_gpu_name_substring"])
if not all(expected_name.lower() in name.lower() for name in gpu_names):
    raise RuntimeError(f"expected T4 GPUs, found {gpu_names}")
print("Numerical stack:", {"numpy": np.__version__, "scipy": scipy.__version__})
print("Visible GPUs:", gpu_names)
print("Active PyTorch allocator config:", os.environ[allocator_variable])

source_manifest = verify_source()
source_manifest_sha = sha256_file(SOURCE_MANIFEST_PATH)
sys.path.insert(0, str(SOURCE_ROOT / "src"))
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))

import train_unet_transformer as train_module  # noqa: E402
from train_unet_transformer import DEFAULT_AUGMENTATIONS, load_dataset_windows, train  # noqa: E402

# %% [markdown]
# ## 3. Competition input and embryo-disjoint split construction
#
# The fixed training function sees only `train` and `test` from each record.
# Both lists contain the outer training embryo. `outer_evaluation` is recorded for
# evaluation and is never passed as training or checkpoint-selection data.

# %%
train_dir = find_competition_dir("train")
zarr_paths = sorted(train_dir.glob("*.zarr"))
paired_stems = [
    path.name[:-5] for path in zarr_paths if (train_dir / f"{path.name[:-5]}.geff").exists()
]
expected_samples = int(validation_cfg["expected_sample_count"])
if len(paired_stems) != expected_samples:
    raise RuntimeError(f"expected {expected_samples} Zarr/GEFF pairs, found {len(paired_stems)}")

embryo_counts: dict[str, int] = {}
for name in paired_stems:
    embryo_counts[embryo_id(name)] = embryo_counts.get(embryo_id(name), 0) + 1
expected_embryo_counts = {
    str(key): int(value) for key, value in validation_cfg["expected_embryo_counts"].items()
}
if embryo_counts != expected_embryo_counts:
    raise RuntimeError(
        f"embryo counts differ: expected {expected_embryo_counts}, found {embryo_counts}"
    )

splits = build_embryo_splits(
    paired_stems,
    validation_cfg["outer_folds"],
    int(validation_cfg["internal_split_seed"]),
)
split_path = SOURCE_ROOT / "dataset_splits.json"
split_path.write_text(json.dumps(splits, indent=2) + "\n")
ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
artifact_split_path = ARTIFACTS_ROOT / "splits.json"
artifact_split_path.write_text(split_path.read_text())

dataset_index = {
    "sample_count": len(paired_stems),
    "paired_stems": paired_stems,
    "embryo_counts": embryo_counts,
    "split_seed": int(validation_cfg["internal_split_seed"]),
}
dataset_index_path = ARTIFACTS_ROOT / "dataset_index.json"
atomic_json(dataset_index_path, dataset_index)

print("Training directory:", train_dir)
for record in splits:
    print(
        f"Fold {record['split']}: {record['outer_train_embryo']} -> "
        f"{record['outer_evaluation_embryo']} | {len(record['train'])} gradient / "
        f"{len(record['test'])} checkpoint-selection / "
        f"{len(record['outer_evaluation'])} outer-evaluation"
    )
print("Split SHA-256:", sha256_file(artifact_split_path))

# %% [markdown]
# ## 4. Per-fold smoke runs and combined runtime gate
#
# Each smoke uses the first name-sorted gradient-update video in its outer fold,
# the full model shape, batch size 8, and two iterations. The conservative
# projections for both full folds are added before the 12-hour gate is evaluated.

# %%
expected_shape = tuple(config["data"]["expected_image_shape"])
frame_counts: dict[str, int] = {}
for path in zarr_paths:
    shape = tuple(zarr.open_group(str(path), mode="r")["0"].shape)
    if shape != expected_shape:
        raise RuntimeError(f"unexpected image shape for {path.name}: {shape}")
    frame_counts[path.name[:-5]] = int(shape[0])

seed = int(config["reproducibility"]["seed"])
window_size = int(model_cfg["window_size"])
batch_size = int(train_cfg["batch_size"])
smoke_records: list[dict[str, Any]] = []
projected_full_seconds = 0.0
smoke_elapsed_total = 0.0

for record in splits:
    fold = int(record["split"])
    debug_stem = sorted(record["train"])[0]
    debug_path = train_dir / debug_stem
    _, debug_windows = load_dataset_windows(
        debug_path,
        window_size=window_size,
        downsample=tuple(model_cfg["downsample"]),
    )
    if not debug_windows:
        raise RuntimeError(f"smoke sample has no valid windows: {debug_stem}")

    seed_everything(seed)
    gc.collect()
    torch.cuda.empty_cache()
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)
    smoke_log_path = ARTIFACTS_ROOT / f"smoke_fold_{fold}.log"
    try:
        smoke_model, smoke_elapsed_seconds = run_with_log(
            train,
            smoke_log_path,
            data_dir=train_dir,
            fold=fold,
            splits_file=split_path,
            method="unet_transformer_smoke_seed42",
            n_epochs=int(smoke_cfg["epochs"]),
            lr=float(train_cfg["learning_rate"]),
            batch_size=batch_size,
            num_workers=int(train_cfg["num_workers"]),
            unet_out_channels=int(model_cfg["unet_out_channels"]),
            unet_layers=list(model_cfg["unet_layers"]),
            unet_weights=None,
            downsample=tuple(model_cfg["downsample"]),
            det_loss_weight=float(train_cfg["det_loss_weight"]),
            det_neg_weight=float(train_cfg["det_neg_weight"]),
            max_iters=int(smoke_cfg["max_iters"]),
            debug_video=debug_path,
            seed=seed,
            window_size=window_size,
            augmentations=DEFAULT_AUGMENTATIONS,
            pool_kernel_um=float(train_cfg["pool_kernel_um"]),
            data_parallel=bool(train_cfg["data_parallel"]),
        )
    except Exception as exc:
        update_metrics(
            {
                "experiment": EXPERIMENT,
                "status": "failed",
                "updated_at": datetime.now(UTC).isoformat(),
                "evidence": {
                    "kaggle": {
                        "kernel_id": TRAIN_KERNEL_ID,
                        "resource": {"gpu_names": gpu_names},
                        "internet_enabled": False,
                    },
                    "artifacts": {
                        "input_file_sha": sha256_file(dataset_index_path),
                        "split_sha": sha256_file(artifact_split_path),
                        "source_manifest_sha": source_manifest_sha,
                    },
                },
                "notes": f"Fold {fold} smoke failed: {type(exc).__name__}: {exc}",
            }
        )
        raise
    torch.cuda.synchronize()

    peak_allocated = [
        torch.cuda.max_memory_allocated(device_index) / (1024**3)
        for device_index in range(torch.cuda.device_count())
    ]
    peak_reserved = [
        torch.cuda.max_memory_reserved(device_index) / (1024**3)
        for device_index in range(torch.cuda.device_count())
    ]
    train_window_upper = sum(
        max(0, frame_counts[name] - window_size + 1) for name in record["train"]
    )
    selection_window_upper = sum(
        max(0, frame_counts[name] - window_size + 1) for name in record["test"]
    )
    smoke_selection_batches = math.ceil(len(debug_windows) / batch_size)
    smoke_equivalent_batches = int(smoke_cfg["max_iters"]) + smoke_selection_batches
    full_batches_per_epoch = math.ceil(train_window_upper / batch_size) + math.ceil(
        selection_window_upper / batch_size
    )
    fold_projected_seconds = (
        smoke_elapsed_seconds
        / smoke_equivalent_batches
        * full_batches_per_epoch
        * int(train_cfg["epochs"])
        * float(smoke_cfg["runtime_projection_multiplier"])
    )
    smoke_record = {
        "fold": fold,
        "outer_train_embryo": record["outer_train_embryo"],
        "outer_evaluation_embryo": record["outer_evaluation_embryo"],
        "debug_stem": debug_stem,
        "debug_windows": len(debug_windows),
        "smoke_train_iterations": int(smoke_cfg["max_iters"]),
        "smoke_checkpoint_selection_batches": smoke_selection_batches,
        "elapsed_seconds": smoke_elapsed_seconds,
        "allocator_environment_variable": allocator_variable,
        "allocator_conf": os.environ[allocator_variable],
        "peak_allocated_gib_by_device": peak_allocated,
        "peak_reserved_gib_by_device": peak_reserved,
        "full_train_window_upper": train_window_upper,
        "full_checkpoint_selection_window_upper": selection_window_upper,
        "projected_full_seconds": fold_projected_seconds,
        "smoke_log_sha256": sha256_file(smoke_log_path),
    }
    del smoke_model
    del debug_windows
    gc.collect()
    torch.cuda.empty_cache()
    smoke_records.append(smoke_record)
    projected_full_seconds += fold_projected_seconds
    smoke_elapsed_total += smoke_elapsed_seconds

training_gate_seconds = float(smoke_cfg["runtime_gate_hours"]) * 3600
run_full_training = projected_full_seconds <= training_gate_seconds
smoke_summary = {
    "folds": smoke_records,
    "projected_all_folds_seconds": projected_full_seconds,
    "gate_limit_seconds": training_gate_seconds,
    "gate_passed": run_full_training,
    "control_retrained": False,
    "active_variant_count": 1,
    "model_config_count": 1,
    "outer_fold_count": len(splits),
    "total_booster_count": 0,
}
atomic_json(ARTIFACTS_ROOT / "smoke_summary.json", smoke_summary)
update_metrics(
    {
        "experiment": EXPERIMENT,
        "status": "debug_completed",
        "updated_at": datetime.now(UTC).isoformat(),
        "cv": None,
        "metric": validation_cfg["primary_metric"],
        "diagnostic_validation": {"training_smoke": smoke_summary},
        "evidence": {
            "kaggle": {
                "kernel_id": TRAIN_KERNEL_ID,
                "resource": {
                    "gpu_names": gpu_names,
                    "visible_gpu_count": len(gpu_names),
                    "allocator_environment_variable": allocator_variable,
                    "allocator_conf": os.environ[allocator_variable],
                },
                "notebook_runtime_seconds": smoke_elapsed_total,
                "internet_enabled": False,
            },
            "artifacts": {
                "input_file_sha": sha256_file(dataset_index_path),
                "split_sha": sha256_file(artifact_split_path),
                "source_manifest_sha": source_manifest_sha,
                "model_count": 0,
            },
        },
        "notes": (
            "Both fold smokes completed; full training starts only if their "
            "combined projection passes the configured gate."
        ),
    }
)
print(json.dumps(smoke_summary, indent=2))
if not run_full_training:
    print(
        "Combined runtime gate stopped both full three-epoch runs; "
        "no training contract was reduced."
    )

# %% [markdown]
# ## 5. Two-fold scratch training and six epoch checkpoints
#
# One model is trained per outer training embryo. The mixed-embryo exp002 control
# is not retrained, and no checkpoint is reused or used for initialization.

# %%
training_records: list[dict[str, Any]] = []
model_records: list[dict[str, Any]] = []
full_elapsed_total = 0.0

if run_full_training:
    for record in splits:
        fold = int(record["split"])
        seed_everything(seed)
        gc.collect()
        torch.cuda.empty_cache()
        full_log_path = ARTIFACTS_ROOT / f"training_fold_{fold}.log"
        try:
            fold_model_dir = MODELS_ROOT / f"fold_{fold}"
            saved_epoch_indices = [int(value) for value in train_cfg["save_epoch_indices"]]
            trained_model, full_elapsed_seconds, captured_checkpoints = (
                run_training_with_epoch_checkpoints(
                    full_log_path,
                    fold_model_dir,
                    saved_epoch_indices,
                    data_dir=train_dir,
                    fold=fold,
                    splits_file=split_path,
                    method="unet_transformer",
                    n_epochs=int(train_cfg["epochs"]),
                    lr=float(train_cfg["learning_rate"]),
                    batch_size=int(train_cfg["batch_size"]),
                    num_workers=int(train_cfg["num_workers"]),
                    unet_out_channels=int(model_cfg["unet_out_channels"]),
                    unet_layers=list(model_cfg["unet_layers"]),
                    unet_weights=None,
                    downsample=tuple(model_cfg["downsample"]),
                    det_loss_weight=float(train_cfg["det_loss_weight"]),
                    det_neg_weight=float(train_cfg["det_neg_weight"]),
                    max_iters=None,
                    debug_video=None,
                    seed=seed,
                    window_size=int(model_cfg["window_size"]),
                    augmentations=DEFAULT_AUGMENTATIONS,
                    pool_kernel_um=float(train_cfg["pool_kernel_um"]),
                    data_parallel=bool(train_cfg["data_parallel"]),
                )
            )
        except Exception as exc:
            update_metrics(
                {
                    "experiment": EXPERIMENT,
                    "status": "failed",
                    "updated_at": datetime.now(UTC).isoformat(),
                    "evidence": {
                        "kaggle": {
                            "kernel_id": TRAIN_KERNEL_ID,
                            "resource": {"gpu_names": gpu_names},
                            "notebook_runtime_seconds": smoke_elapsed_total + full_elapsed_total,
                            "internet_enabled": False,
                        },
                        "artifacts": {
                            "input_file_sha": sha256_file(dataset_index_path),
                            "split_sha": sha256_file(artifact_split_path),
                            "source_manifest_sha": source_manifest_sha,
                            "model_count": sum(
                                int(item["checkpoint_count"]) for item in model_records
                            ),
                            "model_shas": {
                                checkpoint["checkpoint_key"]: checkpoint["checkpoint_sha256"]
                                for item in model_records
                                for checkpoint in item["checkpoints"]
                            },
                        },
                    },
                    "notes": f"Fold {fold} full training failed: {type(exc).__name__}: {exc}",
                }
            )
            raise
        full_elapsed_total += full_elapsed_seconds
        epochs = parse_epoch_records(full_log_path, int(train_cfg["epochs"]))

        source_model_config = (
            SOURCE_ROOT / "weights" / "unet_transformer" / f"split_{fold}" / "config.json"
        )
        if not source_model_config.is_file():
            raise FileNotFoundError(f"fold {fold} training did not save model config")
        if len(captured_checkpoints) != len(saved_epoch_indices):
            raise RuntimeError(
                f"fold {fold} expected {len(saved_epoch_indices)} checkpoints, "
                f"found {len(captured_checkpoints)}"
            )
        if [item["epoch_index"] for item in epochs] != saved_epoch_indices:
            raise RuntimeError(f"fold {fold} training log has unexpected epoch indices")

        model_config_path = fold_model_dir / "config.json"
        model_config_path.write_bytes(source_model_config.read_bytes())
        model_config_sha = sha256_file(model_config_path)
        manifest_checkpoints: list[dict[str, Any]] = []
        bundle_checkpoints: list[dict[str, Any]] = []
        for checkpoint_record, log_record in zip(captured_checkpoints, epochs, strict=True):
            if checkpoint_record["epoch_index"] != log_record["epoch_index"]:
                raise RuntimeError(f"fold {fold} checkpoint and log epochs are misaligned")
            checkpoint_name = str(checkpoint_record["checkpoint"])
            checkpoint_key = f"fold_{fold}/{checkpoint_name}"
            manifest_entry = {
                **checkpoint_record,
                "training_log_metrics": log_record,
            }
            bundle_entry = {
                **checkpoint_record,
                "checkpoint": f"models/{checkpoint_key}",
                "checkpoint_key": checkpoint_key,
            }
            manifest_checkpoints.append(manifest_entry)
            bundle_checkpoints.append(bundle_entry)

        best_proxy_epoch = max(
            manifest_checkpoints,
            key=lambda item: (item["proxy_selection_score"], item["epoch_index"]),
        )
        fold_manifest = {
            "manifest_type": "embryo_holdout_epoch_checkpoint_fold",
            "experiment": EXPERIMENT,
            "source_repository": source_manifest["repository"],
            "source_commit": source_manifest["commit"],
            "source_manifest_sha256": source_manifest_sha,
            "trained_from_scratch": True,
            "upstream_checkpoint_loaded": False,
            "seed": seed,
            "fold": fold,
            "outer_train_embryo": record["outer_train_embryo"],
            "outer_evaluation_embryo": record["outer_evaluation_embryo"],
            "gradient_update_count": len(record["train"]),
            "checkpoint_selection_count": len(record["test"]),
            "outer_evaluation_count": len(record["outer_evaluation"]),
            "epochs": int(train_cfg["epochs"]),
            "checkpoint_count": len(manifest_checkpoints),
            "checkpoints": manifest_checkpoints,
            "model_config": model_config_path.name,
            "model_config_sha256": model_config_sha,
            "split_manifest": "../../splits.json",
            "split_manifest_sha256": sha256_file(artifact_split_path),
            "selection_metric": validation_cfg["checkpoint_selection_metric"],
            "proxy_selected_epoch_index": best_proxy_epoch["epoch_index"],
            "selector_tie_break": validation_cfg["selector_tie_break"],
            "outer_evaluation_used_for_selection": False,
            "augmentation_byte_deterministic": False,
            "gpu_names": gpu_names,
        }
        fold_manifest_path = fold_model_dir / "model_manifest.json"
        atomic_json(fold_manifest_path, fold_manifest)
        model_record = {
            "fold": fold,
            "outer_train_embryo": record["outer_train_embryo"],
            "outer_evaluation_embryo": record["outer_evaluation_embryo"],
            "checkpoint_count": len(bundle_checkpoints),
            "checkpoints": bundle_checkpoints,
            "model_config": f"models/fold_{fold}/config.json",
            "model_config_sha256": model_config_sha,
            "fold_manifest": f"models/fold_{fold}/model_manifest.json",
            "fold_manifest_sha256": sha256_file(fold_manifest_path),
            "proxy_selected_epoch_index": best_proxy_epoch["epoch_index"],
        }
        model_records.append(model_record)
        training_records.append(
            {
                "fold": fold,
                "outer_train_embryo": record["outer_train_embryo"],
                "outer_evaluation_embryo": record["outer_evaluation_embryo"],
                "elapsed_seconds": full_elapsed_seconds,
                "epochs": epochs,
                "checkpoints": bundle_checkpoints,
                "proxy_selected_epoch": best_proxy_epoch,
                "training_log_sha256": sha256_file(full_log_path),
                "model_config_sha256": model_config_sha,
                "fold_manifest_sha256": sha256_file(fold_manifest_path),
            }
        )
        del trained_model
        gc.collect()
        torch.cuda.empty_cache()

    if len(model_records) != 2:
        raise RuntimeError(f"expected two trained fold models, found {len(model_records)}")
    model_bundle = {
        "manifest_type": "embryo_holdout_epoch_checkpoint_bundle",
        "experiment": EXPERIMENT,
        "source_repository": source_manifest["repository"],
        "source_commit": source_manifest["commit"],
        "source_manifest_sha256": source_manifest_sha,
        "split_manifest": "splits.json",
        "split_manifest_sha256": sha256_file(artifact_split_path),
        "trained_from_scratch": True,
        "upstream_checkpoint_loaded": False,
        "epochs": int(train_cfg["epochs"]),
        "model_count": sum(int(item["checkpoint_count"]) for item in model_records),
        "folds": model_records,
    }
    model_bundle_path = ARTIFACTS_ROOT / "model_manifest.json"
    atomic_json(model_bundle_path, model_bundle)
    training_summary = {
        "experiment": EXPERIMENT,
        "smoke_elapsed_seconds": smoke_elapsed_total,
        "full_training_elapsed_seconds": full_elapsed_total,
        "notebook_measured_seconds": smoke_elapsed_total + full_elapsed_total,
        "folds": training_records,
        "model_manifest_sha256": sha256_file(model_bundle_path),
        "split_manifest_sha256": sha256_file(artifact_split_path),
    }
    training_summary_path = ARTIFACTS_ROOT / "training_summary.json"
    atomic_json(training_summary_path, training_summary)

    selection_diagnostics = {
        f"fold_{item['fold']}": {
            "outer_train_embryo": item["outer_train_embryo"],
            "outer_evaluation_embryo": item["outer_evaluation_embryo"],
            "best_proxy_selection_score": item["proxy_selected_epoch"]["proxy_selection_score"],
            "best_proxy_epoch_index": item["proxy_selected_epoch"]["epoch_index"],
            "epochs": item["epochs"],
            "checkpoints": item["checkpoints"],
            "diagnostic_only": True,
        }
        for item in training_records
    }
    update_metrics(
        {
            "experiment": EXPERIMENT,
            "status": "running",
            "updated_at": datetime.now(UTC).isoformat(),
            "cv": None,
            "metric": validation_cfg["primary_metric"],
            "diagnostic_validation": {
                "checkpoint_selection": selection_diagnostics,
                "training_smoke": smoke_summary,
            },
            "evidence": {
                "kaggle": {
                    "kernel_id": TRAIN_KERNEL_ID,
                    "resource": {
                        "gpu_names": gpu_names,
                        "visible_gpu_count": len(gpu_names),
                        "allocator_environment_variable": allocator_variable,
                        "allocator_conf": os.environ[allocator_variable],
                    },
                    "notebook_runtime_seconds": smoke_elapsed_total + full_elapsed_total,
                    "internet_enabled": False,
                },
                "artifacts": {
                    "input_file_sha": sha256_file(dataset_index_path),
                    "split_sha": sha256_file(artifact_split_path),
                    "source_manifest_sha": source_manifest_sha,
                    "model_manifest_sha": sha256_file(model_bundle_path),
                    "model_count": sum(int(item["checkpoint_count"]) for item in model_records),
                    "model_shas": {
                        checkpoint["checkpoint_key"]: checkpoint["checkpoint_sha256"]
                        for item in model_records
                        for checkpoint in item["checkpoints"]
                    },
                    "selected_mode": "all_epoch_checkpoints_pending_two_selector_comparison",
                    "selected_model": None,
                },
            },
            "notes": (
                "Six epoch checkpoints from two embryo-disjoint folds were saved; "
                "checkpoint selection and outer held-out evaluation have not yet run."
            ),
        }
    )
    print(json.dumps(training_summary, indent=2))

# %% [markdown]
# ## 6. Metrics and artifact contract

# %%
if run_full_training:
    required = [
        ARTIFACTS_ROOT / "splits.json",
        ARTIFACTS_ROOT / "dataset_index.json",
        ARTIFACTS_ROOT / "smoke_summary.json",
        ARTIFACTS_ROOT / "training_summary.json",
        ARTIFACTS_ROOT / "model_manifest.json",
        METRICS_PATH,
    ]
    for fold in range(2):
        required.extend(
            [
                *[
                    MODELS_ROOT / f"fold_{fold}" / f"epoch_{epoch_index}.pth"
                    for epoch_index in train_cfg["save_epoch_indices"]
                ],
                MODELS_ROOT / f"fold_{fold}" / "config.json",
                MODELS_ROOT / f"fold_{fold}" / "model_manifest.json",
                ARTIFACTS_ROOT / f"training_fold_{fold}.log",
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing training artifacts: {missing}")
    print("Six epoch checkpoints and split evidence are ready for selector evaluation.")
else:
    print("Only two-fold smoke evidence was produced; no full model bundle is available.")
