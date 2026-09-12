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
# # exp010 deterministic exp006 replay training
#
# This notebook builds two embryo-disjoint outer folds. Within each outer training
# embryo it creates a seed-0 sample holdout only for checkpoint selection, then
# trains the exp006 model from scratch for three epochs with item-keyed
# augmentation and deterministic PyTorch/CUDA settings. The outer evaluation
# embryo is never passed to the training function.

# %% [markdown]
# ## Contents
# 1. Configuration and evidence helpers
# 2. Runtime, offline dependencies, and pinned source
# 3. Competition input and embryo-disjoint split construction
# 4. Per-fold smoke runs and combined runtime gate
# 5. Two-fold scratch training and model artifacts
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

EXPERIMENT = "exp010_exp006_deterministic_replay"
TRAIN_KERNEL_ID = "kentookumura/exp010-exp006-deterministic-replay-train"
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

runtime_environment = {str(key): str(value) for key, value in runtime_cfg["environment"].items()}
if "torch" in sys.modules:
    raise RuntimeError("runtime environment must be configured before importing torch")
for variable, value in runtime_environment.items():
    os.environ[variable] = value

print("Experiment:", EXPERIMENT)
print("Source commit:", config["source"]["commit"])
print("Outer folds:", json.dumps(validation_cfg["outer_folds"], indent=2))
print("Training config:", json.dumps(train_cfg, indent=2))
print("Deterministic runtime environment:", runtime_environment)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_state_dict(path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a state dict: {path}")
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(str(key).encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        try:
            raw = tensor.numpy().tobytes(order="C")
        except TypeError:
            raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(raw)
    return digest.hexdigest()


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
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode("error")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


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
print("Active runtime environment:", runtime_environment)

if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
seed_everything(int(config["reproducibility"]["seed"]))
if not torch.are_deterministic_algorithms_enabled():
    raise RuntimeError("PyTorch deterministic algorithms are not enabled")
if not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark:
    raise RuntimeError("cuDNN deterministic settings are not active")
if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
    raise RuntimeError("TF32 must be disabled for the deterministic replay")

source_manifest = verify_source()
source_manifest_sha = sha256_file(SOURCE_MANIFEST_PATH)
source_patch = source_manifest.get("local_experiment_patches", {}).get(
    "scripts/train_unet_transformer.py", {}
)
if source_patch.get("experiment") != EXPERIMENT:
    raise RuntimeError("the deterministic training source patch is not recorded")
sys.path.insert(0, str(SOURCE_ROOT / "src"))
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))

from train_unet_transformer import DEFAULT_AUGMENTATIONS, load_dataset_windows, train  # noqa: E402

# %% [markdown]
# ## 3. Competition input and embryo-disjoint split construction
#
# The fixed training function sees only `train` and `test` from each record.
# Both lists contain the outer training embryo. `outer_evaluation` is recorded for
# inference and is never passed as training or checkpoint-selection data.

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
# the full model shape, batch size 8, seed 314159, and two iterations. The conservative
# projections for both full folds are added before the 9-hour training gate is evaluated.

# %%
expected_shape = tuple(config["data"]["expected_image_shape"])
frame_counts: dict[str, int] = {}
for path in zarr_paths:
    shape = tuple(zarr.open_group(str(path), mode="r")["0"].shape)
    if shape != expected_shape:
        raise RuntimeError(f"unexpected image shape for {path.name}: {shape}")
    frame_counts[path.name[:-5]] = int(shape[0])

seed = int(config["reproducibility"]["seed"])
if seed != int(train_cfg["seed"]):
    raise RuntimeError("model.training.seed and reproducibility.seed must match")
if seed != int(validation_cfg["seed"]):
    raise RuntimeError("validation.seed and reproducibility.seed must match")
if seed != 314159:
    raise RuntimeError(f"exp010 requires seed 314159, found {seed}")
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
            method="unet_transformer_smoke_seed314159",
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
        "runtime_environment": runtime_environment,
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
                    "runtime_environment": runtime_environment,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
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
# ## 5. Two-fold scratch training and model artifacts
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
            trained_model, full_elapsed_seconds = run_with_log(
                train,
                full_log_path,
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
                            "model_count": len(model_records),
                            "model_shas": {
                                item["checkpoint_key"]: item["checkpoint_sha256"]
                                for item in model_records
                            },
                        },
                    },
                    "notes": f"Fold {fold} full training failed: {type(exc).__name__}: {exc}",
                }
            )
            raise
        full_elapsed_total += full_elapsed_seconds
        epochs = parse_epoch_records(full_log_path, int(train_cfg["epochs"]))

        source_checkpoint = (
            SOURCE_ROOT
            / "weights"
            / "unet_transformer"
            / f"split_{fold}"
            / "edge_predictor_best.pth"
        )
        source_model_config = source_checkpoint.with_name("config.json")
        if not source_checkpoint.is_file() or not source_model_config.is_file():
            raise FileNotFoundError(f"fold {fold} training did not save checkpoint and config")

        fold_model_dir = MODELS_ROOT / f"fold_{fold}"
        fold_model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = fold_model_dir / "edge_predictor_best.pth"
        model_config_path = fold_model_dir / "config.json"
        checkpoint_path.write_bytes(source_checkpoint.read_bytes())
        model_config_path.write_bytes(source_model_config.read_bytes())
        checkpoint_sha = sha256_file(checkpoint_path)
        checkpoint_content_sha = sha256_state_dict(checkpoint_path)
        model_config_sha = sha256_file(model_config_path)
        best_epoch = max(epochs, key=lambda item: item["selection_score"])

        fold_manifest = {
            "manifest_type": "embryo_holdout_fold_model",
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
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_content_sha256": checkpoint_content_sha,
            "model_config": model_config_path.name,
            "model_config_sha256": model_config_sha,
            "split_manifest": "../../splits.json",
            "split_manifest_sha256": sha256_file(artifact_split_path),
            "selection_metric": validation_cfg["checkpoint_selection_metric"],
            "best_epoch": best_epoch,
            "outer_evaluation_used_for_selection": False,
            "augmentation_byte_deterministic": True,
            "augmentation_seed_key": train_cfg["augmentation_seed_key"],
            "deterministic_runtime": config["reproducibility"]["deterministic_runtime"],
            "gpu_names": gpu_names,
        }
        fold_manifest_path = fold_model_dir / "model_manifest.json"
        atomic_json(fold_manifest_path, fold_manifest)
        checkpoint_key = f"fold_{fold}/edge_predictor_best.pth"
        model_record = {
            "fold": fold,
            "outer_train_embryo": record["outer_train_embryo"],
            "outer_evaluation_embryo": record["outer_evaluation_embryo"],
            "checkpoint": f"models/{checkpoint_key}",
            "checkpoint_key": checkpoint_key,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_content_sha256": checkpoint_content_sha,
            "model_config": f"models/fold_{fold}/config.json",
            "model_config_sha256": model_config_sha,
            "fold_manifest": f"models/fold_{fold}/model_manifest.json",
            "fold_manifest_sha256": sha256_file(fold_manifest_path),
        }
        model_records.append(model_record)
        training_records.append(
            {
                "fold": fold,
                "outer_train_embryo": record["outer_train_embryo"],
                "outer_evaluation_embryo": record["outer_evaluation_embryo"],
                "elapsed_seconds": full_elapsed_seconds,
                "epochs": epochs,
                "best_epoch": best_epoch,
                "training_log_sha256": sha256_file(full_log_path),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_content_sha256": checkpoint_content_sha,
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
        "manifest_type": "embryo_holdout_model_bundle",
        "experiment": EXPERIMENT,
        "seed": seed,
        "source_repository": source_manifest["repository"],
        "source_commit": source_manifest["commit"],
        "source_manifest_sha256": source_manifest_sha,
        "split_manifest": "splits.json",
        "split_manifest_sha256": sha256_file(artifact_split_path),
        "trained_from_scratch": True,
        "upstream_checkpoint_loaded": False,
        "epochs": int(train_cfg["epochs"]),
        "augmentation_byte_deterministic": True,
        "deterministic_runtime": config["reproducibility"]["deterministic_runtime"],
        "model_count": len(model_records),
        "folds": model_records,
    }
    model_bundle_path = ARTIFACTS_ROOT / "model_manifest.json"
    atomic_json(model_bundle_path, model_bundle)
    training_summary = {
        "experiment": EXPERIMENT,
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_manifest_sha256": source_manifest_sha,
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
            "best_selection_score": item["best_epoch"]["selection_score"],
            "best_epoch_index": item["best_epoch"]["epoch_index"],
            "epochs": item["epochs"],
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
                        "runtime_environment": runtime_environment,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    },
                    "notebook_runtime_seconds": smoke_elapsed_total + full_elapsed_total,
                    "internet_enabled": False,
                },
                "artifacts": {
                    "input_file_sha": sha256_file(dataset_index_path),
                    "split_sha": sha256_file(artifact_split_path),
                    "source_manifest_sha": source_manifest_sha,
                    "config_sha": sha256_file(CONFIG_PATH),
                    "model_manifest_sha": sha256_file(model_bundle_path),
                    "model_count": len(model_records),
                    "model_shas": {
                        item["checkpoint_key"]: item["checkpoint_sha256"] for item in model_records
                    },
                    "model_content_shas": {
                        item["checkpoint_key"]: item["checkpoint_content_sha256"]
                        for item in model_records
                    },
                    "selected_mode": "two_direction_leave_one_embryo_out",
                    "selected_model": "one_internal-selection checkpoint per outer fold",
                },
            },
            "notes": (
                "Two embryo-disjoint fold models were trained from scratch; "
                "outer held-out inference and official scoring have not yet run."
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
                MODELS_ROOT / f"fold_{fold}" / "edge_predictor_best.pth",
                MODELS_ROOT / f"fold_{fold}" / "config.json",
                MODELS_ROOT / f"fold_{fold}" / "model_manifest.json",
                ARTIFACTS_ROOT / f"training_fold_{fold}.log",
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing training artifacts: {missing}")
    print("Two fold models and split evidence are ready for held-out inference.")
else:
    print("Only two-fold smoke evidence was produced; no full model bundle is available.")
