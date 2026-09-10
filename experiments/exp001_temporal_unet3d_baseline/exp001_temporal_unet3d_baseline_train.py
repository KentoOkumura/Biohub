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
# # exp001 TemporalUNet3D baseline training
#
# This notebook verifies the pinned organizer source, creates the organizer's
# seed-0 sample holdout, runs a same-configuration smoke test, and starts the
# three-epoch scratch run only when the projected runtime is at most 11 hours.
# The published checkpoint bundled with the wheel dataset is never copied or loaded.

# %% [markdown]
# ## 1. Configuration and evidence helpers

# %%
from __future__ import annotations

import contextlib
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

EXPERIMENT = "exp001_temporal_unet3d_baseline"
WORKING_ROOT = Path.cwd()
CONFIG_PATH = WORKING_ROOT / "config.yaml"
SOURCE_ROOT = WORKING_ROOT / "official_source"
SOURCE_MANIFEST_PATH = SOURCE_ROOT / "SOURCE.json"
ARTIFACTS_ROOT = WORKING_ROOT / "artifacts"
MODEL_ARTIFACTS = ARTIFACTS_ROOT / "model"
METRICS_PATH = WORKING_ROOT / "metrics.json"

config = yaml.safe_load(CONFIG_PATH.read_text())
train_cfg = config["model"]["training"]
smoke_cfg = config["model"]["smoke"]
model_cfg = config["model"]["params"]
runtime_cfg = config["runtime"]

print("Experiment:", EXPERIMENT)
print("Source commit:", config["source"]["commit"])
print("Training config:", json.dumps(train_cfg, indent=2))
print("Smoke config:", json.dumps(smoke_cfg, indent=2))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
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
    except Exception as exc:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(buffer.getvalue())
        update_metrics(
            {
                "experiment": EXPERIMENT,
                "status": "failed",
                "updated_at": datetime.now(UTC).isoformat(),
                "notes": f"Training call failed: {type(exc).__name__}: {exc}",
            }
        )
        raise
    elapsed = time.monotonic() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(buffer.getvalue())
    return result, elapsed


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

# Import the numerical stack only after pip finishes. The exact no-deps list
# intentionally omits imagecodecs, whose bundled release requires NumPy >=2.1
# and would replace the NumPy 2.0.2 already loaded by the Kaggle kernel.
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

source_manifest = verify_source()
source_manifest_sha = sha256_file(SOURCE_MANIFEST_PATH)
sys.path.insert(0, str(SOURCE_ROOT / "src"))
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))

from train_unet_transformer import (  # noqa: E402
    DEFAULT_AUGMENTATIONS,
    load_dataset_windows,
    train,
)

# %% [markdown]
# ## 3. Competition input and organizer split 0
#
# The organizer fallback sorts sample names, shuffles them with seed 0, and
# assigns floor(10 percent) to validation. This is a sample-level diagnostic
# holdout and may mix embryo identities.

# %%
train_dir = find_competition_dir("train")
zarr_paths = sorted(train_dir.glob("*.zarr"))
paired_stems = [
    path.name[:-5] for path in zarr_paths if (train_dir / f"{path.name[:-5]}.geff").exists()
]
expected_samples = int(config["validation"]["expected_sample_count"])
if len(paired_stems) != expected_samples:
    raise RuntimeError(f"expected {expected_samples} Zarr/GEFF pairs, found {len(paired_stems)}")

shuffled_stems = list(paired_stems)
random.Random(int(config["validation"]["split_seed"])).shuffle(shuffled_stems)
n_validation = max(1, len(shuffled_stems) // 10)
split_record = {
    "split": 0,
    "train": shuffled_stems[n_validation:],
    "test": shuffled_stems[:n_validation],
}
splits = [split_record]

split_path = SOURCE_ROOT / "dataset_splits.json"
atomic_json(split_path, {"folds": splits})
# The pinned training script expects a JSON list rather than a wrapper object.
split_path.write_text(json.dumps(splits, indent=2) + "\n")
ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
artifact_split_path = ARTIFACTS_ROOT / "split_0.json"
artifact_split_path.write_text(split_path.read_text())

dataset_index = {
    "sample_count": len(paired_stems),
    "paired_stems": paired_stems,
    "split_seed": int(config["validation"]["split_seed"]),
    "train_count": len(split_record["train"]),
    "validation_count": len(split_record["test"]),
}
dataset_index_path = ARTIFACTS_ROOT / "dataset_index.json"
atomic_json(dataset_index_path, dataset_index)

print("Training directory:", train_dir)
print(
    "Split 0:",
    len(split_record["train"]),
    "train /",
    len(split_record["test"]),
    "validation",
)
print("Split SHA-256:", sha256_file(artifact_split_path))

# %% [markdown]
# ## 4. Same-source smoke and 11-hour runtime gate
#
# The smoke uses the first sorted sample, the full model shape, batch size 16,
# the published loss, and two training iterations. Its checkpoint is isolated
# under a smoke-only method name and is never copied into the model artifacts.

# %%
seed = int(config["reproducibility"]["seed"])
seed_everything(seed)
for device_index in range(torch.cuda.device_count()):
    torch.cuda.reset_peak_memory_stats(device_index)

debug_stem = paired_stems[0]
debug_path = train_dir / debug_stem
_, debug_windows = load_dataset_windows(
    debug_path,
    window_size=int(model_cfg["window_size"]),
    downsample=tuple(model_cfg["downsample"]),
)
if not debug_windows:
    raise RuntimeError(f"smoke sample has no valid windows: {debug_stem}")

smoke_log_path = ARTIFACTS_ROOT / "smoke_training_log.txt"
_, smoke_elapsed_seconds = run_with_log(
    train,
    smoke_log_path,
    data_dir=train_dir,
    fold=0,
    splits_file=split_path,
    method="unet_transformer_smoke_seed42",
    n_epochs=int(smoke_cfg["epochs"]),
    lr=float(train_cfg["learning_rate"]),
    batch_size=int(train_cfg["batch_size"]),
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
    window_size=int(model_cfg["window_size"]),
    augmentations=DEFAULT_AUGMENTATIONS,
    pool_kernel_um=float(train_cfg["pool_kernel_um"]),
    data_parallel=bool(train_cfg["data_parallel"]),
)
torch.cuda.synchronize()
peak_gpu_gib = max(
    torch.cuda.max_memory_allocated(device_index) / (1024**3)
    for device_index in range(torch.cuda.device_count())
)

frame_counts: dict[str, int] = {}
expected_shape = tuple(config["data"]["expected_image_shape"])
for path in zarr_paths:
    shape = tuple(zarr.open_group(str(path), mode="r")["0"].shape)
    if shape != expected_shape:
        raise RuntimeError(f"unexpected image shape for {path.name}: {shape}")
    frame_counts[path.name[:-5]] = int(shape[0])

window_size = int(model_cfg["window_size"])
batch_size = int(train_cfg["batch_size"])
train_window_upper = sum(
    max(0, frame_counts[name] - window_size + 1) for name in split_record["train"]
)
validation_window_upper = sum(
    max(0, frame_counts[name] - window_size + 1) for name in split_record["test"]
)
smoke_validation_batches = math.ceil(len(debug_windows) / batch_size)
smoke_equivalent_batches = int(smoke_cfg["max_iters"]) + smoke_validation_batches
full_batches_per_epoch = math.ceil(train_window_upper / batch_size) + math.ceil(
    validation_window_upper / batch_size
)
projected_seconds = (
    smoke_elapsed_seconds
    / smoke_equivalent_batches
    * full_batches_per_epoch
    * int(train_cfg["epochs"])
    * 1.25
)
gate_limit_seconds = float(smoke_cfg["runtime_gate_hours"]) * 3600
run_full_training = projected_seconds <= gate_limit_seconds

smoke_summary = {
    "debug_stem": debug_stem,
    "debug_windows": len(debug_windows),
    "smoke_train_iterations": int(smoke_cfg["max_iters"]),
    "smoke_validation_batches": smoke_validation_batches,
    "elapsed_seconds": smoke_elapsed_seconds,
    "peak_gpu_gib": peak_gpu_gib,
    "full_train_window_upper": train_window_upper,
    "full_validation_window_upper": validation_window_upper,
    "projected_full_seconds": projected_seconds,
    "gate_limit_seconds": gate_limit_seconds,
    "gate_passed": run_full_training,
}
atomic_json(ARTIFACTS_ROOT / "smoke_summary.json", smoke_summary)
update_metrics(
    {
        "experiment": EXPERIMENT,
        "status": "debug_completed",
        "updated_at": datetime.now(UTC).isoformat(),
        "cv": None,
        "metric": "edge_accuracy_times_node_recall",
        "diagnostic_validation": {"smoke": smoke_summary},
        "evidence": {
            "kaggle": {
                "kernel_id": "kentookumura/exp001-temporal-unet3d-baseline-train",
                "resource": {"gpu_names": gpu_names, "visible_gpu_count": len(gpu_names)},
                "internet_enabled": False,
            },
            "artifacts": {
                "input_file_sha": sha256_file(dataset_index_path),
                "split_sha": sha256_file(artifact_split_path),
                "source_manifest_sha": source_manifest_sha,
                "model_count": 0,
            },
        },
        "notes": ("Smoke completed; full training starts only when gate_passed is true."),
    }
)
print(json.dumps(smoke_summary, indent=2))
if not run_full_training:
    print("Runtime gate stopped the full three-epoch run.")

# %% [markdown]
# ## 5. Full three-epoch scratch training
#
# This cell does not run when the conservative smoke projection exceeds 11 hours.
# No pretrained U-Net or public edge-predictor checkpoint is passed to train.

# %%
training_summary: dict[str, Any] | None = None
if run_full_training:
    seed_everything(seed)
    full_log_path = ARTIFACTS_ROOT / "training_log.txt"
    _, full_elapsed_seconds = run_with_log(
        train,
        full_log_path,
        data_dir=train_dir,
        fold=0,
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
    torch.cuda.synchronize()

    epoch_pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+\|\s+edge=([0-9.]+)\s+\|\s+"
        r"det=([0-9.]+)\s+\|\s+test_loss=([0-9.]+)\s+\|\s+"
        r"acc=([0-9.]+)\s+\|\s+recall=([0-9.]+)\s+\|\s+"
        r"best=([0-9.]+).*train=([0-9.]+)s test=([0-9.]+)s"
    )
    epochs = []
    for match in epoch_pattern.finditer(full_log_path.read_text()):
        epochs.append(
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
        )
    if len(epochs) != int(train_cfg["epochs"]):
        raise RuntimeError(f"expected three epoch records, found {len(epochs)}")

    source_checkpoint = (
        SOURCE_ROOT / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
    )
    source_model_config = source_checkpoint.with_name("config.json")
    if not source_checkpoint.is_file() or not source_model_config.is_file():
        raise FileNotFoundError("full training did not save checkpoint and config")

    MODEL_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_ARTIFACTS / "edge_predictor_best.pth"
    model_config_path = MODEL_ARTIFACTS / "config.json"
    checkpoint_path.write_bytes(source_checkpoint.read_bytes())
    model_config_path.write_bytes(source_model_config.read_bytes())

    checkpoint_sha = sha256_file(checkpoint_path)
    model_config_sha = sha256_file(model_config_path)
    best_epoch = max(epochs, key=lambda item: item["selection_score"])
    model_manifest = {
        "experiment": EXPERIMENT,
        "source_repository": source_manifest["repository"],
        "source_commit": source_manifest["commit"],
        "source_manifest_sha256": source_manifest_sha,
        "trained_from_scratch": True,
        "upstream_checkpoint_loaded": False,
        "seed": seed,
        "fold": 0,
        "epochs": int(train_cfg["epochs"]),
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha,
        "model_config": model_config_path.name,
        "model_config_sha256": model_config_sha,
        "split_manifest": "../split_0.json",
        "split_manifest_sha256": sha256_file(artifact_split_path),
        "selection_metric": "edge_accuracy_times_node_recall",
        "best_epoch": best_epoch,
        "validation_is_diagnostic_only": True,
        "gpu_names": gpu_names,
        "augmentation_byte_deterministic": False,
    }
    model_manifest_path = MODEL_ARTIFACTS / "model_manifest.json"
    atomic_json(model_manifest_path, model_manifest)

    training_summary = {
        "elapsed_seconds": full_elapsed_seconds,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "checkpoint_sha256": checkpoint_sha,
        "model_config_sha256": model_config_sha,
        "model_manifest_sha256": sha256_file(model_manifest_path),
    }
    atomic_json(ARTIFACTS_ROOT / "training_summary.json", training_summary)

    update_metrics(
        {
            "experiment": EXPERIMENT,
            "status": "running",
            "updated_at": datetime.now(UTC).isoformat(),
            "cv": None,
            "metric": "edge_accuracy_times_node_recall",
            "diagnostic_validation": {
                "official_seed0_sample_holdout": {
                    "best_selection_score": best_epoch["selection_score"],
                    "best_epoch_index": best_epoch["epoch_index"],
                    "epochs": epochs,
                    "diagnostic_only": True,
                },
                "smoke": smoke_summary,
            },
            "evidence": {
                "kaggle": {
                    "kernel_id": "kentookumura/exp001-temporal-unet3d-baseline-train",
                    "resource": {
                        "gpu_names": gpu_names,
                        "visible_gpu_count": len(gpu_names),
                    },
                    "notebook_runtime_seconds": (smoke_elapsed_seconds + full_elapsed_seconds),
                    "internet_enabled": False,
                },
                "artifacts": {
                    "input_file_sha": sha256_file(dataset_index_path),
                    "split_sha": sha256_file(artifact_split_path),
                    "source_manifest_sha": source_manifest_sha,
                    "model_manifest_sha": sha256_file(model_manifest_path),
                    "model_count": 1,
                    "model_shas": {"edge_predictor_best.pth": checkpoint_sha},
                    "selected_mode": "scratch_three_epochs",
                    "selected_model": "edge_predictor_best.pth",
                },
            },
            "notes": (
                "Three-epoch scratch training finished; inference and submission have not yet run."
            ),
        }
    )
    print(json.dumps(training_summary, indent=2))

# %% [markdown]
# ## 6. Artifact contract

# %%
if run_full_training:
    required = [
        MODEL_ARTIFACTS / "edge_predictor_best.pth",
        MODEL_ARTIFACTS / "config.json",
        MODEL_ARTIFACTS / "model_manifest.json",
        ARTIFACTS_ROOT / "split_0.json",
        ARTIFACTS_ROOT / "training_log.txt",
        ARTIFACTS_ROOT / "training_summary.json",
        METRICS_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing training artifacts: {missing}")
    print("Training artifacts are ready for the inference kernel source.")
else:
    print("Only smoke evidence was produced; no submission checkpoint is available.")
