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
# # exp009 exp006 two-fold probability ensemble inference
#
# This notebook loads both scratch-trained exp006 fold checkpoints. For every
# runtime test video it averages the two models' detection probabilities before
# extracting one shared node set, averages edge probabilities on those shared
# candidates, and applies the fixed threshold and ILP decode once. It builds and
# validates submission.csv but does not issue the competition submission command.

# %% [markdown]
# ## Contents
# 1. Configuration, hashing, and input resolution
# 2. Runtime, offline dependencies, and pinned source
# 3. Resolve and verify both exp006 fold models
# 4. Two-fold probability ensemble helpers
# 5. Dynamic hidden-test inference
# 6. Build and validate the variable-row graph CSV
# 7. Metrics and output contract

# %% [markdown]
# ## 1. Configuration, hashing, and input resolution

# %%
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT = "exp009_exp006_twofold_ensemble"
PARENT_EXPERIMENT = "exp006_embryo_holdout_seed314159"
PARENT_TRAIN_KERNEL_ID = "kentookumura/exp006-embryo-holdout-seed314159-train"
INFERENCE_KERNEL_ID = "kentookumura/exp009-exp006-twofold-ensemble-inference"
COMPETITION_SLUG = "biohub-cell-tracking-during-development"
WORKING_ROOT = Path.cwd()
CONFIG_PATH = WORKING_ROOT / "config.yaml"
SOURCE_ROOT = WORKING_ROOT / "official_source"
SOURCE_MANIFEST_PATH = SOURCE_ROOT / "SOURCE.json"
METRICS_PATH = WORKING_ROOT / "metrics.json"
SUBMISSION_PATH = WORKING_ROOT / "submission.csv"
PREDICTIONS_ROOT = WORKING_ROOT / "predictions"
INFERENCE_SUMMARY_PATH = WORKING_ROOT / "inference_summary.json"

config = yaml.safe_load(CONFIG_PATH.read_text())
inference_cfg = config["model"]["inference"]
ensemble_cfg = inference_cfg["ensemble"]
runtime_cfg = config["runtime"]
started = time.monotonic()

print("Experiment:", EXPERIMENT, flush=True)
print("Parent model experiment:", PARENT_EXPERIMENT, flush=True)
print("Inference config:", json.dumps(inference_cfg, indent=2), flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
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
    candidates = [
        Path("/kaggle/input/competitions") / COMPETITION_SLUG / child,
        Path("/kaggle/input") / COMPETITION_SLUG / child,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    matches = sorted(
        path
        for path in Path("/kaggle/input").rglob(child)
        if path.is_dir() and COMPETITION_SLUG in str(path) and any(path.glob("*.zarr"))
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one competition {child} directory, found {matches}")
    return matches[0]


def find_sample_submission() -> Path:
    candidates = [
        Path("/kaggle/input/competitions") / COMPETITION_SLUG / "sample_submission.csv",
        Path("/kaggle/input") / COMPETITION_SLUG / "sample_submission.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(
        path
        for path in Path("/kaggle/input").rglob("sample_submission.csv")
        if COMPETITION_SLUG in str(path)
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one competition sample submission, found {matches}")
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
    print(f"Verified {len(manifest['files'])} pinned source files.", flush=True)
    return manifest


def load_unique_parent_model_bundle() -> tuple[Path, dict[str, Any]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in Path("/kaggle/input").rglob("model_manifest.json"):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if (
            payload.get("experiment") == PARENT_EXPERIMENT
            and payload.get("manifest_type") == "embryo_holdout_model_bundle"
        ):
            candidates.append((path, payload))
    if len(candidates) != 1:
        paths = [str(path) for path, _ in candidates]
        raise FileNotFoundError(f"expected one exp006 model bundle, found {paths}")
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import scipy  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import zarr  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

expected_gpus = int(runtime_cfg["expected_cuda_device_count"])
gpu_names = [
    torch.cuda.get_device_name(device_index) for device_index in range(torch.cuda.device_count())
]
if len(gpu_names) != expected_gpus:
    raise RuntimeError(f"expected {expected_gpus} visible GPUs for T4 x2, found {gpu_names}")
expected_name = str(runtime_cfg["expected_gpu_name_substring"])
if not all(expected_name.lower() in name.lower() for name in gpu_names):
    raise RuntimeError(f"expected T4 GPUs, found {gpu_names}")
print("Numerical stack:", {"numpy": np.__version__, "scipy": scipy.__version__}, flush=True)
print("Visible GPUs:", gpu_names, flush=True)

source_manifest = verify_source()
source_manifest_sha = sha256_file(SOURCE_MANIFEST_PATH)
sys.path.insert(0, str(SOURCE_ROOT / "src"))
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))

import tracksdata as td  # noqa: E402
from predict_unet_transformer import (  # noqa: E402
    PredictConfig,
    _load_frame,
    load_model,
    pool_kernel_from_um,
    suppress_output,
)
from tracking_cellmot.io import open_dataset, save_graph  # noqa: E402
from train_unet_transformer import extract_pos_features  # noqa: E402

# %% [markdown]
# ## 3. Resolve and verify both exp006 fold models
#
# The public checkpoint in the dependency dataset is never searched or loaded.
# The attached exp006 train kernel must provide exactly two scratch-trained fold
# checkpoints and the manifests that attest their source, seed, and SHA-256.

# %%
model_bundle_path, model_bundle = load_unique_parent_model_bundle()
if not model_bundle.get("trained_from_scratch"):
    raise RuntimeError("model bundle does not attest scratch training")
if model_bundle.get("upstream_checkpoint_loaded"):
    raise RuntimeError("model bundle indicates an upstream checkpoint was loaded")
if model_bundle.get("source_commit") != source_manifest["commit"]:
    raise RuntimeError("training and inference source commits differ")
if int(model_bundle.get("seed", -1)) != 314159:
    raise RuntimeError("model bundle does not attest seed 314159")
if int(model_bundle.get("model_count", -1)) != 2:
    raise RuntimeError("model bundle must contain exactly two fold models")
if len(model_bundle.get("folds", [])) != 2:
    raise RuntimeError("model bundle fold list must contain exactly two records")

train_artifacts_root = model_bundle_path.parent
split_manifest_path = train_artifacts_root / model_bundle["split_manifest"]
if sha256_file(split_manifest_path) != model_bundle["split_manifest_sha256"]:
    raise RuntimeError("training split manifest SHA-256 mismatch")

fold_models: dict[int, dict[str, Any]] = {}
for record in model_bundle["folds"]:
    fold = int(record["fold"])
    checkpoint_path = train_artifacts_root / record["checkpoint"]
    model_config_path = train_artifacts_root / record["model_config"]
    fold_manifest_path = train_artifacts_root / record["fold_manifest"]
    if sha256_file(checkpoint_path) != record["checkpoint_sha256"]:
        raise RuntimeError(f"fold {fold} checkpoint SHA-256 mismatch")
    if sha256_file(model_config_path) != record["model_config_sha256"]:
        raise RuntimeError(f"fold {fold} model config SHA-256 mismatch")
    if sha256_file(fold_manifest_path) != record["fold_manifest_sha256"]:
        raise RuntimeError(f"fold {fold} manifest SHA-256 mismatch")
    fold_manifest = json.loads(fold_manifest_path.read_text())
    if int(fold_manifest.get("seed", -1)) != 314159:
        raise RuntimeError(f"fold {fold} manifest does not attest seed 314159")
    if fold_manifest.get("outer_evaluation_used_for_selection"):
        raise RuntimeError(f"fold {fold} used outer evaluation data for selection")
    fold_models[fold] = {
        **record,
        "checkpoint_path": checkpoint_path,
        "model_config_path": model_config_path,
        "fold_manifest_path": fold_manifest_path,
    }
if set(fold_models) != {0, 1}:
    raise RuntimeError(f"expected folds 0 and 1, found {sorted(fold_models)}")
if [int(fold) for fold in ensemble_cfg["folds"]] != [0, 1]:
    raise RuntimeError("ensemble fold order must be [0, 1]")
if ensemble_cfg["detection_probability"] != "arithmetic_mean":
    raise RuntimeError("detection ensemble must use an arithmetic probability mean")
if ensemble_cfg["edge_probability"] != "arithmetic_mean_on_shared_candidates":
    raise RuntimeError("edge ensemble must average probabilities on shared candidates")
if int(ensemble_cfg["decode_count"]) != 1:
    raise RuntimeError("ensemble graph must be decoded exactly once")

devices = {fold: torch.device(str(ensemble_cfg["device_by_fold"][fold])) for fold in (0, 1)}
if devices != {0: torch.device("cuda:0"), 1: torch.device("cuda:1")}:
    raise RuntimeError(f"unexpected fold-to-device assignment: {devices}")

loaded_models: dict[int, Any] = {}
window_sizes: dict[int, int] = {}
downsamples: dict[int, tuple[int, ...]] = {}
for fold in (0, 1):
    model, window_size, downsample = load_model(
        fold_models[fold]["checkpoint_path"],
        devices[fold],
    )
    loaded_models[fold] = model
    window_sizes[fold] = int(window_size)
    downsamples[fold] = tuple(int(value) for value in downsample)
if len(set(window_sizes.values())) != 1:
    raise RuntimeError(f"fold window sizes differ: {window_sizes}")
if len(set(downsamples.values())) != 1:
    raise RuntimeError(f"fold downsample factors differ: {downsamples}")

window_size = window_sizes[0]
downsample = downsamples[0]
print("Model bundle:", model_bundle_path, flush=True)
print(
    "Checkpoint SHA-256:",
    {fold: fold_models[fold]["checkpoint_sha256"] for fold in (0, 1)},
    flush=True,
)
print("Fold devices:", {fold: str(device) for fold, device in devices.items()}, flush=True)

# %% [markdown]
# ## 4. Two-fold probability ensemble helpers
#
# Detection peaks are extracted only after averaging both models' TTA sigmoid
# probabilities. Edge probabilities are then computed by both models on exactly
# the same detected coordinates and averaged before any threshold or graph decode.

# %%
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
if int(inference_cfg["unet_batch_size"]) != 4:
    raise RuntimeError("the fixed organizer inference batch-size contract changed")


@torch.no_grad()
def encode_with_detection_tta(
    model: Any,
    images: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    device_images = images.unsqueeze(0).to(device)
    unet_output, detection_logits = model.encode(device_images)
    if predict_config.det_tta:
        for dims in [(-1,), (-2,), (-2, -1)]:
            flipped_images = device_images.flip(dims)
            _, flipped_logits = model.encode(flipped_images)
            for frame_offset in range(window_size):
                detection_logits[frame_offset] = detection_logits[frame_offset] + flipped_logits[
                    frame_offset
                ].flip(dims)
            del flipped_images, flipped_logits
        for frame_offset in range(window_size):
            detection_logits[frame_offset] = detection_logits[frame_offset] / 4
    probabilities = [torch.sigmoid(logits) for logits in detection_logits]
    del device_images, detection_logits
    return unet_output, probabilities


def detect_cells_from_mean_probability(
    probabilities: list[torch.Tensor],
    t: int,
    pool_kernel: tuple[int, ...],
) -> np.ndarray:
    mean_probability = torch.stack(
        [probability[0].float().cpu() for probability in probabilities]
    ).mean(dim=0)
    pooled = F.max_pool3d(
        mean_probability.unsqueeze(0),
        pool_kernel,
        stride=1,
        padding=tuple(size // 2 for size in pool_kernel),
    )
    is_peak = (mean_probability.unsqueeze(0) == pooled) & (
        mean_probability.unsqueeze(0) > predict_config.det_threshold
    )
    peak_idx = torch.nonzero(is_peak[0, 0])
    if peak_idx.shape[0] == 0:
        return np.empty((0, 4), dtype=np.int16)
    coords = peak_idx.numpy()
    t_column = np.full((len(coords), 1), t, dtype=np.int16)
    return np.concatenate([t_column, coords], axis=1).astype(np.int16)


def mean_edge_probabilities(
    unet_outputs: dict[int, torch.Tensor],
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    frame_offset: int,
    window_shape: tuple[int, ...],
) -> np.ndarray:
    source_count = len(source_coords)
    target_count = len(target_coords)
    relative_source = source_coords.copy()
    relative_target = target_coords.copy()
    relative_source[:, 0] = frame_offset
    relative_target[:, 0] = frame_offset + 1
    source_position_np = extract_pos_features(relative_source, window_shape)
    target_position_np = extract_pos_features(relative_target, window_shape)

    probabilities: list[torch.Tensor] = []
    for fold in (0, 1):
        device = devices[fold]
        source_tensor = (
            torch.from_numpy(source_coords[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
        )
        target_tensor = (
            torch.from_numpy(target_coords[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
        )
        source_position = torch.from_numpy(source_position_np).unsqueeze(0).to(device)
        target_position = torch.from_numpy(target_position_np).unsqueeze(0).to(device)
        source_valid = torch.ones(1, source_count, dtype=torch.bool, device=device)
        target_valid = torch.ones(1, target_count, dtype=torch.bool, device=device)
        model = loaded_models[fold]
        source_features = model._index_features(
            unet_outputs[fold][:, frame_offset],
            source_tensor,
            source_valid,
        )
        target_features = model._index_features(
            unet_outputs[fold][:, frame_offset + 1],
            target_tensor,
            target_valid,
        )
        downsample_tensor = torch.tensor(downsample, dtype=torch.float32, device=device)
        edge_logits = model.predict_edges(
            source_features,
            target_features,
            source_tensor * downsample_tensor,
            target_tensor * downsample_tensor,
            source_position,
            target_position,
            source_valid,
            target_valid,
        )[0]
        if predict_config.edge_activation == "softmax":
            probability = torch.softmax(edge_logits, dim=0)
        else:
            probability = torch.sigmoid(edge_logits)
        probabilities.append(probability.float().cpu())
    return torch.stack(probabilities).mean(dim=0).numpy().astype(np.float32, copy=False)


def select_graph_edges(
    flat_probabilities: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    source_coords: np.ndarray,
    target_coords: np.ndarray,
) -> list[tuple[int, int, float, float]]:
    target_count = len(target_coords)
    threshold_indices = np.flatnonzero(flat_probabilities > float(predict_config.threshold))
    candidates = sorted(
        [
            (
                float(flat_probabilities[index]),
                int(index // target_count),
                int(index % target_count),
                int(index),
            )
            for index in threshold_indices
        ],
        reverse=True,
    )
    distances = (
        np.linalg.norm(
            source_coords[:, None, 1:].astype(np.float32)
            - target_coords[None, :, 1:].astype(np.float32),
            axis=2,
        )
        .astype(np.float32)
        .reshape(-1)
    )
    selected: list[tuple[int, int, float, float]] = []
    children_count: dict[int, int] = {}
    parents_count: dict[int, int] = {}
    for probability, source_local, target_local, flat_index in candidates:
        n_children = children_count.get(source_local, 0)
        n_parents = parents_count.get(target_local, 0)
        if (
            predict_config.max_children_per_node is not None
            and n_children >= predict_config.max_children_per_node
        ):
            continue
        if (
            predict_config.max_parents_per_node is not None
            and n_parents >= predict_config.max_parents_per_node
        ):
            continue
        selected.append(
            (
                int(source_indices[flat_index]),
                int(target_indices[flat_index]),
                probability,
                float(distances[flat_index]),
            )
        )
        children_count[source_local] = n_children + 1
        parents_count[target_local] = n_parents + 1
    return selected


def build_graph(
    coords: np.ndarray,
    edges: list[tuple[int, int, float, float]],
) -> Any:
    graph = td.graph.InMemoryGraph()
    for key in ["z", "y", "x"]:
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    node_ids = graph.bulk_add_nodes(
        [{"t": int(t), "z": float(z), "y": float(y), "x": float(x)} for t, z, y, x in coords]
    )
    if edges:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.bulk_add_edges(
            [
                {
                    "source_id": node_ids[source],
                    "target_id": node_ids[target],
                    "edge_prob": probability,
                    "edge_dist": distance,
                }
                for source, target, probability, distance in edges
            ]
        )
    return graph


@torch.no_grad()
def predict_video_ensemble(dataset_path: Path) -> Any:
    dataset = open_dataset(
        dataset_path,
        normalize=False,
        load_image=False,
        downsample=downsample,
    )
    if "0.001" not in dataset.quantiles or "0.999" not in dataset.quantiles:
        raise ValueError(f"Zarr attrs missing image_statistics.quantiles for {dataset_path}")
    zarr_array = zarr.open_group(str(dataset.zarr_path), mode="r")["0"]
    q_low = float(dataset.quantiles["0.001"])
    q_high = float(dataset.quantiles["0.999"])
    frame_count = int(dataset.image_shape[0])
    image_shape = (frame_count,) + dataset.image_shape[1:]
    target_shape = list(image_shape[1:])
    downsample_array = np.asarray(downsample, dtype=np.float32)
    voxel_size = tuple(scale * step for scale, step in zip(dataset.scale, downsample, strict=True))
    pool_kernel = pool_kernel_from_um(predict_config.pool_kernel_um, voxel_size)

    seen_frames: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    coord_lists: list[np.ndarray] = []
    coord_offsets: dict[int, tuple[int, int]] = {}
    global_node_count = 0
    graph_edges: list[tuple[int, int, float, float]] = []

    stride = max(window_size - 1, 1)
    window_starts = list(range(0, frame_count - window_size + 1, stride))
    if not window_starts or window_starts[-1] + window_size < frame_count:
        last = max(frame_count - window_size, 0)
        if not window_starts or last != window_starts[-1]:
            window_starts.append(last)

    for window_start in tqdm(
        window_starts,
        desc=f"{dataset_path.name} windows",
        leave=False,
    ):
        frame_indices = list(range(window_start, window_start + window_size))
        images = torch.stack(
            [_load_frame(zarr_array, frame, target_shape, downsample) for frame in frame_indices]
        )
        images = ((images - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)

        unet_outputs: dict[int, torch.Tensor] = {}
        detection_probabilities: dict[int, list[torch.Tensor]] = {}
        for fold in (0, 1):
            unet_output, probabilities = encode_with_detection_tta(
                loaded_models[fold],
                images,
                devices[fold],
            )
            unet_outputs[fold] = unet_output
            detection_probabilities[fold] = probabilities
        del images

        for frame_offset, frame in enumerate(frame_indices):
            if frame in seen_frames:
                continue
            coords = detect_cells_from_mean_probability(
                [
                    detection_probabilities[0][frame_offset],
                    detection_probabilities[1][frame_offset],
                ],
                frame,
                pool_kernel,
            )
            coord_offsets[frame] = (
                global_node_count,
                global_node_count + len(coords),
            )
            global_node_count += len(coords)
            coord_lists.append(coords)
            seen_frames.add(frame)

        coords_so_far = (
            np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
        )
        for frame_offset in range(window_size - 1):
            source_time = frame_indices[frame_offset]
            target_time = frame_indices[frame_offset + 1]
            if (source_time, target_time) in seen_pairs:
                continue
            seen_pairs.add((source_time, target_time))
            source_start, source_end = coord_offsets[source_time]
            target_start, target_end = coord_offsets[target_time]
            source_count = source_end - source_start
            target_count = target_end - target_start
            if source_count == 0 or target_count == 0:
                continue

            source_coords = coords_so_far[source_start:source_end]
            target_coords = coords_so_far[target_start:target_end]
            source_indices = np.arange(source_start, source_end, dtype=np.int32)
            target_indices = np.arange(target_start, target_end, dtype=np.int32)
            flat_source = np.repeat(source_indices, target_count)
            flat_target = np.tile(target_indices, source_count)
            mean_probabilities = mean_edge_probabilities(
                unet_outputs,
                source_coords,
                target_coords,
                frame_offset,
                (window_size,) + image_shape[1:],
            ).reshape(-1)
            if len(mean_probabilities) != source_count * target_count:
                raise RuntimeError("edge score matrix does not match shared candidates")
            graph_edges.extend(
                select_graph_edges(
                    mean_probabilities,
                    flat_source,
                    flat_target,
                    source_coords,
                    target_coords,
                )
            )

        del unet_outputs, detection_probabilities
        torch.cuda.empty_cache()

    if len(seen_frames) != frame_count:
        raise RuntimeError(f"expected {frame_count} frames, found {len(seen_frames)}")
    if len(seen_pairs) != max(frame_count - 1, 0):
        raise RuntimeError(f"expected {frame_count - 1} adjacent pairs, found {len(seen_pairs)}")
    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
    coords = coords.astype(np.float32)
    coords[:, 1:] *= downsample_array
    coords = coords.astype(np.int16)

    graph = build_graph(coords, graph_edges)
    if predict_config.use_ilp and graph.num_edges() > 0:
        solver = td.solvers.ILPSolver(
            edge_weight=predict_config.ilp_edge_weight * td.EdgeAttr("edge_prob"),
            appearance_weight=predict_config.ilp_appearance_weight,
            disappearance_weight=predict_config.ilp_disappearance_weight,
            division_weight=predict_config.ilp_division_weight,
        )
        with suppress_output():
            graph = solver.solve(graph)
    return graph


def graph_rows(graph: Any, dataset_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    return rows


# %% [markdown]
# ## 5. Dynamic hidden-test inference
#
# The runtime competition input is authoritative. No public-test dataset name,
# count, row count, or content hash is used to branch the inference.

# %%
test_dir = find_competition_dir("test")
test_stems = sorted(path.name[:-5] for path in test_dir.glob("*.zarr"))
if not test_stems:
    raise RuntimeError(f"no test Zarr datasets found in {test_dir}")
if len(test_stems) != len(set(test_stems)):
    raise RuntimeError("duplicate test dataset names")

PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
all_rows: list[dict[str, Any]] = []
prediction_records: list[dict[str, Any]] = []
for sample_name in tqdm(test_stems, desc="Two-fold ensemble test videos"):
    video_started = time.monotonic()
    graph = predict_video_ensemble(test_dir / f"{sample_name}.zarr")
    graph_path = PREDICTIONS_ROOT / f"{sample_name}.geff"
    save_graph(graph, graph_path)
    rows = graph_rows(graph, sample_name)
    node_count = sum(row["row_type"] == "node" for row in rows)
    edge_count = sum(row["row_type"] == "edge" for row in rows)
    prediction_records.append(
        {
            "dataset": sample_name,
            "graph": f"predictions/{sample_name}.geff",
            "graph_tree_sha256": sha256_tree(graph_path),
            "node_count": node_count,
            "edge_count": edge_count,
            "elapsed_seconds": time.monotonic() - video_started,
        }
    )
    all_rows.extend(rows)
    del graph
    torch.cuda.empty_cache()
    print(
        f"Completed {sample_name}: nodes={node_count}, edges={edge_count}",
        flush=True,
    )

if {record["dataset"] for record in prediction_records} != set(test_stems):
    raise RuntimeError("prediction records do not cover every runtime test dataset")
print(f"Generated {len(prediction_records)} ensemble GEFF graphs.", flush=True)

# %% [markdown]
# ## 6. Build and validate the variable-row graph CSV

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
submission = pd.DataFrame(all_rows, columns=columns)
if submission.empty:
    raise RuntimeError("submission has no node or edge rows")
if set(submission["dataset"]) != set(test_stems):
    raise RuntimeError("submission datasets do not match runtime test discovery")
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

sample_submission_path = find_sample_submission()
sample_columns = list(pd.read_csv(sample_submission_path, nrows=1).columns)
if sample_columns != expected_csv_columns:
    raise RuntimeError(f"sample submission schema differs: {sample_columns}")

submission_sha = sha256_file(SUBMISSION_PATH)
elapsed_seconds = time.monotonic() - started
inference_summary = {
    "experiment": EXPERIMENT,
    "parent_experiment": PARENT_EXPERIMENT,
    "source_commit": source_manifest["commit"],
    "source_manifest_sha256": source_manifest_sha,
    "model_manifest_sha256": sha256_file(model_bundle_path),
    "checkpoint_sha256": {
        f"fold_{fold}": fold_models[fold]["checkpoint_sha256"] for fold in (0, 1)
    },
    "fold_devices": {f"fold_{fold}": str(devices[fold]) for fold in (0, 1)},
    "ensemble": ensemble_cfg,
    "test_dataset_count": len(test_stems),
    "test_datasets": test_stems,
    "predictions": prediction_records,
    "row_count": len(reloaded),
    "node_row_count": len(node_rows),
    "edge_row_count": len(edge_rows),
    "submission_sha256": submission_sha,
    "elapsed_seconds": elapsed_seconds,
}
atomic_json(INFERENCE_SUMMARY_PATH, inference_summary)

# %% [markdown]
# ## 7. Metrics and output contract

# %%
local_metrics = json.loads(METRICS_PATH.read_text())
metrics = deep_merge(
    local_metrics,
    {
        "experiment": EXPERIMENT,
        "status": "running",
        "updated_at": datetime.now(UTC).isoformat(),
        "cv": None,
        "metric": config["validation"]["metric"],
        "public_lb": None,
        "private_lb": None,
        "evidence": {
            "kaggle": {
                "kernel_id": INFERENCE_KERNEL_ID,
                "resource": {
                    "gpu_names": gpu_names,
                    "visible_gpu_count": len(gpu_names),
                    "fold_devices": {f"fold_{fold}": str(devices[fold]) for fold in (0, 1)},
                },
                "notebook_runtime_seconds": elapsed_seconds,
                "internet_enabled": False,
                "kernel_source_ids": [PARENT_TRAIN_KERNEL_ID],
            },
            "artifacts": {
                "input_file_sha": sha256_file(model_bundle_path),
                "cache_file_sha": sha256_file(INFERENCE_SUMMARY_PATH),
                "model_manifest_sha": sha256_file(model_bundle_path),
                "model_count": 2,
                "model_shas": {
                    f"fold_{fold}/edge_predictor_best.pth": fold_models[fold]["checkpoint_sha256"]
                    for fold in (0, 1)
                },
                "selected_mode": "twofold_probability_mean_then_single_decode",
                "selected_model": "exp006 folds 0 and 1, equal weights",
                "test_prediction_content_sha": submission_sha,
                "submission_sha": submission_sha,
                "row_count": len(reloaded),
                "group_count": len(test_stems),
            },
            "submission_validation": {
                "passed": True,
                "checked_at": datetime.now(UTC).isoformat(),
                "row_count": len(reloaded),
                "id_column": "id",
                "target_columns": columns,
                "duplicate_id_count": 0,
                "missing_value_count": 0,
                "infinite_value_count": 0,
                "errors": [],
                "fallback_rows": 0,
            },
        },
        "notes": (
            "Two exp006 fold probabilities were averaged before one graph decode. "
            "Inference and CSV validation completed; competition submission is separate."
        ),
    },
)
atomic_json(METRICS_PATH, metrics)

required_outputs = [
    SUBMISSION_PATH,
    INFERENCE_SUMMARY_PATH,
    METRICS_PATH,
    PREDICTIONS_ROOT,
]
missing_outputs = [str(path) for path in required_outputs if not path.exists()]
if missing_outputs:
    raise FileNotFoundError(f"missing inference outputs: {missing_outputs}")
print(json.dumps(inference_summary, indent=2), flush=True)
print("Created and validated:", SUBMISSION_PATH, flush=True)
print("No Kaggle competition submission was made.", flush=True)
