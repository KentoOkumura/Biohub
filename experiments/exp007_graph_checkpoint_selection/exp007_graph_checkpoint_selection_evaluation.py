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
# # exp007 checkpoint selection and outer official evaluation
#
# This notebook scores all six epoch checkpoints on the internal selection videos,
# applies the proxy and official graph selectors with the same tie-break, and evaluates
# only their unique selected checkpoints on the outer embryos. Outer labels never enter
# selection, and the notebook never creates a competition submission.

# %% [markdown]
# ## Contents
# 1. Configuration, hashing, and manifest helpers
# 2. Runtime, offline dependencies, and pinned source
# 3. Resolve and validate six epoch checkpoints and the split contract
# 4. Candidate-preserving prediction helpers
# 5. Internal checkpoint scoring and fixed selector choice
# 6. Unique-checkpoint outer smoke and runtime gate
# 7. Outer held-out evaluation with the selected checkpoints
# 8. Metrics, manifests, and artifact contract

# %% [markdown]
# ## 1. Configuration, hashing, and manifest helpers

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

EXPERIMENT = "exp007_graph_checkpoint_selection"
TRAIN_KERNEL_ID = "kentookumura/exp007-graph-checkpoint-selection-train"
EVALUATION_KERNEL_ID = "kentookumura/exp007-graph-checkpoint-selection-evaluation"
WORKING_ROOT = Path.cwd()
CONFIG_PATH = WORKING_ROOT / "config.yaml"
SOURCE_ROOT = WORKING_ROOT / "official_source"
SOURCE_MANIFEST_PATH = SOURCE_ROOT / "SOURCE.json"
METRICS_PATH = WORKING_ROOT / "metrics.json"
ARTIFACTS_ROOT = WORKING_ROOT / "artifacts"
INTERNAL_SELECTION_ROOT = ARTIFACTS_ROOT / "internal_selection"
OUTER_PREDICTIONS_ROOT = ARTIFACTS_ROOT / "outer_predictions"
CHECKPOINT_SELECTION_PATH = ARTIFACTS_ROOT / "checkpoint_selection.json"
PREDICTION_MANIFEST_PATH = ARTIFACTS_ROOT / "prediction_manifest.json"
PER_SAMPLE_METRICS_PATH = ARTIFACTS_ROOT / "per_sample_metrics.json"
OFFICIAL_SUMMARY_PATH = ARTIFACTS_ROOT / "official_metric_summary.json"
EVALUATION_SUMMARY_PATH = ARTIFACTS_ROOT / "evaluation_summary.json"

config = yaml.safe_load(CONFIG_PATH.read_text())
validation_cfg = config["validation"]
inference_cfg = config["model"]["inference"]
runtime_cfg = config["runtime"]
started = time.monotonic()

print("Experiment:", EXPERIMENT)
print("Source commit:", config["source"]["commit"])
print("Prediction config used for evaluation:", json.dumps(inference_cfg, indent=2))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_arrays(arrays: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update(array.dtype.str.encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    mismatches = {}
    for relative, expected in manifest["files"].items():
        observed = sha256_file(SOURCE_ROOT / relative)
        if observed != expected:
            mismatches[relative] = {"expected": expected, "observed": observed}
    if mismatches:
        raise RuntimeError(f"pinned source hash mismatch: {mismatches}")
    print(f"Verified {len(manifest['files'])} pinned source files.")
    return manifest


def load_unique_model_bundle() -> tuple[Path, dict[str, Any]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in Path("/kaggle/input").rglob("model_manifest.json"):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if (
            payload.get("experiment") == EXPERIMENT
            and payload.get("manifest_type") == "embryo_holdout_epoch_checkpoint_bundle"
        ):
            candidates.append((path, payload))
    if len(candidates) != 1:
        paths = [str(path) for path, _ in candidates]
        raise FileNotFoundError(f"expected one exp007 epoch-checkpoint bundle, found {paths}")
    return candidates[0]


def load_train_metrics() -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for path in Path("/kaggle/input").rglob("metrics.json"):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if (
            payload.get("experiment") == EXPERIMENT
            and payload.get("evidence", {}).get("artifacts", {}).get("model_count") == 6
        ):
            candidates.append(payload)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one exp007 train metrics record, found {len(candidates)}"
        )
    return candidates[0]


CANDIDATE_SCHEMA = {
    "version": int(inference_cfg["candidate_schema_version"]),
    "arrays": {
        "coords_tzyx": "int16 [node,4], original-resolution coordinates",
        "detection_probability": "float32 [node], sigmoid probability after detection TTA",
        "edge_source_index": "int32 [candidate], index into coords_tzyx",
        "edge_target_index": "int32 [candidate], index into coords_tzyx",
        "edge_probability": "float32 [candidate], every softmax score before thresholding",
        "edge_distance_downsampled": (
            "float32 [candidate], organizer edge distance in downsampled-grid units"
        ),
        "edge_valid_mask": "bool [candidate], real unpadded node pairs",
        "edge_threshold_mask": "bool [candidate], probability strictly above configured threshold",
        "edge_graph_input_mask": "bool [candidate], edge passed to graph construction before ILP",
        "edge_final_selection_mask": "bool [candidate], edge present after ILP",
        "edge_pair_index": "int16 [candidate], index into pair metadata",
        "pair_t_source": "int16 [pair]",
        "pair_t_target": "int16 [pair]",
        "pair_source_start": "int32 [pair]",
        "pair_source_count": "int32 [pair]",
        "pair_target_start": "int32 [pair]",
        "pair_target_count": "int32 [pair]",
        "pair_score_offset": "int64 [pair]",
        "pair_score_count": "int64 [pair]",
    },
    "flattening": (
        "Each frame-pair score matrix is C-order with source index outer and target index inner."
    ),
    "threshold_semantics": "strict greater-than, identical to pinned organizer prediction source",
}

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
print("Numerical stack:", {"numpy": np.__version__, "scipy": scipy.__version__})
print("Visible GPUs:", gpu_names)

source_manifest = verify_source()
source_manifest_sha = sha256_file(SOURCE_MANIFEST_PATH)
sys.path.insert(0, str(SOURCE_ROOT / "src"))
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))

import tracksdata as td  # noqa: E402
from evaluate import evaluate_pairs  # noqa: E402
from predict_unet_transformer import (  # noqa: E402
    PredictConfig,
    _load_frame,
    load_model,
    pool_kernel_from_um,
    suppress_output,
)
from tracking_cellmot.io import open_dataset, save_graph  # noqa: E402
from tracking_cellmot.metrics import summarise  # noqa: E402
from train_unet_transformer import extract_pos_features  # noqa: E402

# %% [markdown]
# ## 3. Resolve and validate six epoch checkpoints and the split contract
#
# The public checkpoint is not searched or copied. The bundle from the attached
# exp005 train kernel is the sole model input. Its split manifest must prove that
# every evaluation video belongs to the embryo excluded from that fold's training.

# %%
model_bundle_path, model_bundle = load_unique_model_bundle()
train_metrics = load_train_metrics()
if not model_bundle.get("trained_from_scratch"):
    raise RuntimeError("model bundle does not attest scratch training")
if model_bundle.get("upstream_checkpoint_loaded"):
    raise RuntimeError("model bundle indicates an upstream checkpoint was loaded")
if model_bundle.get("source_commit") != source_manifest["commit"]:
    raise RuntimeError("training and inference source commits differ")
if int(model_bundle.get("epochs", -1)) != int(config["model"]["training"]["epochs"]):
    raise RuntimeError("training epoch count differs from the experiment contract")
if int(model_bundle.get("model_count", -1)) != 6 or len(model_bundle.get("folds", [])) != 2:
    raise RuntimeError("model bundle must contain six checkpoints across two folds")

train_artifacts_root = model_bundle_path.parent
train_split_path = train_artifacts_root / model_bundle["split_manifest"]
if sha256_file(train_split_path) != model_bundle["split_manifest_sha256"]:
    raise RuntimeError("training split manifest SHA-256 mismatch")
splits = json.loads(train_split_path.read_text())
if len(splits) != 2:
    raise RuntimeError(f"expected two split records, found {len(splits)}")

expected_epoch_indices = [int(value) for value in config["model"]["training"]["save_epoch_indices"]]
fold_models: dict[int, dict[str, Any]] = {}
for record in model_bundle["folds"]:
    fold = int(record["fold"])
    model_config_path = train_artifacts_root / record["model_config"]
    fold_manifest_path = train_artifacts_root / record["fold_manifest"]
    if sha256_file(model_config_path) != record["model_config_sha256"]:
        raise RuntimeError(f"fold {fold} model config SHA-256 mismatch")
    if sha256_file(fold_manifest_path) != record["fold_manifest_sha256"]:
        raise RuntimeError(f"fold {fold} model manifest SHA-256 mismatch")
    fold_manifest = json.loads(fold_manifest_path.read_text())
    if fold_manifest["outer_evaluation_used_for_selection"]:
        raise RuntimeError(f"fold {fold} used its outer evaluation embryo for selection")
    if fold_manifest["outer_evaluation_embryo"] != record["outer_evaluation_embryo"]:
        raise RuntimeError(f"fold {fold} model and fold manifests disagree")
    if int(record.get("checkpoint_count", -1)) != len(expected_epoch_indices):
        raise RuntimeError(f"fold {fold} does not contain three checkpoint records")

    checkpoints: dict[int, dict[str, Any]] = {}
    for checkpoint_record in record["checkpoints"]:
        epoch_index = int(checkpoint_record["epoch_index"])
        checkpoint_path = train_artifacts_root / checkpoint_record["checkpoint"]
        if sha256_file(checkpoint_path) != checkpoint_record["checkpoint_sha256"]:
            raise RuntimeError(f"fold {fold} epoch {epoch_index} checkpoint SHA mismatch")
        checkpoints[epoch_index] = {
            **checkpoint_record,
            "checkpoint_path": checkpoint_path,
        }
    if sorted(checkpoints) != expected_epoch_indices:
        raise RuntimeError(
            f"fold {fold} expected checkpoint epochs {expected_epoch_indices}, "
            f"found {sorted(checkpoints)}"
        )
    fold_models[fold] = {
        **record,
        "model_config_path": model_config_path,
        "fold_manifest_path": fold_manifest_path,
        "fold_manifest": fold_manifest,
        "checkpoints": checkpoints,
    }
if set(fold_models) != {0, 1}:
    raise RuntimeError(f"expected model folds 0 and 1, found {sorted(fold_models)}")

train_dir = find_competition_dir("train")
paired_stems = sorted(
    path.name[:-5]
    for path in train_dir.glob("*.zarr")
    if (train_dir / f"{path.name[:-5]}.geff").exists()
)
if len(paired_stems) != int(validation_cfg["expected_sample_count"]):
    raise RuntimeError(f"expected 199 train datasets, found {len(paired_stems)}")

split_by_fold = {int(item["split"]): item for item in splits}
sample_to_fold: dict[str, int] = {}
for fold, split in split_by_fold.items():
    expected_spec = next(
        item for item in validation_cfg["outer_folds"] if int(item["fold"]) == fold
    )
    train_embryo = str(expected_spec["train_embryo"])
    evaluation_embryo = str(expected_spec["evaluation_embryo"])
    if split["outer_train_embryo"] != train_embryo:
        raise RuntimeError(f"fold {fold} outer training embryo differs from config")
    if split["outer_evaluation_embryo"] != evaluation_embryo:
        raise RuntimeError(f"fold {fold} outer evaluation embryo differs from config")
    if any(embryo_id(name) != train_embryo for name in split["train"] + split["test"]):
        raise RuntimeError(f"fold {fold} contains evaluation embryo in training or selection")
    if any(embryo_id(name) != evaluation_embryo for name in split["outer_evaluation"]):
        raise RuntimeError(f"fold {fold} outer evaluation list contains another embryo")
    expected_counts = (
        int(expected_spec["expected_gradient_update_count"]),
        int(expected_spec["expected_checkpoint_selection_count"]),
        int(expected_spec["expected_evaluation_count"]),
    )
    observed_counts = (len(split["train"]), len(split["test"]), len(split["outer_evaluation"]))
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"fold {fold} split counts differ: expected {expected_counts}, found {observed_counts}"
        )
    for name in split["outer_evaluation"]:
        if name in sample_to_fold:
            raise RuntimeError(f"sample appears in two outer evaluation folds: {name}")
        sample_to_fold[name] = fold
if set(sample_to_fold) != set(paired_stems):
    raise RuntimeError(
        "outer evaluation folds do not cover every paired training dataset exactly once"
    )

print("Model bundle:", model_bundle_path)
print("Split manifest:", train_split_path)
print(
    "Checkpoint counts:",
    {fold: len(fold_models[fold]["checkpoints"]) for fold in sorted(fold_models)},
)
print(
    "Evaluation counts:",
    {fold: len(split_by_fold[fold]["outer_evaluation"]) for fold in sorted(split_by_fold)},
)

# %% [markdown]
# ## 4. Candidate-preserving prediction helpers
#
# The following code follows the pinned organizer `predict_video` operation order.
# It additionally retains each detection probability, every real source-target
# edge score before thresholding, the strict threshold mask, the graph-input mask,
# and the final ILP selection mask. Those arrays do not feed back into decoding.


# %%
def detect_cells_with_scores(
    det_logits: torch.Tensor,
    t: int,
    det_threshold: float,
    pool_kernel: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    logits = det_logits.unsqueeze(0)
    pad = tuple(k // 2 for k in pool_kernel)
    pooled = F.max_pool3d(logits, pool_kernel, stride=1, padding=pad)
    probabilities = torch.sigmoid(logits)
    is_peak = (logits == pooled) & (probabilities > det_threshold)
    peak_idx = torch.nonzero(is_peak[0, 0])
    if peak_idx.shape[0] == 0:
        return np.empty((0, 4), dtype=np.int16), np.empty((0,), dtype=np.float32)
    coords = peak_idx.float().cpu().numpy()
    scores = probabilities[0, 0][tuple(peak_idx.T)].float().cpu().numpy().astype(np.float32)
    t_col = np.full((len(coords), 1), t, dtype=np.float32)
    return np.concatenate([t_col, coords], axis=1).astype(np.int16), scores


def build_graph_with_node_ids(
    coords: np.ndarray,
    edges: list[tuple[int, int, float, float]],
) -> tuple[td.graph.InMemoryGraph, list[int]]:
    graph = td.graph.InMemoryGraph()
    for key in ["z", "y", "x"]:
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    node_ids = graph.bulk_add_nodes(
        [{"t": int(t), "z": float(z), "y": float(y), "x": float(x)} for t, z, y, x in coords]
    )
    node_ids = [int(node_id) for node_id in node_ids]
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
    return graph, node_ids


@torch.no_grad()
def predict_video_with_candidates(
    model: Any,
    dataset_path: Path,
    device: torch.device,
    predict_cfg: PredictConfig,
    window_size: int,
    downsample: tuple[int, ...],
) -> tuple[np.ndarray, list[tuple[int, int, float, float]], dict[str, np.ndarray]]:
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
    downsample_array = np.array(downsample, dtype=np.float32)
    downsample_tensor = torch.from_numpy(downsample_array).to(device)
    voxel_size = tuple(scale * step for scale, step in zip(dataset.scale, downsample, strict=True))
    pool_kernel = pool_kernel_from_um(predict_cfg.pool_kernel_um, voxel_size)

    seen_frames: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    coord_lists: list[np.ndarray] = []
    detection_score_lists: list[np.ndarray] = []
    coord_offset: dict[int, tuple[int, int]] = {}
    global_node_count = 0
    graph_edges: list[tuple[int, int, float, float]] = []

    edge_source_parts: list[np.ndarray] = []
    edge_target_parts: list[np.ndarray] = []
    edge_probability_parts: list[np.ndarray] = []
    edge_distance_parts: list[np.ndarray] = []
    edge_valid_parts: list[np.ndarray] = []
    edge_threshold_parts: list[np.ndarray] = []
    edge_graph_input_parts: list[np.ndarray] = []
    edge_pair_index_parts: list[np.ndarray] = []
    pair_t_source: list[int] = []
    pair_t_target: list[int] = []
    pair_source_start: list[int] = []
    pair_source_count: list[int] = []
    pair_target_start: list[int] = []
    pair_target_count: list[int] = []
    pair_score_offset: list[int] = []
    pair_score_count: list[int] = []
    score_offset = 0

    stride = max(window_size - 1, 1)
    window_starts = list(range(0, frame_count - window_size + 1, stride))
    if not window_starts or window_starts[-1] + window_size < frame_count:
        last = max(frame_count - window_size, 0)
        if not window_starts or last != window_starts[-1]:
            window_starts.append(last)

    for window_start in tqdm(window_starts, desc=f"{dataset_path.name} windows", leave=False):
        frame_indices = list(range(window_start, window_start + window_size))
        images = torch.stack(
            [_load_frame(zarr_array, frame, target_shape, downsample) for frame in frame_indices]
        )
        images = ((images - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
        images = images.unsqueeze(0).to(device)
        unet_output, detection_logits = model.encode(images)
        if predict_cfg.det_tta:
            for dims in [(-1,), (-2,), (-2, -1)]:
                flipped_images = images.flip(dims)
                _, flipped_logits = model.encode(flipped_images)
                for frame_offset in range(window_size):
                    detection_logits[frame_offset] = detection_logits[
                        frame_offset
                    ] + flipped_logits[frame_offset].flip(dims)
                del flipped_images, flipped_logits
            for frame_offset in range(window_size):
                detection_logits[frame_offset] = detection_logits[frame_offset] / 4
        del images

        for frame_offset, frame in enumerate(frame_indices):
            if frame in seen_frames:
                continue
            coords, detection_scores = detect_cells_with_scores(
                detection_logits[frame_offset][0],
                frame,
                predict_cfg.det_threshold,
                pool_kernel,
            )
            coord_offset[frame] = (global_node_count, global_node_count + len(coords))
            global_node_count += len(coords)
            coord_lists.append(coords)
            detection_score_lists.append(detection_scores)
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
            source_start, source_end = coord_offset[source_time]
            target_start, target_end = coord_offset[target_time]
            source_count = source_end - source_start
            target_count = target_end - target_start
            pair_index = len(pair_t_source)
            count = source_count * target_count
            pair_t_source.append(source_time)
            pair_t_target.append(target_time)
            pair_source_start.append(source_start)
            pair_source_count.append(source_count)
            pair_target_start.append(target_start)
            pair_target_count.append(target_count)
            pair_score_offset.append(score_offset)
            pair_score_count.append(count)
            if count == 0:
                continue

            source_coords = coords_so_far[source_start:source_end]
            target_coords = coords_so_far[target_start:target_end]
            source_indices = np.arange(source_start, source_end, dtype=np.int32)
            target_indices = np.arange(target_start, target_end, dtype=np.int32)
            source_tensor = (
                torch.from_numpy(source_coords[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            )
            target_tensor = (
                torch.from_numpy(target_coords[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            )
            relative_source = source_coords.copy()
            relative_target = target_coords.copy()
            relative_source[:, 0] = frame_offset
            relative_target[:, 0] = frame_offset + 1
            window_shape = (window_size,) + image_shape[1:]
            source_position = (
                torch.from_numpy(extract_pos_features(relative_source, window_shape))
                .unsqueeze(0)
                .to(device)
            )
            target_position = (
                torch.from_numpy(extract_pos_features(relative_target, window_shape))
                .unsqueeze(0)
                .to(device)
            )
            source_valid = torch.ones(1, source_count, dtype=torch.bool, device=device)
            target_valid = torch.ones(1, target_count, dtype=torch.bool, device=device)
            source_features = model._index_features(
                unet_output[:, frame_offset], source_tensor, source_valid
            )
            target_features = model._index_features(
                unet_output[:, frame_offset + 1], target_tensor, target_valid
            )
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
            if predict_cfg.edge_activation == "softmax":
                probabilities = torch.softmax(edge_logits, dim=0).cpu().numpy()
            else:
                probabilities = torch.sigmoid(edge_logits).cpu().numpy()

            flat_probabilities = probabilities.astype(np.float32, copy=False).reshape(-1)
            flat_source = np.repeat(source_indices, target_count)
            flat_target = np.tile(target_indices, source_count)
            distances = (
                np.linalg.norm(
                    source_coords[:, None, 1:].astype(np.float32)
                    - target_coords[None, :, 1:].astype(np.float32),
                    axis=2,
                )
                .astype(np.float32)
                .reshape(-1)
            )
            valid_mask = np.ones(count, dtype=np.bool_)
            threshold_mask = flat_probabilities > float(predict_cfg.threshold)
            graph_input_mask = np.zeros(count, dtype=np.bool_)
            sorted_candidates = sorted(
                [
                    (
                        float(flat_probabilities[index]),
                        int(index // target_count),
                        int(index % target_count),
                        int(index),
                    )
                    for index in np.flatnonzero(threshold_mask)
                ],
                reverse=True,
            )
            children_count: dict[int, int] = {}
            parents_count: dict[int, int] = {}
            for probability, source_local, target_local, flat_index in sorted_candidates:
                n_children = children_count.get(source_local, 0)
                n_parents = parents_count.get(target_local, 0)
                if (
                    predict_cfg.max_children_per_node is not None
                    and n_children >= predict_cfg.max_children_per_node
                ):
                    continue
                if (
                    predict_cfg.max_parents_per_node is not None
                    and n_parents >= predict_cfg.max_parents_per_node
                ):
                    continue
                graph_input_mask[flat_index] = True
                source_global = int(flat_source[flat_index])
                target_global = int(flat_target[flat_index])
                graph_edges.append(
                    (
                        source_global,
                        target_global,
                        probability,
                        float(distances[flat_index]),
                    )
                )
                children_count[source_local] = n_children + 1
                parents_count[target_local] = n_parents + 1

            edge_source_parts.append(flat_source)
            edge_target_parts.append(flat_target)
            edge_probability_parts.append(flat_probabilities)
            edge_distance_parts.append(distances)
            edge_valid_parts.append(valid_mask)
            edge_threshold_parts.append(threshold_mask)
            edge_graph_input_parts.append(graph_input_mask)
            edge_pair_index_parts.append(np.full(count, pair_index, dtype=np.int16))
            score_offset += count
        del unet_output, detection_logits

    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
    detection_probabilities = (
        np.concatenate(detection_score_lists)
        if detection_score_lists
        else np.empty((0,), dtype=np.float32)
    )
    coords = coords.astype(np.float32)
    coords[:, 1:] *= downsample_array
    coords = coords.astype(np.int16)

    def concatenate(parts: list[np.ndarray], dtype: Any) -> np.ndarray:
        return (
            np.concatenate(parts).astype(dtype, copy=False)
            if parts
            else np.empty((0,), dtype=dtype)
        )

    arrays = {
        "coords_tzyx": coords,
        "detection_probability": detection_probabilities,
        "edge_source_index": concatenate(edge_source_parts, np.int32),
        "edge_target_index": concatenate(edge_target_parts, np.int32),
        "edge_probability": concatenate(edge_probability_parts, np.float32),
        "edge_distance_downsampled": concatenate(edge_distance_parts, np.float32),
        "edge_valid_mask": concatenate(edge_valid_parts, np.bool_),
        "edge_threshold_mask": concatenate(edge_threshold_parts, np.bool_),
        "edge_graph_input_mask": concatenate(edge_graph_input_parts, np.bool_),
        "edge_pair_index": concatenate(edge_pair_index_parts, np.int16),
        "pair_t_source": np.asarray(pair_t_source, dtype=np.int16),
        "pair_t_target": np.asarray(pair_t_target, dtype=np.int16),
        "pair_source_start": np.asarray(pair_source_start, dtype=np.int32),
        "pair_source_count": np.asarray(pair_source_count, dtype=np.int32),
        "pair_target_start": np.asarray(pair_target_start, dtype=np.int32),
        "pair_target_count": np.asarray(pair_target_count, dtype=np.int32),
        "pair_score_offset": np.asarray(pair_score_offset, dtype=np.int64),
        "pair_score_count": np.asarray(pair_score_count, dtype=np.int64),
    }
    if len(arrays["coords_tzyx"]) != len(arrays["detection_probability"]):
        raise RuntimeError("detection coordinates and scores are not aligned")
    edge_lengths = {
        len(arrays[key])
        for key in (
            "edge_source_index",
            "edge_target_index",
            "edge_probability",
            "edge_distance_downsampled",
            "edge_valid_mask",
            "edge_threshold_mask",
            "edge_graph_input_mask",
            "edge_pair_index",
        )
    }
    if len(edge_lengths) != 1:
        raise RuntimeError(f"edge candidate arrays are not aligned: {edge_lengths}")
    if int(arrays["edge_graph_input_mask"].sum()) != len(graph_edges):
        raise RuntimeError("graph input mask does not match graph edge count")
    if len(pair_t_source) != max(frame_count - 1, 0):
        raise RuntimeError(
            f"expected {frame_count - 1} consecutive frame pairs, found {len(pair_t_source)}"
        )
    return coords, graph_edges, arrays


def add_final_selection_mask(
    arrays: dict[str, np.ndarray],
    node_ids: list[int],
    selected_graph: Any,
) -> None:
    selected_pairs: set[tuple[int, int]] = set()
    if selected_graph.num_edges() > 0:
        selected_pairs = {
            (int(row["source_id"]), int(row["target_id"]))
            for row in selected_graph.edge_attrs().iter_rows(named=True)
        }
    selected_mask = np.zeros(len(arrays["edge_probability"]), dtype=np.bool_)
    for index in np.flatnonzero(arrays["edge_graph_input_mask"]):
        source = node_ids[int(arrays["edge_source_index"][index])]
        target = node_ids[int(arrays["edge_target_index"][index])]
        selected_mask[index] = (source, target) in selected_pairs
    if int(selected_mask.sum()) != selected_graph.num_edges():
        raise RuntimeError("final selection mask does not match solved graph")
    arrays["edge_final_selection_mask"] = selected_mask


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

INTERNAL_SELECTION_ROOT.mkdir(parents=True, exist_ok=True)
OUTER_PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
internal_prediction_records: list[dict[str, Any]] = []
outer_prediction_records: list[dict[str, Any]] = []


def relative_artifact_path(path: Path) -> str:
    return path.relative_to(ARTIFACTS_ROOT).as_posix()


def write_video_prediction(
    model: Any,
    window_size: int,
    downsample: tuple[int, ...],
    fold: int,
    epoch_index: int,
    sample_name: str,
    stage: str,
    predictions_root: Path,
    candidates_root: Path,
) -> dict[str, Any]:
    video_started = time.monotonic()
    coords, graph_edges, arrays = predict_video_with_candidates(
        model,
        train_dir / sample_name,
        torch.device("cuda:0"),
        predict_config,
        window_size,
        downsample,
    )
    graph, node_ids = build_graph_with_node_ids(coords, graph_edges)
    if predict_config.use_ilp and graph.num_edges() > 0:
        solver = td.solvers.ILPSolver(
            edge_weight=predict_config.ilp_edge_weight * td.EdgeAttr("edge_prob"),
            appearance_weight=predict_config.ilp_appearance_weight,
            disappearance_weight=predict_config.ilp_disappearance_weight,
            division_weight=predict_config.ilp_division_weight,
        )
        with suppress_output():
            graph = solver.solve(graph)
    add_final_selection_mask(arrays, node_ids, graph)

    predictions_root.mkdir(parents=True, exist_ok=True)
    candidates_root.mkdir(parents=True, exist_ok=True)
    graph_path = predictions_root / f"{sample_name}.geff"
    cache_path = candidates_root / f"{sample_name}.npz"
    save_graph(graph, graph_path)
    np.savez_compressed(cache_path, **arrays)
    checkpoint_record = fold_models[fold]["checkpoints"][epoch_index]
    record = {
        "stage": stage,
        "sample": sample_name,
        "embryo": embryo_id(sample_name),
        "fold": fold,
        "epoch_index": epoch_index,
        "outer_train_embryo": fold_models[fold]["outer_train_embryo"],
        "outer_evaluation_embryo": fold_models[fold]["outer_evaluation_embryo"],
        "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
        "graph": relative_artifact_path(graph_path),
        "graph_tree_sha256": sha256_tree(graph_path),
        "candidate_cache": relative_artifact_path(cache_path),
        "candidate_file_sha256": sha256_file(cache_path),
        "candidate_content_sha256": sha256_arrays(arrays),
        "node_count": int(len(arrays["coords_tzyx"])),
        "edge_score_count": int(len(arrays["edge_probability"])),
        "edge_threshold_count": int(arrays["edge_threshold_mask"].sum()),
        "graph_input_edge_count": int(arrays["edge_graph_input_mask"].sum()),
        "final_edge_count": int(arrays["edge_final_selection_mask"].sum()),
        "frame_pair_count": int(len(arrays["pair_t_source"])),
        "elapsed_seconds": time.monotonic() - video_started,
    }
    del arrays, graph
    torch.cuda.empty_cache()
    return record


def load_fold_epoch_model(fold: int, epoch_index: int) -> tuple[Any, int, tuple[int, ...]]:
    checkpoint = fold_models[fold]["checkpoints"][epoch_index]
    model, window_size, downsample = load_model(
        checkpoint["checkpoint_path"],
        torch.device("cuda:0"),
    )
    return model, int(window_size), tuple(int(value) for value in downsample)


def evaluate_prediction_directory(
    predictions_root: Path,
    expected_names: list[str],
    fold: int,
    epoch_index: int,
) -> list[dict[str, Any]]:
    observed_names = sorted(path.stem for path in predictions_root.glob("*.geff"))
    if observed_names != sorted(expected_names):
        raise RuntimeError(
            f"fold {fold} epoch {epoch_index} prediction names differ from the contract"
        )
    metric_rows, skipped = evaluate_pairs(
        predictions_root,
        train_dir,
        max_distance=float(validation_cfg["max_matching_distance_um"]),
    )
    if skipped or len(metric_rows) != len(expected_names):
        raise RuntimeError(
            f"fold {fold} epoch {epoch_index} official evaluation incomplete: "
            f"rows={len(metric_rows)} skipped={skipped}"
        )
    return [
        {
            "sample": sample_name,
            "embryo": embryo_id(sample_name),
            "fold": fold,
            "epoch_index": epoch_index,
            **row,
        }
        for sample_name, row in zip(sorted(expected_names), metric_rows, strict=True)
    ]


def metric_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"sample", "embryo", "fold", "epoch_index", "selector"}
    }


def select_latest_epoch(
    records: list[dict[str, Any]],
    score_key: str,
) -> dict[str, Any]:
    if validation_cfg["selector_tie_break"] != "latest_epoch":
        raise RuntimeError("only the fixed latest-epoch tie break is supported")
    return max(records, key=lambda item: (float(item[score_key]), int(item["epoch_index"])))


# %% [markdown]
# ## 5. Internal checkpoint scoring and fixed selector choice
#
# Every epoch checkpoint predicts exactly the same internal selection videos from
# its outer training embryo. The proxy score comes from the training-loop
# evaluation. The treatment score is the pinned official graph summary over the
# saved internal predictions. Outer-evaluation videos and labels are not used here.

# %%
internal_stage_started = time.monotonic()
internal_checkpoint_records: list[dict[str, Any]] = []
for fold in (0, 1):
    split = split_by_fold[fold]
    internal_names = sorted(split["test"])
    forbidden_outer_names = set(split["outer_evaluation"])
    if set(internal_names) & forbidden_outer_names:
        raise RuntimeError(f"fold {fold} internal and outer selection data overlap")
    for epoch_index in expected_epoch_indices:
        stage_root = INTERNAL_SELECTION_ROOT / f"fold_{fold}" / f"epoch_{epoch_index}"
        predictions_root = stage_root / "predictions"
        candidates_root = stage_root / "candidates"
        model, window_size, downsample = load_fold_epoch_model(fold, epoch_index)
        checkpoint_predictions: list[dict[str, Any]] = []
        for sample_name in tqdm(
            internal_names,
            desc=f"Fold {fold} epoch {epoch_index} internal selection",
        ):
            record = write_video_prediction(
                model,
                window_size,
                downsample,
                fold,
                epoch_index,
                sample_name,
                "internal_checkpoint_selection",
                predictions_root,
                candidates_root,
            )
            checkpoint_predictions.append(record)
            internal_prediction_records.append(record)
        del model
        torch.cuda.empty_cache()

        named_rows = evaluate_prediction_directory(
            predictions_root,
            internal_names,
            fold,
            epoch_index,
        )
        official_summary = summarise([metric_payload(row) for row in named_rows])
        checkpoint_record = fold_models[fold]["checkpoints"][epoch_index]
        internal_checkpoint_records.append(
            {
                "fold": fold,
                "epoch_index": epoch_index,
                "checkpoint": checkpoint_record["checkpoint"],
                "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
                "internal_samples": internal_names,
                "internal_sample_count": len(internal_names),
                "outer_evaluation_used_for_selection": False,
                "proxy_selection_score": float(checkpoint_record["proxy_selection_score"]),
                "official_selection_score": float(official_summary["score"]),
                "official_summary": official_summary,
                "prediction_manifest_sha256": sha256_payload(
                    [[item["sample"], item["graph_tree_sha256"]] for item in checkpoint_predictions]
                ),
                "candidate_content_sha256": sha256_payload(
                    [
                        [item["sample"], item["candidate_content_sha256"]]
                        for item in checkpoint_predictions
                    ]
                ),
                "per_sample_metrics": named_rows,
            }
        )

reference_selector = str(validation_cfg["reference_selector"])
treatment_selector = str(validation_cfg["treatment_selector"])
selector_score_keys = {
    reference_selector: "proxy_selection_score",
    treatment_selector: "official_selection_score",
}
selection_by_fold: dict[int, dict[str, dict[str, Any]]] = {}
for fold in (0, 1):
    fold_records = [item for item in internal_checkpoint_records if item["fold"] == fold]
    selection_by_fold[fold] = {}
    for selector, score_key in selector_score_keys.items():
        selected = select_latest_epoch(fold_records, score_key)
        selection_by_fold[fold][selector] = {
            "selector": selector,
            "score_key": score_key,
            "score": float(selected[score_key]),
            "epoch_index": int(selected["epoch_index"]),
            "checkpoint": selected["checkpoint"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "internal_sample_count": selected["internal_sample_count"],
        }

unique_epochs_by_fold = {
    fold: sorted(
        {int(selection_by_fold[fold][selector]["epoch_index"]) for selector in selector_score_keys}
    )
    for fold in (0, 1)
}
checkpoint_selection = {
    "experiment": EXPERIMENT,
    "selection_data": validation_cfg["selector_data"],
    "outer_evaluation_used_for_selection": False,
    "tie_break": validation_cfg["selector_tie_break"],
    "selectors": list(selector_score_keys),
    "folds": {
        f"fold_{fold}": {
            "outer_train_embryo": fold_models[fold]["outer_train_embryo"],
            "outer_evaluation_embryo": fold_models[fold]["outer_evaluation_embryo"],
            "internal_samples": sorted(split_by_fold[fold]["test"]),
            "epoch_scores": [item for item in internal_checkpoint_records if item["fold"] == fold],
            "selected": selection_by_fold[fold],
            "unique_selected_epoch_indices": unique_epochs_by_fold[fold],
        }
        for fold in (0, 1)
    },
}
atomic_json(CHECKPOINT_SELECTION_PATH, checkpoint_selection)
checkpoint_selection_sha = sha256_file(CHECKPOINT_SELECTION_PATH)
internal_stage_elapsed_seconds = time.monotonic() - internal_stage_started
print(json.dumps(json_safe(checkpoint_selection), indent=2))


# %% [markdown]
# ## 6. Unique-checkpoint outer smoke and runtime gate
#
# One outer video is predicted for each unique selected checkpoint. The measured
# remaining work, the completed internal-selection stage, a conservative
# multiplier, and a fixed reserve must fit the 11.5-hour gate. The gate does not
# reduce videos, epochs, resolution, thresholds, or the number of selectors.

# %%
outer_smoke_records: list[dict[str, Any]] = []
for fold in (0, 1):
    outer_names = sorted(split_by_fold[fold]["outer_evaluation"])
    smoke_count = int(inference_cfg["smoke_videos_per_unique_checkpoint"])
    smoke_names = outer_names[:smoke_count]
    for epoch_index in unique_epochs_by_fold[fold]:
        stage_root = OUTER_PREDICTIONS_ROOT / f"fold_{fold}" / f"epoch_{epoch_index}"
        model, window_size, downsample = load_fold_epoch_model(fold, epoch_index)
        for sample_name in smoke_names:
            record = write_video_prediction(
                model,
                window_size,
                downsample,
                fold,
                epoch_index,
                sample_name,
                "outer_evaluation",
                stage_root / "predictions",
                stage_root / "candidates",
            )
            outer_prediction_records.append(record)
            outer_smoke_records.append(record)
        del model
        torch.cuda.empty_cache()

projected_remaining_outer_seconds = 0.0
for fold in (0, 1):
    outer_count = len(split_by_fold[fold]["outer_evaluation"])
    for epoch_index in unique_epochs_by_fold[fold]:
        matching = [
            item
            for item in outer_smoke_records
            if item["fold"] == fold and item["epoch_index"] == epoch_index
        ]
        if not matching:
            raise RuntimeError(f"missing outer smoke for fold {fold} epoch {epoch_index}")
        mean_seconds = sum(item["elapsed_seconds"] for item in matching) / len(matching)
        projected_remaining_outer_seconds += mean_seconds * (outer_count - len(matching))

elapsed_before_full_outer = time.monotonic() - started
projected_total_seconds = (
    elapsed_before_full_outer
    + projected_remaining_outer_seconds * float(inference_cfg["runtime_projection_multiplier"])
    + float(inference_cfg["runtime_reserve_minutes"]) * 60
)
evaluation_gate_seconds = float(inference_cfg["runtime_gate_hours"]) * 3600
run_full_evaluation = projected_total_seconds <= evaluation_gate_seconds
evaluation_gate = {
    "internal_selection_elapsed_seconds": internal_stage_elapsed_seconds,
    "outer_smoke_keys": [
        [item["fold"], item["epoch_index"], item["sample"]] for item in outer_smoke_records
    ],
    "outer_smoke_elapsed_seconds": sum(item["elapsed_seconds"] for item in outer_smoke_records),
    "unique_selected_epoch_indices": unique_epochs_by_fold,
    "projected_remaining_outer_seconds": projected_remaining_outer_seconds,
    "projection_multiplier": float(inference_cfg["runtime_projection_multiplier"]),
    "reserve_minutes": float(inference_cfg["runtime_reserve_minutes"]),
    "projected_total_seconds": projected_total_seconds,
    "gate_limit_seconds": evaluation_gate_seconds,
    "gate_passed": run_full_evaluation,
}
atomic_json(ARTIFACTS_ROOT / "evaluation_gate.json", evaluation_gate)
print(json.dumps(evaluation_gate, indent=2))


def write_outer_prediction_manifest() -> dict[str, Any]:
    expected_count = sum(
        len(split_by_fold[fold]["outer_evaluation"]) * len(unique_epochs_by_fold[fold])
        for fold in (0, 1)
    )
    payload = {
        "experiment": EXPERIMENT,
        "source_commit": source_manifest["commit"],
        "source_manifest_sha256": source_manifest_sha,
        "model_manifest_sha256": sha256_file(model_bundle_path),
        "split_manifest_sha256": sha256_file(train_split_path),
        "checkpoint_selection_sha256": checkpoint_selection_sha,
        "candidate_schema": CANDIDATE_SCHEMA,
        "candidate_schema_sha256": sha256_payload(CANDIDATE_SCHEMA),
        "completed_count": len(outer_prediction_records),
        "expected_count": expected_count,
        "unique_selected_epoch_indices": unique_epochs_by_fold,
        "predictions": sorted(
            outer_prediction_records,
            key=lambda item: (item["fold"], item["epoch_index"], item["sample"]),
        ),
    }
    atomic_json(PREDICTION_MANIFEST_PATH, payload)
    return payload


if not run_full_evaluation:
    partial_manifest = write_outer_prediction_manifest()
    partial_metrics = deep_merge(
        train_metrics,
        {
            "experiment": EXPERIMENT,
            "status": "debug_completed",
            "updated_at": datetime.now(UTC).isoformat(),
            "cv": None,
            "metric": validation_cfg["primary_metric"],
            "diagnostic_validation": {
                "checkpoint_selection": {
                    "manifest_sha256": checkpoint_selection_sha,
                    "selected": selection_by_fold,
                },
                "evaluation_gate": evaluation_gate,
            },
            "evidence": {
                "kaggle": {
                    "kernel_id": EVALUATION_KERNEL_ID,
                    "resource": {"gpu_names": gpu_names, "visible_gpu_count": len(gpu_names)},
                    "notebook_runtime_seconds": time.monotonic() - started,
                    "internet_enabled": False,
                    "kernel_source_ids": [TRAIN_KERNEL_ID],
                },
                "artifacts": {
                    "cache_file_sha": sha256_file(PREDICTION_MANIFEST_PATH),
                    "feature_schema_sha": partial_manifest["candidate_schema_sha256"],
                    "feature_content_sha": sha256_payload(
                        [item["candidate_content_sha256"] for item in outer_prediction_records]
                    ),
                    "row_count": len(outer_prediction_records),
                    "group_count": len(
                        {(item["fold"], item["epoch_index"]) for item in outer_prediction_records}
                    ),
                    "feature_count": sum(
                        item["edge_score_count"] for item in outer_prediction_records
                    ),
                    "checkpoint_selection_sha": checkpoint_selection_sha,
                },
            },
            "notes": (
                "All six checkpoints were scored on internal selection videos, but "
                "the projected unique-checkpoint outer evaluation exceeded the fixed gate."
            ),
        },
    )
    atomic_json(METRICS_PATH, partial_metrics)


# %% [markdown]
# ## 7. Outer held-out evaluation with the selected checkpoints
#
# Each unique selected checkpoint is run once per fold. Selector summaries reuse
# those fixed saved graphs. Outer ground truth is first read after the checkpoint
# selection manifest has been fixed.

# %%
if run_full_evaluation:
    completed_keys = {
        (item["fold"], item["epoch_index"], item["sample"]) for item in outer_prediction_records
    }
    for fold in (0, 1):
        outer_names = sorted(split_by_fold[fold]["outer_evaluation"])
        for epoch_index in unique_epochs_by_fold[fold]:
            remaining_names = [
                name for name in outer_names if (fold, epoch_index, name) not in completed_keys
            ]
            stage_root = OUTER_PREDICTIONS_ROOT / f"fold_{fold}" / f"epoch_{epoch_index}"
            model, window_size, downsample = load_fold_epoch_model(fold, epoch_index)
            for sample_name in tqdm(
                remaining_names,
                desc=f"Fold {fold} epoch {epoch_index} outer videos",
            ):
                outer_prediction_records.append(
                    write_video_prediction(
                        model,
                        window_size,
                        downsample,
                        fold,
                        epoch_index,
                        sample_name,
                        "outer_evaluation",
                        stage_root / "predictions",
                        stage_root / "candidates",
                    )
                )
            del model
            torch.cuda.empty_cache()

    prediction_manifest = write_outer_prediction_manifest()
    prediction_keys = [
        (item["fold"], item["epoch_index"], item["sample"])
        for item in prediction_manifest["predictions"]
    ]
    expected_keys = {
        (fold, epoch_index, sample_name)
        for fold in (0, 1)
        for epoch_index in unique_epochs_by_fold[fold]
        for sample_name in split_by_fold[fold]["outer_evaluation"]
    }
    if len(prediction_keys) != len(set(prediction_keys)):
        raise RuntimeError("outer prediction manifest contains duplicate keys")
    if set(prediction_keys) != expected_keys:
        raise RuntimeError("outer prediction manifest does not cover every required key")
    for record in prediction_manifest["predictions"]:
        fold = int(record["fold"])
        if embryo_id(record["sample"]) != fold_models[fold]["outer_evaluation_embryo"]:
            raise RuntimeError(f"outer prediction used the wrong fold model: {record['sample']}")

    outer_rows_by_model: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for fold in (0, 1):
        expected_names = sorted(split_by_fold[fold]["outer_evaluation"])
        for epoch_index in unique_epochs_by_fold[fold]:
            predictions_root = (
                OUTER_PREDICTIONS_ROOT / f"fold_{fold}" / f"epoch_{epoch_index}" / "predictions"
            )
            rows = evaluate_prediction_directory(
                predictions_root,
                expected_names,
                fold,
                epoch_index,
            )
            recheck_rows = evaluate_prediction_directory(
                predictions_root,
                expected_names,
                fold,
                epoch_index,
            )
            if json_safe(recheck_rows) != json_safe(rows):
                raise RuntimeError(
                    f"fold {fold} epoch {epoch_index} saved-graph recomputation differs"
                )
            outer_rows_by_model[(fold, epoch_index)] = rows

    selector_rows: dict[str, list[dict[str, Any]]] = {}
    selector_summaries: dict[str, dict[str, Any]] = {}
    selector_prediction_shas: dict[str, str] = {}
    for selector in selector_score_keys:
        rows: list[dict[str, Any]] = []
        selected_graph_records: list[dict[str, Any]] = []
        for fold in (0, 1):
            epoch_index = int(selection_by_fold[fold][selector]["epoch_index"])
            rows.extend(
                {
                    **item,
                    "selector": selector,
                }
                for item in outer_rows_by_model[(fold, epoch_index)]
            )
            selected_graph_records.extend(
                item
                for item in prediction_manifest["predictions"]
                if item["fold"] == fold and item["epoch_index"] == epoch_index
            )
        rows = sorted(rows, key=lambda item: item["sample"])
        if len(rows) != len(paired_stems) or {item["sample"] for item in rows} != set(paired_stems):
            raise RuntimeError(f"selector {selector} does not cover all 199 outer videos")
        selector_rows[selector] = rows
        overall = summarise([metric_payload(row) for row in rows])
        by_embryo = {
            embryo: summarise([metric_payload(row) for row in rows if row["embryo"] == embryo])
            for embryo in sorted(validation_cfg["expected_embryo_counts"])
        }
        selector_summaries[selector] = {
            "overall": overall,
            "by_embryo": by_embryo,
            "sample_count": len(rows),
            "failed_count": 0,
            "selected_epochs": {
                f"fold_{fold}": selection_by_fold[fold][selector]["epoch_index"] for fold in (0, 1)
            },
        }
        selector_prediction_shas[selector] = sha256_payload(
            [
                [
                    item["sample"],
                    item["fold"],
                    item["epoch_index"],
                    item["graph_tree_sha256"],
                ]
                for item in sorted(
                    selected_graph_records,
                    key=lambda value: (value["fold"], value["sample"]),
                )
            ]
        )

    reference_summary = selector_summaries[reference_selector]
    treatment_summary = selector_summaries[treatment_selector]
    score_delta = float(treatment_summary["overall"]["score"]) - float(
        reference_summary["overall"]["score"]
    )
    embryo_score_deltas = {
        embryo: float(treatment_summary["by_embryo"][embryo]["score"])
        - float(reference_summary["by_embryo"][embryo]["score"])
        for embryo in sorted(validation_cfg["expected_embryo_counts"])
    }
    comparison = {
        "reference_selector": reference_selector,
        "treatment_selector": treatment_selector,
        "treatment_minus_reference_score": score_delta,
        "by_embryo_treatment_minus_reference_score": embryo_score_deltas,
        "both_embryos_improved": all(value > 0 for value in embryo_score_deltas.values()),
        "execution_failure_delta": 0,
    }
    atomic_json(
        PER_SAMPLE_METRICS_PATH,
        {
            "selectors": selector_rows,
            "skipped": {selector: [] for selector in selector_score_keys},
        },
    )
    official_summary = {
        "metric": validation_cfg["primary_metric"],
        "max_matching_distance_um": float(validation_cfg["max_matching_distance_um"]),
        "selectors": selector_summaries,
        "comparison": comparison,
        "evaluated_count_per_selector": len(paired_stems),
        "recomputed_from_saved_graphs": True,
        "recomputed_summary_matches": True,
    }
    atomic_json(OFFICIAL_SUMMARY_PATH, official_summary)

    candidate_content_sha = sha256_payload(
        [
            [
                item["fold"],
                item["epoch_index"],
                item["sample"],
                item["candidate_content_sha256"],
            ]
            for item in prediction_manifest["predictions"]
        ]
    )
    combined_oof_prediction_sha = sha256_payload(selector_prediction_shas)
    elapsed_seconds = time.monotonic() - started
    evaluation_summary = {
        "experiment": EXPERIMENT,
        "source_commit": source_manifest["commit"],
        "source_manifest_sha256": source_manifest_sha,
        "model_manifest_sha256": sha256_file(model_bundle_path),
        "split_manifest_sha256": sha256_file(train_split_path),
        "checkpoint_selection_sha256": checkpoint_selection_sha,
        "prediction_manifest_sha256": sha256_file(PREDICTION_MANIFEST_PATH),
        "candidate_schema_sha256": prediction_manifest["candidate_schema_sha256"],
        "candidate_content_sha256": candidate_content_sha,
        "selector_prediction_sha256": selector_prediction_shas,
        "oof_prediction_sha256": combined_oof_prediction_sha,
        "unique_outer_prediction_count": len(prediction_manifest["predictions"]),
        "evaluated_count_per_selector": len(paired_stems),
        "edge_score_count": sum(
            item["edge_score_count"] for item in prediction_manifest["predictions"]
        ),
        "elapsed_seconds": elapsed_seconds,
        "gate": evaluation_gate,
    }
    atomic_json(EVALUATION_SUMMARY_PATH, evaluation_summary)

    model_shas = {
        checkpoint["checkpoint_key"]: checkpoint["checkpoint_sha256"]
        for fold_record in model_bundle["folds"]
        for checkpoint in fold_record["checkpoints"]
    }
    final_metrics = deep_merge(
        train_metrics,
        {
            "experiment": EXPERIMENT,
            "status": "running",
            "updated_at": datetime.now(UTC).isoformat(),
            "cv": {
                "strategy": validation_cfg["strategy"],
                "selectors": json_safe(selector_summaries),
                "comparison": json_safe(comparison),
                "sample_count_per_selector": len(paired_stems),
            },
            "metric": validation_cfg["primary_metric"],
            "public_lb": None,
            "private_lb": None,
            "diagnostic_validation": {
                "checkpoint_selection": {
                    "manifest_sha256": checkpoint_selection_sha,
                    "selected": selection_by_fold,
                },
                "evaluation_gate": evaluation_gate,
            },
            "evidence": {
                "kaggle": {
                    "kernel_id": EVALUATION_KERNEL_ID,
                    "resource": {"gpu_names": gpu_names, "visible_gpu_count": len(gpu_names)},
                    "notebook_runtime_seconds": elapsed_seconds,
                    "internet_enabled": False,
                    "kernel_source_ids": [TRAIN_KERNEL_ID],
                },
                "artifacts": {
                    "cache_file_sha": sha256_file(PREDICTION_MANIFEST_PATH),
                    "feature_schema_sha": prediction_manifest["candidate_schema_sha256"],
                    "feature_content_sha": candidate_content_sha,
                    "row_count": len(prediction_manifest["predictions"]),
                    "group_count": sum(len(value) for value in unique_epochs_by_fold.values()),
                    "feature_count": evaluation_summary["edge_score_count"],
                    "source_manifest_sha": source_manifest_sha,
                    "model_manifest_sha": sha256_file(model_bundle_path),
                    "checkpoint_selection_sha": checkpoint_selection_sha,
                    "model_count": len(model_shas),
                    "model_shas": model_shas,
                    "selected_mode": "same_training_run_two_selectors",
                    "selected_model": {
                        selector: {
                            f"fold_{fold}": selection_by_fold[fold][selector]["checkpoint_sha256"]
                            for fold in (0, 1)
                        }
                        for selector in selector_score_keys
                    },
                    "selector_prediction_shas": selector_prediction_shas,
                    "oof_prediction_sha": combined_oof_prediction_sha,
                    "test_prediction_content_sha": None,
                    "submission_sha": None,
                },
            },
            "notes": (
                "All six checkpoints were scored on the fixed internal videos. "
                "Each unique selected checkpoint was evaluated once on its outer embryo, "
                "and both selector summaries were recomputed from saved graphs."
            ),
        },
    )
    atomic_json(METRICS_PATH, final_metrics)
    print(json.dumps(json_safe(official_summary), indent=2))
    print("No Kaggle competition submission was created or made.")


# %% [markdown]
# ## 8. Metrics, manifests, and artifact contract

# %%
base_outputs = [
    ARTIFACTS_ROOT / "evaluation_gate.json",
    CHECKPOINT_SELECTION_PATH,
    PREDICTION_MANIFEST_PATH,
    METRICS_PATH,
]
if run_full_evaluation:
    base_outputs.extend(
        [
            PER_SAMPLE_METRICS_PATH,
            OFFICIAL_SUMMARY_PATH,
            EVALUATION_SUMMARY_PATH,
        ]
    )
missing_outputs = [str(path) for path in base_outputs if not path.exists()]
if missing_outputs:
    raise FileNotFoundError(f"missing evaluation outputs: {missing_outputs}")
if run_full_evaluation:
    expected_internal_prediction_count = sum(
        len(split_by_fold[fold]["test"]) * len(expected_epoch_indices) for fold in (0, 1)
    )
    if len(list(INTERNAL_SELECTION_ROOT.rglob("*.geff"))) != expected_internal_prediction_count:
        raise RuntimeError("internal selection GEFF count differs from the contract")
    if len(list(INTERNAL_SELECTION_ROOT.rglob("*.npz"))) != expected_internal_prediction_count:
        raise RuntimeError("internal candidate cache count differs from the contract")
    expected_outer_prediction_count = sum(
        len(split_by_fold[fold]["outer_evaluation"]) * len(unique_epochs_by_fold[fold])
        for fold in (0, 1)
    )
    if len(list(OUTER_PREDICTIONS_ROOT.rglob("*.geff"))) != expected_outer_prediction_count:
        raise RuntimeError("outer prediction GEFF count differs from the manifest contract")
    if len(list(OUTER_PREDICTIONS_ROOT.rglob("*.npz"))) != expected_outer_prediction_count:
        raise RuntimeError("outer candidate cache count differs from the manifest contract")
    print("Selector-specific outer metrics and all reusable prediction artifacts are ready.")
else:
    print(
        "The runtime gate stopped full outer evaluation; internal selector evidence "
        "and unique-checkpoint smoke outputs remain diagnostic only."
    )
