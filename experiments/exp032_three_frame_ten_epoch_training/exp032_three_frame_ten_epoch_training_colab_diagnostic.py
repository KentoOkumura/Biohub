# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp032: ten-epoch Colab comparison on fixed outer windows
#
# 1. Imports and offline runtime
# 2. Exact cache, teacher, and model helpers
# 3. Paired diagnostic definitions
# 4. Verify saved inputs and fixed evaluation windows
# 5. Run three inference conditions on each embryo
# 6. Reproduce pilot metrics and save paired results
#
# Use the same three-frame weights with previous points included or masked.
# The separately trained two-frame model is a reference. No weights are trained.
# Threshold curves describe sparse-label predictions; they are not official scores.

# %%
from __future__ import annotations

# ruff: noqa: E402,I001
import hashlib
import os
import zipfile
import importlib
import json
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

EXPERIMENT = "exp032_three_frame_ten_epoch_training"
COMPETITION = "biohub-cell-tracking-during-development"
NOTEBOOK_STARTED = time.perf_counter()

# CLI mode uses the same model and metric code with local inputs and outputs.
CLI_MODE = os.environ.get("EXP032_CLI_MODE") == "1"
DRIVE_PROJECT_ROOT = (
    Path("/content/exp032_cli_bundle") if CLI_MODE else Path("/content/drive/MyDrive/Kaggle/Biohub")
)
if not CLI_MODE:
    from google.colab import drive

    drive.mount("/content/drive")
    BUNDLE_ON_DRIVE = DRIVE_PROJECT_ROOT / "exp032_three_frame_ten_epoch_training_colab_bundle.zip"
    if not BUNDLE_ON_DRIVE.is_file():
        raise FileNotFoundError(f"Upload the exp032 Colab ZIP to {BUNDLE_ON_DRIVE}")
    with zipfile.ZipFile(BUNDLE_ON_DRIVE) as bundle:
        manifest = json.loads(bundle.read("BUNDLE_MANIFEST.json"))
        if (
            manifest.get("experiment") != EXPERIMENT
            or manifest.get("contains_credentials") is not False
        ):
            raise RuntimeError("Unexpected Colab bundle manifest")
        listed = {item["path"] for item in manifest["files"]}
        if set(bundle.namelist()) != listed | {"BUNDLE_MANIFEST.json"}:
            raise RuntimeError("Colab bundle file list differs from its manifest")
        for item in manifest["files"]:
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise RuntimeError(f"Unsafe Colab bundle path: {relative}")
            target = DRIVE_PROJECT_ROOT / relative
            if target.is_file():
                if target.suffix == ".ipynb":
                    continue  # Colab may save outputs into an opened notebook.
                with target.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if digest != item["sha256"]:
                    raise RuntimeError(f"Drive file differs from Colab ZIP: {relative}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name("." + target.name + ".tmp")
            with bundle.open(item["path"]) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            with temporary.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != item["sha256"]:
                temporary.unlink()
                raise RuntimeError(f"Colab ZIP checksum failed: {relative}")
            temporary.replace(target)

if not (DRIVE_PROJECT_ROOT / "project.yml").is_file():
    raise FileNotFoundError(f"Missing project.yml under {DRIVE_PROJECT_ROOT}")
EXPERIMENT_ROOT = DRIVE_PROJECT_ROOT / "experiments" / EXPERIMENT
CONFIG_PATH = EXPERIMENT_ROOT / "config.yaml"
config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
diag_cfg = config["diagnostic"]
colab_cfg = config["runtime"]["colab"]
if CLI_MODE:
    if Path(colab_cfg["cli"]["bundle_root"]) != DRIVE_PROJECT_ROOT:
        raise RuntimeError("CLI bundle root differs from config")
    if colab_cfg["cli"]["cache_acquisition"] != "kaggle_signed_manifest":
        raise RuntimeError("CLI cache must be acquired through the Kaggle signed manifest")
    if colab_cfg["cli"]["geff_acquisition"] != "kaggle_export_signed_manifest":
        raise RuntimeError("CLI GEFF must be acquired from Kaggle directly")
    if colab_cfg["cli"]["saved_predictions_acquisition"] != "kaggle_signed_manifest":
        raise RuntimeError("CLI predictions must be acquired from Kaggle directly")
else:
    if Path(colab_cfg["drive_project_root"]) != DRIVE_PROJECT_ROOT:
        raise RuntimeError("Colab project root in config differs from the notebook")
    if colab_cfg["cache_acquisition"] != "kaggle_kernel_output":
        raise RuntimeError("This Colab notebook requires the Kaggle exp015 cache download")
WORKING_ROOT = (
    Path(colab_cfg["cli"]["run_root"])
    if CLI_MODE
    else DRIVE_PROJECT_ROOT / colab_cfg["persistent_runs_dir"] / colab_cfg["run_id"]
)
if not (WORKING_ROOT / "train_complete.json").is_file():
    raise FileNotFoundError("Complete the ten-epoch train notebook before this diagnostic")
if (WORKING_ROOT / "config.yaml").read_bytes() != CONFIG_PATH.read_bytes():
    raise RuntimeError("The training run used a different configuration")
cache_cfg = config["data"]["cache"]
public_cfg = config["model"]["public_source"]
params_cfg = config["model"]["params"]
teacher_cfg = config["model"]["teacher"]
loss_cfg = config["model"]["loss"]
METRICS_PATH = WORKING_ROOT / "metrics.json"
OUTPUT = WORKING_ROOT / "ten_epoch_diagnostic"
OUTPUT.mkdir(exist_ok=True)
EVENT_LOG = WORKING_ROOT / "run_events.jsonl"


def log_event(event: str, **fields: Any) -> None:
    row = {"time": datetime.now(UTC).isoformat(), "event": event, **fields}
    with EVENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    print(json.dumps(row, sort_keys=True, default=str), flush=True)


log_event("diagnostic_start", run_id=colab_cfg["run_id"])

# %% [markdown]
# ## 1. Colab GEFF runtime
# Install the same offline dependencies as train v4 before reading annotations.

# %%
DEPENDENCIES = (
    "kaggle==2.2.4",
    "blosc2>=3.7,<5",
    "geff>=1.2,<2",
    "geff-spec>=1.1,<2",
    "ilpy>=0.6,<1",
    "polars>=1.42,<2",
    "pyarrow>=20,<25",
    "pyscipopt>=6.0,<7",
    "rustworkx>=0.17,<1",
    "tracksdata>=0.1.0rc6,<0.2",
    "zarr>=3.0.10,<4",
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        *DEPENDENCIES,
    ],
    check=True,
)
importlib.invalidate_caches()
import torch

if not torch.cuda.is_available():
    raise RuntimeError("Select a GPU Colab runtime before staging large inputs")
log_event(
    "runtime_preflight",
    gpu=torch.cuda.get_device_name(0),
    python=sys.version.split()[0],
    torch_version=torch.__version__,
    free_disk_bytes=shutil.disk_usage("/content").free,
)
from torch import nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# %% [markdown]
# ## 2. Exact cache, teacher, and model helpers

# %%
CACHE_SCHEMA_VERSION = 1


CACHE_METADATA_KEY = "__metadata_json__"


REQUIRED_CACHE_ARRAYS = (
    "candidate_ids_src",
    "candidate_ids_tgt",
    "coords_src_grid",
    "coords_tgt_grid",
    "coords_src_physical",
    "coords_tgt_physical",
    "position_features_src",
    "position_features_tgt",
    "candidate_mask_src",
    "candidate_mask_tgt",
    "primary_features_src",
    "primary_features_tgt",
)


@dataclass(frozen=True)
class FrameAnnotation:
    node_ids: np.ndarray
    coords_physical: np.ndarray


@dataclass(frozen=True)
class AnnotationGraph:
    frames: dict[int, FrameAnnotation]
    edges: frozenset[tuple[int, int]]
    content_sha256: str
    outgoing_edges: dict[int, tuple[int, ...]] | None = None


def json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_schema(arrays: dict[str, np.ndarray]) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "dtype": np.asarray(arrays[name]).dtype.str,
            "shape": list(np.asarray(arrays[name]).shape),
        }
        for name in sorted(arrays)
    ]


def array_content_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for item in array_schema(arrays):
        name = str(item["name"])
        array = np.ascontiguousarray(arrays[name])
        digest.update(json.dumps(item, separators=(",", ":"), sort_keys=True).encode())
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_cache_summary(path: Path, cache_cfg: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"cache summary must be an object: {path}")
    recorded_sha = value.get("summary_sha256")
    unsigned = dict(value)
    unsigned.pop("summary_sha256", None)
    observed_sha = json_sha256(unsigned)
    if recorded_sha != observed_sha:
        raise ValueError(f"cache summary self-check failed: {path}")
    expected = {
        "schema_version": int(cache_cfg["schema_version"]),
        "dataset_count": int(cache_cfg["expected_dataset_count"]),
        "window_count": int(cache_cfg["expected_window_count"]),
        "cache_identity_sha256": str(cache_cfg["identity_sha256"]),
        "summary_sha256": str(cache_cfg["summary_sha256"]),
    }
    observed = {key: value.get(key) for key in expected}
    if observed != expected:
        raise ValueError({"cache_summary_mismatch": {"expected": expected, "actual": observed}})
    return value


def discover_cache_paths(cache_root: Path, cache_cfg: dict[str, Any]) -> list[Path]:
    paths = sorted(cache_root.glob("*/*.npz"))
    samples = {path.parent.name for path in paths}
    expected_windows = int(cache_cfg["expected_window_count"])
    expected_samples = int(cache_cfg["expected_dataset_count"])
    if len(paths) != expected_windows or len(samples) != expected_samples:
        raise ValueError(
            {
                "cache_coverage_mismatch": {
                    "expected_samples": expected_samples,
                    "actual_samples": len(samples),
                    "expected_windows": expected_windows,
                    "actual_windows": len(paths),
                }
            }
        )
    return paths


def cache_identity_record(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        if CACHE_METADATA_KEY not in saved.files:
            raise ValueError(f"cache metadata missing: {path}")
        metadata = json.loads(saved[CACHE_METADATA_KEY].tobytes().decode("utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError(f"cache metadata must be an object: {path}")
    schema = metadata.get("array_schema")
    if not isinstance(schema, list):
        raise ValueError(f"cache array schema missing: {path}")
    schema_by_name = {str(item.get("name")): item for item in schema}
    feature_names = (
        "primary_features_src",
        "primary_features_tgt",
        "secondary_features_src",
        "secondary_features_tgt",
    )
    if any(name not in schema_by_name for name in feature_names):
        raise ValueError(f"cache feature schema incomplete: {path}")
    frames = metadata.get("window_frames")
    if metadata.get("dataset") != path.parent.name or not isinstance(frames, list):
        raise ValueError(f"cache identity and path differ: {path}")
    return {
        "dataset": path.parent.name,
        "window_frames": [int(value) for value in frames],
        "candidate_count_src": int(schema_by_name["candidate_mask_src"]["shape"][0]),
        "candidate_count_tgt": int(schema_by_name["candidate_mask_tgt"]["shape"][0]),
        "feature_values": int(
            sum(np.prod(schema_by_name[name]["shape"], dtype=np.int64) for name in feature_names)
        ),
        "cache_bytes": int(path.stat().st_size),
        "cache_schema_sha256": json_sha256(schema),
        "cache_content_sha256": str(metadata.get("array_content_sha256")),
    }


def recompute_cache_identity_sha256(paths: list[Path]) -> str:
    return json_sha256([cache_identity_record(path) for path in sorted(paths)])


def graph_from_geff(path: Path) -> Any:
    import tracksdata as td

    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def load_annotation_graph(
    path: Path,
    voxel_scale_zyx_um: tuple[float, float, float],
) -> AnnotationGraph:
    graph = graph_from_geff(path)
    scale = np.asarray(voxel_scale_zyx_um, dtype=np.float64)
    rows_by_frame: dict[int, list[tuple[int, np.ndarray]]] = {}
    canonical_nodes: list[list[int | float]] = []
    seen: set[int] = set()
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        if node_id in seen:
            raise ValueError(f"duplicate node id in {path}: {node_id}")
        seen.add(node_id)
        frame = int(row["t"])
        raw = np.asarray([row["z"], row["y"], row["x"]], dtype=np.float64)
        physical = raw * scale
        rows_by_frame.setdefault(frame, []).append((node_id, physical))
        canonical_nodes.append([node_id, frame, *map(float, raw)])
    edges = frozenset(
        (int(row["source_id"]), int(row["target_id"]))
        for row in graph.edge_attrs().iter_rows(named=True)
    )
    if any(source not in seen or target not in seen for source, target in edges):
        raise ValueError(f"dangling edge in {path}")
    frames: dict[int, FrameAnnotation] = {}
    for frame, entries in rows_by_frame.items():
        entries.sort(key=lambda item: item[0])
        frames[frame] = FrameAnnotation(
            node_ids=np.asarray([item[0] for item in entries], dtype=np.int64),
            coords_physical=np.asarray([item[1] for item in entries], dtype=np.float32),
        )
    canonical = {
        "nodes": sorted(canonical_nodes, key=lambda row: int(row[0])),
        "edges": [list(edge) for edge in sorted(edges)],
    }
    outgoing: dict[int, list[int]] = {}
    for source, target in edges:
        outgoing.setdefault(source, []).append(target)
    return AnnotationGraph(
        frames=frames,
        edges=edges,
        content_sha256=json_sha256(canonical),
        outgoing_edges={source: tuple(sorted(targets)) for source, targets in outgoing.items()},
    )


def greedy_match_candidates(
    candidate_coords_physical: np.ndarray,
    gt_node_ids: np.ndarray,
    gt_coords_physical: np.ndarray,
    max_distance_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.asarray(candidate_coords_physical, dtype=np.float64)
    gt_ids = np.asarray(gt_node_ids, dtype=np.int64)
    gt_coords = np.asarray(gt_coords_physical, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1:] != (3,):
        raise ValueError("candidate physical coordinates must have shape (N, 3)")
    if gt_coords.ndim != 2 or gt_coords.shape[1:] != (3,) or len(gt_coords) != len(gt_ids):
        raise ValueError("GT ids and coordinates have incompatible shapes")
    matched_ids = np.full(len(candidates), -1, dtype=np.int64)
    matched_distances = np.full(len(candidates), np.nan, dtype=np.float32)
    if len(candidates) == 0 or len(gt_ids) == 0:
        return matched_ids, matched_distances
    distances = np.linalg.norm(candidates[:, None, :] - gt_coords[None, :, :], axis=2)
    nearest_index = distances.argmin(axis=1)
    nearest_distance = distances[np.arange(len(candidates)), nearest_index]
    order = np.argsort(nearest_distance, kind="stable")
    gt_taken = np.zeros(len(gt_ids), dtype=bool)
    for candidate_index in order:
        distance = float(nearest_distance[candidate_index])
        if distance > max_distance_um:
            break
        gt_index = int(nearest_index[candidate_index])
        if gt_taken[gt_index]:
            continue
        gt_taken[gt_index] = True
        matched_ids[candidate_index] = gt_ids[gt_index]
        matched_distances[candidate_index] = distance
    return matched_ids, matched_distances


def build_legacy_edge_target(
    source_matches: np.ndarray,
    target_matches: np.ndarray,
    annotated_edges: frozenset[tuple[int, int]] | set[tuple[int, int]],
    outgoing_edges: dict[int, tuple[int, ...]] | None = None,
) -> np.ndarray:
    source = np.asarray(source_matches, dtype=np.int64)
    target = np.asarray(target_matches, dtype=np.int64)
    matrix = np.zeros((len(source), len(target)), dtype=np.float32)
    target_columns: dict[int, list[int]] = {}
    for column, node_id in enumerate(target):
        if node_id >= 0:
            target_columns.setdefault(int(node_id), []).append(column)
    if outgoing_edges is None:
        temporary: dict[int, list[int]] = {}
        for edge_source, edge_target in annotated_edges:
            temporary.setdefault(int(edge_source), []).append(int(edge_target))
        outgoing_edges = {key: tuple(values) for key, values in temporary.items()}
    for row, source_id in enumerate(source):
        if source_id < 0:
            continue
        for edge_target in outgoing_edges.get(int(source_id), ()):
            for column in target_columns.get(edge_target, []):
                matrix[row, column] = 1.0
    return matrix


def legacy_active_pair_mask(target: np.ndarray) -> np.ndarray:
    matrix = np.asarray(target)
    if matrix.ndim != 2:
        raise ValueError("target must be a matrix")
    active_rows = matrix.sum(axis=1) > 0
    active_cols = matrix.sum(axis=0) > 0
    return active_rows[:, None] | active_cols[None, :]


def legacy_focal_bce(logits: Any, target: Any, gamma: float = 2.0) -> Any:
    import torch
    import torch.nn.functional as functional

    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
    if not mask.any():
        return torch.tensor(0.0, requires_grad=True, device=logits.device)
    probabilities = torch.softmax(logits, dim=0)
    bce = functional.binary_cross_entropy(probabilities, target, reduction="none")
    p_t = probabilities * target + (1 - probabilities) * (1 - target)
    loss = ((1 - p_t) ** gamma) * bce
    return loss[mask].mean()


def _read_cache_payload(
    path: Path,
    *,
    load_all_arrays: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as saved:
        if CACHE_METADATA_KEY not in saved.files:
            raise ValueError(f"cache metadata missing: {path}")
        metadata = json.loads(saved[CACHE_METADATA_KEY].tobytes().decode("utf-8"))
        selected_names = (
            [name for name in saved.files if name != CACHE_METADATA_KEY]
            if load_all_arrays
            else [name for name in REQUIRED_CACHE_ARRAYS if name in saved.files]
        )
        arrays = {name: np.ascontiguousarray(saved[name]) for name in selected_names}
    if not isinstance(metadata, dict):
        raise TypeError(f"cache metadata must be an object: {path}")
    return arrays, metadata


def validate_window_cache(
    path: Path,
    *,
    feature_channels: int,
    expected_primary_checkpoint_sha256: str,
    verify_content: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays, metadata = _read_cache_payload(path, load_all_arrays=verify_content)
    missing = sorted(set(REQUIRED_CACHE_ARRAYS) - set(arrays))
    if missing:
        raise ValueError({"cache_arrays_missing": missing, "path": str(path)})
    expected_metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "experiment": "exp015_oracle_stage_limits",
        "dataset": path.parent.name,
        "primary_checkpoint_sha256": expected_primary_checkpoint_sha256,
    }
    observed_metadata = {key: metadata.get(key) for key in expected_metadata}
    if observed_metadata != expected_metadata:
        raise ValueError(
            {
                "cache_metadata_mismatch": {
                    "expected": expected_metadata,
                    "actual": observed_metadata,
                    "path": str(path),
                }
            }
        )
    frames = metadata.get("window_frames")
    if not isinstance(frames, list) or len(frames) != 2 or int(frames[1]) != int(frames[0]) + 1:
        raise ValueError(f"cache window is not an adjacent frame pair: {path}")
    expected_name = f"{int(frames[0]):06d}_{int(frames[1]):06d}.npz"
    if path.name != expected_name:
        raise ValueError(f"cache filename and window metadata differ: {path}")
    for side in ("src", "tgt"):
        mask = np.asarray(arrays[f"candidate_mask_{side}"], dtype=bool)
        count = len(mask)
        shapes = {
            "grid": np.asarray(arrays[f"coords_{side}_grid"]).shape,
            "physical": np.asarray(arrays[f"coords_{side}_physical"]).shape,
            "position": np.asarray(arrays[f"position_features_{side}"]).shape,
            "primary": np.asarray(arrays[f"primary_features_{side}"]).shape,
        }
        expected_shapes = {
            "grid": (count, 3),
            "physical": (count, 3),
            "position": (count, 32),
            "primary": (count, feature_channels),
        }
        if shapes != expected_shapes:
            raise ValueError(
                {"cache_shape_mismatch": side, "expected": expected_shapes, "actual": shapes}
            )
        if count == 0 or not mask.all():
            raise ValueError(f"cache contains empty or padded candidate rows: {path} {side}")
        for key in (
            f"coords_{side}_grid",
            f"coords_{side}_physical",
            f"position_features_{side}",
            f"primary_features_{side}",
        ):
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"cache contains non-finite values: {path} {key}")
    recorded_schema = {str(item.get("name")): item for item in metadata.get("array_schema", [])}
    actual_schema = {str(item["name"]): item for item in array_schema(arrays)}
    for name in REQUIRED_CACHE_ARRAYS:
        if recorded_schema.get(name) != actual_schema.get(name):
            raise ValueError(f"cache array schema mismatch: {path} {name}")
    if verify_content and metadata.get("array_content_sha256") != array_content_sha256(arrays):
        raise ValueError(f"cache array content mismatch: {path}")
    return arrays, metadata


def build_window_example(
    path: Path,
    annotation: AnnotationGraph,
    *,
    feature_channels: int,
    expected_primary_checkpoint_sha256: str,
    max_matching_distance_um: float,
    downsample_zyx: tuple[float, float, float],
    verify_content: bool = False,
    return_cache_arrays: bool = False,
) -> dict[str, Any]:
    arrays, metadata = validate_window_cache(
        path,
        feature_channels=feature_channels,
        expected_primary_checkpoint_sha256=expected_primary_checkpoint_sha256,
        verify_content=verify_content,
    )
    source_frame, target_frame = map(int, metadata["window_frames"])
    if source_frame not in annotation.frames or target_frame not in annotation.frames:
        raise ValueError(f"annotation is missing a cache frame: {path}")
    source_gt = annotation.frames[source_frame]
    target_gt = annotation.frames[target_frame]
    source_matches, source_distances = greedy_match_candidates(
        arrays["coords_src_physical"],
        source_gt.node_ids,
        source_gt.coords_physical,
        max_matching_distance_um,
    )
    target_matches, target_distances = greedy_match_candidates(
        arrays["coords_tgt_physical"],
        target_gt.node_ids,
        target_gt.coords_physical,
        max_matching_distance_um,
    )
    target = build_legacy_edge_target(
        source_matches,
        target_matches,
        annotation.edges,
        outgoing_edges=annotation.outgoing_edges,
    )
    active_mask = legacy_active_pair_mask(target)
    unknown_pairs = (source_matches < 0)[:, None] | (target_matches < 0)[None, :]
    downsample = np.asarray(downsample_zyx, dtype=np.float32)
    example = {
        "sample": path.parent.name,
        "window_frames": [source_frame, target_frame],
        "features_src": np.concatenate(
            [arrays["primary_features_src"], arrays["position_features_src"]], axis=1
        ).astype(np.float32, copy=False),
        "features_tgt": np.concatenate(
            [arrays["primary_features_tgt"], arrays["position_features_tgt"]], axis=1
        ).astype(np.float32, copy=False),
        "coords_src": (np.asarray(arrays["coords_src_grid"], dtype=np.float32) * downsample),
        "coords_tgt": (np.asarray(arrays["coords_tgt_grid"], dtype=np.float32) * downsample),
        "target": target,
        "teacher_stats": {
            "candidate_nodes": int(len(source_matches) + len(target_matches)),
            "matched_candidate_nodes": int(
                np.count_nonzero(source_matches >= 0) + np.count_nonzero(target_matches >= 0)
            ),
            "gt_nodes": int(len(source_gt.node_ids) + len(target_gt.node_ids)),
            "positive_edges": int(target.sum()),
            "active_pairs": int(active_mask.sum()),
            "active_pairs_with_unknown_endpoint": int((active_mask & unknown_pairs).sum()),
            "division_parents": int(np.count_nonzero(target.sum(axis=1) > 1)),
            "source_match_distance_sum_um": float(np.nansum(source_distances)),
            "source_match_distance_count": int(np.count_nonzero(np.isfinite(source_distances))),
            "target_match_distance_sum_um": float(np.nansum(target_distances)),
            "target_match_distance_count": int(np.count_nonzero(np.isfinite(target_distances))),
        },
    }
    if return_cache_arrays:
        example["_cache_arrays"] = arrays
        example["_cache_metadata"] = metadata
    return example


def collate_window_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch_size = len(examples)
    max_source = max(len(example["features_src"]) for example in examples)
    max_target = max(len(example["features_tgt"]) for example in examples)
    feature_dim = int(examples[0]["features_src"].shape[1])
    features_src = torch.zeros(batch_size, max_source, feature_dim, dtype=torch.float32)
    features_tgt = torch.zeros(batch_size, max_target, feature_dim, dtype=torch.float32)
    coords_src = torch.zeros(batch_size, max_source, 3, dtype=torch.float32)
    coords_tgt = torch.zeros(batch_size, max_target, 3, dtype=torch.float32)
    source_mask = torch.zeros(batch_size, max_source, dtype=torch.bool)
    target_mask = torch.zeros(batch_size, max_target, dtype=torch.bool)
    targets = torch.zeros(batch_size, max_source, max_target, dtype=torch.float32)
    for batch_index, example in enumerate(examples):
        n_source = len(example["features_src"])
        n_target = len(example["features_tgt"])
        features_src[batch_index, :n_source] = torch.from_numpy(example["features_src"])
        features_tgt[batch_index, :n_target] = torch.from_numpy(example["features_tgt"])
        coords_src[batch_index, :n_source] = torch.from_numpy(example["coords_src"])
        coords_tgt[batch_index, :n_target] = torch.from_numpy(example["coords_tgt"])
        source_mask[batch_index, :n_source] = True
        target_mask[batch_index, :n_target] = True
        targets[batch_index, :n_source, :n_target] = torch.from_numpy(example["target"])
    return {
        "features_src": features_src,
        "features_tgt": features_tgt,
        "coords_src": coords_src,
        "coords_tgt": coords_tgt,
        "source_mask": source_mask,
        "target_mask": target_mask,
        "target": targets,
        "metadata": [
            {
                "sample": example["sample"],
                "window_frames": example["window_frames"],
                "teacher_stats": example["teacher_stats"],
            }
            for example in examples
        ],
    }


def move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def canonical_state_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        descriptor = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        digest.update(json.dumps(descriptor, separators=(",", ":"), sort_keys=True).encode())
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_three_frame_example(
    path: Path,
    annotation: AnnotationGraph,
    *,
    feature_channels: int,
    expected_primary_checkpoint_sha256: str,
    max_matching_distance_um: float,
    downsample_zyx: tuple[float, float, float],
) -> dict[str, Any]:
    """Keep the central pair and read earlier points from the preceding image window."""
    central = build_window_example(
        path,
        annotation,
        feature_channels=feature_channels,
        expected_primary_checkpoint_sha256=expected_primary_checkpoint_sha256,
        max_matching_distance_um=max_matching_distance_um,
        downsample_zyx=downsample_zyx,
        return_cache_arrays=True,
    )
    arrays = central.pop("_cache_arrays")
    metadata = central.pop("_cache_metadata")
    source_frame, _ = map(int, metadata["window_frames"])
    central["coords_src_physical"] = np.asarray(arrays["coords_src_physical"], dtype=np.float32)
    central["coords_tgt_physical"] = np.asarray(arrays["coords_tgt_physical"], dtype=np.float32)
    prior_path = path.with_name(f"{source_frame - 1:06d}_{source_frame:06d}.npz")
    if source_frame == 0:
        central["features_prev"] = np.zeros((0, central["features_src"].shape[1]), dtype=np.float32)
        central["coords_prev_physical"] = np.zeros((0, 3), dtype=np.float32)
        central["prior_cache_content_sha256"] = None
        return central
    if not prior_path.is_file():
        raise FileNotFoundError(f"previous window missing within video: {prior_path}")

    prior_arrays, prior_metadata = validate_window_cache(
        prior_path,
        feature_channels=feature_channels,
        expected_primary_checkpoint_sha256=expected_primary_checkpoint_sha256,
    )
    if prior_metadata["window_frames"] != [source_frame - 1, source_frame]:
        raise ValueError(f"incorrect previous window: {prior_path}")
    # Candidate identity and position must survive the overlap. Window-dependent
    # image features need not agree and are deliberately kept separate.
    for label, prior_key, central_key in (
        ("ids", "candidate_ids_tgt", "candidate_ids_src"),
        ("grid", "coords_tgt_grid", "coords_src_grid"),
        ("physical", "coords_tgt_physical", "coords_src_physical"),
    ):
        if not np.array_equal(prior_arrays[prior_key], arrays[central_key]):
            raise ValueError(f"overlapping window {label} mismatch: {prior_path} {path}")
    central["features_prev"] = np.concatenate(
        [prior_arrays["primary_features_src"], prior_arrays["position_features_src"]], axis=1
    ).astype(np.float32, copy=False)
    central["coords_prev_physical"] = np.asarray(
        prior_arrays["coords_src_physical"], dtype=np.float32
    )
    central["prior_cache_content_sha256"] = prior_metadata["array_content_sha256"]
    return central


def collate_three_frame_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch = collate_window_examples(examples)
    batch_size = len(examples)
    feature_dim = int(examples[0]["features_src"].shape[1])
    max_prev = max(len(example["features_prev"]) for example in examples)
    features_prev = torch.zeros(batch_size, max_prev, feature_dim)
    coords_prev_physical = torch.zeros(batch_size, max_prev, 3)
    coords_src_physical = torch.zeros(batch_size, batch["features_src"].shape[1], 3)
    coords_tgt_physical = torch.zeros(batch_size, batch["features_tgt"].shape[1], 3)
    prev_mask = torch.zeros(batch_size, max_prev, dtype=torch.bool)
    for index, example in enumerate(examples):
        n_prev = len(example["features_prev"])
        n_src = len(example["features_src"])
        n_tgt = len(example["features_tgt"])
        features_prev[index, :n_prev] = torch.from_numpy(example["features_prev"])
        coords_prev_physical[index, :n_prev] = torch.from_numpy(example["coords_prev_physical"])
        coords_src_physical[index, :n_src] = torch.from_numpy(example["coords_src_physical"])
        coords_tgt_physical[index, :n_tgt] = torch.from_numpy(example["coords_tgt_physical"])
        prev_mask[index, :n_prev] = True
        batch["metadata"][index]["previous_candidate_count"] = n_prev
        batch["metadata"][index]["prior_cache_content_sha256"] = example[
            "prior_cache_content_sha256"
        ]
    batch.update(
        features_prev=features_prev,
        coords_prev_physical=coords_prev_physical,
        coords_src_physical=coords_src_physical,
        coords_tgt_physical=coords_tgt_physical,
        prev_mask=prev_mask,
    )
    return batch


class LocalAttentionBlock(nn.Module):
    """Public block parameters applied to gathered spatial neighbors."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True, dropout=dropout
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, hidden: torch.Tensor, indices: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        batch_size, node_count, hidden_dim = hidden.shape
        batch_index = torch.arange(batch_size, device=hidden.device)[:, None, None]
        neighbors = hidden[batch_index, indices]
        queries = self.norm1(hidden).reshape(batch_size * node_count, 1, hidden_dim)
        keys = self.norm1(neighbors).reshape(batch_size * node_count, indices.shape[-1], hidden_dim)
        output, _ = self.cross_attn(
            queries,
            keys,
            keys,
            key_padding_mask=~valid.reshape(batch_size * node_count, -1),
            need_weights=False,
        )
        hidden = hidden + output.reshape(batch_size, node_count, hidden_dim)
        return hidden + self.mlp(self.norm2(hidden))


class LocalThreeFrameTracker(nn.Module):
    """Local attention across three point sets, with central-pair edge logits."""

    def __init__(
        self,
        *,
        feat_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_blocks: int,
        dropout: float,
        pair_chunk_size: int,
        attention_radius_um: float,
        max_neighbors: int,
    ) -> None:
        super().__init__()
        if attention_radius_um <= 0 or max_neighbors < 1:
            raise ValueError("local attention radius and neighbor count must be positive")
        self.pair_chunk_size = pair_chunk_size
        self.attention_radius_um = attention_radius_um
        self.max_neighbors = max_neighbors
        self.max_observed_neighbors = 0
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            LocalAttentionBlock(hidden_dim, n_heads, dropout) for _ in range(n_blocks)
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.time_embedding = nn.Embedding(3, hidden_dim)
        nn.init.zeros_(self.time_embedding.weight)

    def load_public_tracker_state(self, state: dict[str, Any]) -> None:
        result = self.load_state_dict(state, strict=False)
        if result.missing_keys != ["time_embedding.weight"] or result.unexpected_keys:
            raise ValueError(
                {
                    "missing_public_parameters": result.missing_keys,
                    "unexpected_public_parameters": result.unexpected_keys,
                }
            )

    def _neighbors(
        self, coords: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            distance = torch.cdist(coords.float(), coords.float())
            distance = distance.masked_fill(~mask[:, None, :], float("inf"))
            neighborhood_size = (distance <= self.attention_radius_um).sum(dim=-1)
            observed_neighbors = int(neighborhood_size[mask].max().item()) if mask.any() else 1
            self.max_observed_neighbors = max(self.max_observed_neighbors, observed_neighbors)
            if observed_neighbors > self.max_neighbors:
                raise RuntimeError(
                    f"neighborhood {observed_neighbors} exceeds limit {self.max_neighbors}"
                )
            k = min(observed_neighbors, coords.shape[1])
            values, indices = torch.topk(distance, k=k, dim=-1, largest=False)
            valid = values <= self.attention_radius_um
            invalid_queries = ~mask
            if invalid_queries.any():
                indices[:, :, 0] = torch.where(
                    invalid_queries, torch.zeros_like(indices[:, :, 0]), indices[:, :, 0]
                )
                valid[:, :, 0] |= invalid_queries
            if not valid.any(dim=-1).all():
                raise RuntimeError("a detection has no local attention neighbor")
            return indices, valid

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        include_previous: bool,
        reverse: bool = False,
    ) -> torch.Tensor:
        prev_mask = batch["prev_mask"] if include_previous else torch.zeros_like(batch["prev_mask"])
        n_prev = batch["features_prev"].shape[1]
        n_src = batch["features_src"].shape[1]
        features = torch.cat(
            [batch["features_prev"], batch["features_src"], batch["features_tgt"]], dim=1
        )
        physical = torch.cat(
            [
                batch["coords_prev_physical"],
                batch["coords_src_physical"],
                batch["coords_tgt_physical"],
            ],
            dim=1,
        )
        mask = torch.cat([prev_mask, batch["source_mask"], batch["target_mask"]], dim=1)
        time_ids = torch.cat(
            [
                torch.zeros(n_prev, device=features.device, dtype=torch.long),
                torch.ones(n_src, device=features.device, dtype=torch.long),
                torch.full(
                    (batch["features_tgt"].shape[1],), 2, device=features.device, dtype=torch.long
                ),
            ]
        )
        hidden = self.norm_in(self.proj(features) + self.time_embedding(time_ids))
        hidden = hidden * mask.unsqueeze(-1)
        indices, valid = self._neighbors(physical, mask)
        for block in self.blocks:
            if self.training and torch.is_grad_enabled():
                hidden = grad_checkpoint(block, hidden, indices, valid, use_reentrant=False)
            else:
                hidden = block(hidden, indices, valid)
            hidden = hidden * mask.unsqueeze(-1)
        hidden = self.norm_out(hidden)
        src = hidden[:, n_prev : n_prev + n_src]
        tgt = hidden[:, n_prev + n_src :]
        coords_src, coords_tgt = batch["coords_src"], batch["coords_tgt"]
        if reverse:
            src, tgt = tgt, src
            coords_src, coords_tgt = coords_tgt, coords_src
        chunks = []
        for start in range(0, src.shape[1], self.pair_chunk_size):
            src_part = src[:, start : start + self.pair_chunk_size]
            coords_part = coords_src[:, start : start + self.pair_chunk_size]

            def score_pairs(
                q: torch.Tensor, k: torch.Tensor, cq: torch.Tensor, ck: torch.Tensor
            ) -> torch.Tensor:
                query = q.unsqueeze(2).expand(-1, -1, k.shape[1], -1)
                key = k.unsqueeze(1).expand(-1, q.shape[1], -1, -1)
                relative = (cq.unsqueeze(2) - ck.unsqueeze(1)) / 100.0
                return self.pair_mlp(torch.cat([query, key, relative], dim=-1)).squeeze(-1)

            if self.training and torch.is_grad_enabled():
                chunk = grad_checkpoint(
                    score_pairs, src_part, tgt, coords_part, coords_tgt, use_reentrant=False
                )
            else:
                chunk = score_pairs(src_part, tgt, coords_part, coords_tgt)
            chunks.append(chunk)
        return torch.cat(chunks, dim=1)


COMPETITION = "biohub-cell-tracking-during-development"


# Shared definitions above match the parent notebook. The run configuration
# was loaded from Drive before installing GEFF dependencies.


def update_metrics(path: Path, updates: dict[str, Any]) -> None:
    current = json.loads(path.read_text(encoding="utf-8"))

    def merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value
        return base

    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(merge(current, updates), indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def unique_existing(paths: list[Path], label: str) -> Path:
    matches: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved.exists() and resolved not in seen:
            matches.append(resolved)
            seen.add(resolved)
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {matches}")
    return matches[0]


LOCAL_INPUT_ROOT = Path("/content") / EXPERIMENT / "inputs"
LOCAL_INPUT_ROOT.mkdir(parents=True, exist_ok=True)


def staged_input(key: str, destination: str, marker: str) -> Path:
    source = DRIVE_PROJECT_ROOT / str(colab_cfg[key])
    if not (source / marker).exists():
        raise FileNotFoundError(f"{key} is missing {marker}: {source}")
    if CLI_MODE:
        return source
    target = LOCAL_INPUT_ROOT / destination
    if not (target / marker).exists():
        source_bytes = sum(item.stat().st_size for item in source.rglob("*") if item.is_file())
        free_bytes = shutil.disk_usage("/content").free
        if free_bytes < source_bytes + 2_000_000_000:
            raise RuntimeError(
                f"Insufficient /content disk for {key}: need {source_bytes + 2_000_000_000}, "
                f"free {free_bytes}"
            )
        shutil.copytree(source, target, dirs_exist_ok=True)
    if not (target / marker).exists():
        raise FileNotFoundError(f"{key} copy is incomplete: {target}")
    return target


def resolve_cache_output_root() -> Path:
    if CLI_MODE:
        target = LOCAL_INPUT_ROOT / "exp015"
        marker = target / str(cache_cfg["summary_file"])
        if not marker.is_file():
            raise FileNotFoundError(f"CLI cache acquisition did not produce {marker}")
        return target
    target = LOCAL_INPUT_ROOT / "exp015"
    marker = target / str(cache_cfg["summary_file"])
    if not marker.is_file():
        from google.colab import userdata

        try:
            kaggle_token = userdata.get("KAGGLE_API_TOKEN")
        except Exception as exc:
            raise RuntimeError(
                "Set KAGGLE_API_TOKEN in Colab Secrets and grant notebook access"
            ) from exc
        if not kaggle_token:
            raise RuntimeError("KAGGLE_API_TOKEN Colab Secret is empty")
        if shutil.disk_usage("/content").free < 7_000_000_000:
            raise RuntimeError("At least 7 GB free under /content is needed for the cache")
        target.mkdir(parents=True, exist_ok=True)
        command = [
            "kaggle",
            "kernels",
            "output",
            str(cache_cfg["kernel_source"]),
            "--path",
            str(target),
            "--file-pattern",
            r"^(window_cache/.*|window_cache_summary\.json)$",
            "--page-size",
            "200",
            "--quiet",
        ]
        log_event("cache_download_start", kernel=cache_cfg["kernel_source"])
        subprocess.run(
            command,
            check=True,
            env={**os.environ, "KAGGLE_API_TOKEN": kaggle_token},
        )
        del kaggle_token
        log_event("cache_download_complete", kernel=cache_cfg["kernel_source"])
    if not marker.is_file():
        raise FileNotFoundError(f"Kaggle cache download did not produce {marker}")
    return target


def resolve_public_artifact_root() -> Path:
    source = DRIVE_PROJECT_ROOT / str(colab_cfg["public_source"])
    required = (
        str(public_cfg["checkpoint"]),
        str(public_cfg["train_script"]),
        str(public_cfg["model_source"]),
    )
    missing = [relative for relative in required if not (source / relative).is_file()]
    if missing:
        raise FileNotFoundError({"public_source": str(source), "missing": missing})
    if CLI_MODE:
        return source
    target = LOCAL_INPUT_ROOT / "public_support"
    package_source = source / "repo/src"
    package_target = target / "repo/src"
    shutil.copytree(package_source, package_target, dirs_exist_ok=True)
    for relative in required:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            shutil.copy2(source / relative, destination)
    log_event("input_staged", name="public_source", source=str(source), target=str(target))
    return target


def resolve_train_dir() -> Path:
    source = DRIVE_PROJECT_ROOT / str(colab_cfg["geff_source"])
    if not source.is_dir() or not any(source.glob("*.geff")):
        raise FileNotFoundError(f"GEFF train directory missing: {source}")
    if CLI_MODE:
        return source
    target = LOCAL_INPUT_ROOT / "train_geff"
    if not target.is_dir():
        shutil.copytree(source, target)
    return target


# %% [markdown]
# ## 3. Paired diagnostic definitions


# %%
def diagnose_pair(
    logits: np.ndarray,
    probabilities: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    threshold_grid: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Describe known edges separately from sparse-teacher negative predictions."""
    logits = np.asarray(logits)
    probabilities = np.asarray(probabilities)
    target = np.asarray(target)
    if logits.shape != target.shape or probabilities.shape != target.shape or target.ndim != 2:
        raise ValueError("logit/probability/target shapes differ")
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise ValueError("nonfinite predictions")
    if not np.allclose(probabilities.sum(axis=0), 1.0, atol=1e-5):
        raise ValueError("probabilities must normalize over source parents")
    positive = target > 0.5
    active = positive.any(axis=1)[:, None] | positive.any(axis=0)[None, :]
    predicted = probabilities > threshold
    best_parent = np.argmax(logits, axis=0)
    division_rows = positive.sum(axis=1) > 1
    edges = []
    for source, child in zip(*np.nonzero(positive), strict=True):
        score = logits[source, child]
        ties = logits[:, child] == score
        rank = 1 + np.count_nonzero(logits[:, child] > score)
        rank += np.count_nonzero(ties[:source])
        competitor = np.arange(logits.shape[0]) != source
        best_other_logit = float(logits[competitor, child].max()) if competitor.any() else None
        best_other_probability = (
            float(probabilities[competitor, child].max()) if competitor.any() else None
        )
        edges.append(
            {
                "source_index": int(source),
                "target_index": int(child),
                "true_parent_rank": int(rank),
                "top1_correct": bool(best_parent[child] == source),
                "tied_parent_count": int(ties.sum()),
                "predicted_parent_index": int(best_parent[child]),
                "true_parent_probability": float(probabilities[source, child]),
                "true_parent_logit": float(score),
                "best_other_probability": best_other_probability,
                "logit_margin": None
                if best_other_logit is None
                else float(score - best_other_logit),
                "probability_margin": None
                if best_other_probability is None
                else float(probabilities[source, child] - best_other_probability),
                "recovered": bool(predicted[source, child]),
                "division_parent": bool(division_rows[source]),
            }
        )
    summary = {
        "positive_edge_count": int(positive.sum()),
        "true_positive_count": int((predicted & positive).sum()),
        "top1_correct_count": sum(row["top1_correct"] for row in edges),
        "false_positive_active_pair_count": int((predicted & ~positive & active).sum()),
        "active_pair_count": int(active.sum()),
        "correct_active_pair_count": int(((predicted == positive) & active).sum()),
        "division_parent_count": int(division_rows.sum()),
        "recovered_division_parent_count": sum(
            bool(predicted[row, positive[row]].all()) for row in np.flatnonzero(division_rows)
        ),
        "threshold_counts": [
            {
                "threshold": float(value),
                "true_positive_count": int(((probabilities > value) & positive).sum()),
                "false_positive_active_pair_count": int(
                    ((probabilities > value) & ~positive & active).sum()
                ),
            }
            for value in threshold_grid
        ],
    }
    return summary, edges


def aggregate_diagnostics(
    windows: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, Any]:
    count_keys = (
        "positive_edge_count",
        "true_positive_count",
        "top1_correct_count",
        "false_positive_active_pair_count",
        "active_pair_count",
        "correct_active_pair_count",
        "division_parent_count",
        "recovered_division_parent_count",
    )
    totals = {key: sum(int(row[key]) for row in windows) for key in count_keys}
    for name, numerator, denominator in (
        ("positive_edge_recall", "true_positive_count", "positive_edge_count"),
        ("top1_accuracy", "top1_correct_count", "positive_edge_count"),
        ("edge_accuracy", "correct_active_pair_count", "active_pair_count"),
        ("division_parent_recall", "recovered_division_parent_count", "division_parent_count"),
    ):
        totals[name] = totals[numerator] / totals[denominator] if totals[denominator] else 0.0
    totals["window_count"] = len(windows)
    totals["legacy_mask_loss"] = float(np.mean([row["legacy_mask_loss"] for row in windows]))
    totals["mean_true_parent_probability"] = (
        float(np.mean([row["true_parent_probability"] for row in edges])) if edges else None
    )
    totals["mean_reciprocal_rank"] = (
        float(np.mean([1.0 / row["true_parent_rank"] for row in edges])) if edges else None
    )
    totals["threshold_curve"] = []
    for index, point in enumerate(windows[0]["threshold_counts"]):
        tp = sum(row["threshold_counts"][index]["true_positive_count"] for row in windows)
        fp = sum(
            row["threshold_counts"][index]["false_positive_active_pair_count"] for row in windows
        )
        totals["threshold_curve"].append(
            {
                "threshold": point["threshold"],
                "true_positive_count": tp,
                "positive_edge_recall": tp / totals["positive_edge_count"]
                if totals["positive_edge_count"]
                else 0.0,
                "false_positive_active_pair_count": fp,
            }
        )
    return totals


def edge_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["sample"],
        row["source_frame"],
        row["target_frame"],
        row["source_id"],
        row["target_id"],
    )


def compare_known_edges(
    reference: list[dict[str, Any]],
    changed: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    before = {edge_identity(row): row for row in reference}
    after = {edge_identity(row): row for row in changed}
    if len(before) != len(reference) or len(after) != len(changed) or before.keys() != after.keys():
        raise ValueError("paired edge identities are duplicated or differ")
    rows = []
    for key in sorted(before):
        a, b = before[key], after[key]
        rows.append(
            {
                "sample": key[0],
                "source_frame": key[1],
                "target_frame": key[2],
                "source_id": key[3],
                "target_id": key[4],
                "reference_rank": a["true_parent_rank"],
                "changed_rank": b["true_parent_rank"],
                "reference_probability": a["true_parent_probability"],
                "changed_probability": b["true_parent_probability"],
                "probability_delta": b["true_parent_probability"] - a["true_parent_probability"],
                "rescued_threshold": not a["recovered"] and b["recovered"],
                "harmed_threshold": a["recovered"] and not b["recovered"],
                "rescued_top1": not a["top1_correct"] and b["top1_correct"],
                "harmed_top1": a["top1_correct"] and not b["top1_correct"],
                "both_top1": a["top1_correct"] and b["top1_correct"],
                "reference_top1": a["top1_correct"],
                "changed_top1": b["top1_correct"],
                "reference_recovered": a["recovered"],
                "changed_recovered": b["recovered"],
                "reference_predicted_parent_id": a["predicted_parent_id"],
                "changed_predicted_parent_id": b["predicted_parent_id"],
                "division_parent": a["division_parent"],
                "displacement_um": a.get("displacement_um"),
                "previous_neighbor_count": a.get("previous_neighbor_count"),
            }
        )
    summary = {
        "positive_edge_count": len(rows),
        **{
            key: sum(row[key] for row in rows)
            for key in (
                "rescued_threshold",
                "harmed_threshold",
                "rescued_top1",
                "harmed_top1",
                "both_top1",
            )
        },
        "threshold_rescues_with_both_top1": sum(
            row["rescued_threshold"] and row["both_top1"] for row in rows
        ),
        "threshold_harms_with_both_top1": sum(
            row["harmed_threshold"] and row["both_top1"] for row in rows
        ),
        "mean_probability_delta": float(np.mean([row["probability_delta"] for row in rows]))
        if rows
        else None,
    }
    summary["recall_delta"] = (
        (summary["rescued_threshold"] - summary["harmed_threshold"]) / len(rows) if rows else None
    )
    summary["top1_delta"] = (
        (summary["rescued_top1"] - summary["harmed_top1"]) / len(rows) if rows else None
    )
    return summary, rows


def paired_video_bootstrap(
    paired_edges: list[dict[str, Any]],
    samples: list[str],
    *,
    repeats: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    if len(set(samples)) != len(samples) or not samples:
        raise ValueError("sample list must be nonempty and unique")
    index = {sample: i for i, sample in enumerate(samples)}
    counts = np.zeros((len(samples), 3), dtype=np.float64)
    for row in paired_edges:
        i = index[row["sample"]]
        counts[i] += [
            1,
            int(row["changed_recovered"]) - int(row["reference_recovered"]),
            int(row["changed_top1"]) - int(row["reference_top1"]),
        ]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(samples), size=(repeats, len(samples)))
    totals = counts[draws].sum(axis=1)
    totals = totals[totals[:, 0] > 0]
    bounds = [(1 - confidence) / 2, 1 - (1 - confidence) / 2]
    return {
        "unit": "video",
        "video_count": len(samples),
        "repeats": repeats,
        "confidence": confidence,
        "nonempty_repeats": len(totals),
        "recall_delta_interval": np.quantile(totals[:, 1] / totals[:, 0], bounds).tolist()
        if len(totals)
        else None,
        "top1_delta_interval": np.quantile(totals[:, 2] / totals[:, 0], bounds).tolist()
        if len(totals)
        else None,
    }


# %% [markdown]
# ## 4. Verify Drive inputs and fixed evaluation windows
# The original exp027 selection, split, model, and prediction hashes are fixed
# before any model is evaluated. The saved exp016 model is never retrained.


# %%
def require_sha(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = file_sha256(path)
    if observed != expected:
        raise RuntimeError({"sha_mismatch": str(path), "expected": expected, "actual": observed})
    return observed


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def active_negative_scores(probabilities: np.ndarray, target: np.ndarray) -> np.ndarray:
    positive = target > 0.5
    active = positive.any(axis=1)[:, None] | positive.any(axis=0)[None, :]
    return probabilities[active & ~positive]


def select_negative_budget_threshold(negative: np.ndarray, budget: int) -> float:
    negative = np.asarray(negative).reshape(-1)
    if budget < 0 or not np.isfinite(negative).all() or np.any((negative < 0) | (negative > 1)):
        raise ValueError("invalid internal teacher-negative scores or budget")
    if len(negative) <= budget:
        return 0.0
    return float(np.partition(negative, len(negative) - budget - 1)[len(negative) - budget - 1])


cache_output_root = resolve_cache_output_root()
cache_root = cache_output_root / str(cache_cfg["directory"])
cache_summary = validate_cache_summary(
    cache_output_root / str(cache_cfg["summary_file"]), cache_cfg
)
cache_paths = discover_cache_paths(cache_root, cache_cfg)
observed_cache_identity = recompute_cache_identity_sha256(cache_paths)
if observed_cache_identity != str(cache_cfg["identity_sha256"]):
    raise RuntimeError("exp015 cache identity differs from the fixed experiment")
train_dir = resolve_train_dir()
public_root = resolve_public_artifact_root()
if require_sha(public_root / str(public_cfg["checkpoint"]), str(public_cfg["checkpoint_sha256"])):
    pass
require_sha(public_root / str(public_cfg["model_source"]), str(public_cfg["model_source_sha256"]))
sys.path.insert(0, str(public_root / "repo/src"))
SimpleNodeTransformer = importlib.import_module("biohub_tracking.models").SimpleNodeTransformer

exp016_root = staged_input("exp016_source", "exp016_train_v3", "model_manifest.json")
exp027_root = staged_input("exp027_source", "exp027_train_v4", "model_manifest.json")
exp027_diagnostic_root = staged_input(
    "exp027_diagnostic_source",
    "exp027_diagnostic_v1",
    "context_diagnostic/prediction_manifest.json",
)
exp027_validation_root = staged_input(
    "exp027_validation_source",
    "exp027_validation_v1",
    "validation_diagnostic/selected_thresholds.json",
)
require_sha(exp016_root / "model_manifest.json", diag_cfg["exp016_model_manifest_sha256"])
require_sha(exp016_root / "split_manifest.json", diag_cfg["parent_split_manifest_sha256"])
require_sha(exp027_root / "model_manifest.json", diag_cfg["parent_model_manifest_sha256"])
require_sha(exp027_root / "split_manifest.json", diag_cfg["parent_split_manifest_sha256"])
require_sha(exp027_root / "pilot_outer_window_selection.json", diag_cfg["parent_selection_sha256"])
require_sha(
    exp027_diagnostic_root / "context_diagnostic/prediction_manifest.json",
    diag_cfg["parent_prediction_manifest_sha256"],
)
require_sha(
    exp027_validation_root / "validation_diagnostic/selected_thresholds.json",
    diag_cfg["parent_validation_thresholds_sha256"],
)
require_sha(WORKING_ROOT / "pilot_outer_window_selection.json", diag_cfg["parent_selection_sha256"])
require_sha(WORKING_ROOT / "split_manifest.json", diag_cfg["parent_split_manifest_sha256"])
train_complete = json.loads((WORKING_ROOT / "train_complete.json").read_text())
new_manifest_path = WORKING_ROOT / "model_manifest.json"
require_sha(new_manifest_path, train_complete["model_manifest_sha256"])
new_manifest = json.loads(new_manifest_path.read_text())
if new_manifest["epochs"] != 10 or len(new_manifest["models"]) != 2:
    raise RuntimeError("ten-epoch three-frame model manifest is incomplete")
if new_manifest["cache_identity_sha256"] != observed_cache_identity:
    raise RuntimeError("training and diagnostic caches differ")
audit_path = WORKING_ROOT / "gt_window_filter_audit.json"
require_sha(audit_path, new_manifest["gt_window_filter_audit_sha256"])
audit = json.loads(audit_path.read_text())
if (
    audit["policy"] != "skip_window_if_any_frame_has_zero_gt_nodes"
    or audit["input_window_count"] != len(cache_paths)
    or audit["skipped_window_count"] != len(audit["skipped_windows"])
):
    raise RuntimeError("training GT window filter audit differs from diagnostic cache")
skipped_paths = {
    "{}/{:06d}_{:06d}.npz".format(row["sample"], *row["window_frames"])
    for row in audit["skipped_windows"]
}
cache_relative_paths = {path.relative_to(cache_root).as_posix() for path in cache_paths}
if len(skipped_paths) != audit["skipped_window_count"] or not skipped_paths <= cache_relative_paths:
    raise RuntimeError("training GT window filter does not match diagnostic cache paths")
eligible_cache_paths = [
    path for path in cache_paths if path.relative_to(cache_root).as_posix() not in skipped_paths
]
if len(eligible_cache_paths) != audit["eligible_window_count"]:
    raise RuntimeError("training and diagnostic eligible window counts differ")
log_event(
    "gt_window_filter_reused",
    eligible=len(eligible_cache_paths),
    skipped=len(skipped_paths),
)
exp016_manifest = json.loads((exp016_root / "model_manifest.json").read_text())
exp027_manifest = json.loads((exp027_root / "model_manifest.json").read_text())
for name, manifest in (("exp016", exp016_manifest), ("exp027", exp027_manifest)):
    if manifest["cache_identity_sha256"] != observed_cache_identity:
        raise RuntimeError(f"{name} used a different feature cache")
    if manifest["public_checkpoint_file_sha256"] != public_cfg["checkpoint_sha256"]:
        raise RuntimeError(f"{name} used a different public checkpoint")
selection = json.loads((WORKING_ROOT / "pilot_outer_window_selection.json").read_text())
splits = json.loads((WORKING_ROOT / "split_manifest.json").read_text())
saved_thresholds = json.loads(
    (exp027_validation_root / "validation_diagnostic/selected_thresholds.json").read_text()
)
prediction_manifest = json.loads(
    (exp027_diagnostic_root / "context_diagnostic/prediction_manifest.json").read_text()
)
prediction_lookup = {row["path"]: row for row in prediction_manifest}
if len(prediction_lookup) != len(prediction_manifest):
    raise RuntimeError("duplicate saved prediction paths")
scale = tuple(float(value) for value in config["data"]["annotation"]["voxel_scale_zyx_um"])
annotations = {
    sample: load_annotation_graph(train_dir / f"{sample}.geff", scale)
    for sample in sorted({path.parent.name for path in cache_paths})
}
annotation_manifest = {
    sample: graph.content_sha256 for sample, graph in sorted(annotations.items())
}
if json_sha256(annotation_manifest) != new_manifest["annotation_content_sha256"]:
    raise RuntimeError("annotation content differs from training")
if diag_cfg["official_graph_evaluation"] or diag_cfg["retrain_model_count"]:
    raise RuntimeError("this stage must only score saved trackers")
seed_everything(int(diag_cfg["bootstrap_seed"]))
if not torch.cuda.is_available():
    raise RuntimeError("Select a GPU Colab runtime for the common-window diagnostic")
device = torch.device("cuda")
log_event(
    "inputs_verified",
    cache_identity_sha256=observed_cache_identity,
    parent_selection_sha256=diag_cfg["parent_selection_sha256"],
)


# %% [markdown]
# ## 5. Model loading and internal-only threshold selection


# %%
def model_row(manifest: dict[str, Any], fold: int, variant: str | None = None) -> dict[str, Any]:
    rows = [
        row
        for row in manifest["models"]
        if int(row["fold"]) == fold and (variant is None or row.get("variant") == variant)
    ]
    if len(rows) != 1:
        raise RuntimeError({"fold": fold, "variant": variant, "model_records": rows})
    return rows[0]


def load_three_frame(fold: int, epoch: int) -> LocalThreeFrameTracker:
    summary_path = WORKING_ROOT / f"three_frame_local_fold_{fold}_summary.json"
    summary = json.loads(summary_path.read_text())
    selected = epoch == int(summary["best_epoch"]) + 1
    if selected:
        row = model_row(new_manifest, fold, "three_frame_local")
        if int(row["best_epoch"]) != epoch - 1:
            raise RuntimeError("selected checkpoint epoch differs from model manifest")
        path = WORKING_ROOT / row["path"]
        require_sha(path, row["file_sha256"])
    else:
        path = (
            WORKING_ROOT
            / "models/three_frame_local"
            / f"fold_{fold}"
            / f"primary_tracker_epoch_{epoch:02d}.pth"
        )
        require_sha(path, summary["epochs"][epoch - 1]["model_file_sha256"])
    payload = torch.load(path, map_location="cpu", weights_only=True)
    recorded_epoch = payload["best_epoch" if selected else "epoch"]
    if (payload["experiment"], payload["fold"], payload["variant"], recorded_epoch) != (
        EXPERIMENT,
        fold,
        "three_frame_local",
        epoch - 1,
    ):
        raise RuntimeError("checkpoint identity mismatch")
    state_sha = canonical_state_sha256(payload["state_dict"])
    if selected and state_sha != row["canonical_state_sha256"]:
        raise RuntimeError("selected model state SHA differs from model manifest")
    if state_sha != summary["epochs"][epoch - 1]["model_state_sha256"]:
        raise RuntimeError("epoch model state SHA mismatch")
    model = LocalThreeFrameTracker(
        feat_dim=int(params_cfg["feature_dim"]),
        hidden_dim=int(params_cfg["hidden_dim"]),
        n_heads=int(params_cfg["n_heads"]),
        n_blocks=int(params_cfg["n_blocks"]),
        dropout=float(params_cfg["dropout"]),
        pair_chunk_size=int(params_cfg["pair_chunk_size"]),
        attention_radius_um=float(params_cfg["attention_radius_um"]),
        max_neighbors=int(params_cfg["max_neighbors"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_exp016(fold: int) -> Any:
    row = model_row(exp016_manifest, fold)
    path = exp016_root / row["path"]
    require_sha(path, row["file_sha256"])
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["fold"] != fold or payload["evaluation_embryo"] != row["evaluation_embryo"]:
        raise RuntimeError("exp016 checkpoint identity mismatch")
    model = SimpleNodeTransformer(
        feat_dim=int(params_cfg["feature_dim"]),
        hidden_dim=int(params_cfg["hidden_dim"]),
        n_heads=int(params_cfg["n_heads"]),
        n_blocks=int(params_cfg["n_blocks"]),
        dropout=float(params_cfg["dropout"]),
        pair_chunk_size=int(params_cfg["pair_chunk_size"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    if canonical_state_sha256(model.state_dict()) != row["canonical_state_sha256"]:
        raise RuntimeError("exp016 checkpoint state SHA mismatch")
    return model.to(device).eval().requires_grad_(False)


def logits_for_model(model: Any, batch: dict[str, Any], kind: str) -> Any:
    with torch.no_grad():
        if kind == "exp016":
            return model(
                batch["features_src"],
                batch["features_tgt"],
                batch["coords_src"],
                batch["coords_tgt"],
                batch["source_mask"],
                batch["target_mask"],
            )
        return model(batch, include_previous=True)


def make_examples(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        build_three_frame_example(
            path,
            annotations[path.parent.name],
            feature_channels=int(cache_cfg["feature_channels"]),
            expected_primary_checkpoint_sha256=str(public_cfg["checkpoint_sha256"]),
            max_matching_distance_um=float(teacher_cfg["max_matching_distance_um"]),
            downsample_zyx=tuple(float(x) for x in params_cfg["downsample_zyx"]),
        )
        for path in paths
    ]


def infer_model(model: Any, paths: list[Path], kind: str):
    batch_size = int(diag_cfg["batch_size"])
    for offset in range(0, len(paths), batch_size):
        chosen_paths = paths[offset : offset + batch_size]
        examples = make_examples(chosen_paths)
        batch = move_batch_to_device(collate_three_frame_examples(examples), device)
        raw = logits_for_model(model, batch, kind)
        for index, (path, example) in enumerate(zip(chosen_paths, examples, strict=True)):
            nsrc, ntgt = example["target"].shape
            logits = raw[index, :nsrc, :ntgt].float().cpu().numpy()
            if not np.isfinite(logits).all():
                raise RuntimeError(f"nonfinite logits: {path}")
            yield path, example, logits
        del raw, batch


def internal_negative_scores(model: Any, paths: list[Path], kind: str) -> np.ndarray:
    pieces = []
    for _, example, logits in infer_model(model, paths, kind):
        probabilities = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
        pieces.append(active_negative_scores(probabilities, example["target"]))
    if not pieces:
        raise RuntimeError("internal validation has no windows")
    return np.concatenate(pieces)


selected_epoch_by_fold = {
    int(row["fold"]): int(row["best_epoch"]) + 1 for row in new_manifest["models"]
}
thresholds: dict[str, dict[str, float]] = {}
for fold in selection["folds"]:
    fold_id = int(fold["fold"])
    split = next(row for row in splits if int(row["fold"]) == fold_id)
    internal_samples = set(split["internal_validation"])
    if internal_samples & set(split["gradient_update"]) or internal_samples & set(
        split["outer_evaluation"]
    ):
        raise RuntimeError("internal validation overlaps gradient or outer samples")
    internal_paths = [path for path in eligible_cache_paths if path.parent.name in internal_samples]
    budget = int(diag_cfg["negative_budget_by_fold"][fold_id])
    if int(saved_thresholds[str(fold_id)]["negative_budget"]) != budget:
        raise RuntimeError("saved two-frame internal negative budget differs")
    thresholds[str(fold_id)] = {
        "exp027_two_frame": float(
            saved_thresholds[str(fold_id)]["modes"]["two_frame_saved"]["selected_threshold"]
        )
    }
    for mode, model in (
        ("selected", load_three_frame(fold_id, selected_epoch_by_fold[fold_id])),
        ("exp016", load_exp016(fold_id)),
    ):
        negative = internal_negative_scores(
            model, internal_paths, "exp016" if mode == "exp016" else "three"
        )
        threshold = select_negative_budget_threshold(negative, budget)
        if int(np.count_nonzero(negative > threshold)) > budget:
            raise RuntimeError("internal threshold violates the teacher-negative budget")
        thresholds[str(fold_id)][mode] = threshold
        log_event(
            "threshold_selected",
            fold=fold_id,
            mode=mode,
            threshold=threshold,
            negative_budget=budget,
            negative_count=len(negative),
        )
        del model
        torch.cuda.empty_cache()
write_json(OUTPUT / "selected_thresholds.json", thresholds)


# %% [markdown]
# ## 6. Score ten epochs and saved controls on identical outer windows

# %%
THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.45, 0.48, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
all_summaries: dict[str, Any] = {}
all_edges: dict[str, Any] = {}
prediction_records: list[dict[str, Any]] = []
epoch_one_sha_comparison: dict[str, Any] = {}
for fold in selection["folds"]:
    fold_id = int(fold["fold"])
    embryo = str(fold["evaluation_embryo"])
    relative_paths = list(fold["selected_windows"])
    paths = [cache_root / name for name in relative_paths]
    if len(paths) != int(diag_cfg["windows_per_embryo"]) or len(
        {p.parent.name for p in paths}
    ) != len(paths):
        raise RuntimeError("expected one predetermined window per video")
    if any(not path.parent.name.startswith(embryo + "_") for path in paths):
        raise RuntimeError("outer embryo differs from fixed selection")
    all_summaries[embryo] = {}
    all_edges[embryo] = {}
    new_epoch_one = json.loads(
        (WORKING_ROOT / f"three_frame_local_fold_{fold_id}_summary.json").read_text()
    )["epochs"][0]["model_state_sha256"]
    parent_epoch_one = model_row(exp027_manifest, fold_id, "three_frame_local")[
        "canonical_state_sha256"
    ]
    epoch_one_sha_comparison[embryo] = {
        "new_epoch_1_state_sha256": new_epoch_one,
        "exp027_epoch_1_state_sha256": parent_epoch_one,
        "exact_match": new_epoch_one == parent_epoch_one,
    }
    evaluated_epochs = sorted({1, selected_epoch_by_fold[fold_id]})
    modes = [(f"epoch_{epoch:02d}", "three", epoch) for epoch in evaluated_epochs]
    modes += [
        ("exp016", "exp016", None),
        ("exp027_two_frame", "saved", None),
        ("exp027_three_frame_1_epoch", "saved", None),
    ]
    for mode, kind, epoch in modes:
        model = (
            load_three_frame(fold_id, epoch)
            if kind == "three"
            else (load_exp016(fold_id) if kind == "exp016" else None)
        )
        rows_fixed, edges_fixed = [], []
        rows_selected, edges_selected = [], []
        selected_threshold = (
            thresholds[str(fold_id)].get("selected")
            if mode == f"epoch_{selected_epoch_by_fold[fold_id]:02d}"
            else thresholds[str(fold_id)].get(mode)
        )
        if kind == "saved":
            stream = []
            for path in paths:
                relative = "pair_logits/" + path.relative_to(cache_root).as_posix()
                manifest_row = prediction_lookup.get(relative)
                if manifest_row is None:
                    raise RuntimeError(f"exp027 prediction missing: {relative}")
                saved_path = exp027_diagnostic_root / "context_diagnostic" / relative
                require_sha(saved_path, manifest_row["file_sha256"])
                with np.load(saved_path, allow_pickle=False) as payload:
                    source_key = (
                        "logits_two_frame_saved"
                        if mode == "exp027_two_frame"
                        else "logits_three_frame_full"
                    )
                    arrays, _ = validate_window_cache(
                        path,
                        feature_channels=int(cache_cfg["feature_channels"]),
                        expected_primary_checkpoint_sha256=str(public_cfg["checkpoint_sha256"]),
                        verify_content=True,
                    )
                    for key in ("candidate_ids_src", "candidate_ids_tgt"):
                        if not np.array_equal(payload[key], arrays[key]):
                            raise RuntimeError(f"saved candidate identity changed: {path}")
                    example = make_examples([path])[0]
                    if not np.array_equal(payload["target"], example["target"]):
                        raise RuntimeError(f"saved GEFF teacher changed: {path}")
                    stream.append((path, example, np.asarray(payload[source_key])))
        else:
            stream = infer_model(model, paths, kind)
        for path, example, logits in stream:
            probabilities = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
            arrays, _ = validate_window_cache(
                path,
                feature_channels=int(cache_cfg["feature_channels"]),
                expected_primary_checkpoint_sha256=str(public_cfg["checkpoint_sha256"]),
                verify_content=True,
            )
            if kind != "saved":
                prediction_path = OUTPUT / "pair_logits" / mode / path.relative_to(cache_root)
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "logits": np.asarray(logits, dtype=np.float32),
                    "target": np.asarray(example["target"]),
                    "candidate_ids_src": np.asarray(arrays["candidate_ids_src"]),
                    "candidate_ids_tgt": np.asarray(arrays["candidate_ids_tgt"]),
                }
                np.savez_compressed(prediction_path, **payload)
                prediction_records.append(
                    {
                        "mode": mode,
                        "path": prediction_path.relative_to(OUTPUT).as_posix(),
                        "file_sha256": file_sha256(prediction_path),
                        "array_content_sha256": array_content_sha256(payload),
                    }
                )
            for threshold_name, threshold, window_rows, edge_rows in (
                ("fixed", float(diag_cfg["probability_threshold"]), rows_fixed, edges_fixed),
                ("internal", selected_threshold, rows_selected, edges_selected),
            ):
                if threshold is None:
                    continue
                diagnostic, edges = diagnose_pair(
                    logits,
                    probabilities,
                    example["target"],
                    threshold=threshold,
                    threshold_grid=THRESHOLD_GRID,
                )
                diagnostic["legacy_mask_loss"] = float(
                    legacy_focal_bce(
                        torch.from_numpy(logits),
                        torch.from_numpy(example["target"]),
                        gamma=float(loss_cfg["focal_gamma"]),
                    ).item()
                )
                diagnostic.update(
                    mode=mode,
                    threshold_source=threshold_name,
                    threshold=threshold,
                    fold=fold_id,
                    embryo=embryo,
                    sample=example["sample"],
                )
                window_rows.append(diagnostic)
                for edge in edges:
                    src, tgt = edge["source_index"], edge["target_index"]
                    edge.update(
                        mode=mode,
                        threshold_source=threshold_name,
                        fold=fold_id,
                        embryo=embryo,
                        sample=example["sample"],
                        source_frame=example["window_frames"][0],
                        target_frame=example["window_frames"][1],
                        source_id=int(arrays["candidate_ids_src"][src]),
                        target_id=int(arrays["candidate_ids_tgt"][tgt]),
                        predicted_parent_id=int(
                            arrays["candidate_ids_src"][edge["predicted_parent_index"]]
                        ),
                    )
                    edge_rows.append(edge)
        fixed_summary = aggregate_diagnostics(rows_fixed, edges_fixed)
        for field in ("logit_margin", "probability_margin"):
            values = [float(row[field]) for row in edges_fixed if row[field] is not None]
            fixed_summary[f"mean_{field}"] = float(np.mean(values)) if values else None
        all_summaries[embryo][mode] = {"fixed": fixed_summary}
        all_edges[embryo][mode] = {"fixed": edges_fixed}
        if rows_selected:
            internal_summary = aggregate_diagnostics(rows_selected, edges_selected)
            for field in ("logit_margin", "probability_margin"):
                values = [float(row[field]) for row in edges_selected if row[field] is not None]
                internal_summary[f"mean_{field}"] = float(np.mean(values)) if values else None
            all_summaries[embryo][mode]["internal"] = internal_summary
            all_edges[embryo][mode]["internal"] = edges_selected
        log_event(
            "outer_mode_complete",
            embryo=embryo,
            mode=mode,
            fixed_recall=all_summaries[embryo][mode]["fixed"]["positive_edge_recall"],
        )
        del model
        torch.cuda.empty_cache()

# The compact train archive retains epoch 1 and the selected weight. Reuse
# its fixed and internal-threshold rows without evaluating it again.
for fold in selection["folds"]:
    embryo = str(fold["evaluation_embryo"])
    selected_mode = f"epoch_{selected_epoch_by_fold[int(fold['fold'])]:02d}"
    all_summaries[embryo]["selected"] = all_summaries[embryo][selected_mode]
    all_edges[embryo]["selected"] = all_edges[embryo][selected_mode]
write_json(OUTPUT / "outer_summaries.json", all_summaries)
for embryo in all_summaries:
    new_first = all_summaries[embryo]["epoch_01"]["fixed"]
    parent_first = all_summaries[embryo]["exp027_three_frame_1_epoch"]["fixed"]
    if (
        new_first["positive_edge_count"] != parent_first["positive_edge_count"]
        or new_first["active_pair_count"] != parent_first["active_pair_count"]
    ):
        raise RuntimeError("epoch-one teacher denominator differs from exp027")
    epoch_one_sha_comparison[embryo]["metric_deltas"] = {
        key: float(new_first[key]) - float(parent_first[key])
        for key in (
            "positive_edge_recall",
            "false_positive_active_pair_count",
            "top1_accuracy",
            "mean_true_parent_probability",
        )
    }
write_json(OUTPUT / "epoch_one_sha_comparison.json", epoch_one_sha_comparison)
write_json(OUTPUT / "prediction_manifest.json", prediction_records)


# %% [markdown]
# ## 7. Paired comparisons, evidence, and completion marker
# Parent ranking and fixed/selected thresholds are reported separately.
# Sparse teacher-negative predictions are not confirmed false links.

# %%
comparisons: dict[str, Any] = {}
for fold in selection["folds"]:
    embryo = str(fold["evaluation_embryo"])
    samples = [Path(name).parent.name for name in fold["selected_windows"]]
    comparisons[embryo] = {}
    for label, before_mode, after_mode, threshold_source in (
        ("selected_vs_epoch_01_fixed", "epoch_01", "selected", "fixed"),
        ("selected_vs_exp016_fixed", "exp016", "selected", "fixed"),
        ("selected_vs_exp027_two_frame_fixed", "exp027_two_frame", "selected", "fixed"),
        ("selected_vs_exp016_internal", "exp016", "selected", "internal"),
        ("selected_vs_exp027_two_frame_internal", "exp027_two_frame", "selected", "internal"),
    ):
        if (
            threshold_source not in all_edges[embryo][before_mode]
            or threshold_source not in all_edges[embryo][after_mode]
        ):
            continue
        before = all_edges[embryo][before_mode][threshold_source]
        after = all_edges[embryo][after_mode][threshold_source]
        paired, paired_rows = compare_known_edges(before, after)
        paired["paired_video_bootstrap"] = paired_video_bootstrap(
            paired_rows,
            samples,
            repeats=int(diag_cfg["bootstrap_samples"]),
            seed=int(diag_cfg["bootstrap_seed"]),
            confidence=float(diag_cfg["bootstrap_confidence"]),
        )
        b = all_summaries[embryo][before_mode][threshold_source]
        a = all_summaries[embryo][after_mode][threshold_source]
        paired["positive_edge_recall_delta"] = a["positive_edge_recall"] - b["positive_edge_recall"]
        paired["teacher_negative_prediction_delta"] = (
            a["false_positive_active_pair_count"] - b["false_positive_active_pair_count"]
        )
        paired["top1_accuracy_delta"] = a["top1_accuracy"] - b["top1_accuracy"]
        paired["preliminary_condition_met"] = (
            paired["positive_edge_recall_delta"] > 0
            and paired["teacher_negative_prediction_delta"] <= 0
            and paired["top1_accuracy_delta"] >= 0
        )
        comparisons[embryo][label] = paired
write_json(OUTPUT / "paired_comparisons.json", comparisons)
summary = {
    "experiment": EXPERIMENT,
    "run_id": colab_cfg["run_id"],
    "created_at": datetime.now(UTC).isoformat(),
    "selected_epoch_by_fold": selected_epoch_by_fold,
    "fixed_window_epochs_by_fold": {
        str(fold): sorted({1, epoch}) for fold, epoch in selected_epoch_by_fold.items()
    },
    "thresholds": thresholds,
    "summaries": all_summaries,
    "comparisons": comparisons,
    "epoch_one_sha_comparison": epoch_one_sha_comparison,
    "cache_identity_sha256": observed_cache_identity,
    "gt_window_filter_audit_sha256": new_manifest["gt_window_filter_audit_sha256"],
    "parent_selection_sha256": diag_cfg["parent_selection_sha256"],
    "exp016_model_manifest_sha256": diag_cfg["exp016_model_manifest_sha256"],
    "parent_model_manifest_sha256": diag_cfg["parent_model_manifest_sha256"],
    "new_model_manifest_sha256": train_complete["model_manifest_sha256"],
    "prediction_manifest_sha256": file_sha256(OUTPUT / "prediction_manifest.json"),
    "prediction_content_sha256": json_sha256(
        [
            {
                "mode": row["mode"],
                "path": row["path"],
                "array_content_sha256": row["array_content_sha256"],
            }
            for row in prediction_records
        ]
    ),
    "official_graph_evaluation": False,
    "submission_created": False,
}
write_json(OUTPUT / "diagnostic_summary.json", summary)
summary_sha = file_sha256(OUTPUT / "diagnostic_summary.json")
update_metrics(
    METRICS_PATH,
    {
        "status": "debug_completed",
        "diagnostic": summary,
        "evidence": {
            "colab": {
                "run_id": colab_cfg["run_id"],
                "resource": torch.cuda.get_device_name(device),
                "diagnostic_summary_sha256": summary_sha,
            }
        },
        "notes": (
            "Ten-epoch Colab training and fixed-window diagnostic complete. "
            "Official graph score not measured; user decision pending."
        ),
    },
)
write_json(
    WORKING_ROOT / "diagnostic_complete.json",
    {
        "experiment": EXPERIMENT,
        "run_id": colab_cfg["run_id"],
        "summary_sha256": summary_sha,
        "completed_at": datetime.now(UTC).isoformat(),
    },
)
log_event("diagnostic_complete", summary_sha256=summary_sha)
print(
    json.dumps(
        {"selected_epoch_by_fold": selected_epoch_by_fold, "comparisons": comparisons},
        indent=2,
        default=float,
    )
)
