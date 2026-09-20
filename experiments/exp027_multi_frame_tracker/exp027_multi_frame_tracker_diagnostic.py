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
# # exp027: saved-model previous-context diagnostic
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
import csv
import hashlib
import importlib
import json
import os
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

EXPERIMENT = "exp027_multi_frame_tracker"
COMPETITION = "biohub-cell-tracking-during-development"
WORKING_ROOT = Path.cwd()
NOTEBOOK_STARTED = time.perf_counter()
if not Path("/kaggle/input").is_dir():
    raise RuntimeError("The authoritative diagnostic must run on Kaggle")
config = yaml.safe_load((WORKING_ROOT / "config.yaml").read_text())
diag_cfg = config["diagnostic"]
cache_cfg = config["data"]["cache"]
public_cfg = config["model"]["public_source"]
params_cfg = config["model"]["params"]
teacher_cfg = config["model"]["teacher"]
loss_cfg = config["model"]["loss"]
METRICS_PATH = WORKING_ROOT / "metrics.json"

# %% [markdown]
# ## 1. Offline GEFF runtime
# Install the same offline dependencies as train v4 before reading annotations.

# %%
OFFLINE_GRAPH_MODULES = {
    "tracksdata": "tracksdata",
    "zarr": "zarr",
    "geff": "geff",
    "geff_spec": "geff_spec",
    "ilpy": "ilpy",
    "polars": "polars",
    "pyscipopt": "pyscipopt",
    "imagecodecs": "imagecodecs",
    "rustworkx": "rustworkx",
    "numcodecs": "numcodecs",
    "donfig": "donfig",
    "bidict": "bidict",
}


OFFLINE_GRAPH_PACKAGE_SPECS = (
    "tracksdata",
    "bidict>=0.23.1",
    "psygnal>=0.14",
    "rich",
    "markdown-it-py",
    "pygments",
    "zarr>=3.0.10,<4",
    "donfig>=0.8",
    "google-crc32c>=1.5",
    "numcodecs>=0.13,<0.16",
    "deprecated",
    "msgpack",
    "wrapt",
    "geff>=1.1.3.1.1",
    "geff-spec<1.2",
    "networkx>=3.2.1",
    "pydantic>=2.11",
    "annotated-types",
    "pydantic-core",
    "typing-extensions>=4.13",
    "typing-inspection",
    "pyscipopt",
    "ilpy>=0.5.1",
    "imagecodecs",
    "rustworkx>=0.17.1",
)


def offline_wheel_dirs() -> list[Path]:
    candidates = [
        Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/wheels"),
        Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1/wheels"),
    ]
    return [path for path in candidates if path.is_dir() and any(path.glob("*.whl"))]


def polars_runtime_ready() -> bool:
    try:
        import polars as pl
        from polars._plr import PySeries

        _ = PySeries
        return hasattr(pl, "Float16")
    except Exception:
        return False


def run_offline_install(
    wheel_dirs: list[Path],
    specs: tuple[str, ...],
    *,
    force_reinstall: bool,
) -> None:
    command = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
    if force_reinstall:
        command.append("--force-reinstall")
    for wheel_dir in wheel_dirs:
        command.extend(["--find-links", str(wheel_dir)])
    command.extend(specs)
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        print((result.stdout or "")[-2000:])
        print((result.stderr or "")[-2000:])
        raise RuntimeError(f"offline graph package install failed: {result.returncode}")


def purge_graph_modules(*, include_polars: bool) -> None:
    roots = set(OFFLINE_GRAPH_MODULES.values())
    if not include_polars:
        roots.discard("polars")
    for module_name in list(sys.modules):
        if any(module_name == root or module_name.startswith(root + ".") for root in roots):
            sys.modules.pop(module_name, None)


def ensure_geff_runtime_dependencies() -> None:
    os.environ.setdefault("POLARS_PREFER_PKG", "32")
    failures: dict[str, str] = {}
    for package_name, module_name in OFFLINE_GRAPH_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as error:
            failures[package_name] = f"{type(error).__name__}: {error}"
    refresh_polars = not polars_runtime_ready()
    if not failures and not refresh_polars:
        return
    wheel_dirs = offline_wheel_dirs()
    if not wheel_dirs:
        raise FileNotFoundError("the public support dataset has no offline wheel directory")
    if refresh_polars:
        run_offline_install(
            wheel_dirs,
            ("polars>=1.36", "polars-runtime-32"),
            force_reinstall=True,
        )
        purge_graph_modules(include_polars=True)
        importlib.invalidate_caches()
        if not polars_runtime_ready():
            raise ImportError("offline Polars refresh did not provide Float16 support")
    run_offline_install(
        wheel_dirs,
        OFFLINE_GRAPH_PACKAGE_SPECS,
        force_reinstall=False,
    )
    purge_graph_modules(include_polars=False)
    importlib.invalidate_caches()
    remaining = {}
    for package_name, module_name in OFFLINE_GRAPH_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as error:
            remaining[package_name] = f"{type(error).__name__}: {error}"
    if remaining or not polars_runtime_ready():
        raise ImportError({"remaining_graph_import_failures": remaining})


ensure_geff_runtime_dependencies()
import torch
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


WORKING_ROOT = Path.cwd()


CONFIG_PATH = WORKING_ROOT / "config.yaml"


config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


cache_cfg = config["data"]["cache"]


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


def resolve_cache_output_root() -> Path:
    direct = Path("/kaggle/input") / str(cache_cfg["kernel_source"]).split("/", 1)[-1]
    if (direct / str(cache_cfg["summary_file"])).is_file():
        return direct
    print("Kaggle input roots:", sorted(str(path) for path in Path("/kaggle/input").iterdir()))
    candidates = [
        path.parent for path in Path("/kaggle/input").rglob(str(cache_cfg["summary_file"]))
    ]
    print("Cache summary candidates:", [str(path) for path in candidates])
    return unique_existing(candidates, "exp015 cache output root")


def resolve_train_dir() -> Path:
    return unique_existing(
        [
            Path(f"/kaggle/input/competitions/{COMPETITION}/train"),
            Path(f"/kaggle/input/{COMPETITION}/train"),
        ],
        "competition train directory",
    )


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
# ## 4. Verify saved inputs and fixed evaluation windows
# A matching manifest binds the diagnostic to train version 4 even if the input
# Notebook is later updated. Validate raw cache content for every used window.


# %%
def require_sha(path: Path, expected: str) -> str:
    observed = file_sha256(path)
    if observed != expected:
        raise RuntimeError({"sha_mismatch": str(path), "expected": expected, "observed": observed})
    return observed


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


train_candidates = [
    path.parent
    for path in Path("/kaggle/input").rglob("model_manifest.json")
    if file_sha256(path) == diag_cfg["model_manifest_sha256"]
]
trained_root = unique_existing(train_candidates, "train v4 model manifest")
model_manifest = json.loads((trained_root / "model_manifest.json").read_text())
require_sha(trained_root / "pilot_outer_window_selection.json", diag_cfg["selection_sha256"])
require_sha(trained_root / "training_summary.json", diag_cfg["training_summary_sha256"])
require_sha(trained_root / "local_tracker_model.py", diag_cfg["model_source_sha256"])
selection = json.loads((trained_root / "pilot_outer_window_selection.json").read_text())
reference_training = json.loads((trained_root / "training_summary.json").read_text())
source_config = yaml.safe_load((trained_root / "config.yaml").read_text())
for key in ("params", "teacher", "loss"):
    if config["model"][key] != source_config["model"][key]:
        raise RuntimeError(f"diagnostic changed the trained model {key}")
if diag_cfg["batch_size"] != source_config["model"]["training"]["batch_size"]:
    raise RuntimeError("diagnostic must reproduce the original batch size")
if model_manifest["stage"] != "pilot" or model_manifest["epochs"] != 1:
    raise RuntimeError("expected the one-epoch pilot")
if diag_cfg["retrain_model_count"] != 0 or diag_cfg["official_graph_evaluation"]:
    raise RuntimeError("this notebook is only a saved-weight pair diagnostic")
cache_output_root = resolve_cache_output_root()
cache_root = cache_output_root / cache_cfg["directory"]
cache_summary = validate_cache_summary(cache_output_root / cache_cfg["summary_file"], cache_cfg)
cache_paths = discover_cache_paths(cache_root, cache_cfg)
identity_sha = recompute_cache_identity_sha256(cache_paths)
if (
    identity_sha != cache_cfg["identity_sha256"]
    or identity_sha != model_manifest["cache_identity_sha256"]
):
    raise RuntimeError("cache identity mismatch")
train_dir = resolve_train_dir()
scale = tuple(config["data"]["annotation"]["voxel_scale_zyx_um"])
annotations = {
    sample: load_annotation_graph(train_dir / f"{sample}.geff", scale)
    for sample in sorted({path.parent.name for path in cache_paths})
}
annotation_manifest = {
    sample: graph.content_sha256 for sample, graph in sorted(annotations.items())
}
if json_sha256(annotation_manifest) != model_manifest["annotation_content_sha256"]:
    raise RuntimeError("annotation content differs from training")
OUTPUT = WORKING_ROOT / "context_diagnostic"
OUTPUT.mkdir(exist_ok=True)
shutil.copy2(
    trained_root / "pilot_outer_window_selection.json", OUTPUT / "pilot_outer_window_selection.json"
)
seed_everything(int(diag_cfg["seed"]))
device = torch.device("cuda")
if not torch.cuda.is_available():
    raise RuntimeError("T4 GPU is required for this runtime contract")
print("Saved training:", trained_root, flush=True)
print("GPU:", torch.cuda.get_device_name(device), flush=True)
print("Same pilot windows; no training; conditions:", diag_cfg["modes"], flush=True)

# %% [markdown]
# ## 5. Run three inference conditions on each embryo
# Evaluate the identical points, target matrix, and batch order for all modes.
# The full and masked three-frame conditions share one loaded model object.

# %%
all_window_rows = []
all_edge_rows = []
input_records = []
prediction_records = []
weight_evidence = []
mode_names = [mode["name"] for mode in diag_cfg["modes"]]
all_summaries = {}
reference_checks = []
processing_started = time.perf_counter()
processed_windows = 0
total_windows = sum(len(fold["selected_windows"]) for fold in selection["folds"])

for fold in selection["folds"]:
    fold_id, embryo = fold["fold"], fold["evaluation_embryo"]
    paths = [cache_root / relative for relative in fold["selected_windows"]]
    if len(paths) != diag_cfg["windows_per_embryo"] or len({p.parent.name for p in paths}) != len(
        paths
    ):
        raise RuntimeError("pilot windows must have one unique video per window")
    if any(not path.parent.name.startswith(embryo + "_") for path in paths):
        raise RuntimeError("evaluation embryo mismatch")
    examples, arrays_by_window = [], []
    for path in paths:
        arrays, metadata = validate_window_cache(
            path,
            feature_channels=cache_cfg["feature_channels"],
            expected_primary_checkpoint_sha256=public_cfg["checkpoint_sha256"],
            verify_content=True,
        )
        current_record = {
            "window": path.relative_to(cache_root).as_posix(),
            "content_sha256": metadata["array_content_sha256"],
        }
        source_frame = metadata["window_frames"][0]
        if source_frame:
            prior_path = path.with_name(f"{source_frame - 1:06d}_{source_frame:06d}.npz")
            _, prior_meta = validate_window_cache(
                prior_path,
                feature_channels=cache_cfg["feature_channels"],
                expected_primary_checkpoint_sha256=public_cfg["checkpoint_sha256"],
                verify_content=True,
            )
            current_record["previous_content_sha256"] = prior_meta["array_content_sha256"]
        input_records.append(current_record)
        example = build_three_frame_example(
            path,
            annotations[path.parent.name],
            feature_channels=cache_cfg["feature_channels"],
            expected_primary_checkpoint_sha256=public_cfg["checkpoint_sha256"],
            max_matching_distance_um=teacher_cfg["max_matching_distance_um"],
            downsample_zyx=tuple(params_cfg["downsample_zyx"]),
        )
        examples.append(example)
        arrays_by_window.append(arrays)
    models = {}
    for variant in {mode["variant"] for mode in diag_cfg["modes"]}:
        manifest_row = next(
            row
            for row in model_manifest["models"]
            if row["fold"] == fold_id and row["variant"] == variant
        )
        checkpoint_path = trained_root / manifest_row["path"]
        require_sha(checkpoint_path, manifest_row["file_sha256"])
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            payload["fold"] != fold_id
            or payload["evaluation_embryo"] != embryo
            or payload["variant"] != variant
        ):
            raise RuntimeError("model identity mismatch")
        model = LocalThreeFrameTracker(
            feat_dim=params_cfg["feature_dim"],
            hidden_dim=params_cfg["hidden_dim"],
            n_heads=params_cfg["n_heads"],
            n_blocks=params_cfg["n_blocks"],
            dropout=params_cfg["dropout"],
            pair_chunk_size=params_cfg["pair_chunk_size"],
            attention_radius_um=params_cfg["attention_radius_um"],
            max_neighbors=params_cfg["max_neighbors"],
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        state_sha = canonical_state_sha256(model.state_dict())
        if state_sha != manifest_row["canonical_state_sha256"]:
            raise RuntimeError("model state SHA mismatch")
        model = model.to(device).eval().requires_grad_(False)
        models[variant] = model
        weight_evidence.append({**manifest_row, "before_state_sha256": state_sha})
    for offset in range(0, len(examples), int(diag_cfg["batch_size"])):
        chosen = examples[offset : offset + int(diag_cfg["batch_size"])]
        batch = move_batch_to_device(collate_three_frame_examples(chosen), device)
        mode_predictions = {}
        for mode in diag_cfg["modes"]:
            mode_name = mode["name"]
            with torch.no_grad():
                raw_logits = models[mode["variant"]](
                    batch, include_previous=mode["include_previous"]
                )
            for local_index, example in enumerate(chosen):
                absolute_index = offset + local_index
                arrays = arrays_by_window[absolute_index]
                source_count, target_count = example["target"].shape
                logits = raw_logits[local_index, :source_count, :target_count].float()
                probabilities = torch.softmax(logits, dim=0)
                diagnostic, edges = diagnose_pair(
                    logits.cpu().numpy(),
                    probabilities.cpu().numpy(),
                    example["target"],
                    threshold=float(diag_cfg["probability_threshold"]),
                    threshold_grid=diag_cfg["threshold_grid"],
                )
                diagnostic["legacy_mask_loss"] = float(
                    legacy_focal_bce(
                        logits,
                        batch["target"][local_index, :source_count, :target_count],
                        gamma=loss_cfg["focal_gamma"],
                    ).item()
                )
                diagnostic.update(
                    mode=mode_name,
                    fold=fold_id,
                    embryo=embryo,
                    sample=example["sample"],
                    source_frame=example["window_frames"][0],
                    target_frame=example["window_frames"][1],
                )
                all_window_rows.append(diagnostic)
                for edge in edges:
                    source, target = edge["source_index"], edge["target_index"]
                    distance = np.linalg.norm(
                        example["coords_prev_physical"] - arrays["coords_src_physical"][source],
                        axis=1,
                    )
                    edge.update(
                        mode=mode_name,
                        fold=fold_id,
                        embryo=embryo,
                        sample=example["sample"],
                        source_frame=example["window_frames"][0],
                        target_frame=example["window_frames"][1],
                        source_id=int(arrays["candidate_ids_src"][source]),
                        target_id=int(arrays["candidate_ids_tgt"][target]),
                        predicted_parent_id=int(
                            arrays["candidate_ids_src"][edge["predicted_parent_index"]]
                        ),
                        displacement_um=float(
                            np.linalg.norm(
                                arrays["coords_src_physical"][source]
                                - arrays["coords_tgt_physical"][target]
                            )
                        ),
                        previous_neighbor_count=int(
                            np.count_nonzero(distance <= params_cfg["attention_radius_um"])
                        ),
                        source_candidate_count=source_count,
                        target_candidate_count=target_count,
                        previous_candidate_count=len(example["features_prev"]),
                    )
                    all_edge_rows.append(edge)
                mode_predictions.setdefault(absolute_index, {})[mode_name] = logits.cpu().numpy()
            del raw_logits
        for absolute_index, predictions in mode_predictions.items():
            path = paths[absolute_index]
            example = examples[absolute_index]
            arrays = arrays_by_window[absolute_index]
            prediction_path = OUTPUT / "pair_logits" / path.parent.name / path.name
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {f"logits_{name}": value for name, value in predictions.items()}
            payload.update(
                target=example["target"],
                candidate_ids_src=arrays["candidate_ids_src"],
                candidate_ids_tgt=arrays["candidate_ids_tgt"],
                coords_src_physical=arrays["coords_src_physical"],
                coords_tgt_physical=arrays["coords_tgt_physical"],
            )
            np.savez_compressed(prediction_path, **payload)
            prediction_records.append(
                {
                    "path": prediction_path.relative_to(OUTPUT).as_posix(),
                    "array_content_sha256": array_content_sha256(payload),
                    "file_sha256": file_sha256(prediction_path),
                }
            )
        processed_windows += len(chosen)
        if processed_windows >= diag_cfg["benchmark_min_windows"]:
            elapsed = time.perf_counter() - processing_started
            setup = processing_started - NOTEBOOK_STARTED
            projected = (
                setup
                + elapsed
                + elapsed
                / processed_windows
                * (total_windows - processed_windows)
                * diag_cfg["runtime_projection_multiplier"]
            )
            if projected > diag_cfg["runtime_gate_hours"] * 3600:
                raise RuntimeError(f"diagnostic runtime projection exceeds gate: {projected}")
        if processed_windows % diag_cfg["log_every_windows"] == 0:
            print(f"Diagnostic windows: {processed_windows}/{total_windows}", flush=True)
        if time.perf_counter() - NOTEBOOK_STARTED > diag_cfg["runtime_gate_hours"] * 3600:
            raise RuntimeError("diagnostic elapsed runtime exceeds gate")
    for variant, model in models.items():
        entry = next(
            row for row in weight_evidence if row["fold"] == fold_id and row["variant"] == variant
        )
        entry["after_state_sha256"] = canonical_state_sha256(model.state_dict())
        if entry["after_state_sha256"] != entry["before_state_sha256"]:
            raise RuntimeError("saved model changed during diagnostic")
    all_summaries[embryo] = {}
    for mode_name in mode_names:
        windows = [
            row for row in all_window_rows if row["embryo"] == embryo and row["mode"] == mode_name
        ]
        edges = [
            row for row in all_edge_rows if row["embryo"] == embryo and row["mode"] == mode_name
        ]
        all_summaries[embryo][mode_name] = aggregate_diagnostics(windows, edges)
    for mode_name, variant in [
        ("two_frame_saved", "two_frame_local"),
        ("three_frame_full", "three_frame_local"),
    ]:
        expected = next(
            row["trained_outer_evaluation"]
            for row in reference_training["folds"]
            if row["fold"] == fold_id and row["variant"] == variant
        )
        actual = all_summaries[embryo][mode_name]
        checks = {
            key: actual[key] == expected[key]
            for key in (
                "positive_edge_count",
                "false_positive_active_pair_count",
                "active_pair_count",
                "division_parent_count",
                "window_count",
            )
        }
        checks["true_positive_count"] = actual["true_positive_count"] == round(
            expected["positive_edge_recall"] * expected["positive_edge_count"]
        )
        checks["division_recovered_count"] = actual["recovered_division_parent_count"] == round(
            expected["division_parent_recall"] * expected["division_parent_count"]
        )
        checks["legacy_mask_loss"] = bool(
            np.isclose(
                actual["legacy_mask_loss"],
                expected["legacy_mask_loss"],
                rtol=0,
                atol=diag_cfg["reference_loss_atol"],
            )
        )
        reference_checks.append({"embryo": embryo, "mode": mode_name, "checks": checks})
    del models, model, payload, batch
    torch.cuda.empty_cache()

# %% [markdown]
# ## 6. Reproduce pilot metrics and save paired results
# Compare full-minus-masked within the same checkpoint separately from the
# full-minus-two-frame difference. Bootstrap resamples videos, not individual edges.

# %%
comparisons = {}
for fold in selection["folds"]:
    embryo = fold["evaluation_embryo"]
    comparisons[embryo] = {}
    samples = [Path(name).parent.name for name in fold["selected_windows"]]
    for label, reference_mode, changed_mode in (
        ("full_minus_masked", "three_frame_masked", "three_frame_full"),
        ("full_minus_two_frame", "two_frame_saved", "three_frame_full"),
        ("masked_minus_two_frame", "two_frame_saved", "three_frame_masked"),
    ):
        before = [
            row
            for row in all_edge_rows
            if row["embryo"] == embryo and row["mode"] == reference_mode
        ]
        after = [
            row for row in all_edge_rows if row["embryo"] == embryo and row["mode"] == changed_mode
        ]
        comparison, paired_rows = compare_known_edges(before, after)
        comparison["paired_video_bootstrap"] = paired_video_bootstrap(
            paired_rows,
            samples,
            repeats=diag_cfg["bootstrap_samples"],
            seed=diag_cfg["bootstrap_seed"],
            confidence=diag_cfg["bootstrap_confidence"],
        )
        comparisons[embryo][label] = comparison
        write_csv(OUTPUT / f"{embryo}_{label}_edges.csv", paired_rows)
        write_csv(
            OUTPUT / f"{embryo}_{label}_changed_edges.csv",
            [
                row
                for row in paired_rows
                if row["rescued_threshold"]
                or row["harmed_threshold"]
                or row["rescued_top1"]
                or row["harmed_top1"]
            ],
        )
write_csv(OUTPUT / "known_edge_predictions.csv", all_edge_rows)
write_json(OUTPUT / "window_metrics.json", all_window_rows)
write_json(OUTPUT / "prediction_manifest.json", prediction_records)
write_json(OUTPUT / "input_manifest.json", input_records)
write_json(OUTPUT / "reference_reproduction.json", reference_checks)
reproduced = all(all(row["checks"].values()) for row in reference_checks)
summary = {
    "source_kernel": diag_cfg["source_kernel"],
    "source_kernel_version": diag_cfg["source_kernel_version"],
    "model_manifest_sha256": diag_cfg["model_manifest_sha256"],
    "selection_sha256": diag_cfg["selection_sha256"],
    "reference_reproduced": reproduced,
    "weight_evidence": weight_evidence,
    "summaries": all_summaries,
    "comparisons": comparisons,
    "notebook_runtime_seconds": time.perf_counter() - NOTEBOOK_STARTED,
    "created_at": datetime.now(UTC).isoformat(),
    "model_training_count": 0,
    "prediction_content_sha256": json_sha256(
        [
            {"path": row["path"], "array_content_sha256": row["array_content_sha256"]}
            for row in prediction_records
        ]
    ),
    "known_edge_predictions_sha256": file_sha256(OUTPUT / "known_edge_predictions.csv"),
    "prediction_manifest_sha256": file_sha256(OUTPUT / "prediction_manifest.json"),
    "input_manifest_sha256": file_sha256(OUTPUT / "input_manifest.json"),
}
write_json(OUTPUT / "diagnostic_summary.json", summary)
update_metrics(
    METRICS_PATH,
    {
        "status": "debug_completed" if reproduced else "failed",
        "context_diagnostic": summary,
        "evidence": {
            "context_diagnostic": {
                "kernel_id": "kentookumura/exp027-multi-frame-tracker-diagnostic",
                "source_kernel_version": diag_cfg["source_kernel_version"],
                "summary_sha256": file_sha256(OUTPUT / "diagnostic_summary.json"),
                "oof_prediction_sha": summary["prediction_content_sha256"],
                "resource": torch.cuda.get_device_name(device),
                "internet_enabled": False,
                "notebook_runtime_seconds": summary["notebook_runtime_seconds"],
            }
        },
    },
)
print(
    json.dumps({"reference_reproduced": reproduced, "comparisons": comparisons}, indent=2),
    flush=True,
)
if not reproduced:
    raise RuntimeError({"pilot_reproduction_failed": reference_checks})
print("Saved-model context diagnostic complete.", flush=True)
