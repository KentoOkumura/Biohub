# %%
# ruff: noqa: E501
# %% [markdown]
# # exp038: nearest-past candidate attention
#
# 固定候補と教師を保ち、物理距離の近傍8点を選択してからattentionを計算する。
# 学習側の前身候補保持率99%、実行費用、両胚の接続診断を順に確認する。
# 全graph推論・公式score・submissionはこのNotebookでは実行しない。
#
# ## Contents
# 1. Imports
# 2. Reachable runtime, cache, teacher, model and diagnostic definitions
# 3. Setup and fixed contract
# 4. Input integrity and embryo splits
# 5. Models and loaders
# 6. Candidate retention and runtime benchmark
# 7. Two-fold training and saved-control comparison
# 8. Metrics, model manifest and artifacts

# %%
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.checkpoint import checkpoint

# %% [markdown]
# ## Definitions: velocity_history

# %%

HistoryFrame = dict[str, np.ndarray]


HistoryByFrame = dict[tuple[str, int], HistoryFrame]


def _validated_candidate_inputs(
    candidate_ids: np.ndarray, coords_physical: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(candidate_ids, dtype=np.int64)
    coords = np.asarray(coords_physical, dtype=np.float32)
    if ids.ndim != 1 or coords.shape != (len(ids), 3):
        raise ValueError("candidate ids and physical coordinates do not align")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("candidate ids must be unique within a frame")
    if not np.isfinite(coords).all():
        raise ValueError("candidate coordinates contain non-finite values")
    return (ids, coords)


def empty_history_frame(candidate_ids: np.ndarray, coords_physical: np.ndarray) -> HistoryFrame:
    ids, coords = _validated_candidate_inputs(candidate_ids, coords_physical)
    return {
        "candidate_ids": ids.copy(),
        "coords_physical": coords.copy(),
        "velocity": np.zeros((len(ids), 3), dtype=np.float32),
        "present": np.zeros(len(ids), dtype=bool),
        "length": np.zeros(len(ids), dtype=np.int64),
        "confidence": np.zeros(len(ids), dtype=np.float32),
    }


def align_history_frame(
    history: HistoryFrame,
    candidate_ids: np.ndarray,
    coords_physical: np.ndarray,
    *,
    coordinate_atol: float = 1e-05,
) -> HistoryFrame:
    ids, coords = _validated_candidate_inputs(candidate_ids, coords_physical)
    history_ids, history_coords = _validated_candidate_inputs(
        history["candidate_ids"], history["coords_physical"]
    )
    if set(map(int, ids)) != set(map(int, history_ids)):
        raise ValueError("history and cache candidate ids differ")
    index = {int(candidate_id): idx for idx, candidate_id in enumerate(history_ids)}
    order = np.asarray([index[int(candidate_id)] for candidate_id in ids], dtype=np.int64)
    if not np.allclose(history_coords[order], coords, rtol=0.0, atol=coordinate_atol):
        raise ValueError("history and cache candidate coordinates differ")
    aligned = {
        "candidate_ids": ids.copy(),
        "coords_physical": coords.copy(),
        "velocity": np.asarray(history["velocity"], dtype=np.float32)[order].copy(),
        "present": np.asarray(history["present"], dtype=bool)[order].copy(),
        "length": np.asarray(history["length"], dtype=np.int64)[order].copy(),
        "confidence": np.asarray(history["confidence"], dtype=np.float32)[order].copy(),
    }
    expected_shapes = {
        "velocity": (len(ids), 3),
        "present": (len(ids),),
        "length": (len(ids),),
        "confidence": (len(ids),),
    }
    observed_shapes = {name: aligned[name].shape for name in expected_shapes}
    if observed_shapes != expected_shapes:
        raise ValueError(
            {"history_shape_mismatch": {"expected": expected_shapes, "actual": observed_shapes}}
        )
    for name in ("velocity", "confidence"):
        if not np.isfinite(aligned[name]).all():
            raise ValueError(f"history contains non-finite {name}")
    return aligned


# %% [markdown]
# ## Definitions: frozen_tracker

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
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
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
        digest.update(b"\x00")
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


def embryo_id(sample_name: str) -> str:
    embryo, separator, _ = sample_name.partition("_")
    if not separator or not embryo:
        raise ValueError(f"sample name has no embryo prefix: {sample_name}")
    return embryo


def build_embryo_splits(
    sample_names: list[str], outer_specs: list[dict[str, Any]], split_seed: int
) -> list[dict[str, Any]]:
    sample_set = set(sample_names)
    if len(sample_set) != len(sample_names):
        raise ValueError("duplicate sample names")
    records: list[dict[str, Any]] = []
    outer_evaluation_seen: set[str] = set()
    for spec in outer_specs:
        fold = int(spec["fold"])
        train_embryo = str(spec["train_embryo"])
        evaluation_embryo = str(spec["evaluation_embryo"])
        train_pool = sorted(name for name in sample_names if embryo_id(name) == train_embryo)
        evaluation = sorted(name for name in sample_names if embryo_id(name) == evaluation_embryo)
        shuffled = list(train_pool)
        random.Random(split_seed).shuffle(shuffled)
        n_internal = max(1, len(shuffled) // 10)
        internal_validation = shuffled[:n_internal]
        gradient_update = shuffled[n_internal:]
        record = {
            "fold": fold,
            "train_embryo": train_embryo,
            "evaluation_embryo": evaluation_embryo,
            "gradient_update": gradient_update,
            "internal_validation": internal_validation,
            "outer_evaluation": evaluation,
        }
        expected = {
            "gradient_update": int(spec["expected_train_sample_count"]),
            "internal_validation": int(spec["expected_internal_validation_sample_count"]),
            "outer_evaluation": int(spec["expected_evaluation_sample_count"]),
        }
        actual = {key: len(record[key]) for key in expected}
        if actual != expected:
            raise ValueError({"fold": fold, "expected": expected, "actual": actual})
        train_names = set(gradient_update) | set(internal_validation)
        evaluation_names = set(evaluation)
        if set(gradient_update) & set(internal_validation):
            raise ValueError(f"fold {fold} internal split overlaps")
        if train_names != set(train_pool):
            raise ValueError(f"fold {fold} internal split does not cover the training embryo")
        if train_names & evaluation_names:
            raise ValueError(f"fold {fold} outer evaluation leaked into training")
        if outer_evaluation_seen & evaluation_names:
            raise ValueError("an outer evaluation sample appears in multiple folds")
        outer_evaluation_seen.update(evaluation_names)
        records.append(record)
    if outer_evaluation_seen != sample_set:
        raise ValueError("outer evaluation folds do not cover every sample exactly once")
    return records


def graph_from_geff(path: Path) -> Any:
    import tracksdata as td

    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def load_annotation_graph(
    path: Path, voxel_scale_zyx_um: tuple[float, float, float]
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
    if any((source not in seen or target not in seen for source, target in edges)):
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
        return (matched_ids, matched_distances)
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
    return (matched_ids, matched_distances)


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
    loss = (1 - p_t) ** gamma * bce
    return loss[mask].mean()


def batch_legacy_focal_bce(
    logits: Any, target: Any, source_mask: Any, target_mask: Any, gamma: float = 2.0
) -> Any:
    import torch

    losses = []
    for batch_index in range(logits.shape[0]):
        n_source = int(source_mask[batch_index].sum().item())
        n_target = int(target_mask[batch_index].sum().item())
        losses.append(
            legacy_focal_bce(
                logits[batch_index, :n_source, :n_target],
                target[batch_index, :n_source, :n_target],
                gamma=gamma,
            )
        )
    return torch.stack(losses).mean()


def _read_cache_payload(
    path: Path, *, load_all_arrays: bool
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
    return (arrays, metadata)


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
        candidate_ids = np.asarray(arrays[f"candidate_ids_{side}"])
        shapes = {
            "candidate_ids": candidate_ids.shape,
            "grid": np.asarray(arrays[f"coords_{side}_grid"]).shape,
            "physical": np.asarray(arrays[f"coords_{side}_physical"]).shape,
            "position": np.asarray(arrays[f"position_features_{side}"]).shape,
            "primary": np.asarray(arrays[f"primary_features_{side}"]).shape,
        }
        expected_shapes = {
            "candidate_ids": (count,),
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
        if not np.issubdtype(candidate_ids.dtype, np.integer):
            raise ValueError(f"cache candidate ids are not integers: {path} {side}")
        if len(np.unique(candidate_ids)) != count:
            raise ValueError(f"cache candidate ids are not unique: {path} {side}")
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
    return (arrays, metadata)


def build_window_example(
    path: Path,
    annotation: AnnotationGraph,
    *,
    feature_channels: int,
    expected_primary_checkpoint_sha256: str,
    max_matching_distance_um: float,
    downsample_zyx: tuple[float, float, float],
    verify_content: bool = False,
    history_by_frame: HistoryByFrame | None = None,
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
        source_matches, target_matches, annotation.edges, outgoing_edges=annotation.outgoing_edges
    )
    active_mask = legacy_active_pair_mask(target)
    unknown_pairs = (source_matches < 0)[:, None] | (target_matches < 0)[None, :]
    downsample = np.asarray(downsample_zyx, dtype=np.float32)
    history_key = (path.parent.name, source_frame)
    if history_by_frame is None:
        source_history = empty_history_frame(
            arrays["candidate_ids_src"], arrays["coords_src_physical"]
        )
    else:
        if history_key not in history_by_frame:
            raise ValueError(f"fixed history is missing {history_key}")
        source_history = align_history_frame(
            history_by_frame[history_key],
            arrays["candidate_ids_src"],
            arrays["coords_src_physical"],
        )
    example = {
        "sample": path.parent.name,
        "window_frames": [source_frame, target_frame],
        "features_src": np.concatenate(
            [arrays["primary_features_src"], arrays["position_features_src"]], axis=1
        ).astype(np.float32, copy=False),
        "features_tgt": np.concatenate(
            [arrays["primary_features_tgt"], arrays["position_features_tgt"]], axis=1
        ).astype(np.float32, copy=False),
        "coords_src": np.asarray(arrays["coords_src_grid"], dtype=np.float32) * downsample,
        "coords_tgt": np.asarray(arrays["coords_tgt_grid"], dtype=np.float32) * downsample,
        "candidate_ids_src": np.asarray(arrays["candidate_ids_src"], dtype=np.int64),
        "candidate_ids_tgt": np.asarray(arrays["candidate_ids_tgt"], dtype=np.int64),
        "coords_src_physical": np.asarray(arrays["coords_src_physical"], dtype=np.float32),
        "coords_tgt_physical": np.asarray(arrays["coords_tgt_physical"], dtype=np.float32),
        "velocity_src": source_history["velocity"],
        "history_present_src": source_history["present"],
        "history_length_src": source_history["length"],
        "history_confidence_src": source_history["confidence"],
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
    coords_src_physical = torch.zeros(batch_size, max_source, 3, dtype=torch.float32)
    coords_tgt_physical = torch.zeros(batch_size, max_target, 3, dtype=torch.float32)
    velocity_src = torch.zeros(batch_size, max_source, 3, dtype=torch.float32)
    history_present_src = torch.zeros(batch_size, max_source, dtype=torch.bool)
    history_length_src = torch.zeros(batch_size, max_source, dtype=torch.int64)
    history_confidence_src = torch.zeros(batch_size, max_source, dtype=torch.float32)
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
        coords_src_physical[batch_index, :n_source] = torch.from_numpy(
            example["coords_src_physical"]
        )
        coords_tgt_physical[batch_index, :n_target] = torch.from_numpy(
            example["coords_tgt_physical"]
        )
        velocity_src[batch_index, :n_source] = torch.from_numpy(example["velocity_src"])
        history_present_src[batch_index, :n_source] = torch.from_numpy(
            example["history_present_src"]
        )
        history_length_src[batch_index, :n_source] = torch.from_numpy(example["history_length_src"])
        history_confidence_src[batch_index, :n_source] = torch.from_numpy(
            example["history_confidence_src"]
        )
        source_mask[batch_index, :n_source] = True
        target_mask[batch_index, :n_target] = True
        targets[batch_index, :n_source, :n_target] = torch.from_numpy(example["target"])
    return {
        "features_src": features_src,
        "features_tgt": features_tgt,
        "coords_src": coords_src,
        "coords_tgt": coords_tgt,
        "coords_src_physical": coords_src_physical,
        "coords_tgt_physical": coords_tgt_physical,
        "velocity_src": velocity_src,
        "history_present_src": history_present_src,
        "history_length_src": history_length_src,
        "history_confidence_src": history_confidence_src,
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


def paths_for_samples(paths: list[Path], samples: list[str]) -> list[Path]:
    allowed = set(samples)
    return [path for path in paths if path.parent.name in allowed]


def filter_nonempty_gt_window_paths(
    paths: list[Path], annotations: dict[str, AnnotationGraph]
) -> tuple[list[Path], dict[str, Any]]:
    """Match the public trainer's exclusion of windows containing an empty GT frame."""
    eligible: list[Path] = []
    skipped: list[dict[str, Any]] = []
    skipped_by_sample: dict[str, int] = {}
    for path in paths:
        sample = path.parent.name
        if sample not in annotations:
            raise ValueError(f"annotation is missing for cache sample: {path}")
        parts = path.stem.split("_")
        if len(parts) != 2:
            raise ValueError(f"cache filename does not encode two frames: {path}")
        source_frame, target_frame = map(int, parts)
        if target_frame != source_frame + 1:
            raise ValueError(f"cache filename is not an adjacent frame pair: {path}")
        missing_frames = [
            frame
            for frame in (source_frame, target_frame)
            if frame not in annotations[sample].frames
        ]
        if not missing_frames:
            eligible.append(path)
            continue
        skipped.append(
            {
                "sample": sample,
                "window_frames": [source_frame, target_frame],
                "empty_gt_frames": missing_frames,
            }
        )
        skipped_by_sample[sample] = skipped_by_sample.get(sample, 0) + 1
    audit = {
        "policy": "skip_window_if_any_frame_has_zero_gt_nodes",
        "public_source_function": "get_window_data",
        "input_window_count": len(paths),
        "eligible_window_count": len(eligible),
        "skipped_window_count": len(skipped),
        "skipped_by_sample": dict(sorted(skipped_by_sample.items())),
        "skipped_windows": skipped,
    }
    return (eligible, audit)


def aggregate_teacher_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    integer_keys = (
        "candidate_nodes",
        "matched_candidate_nodes",
        "gt_nodes",
        "positive_edges",
        "active_pairs",
        "active_pairs_with_unknown_endpoint",
        "division_parents",
        "source_match_distance_count",
        "target_match_distance_count",
    )
    float_keys = ("source_match_distance_sum_um", "target_match_distance_sum_um")
    result: dict[str, Any] = {
        key: int(sum(int(record[key]) for record in records)) for key in integer_keys
    }
    result.update({key: float(sum(float(record[key]) for record in records)) for key in float_keys})
    result["candidate_node_precision_against_sparse_gt"] = safe_ratio(
        result["matched_candidate_nodes"], result["candidate_nodes"]
    )
    result["candidate_node_recall"] = safe_ratio(
        result["matched_candidate_nodes"], result["gt_nodes"]
    )
    result["unknown_endpoint_fraction_of_active_pairs"] = safe_ratio(
        result["active_pairs_with_unknown_endpoint"], result["active_pairs"]
    )
    match_count = result["source_match_distance_count"] + result["target_match_distance_count"]
    match_sum = result["source_match_distance_sum_um"] + result["target_match_distance_sum_um"]
    result["mean_match_distance_um"] = safe_ratio(match_sum, match_count)
    return result


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def tracker_logits(model: Any, batch: dict[str, Any]) -> Any:
    args = (
        batch["features_src"],
        batch["features_tgt"],
        batch["coords_src"],
        batch["coords_tgt"],
        batch["source_mask"],
        batch["target_mask"],
    )
    if bool(getattr(model, "requires_past_candidate_attention", False)):
        return model(
            *args,
            coords_prev_physical=batch["coords_prev_physical"],
            coords_src_physical=batch["coords_src_physical"],
            coords_tgt_physical=batch["coords_tgt_physical"],
            prev_mask=batch["prev_mask"],
            candidate_ids_prev=batch["candidate_ids_prev"],
        )
    if not bool(getattr(model, "requires_velocity_features", False)):
        return model(*args)
    return model(
        *args,
        velocity_src=batch["velocity_src"],
        history_present_src=batch["history_present_src"],
        history_length_src=batch["history_length_src"],
        history_confidence_src=batch["history_confidence_src"],
        coords_src_physical=batch["coords_src_physical"],
        coords_tgt_physical=batch["coords_tgt_physical"],
    )


def train_one_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    scaler: Any,
    device: Any,
    *,
    gamma: float,
    gradient_clip_norm: float,
    use_amp: bool,
) -> dict[str, float]:
    import torch

    model.train()
    total_loss = 0.0
    batches = 0
    windows = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"
        ):
            logits = tracker_logits(model, batch)
            loss = batch_legacy_focal_bce(
                logits, batch["target"], batch["source_mask"], batch["target_mask"], gamma=gamma
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        window_count = len(batch["metadata"])
        total_loss += float(loss.detach().cpu()) * window_count
        batches += 1
        windows += window_count
    return {
        "legacy_mask_loss": safe_ratio(total_loss, windows),
        "batch_count": float(batches),
        "window_count": float(windows),
    }


def evaluate_tracker(
    model: Any, loader: Any, device: Any, *, gamma: float, use_amp: bool
) -> dict[str, Any]:
    import torch

    model.eval()
    loss_sum = 0.0
    loss_windows = 0
    correct_pairs = 0
    active_pairs = 0
    true_positive_edges = 0
    positive_edges = 0
    recovered_division_parents = 0
    division_parents = 0
    teacher_records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                logits = tracker_logits(model, batch)
            for batch_index in range(logits.shape[0]):
                n_source = int(batch["source_mask"][batch_index].sum().item())
                n_target = int(batch["target_mask"][batch_index].sum().item())
                pair_logits = logits[batch_index, :n_source, :n_target].float()
                pair_target = batch["target"][batch_index, :n_source, :n_target].float()
                loss = legacy_focal_bce(pair_logits, pair_target, gamma=gamma)
                loss_sum += float(loss.detach().cpu())
                loss_windows += 1
                probabilities = torch.softmax(pair_logits, dim=0)
                predictions = probabilities > 0.5
                active_rows = pair_target.sum(dim=1) > 0
                active_cols = pair_target.sum(dim=0) > 0
                pair_mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
                if pair_mask.any():
                    correct_pairs += int((predictions[pair_mask] == pair_target[pair_mask]).sum())
                    active_pairs += int(pair_mask.sum())
                positive = pair_target > 0.5
                true_positive_edges += int((predictions & positive).sum())
                positive_edges += int(positive.sum())
                division_rows = pair_target.sum(dim=1) > 1
                for row in torch.nonzero(division_rows, as_tuple=False).flatten():
                    recovered_division_parents += int(bool(predictions[row][positive[row]].all()))
                    division_parents += 1
            teacher_records.extend(item["teacher_stats"] for item in batch["metadata"])
    teacher = aggregate_teacher_stats(teacher_records)
    metrics = {
        "legacy_mask_loss": safe_ratio(loss_sum, loss_windows),
        "edge_accuracy": safe_ratio(correct_pairs, active_pairs),
        "positive_edge_recall": safe_ratio(true_positive_edges, positive_edges),
        "division_parent_recall": safe_ratio(recovered_division_parents, division_parents),
        "candidate_node_recall": teacher["candidate_node_recall"],
        "window_count": loss_windows,
        "active_pair_count": active_pairs,
        "positive_edge_count": positive_edges,
        "division_parent_count": division_parents,
        "teacher": teacher,
    }
    metrics["selection_score"] = metrics["edge_accuracy"] * metrics["candidate_node_recall"]
    return metrics


def evaluate_tracker_diagnostic(
    model: Any,
    loader: Any,
    device: Any,
    *,
    gamma: float,
    use_amp: bool,
    edge_probability_threshold: float = 0.5,
    prediction_dir: Path | None = None,
) -> dict[str, Any]:
    import torch

    if not 0.0 <= edge_probability_threshold <= 1.0:
        raise ValueError("edge probability threshold must be in [0, 1]")
    model.eval()
    loss_sum = 0.0
    loss_windows = 0
    correct_pairs = 0
    active_pairs = 0
    active_negative_pairs = 0
    false_positive_active_pairs = 0
    true_positive_edges = 0
    positive_edges = 0
    top1_correct = 0
    top1_count = 0
    recovered_division_parents = 0
    division_parents = 0
    history_present_sources = 0
    history_sources = 0
    teacher_records: list[dict[str, Any]] = []
    prediction_digest = hashlib.sha256()
    diagnostic_slices: dict[str, dict[str, int]] = {}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                logits = tracker_logits(model, batch)
            for batch_index in range(logits.shape[0]):
                n_source = int(batch["source_mask"][batch_index].sum().item())
                n_target = int(batch["target_mask"][batch_index].sum().item())
                pair_logits = logits[batch_index, :n_source, :n_target].float()
                pair_target = batch["target"][batch_index, :n_source, :n_target].float()
                if not torch.isfinite(pair_logits).all():
                    raise FloatingPointError("non-finite diagnostic logits")
                loss = legacy_focal_bce(pair_logits, pair_target, gamma=gamma)
                loss_sum += float(loss.detach().cpu())
                loss_windows += 1
                probabilities = torch.softmax(pair_logits, dim=0)
                predictions = probabilities >= edge_probability_threshold
                positive = pair_target > 0.5
                active_rows = positive.sum(dim=1) > 0
                active_cols = positive.sum(dim=0) > 0
                pair_mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
                active_negative = pair_mask & ~positive
                if pair_mask.any():
                    correct_pairs += int((predictions[pair_mask] == positive[pair_mask]).sum())
                    active_pairs += int(pair_mask.sum())
                active_negative_pairs += int(active_negative.sum())
                false_positive_active_pairs += int((predictions & active_negative).sum())
                true_positive_edges += int((predictions & positive).sum())
                positive_edges += int(positive.sum())
                if active_cols.any():
                    best_sources = probabilities.argmax(dim=0)
                    active_target_indices = torch.nonzero(active_cols, as_tuple=False).flatten()
                    top1_correct += int(
                        sum(
                            bool(positive[int(best_sources[column]), column])
                            for column in active_target_indices
                        )
                    )
                    top1_count += int(len(active_target_indices))
                division_rows = positive.sum(dim=1) > 1
                for row in torch.nonzero(division_rows, as_tuple=False).flatten():
                    recovered_division_parents += int(bool(predictions[row][positive[row]].all()))
                    division_parents += 1
                history_present_sources += int(
                    batch["history_present_src"][batch_index, :n_source].sum().item()
                )
                history_sources += n_source
                pair_logits_cpu = np.ascontiguousarray(
                    pair_logits.detach().cpu().numpy(), dtype=np.float32
                )
                metadata = batch["metadata"][batch_index]
                for name, row_mask in source_diagnostic_masks(batch, batch_index, n_source).items():
                    counts = diagnostic_slices.setdefault(
                        name,
                        {
                            "sources": 0,
                            "positive_edges": 0,
                            "true_positive_edges": 0,
                            "false_positive_active_pairs": 0,
                            "division_parents": 0,
                            "recovered_division_parents": 0,
                        },
                    )
                    counts["sources"] += int(row_mask.sum())
                    counts["positive_edges"] += int(positive[row_mask].sum())
                    counts["true_positive_edges"] += int((predictions & positive)[row_mask].sum())
                    counts["false_positive_active_pairs"] += int(
                        (predictions & active_negative)[row_mask].sum()
                    )
                    divisions = division_rows & row_mask
                    counts["division_parents"] += int(divisions.sum())
                    counts["recovered_division_parents"] += int(
                        ((predictions | ~positive).all(dim=1) & divisions).sum()
                    )
                if prediction_dir is not None:
                    saved_mask = positive | predictions
                    saved_mask[
                        probabilities.argmax(dim=0), torch.arange(n_target, device=device)
                    ] = True
                    rows, cols = torch.nonzero(saved_mask, as_tuple=True)
                    output_dir = prediction_dir / str(metadata["sample"])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    first, second = metadata["window_frames"]
                    np.savez_compressed(
                        output_dir / f"{first:06d}_{second:06d}.npz",
                        source_ids=np.asarray(metadata["candidate_ids_src"]),
                        target_ids=np.asarray(metadata["candidate_ids_tgt"]),
                        rows=rows.cpu().numpy(),
                        cols=cols.cpu().numpy(),
                        probability=probabilities[rows, cols].cpu().numpy(),
                        positive=positive[rows, cols].cpu().numpy(),
                        active_pair_count=int(pair_mask.sum()),
                        active_negative_pair_count=int(active_negative.sum()),
                    )
                prediction_digest.update(
                    json.dumps(
                        {
                            "sample": metadata["sample"],
                            "window_frames": metadata["window_frames"],
                            "shape": list(pair_logits_cpu.shape),
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                )
                prediction_digest.update(b"\x00")
                prediction_digest.update(pair_logits_cpu.tobytes(order="C"))
            teacher_records.extend(item["teacher_stats"] for item in batch["metadata"])
    teacher = aggregate_teacher_stats(teacher_records)
    metrics = {
        "legacy_mask_loss": safe_ratio(loss_sum, loss_windows),
        "edge_accuracy": safe_ratio(correct_pairs, active_pairs),
        "positive_edge_recall": safe_ratio(true_positive_edges, positive_edges),
        "false_positive_active_pair_rate": safe_ratio(
            false_positive_active_pairs, active_negative_pairs
        ),
        "correct_parent_top1_accuracy": safe_ratio(top1_correct, top1_count),
        "division_parent_recall": safe_ratio(recovered_division_parents, division_parents),
        "candidate_node_recall": teacher["candidate_node_recall"],
        "window_count": loss_windows,
        "active_pair_count": active_pairs,
        "active_negative_pair_count": active_negative_pairs,
        "false_positive_active_pair_count": false_positive_active_pairs,
        "true_positive_edge_count": true_positive_edges,
        "positive_edge_count": positive_edges,
        "correct_parent_top1_count": top1_correct,
        "correct_parent_top1_denominator": top1_count,
        "recovered_division_parent_count": recovered_division_parents,
        "division_parent_count": division_parents,
        "history_present_source_count": history_present_sources,
        "history_source_count": history_sources,
        "prediction_content_sha256": prediction_digest.hexdigest(),
        "diagnostic_slices": diagnostic_slices,
        "edge_probability_threshold": float(edge_probability_threshold),
        "teacher": teacher,
    }
    metrics["history_coverage"] = safe_ratio(history_present_sources, history_sources)
    metrics["selection_score"] = metrics["edge_accuracy"] * metrics["candidate_node_recall"]
    return metrics


def extract_public_tracker_state(full_state: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("transformer.", "module.transformer.")
    for prefix in prefixes:
        selected = {
            key[len(prefix) :]: value for key, value in full_state.items() if key.startswith(prefix)
        }
        if selected:
            return selected
    expected_tracker_roots = {"proj", "norm_in", "blocks", "norm_out", "pair_mlp"}
    observed_roots = {key.split(".", 1)[0] for key in full_state}
    if expected_tracker_roots.issubset(observed_roots):
        return dict(full_state)
    raise ValueError("checkpoint does not contain a SimpleNodeTransformer state")


def canonical_state_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        descriptor = {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        digest.update(json.dumps(descriptor, separators=(",", ":"), sort_keys=True).encode())
        digest.update(b"\x00")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataloader_worker_init(_: int) -> None:
    import torch

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# %% [markdown]
# ## Definitions: past_candidate_attention

# %%

PAST_CANDIDATE_FEATURE_NAMES = (
    "past_displacement_z",
    "past_displacement_y",
    "past_displacement_x",
    "current_displacement_z",
    "current_displacement_y",
    "current_displacement_x",
    "displacement_change_z",
    "displacement_change_y",
    "displacement_change_x",
    "past_displacement_norm",
    "current_displacement_norm",
    "displacement_change_norm",
    "direction_cosine",
)


def select_past_candidates(
    previous: torch.Tensor,
    source: torch.Tensor,
    previous_ids: torch.Tensor,
    previous_mask: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Source-centred physical kNN, with candidate ID breaking exact distance ties."""
    if k < 1:
        raise ValueError("k must be positive")
    count = min(k, previous.shape[1])
    batch, sources = source.shape[:2]
    if not count:
        return (
            previous.new_zeros((batch, sources, 0, 3)),
            previous_mask.new_zeros((batch, sources, 0)),
            previous_ids.new_zeros((batch, sources, 0)),
        )
    id_order = torch.argsort(previous_ids, dim=1, stable=True)
    ordered = torch.gather(previous, 1, id_order[..., None].expand(-1, -1, 3))
    valid = torch.gather(previous_mask, 1, id_order)
    ordered = torch.where(valid[..., None], ordered, torch.zeros_like(ordered))
    distances = ((source[:, :, None] - ordered[:, None]) ** 2).sum(-1)
    distances = distances.masked_fill(~valid[:, None], float("inf"))
    nearest = torch.argsort(distances, dim=-1, stable=True)[..., :count]
    indices = torch.gather(id_order[:, None].expand(-1, sources, -1), 2, nearest)
    selected_mask = torch.gather(previous_mask[:, None].expand(-1, sources, -1), 2, indices)
    selected = torch.gather(
        previous[:, None].expand(-1, sources, -1, -1), 2, indices[..., None].expand(-1, -1, -1, 3)
    )
    selected = torch.where(selected_mask[..., None], selected, torch.zeros_like(selected))
    selected_ids = torch.gather(previous_ids[:, None].expand(-1, sources, -1), 2, indices)
    return (selected, selected_mask, selected_ids)


def build_past_candidate_features(
    *,
    coords_prev_physical: torch.Tensor,
    coords_src_physical: torch.Tensor,
    coords_tgt_physical: torch.Tensor,
    vector_scale_um: float,
    cosine_epsilon: float,
) -> torch.Tensor:
    """Build 13-D features only after gathering the source-specific past points."""
    if vector_scale_um <= 0 or cosine_epsilon <= 0:
        raise ValueError("vector scale and cosine epsilon must be positive")
    if coords_prev_physical.ndim == 3:
        coords_prev_physical = coords_prev_physical[:, None].expand(
            -1, coords_src_physical.shape[1], -1, -1
        )
    if coords_prev_physical.ndim != 4 or coords_prev_physical.shape[-1] != 3:
        raise ValueError("coords_prev_physical must have shape (B, N_source, K, 3)")
    if coords_src_physical.ndim != 3 or coords_src_physical.shape[-1] != 3:
        raise ValueError("coords_src_physical must have shape (B, N_source, 3)")
    if coords_tgt_physical.ndim != 3 or coords_tgt_physical.shape[-1] != 3:
        raise ValueError("coords_tgt_physical must have shape (B, N_target, 3)")
    batch_size = coords_prev_physical.shape[0]
    if coords_src_physical.shape[0] != batch_size or coords_tgt_physical.shape[0] != batch_size:
        raise ValueError("past, source, and target coordinates must share a batch dimension")
    scale = float(vector_scale_um)
    past_displacement = (
        coords_src_physical[:, :, None, None, :] - coords_prev_physical[:, :, None, :, :]
    ) / scale
    current_displacement = (
        coords_tgt_physical[:, :, None, :] - coords_src_physical[:, None, :, :]
    ).transpose(1, 2) / scale
    current_displacement = current_displacement.unsqueeze(3)
    past_displacement = past_displacement.expand(-1, -1, coords_tgt_physical.shape[1], -1, -1)
    current_displacement = current_displacement.expand(
        -1, -1, -1, coords_prev_physical.shape[2], -1
    )
    displacement_change = current_displacement - past_displacement
    past_norm = torch.linalg.vector_norm(past_displacement, dim=-1, keepdim=True)
    current_norm = torch.linalg.vector_norm(current_displacement, dim=-1, keepdim=True)
    change_norm = torch.linalg.vector_norm(displacement_change, dim=-1, keepdim=True)
    dot = (past_displacement * current_displacement).sum(dim=-1, keepdim=True)
    denominator = past_norm * current_norm
    cosine = dot / torch.clamp(denominator, min=float(cosine_epsilon))
    cosine = torch.where(denominator > float(cosine_epsilon), cosine, torch.zeros_like(cosine))
    features = torch.cat(
        [
            past_displacement,
            current_displacement,
            displacement_change,
            past_norm,
            current_norm,
            change_norm,
            cosine,
        ],
        dim=-1,
    )
    if features.shape[-1] != len(PAST_CANDIDATE_FEATURE_NAMES):
        raise RuntimeError("past-candidate feature width changed unexpectedly")
    if not torch.isfinite(features).all():
        raise FloatingPointError("past-candidate features contain non-finite values")
    return features


class PastCandidateAttentionTracker(nn.Module):
    """Public primary tracker plus attention over at most K past points per source."""

    requires_past_candidate_attention = True

    def __init__(
        self,
        base_tracker: nn.Module,
        *,
        feature_dim: int,
        hidden_dim: int,
        source_chunk_size: int,
        target_chunk_size: int,
        past_candidate_chunk_size: int,
        gradient_checkpointing: bool,
        vector_scale_um: float,
        cosine_epsilon: float,
        max_past_candidates: int = 8,
    ) -> None:
        super().__init__()
        if feature_dim != len(PAST_CANDIDATE_FEATURE_NAMES):
            raise ValueError("feature_dim does not match the past-candidate feature contract")
        if (
            hidden_dim < 1
            or source_chunk_size < 1
            or target_chunk_size < 1
            or (past_candidate_chunk_size < 1)
        ):
            raise ValueError("hidden and chunk sizes must be positive")
        self.base_tracker = base_tracker
        self.max_past_candidates = int(max_past_candidates)
        if self.max_past_candidates < 1:
            raise ValueError("max_past_candidates must be positive")
        self.source_chunk_size = int(source_chunk_size)
        self.target_chunk_size = int(target_chunk_size)
        self.past_candidate_chunk_size = int(past_candidate_chunk_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.vector_scale_um = float(vector_scale_um)
        self.cosine_epsilon = float(cosine_epsilon)
        self.candidate_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.GELU(),
        )
        self.attention_score = nn.Linear(hidden_dim, 1, bias=True)
        self.no_past_score = nn.Parameter(torch.zeros(()))
        self.delta_output = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.delta_output.weight)

    def _aggregate_source_chunk(
        self,
        coords_prev_physical: torch.Tensor,
        coords_src_physical: torch.Tensor,
        coords_tgt_physical: torch.Tensor,
        prev_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, source_count, _ = coords_src_physical.shape
        target_count = coords_tgt_physical.shape[1]
        hidden_dim = self.delta_output.in_features
        running_max = self.no_past_score.expand(batch_size, source_count, target_count)
        denominator = torch.ones_like(running_max)
        numerator = torch.zeros(
            batch_size,
            source_count,
            target_count,
            hidden_dim,
            device=coords_src_physical.device,
            dtype=coords_src_physical.dtype,
        )
        for start in range(0, coords_prev_physical.shape[2], self.past_candidate_chunk_size):
            stop = min(start + self.past_candidate_chunk_size, coords_prev_physical.shape[2])
            features = build_past_candidate_features(
                coords_prev_physical=coords_prev_physical[:, :, start:stop],
                coords_src_physical=coords_src_physical,
                coords_tgt_physical=coords_tgt_physical,
                vector_scale_um=self.vector_scale_um,
                cosine_epsilon=self.cosine_epsilon,
            )
            candidate_hidden = self.candidate_mlp(features)
            scores = self.attention_score(candidate_hidden).squeeze(-1)
            valid = prev_mask[:, :, None, start:stop]
            scores = scores.masked_fill(~valid, float("-inf"))
            chunk_max = scores.max(dim=-1).values
            next_max = torch.maximum(running_max, chunk_max)
            old_scale = torch.exp(running_max - next_max)
            weights = torch.exp(scores - next_max.unsqueeze(-1))
            weights = torch.where(valid, weights, torch.zeros_like(weights))
            denominator = denominator * old_scale + weights.sum(dim=-1)
            numerator = numerator * old_scale.unsqueeze(-1) + (
                weights.unsqueeze(-1) * candidate_hidden
            ).sum(dim=-2)
            running_max = next_max
        return numerator / denominator.unsqueeze(-1)

    def forward(
        self,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_t1: torch.Tensor | None = None,
        *,
        coords_prev_physical: torch.Tensor,
        coords_src_physical: torch.Tensor,
        coords_tgt_physical: torch.Tensor,
        prev_mask: torch.Tensor,
        candidate_ids_prev: torch.Tensor,
    ) -> torch.Tensor:
        unbatched = feat_t.ndim == 2
        base_logits = self.base_tracker(feat_t, feat_t1, coords_t, coords_t1, mask_t, mask_t1)
        if unbatched:
            coords_prev_physical = coords_prev_physical.unsqueeze(0)
            coords_src_physical = coords_src_physical.unsqueeze(0)
            coords_tgt_physical = coords_tgt_physical.unsqueeze(0)
            prev_mask = prev_mask.unsqueeze(0)
            candidate_ids_prev = candidate_ids_prev.unsqueeze(0)
            base_logits = base_logits.unsqueeze(0)
            if mask_t is not None:
                mask_t = mask_t.unsqueeze(0)
            if mask_t1 is not None:
                mask_t1 = mask_t1.unsqueeze(0)
        if coords_src_physical.shape[1] == 0 or coords_tgt_physical.shape[1] == 0:
            return base_logits.squeeze(0) if unbatched else base_logits
        source_delta_chunks: list[torch.Tensor] = []
        for source_start in range(0, coords_src_physical.shape[1], self.source_chunk_size):
            source_stop = min(source_start + self.source_chunk_size, coords_src_physical.shape[1])
            selected_previous, selected_mask, _ = select_past_candidates(
                coords_prev_physical,
                coords_src_physical[:, source_start:source_stop],
                candidate_ids_prev,
                prev_mask,
                self.max_past_candidates,
            )
            target_delta_chunks: list[torch.Tensor] = []
            for target_start in range(0, coords_tgt_physical.shape[1], self.target_chunk_size):
                target_stop = min(
                    target_start + self.target_chunk_size, coords_tgt_physical.shape[1]
                )
                aggregate_args = (
                    selected_previous,
                    coords_src_physical[:, source_start:source_stop],
                    coords_tgt_physical[:, target_start:target_stop],
                    selected_mask,
                )
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    aggregated = checkpoint(
                        self._aggregate_source_chunk, *aggregate_args, use_reentrant=False
                    )
                else:
                    aggregated = self._aggregate_source_chunk(*aggregate_args)
                target_delta_chunks.append(self.delta_output(aggregated).squeeze(-1))
            source_delta_chunks.append(torch.cat(target_delta_chunks, dim=2))
        delta_logits = torch.cat(source_delta_chunks, dim=1)
        if mask_t is not None and mask_t1 is not None:
            valid_pairs = mask_t.unsqueeze(-1) & mask_t1.unsqueeze(1)
            delta_logits = delta_logits.masked_fill(~valid_pairs, 0.0)
        logits = base_logits + delta_logits
        return logits.squeeze(0) if unbatched else logits

    def load_public_tracker_state(self, state: dict[str, Any]) -> None:
        self.base_tracker.load_state_dict(state, strict=True)

    def base_state_dict(self) -> dict[str, Any]:
        return self.base_tracker.state_dict()


# %% [markdown]
# ## Definitions: past_candidate_data

# %%


def build_past_candidate_window_example(
    path: Path,
    annotation: AnnotationGraph,
    *,
    feature_channels: int,
    expected_primary_checkpoint_sha256: str,
    max_matching_distance_um: float,
    downsample_zyx: tuple[float, float, float],
) -> dict[str, Any]:
    example = build_window_example(
        path,
        annotation,
        feature_channels=feature_channels,
        expected_primary_checkpoint_sha256=expected_primary_checkpoint_sha256,
        max_matching_distance_um=max_matching_distance_um,
        downsample_zyx=downsample_zyx,
    )
    central_arrays, central_metadata = validate_window_cache(
        path,
        feature_channels=feature_channels,
        expected_primary_checkpoint_sha256=expected_primary_checkpoint_sha256,
    )
    source_frame, target_frame = map(int, central_metadata["window_frames"])
    if [source_frame, target_frame] != example["window_frames"]:
        raise ValueError(f"central window metadata changed while reading {path}")
    prior_path = path.with_name(f"{source_frame - 1:06d}_{source_frame:06d}.npz")
    if source_frame == 0:
        example["candidate_ids_prev"] = np.zeros((0,), dtype=np.int64)
        example["coords_prev_physical"] = np.zeros((0, 3), dtype=np.float32)
        example["prior_cache_content_sha256"] = None
        return example
    if not prior_path.is_file():
        raise FileNotFoundError(f"previous window missing within video: {prior_path}")
    prior_arrays, prior_metadata = validate_window_cache(
        prior_path,
        feature_channels=feature_channels,
        expected_primary_checkpoint_sha256=expected_primary_checkpoint_sha256,
    )
    if prior_metadata["window_frames"] != [source_frame - 1, source_frame]:
        raise ValueError(f"incorrect previous window: {prior_path}")
    for label, prior_key, central_key in (
        ("ids", "candidate_ids_tgt", "candidate_ids_src"),
        ("grid", "coords_tgt_grid", "coords_src_grid"),
        ("physical", "coords_tgt_physical", "coords_src_physical"),
    ):
        if not np.array_equal(prior_arrays[prior_key], central_arrays[central_key]):
            raise ValueError(f"overlapping window {label} mismatch: {prior_path} {path}")
    example["candidate_ids_prev"] = np.asarray(prior_arrays["candidate_ids_src"], dtype=np.int64)
    example["coords_prev_physical"] = np.asarray(
        prior_arrays["coords_src_physical"], dtype=np.float32
    )
    example["prior_cache_content_sha256"] = prior_metadata["array_content_sha256"]
    return example


class PastCandidateWindowDataset:
    def __init__(
        self,
        paths: list[Path],
        annotations: dict[str, AnnotationGraph],
        *,
        feature_channels: int,
        expected_primary_checkpoint_sha256: str,
        max_matching_distance_um: float,
        downsample_zyx: tuple[float, float, float],
        diagnostic_context: dict[str, Any] | None = None,
    ) -> None:
        self.paths = list(paths)
        self.annotations = annotations
        self.feature_channels = feature_channels
        self.expected_primary_checkpoint_sha256 = expected_primary_checkpoint_sha256
        self.max_matching_distance_um = max_matching_distance_um
        self.downsample_zyx = downsample_zyx
        self.diagnostic_context = diagnostic_context

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        example = build_past_candidate_window_example(
            path,
            self.annotations[path.parent.name],
            feature_channels=self.feature_channels,
            expected_primary_checkpoint_sha256=self.expected_primary_checkpoint_sha256,
            max_matching_distance_um=self.max_matching_distance_um,
            downsample_zyx=self.downsample_zyx,
        )
        if self.diagnostic_context is not None:
            example["diagnostic_context"] = past_annotation_context(
                example,
                self.annotations[path.parent.name],
                self.max_matching_distance_um,
                self.diagnostic_context,
            )
        return example


def past_annotation_context(
    example: dict[str, Any], annotation: AnnotationGraph, matching_um: float, cfg: dict[str, Any]
) -> dict[str, Any]:
    """GT-based reporting metadata only; never passed to the model or kNN selector."""
    frame = example["window_frames"][0]
    parents = np.full(len(example["coords_src_physical"]), -1, dtype=np.int64)
    if frame and frame - 1 in annotation.frames:
        past_gt, current_gt = (annotation.frames[frame - 1], annotation.frames[frame])
        previous_matches, _ = greedy_match_candidates(
            example["coords_prev_physical"], past_gt.node_ids, past_gt.coords_physical, matching_um
        )
        source_matches, _ = greedy_match_candidates(
            example["coords_src_physical"],
            current_gt.node_ids,
            current_gt.coords_physical,
            matching_um,
        )
        previous_map = {
            int(node): int(candidate)
            for node, candidate in zip(previous_matches, example["candidate_ids_prev"], strict=True)
            if node >= 0
        }
        source_map = {int(node): index for index, node in enumerate(source_matches) if node >= 0}
        outgoing = annotation.outgoing_edges or {}
        for parent, candidate_id in previous_map.items():
            for child in outgoing.get(parent, ()):
                if child in source_map:
                    parents[source_map[child]] = candidate_id
    return {"known_parent_candidate_ids": parents, "config": cfg}


def collate_past_candidate_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch = collate_window_examples(examples)
    batch_size = len(examples)
    max_prev = max(len(example["coords_prev_physical"]) for example in examples)
    coords_prev_physical = torch.zeros(batch_size, max_prev, 3, dtype=torch.float32)
    prev_mask = torch.zeros(batch_size, max_prev, dtype=torch.bool)
    candidate_ids_prev = torch.zeros(batch_size, max_prev, dtype=torch.int64)
    for index, example in enumerate(examples):
        n_prev = len(example["coords_prev_physical"])
        n_source = len(example["features_src"])
        coords_prev_physical[index, :n_prev] = torch.from_numpy(example["coords_prev_physical"])
        prev_mask[index, :n_prev] = True
        candidate_ids_prev[index, :n_prev] = torch.from_numpy(example["candidate_ids_prev"])
        batch["history_present_src"][index, :n_source] = n_prev > 0
        batch["metadata"][index]["previous_candidate_count"] = n_prev
        batch["metadata"][index]["candidate_ids_src"] = example["candidate_ids_src"]
        batch["metadata"][index]["candidate_ids_tgt"] = example["candidate_ids_tgt"]
        if "diagnostic_context" in example:
            batch["metadata"][index]["diagnostic_context"] = example["diagnostic_context"]
        batch["metadata"][index]["prior_cache_content_sha256"] = example[
            "prior_cache_content_sha256"
        ]
    batch["coords_prev_physical"] = coords_prev_physical
    batch["prev_mask"] = prev_mask
    batch["candidate_ids_prev"] = candidate_ids_prev
    return batch


# %% [markdown]
# ## Definitions: knn_diagnostics

# %%


def write_evidence(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"evidence": str(path), "sha256": file_sha256(path)}), flush=True)


def source_diagnostic_masks(batch: dict[str, Any], index: int, n_source: int) -> dict[str, Any]:
    metadata = batch["metadata"][index]
    context = metadata.get("diagnostic_context")
    if context is None:
        return {}
    cfg = context["config"]
    source = batch["coords_src_physical"][index : index + 1, :n_source]
    previous = batch["coords_prev_physical"][index : index + 1]
    ids = batch["candidate_ids_prev"][index : index + 1]
    valid = batch["prev_mask"][index : index + 1]
    _, selected_valid, selected = select_past_candidates(
        previous, source, ids, valid, int(cfg["k"])
    )
    distance2 = ((source[:, :, None] - previous[:, None]) ** 2).sum(-1)
    density = ((distance2 <= float(cfg["density_radius_um"]) ** 2) & valid[:, None]).sum(-1)[0]
    parents = torch.as_tensor(context["known_parent_candidate_ids"], device=source.device)
    retained = ((selected[0] == parents[:, None]) & selected_valid[0]).any(-1)
    low, high = cfg["density_upper_bounds"]
    return {
        "frame_start": torch.full(
            (n_source,), metadata["window_frames"][0] == 0, device=source.device
        ),
        "density_0": density <= low,
        "density_1": (density > low) & (density <= high),
        "density_2": density > high,
        "past_retained": (parents >= 0) & retained,
        "past_excluded": (parents >= 0) & ~retained,
        "past_unknown": parents < 0,
    }


class RuntimeBudget:
    def __init__(self, started: float, config: dict[str, Any], allocated_gpus: int) -> None:
        self.started = started
        remaining = config["quota_remaining_hours_at_push"]
        used = config["quota_used_hours_at_push"]
        if remaining is None or used is None or (not config["quota_checked_at"]):
            raise RuntimeError("fresh quota and user budget evidence are required")
        self.allocated_gpus = max(1, allocated_gpus)
        hours = min(float(remaining), float(config["weekly_gpu_budget_hours"]) - float(used))
        self.limit_seconds = min(
            float(config["runtime_gate_hours"]) * 3600, hours * 3600 / self.allocated_gpus
        )
        self.multiplier = float(config["runtime_projection_multiplier"])
        self.remaining_work = 0.0
        self.planned_done = 0.0
        self.observed_done = 0.0
        self.check()

    def check(self) -> None:
        elapsed = time.perf_counter() - self.started
        speed_factor = max(1.0, self.observed_done / max(self.planned_done, 1e-09))
        projected = elapsed + self.multiplier * self.remaining_work * speed_factor
        if not np.isfinite(projected) or projected >= self.limit_seconds:
            raise RuntimeError(
                f"runtime/user GPU budget gate: projected={projected:.1f}s limit={self.limit_seconds:.1f}s elapsed={elapsed:.1f}s"
            )

    def observe(self, elapsed: float, planned: float) -> None:
        self.observed_done += elapsed
        self.planned_done += planned
        self.remaining_work = max(0.0, self.remaining_work - planned)
        self.check()


class BudgetedLoader:
    """Account for reading, forward, backward and evaluation between yielded batches."""

    def __init__(self, loader: Any, budget: RuntimeBudget, per_window: float) -> None:
        self.loader = loader
        self.budget = budget
        self.per_window = per_window

    def __iter__(self) -> Any:
        self.budget.check()
        started = time.perf_counter()
        for batch in self.loader:
            yield batch
            self.budget.observe(
                time.perf_counter() - started, self.per_window * len(batch["metadata"])
            )
            started = time.perf_counter()


def candidate_inventory(paths: list[Path], k: int, budget: RuntimeBudget) -> list[dict[str, Any]]:
    records = {path: cache_identity_record(path) for path in paths}
    inventory = []
    for path, record in records.items():
        frame = record["window_frames"][0]
        prior = path.with_name(f"{frame - 1:06d}_{frame:06d}.npz")
        if frame and prior not in records:
            raise FileNotFoundError(prior)
        previous = records[prior]["candidate_count_src"] if frame else 0
        pairs = record["candidate_count_src"] * record["candidate_count_tgt"]
        inventory.append(
            {
                **record,
                "path": path.as_posix(),
                "previous_count": previous,
                "pair_count": pairs,
                "selected_triples": pairs * min(k, previous),
            }
        )
    budget.check()
    return inventory


def stratified_plan(
    rows: list[dict[str, Any]], strata: int, per_stratum: int, seed: int
) -> dict[str, Any]:
    if not rows or strata < 1 or per_stratum < 1:
        raise ValueError("nonempty rows and positive stratification settings required")
    values = np.asarray([row["selected_triples"] for row in rows], dtype=np.float64)
    boundaries = np.quantile(values, np.arange(1, strata) / strata)
    labels = np.searchsorted(boundaries, values, side="right")
    rng = np.random.default_rng(seed)
    sampled = {}
    for label in range(strata):
        indices = np.flatnonzero(labels == label)
        chosen = rng.choice(indices, min(per_stratum, len(indices)), replace=False)
        sampled[str(label)] = sorted(rows[int(index)]["path"] for index in chosen)
    return {"boundaries": boundaries.tolist(), "sampled_paths": sampled}


def stratum_counts(rows: list[dict[str, Any]], boundaries: list[float]) -> list[int]:
    labels = np.searchsorted(boundaries, [row["selected_triples"] for row in rows], side="right")
    return np.bincount(labels, minlength=len(boundaries) + 1).tolist()


def weighted_runtime(counts: list[int], rates: dict[str, float]) -> float:
    measured = [float(value) for value in rates.values()]
    if not measured or not all(np.isfinite(measured)):
        raise ValueError("finite measured rates are required")
    return sum(
        (count * float(rates.get(str(index), max(measured))) for index, count in enumerate(counts))
    )


def retention_window(
    previous: np.ndarray,
    source: np.ndarray,
    previous_ids: np.ndarray,
    annotation: AnnotationGraph,
    source_frame: int,
    selected_ids: np.ndarray,
    matching_um: float,
    cfg: dict[str, Any],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    counts: Counter[str] = Counter(
        known_edges=0,
        matched_edges=0,
        retained_edges=0,
        missing_endpoints=0,
        boundary_windows=0,
        missing_annotation_windows=0,
    )
    buckets: dict[str, Counter[str]] = {}
    if source_frame == 0:
        counts["boundary_windows"] += 1
        return (dict(counts), {})
    if source_frame - 1 not in annotation.frames or source_frame not in annotation.frames:
        counts["missing_annotation_windows"] += 1
        return (dict(counts), {})
    past_gt = annotation.frames[source_frame - 1]
    source_gt = annotation.frames[source_frame]
    past_matches, _ = greedy_match_candidates(
        previous, past_gt.node_ids, past_gt.coords_physical, matching_um
    )
    source_matches, _ = greedy_match_candidates(
        source, source_gt.node_ids, source_gt.coords_physical, matching_um
    )
    past_map = {int(node): i for i, node in enumerate(past_matches) if node >= 0}
    source_map = {int(node): i for i, node in enumerate(source_matches) if node >= 0}
    source_gt_ids = set(map(int, source_gt.node_ids))
    outgoing = annotation.outgoing_edges
    if outgoing is None:
        outgoing = {}
        for p, a in annotation.edges:
            outgoing.setdefault(p, []).append(a)
    for p in past_gt.node_ids:
        children = [int(a) for a in outgoing.get(int(p), ()) if int(a) in source_gt_ids]
        for a in children:
            counts["known_edges"] += 1
            if int(p) not in past_map or a not in source_map:
                counts["missing_endpoints"] += 1
                continue
            pi, ai = (past_map[int(p)], source_map[a])
            kept = int(previous_ids[pi] in selected_ids[ai])
            counts["matched_edges"] += 1
            counts["retained_edges"] += kept
            displacement = float(np.linalg.norm(source[ai] - previous[pi]))
            density = int(
                (np.linalg.norm(previous - source[ai], axis=1) <= cfg["density_radius_um"]).sum()
            )
            displacement_bin = np.searchsorted(
                cfg["displacement_upper_bounds_um"], displacement, side="left"
            )
            labels = (
                f"density_{np.searchsorted(cfg['density_upper_bounds'], density, side='left')}",
                f"displacement_{displacement_bin}",
                f"division_{int(len(children) > 1)}",
            )
            for label in labels:
                bucket = buckets.setdefault(label, Counter(matched_edges=0, retained_edges=0))
                bucket["matched_edges"] += 1
                bucket["retained_edges"] += kept
    return (dict(counts), {key: dict(value) for key, value in buckets.items()})


def audit_candidate_retention(
    paths: list[Path],
    annotations: dict[str, AnnotationGraph],
    splits: list[dict[str, Any]],
    config: dict[str, Any],
    device: Any,
    budget: RuntimeBudget,
) -> dict[str, Any]:
    model_cfg = config["model"]
    cfg = model_cfg["candidate_retention"]
    k = int(model_cfg["past_candidate_attention"]["max_past_candidates"])
    sample_fold = {
        sample: int(split["fold"]) for split in splits for sample in split["gradient_update"]
    }
    totals = {str(split["fold"]): Counter() for split in splits}
    buckets: dict[str, dict[str, Counter[str]]] = {key: {} for key in totals}
    digests = {key: hashlib.sha256() for key in totals}
    last_path = None
    last_arrays = None
    read_kwargs = {
        "feature_channels": int(config["data"]["cache"]["feature_channels"]),
        "expected_primary_checkpoint_sha256": model_cfg["public_source"]["checkpoint_sha256"],
    }
    processed = 0
    for path in sorted(paths):
        if path.parent.name not in sample_fold:
            continue
        arrays, metadata = validate_window_cache(path, **read_kwargs)
        frame = int(metadata["window_frames"][0])
        if frame:
            prior_path = path.with_name(f"{frame - 1:06d}_{frame:06d}.npz")
            prior = (
                last_arrays
                if prior_path == last_path
                else validate_window_cache(prior_path, **read_kwargs)[0]
            )
            for key in ("candidate_ids", "coords_grid", "coords_physical"):
                src_key = (
                    "candidate_ids_src"
                    if key == "candidate_ids"
                    else key.replace("coords_", "coords_src_")
                )
                tgt_key = src_key.replace("_src", "_tgt")
                if not np.array_equal(arrays[src_key], prior[tgt_key]):
                    raise ValueError(f"past/current identity mismatch: {path} {key}")
            previous = prior["coords_src_physical"]
            previous_ids = prior["candidate_ids_src"]
        else:
            previous = np.zeros((0, 3), dtype=np.float32)
            previous_ids = np.zeros(0, dtype=np.int64)
        source = arrays["coords_src_physical"]
        with torch.no_grad():
            _, _, selected = select_past_candidates(
                torch.as_tensor(previous, device=device)[None],
                torch.as_tensor(source, device=device)[None],
                torch.as_tensor(previous_ids, device=device)[None],
                torch.ones((1, len(previous)), dtype=torch.bool, device=device),
                k,
            )
        selected_ids = selected[0].cpu().numpy()
        fold = str(sample_fold[path.parent.name])
        counts, window_buckets = retention_window(
            previous,
            source,
            previous_ids,
            annotations[path.parent.name],
            frame,
            selected_ids,
            float(model_cfg["teacher"]["max_matching_distance_um"]),
            cfg,
        )
        totals[fold].update(counts)
        totals[fold]["window_count"] += 1
        for key, value in window_buckets.items():
            buckets[fold].setdefault(key, Counter()).update(value)
        digests[fold].update(
            json_sha256(
                {
                    "sample": path.parent.name,
                    "frame": frame,
                    "source_ids": arrays["candidate_ids_src"].tolist(),
                    "selected_ids": selected_ids.tolist(),
                }
            ).encode()
        )
        last_path, last_arrays = (path, arrays)
        processed += 1
        if processed % int(model_cfg["training"]["preparation_progress_interval"]) == 0:
            budget.check()
            print(json.dumps({"retention_windows": processed, "fold_counts": totals}), flush=True)
    folds = {}
    for fold, counts in totals.items():
        denominator = counts["matched_edges"]
        rate = counts["retained_edges"] / denominator if denominator else None
        folds[fold] = {
            **dict(counts),
            "conditional_retention": rate,
            "selected_candidate_content_sha256": digests[fold].hexdigest(),
            "buckets": {key: dict(value) for key, value in buckets[fold].items()},
            "passed": rate is not None and rate >= float(cfg["min_retention"]),
        }
    budget.check()
    return {"folds": folds, "passed": all(row["passed"] for row in folds.values()), "config": cfg}


# %% [markdown]
# ## Definitions: runtime_helpers

# %%

COMPETITION = "biohub-cell-tracking-during-development"


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
    wheel_dirs: list[Path], specs: tuple[str, ...], *, force_reinstall: bool
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
    if not failures and (not refresh_polars):
        return
    wheel_dirs = offline_wheel_dirs()
    if not wheel_dirs:
        raise FileNotFoundError("the public support dataset has no offline wheel directory")
    if refresh_polars:
        run_offline_install(wheel_dirs, ("polars>=1.36", "polars-runtime-32"), force_reinstall=True)
        purge_graph_modules(include_polars=True)
        importlib.invalidate_caches()
        if not polars_runtime_ready():
            raise ImportError("offline Polars refresh did not provide Float16 support")
    run_offline_install(wheel_dirs, OFFLINE_GRAPH_PACKAGE_SPECS, force_reinstall=False)
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


def resolve_cache_output_root(cache_cfg: dict[str, Any]) -> Path:
    kernel_slug = str(cache_cfg["kernel_source"]).split("/", 1)[-1]
    candidates = [
        path.parent
        for path in Path("/kaggle/input/notebooks").rglob(str(cache_cfg["summary_file"]))
        if kernel_slug in path.as_posix()
    ]
    return unique_existing(candidates, "exp015 cache output root")


def resolve_control_output_root(control_cfg: dict[str, Any]) -> Path:
    kernel_slug = str(control_cfg["kernel_source"]).split("/", 1)[-1]
    candidates = [
        path.parent
        for path in Path("/kaggle/input/notebooks").rglob(str(control_cfg["manifest_file"]))
        if kernel_slug in path.as_posix()
    ]
    return unique_existing(candidates, "exp016 control output root")


def resolve_public_artifact_root(public_cfg: dict[str, Any]) -> Path:
    slug = str(public_cfg["dataset_ref"]).split("/", 1)[-1]
    candidates = [
        Path(f"/kaggle/input/datasets/pilkwang/{slug}"),
        Path(f"/kaggle/input/{slug}"),
        Path(f"/kaggle/input/{slug}/{slug}"),
    ]
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        candidates.extend(
            path.parents[4]
            for path in input_root.rglob(str(public_cfg["model_source"]))
            if len(path.parents) >= 5
        )
    valid = [
        root
        for root in candidates
        if (root / str(public_cfg["train_script"])).is_file()
        and (root / str(public_cfg["model_source"])).is_file()
        and (root / str(public_cfg["checkpoint"])).is_file()
    ]
    return unique_existing(valid, "public tracker artifact root")


def resolve_train_dir() -> Path:
    return unique_existing(
        [
            Path(f"/kaggle/input/competitions/{COMPETITION}/train"),
            Path(f"/kaggle/input/{COMPETITION}/train"),
        ],
        "competition train directory",
    )


# %% [markdown]
# ## Definitions: settings

# %%

AUTOMATED_EXECUTION_STATUSES = {
    "planned",
    "running",
    "debug_completed",
    "scaffold_completed",
    "failed",
}


PRESERVED_REVIEW_STATUSES = {"usable", "completed", "deprecated", "discarded", "leak-risk"}


EXPERIMENT_STATUSES = AUTOMATED_EXECUTION_STATUSES | PRESERVED_REVIEW_STATUSES


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} must contain valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def update_metrics(path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    """Merge run-owned values into metrics.json without erasing other evidence."""
    current = read_json_object(path)
    safe_updates = dict(updates)
    if "status" in safe_updates and safe_updates["status"] not in EXPERIMENT_STATUSES:
        allowed = ", ".join(sorted(EXPERIMENT_STATUSES))
        raise ValueError(
            f"invalid experiment status {safe_updates['status']!r}; expected one of {allowed}"
        )
    if (
        current.get("status") in PRESERVED_REVIEW_STATUSES
        and safe_updates.get("status") in AUTOMATED_EXECUTION_STATUSES
    ):
        safe_updates.pop("status")
    metrics = deep_merge(current, safe_updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return metrics


# %% [markdown]
# ## Definitions: attention_train_pipeline

# %%

EXPERIMENT = "exp038_past_candidate_knn_attention"


WORKING_ROOT = Path.cwd()


CONFIG_PATH = WORKING_ROOT / "config.yaml"


METRICS_PATH = WORKING_ROOT / "metrics.json"


OUTPUT_MODELS = WORKING_ROOT / "models"


NOTEBOOK_STARTED = time.perf_counter()


# %%
@contextmanager
def recorded_stage(name):
    print({"stage": name}, flush=True)
    try:
        yield
    except Exception as error:
        update_metrics(
            METRICS_PATH,
            {
                "status": "failed",
                "evidence": {
                    "failure": {"stage": name, "type": type(error).__name__, "message": str(error)},
                    "kaggle": {"notebook_runtime_seconds": time.perf_counter() - NOTEBOOK_STARTED},
                },
                "notes": "Stopped at " + name + ": " + str(error),
            },
        )
        raise


# %% [markdown]
# ## Setup and configuration

# %%
with recorded_stage("Setup and configuration"):
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("The authoritative full run must execute on Kaggle.")

    ensure_geff_runtime_dependencies()

    import torch
    from torch.utils.data import DataLoader

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    validation_cfg = config["validation"]

    cache_cfg = config["data"]["cache"]

    model_cfg = config["model"]

    train_cfg = model_cfg["training"]

    attention_cfg = model_cfg["past_candidate_attention"]

    control_cfg = model_cfg["control"]

    teacher_cfg = model_cfg["teacher"]

    loss_cfg = model_cfg["loss"]

    params_cfg = model_cfg["params"]

    expected_trainable = ["primary_SimpleNodeTransformer", "past_candidate_attention_branch"]

    if model_cfg["trainable_components"] != expected_trainable:
        raise RuntimeError("past-candidate tracker trainable-component contract changed")

    if train_cfg["active_variants"] != ["past_candidate_knn_attention"]:
        raise RuntimeError("the train stage requires only past_candidate_knn_attention")

    if control_cfg["retrain"] is not False:
        raise RuntimeError("the exp016 control must not be retrained")

    if int(model_cfg["output"]["model_count"]) != 2:
        raise RuntimeError("the experiment requires exactly two outer-fold models")

    if int(attention_cfg["feature_dim"]) != len(PAST_CANDIDATE_FEATURE_NAMES):
        raise RuntimeError("past-candidate feature width differs from the declared schema")

    if bool(attention_cfg["clip_features"]):
        raise RuntimeError("the agreed feature contract does not clip displacements")

    budget = RuntimeBudget(NOTEBOOK_STARTED, train_cfg, torch.cuda.device_count())

    print(
        {"budget_limit_seconds": budget.limit_seconds, "allocated_gpus": budget.allocated_gpus},
        flush=True,
    )


# %% [markdown]
# ## Input cache and source integrity

# %%
with recorded_stage("Input cache and source integrity"):
    cache_output_root = resolve_cache_output_root(cache_cfg)

    cache_summary_path = cache_output_root / str(cache_cfg["summary_file"])

    cache_summary = validate_cache_summary(cache_summary_path, cache_cfg)

    cache_root = cache_output_root / str(cache_cfg["directory"])

    all_cache_paths = discover_cache_paths(cache_root, cache_cfg)

    observed_cache_identity_sha256 = recompute_cache_identity_sha256(all_cache_paths)

    if observed_cache_identity_sha256 != str(cache_cfg["identity_sha256"]):
        raise RuntimeError("exp015 cache identity differs from config")

    public_root = resolve_public_artifact_root(model_cfg["public_source"])

    train_dir = resolve_train_dir()

    control_root = resolve_control_output_root(control_cfg)

    control_manifest_path = control_root / str(control_cfg["manifest_file"])

    if file_sha256(control_manifest_path) != str(control_cfg["manifest_sha256"]):
        raise RuntimeError("exp016 model manifest SHA differs from config")

    control_manifest = json.loads(control_manifest_path.read_text(encoding="utf-8"))

    if control_manifest.get("experiment") != str(control_cfg["experiment"]):
        raise RuntimeError("exp016 model manifest identifies a different experiment")

    public_cfg = model_cfg["public_source"]

    source_paths = {
        "train_script": public_root / str(public_cfg["train_script"]),
        "model_source": public_root / str(public_cfg["model_source"]),
        "checkpoint": public_root / str(public_cfg["checkpoint"]),
    }

    expected_source_hashes = {
        "train_script": str(public_cfg["train_script_sha256"]),
        "model_source": str(public_cfg["model_source_sha256"]),
        "checkpoint": str(public_cfg["checkpoint_sha256"]),
    }

    observed_source_hashes = {name: file_sha256(path) for name, path in source_paths.items()}

    if observed_source_hashes != expected_source_hashes:
        raise RuntimeError("public tracker source or checkpoint SHA differs from config")


# %% [markdown]
# ## Annotation and fixed embryo splits

# %%
with recorded_stage("Annotation and fixed embryo splits"):
    sample_names = sorted({path.parent.name for path in all_cache_paths})

    observed_embryo_counts = {
        embryo: sum(name.startswith(f"{embryo}_") for name in sample_names)
        for embryo in config["data"]["expected_embryo_counts"]
    }

    if observed_embryo_counts != config["data"]["expected_embryo_counts"]:
        raise RuntimeError({"embryo_count_mismatch": observed_embryo_counts})

    missing_geff = [name for name in sample_names if not (train_dir / f"{name}.geff").is_dir()]

    if missing_geff:
        raise FileNotFoundError({"missing_train_geff": missing_geff[:20]})

    splits = build_embryo_splits(
        sample_names, validation_cfg["outer_folds"], int(validation_cfg["internal_split_seed"])
    )

    scale_zyx_um = tuple(
        float(value) for value in config["data"]["annotation"]["voxel_scale_zyx_um"]
    )

    annotations = {
        sample: load_annotation_graph(train_dir / f"{sample}.geff", scale_zyx_um)
        for sample in sample_names
    }

    eligible_cache_paths, gt_window_filter_audit = filter_nonempty_gt_window_paths(
        all_cache_paths, annotations
    )

    gt_window_filter_path = WORKING_ROOT / "gt_window_filter_audit.json"

    gt_window_filter_path.write_text(
        json.dumps(gt_window_filter_audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    gt_window_filter_audit_sha256 = file_sha256(gt_window_filter_path)

    gt_window_filter_summary = {
        key: value for key, value in gt_window_filter_audit.items() if key != "skipped_windows"
    }

    split_manifest = [
        {
            **record,
            "window_counts": {
                key: len(paths_for_samples(eligible_cache_paths, record[key]))
                for key in ("gradient_update", "internal_validation", "outer_evaluation")
            },
        }
        for record in splits
    ]

    split_manifest_path = WORKING_ROOT / "split_manifest.json"

    split_manifest_path.write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    annotation_manifest = {
        sample: annotation.content_sha256 for sample, annotation in sorted(annotations.items())
    }


# %% [markdown]
# ## Public tracker and device

# %%
with recorded_stage("Public tracker and device"):
    public_src = public_root / "repo" / "src"

    sys.path.insert(0, str(public_src))

    SimpleNodeTransformer = importlib.import_module("biohub_tracking.models").SimpleNodeTransformer

    full_public_state = torch.load(
        source_paths["checkpoint"], map_location="cpu", weights_only=True
    )

    public_tracker_state = extract_public_tracker_state(full_public_state)

    public_tracker_state_sha256 = canonical_state_sha256(public_tracker_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        raise RuntimeError("the authoritative train run requires a Kaggle GPU")

    OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Models, loaders and checkpoint selection

# %%
with recorded_stage("Models, loaders and checkpoint selection"):
    feature_channels = int(cache_cfg["feature_channels"])

    max_matching_distance_um = float(teacher_cfg["max_matching_distance_um"])

    downsample_zyx = tuple(float(value) for value in params_cfg["downsample_zyx"])

    batch_size = int(train_cfg["batch_size"])

    num_workers = int(train_cfg["num_workers"])

    global_seed = int(train_cfg["seed"])

    use_amp = bool(train_cfg["mixed_precision"])

    gamma = float(loss_cfg["focal_gamma"])

    threshold = float(validation_cfg["graph_progression_gate"]["threshold"])

    def new_base_tracker() -> Any:
        tracker = SimpleNodeTransformer(
            feat_dim=int(params_cfg["feature_dim"]),
            hidden_dim=int(params_cfg["hidden_dim"]),
            n_heads=int(params_cfg["n_heads"]),
            n_blocks=int(params_cfg["n_blocks"]),
            dropout=float(params_cfg["dropout"]),
            pair_chunk_size=int(params_cfg["pair_chunk_size"]),
        )
        tracker.load_state_dict(copy.deepcopy(public_tracker_state), strict=True)
        return tracker.to(device)

    def new_attention_tracker() -> Any:
        base_tracker = new_base_tracker()
        base_parameter_count = sum(parameter.numel() for parameter in base_tracker.parameters())
        tracker = PastCandidateAttentionTracker(
            base_tracker,
            max_past_candidates=int(attention_cfg["max_past_candidates"]),
            feature_dim=int(attention_cfg["feature_dim"]),
            hidden_dim=int(attention_cfg["hidden_dim"]),
            source_chunk_size=int(attention_cfg["source_chunk_size"]),
            target_chunk_size=int(attention_cfg["target_chunk_size"]),
            past_candidate_chunk_size=int(attention_cfg["past_candidate_chunk_size"]),
            vector_scale_um=float(attention_cfg["vector_scale_um"]),
            cosine_epsilon=float(attention_cfg["cosine_epsilon"]),
            gradient_checkpointing=bool(attention_cfg["gradient_checkpointing"]),
        ).to(device)
        added_parameter_count = (
            sum(parameter.numel() for parameter in tracker.parameters()) - base_parameter_count
        )
        if added_parameter_count != int(attention_cfg["expected_added_parameter_count"]):
            raise RuntimeError({"past_candidate_parameter_count": added_parameter_count})
        return tracker

    control_records = {int(item["fold"]): item for item in control_cfg["models"]}

    manifest_control_records = {int(item["fold"]): item for item in control_manifest["models"]}

    if set(control_records) != {0, 1} or set(manifest_control_records) != {0, 1}:
        raise RuntimeError("exp016 control manifest must contain folds 0 and 1")

    def load_control_tracker(fold: int) -> Any:
        expected = control_records[fold]
        observed = manifest_control_records[fold]
        for key in ("fold", "evaluation_embryo", "path", "file_sha256", "canonical_state_sha256"):
            if observed.get(key) != expected.get(key):
                raise RuntimeError({"exp016_manifest_mismatch": {"fold": fold, "field": key}})
        model_path = control_root / str(expected["path"])
        if file_sha256(model_path) != str(expected["file_sha256"]):
            raise RuntimeError(f"exp016 fold {fold} model file SHA differs from config")
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        state = payload["state_dict"]
        if canonical_state_sha256(state) != str(expected["canonical_state_sha256"]):
            raise RuntimeError(f"exp016 fold {fold} state SHA differs from config")
        tracker = new_base_tracker()
        tracker.load_state_dict(state, strict=True)
        return tracker

    def make_loader(
        paths: list[Path], *, shuffle: bool, seed: int, diagnostics: bool = False
    ) -> Any:
        dataset = PastCandidateWindowDataset(
            paths,
            annotations,
            feature_channels=feature_channels,
            expected_primary_checkpoint_sha256=str(public_cfg["checkpoint_sha256"]),
            max_matching_distance_um=max_matching_distance_um,
            downsample_zyx=downsample_zyx,
            diagnostic_context={
                **model_cfg["candidate_retention"],
                "k": int(attention_cfg["max_past_candidates"]),
            }
            if diagnostics
            else None,
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_past_candidate_examples,
            generator=generator,
            worker_init_fn=dataloader_worker_init,
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None,
            pin_memory=True,
        )

    def new_optimizer(model: Any) -> Any:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        model_ids = {id(parameter) for parameter in model.parameters()}
        optimizer_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        if optimizer_ids != model_ids:
            raise RuntimeError("optimizer parameters differ from model parameters")
        return optimizer

    def new_scaler() -> Any:
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            return torch.cuda.amp.GradScaler(enabled=use_amp)

    def fit_model(
        model: Any, *, train_paths: list[Path], validation_paths: list[Path], seed: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        optimizer = new_optimizer(model)
        scaler = new_scaler()
        train_loader = BudgetedLoader(
            make_loader(train_paths, shuffle=True, seed=seed), budget, phase_rates["train"]
        )
        validation_loader = BudgetedLoader(
            make_loader(validation_paths, shuffle=False, seed=seed),
            budget,
            phase_rates["internal_validation"],
        )
        best_score = float("-inf")
        best_epoch = -1
        best_state: dict[str, Any] | None = None
        epoch_records: list[dict[str, Any]] = []
        for epoch in range(int(train_cfg["epochs"])):
            epoch_started = time.perf_counter()
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                gamma=gamma,
                gradient_clip_norm=float(train_cfg["gradient_clip_norm"]),
                use_amp=use_amp,
            )
            validation_metrics = evaluate_tracker(
                model, validation_loader, device, gamma=gamma, use_amp=use_amp
            )
            score = float(validation_metrics["selection_score"])
            selected = score >= best_score
            if selected:
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            record = {
                "epoch": epoch,
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "train": train_metrics,
                "internal_validation": validation_metrics,
                "selected": selected,
            }
            epoch_records.append(record)
            print(json.dumps(record, default=float), flush=True)
            torch.save(
                {"state_dict": best_state, "epoch_records": epoch_records},
                OUTPUT_MODELS / f"partial_fold_{seed - global_seed}.pth",
            )
            budget.check()
        if best_state is None:
            raise RuntimeError("model training did not select a checkpoint")
        model.load_state_dict(best_state, strict=True)
        del validation_loader, train_loader, scaler, optimizer
        return (
            best_state,
            {
                "best_epoch": best_epoch,
                "best_internal_selection_score": best_score,
                "epochs": epoch_records,
            },
        )


# %% [markdown]
# ## Candidate counts and retention audit

# %%
with recorded_stage("Candidate counts and retention audit"):
    inventory = candidate_inventory(
        all_cache_paths, int(attention_cfg["max_past_candidates"]), budget
    )

    write_evidence(WORKING_ROOT / "candidate_inventory.json", inventory)

    retention = audit_candidate_retention(
        eligible_cache_paths, annotations, splits, config, device, budget
    )

    write_evidence(WORKING_ROOT / "candidate_retention.json", retention)

    update_metrics(METRICS_PATH, {"evidence": {"candidate_retention": retention}})

    if not retention["passed"]:
        update_metrics(
            METRICS_PATH,
            {"status": "failed", "notes": "Candidate retention gate failed before training."},
        )
        raise RuntimeError("K=8 retention is below the fixed learning-side 99% gate")


# %% [markdown]
# ## Stratified runtime and stress benchmark

# %%
with recorded_stage("Stratified runtime and stress benchmark"):
    inventory_by_path = {row["path"]: row for row in inventory}

    eligible_rows = [inventory_by_path[path.as_posix()] for path in eligible_cache_paths]

    benchmark_seed = global_seed + int(train_cfg["benchmark_seed_offset"])

    seed_everything(benchmark_seed)

    benchmark_model = new_attention_tracker()

    benchmark_optimizer = new_optimizer(benchmark_model)

    benchmark_scaler = new_scaler()

    profiles = {}

    raw_projected_seconds = 0.0

    serialization_seconds = 0.0

    peak_gpu_memory_bytes = 0

    runtime_gate_seconds = budget.limit_seconds

    def timed_pass(
        model: Any,
        paths: list[Path],
        mode: str,
        output: Path,
        optimizer: Any = benchmark_optimizer,
        scaler: Any = benchmark_scaler,
    ) -> float:
        budget.check()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        loader = make_loader(
            paths,
            shuffle=False,
            seed=benchmark_seed,
            diagnostics=mode in ("attention_outer", "control_outer"),
        )
        if mode == "train":
            train_one_epoch(
                model,
                loader,
                optimizer,
                scaler,
                device,
                gamma=gamma,
                gradient_clip_norm=float(train_cfg["gradient_clip_norm"]),
                use_amp=use_amp,
            )
        elif mode == "internal_validation":
            evaluate_tracker(model, loader, device, gamma=gamma, use_amp=use_amp)
        else:
            evaluate_tracker_diagnostic(
                model,
                loader,
                device,
                gamma=gamma,
                use_amp=use_amp,
                edge_probability_threshold=threshold,
                prediction_dir=output,
            )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        budget.check()
        return elapsed

    for split in splits:
        fold = int(split["fold"])
        learning_paths = paths_for_samples(eligible_cache_paths, split["gradient_update"])
        learning_rows = [inventory_by_path[path.as_posix()] for path in learning_paths]
        plan = stratified_plan(
            learning_rows,
            int(train_cfg["benchmark_strata"]),
            int(train_cfg["benchmark_windows_per_stratum"]),
            global_seed,
        )
        control_benchmark = load_control_tracker(fold)
        warmup = learning_paths[: int(train_cfg["benchmark_warmup_windows"])]
        timed_pass(benchmark_model, warmup, "train", WORKING_ROOT / "benchmark_saved")
        torch.cuda.reset_peak_memory_stats(device)
        rates = {
            key: {} for key in ("train", "internal_validation", "attention_outer", "control_outer")
        }
        for stratum, chosen in plan["sampled_paths"].items():
            if not chosen:
                continue
            selected = [Path(path) for path in chosen]
            for mode in rates:
                model = control_benchmark if mode == "control_outer" else benchmark_model
                rates[mode][stratum] = timed_pass(
                    model,
                    selected,
                    mode,
                    WORKING_ROOT / "benchmark_saved" / f"fold_{fold}" / mode / stratum,
                ) / len(selected)
            print({"benchmark_fold": fold, "stratum": stratum, "rates": rates}, flush=True)
        projected = {}
        counts = {}
        mean_rates = {}
        for mode, split_key, repeats in (
            ("train", "gradient_update", int(train_cfg["epochs"])),
            ("internal_validation", "internal_validation", int(train_cfg["epochs"])),
            ("attention_outer", "outer_evaluation", 1),
            ("control_outer", "outer_evaluation", 1),
        ):
            rows = [
                inventory_by_path[path.as_posix()]
                for path in paths_for_samples(eligible_cache_paths, split[split_key])
            ]
            counts[mode] = stratum_counts(rows, plan["boundaries"])
            single_pass = weighted_runtime(counts[mode], rates[mode])
            projected[mode] = repeats * single_pass
            mean_rates[mode] = single_pass / len(rows)
        save_started = time.perf_counter()
        torch.save(
            benchmark_model.state_dict(), WORKING_ROOT / "benchmark_saved" / f"fold_{fold}.pth"
        )
        save_projection = (time.perf_counter() - save_started) * (int(train_cfg["epochs"]) + 1)
        serialization_seconds += save_projection
        raw_projected_seconds += sum(projected.values()) + save_projection
        profiles[str(fold)] = {
            "plan": plan,
            "seconds_per_window": rates,
            "window_counts": counts,
            "projected_seconds": projected,
            "mean_rates": mean_rates,
            "model_save_projected_seconds": save_projection,
        }
        peak_gpu_memory_bytes = max(
            peak_gpu_memory_bytes, int(torch.cuda.max_memory_allocated(device))
        )
        del control_benchmark
        torch.cuda.empty_cache()

    stress_paths = list(
        dict.fromkeys(
            max(eligible_rows, key=lambda row: (row[key], row["path"]))["path"]
            for key in (
                "pair_count",
                "selected_triples",
                "candidate_count_src",
                "candidate_count_tgt",
            )
        )
    )

    stress_seconds = 0.0

    for path in stress_paths:
        stress_seconds += timed_pass(
            benchmark_model, [Path(path)] * batch_size, "train", WORKING_ROOT / "benchmark_saved"
        )

    padded_stress_paths = [
        Path(max(eligible_rows, key=lambda row: row[key])["path"])
        for key in ("candidate_count_src", "candidate_count_tgt")
    ]

    stress_seconds += timed_pass(
        benchmark_model, padded_stress_paths, "train", WORKING_ROOT / "benchmark_saved"
    )

    peak_gpu_memory_bytes = max(peak_gpu_memory_bytes, int(torch.cuda.max_memory_allocated(device)))

    memory_limit = int(
        torch.cuda.get_device_properties(device).total_memory
        * float(train_cfg["max_gpu_memory_fraction"])
    )

    projected_seconds = (
        time.perf_counter() - NOTEBOOK_STARTED + raw_projected_seconds * budget.multiplier
    )

    benchmark_summary = {
        "folds": profiles,
        "stress_paths": stress_paths,
        "mixed_maximum_padding_stress_paths": [path.as_posix() for path in padded_stress_paths],
        "stress_elapsed_seconds": stress_seconds,
        "max_candidate_product": max(row["selected_triples"] for row in eligible_rows),
        "projected_seconds_before_multiplier": raw_projected_seconds,
        "projection_multiplier": budget.multiplier,
        "conservative_projected_seconds": projected_seconds,
        "runtime_gate_seconds": runtime_gate_seconds,
        "elapsed_preparation_seconds": time.perf_counter() - NOTEBOOK_STARTED,
        "model_save_projected_seconds": serialization_seconds,
        "allocated_gpu_count": budget.allocated_gpus,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "memory_limit_bytes": memory_limit,
        "passed": projected_seconds < runtime_gate_seconds
        and peak_gpu_memory_bytes <= memory_limit,
    }

    write_evidence(WORKING_ROOT / "runtime_benchmark.json", benchmark_summary)

    update_metrics(METRICS_PATH, {"evidence": {"benchmark": benchmark_summary}})

    del timed_pass, benchmark_model, benchmark_optimizer, benchmark_scaler

    torch.cuda.empty_cache()

    if not benchmark_summary["passed"]:
        update_metrics(METRICS_PATH, {"status": "failed", "notes": "Runtime/memory gate failed."})
        raise RuntimeError("runtime or memory gate failed before full two-fold training")

    budget.remaining_work = raw_projected_seconds

    budget.check()


# %% [markdown]
# ## Two-fold training and saved-control comparison

# %%
with recorded_stage("Two-fold training and saved-control comparison"):
    training_started = time.perf_counter()

    fold_summaries: list[dict[str, Any]] = []

    model_records: list[dict[str, Any]] = []

    teacher_audit_records: list[dict[str, Any]] = []

    for split in splits:
        fold = int(split["fold"])
        fold_seed = global_seed + fold
        phase_rates = profiles[str(fold)]["mean_rates"]
        train_paths = paths_for_samples(eligible_cache_paths, split["gradient_update"])
        validation_paths = paths_for_samples(eligible_cache_paths, split["internal_validation"])
        evaluation_paths = paths_for_samples(eligible_cache_paths, split["outer_evaluation"])
        evaluation_loader = make_loader(
            evaluation_paths, shuffle=False, seed=fold_seed, diagnostics=True
        )
        control_tracker = load_control_tracker(fold)
        baseline_outer = evaluate_tracker_diagnostic(
            control_tracker,
            BudgetedLoader(evaluation_loader, budget, phase_rates["control_outer"]),
            device,
            gamma=gamma,
            use_amp=use_amp,
            edge_probability_threshold=threshold,
            prediction_dir=WORKING_ROOT / "pair_predictions" / f"fold_{fold}" / "exp016",
        )
        teacher_audit_records.append(baseline_outer["teacher"])
        seed_everything(fold_seed)
        attention_tracker = new_attention_tracker()
        attention_state, attention_fit = fit_model(
            attention_tracker,
            train_paths=train_paths,
            validation_paths=validation_paths,
            seed=fold_seed,
        )
        attention_outer = evaluate_tracker_diagnostic(
            attention_tracker,
            BudgetedLoader(evaluation_loader, budget, phase_rates["attention_outer"]),
            device,
            gamma=gamma,
            use_amp=use_amp,
            edge_probability_threshold=threshold,
            prediction_dir=WORKING_ROOT / "pair_predictions" / f"fold_{fold}" / "knn_attention",
        )
        fold_dir = OUTPUT_MODELS / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model_path = fold_dir / "past_candidate_attention_tracker_best.pth"
        torch.save(
            {
                "experiment": EXPERIMENT,
                "fold": fold,
                "train_embryo": split["train_embryo"],
                "evaluation_embryo": split["evaluation_embryo"],
                "state_dict": attention_state,
                **attention_fit,
            },
            model_path,
        )
        base_fp = int(baseline_outer["false_positive_active_pair_count"])
        attention_fp = int(attention_outer["false_positive_active_pair_count"])
        if base_fp == 0:
            fp_relative_increase = 0.0 if attention_fp == 0 else None
            fp_ok = attention_fp == 0
        else:
            fp_relative_increase = (attention_fp - base_fp) / base_fp
            fp_ok = fp_relative_increase <= float(
                validation_cfg["graph_progression_gate"][
                    "max_false_positive_active_pair_relative_increase"
                ]
            )
        gate_checks = {
            "valid_denominators": all(
                int(row[key]) > 0
                for row in (baseline_outer, attention_outer)
                for key in (
                    "positive_edge_count",
                    "division_parent_count",
                    "active_negative_pair_count",
                )
            ),
            "positive_edge_recall_improved": float(attention_outer["positive_edge_recall"])
            > float(baseline_outer["positive_edge_recall"]),
            "false_positive_active_pair_increase_within_limit": fp_ok,
            "division_recovered_count_non_decrease": int(
                attention_outer["recovered_division_parent_count"]
            )
            >= int(baseline_outer["recovered_division_parent_count"]),
        }
        gate_checks["passed"] = all(gate_checks.values())
        comparison_keys = (
            "legacy_mask_loss",
            "edge_accuracy",
            "positive_edge_recall",
            "false_positive_active_pair_rate",
            "correct_parent_top1_accuracy",
            "division_parent_recall",
            "selection_score",
        )
        fold_summary = {
            "fold": fold,
            "train_embryo": split["train_embryo"],
            "evaluation_embryo": split["evaluation_embryo"],
            "sample_counts": {
                key: len(split[key])
                for key in ("gradient_update", "internal_validation", "outer_evaluation")
            },
            "window_counts": {
                "gradient_update": len(train_paths),
                "internal_validation": len(validation_paths),
                "outer_evaluation": len(evaluation_paths),
            },
            "exp016_saved_control": baseline_outer,
            "attention_fit": attention_fit,
            "attention_outer_evaluation": attention_outer,
            "outer_metric_delta": {
                key: float(attention_outer[key]) - float(baseline_outer[key])
                for key in comparison_keys
            },
            "false_positive_active_pair_relative_increase": fp_relative_increase,
            "graph_progression_gate": gate_checks,
            "model_file": model_path.relative_to(WORKING_ROOT).as_posix(),
            "model_file_sha256": file_sha256(model_path),
            "model_state_sha256": canonical_state_sha256(attention_state),
        }
        fold_summary_path = WORKING_ROOT / f"fold_{fold}_summary.json"
        fold_summary_path.write_text(
            json.dumps(fold_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        fold_summaries.append(fold_summary)
        model_records.append(
            {
                "fold": fold,
                "path": fold_summary["model_file"],
                "file_sha256": fold_summary["model_file_sha256"],
                "canonical_state_sha256": fold_summary["model_state_sha256"],
                "best_epoch": attention_fit["best_epoch"],
                "train_embryo": split["train_embryo"],
                "evaluation_embryo": split["evaluation_embryo"],
            }
        )
        write_evidence(fold_summary_path, fold_summary)
        del evaluation_loader, attention_tracker, control_tracker
        torch.cuda.empty_cache()

    if len(model_records) != int(model_cfg["output"]["model_count"]):
        raise RuntimeError("trained model count differs from config")


# %% [markdown]
# ## Graph progression diagnostic

# %%
with recorded_stage("Graph progression diagnostic"):
    graph_gate_passed = all(
        bool(fold["graph_progression_gate"]["passed"]) for fold in fold_summaries
    )

    graph_gate = {
        "passed": graph_gate_passed,
        "threshold": threshold,
        "required_fold_count": 2,
        "passed_fold_count": sum(
            bool(fold["graph_progression_gate"]["passed"]) for fold in fold_summaries
        ),
        "folds": {str(fold["fold"]): fold["graph_progression_gate"] for fold in fold_summaries},
        "next_action": "request_user_approval_for_full_graph_inference"
        if graph_gate_passed
        else "stop_before_full_graph_inference",
        "official_metric_computed": False,
    }

    graph_gate_path = WORKING_ROOT / "graph_progression_gate.json"

    graph_gate_path.write_text(
        json.dumps(graph_gate, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    teacher_audit = aggregate_teacher_stats(teacher_audit_records)

    teacher_audit.update(
        {
            "matching": teacher_cfg["matching"],
            "max_matching_distance_um": max_matching_distance_um,
            "loss_mask": loss_cfg["mask"],
            "outer_evaluation_coverage": "each of 199 samples exactly once across two folds",
        }
    )

    teacher_audit_path = WORKING_ROOT / "teacher_audit.json"

    teacher_audit_path.write_text(
        json.dumps(teacher_audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# %% [markdown]
# ## Model manifest and experiment metrics

# %%
with recorded_stage("Model manifest and experiment metrics"):
    feature_schema = {
        "cache_schema_version": cache_cfg["schema_version"],
        "arrays": cache_cfg["arrays"],
        "primary_feature_channels": cache_cfg["feature_channels"],
        "tracker_feature_dim": params_cfg["feature_dim"],
        "past_candidate_features": list(PAST_CANDIDATE_FEATURE_NAMES),
        "past_candidate_attention": attention_cfg,
    }

    model_manifest = {
        "experiment": EXPERIMENT,
        "created_at": datetime.now(UTC).isoformat(),
        "model_class": "PastCandidateAttentionTracker",
        "trainable_components": model_cfg["trainable_components"],
        "frozen_components": model_cfg["frozen_components"],
        "public_checkpoint_file_sha256": observed_source_hashes["checkpoint"],
        "public_tracker_initial_state_sha256": public_tracker_state_sha256,
        "exp016_control_manifest_sha256": file_sha256(control_manifest_path),
        "cache_summary_sha256": cache_summary["summary_sha256"],
        "cache_identity_sha256": cache_summary["cache_identity_sha256"],
        "gt_window_filter_audit_sha256": gt_window_filter_audit_sha256,
        "annotation_content_sha256": json_sha256(annotation_manifest),
        "feature_schema_contract_sha256": json_sha256(feature_schema),
        "models": model_records,
    }

    model_manifest_path = WORKING_ROOT / str(model_cfg["output"]["model_manifest"])

    model_manifest_path.write_text(
        json.dumps(model_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    model_manifest_sha = file_sha256(model_manifest_path)

    training_summary = {
        "experiment": EXPERIMENT,
        "status": "debug_completed",
        "official_metric_computed": False,
        "submission_created": False,
        "conditional_validation": validation_cfg["conditional_validation"],
        "elapsed_seconds": time.perf_counter() - training_started,
        "notebook_elapsed_seconds": time.perf_counter() - NOTEBOOK_STARTED,
        "benchmark": benchmark_summary,
        "candidate_retention": retention,
        "gt_window_filter": {
            **gt_window_filter_summary,
            "audit_sha256": gt_window_filter_audit_sha256,
        },
        "teacher_audit": teacher_audit,
        "folds": fold_summaries,
        "graph_progression_gate": graph_gate,
        "model_manifest_sha256": model_manifest_sha,
    }

    training_summary_path = WORKING_ROOT / str(model_cfg["output"]["training_summary"])

    training_summary_path.write_text(
        json.dumps(training_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    update_metrics(
        METRICS_PATH,
        {
            "status": "debug_completed",
            "metric": "fixed_candidate_past_candidate_attention_pair_diagnostics",
            "evidence": {
                "kaggle": {
                    "kernel_source_ids": [
                        str(cache_cfg["kernel_source"]),
                        str(control_cfg["kernel_source"]),
                        str(public_cfg["dataset_ref"]),
                    ],
                    "resource": torch.cuda.get_device_name(device),
                    "notebook_runtime_seconds": training_summary["notebook_elapsed_seconds"],
                    "internet_enabled": False,
                },
                "artifacts": {
                    "input_file_sha": model_manifest["annotation_content_sha256"],
                    "cache_file_sha": cache_summary["summary_sha256"],
                    "feature_schema_sha": model_manifest["feature_schema_contract_sha256"],
                    "feature_content_sha": cache_summary["cache_identity_sha256"],
                    "gt_window_filter_audit_sha": gt_window_filter_audit_sha256,
                    "model_manifest_sha": model_manifest_sha,
                    "candidate_retention_sha": file_sha256(
                        WORKING_ROOT / "candidate_retention.json"
                    ),
                    "runtime_benchmark_sha": file_sha256(WORKING_ROOT / "runtime_benchmark.json"),
                    "oof_prediction_sha": json_sha256(
                        {
                            str(f["fold"]): f["attention_outer_evaluation"][
                                "prediction_content_sha256"
                            ]
                            for f in fold_summaries
                        }
                    ),
                    "row_count": len(eligible_cache_paths),
                    "group_count": len(sample_names),
                    "feature_count": int(params_cfg["feature_dim"])
                    + int(attention_cfg["feature_dim"]),
                    "model_count": len(model_records),
                    "model_shas": {
                        str(record["fold"]): record["file_sha256"] for record in model_records
                    },
                },
            },
            "train_stage": training_summary,
            "notes": "Past-candidate attention pair diagnostic completed against saved exp016 fold models. "
            + (
                "The pair gate passed; full graph inference requires user approval."
                if graph_gate_passed
                else "The pair gate failed; full graph inference was not started."
            ),
        },
    )

    print(json.dumps(training_summary, indent=2, default=float))

    print("Model manifest:", model_manifest_path, model_manifest_sha)

    print("Graph progression gate:", json.dumps(graph_gate, indent=2))

    print("No submission was created.")
