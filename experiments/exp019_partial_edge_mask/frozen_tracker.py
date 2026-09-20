from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CACHE_SCHEMA_VERSION = 1
CACHE_METADATA_KEY = "__metadata_json__"
REQUIRED_CACHE_ARRAYS = (
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
    ).encode("utf-8")
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


def embryo_id(sample_name: str) -> str:
    embryo, separator, _ = sample_name.partition("_")
    if not separator or not embryo:
        raise ValueError(f"sample name has no embryo prefix: {sample_name}")
    return embryo


def build_embryo_splits(
    sample_names: list[str],
    outer_specs: list[dict[str, Any]],
    split_seed: int,
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
    diagnostics: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if diagnostics is not None:
        diagnostics["duplicate_near_gt_candidate_count"] = 0
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
            if diagnostics is not None:
                diagnostics["duplicate_near_gt_candidate_count"] += 1
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


def partial_edge_pair_mask(target: np.ndarray) -> np.ndarray:
    matrix = np.asarray(target)
    if matrix.ndim != 2:
        raise ValueError("target must be a matrix")
    return np.broadcast_to((matrix.sum(axis=0) > 0)[None, :], matrix.shape)


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


def partial_edge_focal_bce(logits: Any, target: Any, gamma: float = 2.0) -> Any:
    import torch
    import torch.nn.functional as functional

    if logits.shape != target.shape or logits.ndim != 2:
        raise ValueError("logits and target must be matching pair matrices")
    active_cols = target.sum(dim=0) > 0
    if not active_cols.any():
        return logits.sum() * 0.0
    probabilities = torch.softmax(logits, dim=0)
    bce = functional.binary_cross_entropy(probabilities, target, reduction="none")
    p_t = probabilities * target + (1 - probabilities) * (1 - target)
    loss = ((1 - p_t) ** gamma) * bce
    return loss[:, active_cols].mean()


def batch_partial_edge_focal_bce(
    logits: Any,
    target: Any,
    source_mask: Any,
    target_mask: Any,
    gamma: float = 2.0,
) -> Any:
    import torch

    losses = []
    for batch_index in range(logits.shape[0]):
        n_source = int(source_mask[batch_index].sum().item())
        n_target = int(target_mask[batch_index].sum().item())
        losses.append(
            partial_edge_focal_bce(
                logits[batch_index, :n_source, :n_target],
                target[batch_index, :n_source, :n_target],
                gamma=gamma,
            )
        )
    return torch.stack(losses).mean()


def batch_legacy_focal_bce(
    logits: Any,
    target: Any,
    source_mask: Any,
    target_mask: Any,
    gamma: float = 2.0,
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
    source_match_diagnostics: dict[str, int] = {}
    target_match_diagnostics: dict[str, int] = {}
    source_matches, source_distances = greedy_match_candidates(
        arrays["coords_src_physical"],
        source_gt.node_ids,
        source_gt.coords_physical,
        max_matching_distance_um,
        diagnostics=source_match_diagnostics,
    )
    target_matches, target_distances = greedy_match_candidates(
        arrays["coords_tgt_physical"],
        target_gt.node_ids,
        target_gt.coords_physical,
        max_matching_distance_um,
        diagnostics=target_match_diagnostics,
    )
    target = build_legacy_edge_target(
        source_matches,
        target_matches,
        annotation.edges,
        outgoing_edges=annotation.outgoing_edges,
    )
    active_mask = legacy_active_pair_mask(target)
    partial_mask = partial_edge_pair_mask(target)
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
            "partial_mask_pairs": int(partial_mask.sum()),
            "partial_mask_pairs_with_unknown_source": int(
                (partial_mask & (source_matches < 0)[:, None]).sum()
            ),
            "removed_legacy_pairs": int((active_mask & ~partial_mask).sum()),
            "duplicate_near_gt_candidates": (
                source_match_diagnostics["duplicate_near_gt_candidate_count"]
                + target_match_diagnostics["duplicate_near_gt_candidate_count"]
            ),
            "division_parents": int(np.count_nonzero(target.sum(axis=1) > 1)),
            "source_match_distance_sum_um": float(np.nansum(source_distances)),
            "source_match_distance_count": int(np.count_nonzero(np.isfinite(source_distances))),
            "target_match_distance_sum_um": float(np.nansum(target_distances)),
            "target_match_distance_count": int(np.count_nonzero(np.isfinite(target_distances))),
        },
    }
    return example


class FrozenFeatureWindowDataset:
    def __init__(
        self,
        paths: list[Path],
        annotations: dict[str, AnnotationGraph],
        *,
        feature_channels: int,
        expected_primary_checkpoint_sha256: str,
        max_matching_distance_um: float,
        downsample_zyx: tuple[float, float, float],
    ) -> None:
        self.paths = list(paths)
        self.annotations = annotations
        self.feature_channels = feature_channels
        self.expected_primary_checkpoint_sha256 = expected_primary_checkpoint_sha256
        self.max_matching_distance_um = max_matching_distance_um
        self.downsample_zyx = downsample_zyx

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        return build_window_example(
            path,
            self.annotations[path.parent.name],
            feature_channels=self.feature_channels,
            expected_primary_checkpoint_sha256=self.expected_primary_checkpoint_sha256,
            max_matching_distance_um=self.max_matching_distance_um,
            downsample_zyx=self.downsample_zyx,
        )


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


def paths_for_samples(paths: list[Path], samples: list[str]) -> list[Path]:
    allowed = set(samples)
    return [path for path in paths if path.parent.name in allowed]


def filter_nonempty_gt_window_paths(
    paths: list[Path],
    annotations: dict[str, AnnotationGraph],
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
    return eligible, audit


def aggregate_teacher_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    integer_keys = (
        "candidate_nodes",
        "matched_candidate_nodes",
        "gt_nodes",
        "positive_edges",
        "active_pairs",
        "active_pairs_with_unknown_endpoint",
        "partial_mask_pairs",
        "partial_mask_pairs_with_unknown_source",
        "removed_legacy_pairs",
        "duplicate_near_gt_candidates",
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
    result["removed_legacy_pair_fraction"] = safe_ratio(
        result["removed_legacy_pairs"], result["active_pairs"]
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
    return model(
        batch["features_src"],
        batch["features_tgt"],
        batch["coords_src"],
        batch["coords_tgt"],
        batch["source_mask"],
        batch["target_mask"],
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
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            logits = tracker_logits(model, batch)
            loss = batch_partial_edge_focal_bce(
                logits,
                batch["target"],
                batch["source_mask"],
                batch["target_mask"],
                gamma=gamma,
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
        "partial_mask_loss": safe_ratio(total_loss, windows),
        "batch_count": float(batches),
        "window_count": float(windows),
    }


def evaluate_tracker(
    model: Any,
    loader: Any,
    device: Any,
    *,
    gamma: float,
    use_amp: bool,
) -> dict[str, Any]:
    import torch

    model.eval()
    loss_sum = 0.0
    legacy_loss_sum = 0.0
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
                loss = partial_edge_focal_bce(pair_logits, pair_target, gamma=gamma)
                loss_sum += float(loss.detach().cpu())
                legacy_loss_sum += float(
                    legacy_focal_bce(pair_logits, pair_target, gamma=gamma).detach().cpu()
                )
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
        "partial_mask_loss": safe_ratio(loss_sum, loss_windows),
        "legacy_mask_loss": safe_ratio(legacy_loss_sum, loss_windows),
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


def dataloader_worker_init(_: int) -> None:
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
