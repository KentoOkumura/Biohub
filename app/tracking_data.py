"""Read-only inputs and spatial diagnostics for the tracking viewer.

Distances and plots use micrometers; GEFF stores original voxel coordinates.
No model inference, prediction edges, or official competition score is computed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

XYZ = ["z", "y", "x"]
NODE_COLUMNS = ["node_id", "t", *XYZ]


def validate_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    nodes = nodes.copy()
    if not set(NODE_COLUMNS).issubset(nodes):
        raise ValueError(f"Nodes require {NODE_COLUMNS}")
    values = nodes[NODE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Node IDs, frames and coordinates must be finite")
    if nodes.node_id.duplicated().any():
        raise ValueError("Node IDs must be unique within a dataset")
    for col in ["node_id", "t"]:
        if (nodes[col] < 0).any() or (nodes[col] % 1 != 0).any():
            raise ValueError(f"{col} must contain nonnegative integers")
        nodes[col] = nodes[col].astype(np.int64)
    return nodes.sort_values(["t", "node_id"]).reset_index(drop=True)


def load_candidates(folder: Path) -> tuple[pd.DataFrame, dict]:
    """Read coordinates only, preserving one copy per frame across overlapping windows.

    The earliest window containing a frame supplies its scores. Coordinates and IDs
    must agree in other windows; window-dependent scores are counted in the receipt.
    Features are intentionally never decompressed or loaded.
    """
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise ValueError(f"No window NPZ files: {folder}")
    frames: dict[int, pd.DataFrame] = {}
    score_changes = 0
    checkpoints = None
    for path in files:
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(saved["__metadata_json__"].tobytes().decode("utf-8"))
            if meta.get("schema_version") != 1 or meta.get("dataset") != folder.name:
                raise ValueError(f"Cache schema or dataset mismatch: {path}")
            if meta.get("coordinate_units", {}).get("physical") != "micrometer":
                raise ValueError(f"Expected physical coordinates in micrometers: {path}")
            identity = tuple(
                meta.get(f"{side}_checkpoint_sha256") for side in ["primary", "secondary"]
            )
            if checkpoints is not None and checkpoints != identity:
                raise ValueError(f"Mixed model checkpoints: {path}")
            checkpoints = identity
            times = meta["window_frames"]
            if len(times) != 2 or any(int(t) != t or t < 0 for t in times) or times[1] <= times[0]:
                raise ValueError(f"Invalid window frames: {path}")
            for side, t in zip(["src", "tgt"], times, strict=True):
                ids = saved[f"candidate_ids_{side}"]
                coords = saved[f"coords_{side}_physical"]
                mask = saved[f"candidate_mask_{side}"]
                scores = saved[f"detection_scores_{side}"]
                n = len(ids)
                if coords.shape != (n, 3) or mask.shape != (n,) or scores.shape != (n,):
                    raise ValueError(f"Candidate array lengths disagree: {path}, {side}")
                if mask.dtype != bool:
                    raise ValueError(f"Candidate mask must be boolean: {path}")
                frame = pd.DataFrame(coords[mask], columns=XYZ)
                frame["node_id"] = ids[mask]
                frame["t"] = int(t)
                frame["score"] = scores[mask]
                frame["window"] = path.stem
                frame = validate_nodes(frame)
                if not np.isfinite(frame.score).all() or not frame.score.between(0, 1).all():
                    raise ValueError(f"Invalid detection score: {path}")
                if t in frames:
                    prior = frames[t]
                    if not np.array_equal(prior[NODE_COLUMNS], frame[NODE_COLUMNS]):
                        raise ValueError(
                            f"Overlapping windows disagree on IDs/coordinates at t={t}"
                        )
                    score_changes += int(not np.array_equal(prior.score, frame.score))
                else:
                    frames[int(t)] = frame
    result = validate_nodes(pd.concat(frames.values(), ignore_index=True))
    receipt = {
        "dataset": folder.name,
        "windows": len(files),
        "frames": sorted(frames),
        "score_changed_frames": score_changes,
        "checkpoints": checkpoints,
        "score_policy": "earliest window containing each frame",
        "units": "µm",
    }
    return result, receipt


def load_geff(path: Path, scale: tuple[float, float, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import zarr

    if len(scale) != 3 or not np.isfinite(scale).all() or min(scale) <= 0:
        raise ValueError("Voxel scale must have three positive finite values")
    group = zarr.open_group(str(path), mode="r")
    nodes = pd.DataFrame(
        {
            "node_id": np.asarray(group["nodes/ids"][:]),
            **{axis: np.asarray(group[f"nodes/props/{axis}/values"][:]) for axis in ["t", *XYZ]},
        }
    )
    nodes = validate_nodes(nodes)
    nodes[XYZ] = nodes[XYZ].astype(float) * np.asarray(scale)
    edges_array = np.asarray(group["edges/ids"][:])
    if edges_array.ndim != 2 or edges_array.shape[1] != 2:
        raise ValueError("GEFF edges/ids must have shape (N, 2)")
    edges = pd.DataFrame(edges_array, columns=["source_id", "target_id"])
    if edges.duplicated().any():
        raise ValueError("Duplicate GEFF edges")
    by_id = nodes.set_index("node_id")
    if not set(edges.to_numpy().ravel()).issubset(by_id.index):
        raise ValueError("GEFF edge references a missing node")
    if len(edges):
        ts = by_id.loc[edges.source_id, "t"].to_numpy()
        tt = by_id.loc[edges.target_id, "t"].to_numpy()
        if (tt <= ts).any():
            raise ValueError("GEFF edges must point forward in time")
    nodes["lineage"] = lineage_ids(nodes, edges)
    nodes["division"] = nodes.node_id.isin(
        edges.source_id.value_counts().loc[lambda s: s >= 2].index
    )
    return nodes, edges


def lineage_ids(nodes: pd.DataFrame, edges: pd.DataFrame) -> list[int]:
    """Weak connected components retain both daughters at divisions."""
    parent = {int(n): int(n) for n in nodes.node_id}

    def root(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for source, target in edges.itertuples(index=False, name=None):
        a, b = root(int(source)), root(int(target))
        parent[max(a, b)] = min(a, b)
    return [root(int(n)) for n in nodes.node_id]


def match_frame(detections: pd.DataFrame, gt: pd.DataFrame, radius: float) -> pd.DataFrame:
    """Maximum-cardinality one-to-one assignment, then minimum physical distance.

    This is an exploratory diagnostic, not the official tracking metric. Matching
    is performed before any display crop or lineage filter.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Matching radius must be positive and finite")
    columns = [
        "gt_id",
        "detection_id",
        "distance_um",
        "nearest_distance_um",
        "candidates_in_radius",
    ]
    if gt.empty:
        return pd.DataFrame(columns=columns)
    if len(set(gt.t) | set(detections.t)) > 1:
        raise ValueError("match_frame accepts exactly one timepoint")
    n, m = len(gt), len(detections)
    rows = pd.DataFrame(
        {
            "gt_id": gt.node_id.to_numpy(),
            "detection_id": pd.array([None] * n, dtype="Int64"),
            "distance_um": np.nan,
            "nearest_distance_um": np.nan,
            "candidates_in_radius": np.zeros(n, dtype=int),
        }
    )
    if not m:
        return rows[columns]
    distances = cdist(gt[XYZ].to_numpy(), detections[XYZ].to_numpy())
    allowed = distances <= radius
    # A forbidden pair costs more than all allowed distances combined, ensuring
    # cardinality takes priority even where nearest-neighbor greedy matching fails.
    penalty = (min(n, m) + 1) * (radius + 1)
    gi, di = linear_sum_assignment(np.where(allowed, distances, penalty))
    rows["nearest_distance_um"] = distances.min(axis=1)
    rows["candidates_in_radius"] = allowed.sum(axis=1)
    for g, d in zip(gi, di, strict=True):
        if allowed[g, d]:
            rows.loc[g, "detection_id"] = int(detections.iloc[d].node_id)
            rows.loc[g, "distance_um"] = distances[g, d]
    return rows[columns]


def analyze_sequence(detections, gt, frames, radius):
    tables, counts = [], []
    for t in frames:
        det_t, gt_t = detections[detections.t == t], gt[gt.t == t]
        matches = match_frame(det_t, gt_t, radius)
        matches["t"] = t
        tables.append(matches)
        nmatch = int(matches.detection_id.notna().sum())
        counts.append(
            {
                "t": t,
                "検出候補": len(det_t),
                "正解": len(gt_t),
                "対応あり": nmatch,
                "未対応の正解": len(gt_t) - nmatch,
                "未対応の検出": len(det_t) - nmatch,
                "正解の対応率": nmatch / len(gt_t) if len(gt_t) else np.nan,
            }
        )
    return pd.concat(tables, ignore_index=True), pd.DataFrame(counts)


def image_array(path: Path):
    import zarr

    group = zarr.open(str(path), mode="r")
    array = group if isinstance(group, zarr.Array) else group["0"]
    if array.ndim != 4:
        raise ValueError("Image must have shape (T, Z, Y, X)")
    return array


def read_projection(path: Path, t: int, plane: str, z_range: tuple[int, int]):
    """Load one timepoint only. Z bounds include both endpoint slices."""
    array = image_array(path)
    lo, hi = z_range
    if not 0 <= t < array.shape[0] or not 0 <= lo <= hi < array.shape[1]:
        raise ValueError("Requested timepoint or Z range is outside the image")
    volume = np.asarray(array[t, lo : hi + 1, :, :])
    axis = {"XY": 0, "XZ": 1, "YZ": 2}[plane]
    return volume.max(axis=axis)
