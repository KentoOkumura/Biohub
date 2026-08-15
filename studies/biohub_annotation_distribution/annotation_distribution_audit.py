# %% [markdown]
# # Biohub official annotation distribution audit

# %%
from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COMPETITION_SLUG = "biohub-cell-tracking-during-development"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
FRAME_COUNT = 100
VOLUME_SHAPE = np.array([64, 256, 256], dtype=np.int64)
SPATIAL_BIN_COUNT = 8


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def nested_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = nested_value(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = nested_value(child, key)
            if found is not None:
                return found
    return None


def decompress_zstd(payload: bytes, expected_size: int) -> bytes:
    try:
        import zstandard

        return zstandard.ZstdDecompressor().decompress(
            payload,
            max_output_size=expected_size,
        )
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        import pyarrow

        return pyarrow.Codec("zstd").decompress(payload, expected_size)
    except (ImportError, ModuleNotFoundError):
        pass

    zstd_command = shutil.which("zstd")
    if zstd_command is None:
        raise RuntimeError("no zstd decoder is available in the Kaggle runtime")
    completed = subprocess.run(
        [zstd_command, "-d", "-c"],
        input=payload,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def read_zarr_v3_array(path: Path) -> np.ndarray:
    metadata = read_json(path / "zarr.json")
    shape = tuple(int(value) for value in metadata["shape"])
    dtype = np.dtype(metadata["data_type"]).newbyteorder("<")
    chunk_shape = tuple(
        int(value) for value in metadata["chunk_grid"]["configuration"]["chunk_shape"]
    )
    if chunk_shape != shape:
        raise NotImplementedError(
            f"expected one full-array chunk: {path}, {chunk_shape=}, {shape=}"
        )
    chunk_path = path / "c" / Path(*("0" for _ in shape))
    payload = chunk_path.read_bytes()
    expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    decoded = decompress_zstd(payload, expected_size)
    if len(decoded) != expected_size:
        raise RuntimeError(
            f"decoded byte size mismatch: {path}, expected={expected_size}, actual={len(decoded)}"
        )
    return np.frombuffer(decoded, dtype=dtype).reshape(shape).copy()


def find_competition_root() -> Path:
    candidates = [
        KAGGLE_INPUT_ROOT / "competitions" / COMPETITION_SLUG,
        KAGGLE_INPUT_ROOT / COMPETITION_SLUG,
    ]
    candidates.extend(KAGGLE_INPUT_ROOT.glob(f"**/{COMPETITION_SLUG}"))
    matches = sorted({path.resolve() for path in candidates if path.is_dir()})
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one competition input root for {COMPETITION_SLUG}, found {matches}"
        )
    return matches[0]


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {key: None for key in ("min", "q05", "q25", "median", "q75", "q95", "max")}
    probs = np.array([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    result = np.quantile(values.astype(np.float64), probs)
    return {
        key: float(value)
        for key, value in zip(
            ("min", "q05", "q25", "median", "q75", "q95", "max"), result, strict=False
        )
    }


class UnionFind:
    def __init__(self, values: np.ndarray) -> None:
        self.parent = {int(value): int(value) for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def component_sizes(node_ids: np.ndarray, edges: np.ndarray) -> np.ndarray:
    union_find = UnionFind(node_ids)
    known = union_find.parent
    for source, target in edges:
        source_id = int(source)
        target_id = int(target)
        if source_id in known and target_id in known:
            union_find.union(source_id, target_id)
    counts = Counter(union_find.find(int(node_id)) for node_id in node_ids)
    return np.asarray(list(counts.values()), dtype=np.int64)


def spatial_histogram(values: np.ndarray, size: int) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64), 0.0, np.nextafter(float(size), 0.0))
    bins = np.floor(clipped / size * SPATIAL_BIN_COUNT).astype(np.int64)
    return np.bincount(bins, minlength=SPATIAL_BIN_COUNT)


competition_root = find_competition_root()
train_root = competition_root / "train"
graph_paths = sorted(train_root.glob("*.geff"))
if len(graph_paths) != 199:
    raise RuntimeError(f"expected 199 train GEFF graphs, found {len(graph_paths)}")

sample_rows: list[dict[str, Any]] = []
frame_rows: list[dict[str, Any]] = []
component_rows: list[dict[str, Any]] = []
spatial_counts = {axis: np.zeros(SPATIAL_BIN_COUNT, dtype=np.int64) for axis in ("z", "y", "x")}
all_component_sizes: list[int] = []

for sample_index, graph_path in enumerate(graph_paths, start=1):
    sample_id = graph_path.name.removesuffix(".geff")
    embryo_id = sample_id.split("_", maxsplit=1)[0]
    node_ids = read_zarr_v3_array(graph_path / "nodes" / "ids").astype(np.int64)
    t = read_zarr_v3_array(graph_path / "nodes" / "props" / "t" / "values").astype(np.int64)
    z = read_zarr_v3_array(graph_path / "nodes" / "props" / "z" / "values").astype(np.int64)
    y = read_zarr_v3_array(graph_path / "nodes" / "props" / "y" / "values").astype(np.int64)
    x = read_zarr_v3_array(graph_path / "nodes" / "props" / "x" / "values").astype(np.int64)
    edges = read_zarr_v3_array(graph_path / "edges" / "ids").astype(np.int64).reshape(-1, 2)
    root_metadata = read_json(graph_path / "zarr.json")
    estimated_nodes = int(nested_value(root_metadata, "estimated_number_of_nodes"))

    if not (len(node_ids) == len(t) == len(z) == len(y) == len(x)):
        raise AssertionError(f"misaligned node arrays: {sample_id}")
    if np.any((t < 0) | (t >= FRAME_COUNT)):
        raise AssertionError(f"timepoint outside 0..{FRAME_COUNT - 1}: {sample_id}")

    counts_by_t = np.bincount(t, minlength=FRAME_COUNT)
    active_frames = np.flatnonzero(counts_by_t)
    first_t = int(active_frames[0]) if active_frames.size else None
    last_t = int(active_frames[-1]) if active_frames.size else None
    span = (last_t - first_t + 1) if first_t is not None and last_t is not None else 0
    missing_inside_span = (
        int(np.count_nonzero(counts_by_t[first_t : last_t + 1] == 0)) if span else 0
    )

    sizes = component_sizes(node_ids, edges)
    all_component_sizes.extend(int(value) for value in sizes)
    for value in sizes:
        component_rows.append(
            {
                "sample_id": sample_id,
                "embryo_id": embryo_id,
                "component_size_nodes": int(value),
            }
        )

    source_counts = Counter(int(source) for source in edges[:, 0]) if len(edges) else Counter()
    division_nodes = sum(count >= 2 for count in source_counts.values())
    id_to_t = {int(node_id): int(time) for node_id, time in zip(node_ids, t, strict=False)}
    edge_dt = np.asarray(
        [id_to_t[int(target)] - id_to_t[int(source)] for source, target in edges],
        dtype=np.int64,
    )

    axis_values = {"z": z, "y": y, "x": x}
    axis_sizes = {"z": 64, "y": 256, "x": 256}
    axis_summary: dict[str, Any] = {}
    for axis, values in axis_values.items():
        summary = quantiles(values)
        size = axis_sizes[axis]
        axis_summary.update({f"{axis}_{key}": value for key, value in summary.items()})
        axis_summary[f"{axis}_bbox_fraction"] = (
            float((values.max() - values.min() + 1) / size) if values.size else None
        )
        spatial_counts[axis] += spatial_histogram(values, size)

    sample_rows.append(
        {
            "sample_id": sample_id,
            "embryo_id": embryo_id,
            "annotated_nodes": int(len(node_ids)),
            "annotated_edges": int(len(edges)),
            "estimated_total_nodes": estimated_nodes,
            "annotation_fraction": float(len(node_ids) / estimated_nodes),
            "active_frame_count": int(active_frames.size),
            "first_annotated_t": first_t,
            "last_annotated_t": last_t,
            "annotated_t_span": int(span),
            "missing_frames_inside_span": missing_inside_span,
            "mean_nodes_per_active_frame": (
                float(len(node_ids) / active_frames.size) if active_frames.size else 0.0
            ),
            "max_nodes_in_one_frame": int(counts_by_t.max(initial=0)),
            "connected_component_count": int(len(sizes)),
            "median_component_size": float(np.median(sizes)) if sizes.size else 0.0,
            "max_component_size": int(sizes.max(initial=0)),
            "division_node_count": int(division_nodes),
            "non_unit_edge_dt_count": int(np.count_nonzero(edge_dt != 1)),
            **axis_summary,
        }
    )

    for frame, count in enumerate(counts_by_t):
        frame_rows.append(
            {
                "sample_id": sample_id,
                "embryo_id": embryo_id,
                "t": frame,
                "annotated_nodes": int(count),
            }
        )

    if sample_index % 25 == 0 or sample_index == len(graph_paths):
        print(f"processed {sample_index}/{len(graph_paths)} GEFF graphs", flush=True)

sample_df = pd.DataFrame(sample_rows).sort_values("sample_id").reset_index(drop=True)
frame_df = pd.DataFrame(frame_rows)
component_df = pd.DataFrame(component_rows)


def series_summary(series: pd.Series) -> dict[str, float]:
    values = series.to_numpy(dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


by_embryo: dict[str, Any] = {}
for embryo_id, group in sample_df.groupby("embryo_id", sort=True):
    by_embryo[str(embryo_id)] = {
        "sample_count": int(len(group)),
        "annotated_nodes": int(group["annotated_nodes"].sum()),
        "estimated_total_nodes": int(group["estimated_total_nodes"].sum()),
        "weighted_annotation_fraction": float(
            group["annotated_nodes"].sum() / group["estimated_total_nodes"].sum()
        ),
        "sample_annotation_fraction": series_summary(group["annotation_fraction"]),
        "active_frame_count": series_summary(group["active_frame_count"]),
    }

time_histogram = (
    frame_df.groupby("t", as_index=False)
    .agg(
        annotated_nodes=("annotated_nodes", "sum"),
        samples_with_annotation=("annotated_nodes", lambda values: int(np.count_nonzero(values))),
    )
    .sort_values("t")
)

spatial_rows: list[dict[str, Any]] = []
for axis, counts in spatial_counts.items():
    total = int(counts.sum())
    for bin_index, count in enumerate(counts):
        spatial_rows.append(
            {
                "axis": axis,
                "bin": bin_index,
                "normalized_start": bin_index / SPATIAL_BIN_COUNT,
                "normalized_end": (bin_index + 1) / SPATIAL_BIN_COUNT,
                "annotated_nodes": int(count),
                "fraction": float(count / total),
            }
        )
spatial_df = pd.DataFrame(spatial_rows)

summary = {
    "schema_version": 1,
    "captured_at_utc": datetime.now(UTC).isoformat(),
    "competition_slug": COMPETITION_SLUG,
    "sample_count": int(len(sample_df)),
    "annotated_nodes": int(sample_df["annotated_nodes"].sum()),
    "annotated_edges": int(sample_df["annotated_edges"].sum()),
    "estimated_total_nodes": int(sample_df["estimated_total_nodes"].sum()),
    "weighted_annotation_fraction": float(
        sample_df["annotated_nodes"].sum() / sample_df["estimated_total_nodes"].sum()
    ),
    "sample_annotation_fraction": series_summary(sample_df["annotation_fraction"]),
    "active_frame_count": series_summary(sample_df["active_frame_count"]),
    "annotated_t_span": series_summary(sample_df["annotated_t_span"]),
    "mean_nodes_per_active_frame": series_summary(sample_df["mean_nodes_per_active_frame"]),
    "connected_component_count": series_summary(sample_df["connected_component_count"]),
    "component_size_nodes": series_summary(component_df["component_size_nodes"]),
    "samples_annotated_in_all_100_frames": int(
        np.count_nonzero(sample_df["active_frame_count"] == FRAME_COUNT)
    ),
    "samples_with_gaps_inside_annotation_span": int(
        np.count_nonzero(sample_df["missing_frames_inside_span"] > 0)
    ),
    "samples_with_non_unit_edge_dt": int(np.count_nonzero(sample_df["non_unit_edge_dt_count"] > 0)),
    "division_node_count": int(sample_df["division_node_count"].sum()),
    "by_embryo": by_embryo,
    "spatial_bin_count": SPATIAL_BIN_COUNT,
    "interpretation_limit": (
        "GEFF records annotated nodes and edges only. Exact locations of true but unannotated "
        "cells are not present; estimated_total_nodes supports sample-level coverage only."
    ),
}

sample_df.to_csv("sample_annotation_distribution.csv", index=False)
time_histogram.to_csv("annotation_time_histogram.csv", index=False)
spatial_df.to_csv("annotation_spatial_histogram.csv", index=False)
component_df.to_csv("annotation_component_sizes.csv", index=False)
Path("annotation_distribution_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
)

print("ANNOTATION_DISTRIBUTION_SUMMARY_START", flush=True)
print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
print("ANNOTATION_DISTRIBUTION_SUMMARY_END", flush=True)
