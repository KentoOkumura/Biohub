# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp015 oracle stage limits diagnostic
#
# This notebook joins the fixed public-model candidate and final graphs with the
# organizer GEFF annotations. Ground truth is used only for diagnostic matching
# and aggregation. It never changes a prediction, threshold, weight, or decode.
#
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Input discovery and fixed exp012 cohort
# 3. GEFF and final-graph loading
# 4. Per-frame one-to-one node matching
# 5. Per-sample stage limits
# 6. Condition and cross-condition summaries
# 7. Artifacts, manifest, and metrics

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

EXPERIMENT = "exp015_oracle_stage_limits"
COMPETITION = "biohub-cell-tracking-during-development"

COUNT_COLUMNS = (
    "known_nodes_total",
    "known_nodes_candidate_matched",
    "known_nodes_final_selected",
    "known_edges_total",
    "known_edges_endpoints_matched",
    "known_edges_candidate_present",
    "known_edges_final_selected",
    "known_divisions_total",
    "known_divisions_candidate_triplet",
    "known_divisions_final_selected",
    "ambiguous_gt_node_count",
    "ambiguous_candidate_node_count",
)
RATIO_SPECS = (
    ("candidate_node_recall", "known_nodes_candidate_matched", "known_nodes_total"),
    ("final_selected_node_recall", "known_nodes_final_selected", "known_nodes_total"),
    ("edge_endpoint_recall", "known_edges_endpoints_matched", "known_edges_total"),
    ("candidate_edge_recall", "known_edges_candidate_present", "known_edges_total"),
    ("final_selected_edge_recall", "known_edges_final_selected", "known_edges_total"),
    (
        "candidate_division_triplet_recall",
        "known_divisions_candidate_triplet",
        "known_divisions_total",
    ),
    (
        "final_selected_division_recall",
        "known_divisions_final_selected",
        "known_divisions_total",
    ),
)


def is_kaggle_runtime() -> bool:
    return Path("/kaggle/input").is_dir() and Path("/kaggle/working").is_dir()


OFFLINE_GRAPH_MODULES = {
    "tracksdata": "tracksdata",
    "zarr": "zarr",
    "pyscipopt": "pyscipopt",
    "geff": "geff",
    "geff_spec": "geff_spec",
    "ilpy": "ilpy",
    "polars": "polars",
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


def run_offline_install(
    wheel_dirs: list[Path],
    specs: tuple[str, ...],
    *,
    force_reinstall: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
    ]
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


def polars_runtime_ready() -> bool:
    try:
        import polars as pl
        from polars._plr import PySeries

        _ = PySeries
        return (
            hasattr(pl, "Float16") and pl.Series([-999999.0], dtype=pl.Float64).dtype == pl.Float64
        )
    except Exception:
        return False


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
        print("Required tracksdata/GEFF packages import successfully.")
        return
    if not is_kaggle_runtime():
        raise ImportError(
            f"missing or incompatible local graph packages: {failures}; "
            f"refresh_polars={refresh_polars}"
        )

    wheel_dirs = offline_wheel_dirs()
    if not wheel_dirs:
        raise FileNotFoundError(
            "offline wheel directory for biohub-tracking-support-pack-50ep-v1 was not found"
        )
    print("Offline wheel dirs:", [str(path) for path in wheel_dirs])
    if refresh_polars:
        print("Refreshing incompatible Polars runtime from offline wheels.")
        run_offline_install(
            wheel_dirs,
            ("polars>=1.36", "polars-runtime-32"),
            force_reinstall=True,
        )
        purge_graph_modules(include_polars=True)
        importlib.invalidate_caches()
        if not polars_runtime_ready():
            raise ImportError("offline Polars refresh did not provide Float16 support")

    print("Installing graph packages from offline wheels:", sorted(failures))
    run_offline_install(
        wheel_dirs,
        OFFLINE_GRAPH_PACKAGE_SPECS,
        force_reinstall=False,
    )
    purge_graph_modules(include_polars=False)
    importlib.invalidate_caches()
    remaining: dict[str, str] = {}
    for package_name, module_name in OFFLINE_GRAPH_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as error:
            remaining[package_name] = f"{type(error).__name__}: {error}"
    if remaining:
        raise ImportError(f"offline graph package imports still fail: {remaining}")
    if not polars_runtime_ready():
        raise ImportError("Polars runtime lost Float16 support after graph package install")
    print("Offline graph package install succeeded.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def unique_existing(paths: Iterable[Path], label: str) -> Path:
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


# %% [markdown]
# ## 2. Input discovery and fixed exp012 cohort
#
# The 16 degraded samples are recomputed from the two fixed exp012 route rows.
# Both exp012 files are verified against the SHA values fixed before this
# experiment. The condition columns come from the fixed exp005 baseline row.


# %%
def resolve_inference_root(working_root: Path, source_cfg: dict[str, Any]) -> Path:
    local = (working_root / str(source_cfg["local_artifact_dir"])).resolve()
    candidates = [local]
    if is_kaggle_runtime():
        kernel_slug = str(source_cfg["kernel_source"]).split("/", 1)[-1]
        notebook_root = Path("/kaggle/input/notebooks")
        candidates.extend(
            path.parent
            for path in notebook_root.rglob(str(source_cfg["manifest"]))
            if kernel_slug in path.as_posix()
        )
    return unique_existing(candidates, "exp015 inference output root")


def resolve_exp012_root(working_root: Path, source_cfg: dict[str, Any]) -> Path:
    local = (working_root / str(source_cfg["local_dir"])).resolve()
    candidates = [local]
    if is_kaggle_runtime():
        kernel_slug = str(source_cfg["kernel_source"]).split("/", 1)[-1]
        notebook_root = Path("/kaggle/input/notebooks")
        candidates.extend(
            path.parent
            for path in notebook_root.rglob("per_sample_readout.csv")
            if kernel_slug in path.as_posix()
        )
    return unique_existing(candidates, "exp012 readout root")


def resolve_train_dir(working_root: Path) -> Path:
    candidates = [
        working_root.parents[1] / "data" / "raw" / "train",
        Path(f"/kaggle/input/competitions/{COMPETITION}/train"),
        Path(f"/kaggle/input/{COMPETITION}/train"),
    ]
    return unique_existing(candidates, "competition train directory")


def load_fixed_conditions(
    root: Path,
    cfg: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    per_sample_path = root / "per_sample_readout.csv"
    paired_path = root / "paired_route_comparison.csv"
    expected_per_sample_sha = str(cfg["per_sample_readout_sha256"])
    expected_paired_sha = str(cfg["paired_route_comparison_sha256"])
    observed_per_sample_sha = sha256_file(per_sample_path)
    observed_paired_sha = sha256_file(paired_path)
    if observed_per_sample_sha != expected_per_sample_sha:
        raise ValueError("exp012 per-sample readout SHA mismatch")
    if observed_paired_sha != expected_paired_sha:
        raise ValueError("exp012 paired comparison SHA mismatch")

    rows = read_csv(per_sample_path)
    baseline_route = str(cfg["baseline_route"])
    comparison_route = str(cfg["comparison_route"])
    by_route: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        route = str(row["route"])
        sample = str(row["sample"])
        if sample in by_route[route]:
            raise ValueError(f"duplicate exp012 row: {route} {sample}")
        by_route[route][sample] = row

    baseline = by_route[baseline_route]
    comparison = by_route[comparison_route]
    if set(baseline) != set(comparison):
        raise ValueError("exp012 baseline and comparison sample sets differ")

    condition_columns = (
        "embryo",
        "brightness_bucket",
        "candidate_density_bucket",
        "boundary_distance_bucket",
        "known_division",
    )
    condition_rows: dict[str, dict[str, Any]] = {}
    degraded_rows: list[dict[str, Any]] = []
    for sample in sorted(baseline):
        base_row = baseline[sample]
        comp_row = comparison[sample]
        base_value = float(base_row["adj_edge_jaccard"])
        comp_value = float(comp_row["adj_edge_jaccard"])
        if not math.isfinite(base_value) or not math.isfinite(comp_value):
            raise ValueError(f"non-finite exp012 paired value: {sample}")
        conditions = {key: base_row[key] for key in condition_columns}
        conditions["known_division"] = str(conditions["known_division"]).lower() == "true"
        condition_rows[sample] = conditions
        delta = comp_value - base_value
        if delta < 0.0:
            degraded_rows.append(
                {
                    "sample": sample,
                    **conditions,
                    "exp005_adj_edge_jaccard": base_value,
                    "exp006_adj_edge_jaccard": comp_value,
                    "adj_edge_jaccard_delta": delta,
                }
            )

    expected_count = int(cfg["expected_degraded_count"])
    if len(degraded_rows) != expected_count:
        raise ValueError(f"expected {expected_count} degraded samples, got {len(degraded_rows)}")
    expected_embryo = str(cfg["expected_degraded_embryo"])
    observed_embryos = {str(row["embryo"]) for row in degraded_rows}
    if observed_embryos != {expected_embryo}:
        raise ValueError(f"degraded sample embryo mismatch: {observed_embryos}")

    evidence = {
        "per_sample_readout_sha256": observed_per_sample_sha,
        "paired_route_comparison_sha256": observed_paired_sha,
    }
    return condition_rows, degraded_rows, evidence


# %% [markdown]
# ## 3. GEFF and final-graph loading


# %%
def graph_from_geff(path: Path):
    import tracksdata as td

    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def load_geff_tables(
    path: Path,
) -> tuple[dict[int, tuple[int, float, float, float]], set[tuple[int, int]], str]:
    graph = graph_from_geff(path)
    nodes: dict[int, tuple[int, float, float, float]] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        if node_id in nodes:
            raise ValueError(f"duplicate node id in {path}: {node_id}")
        nodes[node_id] = (
            int(row["t"]),
            float(row["z"]),
            float(row["y"]),
            float(row["x"]),
        )
    edges = {
        (int(row["source_id"]), int(row["target_id"]))
        for row in graph.edge_attrs().iter_rows(named=True)
    }
    if any(source not in nodes or target not in nodes for source, target in edges):
        raise ValueError(f"dangling edge in {path}")
    canonical = {
        "nodes": [[node_id, *nodes[node_id]] for node_id in sorted(nodes)],
        "edges": [list(edge) for edge in sorted(edges)],
    }
    return nodes, edges, sha256_text(json.dumps(canonical, separators=(",", ":")))


def load_final_graph(
    path: Path,
) -> tuple[set[int], dict[int, tuple[int, float, float, float]], set[tuple[int, int]], str]:
    with np.load(path, allow_pickle=False) as payload:
        node_ids = np.asarray(payload["node_ids"], dtype=np.int64)
        node_tzyx = np.asarray(payload["node_tzyx"], dtype=np.float64)
        edges_array = np.asarray(payload["edges"], dtype=np.int64)
    if node_ids.ndim != 1 or node_tzyx.shape != (len(node_ids), 4):
        raise ValueError(f"invalid final node arrays: {path}")
    if edges_array.ndim != 2 or edges_array.shape[1] != 2:
        raise ValueError(f"invalid final edge array: {path}")
    if len(set(int(value) for value in node_ids)) != len(node_ids):
        raise ValueError(f"duplicate final node id: {path}")
    nodes = {
        int(node_id): (
            int(values[0]),
            float(values[1]),
            float(values[2]),
            float(values[3]),
        )
        for node_id, values in zip(node_ids, node_tzyx, strict=True)
    }
    edges = {(int(source), int(target)) for source, target in edges_array}
    node_set = set(nodes)
    if any(source not in node_set or target not in node_set for source, target in edges):
        raise ValueError(f"dangling final edge: {path}")
    canonical = {
        "nodes": [[node_id, *nodes[node_id]] for node_id in sorted(nodes)],
        "edges": [list(edge) for edge in sorted(edges)],
    }
    return node_set, nodes, edges, sha256_text(json.dumps(canonical, separators=(",", ":")))


# %% [markdown]
# ## 4. Per-frame one-to-one node matching
#
# Matching uses physical distance and is solved independently in each frame.
# Counts of GT nodes or candidates with multiple possible partners inside the
# radius are reported separately. Unmatched candidates are not treated as false
# positives because the organizer annotations are sparse.


# %%
def match_nodes_per_frame(
    gt_nodes: dict[int, tuple[int, float, float, float]],
    candidate_nodes: dict[int, tuple[int, float, float, float]],
    scale_zyx_um: tuple[float, float, float],
    max_distance_um: float,
) -> tuple[dict[int, int], dict[str, Any]]:
    gt_by_time: dict[int, list[int]] = defaultdict(list)
    candidate_by_time: dict[int, list[int]] = defaultdict(list)
    for node_id, values in gt_nodes.items():
        gt_by_time[int(values[0])].append(node_id)
    for node_id, values in candidate_nodes.items():
        candidate_by_time[int(values[0])].append(node_id)

    mapping: dict[int, int] = {}
    distances: list[float] = []
    ambiguous_gt = 0
    ambiguous_candidate = 0
    scale = np.asarray(scale_zyx_um, dtype=np.float64)

    for timepoint in sorted(set(gt_by_time) | set(candidate_by_time)):
        gt_ids = sorted(gt_by_time.get(timepoint, []))
        candidate_ids = sorted(candidate_by_time.get(timepoint, []))
        if not gt_ids or not candidate_ids:
            continue
        gt_pos = np.asarray([gt_nodes[node_id][1:] for node_id in gt_ids]) * scale
        candidate_pos = (
            np.asarray([candidate_nodes[node_id][1:] for node_id in candidate_ids]) * scale
        )
        distance = np.linalg.norm(gt_pos[:, None, :] - candidate_pos[None, :, :], axis=2)
        feasible = distance <= max_distance_um
        ambiguous_gt += int((feasible.sum(axis=1) > 1).sum())
        ambiguous_candidate += int((feasible.sum(axis=0) > 1).sum())

        gated = np.where(feasible, distance, 1.0e9)
        row_indices, column_indices = linear_sum_assignment(gated)
        for row_index, column_index in zip(row_indices, column_indices, strict=True):
            value = float(distance[row_index, column_index])
            if value <= max_distance_um:
                gt_id = gt_ids[int(row_index)]
                candidate_id = candidate_ids[int(column_index)]
                mapping[gt_id] = candidate_id
                distances.append(value)

    diagnostics = {
        "ambiguous_gt_node_count": ambiguous_gt,
        "ambiguous_candidate_node_count": ambiguous_candidate,
        "matching_distance_um_p50": (float(np.median(distances)) if distances else float("nan")),
        "matching_distance_um_max": max(distances) if distances else float("nan"),
    }
    return mapping, diagnostics


# %% [markdown]
# ## 5. Per-sample stage limits


# %%
def evaluate_sample_stages(
    sample: str,
    gt_nodes: dict[int, tuple[int, float, float, float]],
    gt_edges: set[tuple[int, int]],
    candidate_nodes: dict[int, tuple[int, float, float, float]],
    candidate_edges: set[tuple[int, int]],
    final_node_ids: set[int],
    final_edges: set[tuple[int, int]],
    conditions: dict[str, Any],
    scale_zyx_um: tuple[float, float, float],
    max_distance_um: float,
) -> dict[str, Any]:
    mapping, matching = match_nodes_per_frame(
        gt_nodes,
        candidate_nodes,
        scale_zyx_um=scale_zyx_um,
        max_distance_um=max_distance_um,
    )
    endpoint_edges = {edge for edge in gt_edges if edge[0] in mapping and edge[1] in mapping}
    mapped_edge = {edge: (mapping[edge[0]], mapping[edge[1]]) for edge in endpoint_edges}
    candidate_present = {
        edge for edge, prediction_edge in mapped_edge.items() if prediction_edge in candidate_edges
    }
    final_selected = {
        edge
        for edge in candidate_present
        if mapped_edge[edge] in final_edges
        and mapped_edge[edge][0] in final_node_ids
        and mapped_edge[edge][1] in final_node_ids
    }

    gt_children: dict[int, set[int]] = defaultdict(set)
    for source, target in gt_edges:
        gt_children[source].add(target)
    division_parents = {source for source, targets in gt_children.items() if len(targets) >= 2}
    candidate_divisions = 0
    final_divisions = 0
    for source in division_parents:
        candidate_children = {
            target for target in gt_children[source] if (source, target) in candidate_present
        }
        final_children = {
            target for target in gt_children[source] if (source, target) in final_selected
        }
        candidate_divisions += int(len(candidate_children) >= 2)
        final_divisions += int(len(final_children) >= 2)

    final_selected_nodes = {
        gt_id for gt_id, candidate_id in mapping.items() if candidate_id in final_node_ids
    }
    row: dict[str, Any] = {
        "sample": sample,
        **conditions,
        "evaluation_failed": False,
        "failure_reason": "",
        "known_nodes_total": len(gt_nodes),
        "known_nodes_candidate_matched": len(mapping),
        "known_nodes_final_selected": len(final_selected_nodes),
        "known_edges_total": len(gt_edges),
        "known_edges_endpoints_matched": len(endpoint_edges),
        "known_edges_candidate_present": len(candidate_present),
        "known_edges_final_selected": len(final_selected),
        "known_divisions_total": len(division_parents),
        "known_divisions_candidate_triplet": candidate_divisions,
        "known_divisions_final_selected": final_divisions,
        "nodes_missing_at_detection": len(gt_nodes) - len(mapping),
        "edges_missing_due_to_detection": len(gt_edges) - len(endpoint_edges),
        "edges_missing_in_candidate_graph": len(endpoint_edges) - len(candidate_present),
        "candidate_edges_not_selected_final": len(candidate_present) - len(final_selected),
        "divisions_missing_in_candidate_graph": len(division_parents) - candidate_divisions,
        "candidate_divisions_not_selected_final": candidate_divisions - final_divisions,
        **matching,
    }
    for output, numerator, denominator in RATIO_SPECS:
        row[output] = safe_ratio(int(row[numerator]), int(row[denominator]))
    return row


def failed_sample_row(
    sample: str,
    conditions: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample": sample,
        **conditions,
        "evaluation_failed": True,
        "failure_reason": f"{type(error).__name__}: {error}",
    }
    for column in COUNT_COLUMNS:
        row[column] = 0
    for output, _, _ in RATIO_SPECS:
        row[output] = float("nan")
    row["matching_distance_um_p50"] = float("nan")
    row["matching_distance_um_max"] = float("nan")
    return row


# %% [markdown]
# ## 6. Condition and cross-condition summaries


# %%
def summarise_rows(
    rows: list[dict[str, Any]],
    scope: str,
    group_type: str,
    group_value: str,
) -> dict[str, Any]:
    valid = [row for row in rows if not bool(row["evaluation_failed"])]
    result: dict[str, Any] = {
        "scope": scope,
        "group_type": group_type,
        "group_value": group_value,
        "sample_count": len(rows),
        "valid_sample_count": len(valid),
        "failure_count": len(rows) - len(valid),
    }
    for column in COUNT_COLUMNS:
        result[column] = sum(int(row[column]) for row in valid)
    nan_count = 0
    for output, numerator, denominator in RATIO_SPECS:
        result[output] = safe_ratio(result[numerator], result[denominator])
        nan_count += int(not math.isfinite(float(result[output])))
    result["nan_metric_count"] = nan_count
    return result


def rows_for_scopes(
    rows: list[dict[str, Any]],
    degraded_samples: set[str],
    degraded_embryo: str,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "all_199": rows,
        "degraded_16": [row for row in rows if row["sample"] in degraded_samples],
        f"all_{degraded_embryo}": [row for row in rows if str(row["embryo"]) == degraded_embryo],
        f"degraded_16_{degraded_embryo}": [
            row
            for row in rows
            if row["sample"] in degraded_samples and str(row["embryo"]) == degraded_embryo
        ],
    }


def make_condition_summaries(
    rows: list[dict[str, Any]],
    degraded_samples: set[str],
    degraded_embryo: str,
    condition_columns: list[str],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for scope, scoped_rows in rows_for_scopes(rows, degraded_samples, degraded_embryo).items():
        summaries.append(summarise_rows(scoped_rows, scope, "all", "all"))
        for column in condition_columns:
            values = sorted({str(row[column]) for row in scoped_rows})
            for value in values:
                group = [row for row in scoped_rows if str(row[column]) == value]
                summaries.append(summarise_rows(group, scope, column, value))
    return summaries


def make_cross_summaries(
    rows: list[dict[str, Any]],
    degraded_samples: set[str],
    degraded_embryo: str,
    first: str,
    second: str,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for scope, scoped_rows in rows_for_scopes(rows, degraded_samples, degraded_embryo).items():
        keys = sorted({(str(row[first]), str(row[second])) for row in scoped_rows})
        for first_value, second_value in keys:
            group = [
                row
                for row in scoped_rows
                if str(row[first]) == first_value and str(row[second]) == second_value
            ]
            summary = summarise_rows(
                group,
                scope,
                f"{first}_x_{second}",
                f"{first_value}|{second_value}",
            )
            summary[first] = first_value
            summary[second] = second_value
            summaries.append(summary)
    return summaries


# %% [markdown]
# ## 7. Artifacts, manifest, and metrics


# %%
def run_diagnostic(working_root: Path | None = None) -> dict[str, Any]:
    started = time.monotonic()
    working_root = (working_root or Path.cwd()).resolve()
    config_path = working_root / "config.yaml"
    metrics_path = working_root / "metrics.json"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ensure_geff_runtime_dependencies()
    source_cfg = config["oracle"]["inference_source"]
    exp012_cfg = config["data"]["exp012_readout"]

    inference_root = resolve_inference_root(working_root, source_cfg)
    inference_manifest_path = inference_root / str(source_cfg["manifest"])
    inference_manifest = load_json_object(inference_manifest_path)
    if inference_manifest.get("experiment") != EXPERIMENT:
        raise ValueError("inference manifest experiment mismatch")
    expected_count = int(config["validation"]["expected_sample_count"])
    samples = [str(value) for value in inference_manifest.get("samples", [])]
    if len(samples) != expected_count or samples != sorted(set(samples)):
        raise ValueError("inference manifest sample contract mismatch")

    candidate_root = inference_root / str(source_cfg["candidate_graph_dir"])
    final_root = inference_root / str(source_cfg["final_graph_dir"])
    exp012_root = resolve_exp012_root(working_root, exp012_cfg)
    train_root = resolve_train_dir(working_root)
    conditions, degraded_rows, exp012_evidence = load_fixed_conditions(exp012_root, exp012_cfg)
    if set(conditions) != set(samples):
        raise ValueError("exp012 and inference sample sets differ")

    embryo_counts = Counter(str(row["embryo"]) for row in conditions.values())
    expected_embryo_counts = {
        str(key): int(value)
        for key, value in config["validation"]["expected_embryo_counts"].items()
    }
    if dict(embryo_counts) != expected_embryo_counts:
        raise ValueError(f"embryo count mismatch: {dict(embryo_counts)}")

    matching_cfg = config["validation"]["matching"]
    scale = tuple(float(value) for value in matching_cfg["scale_zyx_um"])
    max_distance_um = float(matching_cfg["max_distance_um"])
    per_sample_rows: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []

    for sample in samples:
        condition = conditions[sample]
        gt_path = train_root / f"{sample}.geff"
        candidate_path = candidate_root / f"{sample}.geff"
        final_path = final_root / f"{sample}.npz"
        try:
            if not gt_path.exists() or not candidate_path.exists() or not final_path.is_file():
                raise FileNotFoundError(
                    {
                        "gt": str(gt_path),
                        "candidate": str(candidate_path),
                        "final": str(final_path),
                    }
                )
            gt_nodes, gt_edges, gt_content_sha = load_geff_tables(gt_path)
            candidate_nodes, candidate_edges, candidate_content_sha = load_geff_tables(
                candidate_path
            )
            (
                final_node_ids,
                _,
                final_edges,
                final_content_sha,
            ) = load_final_graph(final_path)
            row = evaluate_sample_stages(
                sample,
                gt_nodes,
                gt_edges,
                candidate_nodes,
                candidate_edges,
                final_node_ids,
                final_edges,
                condition,
                scale_zyx_um=scale,
                max_distance_um=max_distance_um,
            )
            row.update(
                {
                    "gt_graph_content_sha256": gt_content_sha,
                    "candidate_graph_content_sha256": candidate_content_sha,
                    "final_graph_content_sha256": final_content_sha,
                }
            )
            input_records.append(
                {
                    "sample": sample,
                    "gt_graph_content_sha256": gt_content_sha,
                    "candidate_graph_content_sha256": candidate_content_sha,
                    "final_graph_file_sha256": sha256_file(final_path),
                    "final_graph_content_sha256": final_content_sha,
                }
            )
        except Exception as error:
            row = failed_sample_row(sample, condition, error)
        per_sample_rows.append(row)

    degraded_samples = {str(row["sample"]) for row in degraded_rows}
    degraded_embryo = str(exp012_cfg["expected_degraded_embryo"])
    condition_columns = [str(value) for value in config["oracle"]["condition_columns"]]
    crossing = [str(value) for value in config["oracle"]["crossing"]]
    condition_summaries = make_condition_summaries(
        per_sample_rows,
        degraded_samples,
        degraded_embryo,
        condition_columns,
    )
    cross_summaries = make_cross_summaries(
        per_sample_rows,
        degraded_samples,
        degraded_embryo,
        crossing[0],
        crossing[1],
    )
    overall = summarise_rows(per_sample_rows, "all_199", "all", "all")
    degraded = summarise_rows(
        [row for row in per_sample_rows if row["sample"] in degraded_samples],
        "degraded_16",
        "all",
        "all",
    )
    scope_summaries = [
        row
        for row in condition_summaries
        if row["group_type"] == "all" and row["group_value"] == "all"
    ]

    output_root = working_root / str(config["oracle"]["output_dir"])
    paths = {
        "per_sample": output_root / "per_sample_stage_limits.csv",
        "stage_summary": output_root / "stage_summary.csv",
        "condition": output_root / "condition_stage_summary.csv",
        "cross": output_root / "brightness_candidate_density_summary.csv",
        "degraded": output_root / "degraded16_samples.csv",
        "summary": output_root / "oracle_summary.json",
        "manifest": output_root / "oracle_manifest.json",
    }
    write_csv(paths["per_sample"], per_sample_rows)
    write_csv(paths["stage_summary"], [overall, degraded])
    write_csv(paths["condition"], condition_summaries)
    write_csv(paths["cross"], cross_summaries)
    write_csv(paths["degraded"], degraded_rows)

    input_bundle_sha = sha256_text(
        json.dumps(
            {
                "inference_manifest_sha256": sha256_file(inference_manifest_path),
                "exp012": exp012_evidence,
                "sample_graphs": input_records,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    failed_rows = [row for row in per_sample_rows if bool(row["evaluation_failed"])]
    failure_count = len(failed_rows)
    failure_examples = [
        {"sample": str(row["sample"]), "failure_reason": str(row["failure_reason"])}
        for row in failed_rows[:10]
    ]
    summary = {
        "experiment": EXPERIMENT,
        "diagnostic_only": True,
        "failure_examples": failure_examples,
        "matching": {
            "method": str(matching_cfg["method"]),
            "max_distance_um": max_distance_um,
            "scale_zyx_um": list(scale),
        },
        "overall": overall,
        "degraded_16": degraded,
        "scope_summaries": scope_summaries,
        "condition_summary_rows": len(condition_summaries),
        "brightness_candidate_density_rows": len(cross_summaries),
        "limitations": [
            "The fixed public models were trained on the available training movies.",
            "The result is a conditional diagnostic and is not independent CV.",
            "Unmatched predictions are not counted as false positives because GEFF is sparse.",
            "Ground truth is used only for matching and aggregation, never prediction selection.",
            "Condition overlap prevents causal attribution to brightness or candidate density.",
        ],
    }
    atomic_json(paths["summary"], summary)

    artifact_hashes = {
        f"{name}_sha256": sha256_file(path)
        for name, path in paths.items()
        if name not in {"manifest"}
    }
    manifest = {
        "experiment": EXPERIMENT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "diagnostic_only": True,
        "resource": "kaggle_cpu" if is_kaggle_runtime() else "local_cpu",
        "elapsed_seconds": time.monotonic() - started,
        "sample_count": len(per_sample_rows),
        "degraded_sample_count": len(degraded_rows),
        "failure_count": failure_count,
        "input_bundle_sha256": input_bundle_sha,
        "inputs": {
            "inference_manifest_sha256": sha256_file(inference_manifest_path),
            "exp012": exp012_evidence,
            "sample_graphs": input_records,
        },
        "artifact_sha256": artifact_hashes,
    }
    atomic_json(paths["manifest"], manifest)
    manifest_sha = sha256_file(paths["manifest"])

    metrics = load_json_object(metrics_path)
    metrics_update = {
        "diagnostics": {
            "oracle_stage_limits": {
                "diagnostic_only": True,
                "sample_count": len(per_sample_rows),
                "degraded_sample_count": len(degraded_rows),
                "failure_count": failure_count,
                "overall": overall,
                "degraded_16": degraded,
            }
        },
        "evidence": {
            "kaggle": {
                "kernel_source_ids": [
                    str(source_cfg["kernel_source"]),
                    str(exp012_cfg["kernel_source"]),
                ],
                "resource": "cpu",
                "notebook_runtime_seconds": time.monotonic() - started,
                "internet_enabled": False,
            },
            "artifacts": {
                "input_file_sha": input_bundle_sha,
                "feature_schema_sha": sha256_text(
                    json.dumps(
                        {
                            "count_columns": COUNT_COLUMNS,
                            "ratio_specs": RATIO_SPECS,
                            "condition_columns": condition_columns,
                        },
                        sort_keys=True,
                    )
                ),
                "feature_content_sha": artifact_hashes["per_sample_sha256"],
                "row_count": len(per_sample_rows),
                "group_count": len(condition_summaries) + len(cross_summaries),
                "model_count": 0,
                "model_shas": config["model"]["checkpoint_sha256"],
                "selected_mode": "fixed_public_graph_oracle_diagnostic",
                "selected_model": config["model"]["name"],
                "submission_sha": None,
                "oracle_manifest_sha": manifest_sha,
                **artifact_hashes,
            },
        },
        "notes": (
            "Oracle diagnostic executed without changing predictions; "
            "results remain pending user completion judgment."
        ),
    }
    if metrics.get("status") not in {
        "usable",
        "completed",
        "deprecated",
        "discarded",
        "leak-risk",
    }:
        metrics_update["status"] = "debug_completed" if failure_count == 0 else "failed"
    atomic_json(metrics_path, deep_merge(metrics, metrics_update))
    diagnostic_receipt = {
        "experiment": EXPERIMENT,
        "resource": manifest["resource"],
        "elapsed_seconds": manifest["elapsed_seconds"],
        "sample_count": manifest["sample_count"],
        "failure_count": manifest["failure_count"],
        "input_bundle_sha256": input_bundle_sha,
        "inference_manifest_sha256": manifest["inputs"]["inference_manifest_sha256"],
        "exp012": exp012_evidence,
        "artifact_sha256": artifact_hashes,
        "oracle_manifest_sha256": manifest_sha,
    }
    print(
        "DIAGNOSTIC_RECEIPT "
        + json.dumps(json_safe(diagnostic_receipt), ensure_ascii=False, sort_keys=True)
    )
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))
    print("Oracle manifest:", paths["manifest"])
    if failure_count:
        raise RuntimeError(
            f"oracle diagnostic failed for {failure_count} samples: {failure_examples}"
        )
    return summary


if __name__ == "__main__":
    run_diagnostic()
