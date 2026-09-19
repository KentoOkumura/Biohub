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
# # Division triplet candidate diagnostic
#
# Fixed detector centers from exp015 are the only prediction input. Organizer
# annotations are used after candidate generation to measure coverage and
# inspect which labels are safe to use.
#
# ## Contents
# 1. Imports and runtime dependencies
# 2. Configuration and input checks
# 3. Cache and annotation readers
# 4. Candidate generation and annotation matching
# 5. Run the diagnostic
# 6. Summaries and artifacts

# %% [markdown]
# ## 1. Imports and runtime dependencies

# %%
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

EXPERIMENT = "exp020_division_triplet_candidates"
COMPETITION = "biohub-cell-tracking-during-development"
OFFLINE_MODULES = (
    "tracksdata",
    "zarr",
    "pyscipopt",
    "geff",
    "geff_spec",
    "ilpy",
    "polars",
    "imagecodecs",
    "rustworkx",
    "numcodecs",
    "donfig",
    "bidict",
)
OFFLINE_SPECS = (
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


def kaggle_runtime() -> bool:
    return Path("/kaggle/input").is_dir() and Path("/kaggle/working").is_dir()


def offline_wheel_dirs() -> list[Path]:
    paths = [
        Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/wheels"),
        Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1/wheels"),
    ]
    return [path for path in paths if path.is_dir() and any(path.glob("*.whl"))]


def pip_offline(directories: list[Path], packages: tuple[str, ...], reinstall: bool) -> None:
    command = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
    if reinstall:
        command.append("--force-reinstall")
    for directory in directories:
        command.extend(["--find-links", str(directory)])
    command.extend(packages)
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        print((result.stderr or "")[-2000:])
        raise RuntimeError("offline GEFF dependency installation failed")


def ensure_geff() -> None:
    os.environ.setdefault("POLARS_PREFER_PKG", "32")
    failures = []
    for module in OFFLINE_MODULES:
        try:
            importlib.import_module(module)
        except Exception:
            failures.append(module)
    try:
        import polars as pl

        polars_ready = hasattr(pl, "Float16")
    except Exception:
        polars_ready = False
    if not failures and polars_ready:
        return
    if not kaggle_runtime():
        raise ImportError(f"missing local GEFF packages: {failures}")
    wheels = offline_wheel_dirs()
    if not wheels:
        raise FileNotFoundError("support dataset offline wheels are missing")
    if not polars_ready:
        pip_offline(wheels, ("polars>=1.36", "polars-runtime-32"), reinstall=True)
        for name in list(sys.modules):
            if name == "polars" or name.startswith("polars."):
                sys.modules.pop(name, None)
    pip_offline(wheels, OFFLINE_SPECS, reinstall=False)
    for name in list(sys.modules):
        if any(name == root or name.startswith(root + ".") for root in OFFLINE_MODULES):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    import polars as pl

    if not hasattr(pl, "Float16"):
        raise ImportError("offline Polars runtime has no Float16")
    import tracksdata  # noqa: F401


# %% [markdown]
# ## 2. Configuration and input checks
#
# The three geometric ranges are diagnostics fixed before reading annotations.
# They are not selected for training or inference by this notebook.


# %%
def json_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_path() -> Path:
    paths = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "experiments" / EXPERIMENT / "config.yaml",
    ]
    matches = [path for path in paths if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one config.yaml, found {matches}")
    return matches[0]


def unique_path(paths: list[Path], label: str) -> Path:
    matches = sorted({path.resolve() for path in paths if path.exists()})
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label}, found {matches}")
    return matches[0]


def resolve_cache_root(config: dict[str, Any]) -> Path:
    cache = config["data"]["cache"]
    slug = str(cache["kernel_source"]).split("/", 1)[-1]
    candidates = [
        path.parent
        for path in Path("/kaggle/input/notebooks").rglob(str(cache["summary_file"]))
        if slug in path.as_posix()
    ]
    candidates.append(
        Path.cwd().parent / "exp015_oracle_stage_limits" / "artifacts" / "inference_v1"
    )
    return unique_path(candidates, "exp015 cache output")


def resolve_train_dir() -> Path:
    return unique_path(
        [
            Path(f"/kaggle/input/competitions/{COMPETITION}/train"),
            Path(f"/kaggle/input/{COMPETITION}/train"),
            Path.cwd().parent.parent / "data" / "raw" / "train",
        ],
        "competition train directory",
    )


def check_cache_summary(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    cache = config["data"]["cache"]
    path = root / str(cache["summary_file"])
    summary = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(summary)
    recorded_sha = unsigned.pop("summary_sha256", None)
    if recorded_sha != json_sha(unsigned) or recorded_sha != str(cache["summary_sha256"]):
        raise ValueError("exp015 window cache summary content SHA changed")
    expected = {
        "schema_version": int(cache["schema_version"]),
        "dataset_count": int(config["validation"]["expected_sample_count"]),
        "window_count": int(config["validation"]["expected_window_count"]),
        "cache_identity_sha256": str(cache["identity_sha256"]),
    }
    if {key: summary.get(key) for key in expected} != expected:
        raise ValueError("exp015 cache summary contract changed")
    return summary


# %% [markdown]
# ## 3. Cache and annotation readers
#
# Each adjacent window is checked against its metadata. The same frame's
# candidate IDs and physical coordinates must agree across overlapping windows.


# %%
def cache_identity_record(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(saved["__metadata_json__"].tobytes().decode("utf-8"))
    schema = metadata["array_schema"]
    by_name = {str(item["name"]): item for item in schema}
    feature_names = (
        "primary_features_src",
        "primary_features_tgt",
        "secondary_features_src",
        "secondary_features_tgt",
    )
    frames = metadata["window_frames"]
    if metadata["dataset"] != path.parent.name:
        raise ValueError(f"cache sample metadata differs: {path}")
    if path.name != f"{int(frames[0]):06d}_{int(frames[1]):06d}.npz":
        raise ValueError(f"cache window filename differs: {path}")
    return {
        "dataset": path.parent.name,
        "window_frames": [int(value) for value in frames],
        "candidate_count_src": int(by_name["candidate_mask_src"]["shape"][0]),
        "candidate_count_tgt": int(by_name["candidate_mask_tgt"]["shape"][0]),
        "feature_values": int(
            sum(np.prod(by_name[name]["shape"], dtype=np.int64) for name in feature_names)
        ),
        "cache_bytes": int(path.stat().st_size),
        "cache_schema_sha256": json_sha(schema),
        "cache_content_sha256": str(metadata["array_content_sha256"]),
    }


def read_window(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(saved["__metadata_json__"].tobytes().decode("utf-8"))
        fields = {}
        for side in ("src", "tgt"):
            fields[side] = {
                "ids": np.asarray(saved[f"candidate_ids_{side}"], dtype=np.int64),
                "coords": np.asarray(saved[f"coords_{side}_physical"], dtype=np.float64),
                "mask": np.asarray(saved[f"candidate_mask_{side}"], dtype=bool),
            }
    frames = tuple(int(value) for value in metadata["window_frames"])
    if len(frames) != 2 or frames[1] != frames[0] + 1:
        raise ValueError(f"nonadjacent cache window: {path}")
    for side in ("src", "tgt"):
        value = fields[side]
        if value["coords"].shape != (len(value["ids"]), 3):
            raise ValueError(f"invalid candidate coordinates: {path} {side}")
        if len(set(map(int, value["ids"]))) != len(value["ids"]):
            raise ValueError(f"duplicate candidate IDs: {path} {side}")
        if not value["mask"].all():
            raise ValueError(f"padded candidates in cache: {path} {side}")
        if not np.isfinite(value["coords"]).all():
            raise ValueError(f"nonfinite candidate coordinates: {path} {side}")
    return {"frames": frames, **fields}


def load_gt(
    path: Path, scale_zyx: np.ndarray
) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, tuple[int, ...]], str, list[str]]:
    import tracksdata as td

    graph = td.graph.IndexedRXGraph.from_geff(path)
    if isinstance(graph, tuple):
        graph = graph[0]
    frame_nodes: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    raw_nodes = []
    node_times = {}
    node_columns = list(graph.node_attrs().columns)
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        frame = int(row["t"])
        raw = np.asarray([row["z"], row["y"], row["x"]], dtype=np.float64)
        if node_id in node_times or not np.isfinite(raw).all():
            raise ValueError(f"duplicate or invalid GT node: {path} {node_id}")
        node_times[node_id] = frame
        frame_nodes[frame][node_id] = raw * scale_zyx
        raw_nodes.append([node_id, frame, *map(float, raw)])
    outgoing: dict[int, list[int]] = defaultdict(list)
    raw_edges = []
    for row in graph.edge_attrs().iter_rows(named=True):
        source, target = int(row["source_id"]), int(row["target_id"])
        if source not in node_times or target not in node_times:
            raise ValueError(f"dangling GT edge: {path}")
        outgoing[source].append(target)
        raw_edges.append([source, target])
    canonical = {
        "nodes": sorted(raw_nodes, key=lambda row: row[0]),
        "edges": sorted(raw_edges),
    }
    return (
        frame_nodes,
        {source: tuple(sorted(targets)) for source, targets in outgoing.items()},
        json_sha(canonical),
        node_columns,
    )


# %% [markdown]
# ## 4. Candidate generation and annotation matching
#
# Daughter order is removed by sorting the two target candidate IDs. Ground
# truth is read only after the candidate set is fixed for that window.


# %%
def match_frame(
    gt_nodes: dict[int, np.ndarray],
    candidate_ids: np.ndarray,
    candidate_coords: np.ndarray,
    radius: float,
) -> dict[int, int]:
    if not gt_nodes or len(candidate_ids) == 0:
        return {}
    gt_ids = sorted(gt_nodes)
    gt_coords = np.asarray([gt_nodes[node_id] for node_id in gt_ids], dtype=np.float64)
    distances = np.linalg.norm(gt_coords[:, None, :] - candidate_coords[None, :, :], axis=2)
    feasible = distances <= radius
    gated = np.where(feasible, distances, 1.0e9)
    rows, columns = linear_sum_assignment(gated)
    return {
        gt_ids[int(row)]: int(candidate_ids[int(column)])
        for row, column in zip(rows, columns, strict=True)
        if distances[row, column] <= radius
    }


def triplets_for_window(
    source_ids: np.ndarray,
    source_coords: np.ndarray,
    target_ids: np.ndarray,
    target_coords: np.ndarray,
    parent_max_um: float,
    sister_max_um: float,
    max_per_parent: int,
) -> tuple[set[tuple[int, int, int]], int]:
    if len(source_ids) == 0 or len(target_ids) < 2:
        return set(), 0
    tree = cKDTree(target_coords)
    neighbors = tree.query_ball_point(source_coords, parent_max_um)
    triplets: set[tuple[int, int, int]] = set()
    max_observed = 0
    for source_id, nearby in zip(source_ids, neighbors, strict=True):
        count = 0
        for left, right in combinations(sorted(nearby), 2):
            if np.linalg.norm(target_coords[left] - target_coords[right]) > sister_max_um:
                continue
            first, second = sorted((int(target_ids[left]), int(target_ids[right])))
            triplets.add((int(source_id), first, second))
            count += 1
        max_observed = max(max_observed, count)
        if count > max_per_parent:
            raise ValueError(f"triplet guard exceeded for parent {source_id}: {count}")
    return triplets, max_observed


def true_divisions(
    frame: int, gt_by_frame: dict[int, dict[int, np.ndarray]], outgoing: dict[int, tuple[int, ...]]
) -> list[tuple[int, int, int]]:
    current = gt_by_frame.get(frame, {})
    following = gt_by_frame.get(frame + 1, {})
    return [
        (source, targets[0], targets[1])
        for source, targets in outgoing.items()
        if source in current
        and len(targets) == 2
        and all(target in following for target in targets)
    ]


def evaluate_known_division(
    truth: tuple[int, int, int],
    source_match: dict[int, int],
    target_match: dict[int, int],
    triplets: set[tuple[int, int, int]],
) -> tuple[bool, bool, int]:
    mother, first, second = truth
    matched = mother in source_match and first in target_match and second in target_match
    if not matched:
        return False, False, 0
    target_pair = tuple(sorted((target_match[first], target_match[second])))
    positive = (source_match[mother], *target_pair)
    inverse_targets = set(target_match.values())
    wrong = sum(
        1
        for candidate in triplets
        if candidate[0] == positive[0]
        and candidate != positive
        and candidate[1] in inverse_targets
        and candidate[2] in inverse_targets
    )
    return True, positive in triplets, wrong


# %% [markdown]
# ## 5. Run the diagnostic
#
# All samples and 99 windows per sample are required. A partial result is a
# failure, not an estimated coverage rate.


# %%
def run_diagnostic(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    validation = config["validation"]
    cache_root = resolve_cache_root(config)
    train_dir = resolve_train_dir()
    check_cache_summary(cache_root, config)
    window_paths = sorted((cache_root / config["data"]["cache"]["directory"]).glob("*/*.npz"))
    samples = sorted({path.parent.name for path in window_paths})
    if len(window_paths) != int(validation["expected_window_count"]):
        raise ValueError("window cache count mismatch")
    if len(samples) != int(validation["expected_sample_count"]):
        raise ValueError("sample count mismatch")
    embryo_counts = dict(Counter(name.split("_", 1)[0] for name in samples))
    if embryo_counts != {
        str(key): int(value) for key, value in validation["expected_embryo_counts"].items()
    }:
        raise ValueError(f"embryo sample counts differ: {embryo_counts}")
    identity = json_sha([cache_identity_record(path) for path in window_paths])
    if identity != str(config["data"]["cache"]["identity_sha256"]):
        raise ValueError("exp015 cache identity differs")

    geometry = validation["geometry"]
    rows = []
    gt_receipts = []
    annotation_columns = set()
    scale = np.asarray(validation["voxel_scale_zyx_um"], dtype=np.float64)
    by_sample: dict[str, list[Path]] = defaultdict(list)
    for path in window_paths:
        by_sample[path.parent.name].append(path)

    for sample_number, sample in enumerate(samples, start=1):
        paths = by_sample[sample]
        if len(paths) != int(validation["expected_frames_per_sample"]) - 1:
            raise ValueError(f"incomplete windows for {sample}")
        gt_path = train_dir / f"{sample}.geff"
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)
        gt_frames, outgoing, gt_sha, columns = load_gt(gt_path, scale)
        gt_receipts.append([sample, gt_sha])
        annotation_columns.update(columns)
        frame_registry: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        frame_matches: dict[int, dict[int, int]] = {}
        sample_rows = {
            name: {
                "sample": sample,
                "embryo": sample.split("_", 1)[0],
                "geometry": name,
                "window_count": 0,
                "triplet_count": 0,
                "max_triplets_per_parent": 0,
                "known_divisions": 0,
                "three_centers_matched": 0,
                "triplet_recovered": 0,
                "known_wrong_pairs": 0,
                "single_outgoing_annotations": 0,
                "confirmed_continuations": 0,
            }
            for name in geometry
        }
        for path in paths:
            window = read_window(path)
            source_frame, target_frame = window["frames"]
            for frame, side in ((source_frame, "src"), (target_frame, "tgt")):
                value = window[side]
                if frame in frame_registry:
                    ids, coords = frame_registry[frame]
                    if not np.array_equal(ids, value["ids"]) or not np.array_equal(
                        coords, value["coords"]
                    ):
                        raise ValueError(f"overlapping cache frames disagree: {sample} {frame}")
                else:
                    frame_registry[frame] = (value["ids"], value["coords"])
                    frame_matches[frame] = match_frame(
                        gt_frames.get(frame, {}),
                        value["ids"],
                        value["coords"],
                        float(validation["matching_radius_um"]),
                    )
            divisions = true_divisions(source_frame, gt_frames, outgoing)
            singles = sum(
                1
                for mother, children in outgoing.items()
                if mother in gt_frames.get(source_frame, {})
                and len(children) == 1
                and children[0] in gt_frames.get(target_frame, {})
            )
            for name, limits in geometry.items():
                row = sample_rows[name]
                candidates, max_parent = triplets_for_window(
                    window["src"]["ids"],
                    window["src"]["coords"],
                    window["tgt"]["ids"],
                    window["tgt"]["coords"],
                    float(limits["parent_daughter_max_um"]),
                    float(limits["sister_max_um"]),
                    int(validation["max_triplets_per_parent_guard"]),
                )
                row["window_count"] += 1
                row["triplet_count"] += len(candidates)
                row["max_triplets_per_parent"] = max(row["max_triplets_per_parent"], max_parent)
                row["known_divisions"] += len(divisions)
                row["single_outgoing_annotations"] += singles
                for truth in divisions:
                    matched, recovered, wrong = evaluate_known_division(
                        truth,
                        frame_matches[source_frame],
                        frame_matches[target_frame],
                        candidates,
                    )
                    row["three_centers_matched"] += int(matched)
                    row["triplet_recovered"] += int(recovered)
                    row["known_wrong_pairs"] += wrong
        if len(frame_registry) != int(validation["expected_frames_per_sample"]):
            raise ValueError(f"frame coverage differs for {sample}")
        rows.extend(sample_rows.values())
        if sample_number % 20 == 0 or sample_number == len(samples):
            print(f"diagnosed {sample_number}/{len(samples)} samples", flush=True)

    totals = {}
    for name in geometry:
        selected = [row for row in rows if row["geometry"] == name]
        by_embryo = {}
        for group in ("all", *sorted(embryo_counts)):
            subset = (
                selected if group == "all" else [row for row in selected if row["embryo"] == group]
            )
            counts = {
                key: sum(int(row[key]) for row in subset)
                for key in (
                    "window_count",
                    "triplet_count",
                    "known_divisions",
                    "three_centers_matched",
                    "triplet_recovered",
                    "known_wrong_pairs",
                    "single_outgoing_annotations",
                    "confirmed_continuations",
                )
            }
            counts["sample_count"] = len(subset)
            counts["max_triplets_per_parent"] = max(
                row["max_triplets_per_parent"] for row in subset
            )
            counts["triplet_recall"] = (
                counts["triplet_recovered"] / counts["known_divisions"]
                if counts["known_divisions"]
                else None
            )
            by_embryo[group] = counts
        totals[name] = by_embryo
    if any(totals[name]["all"]["known_divisions"] != 151 for name in geometry):
        raise ValueError("known division count differs from exp015's 151")
    summary = {
        "experiment": EXPERIMENT,
        "diagnostic_only": True,
        "geometry": geometry,
        "totals": totals,
        "cache_summary_sha256": file_sha(cache_root / config["data"]["cache"]["summary_file"]),
        "cache_identity_sha256": identity,
        "gt_content_bundle_sha256": json_sha(gt_receipts),
        "gt_node_columns": sorted(annotation_columns),
        "confirmed_continuation_rule": None,
        "confirmed_continuations": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Sparse GEFF has no confirmed mask for absent second daughters.",
            "Wrong pairs require two known daughters and GT-matched candidates.",
            "Public detector saw train embryos; this is not independent CV.",
        ],
    }
    return rows, summary


# %% [markdown]
# ## 6. Summaries and artifacts
#
# Output includes a per-sample table, aggregate coverage, input identity, and
# file hashes. No model or submission file is written.


# %%
def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + chr(10),
        encoding="utf-8",
    )


def save_results(rows: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    output_root = Path("/kaggle/working") if kaggle_runtime() else Path.cwd()
    output_dir = output_root / EXPERIMENT / "artifacts" / "diagnostic_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "per_sample.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    manifest = {
        "experiment": EXPERIMENT,
        "sample_rows": len(rows),
        "summary_sha256": file_sha(summary_path),
        "per_sample_sha256": file_sha(table_path),
        "cache_identity_sha256": summary["cache_identity_sha256"],
        "gt_content_bundle_sha256": summary["gt_content_bundle_sha256"],
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(summary["totals"], indent=2, sort_keys=True), flush=True)
    print(f"diagnostic output: {output_dir}", flush=True)
    return output_dir


if __name__ == "__main__":
    ensure_geff()
    cfg = yaml.safe_load(config_path().read_text(encoding="utf-8"))
    diagnostic_rows, diagnostic_summary = run_diagnostic(cfg)
    save_results(diagnostic_rows, diagnostic_summary)
