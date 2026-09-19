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
# # Division triplet teacher audit
#
# Fixed detector centers from exp015 are the only prediction input. Organizer
# annotations are used after candidate generation to measure coverage and
# count which labels have direct support from annotated edges.
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

EXPERIMENT = "exp021_division_teacher_audit"
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
# Candidate IDs and geometry are fixed without GT. Matching and teacher
# classifications are diagnostics performed after enumeration.


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


def build_incoming(
    outgoing: dict[int, tuple[int, ...]],
    gt_by_frame: dict[int, dict[int, np.ndarray]],
) -> dict[int, int]:
    node_time = {node: frame for frame, nodes in gt_by_frame.items() for node in nodes}
    incoming: dict[int, int] = {}
    for mother, daughters in outgoing.items():
        if len(daughters) > 2:
            raise ValueError(f"GT mother has more than two outgoing edges: {mother}")
        for daughter in daughters:
            if node_time[daughter] != node_time[mother] + 1:
                raise ValueError(f"nonadjacent GT edge: {mother} -> {daughter}")
            if daughter in incoming:
                raise ValueError(f"GT node has multiple incoming edges: {daughter}")
            incoming[daughter] = mother
    return incoming


def known_divisions_in_window(
    frame: int,
    gt_by_frame: dict[int, dict[int, np.ndarray]],
    outgoing: dict[int, tuple[int, ...]],
) -> list[tuple[int, int, int]]:
    current = gt_by_frame.get(frame, {})
    following = gt_by_frame.get(frame + 1, {})
    return [
        (mother, daughters[0], daughters[1])
        for mother, daughters in outgoing.items()
        if mother in current
        and len(daughters) == 2
        and all(daughter in following for daughter in daughters)
    ]


def classify_triplet(
    candidate: tuple[int, int, int],
    source_inverse: dict[int, int],
    target_inverse: dict[int, int],
    target_match: dict[int, int],
    outgoing: dict[int, tuple[int, ...]],
    incoming: dict[int, int],
) -> dict[str, bool]:
    source, first, second = candidate
    mother = source_inverse.get(source)
    result = {
        "gt_parent": mother is not None,
        "both_daughters_gt": False,
        "known_positive": False,
        "strict_wrong": False,
        "foreign_parent_wrong": False,
        "foreign_both_daughters_gt": False,
        "single_outgoing_parent": False,
        "single_outgoing_with_known_child": False,
    }
    if mother is None:
        return result
    daughter_gt = (target_inverse.get(first), target_inverse.get(second))
    both_matched = all(node is not None for node in daughter_gt)
    result["both_daughters_gt"] = both_matched
    daughters = outgoing.get(mother, ())
    complete_division = len(daughters) == 2 and all(
        daughter in target_match for daughter in daughters
    )
    if complete_division and both_matched:
        result["known_positive"] = set(daughter_gt) == set(daughters)
        result["strict_wrong"] = not result["known_positive"]
    foreign = any(
        node is not None and node in incoming and incoming[node] != mother for node in daughter_gt
    )
    result["foreign_parent_wrong"] = foreign
    result["foreign_both_daughters_gt"] = foreign and both_matched
    result["single_outgoing_parent"] = len(daughters) == 1
    result["single_outgoing_with_known_child"] = len(daughters) == 1 and daughters[0] in daughter_gt
    return result


# %% [markdown]
# ## 5. Run the teacher audit
#
# All samples and adjacent windows must be present. The strict counts from
# exp020 are checked before the additional teacher categories are accepted.


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
    if list(geometry) != ["safe_repair"]:
        raise ValueError("teacher audit requires only the fixed safe_repair geometry")
    limits = geometry["safe_repair"]
    rows: list[dict[str, Any]] = []
    gt_receipts = []
    annotation_columns = set()
    scale = np.asarray(validation["voxel_scale_zyx_um"], dtype=np.float64)
    by_sample: dict[str, list[Path]] = defaultdict(list)
    for path in window_paths:
        by_sample[path.parent.name].append(path)

    count_fields = (
        "window_count",
        "triplet_count",
        "known_divisions",
        "three_centers_matched",
        "known_positive",
        "strict_wrong",
        "gt_parent_triplets",
        "both_daughters_gt_triplets",
        "foreign_parent_wrong",
        "foreign_both_daughters_gt",
        "foreign_one_daughter_gt",
        "foreign_on_division_parent",
        "foreign_on_single_edge_parent",
        "foreign_on_no_edge_parent",
        "positive_mothers_with_foreign_wrong",
        "positive_mothers_with_strict_wrong",
        "strict_foreign_overlap",
        "confirmed_wrong_union",
        "foreign_wrong_mothers",
        "single_outgoing_annotations",
        "single_outgoing_parent_triplets",
        "single_outgoing_with_known_child",
    )
    for sample_number, sample in enumerate(samples, start=1):
        paths = by_sample[sample]
        if len(paths) != int(validation["expected_frames_per_sample"]) - 1:
            raise ValueError(f"incomplete windows for {sample}")
        gt_path = train_dir / f"{sample}.geff"
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)
        gt_frames, outgoing, gt_sha, columns = load_gt(gt_path, scale)
        incoming = build_incoming(outgoing, gt_frames)
        gt_receipts.append([sample, gt_sha])
        annotation_columns.update(columns)
        frame_registry: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        frame_matches: dict[int, dict[int, int]] = {}
        row = {
            "sample": sample,
            "embryo": sample.split("_", 1)[0],
            "max_triplets_per_parent": 0,
            **dict.fromkeys(count_fields, 0),
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
            source_match = frame_matches[source_frame]
            target_match = frame_matches[target_frame]
            source_inverse = {candidate: gt for gt, candidate in source_match.items()}
            target_inverse = {candidate: gt for gt, candidate in target_match.items()}
            divisions = known_divisions_in_window(source_frame, gt_frames, outgoing)
            row["known_divisions"] += len(divisions)
            row["three_centers_matched"] += sum(
                mother in source_match and first in target_match and second in target_match
                for mother, first, second in divisions
            )
            row["single_outgoing_annotations"] += sum(
                mother in gt_frames.get(source_frame, {}) and len(daughters) == 1
                for mother, daughters in outgoing.items()
            )
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
            foreign_mothers = set()
            positive_mothers = set()
            strict_mothers = set()
            for candidate in sorted(candidates):
                classes = classify_triplet(
                    candidate, source_inverse, target_inverse, target_match, outgoing, incoming
                )
                row["gt_parent_triplets"] += int(classes["gt_parent"])
                row["both_daughters_gt_triplets"] += int(classes["both_daughters_gt"])
                row["known_positive"] += int(classes["known_positive"])
                if classes["known_positive"]:
                    positive_mothers.add(candidate[0])
                if classes["strict_wrong"]:
                    strict_mothers.add(candidate[0])
                row["strict_wrong"] += int(classes["strict_wrong"])
                row["foreign_parent_wrong"] += int(classes["foreign_parent_wrong"])
                row["foreign_both_daughters_gt"] += int(classes["foreign_both_daughters_gt"])
                row["foreign_one_daughter_gt"] += int(
                    classes["foreign_parent_wrong"] and not classes["both_daughters_gt"]
                )
                row["strict_foreign_overlap"] += int(
                    classes["strict_wrong"] and classes["foreign_parent_wrong"]
                )
                row["confirmed_wrong_union"] += int(
                    classes["strict_wrong"] or classes["foreign_parent_wrong"]
                )
                row["single_outgoing_parent_triplets"] += int(classes["single_outgoing_parent"])
                row["single_outgoing_with_known_child"] += int(
                    classes["single_outgoing_with_known_child"]
                )
                if classes["foreign_parent_wrong"]:
                    foreign_mothers.add(candidate[0])
                    degree = len(outgoing.get(source_inverse[candidate[0]], ()))
                    if degree == 2:
                        row["foreign_on_division_parent"] += 1
                    elif degree == 1:
                        row["foreign_on_single_edge_parent"] += 1
                    else:
                        row["foreign_on_no_edge_parent"] += 1
            row["foreign_wrong_mothers"] += len(foreign_mothers)
            row["positive_mothers_with_foreign_wrong"] += len(positive_mothers & foreign_mothers)
            row["positive_mothers_with_strict_wrong"] += len(positive_mothers & strict_mothers)
        if len(frame_registry) != int(validation["expected_frames_per_sample"]):
            raise ValueError(f"frame coverage differs for {sample}")
        rows.append(row)
        if sample_number % 20 == 0 or sample_number == len(samples):
            print(f"audited {sample_number}/{len(samples)} samples", flush=True)

    totals = {}
    for group in ("all", *sorted(embryo_counts)):
        subset = rows if group == "all" else [row for row in rows if row["embryo"] == group]
        counts = {key: sum(int(row[key]) for row in subset) for key in count_fields}
        counts["sample_count"] = len(subset)
        counts["max_triplets_per_parent"] = max(row["max_triplets_per_parent"] for row in subset)
        totals[group] = counts
    for group, counts in totals.items():
        if counts["foreign_parent_wrong"] != (
            counts["foreign_on_division_parent"]
            + counts["foreign_on_single_edge_parent"]
            + counts["foreign_on_no_edge_parent"]
        ):
            raise ValueError(f"foreign parent degree partition differs for {group}")
        if counts["foreign_parent_wrong"] != (
            counts["foreign_both_daughters_gt"] + counts["foreign_one_daughter_gt"]
        ):
            raise ValueError(f"foreign daughter match partition differs for {group}")
        if counts["confirmed_wrong_union"] != (
            counts["strict_wrong"]
            + counts["foreign_parent_wrong"]
            - counts["strict_foreign_overlap"]
        ):
            raise ValueError(f"negative union differs for {group}")
        if counts["positive_mothers_with_foreign_wrong"] > counts["known_positive"]:
            raise ValueError(f"positive mother overlap exceeds positives for {group}")

    expected = {
        "triplet_count": int(validation["expected_triplet_count"]),
        "known_divisions": int(validation["expected_known_divisions"]),
        "known_positive": int(validation["expected_positive_triplets"]),
        "strict_wrong": int(validation["expected_strict_wrong_pairs"]),
    }
    if {key: totals["all"][key] for key in expected} != expected:
        raise ValueError(f"exp020 result not reproduced: expected {expected}, got {totals['all']}")
    gt_bundle_sha = json_sha(gt_receipts)
    if gt_bundle_sha != str(validation["expected_gt_content_bundle_sha256"]):
        raise ValueError("GT content bundle differs from exp020")
    summary = {
        "experiment": EXPERIMENT,
        "diagnostic_only": True,
        "geometry": geometry,
        "totals": totals,
        "cache_summary_sha256": file_sha(cache_root / config["data"]["cache"]["summary_file"]),
        "cache_identity_sha256": identity,
        "gt_content_bundle_sha256": gt_bundle_sha,
        "gt_node_columns": sorted(annotation_columns),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Single-outgoing GT mothers do not establish absence of a second daughter.",
            "Foreign-parent negatives require correct GT edges, one-to-one matching, "
            "and no merges.",
            "Individual triplets are not a full-graph evaluation of division false positives.",
            "Public detector saw train embryos; this is not independent CV.",
        ],
    }
    return rows, summary


# %% [markdown]
# ## 6. Summaries and artifacts
#
# The per-sample CSV, aggregate JSON, and SHA manifest are the only outputs.
# No model, predictions, or submission file is written.


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
