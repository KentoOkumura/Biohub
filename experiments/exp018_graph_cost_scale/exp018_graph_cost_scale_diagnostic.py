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
# # exp018 graph cost scale diagnostic
#
# This CPU diagnostic re-solves the fixed exp015 pre-ILP candidate graphs.
# Only the scalar multiplying `edge_prob` changes. Candidate generation,
# event costs, ILP constraints, and the official metric stay fixed.
#
# The first stage stops at the raw ILP solution. It does not run motion, gap,
# division, or DeepCenter repair, and it does not create a competition
# submission.
#
# ## Contents
#
# 1. Imports and offline runtime dependencies
# 2. SHA-pinned input and evaluator discovery
# 3. Fixed ILP solve and official metric
# 4. Cross-embryo selection and promotion gate
# 5. Diagnostic execution
# 6. Artifacts, manifest, and metrics

# %% [markdown]
# ## 1. Imports and offline runtime dependencies

# %%
from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

EXPERIMENT = "exp018_graph_cost_scale"
SOURCE_EXPERIMENT = "exp015_oracle_stage_limits"
COMPETITION = "biohub-cell-tracking-during-development"


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
        raise FileNotFoundError("offline support-pack wheel directory was not found")
    print("Offline wheel dirs:", [str(path) for path in wheel_dirs])
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


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        json_safe(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256_text(payload)


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
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
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
# ## 2. SHA-pinned input and evaluator discovery
#
# The candidate graphs were generated without ground-truth access. Ground truth
# enters only below, after the cost variants are fixed, for the official metric.


# %%
def resolve_inference_root(working_root: Path, source_cfg: dict[str, Any]) -> Path:
    manifest_name = str(source_cfg["manifest"])
    local = (working_root / str(source_cfg["local_artifact_dir"])).resolve()
    candidates = [local] if (local / manifest_name).is_file() else []
    if is_kaggle_runtime():
        kernel_slug = str(source_cfg["kernel_source"]).split("/", 1)[-1]
        candidates.extend(
            path.parent
            for path in Path("/kaggle/input").rglob(manifest_name)
            if kernel_slug in path.as_posix()
        )
    return unique_existing(candidates, "exp015 inference output root")


def resolve_exp012_root(working_root: Path, source_cfg: dict[str, Any]) -> Path:
    local = (working_root / str(source_cfg["local_dir"])).resolve()
    candidates = [local] if (local / "per_sample_readout.csv").is_file() else []
    if is_kaggle_runtime():
        kernel_slug = str(source_cfg["kernel_source"]).split("/", 1)[-1]
        candidates.extend(
            path.parent
            for path in Path("/kaggle/input").rglob("per_sample_readout.csv")
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


def verify_exp015_manifest(
    inference_root: Path,
    source_cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[str], Path, dict[str, Any]]:
    manifest_path = inference_root / str(source_cfg["manifest"])
    observed_file_sha = sha256_file(manifest_path)
    if observed_file_sha != str(source_cfg["manifest_file_sha256"]):
        raise ValueError("exp015 inference manifest file SHA mismatch")
    manifest = load_json_object(manifest_path)
    claimed_payload_sha = str(manifest.get("manifest_sha256", ""))
    payload_without_sha = dict(manifest)
    payload_without_sha.pop("manifest_sha256", None)
    observed_payload_sha = canonical_json_sha256(payload_without_sha)
    expected_payload_sha = str(source_cfg["manifest_payload_sha256"])
    if claimed_payload_sha != expected_payload_sha or observed_payload_sha != expected_payload_sha:
        raise ValueError("exp015 inference manifest payload SHA mismatch")
    if manifest.get("experiment") != SOURCE_EXPERIMENT:
        raise ValueError("exp015 inference manifest experiment mismatch")
    if manifest.get("ground_truth_accessed") is not False:
        raise ValueError("candidate graph generation must not access ground truth")
    if manifest.get("input_manifest_sha256") != source_cfg["input_manifest_sha256"]:
        raise ValueError("exp015 fixed input manifest SHA mismatch")

    expected_count = int(source_cfg["expected_candidate_graph_count"])
    samples = [str(value) for value in manifest.get("samples", [])]
    if (
        len(samples) != expected_count
        or samples != sorted(set(samples))
        or int(manifest.get("candidate_graph_count", -1)) != expected_count
    ):
        raise ValueError("exp015 candidate graph sample contract mismatch")
    candidate_root = inference_root / str(source_cfg["candidate_graph_dir"])
    candidate_samples = sorted(path.stem for path in candidate_root.glob("*.geff"))
    if candidate_samples != samples:
        raise ValueError("exp015 manifest and candidate graph samples differ")
    evidence = {
        "manifest_file_sha256": observed_file_sha,
        "manifest_payload_sha256": observed_payload_sha,
        "input_manifest_sha256": manifest["input_manifest_sha256"],
        "kernel_id_no": int(source_cfg["kernel_id_no"]),
    }
    return manifest, samples, candidate_root, evidence


def load_fixed_embryo_map(
    root: Path,
    cfg: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    per_sample_path = root / "per_sample_readout.csv"
    paired_path = root / "paired_route_comparison.csv"
    observed_per_sample_sha = sha256_file(per_sample_path)
    observed_paired_sha = sha256_file(paired_path)
    if observed_per_sample_sha != str(cfg["per_sample_readout_sha256"]):
        raise ValueError("exp012 per-sample readout SHA mismatch")
    if observed_paired_sha != str(cfg["paired_route_comparison_sha256"]):
        raise ValueError("exp012 paired comparison SHA mismatch")

    baseline_route = str(cfg["baseline_route"])
    mapping: dict[str, str] = {}
    for row in read_csv(per_sample_path):
        if str(row["route"]) != baseline_route:
            continue
        sample = str(row["sample"])
        embryo = str(row["embryo"])
        if sample in mapping:
            raise ValueError(f"duplicate exp012 baseline sample: {sample}")
        mapping[sample] = embryo
    evidence = {
        "per_sample_readout_sha256": observed_per_sample_sha,
        "paired_route_comparison_sha256": observed_paired_sha,
    }
    return mapping, evidence


def root_containing_relative_file(path: Path, relative_path: str) -> Path:
    relative_parts = Path(relative_path).parts
    if not relative_parts or path.parts[-len(relative_parts) :] != relative_parts:
        raise ValueError(f"{path} does not end with {relative_path}")
    return path.parents[len(relative_parts) - 1]


def support_repo_candidates(working_root: Path, evaluator_relative_path: str) -> list[Path]:
    candidates: list[Path] = []
    local_repo = (
        working_root.parents[1]
        / "experiments"
        / "exp014_exact_window_cache"
        / "artifacts"
        / "kaggle-v1"
        / "tracking_repo"
    )
    if (local_repo / evaluator_relative_path).is_file():
        candidates.append(local_repo)
    if is_kaggle_runtime():
        explicit = [
            Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"),
            Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1"),
        ]
        candidates.extend(path for path in explicit if (path / evaluator_relative_path).is_file())
        if not candidates:
            candidates.extend(
                root_containing_relative_file(path, evaluator_relative_path)
                for path in Path("/kaggle/input").rglob(evaluator_relative_path)
            )
    return candidates


def load_official_metric_module(
    working_root: Path,
    source_cfg: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    evaluator_relative = str(source_cfg["evaluator_path"])
    expected_evaluator_sha = str(source_cfg["evaluator_sha256"])
    valid_repos = [
        root
        for root in support_repo_candidates(working_root, evaluator_relative)
        if (root / evaluator_relative).is_file()
        and sha256_file(root / evaluator_relative) == expected_evaluator_sha
    ]
    repo_root = unique_existing(valid_repos, "SHA-pinned support repository")
    metrics_path = repo_root / str(source_cfg["metrics_path"])
    division_path = repo_root / str(source_cfg["division_metrics_path"])
    if sha256_file(metrics_path) != str(source_cfg["metrics_sha256"]):
        raise ValueError("official metric source SHA mismatch")
    if sha256_file(division_path) != str(source_cfg["division_metrics_sha256"]):
        raise ValueError("official division metric source SHA mismatch")

    source_root = metrics_path.parent.parent
    for module_name in list(sys.modules):
        if module_name == "biohub_tracking" or module_name.startswith("biohub_tracking."):
            sys.modules.pop(module_name, None)
    sys.path.insert(0, str(source_root))
    metric_module = importlib.import_module("biohub_tracking.metrics")
    loaded_path = Path(inspect.getfile(metric_module)).resolve()
    if loaded_path != metrics_path.resolve():
        raise RuntimeError(f"unexpected official metric import path: {loaded_path}")
    evidence = {
        "support_repo": str(repo_root),
        "evaluator_sha256": expected_evaluator_sha,
        "metrics_sha256": sha256_file(metrics_path),
        "division_metrics_sha256": sha256_file(division_path),
    }
    return metric_module, evidence


# %% [markdown]
# ## 3. Fixed ILP solve and official metric


# %%
def graph_from_geff(path: Path):
    import tracksdata as td

    result = td.graph.IndexedRXGraph.from_geff(path)
    return result[0] if isinstance(result, tuple) else result


def _exact_float(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite graph value: {value}")
    return number.hex()


def candidate_graph_receipt(graph: Any) -> dict[str, Any]:
    node_table = graph.node_attrs()
    edge_table = graph.edge_attrs()
    required_node = {"node_id", "t", "z", "y", "x"}
    required_edge = {"edge_id", "source_id", "target_id", "edge_prob"}
    if not required_node.issubset(set(node_table.columns)):
        raise ValueError(f"candidate node attributes changed: {node_table.columns}")
    if not required_edge.issubset(set(edge_table.columns)):
        raise ValueError(f"candidate edge attributes changed: {edge_table.columns}")

    nodes = sorted(
        (
            int(row["node_id"]),
            int(row["t"]),
            _exact_float(row["z"]),
            _exact_float(row["y"]),
            _exact_float(row["x"]),
        )
        for row in node_table.iter_rows(named=True)
    )
    edges = sorted(
        (
            int(row["edge_id"]),
            int(row["source_id"]),
            int(row["target_id"]),
            _exact_float(row["edge_prob"]),
        )
        for row in edge_table.iter_rows(named=True)
    )
    edge_probabilities = [float.fromhex(row[3]) for row in edges]
    if any(probability < 0.0 or probability > 1.0 for probability in edge_probabilities):
        raise ValueError("candidate edge_prob is outside the fixed probability range [0, 1]")
    payload = {"nodes": nodes, "edges": edges}
    return {
        "content_sha256": canonical_json_sha256(payload),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "edge_probability_min": min(edge_probabilities, default=float("nan")),
        "edge_probability_max": max(edge_probabilities, default=float("nan")),
    }


def solution_topology_receipt(graph: Any) -> dict[str, Any]:
    nodes = sorted(int(value) for value in graph.node_ids())
    edges = sorted(
        (int(row["source_id"]), int(row["target_id"]))
        for row in graph.edge_attrs().iter_rows(named=True)
    )
    return {
        "topology_sha256": canonical_json_sha256({"nodes": nodes, "edges": edges}),
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def solve_candidate_graph(
    candidate_path: Path,
    alpha: float,
    solver_cfg: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import tracksdata as td

    graph = graph_from_geff(candidate_path)
    candidate_receipt = candidate_graph_receipt(graph)
    solver = td.solvers.ILPSolver(
        edge_weight=-float(alpha) * td.EdgeAttr("edge_prob"),
        appearance_weight=float(solver_cfg["appearance_weight"]),
        disappearance_weight=float(solver_cfg["disappearance_weight"]),
        division_weight=float(solver_cfg["division_weight"]),
    )
    solution = solver.solve(graph).detach()
    return solution, candidate_receipt, solution_topology_receipt(solution)


def official_metric_row(
    solution: Any,
    gt_path: Path,
    metric_module: Any,
    *,
    scale: tuple[float, float, float],
    max_distance: float,
) -> dict[str, Any]:
    from geff import GeffMetadata

    gt_graph = graph_from_geff(gt_path)
    result = metric_module.evaluate(
        solution,
        gt_graph,
        scale=scale,
        max_distance=max_distance,
    )
    recall = (
        metric_module.node_recall(solution, gt_graph)
        if solution.num_edges() > 0 and solution.num_nodes() > 0
        else 0.0
    )
    metadata = GeffMetadata.read(gt_path)
    n_total = (metadata.extra or {}).get("estimated_number_of_nodes")
    n_total_value = float(n_total) if n_total is not None else float("nan")
    return dict(metric_module.per_sample_metrics(result, n_total_value, recall))


def solver_runtime_receipt() -> dict[str, Any]:
    import tracksdata as td

    try:
        source = inspect.getsource(td.solvers.ILPSolver)
        source_sha = sha256_text(source)
    except (OSError, TypeError):
        source_sha = None
    package_versions: dict[str, str | None] = {}
    for package in ("tracksdata", "ilpy", "pyscipopt", "geff", "polars"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "ilp_solver_source_sha256": source_sha,
        "package_versions": package_versions,
    }


# %% [markdown]
# ## 4. Cross-embryo selection and promotion gate
#
# Alpha is selected from one embryo's aggregate official score, then evaluated
# once on the other embryo. The held-out rows are never passed to selection.


# %%
def select_alpha(
    summary_rows: list[dict[str, Any]],
    calibration_embryo: str,
    alpha_grid: list[float],
    *,
    baseline_alpha: float,
    score_tolerance: float,
) -> dict[str, Any]:
    candidates = [
        row
        for row in summary_rows
        if str(row["embryo"]) == calibration_embryo and float(row["alpha"]) in alpha_grid
    ]
    if len(candidates) != len(alpha_grid):
        raise ValueError(f"incomplete alpha summary for {calibration_embryo}")
    if len({float(row["alpha"]) for row in candidates}) != len(alpha_grid):
        raise ValueError(f"duplicate alpha summary for {calibration_embryo}")
    scores = [float(row["score"]) for row in candidates]
    if any(not math.isfinite(score) for score in scores):
        raise ValueError(f"non-finite selection score for {calibration_embryo}")
    best_score = max(scores)
    tied = [row for row in candidates if best_score - float(row["score"]) <= score_tolerance]

    def tie_key(row: dict[str, Any]) -> tuple[int, float, float]:
        alpha = float(row["alpha"])
        return (
            0 if math.isclose(alpha, baseline_alpha, rel_tol=0.0, abs_tol=0.0) else 1,
            abs(math.log2(alpha / baseline_alpha)),
            alpha,
        )

    return min(tied, key=tie_key)


def summary_lookup(
    summary_rows: list[dict[str, Any]],
) -> dict[tuple[str, float], dict[str, Any]]:
    lookup: dict[tuple[str, float], dict[str, Any]] = {}
    for row in summary_rows:
        key = (str(row["embryo"]), float(row["alpha"]))
        if key in lookup:
            raise ValueError(f"duplicate embryo/alpha summary: {key}")
        lookup[key] = row
    return lookup


def build_cross_embryo_decisions(
    summary_rows: list[dict[str, Any]],
    directions: list[dict[str, str]],
    alpha_grid: list[float],
    expected_embryo_counts: dict[str, int],
    *,
    baseline_alpha: float,
    minimum_score_delta: float,
    score_tolerance: float,
    require_division_nonregression: bool,
) -> tuple[list[dict[str, Any]], bool]:
    lookup = summary_lookup(summary_rows)
    expected_keys = {(embryo, alpha) for embryo in expected_embryo_counts for alpha in alpha_grid}
    if set(lookup) != expected_keys:
        raise ValueError("embryo/alpha summary coverage changed")

    coverage_ok = all(
        int(row["sample_count"]) == expected_embryo_counts[str(row["embryo"])]
        and int(row["valid_sample_count"]) == expected_embryo_counts[str(row["embryo"])]
        and int(row["failure_count"]) == 0
        and math.isfinite(float(row["score"]))
        and math.isfinite(float(row["division_jaccard"]))
        for row in summary_rows
    )
    decisions: list[dict[str, Any]] = []
    for direction in directions:
        calibration = str(direction["calibration_embryo"])
        heldout = str(direction["heldout_embryo"])
        selected = select_alpha(
            summary_rows,
            calibration,
            alpha_grid,
            baseline_alpha=baseline_alpha,
            score_tolerance=score_tolerance,
        )
        selected_alpha = float(selected["alpha"])
        heldout_selected = lookup[(heldout, selected_alpha)]
        heldout_baseline = lookup[(heldout, baseline_alpha)]
        score_delta = float(heldout_selected["score"]) - float(heldout_baseline["score"])
        division_delta = float(heldout_selected["division_jaccard"]) - float(
            heldout_baseline["division_jaccard"]
        )
        combined_improved = score_delta > minimum_score_delta
        division_held = division_delta >= -score_tolerance
        passed = (
            coverage_ok
            and combined_improved
            and (division_held or not require_division_nonregression)
        )
        decisions.append(
            {
                "calibration_embryo": calibration,
                "heldout_embryo": heldout,
                "selected_alpha": selected_alpha,
                "calibration_selected_score": float(selected["score"]),
                "heldout_selected_score": float(heldout_selected["score"]),
                "heldout_baseline_score": float(heldout_baseline["score"]),
                "heldout_combined_score_delta": score_delta,
                "heldout_selected_division_jaccard": float(heldout_selected["division_jaccard"]),
                "heldout_baseline_division_jaccard": float(heldout_baseline["division_jaccard"]),
                "heldout_division_jaccard_delta": division_delta,
                "complete_coverage": coverage_ok,
                "combined_score_strictly_improved": combined_improved,
                "division_jaccard_not_worse": division_held,
                "direction_passed": passed,
            }
        )
    promotion_gate_passed = len(decisions) == 2 and all(
        bool(row["direction_passed"]) for row in decisions
    )
    return decisions, promotion_gate_passed


def aggregate_embryo_alpha_rows(
    per_sample_rows: list[dict[str, Any]],
    metric_module: Any,
    embryo_ids: list[str],
    alpha_grid: list[float],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for embryo in embryo_ids:
        for alpha in alpha_grid:
            rows = [
                row
                for row in per_sample_rows
                if str(row["embryo"]) == embryo
                and math.isclose(float(row["alpha"]), alpha, rel_tol=0.0, abs_tol=0.0)
            ]
            metric_rows = [
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "sample",
                        "embryo",
                        "alpha",
                        "candidate_graph_content_sha256",
                        "candidate_node_count",
                        "candidate_edge_count",
                        "edge_probability_min",
                        "edge_probability_max",
                        "solution_topology_sha256",
                        "solution_node_count",
                        "solution_edge_count",
                        "failure",
                    }
                }
                for row in rows
            ]
            summary = dict(metric_module.summarise(metric_rows))
            valid_count = int(summary["n"])
            summaries.append(
                {
                    "embryo": embryo,
                    "alpha": alpha,
                    "sample_count": len(rows),
                    "valid_sample_count": valid_count,
                    "failure_count": len(rows) - valid_count,
                    **summary,
                }
            )
    return summaries


# %% [markdown]
# ## 5. Diagnostic execution


# %%
def failed_metric_row(metric_module: Any, error: Exception) -> dict[str, Any]:
    row = dict(metric_module.nan_metrics_row())
    row["failure"] = f"{type(error).__name__}: {error}"
    return row


def run_diagnostic(
    working_root: Path | None = None,
    *,
    shard_name: str | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    working_root = (working_root or Path.cwd()).resolve()
    config = yaml.safe_load((working_root / "config.yaml").read_text(encoding="utf-8"))
    if config["experiment"]["name"] != EXPERIMENT:
        raise ValueError("experiment config mismatch")

    ensure_geff_runtime_dependencies()
    metric_module, evaluator_evidence = load_official_metric_module(
        working_root,
        config["official_metric_source"],
    )
    source_cfg = config["data"]["exp015_candidate_graphs"]
    inference_root = resolve_inference_root(working_root, source_cfg)
    _, samples, candidate_root, inference_evidence = verify_exp015_manifest(
        inference_root,
        source_cfg,
    )
    exp012_cfg = config["data"]["exp012_embryo_map"]
    exp012_root = resolve_exp012_root(working_root, exp012_cfg)
    embryo_map, exp012_evidence = load_fixed_embryo_map(exp012_root, exp012_cfg)
    if set(embryo_map) != set(samples):
        raise ValueError("exp012 embryo map and exp015 candidate samples differ")

    expected_embryo_counts = {
        str(key): int(value)
        for key, value in config["validation"]["expected_embryo_counts"].items()
    }
    observed_counts = Counter(embryo_map.values())
    if dict(observed_counts) != expected_embryo_counts:
        raise ValueError(f"embryo count mismatch: {dict(observed_counts)}")
    train_root = resolve_train_dir(working_root)
    if sorted(path.stem for path in train_root.glob("*.geff")) != samples:
        raise ValueError("competition train GEFF sample set changed")

    solver_cfg = config["model"]["params"]
    approved_alpha_grid = [float(value) for value in solver_cfg["alpha_grid"]]
    if approved_alpha_grid != [0.25, 0.5, 1.0, 2.0, 4.0]:
        raise ValueError("approved alpha grid changed")
    shard_cfg: dict[str, Any] | None = None
    if shard_name is None:
        alpha_grid = approved_alpha_grid
    else:
        configured_shards = config["stages"]["initial_ilp_only"]["alpha_shards"]
        if shard_name not in configured_shards:
            raise ValueError(f"unknown alpha shard: {shard_name}")
        shard_cfg = dict(configured_shards[shard_name])
        shard_alpha = float(shard_cfg["alpha"])
        if shard_alpha not in approved_alpha_grid:
            raise ValueError(f"unapproved alpha shard value: {shard_alpha}")
        alpha_grid = [shard_alpha]
    baseline_alpha = float(solver_cfg["baseline_alpha"])
    matching_cfg = config["validation"]["matching"]
    scale = tuple(float(value) for value in matching_cfg["scale_zyx_um"])
    if len(scale) != 3:
        raise ValueError("expected z/y/x metric scale")
    max_distance = float(matching_cfg["max_distance_um"])

    per_sample_rows: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples, start=1):
        candidate_path = candidate_root / f"{sample}.geff"
        gt_path = train_root / f"{sample}.geff"
        expected_candidate_sha: str | None = None
        for alpha in alpha_grid:
            base = {"sample": sample, "embryo": embryo_map[sample], "alpha": alpha}
            try:
                solution, candidate_receipt, solution_receipt = solve_candidate_graph(
                    candidate_path,
                    alpha,
                    solver_cfg,
                )
                candidate_sha = str(candidate_receipt["content_sha256"])
                if expected_candidate_sha is None:
                    expected_candidate_sha = candidate_sha
                    candidate_records.append({"sample": sample, **candidate_receipt})
                elif candidate_sha != expected_candidate_sha:
                    raise RuntimeError("candidate graph changed between alpha solves")
                metrics = official_metric_row(
                    solution,
                    gt_path,
                    metric_module,
                    scale=scale,
                    max_distance=max_distance,
                )
                row = {
                    **base,
                    "candidate_graph_content_sha256": candidate_sha,
                    "candidate_node_count": int(candidate_receipt["node_count"]),
                    "candidate_edge_count": int(candidate_receipt["edge_count"]),
                    "edge_probability_min": candidate_receipt["edge_probability_min"],
                    "edge_probability_max": candidate_receipt["edge_probability_max"],
                    "solution_topology_sha256": solution_receipt["topology_sha256"],
                    "solution_node_count": int(solution_receipt["node_count"]),
                    "solution_edge_count": int(solution_receipt["edge_count"]),
                    **metrics,
                    "failure": "",
                }
                if sample_index == 1 and alpha == alpha_grid[0]:
                    print("Official metric smoke evaluation passed.", flush=True)
            except Exception as error:
                if sample_index == 1 and alpha == alpha_grid[0]:
                    raise RuntimeError("official metric smoke evaluation failed") from error
                row = {**base, **failed_metric_row(metric_module, error)}
            per_sample_rows.append(row)
        print(f"[{sample_index:03d}/{len(samples)}] {sample}", flush=True)

    embryo_ids = list(expected_embryo_counts)
    embryo_alpha_rows = aggregate_embryo_alpha_rows(
        per_sample_rows,
        metric_module,
        embryo_ids,
        alpha_grid,
    )
    expected_rows = len(samples) * len(alpha_grid)
    failure_rows = [row for row in per_sample_rows if str(row.get("failure", ""))]
    if len(per_sample_rows) != expected_rows:
        raise RuntimeError("per-sample alpha row count mismatch")
    input_receipts_complete = len(candidate_records) == len(samples)
    gate_cfg = config["validation"]["promotion_gate"]
    tie_cfg = config["validation"]["tie_break"]
    if shard_name is not None:
        decisions: list[dict[str, Any]] = []
        promotion_gate_passed: bool | None = None
    elif failure_rows or not input_receipts_complete:
        decisions = [
            {
                "calibration_embryo": str(direction["calibration_embryo"]),
                "heldout_embryo": str(direction["heldout_embryo"]),
                "selected_alpha": None,
                "complete_coverage": False,
                "combined_score_strictly_improved": False,
                "division_jaccard_not_worse": False,
                "direction_passed": False,
                "failure": "incomplete alpha evaluation or candidate input receipt",
            }
            for direction in config["validation"]["selection_directions"]
        ]
        promotion_gate_passed = False
    else:
        decisions, promotion_gate_passed = build_cross_embryo_decisions(
            embryo_alpha_rows,
            config["validation"]["selection_directions"],
            alpha_grid,
            expected_embryo_counts,
            baseline_alpha=baseline_alpha,
            minimum_score_delta=float(gate_cfg["minimum_combined_score_delta"]),
            score_tolerance=float(tie_cfg["score_tolerance"]),
            require_division_nonregression=bool(
                gate_cfg["require_nonnegative_division_jaccard_delta_in_both_directions"]
            ),
        )

    output_dir = (
        str(shard_cfg["output_dir"])
        if shard_cfg is not None
        else str(config["stages"]["initial_ilp_only"]["output_dir"])
    )
    output_root = working_root / output_dir
    paths = {
        "per_sample": output_root / "per_sample_alpha_metrics.csv",
        "embryo_alpha": output_root / "embryo_alpha_summary.csv",
        "summary": output_root / "graph_cost_scale_summary.json",
        "manifest": output_root / "graph_cost_scale_manifest.json",
    }
    if shard_name is None:
        paths["selection"] = output_root / "cross_embryo_selection.csv"
    write_csv(paths["per_sample"], per_sample_rows)
    write_csv(paths["embryo_alpha"], embryo_alpha_rows)
    if shard_name is None:
        write_csv(paths["selection"], decisions)

    elapsed_seconds = time.monotonic() - started
    solution_records = [
        {
            "sample": row["sample"],
            "alpha": row["alpha"],
            "solution_topology_sha256": row.get("solution_topology_sha256"),
            "score_counts": {
                key: row.get(key)
                for key in (
                    "edge_tp",
                    "edge_fp",
                    "edge_fn",
                    "division_tp",
                    "division_fp",
                    "division_fn",
                    "num_pred_nodes",
                )
            },
        }
        for row in per_sample_rows
    ]
    candidate_bundle_sha = canonical_json_sha256(candidate_records)
    solution_bundle_sha = canonical_json_sha256(solution_records)
    stage_name = "initial_ilp_only_alpha_shard" if shard_name is not None else "initial_ilp_only"
    summary = {
        "experiment": EXPERIMENT,
        "stage": stage_name,
        "shard_name": shard_name,
        "diagnostic_only": True,
        "repair_executed": False,
        "sample_count": len(samples),
        "alpha_grid": alpha_grid,
        "per_sample_row_count": len(per_sample_rows),
        "failure_count": len(failure_rows),
        "embryo_alpha": embryo_alpha_rows,
        "directions": decisions,
        "promotion_gate_passed": promotion_gate_passed,
        "next_stage": (
            "aggregate_alpha_shards"
            if shard_name is not None
            else (
                "review_then_implement_fixed_full_repair_in_exp018"
                if promotion_gate_passed
                else "stop_without_full_repair"
            )
        ),
        "candidate_graph_bundle_sha256": candidate_bundle_sha,
        "solution_bundle_sha256": solution_bundle_sha,
        "elapsed_seconds": elapsed_seconds,
    }
    atomic_json(paths["summary"], summary)

    solver_evidence = solver_runtime_receipt()
    output_hashes = {key: sha256_file(path) for key, path in paths.items() if key != "manifest"}
    manifest = {
        "experiment": EXPERIMENT,
        "stage": stage_name,
        "shard_name": shard_name,
        "generated_at": datetime.now(UTC).isoformat(),
        "ground_truth_use": "official_metric_only_after_fixed_variant_generation",
        "repair_executed": False,
        "sample_count": len(samples),
        "alpha_grid": alpha_grid,
        "input": {
            "exp015": inference_evidence,
            "exp012": exp012_evidence,
            "candidate_graph_bundle_sha256": candidate_bundle_sha,
        },
        "official_metric_source": evaluator_evidence,
        "solver": {
            "edge_cost_expression": solver_cfg["edge_cost_expression"],
            "appearance_weight": solver_cfg["appearance_weight"],
            "disappearance_weight": solver_cfg["disappearance_weight"],
            "division_weight": solver_cfg["division_weight"],
            **solver_evidence,
        },
        "output": {
            "solution_bundle_sha256": solution_bundle_sha,
            "file_sha256": output_hashes,
        },
        "failure_count": len(failure_rows),
        "promotion_gate_passed": promotion_gate_passed,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_json(paths["manifest"], manifest)

    if shard_name is not None:
        print("GRAPH_COST_SCALE_ALPHA_SHARD_SUMMARY")
        print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, sort_keys=True))
        print("GRAPH_COST_SCALE_ALPHA_SHARD_MANIFEST_SHA256", manifest["manifest_sha256"])
        if failure_rows:
            examples = [
                {"sample": row["sample"], "alpha": row["alpha"], "failure": row["failure"]}
                for row in failure_rows[:10]
            ]
            raise RuntimeError(f"{len(failure_rows)} alpha evaluations failed; examples={examples}")
        return summary

    metrics_path = working_root / "metrics.json"
    current_metrics = load_json_object(metrics_path)
    updated_metrics = deep_merge(
        current_metrics,
        {
            "status": "debug_completed" if not failure_rows else "failed",
            "metric": config["validation"]["metric"],
            "evidence": {
                "kaggle": {
                    "kernel_source_ids": [
                        source_cfg["kernel_source"],
                        exp012_cfg["kernel_source"],
                    ],
                    "resource": "cpu",
                    "notebook_runtime_seconds": elapsed_seconds,
                    "internet_enabled": False,
                },
                "artifacts": {
                    "input_file_sha": canonical_json_sha256(
                        {"exp015": inference_evidence, "exp012": exp012_evidence}
                    ),
                    "feature_content_sha": candidate_bundle_sha,
                    "row_count": len(per_sample_rows),
                    "group_count": len(embryo_ids),
                    "feature_count": 1,
                    "model_count": 0,
                    "selected_mode": "fixed_candidate_graph_ilp_cost_scale",
                    "per_sample_sha256": output_hashes["per_sample"],
                    "embryo_alpha_sha256": output_hashes["embryo_alpha"],
                    "cross_embryo_selection_sha256": output_hashes["selection"],
                    "summary_sha256": output_hashes["summary"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "solution_bundle_sha256": solution_bundle_sha,
                },
            },
            "diagnostics": {
                "graph_cost_scale": {
                    "stage": "initial_ilp_only",
                    "repair_executed": False,
                    "sample_count": len(samples),
                    "alpha_grid": alpha_grid,
                    "failure_count": len(failure_rows),
                    "embryo_alpha": embryo_alpha_rows,
                    "directions": decisions,
                    "promotion_gate_passed": promotion_gate_passed,
                }
            },
            "notes": (
                "Kaggle CPU ILP-only diagnostic completed. The public-model train graphs "
                "make this a conditional comparison, not independent CV. No submission "
                "was produced."
            ),
        },
    )
    atomic_json(metrics_path, updated_metrics)
    print("GRAPH_COST_SCALE_SUMMARY")
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, sort_keys=True))
    print("GRAPH_COST_SCALE_MANIFEST_SHA256", manifest["manifest_sha256"])
    if failure_rows:
        examples = [
            {"sample": row["sample"], "alpha": row["alpha"], "failure": row["failure"]}
            for row in failure_rows[:10]
        ]
        raise RuntimeError(f"{len(failure_rows)} alpha evaluations failed; examples={examples}")
    return summary


INTEGER_ALPHA_ROW_FIELDS = (
    "candidate_node_count",
    "candidate_edge_count",
    "solution_node_count",
    "solution_edge_count",
    "edge_tp",
    "edge_fp",
    "edge_fn",
    "division_tp",
    "division_fp",
    "division_fn",
    "num_pred_nodes",
)
FLOAT_ALPHA_ROW_FIELDS = (
    "alpha",
    "edge_probability_min",
    "edge_probability_max",
    "node_recall",
    "total_node_ratio",
    "edge_jaccard",
    "adj_edge_jaccard",
)


def parse_alpha_metric_row(row: dict[str, str]) -> dict[str, Any]:
    parsed: dict[str, Any] = dict(row)
    for field in INTEGER_ALPHA_ROW_FIELDS:
        parsed[field] = int(row[field])
    for field in FLOAT_ALPHA_ROW_FIELDS:
        parsed[field] = float(row[field])
    return parsed


def resolve_alpha_shard_root(
    working_root: Path,
    shard_cfg: dict[str, Any],
) -> Path:
    output_dir = Path(str(shard_cfg["output_dir"]))
    manifest_name = "graph_cost_scale_manifest.json"
    local = (working_root / output_dir).resolve()
    candidates = [local] if (local / manifest_name).is_file() else []
    if is_kaggle_runtime():
        kernel_slug = str(shard_cfg["kernel_source"]).split("/", 1)[-1]
        candidates.extend(
            path.parent
            for path in Path("/kaggle/input").rglob(manifest_name)
            if kernel_slug in path.as_posix()
            and path.parent.as_posix().endswith(output_dir.as_posix())
        )
    return unique_existing(candidates, f"alpha shard {shard_cfg['alpha']} output root")


def load_alpha_shard(
    working_root: Path,
    shard_name: str,
    shard_cfg: dict[str, Any],
    *,
    expected_sample_count: int,
    expected_embryo_counts: dict[str, int],
) -> dict[str, Any]:
    root = resolve_alpha_shard_root(working_root, shard_cfg)
    manifest_path = root / "graph_cost_scale_manifest.json"
    summary_path = root / "graph_cost_scale_summary.json"
    per_sample_path = root / "per_sample_alpha_metrics.csv"
    manifest = load_json_object(manifest_path)
    stored_manifest_sha = str(manifest.get("manifest_sha256", ""))
    manifest_payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if canonical_json_sha256(manifest_payload) != stored_manifest_sha:
        raise ValueError(f"alpha shard manifest SHA mismatch: {shard_name}")
    expected_alpha = float(shard_cfg["alpha"])
    if manifest.get("experiment") != EXPERIMENT:
        raise ValueError(f"alpha shard experiment mismatch: {shard_name}")
    if manifest.get("stage") != "initial_ilp_only_alpha_shard":
        raise ValueError(f"alpha shard stage mismatch: {shard_name}")
    if manifest.get("shard_name") != shard_name:
        raise ValueError(f"alpha shard name mismatch: {shard_name}")
    if [float(value) for value in manifest.get("alpha_grid", [])] != [expected_alpha]:
        raise ValueError(f"alpha shard value mismatch: {shard_name}")
    if int(manifest.get("sample_count", -1)) != expected_sample_count:
        raise ValueError(f"alpha shard sample count mismatch: {shard_name}")
    if int(manifest.get("failure_count", -1)) != 0:
        raise ValueError(f"alpha shard contains failed rows: {shard_name}")
    output_hashes = manifest["output"]["file_sha256"]
    if sha256_file(summary_path) != str(output_hashes["summary"]):
        raise ValueError(f"alpha shard summary SHA mismatch: {shard_name}")
    if sha256_file(per_sample_path) != str(output_hashes["per_sample"]):
        raise ValueError(f"alpha shard per-sample SHA mismatch: {shard_name}")
    summary = load_json_object(summary_path)
    if int(summary.get("failure_count", -1)) != 0:
        raise ValueError(f"alpha shard summary contains failures: {shard_name}")
    rows = [parse_alpha_metric_row(row) for row in read_csv(per_sample_path)]
    if len(rows) != expected_sample_count:
        raise ValueError(f"alpha shard row count mismatch: {shard_name}")
    if len({str(row["sample"]) for row in rows}) != expected_sample_count:
        raise ValueError(f"alpha shard sample IDs are not unique: {shard_name}")
    if Counter(str(row["embryo"]) for row in rows) != Counter(expected_embryo_counts):
        raise ValueError(f"alpha shard embryo coverage mismatch: {shard_name}")
    for row in rows:
        if float(row["alpha"]) != expected_alpha:
            raise ValueError(f"alpha shard row value mismatch: {shard_name}")
        if str(row.get("failure", "")):
            raise ValueError(f"alpha shard row failure: {shard_name}")
        for field in ("node_recall", "total_node_ratio", "edge_jaccard", "adj_edge_jaccard"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"non-finite {field} in alpha shard: {shard_name}")
        for field in ("candidate_graph_content_sha256", "solution_topology_sha256"):
            if len(str(row[field])) != 64:
                raise ValueError(f"invalid {field} in alpha shard: {shard_name}")
    return {
        "name": shard_name,
        "root": root,
        "manifest": manifest,
        "summary": summary,
        "rows": rows,
        "manifest_file_sha256": sha256_file(manifest_path),
    }


def official_metric_content_receipt(evidence: dict[str, Any]) -> dict[str, str]:
    fields = ("evaluator_sha256", "metrics_sha256", "division_metrics_sha256")
    receipt = {field: str(evidence.get(field, "")) for field in fields}
    if any(len(value) != 64 for value in receipt.values()):
        raise ValueError("official metric source is missing a content SHA")
    return receipt


def run_aggregate(working_root: Path | None = None) -> dict[str, Any]:
    started = time.monotonic()
    working_root = (working_root or Path.cwd()).resolve()
    config = yaml.safe_load((working_root / "config.yaml").read_text(encoding="utf-8"))
    if config["experiment"]["name"] != EXPERIMENT:
        raise ValueError("experiment config mismatch")

    ensure_geff_runtime_dependencies()
    metric_module, evaluator_evidence = load_official_metric_module(
        working_root,
        config["official_metric_source"],
    )
    alpha_grid = [float(value) for value in config["model"]["params"]["alpha_grid"]]
    if alpha_grid != [0.25, 0.5, 1.0, 2.0, 4.0]:
        raise ValueError("approved alpha grid changed")
    expected_sample_count = int(config["validation"]["expected_sample_count"])
    expected_embryo_counts = {
        str(key): int(value)
        for key, value in config["validation"]["expected_embryo_counts"].items()
    }
    configured_shards = config["stages"]["initial_ilp_only"]["alpha_shards"]
    shard_by_alpha = {
        float(value["alpha"]): (name, value) for name, value in configured_shards.items()
    }
    if sorted(shard_by_alpha) != sorted(alpha_grid):
        raise ValueError("alpha shard configuration does not cover the approved grid")

    shards = [
        load_alpha_shard(
            working_root,
            shard_by_alpha[alpha][0],
            dict(shard_by_alpha[alpha][1]),
            expected_sample_count=expected_sample_count,
            expected_embryo_counts=expected_embryo_counts,
        )
        for alpha in alpha_grid
    ]
    candidate_bundle_shas = {
        str(shard["manifest"]["input"]["candidate_graph_bundle_sha256"]) for shard in shards
    }
    if len(candidate_bundle_shas) != 1:
        raise ValueError("candidate graph bundle differs across alpha shards")
    input_receipt_shas = {
        canonical_json_sha256(
            {
                "exp015": shard["manifest"]["input"]["exp015"],
                "exp012": shard["manifest"]["input"]["exp012"],
            }
        )
        for shard in shards
    }
    if len(input_receipt_shas) != 1:
        raise ValueError("input receipts differ across alpha shards")
    shared_input_receipt_sha = next(iter(input_receipt_shas))
    metric_source_shas = {
        canonical_json_sha256(
            official_metric_content_receipt(shard["manifest"]["official_metric_source"])
        )
        for shard in shards
    }
    if len(metric_source_shas) != 1:
        raise ValueError("official metric source differs across alpha shards")
    if metric_source_shas != {
        canonical_json_sha256(official_metric_content_receipt(evaluator_evidence))
    }:
        raise ValueError("alpha shard metric source differs from aggregate metric source")

    alpha_order = {alpha: index for index, alpha in enumerate(alpha_grid)}
    per_sample_rows = sorted(
        [row for shard in shards for row in shard["rows"]],
        key=lambda row: (str(row["sample"]), alpha_order[float(row["alpha"])]),
    )
    if len(per_sample_rows) != expected_sample_count * len(alpha_grid):
        raise ValueError("combined alpha shard row count mismatch")
    embryo_ids = list(expected_embryo_counts)
    embryo_alpha_rows = aggregate_embryo_alpha_rows(
        per_sample_rows,
        metric_module,
        embryo_ids,
        alpha_grid,
    )
    gate_cfg = config["validation"]["promotion_gate"]
    tie_cfg = config["validation"]["tie_break"]
    decisions, promotion_gate_passed = build_cross_embryo_decisions(
        embryo_alpha_rows,
        config["validation"]["selection_directions"],
        alpha_grid,
        expected_embryo_counts,
        baseline_alpha=float(config["model"]["params"]["baseline_alpha"]),
        minimum_score_delta=float(gate_cfg["minimum_combined_score_delta"]),
        score_tolerance=float(tie_cfg["score_tolerance"]),
        require_division_nonregression=bool(
            gate_cfg["require_nonnegative_division_jaccard_delta_in_both_directions"]
        ),
    )

    output_root = working_root / str(config["stages"]["initial_ilp_only"]["output_dir"])
    paths = {
        "per_sample": output_root / "per_sample_alpha_metrics.csv",
        "embryo_alpha": output_root / "embryo_alpha_summary.csv",
        "selection": output_root / "cross_embryo_selection.csv",
        "summary": output_root / "graph_cost_scale_summary.json",
        "manifest": output_root / "graph_cost_scale_manifest.json",
    }
    write_csv(paths["per_sample"], per_sample_rows)
    write_csv(paths["embryo_alpha"], embryo_alpha_rows)
    write_csv(paths["selection"], decisions)
    solution_records = [
        {
            "sample": row["sample"],
            "alpha": row["alpha"],
            "solution_topology_sha256": row["solution_topology_sha256"],
            "score_counts": {
                key: row[key]
                for key in (
                    "edge_tp",
                    "edge_fp",
                    "edge_fn",
                    "division_tp",
                    "division_fp",
                    "division_fn",
                    "num_pred_nodes",
                )
            },
        }
        for row in per_sample_rows
    ]
    solution_bundle_sha = canonical_json_sha256(solution_records)
    candidate_bundle_sha = next(iter(candidate_bundle_shas))
    shard_runtime_seconds = {
        str(shard["name"]): float(shard["summary"]["elapsed_seconds"]) for shard in shards
    }
    aggregate_runtime_seconds = time.monotonic() - started
    total_shard_runtime_seconds = sum(shard_runtime_seconds.values())
    summary = {
        "experiment": EXPERIMENT,
        "stage": "initial_ilp_only",
        "execution_mode": "independent_sequential_alpha_shards_then_aggregate",
        "diagnostic_only": True,
        "repair_executed": False,
        "sample_count": expected_sample_count,
        "alpha_grid": alpha_grid,
        "per_sample_row_count": len(per_sample_rows),
        "failure_count": 0,
        "embryo_alpha": embryo_alpha_rows,
        "directions": decisions,
        "promotion_gate_passed": promotion_gate_passed,
        "next_stage": (
            "review_then_implement_fixed_full_repair_in_exp018"
            if promotion_gate_passed
            else "stop_without_full_repair"
        ),
        "candidate_graph_bundle_sha256": candidate_bundle_sha,
        "solution_bundle_sha256": solution_bundle_sha,
        "shard_notebook_runtime_seconds": shard_runtime_seconds,
        "total_shard_runtime_seconds": total_shard_runtime_seconds,
        "aggregate_runtime_seconds": aggregate_runtime_seconds,
    }
    atomic_json(paths["summary"], summary)
    output_hashes = {key: sha256_file(path) for key, path in paths.items() if key != "manifest"}
    shard_evidence = {
        str(shard["name"]): {
            "alpha": float(shard["summary"]["alpha_grid"][0]),
            "kernel_source": configured_shards[str(shard["name"])]["kernel_source"],
            "manifest_sha256": shard["manifest"]["manifest_sha256"],
            "manifest_file_sha256": shard["manifest_file_sha256"],
            "solution_bundle_sha256": shard["summary"]["solution_bundle_sha256"],
            "notebook_runtime_seconds": float(shard["summary"]["elapsed_seconds"]),
        }
        for shard in shards
    }
    manifest = {
        "experiment": EXPERIMENT,
        "stage": "initial_ilp_only",
        "execution_mode": "independent_sequential_alpha_shards_then_aggregate",
        "generated_at": datetime.now(UTC).isoformat(),
        "ground_truth_use": "official_metric_only_after_fixed_variant_generation",
        "repair_executed": False,
        "sample_count": expected_sample_count,
        "alpha_grid": alpha_grid,
        "input": {
            "alpha_shards": shard_evidence,
            "candidate_graph_bundle_sha256": candidate_bundle_sha,
            "shared_input_receipt_sha256": shared_input_receipt_sha,
        },
        "official_metric_source": evaluator_evidence,
        "output": {
            "solution_bundle_sha256": solution_bundle_sha,
            "file_sha256": output_hashes,
        },
        "failure_count": 0,
        "promotion_gate_passed": promotion_gate_passed,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_json(paths["manifest"], manifest)

    metrics_path = working_root / "metrics.json"
    current_metrics = load_json_object(metrics_path)
    updated_metrics = deep_merge(
        current_metrics,
        {
            "status": "debug_completed",
            "metric": config["validation"]["metric"],
            "evidence": {
                "kaggle": {
                    "kernel_source_ids": [
                        configured_shards[name]["kernel_source"] for name in configured_shards
                    ],
                    "resource": "cpu",
                    "notebook_runtime_seconds": total_shard_runtime_seconds,
                    "shard_notebook_runtime_seconds": shard_runtime_seconds,
                    "aggregate_runtime_seconds": aggregate_runtime_seconds,
                    "internet_enabled": False,
                },
                "artifacts": {
                    "input_file_sha": shared_input_receipt_sha,
                    "feature_content_sha": candidate_bundle_sha,
                    "row_count": len(per_sample_rows),
                    "group_count": len(embryo_ids),
                    "feature_count": 1,
                    "model_count": 0,
                    "selected_mode": "fixed_candidate_graph_ilp_cost_scale",
                    "per_sample_sha256": output_hashes["per_sample"],
                    "embryo_alpha_sha256": output_hashes["embryo_alpha"],
                    "cross_embryo_selection_sha256": output_hashes["selection"],
                    "summary_sha256": output_hashes["summary"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "solution_bundle_sha256": solution_bundle_sha,
                },
            },
            "diagnostics": {
                "graph_cost_scale": {
                    "stage": "initial_ilp_only",
                    "execution_mode": "independent_sequential_alpha_shards_then_aggregate",
                    "repair_executed": False,
                    "sample_count": expected_sample_count,
                    "alpha_grid": alpha_grid,
                    "failure_count": 0,
                    "embryo_alpha": embryo_alpha_rows,
                    "directions": decisions,
                    "promotion_gate_passed": promotion_gate_passed,
                }
            },
            "notes": (
                "Kaggle CPU ILP-only diagnostic completed from five sequential alpha "
                "shards. The public-model train graphs make this a conditional comparison, "
                "not independent CV. No submission was produced."
            ),
        },
    )
    atomic_json(metrics_path, updated_metrics)
    print("GRAPH_COST_SCALE_AGGREGATE_SUMMARY")
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, sort_keys=True))
    print("GRAPH_COST_SCALE_AGGREGATE_MANIFEST_SHA256", manifest["manifest_sha256"])
    return summary


# %% [markdown]
# ## 6. Artifacts, manifest, and metrics
#
# A successful run writes only metric tables and SHA receipts. The 995 solved
# graphs are intentionally not retained. A passing gate authorizes review of a
# second, fixed-repair stage in this same experiment; it does not run that stage
# automatically.

# %%
if __name__ == "__main__":
    SUMMARY = run_diagnostic()
