# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp012 group error readout
#
# This diagnostic joins fixed embryo-held-out prediction metrics with image and
# candidate conditions. It reports where adjusted edge, division, and node
# recall errors concentrate. It does not train a model, change a prediction,
# choose a decode parameter, or create a competition submission.

# %% [markdown]
# ## Contents
# 1. Imports and runtime configuration
# 2. Input discovery, hashing, and schema checks
# 3. Official metric aggregation helpers
# 4. Candidate-boundary and condition features
# 5. Training-side bucket boundaries
# 6. Group summaries and paired route comparisons
# 7. Diagnostic execution and artifacts

# %% [markdown]
# ## 1. Imports and runtime configuration

# %%
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

EXPERIMENT = "exp012_group_error_readout"
COUNT_COLUMNS = (
    "edge_tp",
    "edge_fp",
    "edge_fn",
    "division_tp",
    "division_fp",
    "division_fn",
    "num_pred_nodes",
)
BUCKET_LABELS = ("Q1_low", "Q2", "Q3", "Q4_high")


def is_kaggle_runtime() -> bool:
    return Path("/kaggle/input").is_dir() and Path("/kaggle/working").is_dir()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(json_safe(row))


# %% [markdown]
# ## 2. Input discovery, hashing, and schema checks


# %%
def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def unique_by_key(
    rows: Iterable[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"duplicate {label} {value!r}")
        result[value] = row
    return result


def resolve_prediction_source(
    working_root: Path,
    source_cfg: dict[str, Any],
) -> tuple[Path, Path, Path]:
    route = str(source_cfg["route"])
    local_dir = (working_root / str(source_cfg["local_artifact_dir"])).resolve()
    local_manifest = local_dir / "prediction_manifest.json"
    local_metrics = local_dir / "per_sample_metrics.json"
    if local_manifest.is_file() and local_metrics.is_file():
        return local_dir, local_manifest, local_metrics

    matches: list[Path] = []
    for path in Path("/kaggle/input").rglob("prediction_manifest.json"):
        try:
            payload = load_json_object(path)
        except (json.JSONDecodeError, OSError, ValueError):
            continue
        if (
            payload.get("experiment") == route
            and (path.parent / "per_sample_metrics.json").is_file()
        ):
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one prediction manifest for {route}, found {matches}"
        )
    manifest_path = matches[0]
    return manifest_path.parent, manifest_path, manifest_path.parent / "per_sample_metrics.json"


def resolve_image_feature_path(working_root: Path, source_cfg: dict[str, Any]) -> Path:
    local_path = (working_root / str(source_cfg["local_path"])).resolve()
    if local_path.is_file():
        return local_path
    matches = sorted(Path("/kaggle/input").rglob("per_sample_candidate_image_diagnostics.csv"))
    matches = [path for path in matches if sha256_file(path) == source_cfg["sha256"]]
    if len(matches) != 1:
        raise FileNotFoundError(
            "expected exactly one pinned image diagnostic CSV, found " + repr(matches)
        )
    return matches[0]


def verify_sha(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"{label} SHA mismatch: expected {expected}, observed {observed}")
    return observed


def read_feature_rows(path: Path, source_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    columns = source_cfg["columns"]
    required = set(columns.values())
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"image diagnostic is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    return unique_by_key(rows, str(columns["id"]), "image-feature sample")


def load_prediction_rows(
    source_dir: Path,
    manifest_path: Path,
    metrics_path: Path,
    source_cfg: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    route = str(source_cfg["route"])
    manifest_sha = verify_sha(
        manifest_path,
        str(source_cfg["prediction_manifest_sha256"]),
        f"{route} prediction manifest",
    )
    metrics_sha = verify_sha(
        metrics_path,
        str(source_cfg["per_sample_metrics_sha256"]),
        f"{route} per-sample metrics",
    )
    manifest = load_json_object(manifest_path)
    metrics = load_json_object(metrics_path)
    if manifest.get("experiment") != route:
        raise ValueError(f"prediction manifest route mismatch for {route}")
    predictions = unique_by_key(manifest.get("predictions", []), "sample", f"{route} prediction")
    metric_rows = unique_by_key(metrics.get("rows", []), "sample", f"{route} metric")
    if set(predictions) != set(metric_rows):
        raise ValueError(f"prediction and metric sample sets differ for {route}")
    expected_count = int(manifest.get("expected_count", -1))
    if (
        len(predictions) != expected_count
        or int(manifest.get("completed_count", -1)) != expected_count
    ):
        raise ValueError(f"prediction manifest is incomplete for {route}")
    return (
        predictions,
        metric_rows,
        {
            "source_dir": str(source_dir),
            "prediction_manifest_sha256": manifest_sha,
            "per_sample_metrics_sha256": metrics_sha,
            "candidate_schema_sha256": str(manifest.get("candidate_schema_sha256")),
        },
    )


# %% [markdown]
# ## 3. Official metric aggregation helpers
#
# The formulas below preserve the exp003 official evaluator contract. Edge and
# division counts are micro-aggregated. Adjusted edge Jaccard is weighted by the
# per-video edge denominator. Failed or NaN evaluation rows remain visible in
# coverage counts and are excluded from the valid aggregate.


# %%
def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def jaccard(tp: float, fp: float, fn: float) -> float:
    denominator = tp + fp + fn
    return tp / denominator if denominator > 0 else float("nan")


def summarise_official(
    rows: list[dict[str, Any]],
    division_weight: float = 0.1,
) -> dict[str, Any]:
    valid = [row for row in rows if is_finite_number(row.get("edge_tp"))]
    failures = sum(bool(row.get("evaluation_failed")) for row in rows)
    nan_count = sum(
        not is_finite_number(row.get("adj_edge_jaccard"))
        for row in rows
        if not row.get("evaluation_failed")
    )
    if not valid:
        return {
            "n_total": len(rows),
            "n_valid": 0,
            "failure_count": failures,
            "nan_count": nan_count,
            "coverage": 0.0 if rows else float("nan"),
            "edge_jaccard": float("nan"),
            "division_jaccard": float("nan"),
            "division_tp": 0,
            "division_fp": 0,
            "division_fn": 0,
            "node_recall": float("nan"),
            "node_recall_error": float("nan"),
            "adj_edge_jaccard": float("nan"),
            "adj_edge_error": float("nan"),
            "n_adj": 0,
            "score": float("nan"),
        }

    totals = {column: sum(float(row[column]) for row in valid) for column in COUNT_COLUMNS}
    adj_rows = [row for row in valid if is_finite_number(row.get("adj_edge_jaccard"))]
    weights = [
        float(row["edge_tp"]) + float(row["edge_fp"]) + float(row["edge_fn"]) for row in adj_rows
    ]
    total_weight = sum(weights)
    adj_edge_jaccard = (
        sum(
            weight * float(row["adj_edge_jaccard"])
            for weight, row in zip(weights, adj_rows, strict=True)
        )
        / total_weight
        if total_weight > 0
        else float("nan")
    )
    division_jaccard = jaccard(totals["division_tp"], totals["division_fp"], totals["division_fn"])
    score = (
        adj_edge_jaccard + division_weight * division_jaccard
        if is_finite_number(division_jaccard)
        else adj_edge_jaccard
    )
    node_recalls = [
        float(row["node_recall"]) for row in valid if is_finite_number(row["node_recall"])
    ]
    node_recall = float(np.mean(node_recalls)) if node_recalls else float("nan")
    return {
        "n_total": len(rows),
        "n_valid": len(valid),
        "failure_count": failures,
        "nan_count": nan_count,
        "coverage": len(valid) / len(rows),
        "edge_jaccard": jaccard(totals["edge_tp"], totals["edge_fp"], totals["edge_fn"]),
        "division_jaccard": division_jaccard,
        "division_tp": int(totals["division_tp"]),
        "division_fp": int(totals["division_fp"]),
        "division_fn": int(totals["division_fn"]),
        "node_recall": node_recall,
        "node_recall_error": 1.0 - node_recall,
        "adj_edge_jaccard": adj_edge_jaccard,
        "adj_edge_error": 1.0 - adj_edge_jaccard,
        "n_adj": len(adj_rows),
        "score": score,
    }


# %% [markdown]
# ## 4. Candidate-boundary and condition features


# %%
def candidate_boundary_statistics(
    coords_tzyx: np.ndarray,
    spatial_shape_zyx: tuple[int, int, int],
    scale_zyx_um: tuple[float, float, float],
    near_distance_um: float,
) -> dict[str, float]:
    coords = np.asarray(coords_tzyx)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"coords_tzyx must have shape [node,4], found {coords.shape}")
    if len(coords) == 0:
        return {
            "candidate_boundary_distance_um_p50": float("nan"),
            "candidate_near_boundary_fraction": float("nan"),
        }
    spatial = coords[:, 1:4].astype(np.float64, copy=False)
    shape = np.asarray(spatial_shape_zyx, dtype=np.float64)
    if np.any(spatial < 0) or np.any(spatial > (shape - 1)):
        raise ValueError("candidate coordinates leave the configured image bounds")
    scale = np.asarray(scale_zyx_um, dtype=np.float64)
    lower_distance = spatial * scale
    upper_distance = (shape - 1 - spatial) * scale
    nearest = np.minimum(lower_distance, upper_distance).min(axis=1)
    return {
        "candidate_boundary_distance_um_p50": float(np.quantile(nearest, 0.5)),
        "candidate_near_boundary_fraction": float(np.mean(nearest <= near_distance_um)),
    }


def readout_rows_for_route(
    route: str,
    source_dir: Path,
    predictions: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
    image_features: dict[str, dict[str, Any]],
    config: dict[str, Any],
    *,
    allow_missing_candidate_cache: bool,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    feature_cfg = config["diagnostic"]["image_feature_source"]
    columns = feature_cfg["columns"]
    boundary_cfg = config["diagnostic"]["boundary"]
    spatial_shape = tuple(int(value) for value in boundary_cfg["spatial_shape_zyx"])
    near_distance = float(boundary_cfg["near_distance_um"])
    rows: list[dict[str, Any]] = []
    content_hashes: dict[str, str] = {}

    for sample in sorted(metrics):
        metric = metrics[sample]
        prediction = predictions[sample]
        if sample not in image_features:
            raise ValueError(f"image-feature row is missing for {sample}")
        feature = image_features[sample]
        frame_count = float(feature[str(columns["frame_count"])])
        if frame_count <= 0:
            raise ValueError(f"invalid image frame count for {sample}: {frame_count}")
        scale = tuple(
            float(feature[str(columns[name])]) for name in ("scale_z", "scale_y", "scale_x")
        )

        candidate_path = source_dir / str(prediction["candidate_cache"])
        if candidate_path.is_file():
            expected_file_sha = str(prediction["candidate_file_sha256"])
            verify_sha(candidate_path, expected_file_sha, f"{route} {sample} candidate cache")
            with np.load(candidate_path, allow_pickle=False) as cache:
                coords = np.asarray(cache["coords_tzyx"])
            boundary = candidate_boundary_statistics(coords, spatial_shape, scale, near_distance)
            content_hashes[sample] = str(prediction["candidate_content_sha256"])
        elif allow_missing_candidate_cache:
            boundary = {
                "candidate_boundary_distance_um_p50": float("nan"),
                "candidate_near_boundary_fraction": float("nan"),
            }
        else:
            raise FileNotFoundError(
                f"candidate cache is missing for {route} {sample}: {candidate_path}"
            )

        division_values = [
            metric.get(name) for name in ("division_tp", "division_fp", "division_fn")
        ]
        required_values = [metric.get(name) for name in COUNT_COLUMNS]
        evaluation_failed = not all(is_finite_number(value) for value in required_values)
        known_division = (
            bool(float(metric["division_tp"]) + float(metric["division_fn"]) > 0)
            if all(is_finite_number(value) for value in division_values)
            else None
        )
        division_jaccard = (
            jaccard(
                *(float(metric[name]) for name in ("division_tp", "division_fp", "division_fn"))
            )
            if all(is_finite_number(value) for value in division_values)
            else float("nan")
        )
        row = {
            "route": route,
            "sample": sample,
            "embryo": str(metric["embryo"]),
            "fold": int(metric["fold"]),
            **{column: metric[column] for column in COUNT_COLUMNS},
            "node_recall": metric["node_recall"],
            "total_node_ratio": metric["total_node_ratio"],
            "edge_jaccard": metric["edge_jaccard"],
            "adj_edge_jaccard": metric["adj_edge_jaccard"],
            "adj_edge_error": (
                1.0 - float(metric["adj_edge_jaccard"])
                if is_finite_number(metric["adj_edge_jaccard"])
                else float("nan")
            ),
            "division_jaccard": division_jaccard,
            "known_division": known_division,
            "evaluation_failed": evaluation_failed,
            "image_normalized_mean": float(feature[str(columns["brightness"])]),
            "candidate_nodes_per_frame": float(prediction["node_count"]) / frame_count,
            **boundary,
        }
        rows.append(row)
    return rows, content_hashes


# %% [markdown]
# ## 5. Training-side bucket boundaries
#
# For each held-out embryo, cut points come from the fixed exp005 baseline rows
# belonging to the other embryo. Outcome columns are never used to choose a cut
# point. The same cut points are then applied to exp005 and exp006.


# %%
def derive_bucket_thresholds(
    rows: list[dict[str, Any]],
    baseline_route: str,
    feature_columns: dict[str, str],
    quantiles: list[float],
) -> dict[str, dict[str, list[float]]]:
    baseline = [row for row in rows if row["route"] == baseline_route]
    embryos = sorted({str(row["embryo"]) for row in baseline})
    thresholds: dict[str, dict[str, list[float]]] = {}
    for evaluation_embryo in embryos:
        reference = [row for row in baseline if row["embryo"] != evaluation_embryo]
        if not reference:
            raise ValueError(f"no opposite-embryo reference for {evaluation_embryo}")
        thresholds[evaluation_embryo] = {}
        for feature_name, column in feature_columns.items():
            values = np.asarray(
                [float(row[column]) for row in reference if is_finite_number(row.get(column))],
                dtype=np.float64,
            )
            if len(values) < len(quantiles) + 1:
                raise ValueError(
                    f"too few training-side values for {evaluation_embryo} {feature_name}"
                )
            thresholds[evaluation_embryo][feature_name] = [
                float(value) for value in np.quantile(values, quantiles)
            ]
    return thresholds


def bucket_label(value: Any, edges: list[float]) -> str:
    if not is_finite_number(value):
        return "missing"
    index = int(np.searchsorted(np.asarray(edges), float(value), side="right"))
    return BUCKET_LABELS[index]


def assign_condition_buckets(
    rows: list[dict[str, Any]],
    thresholds: dict[str, dict[str, list[float]]],
    feature_columns: dict[str, str],
) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row = dict(source)
        embryo = str(row["embryo"])
        for feature_name, column in feature_columns.items():
            row[f"{feature_name}_bucket"] = bucket_label(
                row[column], thresholds[embryo][feature_name]
            )
        output.append(row)
    return output


# %% [markdown]
# ## 6. Group summaries and paired route comparisons


# %%
def mean_finite(rows: list[dict[str, Any]], column: str) -> float:
    values = [float(row[column]) for row in rows if is_finite_number(row.get(column))]
    return float(np.mean(values)) if values else float("nan")


def group_definitions(rows: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("all", "all", rows)]
    columns = {
        "embryo": "embryo",
        "brightness_quantile": "brightness_bucket",
        "candidate_density_quantile": "candidate_density_bucket",
        "boundary_distance_quantile": "boundary_distance_bucket",
        "known_division": "known_division",
    }
    for group_type, column in columns.items():
        values = sorted({str(row[column]) for row in rows})
        groups.extend(
            (group_type, value, [row for row in rows if str(row[column]) == value])
            for value in values
        )
    return groups


def make_group_summaries(
    rows: list[dict[str, Any]],
    division_weight: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    routes = sorted({str(row["route"]) for row in rows})
    for route in routes:
        route_rows = [row for row in rows if row["route"] == route]
        for group_type, group_value, group in group_definitions(route_rows):
            summary = summarise_official(group, division_weight=division_weight)
            result.append(
                {
                    "route": route,
                    "group_type": group_type,
                    "group_value": group_value,
                    **summary,
                    "image_normalized_mean": mean_finite(group, "image_normalized_mean"),
                    "candidate_nodes_per_frame": mean_finite(group, "candidate_nodes_per_frame"),
                    "candidate_boundary_distance_um_p50": mean_finite(
                        group, "candidate_boundary_distance_um_p50"
                    ),
                    "candidate_near_boundary_fraction": mean_finite(
                        group, "candidate_near_boundary_fraction"
                    ),
                }
            )
    return result


def make_paired_comparisons(
    rows: list[dict[str, Any]],
    baseline_route: str,
    comparison_routes: list[str],
    division_weight: float,
) -> list[dict[str, Any]]:
    by_route: dict[str, dict[str, dict[str, Any]]] = {}
    for route in {baseline_route, *comparison_routes}:
        by_route[route] = unique_by_key(
            (row for row in rows if row["route"] == route), "sample", f"{route} readout"
        )
    baseline_rows = list(by_route[baseline_route].values())
    result: list[dict[str, Any]] = []
    for comparison_route in comparison_routes:
        if set(by_route[comparison_route]) != set(by_route[baseline_route]):
            raise ValueError(f"paired route sample sets differ for {comparison_route}")
        for group_type, group_value, baseline_group in group_definitions(baseline_rows):
            samples = [str(row["sample"]) for row in baseline_group]
            comparison_group = [by_route[comparison_route][sample] for sample in samples]
            baseline_summary = summarise_official(baseline_group, division_weight)
            comparison_summary = summarise_official(comparison_group, division_weight)
            deltas = [
                float(by_route[comparison_route][sample]["adj_edge_jaccard"])
                - float(by_route[baseline_route][sample]["adj_edge_jaccard"])
                for sample in samples
                if is_finite_number(by_route[comparison_route][sample]["adj_edge_jaccard"])
                and is_finite_number(by_route[baseline_route][sample]["adj_edge_jaccard"])
            ]
            result.append(
                {
                    "baseline_route": baseline_route,
                    "comparison_route": comparison_route,
                    "condition_source": "baseline_route",
                    "group_type": group_type,
                    "group_value": group_value,
                    "paired_sample_count": len(samples),
                    "valid_pair_count": len(deltas),
                    "baseline_score": baseline_summary["score"],
                    "comparison_score": comparison_summary["score"],
                    "official_score_delta": (
                        float(comparison_summary["score"]) - float(baseline_summary["score"])
                        if is_finite_number(comparison_summary["score"])
                        and is_finite_number(baseline_summary["score"])
                        else float("nan")
                    ),
                    "mean_adj_edge_jaccard_delta": (
                        float(np.mean(deltas)) if deltas else float("nan")
                    ),
                    "improved_sample_count": sum(delta > 0 for delta in deltas),
                    "hurt_sample_count": sum(delta < 0 for delta in deltas),
                    "tied_sample_count": sum(delta == 0 for delta in deltas),
                }
            )
    return result


# %% [markdown]
# ## 7. Diagnostic execution and artifacts


# %%
def validate_expected_samples(rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    expected_count = int(config["validation"]["expected_sample_count"])
    expected_embryos = {
        str(key): int(value)
        for key, value in config["validation"]["expected_embryo_counts"].items()
    }
    for route in sorted({str(row["route"]) for row in rows}):
        route_rows = [row for row in rows if row["route"] == route]
        if len(route_rows) != expected_count:
            raise ValueError(f"{route} has {len(route_rows)} rows; expected {expected_count}")
        observed: dict[str, int] = defaultdict(int)
        for row in route_rows:
            observed[str(row["embryo"])] += 1
        if dict(observed) != expected_embryos:
            raise ValueError(f"{route} embryo counts differ: {dict(observed)}")


def run_diagnostic(working_root: Path | None = None) -> dict[str, Any]:
    started = time.monotonic()
    working_root = working_root or Path.cwd()
    config_path = working_root / "config.yaml"
    metrics_path = working_root / "metrics.json"
    config = yaml.safe_load(config_path.read_text())
    diagnostic_cfg = config["diagnostic"]
    debug = os.environ.get("EXPERIMENT_DEBUG", "0") == "1"
    allow_missing_candidates = debug and not is_kaggle_runtime()

    image_source_cfg = diagnostic_cfg["image_feature_source"]
    image_path = resolve_image_feature_path(working_root, image_source_cfg)
    image_sha = verify_sha(image_path, str(image_source_cfg["sha256"]), "image feature CSV")
    image_features = read_feature_rows(image_path, image_source_cfg)

    all_rows: list[dict[str, Any]] = []
    source_evidence: dict[str, Any] = {}
    candidate_content_evidence: dict[str, dict[str, str]] = {}
    for source_cfg in diagnostic_cfg["prediction_sources"]:
        route = str(source_cfg["route"])
        source_dir, manifest_path, per_sample_path = resolve_prediction_source(
            working_root, source_cfg
        )
        predictions, metric_rows, evidence = load_prediction_rows(
            source_dir, manifest_path, per_sample_path, source_cfg
        )
        route_rows, content_hashes = readout_rows_for_route(
            route,
            source_dir,
            predictions,
            metric_rows,
            image_features,
            config,
            allow_missing_candidate_cache=allow_missing_candidates,
        )
        all_rows.extend(route_rows)
        source_evidence[route] = evidence
        candidate_content_evidence[route] = content_hashes

    validate_expected_samples(all_rows, config)
    baseline_route = str(diagnostic_cfg["baseline_route"])
    comparison_routes = [str(value) for value in diagnostic_cfg["comparison_routes"]]
    feature_columns = {
        str(key): str(value) for key, value in diagnostic_cfg["buckets"]["features"].items()
    }
    quantiles = [float(value) for value in diagnostic_cfg["buckets"]["quantiles"]]
    thresholds = derive_bucket_thresholds(all_rows, baseline_route, feature_columns, quantiles)
    readout_rows = assign_condition_buckets(all_rows, thresholds, feature_columns)
    division_weight = float(config["validation"]["official_division_weight"])
    group_summaries = make_group_summaries(readout_rows, division_weight)
    comparisons = make_paired_comparisons(
        readout_rows, baseline_route, comparison_routes, division_weight
    )

    output_root = working_root / str(diagnostic_cfg["output_dir"])
    per_sample_path = output_root / "per_sample_readout.csv"
    group_path = output_root / "group_error_summary.csv"
    comparison_path = output_root / "paired_route_comparison.csv"
    summary_path = output_root / "group_error_summary.json"
    manifest_path = output_root / "readout_manifest.json"
    write_csv(per_sample_path, readout_rows)
    write_csv(group_path, group_summaries)
    write_csv(comparison_path, comparisons)

    input_bundle_sha = sha256_text(
        json.dumps(
            {
                "image_feature_sha256": image_sha,
                "prediction_sources": source_evidence,
                "candidate_content_sha256": candidate_content_evidence,
            },
            sort_keys=True,
        )
    )
    summary = {
        "experiment": EXPERIMENT,
        "diagnostic_only": True,
        "baseline_route": baseline_route,
        "comparison_routes": comparison_routes,
        "sample_count_per_route": int(config["validation"]["expected_sample_count"]),
        "routes": sorted(source_evidence),
        "threshold_policy": str(diagnostic_cfg["buckets"]["threshold_source"]),
        "bucket_quantiles": quantiles,
        "bucket_thresholds": thresholds,
        "inputs": {
            "image_feature_path": str(image_path),
            "image_feature_sha256": image_sha,
            "prediction_sources": source_evidence,
            "candidate_content_sha256": candidate_content_evidence,
            "input_bundle_sha256": input_bundle_sha,
        },
        "group_summaries": group_summaries,
        "paired_route_comparisons": comparisons,
        "limitations": [
            "Ground truth is used only to aggregate fixed training-data predictions.",
            "Bucket boundaries use opposite-embryo baseline conditions, not held-out outcomes.",
            "Boundary distance describes fixed candidate locations, not a test-time correction.",
            "Only two embryos are available; condition summaries do not establish generalization.",
            "The adopted exp011 public configuration has no fixed prediction artifact yet.",
        ],
    }
    atomic_json(summary_path, summary)

    elapsed_seconds = time.monotonic() - started
    artifact_hashes = {
        "per_sample_readout_sha256": sha256_file(per_sample_path),
        "group_error_summary_csv_sha256": sha256_file(group_path),
        "paired_route_comparison_sha256": sha256_file(comparison_path),
        "group_error_summary_json_sha256": sha256_file(summary_path),
    }
    manifest = {
        "experiment": EXPERIMENT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "diagnostic_only": True,
        "debug": debug,
        "resource": "kaggle_cpu" if is_kaggle_runtime() else "local_cpu_smoke",
        "elapsed_seconds": elapsed_seconds,
        "input_bundle_sha256": input_bundle_sha,
        "source_evidence": source_evidence,
        "artifact_sha256": artifact_hashes,
    }
    atomic_json(manifest_path, manifest)

    metrics = load_json_object(metrics_path)
    metrics_update = {
        "status": "debug_completed",
        "diagnostics": {
            "group_error_readout": {
                "diagnostic_only": True,
                "sample_count_per_route": int(config["validation"]["expected_sample_count"]),
                "route_count": len(source_evidence),
                "group_summary_count": len(group_summaries),
                "paired_comparison_count": len(comparisons),
                "boundary_feature_complete": all(
                    is_finite_number(row["candidate_boundary_distance_um_p50"])
                    for row in readout_rows
                ),
            }
        },
        "evidence": {
            "kaggle": {
                "kernel_source_ids": [
                    str(source["kernel_source"]) for source in diagnostic_cfg["prediction_sources"]
                ]
                + [str(image_source_cfg["kernel_source"])],
                "resource": "cpu",
                "notebook_runtime_seconds": elapsed_seconds,
                "internet_enabled": False,
            },
            "artifacts": {
                "input_file_sha": input_bundle_sha,
                "feature_schema_sha": sha256_text(json.dumps(feature_columns, sort_keys=True)),
                "feature_content_sha": artifact_hashes["per_sample_readout_sha256"],
                "row_count": len(readout_rows),
                "group_count": len(group_summaries),
                "model_count": 0,
                "model_shas": {},
                "selected_mode": "saved_prediction_group_diagnostic",
                "selected_model": None,
                "submission_sha": None,
                "readout_manifest_sha": sha256_file(manifest_path),
                **artifact_hashes,
            },
        },
        "notes": "Diagnostic executed; results remain diagnostic-only pending user review.",
    }
    atomic_json(metrics_path, deep_merge(metrics, metrics_update))
    print(
        json.dumps(
            {
                "routes": sorted(source_evidence),
                "rows": len(readout_rows),
                "groups": len(group_summaries),
                "comparisons": len(comparisons),
                "artifacts": artifact_hashes,
            },
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":
    run_diagnostic()
