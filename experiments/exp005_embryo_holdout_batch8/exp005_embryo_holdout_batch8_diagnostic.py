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
# # exp005 candidate coverage and image-distribution diagnostic
#
# This notebook diagnoses the fixed exp005 embryo-held-out predictions. It
# measures how many annotated ground-truth nodes are present in the saved
# detection candidates before graph selection, compares that coverage with the
# final-graph node recall, and summarizes deterministic image-intensity samples.
# Ground truth is used only for diagnosis. No model, threshold, graph, or
# competition submission is selected or changed here.

# %% [markdown]
# ## Contents
# 1. Configuration, output, and hashing helpers
# 2. Kaggle runtime and pinned input dependencies
# 3. Resolve and validate the fixed exp005 inference artifacts
# 4. Candidate matching and image-sampling helpers
# 5. Run the 199-video diagnostic
# 6. Embryo summaries, image shifts, and associations
# 7. Diagnostic artifacts and metrics update

# %% [markdown]
# ## 1. Configuration, output, and hashing helpers

# %%
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT = "exp005_embryo_holdout_batch8"
INFERENCE_KERNEL_ID = "kentookumura/exp005-embryo-holdout-batch8-inference"
WORKING_ROOT = Path.cwd()
CONFIG_PATH = WORKING_ROOT / "config.yaml"
METRICS_PATH = WORKING_ROOT / "metrics.json"
SOURCE_ROOT = WORKING_ROOT / "official_source"
SOURCE_MANIFEST_PATH = SOURCE_ROOT / "SOURCE.json"
ARTIFACTS_ROOT = WORKING_ROOT / "artifacts" / "diagnostic_v1"
PER_SAMPLE_OUTPUT_PATH = ARTIFACTS_ROOT / "per_sample_candidate_image_diagnostics.csv"
SUMMARY_OUTPUT_PATH = ARTIFACTS_ROOT / "candidate_image_summary.json"
PLOT_OUTPUT_PATH = ARTIFACTS_ROOT / "candidate_image_diagnostics.png"
MANIFEST_OUTPUT_PATH = ARTIFACTS_ROOT / "diagnostic_manifest.json"

config = yaml.safe_load(CONFIG_PATH.read_text())
diagnostic_cfg = config["diagnostics"]["candidate_image_readout"]
validation_cfg = config["validation"]
started = time.monotonic()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def embryo_id(sample_name: str) -> str:
    if "_" not in sample_name:
        raise ValueError(f"sample name has no embryo prefix: {sample_name}")
    return sample_name.split("_", 1)[0]


def find_competition_dir(child: str) -> Path:
    slug = "biohub-cell-tracking-during-development"
    direct = [
        Path("/kaggle/input/competitions") / slug / child,
        Path("/kaggle/input") / slug / child,
    ]
    for path in direct:
        if path.is_dir():
            return path
    matches = sorted(
        path
        for path in Path("/kaggle/input").rglob(child)
        if path.is_dir() and any(path.glob("*.zarr")) and slug in str(path)
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one competition {child} directory, found {matches}")
    return matches[0]


def find_wheels_dir() -> Path:
    candidates = sorted(
        path
        for path in Path("/kaggle/input").rglob("wheels")
        if path.is_dir() and any(path.glob("tracksdata-*.whl"))
    )
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one offline wheels directory, found {candidates}")
    return candidates[0]


def verify_source() -> dict[str, Any]:
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text())
    if manifest["commit"] != config["source"]["commit"]:
        raise RuntimeError("source commit differs from config.yaml")
    mismatches = {}
    for relative, expected in manifest["files"].items():
        observed = sha256_file(SOURCE_ROOT / relative)
        if observed != expected:
            mismatches[relative] = {"expected": expected, "observed": observed}
    if mismatches:
        raise RuntimeError(f"pinned source hash mismatch: {mismatches}")
    return manifest


# %% [markdown]
# ## 2. Kaggle runtime and pinned input dependencies

# %%
if not Path("/kaggle/input").is_dir():
    raise RuntimeError("The authoritative diagnostic must execute on Kaggle.")

wheels_dir = find_wheels_dir()
offline_packages = [
    "bidict==0.23.1",
    "donfig==0.8.1.post1",
    "geff==1.2.0.1.1",
    "geff-spec==1.1.1",
    "ilpy==0.6.0",
    "numcodecs==0.15.1",
    "polars==1.42.0",
    "polars-runtime-32==1.42.0",
    "pyscipopt==6.2.1",
    "rustworkx==0.18.0",
    "tracksdata==0.1.0rc6.dev3+g980c2d30a",
    "zarr==3.2.1",
]
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(wheels_dir),
        "--no-deps",
        *offline_packages,
    ],
    check=True,
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import scipy  # noqa: E402
import tracksdata as td  # noqa: E402
import zarr  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from tracksdata.metrics import DistanceMatching  # noqa: E402
from tracksdata.options import get_options, set_options  # noqa: E402

source_manifest = verify_source()
source_manifest_sha = sha256_file(SOURCE_MANIFEST_PATH)
sys.path.insert(0, str(SOURCE_ROOT / "src"))
from tracking_cellmot.io import open_dataset  # noqa: E402

print(
    "Diagnostic runtime:",
    {
        "resource": "cpu",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_commit": source_manifest["commit"],
    },
)

# %% [markdown]
# ## 3. Resolve and validate the fixed exp005 inference artifacts
#
# The attached inference kernel is the only prediction source. Its 199 cache
# files were generated with detection threshold 0.99. This notebook does not
# reconstruct lower-threshold candidates from that post-threshold cache.

# %%


def find_inference_artifacts() -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in Path("/kaggle/input").rglob("prediction_manifest.json"):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if (
            payload.get("experiment") == EXPERIMENT
            and int(payload.get("completed_count", -1))
            == int(validation_cfg["expected_sample_count"])
            and (path.parent / "candidates").is_dir()
        ):
            matches.append((path, payload))
    if len(matches) != 1:
        raise FileNotFoundError(
            "expected one completed exp005 inference artifact root, found "
            + repr([str(path) for path, _ in matches])
        )
    return matches[0]


prediction_manifest_path, prediction_manifest = find_inference_artifacts()
inference_artifacts_root = prediction_manifest_path.parent
per_sample_metrics_path = inference_artifacts_root / "per_sample_metrics.json"
official_summary_path = inference_artifacts_root / "official_metric_summary.json"
if not per_sample_metrics_path.is_file() or not official_summary_path.is_file():
    raise FileNotFoundError("fixed inference metrics are missing beside prediction manifest")

per_sample_payload = json.loads(per_sample_metrics_path.read_text())
official_summary_payload = json.loads(official_summary_path.read_text())
prediction_records = prediction_manifest["predictions"]
metric_rows = per_sample_payload["rows"]
prediction_by_sample = {str(row["sample"]): row for row in prediction_records}
metric_by_sample = {str(row["sample"]): row for row in metric_rows}
expected_count = int(validation_cfg["expected_sample_count"])
if len(prediction_by_sample) != expected_count or len(metric_by_sample) != expected_count:
    raise RuntimeError("prediction and metric manifests must each contain 199 unique samples")
if set(prediction_by_sample) != set(metric_by_sample):
    raise RuntimeError("prediction and metric sample sets differ")
if float(config["model"]["inference"]["det_threshold"]) != float(
    diagnostic_cfg["fixed_detection_threshold"]
):
    raise RuntimeError("diagnostic threshold differs from the fixed inference threshold")

train_dir = find_competition_dir("train")
paired_stems = sorted(
    path.name[:-5]
    for path in train_dir.glob("*.zarr")
    if (train_dir / f"{path.name[:-5]}.geff").exists()
)
if set(paired_stems) != set(prediction_by_sample):
    raise RuntimeError("Kaggle train pairs differ from the fixed prediction sample set")
for sample_name, record in prediction_by_sample.items():
    candidate_path = inference_artifacts_root / str(record["candidate_cache"])
    if not candidate_path.is_file():
        raise FileNotFoundError(f"candidate cache is missing: {candidate_path}")
    if embryo_id(sample_name) != str(record["embryo"]):
        raise RuntimeError(f"embryo prefix mismatch for {sample_name}")

print(
    "Fixed diagnostic inputs:",
    {
        "inference_kernel": INFERENCE_KERNEL_ID,
        "sample_count": len(paired_stems),
        "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
        "per_sample_metrics_sha256": sha256_file(per_sample_metrics_path),
    },
)

# %% [markdown]
# ## 4. Candidate matching and image-sampling helpers
#
# Candidate coverage uses the same 7 micrometre physical-distance matcher as the
# official node recall. Image statistics use three fixed frames and fixed spatial
# strides per video. They characterize a deterministic sample, not every voxel.

# %%


def build_candidate_graph(coords_tzyx: np.ndarray) -> td.graph.InMemoryGraph:
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    if len(coords_tzyx):
        graph.bulk_add_nodes(
            [
                {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
                for t, z, y, x in coords_tzyx
            ]
        )
    return graph


def matched_candidate_nodes(
    coords_tzyx: np.ndarray,
    truth_graph: td.graph.BaseGraph,
    scale: tuple[float, ...],
    max_distance_um: float,
) -> tuple[int, int, float]:
    gt_count = int(truth_graph.num_nodes())
    if gt_count == 0 or len(coords_tzyx) == 0:
        return 0, gt_count, 0.0 if gt_count else float("nan")
    graph = build_candidate_graph(coords_tzyx)
    previous_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
        graph.match(
            truth_graph,
            matching=DistanceMatching(max_distance=max_distance_um, scale=scale),
        )
    finally:
        set_options(show_progress=previous_progress)
    attrs = graph.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    )
    matched_ids = attrs.filter(
        pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID).is_not_null()
        & (pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID) != -1)
    )[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    matched_count = int(matched_ids.n_unique())
    return matched_count, gt_count, matched_count / gt_count


def lookup_quantile(quantiles: dict[str, Any], target: float) -> float:
    for key, value in quantiles.items():
        try:
            if abs(float(key) - target) <= 1e-9:
                return float(value)
        except (TypeError, ValueError):
            continue
    raise KeyError(f"image quantile {target} is missing")


def finite_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p99": float("nan"),
        }
    q50, q90, q99 = np.quantile(values, [0.5, 0.9, 0.99])
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p50": float(q50),
        f"{prefix}_p90": float(q90),
        f"{prefix}_p99": float(q99),
    }


def sample_image_statistics(
    dataset: Any,
    frame_sample_count: int,
    spatial_stride_zyx: tuple[int, int, int],
) -> dict[str, Any]:
    image_array = zarr.open_group(str(dataset.zarr_path), mode="r")["0"]
    frame_count = int(image_array.shape[0])
    frame_indices = sorted(
        set(int(index) for index in np.rint(np.linspace(0, frame_count - 1, frame_sample_count)))
    )
    z_stride, y_stride, x_stride = spatial_stride_zyx
    samples = [
        np.asarray(
            image_array[
                frame,
                slice(None, None, z_stride),
                slice(None, None, y_stride),
                slice(None, None, x_stride),
            ],
            dtype=np.float32,
        ).reshape(-1)
        for frame in frame_indices
    ]
    raw = np.concatenate(samples)
    q_low = lookup_quantile(dataset.quantiles, 0.001)
    q_high = lookup_quantile(dataset.quantiles, 0.999)
    q_range = q_high - q_low
    if not q_range > 0:
        raise RuntimeError(f"non-positive normalization range: {q_low}, {q_high}")
    normalized = np.maximum((raw - q_low) / (q_range + 1e-6), 0.0)
    return {
        "image_frame_count": frame_count,
        "image_sampled_frame_count": len(frame_indices),
        "image_sampled_voxel_count": int(raw.size),
        "image_attr_q001": q_low,
        "image_attr_q999": q_high,
        "image_attr_qrange": q_range,
        **finite_stats("image_raw", raw),
        **finite_stats("image_normalized", normalized),
        "image_normalized_zero_fraction": float(np.mean(normalized <= 0.0)),
        "image_normalized_above_one_fraction": float(np.mean(normalized > 1.0)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_safe(rows))


# %% [markdown]
# ## 5. Run the 199-video diagnostic

# %%
frame_sample_count = int(diagnostic_cfg["image_sampling"]["frame_count"])
spatial_stride_zyx = tuple(
    int(value) for value in diagnostic_cfg["image_sampling"]["spatial_stride_zyx"]
)
max_distance_um = float(validation_cfg["max_matching_distance_um"])
rows: list[dict[str, Any]] = []

for sample_name in tqdm(paired_stems, desc="candidate/image diagnostic"):
    prediction_record = prediction_by_sample[sample_name]
    metric_row = metric_by_sample[sample_name]
    candidate_path = inference_artifacts_root / str(prediction_record["candidate_cache"])
    with np.load(candidate_path, allow_pickle=False) as candidate_cache:
        coords_tzyx = np.asarray(candidate_cache["coords_tzyx"])
        detection_probability = np.asarray(candidate_cache["detection_probability"])
    if len(coords_tzyx) != int(prediction_record["node_count"]):
        raise RuntimeError(f"candidate node count differs for {sample_name}")
    if len(detection_probability) != len(coords_tzyx):
        raise RuntimeError(f"candidate detection scores are misaligned for {sample_name}")
    if len(detection_probability) and not np.all(
        detection_probability > float(diagnostic_cfg["fixed_detection_threshold"])
    ):
        raise RuntimeError(f"post-threshold candidate cache contains invalid scores: {sample_name}")

    dataset = open_dataset(
        train_dir / f"{sample_name}.zarr",
        normalize=False,
        require_tracks=True,
        load_image=False,
    )
    if dataset.tracks is None:
        raise RuntimeError(f"ground-truth graph is missing for {sample_name}")
    matched_count, gt_count, candidate_recall = matched_candidate_nodes(
        coords_tzyx,
        dataset.tracks,
        tuple(float(value) for value in dataset.scale),
        max_distance_um,
    )
    final_recall = float(metric_row["node_recall"])
    final_matched_count = int(round(final_recall * gt_count))
    if final_matched_count > matched_count:
        raise RuntimeError(
            f"final graph matches more GT nodes than its source candidates for {sample_name}"
        )
    image_stats = sample_image_statistics(
        dataset,
        frame_sample_count=frame_sample_count,
        spatial_stride_zyx=spatial_stride_zyx,
    )
    candidate_count = int(len(coords_tzyx))
    final_node_count = int(metric_row["num_pred_nodes"])
    rows.append(
        {
            "sample": sample_name,
            "embryo": embryo_id(sample_name),
            "fold": int(metric_row["fold"]),
            "gt_annotated_nodes": gt_count,
            "candidate_matched_gt_nodes": matched_count,
            "final_matched_gt_nodes": final_matched_count,
            "candidate_node_recall": candidate_recall,
            "final_node_recall": final_recall,
            "post_candidate_recall_loss": candidate_recall - final_recall,
            "candidate_nodes": candidate_count,
            "final_pred_nodes": final_node_count,
            "candidate_to_final_node_retention": (
                final_node_count / candidate_count if candidate_count else float("nan")
            ),
            "candidate_nodes_per_frame": candidate_count / image_stats["image_frame_count"],
            "candidate_score_min": (
                float(np.min(detection_probability)) if len(detection_probability) else float("nan")
            ),
            "candidate_score_mean": (
                float(np.mean(detection_probability))
                if len(detection_probability)
                else float("nan")
            ),
            "candidate_score_p50": (
                float(np.quantile(detection_probability, 0.5))
                if len(detection_probability)
                else float("nan")
            ),
            "candidate_score_p90": (
                float(np.quantile(detection_probability, 0.9))
                if len(detection_probability)
                else float("nan")
            ),
            "official_adj_edge_jaccard": float(metric_row["adj_edge_jaccard"]),
            "physical_scale_z": float(dataset.scale[0]),
            "physical_scale_y": float(dataset.scale[1]),
            "physical_scale_x": float(dataset.scale[2]),
            **image_stats,
        }
    )

if len(rows) != expected_count:
    raise RuntimeError(f"expected {expected_count} diagnostic rows, found {len(rows)}")
write_csv(PER_SAMPLE_OUTPUT_PATH, rows)

# %% [markdown]
# ## 6. Embryo summaries, image shifts, and associations
#
# Recall summaries include both the official-style per-video mean and a
# GT-node-weighted value. Feature associations are descriptive Spearman
# correlations within each embryo; they are not threshold-selection evidence.

# %%


def finite_values(group: list[dict[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([row[key] for row in group], dtype=np.float64)
    return values[np.isfinite(values)]


def aggregate_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    gt_total = sum(int(row["gt_annotated_nodes"]) for row in group)
    candidate_matched = sum(int(row["candidate_matched_gt_nodes"]) for row in group)
    final_matched = sum(int(row["final_matched_gt_nodes"]) for row in group)
    keys = [
        "candidate_node_recall",
        "final_node_recall",
        "post_candidate_recall_loss",
        "candidate_nodes_per_frame",
        "candidate_to_final_node_retention",
        "candidate_score_p50",
        "image_attr_q001",
        "image_attr_q999",
        "image_attr_qrange",
        "image_normalized_mean",
        "image_normalized_std",
        "image_normalized_p99",
        "image_normalized_zero_fraction",
        "image_normalized_above_one_fraction",
        "official_adj_edge_jaccard",
    ]
    result: dict[str, Any] = {
        "sample_count": len(group),
        "gt_annotated_node_count": gt_total,
        "candidate_matched_gt_node_count": candidate_matched,
        "final_matched_gt_node_count": final_matched,
        "candidate_node_recall_micro": candidate_matched / gt_total,
        "final_node_recall_micro": final_matched / gt_total,
    }
    for key in keys:
        values = finite_values(group, key)
        result[f"{key}_mean"] = float(np.mean(values)) if len(values) else float("nan")
        result[f"{key}_median"] = float(np.median(values)) if len(values) else float("nan")
    return result


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    greater = sum(int(np.sum(value > right)) for value in left)
    less = sum(int(np.sum(value < right)) for value in left)
    return (greater - less) / (len(left) * len(right))


def spearman_record(group: list[dict[str, Any]], feature: str, target: str) -> dict[str, Any]:
    x = np.asarray([row[feature] for row in group], dtype=np.float64)
    y = np.asarray([row[target] for row in group], dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return {"n": int(valid.sum()), "rho": float("nan"), "pvalue": float("nan")}
    result = spearmanr(x[valid], y[valid])
    return {"n": int(valid.sum()), "rho": float(result.statistic), "pvalue": float(result.pvalue)}


grouped = {
    embryo: [row for row in rows if row["embryo"] == embryo]
    for embryo in sorted({str(row["embryo"]) for row in rows})
}
if {embryo: len(group) for embryo, group in grouped.items()} != validation_cfg[
    "expected_embryo_counts"
]:
    raise RuntimeError("diagnostic embryo counts differ from the validation contract")

by_embryo = {embryo: aggregate_group(group) for embryo, group in grouped.items()}
overall = aggregate_group(rows)
for embryo, expected in official_summary_payload["by_embryo"].items():
    observed = by_embryo[embryo]["final_node_recall_mean"]
    if not math.isclose(observed, float(expected["node_recall"]), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"final node recall does not reproduce for embryo {embryo}")

shift_features = [
    "candidate_nodes_per_frame",
    "candidate_score_p50",
    "image_attr_q001",
    "image_attr_q999",
    "image_attr_qrange",
    "image_normalized_mean",
    "image_normalized_std",
    "image_normalized_p99",
    "image_normalized_zero_fraction",
    "image_normalized_above_one_fraction",
]
image_feature_shift = {}
for feature in shift_features:
    values_44b6 = finite_values(grouped["44b6"], feature)
    values_6bba = finite_values(grouped["6bba"], feature)
    image_feature_shift[feature] = {
        "median_44b6": float(np.median(values_44b6)),
        "median_6bba": float(np.median(values_6bba)),
        "median_difference_6bba_minus_44b6": float(np.median(values_6bba) - np.median(values_44b6)),
        "cliffs_delta_6bba_vs_44b6": cliffs_delta(values_6bba, values_44b6),
    }

association_features = [
    "candidate_nodes_per_frame",
    "candidate_score_p50",
    "image_attr_qrange",
    "image_normalized_mean",
    "image_normalized_std",
    "image_normalized_p99",
    "image_normalized_zero_fraction",
    "image_normalized_above_one_fraction",
]
associations = {
    embryo: {
        target: {
            feature: spearman_record(group, feature, target) for feature in association_features
        }
        for target in ("candidate_node_recall", "final_node_recall")
    }
    for embryo, group in grouped.items()
}

summary = {
    "experiment": EXPERIMENT,
    "diagnostic": "fixed_threshold_candidate_coverage_and_image_distribution",
    "diagnostic_only": True,
    "sample_count": len(rows),
    "fixed_detection_threshold": float(diagnostic_cfg["fixed_detection_threshold"]),
    "max_matching_distance_um": max_distance_um,
    "image_sampling": {
        "frame_count": frame_sample_count,
        "spatial_stride_zyx": list(spatial_stride_zyx),
        "deterministic": True,
        "scope": "sampled_voxels_not_full_volume",
    },
    "input": {
        "inference_kernel_id": INFERENCE_KERNEL_ID,
        "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
        "per_sample_metrics_sha256": sha256_file(per_sample_metrics_path),
        "source_manifest_sha256": source_manifest_sha,
        "candidate_schema_sha256": prediction_manifest["candidate_schema_sha256"],
    },
    "overall": overall,
    "by_embryo": by_embryo,
    "feature_shift": image_feature_shift,
    "within_embryo_spearman": associations,
    "worst_candidate_coverage_samples": [
        {
            "sample": row["sample"],
            "embryo": row["embryo"],
            "candidate_node_recall": row["candidate_node_recall"],
            "final_node_recall": row["final_node_recall"],
            "candidate_nodes_per_frame": row["candidate_nodes_per_frame"],
        }
        for row in sorted(rows, key=lambda item: item["candidate_node_recall"])[:10]
    ],
    "largest_post_candidate_losses": [
        {
            "sample": row["sample"],
            "embryo": row["embryo"],
            "candidate_node_recall": row["candidate_node_recall"],
            "final_node_recall": row["final_node_recall"],
            "post_candidate_recall_loss": row["post_candidate_recall_loss"],
        }
        for row in sorted(
            rows,
            key=lambda item: item["post_candidate_recall_loss"],
            reverse=True,
        )[:10]
    ],
    "limitations": [
        "Ground-truth matching is diagnostic only and is not available in test inference.",
        "The cache contains local maxima strictly above 0.99; it cannot measure 0.965 coverage.",
        "Image statistics use deterministic sparse samples and do not represent every voxel.",
        "Associations across only two embryos are descriptive and do not establish causality.",
    ],
}

# %% [markdown]
# ## 7. Diagnostic artifacts and metrics update

# %%
ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
atomic_json(SUMMARY_OUTPUT_PATH, summary)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
plot_features = [
    ("candidate_node_recall", "Candidate node recall"),
    ("final_node_recall", "Final-graph node recall"),
    ("candidate_nodes_per_frame", "Candidates per frame"),
    ("candidate_score_p50", "Median detection probability"),
    ("image_normalized_mean", "Sampled normalized mean"),
    ("image_normalized_std", "Sampled normalized std"),
]
labels = sorted(grouped)
for axis, (feature, title) in zip(axes.flat, plot_features, strict=True):
    axis.boxplot(
        [finite_values(grouped[embryo], feature) for embryo in labels],
        tick_labels=labels,
        showfliers=False,
    )
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
fig.suptitle("exp005 fixed-candidate and sampled-image diagnostic")
fig.tight_layout()
fig.savefig(PLOT_OUTPUT_PATH, dpi=160)
plt.close(fig)

elapsed_seconds = time.monotonic() - started
artifact_hashes = {
    "per_sample_csv_sha256": sha256_file(PER_SAMPLE_OUTPUT_PATH),
    "summary_json_sha256": sha256_file(SUMMARY_OUTPUT_PATH),
    "plot_png_sha256": sha256_file(PLOT_OUTPUT_PATH),
}
diagnostic_manifest = {
    "experiment": EXPERIMENT,
    "diagnostic": summary["diagnostic"],
    "diagnostic_only": True,
    "sample_count": len(rows),
    "elapsed_seconds": elapsed_seconds,
    "inputs": summary["input"],
    "artifacts": artifact_hashes,
}
atomic_json(MANIFEST_OUTPUT_PATH, diagnostic_manifest)

metrics = json.loads(METRICS_PATH.read_text())
metrics_update = {
    "diagnostics": {
        "candidate_image_readout": {
            "diagnostic_only": True,
            "sample_count": len(rows),
            "fixed_detection_threshold": summary["fixed_detection_threshold"],
            "overall": overall,
            "by_embryo": by_embryo,
            "feature_shift": image_feature_shift,
        }
    },
    "evidence": {
        "artifacts": {
            "diagnostic_manifest_sha": sha256_file(MANIFEST_OUTPUT_PATH),
            **artifact_hashes,
        },
        "kaggle": {
            "diagnostic": {
                "resource": "cpu",
                "internet_enabled": False,
                "notebook_runtime_seconds": elapsed_seconds,
                "source_kernel": INFERENCE_KERNEL_ID,
            }
        },
    },
}
atomic_json(METRICS_PATH, deep_merge(metrics, metrics_update))

print("Diagnostic by embryo:")
print(
    json.dumps(
        {
            embryo: {
                "sample_count": values["sample_count"],
                "candidate_node_recall_mean": values["candidate_node_recall_mean"],
                "final_node_recall_mean": values["final_node_recall_mean"],
                "post_candidate_recall_loss_mean": values["post_candidate_recall_loss_mean"],
                "candidate_nodes_per_frame_median": values["candidate_nodes_per_frame_median"],
            }
            for embryo, values in by_embryo.items()
        },
        indent=2,
    )
)
print("Diagnostic runtime seconds:", elapsed_seconds)
print("Diagnostic artifacts:", json.dumps(artifact_hashes, indent=2))
