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
# # exp032: three-frame tracker, ten epochs on Colab
#
# The public image encoder and candidate points remain fixed. This notebook
# trains the three-timepoint local attention model while preserving
# the central-pair teacher, loss, and planned graph decoder.
#
# This train stage reports fixed-candidate edge metrics. It does not claim the
# official graph metric and it does not create a Kaggle submission.

# %% [markdown]
# ## 1. Configuration and runtime guard

# %%
from __future__ import annotations

# ruff: noqa: E402,I001
import copy
import os
import zipfile
import hashlib
import shutil
import importlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT = "exp032_three_frame_ten_epoch_training"
COMPETITION = "biohub-cell-tracking-during-development"
NOTEBOOK_STARTED = time.perf_counter()

# CLI mode uses the same model and metric code with local inputs and outputs.
CLI_MODE = os.environ.get("EXP032_CLI_MODE") == "1"
DRIVE_PROJECT_ROOT = (
    Path("/content/exp032_cli_bundle") if CLI_MODE else Path("/content/drive/MyDrive/Kaggle/Biohub")
)
if not CLI_MODE:
    from google.colab import drive

    drive.mount("/content/drive")
    BUNDLE_ON_DRIVE = DRIVE_PROJECT_ROOT / "exp032_three_frame_ten_epoch_training_colab_bundle.zip"
    if not BUNDLE_ON_DRIVE.is_file():
        raise FileNotFoundError(f"Upload the exp032 Colab ZIP to {BUNDLE_ON_DRIVE}")
    with zipfile.ZipFile(BUNDLE_ON_DRIVE) as bundle:
        manifest = json.loads(bundle.read("BUNDLE_MANIFEST.json"))
        if (
            manifest.get("experiment") != EXPERIMENT
            or manifest.get("contains_credentials") is not False
        ):
            raise RuntimeError("Unexpected Colab bundle manifest")
        listed = {item["path"] for item in manifest["files"]}
        if set(bundle.namelist()) != listed | {"BUNDLE_MANIFEST.json"}:
            raise RuntimeError("Colab bundle file list differs from its manifest")
        for item in manifest["files"]:
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise RuntimeError(f"Unsafe Colab bundle path: {relative}")
            target = DRIVE_PROJECT_ROOT / relative
            if target.is_file():
                if target.suffix == ".ipynb":
                    continue  # Colab may save outputs into an opened notebook.
                with target.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if digest != item["sha256"]:
                    raise RuntimeError(f"Drive file differs from Colab ZIP: {relative}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name("." + target.name + ".tmp")
            with bundle.open(item["path"]) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            with temporary.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != item["sha256"]:
                temporary.unlink()
                raise RuntimeError(f"Colab ZIP checksum failed: {relative}")
            temporary.replace(target)

if not (DRIVE_PROJECT_ROOT / "project.yml").is_file():
    raise FileNotFoundError(f"Missing project.yml under {DRIVE_PROJECT_ROOT}")
EXPERIMENT_ROOT = DRIVE_PROJECT_ROOT / "experiments" / EXPERIMENT
CONFIG_PATH = EXPERIMENT_ROOT / "config.yaml"
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(CONFIG_PATH)
config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
validation_cfg = config["validation"]
cache_cfg = config["data"]["cache"]
model_cfg = config["model"]
train_cfg = model_cfg["training"]
run_stage = str(train_cfg.get("stage", "full"))
pilot_mode = run_stage == "pilot"
if run_stage not in {"pilot", "full"}:
    raise RuntimeError(f"unknown training stage: {run_stage}")
teacher_cfg = model_cfg["teacher"]
loss_cfg = model_cfg["loss"]
params_cfg = model_cfg["params"]
public_cfg = model_cfg["public_source"]
runtime_cfg = config["runtime"]
colab_cfg = runtime_cfg["colab"]
if CLI_MODE:
    if Path(colab_cfg["cli"]["bundle_root"]) != DRIVE_PROJECT_ROOT:
        raise RuntimeError("CLI bundle root differs from config")
    if colab_cfg["cli"]["cache_acquisition"] != "kaggle_signed_manifest":
        raise RuntimeError("CLI cache must be acquired through the Kaggle signed manifest")
    if colab_cfg["cli"]["geff_acquisition"] != "kaggle_export_signed_manifest":
        raise RuntimeError("CLI GEFF must be acquired from Kaggle directly")
else:
    if Path(colab_cfg["drive_project_root"]) != DRIVE_PROJECT_ROOT:
        raise RuntimeError("Colab project root in config differs from the notebook")
    if colab_cfg["cache_acquisition"] != "kaggle_kernel_output":
        raise RuntimeError("This Colab notebook requires the Kaggle exp015 cache download")
WORKING_ROOT = (
    Path(colab_cfg["cli"]["run_root"])
    if CLI_MODE
    else DRIVE_PROJECT_ROOT / colab_cfg["persistent_runs_dir"] / colab_cfg["run_id"]
)
WORKING_ROOT.mkdir(parents=True, exist_ok=True)
METRICS_PATH = WORKING_ROOT / "metrics.json"
OUTPUT_MODELS = WORKING_ROOT / "models"
config_snapshot = WORKING_ROOT / "config.yaml"
if config_snapshot.exists() and config_snapshot.read_bytes() != CONFIG_PATH.read_bytes():
    raise RuntimeError("Saved run uses a different config; choose another run_id")
shutil.copy2(CONFIG_PATH, config_snapshot)
if not METRICS_PATH.exists():
    shutil.copy2(EXPERIMENT_ROOT / "metrics.json", METRICS_PATH)
EVENT_LOG = WORKING_ROOT / "run_events.jsonl"


def log_event(event: str, **fields: Any) -> None:
    row = {"time": datetime.now(UTC).isoformat(), "event": event, **fields}
    with EVENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    print(json.dumps(row, sort_keys=True, default=str), flush=True)


log_event("start", experiment=EXPERIMENT, run_id=colab_cfg["run_id"])

if model_cfg["trainable_components"] != ["primary_LocalThreeFrameTracker"]:
    raise RuntimeError("local primary tracker must be the only trainable component")
if train_cfg["active_variants"] != ["three_frame_local"] or train_cfg["epochs"] != 10:
    raise RuntimeError("this run trains only three_frame_local for ten epochs")
if model_cfg["control"]["retrain"] is not False:
    raise RuntimeError("the public control must not be retrained")
if model_cfg["inference"]["implemented"] is not False:
    raise RuntimeError("this notebook is the train-only stage")

print("Experiment:", EXPERIMENT)
print("Outer folds:", json.dumps(validation_cfg["outer_folds"], indent=2))
print("Trainable components:", model_cfg["trainable_components"])
print("Frozen components:", model_cfg["frozen_components"])

# %% [markdown]
# ## 2. Colab dependencies and experiment helpers

# %%
# Colab uses its own Python environment. Install the GEFF readers before imports.
DEPENDENCIES = (
    "kaggle==2.2.4",
    "blosc2>=3.7,<5",
    "geff>=1.2,<2",
    "geff-spec>=1.1,<2",
    "ilpy>=0.6,<1",
    "polars>=1.42,<2",
    "pyarrow>=20,<25",
    "pyscipopt>=6.0,<7",
    "rustworkx>=0.17,<1",
    "tracksdata>=0.1.0rc6,<0.2",
    "zarr>=3.0.10,<4",
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        *DEPENDENCIES,
    ],
    check=True,
)
importlib.invalidate_caches()
import torch
from torch.utils.data import DataLoader

if not torch.cuda.is_available():
    raise RuntimeError("Select a GPU Colab runtime before staging large inputs")
disk = shutil.disk_usage("/content")
log_event(
    "runtime_preflight",
    gpu=torch.cuda.get_device_name(0),
    python=sys.version.split()[0],
    torch_version=torch.__version__,
    free_disk_bytes=disk.free,
)

# %% [markdown]
# ## 2a. Cache, annotation, and diagnostic helpers

# %%

import hashlib
import random
from dataclasses import dataclass

import numpy as np

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
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
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
    return_cache_arrays: bool = False,
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
        source_matches,
        target_matches,
        annotation.edges,
        outgoing_edges=annotation.outgoing_edges,
    )
    active_mask = legacy_active_pair_mask(target)
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
            "division_parents": int(np.count_nonzero(target.sum(axis=1) > 1)),
            "source_match_distance_sum_um": float(np.nansum(source_distances)),
            "source_match_distance_count": int(np.count_nonzero(np.isfinite(source_distances))),
            "target_match_distance_sum_um": float(np.nansum(target_distances)),
            "target_match_distance_count": int(np.count_nonzero(np.isfinite(target_distances))),
        },
    }
    if return_cache_arrays:
        example["_cache_arrays"] = arrays
        example["_cache_metadata"] = metadata
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


def select_pilot_outer_paths(
    paths: list[Path], samples: list[str], count: int, seed: int
) -> list[Path]:
    """Choose one eligible window per sample without inspecting edge labels."""
    if count < 1:
        raise ValueError("pilot evaluation window count must be positive")
    allowed = set(samples)
    by_sample: dict[str, list[Path]] = {}
    for path in paths:
        if path.parent.name in allowed:
            by_sample.setdefault(path.parent.name, []).append(path)
    if len(by_sample) < count:
        raise ValueError(f"only {len(by_sample)} samples have eligible outer windows; need {count}")

    def rank(value: str) -> bytes:
        return hashlib.sha256(f"{seed}:{value}".encode()).digest()

    selected = [
        min(by_sample[sample], key=lambda path: rank(f"window:{sample}/{path.name}"))
        for sample in sorted(by_sample, key=lambda name: rank(f"sample:{name}"))[:count]
    ]
    return sorted(selected)


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
    return model(batch, include_previous=bool(getattr(model, "include_previous", True)))


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
            loss = batch_legacy_focal_bce(
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
        "legacy_mask_loss": safe_ratio(total_loss, windows),
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
    loss_windows = 0
    correct_pairs = 0
    false_positive_active_pairs = 0
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
                    false_positive_active_pairs += int(
                        (predictions[pair_mask] & (pair_target[pair_mask] < 0.5)).sum()
                    )
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
        "false_positive_active_pair_count": false_positive_active_pairs,
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


# Three-frame extension. The central target and loss above remain unchanged.


def build_three_frame_example(
    path: Path,
    annotation: AnnotationGraph,
    *,
    feature_channels: int,
    expected_primary_checkpoint_sha256: str,
    max_matching_distance_um: float,
    downsample_zyx: tuple[float, float, float],
) -> dict[str, Any]:
    """Keep the central pair and read earlier points from the preceding image window."""
    central = build_window_example(
        path,
        annotation,
        feature_channels=feature_channels,
        expected_primary_checkpoint_sha256=expected_primary_checkpoint_sha256,
        max_matching_distance_um=max_matching_distance_um,
        downsample_zyx=downsample_zyx,
        return_cache_arrays=True,
    )
    arrays = central.pop("_cache_arrays")
    metadata = central.pop("_cache_metadata")
    source_frame, _ = map(int, metadata["window_frames"])
    central["coords_src_physical"] = np.asarray(arrays["coords_src_physical"], dtype=np.float32)
    central["coords_tgt_physical"] = np.asarray(arrays["coords_tgt_physical"], dtype=np.float32)
    prior_path = path.with_name(f"{source_frame - 1:06d}_{source_frame:06d}.npz")
    if source_frame == 0:
        central["features_prev"] = np.zeros((0, central["features_src"].shape[1]), dtype=np.float32)
        central["coords_prev_physical"] = np.zeros((0, 3), dtype=np.float32)
        central["prior_cache_content_sha256"] = None
        return central
    if not prior_path.is_file():
        raise FileNotFoundError(f"previous window missing within video: {prior_path}")

    prior_arrays, prior_metadata = validate_window_cache(
        prior_path,
        feature_channels=feature_channels,
        expected_primary_checkpoint_sha256=expected_primary_checkpoint_sha256,
    )
    if prior_metadata["window_frames"] != [source_frame - 1, source_frame]:
        raise ValueError(f"incorrect previous window: {prior_path}")
    # Candidate identity and position must survive the overlap. Window-dependent
    # image features need not agree and are deliberately kept separate.
    for label, prior_key, central_key in (
        ("ids", "candidate_ids_tgt", "candidate_ids_src"),
        ("grid", "coords_tgt_grid", "coords_src_grid"),
        ("physical", "coords_tgt_physical", "coords_src_physical"),
    ):
        if not np.array_equal(prior_arrays[prior_key], arrays[central_key]):
            raise ValueError(f"overlapping window {label} mismatch: {prior_path} {path}")
    central["features_prev"] = np.concatenate(
        [prior_arrays["primary_features_src"], prior_arrays["position_features_src"]], axis=1
    ).astype(np.float32, copy=False)
    central["coords_prev_physical"] = np.asarray(
        prior_arrays["coords_src_physical"], dtype=np.float32
    )
    central["prior_cache_content_sha256"] = prior_metadata["array_content_sha256"]
    return central


class ThreeFrameWindowDataset:
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
        return build_three_frame_example(
            path,
            self.annotations[path.parent.name],
            feature_channels=self.feature_channels,
            expected_primary_checkpoint_sha256=self.expected_primary_checkpoint_sha256,
            max_matching_distance_um=self.max_matching_distance_um,
            downsample_zyx=self.downsample_zyx,
        )


def collate_three_frame_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch = collate_window_examples(examples)
    batch_size = len(examples)
    feature_dim = int(examples[0]["features_src"].shape[1])
    max_prev = max(len(example["features_prev"]) for example in examples)
    features_prev = torch.zeros(batch_size, max_prev, feature_dim)
    coords_prev_physical = torch.zeros(batch_size, max_prev, 3)
    coords_src_physical = torch.zeros(batch_size, batch["features_src"].shape[1], 3)
    coords_tgt_physical = torch.zeros(batch_size, batch["features_tgt"].shape[1], 3)
    prev_mask = torch.zeros(batch_size, max_prev, dtype=torch.bool)
    for index, example in enumerate(examples):
        n_prev = len(example["features_prev"])
        n_src = len(example["features_src"])
        n_tgt = len(example["features_tgt"])
        features_prev[index, :n_prev] = torch.from_numpy(example["features_prev"])
        coords_prev_physical[index, :n_prev] = torch.from_numpy(example["coords_prev_physical"])
        coords_src_physical[index, :n_src] = torch.from_numpy(example["coords_src_physical"])
        coords_tgt_physical[index, :n_tgt] = torch.from_numpy(example["coords_tgt_physical"])
        prev_mask[index, :n_prev] = True
        batch["metadata"][index]["previous_candidate_count"] = n_prev
        batch["metadata"][index]["prior_cache_content_sha256"] = example[
            "prior_cache_content_sha256"
        ]
    batch.update(
        features_prev=features_prev,
        coords_prev_physical=coords_prev_physical,
        coords_src_physical=coords_src_physical,
        coords_tgt_physical=coords_tgt_physical,
        prev_mask=prev_mask,
    )
    return batch


# %% [markdown]
# ## 2a. Local three-frame attention model

# %%


from torch import nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class LocalAttentionBlock(nn.Module):
    """Public block parameters applied to gathered spatial neighbors."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True, dropout=dropout
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, hidden: torch.Tensor, indices: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        batch_size, node_count, hidden_dim = hidden.shape
        batch_index = torch.arange(batch_size, device=hidden.device)[:, None, None]
        neighbors = hidden[batch_index, indices]
        queries = self.norm1(hidden).reshape(batch_size * node_count, 1, hidden_dim)
        keys = self.norm1(neighbors).reshape(batch_size * node_count, indices.shape[-1], hidden_dim)
        output, _ = self.cross_attn(
            queries,
            keys,
            keys,
            key_padding_mask=~valid.reshape(batch_size * node_count, -1),
            need_weights=False,
        )
        hidden = hidden + output.reshape(batch_size, node_count, hidden_dim)
        return hidden + self.mlp(self.norm2(hidden))


class LocalThreeFrameTracker(nn.Module):
    """Local attention across three point sets, with central-pair edge logits."""

    def __init__(
        self,
        *,
        feat_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_blocks: int,
        dropout: float,
        pair_chunk_size: int,
        attention_radius_um: float,
        max_neighbors: int,
    ) -> None:
        super().__init__()
        if attention_radius_um <= 0 or max_neighbors < 1:
            raise ValueError("local attention radius and neighbor count must be positive")
        self.pair_chunk_size = pair_chunk_size
        self.attention_radius_um = attention_radius_um
        self.max_neighbors = max_neighbors
        self.max_observed_neighbors = 0
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            LocalAttentionBlock(hidden_dim, n_heads, dropout) for _ in range(n_blocks)
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.time_embedding = nn.Embedding(3, hidden_dim)
        nn.init.zeros_(self.time_embedding.weight)

    def load_public_tracker_state(self, state: dict[str, Any]) -> None:
        result = self.load_state_dict(state, strict=False)
        if result.missing_keys != ["time_embedding.weight"] or result.unexpected_keys:
            raise ValueError(
                {
                    "missing_public_parameters": result.missing_keys,
                    "unexpected_public_parameters": result.unexpected_keys,
                }
            )

    def _neighbors(
        self, coords: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            distance = torch.cdist(coords.float(), coords.float())
            distance = distance.masked_fill(~mask[:, None, :], float("inf"))
            neighborhood_size = (distance <= self.attention_radius_um).sum(dim=-1)
            observed_neighbors = int(neighborhood_size[mask].max().item()) if mask.any() else 1
            self.max_observed_neighbors = max(self.max_observed_neighbors, observed_neighbors)
            if observed_neighbors > self.max_neighbors:
                raise RuntimeError(
                    f"neighborhood {observed_neighbors} exceeds limit {self.max_neighbors}"
                )
            k = min(observed_neighbors, coords.shape[1])
            values, indices = torch.topk(distance, k=k, dim=-1, largest=False)
            valid = values <= self.attention_radius_um
            invalid_queries = ~mask
            if invalid_queries.any():
                indices[:, :, 0] = torch.where(
                    invalid_queries, torch.zeros_like(indices[:, :, 0]), indices[:, :, 0]
                )
                valid[:, :, 0] |= invalid_queries
            if not valid.any(dim=-1).all():
                raise RuntimeError("a detection has no local attention neighbor")
            return indices, valid

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        include_previous: bool,
        reverse: bool = False,
    ) -> torch.Tensor:
        prev_mask = batch["prev_mask"] if include_previous else torch.zeros_like(batch["prev_mask"])
        n_prev = batch["features_prev"].shape[1]
        n_src = batch["features_src"].shape[1]
        features = torch.cat(
            [batch["features_prev"], batch["features_src"], batch["features_tgt"]], dim=1
        )
        physical = torch.cat(
            [
                batch["coords_prev_physical"],
                batch["coords_src_physical"],
                batch["coords_tgt_physical"],
            ],
            dim=1,
        )
        mask = torch.cat([prev_mask, batch["source_mask"], batch["target_mask"]], dim=1)
        time_ids = torch.cat(
            [
                torch.zeros(n_prev, device=features.device, dtype=torch.long),
                torch.ones(n_src, device=features.device, dtype=torch.long),
                torch.full(
                    (batch["features_tgt"].shape[1],), 2, device=features.device, dtype=torch.long
                ),
            ]
        )
        hidden = self.norm_in(self.proj(features) + self.time_embedding(time_ids))
        hidden = hidden * mask.unsqueeze(-1)
        indices, valid = self._neighbors(physical, mask)
        for block in self.blocks:
            if self.training and torch.is_grad_enabled():
                hidden = grad_checkpoint(block, hidden, indices, valid, use_reentrant=False)
            else:
                hidden = block(hidden, indices, valid)
            hidden = hidden * mask.unsqueeze(-1)
        hidden = self.norm_out(hidden)
        src = hidden[:, n_prev : n_prev + n_src]
        tgt = hidden[:, n_prev + n_src :]
        coords_src, coords_tgt = batch["coords_src"], batch["coords_tgt"]
        if reverse:
            src, tgt = tgt, src
            coords_src, coords_tgt = coords_tgt, coords_src
        chunks = []
        for start in range(0, src.shape[1], self.pair_chunk_size):
            src_part = src[:, start : start + self.pair_chunk_size]
            coords_part = coords_src[:, start : start + self.pair_chunk_size]

            def score_pairs(
                q: torch.Tensor, k: torch.Tensor, cq: torch.Tensor, ck: torch.Tensor
            ) -> torch.Tensor:
                query = q.unsqueeze(2).expand(-1, -1, k.shape[1], -1)
                key = k.unsqueeze(1).expand(-1, q.shape[1], -1, -1)
                relative = (cq.unsqueeze(2) - ck.unsqueeze(1)) / 100.0
                return self.pair_mlp(torch.cat([query, key, relative], dim=-1)).squeeze(-1)

            if self.training and torch.is_grad_enabled():
                chunk = grad_checkpoint(
                    score_pairs, src_part, tgt, coords_part, coords_tgt, use_reentrant=False
                )
            else:
                chunk = score_pairs(src_part, tgt, coords_part, coords_tgt)
            chunks.append(chunk)
        return torch.cat(chunks, dim=1)


# %% [markdown]
# ## 2b. Metrics persistence


# %%
def update_metrics(path: Path, updates: dict[str, Any]) -> None:
    current = json.loads(path.read_text(encoding="utf-8"))

    def merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value
        return base

    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(merge(current, updates), indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


# %% [markdown]
# ## 3. Resolve and verify the fixed inputs


# %%
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


# Mount Drive first, then copy large immutable inputs to Colab's local disk.
LOCAL_INPUT_ROOT = Path("/content") / EXPERIMENT / "inputs"
LOCAL_INPUT_ROOT.mkdir(parents=True, exist_ok=True)


def staged_input(key: str, destination: str, marker: str) -> Path:
    source = DRIVE_PROJECT_ROOT / str(colab_cfg[key])
    if not (source / marker).exists():
        raise FileNotFoundError(f"{key} is missing {marker}: {source}")
    target = LOCAL_INPUT_ROOT / destination
    if not (target / marker).exists():
        source_bytes = sum(item.stat().st_size for item in source.rglob("*") if item.is_file())
        free_bytes = shutil.disk_usage("/content").free
        if free_bytes < source_bytes + 2_000_000_000:
            raise RuntimeError(
                f"Insufficient /content disk for {key}: need {source_bytes + 2_000_000_000}, "
                f"free {free_bytes}"
            )
        shutil.copytree(source, target, dirs_exist_ok=True)
    if not (target / marker).exists():
        raise FileNotFoundError(f"{key} copy is incomplete: {target}")
    log_event("input_staged", name=key, source=str(source), target=str(target))
    return target


def resolve_cache_output_root() -> Path:
    if CLI_MODE:
        target = LOCAL_INPUT_ROOT / "exp015"
        marker = target / str(cache_cfg["summary_file"])
        if not marker.is_file():
            raise FileNotFoundError(f"CLI cache acquisition did not produce {marker}")
        return target
    target = LOCAL_INPUT_ROOT / "exp015"
    marker = target / str(cache_cfg["summary_file"])
    if not marker.is_file():
        from google.colab import userdata

        try:
            kaggle_token = userdata.get("KAGGLE_API_TOKEN")
        except Exception as exc:
            raise RuntimeError(
                "Set KAGGLE_API_TOKEN in Colab Secrets and grant notebook access"
            ) from exc
        if not kaggle_token:
            raise RuntimeError("KAGGLE_API_TOKEN Colab Secret is empty")
        if shutil.disk_usage("/content").free < 7_000_000_000:
            raise RuntimeError("At least 7 GB free under /content is needed for the cache")
        target.mkdir(parents=True, exist_ok=True)
        command = [
            "kaggle",
            "kernels",
            "output",
            str(cache_cfg["kernel_source"]),
            "--path",
            str(target),
            "--file-pattern",
            r"^(window_cache/.*|window_cache_summary\.json)$",
            "--page-size",
            "200",
            "--quiet",
        ]
        log_event("cache_download_start", kernel=cache_cfg["kernel_source"])
        subprocess.run(
            command,
            check=True,
            env={**os.environ, "KAGGLE_API_TOKEN": kaggle_token},
        )
        del kaggle_token
        log_event("cache_download_complete", kernel=cache_cfg["kernel_source"])
    if not marker.is_file():
        raise FileNotFoundError(f"Kaggle cache download did not produce {marker}")
    return target


def resolve_public_artifact_root() -> Path:
    source = DRIVE_PROJECT_ROOT / str(colab_cfg["public_source"])
    required = (
        str(public_cfg["checkpoint"]),
        str(public_cfg["train_script"]),
        str(public_cfg["model_source"]),
    )
    missing = [relative for relative in required if not (source / relative).is_file()]
    if missing:
        raise FileNotFoundError({"public_source": str(source), "missing": missing})
    if CLI_MODE:
        return source
    target = LOCAL_INPUT_ROOT / "public_support"
    package_source = source / "repo/src"
    package_target = target / "repo/src"
    shutil.copytree(package_source, package_target, dirs_exist_ok=True)
    for relative in required:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            shutil.copy2(source / relative, destination)
    log_event("input_staged", name="public_source", source=str(source), target=str(target))
    return target


def resolve_train_dir() -> Path:
    source = DRIVE_PROJECT_ROOT / str(colab_cfg["geff_source"])
    if not source.is_dir() or not any(source.glob("*.geff")):
        raise FileNotFoundError(f"GEFF train directory missing: {source}")
    if CLI_MODE:
        return source
    target = LOCAL_INPUT_ROOT / "train_geff"
    if not target.is_dir():
        shutil.copytree(source, target)
    log_event("input_staged", name="geff_source", source=str(source), target=str(target))
    return target


cache_output_root = resolve_cache_output_root()
cache_summary_path = cache_output_root / str(cache_cfg["summary_file"])
cache_summary = validate_cache_summary(cache_summary_path, cache_cfg)
cache_root = cache_output_root / str(cache_cfg["directory"])
cache_paths = discover_cache_paths(cache_root, cache_cfg)
observed_cache_identity_sha256 = recompute_cache_identity_sha256(cache_paths)
if observed_cache_identity_sha256 != str(cache_cfg["identity_sha256"]):
    raise RuntimeError(
        {
            "cache_identity_mismatch": {
                "expected": cache_cfg["identity_sha256"],
                "actual": observed_cache_identity_sha256,
            }
        }
    )
public_root = resolve_public_artifact_root()
train_dir = resolve_train_dir()

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
    raise RuntimeError(
        {
            "public_artifact_checksum_mismatch": {
                "expected": expected_source_hashes,
                "actual": observed_source_hashes,
            }
        }
    )

sample_names = sorted({path.parent.name for path in cache_paths})
expected_embryo_counts = {
    embryo: sum(name.startswith(f"{embryo}_") for name in sample_names)
    for embryo in config["data"]["expected_embryo_counts"]
}
if expected_embryo_counts != config["data"]["expected_embryo_counts"]:
    raise RuntimeError({"embryo_count_mismatch": expected_embryo_counts})
missing_geff = [name for name in sample_names if not (train_dir / f"{name}.geff").is_dir()]
if missing_geff:
    raise FileNotFoundError({"missing_train_geff": missing_geff[:20]})

splits = build_embryo_splits(
    sample_names,
    validation_cfg["outer_folds"],
    int(validation_cfg["internal_split_seed"]),
)
print("Cache output:", cache_output_root)
print("Cache summary SHA:", cache_summary["summary_sha256"])
print("Public artifact:", public_root)
print("Train GEFF:", train_dir)

# %% [markdown]
# ## 4. Load sparse annotations and the public tracker initialization

# %%
scale_zyx_um = tuple(float(value) for value in config["data"]["annotation"]["voxel_scale_zyx_um"])
annotations = {
    sample: load_annotation_graph(train_dir / f"{sample}.geff", scale_zyx_um)
    for sample in sample_names
}
cache_paths, gt_window_filter_audit = filter_nonempty_gt_window_paths(cache_paths, annotations)
gt_window_filter_path = WORKING_ROOT / "gt_window_filter_audit.json"
gt_window_filter_path.write_text(
    json.dumps(gt_window_filter_audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
gt_window_filter_audit_sha256 = file_sha256(gt_window_filter_path)
gt_window_filter_summary = {
    key: value for key, value in gt_window_filter_audit.items() if key != "skipped_windows"
}
if not cache_paths:
    raise RuntimeError("no cache windows remain after applying the public empty-GT policy")
selected_outer_paths_by_fold: dict[int, list[Path]] = {}
pilot_selection_sha256: str | None = None
if pilot_mode:
    pilot_count = int(train_cfg["outer_evaluation_windows_per_embryo"])
    pilot_seed = int(train_cfg["outer_evaluation_sampling_seed"])
    pilot_selection_manifest = {
        "stage": "pilot",
        "policy": "sha256_ranked_samples_one_sha256_ranked_eligible_window_each",
        "seed": pilot_seed,
        "windows_per_embryo": pilot_count,
        "folds": [],
    }
    for record in splits:
        fold = int(record["fold"])
        selected = select_pilot_outer_paths(
            cache_paths, record["outer_evaluation"], pilot_count, pilot_seed
        )
        selected_outer_paths_by_fold[fold] = selected
        pilot_selection_manifest["folds"].append(
            {
                "fold": fold,
                "evaluation_embryo": record["evaluation_embryo"],
                "eligible_outer_window_count": len(
                    paths_for_samples(cache_paths, record["outer_evaluation"])
                ),
                "selected_windows": [path.relative_to(cache_root).as_posix() for path in selected],
            }
        )
    pilot_selection_path = WORKING_ROOT / "pilot_outer_window_selection.json"
    pilot_selection_path.write_text(
        json.dumps(pilot_selection_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pilot_selection_sha256 = file_sha256(pilot_selection_path)
    print("Pilot outer evaluation:", pilot_count, "windows per embryo", pilot_selection_sha256)
else:
    selected_outer_paths_by_fold = {
        int(record["fold"]): paths_for_samples(cache_paths, record["outer_evaluation"])
        for record in splits
    }

split_manifest = [
    {
        **record,
        "window_counts": {
            key: len(paths_for_samples(cache_paths, record[key]))
            for key in ("gradient_update", "internal_validation", "outer_evaluation")
        },
    }
    for record in splits
]
(WORKING_ROOT / "split_manifest.json").write_text(
    json.dumps(split_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
if (
    file_sha256(WORKING_ROOT / "split_manifest.json")
    != config["diagnostic"]["parent_split_manifest_sha256"]
):
    raise RuntimeError("split manifest differs from exp027")
if pilot_selection_sha256 != config["diagnostic"]["parent_selection_sha256"]:
    raise RuntimeError("selected outer windows differ from exp027")
log_event(
    "fixed_windows_verified",
    split_sha256=config["diagnostic"]["parent_split_manifest_sha256"],
    selection_sha256=pilot_selection_sha256,
)
print("GT window filter:", gt_window_filter_summary)
print("Split window counts:", [record["window_counts"] for record in split_manifest])
annotation_manifest = {
    sample: annotation.content_sha256 for sample, annotation in sorted(annotations.items())
}


public_src = public_root / "repo" / "src"
sys.path.insert(0, str(public_src))
full_public_state = torch.load(source_paths["checkpoint"], map_location="cpu", weights_only=True)
public_tracker_state = extract_public_tracker_state(full_public_state)
public_tracker_state_sha256 = canonical_state_sha256(public_tracker_state)


def new_initialized_tracker(
    device: torch.device, *, include_previous: bool
) -> LocalThreeFrameTracker:
    tracker = LocalThreeFrameTracker(
        feat_dim=int(params_cfg["feature_dim"]),
        hidden_dim=int(params_cfg["hidden_dim"]),
        n_heads=int(params_cfg["n_heads"]),
        n_blocks=int(params_cfg["n_blocks"]),
        dropout=float(params_cfg["dropout"]),
        pair_chunk_size=int(params_cfg["pair_chunk_size"]),
        attention_radius_um=float(params_cfg["attention_radius_um"]),
        max_neighbors=int(params_cfg["max_neighbors"]),
    )
    tracker.load_public_tracker_state(copy.deepcopy(public_tracker_state))
    tracker.include_previous = include_previous
    return tracker.to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("Select a GPU Colab runtime before training")
OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)
print("Device:", device, torch.cuda.get_device_name(device))
print("Initial public tracker state SHA:", public_tracker_state_sha256)

# %% [markdown]
# ## 5. Three-frame windows, model initialization, and runtime benchmark

# %%
feature_channels = int(cache_cfg["feature_channels"])
max_matching_distance_um = float(teacher_cfg["max_matching_distance_um"])
downsample_zyx = tuple(float(value) for value in params_cfg["downsample_zyx"])
batch_size = int(train_cfg["batch_size"])
num_workers = int(train_cfg["num_workers"])
global_seed = int(train_cfg["seed"])
use_amp = bool(train_cfg["mixed_precision"])
variants = list(train_cfg["active_variants"])


def make_loader(paths: list[Path], *, shuffle: bool, seed: int) -> DataLoader:
    dataset = ThreeFrameWindowDataset(
        paths,
        annotations,
        feature_channels=feature_channels,
        expected_primary_checkpoint_sha256=str(public_cfg["checkpoint_sha256"]),
        max_matching_distance_um=max_matching_distance_um,
        downsample_zyx=downsample_zyx,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_three_frame_examples,
        generator=generator,
        worker_init_fn=dataloader_worker_init,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
    )


def new_optimizer(tracker: LocalThreeFrameTracker) -> Any:
    optimizer = torch.optim.AdamW(
        tracker.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    model_ids = {id(parameter) for parameter in tracker.parameters()}
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    if optimizer_ids != model_ids:
        raise RuntimeError("optimizer parameter boundary differs from tracker")
    return optimizer


def new_scaler() -> Any:
    return torch.amp.GradScaler("cuda", enabled=use_amp)


benchmark_records: list[dict[str, Any]] = []
projected_seconds_total = 0.0
for record in splits:
    fold = int(record["fold"])
    train_paths = paths_for_samples(cache_paths, record["gradient_update"])
    validation_paths = paths_for_samples(cache_paths, record["internal_validation"])
    evaluation_paths = selected_outer_paths_by_fold[fold]
    benchmark_count = min(int(train_cfg["benchmark_windows_per_fold"]), len(train_paths))
    benchmark_paths = sorted(train_paths, key=lambda path: path.stat().st_size, reverse=True)[
        :benchmark_count
    ]
    for variant in variants:
        fold_seed = global_seed + fold
        seed_everything(fold_seed)
        tracker = new_initialized_tracker(device, include_previous=variant == "three_frame_local")
        optimizer = new_optimizer(tracker)
        scaler = new_scaler()
        loader = make_loader(benchmark_paths, shuffle=False, seed=fold_seed)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        benchmark_metrics = train_one_epoch(
            tracker,
            loader,
            optimizer,
            scaler,
            device,
            gamma=float(loss_cfg["focal_gamma"]),
            gradient_clip_norm=float(train_cfg["gradient_clip_norm"]),
            use_amp=use_amp,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        work_windows = int(train_cfg["epochs"]) * (
            len(train_paths) + len(validation_paths) + len(evaluation_paths)
        ) + len(evaluation_paths)
        projected = (
            elapsed
            / benchmark_count
            * work_windows
            * float(train_cfg["runtime_projection_multiplier"])
        )
        projected_seconds_total += projected
        benchmark_records.append(
            {
                "fold": fold,
                "variant": variant,
                "window_count": benchmark_count,
                "max_observed_neighbors": tracker.max_observed_neighbors,
                "elapsed_seconds": elapsed,
                "work_windows_projected": work_windows,
                "conservative_projected_seconds": projected,
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "train_metrics": benchmark_metrics,
            }
        )
        del loader, scaler, optimizer, tracker
        torch.cuda.empty_cache()

benchmark_summary = {
    "records": benchmark_records,
    "conservative_projected_seconds_total": projected_seconds_total,
    "runtime_gate_seconds": float(train_cfg["runtime_gate_hours"]) * 3600.0,
    "stage": run_stage,
    "pilot_selection_sha256": pilot_selection_sha256,
}
(WORKING_ROOT / "benchmark_summary.json").write_text(
    json.dumps(benchmark_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
benchmark_summary_sha256 = file_sha256(WORKING_ROOT / "benchmark_summary.json")
print(json.dumps(benchmark_summary, indent=2))
if projected_seconds_total > benchmark_summary["runtime_gate_seconds"]:
    log_event(
        "runtime_projection_warning",
        seconds=projected_seconds_total,
        gate_seconds=benchmark_summary["runtime_gate_seconds"],
    )

# %% [markdown]
# ## 6. Fit one three-frame tracker for each embryo-held-out fold

# %%
fold_summaries: list[dict[str, Any]] = []
model_manifest_records: list[dict[str, Any]] = []
training_started = time.perf_counter()

for record in splits:
    fold = int(record["fold"])
    train_paths = paths_for_samples(cache_paths, record["gradient_update"])
    validation_paths = paths_for_samples(cache_paths, record["internal_validation"])
    evaluation_paths = selected_outer_paths_by_fold[fold]
    for variant in variants:
        fold_seed = global_seed + fold
        seed_everything(fold_seed)
        tracker = new_initialized_tracker(device, include_previous=variant == "three_frame_local")
        optimizer = new_optimizer(tracker)
        scaler = new_scaler()
        train_loader = make_loader(train_paths, shuffle=True, seed=fold_seed)
        validation_loader = make_loader(validation_paths, shuffle=False, seed=fold_seed)
        evaluation_loader = make_loader(evaluation_paths, shuffle=False, seed=fold_seed)
        model_dir = OUTPUT_MODELS / variant / f"fold_{fold}"
        model_dir.mkdir(parents=True, exist_ok=True)
        resume_path = model_dir / "resume_state.pth"
        best_score = float("-inf")
        best_epoch = -1
        best_state: dict[str, Any] | None = None
        epochs: list[dict[str, Any]] = []
        start_epoch = 0
        if resume_path.is_file():
            saved = torch.load(resume_path, map_location="cpu", weights_only=False)
            if (saved["experiment"], saved["fold"], saved["variant"]) != (
                EXPERIMENT,
                fold,
                variant,
            ):
                raise RuntimeError("resume checkpoint belongs to another run")
            if saved["cache_identity_sha256"] != observed_cache_identity_sha256:
                raise RuntimeError("resume checkpoint cache identity differs")
            if saved["public_tracker_initial_state_sha256"] != public_tracker_state_sha256:
                raise RuntimeError("resume checkpoint initialization differs")
            tracker.load_state_dict(saved["state_dict"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            train_loader.generator.set_state(saved["loader_generator_state"])
            random.setstate(saved["python_rng"])
            np.random.set_state(saved["numpy_rng"])
            torch.set_rng_state(saved["torch_rng"])
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
            epochs = saved["epochs"]
            best_score = saved["best_score"]
            best_epoch = saved["best_epoch"]
            best_state = saved["best_state"]
            start_epoch = saved["next_epoch"]
            if start_epoch != len(epochs) or start_epoch > int(train_cfg["epochs"]):
                raise RuntimeError("resume epoch history is inconsistent")
            log_event("resume", fold=fold, next_epoch=start_epoch)
        for epoch in range(start_epoch, int(train_cfg["epochs"])):
            epoch_started = time.perf_counter()
            train_metrics = train_one_epoch(
                tracker,
                train_loader,
                optimizer,
                scaler,
                device,
                gamma=float(loss_cfg["focal_gamma"]),
                gradient_clip_norm=float(train_cfg["gradient_clip_norm"]),
                use_amp=use_amp,
            )
            validation_metrics = evaluate_tracker(
                tracker,
                validation_loader,
                device,
                gamma=float(loss_cfg["focal_gamma"]),
                use_amp=use_amp,
            )
            outer_metrics = evaluate_tracker(
                tracker,
                evaluation_loader,
                device,
                gamma=float(loss_cfg["focal_gamma"]),
                use_amp=use_amp,
            )
            selection_score = float(validation_metrics["selection_score"])
            is_best = selection_score >= best_score
            if is_best:
                best_score = selection_score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in tracker.state_dict().items()
                }
            epoch_record = {
                "epoch": epoch,
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "train": train_metrics,
                "internal_validation": validation_metrics,
                "outer_evaluation": outer_metrics,
                "selected": is_best,
            }
            epochs.append(epoch_record)
            epoch_path = model_dir / f"primary_tracker_epoch_{epoch + 1:02d}.pth"
            epoch_payload = {
                "experiment": EXPERIMENT,
                "fold": fold,
                "variant": variant,
                "epoch": epoch,
                "state_dict": {
                    key: value.detach().cpu().clone() for key, value in tracker.state_dict().items()
                },
            }
            temporary_epoch = epoch_path.with_name("." + epoch_path.name + ".tmp")
            torch.save(epoch_payload, temporary_epoch)
            temporary_epoch.replace(epoch_path)
            epoch_record["model_file_sha256"] = file_sha256(epoch_path)
            epoch_record["model_state_sha256"] = canonical_state_sha256(epoch_payload["state_dict"])
            checkpoint = {
                "experiment": EXPERIMENT,
                "fold": fold,
                "variant": variant,
                "cache_identity_sha256": observed_cache_identity_sha256,
                "public_tracker_initial_state_sha256": public_tracker_state_sha256,
                "next_epoch": epoch + 1,
                "state_dict": epoch_payload["state_dict"],
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "loader_generator_state": train_loader.generator.get_state(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "epochs": epochs,
            }
            temporary_resume = resume_path.with_name("." + resume_path.name + ".tmp")
            torch.save(checkpoint, temporary_resume)
            temporary_resume.replace(resume_path)
            if CLI_MODE:
                immutable_resume = model_dir / f"resume_state_epoch_{epoch + 1:02d}.pth"
                shutil.copy2(resume_path, immutable_resume)
                receipt = {
                    "experiment": EXPERIMENT,
                    "run_id": colab_cfg["run_id"],
                    "config_sha256": file_sha256(CONFIG_PATH),
                    "fold": fold,
                    "epoch": epoch + 1,
                    "files": [
                        {
                            "path": epoch_path.relative_to(WORKING_ROOT).as_posix(),
                            "bytes": epoch_path.stat().st_size,
                            "sha256": file_sha256(epoch_path),
                        },
                        {
                            "path": immutable_resume.relative_to(WORKING_ROOT).as_posix(),
                            "bytes": immutable_resume.stat().st_size,
                            "sha256": file_sha256(immutable_resume),
                        },
                    ],
                }
                receipt_path = model_dir / f"epoch_{epoch + 1:02d}_receipt.json"
                receipt_path.write_text(
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            log_event(
                "epoch_complete",
                fold=fold,
                epoch=epoch + 1,
                selection_score=selection_score,
                state_sha256=epoch_record["model_state_sha256"],
            )
        if best_state is None:
            raise RuntimeError(f"fold {fold} {variant} did not select a checkpoint")
        tracker.load_state_dict(best_state, strict=True)
        trained_outer = evaluate_tracker(
            tracker,
            evaluation_loader,
            device,
            gamma=float(loss_cfg["focal_gamma"]),
            use_amp=use_amp,
        )
        model_path = model_dir / "primary_tracker_best.pth"
        torch.save(
            {
                "experiment": EXPERIMENT,
                "fold": fold,
                "variant": variant,
                "train_embryo": record["train_embryo"],
                "evaluation_embryo": record["evaluation_embryo"],
                "best_epoch": best_epoch,
                "selection_score": best_score,
                "state_dict": best_state,
            },
            model_path,
        )
        state_sha = canonical_state_sha256(best_state)
        summary = {
            "fold": fold,
            "variant": variant,
            "train_embryo": record["train_embryo"],
            "evaluation_embryo": record["evaluation_embryo"],
            "sample_counts": {
                key: len(record[key])
                for key in ("gradient_update", "internal_validation", "outer_evaluation")
            },
            "window_counts": {
                "gradient_update": len(train_paths),
                "internal_validation": len(validation_paths),
                "outer_evaluation": len(evaluation_paths),
            },
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_internal_selection_score": best_score,
            "max_observed_neighbors": tracker.max_observed_neighbors,
            "trained_outer_evaluation": trained_outer,
            "model_file": model_path.relative_to(WORKING_ROOT).as_posix(),
            "model_file_sha256": file_sha256(model_path),
            "model_state_sha256": state_sha,
        }
        (WORKING_ROOT / f"{variant}_fold_{fold}_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        fold_summaries.append(summary)
        model_manifest_records.append(
            {
                "fold": fold,
                "variant": variant,
                "path": summary["model_file"],
                "file_sha256": summary["model_file_sha256"],
                "canonical_state_sha256": state_sha,
                "best_epoch": best_epoch,
                "train_embryo": record["train_embryo"],
                "evaluation_embryo": record["evaluation_embryo"],
            }
        )
        del evaluation_loader, validation_loader, train_loader, scaler, optimizer, tracker
        torch.cuda.empty_cache()

# %% [markdown]
# ## 7. Model manifest and output evidence

# %%
diagnostic_gate = {
    record["evaluation_embryo"]: {
        "status": "pending_saved_control_comparison",
        "selected_epoch": record["best_epoch"] + 1,
        "selected_outer_evaluation": record["trained_outer_evaluation"],
    }
    for record in fold_summaries
}
all_embryos_pass = False
(WORKING_ROOT / "diagnostic_gate.json").write_text(
    json.dumps(diagnostic_gate, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)

model_manifest = {
    "experiment": EXPERIMENT,
    "stage": run_stage,
    "epochs": int(train_cfg["epochs"]),
    "outer_evaluation_windows_per_embryo": (pilot_count if pilot_mode else None),
    "pilot_selection_sha256": pilot_selection_sha256,
    "created_at": datetime.now(UTC).isoformat(),
    "model_class": "LocalThreeFrameTracker",
    "attention_radius_um": float(params_cfg["attention_radius_um"]),
    "max_neighbors": int(params_cfg["max_neighbors"]),
    "public_checkpoint_file_sha256": observed_source_hashes["checkpoint"],
    "public_tracker_initial_state_sha256": public_tracker_state_sha256,
    "cache_summary_sha256": cache_summary["summary_sha256"],
    "cache_identity_sha256": cache_summary["cache_identity_sha256"],
    "gt_window_filter_audit_sha256": gt_window_filter_audit_sha256,
    "annotation_content_sha256": json_sha256(annotation_manifest),
    "models": model_manifest_records,
}
model_manifest_path = WORKING_ROOT / str(model_cfg["output"]["model_manifest"])
model_manifest_path.write_text(
    json.dumps(model_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
model_manifest_sha = file_sha256(model_manifest_path)
training_summary = {
    "experiment": EXPERIMENT,
    "status": "pilot_train_completed" if pilot_mode else "train_stage_completed",
    "stage": run_stage,
    "epochs": int(train_cfg["epochs"]),
    "outer_evaluation_windows_per_embryo": (pilot_count if pilot_mode else None),
    "pilot_selection_sha256": pilot_selection_sha256,
    "benchmark_summary_sha256": benchmark_summary_sha256,
    "official_metric_computed": False,
    "submission_created": False,
    "conditional_validation": validation_cfg["conditional_validation"],
    "epoch_elapsed_seconds_sum": sum(
        float(epoch["elapsed_seconds"])
        for fold_summary in fold_summaries
        for epoch in fold_summary["epochs"]
    ),
    "last_session_training_seconds": time.perf_counter() - training_started,
    "last_session_notebook_seconds": time.perf_counter() - NOTEBOOK_STARTED,
    "benchmark": benchmark_summary,
    "gt_window_filter": gt_window_filter_summary,
    "folds": fold_summaries,
    "diagnostic_gate": diagnostic_gate,
    "all_embryos_pass": all_embryos_pass,
    "model_manifest_sha256": model_manifest_sha,
}
summary_path = WORKING_ROOT / str(model_cfg["output"]["training_summary"])
summary_path.write_text(
    json.dumps(training_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
update_metrics(
    METRICS_PATH,
    {
        "status": "running",
        "metric": "fixed_candidate_public_teacher_edge_metrics",
        "evidence": {
            "colab": {
                "run_id": colab_cfg["run_id"],
                "resource": torch.cuda.get_device_name(device),
                "last_session_notebook_seconds": training_summary["last_session_notebook_seconds"],
                "epoch_elapsed_seconds_sum": training_summary["epoch_elapsed_seconds_sum"],
                "run_root": str(WORKING_ROOT),
            },
            "artifacts": {
                "input_file_sha": model_manifest["annotation_content_sha256"],
                "cache_file_sha": cache_summary["summary_sha256"],
                "feature_content_sha": cache_summary["cache_identity_sha256"],
                "pilot_selection_sha256": pilot_selection_sha256,
                "benchmark_summary_sha256": benchmark_summary_sha256,
                "model_manifest_sha": model_manifest_sha,
                "model_count": len(model_manifest_records),
                "model_shas": {
                    f"{record['variant']}_fold_{record['fold']}": record["file_sha256"]
                    for record in model_manifest_records
                },
            },
        },
        "train_stage": training_summary,
        "notes": (
            "Ten Colab epochs and 64 fixed outer windows per embryo; "
            "saved controls and official graph score pending."
        ),
    },
)
log_event("training_complete", model_manifest_sha256=model_manifest_sha)
(WORKING_ROOT / "train_complete.json").write_text(
    json.dumps(
        {
            "experiment": EXPERIMENT,
            "run_id": colab_cfg["run_id"],
            "model_manifest_sha256": model_manifest_sha,
            "completed_at": datetime.now(UTC).isoformat(),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(json.dumps({"gate": diagnostic_gate, "all_embryos_pass": all_embryos_pass}, indent=2))
print("Model manifest:", model_manifest_path, model_manifest_sha)
print("No submission was created.")
