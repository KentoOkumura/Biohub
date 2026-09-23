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
# # exp039: DoG responses as primary tracker node features
#
# The public image encoder, detector candidates, normalization, secondary model,
# and graph decode are fixed. This notebook reads the exact exp015 train-window cache,
# reads separately generated original-resolution DoG responses, normalizes them
# within each fold's gradient-update samples, and compares their use against
# a zero-input 66-dimensional control under the public 5-micrometer teacher.
#
# This train stage reports fixed-candidate edge metrics. It does not claim the
# official graph metric, run full-graph inference, or create a Kaggle submission.

# %% [markdown]
# ## 1. Configuration and runtime guard

# %%
from __future__ import annotations

# ruff: noqa: E402
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT = "exp039_dog_features"
COMPETITION = "biohub-cell-tracking-during-development"
WORKING_ROOT = Path.cwd()
CONFIG_PATH = WORKING_ROOT / "config.yaml"
METRICS_PATH = WORKING_ROOT / "metrics.json"
OUTPUT_MODELS = WORKING_ROOT / "models"
NOTEBOOK_STARTED = time.perf_counter()

if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
    raise RuntimeError("The authoritative full run must execute on Kaggle.")

config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
validation_cfg = config["validation"]
cache_cfg = config["data"]["cache"]
model_cfg = config["model"]
train_cfg = model_cfg["training"]
teacher_cfg = model_cfg["teacher"]
loss_cfg = model_cfg["loss"]
params_cfg = model_cfg["params"]
dog_cfg = config["data"]["dog_features"]
ablation_cfg = model_cfg["feature_ablation"]
public_cfg = model_cfg["public_source"]
active_variants = list(train_cfg["active_variants"])
runtime_cfg = config["runtime"]

if model_cfg["trainable_components"] != ["primary_SimpleNodeTransformer"]:
    raise RuntimeError("only the primary tracker is trainable")
if active_variants != ["zero_dog_control", "dog_node_features"]:
    raise RuntimeError("the train stage requires the approved same-structure variants")
if ablation_cfg["active_variants"] != active_variants:
    raise RuntimeError("training and feature-ablation variants differ")
if model_cfg["control"]["retrain"] is not True:
    raise RuntimeError("the 66-input zero-DOG control must be retrained")
if model_cfg["inference"]["implemented"] is not False:
    raise RuntimeError("full-graph inference must pass the pair-level gate first")

print("Experiment:", EXPERIMENT)
print("Outer folds:", json.dumps(validation_cfg["outer_folds"], indent=2))
print("Trainable components:", model_cfg["trainable_components"])
print("Frozen components:", model_cfg["frozen_components"])

# %% [markdown]
# ## 2. Offline dependencies and experiment helpers

# %%
OFFLINE_GRAPH_MODULES = {
    "tracksdata": "tracksdata",
    "zarr": "zarr",
    "geff": "geff",
    "geff_spec": "geff_spec",
    "ilpy": "ilpy",
    "polars": "polars",
    "pyscipopt": "pyscipopt",
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


def polars_runtime_ready() -> bool:
    try:
        import polars as pl
        from polars._plr import PySeries

        _ = PySeries
        return hasattr(pl, "Float16")
    except Exception:
        return False


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
        return
    wheel_dirs = offline_wheel_dirs()
    if not wheel_dirs:
        raise FileNotFoundError("the public support dataset has no offline wheel directory")
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
    remaining = {}
    for package_name, module_name in OFFLINE_GRAPH_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as error:
            remaining[package_name] = f"{type(error).__name__}: {error}"
    if remaining or not polars_runtime_ready():
        raise ImportError({"remaining_graph_import_failures": remaining})


ensure_geff_runtime_dependencies()

import torch
from dog_features import canonical_sha256, expand_public_tracker_projection, fit_dog_normalizer
from frozen_tracker import (
    FrozenFeatureWindowDataset,
    aggregate_teacher_stats,
    build_embryo_splits,
    canonical_state_sha256,
    collate_window_examples,
    dataloader_worker_init,
    discover_cache_paths,
    evaluate_dog_progression_gate,
    evaluate_tracker,
    extract_public_tracker_state,
    file_sha256,
    filter_nonempty_gt_window_paths,
    json_sha256,
    load_annotation_graph,
    move_batch_to_device,
    paths_for_samples,
    recompute_cache_identity_sha256,
    seed_everything,
    tracker_logits,
    train_one_epoch,
    validate_cache_summary,
)
from settings import update_metrics
from torch.utils.data import DataLoader

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


def resolve_cache_output_root() -> Path:
    kernel_slug = str(cache_cfg["kernel_source"]).split("/", 1)[-1]
    candidates = [
        path.parent
        for path in Path("/kaggle/input/notebooks").rglob(str(cache_cfg["summary_file"]))
        if kernel_slug in path.as_posix()
    ]
    return unique_existing(candidates, "exp015 cache output root")


def resolve_public_artifact_root() -> Path:
    slug = str(public_cfg["dataset_ref"]).split("/", 1)[-1]
    candidates = [
        Path(f"/kaggle/input/datasets/pilkwang/{slug}"),
        Path(f"/kaggle/input/{slug}"),
        Path(f"/kaggle/input/{slug}/{slug}"),
    ]
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        candidates.extend(
            path.parents[4]
            for path in input_root.rglob(str(public_cfg["model_source"]))
            if len(path.parents) >= 5
        )
    valid = [
        root
        for root in candidates
        if (root / str(public_cfg["train_script"])).is_file()
        and (root / str(public_cfg["model_source"])).is_file()
        and (root / str(public_cfg["checkpoint"])).is_file()
    ]
    return unique_existing(valid, "public tracker artifact root")


def resolve_train_dir() -> Path:
    return unique_existing(
        [
            Path(f"/kaggle/input/competitions/{COMPETITION}/train"),
            Path(f"/kaggle/input/{COMPETITION}/train"),
        ],
        "competition train directory",
    )


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
feature_slug = str(runtime_cfg["kaggle"]["train"]["feature_kernel_source"]).split("/", 1)[-1]
feature_manifest_path = unique_existing(
    [
        path
        for path in Path("/kaggle/input/notebooks").rglob("dog_feature_manifest.json")
        if feature_slug in path.as_posix()
    ],
    "exp039 DoG feature manifest",
)
feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
unsigned_feature_manifest = {
    key: value for key, value in feature_manifest.items() if key != "manifest_content_sha256"
}
if canonical_sha256(unsigned_feature_manifest) != feature_manifest["manifest_content_sha256"]:
    raise RuntimeError("DoG feature manifest self-check failed")
if (
    feature_manifest["experiment"] != EXPERIMENT
    or feature_manifest["dog_contract"] != dog_cfg
    or feature_manifest["source_cache_summary_sha256"] != cache_summary["summary_sha256"]
    or feature_manifest["source_cache_identity_sha256"] != cache_summary["cache_identity_sha256"]
    or feature_manifest["dataset_count"] != len({path.parent.name for path in cache_paths})
    or set(feature_manifest["raw_image_sources"]) != {path.parent.name for path in cache_paths}
):
    raise RuntimeError("DoG feature manifest differs from fixed inputs")
feature_root = feature_manifest_path.parent / str(dog_cfg["feature_output"])
if len(feature_manifest["frames"]) != int(feature_manifest["frame_count"]):
    raise RuntimeError("incomplete DoG feature manifest")
for sample in sorted({path.parent.name for path in cache_paths}):
    frame_paths = sorted((feature_root / sample).glob("*.npz"))
    if len(frame_paths) != int(config["data"]["expected_frames_per_sample"]):
        raise RuntimeError(f"incomplete DoG feature files: {sample}")
print("DoG feature source:", feature_manifest_path)
print("DoG feature manifest SHA:", feature_manifest["manifest_content_sha256"])

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
print("GT window filter:", gt_window_filter_summary)
print("Split window counts:", [record["window_counts"] for record in split_manifest])
annotation_manifest = {
    sample: annotation.content_sha256 for sample, annotation in sorted(annotations.items())
}

public_src = public_root / "repo" / "src"
sys.path.insert(0, str(public_src))
SimpleNodeTransformer = importlib.import_module("biohub_tracking.models").SimpleNodeTransformer

full_public_state = torch.load(source_paths["checkpoint"], map_location="cpu", weights_only=True)
public_tracker_state = extract_public_tracker_state(full_public_state)
public_tracker_state_sha256 = canonical_state_sha256(public_tracker_state)


def new_initialized_tracker(device: torch.device, variant: str) -> Any:
    if variant not in active_variants:
        raise ValueError(f"unknown DoG variant: {variant}")
    tracker = SimpleNodeTransformer(
        feat_dim=int(params_cfg["feature_dim"]),
        hidden_dim=int(params_cfg["hidden_dim"]),
        n_heads=int(params_cfg["n_heads"]),
        n_blocks=int(params_cfg["n_blocks"]),
        dropout=float(params_cfg["dropout"]),
        pair_chunk_size=int(params_cfg["pair_chunk_size"]),
    )
    expanded = expand_public_tracker_projection(
        public_tracker_state, extra_channels=int(dog_cfg["feature_dimension"])
    )
    tracker.load_state_dict(expanded, strict=True)
    return tracker.to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("the authoritative train run requires a Kaggle GPU")
OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)
print("Device:", device, torch.cuda.get_device_name(device))
print("Initial tracker canonical state SHA:", public_tracker_state_sha256)

# %% [markdown]
# ## 5. Dataset, optimizer boundary, and runtime benchmark

# %%
feature_channels = int(cache_cfg["feature_channels"])
max_matching_distance_um = float(teacher_cfg["max_matching_distance_um"])
downsample_zyx = tuple(float(value) for value in params_cfg["downsample_zyx"])
batch_size = int(train_cfg["batch_size"])
num_workers = int(train_cfg["num_workers"])
global_seed = int(train_cfg["seed"])
use_amp = bool(train_cfg["mixed_precision"])


def make_loader(
    paths: list[Path],
    *,
    shuffle: bool,
    seed: int,
    dog_normalizer: dict[str, Any],
    use_dog_values: bool,
) -> DataLoader:
    dataset = FrozenFeatureWindowDataset(
        paths,
        annotations,
        feature_channels=feature_channels,
        expected_primary_checkpoint_sha256=str(public_cfg["checkpoint_sha256"]),
        max_matching_distance_um=max_matching_distance_um,
        downsample_zyx=downsample_zyx,
        dog_feature_root=feature_root,
        dog_normalizer=dog_normalizer,
        use_dog_values=use_dog_values,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_window_examples,
        generator=generator,
        worker_init_fn=dataloader_worker_init,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
    )


def new_optimizer(tracker: Any) -> Any:
    optimizer = torch.optim.AdamW(
        tracker.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    model_parameter_ids = {id(parameter) for parameter in tracker.parameters()}
    optimizer_parameter_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    if optimizer_parameter_ids != model_parameter_ids:
        raise RuntimeError("optimizer parameters differ from primary tracker parameters")
    return optimizer


def new_scaler() -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=use_amp)


normalizers_by_fold: dict[int, dict[str, Any]] = {}
normalizer_records: list[dict[str, Any]] = []
for split_record in splits:
    fold = int(split_record["fold"])
    gradient_paths = paths_for_samples(cache_paths, split_record["gradient_update"])
    statistics = fit_dog_normalizer(
        gradient_paths,
        feature_root,
        standard_deviation_floor=float(dog_cfg["normalizer_standard_deviation_floor"]),
    )
    normalizers_by_fold[fold] = statistics
    normalizer_records.append(
        {
            "fold": fold,
            "train_embryo": split_record["train_embryo"],
            "gradient_update_samples": list(split_record["gradient_update"]),
            "gradient_update_window_count": len(gradient_paths),
            "gradient_update_paths_sha256": json_sha256(
                [path.relative_to(cache_root).as_posix() for path in sorted(gradient_paths)]
            ),
            "statistics": statistics,
        }
    )
normalizer_manifest = {
    "experiment": EXPERIMENT,
    "fit_scope": "gradient_update_candidates_only",
    "outer_evaluation_or_internal_validation_used_for_fit": False,
    "dog_feature_manifest_content_sha256": feature_manifest["manifest_content_sha256"],
    "folds": normalizer_records,
}
normalizer_path = WORKING_ROOT / str(model_cfg["output"]["normalizer_manifest"])
normalizer_path.write_text(
    json.dumps(normalizer_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
normalizer_manifest_sha256 = file_sha256(normalizer_path)


benchmark_records: list[dict[str, Any]] = []
raw_projected_seconds_total = 0.0
calibrated_projected_seconds_total = 0.0
projection_calibration = train_cfg["runtime_projection_calibration"]
calibration_factor = (
    float(projection_calibration["parent_observed_notebook_seconds"])
    / float(projection_calibration["parent_conservative_projected_seconds"])
    * float(projection_calibration["reserve_multiplier"])
)
for record in splits:
    fold = int(record["fold"])
    fold_seed = global_seed + fold
    train_paths = paths_for_samples(cache_paths, record["gradient_update"])
    validation_paths = paths_for_samples(cache_paths, record["internal_validation"])
    evaluation_paths = paths_for_samples(cache_paths, record["outer_evaluation"])
    benchmark_count = min(int(train_cfg["benchmark_windows_per_fold"]), len(train_paths))
    benchmark_paths = sorted(
        train_paths,
        key=lambda path: path.stat().st_size,
        reverse=True,
    )[:benchmark_count]
    seed_everything(fold_seed)
    benchmark_tracker = new_initialized_tracker(device, "dog_node_features")
    benchmark_optimizer = new_optimizer(benchmark_tracker)
    benchmark_scaler = new_scaler()
    benchmark_loader = make_loader(
        benchmark_paths,
        shuffle=False,
        seed=fold_seed,
        dog_normalizer=normalizers_by_fold[fold],
        use_dog_values=True,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    benchmark_metrics = train_one_epoch(
        benchmark_tracker,
        benchmark_loader,
        benchmark_optimizer,
        benchmark_scaler,
        device,
        gamma=float(loss_cfg["focal_gamma"]),
        gradient_clip_norm=float(train_cfg["gradient_clip_norm"]),
        use_amp=use_amp,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    work_windows_per_variant = int(train_cfg["epochs"]) * (
        len(train_paths) + len(validation_paths)
    ) + 2 * len(evaluation_paths)
    work_windows = len(active_variants) * work_windows_per_variant
    raw_projected = (
        elapsed / benchmark_count * work_windows * float(train_cfg["runtime_projection_multiplier"])
    )
    calibrated_projected = raw_projected * calibration_factor
    raw_projected_seconds_total += raw_projected
    calibrated_projected_seconds_total += calibrated_projected
    benchmark_records.append(
        {
            "fold": fold,
            "window_count": benchmark_count,
            "elapsed_seconds": elapsed,
            "variant_count": len(active_variants),
            "work_windows_per_variant": work_windows_per_variant,
            "work_windows_projected": work_windows,
            "raw_conservative_projected_seconds": raw_projected,
            "calibrated_projected_seconds": calibrated_projected,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "train_metrics": benchmark_metrics,
        }
    )
    del benchmark_loader, benchmark_scaler, benchmark_optimizer, benchmark_tracker
    torch.cuda.empty_cache()

benchmark_summary = {
    "records": benchmark_records,
    "raw_conservative_projected_seconds_total": raw_projected_seconds_total,
    "calibrated_projected_seconds_total": calibrated_projected_seconds_total,
    "runtime_gate_seconds": float(train_cfg["runtime_gate_hours"]) * 3600.0,
    "projection_multiplier": float(train_cfg["runtime_projection_multiplier"]),
    "calibration": {
        **projection_calibration,
        "effective_factor": calibration_factor,
        "gate_uses": "calibrated_projected_seconds_total",
    },
    "variant_count": len(active_variants),
}
(WORKING_ROOT / "benchmark_summary.json").write_text(
    json.dumps(benchmark_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(benchmark_summary, indent=2))
if calibrated_projected_seconds_total > benchmark_summary["runtime_gate_seconds"]:
    runtime_gate_hours = benchmark_summary["runtime_gate_seconds"] / 3600.0
    update_metrics(
        METRICS_PATH,
        {
            "status": "failed",
            "notes": (
                "Stopped before full training because the calibrated runtime projection "
                f"exceeded the configured {runtime_gate_hours:g}-hour gate."
            ),
            "train_stage": {"benchmark": benchmark_summary},
        },
    )
    raise RuntimeError(
        "calibrated train projection exceeds the configured "
        f"{runtime_gate_hours:g}-hour runtime gate"
    )

# %% [markdown]
# ## 6. Same-structure DoG ablation across two embryo-held-out folds

# %%
fold_summaries: list[dict[str, Any]] = []
model_manifest_records: list[dict[str, Any]] = []
teacher_audit_records: list[dict[str, Any]] = []
outer_metrics_by_variant: dict[str, dict[int, dict[str, Any]]] = {
    variant: {} for variant in active_variants
}
initial_state_sha_by_fold: dict[int, str] = {}
initial_logits_by_fold: dict[int, torch.Tensor] = {}
training_started = time.perf_counter()

training_jobs = [(record, variant) for record in splits for variant in active_variants]
for record, variant in training_jobs:
    fold = int(record["fold"])
    fold_seed = global_seed + fold
    train_paths = paths_for_samples(cache_paths, record["gradient_update"])
    validation_paths = paths_for_samples(cache_paths, record["internal_validation"])
    evaluation_paths = paths_for_samples(cache_paths, record["outer_evaluation"])
    normalizer = normalizers_by_fold[fold]
    seed_everything(fold_seed)
    tracker = new_initialized_tracker(device, variant)
    initial_state_sha = canonical_state_sha256(tracker.state_dict())
    expected_initial_state_sha = initial_state_sha_by_fold.setdefault(fold, initial_state_sha)
    if initial_state_sha != expected_initial_state_sha:
        raise RuntimeError(f"fold {fold} variants do not share identical initialization")
    optimizer = new_optimizer(tracker)
    scaler = new_scaler()
    train_loader = make_loader(
        train_paths,
        shuffle=True,
        seed=fold_seed,
        dog_normalizer=normalizer,
        use_dog_values=bool(ablation_cfg[variant]["use_dog_values"]),
    )
    validation_loader = make_loader(
        validation_paths,
        shuffle=False,
        seed=fold_seed,
        dog_normalizer=normalizer,
        use_dog_values=bool(ablation_cfg[variant]["use_dog_values"]),
    )
    evaluation_loader = make_loader(
        evaluation_paths,
        shuffle=False,
        seed=fold_seed,
        dog_normalizer=normalizer,
        use_dog_values=bool(ablation_cfg[variant]["use_dog_values"]),
    )
    tracker.eval()
    initial_batch = move_batch_to_device(next(iter(evaluation_loader)), device)
    with torch.no_grad():
        initial_logits = tracker_logits(tracker, initial_batch).detach().cpu()
    reference_logits = initial_logits_by_fold.setdefault(fold, initial_logits)
    if not torch.equal(initial_logits, reference_logits):
        raise RuntimeError(f"fold {fold} variants do not have identical initial logits")
    del initial_batch, initial_logits, reference_logits

    baseline_outer = evaluate_tracker(
        tracker,
        evaluation_loader,
        device,
        gamma=float(loss_cfg["focal_gamma"]),
        use_amp=use_amp,
        with_diagnostics=True,
    )
    if variant == active_variants[0]:
        teacher_audit_records.append(baseline_outer["teacher"])
    best_score = float("-inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    epochs: list[dict[str, Any]] = []
    for epoch in range(int(train_cfg["epochs"])):
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
            "selected": is_best,
        }
        epochs.append(epoch_record)
        print(json.dumps({"variant": variant, "fold": fold, **epoch_record}, default=float))
    if best_state is None:
        raise RuntimeError(f"fold {fold} did not select a checkpoint")
    tracker.load_state_dict(best_state, strict=True)
    trained_outer = evaluate_tracker(
        tracker,
        evaluation_loader,
        device,
        gamma=float(loss_cfg["focal_gamma"]),
        use_amp=use_amp,
        with_diagnostics=True,
    )
    outer_metrics_by_variant[variant][fold] = trained_outer

    fold_dir = OUTPUT_MODELS / variant / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_path = fold_dir / "primary_tracker_best.pth"
    torch.save(
        {
            "experiment": EXPERIMENT,
            "variant": variant,
            "fold": fold,
            "dog_normalizer": normalizer,
            "dog_normalizer_sha256": json_sha256(normalizer),
            "train_embryo": record["train_embryo"],
            "evaluation_embryo": record["evaluation_embryo"],
            "best_epoch": best_epoch,
            "selection_score": best_score,
            "state_dict": best_state,
        },
        model_path,
    )
    state_sha = canonical_state_sha256(best_state)
    fold_summary = {
        "variant": variant,
        "use_dog_values": bool(ablation_cfg[variant]["use_dog_values"]),
        "fold": fold,
        "initial_state_sha256": initial_state_sha,
        "train_embryo": record["train_embryo"],
        "evaluation_embryo": record["evaluation_embryo"],
        "sample_counts": {
            "gradient_update": len(record["gradient_update"]),
            "internal_validation": len(record["internal_validation"]),
            "outer_evaluation": len(record["outer_evaluation"]),
        },
        "window_counts": {
            "gradient_update": len(train_paths),
            "internal_validation": len(validation_paths),
            "outer_evaluation": len(evaluation_paths),
        },
        "dog_normalizer": normalizer,
        "dog_normalizer_sha256": json_sha256(normalizer),
        "initial_outer_evaluation": baseline_outer,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_internal_selection_score": best_score,
        "trained_outer_evaluation": trained_outer,
        "outer_metric_delta": {
            key: float(trained_outer[key]) - float(baseline_outer[key])
            for key in (
                "legacy_mask_loss",
                "edge_accuracy",
                "positive_edge_recall",
                "division_parent_recall",
                "selection_score",
            )
        },
        "model_file": model_path.relative_to(WORKING_ROOT).as_posix(),
        "model_file_sha256": file_sha256(model_path),
        "model_state_sha256": state_sha,
    }
    fold_summary_path = WORKING_ROOT / f"{variant}_fold_{fold}_summary.json"
    fold_summary_path.write_text(
        json.dumps(fold_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fold_summaries.append(fold_summary)
    model_manifest_records.append(
        {
            "variant": variant,
            "fold": fold,
            "path": fold_summary["model_file"],
            "file_sha256": fold_summary["model_file_sha256"],
            "canonical_state_sha256": state_sha,
            "initial_state_sha256": initial_state_sha,
            "best_epoch": best_epoch,
            "train_embryo": record["train_embryo"],
            "evaluation_embryo": record["evaluation_embryo"],
        }
    )
    del evaluation_loader, validation_loader, train_loader, scaler, optimizer, tracker
    torch.cuda.empty_cache()

expected_model_count = int(model_cfg["output"]["model_count"])
if len(model_manifest_records) != expected_model_count:
    raise RuntimeError(
        f"expected {expected_model_count} trained trackers, got {len(model_manifest_records)}"
    )
progression_gate = evaluate_dog_progression_gate(
    outer_metrics_by_variant,
    validation_cfg["progression_gate"],
)


# %% [markdown]
# ## 7. Teacher audit, model manifest, and metrics

# %%
teacher_audit = aggregate_teacher_stats(teacher_audit_records)
teacher_audit.update(
    {
        "matching": teacher_cfg["matching"],
        "max_matching_distance_um": max_matching_distance_um,
        "loss_mask": loss_cfg["mask"],
        "outer_evaluation_coverage": "each of 199 samples exactly once across two folds",
    }
)
(WORKING_ROOT / "teacher_audit.json").write_text(
    json.dumps(teacher_audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)

model_manifest = {
    "experiment": EXPERIMENT,
    "created_at": datetime.now(UTC).isoformat(),
    "model_class": "SimpleNodeTransformer",
    "active_variants": active_variants,
    "trainable_components": model_cfg["trainable_components"],
    "frozen_components": model_cfg["frozen_components"],
    "public_checkpoint_file_sha256": observed_source_hashes["checkpoint"],
    "public_tracker_initial_state_sha256": public_tracker_state_sha256,
    "public_train_script_sha256": observed_source_hashes["train_script"],
    "public_model_source_sha256": observed_source_hashes["model_source"],
    "cache_summary_sha256": cache_summary["summary_sha256"],
    "cache_identity_sha256": cache_summary["cache_identity_sha256"],
    "dog_feature_manifest_content_sha256": feature_manifest["manifest_content_sha256"],
    "gt_window_filter_audit_sha256": gt_window_filter_audit_sha256,
    "dog_normalizer_manifest_sha256": normalizer_manifest_sha256,
    "dog_contract": dog_cfg,
    "annotation_content_sha256": json_sha256(annotation_manifest),
    "feature_schema_contract_sha256": json_sha256(
        {
            "cache_schema_version": cache_cfg["schema_version"],
            "arrays": cache_cfg["arrays"],
            "primary_feature_channels": cache_cfg["feature_channels"],
            "tracker_feature_dim": params_cfg["feature_dim"],
            "dog_feature_dim": dog_cfg["feature_dimension"],
            "dog_feature_manifest_content_sha256": feature_manifest["manifest_content_sha256"],
        }
    ),
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
    "status": (
        "train_stage_completed_gate_passed"
        if progression_gate["passed"]
        else "train_stage_completed_gate_failed"
    ),
    "official_metric_computed": False,
    "full_graph_inference_run": False,
    "submission_created": False,
    "conditional_validation": validation_cfg["conditional_validation"],
    "elapsed_seconds": time.perf_counter() - training_started,
    "notebook_elapsed_seconds": time.perf_counter() - NOTEBOOK_STARTED,
    "benchmark": benchmark_summary,
    "gt_window_filter": {
        **gt_window_filter_summary,
        "audit_sha256": gt_window_filter_audit_sha256,
    },
    "teacher_audit": teacher_audit,
    "dog_normalizer_manifest_sha256": normalizer_manifest_sha256,
    "progression_gate": progression_gate,
    "folds": fold_summaries,
    "model_manifest_sha256": model_manifest_sha,
}
training_summary_path = WORKING_ROOT / str(model_cfg["output"]["training_summary"])
training_summary_path.write_text(
    json.dumps(training_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)

update_metrics(
    METRICS_PATH,
    {
        "status": "running",
        "metric": "fixed_candidate_public_teacher_edge_metrics",
        "evidence": {
            "kaggle": {
                "kernel_source_ids": [
                    str(cache_cfg["kernel_source"]),
                    str(public_cfg["dataset_ref"]),
                    str(runtime_cfg["kaggle"]["train"]["feature_kernel_source"]),
                ],
                "resource": torch.cuda.get_device_name(device),
                "notebook_runtime_seconds": training_summary["notebook_elapsed_seconds"],
                "internet_enabled": False,
            },
            "artifacts": {
                "input_file_sha": model_manifest["annotation_content_sha256"],
                "cache_file_sha": cache_summary["summary_sha256"],
                "feature_schema_sha": model_manifest["feature_schema_contract_sha256"],
                "feature_content_sha": feature_manifest["manifest_content_sha256"],
                "gt_window_filter_audit_sha": gt_window_filter_audit_sha256,
                "dog_normalizer_manifest_sha": normalizer_manifest_sha256,
                "row_count": len(cache_paths),
                "group_count": len(sample_names),
                "feature_count": int(params_cfg["feature_dim"]),
                "dog_feature_count": int(dog_cfg["feature_dimension"]),
                "model_manifest_sha": model_manifest_sha,
                "model_count": len(model_manifest_records),
                "model_shas": {
                    f"{record['variant']}/fold_{record['fold']}": record["file_sha256"]
                    for record in model_manifest_records
                },
            },
        },
        "train_stage": training_summary,
        "notes": (
            "DoG pair-level progression gate passed; full-graph inference is the next stage."
            if progression_gate["passed"]
            else "DoG pair-level progression gate failed; stopped before full-graph inference "
            "as required."
        ),
    },
)

print(json.dumps(training_summary, indent=2, default=float))
print("Model manifest:", model_manifest_path, model_manifest_sha)
print("No submission was created.")
