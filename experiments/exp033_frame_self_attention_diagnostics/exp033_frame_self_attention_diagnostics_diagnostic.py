# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Saved tracker pair-ranking diagnostic
#
# Re-evaluate the exp016 current tracker and exp025 Model A, Model B and
# identity-initialized Model B on exactly the same fixed two-frame windows.
# The public image encoder, candidates, teacher, and checkpoint selection stay
# fixed. This notebook writes exact PR curves and conditional teacher-error
# readouts. It does not train, restore a full graph, or submit predictions.

# %% [markdown]
# ## 1. Runtime and configuration

# %%
from __future__ import annotations

# ruff: noqa: E402
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

EXPERIMENT = "exp033_frame_self_attention_diagnostics"
WORKING_ROOT = Path.cwd()
INPUT_ROOT = Path("/kaggle/input")
if not INPUT_ROOT.is_dir() or not Path("/kaggle/working").is_dir():
    raise RuntimeError("The authoritative diagnostic run must execute on Kaggle.")
config = yaml.safe_load((WORKING_ROOT / "config.yaml").read_text())
if config["experiment"]["name"] != EXPERIMENT:
    raise RuntimeError("packaged diagnostic configuration mismatch")
if config["model"]["training"] != "disabled":
    raise RuntimeError("diagnostic must not train the tracker")
started = time.perf_counter()
print("Experiment:", EXPERIMENT)
print("Variants:", list(config["model"]["checkpoints"]))
print("Fold specs:", config["validation"]["outer_folds"])
print("Target precision:", config["validation"]["target_precision"])

# %% [markdown]
# ## 2. Offline GEFF dependency check

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

# %% [markdown]
# ## 3. Fixed cache, annotations, split, and checkpoint identity

# %%
import torch
from diagnostic_runner import (
    SPLITS,
    VARIANTS,
    benchmark_inference,
    fit_training_boundaries,
    reproduce_fixed_counts,
    resolve_inputs,
    resolve_models,
    summarize_group,
    verify_outer_reproduction,
    write_json,
    write_sample_shard,
)
from frozen_tracker import file_sha256, json_sha256, paths_for_samples
from settings import update_metrics

reference_path = WORKING_ROOT / "reference_pair_metrics.json"
reference = json.loads(reference_path.read_text())
inputs = resolve_inputs(config, INPUT_ROOT)
print("Cache windows after public empty-GT policy:", len(inputs["cache_paths"]))
print("Fixed cache identity SHA:", inputs["cache_identity_sha256"])
print("Annotation groups:", len(inputs["annotations"]))
print("Reference 0.5 metrics SHA:", file_sha256(reference_path))
diagnostic_root = WORKING_ROOT / "diagnostic"
diagnostic_root.mkdir(exist_ok=True)

# %% [markdown]
# ## 4. Benchmark and training-side bucket boundaries
#
# Each fold loads four saved checkpoints. Eight large windows provide an
# initial runtime and memory check. Candidate-count, nearest-neighbor, and
# known-positive displacement quartiles come only from that fold's
# gradient-update samples; the outer embryo never changes a boundary.

# %%
fold_models = {}
fold_audits = {}
fold_boundaries = {}
fold_benchmarks = {}
for fold_record in inputs["splits"]:
    fold = int(fold_record["fold"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Kaggle T4 is required for the full checkpoint diagnostic")
    models, audit = resolve_models(config, INPUT_ROOT, fold_record, device)
    fold_models[fold] = models
    fold_audits[str(fold)] = audit
    evaluation_paths = paths_for_samples(inputs["cache_paths"], fold_record["outer_evaluation"])
    internal_paths = paths_for_samples(inputs["cache_paths"], fold_record["internal_validation"])
    benchmark = benchmark_inference(evaluation_paths, inputs["annotations"], config, models, device)
    projected = (
        benchmark["seconds_per_window"]
        * (len(evaluation_paths) + len(internal_paths))
        * float(config["runtime"]["projection_multiplier"])
    )
    benchmark["projected_seconds"] = projected
    benchmark["evaluation_window_count"] = len(evaluation_paths)
    benchmark["internal_window_count"] = len(internal_paths)
    fold_benchmarks[str(fold)] = benchmark
    if projected > float(config["runtime"]["full_notebook_hours_limit"]) * 3600:
        raise RuntimeError(f"fold {fold} projection exceeds the 12-hour notebook limit")
    training_paths = paths_for_samples(inputs["cache_paths"], fold_record["gradient_update"])
    fold_boundaries[str(fold)] = fit_training_boundaries(
        training_paths, inputs["annotations"], config
    )
    print(
        "Fold",
        fold,
        "benchmark",
        benchmark,
        "training-side boundaries",
        fold_boundaries[str(fold)],
    )
projected_total = sum(item["projected_seconds"] for item in fold_benchmarks.values())
projected_total += time.perf_counter() - started
available_hours = min(
    float(config["runtime"]["full_notebook_hours_limit"]),
    float(config["runtime"]["available_gpu_hours_limit"]),
)
if projected_total > available_hours * 3600:
    raise RuntimeError(f"diagnostic projection exceeds {available_hours:.2f} available GPU hours")
print("Combined projected seconds including setup:", projected_total)
write_json(
    diagnostic_root / "input_and_model_audit.json",
    {
        "cache_summary_sha256": inputs["cache_summary"]["summary_sha256"],
        "cache_identity_sha256": inputs["cache_identity_sha256"],
        "annotation_sha256": inputs["annotation_sha256"],
        "empty_gt_window_filter": inputs["filter_audit"],
        "model_checkpoints": fold_audits,
        "benchmarks": fold_benchmarks,
        "bucket_boundaries": fold_boundaries,
    },
)

# %% [markdown]
# ## 5. Save aligned pair probabilities by sample
#
# Source-axis softmax is applied to the complete window before selecting
# active teacher-mask pairs. A shard holds pair row/column indices, candidate
# IDs and physical coordinates once per window, labels, unknown-endpoint
# flags, and all four probabilities. This keeps the full image cache out of
# the generated output.

# %%
shard_records = {}
for fold_record in inputs["splits"]:
    fold = int(fold_record["fold"])
    shard_records[str(fold)] = {}
    for split in SPLITS:
        split_samples = fold_record[split]
        split_paths = paths_for_samples(inputs["cache_paths"], split_samples)
        paths_by_sample = {
            sample: [path for path in split_paths if path.parent.name == sample]
            for sample in split_samples
        }
        records = []
        for index, sample in enumerate(sorted(paths_by_sample), start=1):
            sample_paths = paths_by_sample[sample]
            if not sample_paths:
                raise RuntimeError(f"no fixed windows for {sample}")
            record = write_sample_shard(
                sample_paths,
                inputs["annotations"],
                config,
                fold_models[fold],
                device,
                diagnostic_root / "pair_shards" / f"fold_{fold}" / split / f"{sample}.npz",
            )
            records.append(record)
            if index % 10 == 0 or index == len(paths_by_sample):
                print(f"Fold {fold} {split}: {index}/{len(paths_by_sample)} samples")
        shard_records[str(fold)][split] = records
shard_manifest_sha = write_json(diagnostic_root / "pair_shard_manifest.json", shard_records)
print("Pair shard manifest SHA:", shard_manifest_sha)

# %% [markdown]
# ## 6. Reproduce the saved 0.5 results before comparing rankings

# %%
for fold_record in inputs["splits"]:
    fold = str(fold_record["fold"])
    embryo = fold_record["evaluation_embryo"]
    records = shard_records[fold]["outer_evaluation"]
    for variant in VARIANTS:
        observed = reproduce_fixed_counts(records, variant)
        verify_outer_reproduction(observed, reference, embryo, variant)
        print(
            "Reproduced",
            embryo,
            variant,
            observed["fixed_0_5"]["positive_edge_recall"],
            observed["fixed_0_5"]["positive_pair_precision"],
        )

# %% [markdown]
# ## 7. PR curves, matched-endpoint sensitivity, and conditional error buckets
#
# All equal float32 scores enter a PR point together. Precision 0.95 is a
# descriptive comparison only; its threshold is not used for graph decoding.
# Sparse GEFF annotation means active unannotated pairs are conditional
# teacher negatives, not confirmed biological errors.

# %%
summary = {}
for fold_record in inputs["splits"]:
    fold = str(fold_record["fold"])
    summary[fold] = {}
    for split in SPLITS:
        records = shard_records[fold][split]
        summary[fold][split] = {}
        for variant in VARIANTS:
            output_dir = diagnostic_root / "curves" / f"fold_{fold}" / split
            result = summarize_group(
                records,
                variant,
                fold_boundaries[fold],
                output_dir,
                config,
            )
            summary[fold][split][variant] = result
            print(
                "Fold",
                fold,
                split,
                variant,
                "recall at precision 0.95:",
                result["pr_curves"]["active_mask"]["recall_at_target_precision"],
                "unknown endpoint pairs:",
                result["fixed_0_5"]["unknown_endpoint_pair_count"],
            )
summary_sha = write_json(diagnostic_root / "diagnostic_summary.json", summary)

# %% [markdown]
# ## 8. Evidence and status
#
# This notebook writes execution evidence. The user decides whether the
# experiment is complete and whether any later model change is worth testing.

# %%
elapsed = time.perf_counter() - started
artifact_bytes = sum(
    record["bytes"]
    for folds in shard_records.values()
    for records in folds.values()
    for record in records
)
update_metrics(
    WORKING_ROOT / "metrics.json",
    {
        "status": "debug_completed",
        "metric": "conditional_fixed_candidate_pair_ranking",
        "evidence": {
            "kaggle": {
                "kernel_source_ids": config["runtime"]["kaggle"]["diagnostic"]["kernel_sources"],
                "resource": torch.cuda.get_device_name(device),
                "notebook_runtime_seconds": elapsed,
                "internet_enabled": False,
            },
            "artifacts": {
                "input_file_sha": json_sha256(inputs["annotation_sha256"]),
                "cache_file_sha": inputs["cache_summary"]["summary_sha256"],
                "feature_schema_sha": json_sha256(
                    {
                        "schema_version": config["data"]["cache"]["schema_version"],
                        "arrays": config["data"]["cache"]["arrays"],
                        "feature_channels": config["data"]["cache"]["feature_channels"],
                    }
                ),
                "feature_content_sha": inputs["cache_identity_sha256"],
                "row_count": len(inputs["cache_paths"]),
                "group_count": len(inputs["annotations"]),
                "model_count": 8,
                "model_manifest_sha": json_sha256(
                    {
                        f"{fold}/{variant}": value["manifest_sha256"]
                        for fold, entries in fold_audits.items()
                        for variant, value in entries.items()
                    }
                ),
                "model_shas": {
                    f"{fold}/{variant}": value["checkpoint_sha256"]
                    for fold, entries in fold_audits.items()
                    for variant, value in entries.items()
                },
                "oof_prediction_sha": shard_manifest_sha,
            },
        },
        "pair_ranking_diagnostic": {
            "target_precision": config["validation"]["target_precision"],
            "reference_0_5_metrics_sha256": file_sha256(reference_path),
            "pair_shard_manifest_sha256": shard_manifest_sha,
            "diagnostic_summary_sha256": summary_sha,
            "pair_shard_bytes": artifact_bytes,
            "fold_benchmarks": fold_benchmarks,
            "bucket_boundaries": fold_boundaries,
            "official_graph_score_computed": False,
            "submission_created": False,
        },
    },
)
print("Diagnostic summary SHA:", summary_sha)
print("Pair shard bytes:", artifact_bytes)
print("Notebook elapsed seconds:", elapsed)
