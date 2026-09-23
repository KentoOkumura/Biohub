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
# # exp039: DoG responses at fixed candidates
#
# 1. Resolve the exp015 fixed-candidate cache and competition images.
# 2. Benchmark original-resolution 3D Gaussian smoothing on both embryos.
# 3. Save two signed DoG responses per fixed candidate and verify each role.
# 4. Record content hashes and timing. No detection candidates are generated.

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
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import yaml

EXPERIMENT = "exp039_dog_features"
WORK = Path.cwd()
CONFIG = yaml.safe_load((WORK / "config.yaml").read_text())
CACHE_CFG = CONFIG["data"]["cache"]
DOG_CFG = CONFIG["data"]["dog_features"]
DOWNSAMPLE = tuple(CONFIG["model"]["params"]["downsample_zyx"])
FEATURE_ROOT = WORK / DOG_CFG["feature_output"]
FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
STARTED = time.perf_counter()
PARALLEL_WORKERS = int(DOG_CFG["parallel_frame_workers"])
if PARALLEL_WORKERS < 1 or (os.cpu_count() and PARALLEL_WORKERS > os.cpu_count()):
    raise ValueError("invalid DoG frame worker count for this CPU")

if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
    raise RuntimeError("Authoritative raw-image feature extraction must run on Kaggle")

# %% [markdown]
# ## 1. Offline runtime and verified input cache


# %%
def ensure_zarr() -> Any:
    try:
        installed = importlib.import_module("zarr")
        if int(installed.__version__.split(".")[0]) >= 3:
            return installed
    except (ImportError, ValueError):
        pass
    wheel_dirs = [
        path
        for path in Path("/kaggle/input").rglob("wheels")
        if path.is_dir() and "biohub-tracking-support-pack-50ep-v1" in path.as_posix()
    ]
    if not wheel_dirs:
        raise FileNotFoundError("the support dataset has no offline Zarr wheels")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--force-reinstall",
    ]
    for wheel_dir in wheel_dirs:
        command.extend(["--find-links", str(wheel_dir)])
    command.extend(
        ["zarr>=3.0.10,<4", "numcodecs>=0.13,<0.16", "donfig>=0.8", "google-crc32c>=1.5"]
    )
    subprocess.run(command, check=True)
    return importlib.import_module("zarr")


zarr = ensure_zarr()
from dog_features import (
    canonical_sha256,
    frame_feature_payload,
    load_window_side_features,
    process_frames_parallel,
    read_cache_role,
    save_frame_features,
)
from frozen_tracker import (
    discover_cache_paths,
    file_sha256,
    recompute_cache_identity_sha256,
    validate_cache_summary,
)


def only_existing(paths: list[Path], label: str) -> Path:
    matches = sorted({path.resolve() for path in paths if path.exists()})
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label}, found {matches}")
    return matches[0]


cache_slug = str(CACHE_CFG["kernel_source"]).split("/", 1)[-1]
cache_root_candidates = [
    p.parent
    for p in Path("/kaggle/input/notebooks").rglob(str(CACHE_CFG["summary_file"]))
    if cache_slug in p.as_posix()
]
cache_output_root = only_existing(cache_root_candidates, "exp015 cache output")
cache_summary = validate_cache_summary(
    cache_output_root / str(CACHE_CFG["summary_file"]), CACHE_CFG
)
cache_root = cache_output_root / str(CACHE_CFG["directory"])
cache_paths = discover_cache_paths(cache_root, CACHE_CFG)
if recompute_cache_identity_sha256(cache_paths) != str(CACHE_CFG["identity_sha256"]):
    raise RuntimeError("exp015 cache identity mismatch")
competition = "biohub-cell-tracking-during-development"
train_dir = only_existing(
    [
        Path(f"/kaggle/input/competitions/{competition}/train"),
        Path(f"/kaggle/input/{competition}/train"),
    ],
    "competition train directory",
)
samples = sorted({path.parent.name for path in cache_paths})
assert len(samples) == int(CACHE_CFG["expected_dataset_count"])
print("Fixed candidates:", len(samples), "datasets,", len(cache_paths), "windows")
print("Cache SHA:", cache_summary["summary_sha256"], cache_summary["cache_identity_sha256"])

# %% [markdown]
# ## 2. Original-frame responses and runtime benchmark


# %%
raw_image_sources: dict[str, dict[str, Any]] = {}


def image_array(sample: str) -> Any:
    image_path = train_dir / f"{sample}.zarr"
    if not image_path.is_dir():
        raise FileNotFoundError(image_path)
    metadata_path = image_path / "0" / "zarr.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    group = zarr.open_group(str(image_path), mode="r")
    array = group["0"]
    if array.ndim != 4 or int(array.shape[0]) != int(CONFIG["data"]["expected_frames_per_sample"]):
        raise ValueError(f"unexpected image time/ZYX shape: {sample} {array.shape}")
    raw_image_sources[sample] = {
        "input_relative_path": image_path.relative_to(Path("/kaggle/input")).as_posix(),
        "array_shape": list(array.shape),
        "array_dtype": str(array.dtype),
        "zarr_array_metadata_sha256": file_sha256(metadata_path),
    }
    return array


def requests_for_frame(sample: str, frame: int, frame_count: int) -> dict[str, Any]:
    sample_cache = cache_root / sample
    requests: dict[str, Any] = {}
    for side, stem in (
        ("src", f"{frame:06d}_{frame + 1:06d}"),
        ("tgt", f"{frame - 1:06d}_{frame:06d}"),
    ):
        if side == "src" and frame == frame_count - 1:
            continue
        if side == "tgt" and frame == 0:
            continue
        path = sample_cache / f"{stem}.npz"
        arrays, metadata = read_cache_role(path, side)
        if metadata["window_frames"][0 if side == "src" else 1] != frame:
            raise ValueError(f"cache role belongs to another frame: {path}")
        requests[side] = (path, arrays, metadata)
    return requests


feature_records: dict[tuple[str, int], dict[str, Any]] = {}


def process_frame(sample: str, array: Any, frame: int) -> dict[str, Any]:
    path = FEATURE_ROOT / sample / f"{frame:06d}.npz"
    requests = requests_for_frame(sample, frame, int(array.shape[0]))
    started = time.perf_counter()
    image = np.asarray(array[frame], dtype=np.float32)
    arrays, roles = frame_feature_payload(image, requests, DOG_CFG, downsample_zyx=DOWNSAMPLE)
    content_sha = save_frame_features(
        path,
        arrays,
        {"experiment": EXPERIMENT, "dataset": sample, "frame": frame, "roles": roles},
    )
    for side, (cache_path, _, _) in requests.items():
        restored = load_window_side_features(path, cache_path, side)
        if not np.array_equal(restored, arrays[f"{side}_responses"]):
            raise RuntimeError(f"DoG feature round trip differs: {path} {side}")
    record = {
        "dataset": sample,
        "frame": frame,
        "feature_content_sha256": content_sha,
        "roles": roles,
        "candidate_count": sum(len(arrays[f"{side}_responses"]) for side in roles),
        "elapsed_seconds": time.perf_counter() - started,
        "file_bytes": path.stat().st_size,
    }
    feature_records[(sample, frame)] = record
    return record


first_by_embryo = {
    embryo: min(sample for sample in samples if sample.startswith(f"{embryo}_"))
    for embryo in CONFIG["data"]["expected_embryo_counts"]
}
benchmark_rows = []
benchmark_by_embryo = {}
setup_seconds = time.perf_counter() - STARTED
for embryo, sample in first_by_embryo.items():
    array = image_array(sample)
    indices = np.linspace(
        0, int(array.shape[0]) - 1, int(DOG_CFG["compute_benchmark_frames_per_embryo"]), dtype=int
    ).tolist()
    started = time.perf_counter()
    rows = process_frames_parallel(indices, partial(process_frame, sample, array), PARALLEL_WORKERS)
    elapsed = time.perf_counter() - started
    benchmark_rows.extend(rows)
    benchmark_by_embryo[embryo] = {
        "dataset": sample,
        "observed_frames": len(rows),
        "batch_wall_seconds": elapsed,
        "effective_wall_seconds_per_frame": elapsed / len(rows),
        "maximum_individual_frame_seconds": max(row["elapsed_seconds"] for row in rows),
    }
    # Recompute one frame without concurrency to guard the signed response values.
    reference = rows[0]
    repeated = process_frame(sample, array, int(reference["frame"]))
    if repeated["feature_content_sha256"] != reference["feature_content_sha256"]:
        raise RuntimeError(f"concurrent and sequential DoG values differ: {sample}")
max_seconds_per_frame = max(
    record["effective_wall_seconds_per_frame"] for record in benchmark_by_embryo.values()
)
projected_seconds = setup_seconds + (
    max_seconds_per_frame
    * len(samples)
    * int(CONFIG["data"]["expected_frames_per_sample"])
    * float(DOG_CFG["compute_projection_multiplier"])
)
benchmark = {
    "samples": first_by_embryo,
    "observed_frames": len(benchmark_rows),
    "parallel_workers": PARALLEL_WORKERS,
    "setup_seconds": setup_seconds,
    "by_embryo": benchmark_by_embryo,
    "maximum_effective_wall_seconds_per_frame": max_seconds_per_frame,
    "projected_seconds": projected_seconds,
    "gate_seconds": float(DOG_CFG["compute_gate_hours"]) * 3600,
    "peak_image_bytes": max(
        int(np.prod(image_array(sample).shape[1:])) * 4 for sample in first_by_embryo.values()
    ),
}
(WORK / "dog_feature_benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
print("DoG feature benchmark:", benchmark)
if projected_seconds > benchmark["gate_seconds"]:
    raise RuntimeError("original-resolution DoG feature extraction exceeds its runtime gate")

# %% [markdown]
# ## 3. Compute fixed-candidate features for every training frame

# %%
for sample_index, sample in enumerate(samples, start=1):
    array = image_array(sample)
    pending_frames = [
        frame for frame in range(int(array.shape[0])) if (sample, frame) not in feature_records
    ]
    process_frames_parallel(pending_frames, partial(process_frame, sample, array), PARALLEL_WORKERS)
    print(f"[{sample_index}/{len(samples)}] {sample}: DoG features saved", flush=True)
    if time.perf_counter() - STARTED > benchmark["gate_seconds"]:
        raise RuntimeError("DoG feature Notebook exceeded its runtime gate")

# %% [markdown]
# ## 4. Feature manifest and evidence

# %%
ordered = [feature_records[key] for key in sorted(feature_records)]
if len(ordered) != len(samples) * int(CONFIG["data"]["expected_frames_per_sample"]):
    raise RuntimeError("incomplete DoG frame coverage")
manifest = {
    "experiment": EXPERIMENT,
    "created_at": datetime.now(UTC).isoformat(),
    "source_cache_summary_sha256": cache_summary["summary_sha256"],
    "source_cache_identity_sha256": cache_summary["cache_identity_sha256"],
    "dog_contract": DOG_CFG,
    "raw_image_sources": dict(sorted(raw_image_sources.items())),
    "dataset_count": len(samples),
    "window_count": len(cache_paths),
    "frame_count": len(ordered),
    "benchmark": benchmark,
    "frames": ordered,
}
manifest["manifest_content_sha256"] = canonical_sha256(
    {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
)
manifest_path = WORK / "dog_feature_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
print("Feature manifest:", manifest_path, manifest["manifest_content_sha256"])
print("Elapsed seconds:", time.perf_counter() - STARTED)
