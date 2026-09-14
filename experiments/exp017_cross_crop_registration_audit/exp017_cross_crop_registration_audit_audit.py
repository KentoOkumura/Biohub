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
# # exp017 cross-crop registration audit
#
# This CPU-only audit asks whether crops from the same embryo contain a common
# field of view. It screens all same-embryo pairs with multi-timepoint image
# registration, checks negative controls, confirms top pairs in 3D, and then
# measures whether sparse GEFF annotations agree with the same transform.
# It does not train a model, run test inference, or create a submission.

# %%
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

EXPERIMENT = "exp017_cross_crop_registration_audit"
COMPETITION_SLUG = "biohub-cell-tracking-during-development"
OUTPUT_SUBDIR = Path("artifacts/audit_v1")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    embryo_id: str
    crop_id: str
    image_path: Path
    graph_path: Path
    shape: tuple[int, int, int, int]
    chunk_shape: tuple[int, int, int, int]
    dtype: np.dtype


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(json_safe(row))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def find_competition_root(input_root: Path = Path("/kaggle/input")) -> Path:
    candidates = [
        input_root / "competitions" / COMPETITION_SLUG,
        input_root / COMPETITION_SLUG,
    ]
    candidates.extend(input_root.glob(f"**/{COMPETITION_SLUG}"))
    matches = sorted({path.resolve() for path in candidates if path.is_dir()})
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one competition root for {COMPETITION_SLUG}, found {matches}"
        )
    return matches[0]


def image_array_metadata(image_path: Path) -> tuple[tuple[int, ...], tuple[int, ...], np.dtype]:
    metadata = read_json_object(image_path / "0" / "zarr.json")
    shape = tuple(int(value) for value in metadata["shape"])
    chunk_shape = tuple(
        int(value) for value in metadata["chunk_grid"]["configuration"]["chunk_shape"]
    )
    dtype = np.dtype(metadata["data_type"]).newbyteorder("<")
    return shape, chunk_shape, dtype


def discover_samples(train_root: Path, config: dict[str, Any]) -> list[Sample]:
    samples: list[Sample] = []
    for image_path in sorted(train_root.glob("*.zarr")):
        sample_id = image_path.name.removesuffix(".zarr")
        graph_path = train_root / f"{sample_id}.geff"
        if not graph_path.is_dir():
            raise FileNotFoundError(f"missing GEFF pair for {sample_id}")
        embryo_id, separator, crop_id = sample_id.partition("_")
        if not separator:
            raise ValueError(f"sample does not follow embryo_crop naming: {sample_id}")
        shape, chunk_shape, dtype = image_array_metadata(image_path)
        samples.append(
            Sample(
                sample_id=sample_id,
                embryo_id=embryo_id,
                crop_id=crop_id,
                image_path=image_path,
                graph_path=graph_path,
                shape=shape,
                chunk_shape=chunk_shape,
                dtype=dtype,
            )
        )

    expected_count = int(config["validation"]["expected_sample_count"])
    if len(samples) != expected_count:
        raise RuntimeError(f"expected {expected_count} train samples, found {len(samples)}")
    expected_shape = tuple(int(value) for value in config["data"]["expected_image_shape_tzyx"])
    expected_counts = {
        str(key): int(value)
        for key, value in config["validation"]["expected_embryo_counts"].items()
    }
    actual_counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        if sample.shape != expected_shape or sample.chunk_shape != (1, *expected_shape[1:]):
            raise RuntimeError(
                f"unexpected image layout for {sample.sample_id}: "
                f"shape={sample.shape}, chunk={sample.chunk_shape}"
            )
        actual_counts[sample.embryo_id] += 1
    if dict(sorted(actual_counts.items())) != dict(sorted(expected_counts.items())):
        raise RuntimeError(
            f"unexpected embryo counts: expected={expected_counts}, actual={dict(actual_counts)}"
        )
    return samples


def frame_chunk_path(sample: Sample, frame: int) -> Path:
    return sample.image_path / "0" / "c" / str(frame) / "0" / "0" / "0"


def read_image_frame(sample: Sample, frame: int) -> np.ndarray:
    import blosc2

    if frame < 0 or frame >= sample.shape[0]:
        raise IndexError(f"frame outside sample: {sample.sample_id} t={frame}")
    path = frame_chunk_path(sample, frame)
    decoded = blosc2.decompress(path.read_bytes())
    expected_size = int(np.prod(sample.shape[1:], dtype=np.int64)) * sample.dtype.itemsize
    if len(decoded) != expected_size:
        raise RuntimeError(
            f"decoded byte mismatch for {sample.sample_id} t={frame}: "
            f"expected={expected_size}, actual={len(decoded)}"
        )
    return np.frombuffer(decoded, dtype=sample.dtype).reshape(sample.shape[1:]).copy()


def quick_file_signature(path: Path, sample_bytes: int) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode())
    with path.open("rb") as stream:
        digest.update(stream.read(sample_bytes))
        if size > sample_bytes:
            stream.seek(max(0, size - sample_bytes))
            digest.update(stream.read(sample_bytes))
    return digest.hexdigest()


def files_equal(left: Path, right: Path, block_size: int = 4 * 1024 * 1024) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while True:
            left_block = left_stream.read(block_size)
            right_block = right_stream.read(block_size)
            if left_block != right_block:
                return False
            if not left_block:
                return True


def find_freeze_runs(sample: Sample, sample_bytes: int) -> tuple[list[dict[str, Any]], list[str]]:
    paths = [frame_chunk_path(sample, frame) for frame in range(sample.shape[0])]
    signatures = [quick_file_signature(path, sample_bytes) for path in paths]
    runs: list[dict[str, Any]] = []
    start = 0
    while start < len(signatures):
        end = start
        while end + 1 < len(signatures) and signatures[end + 1] == signatures[start]:
            end += 1
        if end > start:
            verified = files_equal(paths[start], paths[start + 1]) and files_equal(
                paths[start], paths[end]
            )
            if verified:
                runs.append(
                    {
                        "sample_id": sample.sample_id,
                        "embryo_id": sample.embryo_id,
                        "start_frame": start,
                        "end_frame": end,
                        "length": end - start + 1,
                        "quick_signature": signatures[start],
                        "byte_verified_first_and_last": True,
                    }
                )
        start = end + 1
    return runs, signatures


def block_mean(array: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    if array.ndim != len(factors):
        raise ValueError(f"rank/factor mismatch: {array.shape}, {factors}")
    slices = tuple(
        slice(0, size - size % factor) for size, factor in zip(array.shape, factors, strict=True)
    )
    cropped = np.asarray(array[slices], dtype=np.float32)
    reshape: list[int] = []
    mean_axes: list[int] = []
    for axis, factor in enumerate(factors):
        reshape.extend([cropped.shape[axis] // factor, factor])
        mean_axes.append(2 * axis + 1)
    return cropped.reshape(reshape).mean(axis=tuple(mean_axes))


def robust_highpass(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, [1.0, 99.5])
    if not math.isfinite(float(high - low)) or high <= low:
        return np.zeros_like(image, dtype=np.float32)
    normalized = np.clip((image - low) / (high - low), 0.0, 1.0)
    highpass = gaussian_filter(normalized, 0.7) - gaussian_filter(normalized, 3.0)
    scale = float(np.std(highpass))
    if scale <= 1e-8:
        return np.zeros_like(highpass, dtype=np.float32)
    return np.asarray((highpass - np.mean(highpass)) / scale, dtype=np.float32)


def make_thumbnail(volume: np.ndarray, downsample_yx: int) -> np.ndarray:
    mip = np.asarray(volume, dtype=np.float32).max(axis=0)
    reduced = block_mean(mip, (downsample_yx, downsample_yx))
    return robust_highpass(reduced)


def make_lowres_volume(volume: np.ndarray, factors: tuple[int, int, int]) -> np.ndarray:
    reduced = block_mean(np.asarray(volume, dtype=np.float32), factors)
    low, high = np.percentile(reduced, [1.0, 99.5])
    if high <= low:
        return np.zeros_like(reduced, dtype=np.float32)
    normalized = np.clip((reduced - low) / (high - low), 0.0, 1.0)
    background = gaussian_filter(normalized, (1.0, 2.0, 2.0))
    highpass = normalized - background
    scale = float(np.std(highpass))
    return np.asarray(highpass / max(scale, 1e-8), dtype=np.float32)


def apply_d4(array: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return array
    if transform == "rot90":
        return np.rot90(array, 1, axes=(-2, -1))
    if transform == "rot180":
        return np.rot90(array, 2, axes=(-2, -1))
    if transform == "rot270":
        return np.rot90(array, 3, axes=(-2, -1))
    if transform == "flip_x":
        return np.flip(array, axis=-1)
    if transform == "flip_y":
        return np.flip(array, axis=-2)
    if transform == "transpose":
        return np.swapaxes(array, -2, -1)
    if transform == "anti_transpose":
        return np.flip(np.flip(np.swapaxes(array, -2, -1), axis=-2), axis=-1)
    raise ValueError(f"unknown D4 transform: {transform}")


def overlap_slices(
    shape_a: tuple[int, ...], shape_b: tuple[int, ...], shift: tuple[int, ...]
) -> tuple[tuple[slice, ...], tuple[slice, ...], int]:
    if not (len(shape_a) == len(shape_b) == len(shift)):
        raise ValueError("shape and shift ranks must agree")
    slices_a: list[slice] = []
    slices_b: list[slice] = []
    count = 1
    for size_a, size_b, delta in zip(shape_a, shape_b, shift, strict=True):
        start_a = max(0, int(delta))
        end_a = min(size_a, int(delta) + size_b)
        length = max(0, end_a - start_a)
        start_b = max(0, -int(delta))
        slices_a.append(slice(start_a, start_a + length))
        slices_b.append(slice(start_b, start_b + length))
        count *= length
    return tuple(slices_a), tuple(slices_b), count


def overlap_ncc(
    reference: np.ndarray,
    moving: np.ndarray,
    shift: tuple[int, ...],
    min_overlap_fraction: float,
) -> tuple[float, float]:
    slices_a, slices_b, count = overlap_slices(reference.shape, moving.shape, shift)
    overlap_fraction = count / min(reference.size, moving.size)
    if overlap_fraction < min_overlap_fraction or count < 8:
        return float("nan"), float(overlap_fraction)
    left = np.asarray(reference[slices_a], dtype=np.float64).reshape(-1)
    right = np.asarray(moving[slices_b], dtype=np.float64).reshape(-1)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return float("nan"), float(overlap_fraction)
    return float(np.dot(left, right) / denominator), float(overlap_fraction)


def phase_shift_from_ffts(
    reference_fft: np.ndarray, moving_fft: np.ndarray
) -> tuple[tuple[int, int], float]:
    cross_power = reference_fft * np.conj(moving_fft)
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    surface = np.abs(np.fft.ifft2(cross_power))
    peak_index = np.unravel_index(int(np.argmax(surface)), surface.shape)
    shift = []
    for index, size in zip(peak_index, surface.shape, strict=True):
        shift.append(int(index if index <= size // 2 else index - size))
    peak_ratio = float(surface[peak_index] / max(float(np.median(surface)), 1e-12))
    return (shift[0], shift[1]), peak_ratio


def equivalent_phase_shifts(
    base_shift: tuple[int, int], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    return sorted(
        {
            (base_shift[0] + ky * shape[0], base_shift[1] + kx * shape[1])
            for ky in (-1, 0, 1)
            for kx in (-1, 0, 1)
        }
    )


def select_phase_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    base_shift: tuple[int, int],
    min_overlap_fraction: float,
) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for shift in equivalent_phase_shifts(base_shift, reference.shape):
        ncc, overlap = overlap_ncc(reference, moving, shift, min_overlap_fraction)
        if not math.isfinite(ncc):
            continue
        objective = float(ncc * math.sqrt(overlap))
        row: dict[str, float | int] = {
            "shift_y": shift[0],
            "shift_x": shift[1],
            "ncc": ncc,
            "overlap_fraction": overlap,
            "objective": objective,
        }
        if best is None or float(row["objective"]) > float(best["objective"]):
            best = row
    if best is None:
        return {
            "shift_y": 0,
            "shift_x": 0,
            "ncc": float("nan"),
            "overlap_fraction": 0.0,
            "objective": float("nan"),
        }
    return best


def best_ncc_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    min_overlap_fraction: float,
) -> dict[str, float | int]:
    reference = np.asarray(reference, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    ones_reference = np.ones_like(reference)
    ones_moving = np.ones_like(moving)
    reversed_moving = tuple(slice(None, None, -1) for _ in range(moving.ndim))
    sum_ab = fftconvolve(reference, moving[reversed_moving], mode="full")
    count = fftconvolve(ones_reference, ones_moving[reversed_moving], mode="full")
    sum_a = fftconvolve(reference, ones_moving[reversed_moving], mode="full")
    sum_b = fftconvolve(ones_reference, moving[reversed_moving], mode="full")
    sum_aa = fftconvolve(reference * reference, ones_moving[reversed_moving], mode="full")
    sum_bb = fftconvolve(ones_reference, (moving * moving)[reversed_moving], mode="full")

    count = np.maximum(np.rint(count), 1.0)
    numerator = sum_ab - sum_a * sum_b / count
    variance_a = np.maximum(sum_aa - sum_a * sum_a / count, 0.0)
    variance_b = np.maximum(sum_bb - sum_b * sum_b / count, 0.0)
    denominator = np.sqrt(variance_a * variance_b)
    ncc = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, -np.inf),
        where=denominator > 1e-10,
    )
    overlap = count / min(reference.size, moving.size)
    valid = overlap >= min_overlap_fraction
    objective = np.where(valid, ncc * np.sqrt(overlap), -np.inf)
    flat_index = int(np.argmax(objective))
    peak_index = np.unravel_index(flat_index, objective.shape)
    if not math.isfinite(float(objective[peak_index])):
        return {
            "shift_y": 0,
            "shift_x": 0,
            "ncc": float("nan"),
            "overlap_fraction": 0.0,
            "objective": float("nan"),
        }
    shifts = tuple(
        int(index - (size - 1)) for index, size in zip(peak_index, moving.shape, strict=True)
    )
    result: dict[str, float | int] = {
        "ncc": float(ncc[peak_index]),
        "overlap_fraction": float(overlap[peak_index]),
        "objective": float(objective[peak_index]),
    }
    axis_names = (
        ("shift_y", "shift_x") if reference.ndim == 2 else ("shift_z", "shift_y", "shift_x")
    )
    result.update({name: shift for name, shift in zip(axis_names, shifts, strict=True)})
    return result


def median_and_mad(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return median, float(np.median(np.abs(array - median)))


def screen_pair(
    sample_a: Sample,
    sample_b: Sample,
    frames: list[int],
    thumbnails: dict[tuple[str, int], np.ndarray],
    ffts: dict[tuple[str, int], np.ndarray],
    min_overlap_fraction: float,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for frame in frames:
        key_a = (sample_a.sample_id, frame)
        key_b = (sample_b.sample_id, frame)
        base_shift, peak_ratio = phase_shift_from_ffts(ffts[key_a], ffts[key_b])
        selected = select_phase_shift(
            thumbnails[key_a], thumbnails[key_b], base_shift, min_overlap_fraction
        )
        selected.update({"frame": frame, "phase_peak_ratio": peak_ratio})
        results.append(selected)
    ncc = [float(row["ncc"]) for row in results]
    objectives = [float(row["objective"]) for row in results]
    shifts_y = [float(row["shift_y"]) for row in results]
    shifts_x = [float(row["shift_x"]) for row in results]
    median_y, mad_y = median_and_mad(shifts_y)
    median_x, mad_x = median_and_mad(shifts_x)
    return {
        "sample_a": sample_a.sample_id,
        "sample_b": sample_b.sample_id,
        "embryo_id": sample_a.embryo_id,
        "phase_ncc_median": float(np.nanmedian(ncc)),
        "phase_objective_median": float(np.nanmedian(objectives)),
        "phase_peak_ratio_median": float(
            np.median([float(row["phase_peak_ratio"]) for row in results])
        ),
        "shift_y_median_ds": median_y,
        "shift_x_median_ds": median_x,
        "shift_mad_ds": float(math.hypot(mad_y, mad_x)),
        "frame_results_json": json.dumps(json_safe(results), sort_keys=True),
    }


def registration_for_transform(
    sample_a: Sample,
    sample_b: Sample,
    frames: list[int],
    mismatch_offset: int,
    thumbnails: dict[tuple[str, int], np.ndarray],
    transform: str,
    min_overlap_fraction: float,
) -> dict[str, Any]:
    same_results: list[dict[str, Any]] = []
    mismatch_results: list[dict[str, Any]] = []
    frame_count = sample_a.shape[0]
    for frame in frames:
        reference = thumbnails[(sample_a.sample_id, frame)]
        moving = apply_d4(thumbnails[(sample_b.sample_id, frame)], transform)
        same = best_ncc_shift(reference, moving, min_overlap_fraction)
        same["frame_a"] = frame
        same["frame_b"] = frame
        same_results.append(same)

        mismatch_frame = frame + mismatch_offset
        if mismatch_frame >= frame_count:
            mismatch_frame = frame - mismatch_offset
        mismatch = best_ncc_shift(
            reference,
            apply_d4(thumbnails[(sample_b.sample_id, mismatch_frame)], transform),
            min_overlap_fraction,
        )
        mismatch["frame_a"] = frame
        mismatch["frame_b"] = mismatch_frame
        mismatch_results.append(mismatch)

    same_ncc = [float(row["ncc"]) for row in same_results]
    same_overlap = [float(row["overlap_fraction"]) for row in same_results]
    same_objective = [float(row["objective"]) for row in same_results]
    mismatch_ncc = [float(row["ncc"]) for row in mismatch_results]
    shifts_y = [float(row["shift_y"]) for row in same_results]
    shifts_x = [float(row["shift_x"]) for row in same_results]
    median_y, mad_y = median_and_mad(shifts_y)
    median_x, mad_x = median_and_mad(shifts_x)
    shift_mad = float(math.hypot(mad_y, mad_x))
    same_ncc_median = float(np.nanmedian(same_ncc))
    mismatch_ncc_median = float(np.nanmedian(mismatch_ncc))
    objective_median = float(np.nanmedian(same_objective))
    return {
        "sample_a": sample_a.sample_id,
        "sample_b": sample_b.sample_id,
        "embryo_id": (
            sample_a.embryo_id if sample_a.embryo_id == sample_b.embryo_id else "cross_embryo_null"
        ),
        "transform": transform,
        "thumbnail_ncc_median": same_ncc_median,
        "thumbnail_overlap_median": float(np.nanmedian(same_overlap)),
        "thumbnail_objective_median": objective_median,
        "shift_y_median_ds": median_y,
        "shift_x_median_ds": median_x,
        "shift_mad_ds": shift_mad,
        "mismatch_ncc_median": mismatch_ncc_median,
        "same_vs_mismatch_margin": same_ncc_median - mismatch_ncc_median,
        "selection_score": objective_median - 0.02 * shift_mad,
        "same_frame_results_json": json.dumps(json_safe(same_results), sort_keys=True),
        "mismatch_frame_results_json": json.dumps(json_safe(mismatch_results), sort_keys=True),
    }


def refine_pair(
    sample_a: Sample,
    sample_b: Sample,
    frames: list[int],
    mismatch_offset: int,
    thumbnails: dict[tuple[str, int], np.ndarray],
    transforms: list[str],
    min_overlap_fraction: float,
) -> dict[str, Any]:
    alternatives = [
        registration_for_transform(
            sample_a,
            sample_b,
            frames,
            mismatch_offset,
            thumbnails,
            transform,
            min_overlap_fraction,
        )
        for transform in transforms
    ]
    selected = max(alternatives, key=lambda row: float(row["selection_score"]))
    selected = dict(selected)
    selected["transform_alternatives_json"] = json.dumps(
        [
            {
                key: row[key]
                for key in (
                    "transform",
                    "thumbnail_ncc_median",
                    "thumbnail_overlap_median",
                    "shift_mad_ds",
                    "same_vs_mismatch_margin",
                    "selection_score",
                )
            }
            for row in alternatives
        ],
        sort_keys=True,
    )
    return selected


def pair_key(sample_a: str, sample_b: str) -> tuple[str, str]:
    return tuple(sorted((sample_a, sample_b)))


def select_screen_pairs(
    rows: list[dict[str, Any]], per_sample: int, maximum: int
) -> list[tuple[str, str]]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[str(row["sample_a"])].append(row)
        by_sample[str(row["sample_b"])].append(row)
    selected: set[tuple[str, str]] = set()
    for sample_rows in by_sample.values():
        ranked = sorted(
            sample_rows,
            key=lambda row: float(row["phase_objective_median"]),
            reverse=True,
        )
        for row in ranked[:per_sample]:
            selected.add(pair_key(str(row["sample_a"]), str(row["sample_b"])))
    scores = {
        pair_key(str(row["sample_a"]), str(row["sample_b"])): float(row["phase_objective_median"])
        for row in rows
    }
    return sorted(selected, key=lambda key: scores[key], reverse=True)[:maximum]


def freeze_candidate_pairs(freeze_rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    groups: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    for row in freeze_rows:
        groups[
            (
                str(row["embryo_id"]),
                int(row["start_frame"]),
                int(row["end_frame"]),
            )
        ].append(str(row["sample_id"]))
    pairs: set[tuple[str, str]] = set()
    for samples in groups.values():
        for index, sample_a in enumerate(sorted(set(samples))):
            for sample_b in sorted(set(samples))[index + 1 :]:
                pairs.add(pair_key(sample_a, sample_b))
    return pairs


def identical_freeze_schedule_pairs(
    freeze_rows: list[dict[str, Any]],
) -> set[tuple[str, str]]:
    schedule_by_sample: dict[str, set[tuple[int, int]]] = defaultdict(set)
    embryo_by_sample: dict[str, str] = {}
    for row in freeze_rows:
        sample_id = str(row["sample_id"])
        embryo_by_sample[sample_id] = str(row["embryo_id"])
        schedule_by_sample[sample_id].add((int(row["start_frame"]), int(row["end_frame"])))
    groups: dict[tuple[str, tuple[tuple[int, int], ...]], list[str]] = defaultdict(list)
    for sample_id, schedule in schedule_by_sample.items():
        groups[(embryo_by_sample[sample_id], tuple(sorted(schedule)))].append(sample_id)
    pairs: set[tuple[str, str]] = set()
    for (_, schedule), sample_ids in groups.items():
        if not schedule:
            continue
        ordered = sorted(sample_ids)
        for index, sample_a in enumerate(ordered):
            for sample_b in ordered[index + 1 :]:
                pairs.add(pair_key(sample_a, sample_b))
    return pairs


def deterministic_null_pairs(samples: list[Sample], count: int, seed: int) -> list[tuple[str, str]]:
    by_embryo: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        by_embryo[sample.embryo_id].append(sample.sample_id)
    embryo_ids = sorted(by_embryo)
    if len(embryo_ids) != 2:
        raise RuntimeError(f"null-pair construction expects two embryos, got {embryo_ids}")
    all_pairs = [
        (sample_a, sample_b)
        for sample_a in sorted(by_embryo[embryo_ids[0]])
        for sample_b in sorted(by_embryo[embryo_ids[1]])
    ]
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(all_pairs), size=min(count, len(all_pairs)), replace=False)
    return [all_pairs[int(index)] for index in sorted(indices)]


def best_z_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    shift_y: int,
    shift_x: int,
    min_overlap_fraction: float,
) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for shift_z in range(-(moving.shape[0] - 1), reference.shape[0]):
        ncc, overlap = overlap_ncc(
            reference,
            moving,
            (shift_z, shift_y, shift_x),
            min_overlap_fraction,
        )
        if not math.isfinite(ncc):
            continue
        objective = float(ncc * math.sqrt(overlap))
        row: dict[str, float | int] = {
            "shift_z": shift_z,
            "shift_y": shift_y,
            "shift_x": shift_x,
            "ncc": ncc,
            "overlap_fraction": overlap,
            "objective": objective,
        }
        if best is None or float(row["objective"]) > float(best["objective"]):
            best = row
    if best is None:
        return {
            "shift_z": 0,
            "shift_y": shift_y,
            "shift_x": shift_x,
            "ncc": float("nan"),
            "overlap_fraction": 0.0,
            "objective": float("nan"),
        }
    return best


def confirm_volume_pair(
    sample_a: Sample,
    sample_b: Sample,
    registration: dict[str, Any],
    frames: list[int],
    volume_cache: dict[tuple[str, int], np.ndarray],
    min_overlap_fraction: float,
) -> dict[str, Any]:
    transform = str(registration["transform"])
    shift_y = int(round(float(registration["shift_y_median_ds"])))
    shift_x = int(round(float(registration["shift_x_median_ds"])))
    frame_results: list[dict[str, Any]] = []
    for frame in frames:
        reference = volume_cache[(sample_a.sample_id, frame)]
        moving = apply_d4(volume_cache[(sample_b.sample_id, frame)], transform)
        result = best_z_shift(
            reference,
            moving,
            shift_y,
            shift_x,
            min_overlap_fraction,
        )
        result["frame"] = frame
        frame_results.append(result)
    z_shifts = [float(row["shift_z"]) for row in frame_results]
    median_z, mad_z = median_and_mad(z_shifts)
    return {
        "sample_a": sample_a.sample_id,
        "sample_b": sample_b.sample_id,
        "embryo_id": sample_a.embryo_id,
        "transform": transform,
        "shift_y_median_ds": float(registration["shift_y_median_ds"]),
        "shift_x_median_ds": float(registration["shift_x_median_ds"]),
        "shift_mad_ds": float(registration["shift_mad_ds"]),
        "thumbnail_ncc_median": float(registration["thumbnail_ncc_median"]),
        "thumbnail_overlap_median": float(registration["thumbnail_overlap_median"]),
        "same_vs_mismatch_margin": float(registration["same_vs_mismatch_margin"]),
        "volume_ncc_median": float(np.nanmedian([float(row["ncc"]) for row in frame_results])),
        "volume_overlap_median": float(
            np.nanmedian([float(row["overlap_fraction"]) for row in frame_results])
        ),
        "shift_z_median_ds": median_z,
        "shift_z_mad_ds": mad_z,
        "volume_frame_results_json": json.dumps(json_safe(frame_results), sort_keys=True),
    }


def nested_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = nested_value(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = nested_value(child, key)
            if found is not None:
                return found
    return None


def decompress_zstd(payload: bytes, expected_size: int) -> bytes:
    try:
        import zstandard

        return zstandard.ZstdDecompressor().decompress(
            payload,
            max_output_size=expected_size,
        )
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        import pyarrow

        return pyarrow.Codec("zstd").decompress(payload, expected_size)
    except (ImportError, ModuleNotFoundError):
        pass
    zstd_command = shutil.which("zstd")
    if zstd_command is None:
        raise RuntimeError("no zstd decoder is available")
    completed = subprocess.run(
        [zstd_command, "-d", "-c"],
        input=payload,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def read_zarr_v3_array(path: Path) -> np.ndarray:
    metadata = read_json_object(path / "zarr.json")
    shape = tuple(int(value) for value in metadata["shape"])
    dtype = np.dtype(metadata["data_type"]).newbyteorder("<")
    chunk_shape = tuple(
        int(value) for value in metadata["chunk_grid"]["configuration"]["chunk_shape"]
    )
    if chunk_shape != shape:
        raise NotImplementedError(
            f"expected one full-array GEFF chunk: {path}, shape={shape}, chunk={chunk_shape}"
        )
    chunk_path = path / "c" / Path(*("0" for _ in shape))
    expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    decoded = decompress_zstd(chunk_path.read_bytes(), expected_size)
    if len(decoded) != expected_size:
        raise RuntimeError(
            f"GEFF decoded byte mismatch: {path}, expected={expected_size}, actual={len(decoded)}"
        )
    return np.frombuffer(decoded, dtype=dtype).reshape(shape).copy()


def load_geff_arrays(graph_path: Path) -> dict[str, Any]:
    node_ids = read_zarr_v3_array(graph_path / "nodes" / "ids").astype(np.int64)
    properties = {
        axis: read_zarr_v3_array(graph_path / "nodes" / "props" / axis / "values")
        for axis in ("t", "z", "y", "x")
    }
    if not all(len(values) == len(node_ids) for values in properties.values()):
        raise RuntimeError(f"misaligned GEFF node arrays: {graph_path}")
    edges = read_zarr_v3_array(graph_path / "edges" / "ids").astype(np.int64).reshape(-1, 2)
    root_metadata = read_json_object(graph_path / "zarr.json")
    return {
        "node_ids": node_ids,
        "t": np.asarray(properties["t"], dtype=np.int64),
        "zyx": np.column_stack(
            [
                np.asarray(properties["z"], dtype=np.float64),
                np.asarray(properties["y"], dtype=np.float64),
                np.asarray(properties["x"], dtype=np.float64),
            ]
        ),
        "edges": edges,
        "estimated_number_of_nodes": nested_value(root_metadata, "estimated_number_of_nodes"),
    }


def transform_xy_coordinates(
    y: np.ndarray,
    x: np.ndarray,
    transform: str,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    if height != width and transform in {"rot90", "rot270", "transpose", "anti_transpose"}:
        raise ValueError("axis-swapping D4 transforms require square images")
    if transform == "identity":
        return y, x
    if transform == "rot90":
        return width - 1 - x, y
    if transform == "rot180":
        return height - 1 - y, width - 1 - x
    if transform == "rot270":
        return x, height - 1 - y
    if transform == "flip_x":
        return y, width - 1 - x
    if transform == "flip_y":
        return height - 1 - y, x
    if transform == "transpose":
        return x, y
    if transform == "anti_transpose":
        return width - 1 - x, height - 1 - y
    raise ValueError(f"unknown D4 transform: {transform}")


def greedy_spatial_matches(
    nodes_a: dict[str, Any],
    nodes_b_mapped: dict[str, Any],
    scale_zyx_um: np.ndarray,
    max_distance_um: float,
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    shared_times = sorted(set(nodes_a["t"].tolist()) & set(nodes_b_mapped["t"].tolist()))
    for frame in shared_times:
        indices_a = np.flatnonzero(nodes_a["t"] == frame)
        indices_b = np.flatnonzero(nodes_b_mapped["t"] == frame)
        if not len(indices_a) or not len(indices_b):
            continue
        points_a = nodes_a["zyx"][indices_a] * scale_zyx_um
        points_b = nodes_b_mapped["zyx"][indices_b] * scale_zyx_um
        tree = cKDTree(points_a)
        distances, local_a = tree.query(
            points_b,
            k=1,
            distance_upper_bound=max_distance_um,
        )
        for local_b, (distance, index_a) in enumerate(zip(distances, local_a, strict=True)):
            if math.isfinite(float(distance)) and int(index_a) < len(indices_a):
                candidates.append(
                    (float(distance), int(indices_a[int(index_a)]), int(indices_b[local_b]))
                )
    matches: list[tuple[int, int, float]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for distance, index_a, index_b in sorted(candidates):
        if index_a in used_a or index_b in used_b:
            continue
        used_a.add(index_a)
        used_b.add(index_b)
        matches.append((index_a, index_b, distance))
    return matches


def geff_pair_evidence(
    sample_a: Sample,
    sample_b: Sample,
    registration: dict[str, Any],
    volume_confirmation: dict[str, Any],
    thumbnail_downsample_yx: int,
    volume_downsample_z: int,
    scale_zyx_um: np.ndarray,
    max_distance_um: float,
) -> dict[str, Any]:
    nodes_a = load_geff_arrays(sample_a.graph_path)
    nodes_b = load_geff_arrays(sample_b.graph_path)
    mapped_b = np.asarray(nodes_b["zyx"], dtype=np.float64).copy()
    mapped_y, mapped_x = transform_xy_coordinates(
        mapped_b[:, 1],
        mapped_b[:, 2],
        str(registration["transform"]),
        sample_b.shape[2],
        sample_b.shape[3],
    )
    mapped_b[:, 0] += float(volume_confirmation["shift_z_median_ds"]) * volume_downsample_z
    mapped_b[:, 1] = mapped_y + float(registration["shift_y_median_ds"]) * thumbnail_downsample_yx
    mapped_b[:, 2] = mapped_x + float(registration["shift_x_median_ds"]) * thumbnail_downsample_yx
    in_bounds = (
        (mapped_b[:, 0] >= 0)
        & (mapped_b[:, 0] < sample_a.shape[1])
        & (mapped_b[:, 1] >= 0)
        & (mapped_b[:, 1] < sample_a.shape[2])
        & (mapped_b[:, 2] >= 0)
        & (mapped_b[:, 2] < sample_a.shape[3])
    )
    mapped_nodes = {"t": nodes_b["t"], "zyx": mapped_b}
    matches = greedy_spatial_matches(
        nodes_a,
        mapped_nodes,
        scale_zyx_um,
        max_distance_um,
    )
    matched_b_to_a = {
        int(nodes_b["node_ids"][index_b]): int(nodes_a["node_ids"][index_a])
        for index_a, index_b, _ in matches
    }
    edges_a = {tuple(int(value) for value in edge) for edge in nodes_a["edges"]}
    mapped_edges: list[tuple[int, int]] = []
    one_ended_edges = 0
    for source, target in nodes_b["edges"]:
        source_id = int(source)
        target_id = int(target)
        source_matched = source_id in matched_b_to_a
        target_matched = target_id in matched_b_to_a
        if source_matched and target_matched:
            mapped_edges.append((matched_b_to_a[source_id], matched_b_to_a[target_id]))
        elif source_matched != target_matched:
            one_ended_edges += 1
    matching_edges = sum(edge in edges_a for edge in mapped_edges)

    ids_a = {int(value) for value in nodes_a["node_ids"]}
    ids_b = {int(value) for value in nodes_b["node_ids"]}
    shared_ids = ids_a & ids_b
    shared_id_matches = sum(
        1
        for index_a, index_b, _ in matches
        if int(nodes_a["node_ids"][index_a]) == int(nodes_b["node_ids"][index_b])
    )
    distances = [distance for _, _, distance in matches]
    eligible_b = int(np.count_nonzero(in_bounds))
    return {
        "sample_a": sample_a.sample_id,
        "sample_b": sample_b.sample_id,
        "embryo_id": sample_a.embryo_id,
        "transform": registration["transform"],
        "mapped_shift_z_vox": float(volume_confirmation["shift_z_median_ds"]) * volume_downsample_z,
        "mapped_shift_y_vox": float(registration["shift_y_median_ds"]) * thumbnail_downsample_yx,
        "mapped_shift_x_vox": float(registration["shift_x_median_ds"]) * thumbnail_downsample_yx,
        "annotated_nodes_a": len(nodes_a["node_ids"]),
        "annotated_nodes_b": len(nodes_b["node_ids"]),
        "mapped_b_nodes_inside_a": eligible_b,
        "spatial_node_matches": len(matches),
        "spatial_match_fraction_of_mapped_b": len(matches) / max(eligible_b, 1),
        "spatial_match_distance_um_median": (
            float(np.median(distances)) if distances else float("nan")
        ),
        "shared_node_ids": len(shared_ids),
        "shared_id_and_spatial_matches": shared_id_matches,
        "mapped_edges_with_both_nodes_matched": len(mapped_edges),
        "mapped_edges_present_in_a": matching_edges,
        "mapped_edge_agreement": matching_edges / max(len(mapped_edges), 1),
        "one_ended_b_edges_after_node_matching": one_ended_edges,
        "estimated_nodes_a": nodes_a["estimated_number_of_nodes"],
        "estimated_nodes_b": nodes_b["estimated_number_of_nodes"],
    }


def connected_components_from_pairs(
    pairs: list[tuple[str, str]], all_samples: list[str]
) -> list[list[str]]:
    parent = {sample: sample for sample in all_samples}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for sample_a, sample_b in pairs:
        root_a = find(sample_a)
        root_b = find(sample_b)
        if root_a != root_b:
            parent[root_b] = root_a
    groups: dict[str, list[str]] = defaultdict(list)
    for sample in all_samples:
        groups[find(sample)].append(sample)
    return sorted(
        [sorted(group) for group in groups.values() if len(group) > 1],
        key=lambda group: (-len(group), group),
    )


def dynamic_fieldnames(rows: list[dict[str, Any]], fallback: list[str]) -> list[str]:
    fields = list(fallback)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def save_registration_figure(
    path: Path,
    rows: list[dict[str, Any]],
    thumbnails: dict[tuple[str, int], np.ndarray],
    frame: int,
) -> None:
    import matplotlib.pyplot as plt

    selected = rows[: min(6, len(rows))]
    if not selected:
        return
    figure, axes = plt.subplots(len(selected), 3, figsize=(9, 3 * len(selected)), squeeze=False)
    for row_index, row in enumerate(selected):
        reference = thumbnails[(str(row["sample_a"]), frame)]
        moving = apply_d4(thumbnails[(str(row["sample_b"]), frame)], str(row["transform"]))
        shift = (
            int(round(float(row["shift_y_median_ds"]))),
            int(round(float(row["shift_x_median_ds"]))),
        )
        slices_a, slices_b, _ = overlap_slices(reference.shape, moving.shape, shift)
        left = reference[slices_a]
        right = moving[slices_b]
        low = min(float(np.percentile(left, 2)), float(np.percentile(right, 2)))
        high = max(float(np.percentile(left, 98)), float(np.percentile(right, 98)))

        value_range = max(high - low, 1e-8)
        overlay = np.zeros((*left.shape, 3), dtype=np.float32)
        overlay[..., 0] = np.clip((left - low) / value_range, 0.0, 1.0)
        overlay[..., 1] = np.clip((right - low) / value_range, 0.0, 1.0)
        axes[row_index, 0].imshow(left, cmap="gray")
        axes[row_index, 1].imshow(right, cmap="gray")
        axes[row_index, 2].imshow(overlay)
        axes[row_index, 0].set_ylabel(f"{row['sample_a']}\n{row['sample_b']}", fontsize=7)
        axes[row_index, 2].set_title(
            f"{row['transform']} NCC={float(row['thumbnail_ncc_median']):.3f}",
            fontsize=8,
        )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_title("crop A overlap")
    axes[0, 1].set_title("transformed crop B overlap")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


# %% [markdown]
# ## Execute the staged audit


# %%
def main() -> None:
    started = time.monotonic()
    working_root = Path.cwd()
    config_path = working_root / "config.yaml"
    metrics_path = working_root / "metrics.json"
    if not config_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            "config.yaml and metrics.json must be present in the notebook working directory"
        )
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise TypeError("config.yaml must contain a mapping")
    output_dir = working_root / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    competition_root = find_competition_root()
    train_root = competition_root / "train"
    samples = discover_samples(train_root, config)
    sample_by_id = {sample.sample_id: sample for sample in samples}
    print(f"DISCOVERED {len(samples)} samples at {train_root}", flush=True)

    sample_bytes = int(config["audit"]["freeze_signature_bytes"])
    freeze_rows: list[dict[str, Any]] = []
    signature_bundle_by_sample: dict[str, str] = {}
    for index, sample in enumerate(samples, start=1):
        runs, signatures = find_freeze_runs(sample, sample_bytes)
        freeze_rows.extend(runs)
        signature_bundle_by_sample[sample.sample_id] = sha256_json(signatures)
        if index % 20 == 0 or index == len(samples):
            print(
                f"FREEZE_SCAN {index}/{len(samples)} runs={len(freeze_rows)}",
                flush=True,
            )

    inventory_rows = [
        {
            "sample_id": sample.sample_id,
            "embryo_id": sample.embryo_id,
            "crop_id": sample.crop_id,
            "shape_tzyx": "x".join(str(value) for value in sample.shape),
            "chunk_shape_tzyx": "x".join(str(value) for value in sample.chunk_shape),
            "dtype": str(sample.dtype),
            "image_metadata_sha256": sha256_file(sample.image_path / "0" / "zarr.json"),
            "geff_metadata_sha256": sha256_file(sample.graph_path / "zarr.json"),
            "frame_quick_signature_bundle_sha256": signature_bundle_by_sample[sample.sample_id],
        }
        for sample in samples
    ]
    write_csv(
        output_dir / "sample_inventory.csv",
        inventory_rows,
        dynamic_fieldnames(inventory_rows, ["sample_id"]),
    )
    write_csv(
        output_dir / "freeze_runs.csv",
        freeze_rows,
        dynamic_fieldnames(
            freeze_rows,
            [
                "sample_id",
                "embryo_id",
                "start_frame",
                "end_frame",
                "length",
                "quick_signature",
                "byte_verified_first_and_last",
            ],
        ),
    )

    frames = [int(value) for value in config["data"]["thumbnail_frames"]]
    mismatch_offset = int(config["data"]["mismatch_frame_offset"])
    frame_count = int(config["data"]["expected_frames_per_sample"])
    mismatch_frames = [
        frame + mismatch_offset
        if frame + mismatch_offset < frame_count
        else frame - mismatch_offset
        for frame in frames
    ]
    thumbnail_frames = sorted(set(frames + mismatch_frames))
    downsample_yx = int(config["data"]["thumbnail_downsample_yx"])
    thumbnails: dict[tuple[str, int], np.ndarray] = {}
    for index, sample in enumerate(samples, start=1):
        for frame in thumbnail_frames:
            thumbnails[(sample.sample_id, frame)] = make_thumbnail(
                read_image_frame(sample, frame), downsample_yx
            )
        if index % 20 == 0 or index == len(samples):
            print(
                f"THUMBNAILS {index}/{len(samples)} cached={len(thumbnails)}",
                flush=True,
            )
    ffts = {key: np.fft.fft2(value) for key, value in thumbnails.items() if key[1] in frames}

    min_overlap_2d = float(config["audit"]["min_overlap_fraction_2d"])
    samples_by_embryo: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        samples_by_embryo[sample.embryo_id].append(sample)
    phase_rows: list[dict[str, Any]] = []
    for embryo_id, embryo_samples in sorted(samples_by_embryo.items()):
        ordered = sorted(embryo_samples, key=lambda sample: sample.sample_id)
        total_pairs = len(ordered) * (len(ordered) - 1) // 2
        completed = 0
        for left_index, sample_a in enumerate(ordered):
            for sample_b in ordered[left_index + 1 :]:
                phase_rows.append(
                    screen_pair(
                        sample_a,
                        sample_b,
                        frames,
                        thumbnails,
                        ffts,
                        min_overlap_2d,
                    )
                )
                completed += 1
                if completed % 1000 == 0 or completed == total_pairs:
                    print(
                        f"PHASE_SCREEN embryo={embryo_id} {completed}/{total_pairs}",
                        flush=True,
                    )
    phase_rows.sort(key=lambda row: float(row["phase_objective_median"]), reverse=True)
    write_csv(
        output_dir / "phase_screen.csv",
        phase_rows,
        dynamic_fieldnames(phase_rows, ["sample_a", "sample_b"]),
    )

    max_identity = int(config["audit"]["max_identity_refinement_pairs"])
    selected_pairs = select_screen_pairs(
        phase_rows,
        int(config["audit"]["top_phase_pairs_per_sample"]),
        max_identity,
    )
    phase_scores = {
        pair_key(str(row["sample_a"]), str(row["sample_b"])): float(row["phase_objective_median"])
        for row in phase_rows
    }
    freeze_pairs = freeze_candidate_pairs(freeze_rows)
    identical_schedule_pairs = identical_freeze_schedule_pairs(freeze_rows)
    selected_set = set(selected_pairs) | freeze_pairs
    selected_pairs = sorted(
        selected_set,
        key=lambda key: phase_scores.get(key, -math.inf),
        reverse=True,
    )[:max_identity]
    identity_rows: list[dict[str, Any]] = []
    for index, (sample_a_id, sample_b_id) in enumerate(selected_pairs, start=1):
        identity_rows.append(
            refine_pair(
                sample_by_id[sample_a_id],
                sample_by_id[sample_b_id],
                frames,
                mismatch_offset,
                thumbnails,
                ["identity"],
                min_overlap_2d,
            )
        )
        if index % 100 == 0 or index == len(selected_pairs):
            print(f"IDENTITY_REFINE {index}/{len(selected_pairs)}", flush=True)
    identity_rows.sort(key=lambda row: float(row["selection_score"]), reverse=True)

    max_d4 = int(config["audit"]["max_d4_refinement_pairs"])
    score_selected_d4_pairs = [
        pair_key(str(row["sample_a"]), str(row["sample_b"])) for row in identity_rows[:max_d4]
    ]
    d4_pairs = sorted(identical_schedule_pairs)
    d4_pairs.extend(key for key in score_selected_d4_pairs if key not in identical_schedule_pairs)
    d4_pairs = d4_pairs[:max_d4]
    transforms = [str(value) for value in config["audit"]["transforms"]]
    registration_rows: list[dict[str, Any]] = []
    for index, (sample_a_id, sample_b_id) in enumerate(d4_pairs, start=1):
        registration_rows.append(
            refine_pair(
                sample_by_id[sample_a_id],
                sample_by_id[sample_b_id],
                frames,
                mismatch_offset,
                thumbnails,
                transforms,
                min_overlap_2d,
            )
        )
        if index % 20 == 0 or index == len(d4_pairs):
            print(f"D4_REFINE {index}/{len(d4_pairs)}", flush=True)
    registration_rows.sort(key=lambda row: float(row["selection_score"]), reverse=True)

    null_pairs = deterministic_null_pairs(
        samples,
        int(config["audit"]["null_pair_count"]),
        int(config.get("reproducibility", {}).get("seed", 42)),
    )
    null_rows: list[dict[str, Any]] = []
    for index, (sample_a_id, sample_b_id) in enumerate(null_pairs, start=1):
        null_rows.append(
            refine_pair(
                sample_by_id[sample_a_id],
                sample_by_id[sample_b_id],
                frames,
                mismatch_offset,
                thumbnails,
                transforms,
                min_overlap_2d,
            )
        )
        if index % 20 == 0 or index == len(null_pairs):
            print(f"NULL_REFINE {index}/{len(null_pairs)}", flush=True)
    null_rows.sort(key=lambda row: float(row["selection_score"]), reverse=True)

    thresholds = config["audit"]["evidence_thresholds"]
    null_quantile = float(thresholds["null_quantile"])
    null_ncc_threshold = float(
        np.quantile(
            [float(row["thumbnail_ncc_median"]) for row in null_rows],
            null_quantile,
        )
    )
    thumbnail_threshold = max(float(thresholds["min_thumbnail_ncc"]), null_ncc_threshold)
    for rank, row in enumerate(registration_rows, start=1):
        row["rank"] = rank
        row["identical_repeat_schedule"] = (
            pair_key(str(row["sample_a"]), str(row["sample_b"])) in identical_schedule_pairs
        )
        row["passes_2d_evidence"] = bool(
            float(row["thumbnail_ncc_median"]) >= thumbnail_threshold
            and float(row["same_vs_mismatch_margin"])
            >= float(thresholds["min_same_vs_mismatch_margin"])
            and float(row["shift_mad_ds"]) <= float(thresholds["max_shift_mad_downsampled_px"])
        )
    write_csv(
        output_dir / "registration_pairs.csv",
        registration_rows,
        dynamic_fieldnames(registration_rows, ["sample_a", "sample_b"]),
    )
    write_csv(
        output_dir / "null_pairs.csv",
        null_rows,
        dynamic_fieldnames(null_rows, ["sample_a", "sample_b"]),
    )

    max_volume = int(config["audit"]["max_volume_confirmation_pairs"])
    volume_top_score_pairs = int(config["audit"]["volume_top_score_pairs"])
    repeat_schedule_candidates = [
        row for row in registration_rows if bool(row["identical_repeat_schedule"])
    ]
    volume_candidate_keys = {
        pair_key(str(row["sample_a"]), str(row["sample_b"])) for row in repeat_schedule_candidates
    }
    volume_candidates = list(repeat_schedule_candidates)
    for row in registration_rows[:volume_top_score_pairs]:
        key = pair_key(str(row["sample_a"]), str(row["sample_b"]))
        if key not in volume_candidate_keys:
            volume_candidates.append(row)
            volume_candidate_keys.add(key)
    volume_candidates = volume_candidates[:max_volume]
    volume_frames = [int(value) for value in config["data"]["volume_frames"]]
    volume_factors = tuple(int(value) for value in config["data"]["volume_downsample_zyx"])
    volume_sample_ids = sorted(
        {str(row[key]) for row in volume_candidates for key in ("sample_a", "sample_b")}
    )
    volume_cache: dict[tuple[str, int], np.ndarray] = {}
    for index, sample_id in enumerate(volume_sample_ids, start=1):
        sample = sample_by_id[sample_id]
        for frame in volume_frames:
            volume_cache[(sample_id, frame)] = make_lowres_volume(
                read_image_frame(sample, frame), volume_factors
            )
        if index % 10 == 0 or index == len(volume_sample_ids):
            print(f"VOLUMES {index}/{len(volume_sample_ids)}", flush=True)

    min_overlap_3d = float(config["audit"]["min_overlap_fraction_3d"])
    volume_rows: list[dict[str, Any]] = []
    for index, registration in enumerate(volume_candidates, start=1):
        sample_a = sample_by_id[str(registration["sample_a"])]
        sample_b = sample_by_id[str(registration["sample_b"])]
        row = confirm_volume_pair(
            sample_a,
            sample_b,
            registration,
            volume_frames,
            volume_cache,
            min_overlap_3d,
        )
        row["passes_2d_evidence"] = bool(registration["passes_2d_evidence"])
        row["identical_repeat_schedule"] = bool(registration["identical_repeat_schedule"])
        row["passes_registration_evidence"] = bool(
            row["passes_2d_evidence"]
            and float(row["volume_ncc_median"]) >= float(thresholds["min_volume_ncc"])
        )
        volume_rows.append(row)
        print(f"VOLUME_CONFIRM {index}/{len(volume_candidates)}", flush=True)
    volume_rows.sort(
        key=lambda row: (
            bool(row["passes_registration_evidence"]),
            float(row["volume_ncc_median"]),
            float(row["thumbnail_ncc_median"]),
        ),
        reverse=True,
    )
    write_csv(
        output_dir / "volume_confirmation.csv",
        volume_rows,
        dynamic_fieldnames(volume_rows, ["sample_a", "sample_b"]),
    )

    registration_lookup = {
        pair_key(str(row["sample_a"]), str(row["sample_b"])): row for row in registration_rows
    }
    annotation_config = config["audit"]["annotation"]
    scale_zyx_um = np.asarray(config["data"]["voxel_scale_zyx_um"], dtype=np.float64)
    geff_rows: list[dict[str, Any]] = []
    for index, volume_row in enumerate(volume_rows, start=1):
        key = pair_key(str(volume_row["sample_a"]), str(volume_row["sample_b"]))
        registration = registration_lookup[key]
        sample_a = sample_by_id[str(volume_row["sample_a"])]
        sample_b = sample_by_id[str(volume_row["sample_b"])]
        geff_row = geff_pair_evidence(
            sample_a,
            sample_b,
            registration,
            volume_row,
            downsample_yx,
            volume_factors[0],
            scale_zyx_um,
            float(annotation_config["max_match_distance_um"]),
        )
        geff_row["passes_registration_evidence"] = bool(volume_row["passes_registration_evidence"])
        geff_rows.append(geff_row)
        print(f"GEFF_CONFIRM {index}/{len(volume_rows)}", flush=True)
    geff_rows.sort(
        key=lambda row: (
            bool(row["passes_registration_evidence"]),
            int(row["spatial_node_matches"]),
        ),
        reverse=True,
    )
    write_csv(
        output_dir / "geff_pair_evidence.csv",
        geff_rows,
        dynamic_fieldnames(geff_rows, ["sample_a", "sample_b"]),
    )

    save_registration_figure(
        output_dir / "top_registration_examples.png",
        volume_rows,
        thumbnails,
        frame=frames[len(frames) // 2],
    )
    repeat_schedule_volume_rows = sorted(
        [row for row in volume_rows if bool(row["identical_repeat_schedule"])],
        key=lambda row: (
            float(row["volume_ncc_median"]),
            float(row["thumbnail_ncc_median"]),
        ),
        reverse=True,
    )
    save_registration_figure(
        output_dir / "repeat_schedule_registration_examples.png",
        repeat_schedule_volume_rows,
        thumbnails,
        frame=frames[len(frames) // 2],
    )

    positive_pairs = [
        pair_key(str(row["sample_a"]), str(row["sample_b"]))
        for row in volume_rows
        if row["passes_registration_evidence"]
    ]
    freeze_groups: dict[str, list[str]] = defaultdict(list)
    for row in freeze_rows:
        key = f"{row['embryo_id']}:{row['start_frame']}-{row['end_frame']}"
        freeze_groups[key].append(str(row["sample_id"]))
    shared_freeze_groups = {
        key: sorted(set(values))
        for key, values in sorted(freeze_groups.items())
        if len(set(values)) > 1
    }
    components = connected_components_from_pairs(
        positive_pairs, [sample.sample_id for sample in samples]
    )
    summary = {
        "experiment": EXPERIMENT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "sample_count": len(samples),
        "embryo_counts": {
            embryo: len(group) for embryo, group in sorted(samples_by_embryo.items())
        },
        "freeze": {
            "run_count": len(freeze_rows),
            "samples_with_runs": len({str(row["sample_id"]) for row in freeze_rows}),
            "shared_schedule_groups": shared_freeze_groups,
        },
        "registration": {
            "same_embryo_pairs_screened": len(phase_rows),
            "identity_pairs_refined": len(identity_rows),
            "d4_pairs_refined": len(registration_rows),
            "identical_repeat_schedule_pairs": len(identical_schedule_pairs),
            "identical_repeat_schedule_pairs_d4_refined": sum(
                bool(row["identical_repeat_schedule"]) for row in registration_rows
            ),
            "cross_embryo_null_pairs": len(null_rows),
            "volume_pairs_confirmed": len(volume_rows),
            "thumbnail_ncc_fixed_threshold": float(thresholds["min_thumbnail_ncc"]),
            "thumbnail_ncc_null_quantile": null_quantile,
            "thumbnail_ncc_null_threshold": null_ncc_threshold,
            "thumbnail_ncc_effective_threshold": thumbnail_threshold,
            "passes_2d_count": sum(bool(row["passes_2d_evidence"]) for row in registration_rows),
            "passes_2d_and_3d_count": len(positive_pairs),
            "identical_repeat_schedule_pairs_passing_2d_and_3d": sum(
                bool(row["identical_repeat_schedule"]) and bool(row["passes_registration_evidence"])
                for row in volume_rows
            ),
            "positive_components": components,
            "top_pairs": [
                {
                    key: row[key]
                    for key in (
                        "sample_a",
                        "sample_b",
                        "transform",
                        "thumbnail_ncc_median",
                        "volume_ncc_median",
                        "same_vs_mismatch_margin",
                        "shift_mad_ds",
                        "passes_registration_evidence",
                    )
                }
                for row in volume_rows[:10]
            ],
        },
        "geff": {
            "pairs_checked": len(geff_rows),
            "positive_image_pairs_with_spatial_annotation_matches": sum(
                bool(row["passes_registration_evidence"]) and int(row["spatial_node_matches"]) > 0
                for row in geff_rows
            ),
            "positive_image_pairs_with_edge_agreement": sum(
                bool(row["passes_registration_evidence"])
                and int(row["mapped_edges_present_in_a"]) > 0
                for row in geff_rows
            ),
        },
        "runtime_seconds": time.monotonic() - started,
        "interpretation_limits": [
            (
                "A positive pair supports a shared field of view, not a unique "
                "embryo-wide coordinate system."
            ),
            (
                "A negative result does not rule out non-overlapping adjacency, "
                "arbitrary time offsets, or non-rigid transforms."
            ),
            "Sparse GEFF annotations can yield zero matches even when image registration is valid.",
            (
                "One-ended annotated edges are reported descriptively and are not "
                "proven cross-crop continuations."
            ),
        ],
    }
    atomic_json(output_dir / "audit_summary.json", summary)

    artifact_rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifact_rows.append(
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    artifact_manifest = {
        "experiment": EXPERIMENT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input_manifest_sha256": sha256_json(inventory_rows),
        "files": artifact_rows,
    }
    atomic_json(output_dir / "artifact_manifest.json", artifact_manifest)
    artifact_manifest_sha = sha256_file(output_dir / "artifact_manifest.json")

    metrics = read_json_object(metrics_path)
    metrics["status"] = "debug_completed"
    metrics["metric"] = "cross_crop_registration_evidence"
    evidence = metrics.setdefault("evidence", {})
    kaggle_evidence = evidence.setdefault("kaggle", {})
    kaggle_evidence["resource"] = "cpu"
    kaggle_evidence["internet_enabled"] = False
    artifact_evidence = evidence.setdefault("artifacts", {})
    artifact_evidence.update(
        {
            "input_file_sha": artifact_manifest["input_manifest_sha256"],
            "feature_schema_sha": sha256_json(
                {
                    path.name: path.open().readline().strip().split(",")
                    for path in sorted(output_dir.glob("*.csv"))
                }
            ),
            "feature_content_sha": artifact_manifest_sha,
            "row_count": len(samples),
            "group_count": len(samples_by_embryo),
            "feature_count": len(artifact_rows),
            "selected_mode": "cpu_multistage_registration_audit",
            "artifact_manifest_sha": artifact_manifest_sha,
        }
    )
    evidence["registration_audit"] = summary
    metrics["notes"] = (
        "Kaggle CPU audit finished. This diagnostic does not train a model, produce "
        "predictions, or create a competition submission."
    )
    atomic_json(metrics_path, metrics)
    print("AUDIT_SUMMARY", json.dumps(json_safe(summary), sort_keys=True), flush=True)
    print(f"ARTIFACTS {output_dir}", flush=True)


if __name__ == "__main__":
    main()
