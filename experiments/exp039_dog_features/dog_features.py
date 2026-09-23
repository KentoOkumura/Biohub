"""Raw-image Difference-of-Gaussians responses for fixed exp015 candidates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

SIDES = ("src", "tgt")
T = TypeVar("T")


def process_frames_parallel(
    frames: list[int], process_one: Callable[[int], T], workers: int
) -> list[T]:
    """Process independent frames concurrently while returning results in frame order."""
    if workers < 1:
        raise ValueError("parallel workers must be positive")
    if len(frames) != len(set(frames)):
        raise ValueError("parallel frame batch contains duplicate frames")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(process_one, frames))


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def content_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(
            json.dumps(
                {"name": name, "dtype": str(array.dtype), "shape": list(array.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def normalize_frame(frame: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    image = np.asarray(frame, dtype=np.float32)
    if image.ndim != 3 or not image.size or not np.isfinite(image).all():
        raise ValueError("expected a nonempty finite z/y/x image")
    low, high = np.percentile(image, percentiles)
    if high <= low:
        high = low + 1.0
    return np.maximum((image - low) / (high - low), 0.0).astype(np.float32)


def dog_volumes(
    frame: np.ndarray, cfg: dict[str, Any], *, already_normalized: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    spacing = np.asarray(cfg["spatial_spacing_um"], dtype=np.float64)
    pairs = np.asarray(cfg["gaussian_pairs_um"], dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spatial spacing must be three positive finite micrometer values")
    if pairs.shape != (2, 2) or not np.isfinite(pairs).all() or np.any(pairs[:, 0] >= pairs[:, 1]):
        raise ValueError("expected two ordered finite Gaussian pairs")
    image = (
        np.asarray(frame, dtype=np.float32)
        if already_normalized
        else normalize_frame(frame, tuple(cfg["frame_percentiles"]))
    )
    responses = []
    for small, large in pairs:
        small_blur = gaussian_filter(
            image,
            sigma=small / spacing,
            order=0,
            mode=str(cfg["gaussian_mode"]),
            truncate=float(cfg["gaussian_truncate"]),
            output=np.float32,
        )
        large_blur = gaussian_filter(
            image,
            sigma=large / spacing,
            order=0,
            mode=str(cfg["gaussian_mode"]),
            truncate=float(cfg["gaussian_truncate"]),
            output=np.float32,
        )
        responses.append(np.ascontiguousarray(small_blur - large_blur))
    return responses[0], responses[1]


def sample_candidates(
    responses: tuple[np.ndarray, np.ndarray],
    coords_physical: np.ndarray,
    coords_grid: np.ndarray,
    *,
    spacing_um: tuple[float, float, float],
    downsample_zyx: tuple[float, float, float],
    tolerance_voxel: float,
) -> np.ndarray:
    physical = np.asarray(coords_physical, dtype=np.float64)
    grid = np.asarray(coords_grid, dtype=np.float64)
    if physical.shape != grid.shape or physical.ndim != 2 or physical.shape[1] != 3:
        raise ValueError("candidate physical and grid coordinates must have shape [N,3]")
    if not np.isfinite(physical).all() or not np.isfinite(grid).all():
        raise ValueError("candidate coordinates are nonfinite")
    if len(physical) == 0:
        return np.empty((0, 2), dtype=np.float32)
    voxel = physical / np.asarray(spacing_um)
    if not np.allclose(
        voxel,
        grid * np.asarray(downsample_zyx),
        rtol=0,
        atol=tolerance_voxel,
    ):
        raise ValueError("candidate physical and grid coordinates disagree")
    shape = np.asarray(responses[0].shape, dtype=np.float64)
    if any(response.shape != tuple(shape.astype(int)) for response in responses):
        raise ValueError("DoG response volumes have different shapes")
    if np.any(voxel < 0) or np.any(voxel > shape - 1):
        raise ValueError("candidate lies outside its original image frame")
    sampled = np.column_stack(
        [
            map_coordinates(response, voxel.T, order=1, mode="nearest", prefilter=False)
            for response in responses
        ]
    ).astype(np.float32)
    if not np.isfinite(sampled).all():
        raise ValueError("sampled DoG features are nonfinite")
    return np.ascontiguousarray(sampled)


def read_cache_role(path: Path, side: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if side not in SIDES:
        raise ValueError(side)
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(saved["__metadata_json__"].tobytes().decode("utf-8"))
        names = {
            "candidate_ids": f"candidate_ids_{side}",
            "coords_grid": f"coords_{side}_grid",
            "coords_physical": f"coords_{side}_physical",
            "candidate_mask": f"candidate_mask_{side}",
        }
        arrays = {key: np.ascontiguousarray(saved[name]) for key, name in names.items()}
    if metadata.get("experiment") != "exp015_oracle_stage_limits":
        raise ValueError(f"wrong cache source: {path}")
    if metadata.get("dataset") != path.parent.name:
        raise ValueError(f"cache dataset mismatch: {path}")
    frames = metadata.get("window_frames")
    if frames != [int(v) for v in path.stem.split("_")]:
        raise ValueError(f"cache frame mismatch: {path}")
    count = len(arrays["candidate_ids"])
    if arrays["coords_grid"].shape != (count, 3) or arrays["coords_physical"].shape != (count, 3):
        raise ValueError(f"candidate coordinate shape mismatch: {path}")
    if arrays["candidate_mask"].shape != (count,) or not arrays["candidate_mask"].all():
        raise ValueError(f"candidate mask mismatch: {path}")
    recorded_sha = metadata.get("array_content_sha256")
    if not isinstance(recorded_sha, str) or len(recorded_sha) != 64:
        raise ValueError(f"cache has no content SHA: {path}")
    return arrays, metadata


def frame_feature_payload(
    frame: np.ndarray,
    requests: dict[str, tuple[Path, dict[str, np.ndarray], dict[str, Any]]],
    cfg: dict[str, Any],
    *,
    downsample_zyx: tuple[float, float, float],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    image = normalize_frame(frame, tuple(cfg["frame_percentiles"]))
    responses = dog_volumes(image, cfg, already_normalized=True)
    output: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    for side, (path, arrays, metadata) in sorted(requests.items()):
        sampled = sample_candidates(
            responses,
            arrays["coords_physical"],
            arrays["coords_grid"],
            spacing_um=tuple(cfg["spatial_spacing_um"]),
            downsample_zyx=downsample_zyx,
            tolerance_voxel=float(cfg["coordinate_tolerance_voxel"]),
        )
        output[f"{side}_candidate_ids"] = arrays["candidate_ids"]
        output[f"{side}_coords_grid"] = arrays["coords_grid"]
        output[f"{side}_coords_physical"] = arrays["coords_physical"]
        output[f"{side}_candidate_mask"] = arrays["candidate_mask"]
        output[f"{side}_responses"] = sampled
        voxel = arrays["coords_physical"] / np.asarray(cfg["spatial_spacing_um"])
        output[f"{side}_intensity"] = map_coordinates(
            image, voxel.T, order=1, mode="nearest", prefilter=False
        ).astype(np.float32)
        max_sigma = max(max(pair) for pair in cfg["gaussian_pairs_um"])
        margin = float(cfg["gaussian_truncate"]) * max_sigma / np.asarray(cfg["spatial_spacing_um"])
        output[f"{side}_boundary"] = np.any(
            (voxel < margin) | (voxel > np.asarray(image.shape) - 1 - margin), axis=1
        ).astype(np.bool_)
        records[side] = {
            "window_frames": metadata["window_frames"],
            "window_name": path.name,
            "cache_content_sha256": metadata["array_content_sha256"],
            "candidate_count": len(sampled),
        }
    if not records:
        raise ValueError("frame has no adjacent cache window")
    return output, records


def save_frame_features(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    unsigned = {**metadata, "array_content_sha256": content_sha256(arrays)}
    payload = {
        **arrays,
        "__metadata_json__": np.frombuffer(
            json.dumps(unsigned, sort_keys=True).encode(), dtype=np.uint8
        ),
    }
    np.savez_compressed(path, **payload)
    return unsigned["array_content_sha256"]


def load_window_side_features(
    path: Path, cache_path: Path, side: str, *, with_aux: bool = False
) -> np.ndarray | dict[str, np.ndarray]:
    if side not in SIDES:
        raise ValueError(side)
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(saved["__metadata_json__"].tobytes().decode("utf-8"))
        arrays = {
            name: np.ascontiguousarray(saved[name])
            for name in saved.files
            if name != "__metadata_json__"
        }
    if content_sha256(arrays) != metadata.get("array_content_sha256"):
        raise ValueError(f"DoG feature content SHA mismatch: {path}")
    if metadata.get("dataset") != cache_path.parent.name:
        raise ValueError(f"DoG dataset mismatch: {path}")
    role_record = metadata.get("roles", {}).get(side)
    if not role_record or role_record.get("window_name") != cache_path.name:
        raise ValueError(f"DoG frame/window role mismatch: {path} {cache_path} {side}")
    original, cache_meta = read_cache_role(cache_path, side)
    if role_record["cache_content_sha256"] != cache_meta["array_content_sha256"]:
        raise ValueError(f"DoG feature references wrong cache: {path}")
    for name in ("candidate_ids", "coords_grid", "coords_physical", "candidate_mask"):
        if not np.array_equal(arrays[f"{side}_{name}"], original[name]):
            raise ValueError(f"DoG candidate {name} mismatch: {path}")
    values = arrays[f"{side}_responses"]
    if values.shape != (len(original["candidate_ids"]), 2) or not np.isfinite(values).all():
        raise ValueError(f"invalid DoG response array: {path}")
    if not with_aux:
        return values
    intensity = arrays[f"{side}_intensity"]
    boundary = arrays[f"{side}_boundary"]
    if intensity.shape != (len(values),) or boundary.shape != (len(values),):
        raise ValueError(f"invalid DoG auxiliary array: {path}")
    if not np.isfinite(intensity).all() or boundary.dtype != np.bool_:
        raise ValueError(f"invalid DoG intensity/boundary: {path}")
    return {"responses": values, "intensity": intensity, "boundary": boundary}


def fit_dog_normalizer(
    cache_paths: list[Path], feature_root: Path, standard_deviation_floor: float
) -> dict[str, Any]:
    count = 0
    response_chunks: list[np.ndarray] = []
    intensity_chunks: list[np.ndarray] = []
    total = np.zeros(2, dtype=np.float64)
    total_squared = np.zeros(2, dtype=np.float64)
    for cache_path in sorted(cache_paths):
        src, tgt = map(int, cache_path.stem.split("_"))
        for side, frame in (("src", src), ("tgt", tgt)):
            feature = load_window_side_features(
                feature_root / cache_path.parent.name / f"{frame:06d}.npz",
                cache_path,
                side,
                with_aux=True,
            )
            values = feature["responses"].astype(np.float64)
            response_chunks.append(feature["responses"])
            intensity_chunks.append(feature["intensity"])
            count += len(values)
            total += values.sum(axis=0)
            total_squared += np.square(values).sum(axis=0)
    if not count:
        raise ValueError("cannot fit DoG normalizer without candidates")
    mean = total / count
    std = np.sqrt(np.maximum(total_squared / count - np.square(mean), 0.0))
    response_values = np.concatenate(response_chunks)
    intensity_values = np.concatenate(intensity_chunks)
    return {
        "count": count,
        "response_quartiles": np.percentile(response_values, [25, 50, 75], axis=0).T.tolist(),
        "intensity_quartiles": np.percentile(intensity_values, [25, 50, 75]).tolist(),
        "mean": mean.tolist(),
        "observed_standard_deviation": std.tolist(),
        "standard_deviation": np.maximum(std, standard_deviation_floor).tolist(),
        "scope": "gradient_update_candidates_only",
    }


def standardize_dog(values: np.ndarray, normalizer: dict[str, Any], use_values: bool) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("DoG input must have two finite columns")
    if not use_values:
        return np.zeros_like(values, dtype=np.float32)
    result = (values.astype(np.float64) - np.asarray(normalizer["mean"])) / np.asarray(
        normalizer["standard_deviation"]
    )
    if not np.isfinite(result).all():
        raise ValueError("standardized DoG is nonfinite")
    return result.astype(np.float32)


def expand_public_tracker_projection(state: dict[str, Any], extra_channels: int) -> dict[str, Any]:
    """Copy the public tracker and initialize only the added input columns to zero."""
    import torch

    if extra_channels != 2 or "proj.weight" not in state:
        raise ValueError("expected two DoG channels and public projection weights")
    weight = state["proj.weight"]
    if weight.ndim != 2 or weight.shape[1] != 64:
        raise ValueError("public projection must have 64 input channels")
    expanded = {name: value.detach().clone() for name, value in state.items()}
    expanded["proj.weight"] = torch.cat(
        (weight.detach().clone(), torch.zeros_like(weight[:, :extra_channels])), dim=1
    )
    return expanded
