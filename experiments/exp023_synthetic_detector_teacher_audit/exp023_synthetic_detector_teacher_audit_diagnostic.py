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
# # Synthetic lineage labels on fixed public detector candidates
#
# This GPU diagnostic runs only the frozen public detector and image encoders
# on two synthetic sequences. Synthetic GT is used after detection for audit.
# No tracker is trained or scored, and no submission is produced.
#
# ## Contents
# 1. Imports and input integrity
# 2. Graph validation and GT correspondence
# 3. Frozen detector and two-frame feature extraction
# 4. Triplet teacher classification
# 5. Run and save diagnostic evidence

# %%
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree, distance

EXPERIMENT = "exp023_synthetic_detector_teacher_audit"
COUNT_FIELDS = (
    "gt_nodes_src",
    "gt_nodes_tgt",
    "gt_divisions",
    "gt_geometric_divisions",
    "detected_src",
    "detected_tgt",
    "matched_src",
    "matched_tgt",
    "matched_division_mothers",
    "matched_all_division_centers",
    "detected_divisions",
    "positive_with_wrong_mother",
    "positive_triplets",
    "wrong_division_triplets",
    "continuation_triplets",
    "unmatched_triplets",
    "all_triplets",
    "max_triplets_per_mother",
)
FEATURE_SCHEMA = {
    "coords_grid": "int16[N,3]",
    "coords_physical": "float32[N,3]",
    "detection_score": "float32[N]",
    "primary_features": "float32[N,32]; 8-view inverse-D4 average",
    "secondary_features": "float32[N,32]; original view",
    "matched_gt_row": "int32[N]; -1 means unmatched",
    "matched_triplet_ids": "int32[M,3]; source index, target indices",
    "matched_triplet_class": "int8[M]; 1 correct division, 2 wrong division, 3 continuation",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def config_path() -> Path:
    candidates = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "experiments" / EXPERIMENT / "config.yaml",
    ]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise RuntimeError(f"expected one config.yaml: {found}")
    return found[0]


def synthetic_source(config: dict[str, Any]) -> Path:
    root = Path("/kaggle/input/notebooks")
    slug = config["data"]["kernel_source"].split("/", 1)[1]
    matches = [
        path.parent
        for path in root.rglob("manifest.json")
        if path.parent.name == config["data"]["output_directory"] and slug in str(path)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one synthetic source: {matches}")
    return matches[0]


def artifact_root(slug: str) -> Path:
    roots = [Path("/kaggle/input/datasets/pilkwang") / slug, Path("/kaggle/input") / slug]
    roots += [
        path.parent
        for path in Path("/kaggle/input").rglob("ARTIFACT_MANIFEST.json")
        if slug in str(path)
    ]
    found = sorted(
        {path.resolve() for path in roots if (path / "ARTIFACT_MANIFEST.json").is_file()}
    )
    if len(found) != 1:
        raise RuntimeError(f"expected one model artifact {slug}: {found}")
    return found[0]


def validate_graph(nodes: np.ndarray, edges: np.ndarray, divisions: np.ndarray, t_count: int):
    if nodes.dtype != np.float32 or nodes.ndim != 2 or nodes.shape[1] != 5:
        raise ValueError("nodes must be float32[N,5]")
    if edges.dtype != np.int32 or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must be int32[E,2]")
    if divisions.dtype != np.int32 or divisions.ndim != 1:
        raise ValueError("divisions must be int32[D]")
    if not np.isfinite(nodes).all() or not np.array_equal(nodes[:, 0], nodes[:, 0].astype(int)):
        raise ValueError("invalid node values")
    if set(nodes[:, 0].astype(int)) != set(range(t_count)):
        raise ValueError("missing or extra frame")
    if np.any(edges < 0) or np.any(edges >= len(nodes)):
        raise ValueError("edge out of bounds")
    if not (nodes[edges[:, 1], 0] == nodes[edges[:, 0], 0] + 1).all():
        raise ValueError("nonadjacent edge")
    outgoing = np.bincount(edges[:, 0], minlength=len(nodes))
    incoming = np.bincount(edges[:, 1], minlength=len(nodes))
    if np.any(outgoing[nodes[:, 0] == t_count - 1]):
        raise ValueError("final frame has outgoing edge")
    if not np.isin(outgoing[nodes[:, 0] < t_count - 1], [1, 2]).all():
        raise ValueError("interior output degree must be 1 or 2")
    if np.any(incoming[nodes[:, 0] == 0]) or not (incoming[nodes[:, 0] > 0] == 1).all():
        raise ValueError("incoming degree must be 1 after first frame")
    if not np.array_equal(np.sort(divisions), np.flatnonzero(outgoing == 2)):
        raise ValueError("division list does not match output degree")
    children: list[list[int]] = [[] for _ in range(len(nodes))]
    for parent, child in edges:
        children[int(parent)].append(int(child))
    return children


def match_candidates(
    candidate_um: np.ndarray, gt_um: np.ndarray, gt_ids: np.ndarray, max_um: float
) -> np.ndarray:
    mapped = np.full(len(candidate_um), -1, dtype=np.int32)
    if not len(candidate_um) or not len(gt_um):
        return mapped
    costs = distance.cdist(candidate_um, gt_um)
    gated = np.where(costs <= max_um, costs, 1e6)
    rows, cols = linear_sum_assignment(gated)
    for row, col in zip(rows, cols, strict=True):
        if costs[row, col] <= max_um:
            mapped[row] = int(gt_ids[col])
    return mapped


def classify_window_triplets(
    src_um: np.ndarray,
    tgt_um: np.ndarray,
    src_match: np.ndarray,
    tgt_match: np.ndarray,
    children: list[list[int]],
    radius: float,
    sister_radius: float,
    mother_guard: int,
    window_guard: int,
) -> tuple[Counter, np.ndarray, np.ndarray]:
    counts: Counter[str] = Counter()
    triplet_ids = []
    triplet_class = []
    tree = cKDTree(tgt_um) if len(tgt_um) else None
    nearby = tree.query_ball_point(src_um, radius) if tree is not None else [[] for _ in src_um]
    for mother, local_ids in enumerate(nearby):
        candidates = sorted(int(index) for index in local_ids)
        triplets = positive = wrong = 0
        for first, second in combinations(candidates, 2):
            if np.linalg.norm(tgt_um[first] - tgt_um[second]) > sister_radius:
                continue
            triplets += 1
            counts["all_triplets"] += 1
            if triplets > mother_guard or counts["all_triplets"] > window_guard:
                raise RuntimeError("triplet guard exceeded")
            gt_mother, gt_first, gt_second = (
                int(src_match[mother]),
                int(tgt_match[first]),
                int(tgt_match[second]),
            )
            if min(gt_mother, gt_first, gt_second) < 0:
                counts["unmatched_triplets"] += 1
                continue
            gt_children = children[gt_mother]
            if len(gt_children) == 2:
                if {gt_first, gt_second} == set(gt_children):
                    label = 1
                    positive += 1
                    counts["positive_triplets"] += 1
                else:
                    label = 2
                    wrong += 1
                    counts["wrong_division_triplets"] += 1
            else:
                label = 3
                counts["continuation_triplets"] += 1
            triplet_ids.append((mother, first, second))
            triplet_class.append(label)
        counts["max_triplets_per_mother"] = max(counts["max_triplets_per_mother"], triplets)
        counts["detected_divisions"] += int(positive == 1)
        counts["positive_with_wrong_mother"] += int(positive == 1 and wrong > 0)
    if counts["all_triplets"] != sum(
        counts[k]
        for k in (
            "positive_triplets",
            "wrong_division_triplets",
            "continuation_triplets",
            "unmatched_triplets",
        )
    ):
        raise AssertionError("triplet partition mismatch")
    ids = np.asarray(triplet_ids, dtype=np.int32).reshape(-1, 3)
    labels = np.asarray(triplet_class, dtype=np.int8)
    return counts, ids, labels


def normalize_synthetic_volumes(
    volumes: np.ndarray, low: float, high: float
) -> tuple[np.ndarray, float, float]:
    q_low, q_high = np.quantile(volumes, [low, high])
    if q_high <= q_low:
        raise ValueError("invalid synthetic image quantiles")
    normalized = np.maximum(
        (volumes.astype(np.float32) - q_low) / (q_high - q_low + 1e-6), 0
    ).astype(np.float32)
    return normalized, float(q_low), float(q_high)


def gt_window_counts(
    gt_um: np.ndarray,
    nodes: np.ndarray,
    children: list[list[int]],
    t: int,
    src_match: np.ndarray,
    tgt_match: np.ndarray,
    mother_radius: float,
    sister_radius: float,
) -> Counter:
    counts: Counter[str] = Counter()
    matched_src = set(int(value) for value in src_match if value >= 0)
    matched_tgt = set(int(value) for value in tgt_match if value >= 0)
    for parent in np.flatnonzero(nodes[:, 0] == t):
        daughters = children[int(parent)]
        if len(daughters) != 2:
            continue
        counts["gt_divisions"] += 1
        geometric = (
            all(
                np.linalg.norm(gt_um[parent] - gt_um[child]) <= mother_radius for child in daughters
            )
            and np.linalg.norm(gt_um[daughters[0]] - gt_um[daughters[1]]) <= sister_radius
        )
        counts["gt_geometric_divisions"] += int(geometric)
        counts["matched_division_mothers"] += int(int(parent) in matched_src)
        counts["matched_all_division_centers"] += int(
            int(parent) in matched_src and all(child in matched_tgt for child in daughters)
        )
    return counts


# %% [markdown]
# ## Frozen public detector and two-frame feature extraction
#
# The selected model inputs a two-frame 64^3 pooled volume. The synthetic
# source lacks Zarr quantile attributes; the same quantile definition is
# computed over its six frames without consulting GT.


# %%
def load_detector(root: Path, expected_sha: str, config: dict[str, Any], device):
    import torch
    from torch import nn

    source_root = root / "repo" / "src"
    source_path = source_root / "biohub_tracking" / "models" / "temporal_unet.py"
    if (
        not source_path.is_file()
        or sha256_file(source_path) != config["model"]["temporal_unet_source_sha256"]
    ):
        raise ValueError("selected TemporalUNet3D source SHA mismatch")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from biohub_tracking.models.temporal_unet import TemporalUNet3D

    weights = root / config["model"]["checkpoint_relative"]
    if sha256_file(weights) != expected_sha:
        raise ValueError(f"checkpoint SHA mismatch: {weights}")
    model_config = json.loads((weights.parent / "config.json").read_text())
    params = config["model"]["params"]
    if (
        model_config["window_size"] != params["window_size"]
        or list(model_config["downsample"]) != params["downsample_zyx"]
    ):
        raise ValueError("checkpoint model config mismatch")
    channels = int(model_config["unet_out_channels"])
    if channels != config["validation"]["feature_channels_per_branch"]:
        raise ValueError("unexpected feature channels")
    unet = TemporalUNet3D(in_channels=1, out_channels=channels, layers=model_config["unet_layers"])
    head = nn.Conv3d(channels, 1, kernel_size=1)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    unet.load_state_dict(
        {
            key.removeprefix("unet."): value
            for key, value in state.items()
            if key.startswith("unet.")
        },
        strict=True,
    )
    head.load_state_dict(
        {
            key.removeprefix("detect_head."): value
            for key, value in state.items()
            if key.startswith("detect_head.")
        },
        strict=True,
    )
    return unet.to(device).eval(), head.to(device).eval()


def d4_view(image, index: int):
    import torch

    if index == 0:
        return image
    if index <= 3:
        return image.flip({1: (-1,), 2: (-2,), 3: (-2, -1)}[index])
    if index == 4:
        return torch.rot90(image, 1, dims=(-2, -1))
    if index == 5:
        return torch.rot90(image, 3, dims=(-2, -1))
    if index == 6:
        return image.transpose(-1, -2)
    if index == 7:
        return torch.rot90(image, 1, dims=(-2, -1)).transpose(-1, -2)
    raise ValueError(index)


def d4_inverse(image, index: int):
    import torch

    if index <= 3:
        return d4_view(image, index)
    if index == 4:
        return torch.rot90(image, -1, dims=(-2, -1))
    if index == 5:
        return torch.rot90(image, -3, dims=(-2, -1))
    if index == 6:
        return image.transpose(-1, -2)
    if index == 7:
        return torch.rot90(image.transpose(-1, -2), -1, dims=(-2, -1))
    raise ValueError(index)


def encode_d4(unet, head, image, keep_feature_tta: bool):
    import torch

    feature_sum = None
    original_feature = None
    logit_sum = None
    for view in range(8):
        feature = unet(d4_view(image, view).unsqueeze(2))
        logit = torch.stack([head(feature[:, frame]) for frame in range(2)], dim=1)
        aligned_logit = d4_inverse(logit, view)
        logit_sum = aligned_logit if logit_sum is None else logit_sum + aligned_logit
        if keep_feature_tta:
            aligned = d4_inverse(feature, view)
            feature_sum = aligned if feature_sum is None else feature_sum + aligned
        elif view == 0:
            original_feature = feature
    return logit_sum / 8, feature_sum / 8 if keep_feature_tta else original_feature


def blend_logits(primary, secondary, weight: float):
    import torch

    blended = []
    for frame in range(2):
        p = primary[:, frame]
        s = secondary[:, frame]
        ratio = p.float().std(unbiased=False) / s.float().std(unbiased=False).clamp_min(1e-4)
        ratio = ratio.clamp(0.5, 2.0)
        aligned = (s - s.mean()) * ratio + p.mean()
        blended.append((1.0 - weight) * p + weight * aligned)
    return torch.stack(blended, dim=1)


def detect_frame(logits, threshold: float, pool_um: float, voxel_um: list[float]):
    import torch
    import torch.nn.functional as functional

    kernel = []
    for um in voxel_um:
        size = max(1, round(pool_um / um))
        kernel.append(size if size % 2 else size + 1)
    volume = logits.unsqueeze(0).unsqueeze(0)
    peaks = functional.max_pool3d(
        volume, tuple(kernel), stride=1, padding=tuple(size // 2 for size in kernel)
    )
    mask = (volume == peaks) & (torch.sigmoid(volume) > threshold)
    coords = torch.nonzero(mask[0, 0], as_tuple=False)
    scores = torch.sigmoid(logits[coords[:, 0], coords[:, 1], coords[:, 2]])
    return coords, scores


def features_at(feature_map, coords):
    if not len(coords):
        return np.empty((0, feature_map.shape[0]), dtype=np.float32)
    selected = feature_map[:, coords[:, 0], coords[:, 1], coords[:, 2]].T
    result = selected.float().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite image feature")
    return result


# %% [markdown]
# ## Run fixed candidates, match GT, and save evidence


# %%
def main() -> None:
    import torch

    started = time.perf_counter()
    config = yaml.safe_load(config_path().read_text(encoding="utf-8"))
    validation = config["validation"]
    source = synthetic_source(config)
    source_manifest = source / "manifest.json"
    source_metadata = source / "metadata.json"
    if sha256_file(source_manifest) != validation["source_manifest_sha256"]:
        raise ValueError("synthetic manifest SHA mismatch")
    if sha256_file(source_metadata) != validation["source_metadata_sha256"]:
        raise ValueError("synthetic metadata SHA mismatch")
    manifest = json.loads(source_manifest.read_text())
    selected = manifest["sequences"][: validation["sample_count"]]
    if len(selected) != validation["sample_count"]:
        raise ValueError("not enough synthetic sequences")
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle T4 GPU is required")
    device = torch.device("cuda")
    roots = {}
    for branch in ("primary", "secondary"):
        slug = config["model"][f"{branch}_dataset"].split("/", 1)[1]
        root = artifact_root(slug)
        if (
            sha256_file(root / "ARTIFACT_MANIFEST.json")
            != config["model"][f"{branch}_artifact_manifest_sha256"]
        ):
            raise ValueError(f"{branch} artifact manifest SHA mismatch")
        roots[branch] = root
    primary = load_detector(
        roots["primary"], config["model"]["primary_checkpoint_sha256"], config, device
    )
    secondary = load_detector(
        roots["secondary"], config["model"]["secondary_checkpoint_sha256"], config, device
    )
    output = Path("/kaggle/working/synthetic_detector_teacher_audit")
    feature_dir = output / "window_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    records = []
    feature_files = []
    input_files = []
    total_gpu_seconds = 0.0
    voxel = np.asarray(validation["pooled_voxel_zyx_um"], dtype=np.float32)
    native_voxel = np.asarray(validation["native_voxel_zyx_um"], dtype=np.float32)
    torch.cuda.reset_peak_memory_stats()
    for sequence in selected:
        path = source / sequence["file"]
        if path.stat().st_size != sequence["bytes"]:
            raise ValueError("synthetic sequence byte size mismatch")
        file_sha = sha256_file(path)
        input_files.append({"file": sequence["file"], "sha256": file_sha})
        with np.load(path, allow_pickle=False) as archive:
            volumes = archive["volumes"]
            nodes = archive["nodes"]
            edges = archive["edges"]
            divisions = archive["divisions"]
            pooled_voxel = archive["voxel_um_pooled"]
        expected_shape = (
            validation["expected_sequence_length"],
            *validation["expected_volume_zyx"],
        )
        if volumes.shape != expected_shape or volumes.dtype != np.uint16:
            raise ValueError("synthetic volume shape/dtype mismatch")
        if not np.allclose(pooled_voxel, voxel):
            raise ValueError("synthetic pooled voxel mismatch")
        if (
            len(nodes) != sequence["n_nodes"]
            or len(edges) != sequence["n_edges"]
            or len(divisions) != sequence["n_divisions"]
        ):
            raise ValueError("synthetic manifest row count mismatch")
        children = validate_graph(nodes, edges, divisions, len(volumes))
        gt_um = nodes[:, 1:4].astype(np.float64) * native_voxel
        normalized, q_low, q_high = normalize_synthetic_volumes(
            volumes, validation["quantile_low"], validation["quantile_high"]
        )
        for t in range(len(volumes) - 1):
            window = torch.from_numpy(normalized[t : t + 2]).unsqueeze(0).to(device)
            torch.cuda.synchronize()
            gpu_started = time.perf_counter()
            with torch.inference_mode():
                primary_logits, primary_features = encode_d4(*primary, window, True)
                secondary_logits, secondary_features = encode_d4(*secondary, window, False)
                logits = blend_logits(
                    primary_logits,
                    secondary_logits,
                    config["model"]["params"]["detection_blend_secondary"],
                )
                detections = [
                    detect_frame(
                        logits[0, frame, 0],
                        config["model"]["params"]["detector_threshold"],
                        config["model"]["params"]["pool_kernel_um"],
                        validation["pooled_voxel_zyx_um"],
                    )
                    for frame in range(2)
                ]
            torch.cuda.synchronize()
            gpu_seconds = time.perf_counter() - gpu_started
            total_gpu_seconds += gpu_seconds
            parts = []
            for frame, (coord_tensor, score_tensor) in enumerate(detections):
                if len(coord_tensor) > validation["max_candidates_per_frame_guard"]:
                    raise RuntimeError("candidate-per-frame guard exceeded")
                primary_part = features_at(primary_features[0, frame], coord_tensor)
                secondary_part = features_at(secondary_features[0, frame], coord_tensor)
                coords_grid = coord_tensor.cpu().numpy().astype(np.int16)
                coords_um = coords_grid.astype(np.float32) * voxel
                scores = score_tensor.float().cpu().numpy().astype(np.float32)
                gt_ids = np.flatnonzero(nodes[:, 0] == t + frame)
                mapping = match_candidates(
                    coords_um, gt_um[gt_ids], gt_ids, validation["gt_match_max_um"]
                )
                if (
                    primary_part.shape != (len(coords_grid), 32)
                    or secondary_part.shape != primary_part.shape
                ):
                    raise ValueError("feature shape mismatch")
                parts.append(
                    (coords_grid, coords_um, scores, primary_part, secondary_part, mapping)
                )
            src, tgt = parts
            counts, triplet_ids, triplet_class = classify_window_triplets(
                src[1],
                tgt[1],
                src[5],
                tgt[5],
                children,
                validation["parent_daughter_max_um"],
                validation["sister_max_um"],
                validation["max_triplets_per_parent_guard"],
                validation["max_triplets_per_window_guard"],
            )
            counts.update(
                gt_window_counts(
                    gt_um,
                    nodes,
                    children,
                    t,
                    src[5],
                    tgt[5],
                    validation["parent_daughter_max_um"],
                    validation["sister_max_um"],
                )
            )
            counts.update(
                gt_nodes_src=int(np.count_nonzero(nodes[:, 0] == t)),
                gt_nodes_tgt=int(np.count_nonzero(nodes[:, 0] == t + 1)),
                detected_src=len(src[0]),
                detected_tgt=len(tgt[0]),
                matched_src=int(np.count_nonzero(src[5] >= 0)),
                matched_tgt=int(np.count_nonzero(tgt[5] >= 0)),
            )
            feature_path = feature_dir / f"{path.stem}_t{t:02d}.npz"
            np.savez_compressed(
                feature_path,
                coords_src_grid=src[0],
                coords_tgt_grid=tgt[0],
                coords_src_physical=src[1],
                coords_tgt_physical=tgt[1],
                detection_scores_src=src[2],
                detection_scores_tgt=tgt[2],
                primary_features_src=src[3],
                primary_features_tgt=tgt[3],
                secondary_features_src=src[4],
                secondary_features_tgt=tgt[4],
                matched_gt_row_src=src[5],
                matched_gt_row_tgt=tgt[5],
                matched_triplet_ids=triplet_ids,
                matched_triplet_class=triplet_class,
            )
            feature_files.append(
                {
                    "file": feature_path.relative_to(output).as_posix(),
                    "sha256": sha256_file(feature_path),
                }
            )
            record = {
                "sequence": sequence["file"],
                "t": t,
                "q_low": float(q_low),
                "q_high": float(q_high),
                "gpu_forward_seconds": gpu_seconds,
                **{key: counts[key] for key in COUNT_FIELDS},
            }
            records.append(record)
            print(
                f"{path.stem} t={t}: src {record['detected_src']}/{record['gt_nodes_src']} "
                f"tgt {record['detected_tgt']}/{record['gt_nodes_tgt']} "
                f"division {record['detected_divisions']}/{record['gt_divisions']}",
                flush=True,
            )
            del (
                primary_logits,
                secondary_logits,
                primary_features,
                secondary_features,
                logits,
                window,
            )
            torch.cuda.empty_cache()
    if len(records) != validation["expected_windows"]:
        raise ValueError("window count mismatch")
    totals = {
        key: (
            max(row[key] for row in records)
            if key == "max_triplets_per_mother"
            else sum(row[key] for row in records)
        )
        for key in COUNT_FIELDS
    }
    per_window = output / "per_window.csv"
    with per_window.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sequence", "t", "q_low", "q_high", "gpu_forward_seconds", *COUNT_FIELDS],
        )
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "experiment": EXPERIMENT,
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_metadata_sha256": sha256_file(source_metadata),
        "checkpoint_sha256": {
            branch: config["model"][f"{branch}_checkpoint_sha256"]
            for branch in ("primary", "secondary")
        },
        "feature_schema_sha256": json_sha(FEATURE_SCHEMA),
        "selected_sequences": len(selected),
        "window_count": len(records),
        "totals": totals,
        "gpu_forward_seconds": total_gpu_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "diagnostic_elapsed_seconds": time.perf_counter() - started,
        "preprocessing_difference": (
            "Synthetic 6-frame quantiles replace real Zarr image_statistics quantiles."
        ),
        "interpretation_limit": (
            "GT-center and detected-center counts use different candidate sources; "
            "only synthetic GT is complete."
        ),
    }
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    write_json(
        output / "manifest.json",
        {
            "input": {
                "source_manifest_sha256": summary["source_manifest_sha256"],
                "source_metadata_sha256": summary["source_metadata_sha256"],
                "sequence_files": input_files,
                "checkpoint_sha256": summary["checkpoint_sha256"],
            },
            "output": {
                "per_window_sha256": sha256_file(per_window),
                "summary_sha256": sha256_file(summary_path),
                "feature_schema_sha256": summary["feature_schema_sha256"],
                "window_files": feature_files,
                "window_file_bundle_sha256": json_sha(feature_files),
            },
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


# %%
if __name__ == "__main__":
    main()
