"""Exp043 pair features and sparse GEFF labels for exp044."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import frozen_tracker as labels

GRID_SPACING_UM = np.asarray([1.625, 1.625, 1.625], dtype=np.float32)
DOWNSAMPLE = np.asarray([1.0, 4.0, 4.0], dtype=np.float32)
REQUIRED = {
    "window_frames",
    "candidate_ids_prev",
    "candidate_ids_src",
    "candidate_ids_tgt",
    "coords_prev_grid",
    "coords_src_grid",
    "coords_tgt_grid",
    "primary_features_src",
    "primary_features_tgt",
    "position_features_src",
    "position_features_tgt",
    "primary_forward_logits",
    "primary_reverse_logits",
    "secondary_logits",
    "x138_fused_logits",
}


def read_capture(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != REQUIRED:
            raise ValueError(
                {
                    "missing": sorted(REQUIRED - set(saved.files)),
                    "extra": sorted(set(saved.files) - REQUIRED),
                }
            )
        arrays = {key: saved[key] for key in REQUIRED}
    frames = arrays["window_frames"].tolist()
    if len(frames) != 2 or frames[1] != frames[0] + 1:
        raise ValueError(f"non-adjacent frame pair: {path}")
    if path.name != f"{frames[0]:06d}_{frames[1]:06d}.npz":
        raise ValueError(f"frame/name mismatch: {path}")
    n_src, n_tgt = len(arrays["candidate_ids_src"]), len(arrays["candidate_ids_tgt"])
    n_prev = len(arrays["candidate_ids_prev"])
    for side, n in (("prev", n_prev), ("src", n_src), ("tgt", n_tgt)):
        if arrays[f"coords_{side}_grid"].shape != (n, 3):
            raise ValueError(f"invalid {side} coordinates: {path}")
        if not np.isfinite(arrays[f"coords_{side}_grid"]).all():
            raise ValueError(f"non-finite {side} coordinates: {path}")
        if len(np.unique(arrays[f"candidate_ids_{side}"])) != n:
            raise ValueError(f"duplicate {side} ids: {path}")
    for side, n in (("src", n_src), ("tgt", n_tgt)):
        if arrays[f"primary_features_{side}"].shape != (n, 32):
            raise ValueError(f"invalid {side} image features: {path}")
        if arrays[f"position_features_{side}"].shape != (n, 32):
            raise ValueError(f"invalid {side} position features: {path}")
        for key in (f"primary_features_{side}", f"position_features_{side}"):
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"non-finite {key}: {path}")
    for key, shape in (
        ("primary_forward_logits", (n_src, n_tgt)),
        ("primary_reverse_logits", (n_tgt, n_src)),
        ("secondary_logits", (n_src, n_tgt)),
        ("x138_fused_logits", (n_src, n_tgt)),
    ):
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"invalid {key}: {path}")
    return arrays


def known_parent_retention(
    arrays: dict[str, np.ndarray], annotation: labels.AnnotationGraph, radius_um: float, k: int
) -> tuple[int, int]:
    frame = int(arrays["window_frames"][0])
    if frame == 0 or frame - 1 not in annotation.frames or frame not in annotation.frames:
        return 0, 0
    prev_gt, src_gt = annotation.frames[frame - 1], annotation.frames[frame]
    prev_ids, _ = labels.greedy_match_candidates(
        arrays["coords_prev_grid"] * GRID_SPACING_UM,
        prev_gt.node_ids,
        prev_gt.coords_physical,
        radius_um,
    )
    src_ids, _ = labels.greedy_match_candidates(
        arrays["coords_src_grid"] * GRID_SPACING_UM,
        src_gt.node_ids,
        src_gt.coords_physical,
        radius_um,
    )
    previous_by_gt = {int(gt): idx for idx, gt in enumerate(prev_ids) if gt >= 0}
    recovered = eligible = 0
    for source_index, gt in enumerate(src_ids):
        if gt < 0:
            continue
        known = [
            previous_by_gt[parent]
            for parent, children in (annotation.outgoing_edges or {}).items()
            if gt in children and parent in previous_by_gt
        ]
        if not known:
            continue
        eligible += 1
        distances = np.sum(
            (arrays["coords_prev_grid"] - arrays["coords_src_grid"][source_index]) ** 2, axis=1
        )
        order = np.lexsort((arrays["candidate_ids_prev"], distances))[:k]
        recovered += int(any(index in order for index in known))
    return recovered, eligible


class X138PairDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        annotations: dict[str, labels.AnnotationGraph],
        *,
        match_radius_um: float = 7.0,
    ) -> None:
        self.paths = paths
        self.annotations = annotations
        self.match_radius_um = match_radius_um

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[index]
        arrays = read_capture(path)
        annotation = self.annotations[path.parent.name]
        src_frame, tgt_frame = arrays["window_frames"].tolist()
        if src_frame not in annotation.frames or tgt_frame not in annotation.frames:
            raise ValueError(f"annotation is missing frame {src_frame} or {tgt_frame}: {path}")
        src_gt, tgt_gt = annotation.frames[src_frame], annotation.frames[tgt_frame]
        src_match, _ = labels.greedy_match_candidates(
            arrays["coords_src_grid"] * GRID_SPACING_UM,
            src_gt.node_ids,
            src_gt.coords_physical,
            self.match_radius_um,
        )
        tgt_match, _ = labels.greedy_match_candidates(
            arrays["coords_tgt_grid"] * GRID_SPACING_UM,
            tgt_gt.node_ids,
            tgt_gt.coords_physical,
            self.match_radius_um,
        )
        target = labels.build_legacy_edge_target(
            src_match, tgt_match, annotation.edges, outgoing_edges=annotation.outgoing_edges
        )
        features_src = np.concatenate(
            (arrays["primary_features_src"], arrays["position_features_src"]), axis=1
        )
        features_tgt = np.concatenate(
            (arrays["primary_features_tgt"], arrays["position_features_tgt"]), axis=1
        )
        values = {
            "features_src": features_src,
            "features_tgt": features_tgt,
            "coords_src": arrays["coords_src_grid"] * DOWNSAMPLE,
            "coords_tgt": arrays["coords_tgt_grid"] * DOWNSAMPLE,
            "coords_prev_physical": arrays["coords_prev_grid"] * GRID_SPACING_UM,
            "coords_src_physical": arrays["coords_src_grid"] * GRID_SPACING_UM,
            "coords_tgt_physical": arrays["coords_tgt_grid"] * GRID_SPACING_UM,
            "candidate_ids_prev": arrays["candidate_ids_prev"],
            "prev_mask": np.ones(len(arrays["candidate_ids_prev"]), dtype=bool),
            "secondary_logits": arrays["secondary_logits"],
            "primary_forward_logits": arrays["primary_forward_logits"],
            "primary_reverse_logits": arrays["primary_reverse_logits"],
            "x138_fused_logits": arrays["x138_fused_logits"],
            "target": target,
        }
        result: dict[str, torch.Tensor | str] = {
            key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in values.items()
        }
        result["path"] = str(path)
        return result
