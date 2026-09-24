from __future__ import annotations

from pathlib import Path
from typing import Any

import frozen_tracker as base
import numpy as np


def build_past_candidate_window_example(
    path: Path,
    annotation: base.AnnotationGraph,
    *,
    feature_channels: int,
    expected_primary_checkpoint_sha256: str,
    max_matching_distance_um: float,
    downsample_zyx: tuple[float, float, float],
) -> dict[str, Any]:
    example = base.build_window_example(
        path,
        annotation,
        feature_channels=feature_channels,
        expected_primary_checkpoint_sha256=expected_primary_checkpoint_sha256,
        max_matching_distance_um=max_matching_distance_um,
        downsample_zyx=downsample_zyx,
    )
    central_arrays, central_metadata = base.validate_window_cache(
        path,
        feature_channels=feature_channels,
        expected_primary_checkpoint_sha256=expected_primary_checkpoint_sha256,
    )
    source_frame, target_frame = map(int, central_metadata["window_frames"])
    if [source_frame, target_frame] != example["window_frames"]:
        raise ValueError(f"central window metadata changed while reading {path}")
    prior_path = path.with_name(f"{source_frame - 1:06d}_{source_frame:06d}.npz")
    if source_frame == 0:
        example["candidate_ids_prev"] = np.zeros((0,), dtype=np.int64)
        example["coords_prev_physical"] = np.zeros((0, 3), dtype=np.float32)
        example["prior_cache_content_sha256"] = None
        return example
    if not prior_path.is_file():
        raise FileNotFoundError(f"previous window missing within video: {prior_path}")
    prior_arrays, prior_metadata = base.validate_window_cache(
        prior_path,
        feature_channels=feature_channels,
        expected_primary_checkpoint_sha256=expected_primary_checkpoint_sha256,
    )
    if prior_metadata["window_frames"] != [source_frame - 1, source_frame]:
        raise ValueError(f"incorrect previous window: {prior_path}")
    for label, prior_key, central_key in (
        ("ids", "candidate_ids_tgt", "candidate_ids_src"),
        ("grid", "coords_tgt_grid", "coords_src_grid"),
        ("physical", "coords_tgt_physical", "coords_src_physical"),
    ):
        if not np.array_equal(prior_arrays[prior_key], central_arrays[central_key]):
            raise ValueError(f"overlapping window {label} mismatch: {prior_path} {path}")
    example["candidate_ids_prev"] = np.asarray(prior_arrays["candidate_ids_src"], dtype=np.int64)
    example["coords_prev_physical"] = np.asarray(
        prior_arrays["coords_src_physical"], dtype=np.float32
    )
    example["prior_cache_content_sha256"] = prior_metadata["array_content_sha256"]
    return example


class PastCandidateWindowDataset:
    def __init__(
        self,
        paths: list[Path],
        annotations: dict[str, base.AnnotationGraph],
        *,
        feature_channels: int,
        expected_primary_checkpoint_sha256: str,
        max_matching_distance_um: float,
        downsample_zyx: tuple[float, float, float],
        diagnostic_context: dict[str, Any] | None = None,
    ) -> None:
        self.paths = list(paths)
        self.annotations = annotations
        self.feature_channels = feature_channels
        self.expected_primary_checkpoint_sha256 = expected_primary_checkpoint_sha256
        self.max_matching_distance_um = max_matching_distance_um
        self.downsample_zyx = downsample_zyx
        self.diagnostic_context = diagnostic_context

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        example = build_past_candidate_window_example(
            path,
            self.annotations[path.parent.name],
            feature_channels=self.feature_channels,
            expected_primary_checkpoint_sha256=self.expected_primary_checkpoint_sha256,
            max_matching_distance_um=self.max_matching_distance_um,
            downsample_zyx=self.downsample_zyx,
        )
        if self.diagnostic_context is not None:
            example["diagnostic_context"] = past_annotation_context(
                example,
                self.annotations[path.parent.name],
                self.max_matching_distance_um,
                self.diagnostic_context,
            )
        return example


def past_annotation_context(
    example: dict[str, Any],
    annotation: base.AnnotationGraph,
    matching_um: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """GT-based reporting metadata only; never passed to the model or kNN selector."""
    frame = example["window_frames"][0]
    parents = np.full(len(example["coords_src_physical"]), -1, dtype=np.int64)
    if frame and frame - 1 in annotation.frames:
        past_gt, current_gt = annotation.frames[frame - 1], annotation.frames[frame]
        previous_matches, _ = base.greedy_match_candidates(
            example["coords_prev_physical"], past_gt.node_ids, past_gt.coords_physical, matching_um
        )
        source_matches, _ = base.greedy_match_candidates(
            example["coords_src_physical"],
            current_gt.node_ids,
            current_gt.coords_physical,
            matching_um,
        )
        previous_map = {
            int(node): int(candidate)
            for node, candidate in zip(previous_matches, example["candidate_ids_prev"], strict=True)
            if node >= 0
        }
        source_map = {int(node): index for index, node in enumerate(source_matches) if node >= 0}
        outgoing = annotation.outgoing_edges or {}
        for parent, candidate_id in previous_map.items():
            for child in outgoing.get(parent, ()):
                if child in source_map:
                    parents[source_map[child]] = candidate_id
    return {"known_parent_candidate_ids": parents, "config": cfg}


def collate_past_candidate_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch = base.collate_window_examples(examples)
    batch_size = len(examples)
    max_prev = max(len(example["coords_prev_physical"]) for example in examples)
    coords_prev_physical = torch.zeros(batch_size, max_prev, 3, dtype=torch.float32)
    prev_mask = torch.zeros(batch_size, max_prev, dtype=torch.bool)
    candidate_ids_prev = torch.zeros(batch_size, max_prev, dtype=torch.int64)
    for index, example in enumerate(examples):
        n_prev = len(example["coords_prev_physical"])
        n_source = len(example["features_src"])
        coords_prev_physical[index, :n_prev] = torch.from_numpy(example["coords_prev_physical"])
        prev_mask[index, :n_prev] = True
        candidate_ids_prev[index, :n_prev] = torch.from_numpy(example["candidate_ids_prev"])
        # The inherited diagnostic counts this legacy field. Reuse it only as a
        # count of source nodes that have any earlier-frame candidate context.
        batch["history_present_src"][index, :n_source] = n_prev > 0
        batch["metadata"][index]["previous_candidate_count"] = n_prev
        batch["metadata"][index]["candidate_ids_src"] = example["candidate_ids_src"]
        batch["metadata"][index]["candidate_ids_tgt"] = example["candidate_ids_tgt"]
        if "diagnostic_context" in example:
            batch["metadata"][index]["diagnostic_context"] = example["diagnostic_context"]
        batch["metadata"][index]["prior_cache_content_sha256"] = example[
            "prior_cache_content_sha256"
        ]
    batch["coords_prev_physical"] = coords_prev_physical
    batch["prev_mask"] = prev_mask
    batch["candidate_ids_prev"] = candidate_ids_prev
    return batch
