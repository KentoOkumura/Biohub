"""Join fixed two-frame cache windows into a five-frame tracker example."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import frozen_tracker as base
import numpy as np


def _load_history_cache(
    path: Path, *, feature_channels: int, expected_primary_checkpoint_sha256: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Validate a history window while permitting a genuinely empty detector frame."""
    arrays, metadata = base._read_cache_payload(path, load_all_arrays=True)
    missing = set(base.REQUIRED_CACHE_ARRAYS) - set(arrays)
    if missing:
        raise ValueError({"cache_arrays_missing": sorted(missing), "path": str(path)})
    frames = metadata.get("window_frames")
    expected = {
        "schema_version": base.CACHE_SCHEMA_VERSION,
        "experiment": "exp015_oracle_stage_limits",
        "dataset": path.parent.name,
        "primary_checkpoint_sha256": expected_primary_checkpoint_sha256,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"history cache metadata mismatch: {path}")
    if not isinstance(frames, list) or len(frames) != 2 or int(frames[1]) != int(frames[0]) + 1:
        raise ValueError(f"history cache frames are not adjacent: {path}")
    if path.name != f"{int(frames[0]):06d}_{int(frames[1]):06d}.npz":
        raise ValueError(f"history cache name and frames differ: {path}")
    if metadata.get("array_content_sha256") != base.array_content_sha256(arrays):
        raise ValueError(f"history cache content mismatch: {path}")
    schema = {str(item.get("name")): item for item in metadata.get("array_schema", [])}
    actual_schema = {str(item["name"]): item for item in base.array_schema(arrays)}
    if schema != actual_schema:
        raise ValueError(f"history cache array schema mismatch: {path}")
    for side in ("src", "tgt"):
        count = len(arrays[f"candidate_ids_{side}"])
        mask = arrays[f"candidate_mask_{side}"]
        if mask.shape != (count,) or mask.dtype != np.bool_ or not mask.all():
            raise ValueError(f"invalid history candidate mask: {path} {side}")
        if not np.issubdtype(arrays[f"candidate_ids_{side}"].dtype, np.integer):
            raise ValueError(f"history candidate IDs are not integers: {path} {side}")
        if len(np.unique(arrays[f"candidate_ids_{side}"])) != count:
            raise ValueError(f"duplicate history candidate IDs: {path} {side}")
        expected_shapes = {
            f"coords_{side}_grid": (count, 3),
            f"coords_{side}_physical": (count, 3),
            f"primary_features_{side}": (count, feature_channels),
            f"position_features_{side}": (count, 32),
        }
        for name, shape in expected_shapes.items():
            if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError(f"invalid history array: {path} {name}")
    return arrays, metadata


def _check_overlap(
    earlier: dict[str, np.ndarray], later: dict[str, np.ndarray], *, path: Path
) -> None:
    earlier_ids = np.asarray(earlier["candidate_ids_tgt"])
    later_ids = np.asarray(later["candidate_ids_src"])
    if len(earlier_ids) != len(later_ids) or set(earlier_ids.tolist()) != set(later_ids.tolist()):
        raise ValueError(f"overlapping window candidate IDs differ: {path}")
    earlier_index = {int(value): index for index, value in enumerate(earlier_ids)}
    order = [earlier_index[int(value)] for value in later_ids]
    for unit in ("grid", "physical"):
        if not np.array_equal(
            earlier[f"coords_tgt_{unit}"][order],
            later[f"coords_src_{unit}"],
        ):
            raise ValueError(f"overlapping window coords_{unit} differs: {path}")


def build_past_feature_window_example(
    path: Path,
    annotation: base.AnnotationGraph,
    *,
    feature_channels: int,
    expected_primary_checkpoint_sha256: str,
    max_matching_distance_um: float,
    downsample_zyx: tuple[float, float, float],
    history_frames: int,
) -> dict[str, Any]:
    if history_frames != 3:
        raise ValueError("the initial experiment requires three previous frames")
    example = base.build_window_example(
        path,
        annotation,
        feature_channels=feature_channels,
        expected_primary_checkpoint_sha256=expected_primary_checkpoint_sha256,
        max_matching_distance_um=max_matching_distance_um,
        downsample_zyx=downsample_zyx,
    )
    central, central_meta = base.validate_window_cache(
        path,
        feature_channels=feature_channels,
        expected_primary_checkpoint_sha256=expected_primary_checkpoint_sha256,
    )
    source_frame, target_frame = map(int, central_meta["window_frames"])
    if [source_frame, target_frame] != example["window_frames"]:
        raise ValueError(f"central cache frames changed: {path}")
    if len(np.unique(central["candidate_ids_src"])) != len(central["candidate_ids_src"]):
        raise ValueError(f"duplicate central source candidate IDs: {path}")
    history: list[tuple[int, dict[str, np.ndarray], dict[str, Any]]] = []
    later = central
    for frame in range(source_frame - 1, max(-1, source_frame - history_frames - 1), -1):
        prior_path = path.with_name(f"{frame:06d}_{frame + 1:06d}.npz")
        if not prior_path.is_file():
            raise FileNotFoundError(f"past window missing within video: {prior_path}")
        prior, prior_meta = _load_history_cache(
            prior_path,
            feature_channels=feature_channels,
            expected_primary_checkpoint_sha256=expected_primary_checkpoint_sha256,
        )
        if prior_meta["window_frames"] != [frame, frame + 1]:
            raise ValueError(f"incorrect past window frames: {prior_path}")
        _check_overlap(prior, later, path=prior_path)
        history.append((frame, prior, prior_meta))
        later = prior
    history.reverse()
    features = [
        np.concatenate((item["primary_features_src"], item["position_features_src"]), axis=1)
        for _, item, _ in history
    ]
    coords = [np.asarray(item["coords_src_physical"], dtype=np.float32) for _, item, _ in history]
    frames = [
        np.full(len(item["candidate_ids_src"]), frame, dtype=np.float32)
        for frame, item, _ in history
    ]
    feature_dim = int(example["features_src"].shape[1])
    example["features_past"] = (
        np.concatenate(features, axis=0).astype(np.float32, copy=False)
        if features
        else np.zeros((0, feature_dim), dtype=np.float32)
    )
    example["coords_past_physical"] = (
        np.concatenate(coords, axis=0) if coords else np.zeros((0, 3), dtype=np.float32)
    )
    example["frames_past"] = np.concatenate(frames) if frames else np.zeros((0,), dtype=np.float32)
    example["past_candidate_count"] = len(example["frames_past"])
    example["past_frame_count"] = sum(
        bool(len(item["candidate_ids_src"])) for _, item, _ in history
    )
    example["past_sources"] = [
        {
            "frame": frame,
            "window": [frame, frame + 1],
            "content_sha256": meta.get("array_content_sha256"),
            "candidate_ids": item["candidate_ids_src"].tolist(),
        }
        for frame, item, meta in history
    ]
    return example


class PastFeatureWindowDataset:
    def __init__(
        self,
        paths: list[Path],
        annotations: dict[str, base.AnnotationGraph],
        *,
        feature_channels: int,
        expected_primary_checkpoint_sha256: str,
        max_matching_distance_um: float,
        downsample_zyx: tuple[float, float, float],
        history_frames: int,
        diagnostic_mode: str = "full",
    ) -> None:
        if diagnostic_mode not in {"full", "no_past", "reverse_time"}:
            raise ValueError(f"unknown diagnostic mode: {diagnostic_mode}")
        self.paths = list(paths)
        self.annotations = annotations
        self.feature_channels = feature_channels
        self.expected_primary_checkpoint_sha256 = expected_primary_checkpoint_sha256
        self.max_matching_distance_um = max_matching_distance_um
        self.downsample_zyx = downsample_zyx
        self.history_frames = history_frames
        self.diagnostic_mode = diagnostic_mode

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        example = build_past_feature_window_example(
            path,
            self.annotations[path.parent.name],
            feature_channels=self.feature_channels,
            expected_primary_checkpoint_sha256=self.expected_primary_checkpoint_sha256,
            max_matching_distance_um=self.max_matching_distance_um,
            downsample_zyx=self.downsample_zyx,
            history_frames=self.history_frames,
        )
        example["diagnostic_mode"] = self.diagnostic_mode
        return example


def collate_past_feature_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch = base.collate_window_examples(examples)
    batch_size = len(examples)
    max_past = max(len(example["frames_past"]) for example in examples)
    feature_dim = int(examples[0]["features_src"].shape[1])
    batch["features_past"] = torch.zeros(batch_size, max_past, feature_dim, dtype=torch.float32)
    batch["coords_past_physical"] = torch.zeros(batch_size, max_past, 3, dtype=torch.float32)
    batch["frames_past"] = torch.zeros(batch_size, max_past, dtype=torch.float32)
    batch["past_mask"] = torch.zeros(batch_size, max_past, dtype=torch.bool)
    batch["frames_src"] = torch.zeros(
        batch_size, batch["features_src"].shape[1], dtype=torch.float32
    )
    batch["frames_tgt"] = torch.zeros(
        batch_size, batch["features_tgt"].shape[1], dtype=torch.float32
    )
    for index, example in enumerate(examples):
        past_count = int(example["past_candidate_count"])
        source_count = len(example["features_src"])
        target_count = len(example["features_tgt"])
        batch["features_past"][index, :past_count] = torch.from_numpy(example["features_past"])
        batch["coords_past_physical"][index, :past_count] = torch.from_numpy(
            example["coords_past_physical"]
        )
        frames = example["frames_past"].copy()
        mode = example.get("diagnostic_mode", "full")
        if mode == "reverse_time":
            distinct = sorted(set(frames.tolist()))
            reversed_frame = dict(zip(distinct, reversed(distinct), strict=True))
            frames = np.asarray([reversed_frame[value] for value in frames], dtype=np.float32)
        batch["frames_past"][index, :past_count] = torch.from_numpy(frames)
        batch["past_mask"][index, :past_count] = mode != "no_past"
        batch["frames_src"][index, :source_count] = float(example["window_frames"][0])
        batch["frames_tgt"][index, :target_count] = float(example["window_frames"][1])
        batch["history_present_src"][index, :source_count] = past_count > 0 and mode != "no_past"
        batch["metadata"][index].update(
            {
                "past_sources": example["past_sources"],
                "past_candidate_count": past_count,
                "past_frame_count": example["past_frame_count"],
                "diagnostic_mode": mode,
                "candidate_ids_src": example["candidate_ids_src"].tolist(),
                "candidate_ids_tgt": example["candidate_ids_tgt"].tolist(),
                "coords_src_physical": example["coords_src_physical"].tolist(),
                "coords_tgt_physical": example["coords_tgt_physical"].tolist(),
            }
        )
    return batch


def tracker_logits(model: Any, batch: dict[str, Any]) -> Any:
    args = (
        batch["features_src"],
        batch["features_tgt"],
        batch["coords_src"],
        batch["coords_tgt"],
        batch["source_mask"],
        batch["target_mask"],
    )
    if not bool(getattr(model, "requires_past_feature_attention", False)):
        return model(*args)
    return model(
        *args,
        features_past=batch["features_past"],
        coords_past_physical=batch["coords_past_physical"],
        frames_past=batch["frames_past"],
        coords_src_physical=batch["coords_src_physical"],
        coords_tgt_physical=batch["coords_tgt_physical"],
        frames_src=batch["frames_src"],
        frames_tgt=batch["frames_tgt"],
        past_mask=batch["past_mask"],
    )


def _with_dispatch(function: Any, *args: Any, **kwargs: Any) -> Any:
    original = base.tracker_logits
    base.tracker_logits = tracker_logits
    try:
        return function(*args, **kwargs)
    finally:
        base.tracker_logits = original


def train_one_epoch(*args: Any, **kwargs: Any) -> dict[str, float]:
    return _with_dispatch(base.train_one_epoch, *args, **kwargs)


def evaluate_tracker(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _with_dispatch(base.evaluate_tracker, *args, **kwargs)


def evaluate_tracker_diagnostic(*args: Any, **kwargs: Any) -> dict[str, Any]:
    model = args[0]
    attention = getattr(model, "past_attention", None)
    if attention is not None:
        attention.reset_null_mass()
    metrics = _with_dispatch(base.evaluate_tracker_diagnostic, *args, **kwargs)
    metrics["no_past_attention_mass_fraction"] = (
        attention.null_mass_fraction() if attention is not None else None
    )
    metrics["past_context_source_count"] = metrics.pop("history_present_source_count")
    metrics["source_count"] = metrics.pop("history_source_count")
    metrics["past_context_coverage"] = metrics.pop("history_coverage")
    return metrics
