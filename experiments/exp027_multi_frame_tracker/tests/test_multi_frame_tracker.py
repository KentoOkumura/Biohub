from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

from frozen_tracker import (  # noqa: E402
    AnnotationGraph,
    FrameAnnotation,
    array_content_sha256,
    array_schema,
    build_three_frame_example,
    select_pilot_outer_paths,
)

SHA = "a" * 64


def write_cache(path: Path, frame: int, *, target_id: int) -> None:
    arrays = {}
    for side, node_id, position in (
        ("src", frame, float(frame)),
        ("tgt", target_id, float(frame + 1)),
    ):
        coords = np.asarray([[position, 0, 0]], dtype=np.float32)
        arrays[f"candidate_ids_{side}"] = np.asarray([node_id], dtype=np.int64)
        arrays[f"coords_{side}_grid"] = coords
        arrays[f"coords_{side}_physical"] = coords
        arrays[f"position_features_{side}"] = np.zeros((1, 32), dtype=np.float32)
        arrays[f"primary_features_{side}"] = np.full((1, 32), float(frame + 1), dtype=np.float32)
        arrays[f"candidate_mask_{side}"] = np.ones(1, dtype=bool)
    metadata = {
        "schema_version": 1,
        "experiment": "exp015_oracle_stage_limits",
        "dataset": path.parent.name,
        "window_frames": [frame, frame + 1],
        "primary_checkpoint_sha256": SHA,
        "array_schema": array_schema(arrays),
        "array_content_sha256": array_content_sha256(arrays),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **arrays,
        __metadata_json__=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
    )


def annotation() -> AnnotationGraph:
    frames = {
        frame: FrameAnnotation(
            node_ids=np.asarray([100 + frame], dtype=np.int64),
            coords_physical=np.asarray([[float(frame), 0, 0]], dtype=np.float32),
        )
        for frame in range(3)
    }
    return AnnotationGraph(
        frames=frames,
        edges=frozenset({(100, 101), (101, 102)}),
        content_sha256=SHA,
        outgoing_edges={100: (101,), 101: (102,)},
    )


def example(path: Path) -> dict:
    return build_three_frame_example(
        path,
        annotation(),
        feature_channels=32,
        expected_primary_checkpoint_sha256=SHA,
        max_matching_distance_um=5.0,
        downsample_zyx=(1, 1, 1),
    )


def test_previous_frame_comes_from_prior_window_and_central_target_is_unchanged(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "44b6_sample"
    first = sample / "000000_000001.npz"
    second = sample / "000001_000002.npz"
    write_cache(first, 0, target_id=1)
    write_cache(second, 1, target_id=2)
    initial = example(first)
    three = example(second)
    assert initial["features_prev"].shape == (0, 64)
    assert three["features_prev"].shape == (1, 64)
    assert three["features_prev"][0, 0] == 1
    assert three["features_src"][0, 0] == 2
    assert three["target"].shape == (1, 1)
    assert three["target"][0, 0] == 1
    assert three["window_frames"] == [1, 2]


def test_changed_overlapping_candidate_identity_stops(tmp_path: Path) -> None:
    sample = tmp_path / "44b6_sample"
    first = sample / "000000_000001.npz"
    second = sample / "000001_000002.npz"
    write_cache(first, 0, target_id=99)
    write_cache(second, 1, target_id=2)
    with pytest.raises(ValueError, match="overlapping window ids mismatch"):
        example(second)


def test_pilot_evaluation_uses_one_seeded_window_per_sample(tmp_path: Path) -> None:
    samples = [f"6bba_sample_{index}" for index in range(5)]
    paths = [
        tmp_path / sample / f"{frame:06d}_{frame + 1:06d}.npz"
        for sample in samples
        for frame in range(3)
    ]
    first = select_pilot_outer_paths(paths, samples, count=3, seed=42)
    again = select_pilot_outer_paths(list(reversed(paths)), samples, count=3, seed=42)
    assert first == again
    assert len(first) == 3
    assert len({path.parent.name for path in first}) == 3
    assert all(path in paths for path in first)
    with pytest.raises(ValueError, match="only 5 samples"):
        select_pilot_outer_paths(paths, samples, count=6, seed=42)
