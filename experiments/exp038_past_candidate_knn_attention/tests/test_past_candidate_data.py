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
)
from past_candidate_data import build_past_candidate_window_example  # noqa: E402

SHA = "a" * 64


def write_cache(path: Path, frame: int, *, source_id: int, target_id: int) -> None:
    arrays: dict[str, np.ndarray] = {}
    for side, node_id, position in (
        ("src", source_id, float(frame)),
        ("tgt", target_id, float(frame + 1)),
    ):
        coords = np.asarray([[position, 0, 0]], dtype=np.float32)
        arrays[f"candidate_ids_{side}"] = np.asarray([node_id], dtype=np.int64)
        arrays[f"coords_{side}_grid"] = coords
        arrays[f"coords_{side}_physical"] = coords
        arrays[f"position_features_{side}"] = np.zeros((1, 32), dtype=np.float32)
        arrays[f"primary_features_{side}"] = np.ones((1, 32), dtype=np.float32)
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


def build(path: Path) -> dict:
    return build_past_candidate_window_example(
        path,
        annotation(),
        feature_channels=32,
        expected_primary_checkpoint_sha256=SHA,
        max_matching_distance_um=5.0,
        downsample_zyx=(1, 1, 1),
    )


def test_previous_candidates_come_from_prior_window_without_predicting_links(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "44b6_sample"
    first = sample / "000000_000001.npz"
    second = sample / "000001_000002.npz"
    write_cache(first, 0, source_id=0, target_id=1)
    write_cache(second, 1, source_id=1, target_id=2)
    initial = build(first)
    central = build(second)
    assert initial["coords_prev_physical"].shape == (0, 3)
    assert central["candidate_ids_prev"].tolist() == [0]
    assert central["coords_prev_physical"].tolist() == [[0.0, 0.0, 0.0]]
    assert central["target"].tolist() == [[1.0]]


def test_overlapping_candidate_identity_or_coordinates_must_match(tmp_path: Path) -> None:
    sample = tmp_path / "44b6_sample"
    first = sample / "000000_000001.npz"
    second = sample / "000001_000002.npz"
    write_cache(first, 0, source_id=0, target_id=99)
    write_cache(second, 1, source_id=1, target_id=2)
    with pytest.raises(ValueError, match="overlapping window ids mismatch"):
        build(second)


def test_missing_previous_window_is_not_treated_as_video_start(tmp_path: Path) -> None:
    second = tmp_path / "44b6_sample" / "000001_000002.npz"
    write_cache(second, 1, source_id=1, target_id=2)
    with pytest.raises(FileNotFoundError, match="previous window missing"):
        build(second)


def test_gt_changes_do_not_change_candidate_inputs(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from past_candidate_data import collate_past_candidate_examples

    first = tmp_path / "44b6_sample" / "000000_000001.npz"
    second = first.with_name("000001_000002.npz")
    write_cache(first, 0, source_id=0, target_id=1)
    write_cache(second, 1, source_id=1, target_id=2)
    original = build(second)
    empty_edges = AnnotationGraph(annotation().frames, frozenset(), SHA, {})
    changed = build_past_candidate_window_example(
        second,
        empty_edges,
        feature_channels=32,
        expected_primary_checkpoint_sha256=SHA,
        max_matching_distance_um=5.0,
        downsample_zyx=(1, 1, 1),
    )
    assert changed["target"].sum() == 0 and original["target"].sum() == 1
    for key in ("coords_prev_physical", "coords_src_physical", "candidate_ids_prev"):
        np.testing.assert_array_equal(original[key], changed[key])
    batch = collate_past_candidate_examples([original, changed])
    assert batch["candidate_ids_prev"].tolist() == [[0], [0]]


def test_diagnostic_slices_and_saved_predictions(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from frozen_tracker import evaluate_tracker_diagnostic
    from past_candidate_data import collate_past_candidate_examples, past_annotation_context

    class ConstantTracker(torch.nn.Module):
        def forward(self, features_src, features_tgt, *args):
            return features_src.new_zeros(
                (len(features_src), features_src.shape[1], features_tgt.shape[1])
            )

    first = tmp_path / "44b6_sample" / "000000_000001.npz"
    second = first.with_name("000001_000002.npz")
    write_cache(first, 0, source_id=0, target_id=1)
    write_cache(second, 1, source_id=1, target_id=2)
    examples = [build(first), build(second)]
    cfg = {"k": 8, "density_radius_um": 10.0, "density_upper_bounds": [8, 16]}
    for example in examples:
        example["diagnostic_context"] = past_annotation_context(example, annotation(), 5.0, cfg)
    metrics = evaluate_tracker_diagnostic(
        ConstantTracker(),
        [collate_past_candidate_examples(examples)],
        torch.device("cpu"),
        gamma=2.0,
        use_amp=False,
        prediction_dir=tmp_path / "predictions",
    )
    assert metrics["positive_edge_recall"] == 1
    assert metrics["diagnostic_slices"]["frame_start"]["sources"] == 1
    assert metrics["diagnostic_slices"]["past_retained"]["sources"] == 1
    assert metrics["diagnostic_slices"]["past_excluded"]["sources"] == 0
    saved = np.load(tmp_path / "predictions" / "44b6_sample" / second.name)
    assert saved["source_ids"].tolist() == [1]
    assert saved["probability"].tolist() == [1.0]
