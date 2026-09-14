from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

from app.tracking_data import (
    analyze_sequence,
    load_candidates,
    load_geff,
    match_frame,
    read_projection,
)
from app.tracking_plots import detail_table, spatial_figure


def nodes(ids, positions, times=None):
    table = pd.DataFrame(positions, columns=["z", "y", "x"])
    table["node_id"] = ids
    table["t"] = times if times is not None else 0
    return table


def write_window(folder, times, a, b, scores=None, masks=None):
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "dataset": folder.name,
        "window_frames": times,
        "coordinate_units": {"physical": "micrometer"},
    }
    arrays = {"__metadata_json__": np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)}
    for i, (side, frame) in enumerate(zip(["src", "tgt"], [a, b], strict=True)):
        arrays[f"candidate_ids_{side}"] = frame.node_id.to_numpy(dtype=np.int64)
        arrays[f"coords_{side}_physical"] = frame[["z", "y", "x"]].to_numpy(dtype=float)
        arrays[f"candidate_mask_{side}"] = (
            np.ones(len(frame), dtype=bool) if masks is None else masks[i]
        )
        arrays[f"detection_scores_{side}"] = np.full(
            len(frame), 0.9 if scores is None else scores[i]
        )
    np.savez(folder / f"{times[0]:06d}_{times[1]:06d}.npz", **arrays)


def write_geff(path):
    group = zarr.open_group(str(path), mode="w")
    for name, data in {
        "nodes/ids": [10, 20, 30, 40],
        "nodes/props/t/values": [0, 1, 2, 2],
        "nodes/props/z/values": [2, 2, 2, 2],
        "nodes/props/y/values": [3, 3, 3, 4],
        "nodes/props/x/values": [4, 5, 6, 5],
        "edges/ids": [[10, 20], [20, 30], [20, 40]],
    }.items():
        group.create_array(name, data=np.array(data))


def test_assignment_avoids_greedy_conflict_and_is_inclusive():
    gt = nodes([10, 11], [[0, 0, 0], [0, 0, 2]])
    det = nodes([20, 21], [[0, 0, 1], [0, 0, -1]])
    result = match_frame(det, gt, 1)
    assert dict(zip(result.gt_id, result.detection_id, strict=True)) == {10: 21, 11: 20}
    assert result.distance_um.tolist() == [1, 1]
    assert result.candidates_in_radius.tolist() == [2, 1]


def test_unmatched_has_nearest_distance_without_inventing_a_match():
    gt = nodes([10, 11], [[0, 0, 0], [0, 0, 20]])
    det = nodes([20], [[0, 0, 1]])
    result = match_frame(det, gt, 7)
    assert result.detection_id.iloc[0] == 20
    assert pd.isna(result.detection_id.iloc[1])
    assert result.nearest_distance_um.tolist() == [1, 19]
    assert match_frame(det.iloc[:0], gt, 7).detection_id.isna().all()
    assert match_frame(det, gt.iloc[:0], 7).empty
    with pytest.raises(ValueError, match="one timepoint"):
        match_frame(det.assign(t=1), gt, 7)


def test_cache_mask_overlap_score_context_and_inconsistent_coordinates(tmp_path):
    folder = tmp_path / "sample"
    a = nodes([1, 99], [[1, 2, 3], [999, 999, 999]])
    b = nodes([2], [[2, 3, 4]])
    c = nodes([3], [[3, 4, 5]])
    write_window(folder, [0, 1], a, b, masks=[np.array([True, False]), np.array([True])])
    write_window(folder, [1, 2], b, c, scores=[0.8, 0.9])
    det, receipt = load_candidates(folder)
    assert det.node_id.tolist() == [1, 2, 3]
    assert det.t.tolist() == [0, 1, 2]
    assert det.score.tolist() == [0.9, 0.9, 0.9]
    assert receipt["score_changed_frames"] == 1
    write_window(folder, [1, 2], b.assign(x=999), c)
    with pytest.raises(ValueError, match="Overlapping windows disagree"):
        load_candidates(folder)


def test_empty_candidate_frame_remains_observed(tmp_path):
    folder = tmp_path / "sample"
    a, b = nodes([], []), nodes([1], [[0, 0, 0]])
    write_window(folder, [0, 1], a, b)
    det, receipt = load_candidates(folder)
    gt = nodes([10, 11, 12], [[0, 0, 0]] * 3, [0, 1, 2])
    matches, summary = analyze_sequence(det, gt, receipt["frames"], 7)
    table = detail_table(gt, det, matches)
    assert summary["未対応の正解"].tolist() == [1, 0]
    assert table.state.tolist() == ["未対応", "対応あり", "キャッシュなし"]


def test_geff_scale_division_and_daughters_share_lineage(tmp_path):
    path = tmp_path / "sample.geff"
    write_geff(path)
    gt, edges = load_geff(path, (1.625, 0.40625, 0.40625))
    assert gt.iloc[0].z == 3.25
    assert gt.iloc[0].x == 1.625
    assert gt.lineage.tolist() == [10, 10, 10, 10]
    assert gt[gt.division].node_id.tolist() == [20]
    matches, _ = analyze_sequence(gt.assign(score=1), gt, [0, 1, 2], 7)
    fig = spatial_figure(gt, gt, edges, matches, t=2, plane="3D", lineage=10)
    trail = next(trace for trace in fig.data if trace.name == "正解の軌跡・分裂")
    assert len(trail.x) == 9  # All three real graph edges, including both daughters.


def test_projection_keeps_axes_and_includes_last_z_slice(tmp_path):
    path = tmp_path / "sample.zarr"
    volume = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    zarr.open_group(str(path), mode="w").create_array("0", data=volume)
    for plane, axis in [("XY", 0), ("XZ", 1), ("YZ", 2)]:
        np.testing.assert_array_equal(
            read_projection(path, 1, plane, (1, 2)), volume[1, 1:3].max(axis=axis)
        )
    with pytest.raises(ValueError, match="outside"):
        read_projection(path, 2, "XY", (0, 2))


def test_notebook_controls_recompute_and_preserve_lineage(tmp_path, monkeypatch):
    from app import tracking_notebook as module

    monkeypatch.setattr(module, "display", lambda *args: None)
    monkeypatch.setattr(module, "clear_output", lambda **kwargs: None)
    train, cache = tmp_path / "train", tmp_path / "window_cache"
    train.mkdir()
    write_geff(train / "sample.geff")
    gt, _ = load_geff(train / "sample.geff", (1.0, 1.0, 1.0))
    det = gt.copy()
    det["node_id"] += 100
    for a, b in [(0, 1), (1, 2)]:
        write_window(cache / "sample", [a, b], det[det.t == a], det[det.t == b])
    volume = np.ones((3, 5, 8, 8), dtype=np.uint16)
    zarr.open_group(str(train / "sample.zarr"), mode="w").create_array("0", data=volume)
    config = {
        "voxel_scale_um": [1.0, 1.0, 1.0],
        "matching_radius_um": 7.0,
        "minimum_detection_score": 0.0,
        "trail_frames": 5,
        "playback_interval_seconds": 0.7,
    }
    viewer = module.TrackingNotebookViewer(cache, train, config)
    assert viewer.matches.detection_id.notna().sum() == 4
    assert viewer.play.layout.display == "none"
    assert "検出キャッシュあり: 1件 / 正解 train: 1件" in viewer.dataset_note.value
    assert all(
        button.icon == ""
        for button in [
            viewer.previous_frame,
            viewer.play_pause,
            viewer.next_frame,
            viewer.export,
        ]
    )
    viewer.node.value = 20
    assert viewer.frame.value == 1
    assert viewer.play.value == 1
    assert viewer.lineage.value == 10
    viewer.next_frame.click()
    assert viewer.frame.value == 2
    viewer.previous_frame.click()
    assert viewer.frame.value == 1
    viewer.play_pause.click()
    assert viewer.play.playing
    assert viewer.play_pause.description == "一時停止"
    viewer.play_pause.click()
    assert not viewer.play.playing
    assert viewer.play_pause.description == "再生"
    viewer.plane.value = "XZ"
    viewer.zrange.value = (1, 2)
    assert viewer.matches.detection_id.notna().sum() == 4
    viewer.score.value = 1.0
    assert viewer.matches.detection_id.isna().all()
    viewer._jump(1)
    assert viewer.frame.value == 2
    monkeypatch.chdir(tmp_path)
    viewer._export()
    assert pd.read_csv(tmp_path / "tracking_matches_sample.csv").state.eq("未対応").all()
    viewer.play.value = 0
    viewer._play_frame({"new": 0})
    assert viewer.frame.value == 0


def test_local_streamlit_no_gt_is_explicit(tmp_path):
    from streamlit.testing.v1 import AppTest

    folder = tmp_path / "window_cache/sample"
    write_window(folder, [0, 1], nodes([1], [[2, 3, 4]]), nodes([2], [[3, 4, 5]]))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/tracking_viewer.py"))
    app.run(timeout=30)
    app.sidebar.text_input[0].set_value(str(folder.parent))
    app.sidebar.text_input[1].set_value(str(tmp_path / "train"))
    app.run(timeout=30)
    assert not app.exception
    assert any("正解 GEFF はローカルにありません" in info.value for info in app.info)
    assert app.metric[0].value == "1"
    app.button[1].click().run(timeout=30)
    assert not app.exception
    assert app.select_slider[0].value == 1


def test_kaggle_viewer_uses_exp015_train_cache_only():
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads((root / "app/kaggle/kernel-metadata.json").read_text())
    assert metadata["kernel_sources"] == ["kentookumura/exp015-oracle-stage-limits-inference"]
