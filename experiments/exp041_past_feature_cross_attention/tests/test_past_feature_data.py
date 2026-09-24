from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

import frozen_tracker as base  # noqa: E402
import past_feature_data as data  # noqa: E402


def cache_arrays(source_id: int, target_id: int, *, source_value: float) -> dict[str, np.ndarray]:
    return {
        "candidate_ids_src": np.array([source_id], dtype=np.int64),
        "candidate_ids_tgt": np.array([target_id], dtype=np.int64),
        "coords_src_grid": np.array([[float(source_id), 0, 0]], dtype=np.float32),
        "coords_tgt_grid": np.array([[float(target_id), 0, 0]], dtype=np.float32),
        "coords_src_physical": np.array([[float(source_id), 0, 0]], dtype=np.float32),
        "coords_tgt_physical": np.array([[float(target_id), 0, 0]], dtype=np.float32),
        "position_features_src": np.full((1, 32), source_value, dtype=np.float32),
        "position_features_tgt": np.full((1, 32), source_value + 1, dtype=np.float32),
        "primary_features_src": np.full((1, 32), source_value, dtype=np.float32),
        "primary_features_tgt": np.full((1, 32), source_value + 1, dtype=np.float32),
        "secondary_features_src": np.full((1, 32), source_value + 2, dtype=np.float32),
        "secondary_features_tgt": np.full((1, 32), source_value + 3, dtype=np.float32),
        "candidate_mask_src": np.array([True]),
        "candidate_mask_tgt": np.array([True]),
    }


def make_cache(path: Path, arrays: dict[str, np.ndarray], frames: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "experiment": "exp015_oracle_stage_limits",
        "dataset": path.parent.name,
        "primary_checkpoint_sha256": "test-sha",
        "window_frames": frames,
        "array_schema": base.array_schema(arrays),
        "array_content_sha256": base.array_content_sha256(arrays),
    }
    np.savez_compressed(
        path,
        **arrays,
        __metadata_json__=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
    )


def test_empty_history_frame_is_valid_but_corruption_is_rejected(tmp_path: Path) -> None:
    arrays = cache_arrays(2, 3, source_value=2)
    for suffix in ("src",):
        for key in ("candidate_ids", "candidate_mask"):
            arrays[f"{key}_{suffix}"] = arrays[f"{key}_{suffix}"][:0]
        for key in (
            "coords_src_grid",
            "coords_src_physical",
            "position_features_src",
            "primary_features_src",
            "secondary_features_src",
        ):
            arrays[key] = arrays[key][:0]
    path = tmp_path / "sample" / "000002_000003.npz"
    make_cache(path, arrays, [2, 3])
    loaded, _ = data._load_history_cache(
        path, feature_channels=32, expected_primary_checkpoint_sha256="test-sha"
    )
    assert loaded["candidate_ids_src"].size == 0
    arrays["primary_features_tgt"][0, 0] = 999
    with np.load(path, allow_pickle=False) as saved:
        original = {key: saved[key] for key in saved.files}
    original["primary_features_tgt"] = arrays["primary_features_tgt"]
    np.savez_compressed(path, **original)
    with pytest.raises(ValueError, match="content mismatch"):
        data._load_history_cache(
            path, feature_channels=32, expected_primary_checkpoint_sha256="test-sha"
        )


def test_three_previous_source_windows_and_overlap_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = tmp_path / "sample"
    central = sample / "000005_000006.npz"
    windows = {
        2: cache_arrays(2, 3, source_value=2),
        3: cache_arrays(3, 4, source_value=3),
        4: cache_arrays(4, 5, source_value=4),
        5: cache_arrays(5, 6, source_value=5),
    }
    for frame, arrays in windows.items():
        make_cache(sample / f"{frame:06d}_{frame + 1:06d}.npz", arrays, [frame, frame + 1])
    example = {
        "window_frames": [5, 6],
        "features_src": np.zeros((1, 64), dtype=np.float32),
        "features_tgt": np.zeros((1, 64), dtype=np.float32),
    }
    monkeypatch.setattr(base, "build_window_example", lambda *_args, **_kwargs: dict(example))
    result = data.build_past_feature_window_example(
        central,
        None,
        feature_channels=32,
        expected_primary_checkpoint_sha256="test-sha",
        max_matching_distance_um=5.0,
        downsample_zyx=(1.0, 4.0, 4.0),
        history_frames=3,
    )
    assert result["frames_past"].tolist() == [2, 3, 4]
    assert result["features_past"][:, 0].tolist() == [2, 3, 4]
    assert [entry["candidate_ids"] for entry in result["past_sources"]] == [[2], [3], [4]]
    assert all(entry["content_sha256"] for entry in result["past_sources"])
    windows[4]["coords_tgt_physical"][0, 0] = 999
    make_cache(sample / "000004_000005.npz", windows[4], [4, 5])
    with pytest.raises(ValueError, match="overlapping window coords_physical differs"):
        data.build_past_feature_window_example(
            central,
            None,
            feature_channels=32,
            expected_primary_checkpoint_sha256="test-sha",
            max_matching_distance_um=5.0,
            downsample_zyx=(1.0, 4.0, 4.0),
            history_frames=3,
        )


def test_diagnostic_modes_change_only_mask_or_frame_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    example = {
        "features_src": np.zeros((1, 64), dtype=np.float32),
        "features_tgt": np.zeros((1, 64), dtype=np.float32),
        "features_past": np.arange(3 * 64, dtype=np.float32).reshape(3, 64),
        "coords_past_physical": np.arange(9, dtype=np.float32).reshape(3, 3),
        "frames_past": np.array([2, 3, 4], dtype=np.float32),
        "past_candidate_count": 3,
        "past_frame_count": 3,
        "past_sources": [],
        "window_frames": [5, 6],
        "candidate_ids_src": np.array([5]),
        "candidate_ids_tgt": np.array([6]),
        "coords_src_physical": np.zeros((1, 3), dtype=np.float32),
        "coords_tgt_physical": np.zeros((1, 3), dtype=np.float32),
    }

    def base_collate(_examples: object) -> dict[str, object]:
        return {
            "features_src": torch.zeros(1, 1, 64),
            "features_tgt": torch.zeros(1, 1, 64),
            "history_present_src": torch.zeros(1, 1, dtype=torch.bool),
            "metadata": [{}],
        }

    monkeypatch.setattr(base, "collate_window_examples", base_collate)
    full = data.collate_past_feature_examples([dict(example, diagnostic_mode="full")])
    absent = data.collate_past_feature_examples([dict(example, diagnostic_mode="no_past")])
    reversed_time = data.collate_past_feature_examples(
        [dict(example, diagnostic_mode="reverse_time")]
    )
    assert full["past_mask"].tolist() == [[True, True, True]]
    assert absent["past_mask"].tolist() == [[False, False, False]]
    assert reversed_time["frames_past"].tolist() == [[4.0, 3.0, 2.0]]
    torch.testing.assert_close(full["features_past"], reversed_time["features_past"])
    torch.testing.assert_close(full["coords_past_physical"], reversed_time["coords_past_physical"])
