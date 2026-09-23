from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

from dog_features import (  # noqa: E402
    dog_volumes,
    expand_public_tracker_projection,
    fit_dog_normalizer,
    frame_feature_payload,
    load_window_side_features,
    normalize_frame,
    process_frames_parallel,
    read_cache_role,
    sample_candidates,
    save_frame_features,
    standardize_dog,
)
from frozen_tracker import (  # noqa: E402
    AnnotationGraph,
    FrameAnnotation,
    array_content_sha256,
    array_schema,
    build_window_example,
    diagnose_fixed_window,
    evaluate_dog_progression_gate,
)


def config() -> dict:
    return {
        "spatial_spacing_um": [1.0, 2.0, 2.0],
        "gaussian_pairs_um": [[1.5, 4.0], [2.2, 5.5]],
        "frame_percentiles": [1.0, 99.7],
        "gaussian_mode": "reflect",
        "gaussian_truncate": 4.0,
        "coordinate_tolerance_voxel": 1e-4,
    }


def make_cache(path: Path) -> None:
    path.parent.mkdir(parents=True)
    src = np.array([[3.0, 4.0, 4.0], [3.0, 5.0, 5.0]], dtype=np.float32)
    tgt = np.array([[3.0, 4.0, 4.0], [3.0, 6.0, 6.0]], dtype=np.float32)
    arrays = {
        "candidate_ids_src": np.array([10, 11]),
        "candidate_ids_tgt": np.array([20, 21]),
        "coords_src_grid": src,
        "coords_tgt_grid": tgt,
        "coords_src_physical": src * np.array([1.0, 2.0, 2.0]),
        "coords_tgt_physical": tgt * np.array([1.0, 2.0, 2.0]),
        "candidate_mask_src": np.array([True, True]),
        "candidate_mask_tgt": np.array([True, True]),
        "position_features_src": np.zeros((2, 32), dtype=np.float32),
        "position_features_tgt": np.zeros((2, 32), dtype=np.float32),
        "primary_features_src": np.zeros((2, 32), dtype=np.float32),
        "primary_features_tgt": np.zeros((2, 32), dtype=np.float32),
    }
    meta = {
        "schema_version": 1,
        "experiment": "exp015_oracle_stage_limits",
        "dataset": path.parent.name,
        "window_frames": [0, 1],
        "primary_checkpoint_sha256": "test-checkpoint",
        "array_schema": array_schema(arrays),
        "array_content_sha256": array_content_sha256(arrays),
    }
    np.savez(
        path, **arrays, __metadata_json__=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    )


def test_original_resolution_dog_sign_anisotropy_and_interpolation() -> None:
    frame = np.zeros((9, 13, 13), dtype=np.float32)
    frame[4, 6, 6] = 100
    frame[4, 2, 2] = -100
    cfg = config()
    actual = dog_volumes(frame, cfg)
    normalized = normalize_frame(frame, (1.0, 99.7))
    for index, (small, large) in enumerate(cfg["gaussian_pairs_um"]):
        expected = gaussian_filter(normalized, np.array([small, small, small]) / [1, 2, 2])
        expected -= gaussian_filter(normalized, np.array([large, large, large]) / [1, 2, 2])
        np.testing.assert_allclose(actual[index], expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dog_volumes(np.ones_like(frame), cfg)[0], 0, atol=1e-6)
    grid = np.array([[4.0, 6.5, 6.0]])
    physical = grid * np.array(cfg["spatial_spacing_um"])
    sampled = sample_candidates(
        actual,
        physical,
        grid,
        spacing_um=tuple(cfg["spatial_spacing_um"]),
        downsample_zyx=(1, 1, 1),
        tolerance_voxel=1e-4,
    )
    for component in range(2):
        np.testing.assert_allclose(
            sampled[0, component],
            (actual[component][4, 6, 6] + actual[component][4, 7, 6]) / 2,
            atol=1e-6,
        )
    with pytest.raises(ValueError, match="disagree"):
        sample_candidates(
            actual,
            physical + 2,
            grid,
            spacing_um=tuple(cfg["spatial_spacing_um"]),
            downsample_zyx=(1, 1, 1),
            tolerance_voxel=1e-4,
        )


def test_window_role_identity_fold_fit_and_zero_control(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache" / "44b6_sample" / "000000_000001.npz"
    make_cache(cache_path)
    feature_root = tmp_path / "features"
    for frame, side in ((0, "src"), (1, "tgt")):
        arrays, meta = read_cache_role(cache_path, side)
        image = np.zeros((9, 13, 13), dtype=np.float32)
        image[4, 6, 6] = 10 + frame
        output, roles = frame_feature_payload(
            image, {side: (cache_path, arrays, meta)}, config(), downsample_zyx=(1, 1, 1)
        )
        save_frame_features(
            feature_root / cache_path.parent.name / f"{frame:06d}.npz",
            output,
            {"dataset": cache_path.parent.name, "frame": frame, "roles": roles},
        )
        np.testing.assert_array_equal(
            load_window_side_features(
                feature_root / cache_path.parent.name / f"{frame:06d}.npz", cache_path, side
            ),
            output[f"{side}_responses"],
        )
    normalizer = fit_dog_normalizer([cache_path], feature_root, 1e-6)
    assert normalizer["count"] == 4
    features = load_window_side_features(
        feature_root / cache_path.parent.name / "000000.npz", cache_path, "src"
    )
    assert standardize_dog(features, normalizer, use_values=False).sum() == 0
    annotation = AnnotationGraph(
        frames={
            0: FrameAnnotation(
                np.array([101, 102]), np.array([[3, 8, 8], [3, 10, 10]], dtype=np.float32)
            ),
            1: FrameAnnotation(
                np.array([201, 202]), np.array([[3, 8, 8], [3, 12, 12]], dtype=np.float32)
            ),
        },
        edges=frozenset({(101, 201), (102, 202)}),
        content_sha256="test",
    )
    example = build_window_example(
        cache_path,
        annotation,
        feature_channels=32,
        expected_primary_checkpoint_sha256="test-checkpoint",
        max_matching_distance_um=5.0,
        downsample_zyx=(1, 1, 1),
        dog_feature_root=feature_root,
        dog_normalizer=normalizer,
        use_dog_values=True,
    )
    assert example["features_src"].shape == (2, 66)
    assert example["features_tgt"].shape == (2, 66)
    assert example["known_parent_by_target"].tolist() == [101, 102]
    zero_example = build_window_example(
        cache_path,
        annotation,
        feature_channels=32,
        expected_primary_checkpoint_sha256="test-checkpoint",
        max_matching_distance_um=5.0,
        downsample_zyx=(1, 1, 1),
        dog_feature_root=feature_root,
        dog_normalizer=normalizer,
        use_dog_values=False,
    )
    np.testing.assert_array_equal(zero_example["features_src"][:, -2:], 0)
    with np.load(feature_root / cache_path.parent.name / "000000.npz") as saved:
        bad = {key: saved[key] for key in saved.files}
    bad["src_candidate_ids"] = np.array([11, 10])
    np.savez(feature_root / cache_path.parent.name / "000000.npz", **bad)
    with pytest.raises(ValueError, match="content SHA"):
        load_window_side_features(
            feature_root / cache_path.parent.name / "000000.npz", cache_path, "src"
        )


def test_known_wrong_edges_siblings_and_gate() -> None:
    metadata = {
        "matched_source_ids": np.array([101, 102]),
        "known_parent_by_target": np.array([101, 102]),
        "dog_raw_src": None,
        "dog_raw_tgt": None,
    }
    target = np.array([[1, 0], [0, 1]], dtype=bool)
    predictions = np.array([[1, 1], [0, 1]], dtype=bool)
    diagnostic = diagnose_fixed_window(predictions, target, metadata)
    assert diagnostic["known_wrong_edge_count"] == 1
    assert diagnostic["known_wrong_negative_pair_count"] == 2
    assert diagnostic["known_wrong_sibling_pair_count"] == 1
    assert diagnostic["known_sibling_opportunity_count"] == 2
    cfg = {
        "variants": ["zero", "dog"],
        "strictly_improve": ["positive_edge_recall"],
        "no_decrease": ["edge_accuracy", "division_parent_recall"],
        "no_increase": ["known_wrong_edge_count", "known_wrong_sibling_pair_count"],
        "required_outer_windows": {"6bba": 1, "44b6": 1},
        "required_positive_edges": {"6bba": 2, "44b6": 2},
        "required_division_parents": {"6bba": 1, "44b6": 1},
    }
    base = {
        "window_count": 1,
        "positive_edge_count": 2,
        "division_parent_count": 1,
        "positive_edge_recall": 0.5,
        "edge_accuracy": 0.9,
        "division_parent_recall": 1.0,
        "known_wrong_edge_count": 1,
        "known_wrong_sibling_pair_count": 1,
        "known_wrong_negative_pair_count": 2,
        "known_sibling_opportunity_count": 2,
    }
    better = {
        **base,
        "positive_edge_recall": 1.0,
        "known_wrong_edge_count": 0,
        "known_wrong_sibling_pair_count": 0,
    }
    folds = {"zero": {0: base, 1: base}, "dog": {0: better, 1: better}}
    assert evaluate_dog_progression_gate(folds, cfg)["passed"]
    folds["dog"][1] = {**better, "division_parent_recall": 0.0}
    assert not evaluate_dog_progression_gate(folds, cfg)["passed"]


def test_public_projection_preserves_initial_output() -> None:
    torch = pytest.importorskip("torch")
    public = {
        "proj.weight": torch.arange(128 * 64, dtype=torch.float32).reshape(128, 64),
        "proj.bias": torch.arange(128, dtype=torch.float32),
    }
    expanded = expand_public_tracker_projection(public, 2)
    assert tuple(expanded["proj.weight"].shape) == (128, 66)
    x = torch.randn(4, 64)
    y = torch.cat([x, torch.randn(4, 2)], dim=1)
    torch.testing.assert_close(
        torch.nn.functional.linear(x, public["proj.weight"], public["proj.bias"]),
        torch.nn.functional.linear(y, expanded["proj.weight"], expanded["proj.bias"]),
        rtol=1e-5,
        atol=1e-6,
    )
    assert tuple(public["proj.weight"].shape) == (128, 64)


def test_parallel_frame_execution_preserves_dog_values_and_order() -> None:
    rng = np.random.default_rng(7)
    frames = [rng.normal(size=(9, 13, 13)).astype(np.float32) for _ in range(4)]
    expected = [dog_volumes(frame, config()) for frame in frames]
    actual = process_frames_parallel(
        list(range(len(frames))), lambda index: dog_volumes(frames[index], config()), 2
    )
    for actual_pair, expected_pair in zip(actual, expected, strict=True):
        for actual_response, expected_response in zip(actual_pair, expected_pair, strict=True):
            np.testing.assert_array_equal(actual_response, expected_response)
    with pytest.raises(ValueError, match="duplicate"):
        process_frames_parallel([0, 0], lambda index: index, 2)
    with pytest.raises(ValueError, match="positive"):
        process_frames_parallel([0], lambda index: index, 0)
