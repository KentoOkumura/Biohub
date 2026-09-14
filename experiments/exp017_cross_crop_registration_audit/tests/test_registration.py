from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml

EXP = Path(__file__).resolve().parents[1]
SOURCE = EXP / "exp017_cross_crop_registration_audit_audit.py"


def load_audit_module():
    module_name = "exp017_cross_crop_registration_audit_audit"
    spec = importlib.util.spec_from_file_location(module_name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def correlated_crops() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(123)
    scene = generator.normal(size=(96, 96))
    return scene[:64, :64], scene[10:74, 15:79]


def test_cpu_only_audit_contract():
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    assert config["experiment"]["notebooks"] == ["audit"]
    assert config["lineage"]["hypothesis_id"] == "N/A"
    assert config["lineage"]["backlog_candidate"] == "N/A"
    assert config["model"]["name"] == "none_image_registration_audit"
    assert config["runtime"]["kaggle"]["enable_gpu"] is False
    assert config["runtime"]["kaggle"]["enable_internet"] is False
    assert config["runtime"]["kaggle"]["audit"]["enable_gpu"] is False
    source = SOURCE.read_text()
    assert "submission.csv" not in source
    assert "torch" not in source
    assert "__file__" not in source


def test_best_ncc_shift_recovers_partial_overlap_translation():
    audit = load_audit_module()
    reference, moving = correlated_crops()
    result = audit.best_ncc_shift(reference, moving, min_overlap_fraction=0.2)
    assert (result["shift_y"], result["shift_x"]) == (10, 15)
    assert result["ncc"] > 0.999999
    assert result["overlap_fraction"] == (54 * 49) / (64 * 64)


def test_phase_candidate_recovers_equivalent_partial_overlap_shift():
    audit = load_audit_module()
    reference, moving = correlated_crops()
    base_shift, _ = audit.phase_shift_from_ffts(np.fft.fft2(reference), np.fft.fft2(moving))
    result = audit.select_phase_shift(
        reference,
        moving,
        base_shift,
        min_overlap_fraction=0.2,
    )
    assert (result["shift_y"], result["shift_x"]) == (10, 15)
    assert result["ncc"] > 0.999999


def test_d4_transform_and_shift_can_be_recovered():
    audit = load_audit_module()
    reference, aligned_moving = correlated_crops()
    rotated_moving = audit.apply_d4(aligned_moving, "rot270")
    restored = audit.apply_d4(rotated_moving, "rot90")
    result = audit.best_ncc_shift(reference, restored, min_overlap_fraction=0.2)
    assert (result["shift_y"], result["shift_x"]) == (10, 15)
    assert result["ncc"] > 0.999999


def test_coordinate_transforms_match_array_d4_operations():
    audit = load_audit_module()
    size = 7
    for transform in (
        "identity",
        "rot90",
        "rot180",
        "rot270",
        "flip_x",
        "flip_y",
        "transpose",
        "anti_transpose",
    ):
        for y, x in ((0, 0), (1, 5), (6, 2)):
            array = np.zeros((size, size), dtype=np.int8)
            array[y, x] = 1
            transformed = audit.apply_d4(array, transform)
            expected_y, expected_x = np.argwhere(transformed == 1)[0]
            observed_y, observed_x = audit.transform_xy_coordinates(
                np.asarray([y]),
                np.asarray([x]),
                transform,
                size,
                size,
            )
            assert (int(observed_y[0]), int(observed_x[0])) == (
                int(expected_y),
                int(expected_x),
            )


def test_best_z_shift_uses_fixed_xy_registration():
    audit = load_audit_module()
    generator = np.random.default_rng(456)
    scene = generator.normal(size=(24, 40, 40))
    reference = scene[:16, :32, :32]
    moving = scene[3:19, 4:36, 5:37]
    result = audit.best_z_shift(
        reference,
        moving,
        shift_y=4,
        shift_x=5,
        min_overlap_fraction=0.2,
    )
    assert (result["shift_z"], result["shift_y"], result["shift_x"]) == (3, 4, 5)
    assert result["ncc"] > 0.999999


def test_unrelated_noise_does_not_look_like_exact_registration():
    audit = load_audit_module()
    generator = np.random.default_rng(789)
    reference = generator.normal(size=(64, 64))
    moving = generator.normal(size=(64, 64))
    result = audit.best_ncc_shift(reference, moving, min_overlap_fraction=0.2)
    assert result["ncc"] < 0.25


def test_identical_nonempty_repeat_schedules_are_selected_as_pairs():
    audit = load_audit_module()
    rows = [
        {"sample_id": "6bba_a", "embryo_id": "6bba", "start_frame": 2, "end_frame": 3},
        {"sample_id": "6bba_a", "embryo_id": "6bba", "start_frame": 8, "end_frame": 9},
        {"sample_id": "6bba_b", "embryo_id": "6bba", "start_frame": 2, "end_frame": 3},
        {"sample_id": "6bba_b", "embryo_id": "6bba", "start_frame": 8, "end_frame": 9},
        {"sample_id": "6bba_c", "embryo_id": "6bba", "start_frame": 2, "end_frame": 3},
        {"sample_id": "44b6_a", "embryo_id": "44b6", "start_frame": 2, "end_frame": 3},
    ]
    assert audit.identical_freeze_schedule_pairs(rows) == {("6bba_a", "6bba_b")}
