from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from trackastra_association import (  # noqa: E402
    AnnotationGraph,
    FrameAnnotation,
    TrackingTransformer,
    accumulate_adjacent_scores,
    build_context_window,
    build_sparse_teacher,
    eligible_pair_frames,
    masked_association_loss,
    mean_adjacent_scores,
    parental_log_probabilities,
    parental_softmax,
    probability_candidate_edges,
    select_internal_threshold,
    training_window_schedule,
    window_loss,
)


def annotation() -> AnnotationGraph:
    frames = {
        0: FrameAnnotation(np.array([1]), np.array([[0.0, 0.0, 0.0]])),
        1: FrameAnnotation(np.array([2, 3, 9]), np.zeros((3, 3))),
        2: FrameAnnotation(np.array([4, 5, 10]), np.zeros((3, 3))),
    }
    # 1 divides into 2/3; 9 and 10 are unrelated known cells.
    return AnnotationGraph(
        frames=frames,
        edges=frozenset({(1, 2), (1, 3), (2, 4), (3, 5), (9, 10)}),
        content_sha256="synthetic",
    )


def test_sparse_teacher_preserves_divisions_and_unknown() -> None:
    times = np.array([0, 1, 1, 1, 1, 2, 2, 2])
    matched = np.array([1, 2, 3, 9, -1, 4, 5, 10])
    xyz = np.zeros((len(times), 3), dtype=np.float32)
    result = build_sparse_teacher(times, xyz, matched, annotation(), cutoff_um=5.0)
    target, mask, weight = result["target"], result["mask"], result["weight"]
    assert target[0, 1] == target[0, 2] == 1
    assert weight[0, 1] == weight[0, 2] == 11
    assert target[0, 5] == target[0, 6] == 1
    assert target[3, 7] == 1
    assert mask[3, 5] and target[3, 5] == 0
    assert not mask[4, 1] and not mask[0, 4]
    assert mask[2, 5] and target[2, 5] == 0  # Sibling 3 is a known wrong parent.
    assert result["stats"]["positive_dt2"] == 2


def test_parental_softmax_retains_abstention_and_unknown_competition() -> None:
    logits = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)
    times = torch.tensor([0, 0, 1])
    xyz = torch.zeros((3, 3))
    probabilities = parental_softmax(logits, times, xyz, cutoff_um=5.0)
    torch.testing.assert_close(probabilities[:2, 2], torch.full((2,), 1 / 3))
    teacher = {
        "target": np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.float32),
        "mask": np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=bool),
        "weight": np.array([[0, 0, 2], [0, 0, 0], [0, 0, 0]], dtype=np.float32),
    }
    log_probabilities, log_complements = parental_log_probabilities(
        logits, times, xyz, cutoff_um=5.0, include_complement=True
    )
    loss = masked_association_loss(logits, log_probabilities, log_complements, teacher)
    assert loss is not None
    assert loss.item() == pytest.approx(2 * (np.log(3) + 0.01 * np.log(2)))
    loss.backward()
    assert logits.grad[1, 2] > 0  # Unknown parent competes in the denominator.
    assert logits.grad[2, 0] == 0


def test_masked_loss_keeps_gradients_for_near_certain_mistakes() -> None:
    times = torch.tensor([0, 0, 1])
    xyz = torch.zeros((3, 3))
    for target_row, expected_gradient in ((0, -2.0), (1, 1.0)):
        logits = torch.zeros((3, 3), requires_grad=True)
        with torch.no_grad():
            logits[1, 2] = 20.0
        log_probabilities, log_complements = parental_log_probabilities(
            logits, times, xyz, cutoff_um=5.0, include_complement=True
        )
        target = np.zeros((3, 3), dtype=np.float32)
        mask = np.zeros((3, 3), dtype=bool)
        weight = np.zeros((3, 3), dtype=np.float32)
        target[target_row, 2] = float(target_row == 0)
        mask[target_row, 2] = True
        weight[target_row, 2] = 2.0 if target_row == 0 else 1.0
        loss = masked_association_loss(
            logits,
            log_probabilities,
            log_complements,
            {"target": target, "mask": mask, "weight": weight},
            auxiliary_weight=0.0,
        )
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        assert logits.grad[target_row, 2].item() == pytest.approx(expected_gradient, abs=1e-5)
        assert torch.isfinite(logits.grad).all()


def test_context_features_stay_inside_window_and_overlap_means() -> None:
    pairs = []
    registry = {}
    for frame in range(4):
        registry[frame] = {
            "ids": np.array([frame], dtype=np.int64),
            "physical": np.array([[frame, 0, 0]], dtype=np.float32),
            "grid": np.array([[frame, 0, 0]], dtype=np.float32),
        }
    for frame in range(3):
        pairs.append(
            {
                "frames": (frame, frame + 1),
                "arrays": {
                    "candidate_ids_src": np.array([frame]),
                    "candidate_ids_tgt": np.array([frame + 1]),
                    "primary_features_src": np.full((1, 32), frame, dtype=np.float32),
                    "primary_features_tgt": np.full((1, 32), frame + 10, dtype=np.float32),
                },
            }
        )
    video = {"sample": "a", "frames": list(range(4)), "pairs": pairs, "registry": registry}
    left = build_context_window(video, 0, window_size=3)
    right = build_context_window(video, 1, window_size=3)
    assert left["features"][:, 0].tolist() == [0, 1, 11]
    assert right["features"][:, 0].tolist() == [1, 2, 12]
    left_scores = np.zeros((3, 3), dtype=np.float32)
    right_scores = np.zeros((3, 3), dtype=np.float32)
    left_scores[1, 2] = 0.2
    right_scores[0, 1] = 0.6
    sums = {}
    accumulate_adjacent_scores(sums, left, left_scores)
    accumulate_adjacent_scores(sums, right, right_scores)
    means = mean_adjacent_scores(sums)
    assert means[(1, 1, 2)] == pytest.approx(0.4)
    edges = probability_candidate_edges(means, video, threshold=0.35)
    assert (1, 2, pytest.approx(0.4), pytest.approx(1.0)) in edges


def test_threshold_keeps_false_alarm_count() -> None:
    assert select_internal_threshold(np.array([0.1, 0.4, 0.4, 0.9]), 1) == 0.4
    assert select_internal_threshold(np.array([0.1, 0.4, 0.4, 0.9]), 0) == 0.9


def test_model_has_encoder_decoder_rope_and_finite_logits() -> None:
    model = TrackingTransformer(
        coord_dim=3,
        feat_dim=32,
        d_model=128,
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dropout=0.0,
        window=6,
        spatial_pos_cutoff=256,
        attn_positional_bias="rope",
        causal_norm="quiet_softmax",
    ).eval()
    coords = torch.tensor([[[0.0, 0, 0, 0], [1.0, 1, 0, 0]]])
    with torch.no_grad():
        logits = model(coords, torch.zeros((1, 2, 32)))
    assert logits.shape == (1, 2, 2)
    assert torch.isfinite(logits).all()
    assert len(model.encoder) == len(model.decoder) == 4
    teacher = {
        "target": np.array([[0, 1], [0, 0]], dtype=np.float32),
        "mask": np.array([[0, 1], [0, 0]], dtype=bool),
        "weight": np.array([[0, 2], [0, 0]], dtype=np.float32),
    }
    window = {
        "coords": coords[0].numpy(),
        "features": np.zeros((2, 32), dtype=np.float32),
        "times": np.array([0, 1]),
        "physical": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        "teacher": teacher,
    }
    loss = window_loss(model, window, torch.device("cpu"), cutoff_um=256, auxiliary_weight=0.01)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


def test_pair_evaluation_excludes_empty_gt_frames() -> None:
    video = {"pairs": [{"frames": (0, 1)}, {"frames": (1, 2)}, {"frames": (2, 3)}]}
    annotated = annotation()
    assert eligible_pair_frames(video, annotated) == {0, 1}


def test_training_window_schedule_balances_teachers_and_rotates_coverage() -> None:
    manifest = []
    teacher_types = ("division", "two_step", "adjacent", "known_negative_only")
    for group, teacher_type in enumerate(teacher_types):
        for index in range(4):
            manifest.append(
                {
                    "sample": "training_embryo_video",
                    "start": group * 4 + index,
                    "tokens": 10 * (group + 1),
                    "positive_dt1": int(teacher_type in ("division", "adjacent")),
                    "positive_dt2": int(teacher_type == "two_step"),
                    "division_positive": int(teacher_type == "division"),
                    "known_negative": 1,
                }
            )
    result = training_window_schedule(manifest, windows_per_epoch=8, epochs=2, seed=42)
    assert result == training_window_schedule(
        list(reversed(manifest)), windows_per_epoch=8, epochs=2, seed=42
    )
    assert len(result["epochs"]) == 2
    for epoch in result["epochs"]:
        assert len(epoch) == len({(row["sample"], row["start"]) for row in epoch}) == 8
        assert {
            kind: sum(row["teacher_type"] == kind for row in epoch) for kind in teacher_types
        } == {kind: 2 for kind in teacher_types}
    assert len({row["start"] for epoch in result["epochs"] for row in epoch}) == 16
    with pytest.raises(ValueError, match="exceeds"):
        training_window_schedule(manifest, windows_per_epoch=17, epochs=1, seed=42)


def test_training_window_schedule_retains_rare_division_window() -> None:
    manifest = [
        {
            "sample": "44b6_training_video",
            "start": index,
            "tokens": 100 + index,
            "positive_dt1": 1,
            "positive_dt2": 0,
            "division_positive": int(index == 0),
            "known_negative": 1,
        }
        for index in range(101)
    ]
    schedule = training_window_schedule(manifest, windows_per_epoch=4, epochs=2, seed=42)
    assert all(
        any(row["teacher_type"] == "division" for row in epoch) for epoch in schedule["epochs"]
    )


def test_runtime_budget_asset_matches_fixed_sampling_config() -> None:
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    training = config["model"]["training"]
    asset_path = EXP / training["runtime_budget_asset"]
    data = asset_path.read_bytes()
    budget = json.loads(data)
    assert hashlib.sha256(data).hexdigest() == training["runtime_budget_asset_sha256"]
    assert not training["profiling_only"]
    assert budget["windows_per_epoch_per_fold"] == training["windows_per_epoch"] == 4096
    assert budget["runtime_multiplier"] == training["runtime_projection_multiplier"]
    assert budget["projected_seconds"] < budget["runtime_gate_seconds"] == 43200
    assert set(budget["training_window_manifest_sha256_by_fold"]) == {"0", "1"}
    assert set(training["expected_training_schedule_sha256_by_fold"]) == {0, 1}
    assert all(
        len(value) == 64 for value in training["expected_training_schedule_sha256_by_fold"].values()
    )
    assert all(
        fold["distinct_training_windows"] == fold["population_windows"]
        for fold in budget["folds"].values()
    )
