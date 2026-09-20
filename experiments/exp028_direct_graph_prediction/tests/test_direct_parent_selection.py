from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from exp028_direct_graph_prediction_train import (  # noqa: E402
    build_partial_teacher,
    decode_parent_choices,
    greedy_match_candidates,
    selection_counts,
)


def test_partial_teacher_distinguishes_known_parent_missing_candidate_and_unknown() -> None:
    source_matches = np.array([10, -1, 20])
    target_matches = np.array([11, 21, 31, 41, -1])
    incoming = {11: [10], 21: [999], 31: [10, 20], 41: []}
    labels, stats = build_partial_teacher(source_matches, target_matches, incoming)
    assert labels.tolist() == [0, 3, -1, -1, -1]
    assert stats == {
        "known_parent": 1,
        "known_parent_outside_candidates": 1,
        "unknown": 2,
        "ambiguous_incoming": 1,
    }


def test_greedy_matching_keeps_one_candidate_per_gt() -> None:
    candidates = np.array([[0.0, 0, 0], [0.2, 0, 0], [9.0, 0, 0]])
    gt = [(7, np.array([0.0, 0, 0]))]
    assert greedy_match_candidates(candidates, gt, 5.0).tolist() == [7, -1, -1]


def test_direct_decode_enforces_parent_capacity_and_unique_child() -> None:
    source = np.array([100, 200])
    target = np.array([11, 12, 13, 14])
    logits = np.array(
        [
            [7.0, -1.0, -3.0],
            [7.0, -1.0, -3.0],
            [7.0, 6.0, -3.0],
            [-2.0, 7.0, -3.0],
        ]
    )
    choices, probabilities = decode_parent_choices(logits, source, target, threshold=0.2)
    assert choices.tolist() == [0, 0, 1, 1]
    assert np.all(probabilities > 0)
    assert len(choices) == len(target)


def test_null_and_ties_have_deterministic_results() -> None:
    choices, _ = decode_parent_choices(
        np.array([[1.0, 1.0, -2.0], [0.0, 0.0, 5.0]]),
        np.array([9, 3]),
        np.array([8, 7]),
        threshold=0.2,
    )
    assert choices.tolist() == [1, -1]
    choices, _ = decode_parent_choices(
        np.zeros((2, 1)), np.array([], dtype=int), np.array([1, 2]), threshold=0.5
    )
    assert choices.tolist() == [-1, -1]


def test_sparse_metric_counts_only_observable_conflicts() -> None:
    report = selection_counts(np.array([0, -1, 0, 0]), np.array([0, 2, 2, -1]), 2)
    assert report["known_edges"] == 1
    assert report["known_null"] == 2
    assert report["true_positive_edges"] == 1
    assert report["observable_false_edges"] == 1
    assert report["unknown_children"] == 1


def test_duplicate_candidate_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        decode_parent_choices(np.zeros((1, 3)), np.array([1, 1]), np.array([2]), threshold=0.5)


def test_split_matches_parent_internal_sample_rule() -> None:
    from exp028_direct_graph_prediction_train import split_samples

    names = [f"44b6_{index:04d}" for index in range(71)] + [
        f"6bba_{index:04d}" for index in range(128)
    ]
    split = split_samples(names, "44b6", "6bba", 0, 0.1)
    assert [len(split[key]) for key in ("gradient", "internal", "outer")] == [64, 7, 128]
    assert len(set(split["gradient"]) & set(split["internal"])) == 0


def test_window_candidate_identity_is_checked() -> None:
    from exp028_direct_graph_prediction_inference import register_frame

    registry = {}
    register_frame(
        registry,
        frame=0,
        ids=np.array([0, 1]),
        grid=np.array([[1, 2, 3], [4, 5, 6]]),
    )
    register_frame(
        registry,
        frame=0,
        ids=np.array([0, 1]),
        grid=np.array([[1, 2, 3], [4, 5, 6]]),
    )
    with pytest.raises(RuntimeError, match="candidate_id_changed"):
        register_frame(registry, frame=0, ids=np.array([0]), grid=np.array([[9, 2, 3]]))


def test_pinned_repair_patch_skips_original_prediction() -> None:
    from exp028_direct_graph_prediction_inference import patched_repair_source

    original = (
        EXP.parent / "exp015_oracle_stage_limits" / "exp015_oracle_stage_limits_inference.py"
    ).read_text()
    patched = patched_repair_source(original)
    assert "run_direct_prediction(REPO_DIR, METHOD, test_stems)" in patched
    assert "available_gpu_count = _torch.cuda.device_count()" not in patched
    assert "DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()" in patched
    assert "print(f'Wrote {RUN_STATS_PATH}')" in patched


def test_saved_control_counts_conflicting_edges_without_unknown_negative() -> None:
    from exp028_direct_graph_prediction_train import control_window_counts

    selected = np.array(
        [
            [True, True, False, True],
            [False, False, True, False],
        ]
    )
    labels = np.array([0, 2, 1, -1])
    report = control_window_counts(selected, labels)
    assert report["known_edges"] == 2
    assert report["known_null"] == 1
    assert report["true_positive_edges"] == 2
    assert report["observable_false_edges"] == 1
    assert report["selected_edges"] == 4


def test_early_gate_requires_both_embryos_and_each_metric() -> None:
    from exp028_direct_graph_prediction_train import assess_early_gate

    config = {
        "validation": {
            "early_gate": {
                "maximum_structural_violations": 0,
                "minimum_known_division_parents": 1,
                "require_same_unit_control": True,
                "maximum_known_edge_recall_drop": 0.01,
                "maximum_observable_false_edge_rate_increase": 0.0,
                "minimum_division_recall_delta": 0.0,
            }
        }
    }
    baseline = {
        "known_edge_recall": 0.9,
        "observable_false_edge_rate": 0.1,
        "known_division_recall": 0.2,
        "known_edges": 100,
        "known_null": 10,
        "known_division_parents": 10,
    }
    direct = {
        "known_edge_recall": 0.9,
        "observable_false_edge_rate": 0.1,
        "known_division_recall": 0.2,
        "structural_violations": 0,
        "known_division_parents": 10,
        "known_edges": 100,
        "known_null": 10,
    }
    rows = [
        {
            "fold": fold,
            "evaluation_embryo": embryo,
            "outer": dict(direct),
            "control_two_frame": dict(baseline),
        }
        for fold, embryo in ((0, "6bba"), (1, "44b6"))
    ]
    assert assess_early_gate(config, rows)["pass"] is True
    rows[1]["outer"]["known_division_recall"] = 0.1
    assert assess_early_gate(config, rows)["pass"] is False
    rows[1]["outer"]["known_division_recall"] = 0.2
    rows[1]["outer"]["known_edges"] = 99
    assert assess_early_gate(config, rows)["pass"] is False


def test_window_reader_uses_saved_score_and_consistent_features(tmp_path: Path) -> None:
    from exp028_direct_graph_prediction_inference import read_window as inference_read_window
    from exp028_direct_graph_prediction_train import read_window as train_read_window

    window = tmp_path / "44b6_0001" / "000000_000001.npz"
    window.parent.mkdir()
    metadata = {
        "schema_version": 1,
        "experiment": "exp015_oracle_stage_limits",
        "dataset": window.parent.name,
        "window_frames": [0, 1],
        "primary_checkpoint_sha256": "fixed-checkpoint",
    }
    arrays = {
        "__metadata_json__": np.frombuffer(
            __import__("json").dumps(metadata).encode(), dtype=np.uint8
        )
    }
    for side, node_id, score in (("src", 10, 0.25), ("tgt", 11, 0.75)):
        arrays[f"candidate_ids_{side}"] = np.array([node_id])
        arrays[f"coords_{side}_physical"] = np.array([[1.0, 2.0, 3.0]])
        arrays[f"coords_{side}_grid"] = np.array([[1.0, 2.0, 3.0]])
        arrays[f"primary_features_{side}"] = np.full((1, 32), 2.0)
        arrays[f"position_features_{side}"] = np.full((1, 32), 3.0)
        arrays[f"detection_scores_{side}"] = np.array([score])
    np.savez(window, **arrays)
    trained = train_read_window(window, "fixed-checkpoint")
    inferred = inference_read_window(window, "fixed-checkpoint")
    for side, score in (("src", 0.25), ("tgt", 0.75)):
        features = trained[f"features_{side}"]
        assert features.shape == (1, 65)
        assert features[0, -1] == pytest.approx(score)
        np.testing.assert_array_equal(features, inferred[f"features_{side}"])
    arrays["detection_scores_tgt"] = np.array([float("nan")])
    np.savez(window, **arrays)
    with pytest.raises(RuntimeError, match="invalid_detection_scores"):
        train_read_window(window, "fixed-checkpoint")


def test_compact_graph_conversion_matches_public_evaluator_coordinates() -> None:
    from exp028_direct_graph_prediction_inference import normalise_compact_graph_arrays

    nodes, edges = normalise_compact_graph_arrays(
        np.array([9, 5]),
        np.array([[0, -0.3, 1.6, 2.4], [1, 3.1, 4.7, 5.2]]),
        np.array([[9, 5]]),
    )
    assert nodes == [
        {"t": 0, "z": 0, "y": 2, "x": 2},
        {"t": 1, "z": 3, "y": 5, "x": 5},
    ]
    assert edges == [(0, 1)]
    with pytest.raises(ValueError, match="dangling"):
        normalise_compact_graph_arrays(np.array([9, 5]), np.zeros((2, 4)), np.array([[9, 99]]))
