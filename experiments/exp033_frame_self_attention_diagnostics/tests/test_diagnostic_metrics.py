from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

EXPERIMENT = Path(__file__).resolve().parents[1]


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics = load_module("diagnostic_metrics")


def test_pr_groups_tied_scores_before_precision_gate():
    curve = metrics.precision_recall_by_score(
        np.array([0.9, 0.9, 0.8], dtype=np.float32),
        np.array([1, 0, 1], dtype=np.bool_),
        0.95,
    )
    assert curve["threshold"] == pytest.approx([0.9, 0.8])
    assert curve["precision"] == pytest.approx([0.5, 2 / 3])
    assert curve["target_reachable"] is False
    assert curve["recall_at_target_precision"] is None


def test_pr_recall_at_fixed_precision_and_empty_positive_case():
    curve = metrics.precision_recall_by_score(
        np.array([0.9, 0.8, 0.7], dtype=np.float32),
        np.array([1, 1, 0], dtype=np.bool_),
        0.95,
    )
    assert curve["recall_at_target_precision"] == 1.0
    assert curve["threshold_at_target_precision"] == pytest.approx(0.8)
    empty = metrics.precision_recall_by_score(
        np.array([0.2], dtype=np.float32), np.array([0]), 0.95
    )
    assert empty["recall_at_target_precision"] is None


def test_fixed_threshold_uses_strict_greater_than_and_active_mask():
    readout = metrics.fixed_threshold_counts(
        np.array([0.5, 0.8, 0.9], dtype=np.float32),
        np.array([1, 1, 0], dtype=np.bool_),
        np.array([1, 1, 0], dtype=np.bool_),
    )
    assert readout["true_positive_edge_count"] == 1
    assert readout["false_positive_pair_count"] == 0
    assert readout["positive_edge_recall"] == 0.5
    assert readout["positive_pair_precision"] == 1.0


def test_nearest_neighbor_and_training_quantiles_keep_missing_separate():
    scipy = pytest.importorskip("scipy")
    assert scipy
    distances = metrics.nearest_neighbor_distance_um(np.array([[0, 0, 0], [0, 3, 0], [0, 8, 0]]))
    assert distances.tolist() == pytest.approx([3, 3, 5])
    assert np.isnan(metrics.nearest_neighbor_distance_um(np.zeros((1, 3)))[0])
    assert metrics.quantile_boundaries(np.array([1, 1, 1]), [0.25, 0.5, 0.75]) == [1.0]


def test_bucket_readout_retains_counts_when_support_is_small():
    rows = metrics.bucket_readout(
        np.array([1.0, 3.0, np.nan]),
        np.array([0.9, 0.1, 0.6]),
        np.array([1, 1, 0]),
        [2.0],
        min_positive_edges=2,
    )
    by_bucket = {row["bucket"]: row for row in rows}
    assert by_bucket[0]["positive_edge_count"] == 1
    assert by_bucket[0]["positive_edge_recall"] is None
    assert by_bucket[-1]["missing_value"] is True
    assert by_bucket[-1]["false_positive_pair_count"] == 1


def test_reference_contains_all_eight_fixed_comparisons():
    reference = json.loads((EXPERIMENT / "reference_pair_metrics.json").read_text())
    assert set(reference) == {"6bba", "44b6"}
    for rows in reference.values():
        assert set(rows) == {
            "control",
            "model_a",
            "model_b",
            "model_b_identity_init",
        }
        counts = {row["positive_edge_count"] for row in rows.values()}
        assert len(counts) == 1


def test_sample_shard_summary_preserves_pair_alignment(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENT))
    runner = load_module("diagnostic_runner")
    shard = tmp_path / "sample.npz"
    payload = {
        "window": np.zeros(4, dtype=np.uint16),
        "source_row": np.array([0, 0, 1, 1], dtype=np.uint16),
        "target_col": np.array([0, 1, 0, 1], dtype=np.uint16),
        "label": np.array([1, 0, 0, 1], dtype=np.bool_),
        "unknown": np.array([0, 0, 0, 1], dtype=np.bool_),
        "source_offsets": np.array([0, 2], dtype=np.int32),
        "target_offsets": np.array([0, 2], dtype=np.int32),
        "source_coords": np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float32),
        "target_coords": np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float32),
        "target_nearest_um": np.array([2, 2], dtype=np.float32),
        "division_rows": np.zeros(2, dtype=np.bool_),
        "current": np.array([0.9, 0.1, 0.2, 0.8], dtype=np.float32),
    }
    np.savez_compressed(shard, **payload)
    record = {
        "path": str(shard),
        "sha256": runner.file_sha256(shard),
        "fixed_totals": {
            "current": {
                "window_count": 1,
                "division_parent_count": 0,
                "recovered_division_parent_count": 0,
            }
        },
    }
    edges = {
        "target_candidate_count": [1.0],
        "target_nearest_um": [1.0],
        "pair_displacement_um": [1.0],
    }
    config = {
        "validation": {
            "target_precision": 0.95,
            "min_positive_edges_per_bucket": 1,
            "min_division_parents_for_ratio": 1,
        }
    }
    result = runner.summarize_group([record], "current", edges, tmp_path / "curves", config)
    assert result["fixed_0_5"]["positive_edge_count"] == 2
    assert result["fixed_0_5"]["true_positive_edge_count"] == 2
    assert result["fixed_0_5"]["unknown_endpoint_pair_count"] == 1
    assert result["pr_curves"]["active_mask"]["recall_at_target_precision"] == 1.0
    assert result["pr_curves"]["matched_endpoints"]["positive_count"] == 1


def test_candidate_ids_are_read_from_npz_outside_feature_validator(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENT))
    runner = load_module("diagnostic_runner")
    path = tmp_path / "window.npz"
    np.savez(
        path,
        candidate_ids_src=np.array([10, 11], dtype=np.int64),
        candidate_ids_tgt=np.array([20], dtype=np.int64),
    )
    source_ids, target_ids = runner.read_candidate_ids(path, 2, 1)
    assert source_ids.tolist() == [10, 11]
    assert target_ids.tolist() == [20]
    with pytest.raises(ValueError, match="candidate ID shape mismatch"):
        runner.read_candidate_ids(path, 1, 1)
