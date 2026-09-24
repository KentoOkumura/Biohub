from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
pytest.importorskip("torch")

from frozen_tracker import AnnotationGraph, FrameAnnotation  # noqa: E402
from knn_diagnostics import (  # noqa: E402
    RuntimeBudget,
    retention_window,
    stratified_plan,
    stratum_counts,
    weighted_runtime,
)


def test_retention_counts_detection_failure_separately_and_detects_dropped_parent() -> None:
    frames = {
        0: FrameAnnotation(np.array([100, 101]), np.array([[0.0, 0, 0], [50.0, 0, 0]])),
        1: FrameAnnotation(
            np.array([200, 201, 202]), np.array([[1.0, 0, 0], [2.0, 0, 0], [50.0, 0, 0]])
        ),
    }
    annotation = AnnotationGraph(
        frames, frozenset({(100, 200), (100, 201), (101, 202)}), "x", {100: (200, 201), 101: (202,)}
    )
    cfg = {
        "density_radius_um": 10.0,
        "density_upper_bounds": [8, 16],
        "displacement_upper_bounds_um": [5.0, 10.0],
    }
    counts, buckets = retention_window(
        np.array([[0.0, 0, 0]], dtype=np.float32),
        np.array([[1.0, 0, 0], [2.0, 0, 0]], dtype=np.float32),
        np.array([3]),
        annotation,
        1,
        np.array([[3], [99]]),
        5.0,
        cfg,
    )
    assert counts["known_edges"] == 3
    assert counts["matched_edges"] == 2
    assert counts["missing_endpoints"] == 1
    assert counts["retained_edges"] == 1
    assert buckets["division_1"] == {"matched_edges": 2, "retained_edges": 1}
    boundary, _ = retention_window(
        np.empty((0, 3)), np.empty((0, 3)), np.array([]), annotation, 0, np.empty((0, 0)), 5.0, cfg
    )
    assert boundary["boundary_windows"] == 1 and boundary["matched_edges"] == 0


def test_stratification_uses_actual_work_and_weights_all_windows() -> None:
    rows = [{"path": f"window_{i:03}", "selected_triples": i} for i in range(100)]
    plan = stratified_plan(rows, 4, 16, 42)
    assert plan == stratified_plan(rows, 4, 16, 42)
    assert sum(map(len, plan["sampled_paths"].values())) == 64
    assert stratum_counts(rows, plan["boundaries"]) == [25, 25, 25, 25]
    assert weighted_runtime([1, 2, 10, 4], {"0": 1.0, "1": 2.0, "2": 3.0, "3": 4.0}) == 51
    assert weighted_runtime([0, 0, 0, 4], {"0": 1.0, "2": 3.0}) == 12
    identical = [{"path": str(i), "selected_triples": 8} for i in range(20)]
    assert len(stratified_plan(identical, 4, 16, 42)["sampled_paths"]["3"]) == 16


def test_budget_requires_quota_and_honours_smaller_user_limit() -> None:
    cfg = {
        "quota_remaining_hours_at_push": 13.0,
        "quota_used_hours_at_push": 32.0,
        "quota_checked_at": "2026-09-23",
        "weekly_gpu_budget_hours": 45.0,
        "runtime_gate_hours": 12.0,
        "runtime_projection_multiplier": 1.5,
    }
    budget = RuntimeBudget(time.perf_counter(), cfg, 2)
    assert budget.limit_seconds == 6.5 * 3600
    budget.remaining_work = 7 * 3600
    with pytest.raises(RuntimeError, match="budget gate"):
        budget.check()
    with pytest.raises(RuntimeError, match="budget gate"):
        RuntimeBudget(time.perf_counter(), {**cfg, "weekly_gpu_budget_hours": 30.0}, 1)
    with pytest.raises(RuntimeError, match="fresh quota"):
        RuntimeBudget(time.perf_counter(), {**cfg, "quota_remaining_hours_at_push": None}, 1)
