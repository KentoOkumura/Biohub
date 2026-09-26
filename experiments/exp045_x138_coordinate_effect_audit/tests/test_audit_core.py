"""Contracts for the fixed video set and partial-annotation readouts."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "audit_core.py"
SPEC = importlib.util.spec_from_file_location("exp045_audit_core", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
core = importlib.util.module_from_spec(SPEC)

sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


def _ranked(embryo: str) -> list[str]:
    return sorted(
        (f"{embryo}_{index:03d}" for index in range(25)),
        key=core.video_rank,
    )


def test_head_unseen_video_selection_matches_original_rank() -> None:
    ranked = {embryo: _ranked(embryo) for embryo in ("44b6", "6bba")}
    trained = [stem for values in ranked.values() for stem in values[:10]]
    eligible = [stem for values in ranked.values() for stem in reversed(values)]
    selected = core.select_videos(iter(eligible), iter(trained))
    assert selected == {embryo: values[10:20] for embryo, values in ranked.items()}
    assert not set(trained).intersection(sum(selected.values(), []))


def test_selection_rejects_manifest_drift_and_missing_video() -> None:
    ranked = {embryo: _ranked(embryo) for embryo in ("44b6", "6bba")}
    eligible = [stem for values in ranked.values() for stem in values]
    trained = [stem for values in ranked.values() for stem in values[:10]]
    trained[0] = ranked["44b6"][11]
    with pytest.raises(ValueError, match="manifest differs"):
        core.select_videos(eligible, trained)
    with pytest.raises(ValueError, match="fewer than"):
        core.select_videos(eligible[:18], trained[:10])


def test_physical_match_is_one_to_one_and_time_local() -> None:
    candidates = np.array(
        [
            [0, 0, 1, 1, 1],
            [1, 0, 1, 10, 10],
            [2, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    gt = np.array(
        [
            [10, 0, 1, 1, 1],
            [11, 0, 1, 10, 10],
            [12, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    match = core.match_known_centers(candidates, gt, radius_um=1.0)
    assert match.gt_to_candidate == {10: 0, 11: 1, 12: 2}
    assert all(value == 0 for value in match.matched_distance_um.values())
    assert match.ambiguous_gt == 0


def test_stage_counts_use_only_annotated_parent_contradictions() -> None:
    match = core.Match(
        {10: 0, 11: 1, 12: 2, 13: 3},
        {0: 10, 1: 11, 2: 12, 3: 13},
        {},
        0,
        0,
    )
    truth = {(10, 11), (10, 12)}
    result = core.readout_stages(
        truth,
        match,
        {"candidate": {(0, 1), (0, 2), (3, 1), (3, 2)}},
    )["candidate"]
    assert result["known_edge_denominator"] == 2
    assert result["known_edges_selected"] == 2
    assert result["known_divisions_selected"] == 1
    assert result["annotated_parent_contradictions"] == 2


def test_rank_of_known_edges_preserves_absent_pair() -> None:
    result = core.scored_known_edge_ranks(
        iter([(99, 1, 0.9), (0, 2, 0.8), (0, 1, 0.6), (0, 2, 0.7)]),
        {(0, 1), (0, 2), (0, 3)},
    )
    assert result[(0, 2)] == (0.8, 1)
    assert result[(0, 1)] == (0.6, 2)
    assert result[(0, 3)] == (None, None)


def test_sparse_graph_ids_map_to_candidate_rows_and_exclude_added_nodes() -> None:
    candidates = np.array(
        [
            [0, 1, 2, 3],
            [0, 4, 5, 6],
            [1, 7, 8, 9],
            [1, 10, 11, 12],
        ],
        dtype=float,
    )
    graph_rows = np.array([[1, 0, 4, 5, 6], [3, 1, 10, 11, 12]], dtype=float)
    ids = core.validate_initial_graph_ids(candidates, graph_rows)
    assert ids == {1, 3}
    assert core.trace_original_edges({(1, 3), (1, 2), (2, 3)}, ids) == {(1, 3)}
    wrong = graph_rows.copy()
    wrong[0, 2] = 9
    with pytest.raises(ValueError, match="do not map"):
        core.validate_initial_graph_ids(candidates, wrong)


def test_candidate_cache_and_gt_use_native_yx_spacing() -> None:
    candidate = np.array([[0, 0, 0, 4, 0]], dtype=float)
    known = np.array([[10, 0, 0, 0, 0]], dtype=float)
    match = core.match_known_centers(candidate, known, radius_um=2.0)
    assert match.gt_to_candidate == {10: 0}
    assert match.matched_distance_um[10] == pytest.approx(1.625)
