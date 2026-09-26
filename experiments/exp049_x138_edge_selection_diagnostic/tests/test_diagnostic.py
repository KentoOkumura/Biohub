"""Small contracts that do not require saved Kaggle outputs."""

import runpy
from pathlib import Path

import numpy as np
import pytest

MODULE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1] / "exp049_x138_edge_selection_diagnostic_diagnostic.py"
    ),
    run_name="exp049_test",
)


def test_candidate_selection_preserves_baseline_and_tie_break() -> None:
    coords = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0], [1, 0, 1, 0]])
    payload = {
        "coords": coords,
        "edge_src": np.array([0, 1, 2, 3]),
        "edge_tgt": np.array([4, 4, 4, 4]),
        "edge_prob": np.array([0.5, 0.3, 0.3, 0.2]),
        "admitted": np.array([[0, 4, 0.5, 1.0]]),
    }
    selected = MODULE["select_edges"](payload)
    assert MODULE["pairs"](selected["baseline"]) == {(0, 4)}
    assert MODULE["pairs"](selected["expanded"]) == {(0, 4), (1, 4), (2, 4)}
    assert len(selected["added"]) == 2


def test_forced_ilp_is_diagnostic_and_respects_parent_constraint() -> None:
    edges = np.array(
        [
            [0, 1, 0.9, 1],
            [1, 2, 0.9, 1],
            [2, 3, 0.9, 1],
            [4, 2, 0.1, 1],
        ]
    )
    solve = MODULE["solve_ilp"]
    normal = solve(5, edges)
    forced = solve(5, edges, {(4, 2)})
    assert normal["optimal"] and forced["optimal"]
    assert (4, 2) not in normal["selected_edges"]
    assert (4, 2) in forced["selected_edges"]
    assert (1, 2) not in forced["selected_edges"]
    assert forced["objective"] >= normal["objective"] - 1e-8
    with pytest.raises(ValueError, match="forced edge missing"):
        solve(5, edges, {(5, 2)})


def test_first_lost_and_forced_case_cap() -> None:
    edge = (1, 2)
    stages = {stage: {edge} for stage in MODULE["STAGES"]}
    stages["ilp"] = set()
    assert MODULE["first_lost"](stages, {edge}) == "ilp"
    rows = []
    for embryo in ("44b6", "6bba"):
        for i in range(9):
            rows.append(
                {
                    "stem": f"{embryo}_a{i}",
                    "embryo": embryo,
                    "arm": "expanded",
                    "type": "edge",
                    "event_edges": [[i, i + 1]],
                    "stages": {"candidate": True, "ilp": False},
                }
            )
    selected = MODULE["pick_forced_cases"](rows)
    assert len(selected) == 6
