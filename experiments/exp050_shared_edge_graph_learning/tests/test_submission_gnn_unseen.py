"""Check the frozen GNN choice for labeled and unseen test embryos."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "exp050_shared_edge_graph_learning_submission_gnn.py"


def load_scoring_function(score_video_sets):
    module = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "score_submission_gnn_sets"
    )
    namespace = {"np": np, "score_video_sets": score_video_sets}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[function.name]


def test_unseen_embryo_averages_matching_set_scores_before_decoding():
    calls = []
    catalog = [(0, (), ()), (0, (1,), (0,)), (0, (1, 2), (0, 1))]
    by_parent = [[0, 1, 2], [], []]
    scores = {
        0: np.array([0.0, 0.3, 0.5]),
        1: np.array([0.0, 0.9, -0.1]),
    }

    def score_video_sets(model, coords, features, edges, *, device):
        calls.append(model)
        assert device == "cpu"
        return catalog, by_parent, scores[model], np.zeros(2)

    score = load_scoring_function(score_video_sets)
    args = ({0: 0, 1: 1}, None, None, None)
    result = score(args[0], "new_embryo", *args[1:], device="cpu")
    assert calls == [0, 1]
    assert result[0] == catalog and result[1] == by_parent
    np.testing.assert_allclose(result[2], [0.0, 0.6, 0.2])
    assert result[3] == (0, 1)

    calls.clear()
    assert score(args[0], "44b6", *args[1:], device="cpu")[3] == (1,)
    assert calls == [1]
    calls.clear()
    assert score(args[0], "6bba", *args[1:], device="cpu")[3] == (0,)
    assert calls == [0]


def test_unseen_embryo_rejects_misaligned_candidate_catalogs():
    calls = []

    def score_video_sets(model, coords, features, edges, *, device):
        calls.append(model)
        catalog = [(0, (1 if model == 0 else 2,), (0,))]
        return catalog, [[0]], np.array([1.0]), np.zeros(1)

    score = load_scoring_function(score_video_sets)
    with pytest.raises(RuntimeError, match="different candidate set catalogs"):
        score({0: 0, 1: 1}, "new_embryo", None, None, None, device="cpu")
    assert calls == [0, 1]
