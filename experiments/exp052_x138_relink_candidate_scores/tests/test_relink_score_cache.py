"""Score-cache mapping tests without importing the executable Kaggle Notebook."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / (
    "exp052_x138_relink_candidate_scores_diagnostic.py"
)


def score_map_function(tmp_path: Path):
    tree = ast.parse(NOTEBOOK.read_text())
    names = {"validate_initial_graph_ids", "_exp052_score_map"}
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    namespace = {
        "Path": Path,
        "np": np,
        "AUDIT_ROOT": tmp_path,
        "_audit_sha256": lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["_exp052_score_map"]


def write_cache(tmp_path: Path, *, selected_probability: float = 0.6) -> None:
    cache = tmp_path / "baseline" / "candidate_cache"
    cache.mkdir(parents=True)
    np.savez_compressed(
        cache / "video.npz",
        coords=np.array([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 1]]),
        edge_src=np.array([0, 1, 0, 1]),
        edge_tgt=np.array([2, 2, 3, 3]),
        edge_prob=np.array([0.6, 0.4, 0.3, 0.7]),
        admitted=np.array([[0, 2, selected_probability, 1], [1, 3, 0.7, 1]]),
    )


def test_unselected_scores_are_limited_to_initial_ilp_nodes(tmp_path: Path) -> None:
    write_cache(tmp_path)
    score_map = score_map_function(tmp_path)
    # Candidate ID 1 was discarded by the ILP. A newly readmitted node could reuse it.
    nodes = {
        0: {"t": 0, "z": 0, "y": 0, "x": 0},
        2: {"t": 1, "z": 0, "y": 0, "x": 1},
        3: {"t": 1, "z": 1, "y": 0, "x": 1},
    }
    extra, audit = score_map("video", nodes, [{"source_id": 0, "target_id": 2, "edge_prob": 0.6}])
    assert extra == {(0, 3): 0.3}
    assert audit["new_scores_available_on_initial_nodes"] == 1
    assert audit["selected_edge_count"] == 1


def test_selected_score_mismatch_stops_replay(tmp_path: Path) -> None:
    write_cache(tmp_path, selected_probability=0.61)
    score_map = score_map_function(tmp_path)
    nodes = {
        0: {"t": 0, "z": 0, "y": 0, "x": 0},
        2: {"t": 1, "z": 0, "y": 0, "x": 1},
    }
    with pytest.raises(RuntimeError, match="original candidate score differs"):
        score_map("video", nodes, [{"source_id": 0, "target_id": 2, "edge_prob": 0.6}])


def test_graph_coordinate_mismatch_stops_replay(tmp_path: Path) -> None:
    write_cache(tmp_path)
    score_map = score_map_function(tmp_path)
    nodes = {
        0: {"t": 0, "z": 0, "y": 0, "x": 0},
        2: {"t": 1, "z": 0, "y": 9, "x": 1},
    }
    with pytest.raises(ValueError, match="initial graph IDs do not map"):
        score_map("video", nodes, [{"source_id": 0, "target_id": 2, "edge_prob": 0.6}])
