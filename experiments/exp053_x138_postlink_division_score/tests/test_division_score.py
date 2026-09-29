# ruff: noqa: E402
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from division_features import (
    FEATURE_NAMES,
    build_edge_index,
    feature_vector,
    mask_context,
)
from scored_division import add_scored_divisions_postlink


def node(t, z, y, x):
    return {"t": t, "z": z, "y": y, "x": x}


def test_context_mask_keeps_central_geometry_and_removes_pre_post_information():
    nodes = {
        0: node(0, 0, 0, 0),
        1: node(1, 0, 1, 0),
        2: node(2, 0, 2, 0),
        3: node(2, 0, 1, 1),
        4: node(3, 0, 3, 0),
        5: node(3, 0, 1, 2),
    }
    incoming, outgoing = build_edge_index(
        [{"source_id": a, "target_id": b} for a, b in ((0, 1), (1, 2), (2, 4), (3, 5))]
    )
    features = feature_vector(nodes, incoming, outgoing, 1, 2, 3)
    masked = mask_context(features)
    assert len(features) == len(FEATURE_NAMES)
    assert (features[:5] == masked[:5]).all()
    assert (masked[5:] == 0).all()
    assert features[5] == 1
    assert features[11] == 1
    assert features[12] == 1


def test_postlink_addition_keeps_one_parent_and_two_daughters_at_most():
    nodes = {
        0: node(0, 0, 0, 0),
        1: node(1, 0, 1, 0),
        2: node(1, 0, 2, 0),
        3: node(1, 0, 1, 1),
    }
    edges = [{"source_id": 0, "target_id": 1, "edge_prob": 0.9}]
    model = {
        "selected_variant": "four_frame",
        "feature_names": FEATURE_NAMES,
        "mean": [0] * len(FEATURE_NAMES),
        "scale": [1] * len(FEATURE_NAMES),
        "coef": [0] * len(FEATURE_NAMES),
        "intercept": 50,
        "threshold": 0.5,
    }
    stats = {}
    result = add_scored_divisions_postlink(nodes, edges, stats, model, lambda _candidate: True)
    incoming, outgoing = build_edge_index(result)
    assert len(outgoing[0]) == 2
    assert len(incoming) == len(result)
    assert stats["scored_divisions_added"] == 1
