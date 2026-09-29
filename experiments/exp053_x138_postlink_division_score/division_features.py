"""Four-frame geometric features for one proposed second daughter.

The same feature function is used for sparse GEFF training and the
post-link graph in x138 inference. Coordinates are in native voxel units.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

FEATURE_NAMES = (
    "mother_to_first_um",
    "mother_to_second_um",
    "sister_distance_um",
    "daughter_distance_asymmetry_um",
    "daughter_displacement_cosine",
    "predecessor_present",
    "predecessor_step_um",
    "predecessor_to_first_cosine",
    "predecessor_to_second_cosine",
    "predecessor_extrapolation_first_um",
    "predecessor_extrapolation_second_um",
    "first_successor_present",
    "second_successor_present",
    "first_successor_step_um",
    "second_successor_step_um",
    "first_straightness_cosine",
    "second_straightness_cosine",
    "next_sister_distance_um",
    "daughter_divergence_um",
)
CENTRAL_FEATURE_COUNT = 5
SPACING_UM = np.array((1.625, 0.40625, 0.40625), dtype=np.float64)


def position_um(node: Mapping[str, object]) -> np.ndarray:
    return (
        np.array(
            (float(node["z"]), float(node["y"]), float(node["x"])),
            dtype=np.float64,
        )
        * SPACING_UM
    )


def _norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = _norm(a) * _norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-9 else 0.0


def build_edge_index(
    edges: Sequence[Mapping[str, object]] | np.ndarray,
) -> tuple[dict[int, int], dict[int, list[int]]]:
    incoming: dict[int, int] = {}
    outgoing: dict[int, list[int]] = {}
    if isinstance(edges, np.ndarray):
        pairs = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        iterator = ((int(source), int(target)) for source, target in pairs)
    else:
        iterator = ((int(edge["source_id"]), int(edge["target_id"])) for edge in edges)
    for source, target in iterator:
        if target in incoming and incoming[target] != source:
            raise ValueError(f"multiple parents for node {target}")
        incoming[target] = source
        outgoing.setdefault(source, []).append(target)
    for children in outgoing.values():
        children.sort()
    return incoming, outgoing


def _successor(
    node_id: int,
    positions: Mapping[int, np.ndarray],
    outgoing: Mapping[int, Sequence[int]],
) -> int | None:
    children = [child for child in outgoing.get(node_id, ()) if child in positions]
    if not children:
        return None
    return min(children, key=lambda child: (_norm(positions[child] - positions[node_id]), child))


def feature_vector(
    nodes: Mapping[int, Mapping[str, object]],
    incoming: Mapping[int, int],
    outgoing: Mapping[int, Sequence[int]],
    mother_id: int,
    first_id: int,
    second_id: int,
) -> np.ndarray:
    positions = {
        node_id: position_um(nodes[node_id]) for node_id in (mother_id, first_id, second_id)
    }
    mother = positions[mother_id]
    first = positions[first_id]
    second = positions[second_id]
    first_step = first - mother
    second_step = second - mother
    first_distance = _norm(first_step)
    second_distance = _norm(second_step)
    sister_distance = _norm(first - second)
    result = [
        first_distance,
        second_distance,
        sister_distance,
        abs(first_distance - second_distance),
        _cosine(first_step, second_step),
    ]
    predecessor_id = incoming.get(mother_id)
    if (
        predecessor_id is not None
        and predecessor_id in nodes
        and int(nodes[predecessor_id]["t"]) == int(nodes[mother_id]["t"]) - 1
    ):
        predecessor = position_um(nodes[predecessor_id])
        previous_step = mother - predecessor
        extrapolated = mother + previous_step
        result.extend(
            (
                1.0,
                _norm(previous_step),
                _cosine(previous_step, first_step),
                _cosine(previous_step, second_step),
                _norm(extrapolated - first),
                _norm(extrapolated - second),
            )
        )
    else:
        result.extend((0.0,) * 6)

    successor_positions: list[np.ndarray | None] = []
    for daughter_id in (first_id, second_id):
        successor_id = _successor(
            daughter_id,
            {
                daughter_id: position_um(nodes[daughter_id]),
                **{
                    child: position_um(nodes[child])
                    for child in outgoing.get(daughter_id, ())
                    if child in nodes and int(nodes[child]["t"]) == int(nodes[daughter_id]["t"]) + 1
                },
            },
            outgoing,
        )
        successor_positions.append(
            position_um(nodes[successor_id]) if successor_id is not None else None
        )
    first_next, second_next = successor_positions
    first_next_step = first_next - first if first_next is not None else None
    second_next_step = second_next - second if second_next is not None else None
    result.extend(
        (
            float(first_next is not None),
            float(second_next is not None),
            _norm(first_next_step) if first_next_step is not None else 0.0,
            _norm(second_next_step) if second_next_step is not None else 0.0,
            _cosine(first_step, first_next_step) if first_next_step is not None else 0.0,
            _cosine(second_step, second_next_step) if second_next_step is not None else 0.0,
            _norm(first_next - second_next)
            if first_next is not None and second_next is not None
            else 0.0,
            _norm(first_next - second_next) - sister_distance
            if first_next is not None and second_next is not None
            else 0.0,
        )
    )
    output = np.asarray(result, dtype=np.float32)
    if len(output) != len(FEATURE_NAMES) or not np.isfinite(output).all():
        raise ValueError("invalid four-frame division features")
    return output


def mask_context(features: np.ndarray) -> np.ndarray:
    output = np.asarray(features, dtype=np.float32).copy()
    output[..., CENTRAL_FEATURE_COUNT:] = 0.0
    return output


def score_model(model: Mapping[str, object], features: np.ndarray) -> float:
    if tuple(model["feature_names"]) != FEATURE_NAMES:
        raise ValueError("division feature schema mismatch")
    if model.get("selected_variant") != "four_frame":
        raise ValueError("selected division variant mismatch")
    x = np.asarray(features, dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    if x.shape != mean.shape or x.shape != scale.shape or x.shape != coef.shape:
        raise ValueError("division model dimension mismatch")
    if not np.isfinite(x).all() or not np.isfinite(coef).all():
        raise ValueError("non-finite division model input")
    logit = float(np.dot((x - mean) / scale, coef) + float(model["intercept"]))
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, logit))))
