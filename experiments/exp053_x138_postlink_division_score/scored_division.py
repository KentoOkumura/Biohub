"""Greedy legal second-daughter additions after x138 motion relinking."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from division_features import build_edge_index, feature_vector, position_um, score_model
from scipy.spatial import cKDTree


def load_model(path: Path, expected_sha256: str) -> dict[str, object]:
    if len(expected_sha256) != 64:
        raise ValueError("expected model SHA256 is missing")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("division model SHA256 mismatch")
    model = json.loads(raw)
    threshold = float(model["threshold"])
    if not 0.0 < threshold <= 1.0:
        raise ValueError("invalid division threshold")
    return model


def add_scored_divisions_postlink(
    nodes: Mapping[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    model: Mapping[str, object],
    accept_candidate: Callable[[int], bool],
    parent_max_um: float = 9.0,
    sister_max_um: float = 14.0,
    first_max_um: float = 10.0,
    frame_fraction_cap: float = 0.0076,
    global_fraction_cap: float = 0.00375,
    enabled: bool = True,
) -> list[dict[str, object]]:
    if not enabled or not edges or not nodes:
        return edges
    incoming, outgoing = build_edge_index(edges)
    ids_by_t: dict[int, list[int]] = defaultdict(list)
    for node_id, node in nodes.items():
        ids_by_t[int(node["t"])].append(node_id)
    for ids in ids_by_t.values():
        ids.sort()

    global_cap = max(1, int(round(max(1, len(edges)) * global_fraction_cap)))
    proposals: list[tuple[float, int, int, int, float]] = []
    frame_caps: dict[int, int] = {}
    for t in sorted(ids_by_t):
        source_ids = [node_id for node_id in ids_by_t[t] if len(outgoing.get(node_id, ())) == 1]
        orphan_ids = [node_id for node_id in ids_by_t.get(t + 1, ()) if node_id not in incoming]
        if not source_ids or not orphan_ids:
            continue
        frame_caps[t] = max(1, int(round(len(source_ids) * frame_fraction_cap)))
        orphan_positions = np.stack([position_um(nodes[node_id]) for node_id in orphan_ids])
        tree = cKDTree(orphan_positions)
        for mother_id in source_ids:
            first_id = outgoing[mother_id][0]
            if first_id not in nodes or int(nodes[first_id]["t"]) != t + 1:
                continue
            mother_pos = position_um(nodes[mother_id])
            first_pos = position_um(nodes[first_id])
            if np.linalg.norm(first_pos - mother_pos) > first_max_um:
                continue
            for index in tree.query_ball_point(mother_pos, r=parent_max_um):
                second_id = orphan_ids[int(index)]
                if second_id == first_id:
                    continue
                second_pos = orphan_positions[int(index)]
                if np.linalg.norm(second_pos - first_pos) > sister_max_um:
                    continue
                features = feature_vector(nodes, incoming, outgoing, mother_id, first_id, second_id)
                probability = score_model(model, features)
                stats["scored_division_candidates"] = stats.get("scored_division_candidates", 0) + 1
                if probability >= float(model["threshold"]):
                    proposals.append(
                        (
                            probability,
                            mother_id,
                            first_id,
                            second_id,
                            float(np.linalg.norm(second_pos - mother_pos)),
                        )
                    )

    stats["scored_division_above_threshold"] = len(proposals)
    proposals.sort(key=lambda item: (-item[0], item[1], item[3]))
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    selected_per_frame: dict[int, int] = defaultdict(int)
    added: list[dict[str, object]] = []
    for probability, mother_id, _, second_id, distance in proposals:
        if len(added) >= global_cap:
            stats["scored_division_global_cap_reached"] = 1
            break
        t = int(nodes[mother_id]["t"])
        if selected_per_frame[t] >= frame_caps[t]:
            continue
        if mother_id in used_sources or second_id in used_targets:
            continue
        if second_id in incoming:
            continue
        if not accept_candidate(second_id):
            stats["scored_division_deepcenter_rejected"] = (
                stats.get("scored_division_deepcenter_rejected", 0) + 1
            )
            continue
        added.append(
            {
                "source_id": mother_id,
                "target_id": second_id,
                "edge_prob": probability,
                "distance_um": distance,
                "scored_division": 1,
            }
        )
        used_sources.add(mother_id)
        used_targets.add(second_id)
        selected_per_frame[t] += 1
    stats["scored_divisions_added"] = len(added)
    stats["safe_divisions_added"] = len(added)
    return [*edges, *added]
