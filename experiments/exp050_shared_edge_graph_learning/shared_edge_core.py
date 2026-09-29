"""Shared edge scoring and exact mother-daughter set selection for exp050."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from time import monotonic

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array
from torch import nn
from torch.nn import functional as F

UNKNOWN_PARENT = -1
KNOWN_NO_PARENT = -2


@dataclass(frozen=True)
class SetOption:
    mother: int
    daughters: tuple[int, ...]
    edge_ids: tuple[int, ...]


def select_expanded_edges(
    coords: np.ndarray,
    scored_src: np.ndarray,
    scored_tgt: np.ndarray,
    scored_probability: np.ndarray,
    admitted: np.ndarray,
) -> np.ndarray:
    """Reproduce exp047's strict thresholds and daughter-side top three."""
    coords = np.asarray(coords, dtype=np.float64)
    src = np.asarray(scored_src, dtype=np.int64)
    tgt = np.asarray(scored_tgt, dtype=np.int64)
    prob = np.asarray(scored_probability, dtype=np.float64)
    admitted = np.asarray(admitted, dtype=np.float64).reshape(-1, 4)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("coords must have t,z,y,x columns")
    if src.shape != tgt.shape or src.shape != prob.shape or src.ndim != 1:
        raise ValueError("score vectors must match")
    if not np.isfinite(coords).all() or not np.isfinite(prob).all():
        raise ValueError("nonfinite input")
    if np.any(prob <= 0) or np.any(prob > 1):
        raise ValueError("invalid probability")
    if (
        np.any(src < 0)
        or np.any(tgt < 0)
        or np.any(src >= len(coords))
        or np.any(tgt >= len(coords))
    ):
        raise ValueError("edge ID out of range")
    if np.any(coords[src, 0] + 1 != coords[tgt, 0]):
        raise ValueError("non-adjacent edge")
    scored = {(int(s), int(t)): float(p) for s, t, p in zip(src, tgt, prob, strict=True)}
    if len(scored) != len(src):
        raise ValueError("duplicate scored edge")
    baseline = {(int(r[0]), int(r[1])) for r in admitted}
    if len(baseline) != len(admitted):
        raise ValueError("duplicate admitted edge")
    if {pair for pair, p in scored.items() if p > 0.48} != baseline:
        raise ValueError("baseline_cache_mismatch")
    for row in admitted:
        pair = (int(row[0]), int(row[1]))
        if not np.isclose(float(row[2]), scored[pair], rtol=2e-6, atol=1e-8):
            raise ValueError("baseline probability mismatch")
    eligible = np.flatnonzero(prob > 0.10)
    order = eligible[np.lexsort((src[eligible], -prob[eligible], tgt[eligible]))]
    selected: list[int] = []
    daughter_count: dict[int, int] = defaultdict(int)
    for idx in order:
        daughter = int(tgt[idx])
        if daughter_count[daughter] < 3:
            selected.append(int(idx))
        daughter_count[daughter] += 1
    added = []
    for idx in selected:
        pair = (int(src[idx]), int(tgt[idx]))
        if pair in baseline:
            continue
        distance = float(np.linalg.norm(coords[src[idx], 1:] - coords[tgt[idx], 1:]))
        added.append((pair[0], pair[1], float(prob[idx]), distance))
    expanded = np.concatenate((admitted, np.asarray(added, dtype=np.float64).reshape(-1, 4)))
    if len({(int(r[0]), int(r[1])) for r in expanded}) != len(expanded):
        raise ValueError("duplicate expanded edge")
    return expanded


def enumerate_sets(
    coords: np.ndarray, edges: np.ndarray
) -> tuple[list[SetOption], list[list[int]]]:
    """Enumerate every pair from the fixed edge list, with no mother-side cap."""
    coords = np.asarray(coords)
    edges = np.asarray(edges, dtype=np.float64).reshape(-1, 4)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("invalid coordinates")
    by_mother: list[list[int]] = [[] for _ in range(len(coords))]
    for edge_id, row in enumerate(edges):
        mother, daughter = int(row[0]), int(row[1])
        if mother < 0 or mother >= len(coords) or daughter < 0 or daughter >= len(coords):
            raise ValueError("edge ID out of range")
        if int(coords[mother, 0]) + 1 != int(coords[daughter, 0]):
            raise ValueError("non-adjacent edge")
        by_mother[mother].append(edge_id)
    catalog: list[SetOption] = []
    by_parent: list[list[int]] = [[] for _ in range(len(coords))]
    for mother, edge_ids in enumerate(by_mother):
        edge_ids.sort(key=lambda e: int(edges[e, 1]))
        if len({int(edges[e, 1]) for e in edge_ids}) != len(edge_ids):
            raise ValueError("duplicate mother-daughter edge")
        choices = [()] + [(e,) for e in edge_ids] + list(combinations(edge_ids, 2))
        for edge_tuple in choices:
            daughters = tuple(int(edges[e, 1]) for e in edge_tuple)
            by_parent[mother].append(len(catalog))
            catalog.append(SetOption(mother, daughters, edge_tuple))
    return catalog, by_parent


class MessageBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.self_linear = nn.Linear(width, width)
        self.neighbor_linear = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        total = torch.zeros_like(node)
        degree = torch.zeros((node.shape[0], 1), device=node.device, dtype=node.dtype)
        total.index_add_(0, src, node[tgt])
        total.index_add_(0, tgt, node[src])
        one = torch.ones((len(src), 1), device=node.device, dtype=node.dtype)
        degree.index_add_(0, src, one)
        degree.index_add_(0, tgt, one)
        update = self.self_linear(node) + self.neighbor_linear(total / degree.clamp_min(1))
        return F.relu(self.norm(node + self.dropout(update)))


class SharedEdgeGNN(nn.Module):
    def __init__(self, feature_width: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.node = nn.Linear(feature_width + 4, hidden)
        self.blocks = nn.ModuleList([MessageBlock(hidden, dropout) for _ in range(2)])
        edge_width = 2 * hidden + 5
        self.edge = nn.Sequential(nn.Linear(edge_width, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        pair_width = 3 * hidden + 10
        self.pair = nn.Sequential(nn.Linear(pair_width, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.edge[-1].weight)
        nn.init.zeros_(self.edge[-1].bias)
        nn.init.zeros_(self.pair[-1].weight)
        nn.init.constant_(self.pair[-1].bias, float(np.log(0.6 / 0.4)))

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        edges: torch.Tensor,
        catalog: list[SetOption],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or coords.shape != (len(features), 4):
            raise ValueError("node feature or coordinate shape mismatch")
        if edges.ndim != 2 or edges.shape[1] != 4:
            raise ValueError("edges must be Mx4")
        src, tgt = edges[:, 0].long(), edges[:, 1].long()
        physical = coords * coords.new_tensor([1.0, 1.625, 0.40625, 0.40625])
        node = F.relu(self.node(torch.cat((features, physical / 20.0), dim=1)))
        for block in self.blocks:
            node = block(node, src, tgt)
        displacement = (physical[tgt, 1:] - physical[src, 1:]) / 20.0
        distance = torch.linalg.vector_norm(displacement, dim=1, keepdim=True)
        original_probability = edges[:, 2].clamp(1e-6, 1 - 1e-6)
        edge_input = torch.cat(
            (node[src], node[tgt], displacement, distance, original_probability[:, None]), dim=1
        )
        edge_probability = torch.sigmoid(
            torch.logit(original_probability) + self.edge(edge_input).squeeze(1)
        )
        pair_options = [option for option in catalog if len(option.edge_ids) == 2]
        if pair_options:
            a_ids = torch.tensor(
                [option.edge_ids[0] for option in pair_options], device=features.device
            )
            b_ids = torch.tensor(
                [option.edge_ids[1] for option in pair_options], device=features.device
            )
            mothers = src[a_ids]
            daughters_a, daughters_b = tgt[a_ids], tgt[b_ids]
            pair_input = torch.cat(
                (
                    node[mothers],
                    node[daughters_a] + node[daughters_b],
                    torch.abs(node[daughters_a] - node[daughters_b]),
                    displacement[a_ids] + displacement[b_ids],
                    torch.abs(displacement[a_ids] - displacement[b_ids]),
                    distance[a_ids] + distance[b_ids],
                    torch.abs(distance[a_ids] - distance[b_ids]),
                    original_probability[a_ids, None] + original_probability[b_ids, None],
                    torch.abs(
                        original_probability[a_ids, None] - original_probability[b_ids, None]
                    ),
                ),
                dim=1,
            )
            pair_cost = 2.0 * torch.sigmoid(self.pair(pair_input).squeeze(1))
        else:
            pair_cost = edge_probability.new_empty((0,))
        scores: list[torch.Tensor] = []
        pair_index = 0
        zero = edge_probability.sum() * 0
        for option in catalog:
            if not option.edge_ids:
                scores.append(zero)
            elif len(option.edge_ids) == 1:
                scores.append(edge_probability[option.edge_ids[0]])
            else:
                scores.append(
                    edge_probability[option.edge_ids[0]]
                    + edge_probability[option.edge_ids[1]]
                    - pair_cost[pair_index]
                )
                pair_index += 1
        return edge_probability, pair_cost, torch.stack(scores)


def partial_teacher_loss(
    scores: torch.Tensor,
    edge_probability: torch.Tensor,
    catalog: list[SetOption],
    by_parent: list[list[int]],
    edges: np.ndarray,
    known_parent: Mapping[int, int],
    margin: float = 0.05,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Use only known parent assignments and contradictions as supervision."""
    edges = np.asarray(edges)
    n = len(by_parent)
    known_children: dict[int, set[int]] = defaultdict(set)
    for daughter, parent in known_parent.items():
        if parent >= 0:
            known_children[parent].add(daughter)
    partial: list[torch.Tensor] = []
    division: list[torch.Tensor] = []
    parent_margin: list[torch.Tensor] = []
    stats: dict[str, int] = defaultdict(int)
    for mother in range(n):
        positives = known_children.get(mother, set())
        options = by_parent[mother]
        if len(positives) > 2:
            stats["over_capacity_mothers"] += 1
            continue
        union = {daughter for idx in options for daughter in catalog[idx].daughters}
        if not positives <= union:
            stats["missing_positive_mothers"] += 1
            continue
        allowed = []
        for idx in options:
            daughters = set(catalog[idx].daughters)
            contradictions = any(d in known_parent and known_parent[d] != mother for d in daughters)
            allowed.append(positives <= daughters and not contradictions)
        if any(allowed) and not all(allowed):
            local = scores[options]
            permitted = local[torch.tensor(allowed, device=scores.device)]
            partial.append(torch.logsumexp(local, dim=0) - torch.logsumexp(permitted, dim=0))
            stats["informative_mothers"] += 1
        if len(positives) == 2:
            full = next((idx for idx in options if set(catalog[idx].daughters) == positives), None)
            if full is not None:
                subsets = [idx for idx in options if set(catalog[idx].daughters) < positives]
                if len(subsets) == 3:
                    division.append(F.softplus(margin + scores[subsets] - scores[full]).mean())
                    stats["known_divisions_with_candidates"] += 1
    by_daughter: dict[int, list[int]] = defaultdict(list)
    for edge_id, row in enumerate(edges):
        by_daughter[int(row[1])].append(edge_id)
    for daughter, parent in known_parent.items():
        if parent < 0:
            continue
        positive = next(
            (e for e in by_daughter.get(daughter, []) if int(edges[e, 0]) == parent), None
        )
        if positive is None:
            stats["known_parent_missing"] += 1
            continue
        others = [e for e in by_daughter[daughter] if e != positive]
        if others:
            parent_margin.append(
                F.softplus(margin + edge_probability[others] - edge_probability[positive]).mean()
            )
            stats["known_parent_competitions"] += 1
    zero = scores.sum() * 0
    terms = [
        torch.stack(group).mean() if group else zero for group in (partial, division, parent_margin)
    ]
    stats["known_divisions_with_candidates"] += 0
    stats["partial_terms"] = len(partial)
    stats["division_terms"] = len(division)
    stats["parent_terms"] = len(parent_margin)
    return sum(terms), dict(stats)


def solve_set_ilp(
    coords: np.ndarray,
    edges: np.ndarray,
    catalog: list[SetOption],
    by_parent: list[list[int]],
    scores: np.ndarray,
    *,
    appearance_cost: float = 0.0,
    disappearance_cost: float = 2.0,
    timeout_seconds: float = 1200.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Select nodes and mother sets with the baseline birth/death semantics."""
    coords = np.asarray(coords)
    edges = np.asarray(edges, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64)
    n, k = len(coords), len(catalog)
    if len(by_parent) != n or scores.shape != (k,) or not np.isfinite(scores).all():
        raise ValueError("set catalog or score mismatch")
    if n == 0:
        return np.empty((0, 4)), {"optimal": True, "status": 0, "objective": 0.0}
    # Variables: selected node, appearance, disappearance, mother set.
    objective = np.concatenate(
        (np.zeros(n), np.full(n, appearance_cost), np.full(n, disappearance_cost), -scores)
    )
    row, col, value = [], [], []

    def add(r: int, c: int, v: float) -> None:
        row.append(r)
        col.append(c)
        value.append(v)

    for node in range(n):
        # One set for each selected mother.
        add(node, node, -1)
        for set_id in by_parent[node]:
            add(node, 3 * n + set_id, 1)
        # Incoming edge or appearance for each selected node.
        add(n + node, node, -1)
        add(n + node, n + node, 1)
        # Empty set is exactly one disappearance.
        add(2 * n + node, 2 * n + node, -1)
    for set_id, option in enumerate(catalog):
        for daughter in option.daughters:
            add(n + daughter, 3 * n + set_id, 1)
        if not option.daughters:
            add(2 * n + option.mother, 3 * n + set_id, 1)
    matrix = coo_array(
        (np.asarray(value, dtype=np.float64), (row, col)), shape=(3 * n, 3 * n + k)
    ).tocsr()
    solve_started = monotonic()
    result = milp(
        objective,
        integrality=np.ones(3 * n + k, dtype=np.uint8),
        bounds=Bounds(np.zeros(3 * n + k), np.ones(3 * n + k)),
        constraints=LinearConstraint(matrix, np.zeros(3 * n), np.zeros(3 * n)),
        options={"time_limit": timeout_seconds, "mip_rel_gap": 0.0},
    )
    elapsed = monotonic() - solve_started
    if result.status != 0 or result.x is None or elapsed > timeout_seconds:
        raise RuntimeError(
            f"set ILP not optimal within {timeout_seconds}s: "
            f"{result.status} {result.message} ({elapsed:.1f}s)"
        )
    selected = np.flatnonzero(result.x[3 * n :] > 0.5)
    chosen_edge_ids = [edge_id for set_id in selected for edge_id in catalog[set_id].edge_ids]
    selected_edges = edges[chosen_edge_ids].reshape(-1, 4)
    incoming = defaultdict(int)
    outgoing = defaultdict(int)
    for edge in selected_edges:
        outgoing[int(edge[0])] += 1
        incoming[int(edge[1])] += 1
    if any(x > 1 for x in incoming.values()) or any(x > 2 for x in outgoing.values()):
        raise RuntimeError("ILP degree contract violated")
    return selected_edges, {
        "optimal": True,
        "status": int(result.status),
        "objective": float(result.fun),
        "seconds": elapsed,
        "selected_nodes": int(np.count_nonzero(result.x[:n] > 0.5)),
        "selected_node_ids": np.flatnonzero(result.x[:n] > 0.5).tolist(),
        "selected_edges": len(selected_edges),
        "selected_divisions": sum(len(catalog[i].daughters) == 2 for i in selected),
        "set_variables": k,
    }


@dataclass(frozen=True)
class Window:
    time: int
    global_nodes: np.ndarray
    global_edges: np.ndarray
    coords: np.ndarray
    features: np.ndarray
    edges: np.ndarray
    catalog: list[SetOption]
    by_parent: list[list[int]]


def split_windows(coords: np.ndarray, features: np.ndarray, edges: np.ndarray) -> list[Window]:
    """Use exactly one adjacent-frame bipartite graph per observed frame pair."""
    coords = np.asarray(coords, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.float64).reshape(-1, 4)
    if coords.ndim != 2 or coords.shape[1] != 4 or features.shape[0] != len(coords):
        raise ValueError("coordinate/feature shape mismatch")
    if not np.isfinite(coords).all() or not np.isfinite(features).all():
        raise ValueError("nonfinite coordinate or feature")
    windows = []
    for time in sorted(set(coords[:, 0].astype(int))):
        outgoing = np.flatnonzero(coords[edges[:, 0].astype(int), 0] == time)
        if not len(outgoing):
            continue
        nodes = np.flatnonzero((coords[:, 0] == time) | (coords[:, 0] == time + 1))
        lookup = np.full(len(coords), -1, dtype=np.int64)
        lookup[nodes] = np.arange(len(nodes))
        local_edges = edges[outgoing].copy()
        local_edges[:, :2] = lookup[local_edges[:, :2].astype(np.int64)]
        catalog, by_parent = enumerate_sets(coords[nodes], local_edges)
        windows.append(
            Window(
                time,
                nodes,
                outgoing,
                coords[nodes],
                features[nodes],
                local_edges,
                catalog,
                by_parent,
            )
        )
    return windows


def score_video_sets(
    model: SharedEdgeGNN,
    coords: np.ndarray,
    features: np.ndarray,
    edges: np.ndarray,
    *,
    device: str = "cpu",
) -> tuple[list[SetOption], list[list[int]], np.ndarray, np.ndarray]:
    """Run each two-frame window once and align scores to a video-wide set catalog."""
    global_catalog, by_parent = enumerate_sets(coords, edges)
    key_to_index = {(o.mother, o.daughters): i for i, o in enumerate(global_catalog)}
    scores = np.zeros(len(global_catalog), dtype=np.float64)
    probabilities = np.asarray(edges[:, 2], dtype=np.float64).copy()
    seen = np.zeros(len(global_catalog), dtype=bool)
    model.eval()
    with torch.no_grad():
        for window in split_windows(coords, features, edges):
            p, _, local_scores = model(
                torch.as_tensor(window.features, device=device),
                torch.as_tensor(window.coords, device=device),
                torch.as_tensor(window.edges, dtype=torch.float32, device=device),
                window.catalog,
            )
            probabilities[window.global_edges] = p.detach().cpu().numpy()
            for local_option, score in zip(window.catalog, local_scores, strict=True):
                if int(window.coords[local_option.mother, 0]) != window.time:
                    continue
                mother = int(window.global_nodes[local_option.mother])
                daughters = tuple(int(window.global_nodes[d]) for d in local_option.daughters)
                index = key_to_index[(mother, daughters)]
                if seen[index]:
                    raise RuntimeError("duplicate set score")
                scores[index] = float(score)
                seen[index] = True
    if not np.isfinite(scores).all() or not np.isfinite(probabilities).all():
        raise RuntimeError("nonfinite model score")
    if any(not seen[i] and option.edge_ids for i, option in enumerate(global_catalog)):
        raise RuntimeError("missing nonempty set score")
    return global_catalog, by_parent, scores, probabilities
