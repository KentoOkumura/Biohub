"""Pure selection and fixed-ID readouts for the exp045 coordinate audit."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

GRID_SPACING_UM = np.array([1.625, 1.625, 1.625], dtype=np.float64)
NATIVE_SPACING_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)


def video_rank(stem: str, seed: int = 42) -> tuple[str, str]:
    return hashlib.sha256(f"{seed}:{stem}".encode()).hexdigest(), stem


def select_videos(
    eligible: Iterable[str],
    trained: Iterable[str],
    *,
    embryos: tuple[str, str] = ("44b6", "6bba"),
    per_embryo: int = 10,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Take ranks 11-20 per embryo and prove ranks 1-10 trained the head."""
    eligible_list = list(eligible)
    trained_list = list(trained)
    eligible_set = set(eligible_list)
    trained_set = set(trained_list)
    if len(eligible_set) != len(eligible_list) or len(trained_set) != len(trained_list):
        raise ValueError("duplicate video stems")
    if not trained_set <= eligible_set:
        raise ValueError("head training video absent from eligible train images")
    selection: dict[str, list[str]] = {}
    for embryo in embryos:
        ranked = sorted(
            (stem for stem in eligible_set if stem.split("_", 1)[0] == embryo),
            key=lambda stem: video_rank(stem, seed),
        )
        if len(ranked) < 2 * per_embryo:
            raise ValueError(f"{embryo}: fewer than {2 * per_embryo} eligible videos")
        if set(ranked[:per_embryo]) != trained_set.intersection(ranked):
            raise ValueError(f"{embryo}: head manifest differs from original hash selection")
        selection[embryo] = ranked[per_embryo : 2 * per_embryo]
    if trained_set.intersection(stem for group in selection.values() for stem in group):
        raise ValueError("selected a head training video")
    if len(trained_set) != len(embryos) * per_embryo:
        raise ValueError("unexpected head training video count")
    return selection


@dataclass(frozen=True)
class Match:
    gt_to_candidate: dict[int, int]
    candidate_to_gt: dict[int, int]
    matched_distance_um: dict[int, float]
    ambiguous_gt: int
    ambiguous_candidate: int


def match_known_centers(
    candidate_rows: np.ndarray,
    gt_rows: np.ndarray,
    *,
    radius_um: float = 7.0,
) -> Match:
    """Rows are [id, t, z, y, x] on the native Zarr/GEFF voxel grid."""
    candidate_rows = np.asarray(candidate_rows, dtype=np.float64)
    gt_rows = np.asarray(gt_rows, dtype=np.float64)
    if candidate_rows.ndim != 2 or candidate_rows.shape[1] != 5:
        raise ValueError("candidate rows must have 5 columns")
    if gt_rows.ndim != 2 or gt_rows.shape[1] != 5:
        raise ValueError("GT rows must have 5 columns")
    if len(set(candidate_rows[:, 0])) != len(candidate_rows):
        raise ValueError("duplicate candidate ID")
    if len(set(gt_rows[:, 0])) != len(gt_rows):
        raise ValueError("duplicate GT node ID")
    gt_to_candidate: dict[int, int] = {}
    candidate_to_gt: dict[int, int] = {}
    matched_distance_um: dict[int, float] = {}
    ambiguous_gt = 0
    ambiguous_candidate = 0
    for t in sorted(set(candidate_rows[:, 1]) | set(gt_rows[:, 1])):
        candidates = candidate_rows[candidate_rows[:, 1] == t]
        known = gt_rows[gt_rows[:, 1] == t]
        if len(candidates) == 0 or len(known) == 0:
            continue
        distances = cdist(
            candidates[:, 2:5] * NATIVE_SPACING_UM,
            known[:, 2:5] * NATIVE_SPACING_UM,
        )
        in_radius = distances <= radius_um
        ambiguous_gt += int((in_radius.sum(axis=0) > 1).sum())
        ambiguous_candidate += int((in_radius.sum(axis=1) > 1).sum())
        rows, cols = linear_sum_assignment(np.where(in_radius, distances, 1e9))
        for row, col in zip(rows, cols, strict=True):
            if not in_radius[row, col]:
                continue
            candidate_id = int(candidates[row, 0])
            gt_id = int(known[col, 0])
            gt_to_candidate[gt_id] = candidate_id
            candidate_to_gt[candidate_id] = gt_id
            matched_distance_um[gt_id] = float(distances[row, col])
    return Match(
        gt_to_candidate,
        candidate_to_gt,
        matched_distance_um,
        ambiguous_gt,
        ambiguous_candidate,
    )


def known_edges(
    gt_edges: Iterable[tuple[int, int]], match: Match
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    all_edges = {(int(source), int(target)) for source, target in gt_edges}
    mapped = {
        (match.gt_to_candidate[source], match.gt_to_candidate[target])
        for source, target in all_edges
        if source in match.gt_to_candidate and target in match.gt_to_candidate
    }
    return all_edges, mapped


def readout_stages(
    gt_edges: Iterable[tuple[int, int]],
    match: Match,
    stages: Mapping[str, Iterable[tuple[int, int]]],
) -> dict[str, dict[str, int]]:
    """Count known connections and complete annotated daughter sets by stage."""
    all_gt, mapped = known_edges(gt_edges, match)
    daughters: dict[int, set[int]] = defaultdict(set)
    for source, target in all_gt:
        daughters[source].add(target)
    divisions = {
        source: {
            (match.gt_to_candidate[source], match.gt_to_candidate[target]) for target in targets
        }
        for source, targets in daughters.items()
        if len(targets) >= 2
        and source in match.gt_to_candidate
        and all(target in match.gt_to_candidate for target in targets)
    }
    annotated_parent = {target: source for source, target in all_gt}
    output: dict[str, dict[str, int]] = {}
    for name, stage_edges in stages.items():
        selected = {(int(source), int(target)) for source, target in stage_edges}
        contradictory = 0
        for source, target in selected:
            gt_source = match.candidate_to_gt.get(source)
            gt_target = match.candidate_to_gt.get(target)
            if gt_source is not None and gt_target is not None:
                known_parent = annotated_parent.get(gt_target)
                contradictory += int(known_parent is not None and known_parent != gt_source)
        output[name] = {
            "known_edge_denominator": len(all_gt),
            "both_endpoints_detected": len(mapped),
            "known_edges_selected": len(mapped & selected),
            "known_division_denominator": sum(len(targets) >= 2 for targets in daughters.values()),
            "all_division_nodes_detected": len(divisions),
            "known_divisions_selected": sum(edges <= selected for edges in divisions.values()),
            "annotated_parent_contradictions": contradictory,
            "selected_edges": len(selected),
        }
    return output


def scored_known_edge_ranks(
    scores: Iterable[tuple[int, int, float]],
    matched_known_edges: set[tuple[int, int]],
) -> dict[tuple[int, int], tuple[float | None, int | None]]:
    known_sources = {int(source) for source, _ in matched_known_edges}
    by_source: dict[int, dict[int, float]] = defaultdict(dict)
    for source, target, score in scores:
        source = int(source)
        if source not in known_sources:
            continue
        target = int(target)
        value = float(score)
        if not np.isfinite(value):
            raise ValueError("nonfinite edge score")
        by_source[source][target] = max(by_source[source].get(target, -np.inf), value)
    result = {}
    for source, target in matched_known_edges:
        ranked = sorted(by_source.get(source, {}).items(), key=lambda item: (-item[1], item[0]))
        found = next(
            ((value, rank) for rank, (other, value) in enumerate(ranked, 1) if other == target),
            None,
        )
        result[source, target] = found if found is not None else (None, None)
    return result


def validate_initial_graph_ids(candidate_coords: np.ndarray, graph_rows: np.ndarray) -> set[int]:
    """Graph rows [id,t,z,y,x] may be a sparse subset of candidate [t,z,y,x]."""
    candidate_coords = np.asarray(candidate_coords, dtype=np.float64)
    graph_rows = np.asarray(graph_rows, dtype=np.float64)
    if candidate_coords.ndim != 2 or candidate_coords.shape[1] != 4:
        raise ValueError("candidate coordinates must have 4 columns")
    if graph_rows.ndim != 2 or graph_rows.shape[1] != 5:
        raise ValueError("graph rows must have 5 columns")
    ids = graph_rows[:, 0]
    if (
        len(ids) == 0
        or not np.isfinite(graph_rows).all()
        or not np.array_equal(ids, np.floor(ids))
        or np.any(ids < 0)
        or np.any(ids >= len(candidate_coords))
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("invalid or empty initial graph IDs")
    if not np.allclose(graph_rows[:, 1:5], candidate_coords[ids.astype(int)], rtol=0, atol=1e-4):
        raise ValueError("initial graph IDs do not map to cached candidate coordinates")
    return set(ids.astype(int))


def trace_original_edges(
    edges: Iterable[tuple[int, int]], original_ids: set[int]
) -> set[tuple[int, int]]:
    """Exclude nodes created after the initial ILP graph, even if IDs are reused."""
    return {
        (int(source), int(target))
        for source, target in edges
        if int(source) in original_ids and int(target) in original_ids
    }
