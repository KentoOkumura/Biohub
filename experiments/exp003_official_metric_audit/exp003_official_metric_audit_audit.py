# %% [markdown]
# # exp003 Official metric audit
#
# Compare the complete pinned official evaluator with the archived public
# notebook evaluator. Fixed synthetic graphs and four existing predictions
# are diagnostic inputs; these results are not independent-embryo CV.
#
# ## Contents
# 1. Imports, configuration, and offline dependencies
# 2. Complete official division and edge evaluation
# 3. Archived public evaluator
# 4. Graph conversion, input contracts, and result helpers
# 5. Synthetic graph checks
# 6. Fixed predictions against official annotations
# 7. Aggregation, evidence, and artifact persistence

# %% [markdown]
# ## 1. Imports, configuration, and offline dependencies

# %%
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, NamedTuple

import yaml

WORKING = Path("/kaggle/working")
if not Path("/kaggle/input").is_dir():
    raise RuntimeError("The authoritative audit must execute on Kaggle.")
CONFIG = yaml.safe_load((WORKING / "config.yaml").read_text())
AUDIT = CONFIG["audit"]
ARTIFACTS = WORKING / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
ASSETS = WORKING / "assets"
STARTED = time.monotonic()
wheels = sorted(
    p
    for p in Path("/kaggle/input").rglob("wheels")
    if p.is_dir() and any(p.glob("tracksdata-*.whl"))
)
if len(wheels) != 1:
    raise RuntimeError(f"expected one offline wheel directory, found {wheels}")
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(wheels[0]),
        "--no-deps",
        *AUDIT["offline_packages"],
    ],
    check=True,
)

import numpy as np
import polars as pl
import tracksdata as td
from geff import GeffMetadata
from scipy.optimize import linear_sum_assignment

print("AUDIT_CONFIG", json.dumps(AUDIT), flush=True)
SOURCE_MANIFEST = json.loads((ASSETS / "source_manifest.json").read_text())
for filename, expected in SOURCE_MANIFEST["files"].items():
    observed = hashlib.sha256((ASSETS / filename).read_bytes()).hexdigest()
    if observed != expected:
        raise RuntimeError(f"source SHA mismatch: {filename}")
assert SOURCE_MANIFEST["commit"] == AUDIT["source_commit"]
print("SOURCE_VERIFIED", SOURCE_MANIFEST["commit"], flush=True)


# %% [markdown]
# ## 2. Complete official division and edge evaluation

# %%


class DivisionCounts(NamedTuple):
    """Counts for division event evaluation."""

    tp: int
    fn: int
    fp: int


class DivisionScores(NamedTuple):
    """Result of :func:`score_divisions`.

    Attributes
    ----------
    scores : dict[int, int]
        Mapping from GT dividing-node ID to 1 (recovered) or 0 (not).
    tp_forks : set[int]
        Predicted dividing nodes paired to GT divisions.
    fp_forks : set[int]
        Predicted dividing nodes that were considered for a GT division
        but did not become a true positive, including local-topology
        rejects, bipartite leftovers, evaluable spurious forks, malformed
        local branches, and forks whose branch evidence spans distinct GT
        components.
    """

    scores: dict[int, int]
    tp_forks: set[int]
    fp_forks: set[int]


def _reset_matching_attrs(graph: td.graph.BaseGraph) -> None:
    """Reset any pre-existing match attrs in place so a fresh ``.match()`` isn't
    contaminated by stale values carried in from a previous matching pass."""
    node_keys = graph.node_attr_keys()
    if td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID in node_keys:
        node_ids = graph.node_ids()
        if len(node_ids) > 0:
            reset: dict = {td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID: -1}
            if td.DEFAULT_ATTR_KEYS.MATCH_SCORE in node_keys:
                reset[td.DEFAULT_ATTR_KEYS.MATCH_SCORE] = 0.0
            graph.update_node_attrs(node_ids=node_ids, attrs=reset)
    if td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK in graph.edge_attr_keys():
        edge_ids = graph.edge_ids()
        if len(edge_ids) > 0:
            graph.update_edge_attrs(
                edge_ids=edge_ids,
                attrs={td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK: False},
            )


def extract_divisions(
    graph: td.graph.BaseGraph,
) -> dict[int, td.graph.BaseGraph]:
    """Extract individual division events as separate subgraphs.

    Each division event includes the parent of the dividing node, the
    dividing node, its children, and the grandchildren::

        parent → divider → child1 → grandchild1
                         → child2 → grandchild2

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The input tracking graph.

    Returns
    -------
    dict[int, td.graph.BaseGraph]
        Mapping from dividing node ID to a subgraph containing the
        parent, divider, children, and grandchildren.
    """
    divisions: dict[int, td.graph.BaseGraph] = {}
    for div_node in graph.dividing_nodes():
        parents = graph.predecessors(div_node)
        children = graph.successors(div_node)
        grandchildren = [gc for child in children for gc in graph.successors(child)]
        keep = [*parents, div_node, *children, *grandchildren]
        divisions[div_node] = graph.filter(node_ids=keep).subgraph()
    return divisions


def match_divisions(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> dict[int, td.graph.BaseGraph]:
    """Match the predicted graph against each GT division subgraph.

    Extracts division events from *gt_graph* via :func:`extract_divisions`,
    then runs ``pred_graph.match(gt_div, ...)`` for each one independently.
    A fresh copy of *pred_graph* is used per division so matchings don't
    interfere.

    Parameters
    ----------
    pred_graph : td.graph.BaseGraph
        The predicted tracking graph.
    gt_graph : td.graph.BaseGraph
        The ground-truth tracking graph.
    scale : tuple[float, ...] | None
        Physical voxel scale used for centroid-distance matching.
    max_distance : float
        Maximum centroid distance for a match.

    Returns
    -------
    dict[int, td.graph.BaseGraph]
        Mapping from GT dividing-node ID to the matched copy of
        *pred_graph* for that division.
    """
    from tracksdata.metrics import DistanceMatching

    matching = DistanceMatching(max_distance=max_distance, scale=scale)

    gt_divisions = extract_divisions(gt_graph)
    matched: dict[int, td.graph.BaseGraph] = {}

    from tracksdata.options import get_options, set_options

    prev_show_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
        for div_node, gt_div in gt_divisions.items():
            pred_copy = pred_graph.copy()
            _reset_matching_attrs(pred_copy)
            with warnings.catch_warnings():
                from scipy.sparse import SparseEfficiencyWarning

                warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
                pred_copy.match(gt_div, matching=matching)
            matched[div_node] = pred_copy
    finally:
        set_options(show_progress=prev_show_progress)

    return matched


def _match_full(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None,
    max_distance: float,
) -> td.graph.BaseGraph:
    """Match the full pred graph against the full GT graph, return the matched copy."""
    from tracksdata.metrics import DistanceMatching

    matching = DistanceMatching(max_distance=max_distance, scale=scale)

    pred_copy = pred_graph.copy()
    _reset_matching_attrs(pred_copy)

    from tracksdata.options import get_options, set_options

    prev_show_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
        with warnings.catch_warnings():
            from scipy.sparse import SparseEfficiencyWarning

            warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
            pred_copy.match(gt_graph, matching=matching)
    finally:
        set_options(show_progress=prev_show_progress)

    return pred_copy


def _matched_node_attrs(graph: td.graph.BaseGraph) -> pl.DataFrame:
    """Return pred/GT node-ID pairs for matched prediction nodes."""
    node_attrs = graph.node_attrs(
        attr_keys=[
            td.DEFAULT_ATTR_KEYS.NODE_ID,
            td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
        ],
    )
    return node_attrs.filter(
        pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID).is_not_null()
        & (pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID) != -1)
    )


def _matched_division_nodes(
    matched_attrs: pl.DataFrame,
    gt_div: td.graph.BaseGraph,
    divider_id: int,
) -> tuple[set[int], list[set[int]]] | None:
    """Group matched pred nodes by their role in a GT division window.

    The parent side contains the GT divider (the parent cell) and its
    immediate predecessor (the grandparent). Each daughter side contains
    one GT child and its immediate successors (the grandchildren).
    """
    if matched_attrs.is_empty():
        return None

    node_to_gt = dict(
        zip(
            matched_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
            matched_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list(),
            strict=True,
        )
    )
    gt_children = gt_div.successors(divider_id)
    if len(gt_children) < 2:
        return None

    gt_parent_ids = {divider_id, *gt_div.predecessors(divider_id)}
    parent_ids = {pred_id for pred_id, gt_id in node_to_gt.items() if gt_id in gt_parent_ids}
    daughter_ids = [
        {
            pred_id
            for pred_id, gt_id in node_to_gt.items()
            if gt_id in {child, *gt_div.successors(child)}
        }
        for child in gt_children
    ]
    if not parent_ids or sum(bool(ids) for ids in daughter_ids) < 2:
        return None
    return parent_ids, daughter_ids


def _is_strongly_connected_division(
    pred_graph: td.graph.BaseGraph,
    pred_div: int,
    parent_ids: set[int],
    daughter_ids: list[set[int]],
) -> bool:
    """Check a predicted division's local directed topology.

    The prediction window mirrors :func:`extract_divisions`: an immediate
    predecessor (grandparent), *pred_div* (parent), its children, and their
    children (grandchildren). The parent match must be the fork itself or
    its immediate predecessor. Matches from at least two GT daughter
    lineages must occur in two distinct predicted child lineages.

    Parameters
    ----------
    pred_graph : td.graph.BaseGraph
        The predicted tracking graph.
    pred_div : int
        Candidate predicted dividing node (the parent/fork).
    parent_ids : set[int]
        Prediction node IDs matched to the GT parent side (grandparent or
        dividing parent).
    daughter_ids : list[set[int]]
        Prediction node IDs matched to each GT daughter lineage (child or
        grandchild), grouped by lineage.

    Returns
    -------
    bool
        Whether the local prediction topology connects the parent side to
        at least two distinct daughter lineages through *pred_div*.
    """
    pred_parent_ids = {pred_div, *pred_graph.predecessors(pred_div)}
    if pred_parent_ids.isdisjoint(parent_ids):
        return False

    pred_lineages = [
        {child, *pred_graph.successors(child)} for child in pred_graph.successors(pred_div)
    ]
    lineage_edges = {
        gt_lineage: {
            pred_lineage
            for pred_lineage, pred_ids in enumerate(pred_lineages)
            if not matched_ids.isdisjoint(pred_ids)
        }
        for gt_lineage, matched_ids in enumerate(daughter_ids)
    }
    return len(_bipartite_max_matching(list(lineage_edges), lineage_edges)) >= 2


def _bipartite_max_matching(
    left: list[int],
    edges: dict[int, set[int]],
) -> dict[int, int]:
    """Maximum-cardinality bipartite matching via DFS augmenting paths.

    *edges* maps each left-side vertex to the set of adjacent right-side
    vertices. Returns only the matched pairs as a ``left → right`` dict.
    """
    match_r: dict[int, int] = {}
    match_l: dict[int, int] = {}

    def augment(u: int, seen: set[int]) -> bool:
        for v in edges.get(u, ()):
            if v in seen:
                continue
            seen.add(v)
            if v not in match_r or augment(match_r[v], seen):
                match_l[u] = v
                match_r[v] = u
                return True
        return False

    for u in left:
        augment(u, set())

    return match_l


def score_divisions(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> DivisionScores:
    """Score each GT division: 1 if the prediction recovers it, 0 otherwise.

    For each GT division, the predicted graph is matched against its
    parent/divider/children/grandchildren window. Candidate pred forks are
    restricted to the matched parent-side nodes and their immediate
    successors. A candidate is valid only when its local topology contains
    a matched parent and matches from two GT daughter lineages on distinct
    predicted child branches. A fork is rejected when two direct-child
    branches have nearest matched evidence in distinct reliable GT components.
    An unmatched child may use unambiguous grandchild evidence as a fallback;
    matched children take precedence over downstream matches.

    A maximum-cardinality bipartite matching is then computed so each pred
    fork serves at most one GT division, and each GT division is paired
    with at most one pred fork. A GT division scores 1 only if paired;
    rejected candidates and valid candidates left unpaired are returned as
    false-positive forks.

    Parameters
    ----------
    pred_graph : td.graph.BaseGraph
        The predicted tracking graph.
    gt_graph : td.graph.BaseGraph
        The ground-truth tracking graph.
    scale : tuple[float, ...] | None
        Physical voxel scale used for centroid-distance matching.
    max_distance : float
        Maximum centroid distance for a match.

    Returns
    -------
    DivisionScores
        The per-division scores and the predicted forks classified as true
        positives or false positives. False-positive forks include local
        topology rejects, cross-GT-component branches, locally merged branches,
        evaluable spurious forks, and valid candidates left unmatched by the
        bipartite pairing.
    """
    matched = match_divisions(
        pred_graph,
        gt_graph,
        scale,
        max_distance,
    )
    gt_divisions = extract_divisions(gt_graph)
    pred_div_nodes = {
        node_id for node_id in pred_graph.node_ids() if pred_graph.out_degree(node_id) >= 2
    }
    evaluable_forks, cross_component_forks, malformed_forks = _pred_division_fork_sets(
        pred_graph, gt_graph, scale, max_distance
    )
    invalid_forks = cross_component_forks | malformed_forks

    candidates: dict[int, set[int]] = {}
    considered: set[int] = set()
    for div_node, matched_pred in matched.items():
        matched_nodes = _matched_division_nodes(
            _matched_node_attrs(matched_pred), gt_divisions[div_node], div_node
        )
        if matched_nodes is None:
            candidates[div_node] = set()
            continue

        parent_ids, daughter_ids = matched_nodes
        local_nodes = parent_ids | {
            successor
            for parent_id in parent_ids
            for successor in matched_pred.successors(parent_id)
        }
        local_forks = local_nodes & pred_div_nodes
        considered |= local_forks
        candidates[div_node] = {
            pred_div
            for pred_div in local_forks - invalid_forks
            if _is_strongly_connected_division(matched_pred, pred_div, parent_ids, daughter_ids)
        }

    pairing = _bipartite_max_matching(list(candidates), candidates)
    scores = {div: int(div in pairing) for div in candidates}
    tp_forks = set(pairing.values())
    # Use a set union so forks supported by multiple FP rules are counted once.
    # Invalid forks were excluded from the pairing above and therefore cannot
    # also be true positives.
    fp_forks = (considered | evaluable_forks | invalid_forks) - tp_forks
    return DivisionScores(scores=scores, tp_forks=tp_forks, fp_forks=fp_forks)


def _gt_weak_component_ids(graph: td.graph.BaseGraph) -> dict[int, int]:
    """Map each GT node to its weakly connected component ID."""
    component_ids: dict[int, int] = {}
    for seed in graph.node_ids():
        if seed in component_ids:
            continue
        component_ids[seed] = seed
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in graph.successors(current) + graph.predecessors(current):
                if neighbor not in component_ids:
                    component_ids[neighbor] = seed
                    stack.append(neighbor)
    return component_ids


def _branch_component_evidence(
    graph: td.graph.BaseGraph,
    pred_div: int,
    child: int,
    pred_to_gt: dict[int, int],
    gt_component: dict[int, int],
) -> tuple[int | None, bool]:
    """Return one GT component for a predicted child branch.

    Direct-child evidence takes precedence over grandchildren so downstream
    errors do not invalidate a correctly matched division. Grandchildren are
    fallback evidence only when the child is unmatched. The boolean marks a
    locally merged branch that cannot be assigned uniquely to this fork.
    """
    if set(graph.predecessors(child)) != {pred_div}:
        return None, True
    if child in pred_to_gt:
        return gt_component[pred_to_gt[child]], False

    grandchildren = graph.successors(child)
    if any(set(graph.predecessors(node)) != {child} for node in grandchildren):
        return None, True

    components = {gt_component[pred_to_gt[node]] for node in grandchildren if node in pred_to_gt}
    if len(components) == 1:
        return next(iter(components)), False
    return None, False


def _pred_division_fork_sets(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None,
    max_distance: float,
) -> tuple[set[int], set[int], set[int]]:
    """Return evaluable, cross-component, and malformed predicted forks.

    Cross-component evidence must come from distinct direct-child branches.
    A matched child identifies its branch; otherwise an unambiguous matched
    grandchild may identify it. Merged local branches are malformed.
    """
    matched_pred = _match_full(pred_graph, gt_graph, scale, max_distance)
    matched_attrs = _matched_node_attrs(matched_pred)
    pred_to_gt = dict(
        zip(
            matched_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
            matched_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list(),
            strict=True,
        )
    )

    pred_forks = {
        node_id for node_id in matched_pred.node_ids() if matched_pred.out_degree(node_id) >= 2
    }
    evaluable_forks = {
        pred_id
        for pred_id in pred_forks
        if pred_id in pred_to_gt and gt_graph.out_degree(pred_to_gt[pred_id]) >= 1
    }

    gt_component = _gt_weak_component_ids(gt_graph)
    cross_component_forks: set[int] = set()
    malformed_forks: set[int] = set()
    for pred_id in pred_forks:
        branch_evidence: list[int] = []
        for child in matched_pred.successors(pred_id):
            component, malformed = _branch_component_evidence(
                matched_pred, pred_id, child, pred_to_gt, gt_component
            )
            if malformed:
                malformed_forks.add(pred_id)
                break
            if component is not None:
                branch_evidence.append(component)
        else:
            if len(set(branch_evidence)) >= 2:
                cross_component_forks.add(pred_id)

    return evaluable_forks, cross_component_forks, malformed_forks


def count_matched_pred_divisions(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> int:
    """Count predicted division nodes whose matched GT node is annotated.

    Matches the full predicted graph against the full GT graph.  Among
    predicted nodes that were matched to a GT node, counts how many are
    dividing (out-degree >= 2) in the prediction *and* whose matched GT
    node has at least one child.  A matched GT node with no children marks
    the end of the annotation — we can't tell whether the cell actually
    divided there, so such predicted divisions are excluded from the count
    (and therefore from the FP tally).

    Parameters
    ----------
    pred_graph : td.graph.BaseGraph
        The predicted tracking graph.
    gt_graph : td.graph.BaseGraph
        The ground-truth tracking graph.
    scale : tuple[float, ...] | None
        Physical voxel scale used for centroid-distance matching.
    max_distance : float
        Maximum centroid distance for a match.

    Returns
    -------
    int
        Number of matched predicted division nodes.
    """
    evaluable_forks, _, _ = _pred_division_fork_sets(pred_graph, gt_graph, scale, max_distance)
    return len(evaluable_forks)


def evaluate_divisions(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> DivisionCounts:
    """Compute TP, FN, and FP counts for division events.

    - **TP**: GT divisions correctly recovered in the prediction
      (matched nodes connected and forking).
    - **FN**: GT divisions not recovered.
    - **FP**: Spurious predicted divisions, including forks matched to an
      annotated GT node, local-topology rejects, bipartite leftovers, and
      forks whose distinct child branches have nearest matched evidence in
      distinct GT components, and forks with locally merged branches. Fork IDs
      are unioned, so a fork supported by multiple rules counts once.

    Parameters
    ----------
    pred_graph : td.graph.BaseGraph
        The predicted tracking graph.
    gt_graph : td.graph.BaseGraph
        The ground-truth tracking graph.
    scale : tuple[float, ...] | None
        Physical voxel scale used for centroid-distance matching.
    max_distance : float
        Maximum centroid distance for a match.

    Returns
    -------
    DivisionCounts
        Named tuple with ``tp``, ``fn``, and ``fp`` fields.
    """
    result = score_divisions(
        pred_graph,
        gt_graph,
        scale,
        max_distance,
    )
    tp = sum(result.scores.values())
    fn = len(result.scores) - tp
    return DivisionCounts(tp=tp, fn=fn, fp=len(result.fp_forks))


# %%
from typing import NamedTuple


class EvaluationResult(NamedTuple):
    """Counts returned by :func:`evaluate`."""

    edge_tp: int
    edge_fp: int
    edge_fn: int
    division_tp: int
    division_fp: int
    division_fn: int
    num_pred_nodes: int


class DatasetsResult(NamedTuple):
    """Cumulative (micro-averaged) Jaccards plus the combined score."""

    edge_jaccard: float
    division_jaccard: float
    score: float


# Penalty coefficient for the adjusted edge Jaccard:
#   J_adj = max(0, J · (1 - ADJUSTMENT_ALPHA · total_node_ratio))
ADJUSTMENT_ALPHA: float = 0.1

# Weight of the division Jaccard in the combined run-level score:
#   score = adj_edge_jaccard + SCORE_DIVISION_WEIGHT · division_jaccard
SCORE_DIVISION_WEIGHT: float = 0.1

COUNT_COLUMNS: tuple[str, ...] = (
    "edge_tp",
    "edge_fp",
    "edge_fn",
    "division_tp",
    "division_fp",
    "division_fn",
    "num_pred_nodes",
)
METRIC_COLUMNS: tuple[str, ...] = COUNT_COLUMNS + (
    "node_recall",
    "total_node_ratio",
    "edge_jaccard",
    "adj_edge_jaccard",
)


def _jaccard(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return tp / denom if denom > 0 else float("nan")


# function is split for easier testing
def _evaluate_matched_graph(
    graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
) -> pl.DataFrame:
    edge_attrs = graph.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK])
    # Guard against duplicate edges (same source→target pair appearing multiple times).
    # tracksdata's match() inner-join marks all duplicates as matched, which inflates
    # the intersection count and can push scores above 1.0. Sort matched rows first
    # so the dedup keeps the matched copy when duplicates disagree on the mask.
    edge_attrs = edge_attrs.sort(
        td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK,
        descending=True,
    ).unique(
        subset=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET],
        keep="first",
    )
    node_attrs = graph.node_attrs(
        attr_keys=[
            td.DEFAULT_ATTR_KEYS.NODE_ID,
            td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
            td.DEFAULT_ATTR_KEYS.T,
        ]
    )

    # Drop edges that do not connect consecutive frames, i.e. keep only edges where
    # t_target == t_source + 1. This removes backward-in-time edges (t_target <= t_source)
    # and any edge spanning more than a single time step (t_target - t_source > 1).
    node_times = node_attrs.select(td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.T)
    edge_attrs = (
        edge_attrs.join(
            node_times.rename({td.DEFAULT_ATTR_KEYS.T: "_source_t"}),
            left_on=td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
            right_on=td.DEFAULT_ATTR_KEYS.NODE_ID,
            how="left",
        )
        .join(
            node_times.rename({td.DEFAULT_ATTR_KEYS.T: "_target_t"}),
            left_on=td.DEFAULT_ATTR_KEYS.EDGE_TARGET,
            right_on=td.DEFAULT_ATTR_KEYS.NODE_ID,
            how="left",
        )
        .filter(pl.col("_target_t") - pl.col("_source_t") == 1)
        .drop("_source_t", "_target_t")
    )

    # Collapse merges: when several predicted nodes match the same ground-truth
    # node, multiple predicted edges can map onto the same ground-truth edge
    # (identical matched source/target pair). tracksdata marks all of them as
    # matched, inflating the intersection. Keep only the edge with the lowest
    # EDGE_ID per matched GT edge and discard the rest with a warning.
    matched_ids = node_attrs.select(
        td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
    )
    edge_attrs = edge_attrs.join(
        matched_ids.rename({td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID: "_matched_source"}),
        left_on=td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
        right_on=td.DEFAULT_ATTR_KEYS.NODE_ID,
        how="left",
    ).join(
        matched_ids.rename({td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID: "_matched_target"}),
        left_on=td.DEFAULT_ATTR_KEYS.EDGE_TARGET,
        right_on=td.DEFAULT_ATTR_KEYS.NODE_ID,
        how="left",
    )
    # Only edges whose endpoints both match a GT node can collapse onto a GT edge.
    both_matched = (
        pl.col("_matched_source").is_not_null()
        & pl.col("_matched_target").is_not_null()
        & (pl.col("_matched_source") != -1)
        & (pl.col("_matched_target") != -1)
    )
    edge_attrs = edge_attrs.with_columns(
        (
            both_matched
            & (
                pl.col(td.DEFAULT_ATTR_KEYS.EDGE_ID)
                != pl.col(td.DEFAULT_ATTR_KEYS.EDGE_ID)
                .min()
                .over("_matched_source", "_matched_target")
            )
        ).alias("_is_merge_dup")
    )
    n_merge_dropped = int(edge_attrs["_is_merge_dup"].sum())
    if n_merge_dropped > 0:
        warnings.warn(
            f"Dropped {n_merge_dropped} merged edge(s) mapping onto the same "
            "ground-truth edge; kept the lowest edge id per merge.",
            stacklevel=2,
        )
    edge_attrs = edge_attrs.filter(~pl.col("_is_merge_dup")).drop(
        "_matched_source", "_matched_target", "_is_merge_dup"
    )

    # Cap out-degree: a dividing cell has at most two children, so a predicted node
    # with more than two outgoing edges is biologically invalid. Keep the two edges
    # with the lowest EDGE_ID per source and drop the rest with a warning.
    edge_attrs = edge_attrs.with_columns(
        pl.col(td.DEFAULT_ATTR_KEYS.EDGE_ID)
        .rank("ordinal")
        .over(td.DEFAULT_ATTR_KEYS.EDGE_SOURCE)
        .alias("_out_rank")
    )
    n_outdeg_dropped = int((edge_attrs["_out_rank"] > 2).sum())
    if n_outdeg_dropped > 0:
        warnings.warn(
            f"Dropped {n_outdeg_dropped} outgoing edge(s) from nodes with more than "
            "two children; kept the two lowest edge ids per source.",
            stacklevel=2,
        )
    edge_attrs = edge_attrs.filter(pl.col("_out_rank") <= 2).drop("_out_rank")

    # I'm assuming valid ground-truth edges are always 100% correct if they have an edge.
    # Therefore, we don't have cases where the cell divided, but not in the ground truth.
    gt_node_ids = gt_graph.node_ids()
    gt_node_attrs = pl.DataFrame(
        {
            td.DEFAULT_ATTR_KEYS.NODE_ID: gt_node_ids,
            "out_degree": gt_graph.out_degree(gt_node_ids),
            "in_degree": gt_graph.in_degree(gt_node_ids),
        }
    ).with_columns(
        (pl.col("out_degree") > 0).alias("out_valid"),
        (pl.col("in_degree") > 0).alias("in_valid"),
    )

    # merging ground truth graph into the predicted graph
    node_attrs = node_attrs.join(
        gt_node_attrs,
        left_on=td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
        right_on=td.DEFAULT_ATTR_KEYS.NODE_ID,
        how="left",
    ).with_columns(
        pl.col("out_valid").fill_null(False),
        pl.col("in_valid").fill_null(False),
    )

    # merge out valid into source and in valid into target
    edge_attrs = edge_attrs.join(
        node_attrs.select(td.DEFAULT_ATTR_KEYS.NODE_ID, "out_valid"),
        left_on=td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
        right_on=td.DEFAULT_ATTR_KEYS.NODE_ID,
        how="left",
    ).join(
        node_attrs.select(td.DEFAULT_ATTR_KEYS.NODE_ID, "in_valid"),
        left_on=td.DEFAULT_ATTR_KEYS.EDGE_TARGET,
        right_on=td.DEFAULT_ATTR_KEYS.NODE_ID,
        how="left",
    )

    edge_attrs = edge_attrs.with_columns(
        (pl.col("out_valid") | pl.col("in_valid")).alias("pred_valid"),
    )

    # sanity check that `pred_valid` is a superset of all matched edges
    assert edge_attrs.filter(td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK)["pred_valid"].all()

    return edge_attrs


def _compute_score(
    edge_attrs: pl.DataFrame,
    gt_num_edges: int,
    metric: Literal["jaccard", "dice"],
) -> float:
    intersection = int(edge_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK].sum())
    n_valid_pred_edges = int(edge_attrs["pred_valid"].sum())

    if metric == "jaccard":
        num = intersection
        denom = gt_num_edges + n_valid_pred_edges - intersection
    elif metric == "dice":
        num = 2 * intersection
        denom = gt_num_edges + n_valid_pred_edges
    else:
        raise ValueError(f"Invalid metric: {metric}")

    return num / denom if denom > 0 else float("nan")


def _evaluate(
    graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    metric: Literal["jaccard", "dice"],
    scale: tuple[float, ...] | None,
    max_distance: float,
) -> float:
    if td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID in graph.node_attr_keys():
        warnings.warn("Graph already matched, overwriting previous matching.")
        # Reset matching attributes to defaults before re-matching
        all_node_ids = graph.node_ids()
        graph.update_node_attrs(
            node_ids=all_node_ids,
            attrs={
                td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID: -1,
                td.DEFAULT_ATTR_KEYS.MATCH_SCORE: 0.0,
            },
        )
        all_edge_ids = graph.edge_ids()
        if len(all_edge_ids) > 0:
            graph.update_edge_attrs(
                edge_ids=all_edge_ids,
                attrs={td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK: False},
            )

    from tracksdata.metrics import DistanceMatching

    matching = DistanceMatching(max_distance=max_distance, scale=scale)

    if graph.num_edges() == 0 or graph.num_nodes() == 0:
        warnings.warn("Predicted graph has no edges or no nodes, returning score 0.0.")
        return 0.0

    from tracksdata.options import get_options, set_options

    prev_show_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
        with warnings.catch_warnings():
            from scipy.sparse import SparseEfficiencyWarning

            warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
            graph.match(gt_graph, matching=matching)
    finally:
        set_options(show_progress=prev_show_progress)

    edge_attrs = _evaluate_matched_graph(graph, gt_graph)

    return _compute_score(edge_attrs, gt_graph.num_edges(), metric)


def evaluate(
    graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> EvaluationResult:
    """
    Evaluate a predicted graph against a ground-truth graph using
    centroid-distance node matching.

    Computes edge TP/FP/FN, division TP/FP/FN (via
    :func:`tracking_cellmot.division_metrics.evaluate_divisions`), and the
    total number of predicted nodes (irrespective of matching).

    Parameters
    ----------
    graph : tracksdata.graph.BaseGraph
        The predicted graph. Matching attributes are written onto *graph*
        as a side effect.
    gt_graph : tracksdata.graph.BaseGraph
        The ground truth graph.
    scale : tuple[float, ...] | None, optional
        Physical scale for each spatial dimension (e.g., (z, y, x)) to
        account for anisotropy. If None, assumes isotropic data.
    max_distance : float, optional
        Maximum distance between centroids to be considered as a match.

    Returns
    -------
    EvaluationResult
    """

    # Match graph against gt_graph (in place); discard the returned score.
    _evaluate(graph, gt_graph, "jaccard", scale, max_distance)

    if graph.num_edges() == 0:
        edge_tp = 0
        edge_fp = 0
        edge_fn = gt_graph.num_edges()
    else:
        edge_attrs = _evaluate_matched_graph(graph, gt_graph)
        edge_tp = int(edge_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK].sum())
        edge_valid_pred = int(edge_attrs["pred_valid"].sum())
        edge_fp = edge_valid_pred - edge_tp
        edge_fn = gt_graph.num_edges() - edge_tp

    div = evaluate_divisions(
        graph,
        gt_graph,
        scale=scale,
        max_distance=max_distance,
    )

    return EvaluationResult(
        edge_tp=edge_tp,
        edge_fp=edge_fp,
        edge_fn=edge_fn,
        division_tp=div.tp,
        division_fp=div.fp,
        division_fn=div.fn,
        num_pred_nodes=graph.num_nodes(),
    )


def evaluate_datasets(
    graph_pairs: list[tuple[td.graph.BaseGraph, td.graph.BaseGraph]],
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> DatasetsResult:
    """Run :func:`evaluate` on each (pred, gt) pair and return cumulative
    (micro-averaged) edge and division Jaccard.

    Per-pair TP/FP/FN counts are summed across the whole list before the
    Jaccard is computed, so larger datasets dominate the score naturally.

    Parameters
    ----------
    graph_pairs : list of (pred_graph, gt_graph)
        Predicted / ground-truth graph pairs. Each *pred_graph* is mutated
        in place by matching (same side effect as :func:`evaluate`).
    scale : tuple[float, ...] | None, optional
        Physical voxel scale used for centroid-distance matching.
    max_distance : float, optional
        Maximum centroid distance for a match.

    Returns
    -------
    DatasetsResult
        Named tuple with ``edge_jaccard``, ``division_jaccard``, and the
        combined ``score = edge_jaccard + SCORE_DIVISION_WEIGHT *
        division_jaccard``. If no divisions exist anywhere in the input
        the division term is dropped and ``score = edge_jaccard``.
    """
    edge_tp = edge_fp = edge_fn = 0
    div_tp = div_fp = div_fn = 0
    for pred, gt in graph_pairs:
        r = evaluate(pred, gt, scale=scale, max_distance=max_distance)
        edge_tp += r.edge_tp
        edge_fp += r.edge_fp
        edge_fn += r.edge_fn
        div_tp += r.division_tp
        div_fp += r.division_fp
        div_fn += r.division_fn

    edge_jaccard = _jaccard(edge_tp, edge_fp, edge_fn)
    has_divisions = (div_tp + div_fp + div_fn) > 0
    division_jaccard = _jaccard(div_tp, div_fp, div_fn) if has_divisions else float("nan")
    score = (
        edge_jaccard + SCORE_DIVISION_WEIGHT * division_jaccard if has_divisions else edge_jaccard
    )

    return DatasetsResult(
        edge_jaccard=edge_jaccard,
        division_jaccard=division_jaccard,
        score=score,
    )


def _matched_node_ids(graph: td.graph.BaseGraph) -> pl.DataFrame:
    """Return a DataFrame with NODE_ID and MATCHED_NODE_ID (as Int64) for *graph*."""
    node_attrs = graph.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    )
    return node_attrs


def node_recall(
    graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
) -> float:
    """Fraction of GT nodes that were matched by a predicted node.

    The predicted graph must already be matched (e.g. via :func:`evaluate` or
    ``graph.match``).
    """
    node_attrs = _matched_node_ids(graph)
    matched = node_attrs.filter(
        pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID).is_not_null()
        & (pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID) != -1)
    )
    n_matched_gt = matched[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].n_unique()
    return n_matched_gt / gt_graph.num_nodes()


def per_sample_metrics(
    er: EvaluationResult,
    n_total: float,
    node_recall: float,
) -> dict:
    """Derive per-sample metric columns from an :class:`EvaluationResult`.

    Computes ``edge_jaccard``, ``total_node_ratio`` (``(N_pred − N_total) / N_total``),
    and the adjusted edge Jaccard ``J_adj = max(0, J · (1 − α · total_node_ratio))``
    with α = :data:`ADJUSTMENT_ALPHA`.

    Parameters
    ----------
    er
        Counts for one (pred, gt) pair — see :func:`evaluate`.
    n_total
        Target node count (e.g. from the GEFF ``estimated_number_of_nodes``
        metadata extra). Pass ``float("nan")`` when unavailable; that makes
        ``total_node_ratio`` and ``adj_edge_jaccard`` also NaN.
    node_recall
        Fraction of GT nodes matched by a predicted node.

    Returns
    -------
    dict
        One entry per key in :data:`METRIC_COLUMNS`.
    """
    if n_total > 0:
        total_node_ratio = (er.num_pred_nodes - n_total) / n_total
    else:
        total_node_ratio = float("nan")

    edge_denom = er.edge_tp + er.edge_fp + er.edge_fn
    edge_jaccard = er.edge_tp / edge_denom if edge_denom > 0 else float("nan")
    if edge_jaccard == edge_jaccard and total_node_ratio == total_node_ratio:
        adj_edge_jaccard = max(
            0.0,
            edge_jaccard * (1 - ADJUSTMENT_ALPHA * total_node_ratio),
        )
    else:
        adj_edge_jaccard = float("nan")

    return {
        "edge_tp": er.edge_tp,
        "edge_fp": er.edge_fp,
        "edge_fn": er.edge_fn,
        "division_tp": er.division_tp,
        "division_fp": er.division_fp,
        "division_fn": er.division_fn,
        "num_pred_nodes": er.num_pred_nodes,
        "node_recall": node_recall,
        "total_node_ratio": total_node_ratio,
        "edge_jaccard": edge_jaccard,
        "adj_edge_jaccard": adj_edge_jaccard,
    }


def nan_metrics_row() -> dict:
    """Return a dict with every :data:`METRIC_COLUMNS` key set to NaN."""
    return {col: float("nan") for col in METRIC_COLUMNS}


def summarise(rows: list[dict]) -> dict:
    """Aggregate per-sample metric rows into a run-level summary.

    - ``edge_jaccard`` / ``division_jaccard``: micro-averaged across valid rows
      (TP/FP/FN summed, then Jaccard).
    - ``adj_edge_jaccard``: per-sample adjusted Jaccard weight-averaged by
      sample size ``w_i = TP_i + FP_i + FN_i``; rows with NaN are skipped.
    - ``score``: ``adj_edge_jaccard + SCORE_DIVISION_WEIGHT · division_jaccard``.

    Parameters
    ----------
    rows
        Per-sample dicts as produced by :func:`per_sample_metrics`. Rows with
        NaN ``edge_tp`` are treated as failed evaluations and skipped.
    """
    valid = [r for r in rows if r["edge_tp"] == r["edge_tp"]]
    if not valid:
        return {
            "n": 0,
            "edge_jaccard": float("nan"),
            "division_jaccard": float("nan"),
            "division_tp": 0,
            "division_fp": 0,
            "division_fn": 0,
            "node_recall": float("nan"),
            "adj_edge_jaccard": float("nan"),
            "n_adj": 0,
            "score": float("nan"),
        }
    totals = {c: sum(r[c] for r in valid) for c in COUNT_COLUMNS}

    adj_rows = [r for r in valid if r["adj_edge_jaccard"] == r["adj_edge_jaccard"]]
    weights = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in adj_rows]
    total_w = sum(weights)
    if total_w > 0:
        adj_edge_jaccard = (
            sum(w * r["adj_edge_jaccard"] for w, r in zip(weights, adj_rows)) / total_w
        )
    else:
        adj_edge_jaccard = float("nan")

    division_total = totals["division_tp"] + totals["division_fp"] + totals["division_fn"]
    if division_total == 0:
        warnings.warn(
            "No divisions present across any sample in this split; "
            "dropping division term from the combined score."
        )
        division_jaccard = float("nan")
        score = adj_edge_jaccard
    else:
        division_jaccard = _jaccard(
            totals["division_tp"],
            totals["division_fp"],
            totals["division_fn"],
        )
        score = adj_edge_jaccard + SCORE_DIVISION_WEIGHT * division_jaccard
    return {
        "n": len(valid),
        "edge_jaccard": _jaccard(
            totals["edge_tp"],
            totals["edge_fp"],
            totals["edge_fn"],
        ),
        "division_jaccard": division_jaccard,
        "division_tp": totals["division_tp"],
        "division_fp": totals["division_fp"],
        "division_fn": totals["division_fn"],
        "node_recall": sum(r["node_recall"] for r in valid) / len(valid),
        "adj_edge_jaccard": adj_edge_jaccard,
        "n_adj": len(adj_rows),
        "score": score,
    }


# %% [markdown]
# ## 3. Archived public evaluator


# %%
def match_nodes_bipartite(pred_nodes: dict, gt_nodes: dict, max_dist: float = 7.0):
    pred_by_t: dict[int, list[int]] = {}
    for pid, (t, *_r) in pred_nodes.items():
        pred_by_t.setdefault(int(t), []).append(pid)
    gt_by_t: dict[int, list[int]] = {}
    for gid, (t, *_r) in gt_nodes.items():
        gt_by_t.setdefault(int(t), []).append(gid)

    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    for t, p_ids in pred_by_t.items():
        g_ids = gt_by_t.get(t, [])
        if not g_ids:
            continue
        voxel_scale = np.array(VOXEL_SCALE_UM, dtype=float)
        p_pos = np.array([pred_nodes[p][1:] for p in p_ids], dtype=float) * voxel_scale
        g_pos = np.array([gt_nodes[g][1:] for g in g_ids], dtype=float) * voxel_scale
        diff = p_pos[:, None, :] - g_pos[None, :, :]
        cost = np.sqrt((diff**2).sum(axis=-1))
        BIG = 1e6
        cost_gated = np.where(cost <= max_dist, cost, BIG)
        row_ind, col_ind = linear_sum_assignment(cost_gated)
        for r, c in zip(row_ind, col_ind):
            if cost_gated[r, c] >= BIG:
                continue
            pred_to_gt[p_ids[r]] = g_ids[c]
            gt_to_pred[g_ids[c]] = p_ids[r]
    return pred_to_gt, gt_to_pred


def compute_edge_confusion(pred_edges, gt_edges, pred_to_gt, gt_to_pred):
    gt_edge_set = set(gt_edges)
    gt_outgoing: dict[int, set[int]] = {}
    gt_incoming_source: dict[int, int] = {}
    for s, t in gt_edge_set:
        gt_outgoing.setdefault(s, set()).add(t)
        gt_incoming_source[t] = s

    tp = 0
    fp = 0
    matched_gt_edges = set()
    for s, t in pred_edges:
        ms = pred_to_gt.get(s)
        mt = pred_to_gt.get(t)
        is_tp = ms is not None and mt is not None and mt in gt_outgoing.get(ms, ())
        if is_tp:
            tp += 1
            matched_gt_edges.add((ms, mt))
            continue
        is_fp = (mt is not None and mt in gt_incoming_source) or (
            ms is not None and bool(gt_outgoing.get(ms))
        )
        if is_fp:
            fp += 1
    fn = len(gt_edge_set - matched_gt_edges)
    return tp, fp, fn


def edge_jaccard(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return tp / denom if denom else 0.0


def adjusted_jaccard(jaccard: float, t_pred: int, t_true, a: float = 0.1) -> float:
    if not t_true or t_true <= 0:
        return jaccard
    return max(0.0, jaccard * (1.0 - a * (t_pred - t_true) / t_true))


def weakly_connected_components(node_ids, edges):
    parent = {n: n for n in node_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for s, t in edges:
        if s in parent and t in parent:
            union(s, t)
    return {n: find(n) for n in node_ids}


def compute_division_confusion(pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred):
    gt_out: dict[int, set[int]] = {}
    gt_in: dict[int, int] = {}
    for s, t in gt_edges:
        gt_out.setdefault(s, set()).add(t)
        gt_in[t] = s

    pred_out: dict[int, set[int]] = {}
    for s, t in pred_edges:
        pred_out.setdefault(s, set()).add(t)

    pred_node_ids = list(pred_nodes.keys())
    pred_edge_list = list(pred_edges)
    components = weakly_connected_components(pred_node_ids, pred_edge_list)
    fork_components = {
        components[n] for n, outs in pred_out.items() if len(outs) >= 2 and n in components
    }
    gt_division_sources = [s for s, outs in gt_out.items() if len(outs) >= 2]

    def lineage_descendants(root_child: int) -> set[int]:
        seen = {root_child}
        stack = [root_child]
        while stack:
            cur = stack.pop()
            for nxt in gt_out.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    tp = 0
    fn = 0
    tp_gt_sources: set[int] = set()

    for gsrc in gt_division_sources:
        children = sorted(gt_out[gsrc])
        if len(children) < 2:
            continue
        anchor_candidates = [gsrc]
        if gsrc in gt_in:
            anchor_candidates.append(gt_in[gsrc])
        anchor_pred_nodes = [gt_to_pred[a] for a in anchor_candidates if a in gt_to_pred]

        lineage_hit_components: list[set[int]] = []
        ok = True
        for child in children[:2]:
            lineage = lineage_descendants(child)
            hit_comp_ids = {
                components[p_id]
                for gt_id in lineage
                if (p_id := gt_to_pred.get(gt_id)) is not None and p_id in components
            }
            if not hit_comp_ids:
                ok = False
                break
            lineage_hit_components.append(hit_comp_ids)

        if not ok or not anchor_pred_nodes:
            fn += 1
            continue

        anchor_comp_ids = {components[p] for p in anchor_pred_nodes if p in components}
        if not anchor_comp_ids:
            fn += 1
            continue

        found = any(
            comp_id in lineage_hit_components[0]
            and comp_id in lineage_hit_components[1]
            and comp_id in fork_components
            for comp_id in anchor_comp_ids
        )
        if found:
            tp += 1
            tp_gt_sources.add(gsrc)
        else:
            fn += 1

    fp = 0
    for n, outs in pred_out.items():
        if len(outs) < 2:
            continue
        g = pred_to_gt.get(n)
        if g is None or g not in gt_out or g in tp_gt_sources:
            continue
        fp += 1

    return tp, fp, fn


# %% [markdown]
# ## 4. Graph conversion, input contracts, and result helpers

# %%
assert ADJUSTMENT_ALPHA == AUDIT["official_adjustment_alpha"]
assert SCORE_DIVISION_WEIGHT == AUDIT["official_division_weight"]
VOXEL_SCALE_UM = tuple(AUDIT["scale_zyx_um"])


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def save_json(path, payload):
    path.write_text(
        json.dumps(clean_json(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def build_audit_graph(nodes, edges):
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    original_ids = list(nodes)
    new_ids = graph.bulk_add_nodes(
        [
            {
                "t": int(nodes[i][0]),
                "z": float(nodes[i][1]),
                "y": float(nodes[i][2]),
                "x": float(nodes[i][3]),
            }
            for i in original_ids
        ]
    )
    mapping = dict(zip(original_ids, new_ids, strict=True))
    if edges:
        graph.bulk_add_edges(
            [
                {"source_id": mapping[source], "target_id": mapping[target]}
                for source, target in edges
            ]
        )
    return graph, mapping


def graph_arrays(graph):
    nodes = {
        int(row["node_id"]): [int(row["t"]), float(row["z"]), float(row["y"]), float(row["x"])]
        for row in graph.node_attrs().iter_rows(named=True)
    }
    edges = [
        (int(row["source_id"]), int(row["target_id"]))
        for row in graph.edge_attrs().iter_rows(named=True)
    ]
    return nodes, edges


def read_csv_graphs(path):
    grouped = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            entry = grouped.setdefault(row["dataset"], {"nodes": {}, "edges": []})
            if row["row_type"] == "node":
                node_id = int(row["node_id"])
                if node_id in entry["nodes"]:
                    raise ValueError(f"duplicate node ID: {row['dataset']} {node_id}")
                entry["nodes"][node_id] = [
                    int(row["t"]),
                    float(row["z"]),
                    float(row["y"]),
                    float(row["x"]),
                ]
            elif row["row_type"] == "edge":
                entry["edges"].append((int(row["source_id"]), int(row["target_id"])))
            else:
                raise ValueError(f"unexpected row type: {row['row_type']}")
    for name, entry in grouped.items():
        for source, target in entry["edges"]:
            if source not in entry["nodes"] or target not in entry["nodes"]:
                raise ValueError(f"edge references absent node in {name}")
    return grouped


def evaluate_both(name, pred_nodes, pred_edges, gt_nodes, gt_edges, n_total, scale):
    global VOXEL_SCALE_UM
    VOXEL_SCALE_UM = tuple(scale)
    pred_graph, pred_ids = build_audit_graph(pred_nodes, pred_edges)
    gt_graph, gt_ids = build_audit_graph(gt_nodes, gt_edges)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        official = evaluate(
            pred_graph, gt_graph, scale=tuple(scale), max_distance=AUDIT["max_distance_um"]
        )
        recall = node_recall(pred_graph, gt_graph) if pred_graph.num_edges() else 0.0
        official_row = per_sample_metrics(official, n_total, recall)
        official_summary = summarise([official_row])
    pred_to_gt, gt_to_pred = match_nodes_bipartite(
        pred_nodes, gt_nodes, max_dist=AUDIT["max_distance_um"]
    )
    edge_counts = compute_edge_confusion(pred_edges, gt_edges, pred_to_gt, gt_to_pred)
    div_counts = compute_division_confusion(
        pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred
    )
    edge_score = edge_jaccard(*edge_counts)
    proxy_score = adjusted_jaccard(edge_score, len(pred_nodes), n_total, a=ADJUSTMENT_ALPHA)
    proxy_score += SCORE_DIVISION_WEIGHT * edge_jaccard(*div_counts)
    inverse_pred, inverse_gt = (
        {v: k for k, v in mapping.items()} for mapping in (pred_ids, gt_ids)
    )
    official_matches = set()
    if td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID in pred_graph.node_attr_keys():
        for row in pred_graph.node_attrs().iter_rows(named=True):
            matched = row.get(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID)
            if matched is not None and matched != -1:
                official_matches.add((inverse_pred[int(row["node_id"])], inverse_gt[int(matched)]))
    proxy_matches = set(pred_to_gt.items())
    return {
        "name": name,
        "official_counts": official._asdict(),
        "official_row": official_row,
        "official_summary": official_summary,
        "proxy_edge_tp_fp_fn": edge_counts,
        "proxy_division_tp_fp_fn": div_counts,
        "proxy_score": proxy_score,
        "score_delta_official_minus_proxy": official_summary["score"] - proxy_score,
        "node_matching_symmetric_difference_count": len(
            official_matches.symmetric_difference(proxy_matches)
        ),
        "warnings": [str(item.message) for item in caught],
        "pred_nodes": len(pred_nodes),
        "pred_edges": len(pred_edges),
        "gt_nodes": len(gt_nodes),
        "gt_edges": len(gt_edges),
    }


# %% [markdown]
# ## 5. Synthetic graph checks

# %%
fixture_definitions = json.loads((ASSETS / "fixtures.json").read_text())
assert len(fixture_definitions) == AUDIT["fixture_count"]
fixture_results = []
failures = []
for fixture in fixture_definitions:
    try:
        result = evaluate_both(
            fixture["name"],
            {int(k): v for k, v in fixture["pred_nodes"].items()},
            [tuple(e) for e in fixture["pred_edges"]],
            {int(k): v for k, v in fixture["gt_nodes"].items()},
            [tuple(e) for e in fixture["gt_edges"]],
            fixture["estimated_total_nodes"],
            AUDIT["fixture_scale_zyx_um"],
        )
        for field, expected in fixture["expected_official"].items():
            if result["official_counts"][field] != expected:
                raise AssertionError(
                    f"{fixture['name']} {field}: {result['official_counts'][field]} != {expected}"
                )
        fixture_results.append(result)
        print("FIXTURE", json.dumps(clean_json(result)), flush=True)
    except Exception as error:
        failures.append({"name": fixture["name"], "error": repr(error)})
        print("FIXTURE_FAILED", failures[-1], flush=True)
save_json(ARTIFACTS / "fixture_results.json", fixture_results)
save_json(ARTIFACTS / "failures.json", failures)
if failures:
    raise RuntimeError(f"{len(failures)} synthetic checks failed; see saved evidence.")


# %% [markdown]
# ## 6. Fixed predictions against official annotations
#
# These four public samples overlap the competition's training data. They are
# used only to compare evaluators. No CV or leaderboard claim is made.

# %%
csv_candidates = sorted(Path("/kaggle/input").rglob("submission.csv"))
prediction_paths = [p for p in csv_candidates if sha256_file(p) == AUDIT["prediction_sha256"]]
if not prediction_paths:
    raise FileNotFoundError("fixed prediction CSV with expected SHA is absent")
prediction_path = prediction_paths[0]
prediction_graphs = read_csv_graphs(prediction_path)
if sorted(prediction_graphs) != sorted(AUDIT["sample_ids"]):
    raise RuntimeError(f"unexpected prediction sample IDs: {sorted(prediction_graphs)}")
train_dirs = [
    p
    for p in Path("/kaggle/input").rglob("train")
    if p.is_dir() and all((p / f"{sample}.geff").is_dir() for sample in AUDIT["sample_ids"])
]
if len(train_dirs) != 1:
    raise RuntimeError(f"expected one matching train GEFF directory: {train_dirs}")
real_results = []
for sample in AUDIT["sample_ids"]:
    try:
        gt_path = train_dirs[0] / f"{sample}.geff"
        loaded = td.graph.IndexedRXGraph.from_geff(gt_path)
        gt_graph = loaded[0] if isinstance(loaded, tuple) else loaded
        gt_nodes, gt_edges = graph_arrays(gt_graph)
        metadata = GeffMetadata.read(gt_path)
        n_total = (metadata.extra or {}).get("estimated_number_of_nodes")
        if n_total is None or float(n_total) <= 0:
            raise ValueError(f"missing estimated total node count: {sample}")
        prediction = prediction_graphs[sample]
        result = evaluate_both(
            sample,
            prediction["nodes"],
            prediction["edges"],
            gt_nodes,
            gt_edges,
            float(n_total),
            AUDIT["scale_zyx_um"],
        )
        result["estimated_total_nodes_for_scoring_only"] = float(n_total)
        real_results.append(result)
        save_json(ARTIFACTS / "sample_results.json", real_results)
        print("SAMPLE", json.dumps(clean_json(result)), flush=True)
    except Exception as error:
        failures.append({"name": sample, "error": repr(error)})
        print("SAMPLE_FAILED", failures[-1], flush=True)
save_json(ARTIFACTS / "failures.json", failures)
if failures or len(real_results) != len(AUDIT["sample_ids"]):
    raise RuntimeError("real sample audit incomplete; inspect saved evidence")


# %% [markdown]
# ## 7. Aggregation, evidence, and artifact persistence

# %%
official_summary = summarise([row["official_row"] for row in real_results])
official_order = sorted(
    real_results, key=lambda row: row["official_summary"]["score"], reverse=True
)
proxy_order = sorted(real_results, key=lambda row: row["proxy_score"], reverse=True)
summary = {
    "diagnostic_only": True,
    "comparison": "fixed predictions; evaluator comparison, not independent CV",
    "official_summary": official_summary,
    "sample_order_by_official_score": [r["name"] for r in official_order],
    "sample_order_by_proxy_score": [r["name"] for r in proxy_order],
    "fixture_count": len(fixture_results),
    "sample_count": len(real_results),
    "failed_count": len(failures),
    "expected_sample_ids": AUDIT["sample_ids"],
    "max_absolute_sample_score_difference": max(
        abs(row["score_delta_official_minus_proxy"]) for row in real_results
    ),
    "notebook_runtime_seconds": time.monotonic() - STARTED,
}
save_json(ARTIFACTS / "audit_summary.json", summary)
with (ARTIFACTS / "sample_comparison.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "sample_id",
            "official_score",
            "proxy_score",
            "difference",
            "official_edge_tp",
            "official_edge_fp",
            "official_edge_fn",
            "official_division_tp",
            "official_division_fp",
            "official_division_fn",
            "node_matching_difference_count",
        ],
    )
    writer.writeheader()
    for row in real_results:
        writer.writerow(
            {
                "sample_id": row["name"],
                "official_score": row["official_summary"]["score"],
                "proxy_score": row["proxy_score"],
                "difference": row["score_delta_official_minus_proxy"],
                **{
                    "official_" + key: value
                    for key, value in row["official_counts"].items()
                    if key != "num_pred_nodes"
                },
                "node_matching_difference_count": row["node_matching_symmetric_difference_count"],
            }
        )
environment = {
    "python": platform.python_version(),
    "packages": {
        name: importlib.metadata.version(name)
        for name in ["numpy", "scipy", "polars", "tracksdata", "geff"]
    },
    "source_manifest_sha256": sha256_file(ASSETS / "source_manifest.json"),
    "fixture_sha256": sha256_file(ASSETS / "fixtures.json"),
    "prediction_sha256": sha256_file(prediction_path),
    "resource": "CPU",
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
}
save_json(ARTIFACTS / "environment.json", environment)
artifact_manifest = {p.name: sha256_file(p) for p in sorted(ARTIFACTS.iterdir()) if p.is_file()}
save_json(ARTIFACTS / "artifact_manifest.json", artifact_manifest)
metrics_path = WORKING / "metrics.json"
metric_record = json.loads(metrics_path.read_text())
metric_record.update(
    status="debug_completed",
    updated_at=datetime.now(timezone.utc).isoformat(),
    notes="Full CPU audit executed; experiment completion awaits user judgement.",
)
metric_record["diagnostic_validation"] = {"official_metric_audit": summary}
metric_record["evidence"]["artifacts"].update(
    input_file_sha=environment["prediction_sha256"],
    row_count=len(real_results),
    group_count=len(AUDIT["sample_ids"]),
    model_count=0,
    source_manifest_sha=environment["source_manifest_sha256"],
    audit_manifest_sha=sha256_file(ARTIFACTS / "artifact_manifest.json"),
)
metric_record["evidence"]["kaggle"].update(
    notebook_runtime_seconds=summary["notebook_runtime_seconds"],
    internet_enabled=False,
    resource={"cpu_only": True},
)
save_json(metrics_path, metric_record)
print("AUDIT_SUMMARY", json.dumps(clean_json(summary)), flush=True)
