from __future__ import annotations

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

    pred_lineages = [{child, *pred_graph.successors(child)} for child in pred_graph.successors(pred_div)]
    lineage_edges = {
        gt_lineage: {
            pred_lineage for pred_lineage, pred_ids in enumerate(pred_lineages) if not matched_ids.isdisjoint(pred_ids)
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
