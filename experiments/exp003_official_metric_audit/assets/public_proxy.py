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
        cost = np.sqrt((diff ** 2).sum(axis=-1))
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
