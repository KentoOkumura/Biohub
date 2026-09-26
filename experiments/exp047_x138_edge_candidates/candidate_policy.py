"""Fixed exp047 candidate-edge policy on exp043 fused softmax scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BASELINE_THRESHOLD = 0.48
EXPANDED_THRESHOLD = 0.10
MAX_PARENTS = 3


@dataclass(frozen=True)
class CandidateEdges:
    baseline: np.ndarray
    expanded: np.ndarray
    added: np.ndarray


def select_candidate_edges(
    coords: np.ndarray,
    edge_src: np.ndarray,
    edge_tgt: np.ndarray,
    edge_prob: np.ndarray,
    admitted: np.ndarray,
) -> CandidateEdges:
    """Return Nx4 (source, target, probability, distance) arrays.

    The parent source prediction's admitted edges are authoritative for the
    baseline. An added edge keeps the *same* fused softmax probability. The
    distance attribute is informational; the unchanged ILP consumes edge_prob.
    """
    coords = np.asarray(coords)
    src = np.asarray(edge_src)
    tgt = np.asarray(edge_tgt)
    prob = np.asarray(edge_prob)
    baseline = np.asarray(admitted, dtype=np.float64).reshape(-1, 4)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("coords must be Nx4")
    if (
        src.ndim != 1
        or tgt.ndim != 1
        or prob.ndim != 1
        or len(src) != len(tgt)
        or len(src) != len(prob)
    ):
        raise ValueError("score arrays must be equal-length vectors")
    if (
        not np.isfinite(coords).all()
        or not np.isfinite(prob).all()
        or not np.isfinite(baseline).all()
    ):
        raise ValueError("nonfinite candidate data")
    if np.any(prob <= 0) or np.any(prob > 1):
        raise ValueError("invalid softmax probability")
    if (
        np.any(src < 0)
        or np.any(tgt < 0)
        or np.any(src >= len(coords))
        or np.any(tgt >= len(coords))
    ):
        raise ValueError("score edge IDs outside node range")
    if len(baseline) and (np.any(baseline[:, :2] < 0) or np.any(baseline[:, :2] >= len(coords))):
        raise ValueError("admitted edge IDs outside node range")
    if len(baseline) and (
        np.any(baseline[:, 2] <= BASELINE_THRESHOLD) or np.any(baseline[:, 2] > 1)
    ):
        raise ValueError("baseline threshold drift")
    if np.any(coords[src, 0] + 1 != coords[tgt, 0]):
        raise ValueError("score edge does not link adjacent frames")
    eligible = np.flatnonzero(prob > EXPANDED_THRESHOLD)
    src = src[eligible]
    tgt = tgt[eligible]
    prob = prob[eligible]
    scored = {(int(s), int(t)): float(p) for s, t, p in zip(src, tgt, prob, strict=True)}
    if len(scored) != len(src):
        raise ValueError("duplicate score edge")
    baseline_pairs = {(int(row[0]), int(row[1])) for row in baseline}
    if len(baseline_pairs) != len(baseline):
        raise ValueError("duplicate admitted edge")
    scored_baseline = {pair for pair, p in scored.items() if p > BASELINE_THRESHOLD}
    if scored_baseline != baseline_pairs:
        raise ValueError(
            {
                "baseline_cache_mismatch": {
                    "missing": len(scored_baseline - baseline_pairs),
                    "extra": len(baseline_pairs - scored_baseline),
                }
            }
        )
    for row in baseline:
        pair = (int(row[0]), int(row[1]))
        if not np.isclose(float(row[2]), scored[pair], rtol=2e-6, atol=1e-8):
            raise ValueError("baseline softmax probability mismatch")

    # np.lexsort uses the last key first: daughter, decreasing score, parent ID.
    order = np.lexsort((src, -prob, tgt))
    chosen: list[int] = []
    last_target = -1
    rank = 0
    for idx in order:
        daughter = int(tgt[idx])
        if daughter != last_target:
            last_target, rank = daughter, 0
        if rank < MAX_PARENTS:
            chosen.append(int(idx))
        rank += 1
    additional_indices = [
        idx for idx in chosen if (int(src[idx]), int(tgt[idx])) not in baseline_pairs
    ]
    additional = np.empty((len(additional_indices), 4), dtype=np.float64)
    if additional_indices:
        ids = np.asarray(additional_indices, dtype=np.int64)
        additional[:, 0] = src[ids]
        additional[:, 1] = tgt[ids]
        additional[:, 2] = prob[ids]
        additional[:, 3] = np.linalg.norm(
            coords[src[ids], 1:].astype(np.float64) - coords[tgt[ids], 1:].astype(np.float64),
            axis=1,
        )
    expanded = np.concatenate([baseline, additional], axis=0)
    if len({(int(row[0]), int(row[1])) for row in expanded}) != len(expanded):
        raise ValueError("expanded candidate duplicate")
    if len(expanded):
        _, daughter_counts = np.unique(expanded[:, 1].astype(np.int64), return_counts=True)
        if np.any(daughter_counts > MAX_PARENTS):
            raise ValueError("expanded candidate parent count exceeds three")
    return CandidateEdges(baseline=baseline, expanded=expanded, added=additional)
