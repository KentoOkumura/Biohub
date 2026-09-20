# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp025: fixed divisions and 3D Kalman association
#
# Compare image-only assignment and assignment with self-predicted motion.
# Both variants keep identical observed centers, ordinary candidate edges,
# image probabilities and reserved parent/two-daughter predictions.
# Ground truth is used only by calibration and evaluation functions.
#
# ## Contents
# 1. Imports, configuration and evidence
# 2. Fixed graphs and division reservations
# 3. Six-dimensional Kalman state
# 4. Sparse components and assignment with no match
# 5. Sequential tracking and invariant checks
# 6. Training-side noise and cost calibration
# 7. Input discovery, cached tracker replay and official evaluation
# 8. Setup, calibration, outer evaluation and artifacts

# %% [markdown]
# ## 1. Imports, configuration and evidence

# %%
from __future__ import annotations

import fnmatch
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

EXPERIMENT = "exp025_kalman_hungarian_links"


def json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def require_sha(path: Path, expected: str) -> Path:
    if not path.is_file() or file_sha(path) != expected:
        raise ValueError(f"input checksum mismatch: {path}")
    return path


def check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("experiment runtime gate exceeded")


# %% [markdown]
# ## 2. Fixed graphs and division reservations
#
# Graph positions are physical z/y/x in micrometers. Ordinary candidates keep
# their original detector coordinates. Already saved repair-only division
# endpoints are immutable reserved inputs; they never create ordinary edges.


# %%
@dataclass
class Graph:
    ids: np.ndarray
    frames: np.ndarray
    positions: np.ndarray
    edges: np.ndarray
    probabilities: np.ndarray

    def validate(self, *, scored: bool = True) -> None:
        n = len(self.ids)
        if self.ids.shape != (n,) or len(set(self.ids.tolist())) != n:
            raise ValueError("duplicate or invalid node IDs")
        if self.frames.shape != (n,) or self.positions.shape != (n, 3):
            raise ValueError("invalid node arrays")
        if self.edges.shape != (len(self.edges), 2):
            raise ValueError("invalid edge array")
        if not np.isfinite(self.positions).all() or not np.isfinite(self.frames).all():
            raise ValueError("non-finite node")
        if not np.equal(self.frames, np.floor(self.frames)).all():
            raise ValueError("fractional frame")
        if len(set(map(tuple, self.edges))) != len(self.edges):
            raise ValueError("duplicate edges")
        by_id = dict(zip(map(int, self.ids), map(int, self.frames), strict=True))
        for a, b in self.edges:
            if int(a) not in by_id or int(b) not in by_id:
                raise ValueError("dangling edge")
            if by_id[int(b)] != by_id[int(a)] + 1:
                raise ValueError("non-adjacent edge")
        if scored and (
            self.probabilities.shape != (len(self.edges),)
            or not np.isfinite(self.probabilities).all()
            or np.any(self.probabilities <= 0)
            or np.any(self.probabilities > 1)
        ):
            raise ValueError("invalid edge probabilities")

    def digest(self) -> str:
        nodes = sorted(
            [int(i), int(t), *map(float, p)]
            for i, t, p in zip(self.ids, self.frames, self.positions, strict=True)
        )
        edges = sorted(
            [int(a), int(b), float(p)]
            for (a, b), p in zip(self.edges, self.probabilities, strict=True)
        )
        return json_sha({"nodes": nodes, "edges": edges})


def load_geff(path: Path, scale: np.ndarray, *, scored: bool = True) -> Graph:
    import zarr

    group = zarr.open_group(str(path), mode="r")
    ids = np.asarray(group["nodes/ids"], dtype=np.int64)
    frames = np.asarray(group["nodes/props/t/values"])
    positions = (
        np.column_stack(
            [np.asarray(group[f"nodes/props/{axis}/values"], dtype=np.float64) for axis in "zyx"]
        )
        * scale
    )
    edges = np.asarray(group["edges/ids"], dtype=np.int64).reshape(-1, 2)
    probabilities = (
        np.asarray(group["edges/props/edge_prob/values"], dtype=np.float64)
        if scored
        else np.ones(len(edges))
    )
    result = Graph(ids, frames, positions, edges, probabilities)
    result.validate(scored=scored)
    return result


def load_compact(path: Path, scale: np.ndarray) -> Graph:
    with np.load(path, allow_pickle=False) as saved:
        ids = saved["node_ids"].astype(np.int64)
        tzyx = saved["node_tzyx"].astype(np.float64)
        edges = saved["edges"].astype(np.int64).reshape(-1, 2)
    result = Graph(ids, tzyx[:, 0], tzyx[:, 1:] * scale, edges, np.ones(len(edges)))
    result.validate(scored=False)
    return result


def reserve_divisions(candidates: Graph, reference: Graph) -> tuple[Graph, np.ndarray, dict]:
    candidates.validate()
    reference.validate(scored=False)
    outgoing = Counter(map(int, reference.edges[:, 0]))
    if any(count > 2 for count in outgoing.values()):
        raise ValueError("reference has more than two daughters")
    reservations = np.asarray(
        sorted((int(a), int(b)) for a, b in reference.edges if outgoing[int(a)] == 2),
        dtype=np.int64,
    ).reshape(-1, 2)
    if len(set(reservations[:, 1].tolist())) != len(reservations):
        raise ValueError("reserved daughter has multiple parents")
    missing = sorted(set(reservations.flatten().tolist()) - set(candidates.ids.tolist()))
    reference_index = {int(i): n for n, i in enumerate(reference.ids)}
    indices = [reference_index[i] for i in missing]
    graph = Graph(
        np.concatenate([candidates.ids, np.asarray(missing, dtype=np.int64)]),
        np.concatenate([candidates.frames, reference.frames[indices]]),
        np.concatenate([candidates.positions, reference.positions[indices]], axis=0),
        candidates.edges.copy(),
        candidates.probabilities.copy(),
    )
    graph.validate()
    frame_by_id = dict(zip(map(int, graph.ids), map(int, graph.frames), strict=True))
    if any(frame_by_id[int(b)] != frame_by_id[int(a)] + 1 for a, b in reservations):
        raise ValueError("reservation and candidate frame identity disagree")
    support = set(map(tuple, candidates.edges))
    return (
        graph,
        reservations,
        {
            "candidate_sha256": candidates.digest(),
            "input_sha256": graph.digest(),
            "reservation_sha256": json_sha(reservations.tolist()),
            "reserved_divisions": len(reservations) // 2,
            "reserved_only_node_ids": missing,
            "reserved_edges_outside_ordinary_support": sum(
                tuple(edge) not in support for edge in reservations
            ),
        },
    )


# %% [markdown]
# ## 3. Six-dimensional Kalman state
#
# Use a unit frame interval and a random acceleration covariance. The filter's
# state is internal; observed output coordinates remain unchanged. A daughter
# starts a new identity with the mother's velocity, inflated velocity covariance
# and zero position/velocity cross-covariance.


# %%
@dataclass
class State:
    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    history: int


def validate_state(state: State) -> None:
    p = state.covariance
    if state.mean.shape != (6,) or p.shape != (6, 6):
        raise ValueError("invalid Kalman state dimensions")
    if not np.isfinite(state.mean).all() or not np.isfinite(p).all():
        raise ValueError("non-finite Kalman state")
    if not np.allclose(p, p.T, atol=1e-9, rtol=1e-9):
        raise ValueError("asymmetric covariance")
    np.linalg.cholesky(p)


def initial_state(
    position: np.ndarray, track_id: int, noise: dict, mother: State | None = None
) -> State:
    mean = np.concatenate([position, np.zeros(3) if mother is None else mother.mean[3:]])
    covariance = np.zeros((6, 6), dtype=np.float64)
    covariance[:3, :3] = np.diag(noise["observation_variance"])
    covariance[3:, 3:] = (
        np.diag(noise["initial_velocity_variance"])
        if mother is None
        else mother.covariance[3:, 3:] * float(noise["daughter_covariance_multiplier"])
    )
    result = State(track_id, mean, covariance, 1)
    validate_state(result)
    return result


def predict(state: State, noise: dict) -> State:
    transition = np.eye(6)
    transition[:3, 3:] = np.eye(3)
    acceleration = np.vstack([0.5 * np.eye(3), np.eye(3)])
    process = acceleration @ np.diag(noise["acceleration_variance"]) @ acceleration.T
    covariance = transition @ state.covariance @ transition.T + process
    return State(state.track_id, transition @ state.mean, covariance, state.history)


def innovation_cost(state: State, position: np.ndarray, noise: dict) -> float:
    covariance = state.covariance[:3, :3] + np.diag(noise["observation_variance"])
    chol = np.linalg.cholesky(covariance)
    error = np.linalg.solve(chol, position - state.mean[:3])
    return float(0.5 * (error @ error + 2 * np.log(np.diag(chol)).sum() + 3 * np.log(2 * np.pi)))


def update(state: State, position: np.ndarray, noise: dict) -> State:
    observation = np.diag(noise["observation_variance"])
    innovation = state.covariance[:3, :3] + observation
    gain = np.linalg.solve(innovation, state.covariance[:3, :]).T
    mean = state.mean + gain @ (position - state.mean[:3])
    identity_minus_gain = np.eye(6)
    identity_minus_gain[:, :3] -= gain
    covariance = (
        identity_minus_gain @ state.covariance @ identity_minus_gain.T + gain @ observation @ gain.T
    )
    covariance = (covariance + covariance.T) / 2
    result = State(state.track_id, mean, covariance, state.history + 1)
    validate_state(result)
    return result


# %% [markdown]
# ## 4. Sparse components and assignment with no match
#
# Absent edges have infinite cost. A source and a target each pay half the
# configured no-match cost. Sorted node IDs define matrix order. Components
# are exact graph components; memory limits stop execution instead of pruning.


# %%
def assign_with_no_match(
    edges: list[tuple[int, int]], costs: list[float], no_match_cost: float, max_matrix_bytes: int
) -> list[tuple[int, int]]:
    if not math.isfinite(no_match_cost) or no_match_cost < 0:
        raise ValueError("invalid no-match cost")
    if len(edges) != len(costs) or not np.isfinite(costs).all():
        raise ValueError("invalid assignment costs")
    if len(set(edges)) != len(edges):
        raise ValueError("duplicate assignment edges")
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for a, b in edges:
        adjacency[0, a].append((1, b))
        adjacency[1, b].append((0, a))
    costs_by_edge = dict(zip(edges, costs, strict=True))
    visited = set()
    chosen = []
    for start in sorted(adjacency):
        if start in visited:
            continue
        stack, component = [start], []
        visited.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        sources = sorted(i for side, i in component if side == 0)
        targets = sorted(i for side, i in component if side == 1)
        n, m = len(sources), len(targets)
        if (n + m) ** 2 * np.dtype(np.float64).itemsize > max_matrix_bytes:
            raise MemoryError(f"assignment component too large: {n} x {m}")
        matrix = np.full((n + m, n + m), np.inf)
        target_index = {b: j for j, b in enumerate(targets)}
        for i, a in enumerate(sources):
            for _, b in adjacency[0, a]:
                matrix[i, target_index[b]] = costs_by_edge[a, b]
        matrix[np.arange(n), m + np.arange(n)] = no_match_cost / 2
        matrix[n + np.arange(m), np.arange(m)] = no_match_cost / 2
        matrix[n:, m:] = 0
        rows, columns = linear_sum_assignment(matrix)
        for i, j in zip(rows, columns, strict=True):
            if i < n and j < m:
                chosen.append((sources[i], targets[j]))
    return sorted(chosen)


# %% [markdown]
# ## 5. Sequential tracking and invariant checks


# %%
def validate_output(graph: Graph, reservations: np.ndarray, selected: np.ndarray) -> None:
    selected_set = set(map(tuple, selected))
    reserved_set = set(map(tuple, reservations))
    if len(selected_set) != len(selected) or not reserved_set <= selected_set:
        raise ValueError("duplicate output or lost reserved division")
    if not selected_set <= set(map(tuple, graph.edges)) | reserved_set:
        raise ValueError("output edge outside fixed support")
    incoming = Counter(int(b) for _, b in selected)
    outgoing = Counter(int(a) for a, _ in selected)
    mothers = set(reservations[:, 0].tolist())
    if any(n > 1 for n in incoming.values()):
        raise ValueError("multiple parents")
    if any(n != 2 if a in mothers else n > 1 for a, n in outgoing.items()):
        raise ValueError("changed division capacity")
    frame = dict(zip(map(int, graph.ids), map(int, graph.frames), strict=True))
    if any(frame[int(b)] != frame[int(a)] + 1 for a, b in selected):
        raise ValueError("non-adjacent output edge")


def decode(
    graph: Graph,
    reservations: np.ndarray,
    noise: dict,
    *,
    motion_weight: float,
    no_match_cost: float,
    cfg: dict,
) -> dict:
    graph.validate()
    if not math.isfinite(motion_weight) or motion_weight < 0:
        raise ValueError("invalid motion weight")
    reservations = np.asarray(reservations, dtype=np.int64).reshape(-1, 2)
    validate_output(graph, reservations, reservations)
    index = {int(i): n for n, i in enumerate(graph.ids)}
    nodes_by_frame: dict[int, list[int]] = defaultdict(list)
    for i, t in zip(graph.ids, graph.frames, strict=True):
        nodes_by_frame[int(t)].append(int(i))
    frame_edges: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for (a, b), p in zip(graph.edges, graph.probabilities, strict=True):
        frame_edges[int(graph.frames[index[int(a)]])].append((int(a), int(b), float(p)))
    reserved_by_frame: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for a, b in reservations:
        reserved_by_frame[int(graph.frames[index[int(a)]])].append((int(a), int(b)))
    active: dict[int, State] = {}
    selected, next_track_id = [], 0
    n = len(graph.ids)
    means, covariances = np.zeros((n, 6)), np.zeros((n, 6, 6))
    track_ids, histories = np.full(n, -1, dtype=np.int64), np.zeros(n, dtype=np.int64)
    association_diagnostics = []
    for frame in sorted(nodes_by_frame):
        ids = sorted(nodes_by_frame[frame])
        frame_ids = set(ids)
        active = {i: state for i, state in active.items() if i in frame_ids}
        for i in ids:
            if i not in active:
                active[i] = initial_state(graph.positions[index[i]], next_track_id, noise)
                next_track_id += 1
            state = active[i]
            validate_state(state)
            row = index[i]
            means[row], covariances[row] = state.mean, state.covariance
            track_ids[row], histories[row] = state.track_id, state.history
        reserved = sorted(reserved_by_frame[frame])
        mothers, daughters = {a for a, _ in reserved}, {b for _, b in reserved}
        predictions = {i: predict(active[i], noise) for i in ids}
        pairs, costs = [], []
        for a, b, probability in sorted(frame_edges[frame]):
            if a in mothers or b in daughters:
                continue
            image_cost = -math.log(probability)
            nll = innovation_cost(predictions[a], graph.positions[index[b]], noise)
            motion = (nll - float(noise["nll_center"])) / float(noise["nll_scale"])
            pairs.append((a, b))
            costs.append(image_cost + motion_weight * motion)
        normal = assign_with_no_match(
            pairs, costs, no_match_cost, int(cfg["max_assignment_matrix_bytes"])
        )
        next_active = {}
        for a, b in normal:
            next_active[b] = update(predictions[a], graph.positions[index[b]], noise)
            association_diagnostics.append(
                [
                    a,
                    b,
                    active[a].history,
                    innovation_cost(predictions[a], graph.positions[index[b]], noise),
                ]
            )
        for a, b in reserved:
            next_active[b] = initial_state(
                graph.positions[index[b]], next_track_id, noise, predictions[a]
            )
            next_track_id += 1
        selected.extend(normal + reserved)
        active = next_active
    selected_array = np.asarray(sorted(selected), dtype=np.int64).reshape(-1, 2)
    validate_output(graph, reservations, selected_array)
    if np.any(track_ids < 0):
        raise ValueError("uninitialized output state")
    return {
        "edges": selected_array,
        "means": means,
        "covariances": covariances,
        "track_ids": track_ids,
        "histories": histories,
        "association_diagnostics": np.asarray(association_diagnostics).reshape(-1, 4),
        "track_count": next_track_id,
        "no_match_sources": n - len(set(selected_array[:, 0].tolist())),
        "no_match_targets": n - len(set(selected_array[:, 1].tolist())),
    }


# %% [markdown]
# ## 6. Training-side noise and cost calibration
#
# Reuse the parent's 5 micrometer greedy matching for noise diagnostics, not
# the official evaluator's 7 micrometer matching. Unannotated detections remain
# unknown. No function below receives an outer-embryo label during selection.


# %%
def matched_nodes(graph: Graph, truth: Graph, radius: float) -> dict[int, int]:
    result = {}
    for frame in sorted(set(graph.frames.tolist())):
        candidate = np.flatnonzero(graph.frames == frame)
        annotated = np.flatnonzero(truth.frames == frame)
        candidate = candidate[np.argsort(graph.ids[candidate], kind="stable")]
        annotated = annotated[np.argsort(truth.ids[annotated], kind="stable")]
        if not len(candidate) or not len(annotated):
            continue
        # Each candidate proposes only its nearest annotation, as in exp016.
        distances = np.linalg.norm(
            graph.positions[candidate, None] - truth.positions[None, annotated], axis=2
        )
        nearest = distances.argmin(axis=1)
        nearest_distance = distances[np.arange(len(candidate)), nearest]
        taken = set()
        for i in np.argsort(nearest_distance, kind="stable"):
            j = int(nearest[i])
            if nearest_distance[i] <= radius and j not in taken:
                taken.add(j)
                result[int(graph.ids[candidate[i]])] = int(truth.ids[annotated[j]])
    return result


def fit_noise(records: list[tuple[str, Graph, Graph]], train_embryo: str, cfg: dict) -> dict:
    residuals, velocities, accelerations = [], [], []
    for sample, graph, truth in records:
        if sample.split("_", 1)[0] != train_embryo:
            raise ValueError("outer embryo supplied to noise fit")
        match = matched_nodes(graph, truth, float(cfg["teacher_matching_um"]))
        observed = {int(i): p for i, p in zip(graph.ids, graph.positions, strict=True)}
        annotated = {int(i): p for i, p in zip(truth.ids, truth.positions, strict=True)}
        residuals.extend(observed[i] - annotated[j] for i, j in match.items())
        children: dict[int, list[int]] = defaultdict(list)
        parents: dict[int, list[int]] = defaultdict(list)
        for a, b in truth.edges:
            children[int(a)].append(int(b))
            parents[int(b)].append(int(a))
        for a, targets in children.items():
            if len(targets) != 1:
                continue
            b = targets[0]
            if len(parents[b]) != 1:
                continue
            velocities.append(annotated[b] - annotated[a])
            if len(children.get(b, [])) == 1:
                c = children[b][0]
                if len(parents[c]) == 1:
                    accelerations.append(annotated[c] - 2 * annotated[b] + annotated[a])
    counts = [len(residuals), len(velocities), len(accelerations)]
    if min(counts) < int(cfg["minimum_noise_observations"]):
        raise ValueError(f"insufficient training-only noise observations: {counts}")
    result = {
        "observation_variance": np.maximum(
            np.mean(np.square(residuals), axis=0), cfg["observation_variance_floor"]
        ).tolist(),
        "acceleration_variance": np.maximum(
            np.mean(np.square(accelerations), axis=0), cfg["acceleration_variance_floor"]
        ).tolist(),
        "initial_velocity_variance": np.maximum(
            np.mean(np.square(velocities), axis=0), cfg["velocity_variance_floor"]
        ).tolist(),
        "daughter_covariance_multiplier": cfg["daughter_covariance_multiplier"],
        "nll_center": 0.0,
        "nll_scale": 1.0,
        "fit_embryo": train_embryo,
        "counts": dict(zip(["residuals", "velocities", "accelerations"], counts, strict=True)),
        "fit_samples": sorted(sample for sample, _, _ in records),
        "training_input_sha256": json_sha(
            [[sample, graph.digest(), truth.digest()] for sample, graph, truth in records]
        ),
    }
    # Analytic innovation scale for the initial prior, estimated only from fit data.
    covariance = np.asarray(result["observation_variance"]) * 2
    covariance += np.asarray(result["initial_velocity_variance"])
    covariance += 0.25 * np.asarray(result["acceleration_variance"])
    result["nll_center"] = float(0.5 * (np.log(covariance).sum() + 3 * np.log(2 * np.pi)))
    result["nll_scale"] = float(cfg["motion_nll_scale"])
    return result


def validate_split(split: dict, samples: list[str]) -> None:
    fit = set(split["gradient_update"])
    internal = set(split["internal_validation"])
    outer = set(split["outer_evaluation"])
    if not fit or not internal or not outer or fit & internal or (fit | internal) & outer:
        raise ValueError("overlapping or empty calibration split")
    if fit | internal | outer != set(samples) or len(set(samples)) != len(samples):
        raise ValueError("split does not cover input samples")
    if any(s.split("_", 1)[0] != split["train_embryo"] for s in fit | internal):
        raise ValueError("foreign embryo in calibration")
    if any(s.split("_", 1)[0] != split["evaluation_embryo"] for s in outer):
        raise ValueError("foreign embryo in outer evaluation")
    if split["train_embryo"] == split["evaluation_embryo"]:
        raise ValueError("train and evaluation embryos coincide")


def choose_costs(rows: list[dict], cfg: dict) -> dict:
    baseline = [r for r in rows if r["motion_weight"] == 0]
    expected_null = list(map(float, cfg["no_match_cost_grid"]))
    if sorted(r["no_match_cost"] for r in baseline) != sorted(expected_null):
        raise ValueError("incomplete image-only calibration grid")
    if any(not math.isfinite(float(r["combined_score"])) for r in rows):
        raise ValueError("non-finite calibration metric")
    best_image = max(
        baseline, key=lambda r: (r["combined_score"], -expected_null.index(r["no_match_cost"]))
    )
    null = best_image["no_match_cost"]
    motion = [r for r in rows if r["no_match_cost"] == null and r["motion_weight"] > 0]
    expected_weights = list(map(float, cfg["motion_weight_grid"]))
    if sorted(r["motion_weight"] for r in motion) != sorted(expected_weights):
        raise ValueError("incomplete motion calibration grid")
    best_motion = max(
        motion, key=lambda r: (r["combined_score"], -expected_weights.index(r["motion_weight"]))
    )
    return {"no_match_cost": null, "motion_weight": best_motion["motion_weight"]}


def diagnostic_counts(graph: Graph, truth: Graph, output: dict, cfg: dict) -> list[dict]:
    match = matched_nodes(graph, truth, float(cfg["teacher_matching_um"]))
    annotated_edges = set(map(tuple, truth.edges))
    gt_children = Counter(int(a) for a, _ in truth.edges)
    index = {int(i): n for n, i in enumerate(graph.ids)}
    spacing = np.full(len(graph.ids), np.inf)
    for frame in set(graph.frames.tolist()):
        rows = np.flatnonzero(graph.frames == frame)
        if len(rows) > 1:
            spacing[rows] = cKDTree(graph.positions[rows]).query(graph.positions[rows], k=2)[0][
                :, 1
            ]
    selected = set(map(tuple, output["edges"]))
    inverse = {gt: candidate for candidate, gt in match.items()}
    known = {
        (inverse[int(a)], inverse[int(b)])
        for a, b in truth.edges
        if int(a) in inverse and int(b) in inverse
    }
    groups: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for a in graph.ids:
        row = index[int(a)]
        history = int(output["histories"][row])
        bucket = "1" if history == 1 else "2-4" if history <= 4 else "5+"
        density = "near" if spacing[row] < float(cfg["density_distance_um"]) else "far"
        group = groups[bucket, density]
        group["nodes"] += 1
    for a, b in selected | known:
        row = index[int(a)]
        history = int(output["histories"][row])
        bucket = "1" if history == 1 else "2-4" if history <= 4 else "5+"
        density = "near" if spacing[row] < float(cfg["density_distance_um"]) else "far"
        group = groups[bucket, density]
        if (a, b) in known:
            group["known_edges"] += 1
            group["recovered_known_edges"] += int((a, b) in selected)
        if (a, b) not in selected:
            continue
        group["selected_edges"] += 1
        if a not in match or b not in match:
            group["unknown_selected_edges"] += 1
        elif (match[a], match[b]) not in annotated_edges:
            # Only mark wrong when source has an annotated successor at this step.
            group["known_wrong_edges" if gt_children[match[a]] else "unknown_selected_edges"] += 1
    return [
        {"history": h, "density": d, **dict(values)} for (h, d), values in sorted(groups.items())
    ]


# %% [markdown]
# ## 7. Input discovery, cached tracker replay and official evaluation
#
# The following cache-reader and tracker functions retain the parent computation.
# Only required definitions are embedded; no experiment-local imports are needed.

# %%

# Extracted from window_cache.py
# source SHA256 47a9f02234e50393cd69a48c9cbe7f0951c24cb31a7474aef28adec3a44b1d94

CACHE_SCHEMA_VERSION = 1

METADATA_KEY = "__metadata_json__"


def array_schema(arrays: dict[str, np.ndarray]) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "dtype": np.asarray(arrays[name]).dtype.str,
            "shape": list(np.asarray(arrays[name]).shape),
        }
        for name in sorted(arrays)
    ]


def array_content_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for item in array_schema(arrays):
        name = str(item["name"])
        array = np.ascontiguousarray(arrays[name])
        digest.update(json.dumps(item, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def read_window_cache(
    path: Path,
    *,
    expected_metadata: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as saved:
        if METADATA_KEY not in saved.files:
            raise ValueError(f"cache metadata missing: {path}")
        metadata = json.loads(saved[METADATA_KEY].tobytes().decode("utf-8"))
        arrays = {
            name: np.ascontiguousarray(saved[name]) for name in saved.files if name != METADATA_KEY
        }
    read_seconds = time.perf_counter() - started
    if not isinstance(metadata, dict):
        raise TypeError(f"cache metadata must be an object: {path}")
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                {
                    "cache_metadata_mismatch": key,
                    "expected": expected,
                    "actual": metadata.get(key),
                    "path": str(path),
                }
            )
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported cache schema: {metadata.get('schema_version')}")
    actual_schema = array_schema(arrays)
    actual_content_sha = array_content_sha256(arrays)
    if metadata.get("array_schema") != actual_schema:
        raise ValueError(f"cache array schema mismatch: {path}")
    if metadata.get("array_content_sha256") != actual_content_sha:
        raise ValueError(f"cache array content mismatch: {path}")
    return arrays, {
        "read_seconds": float(read_seconds),
        "schema_sha256": json_sha256(actual_schema),
        "content_sha256": actual_content_sha,
    }


# Extracted from graph_inference.py
# source SHA256 168c8c936178d58910074f31a09e604784aef6f236322fcd4b233de850e6be54

EXPECTED_PUBLIC_PRIMARY_CHECKPOINT_SHA256 = (
    "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
)

REQUIRED_REPLAY_ARRAYS = (
    "candidate_ids_src",
    "candidate_ids_tgt",
    "coords_src_grid",
    "coords_tgt_grid",
    "detection_scores_src",
    "detection_scores_tgt",
    "position_features_src",
    "position_features_tgt",
    "candidate_mask_src",
    "candidate_mask_tgt",
    "primary_features_src",
    "primary_features_tgt",
    "secondary_features_src",
    "secondary_features_tgt",
)


def _extract_tracker_state(full_state: dict[str, Any]) -> dict[str, Any]:
    nested_state = full_state.get("state_dict")
    if isinstance(nested_state, dict):
        return _extract_tracker_state(nested_state)
    for prefix in ("transformer.", "module.transformer."):
        selected = {
            key[len(prefix) :]: value for key, value in full_state.items() if key.startswith(prefix)
        }
        if selected:
            return selected
    tracker_roots = {"proj", "norm_in", "blocks", "norm_out", "pair_mlp"}
    observed_roots = {key.split(".", 1)[0] for key in full_state}
    if tracker_roots.issubset(observed_roots):
        return dict(full_state)
    raise ValueError("checkpoint does not contain a SimpleNodeTransformer state")


def _load_tracker(checkpoint_path: Path, device: Any, model_params: dict[str, Any]) -> Any:
    import torch
    from biohub_tracking.models import SimpleNodeTransformer

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"tracker checkpoint is not a state dictionary: {checkpoint_path}")
    tracker_state = _extract_tracker_state(state)
    tracker = SimpleNodeTransformer(
        feat_dim=int(model_params["feature_dim"]),
        hidden_dim=int(model_params["hidden_dim"]),
        n_heads=int(model_params["n_heads"]),
        n_blocks=int(model_params["n_blocks"]),
        dropout=float(model_params["dropout"]),
        pair_chunk_size=int(model_params["pair_chunk_size"]),
    )
    tracker.load_state_dict(tracker_state, strict=True)
    tracker.to(device)
    tracker.eval()
    return tracker


def _read_cache_window(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    source_frame, target_frame = map(int, path.stem.split("_"))
    arrays, receipt = read_window_cache(
        path,
        expected_metadata={
            "experiment": "exp015_oracle_stage_limits",
            "dataset": path.parent.name,
            "window_frames": [source_frame, target_frame],
            "primary_checkpoint_sha256": EXPECTED_PUBLIC_PRIMARY_CHECKPOINT_SHA256,
        },
    )
    missing = sorted(set(REQUIRED_REPLAY_ARRAYS) - set(arrays))
    if missing:
        raise ValueError({"cache_arrays_missing": missing, "path": str(path)})
    for side in ("src", "tgt"):
        candidate_ids = np.asarray(arrays[f"candidate_ids_{side}"], dtype=np.int64)
        coords = np.asarray(arrays[f"coords_{side}_grid"])
        mask = np.asarray(arrays[f"candidate_mask_{side}"], dtype=bool)
        if candidate_ids.ndim != 1 or coords.shape != (len(candidate_ids), 3):
            raise ValueError(f"invalid cached candidates: {path} {side}")
        if mask.shape != (len(candidate_ids),) or not mask.all():
            raise ValueError(f"cached candidates contain padding: {path} {side}")
        for name in (
            f"position_features_{side}",
            f"primary_features_{side}",
            f"secondary_features_{side}",
        ):
            if np.asarray(arrays[name]).shape != (len(candidate_ids), 32):
                raise ValueError(f"invalid cached feature shape: {path} {name}")
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f"non-finite cached feature: {path} {name}")
    return arrays, receipt


def _cached_tracker_logits(
    tracker: Any,
    arrays: dict[str, Any],
    *,
    feature_prefix: str,
    reverse: bool,
    device: Any,
    downsample_zyx: tuple[float, float, float],
) -> Any:
    import numpy as np
    import torch

    source_side, target_side = ("tgt", "src") if reverse else ("src", "tgt")

    def tensor(name: str, *, dtype: Any | None = None) -> Any:
        value = np.asarray(arrays[name])
        result = torch.from_numpy(value).unsqueeze(0).to(device)
        return result.to(dtype=dtype) if dtype is not None else result

    features_source = torch.cat(
        [
            tensor(f"{feature_prefix}_features_{source_side}", dtype=torch.float32),
            tensor(f"position_features_{source_side}", dtype=torch.float32),
        ],
        dim=-1,
    )
    features_target = torch.cat(
        [
            tensor(f"{feature_prefix}_features_{target_side}", dtype=torch.float32),
            tensor(f"position_features_{target_side}", dtype=torch.float32),
        ],
        dim=-1,
    )
    scale = torch.as_tensor(downsample_zyx, dtype=torch.float32, device=device)
    coords_source = tensor(f"coords_{source_side}_grid", dtype=torch.float32) * scale
    coords_target = tensor(f"coords_{target_side}_grid", dtype=torch.float32) * scale
    mask_source = tensor(f"candidate_mask_{source_side}", dtype=torch.bool)
    mask_target = tensor(f"candidate_mask_{target_side}", dtype=torch.bool)
    with torch.no_grad():
        return tracker(
            features_source,
            features_target,
            coords_source,
            coords_target,
            mask_source,
            mask_target,
        )


def fuse_cached_edge_logits(
    primary_tracker: Any,
    secondary_tracker: Any,
    arrays: dict[str, Any],
    *,
    secondary_logits: Any | None = None,
    device: Any,
    downsample_zyx: tuple[float, float, float],
    bidirectional_weight: float,
    secondary_edge_weight: float,
    secondary_low_margin_max: float,
    secondary_mix_temperature: float,
) -> Any:
    import torch

    edge_logits = _cached_tracker_logits(
        primary_tracker,
        arrays,
        feature_prefix="primary",
        reverse=False,
        device=device,
        downsample_zyx=downsample_zyx,
    )
    if bidirectional_weight > 0.0:
        reverse_native = _cached_tracker_logits(
            primary_tracker,
            arrays,
            feature_prefix="primary",
            reverse=True,
            device=device,
            downsample_zyx=downsample_zyx,
        )
        reverse_logits = reverse_native.transpose(1, 2)
        forward_center = edge_logits.mean(dim=1, keepdim=True)
        forward_scale = edge_logits.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        reverse_center = reverse_logits.mean(dim=1, keepdim=True)
        reverse_scale = (
            reverse_logits.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        )
        reverse_ratio = (forward_scale / reverse_scale).clamp(0.5, 2.0).to(reverse_logits.dtype)
        reverse_aligned = (reverse_logits - reverse_center) * reverse_ratio + forward_center
        forward_prob = torch.softmax(edge_logits.float(), dim=1).clamp_min(1e-8)
        reverse_prob = torch.softmax(reverse_aligned.float(), dim=1).clamp_min(1e-8)
        harmonic_prob = 1.0 / (
            (1.0 - bidirectional_weight) / forward_prob + bidirectional_weight / reverse_prob
        )
        harmonic_prob = harmonic_prob / harmonic_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
        harmonic_logits = torch.log(harmonic_prob.clamp_min(1e-8))
        harmonic_center = harmonic_logits.mean(dim=1, keepdim=True)
        harmonic_scale = harmonic_logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        harmonic_ratio = (forward_scale / harmonic_scale).clamp(0.5, 2.0)
        edge_logits = ((harmonic_logits - harmonic_center) * harmonic_ratio + forward_center).to(
            reverse_aligned.dtype
        )

    if secondary_logits is None:
        secondary_logits = _cached_tracker_logits(
            secondary_tracker,
            arrays,
            feature_prefix="secondary",
            reverse=False,
            device=device,
            downsample_zyx=downsample_zyx,
        )
    primary_center = edge_logits.mean(dim=1, keepdim=True)
    primary_scale = edge_logits.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    secondary_center = secondary_logits.mean(dim=1, keepdim=True)
    secondary_scale = (
        secondary_logits.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    )
    secondary_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)
    secondary_aligned = (secondary_logits - secondary_center) * secondary_ratio + primary_center
    n_source = int(edge_logits.shape[1])
    if n_source >= 2:
        primary_probs = torch.softmax(edge_logits[0], dim=0)
        secondary_probs = torch.softmax(secondary_aligned[0], dim=0)
        primary_top2 = torch.topk(primary_probs, k=2, dim=0)
        secondary_top2 = torch.topk(secondary_probs, k=2, dim=0)
        primary_margin = primary_top2.values[0] - primary_top2.values[1]
        same_parent = primary_top2.indices[0].eq(secondary_top2.indices[0])
        uncertainty = (
            (secondary_low_margin_max - primary_margin) / secondary_low_margin_max
        ).clamp(0.0, 1.0)
        local_weight = secondary_edge_weight * uncertainty
        local_weight = torch.where(same_parent, local_weight, torch.zeros_like(local_weight))
        blend_weight: Any = local_weight.view(1, 1, -1)
    else:
        blend_weight = 0.0
    edge_logits = (1.0 - blend_weight) * edge_logits + blend_weight * secondary_aligned
    if secondary_mix_temperature != 1.0:
        mixed_center = edge_logits.mean(dim=1, keepdim=True)
        edge_logits = mixed_center + (edge_logits - mixed_center) / secondary_mix_temperature
    return edge_logits


def select_cached_candidate_edges(
    edge_logits: Any,
    arrays: dict[str, Any],
    *,
    threshold: float,
) -> list[tuple[int, int, float, float]]:
    import numpy as np
    import torch

    probabilities = torch.softmax(edge_logits[0], dim=0).detach().cpu().numpy()
    source_ids = np.asarray(arrays["candidate_ids_src"], dtype=np.int64)
    target_ids = np.asarray(arrays["candidate_ids_tgt"], dtype=np.int64)
    source_coords = np.asarray(arrays["coords_src_grid"], dtype=np.float32)
    target_coords = np.asarray(arrays["coords_tgt_grid"], dtype=np.float32)
    ranked = sorted(
        [
            (probabilities[i, j], i, j)
            for i in range(len(source_ids))
            for j in range(len(target_ids))
            if probabilities[i, j] > threshold
        ],
        reverse=True,
    )
    return [
        (
            int(source_ids[i]),
            int(target_ids[j]),
            float(probability),
            float(np.linalg.norm(source_coords[i] - target_coords[j])),
        )
        for probability, i, j in ranked
    ]


def _register_cached_frame(
    registry: dict[int, tuple[int, int, int, int]],
    *,
    frame: int,
    candidate_ids: Any,
    coords_grid: Any,
    downsample_zyx: tuple[int, int, int],
) -> None:
    import numpy as np

    ids = np.asarray(candidate_ids, dtype=np.int64)
    grid = np.asarray(coords_grid, dtype=np.float32)
    scaled = (grid * np.asarray(downsample_zyx, dtype=np.float32)).astype(np.int16)
    for node_id, zyx in zip(ids, scaled, strict=True):
        record = (int(frame), int(zyx[0]), int(zyx[1]), int(zyx[2]))
        previous = registry.setdefault(int(node_id), record)
        if previous != record:
            raise ValueError({"candidate_id_changed_across_windows": int(node_id)})


def _coords_from_registry(registry: dict[int, tuple[int, int, int, int]]) -> Any:
    import numpy as np

    ids = sorted(registry)
    if ids != list(range(len(ids))):
        raise ValueError("cached candidate IDs are not contiguous from zero")
    return np.asarray([registry[index] for index in ids], dtype=np.int16).reshape((-1, 4))


# Extracted from exp018_graph_cost_scale_diagnostic.py
# source SHA256 1c1975fa2340913daeb1d7d7bce85e82fa0534fe60974fee650fc7a304cc7093


def is_kaggle_runtime() -> bool:
    return Path("/kaggle/input").is_dir() and Path("/kaggle/working").is_dir()


OFFLINE_GRAPH_MODULES = {
    "tracksdata": "tracksdata",
    "zarr": "zarr",
    "pyscipopt": "pyscipopt",
    "geff": "geff",
    "geff_spec": "geff_spec",
    "ilpy": "ilpy",
    "polars": "polars",
    "imagecodecs": "imagecodecs",
    "rustworkx": "rustworkx",
    "numcodecs": "numcodecs",
    "donfig": "donfig",
    "bidict": "bidict",
}

OFFLINE_GRAPH_PACKAGE_SPECS = (
    "tracksdata",
    "bidict>=0.23.1",
    "psygnal>=0.14",
    "rich",
    "markdown-it-py",
    "pygments",
    "zarr>=3.0.10,<4",
    "donfig>=0.8",
    "google-crc32c>=1.5",
    "numcodecs>=0.13,<0.16",
    "deprecated",
    "msgpack",
    "wrapt",
    "geff>=1.1.3.1.1",
    "geff-spec<1.2",
    "networkx>=3.2.1",
    "pydantic>=2.11",
    "annotated-types",
    "pydantic-core",
    "typing-extensions>=4.13",
    "typing-inspection",
    "pyscipopt",
    "ilpy>=0.5.1",
    "imagecodecs",
    "rustworkx>=0.17.1",
)


def offline_wheel_dirs() -> list[Path]:
    candidates = [
        Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/wheels"),
        Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1/wheels"),
    ]
    return [path for path in candidates if path.is_dir() and any(path.glob("*.whl"))]


def run_offline_install(
    wheel_dirs: list[Path],
    specs: tuple[str, ...],
    *,
    force_reinstall: bool,
) -> None:
    command = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
    if force_reinstall:
        command.append("--force-reinstall")
    for wheel_dir in wheel_dirs:
        command.extend(["--find-links", str(wheel_dir)])
    command.extend(specs)
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        print((result.stdout or "")[-2000:])
        print((result.stderr or "")[-2000:])
        raise RuntimeError(f"offline graph package install failed: {result.returncode}")


def purge_graph_modules(*, include_polars: bool) -> None:
    roots = set(OFFLINE_GRAPH_MODULES.values())
    if not include_polars:
        roots.discard("polars")
    for module_name in list(sys.modules):
        if any(module_name == root or module_name.startswith(root + ".") for root in roots):
            sys.modules.pop(module_name, None)


def polars_runtime_ready() -> bool:
    try:
        import polars as pl
        from polars._plr import PySeries

        _ = PySeries
        return (
            hasattr(pl, "Float16") and pl.Series([-999999.0], dtype=pl.Float64).dtype == pl.Float64
        )
    except Exception:
        return False


def ensure_geff_runtime_dependencies() -> None:
    os.environ.setdefault("POLARS_PREFER_PKG", "32")
    failures: dict[str, str] = {}
    for package_name, module_name in OFFLINE_GRAPH_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as error:
            failures[package_name] = f"{type(error).__name__}: {error}"
    refresh_polars = not polars_runtime_ready()
    if not failures and not refresh_polars:
        print("Required tracksdata/GEFF packages import successfully.")
        return
    if not is_kaggle_runtime():
        raise ImportError(
            f"missing or incompatible local graph packages: {failures}; "
            f"refresh_polars={refresh_polars}"
        )

    wheel_dirs = offline_wheel_dirs()
    if not wheel_dirs:
        raise FileNotFoundError("offline support-pack wheel directory was not found")
    print("Offline wheel dirs:", [str(path) for path in wheel_dirs])
    if refresh_polars:
        run_offline_install(
            wheel_dirs,
            ("polars>=1.36", "polars-runtime-32"),
            force_reinstall=True,
        )
        purge_graph_modules(include_polars=True)
        importlib.invalidate_caches()
        if not polars_runtime_ready():
            raise ImportError("offline Polars refresh did not provide Float16 support")

    run_offline_install(
        wheel_dirs,
        OFFLINE_GRAPH_PACKAGE_SPECS,
        force_reinstall=False,
    )
    purge_graph_modules(include_polars=False)
    importlib.invalidate_caches()
    remaining: dict[str, str] = {}
    for package_name, module_name in OFFLINE_GRAPH_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as error:
            remaining[package_name] = f"{type(error).__name__}: {error}"
    if remaining:
        raise ImportError(f"offline graph package imports still fail: {remaining}")
    if not polars_runtime_ready():
        raise ImportError("Polars runtime lost Float16 support after graph package install")
    print("Offline graph package install succeeded.")


json_sha256 = json_sha


def cache_identity_record(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        if METADATA_KEY not in saved.files:
            raise ValueError(f"cache metadata missing: {path}")
        metadata = json.loads(saved[METADATA_KEY].tobytes().decode("utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError(f"cache metadata must be an object: {path}")
    schema = metadata.get("array_schema")
    if not isinstance(schema, list):
        raise ValueError(f"cache array schema missing: {path}")
    schema_by_name = {str(item.get("name")): item for item in schema}
    feature_names = (
        "primary_features_src",
        "primary_features_tgt",
        "secondary_features_src",
        "secondary_features_tgt",
    )
    if any(name not in schema_by_name for name in feature_names):
        raise ValueError(f"cache feature schema incomplete: {path}")
    frames = metadata.get("window_frames")
    if metadata.get("dataset") != path.parent.name or not isinstance(frames, list):
        raise ValueError(f"cache identity and path differ: {path}")
    return {
        "dataset": path.parent.name,
        "window_frames": [int(value) for value in frames],
        "candidate_count_src": int(schema_by_name["candidate_mask_src"]["shape"][0]),
        "candidate_count_tgt": int(schema_by_name["candidate_mask_tgt"]["shape"][0]),
        "feature_values": int(
            sum(np.prod(schema_by_name[name]["shape"], dtype=np.int64) for name in feature_names)
        ),
        "cache_bytes": int(path.stat().st_size),
        "cache_schema_sha256": json_sha256(schema),
        "cache_content_sha256": str(metadata.get("array_content_sha256")),
    }


def recompute_cache_identity_sha256(paths: list[Path]) -> str:
    return json_sha256([cache_identity_record(path) for path in sorted(paths)])


def artifact_paths(root: Path, pattern: str):
    """Search model/support artifacts without walking image or graph chunks."""
    for directory, names, files in os.walk(root):
        names[:] = sorted(
            name
            for name in names
            if name not in {"competitions", "window_cache"}
            and not name.endswith((".geff", ".zarr"))
        )
        for name in sorted(files):
            if fnmatch.fnmatchcase(name, pattern):
                yield Path(directory) / name


def resolve_pinned(root: Path, filename: str, expected_sha: str) -> Path:
    matches = [p for p in artifact_paths(root, filename) if file_sha(p) == expected_sha]
    if not matches:
        raise FileNotFoundError(f"no checksum-verified {filename} under {root}")
    return sorted(matches)[0]


def candidate_tree_sha(root: Path) -> str:
    return json_sha(
        [
            [p.relative_to(root).as_posix(), file_sha(p)]
            for p in sorted(root.rglob("*"))
            if p.is_file()
        ]
    )


def materialize_candidates(input_root: Path, output_root: Path, manifest: dict) -> dict[str, Path]:
    destination = output_root / "verified_candidates"
    destination.mkdir(parents=True, exist_ok=True)
    index: dict[str, list[Path]] = defaultdict(list)
    for path in artifact_paths(input_root, "batch_*"):
        index[path.name].append(path)
    candidates = {}
    layouts = []
    for record in manifest["archives"]:
        paths = index[record["name"]] + index[record["name"].replace(".zip", "_archive.bin")]
        valid = [
            p
            for p in paths
            if p.stat().st_size == record["bytes"] and file_sha(p) == record["sha256"]
        ]
        if valid:
            with zipfile.ZipFile(sorted(valid)[0]) as archive:
                members = []
                for info in archive.infolist():
                    relative = Path(info.filename)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("unsafe archive member")
                    if info.is_dir() or relative.parts[0] != "oracle_candidate_graphs":
                        continue
                    if len(relative.parts) < 3:
                        raise ValueError("invalid graph archive layout")
                    sample = relative.parts[1].removesuffix(".geff")
                    if sample not in record["samples"]:
                        raise ValueError("unexpected sample in archive")
                    target = destination.joinpath(*relative.parts[1:])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(info))
                    members.append(sample)
                if set(members) != set(record["samples"]):
                    raise ValueError("incomplete candidate archive")
            graph_root = destination
            layout = "sha_verified_archive"
        else:
            receipts = [
                p for p in index[record["receipt_name"]] if file_sha(p) == record["receipt_sha256"]
            ]
            expanded = [
                p.parent / Path(record["name"]).stem / "oracle_candidate_graphs" for p in receipts
            ]
            expanded = [p for p in expanded if p.is_dir()]
            if len(expanded) != 1:
                raise FileNotFoundError(
                    f"checksum-pinned archive or expanded graphs missing: {record['name']}"
                )
            graph_root = expanded[0]
            if sorted(p.stem for p in graph_root.glob("*.geff")) != sorted(record["samples"]):
                raise ValueError("expanded candidate sample coverage changed")
            layout = "sha_verified_expanded_graphs"
        for sample in record["samples"]:
            graph_path = graph_root / f"{sample}.geff"
            if candidate_tree_sha(graph_path) != record["candidate_tree_shas"][sample]:
                raise ValueError(f"candidate graph content changed: {sample}")
            if sample in candidates:
                raise ValueError("duplicate candidate sample")
            candidates[sample] = graph_path
        layouts.append({"archive": record["name"], "layout": layout})
    if len(candidates) != manifest["expected_sample_count"]:
        raise ValueError("missing candidate samples")
    write_json(output_root / "candidate_input_layouts.json", layouts)
    print(f"INPUT_VERIFIED candidate_graphs={len(candidates)}", flush=True)
    return candidates


def support_runtime(input_root: Path, source: dict, output_root: Path) -> tuple[Any, Path]:
    ensure_geff_runtime_dependencies()
    try:
        evaluator = resolve_pinned(input_root, "evaluate.py", source["evaluator_sha256"])
    except FileNotFoundError:
        evaluator = materialize_public_repository(
            input_root, output_root, source["evaluator_sha256"]
        )
    root = evaluator.parents[1]
    for key in ("metrics", "division_metrics"):
        relative = Path(source[f"{key}_path"]).relative_to("repo")
        require_sha(root / relative, source[f"{key}_sha256"])
    for name in list(sys.modules):
        if name == "biohub_tracking" or name.startswith("biohub_tracking."):
            del sys.modules[name]
    sys.path.insert(0, str(root / "src"))
    metric = importlib.import_module("biohub_tracking.metrics")
    return metric, root


def as_tracksdata(
    graph: Graph,
    edges: np.ndarray,
    scale: np.ndarray,
    *,
    rounding: bool,
    return_mapping: bool = False,
) -> Any:
    import polars as pl
    import tracksdata as td

    result = td.graph.InMemoryGraph()
    for axis in "zyx":
        result.add_node_attr_key(axis, pl.Float64, -999999.0)
    positions = graph.positions / scale
    if rounding:
        positions = np.maximum(0, np.rint(positions))
    order = np.argsort(graph.ids, kind="stable")
    new_ids = result.bulk_add_nodes(
        [
            {"t": int(graph.frames[i]), **dict(zip("zyx", map(float, positions[i]), strict=True))}
            for i in order
        ]
    )
    id_map = dict(zip(map(int, graph.ids[order]), map(int, new_ids), strict=True))
    if len(edges):
        result.bulk_add_edges(
            [{"source_id": id_map[int(a)], "target_id": id_map[int(b)]} for a, b in edges]
        )
    return (result, id_map) if return_mapping else result


def official_row(
    graph: Graph, edges: np.ndarray, truth_path: Path, metric: Any, validation: dict
) -> dict:
    import tracksdata as td
    from geff import GeffMetadata

    scale = np.asarray(validation["scale_zyx_um"])
    prediction = as_tracksdata(graph, edges, scale, rounding=True)
    truth = td.graph.IndexedRXGraph.from_geff(truth_path)
    truth = truth[0] if isinstance(truth, tuple) else truth
    result = metric.evaluate(
        prediction,
        truth,
        scale=tuple(scale),
        max_distance=float(validation["official_matching_um"]),
    )
    recall = (
        metric.node_recall(prediction, truth)
        if prediction.num_nodes() and prediction.num_edges()
        else 0.0
    )
    metadata = GeffMetadata.read(truth_path)
    total = (metadata.extra or {}).get("estimated_number_of_nodes")
    if total is None:
        raise ValueError(f"missing estimated node count: {truth_path}")
    row = dict(metric.per_sample_metrics(result, float(total), recall))
    return {str(k): float(v) if isinstance(v, (np.number, float)) else v for k, v in row.items()}


def summarize(rows: list[dict], metric: Any, expected: int) -> dict:
    result = dict(metric.summarise(rows))
    result["combined_score"] = result["score"]
    result["adjusted_edge_jaccard"] = result["adj_edge_jaccard"]
    if len(rows) != expected or int(result["n"]) != expected or int(result["n_adj"]) != expected:
        raise ValueError("official evaluator skipped a sample")
    result = {str(k): v.item() if isinstance(v, np.generic) else v for k, v in result.items()}
    for key in ("combined_score", "adjusted_edge_jaccard", "division_jaccard"):
        if not math.isfinite(float(result[key])):
            raise ValueError(f"non-finite official {key}")
    return result


def fold_tracker_checkpoint(train_root: Path, split: dict, cfg: dict) -> Path:
    manifest_path = require_sha(
        train_root / "model_manifest.json", cfg["data"]["model_manifest_sha256"]
    )
    models = json.loads(manifest_path.read_text())["models"]
    matches = [row for row in models if row["fold"] == split["fold"]]
    if len(matches) != 1:
        raise ValueError("missing fold model")
    row = matches[0]
    expected = cfg["model"]["fold_checkpoints"][str(split["fold"])]
    if (
        row["train_embryo"] != split["train_embryo"]
        or row["evaluation_embryo"] != split["evaluation_embryo"]
        or row["file_sha256"] != expected["sha256"]
        or row["path"] != expected["relative_path"]
    ):
        raise ValueError("wrong fold checkpoint or training provenance")
    return require_sha(train_root / row["path"], expected["sha256"])


def replay_internal_graph(
    sample: str,
    split: dict,
    cache_root: Path,
    primary: Any,
    secondary: Any,
    cfg: dict,
    deadline: float,
) -> tuple[Graph, Graph, dict]:
    import polars as pl
    import tracksdata as td

    if sample not in split["internal_validation"] or not sample.startswith(
        split["train_embryo"] + "_"
    ):
        raise ValueError("replay calibration sample is not training-side internal validation")
    paths = sorted((cache_root / sample).glob("*.npz"))
    if len(paths) != cfg["data"]["expected_windows_per_sample"]:
        raise ValueError("incomplete cache sequence")
    registry, candidate_edges, window_evidence = {}, [], []
    replay = cfg["model"]["replay"]
    downsample = tuple(replay["downsample_zyx"])
    for frame, path in enumerate(paths):
        check_deadline(deadline)
        if tuple(map(int, path.stem.split("_"))) != (frame, frame + 1):
            raise ValueError("unordered cache windows")
        arrays, receipt = _read_cache_window(path)
        window_evidence.append([path.name, receipt["content_sha256"]])
        for side, t in (("src", frame), ("tgt", frame + 1)):
            _register_cached_frame(
                registry,
                frame=t,
                candidate_ids=arrays[f"candidate_ids_{side}"],
                coords_grid=arrays[f"coords_{side}_grid"],
                downsample_zyx=downsample,
            )
        logits = fuse_cached_edge_logits(
            primary,
            secondary,
            arrays,
            device=cfg["runtime"]["device"],
            downsample_zyx=downsample,
            bidirectional_weight=replay["bidirectional_weight"],
            secondary_edge_weight=replay["secondary_edge_weight"],
            secondary_low_margin_max=replay["secondary_low_margin_max"],
            secondary_mix_temperature=replay["secondary_mix_temperature"],
        )
        candidate_edges.extend(
            select_cached_candidate_edges(logits, arrays, threshold=replay["edge_threshold"])
        )
    coords = _coords_from_registry(registry)
    scale = np.asarray(cfg["validation"]["scale_zyx_um"])
    graph = Graph(
        np.arange(len(coords)),
        coords[:, 0],
        coords[:, 1:] * scale,
        np.asarray([(a, b) for a, b, _, _ in candidate_edges], dtype=np.int64).reshape(-1, 2),
        np.asarray([p for _, _, p, _ in candidate_edges]),
    )
    graph.validate()
    # Internal reservations use this fold's existing ILP costs, without repair.
    baseline, id_map = as_tracksdata(
        graph, np.empty((0, 2), dtype=np.int64), scale, rounding=False, return_mapping=True
    )
    reverse_ids = {new: old for old, new in id_map.items()}
    baseline.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
    if candidate_edges:
        baseline.bulk_add_edges(
            [
                {"source_id": id_map[a], "target_id": id_map[b], "edge_prob": p}
                for a, b, p, _ in candidate_edges
            ]
        )
        baseline = (
            td.solvers.ILPSolver(
                edge_weight=float(replay["ilp_edge_weight"]) * td.EdgeAttr("edge_prob"),
                appearance_weight=float(replay["ilp_appearance_weight"]),
                disappearance_weight=float(replay["ilp_disappearance_weight"]),
                division_weight=float(replay["ilp_division_weight"]),
            )
            .solve(baseline)
            .detach()
        )
    edges = np.asarray(
        [
            (reverse_ids[int(r["source_id"])], reverse_ids[int(r["target_id"])])
            for r in baseline.edge_attrs().iter_rows(named=True)
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    reference = Graph(graph.ids, graph.frames, graph.positions, edges, np.ones(len(edges)))
    reference.validate(scored=False)
    return (
        graph,
        reference,
        {
            "sample": sample,
            "fold": split["fold"],
            "train_embryo": split["train_embryo"],
            "window_content_sha256": json_sha(window_evidence),
            "candidate_sha256": graph.digest(),
            "window_count": len(paths),
            "reservation_source": "same_fold_internal_ilp_prediction",
        },
    )


def save_prediction(path: Path, graph: Graph, output: dict, scale: np.ndarray) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    graph_path = path.with_suffix(".npz")
    state_path = path.with_name(path.name + "_states.npz")
    np.savez_compressed(
        graph_path,
        node_ids=graph.ids,
        node_tzyx=np.column_stack([graph.frames, graph.positions / scale]),
        edges=output["edges"],
    )
    np.savez_compressed(
        state_path,
        node_ids=graph.ids,
        means=output["means"],
        covariances=output["covariances"],
        track_ids=output["track_ids"],
        histories=output["histories"],
        association_diagnostics=output["association_diagnostics"],
    )
    return {
        "graph": str(graph_path),
        "graph_sha256": file_sha(graph_path),
        "states": str(state_path),
        "states_sha256": file_sha(state_path),
        "input_sha256": graph.digest(),
        "edge_content_sha256": json_sha(output["edges"].tolist()),
        "track_count": output["track_count"],
        "no_match_sources": output["no_match_sources"],
        "no_match_targets": output["no_match_targets"],
    }


def projection_gate(
    elapsed: float, replay_seconds: float, decode_seconds: float, cfg: dict
) -> dict:
    internal = sum(len(s["internal_validation"]) for s in cfg["splits"])
    outer = sum(len(s["outer_evaluation"]) for s in cfg["splits"])
    p = cfg["model"]["params"]
    trials = len(p["no_match_cost_grid"]) + len(p["motion_weight_grid"])
    runtime = cfg["runtime"]
    projected = (
        elapsed
        + runtime["projection_multiplier"]
        * (internal * replay_seconds + (internal * trials + outer * 3) * decode_seconds)
        + runtime["evaluation_reserve_seconds"]
    )
    result = {
        "elapsed_seconds": elapsed,
        "projected_seconds": projected,
        "limit_seconds": runtime["time_limit_seconds"],
        "passed": projected < runtime["time_limit_seconds"],
    }
    if not result["passed"]:
        raise TimeoutError(f"runtime projection gate failed: {result}")
    return result


def calibrate_fold(
    split: dict,
    cfg: dict,
    candidate_paths: dict[str, Path],
    truth_paths: dict[str, Path],
    cache_root: Path,
    primary: Any,
    secondary: Any,
    metric: Any,
    output_root: Path,
    started: float,
    deadline: float,
) -> tuple[dict, dict]:
    scale = np.asarray(cfg["validation"]["scale_zyx_um"])
    params = {**cfg["model"]["params"], **cfg["validation"]}
    records = [
        (
            sample,
            load_geff(candidate_paths[sample], scale),
            load_geff(truth_paths[sample], scale, scored=False),
        )
        for sample in sorted(split["gradient_update"])
    ]
    noise = fit_noise(records, split["train_embryo"], params)
    del records
    fold_root = output_root / f"fold_{split['fold']}"
    write_json(fold_root / "noise.json", noise)
    internal = {}
    replay_receipts = []
    max_replay, max_decode = 0.0, 0.0
    for sample in sorted(split["internal_validation"]):
        check_deadline(deadline)
        before = time.monotonic()
        candidate, reference, receipt = replay_internal_graph(
            sample, split, cache_root, primary, secondary, cfg, deadline
        )
        max_replay = max(max_replay, time.monotonic() - before)
        graph, reserved, reservation_receipt = reserve_divisions(candidate, reference)
        internal[sample] = (graph, reserved)
        replay_receipts.append({**receipt, **reservation_receipt})
        before = time.monotonic()
        smoke = decode(
            graph,
            reserved,
            noise,
            motion_weight=0.0,
            no_match_cost=params["no_match_cost_grid"][0],
            cfg=params,
        )
        official_row(graph, smoke["edges"], truth_paths[sample], metric, cfg["validation"])
        max_decode = max(max_decode, time.monotonic() - before)
        gate = projection_gate(time.monotonic() - started, max_replay, max_decode, cfg)
        write_json(fold_root / "runtime_projection.json", gate)
        print(
            f"INTERNAL_INPUT fold={split['fold']} sample={sample} "
            f"projected={gate['projected_seconds']:.1f}",
            flush=True,
        )
    write_json(fold_root / "internal_replay_receipts.json", replay_receipts)

    def trial(weight: float, null: float) -> dict:
        rows = []
        for sample, (graph, reserved) in internal.items():
            check_deadline(deadline)
            prediction = decode(
                graph, reserved, noise, motion_weight=weight, no_match_cost=null, cfg=params
            )
            rows.append(
                official_row(
                    graph, prediction["edges"], truth_paths[sample], metric, cfg["validation"]
                )
            )
        summary = summarize(rows, metric, len(internal))
        return {"motion_weight": weight, "no_match_cost": null, **summary}

    rows = [trial(0.0, float(null)) for null in params["no_match_cost_grid"]]
    best_null = max(rows, key=lambda row: row["combined_score"])["no_match_cost"]
    rows.extend(trial(float(weight), best_null) for weight in params["motion_weight_grid"])
    selected = choose_costs(rows, params)
    receipt = {
        "fold": split["fold"],
        "train_embryo": split["train_embryo"],
        "evaluation_embryo": split["evaluation_embryo"],
        "fit_samples": sorted(split["gradient_update"]),
        "selection_samples": sorted(split["internal_validation"]),
        "noise_sha256": json_sha(noise),
        "selected": selected,
        "trials": json_finite(rows),
        "internal_reservations": "same_fold_ilp_before_repair",
        "outer_reservations": "saved_exp016_after_repair",
    }
    write_json(fold_root / "calibration.json", receipt)
    return noise, selected


def promotion_decision(summaries: dict, embryos: list[str]) -> dict:
    decisions = {}
    for embryo in embryos:
        baseline = summaries["image_only"][embryo]
        motion = summaries["kalman"][embryo]
        delta = float(motion["combined_score"]) - float(baseline["combined_score"])
        division_delta = float(motion["division_jaccard"]) - float(baseline["division_jaccard"])
        decisions[embryo] = {
            "combined_score_delta": delta,
            "division_jaccard_delta": division_delta,
            "condition_met": delta > 0 and division_delta >= 0,
        }
    return {
        "by_embryo": decisions,
        "conditions_met": all(r["condition_met"] for r in decisions.values()),
        "adoption_decision": "pending_user",
    }


def json_finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(v) for v in value]
    if isinstance(value, np.generic):
        return json_finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def resolve_truth_paths(input_root: Path, samples: list[str], competition: str) -> dict[str, Path]:
    roots = [
        input_root / "competitions" / competition / "train",
        input_root / competition / "train",
    ]
    matches = [
        root
        for root in roots
        if root.is_dir() and all((root / f"{sample}.geff").is_dir() for sample in samples)
    ]
    if len(matches) != 1:
        raise ValueError("expected one complete competition train/*.geff directory")
    return {sample: matches[0] / f"{sample}.geff" for sample in samples}


def materialize_public_repository(input_root: Path, output_root: Path, expected_sha: str) -> Path:
    for archive_path in sorted(artifact_paths(input_root, "repo.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            targets = [name for name in archive.namelist() if name.endswith("scripts/evaluate.py")]
            for name in targets:
                if hashlib.sha256(archive.read(name)).hexdigest() != expected_sha:
                    continue
                prefix = Path(name).parents[1]
                destination = output_root / "support_repo"
                for info in archive.infolist():
                    relative = Path(info.filename)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("unsafe public repository archive")
                    if info.is_dir() or not relative.is_relative_to(prefix):
                        continue
                    target = destination / relative.relative_to(prefix)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(info))
                return require_sha(destination / "scripts/evaluate.py", expected_sha)
    raise FileNotFoundError("no checksum-verified support repository or repo.zip")


def resolve_checkpoint(input_root: Path, output_root: Path, expected_sha: str) -> Path:
    try:
        return resolve_pinned(input_root, "edge_predictor_best.pth", expected_sha)
    except FileNotFoundError:
        for archive_path in sorted(artifact_paths(input_root, "weights.zip")):
            with zipfile.ZipFile(archive_path) as archive:
                for name in archive.namelist():
                    if not name.endswith("edge_predictor_best.pth"):
                        continue
                    digest = hashlib.sha256()
                    with archive.open(name) as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                    if digest.hexdigest() != expected_sha:
                        continue
                    destination = output_root / "secondary_tracker.pth"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(name) as handle, destination.open("wb") as target:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            target.write(block)
                    return require_sha(destination, expected_sha)
    raise FileNotFoundError("no checksum-verified secondary checkpoint")


# %% [markdown]
# ## 8. Setup, calibration, outer evaluation and artifacts
#
# The first full run and official scoring are restricted to Kaggle. Calibration
# uses only each direction's training-side labels. There is no submission step.
# All acceptance decisions remain pending user review.


# %%
def require_kaggle_execution() -> None:
    if not is_kaggle_runtime():
        raise RuntimeError("full diagnostic and official evaluation must run on Kaggle")


# %% [markdown]
# ### Setup and verify all fixed inputs

# %%
if __name__ == "__main__":
    require_kaggle_execution()
    config_path = Path.cwd() / "config.yaml"
    output_root = Path("/kaggle/working/diagnostic")
    input_root = Path("/kaggle/input")
    print(config_path.read_text())
    if not is_kaggle_runtime():
        raise RuntimeError("full diagnostic and official evaluation must run on Kaggle")
    import torch

    started = time.monotonic()
    cfg = yaml.safe_load(config_path.read_text())
    torch.set_num_threads(int(cfg["runtime"]["torch_threads"]))
    deadline = started + float(cfg["runtime"]["time_limit_seconds"])
    validation, data = cfg["validation"], cfg["data"]
    params = {**cfg["model"]["params"], **validation}
    scale = np.asarray(validation["scale_zyx_um"])
    split_path = require_sha(
        config_path.parent / validation["split_file"], validation["split_sha256"]
    )
    splits = json.loads(split_path.read_text())
    cfg["splits"] = splits
    manifest_path = require_sha(config_path.parent / data["manifest_file"], data["manifest_sha256"])
    manifest = json.loads(manifest_path.read_text())
    metric, repo = support_runtime(input_root, cfg["official_metric_source"], output_root)
    require_sha(
        repo / "src/biohub_tracking/models/simple_node_transformer.py",
        cfg["model"]["tracker_source_sha256"],
    )
    print("INPUT_VERIFIED support_repository", flush=True)
    candidate_paths = materialize_candidates(input_root, output_root, manifest)
    samples = sorted(candidate_paths)
    if dict(Counter(s.split("_", 1)[0] for s in samples)) != validation["expected_embryo_counts"]:
        raise ValueError("embryo coverage changed")
    for split in splits:
        validate_split(split, samples)
    if len(splits) != 2 or {s["fold"] for s in splits} != {0, 1}:
        raise ValueError("expected exactly the two parent folds")
    cache_matches = []
    for path in artifact_paths(input_root, "window_cache_summary.json"):
        value = json.loads(path.read_text())
        unsigned = {k: v for k, v in value.items() if k != "summary_sha256"}
        if value.get("summary_sha256") == data["cache_summary_sha256"] == json_sha(unsigned):
            cache_matches.append(path)
    if not cache_matches:
        raise ValueError("no valid cache summary payload")
    cache_summary_path = sorted(cache_matches)[0]
    cache_summary = json.loads(cache_summary_path.read_text())
    if (
        cache_summary.get("cache_identity_sha256", cache_summary.get("identity_sha256"))
        != data["cache_identity_sha256"]
    ):
        raise ValueError("cache identity mismatch")
    cache_root = cache_summary_path.parent / "window_cache"
    cache_paths = sorted(cache_root.glob("*/*.npz"))
    if (
        len(cache_paths) != len(samples) * data["expected_windows_per_sample"]
        or {p.parent.name for p in cache_paths} != set(samples)
        or recompute_cache_identity_sha256(cache_paths) != data["cache_identity_sha256"]
    ):
        raise ValueError("cache coverage or content identity mismatch")
    print(f"INPUT_VERIFIED cache_windows={len(cache_paths)}", flush=True)
    train_root = resolve_pinned(
        input_root, "model_manifest.json", data["model_manifest_sha256"]
    ).parent
    reference_root = (
        resolve_pinned(
            input_root, "repaired_graph_manifest.json", data["reference_manifest_sha256"]
        ).parent
        / "oracle_final_graphs"
    )
    reference_paths = {}
    for name, sha in manifest["reference_graphs"]:
        reference_paths[Path(name).stem] = require_sha(reference_root / name, sha)
    if sorted(reference_paths) != samples:
        raise ValueError("reference sample coverage changed")
    truth_paths = resolve_truth_paths(input_root, samples, data["competition_slug"])
    secondary_path = resolve_checkpoint(input_root, output_root, cfg["model"]["secondary_sha256"])
    secondary = _load_tracker(
        secondary_path, cfg["runtime"]["device"], cfg["model"]["tracker_params"]
    )
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT,
                "samples": len(samples),
                "additional_training": 0,
                "variants": ["image_only", "kalman"],
                "device": cfg["runtime"]["device"],
            }
        ),
        flush=True,
    )

# %% [markdown]
# ### Fit training-side noise and select association costs
# Each fold recalculates internal-validation scores using its own checkpoint.

# %%
if __name__ == "__main__":
    calibrated = {}
    for split in splits:
        checkpoint = fold_tracker_checkpoint(train_root, split, cfg)
        primary = _load_tracker(
            checkpoint, cfg["runtime"]["device"], cfg["model"]["tracker_params"]
        )
        calibrated[split["fold"]] = calibrate_fold(
            split,
            cfg,
            candidate_paths,
            truth_paths,
            cache_root,
            primary,
            secondary,
            metric,
            output_root,
            started,
            deadline,
        )
        del primary
    del secondary

# %% [markdown]
# ### Evaluate both locked variants and the unchanged exp016 reference
# Every sample uses the frozen costs selected from the opposite embryo.

# %%
if __name__ == "__main__":
    rows = {variant: {} for variant in ("image_only", "kalman", "exp016_reference")}
    predictions, diagnostics, reservations_evidence = [], [], []
    for split in splits:
        noise, costs = calibrated[split["fold"]]
        for sample in sorted(split["outer_evaluation"]):
            check_deadline(deadline)
            candidate = load_geff(candidate_paths[sample], scale)
            reference = load_compact(reference_paths[sample], scale)
            graph, reserved, receipt = reserve_divisions(candidate, reference)
            reservations_evidence.append({"sample": sample, **receipt})
            truth = load_geff(truth_paths[sample], scale, scored=False)
            for variant, weight in (("image_only", 0.0), ("kalman", costs["motion_weight"])):
                before_digest = graph.digest()
                prediction = decode(
                    graph,
                    reserved,
                    noise,
                    motion_weight=weight,
                    no_match_cost=costs["no_match_cost"],
                    cfg=params,
                )
                if graph.digest() != before_digest:
                    raise ValueError("decoder mutated fixed input")
                rows[variant][sample] = official_row(
                    graph, prediction["edges"], truth_paths[sample], metric, validation
                )
                output_receipt = save_prediction(
                    output_root / "graphs" / variant / sample, graph, prediction, scale
                )
                predictions.append({"sample": sample, "variant": variant, **output_receipt})
                diagnostics.extend(
                    {"sample": sample, "variant": variant, **r}
                    for r in diagnostic_counts(graph, truth, prediction, params)
                )
            rows["exp016_reference"][sample] = official_row(
                reference, reference.edges, truth_paths[sample], metric, validation
            )
            print(
                f"OUTER fold={split['fold']} sample={sample} "
                f"elapsed={time.monotonic() - started:.1f}",
                flush=True,
            )
            write_json(output_root / "partial_official_rows.json", json_finite(rows))
    if (
        sum(len(r["reserved_only_node_ids"]) for r in reservations_evidence)
        != data["expected_reserved_only_nodes"]
    ):
        raise ValueError("reserved-only node count changed")
    if (
        sum(r["reserved_divisions"] for r in reservations_evidence)
        != data["expected_reserved_divisions"]
    ):
        raise ValueError("reserved division count changed")
# %% [markdown]
# ### Aggregate official metrics and preserve evidence
# Validate the unchanged reference against its saved official result.
# Success conditions do not constitute an adoption decision.

# %%
if __name__ == "__main__":
    summaries = {}
    for variant, sample_rows in rows.items():
        if sorted(sample_rows) != samples:
            raise ValueError("incomplete outer evaluation")
        summaries[variant] = {"all": summarize(list(sample_rows.values()), metric, len(samples))}
        for embryo, count in validation["expected_embryo_counts"].items():
            embryo_rows = [r for s, r in sample_rows.items() if s.startswith(embryo + "_")]
            summaries[variant][embryo] = summarize(embryo_rows, metric, count)
    parent_path = require_sha(
        config_path.parent / data["parent_evaluation_file"], data["parent_evaluation_sha256"]
    )
    parent_metrics = json.loads(parent_path.read_text())["retrained"]
    expected_reference = {"all": parent_metrics["overall"], **parent_metrics["by_embryo"]}
    for group, expected_metrics in expected_reference.items():
        for key in ("score", "adj_edge_jaccard", "division_jaccard", "edge_jaccard"):
            if not math.isclose(
                summaries["exp016_reference"][group][key],
                expected_metrics[key],
                rel_tol=0,
                abs_tol=validation["reference_metric_tolerance"],
            ):
                raise ValueError(f"saved parent official metric mismatch: {group}/{key}")
    decision = promotion_decision(summaries, sorted(validation["expected_embryo_counts"]))
    write_json(output_root / "official_metrics.json", json_finite(summaries))
    write_json(output_root / "diagnostics.json", diagnostics)
    write_json(output_root / "predictions.json", predictions)
    write_json(output_root / "reservations.json", reservations_evidence)
    packages = {
        p: importlib.metadata.version(p)
        for p in ("numpy", "scipy", "torch", "tracksdata", "geff", "zarr")
    }
    receipt = {
        "experiment": EXPERIMENT,
        "execution_environment": "kaggle",
        "config_sha256": file_sha(config_path),
        "parent_model_manifest_sha256": data["model_manifest_sha256"],
        "fold_model_sha256": {
            fold: row["sha256"] for fold, row in cfg["model"]["fold_checkpoints"].items()
        },
        "official_metric_source": cfg["official_metric_source"],
        "input_manifest_sha256": file_sha(manifest_path),
        "split_sha256": file_sha(split_path),
        "sample_count": len(samples),
        "notebook_runtime_seconds": time.monotonic() - started,
        "packages": packages,
        "official_metrics_sha256": file_sha(output_root / "official_metrics.json"),
        "oof_prediction_sha": json_sha(predictions),
        "decision": decision,
        "additional_training": 0,
        "image_encoder_forward_count": 0,
        "submission_created": False,
    }
    write_json(output_root / "execution_receipt.json", receipt)
    metrics_path = config_path.parent / "metrics.json"
    metrics = (
        json.loads(metrics_path.read_text())
        if metrics_path.exists()
        else {"experiment": EXPERIMENT}
    )
    if metrics.get("status") not in {"usable", "completed", "deprecated", "discarded", "leak-risk"}:
        metrics["status"] = "debug_completed"
    metrics["cv"] = summaries["kalman"]["all"]["combined_score"]
    metrics.setdefault("evidence", {})["kalman_graph_diagnostic"] = receipt
    metrics["official_graph_evaluation"] = json_finite(summaries)
    write_json(metrics_path, metrics)
    print(json.dumps(receipt, indent=2))
