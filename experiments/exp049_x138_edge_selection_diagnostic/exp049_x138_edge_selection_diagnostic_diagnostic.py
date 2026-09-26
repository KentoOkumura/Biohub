# %% [markdown]
# # exp049: fixed edge selection diagnostic
#
# The exp047 predictions remain fixed. This notebook reads score caches and
# original-ID stage arrays, checks their identity, then reports known GEFF edges.
# Truth-constrained ILP solutions are diagnostic only.

# %% [markdown]
# ## 1. Imports and fixed configuration

# %%
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sparse

if importlib.util.find_spec("zarr") is None:
    input_root = Path("/kaggle/input")
    wheel_dirs = []
    for child in input_root.iterdir():
        for directory in (child / "wheels", child):
            if any(directory.glob("zarr-*.whl")):
                wheel_dirs.append(directory)
        for grandchild in child.iterdir():
            if grandchild.is_dir():
                for directory in (grandchild / "wheels", grandchild):
                    if any(directory.glob("zarr-*.whl")):
                        wheel_dirs.append(directory)
    wheel_dirs = list(dict.fromkeys(wheel_dirs))
    if len(wheel_dirs) != 1:
        raise RuntimeError(f"expected one offline Zarr wheel directory: {wheel_dirs}")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--find-links",
        str(wheel_dirs[0]),
        "zarr>=3.0.10,<4",
        "donfig",
        "google-crc32c",
        "numcodecs",
        "typing-extensions",
    ]
    install = subprocess.run(command, text=True, capture_output=True, check=False)
    if install.returncode:
        raise RuntimeError(f"offline Zarr install failed: {install.stderr[-2000:]}")
    print("Installed Zarr from offline support pack:", wheel_dirs[0], flush=True)

import zarr
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.spatial.distance import cdist

EXPERIMENT = "exp049_x138_edge_selection_diagnostic"
ARMS = ("baseline", "expanded")
STAGES = (
    "candidate",
    "ilp",
    "edge_filter",
    "relink",
    "gap1",
    "gap2",
    "low_detection",
    "division",
    "short_track",
    "final",
)
SPACING = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
BASELINE_THRESHOLD = 0.48
EXPANDED_THRESHOLD = 0.10
MAX_PARENTS = 3
ILP_COSTS = {"edge": -1.0, "appearance": 0.0, "disappearance": 2.0, "division": 1.2}
ILP_TIMEOUT = 1200.0
MAX_FORCED_PER_EMBRYO_TYPE = 3
EXPECTED_SELECTION_SHA = "ba7eb44348be9ee18a0a15180276dbb92399a5c5995e26ae3b3d667b8cc30daa"
EXPECTED_OFFICIAL_SHA = "bce246fa9783db769ece67d9959c1a5939edb74c37f0df9157070219566ed59e"


@dataclass(frozen=True)
class Inputs:
    cache_root: Path
    gt_root: Path
    stage_root: Path | None
    evidence_root: Path
    output_root: Path


def input_paths() -> Inputs:
    """Resolve local smoke inputs or the mounted exp047 Kaggle output."""
    here = Path.cwd()
    exp047 = (
        here.parent / "exp047_x138_edge_candidates"
        if here.name == EXPERIMENT
        else here / "experiments/exp047_x138_edge_candidates"
    )
    input_root = Path("/kaggle/input")
    kaggle_competition = Path(
        "/kaggle/input/competitions/biohub-cell-tracking-during-development/train"
    )
    if not kaggle_competition.exists():
        kaggle_competition = Path("/kaggle/input/biohub-cell-tracking-during-development/train")
    if input_root.exists():
        children = sorted(child for child in input_root.iterdir() if child.is_dir())
        print("Kaggle input roots:", [child.name for child in children], flush=True)
        matches = []
        for child in children:
            if "exp047" not in child.name and child.name != "notebooks":
                continue
            for receipt in child.rglob("receipt.json"):
                source = receipt.parent
                if (source / "fixed_id_diagnostic.json").is_file():
                    matches.append(source)
        if len(matches) != 1:
            raise RuntimeError(f"expected one exp047 output root, found {matches}")
        kaggle_source = matches[0]
        print("exp047 output root:", kaggle_source, flush=True)
        default_cache = kaggle_source / "baseline/candidate_cache"
        default_evidence = kaggle_source
        default_stage = kaggle_source / "stages"
        default_gt = kaggle_competition
        default_output = Path("/kaggle/working/exp049_diagnostic")
    else:
        default_cache = exp047 / "artifacts/outer_cache_dataset"
        default_evidence = exp047 / "artifacts/evidence/inference_v1"
        default_stage = None
        default_gt = Path("/tmp/exp045-gt-local/train")
        default_output = here / "artifacts/diagnostic"
    stage_text = os.environ.get("EXP049_STAGE_ROOT", "").strip()
    return Inputs(
        cache_root=Path(os.environ.get("EXP049_CACHE_ROOT", default_cache)),
        gt_root=Path(os.environ.get("EXP049_GT_ROOT", default_gt)),
        stage_root=Path(stage_text) if stage_text else default_stage,
        evidence_root=Path(os.environ.get("EXP049_EVIDENCE_ROOT", default_evidence)),
        output_root=Path(os.environ.get("EXP049_OUTPUT_ROOT", default_output)),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pairs(rows: np.ndarray) -> set[tuple[int, int]]:
    return {(int(row[0]), int(row[1])) for row in rows}


# %% [markdown]
# ## 2. Input, cache, and original ID checks


# %%
def selection_and_manifest(paths: Inputs) -> tuple[list[str], dict, dict, dict]:
    selection_path = paths.evidence_root / "exp047_selection.json"
    if not selection_path.exists():
        selection_path = paths.evidence_root.parent / "exp047_selection.json"
    official_path = paths.evidence_root / "official_metric.json"
    if sha256(selection_path) != EXPECTED_SELECTION_SHA:
        raise ValueError("exp047 selection SHA mismatch")
    if sha256(official_path) != EXPECTED_OFFICIAL_SHA:
        raise ValueError("exp047 official metric SHA mismatch")
    selection = json.loads(selection_path.read_text())
    stems = sorted(stem for group in selection["selected"].values() for stem in group)
    if len(stems) != 20 or len(set(stems)) != 20:
        raise ValueError("expected 20 unique head-unseen videos")
    if any(
        sum(stem.startswith(f"{embryo}_") for stem in stems) != 10 for embryo in ("44b6", "6bba")
    ):
        raise ValueError("expected ten videos per embryo")
    manifest_path = paths.cache_root / "cache_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["selection_sha256"] != EXPECTED_SELECTION_SHA:
            raise ValueError("cache selection SHA differs from exp047")
    else:
        receipt = json.loads((paths.evidence_root / "receipt.json").read_text())
        if receipt["selected_videos"] != selection["selected"]:
            raise ValueError("exp047 receipt selection differs")
        manifest = {"videos": receipt["processing"]}
    if set(manifest["videos"]) != set(stems):
        raise ValueError("cache video set differs from exp047")
    fixed = json.loads((paths.evidence_root / "fixed_id_diagnostic.json").read_text())
    if set(fixed) != set(stems):
        raise ValueError("fixed-ID evidence selection differs")
    official = json.loads(official_path.read_text())
    return stems, manifest, fixed, official


def select_edges(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Exact exp047 candidate rule, including authoritative admitted baseline."""
    coords = np.asarray(payload["coords"], dtype=np.float64)
    src = np.asarray(payload["edge_src"], dtype=np.int64)
    tgt = np.asarray(payload["edge_tgt"], dtype=np.int64)
    prob = np.asarray(payload["edge_prob"], dtype=np.float64)
    baseline = np.asarray(payload["admitted"], dtype=np.float64).reshape(-1, 4)
    if coords.ndim != 2 or coords.shape[1] != 4 or len(src) != len(tgt) or len(src) != len(prob):
        raise ValueError("malformed score cache")
    if (
        not np.isfinite(coords).all()
        or not np.isfinite(prob).all()
        or not np.isfinite(baseline).all()
    ):
        raise ValueError("nonfinite cache values")
    if (
        np.any(src < 0)
        or np.any(tgt < 0)
        or np.any(src >= len(coords))
        or np.any(tgt >= len(coords))
    ):
        raise ValueError("score edge ID out of range")
    if (
        np.any(coords[src, 0] + 1 != coords[tgt, 0])
        or np.any(prob <= EXPANDED_THRESHOLD)
        or np.any(prob > 1)
    ):
        raise ValueError("score timing or threshold mismatch")
    if np.any(baseline[:, 2] <= BASELINE_THRESHOLD):
        raise ValueError("baseline threshold mismatch")
    score_pairs = pairs(np.column_stack((src, tgt)))
    base_pairs = pairs(baseline)
    if len(score_pairs) != len(src) or len(base_pairs) != len(baseline):
        raise ValueError("duplicate scored or baseline edge")
    if {
        (int(s), int(t)) for s, t, p in zip(src, tgt, prob, strict=True) if p > BASELINE_THRESHOLD
    } != base_pairs:
        raise ValueError("baseline edge ID mismatch")
    score_by_pair = {(int(s), int(t)): float(p) for s, t, p in zip(src, tgt, prob, strict=True)}
    for s, t, p, _ in baseline:
        if not np.isclose(score_by_pair[int(s), int(t)], p, rtol=2e-6, atol=1e-8):
            raise ValueError("baseline edge score mismatch")
    order = np.lexsort((src, -prob, tgt))
    chosen = []
    last_target, rank = -1, 0
    for index in order:
        target = int(tgt[index])
        if target != last_target:
            last_target, rank = target, 0
        if rank < MAX_PARENTS and (int(src[index]), target) not in base_pairs:
            chosen.append(int(index))
        rank += 1
    extra = np.empty((len(chosen), 4), dtype=np.float64)
    if chosen:
        indices = np.asarray(chosen, dtype=np.int64)
        extra[:, :2] = np.column_stack((src[indices], tgt[indices]))
        extra[:, 2] = prob[indices]
        extra[:, 3] = np.linalg.norm(coords[src[indices], 1:] - coords[tgt[indices], 1:], axis=1)
    expanded = np.concatenate((baseline, extra))
    if len(pairs(expanded)) != len(expanded):
        raise ValueError("expanded candidate duplicates")
    return {"baseline": baseline, "expanded": expanded, "added": extra}


def load_cache(path: Path, manifest_row: dict) -> dict[str, np.ndarray]:
    expected_sha = manifest_row.get("reduced_sha256", manifest_row.get("cache_sha256"))
    if sha256(path) != expected_sha:
        raise ValueError(f"cache SHA mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    selected = select_edges(payload)
    if (len(selected["baseline"]), len(selected["expanded"]), len(selected["added"])) != (
        manifest_row["baseline_candidates"],
        manifest_row["expanded_candidates"],
        manifest_row["added_candidates"],
    ):
        raise ValueError("candidate count differs from exp047 cache manifest")
    return payload


def load_gt(path: Path) -> tuple[np.ndarray, set[tuple[int, int]]]:
    group = zarr.open_group(str(path), mode="r")
    ids = np.asarray(group["nodes/ids"][:], dtype=np.int64)
    columns = [np.asarray(group[f"nodes/props/{axis}/values"][:]) for axis in ("t", "z", "y", "x")]
    nodes = np.column_stack((ids, *columns)).astype(np.float64)
    edges = pairs(np.asarray(group["edges/ids"][:], dtype=np.int64))
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate GEFF node ID")
    return nodes, edges


def match_gt(coords: np.ndarray, gt_nodes: np.ndarray, radius_um: float = 7.0) -> dict[int, int]:
    """Exp047 per-frame one-to-one assignment at the original candidate IDs."""
    mapped = {}
    for t in sorted(set(coords[:, 0].astype(int)) | set(gt_nodes[:, 1].astype(int))):
        candidate_ids = np.flatnonzero(coords[:, 0] == t)
        gt_rows = gt_nodes[gt_nodes[:, 1] == t]
        if not len(candidate_ids) or not len(gt_rows):
            continue
        distance = cdist(coords[candidate_ids, 1:] * SPACING, gt_rows[:, 2:5] * SPACING)
        within = distance <= radius_um
        rows, cols = linear_sum_assignment(np.where(within, distance, 1e9))
        for row, col in zip(rows, cols, strict=True):
            if within[row, col]:
                mapped[int(gt_rows[col, 0])] = int(candidate_ids[row])
    return mapped


def known_events(
    gt_edges: set[tuple[int, int]], gt_to_candidate: dict[int, int]
) -> tuple[set[tuple[int, int]], dict[int, set[tuple[int, int]]]]:
    mapped = {
        (gt_to_candidate[s], gt_to_candidate[t])
        for s, t in gt_edges
        if s in gt_to_candidate and t in gt_to_candidate
    }
    daughters: dict[int, set[int]] = defaultdict(set)
    for s, t in gt_edges:
        daughters[s].add(t)
    divisions = {
        gt_to_candidate[s]: {(gt_to_candidate[s], gt_to_candidate[t]) for t in targets}
        for s, targets in daughters.items()
        if len(targets) >= 2 and s in gt_to_candidate and all(t in gt_to_candidate for t in targets)
    }
    return mapped, divisions


def load_stage_edges(path: Path, original_ids: set[int]) -> set[tuple[int, int]]:
    with np.load(path, allow_pickle=False) as payload:
        edges = np.asarray(payload["edges"], dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(edges[:, :2]).all():
        raise ValueError(f"nonfinite stage edge ID: {path}")
    return {
        (int(s), int(t)) for s, t, _ in edges if int(s) in original_ids and int(t) in original_ids
    }


def original_ids_from_ilp(path: Path, coords: np.ndarray) -> set[int]:
    with np.load(path, allow_pickle=False) as payload:
        nodes = np.asarray(payload["nodes"], dtype=np.float64).reshape(-1, 5)
    if (
        not len(nodes)
        or not np.isfinite(nodes).all()
        or not np.array_equal(nodes[:, 0], np.floor(nodes[:, 0]))
    ):
        raise ValueError(f"invalid initial ILP node IDs: {path}")
    ids = nodes[:, 0].astype(int)
    if np.any(ids < 0) or np.any(ids >= len(coords)) or len(set(ids)) != len(ids):
        raise ValueError(f"initial ILP node ID out of range: {path}")
    if not np.allclose(nodes[:, 1:], coords[ids], rtol=0, atol=1e-4):
        raise ValueError(f"initial ILP node ID/coordinate mismatch: {path}")
    return set(ids)


# %% [markdown]
# ## 3. Score ranks and stage transitions


# %%
def rank_known_edges(
    src: np.ndarray, tgt: np.ndarray, prob: np.ndarray, known: set[tuple[int, int]]
) -> dict[tuple[int, int], dict]:
    by_target: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for s, t, p in zip(src, tgt, prob, strict=True):
        by_target[int(t)].append((int(s), float(p)))
    result = {}
    for edge in sorted(known):
        ranked = sorted(by_target.get(edge[1], []), key=lambda item: (-item[1], item[0]))
        found = next(
            ((rank, score) for rank, (s, score) in enumerate(ranked, 1) if s == edge[0]), None
        )
        result[edge] = {
            "score": None if found is None else found[1],
            "rank_among_captured_parents": None if found is None else found[0],
            "captured_parent_count": len(ranked),
        }
    return result


def first_lost(stages: dict[str, set[tuple[int, int]]], event: set[tuple[int, int]]) -> str | None:
    for stage in STAGES:
        if not event <= stages[stage]:
            return stage
    return None


def event_rows(
    stem: str,
    arm: str,
    known: set[tuple[int, int]],
    divisions: dict[int, set[tuple[int, int]]],
    stages: dict[str, set[tuple[int, int]]],
    ranks: dict,
) -> list[dict]:
    rows = []
    selected_parents: dict[int, list[int]] = defaultdict(list)
    selected_children: dict[int, list[int]] = defaultdict(list)
    for source, target in stages["ilp"]:
        selected_parents[target].append(source)
        selected_children[source].append(target)
    for edge in sorted(known):
        rows.append(
            {
                "stem": stem,
                "embryo": stem.split("_", 1)[0],
                "arm": arm,
                "type": "edge",
                "event_edges": [list(edge)],
                "score": ranks[edge]["score"],
                "rank": ranks[edge]["rank_among_captured_parents"],
                "ilp_edge_cost": None
                if ranks[edge]["score"] is None
                else ILP_COSTS["edge"] * ranks[edge]["score"],
                "ilp_selected_parents_for_daughter": sorted(selected_parents[edge[1]]),
                "ilp_selected_children_for_parent": sorted(selected_children[edge[0]]),
                "first_lost": first_lost(stages, {edge}),
                "stages": {stage: edge in stages[stage] for stage in STAGES},
            }
        )
    for parent, event in sorted(divisions.items()):
        rows.append(
            {
                "stem": stem,
                "embryo": stem.split("_", 1)[0],
                "arm": arm,
                "type": "division",
                "parent": parent,
                "event_edges": [list(edge) for edge in sorted(event)],
                "score": None,
                "rank": None,
                "ilp_selected_parents_for_daughters": {
                    str(target): sorted(selected_parents[target]) for _, target in event
                },
                "ilp_selected_children_for_parent": sorted(selected_children[parent]),
                "first_lost": first_lost(stages, event),
                "stages": {stage: event <= stages[stage] for stage in STAGES},
            }
        )
    return rows


def annotated_parent_contradictions(
    selected: set[tuple[int, int]],
    gt_edges: set[tuple[int, int]],
    gt_to_candidate: dict[int, int],
) -> int:
    candidate_to_gt = {candidate: gt for gt, candidate in gt_to_candidate.items()}
    known_parent = {target: source for source, target in gt_edges}
    count = 0
    for source, target in selected:
        gt_source = candidate_to_gt.get(source)
        gt_target = candidate_to_gt.get(target)
        if gt_source is not None and gt_target is not None:
            parent = known_parent.get(gt_target)
            count += int(parent is not None and parent != gt_source)
    return count


def check_saved_counts(
    stem: str,
    arm: str,
    stages: dict[str, set[tuple[int, int]]],
    known: set[tuple[int, int]],
    divisions: dict[int, set[tuple[int, int]]],
    fixed: dict,
    gt_edges: set[tuple[int, int]],
    gt_to_candidate: dict[int, int],
) -> None:
    for stage in STAGES:
        expected = fixed[stem]["arms"][arm][stage]
        actual = {
            "selected_edges": len(stages[stage]),
            "known_edges_selected": len(known & stages[stage]),
            "known_divisions_selected": sum(event <= stages[stage] for event in divisions.values()),
            "annotated_parent_contradictions": annotated_parent_contradictions(
                stages[stage], gt_edges, gt_to_candidate
            ),
            "both_endpoints_detected": len(known),
            "known_edge_denominator": len(gt_edges),
            "all_division_nodes_detected": len(divisions),
        }
        for key, value in actual.items():
            if value != expected[key]:
                raise ValueError(f"{stem}/{arm}/{stage} {key}: {value} != {expected[key]}")


def pick_forced_cases(rows: list[dict]) -> list[dict]:
    """Expanded known events present in candidates but absent from normal ILP."""
    selected = []
    for embryo in ("44b6", "6bba"):
        for kind in ("edge", "division"):
            eligible = [
                r
                for r in rows
                if r["embryo"] == embryo
                and r["arm"] == "expanded"
                and r["type"] == kind
                and r["stages"]["candidate"]
                and not r["stages"]["ilp"]
            ]
            eligible.sort(
                key=lambda r: hashlib.sha256(
                    json.dumps([r["stem"], r["event_edges"]], separators=(",", ":")).encode()
                ).hexdigest()
            )
            selected.extend(eligible[:MAX_FORCED_PER_EMBRYO_TYPE])
    if len(selected) > 12:
        raise AssertionError("forced-case limit exceeded")
    return selected


# %% [markdown]
# ## 4. Original ILP objective and truth-constrained diagnostic solve


# %%
def objective_from_ilp_stage(path: Path, candidates: np.ndarray) -> float:
    """Evaluate exp047's exact saved ILP solution under its fixed objective."""
    with np.load(path, allow_pickle=False) as payload:
        nodes = np.asarray(payload["nodes"], dtype=np.float64).reshape(-1, 5)
        edges = np.asarray(payload["edges"], dtype=np.float64).reshape(-1, 3)
    selected_nodes = {int(row[0]) for row in nodes}
    scores = {(int(s), int(t)): float(p) for s, t, p, _ in candidates}
    selected_edges = pairs(edges)
    if not selected_edges <= scores.keys() or any(
        s not in selected_nodes or t not in selected_nodes for s, t in selected_edges
    ):
        raise ValueError(f"saved ILP edge does not match candidates or nodes: {path}")
    indegree = Counter(t for _, t in selected_edges)
    outdegree = Counter(s for s, _ in selected_edges)
    if any(indegree[node] > 1 or outdegree[node] > 2 for node in selected_nodes):
        raise ValueError(f"saved ILP degree constraint mismatch: {path}")
    return (
        ILP_COSTS["edge"] * sum(scores[edge] for edge in selected_edges)
        + ILP_COSTS["appearance"] * sum(indegree[node] == 0 for node in selected_nodes)
        + ILP_COSTS["disappearance"] * sum(outdegree[node] == 0 for node in selected_nodes)
        + ILP_COSTS["division"] * sum(outdegree[node] == 2 for node in selected_nodes)
    )


def solve_ilp(
    node_count: int, candidates: np.ndarray, forced: set[tuple[int, int]] = frozenset()
) -> dict:
    """Binary formulation from exp047 tracksdata ILPSolver; no overlap constraints exist."""
    edges = np.asarray(candidates, dtype=np.float64).reshape(-1, 4)
    n, m = node_count, len(edges)
    if not n or not m:
        raise ValueError("empty ILP input")
    edge_pairs = [(int(s), int(t)) for s, t in edges[:, :2]]
    edge_index = {pair: i for i, pair in enumerate(edge_pairs)}
    if len(edge_index) != m or not forced <= edge_index.keys():
        raise ValueError("forced edge missing or duplicate ILP edge")
    if any(s < 0 or s >= n or t < 0 or t >= n for s, t in edge_pairs):
        raise ValueError("ILP edge ID out of range")
    # Variable blocks: node, appearance, disappearance, division, edge.
    count = 4 * n + m
    cost = np.concatenate(
        (
            np.zeros(n),
            np.full(n, ILP_COSTS["appearance"]),
            np.full(n, ILP_COSTS["disappearance"]),
            np.full(n, ILP_COSTS["division"]),
            ILP_COSTS["edge"] * edges[:, 2],
        )
    )
    row, col, value = [], [], []

    def add(r: int, c: int, v: float) -> None:
        row.append(r)
        col.append(c)
        value.append(v)

    for i in range(n):
        add(i, i, -1)
        add(i, n + i, 1)
        add(n + i, i, -1)
        add(n + i, 2 * n + i, 1)
        add(n + i, 3 * n + i, -1)
        add(2 * n + i, i, -1)
        add(2 * n + i, 3 * n + i, 1)
    for i, (s, t) in enumerate(edge_pairs):
        add(t, 4 * n + i, 1)
        add(n + s, 4 * n + i, 1)
    matrix = sparse.coo_matrix((value, (row, col)), shape=(3 * n, count)).tocsr()
    lower = np.concatenate((np.zeros(2 * n), np.full(n, -np.inf)))
    upper = np.zeros(3 * n)
    lower_bounds = np.zeros(count)
    for edge in forced:
        lower_bounds[4 * n + edge_index[edge]] = 1
    started = time.monotonic()
    result = milp(
        cost,
        integrality=np.ones(count, dtype=np.uint8),
        bounds=Bounds(lower_bounds, np.ones(count)),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": ILP_TIMEOUT, "mip_rel_gap": 0.0},
    )
    elapsed = time.monotonic() - started
    selected = (
        set()
        if result.x is None
        else {edge_pairs[i] for i, flag in enumerate(result.x[4 * n :]) if flag > 0.5}
    )
    return {
        "status": int(result.status),
        "message": str(result.message),
        "optimal": result.status == 0,
        "seconds": elapsed,
        "objective": None if result.fun is None else float(result.fun),
        "selected_edges": selected,
        "solver": "scipy.optimize.milp HiGHS",
    }


# %% [markdown]
# ## 5. Pilot gate, 20-video readout, and saved evidence


# %%
def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def preflight(paths: Inputs, stems: list[str], manifest: dict, fixed: dict) -> dict:
    output = {}
    for stem in stems:
        path = paths.cache_root / f"{stem}.npz"
        payload = load_cache(path, manifest["videos"][stem])
        selected = select_edges(payload)
        row = {
            "cache_sha256": sha256(path),
            "candidate_counts": {arm: len(selected[arm]) for arm in ARMS},
        }
        for arm in ARMS:
            expected = fixed[stem]["arms"][arm]["candidate"]["selected_edges"]
            if len(selected[arm]) != expected:
                raise ValueError(f"{stem}/{arm} candidate count differs from exp047")
        output[stem] = row
    return output


def main() -> None:
    paths = input_paths()
    stems, manifest, fixed, official = selection_and_manifest(paths)
    if os.environ.get("EXP049_SKIP_PREFLIGHT") == "1":
        previous = json.loads((paths.output_root / "input_checks.json").read_text())
        if previous["status"] != "candidate_cache_verified" or set(previous["videos"]) != set(
            stems
        ):
            raise ValueError("preflight record missing or incomplete")
        checks = previous["videos"]
    else:
        checks = preflight(paths, stems, manifest, fixed)
    write_json(
        paths.output_root / "input_checks.json",
        {
            "status": "candidate_cache_verified",
            "selection_sha256": EXPECTED_SELECTION_SHA,
            "official_metric_sha256": EXPECTED_OFFICIAL_SHA,
            "videos": checks,
        },
    )
    if paths.stage_root is None:
        print(
            "Candidate cache verified. Stage edge-ID files are absent; "
            "stopping before cause classification."
        )
        return
    pilots = [
        selection_stems[0]
        for selection_stems in (
            sorted(stem for stem in stems if stem.startswith("44b6_")),
            sorted(stem for stem in stems if stem.startswith("6bba_")),
        )
    ]
    rows = []
    normal_solves = {}
    for index, stem in enumerate([*pilots, *(s for s in stems if s not in pilots)]):
        payload = load_cache(paths.cache_root / f"{stem}.npz", manifest["videos"][stem])
        coords = np.asarray(payload["coords"], dtype=np.float64)
        selected = select_edges(payload)
        gt_nodes, gt_edges = load_gt(paths.gt_root / f"{stem}.geff")
        mapping = match_gt(coords, gt_nodes)
        known, divisions = known_events(gt_edges, mapping)
        ranks = rank_known_edges(
            payload["edge_src"], payload["edge_tgt"], payload["edge_prob"], known
        )
        for arm in ARMS:
            stages = {"candidate": pairs(selected[arm])}
            original_ids = original_ids_from_ilp(paths.stage_root / arm / stem / "ilp.npz", coords)
            for stage in STAGES[1:]:
                path = paths.stage_root / arm / stem / f"{stage}.npz"
                if not path.is_file():
                    raise FileNotFoundError(f"missing exp047 stage edge IDs: {path}")
                stages[stage] = load_stage_edges(path, original_ids)
            check_saved_counts(stem, arm, stages, known, divisions, fixed, gt_edges, mapping)
            ilp_path = paths.stage_root / arm / stem / "ilp.npz"
            saved_objective = objective_from_ilp_stage(ilp_path, selected[arm])
            if index < 2:
                normal = solve_ilp(len(coords), selected[arm])
                if not normal["optimal"] or normal["selected_edges"] != stages["ilp"]:
                    raise RuntimeError(f"{stem}/{arm}: normal ILP differs from exp047; stopping")
                if not np.isclose(normal["objective"], saved_objective, rtol=0, atol=1e-5):
                    raise RuntimeError(f"{stem}/{arm}: normal objective differs from exp047")
            normal_solves[stem, arm] = saved_objective
            rows.extend(event_rows(stem, arm, known, divisions, stages, ranks))
        if index == 1:
            print("Both embryo pilots matched exp047 stage IDs and normal ILP")
            write_json(paths.output_root / "pilot_events.json", rows)
            if os.environ.get("EXP049_PILOT_ONLY") == "1":
                return
    if os.environ.get("EXP049_STAGE_ONLY") == "1":
        write_json(paths.output_root / "stage_events.json", rows)
        print("Stage readout complete; truth-constrained ILP was skipped for stage-only smoke.")
        return
    forced_rows = []
    for case in pick_forced_cases(rows):
        stem = case["stem"]
        payload = load_cache(paths.cache_root / f"{stem}.npz", manifest["videos"][stem])
        expanded = select_edges(payload)["expanded"]
        answer = solve_ilp(
            len(payload["coords"]), expanded, {tuple(edge) for edge in case["event_edges"]}
        )
        forced_rows.append(
            {
                "stem": stem,
                "embryo": case["embryo"],
                "type": case["type"],
                "event_edges": case["event_edges"],
                "optimal": answer["optimal"],
                "status": answer["status"],
                "message": answer["message"],
                "seconds": answer["seconds"],
                "normal_objective": normal_solves[stem, "expanded"],
                "forced_objective": answer["objective"],
                "objective_delta": None
                if answer["objective"] is None
                else answer["objective"] - normal_solves[stem, "expanded"],
            }
        )
    summary = {}
    for embryo in ("44b6", "6bba"):
        subset = [row for row in rows if row["embryo"] == embryo]
        summary[embryo] = {
            arm: {
                kind: {
                    "denominator": len(
                        group := [r for r in subset if r["arm"] == arm and r["type"] == kind]
                    ),
                    "first_lost": dict(Counter(str(r["first_lost"]) for r in group)),
                    "rank_unavailable": sum(r["rank"] is None for r in group),
                }
                for kind in ("edge", "division")
            }
            for arm in ARMS
        }
    write_json(paths.output_root / "edge_events.json", rows)
    write_json(paths.output_root / "forced_ilp.json", forced_rows)
    write_json(
        paths.output_root / "summary.json",
        {
            "status": "diagnostic_completed",
            "by_embryo": summary,
            "saved_official_metric": {
                arm: {"by_embryo": official[arm]["by_embryo"], "rows": official[arm]["rows"]}
                for arm in ARMS
            },
            "solver": "scipy.optimize.milp HiGHS; normal edge set verified against exp047",
            "unknown_edges_are_not_negative": True,
        },
    )
    print("Diagnostic complete:", paths.output_root)


# %%
if __name__ == "__main__":
    main()
