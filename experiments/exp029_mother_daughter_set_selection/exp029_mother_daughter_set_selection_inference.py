# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Direct parent selection: fixed-graph inference
#
# This notebook applies saved mother-daughter set models to the fixed cache.
# Each two-frame window is decoded with one daughter set per mother and no
# duplicate daughter assignment, then the pinned graph repair is evaluated.

# %% [markdown]
# ## 1. Imports and deterministic helpers

# %%
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

EXPERIMENT = "exp029_mother_daughter_set_selection"
COMPETITION = "biohub-cell-tracking-during-development"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_unique(paths: list[Path], label: str) -> Path:
    valid = sorted(set(path.resolve() for path in paths if path.exists()))
    if len(valid) != 1:
        raise RuntimeError({label: [str(path) for path in valid]})
    return valid[0]


def ensure_tracksdata() -> None:
    try:
        import tracksdata  # noqa: F401

        return
    except ImportError:
        pass
    wheel_dirs = [
        path
        for path in Path("/kaggle/input").rglob("wheels")
        if "biohub-tracking-support-pack" in path.as_posix() and path.is_dir()
    ]
    wheel_dir = resolve_unique(wheel_dirs, "offline wheel directory")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(wheel_dir),
        "tracksdata",
        "geff",
        "geff-spec<1.2",
        "zarr>=3,<4",
        "polars",
        "pyscipopt",
        "ilpy",
        "rustworkx",
    ]
    subprocess.run(command, check=True)
    import tracksdata  # noqa: F401


# %% [markdown]
# ## 2. Fixed cache and annotation readers


# %%
def resolve_cache_root(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    cache_cfg = config["data"]["cache"]
    slug = str(cache_cfg["kernel_source"]).split("/", 1)[-1]
    summary_path = resolve_unique(
        [
            path
            for path in Path("/kaggle/input/notebooks").rglob(cache_cfg["summary_file"])
            if slug in path.as_posix()
        ],
        "exp015 cache summary",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if summary.get("summary_sha256") != json_sha256(unsigned):
        raise RuntimeError("cache summary self-hash mismatch")
    for key, expected in (
        ("summary_sha256", cache_cfg["summary_sha256"]),
        ("cache_identity_sha256", cache_cfg["identity_sha256"]),
        ("dataset_count", cache_cfg["expected_dataset_count"]),
        ("window_count", cache_cfg["expected_window_count"]),
    ):
        if summary.get(key) != expected:
            raise RuntimeError({"cache_summary_mismatch": key})
    root = summary_path.parent / cache_cfg["directory"]
    paths = sorted(root.glob("*/*.npz"))
    if len(paths) != cache_cfg["expected_window_count"]:
        raise RuntimeError("cache window count mismatch")
    return root, summary


def resolve_train_dir() -> Path:
    return resolve_unique(
        [
            Path(f"/kaggle/input/competitions/{COMPETITION}/train"),
            Path(f"/kaggle/input/{COMPETITION}/train"),
        ],
        "competition train directory",
    )


def read_window(path: Path, expected_checkpoint_sha: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(saved["__metadata_json__"].tobytes().decode("utf-8"))
        keys = (
            "candidate_ids_src",
            "candidate_ids_tgt",
            "coords_src_physical",
            "coords_tgt_physical",
            "coords_src_grid",
            "coords_tgt_grid",
            "primary_features_src",
            "primary_features_tgt",
            "position_features_src",
            "position_features_tgt",
            "detection_scores_src",
            "detection_scores_tgt",
        )
        arrays = {key: np.ascontiguousarray(saved[key]) for key in keys}
    frames = metadata.get("window_frames")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("experiment") != "exp015_oracle_stage_limits"
        or metadata.get("dataset") != path.parent.name
        or metadata.get("primary_checkpoint_sha256") != expected_checkpoint_sha
        or not isinstance(frames, list)
        or len(frames) != 2
        or int(frames[1]) != int(frames[0]) + 1
    ):
        raise RuntimeError({"cache_metadata_mismatch": str(path)})
    if path.name != f"{int(frames[0]):06d}_{int(frames[1]):06d}.npz":
        raise RuntimeError("cache filename does not match frames")
    for side in ("src", "tgt"):
        ids = arrays[f"candidate_ids_{side}"]
        if len(ids) == 0 or len(set(map(int, ids))) != len(ids):
            raise RuntimeError({"empty_or_duplicate_candidates": str(path), "side": side})
        grid = np.asarray(arrays[f"coords_{side}_grid"])
        score = np.asarray(arrays[f"detection_scores_{side}"]).reshape(-1)
        if grid.shape != (len(ids), 3) or not np.isfinite(grid).all():
            raise RuntimeError({"invalid_grid_coordinates": str(path), "side": side})
        if len(score) != len(ids) or not np.isfinite(score).all():
            raise RuntimeError({"invalid_detection_scores": str(path), "side": side})
        for suffix, columns in (("coords", 3), ("primary_features", 32), ("position_features", 32)):
            key = f"{suffix}_{side}_physical" if suffix == "coords" else f"{suffix}_{side}"
            value = arrays[key]
            if value.shape != (len(ids), columns) or not np.isfinite(value).all():
                raise RuntimeError({"invalid_cache_array": str(path), "name": key})
    arrays["frames"] = tuple(map(int, frames))
    arrays["sample"] = path.parent.name
    arrays["features_src"] = np.concatenate(
        [
            arrays["primary_features_src"],
            arrays["position_features_src"],
            np.asarray(arrays["detection_scores_src"]).reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    arrays["features_tgt"] = np.concatenate(
        [
            arrays["primary_features_tgt"],
            arrays["position_features_tgt"],
            np.asarray(arrays["detection_scores_tgt"]).reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    return arrays


def load_annotation(path: Path, scale_zyx_um: tuple[float, float, float]) -> dict[str, Any]:
    import tracksdata as td

    graph = td.graph.IndexedRXGraph.from_geff(path)
    if isinstance(graph, tuple):
        graph = graph[0]
    scale = np.asarray(scale_zyx_um, dtype=np.float64)
    frames: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    seen: set[int] = set()
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        if node_id in seen:
            raise RuntimeError("duplicate GT node id")
        seen.add(node_id)
        frame = int(row["t"])
        coord = np.asarray([row["z"], row["y"], row["x"]], dtype=np.float64) * scale
        frames[frame].append((node_id, coord))
    edges = [
        (int(row["source_id"]), int(row["target_id"]))
        for row in graph.edge_attrs().iter_rows(named=True)
    ]
    if any(source not in seen or target not in seen for source, target in edges):
        raise RuntimeError("dangling GT edge")
    incoming: dict[int, list[int]] = defaultdict(list)
    for source, target in edges:
        incoming[target].append(source)
    return {"frames": frames, "incoming": incoming, "edge_count": len(edges)}


def greedy_match_candidates(
    candidate_coords: np.ndarray,
    gt_entries: list[tuple[int, np.ndarray]],
    max_distance_um: float,
) -> np.ndarray:
    result = np.full(len(candidate_coords), -1, dtype=np.int64)
    if len(candidate_coords) == 0 or not gt_entries:
        return result
    ids = np.asarray([item[0] for item in gt_entries], dtype=np.int64)
    coords = np.asarray([item[1] for item in gt_entries], dtype=np.float64)
    distances = np.linalg.norm(
        np.asarray(candidate_coords)[:, None, :] - coords[None, :, :], axis=2
    )
    nearest = distances.argmin(axis=1)
    nearest_distance = distances[np.arange(len(candidate_coords)), nearest]
    taken = np.zeros(len(ids), dtype=bool)
    for index in np.argsort(nearest_distance, kind="stable"):
        if nearest_distance[index] > max_distance_um:
            break
        match = int(nearest[index])
        if not taken[match]:
            result[index] = ids[match]
            taken[match] = True
    return result


def build_partial_teacher(
    source_matches: np.ndarray,
    target_matches: np.ndarray,
    incoming: dict[int, list[int]],
) -> tuple[np.ndarray, dict[str, int]]:
    source = np.asarray(source_matches, dtype=np.int64)
    target = np.asarray(target_matches, dtype=np.int64)
    labels = np.full(len(target), -1, dtype=np.int64)
    matched_source = {int(node_id): index for index, node_id in enumerate(source) if node_id >= 0}
    stats = {
        "known_parent": 0,
        "known_parent_outside_candidates": 0,
        "unknown": 0,
        "ambiguous_incoming": 0,
    }
    for child, node_id in enumerate(target):
        parents = incoming.get(int(node_id), []) if node_id >= 0 else []
        if len(parents) == 1:
            parent_index = matched_source.get(int(parents[0]))
            if parent_index is None:
                labels[child] = len(source)
                stats["known_parent_outside_candidates"] += 1
            else:
                labels[child] = parent_index
                stats["known_parent"] += 1
        elif len(parents) > 1:
            stats["ambiguous_incoming"] += 1
        else:
            stats["unknown"] += 1
    return labels, stats


def make_example(
    path: Path, annotation: dict[str, Any], checkpoint_sha: str, max_distance_um: float
) -> dict[str, Any]:
    arrays = read_window(path, checkpoint_sha)
    source_frame, target_frame = arrays["frames"]
    frames = annotation["frames"]
    if source_frame not in frames or target_frame not in frames:
        raise RuntimeError("GT frame missing")
    source_matches = greedy_match_candidates(
        arrays["coords_src_physical"], frames[source_frame], max_distance_um
    )
    target_matches = greedy_match_candidates(
        arrays["coords_tgt_physical"], frames[target_frame], max_distance_um
    )
    labels, stats = build_partial_teacher(source_matches, target_matches, annotation["incoming"])
    arrays["labels"] = labels
    arrays["source_matches"] = source_matches
    arrays["target_matches"] = target_matches
    arrays["teacher_stats"] = stats
    return arrays


# %% [markdown]
# ## 3. Transformer and constrained parent selection


# %%
def enumerate_daughter_sets(
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    *,
    maximum_distance_um: float,
    maximum_daughters_per_mother: int,
) -> list[list[tuple[int, ...]]]:
    """Enumerate unordered daughter ID sets without consulting annotations."""
    source = np.asarray(source_coords, dtype=np.float64)
    target = np.asarray(target_coords, dtype=np.float64)
    if source.shape != (len(source_ids), 3) or target.shape != (len(target_ids), 3):
        raise ValueError("candidate coordinates have the wrong shape")
    if len(set(map(int, source_ids))) != len(source_ids) or len(set(map(int, target_ids))) != len(
        target_ids
    ):
        raise ValueError("duplicate candidate IDs")
    if maximum_distance_um <= 0 or maximum_daughters_per_mother <= 0:
        raise ValueError("candidate limits must be positive")
    result: list[list[tuple[int, ...]]] = []
    for coord in source:
        distances = np.linalg.norm(target - coord, axis=1)
        nearby = [j for j in range(len(target)) if distances[j] <= maximum_distance_um]
        nearby.sort(key=lambda j: (float(distances[j]), int(target_ids[j])))
        nearby = nearby[:maximum_daughters_per_mother]
        singles = [(j,) for j in nearby]
        pairs = [
            tuple(sorted((a, b), key=lambda j: int(target_ids[j])))
            for offset, a in enumerate(nearby)
            for b in nearby[offset + 1 :]
        ]
        pairs.sort(key=lambda pair: tuple(int(target_ids[j]) for j in pair))
        result.append([(), *singles, *pairs])
    return result


def make_set_supervision(
    labels: np.ndarray,
    daughter_sets: list[list[tuple[int, ...]]],
    n_source: int,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, int]]:
    """Marginalize unknown links; only observed parents and known no-match are negative."""
    labels = np.asarray(labels, dtype=np.int64)
    if np.any(labels < -1) or np.any(labels > n_source):
        raise ValueError("invalid parent labels")
    allowed: list[np.ndarray] = []
    supervised = np.zeros(n_source, dtype=bool)
    counts = defaultdict(int)
    for parent, options in enumerate(daughter_sets):
        positive = set(np.flatnonzero(labels == parent).tolist())
        negative = set(np.flatnonzero((labels >= 0) & (labels != parent)).tolist())
        counts[f"known_outdegree_{min(len(positive), 3)}"] += 1
        if len(positive) == 2:
            counts["known_two_daughter_parent"] += 1
        if len(positive) > 2:
            counts["over_capacity_parent"] += 1
            allowed.append(np.zeros(len(options), dtype=bool))
            continue
        candidate_union = {j for option in options for j in option}
        counts["known_negative_candidate_daughters"] += len(negative & candidate_union)
        counts["unknown_candidate_daughters"] += int(
            sum(bool(labels[j] < 0) for j in candidate_union)
        )
        if not positive.issubset(candidate_union):
            counts["parent_with_missing_known_daughter"] += 1
            counts["missing_known_daughters"] += len(positive - candidate_union)
            allowed.append(np.zeros(len(options), dtype=bool))
            continue
        if len(positive) == 2:
            if any(set(option) == positive for option in options):
                counts["known_two_daughter_set_recovered"] += 1
        mask = np.asarray(
            [positive.issubset(option) and not negative.intersection(option) for option in options],
            dtype=bool,
        )
        if not mask.any():
            counts["no_compatible_set"] += 1
        elif not mask.all():
            supervised[parent] = True
            counts["supervised_parents"] += 1
            if positive:
                counts["supervised_positive_parents"] += 1
            else:
                counts["supervised_negative_only_parents"] += 1
        allowed.append(mask)
    counts["candidate_sets"] = sum(map(len, daughter_sets))
    counts["mothers"] = n_source
    return allowed, supervised, {key: int(value) for key, value in counts.items()}


def make_model(config: dict[str, Any]) -> Any:
    import torch
    from torch import nn

    class MotherDaughterSetTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = int(config["hidden_dim"])
            self.feature = nn.Linear(int(config["input_feature_dim"]), width)
            self.position = nn.Linear(3, width)
            self.frame = nn.Embedding(2, width)
            block = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=int(config["n_heads"]),
                dim_feedforward=width * 4,
                dropout=float(config["dropout"]),
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(block, num_layers=int(config["n_layers"]))
            self.empty = nn.Linear(width, 1)
            self.singleton = nn.Sequential(
                nn.Linear(width * 2 + 3, width), nn.GELU(), nn.Linear(width, 1)
            )
            self.pair = nn.Sequential(
                nn.Linear(width * 3 + 6, width), nn.GELU(), nn.Linear(width, 1)
            )

        def forward(self, batch: dict[str, Any]) -> Any:
            src = batch["features_src"]
            tgt = batch["features_tgt"]
            src_coord = batch["coords_src"] / 20.0
            tgt_coord = batch["coords_tgt"] / 20.0
            src_mask = batch["source_mask"]
            tgt_mask = batch["target_mask"]
            tokens = torch.cat(
                [
                    self.feature(src) + self.position(src_coord) + self.frame.weight[0],
                    self.feature(tgt) + self.position(tgt_coord) + self.frame.weight[1],
                ],
                dim=1,
            )
            encoded = self.encoder(
                tokens, src_key_padding_mask=~torch.cat([src_mask, tgt_mask], dim=1)
            )
            n_src = src.shape[1]
            mother = encoded[:, :n_src]
            daughter = encoded[:, n_src:]
            indices = batch["set_children"]
            a_index = indices[..., 0].clamp(min=0)
            b_index = indices[..., 1].clamp(min=0)
            batch_index = torch.arange(src.shape[0], device=src.device)[:, None, None]
            a = daughter[batch_index, a_index]
            b = daughter[batch_index, b_index]
            ca = tgt_coord[batch_index, a_index]
            cb = tgt_coord[batch_index, b_index]
            p = mother[:, :, None, :].expand_as(a)
            cp = src_coord[:, :, None, :].expand_as(ca)
            singleton = self.singleton(torch.cat([p, a, ca - cp], dim=-1)).squeeze(-1)
            pair = self.pair(
                torch.cat(
                    [p, a + b, torch.abs(a - b), ca + cb - 2 * cp, torch.abs(ca - cb)], dim=-1
                )
            ).squeeze(-1)
            empty = self.empty(mother).expand_as(singleton)
            logits = torch.where(
                indices[..., 0] < 0, empty, torch.where(indices[..., 1] < 0, singleton, pair)
            )
            return logits.masked_fill(~batch["set_mask"], -1e9)

    return MotherDaughterSetTransformer()


def partial_set_loss(logits: Any, allowed: Any, supervised: Any) -> Any:
    import torch

    if not bool(supervised.any()):
        return logits.sum() * 0.0
    feasible = torch.logsumexp(logits.masked_fill(~allowed, -1e9), dim=-1)
    total = torch.logsumexp(logits, dim=-1)
    return (total - feasible)[supervised].mean()


def decode_daughter_sets(
    logits: np.ndarray,
    daughter_sets: list[list[tuple[int, ...]]],
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    *,
    edge_cost: float,
    maximum_decode_sets_per_mother: int,
    time_limit_seconds: float,
    relative_gap: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Select one set per mother globally, with each daughter assigned at most once."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_array

    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != len(source_ids):
        raise ValueError("set score shape mismatch")
    if len(daughter_sets) != len(source_ids) or maximum_decode_sets_per_mother < 1:
        raise ValueError("set catalog mismatch")
    choices: list[tuple[int, int, tuple[int, ...], float]] = []
    pruned = 0
    for parent, options in enumerate(daughter_sets):
        if not options or options[0] != () or len(options) > scores.shape[1]:
            raise ValueError("invalid daughter set catalog")
        if not np.isfinite(scores[parent, : len(options)]).all():
            raise ValueError("nonfinite set logits")
        ranked = sorted(
            range(1, len(options)),
            key=lambda k: (
                -(float(scores[parent, k]) - edge_cost * len(options[k])),
                tuple(int(target_ids[j]) for j in options[k]),
            ),
        )
        keep = [0, *ranked[: maximum_decode_sets_per_mother - 1]]
        pruned += len(options) - len(keep)
        for option_index in keep:
            option = options[option_index]
            weight = float(
                scores[parent, option_index] - scores[parent, 0] - edge_cost * len(option)
            )
            choices.append((parent, option_index, option, weight))
    n_variables = len(choices)
    n_rows = len(source_ids) + len(target_ids)
    row_indices = []
    col_indices = []
    for col, (parent, _, option, _) in enumerate(choices):
        row_indices.append(parent)
        col_indices.append(col)
        for child in option:
            row_indices.append(len(source_ids) + child)
            col_indices.append(col)
    matrix = coo_array(
        (
            np.ones(len(row_indices), dtype=np.float64),
            (np.asarray(row_indices, dtype=np.int32), np.asarray(col_indices, dtype=np.int32)),
        ),
        shape=(n_rows, n_variables),
    ).tocsr()
    lower = np.r_[np.ones(len(source_ids)), np.zeros(len(target_ids))]
    upper = np.ones(n_rows)
    objective = -np.asarray([row[3] for row in choices])
    objective += np.arange(n_variables) * 1e-10
    result = milp(
        objective,
        integrality=np.ones(n_variables, dtype=np.int32),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": time_limit_seconds, "mip_rel_gap": relative_gap},
    )
    if result.status != 0 or result.x is None:
        raise RuntimeError({"set_selection_milp_status": result.status, "message": result.message})
    chosen = np.full(len(target_ids), -1, dtype=np.int64)
    for flag, (parent, _, option, _) in zip(result.x, choices, strict=True):
        if flag > 0.5:
            for child in option:
                if chosen[child] != -1:
                    raise RuntimeError("daughter assigned to multiple mothers")
                chosen[child] = parent
    return chosen, {
        "candidate_sets": sum(map(len, daughter_sets)),
        "optimized_sets": n_variables,
        "pruned_sets": pruned,
    }


def selection_counts(choices: np.ndarray, labels: np.ndarray, n_source: int) -> dict[str, int]:
    known_parent = (labels >= 0) & (labels < n_source)
    known_null = labels == n_source
    predicted = choices >= 0
    true_positive = int(np.count_nonzero(known_parent & (choices == labels)))
    false_edge = int(
        np.count_nonzero(
            (known_parent & predicted & (choices != labels)) | (known_null & predicted)
        )
    )
    return {
        "known_edges": int(known_parent.sum()),
        "known_null": int(known_null.sum()),
        "true_positive_edges": true_positive,
        "observable_false_edges": false_edge,
        "selected_edges": int(predicted.sum()),
        "correct_null": int(np.count_nonzero(known_null & ~predicted)),
        "selected_unknown_children": int(np.count_nonzero((labels < 0) & predicted)),
        "unknown_children": int(np.count_nonzero(labels < 0)),
    }


# %% [markdown]
# ## 4. Fold artifacts, runtime gate, and direct graph assembly

# %%


def resolve_train_output() -> Path:
    slug = "exp029-mother-daughter-set-selection-train"
    return resolve_unique(
        [
            path.parent
            for path in Path("/kaggle/input/notebooks").rglob("model_manifest.json")
            if slug in path.as_posix()
        ],
        "exp029 train output",
    )


def load_fold_models(
    config: dict[str, Any], train_output: Path, device: Any
) -> tuple[dict[int, Any], dict[int, float], dict[str, Any]]:
    import torch

    manifest_path = train_output / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != EXPERIMENT:
        raise RuntimeError("wrong train manifest")
    if manifest.get("cache_identity_sha256") != config["data"]["cache"]["identity_sha256"]:
        raise RuntimeError("train and inference cache identity mismatch")
    models = {}
    edge_costs = {}
    for record in manifest["folds"]:
        fold = int(record["fold"])
        path = train_output / "models" / str(record["model_file"])
        if file_sha256(path) != record["model_sha256"]:
            raise RuntimeError({"fold_model_sha_mismatch": fold})
        model = make_model(config["model"]).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        models[fold] = model
        edge_costs[fold] = float(record["edge_cost"])
    if set(models) != {0, 1}:
        raise RuntimeError("expected exactly two fold models")
    return models, edge_costs, manifest


def load_public_graph_module(repo_dir: Path) -> Any:
    source = repo_dir / "scripts" / "predict_unet_transformer.py"
    sys.path.insert(0, str(repo_dir / "src"))
    spec = importlib.util.spec_from_file_location("exp029_fixed_graph_io", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("public graph writer missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def register_frame(
    registry: dict[int, tuple[int, int, int, int]], *, frame: int, ids: np.ndarray, grid: np.ndarray
) -> None:
    voxel = (np.asarray(grid, dtype=np.float32) * np.array([1, 4, 4], dtype=np.float32)).astype(
        np.int16
    )
    for node_id, coord in zip(np.asarray(ids, dtype=np.int64), voxel, strict=True):
        row = (int(frame), int(coord[0]), int(coord[1]), int(coord[2]))
        previous = registry.setdefault(int(node_id), row)
        if previous != row:
            raise RuntimeError({"candidate_id_changed_across_windows": int(node_id)})


def predict_sample(
    sample: str,
    sample_paths: list[Path],
    model: Any,
    edge_cost: float,
    checkpoint_sha: str,
    device: Any,
    candidate_cfg: dict[str, Any],
    decoding_cfg: dict[str, Any],
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    import torch
    from scipy.special import logsumexp

    registry: dict[int, tuple[int, int, int, int]] = {}
    edges = []
    with torch.no_grad():
        for window_index, path in enumerate(sample_paths):
            arrays = read_window(path, checkpoint_sha)
            source_frame, target_frame = arrays["frames"]
            if (source_frame, target_frame) != (window_index, window_index + 1):
                raise RuntimeError({"noncontiguous_window": sample, "path": path.name})
            for side, frame in (("src", source_frame), ("tgt", target_frame)):
                register_frame(
                    registry,
                    frame=frame,
                    ids=arrays[f"candidate_ids_{side}"],
                    grid=arrays[f"coords_{side}_grid"],
                )
            n_source = len(arrays["candidate_ids_src"])
            n_target = len(arrays["candidate_ids_tgt"])
            daughter_sets = enumerate_daughter_sets(
                arrays["coords_src_physical"],
                arrays["coords_tgt_physical"],
                arrays["candidate_ids_src"],
                arrays["candidate_ids_tgt"],
                maximum_distance_um=float(candidate_cfg["maximum_distance_um"]),
                maximum_daughters_per_mother=int(candidate_cfg["maximum_daughters_per_mother"]),
            )
            max_sets = max(map(len, daughter_sets))
            set_children = torch.full((1, n_source, max_sets, 2), -1, dtype=torch.long)
            set_mask = torch.zeros((1, n_source, max_sets), dtype=torch.bool)
            for parent, options in enumerate(daughter_sets):
                for option_index, option in enumerate(options):
                    set_children[0, parent, option_index, : len(option)] = torch.as_tensor(option)
                set_mask[0, parent, : len(options)] = True
            batch = {
                "features_src": torch.from_numpy(arrays["features_src"]).unsqueeze(0).to(device),
                "features_tgt": torch.from_numpy(arrays["features_tgt"]).unsqueeze(0).to(device),
                "coords_src": torch.from_numpy(arrays["coords_src_physical"].astype(np.float32))
                .unsqueeze(0)
                .to(device),
                "coords_tgt": torch.from_numpy(arrays["coords_tgt_physical"].astype(np.float32))
                .unsqueeze(0)
                .to(device),
                "source_mask": torch.ones((1, n_source), dtype=torch.bool, device=device),
                "target_mask": torch.ones((1, n_target), dtype=torch.bool, device=device),
                "set_children": set_children.to(device),
                "set_mask": set_mask.to(device),
            }
            logits = model(batch)[0].detach().cpu().numpy()
            chosen, _ = decode_daughter_sets(
                logits,
                daughter_sets,
                arrays["candidate_ids_src"],
                arrays["candidate_ids_tgt"],
                edge_cost=edge_cost,
                maximum_decode_sets_per_mother=int(candidate_cfg["maximum_decode_sets_per_mother"]),
                time_limit_seconds=float(decoding_cfg["time_limit_seconds_per_window"]),
                relative_gap=float(decoding_cfg["relative_gap"]),
            )
            for parent in range(n_source):
                children = tuple(np.flatnonzero(chosen == parent).tolist())
                if not children:
                    continue
                option = next(
                    (
                        index
                        for index, daughter_set in enumerate(daughter_sets[parent])
                        if set(daughter_set) == set(children)
                    ),
                    None,
                )
                if option is None:
                    raise RuntimeError("decoded daughter set missing from catalog")
                probability = float(
                    np.exp(
                        logits[parent, option]
                        - logsumexp(logits[parent, : len(daughter_sets[parent])])
                    )
                )
                for child in children:
                    distance = float(
                        np.linalg.norm(
                            arrays["coords_src_grid"][parent] - arrays["coords_tgt_grid"][child]
                        )
                    )
                    edges.append(
                        (
                            int(arrays["candidate_ids_src"][parent]),
                            int(arrays["candidate_ids_tgt"][child]),
                            probability,
                            distance,
                        )
                    )
    ids = sorted(registry)
    if ids != list(range(len(ids))):
        raise RuntimeError({"candidate_ids_noncontiguous": sample})
    coords = np.asarray([registry[index] for index in ids], dtype=np.int16).reshape((-1, 4))
    return coords, edges


def run_direct_prediction(repo_dir: Path, method: str, test_stems: list[str]) -> None:
    import torch

    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    if not config["model"]["inference"]["early_gate_passed"]:
        raise RuntimeError("two-frame gate is not recorded as passed")
    gate_sha = config["model"]["inference"].get("early_gate_evidence_sha256")
    if not isinstance(gate_sha, str) or len(gate_sha) != 64:
        raise RuntimeError("two-frame gate evidence SHA missing")
    cache_root, summary = resolve_cache_root(config)
    train_output = resolve_train_output()
    gate_path = train_output / "early_gate.json"
    if file_sha256(gate_path) != gate_sha:
        raise RuntimeError("recorded two-frame gate SHA does not match train output")
    gate_record = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        not gate_record.get("pass")
        or not gate_record.get("same_cache_and_teacher")
        or len(gate_record.get("folds", [])) != 2
        or not all(row.get("pass") for row in gate_record["folds"])
    ):
        raise RuntimeError("two-frame gate did not pass for both embryos")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    models, edge_costs, manifest = load_fold_models(config, train_output, device)
    writer = load_public_graph_module(repo_dir)
    sample_names = sorted(path.name for path in cache_root.iterdir() if path.is_dir())
    if sample_names != sorted(test_stems) or len(sample_names) != 199:
        raise RuntimeError("cache sample coverage differs from exp015")
    fold_by_embryo = config["model"]["inference"]["fold_model_by_evaluation_embryo"]
    checkpoint_sha = manifest["checkpoint_sha256"]
    output_root = Path("/kaggle/working/pre_repair_graphs")
    output_root.mkdir(exist_ok=True)
    record = {
        "sample_count": len(sample_names),
        "samples": [],
        "cache_summary_sha256": summary["summary_sha256"],
        "model_manifest_sha256": file_sha256(train_output / "model_manifest.json"),
        "early_gate_evidence_sha256": gate_sha,
        "main_image_encoder_forward_count": 0,
        "set_selection_optimizer": "scipy_milp",
        "use_secondary_tracker": False,
        "use_bidirectional_fusion": False,
    }
    started = time.perf_counter()
    benchmark_names = [
        next(name for name in sample_names if name.startswith(embryo + "_"))
        for embryo in sorted(fold_by_embryo)
    ]
    execution_order = benchmark_names + [
        name for name in sample_names if name not in benchmark_names
    ]
    for index, sample in enumerate(execution_order):
        embryo = sample.split("_", 1)[0]
        fold = int(fold_by_embryo[embryo])
        paths = sorted((cache_root / sample).glob("*.npz"))
        if len(paths) != 99:
            raise RuntimeError({"window_count": sample, "actual": len(paths)})
        sample_start = time.perf_counter()
        coords, edges = predict_sample(
            sample,
            paths,
            models[fold],
            edge_costs[fold],
            checkpoint_sha,
            device,
            config["model"]["candidate_sets"],
            config["model"]["decoding"],
        )
        graph = writer.build_graph(coords, edges)
        destination = repo_dir / "predictions" / sample / method / "split_0" / f"{sample}.geff"
        destination.parent.mkdir(parents=True, exist_ok=True)
        writer.save_graph(graph, destination)
        shutil.copytree(destination, output_root / destination.name)
        row = {
            "sample": sample,
            "embryo": embryo,
            "fold": fold,
            "nodes": len(coords),
            "edges": len(edges),
            "seconds": time.perf_counter() - sample_start,
        }
        record["samples"].append(row)
        print("DIRECT_GRAPH", json.dumps(row, sort_keys=True), flush=True)
        if index == 1:
            projected = (time.perf_counter() - started) / 2 * len(sample_names) * 1.5 + 3600
            record["projected_total_seconds_with_repair_reserve"] = projected
            if projected > 12 * 3600:
                raise RuntimeError({"inference_runtime_gate_seconds": projected})
        if time.perf_counter() - started > 12 * 3600:
            raise RuntimeError("inference runtime gate exceeded")
    record["direct_prediction_seconds"] = time.perf_counter() - started
    record["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
    receipt = Path("/kaggle/working/direct_graph_receipt.json")
    receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("DIRECT_GRAPH_RECEIPT_SHA256", file_sha256(receipt), flush=True)
    del models
    torch.cuda.empty_cache()


# %% [markdown]
# ## 5. Execute the pinned graph repair and evaluate before and after repair


# %%
def patched_repair_source(source: str) -> str:
    start_marker = "start_time = time.time()\navailable_gpu_count = _torch.cuda.device_count()"
    end_marker = "print(f'Prediction completed in {predict_seconds / 60:.2f} minutes')"
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise RuntimeError("pinned exp015 prediction markers changed")
    start = source.index(start_marker)
    end = source.index(end_marker, start) + len(end_marker)
    replacement = (
        "start_time = time.time()\n"
        "run_direct_prediction(REPO_DIR, METHOD, test_stems)\n"
        "predict_seconds = time.time() - start_time\n"
        "print(f'Direct prediction completed in {predict_seconds / 60:.2f} minutes')"
    )
    source = source[:start] + replacement + source[end:]
    stop_marker = "print(f'Wrote {RUN_STATS_PATH}')"
    if source.count(stop_marker) != 1:
        raise RuntimeError("pinned exp015 repair end marker changed")
    source = source[: source.index(stop_marker) + len(stop_marker)] + "\n"
    compile(source, "exp015_fixed_repair_for_direct_graph.py", "exec")
    return source


def normalise_compact_graph_arrays(
    node_ids: np.ndarray, node_tzyx: np.ndarray, edges: np.ndarray
) -> tuple[list[dict[str, int]], list[tuple[int, int]]]:
    old_ids = np.asarray(node_ids, dtype=np.int64)
    coordinates = np.asarray(node_tzyx, dtype=np.float64)
    old_edges = np.asarray(edges, dtype=np.int64)
    if old_ids.ndim != 1 or coordinates.shape != (len(old_ids), 4):
        raise ValueError("invalid compact node arrays")
    if old_edges.shape != (len(old_edges), 2):
        raise ValueError("invalid compact edge array")
    if len(set(old_ids.tolist())) != len(old_ids) or not np.isfinite(coordinates).all():
        raise ValueError("invalid compact node ids or coordinates")
    old_to_new = {int(node_id): index for index, node_id in enumerate(old_ids)}
    nodes = [
        {
            "t": int(row[0]),
            "z": max(0, int(round(float(row[1])))),
            "y": max(0, int(round(float(row[2])))),
            "x": max(0, int(round(float(row[3])))),
        }
        for row in coordinates
    ]
    indexed_edges = []
    for source, target in old_edges:
        if int(source) not in old_to_new or int(target) not in old_to_new:
            raise ValueError("dangling compact edge")
        indexed_edges.append((old_to_new[int(source)], old_to_new[int(target)]))
    return nodes, indexed_edges


def compact_graph_to_geff(source: Path, destination: Path) -> None:
    import polars as pl
    import tracksdata as td
    from biohub_tracking.io import save_graph

    with np.load(source, allow_pickle=False) as payload:
        nodes, indexed_edges = normalise_compact_graph_arrays(
            payload["node_ids"], payload["node_tzyx"], payload["edges"]
        )
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    node_ids = [int(value) for value in graph.bulk_add_nodes(nodes)]
    if indexed_edges:
        graph.bulk_add_edges(
            [
                {"source_id": node_ids[source], "target_id": node_ids[target]}
                for source, target in indexed_edges
            ]
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_graph(graph, destination)


def evaluate_graph_root(root: Path, public_evaluator: Any, label: str) -> dict[str, Any]:
    from biohub_tracking.metrics import summarise

    paths = sorted(root.glob("*.geff"))
    if len(paths) != 199 or len({path.stem for path in paths}) != 199:
        raise RuntimeError({"graph_count": label, "actual": len(paths)})
    run = {"username": "exp029", "method": label, "split": "split_0", "dir": root, "geffs": paths}
    rows = public_evaluator.evaluate_run(run, max_distance=7.0)
    if len(rows) != 199 or [str(row.get("dataset")) for row in rows] != [
        path.stem for path in paths
    ]:
        raise RuntimeError({"official_evaluation_coverage": label})
    invalid = [
        str(row.get("dataset"))
        for row in rows
        if not math.isfinite(float(row.get("edge_tp", float("nan"))))
    ]
    if invalid:
        raise RuntimeError({"official_evaluation_invalid_rows": label, "samples": invalid})
    overall = summarise(rows)
    by_embryo = {
        embryo: summarise([row for row in rows if str(row.get("dataset")).startswith(embryo + "_")])
        for embryo in ("44b6", "6bba")
    }
    return {"overall": overall, "by_embryo": by_embryo, "rows": rows}


def main() -> None:
    import torch

    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("full graph inference must run on Kaggle")
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    if config["experiment"]["name"] != EXPERIMENT:
        raise RuntimeError("wrong experiment config")
    gate = config["model"]["inference"]
    if not gate["early_gate_passed"] or not gate.get("early_gate_evidence_sha256"):
        raise RuntimeError("two-frame gate must pass before full graph inference")
    base = Path("exp015_inference_base.py")
    if file_sha256(base) != "d9a7f46dda21a01d03ffa0072e4e4396dd2542ec4d4996b934bec5b116f49e04":
        raise RuntimeError("fixed repair source SHA mismatch")
    source = patched_repair_source(base.read_text(encoding="utf-8"))
    scope = {"__name__": "__main__", "run_direct_prediction": run_direct_prediction}
    started = time.perf_counter()
    exec(compile(source, str(base), "exec"), scope)
    repaired_compact = Path(scope["ORACLE_FINAL_GRAPH_DIR"])
    repaired_root = Path("/kaggle/working/post_repair_graphs")
    repaired_root.mkdir(exist_ok=True)
    for path in sorted(repaired_compact.glob("*.npz")):
        compact_graph_to_geff(path, repaired_root / f"{path.stem}.geff")
    if len(list(repaired_root.glob("*.geff"))) != 199:
        raise RuntimeError("repaired graph coverage mismatch")
    source_path = Path(scope["REPO_DIR"]) / "scripts" / "evaluate.py"
    if (
        file_sha256(source_path)
        != "614813cc51c3581c6ccda4bb20725a19da8ecac4a27620654bfca58319cffa3c"
    ):
        raise RuntimeError("official evaluator SHA mismatch")
    spec = importlib.util.spec_from_file_location("exp029_official_evaluator", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("official evaluator unavailable")
    evaluator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = evaluator
    spec.loader.exec_module(evaluator)
    original_open_dataset = evaluator.open_dataset

    def metadata_only(*args: object, **kwargs: object) -> object:
        kwargs["load_image"] = False
        data = original_open_dataset(*args, **kwargs)
        if getattr(data, "image", None) is not None:
            raise RuntimeError("official evaluator loaded unused image tensor")
        return data

    evaluator.open_dataset = metadata_only
    pre = evaluate_graph_root(
        Path("/kaggle/working/pre_repair_graphs"), evaluator, "direct_before_repair"
    )
    post = evaluate_graph_root(repaired_root, evaluator, "direct_after_repair")
    official = {
        "before_repair": pre,
        "after_repair": post,
        "fixed_repair_source_sha256": file_sha256(base),
        "official_evaluator_sha256": file_sha256(source_path),
        "direct_graph_receipt_sha256": file_sha256(
            Path("/kaggle/working/direct_graph_receipt.json")
        ),
        "notebook_runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    output = Path("/kaggle/working/official_graph_evaluation.json")
    output.write_text(
        json.dumps(official, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )
    metrics_path = Path("/kaggle/working/metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["status"] = "running"
    metrics["metric"] = "official_adjusted_edge_jaccard_plus_0.1_division_jaccard"
    metrics.setdefault("evidence", {})["inference_stage"] = {
        "official_evaluation_sha256": file_sha256(output),
        "direct_graph_receipt_sha256": official["direct_graph_receipt_sha256"],
        "fixed_repair_source_sha256": official["fixed_repair_source_sha256"],
        "official_evaluator_sha256": official["official_evaluator_sha256"],
        "notebook_runtime_seconds": official["notebook_runtime_seconds"],
        "peak_gpu_memory_bytes": official["peak_gpu_memory_bytes"],
        "before_repair": pre["overall"],
        "after_repair": post["overall"],
        "by_embryo_before": pre["by_embryo"],
        "by_embryo_after": post["by_embryo"],
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )
    print("OFFICIAL_EVALUATION_SHA256", file_sha256(output), flush=True)
    print(
        json.dumps(
            {
                "before": pre["overall"],
                "after": post["overall"],
                "by_embryo_before": pre["by_embryo"],
                "by_embryo_after": post["by_embryo"],
            },
            default=float,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
