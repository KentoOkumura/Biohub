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
# # Mother-daughter set selection from fixed detections
#
# This notebook reads fixed two-frame candidates and organizer GEFF annotations.
# Each mother scores empty, singleton, and unordered daughter-pair sets.
# A partial-label marginal loss excludes only observed contradictions; global
# decoding prevents a daughter from being selected by two mothers.
# Two embryo-held-out models are evaluated before any full-graph inference.

# %% [markdown]
# ## 1. Imports and deterministic helpers

# %%
from __future__ import annotations

import hashlib
import importlib
import json
import os
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


OFFLINE_GRAPH_MODULES = {
    "tracksdata": "tracksdata",
    "zarr": "zarr",
    "geff": "geff",
    "geff_spec": "geff_spec",
    "ilpy": "ilpy",
    "polars": "polars",
    "pyscipopt": "pyscipopt",
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


def polars_runtime_ready() -> bool:
    try:
        import polars as pl
        from polars._plr import PySeries

        _ = PySeries
        return hasattr(pl, "Float16")
    except Exception:
        return False


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
        return
    wheel_dirs = offline_wheel_dirs()
    if not wheel_dirs:
        raise FileNotFoundError("the public support dataset has no offline wheel directory")
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
    remaining = {}
    for package_name, module_name in OFFLINE_GRAPH_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as error:
            remaining[package_name] = f"{type(error).__name__}: {error}"
    if remaining or not polars_runtime_ready():
        raise ImportError({"remaining_graph_import_failures": remaining})


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
    path: Path,
    annotation: dict[str, Any],
    checkpoint_sha: str,
    max_distance_um: float,
    candidate_cfg: dict[str, Any],
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
    daughter_sets = enumerate_daughter_sets(
        arrays["coords_src_physical"],
        arrays["coords_tgt_physical"],
        arrays["candidate_ids_src"],
        arrays["candidate_ids_tgt"],
        maximum_distance_um=float(candidate_cfg["maximum_distance_um"]),
        maximum_daughters_per_mother=int(candidate_cfg["maximum_daughters_per_mother"]),
    )
    allowed, supervised, set_stats = make_set_supervision(
        labels, daughter_sets, len(source_matches)
    )
    arrays["teacher_stats"] = stats
    arrays["daughter_sets"] = daughter_sets
    arrays["allowed_sets"] = allowed
    arrays["supervised_mothers"] = supervised
    arrays["set_teacher_stats"] = set_stats
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
# ## 4. Batches and embryo splits


# %%
class WindowDataset:
    def __init__(
        self,
        paths: list[Path],
        annotations: dict[str, dict[str, Any]],
        checkpoint_sha: str,
        max_distance_um: float,
        candidate_cfg: dict[str, Any],
    ) -> None:
        self.candidate_cfg = candidate_cfg
        self.paths = paths
        self.annotations = annotations
        self.checkpoint_sha = checkpoint_sha
        self.max_distance_um = max_distance_um

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        return make_example(
            path,
            self.annotations[path.parent.name],
            self.checkpoint_sha,
            self.max_distance_um,
            self.candidate_cfg,
        )


def collate_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch_size = len(examples)
    source_count = max(len(item["candidate_ids_src"]) for item in examples)
    target_count = max(len(item["candidate_ids_tgt"]) for item in examples)
    set_count = max(len(options) for item in examples for options in item["daughter_sets"])
    result = {
        "features_src": torch.zeros(batch_size, source_count, 65),
        "features_tgt": torch.zeros(batch_size, target_count, 65),
        "coords_src": torch.zeros(batch_size, source_count, 3),
        "coords_tgt": torch.zeros(batch_size, target_count, 3),
        "source_mask": torch.zeros(batch_size, source_count, dtype=torch.bool),
        "target_mask": torch.zeros(batch_size, target_count, dtype=torch.bool),
        "labels": torch.full((batch_size, target_count), -1, dtype=torch.long),
        "set_children": torch.full((batch_size, source_count, set_count, 2), -1, dtype=torch.long),
        "set_mask": torch.zeros((batch_size, source_count, set_count), dtype=torch.bool),
        "allowed_sets": torch.zeros((batch_size, source_count, set_count), dtype=torch.bool),
        "supervised_mothers": torch.zeros((batch_size, source_count), dtype=torch.bool),
        "metadata": [],
    }
    for batch_index, item in enumerate(examples):
        n_source = len(item["candidate_ids_src"])
        n_target = len(item["candidate_ids_tgt"])
        for side, count in (("src", n_source), ("tgt", n_target)):
            result[f"features_{side}"][batch_index, :count] = torch.from_numpy(
                item[f"features_{side}"]
            )
            result[f"coords_{side}"][batch_index, :count] = torch.from_numpy(
                item[f"coords_{side}_physical"].astype(np.float32)
            )
            result[f"{'source' if side == 'src' else 'target'}_mask"][batch_index, :count] = True
        labels = item["labels"].copy()
        labels[labels == n_source] = source_count
        result["labels"][batch_index, :n_target] = torch.from_numpy(labels)
        for parent, options in enumerate(item["daughter_sets"]):
            for option_index, option in enumerate(options):
                result["set_children"][batch_index, parent, option_index, : len(option)] = (
                    torch.as_tensor(option)
                )
            result["set_mask"][batch_index, parent, : len(options)] = True
            result["allowed_sets"][batch_index, parent, : len(options)] = torch.from_numpy(
                item["allowed_sets"][parent]
            )
        result["supervised_mothers"][batch_index, :n_source] = torch.from_numpy(
            item["supervised_mothers"]
        )
        result["metadata"].append(item)
    return result


def split_samples(
    sample_names: list[str], train_embryo: str, eval_embryo: str, seed: int, fraction: float
) -> dict[str, list[str]]:
    train_pool = [name for name in sample_names if name.startswith(train_embryo + "_")]
    evaluation = [name for name in sample_names if name.startswith(eval_embryo + "_")]
    if len(train_pool) + len(evaluation) != len(sample_names):
        raise RuntimeError("unexpected embryo name or split leakage")
    shuffled = list(train_pool)
    random.Random(seed).shuffle(shuffled)
    n_internal = max(1, int(len(train_pool) * fraction))
    internal = shuffled[:n_internal]
    gradient = shuffled[n_internal:]
    return {"gradient": gradient, "internal": internal, "outer": evaluation}


def make_loader(
    paths: list[Path],
    annotations: dict[str, dict[str, Any]],
    checkpoint_sha: str,
    matching_um: float,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
    candidate_cfg: dict[str, Any],
) -> Any:
    import torch
    from torch.utils.data import DataLoader

    dataset = WindowDataset(paths, annotations, checkpoint_sha, matching_um, candidate_cfg)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_examples,
        generator=generator,
    )


def move_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
    }


# %% [markdown]
# ## 5. Training, internal selection, and two-frame diagnostics


# %%
def run_epoch(model: Any, loader: Any, optimizer: Any | None, device: Any) -> dict[str, float]:
    import torch

    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    supervised = 0
    teacher = defaultdict(int)
    for batch in loader:
        for item in batch["metadata"]:
            for key, value in item["set_teacher_stats"].items():
                teacher[key] += value
        batch = move_to_device(batch, device)
        with torch.set_grad_enabled(training):
            logits = model(batch)
            loss = partial_set_loss(logits, batch["allowed_sets"], batch["supervised_mothers"])
            if training and bool(batch["supervised_mothers"].any()):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        count = int(batch["supervised_mothers"].sum().item())
        loss_sum += float(loss.detach().item()) * count
        supervised += count
    return {
        "loss": loss_sum / max(supervised, 1),
        "supervised_mothers": supervised,
        "set_teacher": dict(teacher),
    }


def evaluate_costs(
    model: Any,
    loader: Any,
    device: Any,
    edge_costs: list[float],
    maximum_decode_sets_per_mother: int,
    time_limit_seconds: float,
    relative_gap: float,
) -> list[dict[str, Any]]:
    import torch

    if not edge_costs or len(set(edge_costs)) != len(edge_costs):
        raise ValueError("edge cost grid must be nonempty and unique")
    model.eval()
    counts_by_cost = {cost: defaultdict(int) for cost in edge_costs}
    teacher = defaultdict(int)
    set_teacher = defaultdict(int)
    with torch.no_grad():
        for batch in loader:
            logits = model(move_to_device(batch, device)).cpu().numpy()
            for row, item in enumerate(batch["metadata"]):
                n_source = len(item["candidate_ids_src"])
                window_logits = logits[row, :n_source]
                by_parent: dict[int, list[int]] = defaultdict(list)
                positives = np.flatnonzero((item["labels"] >= 0) & (item["labels"] < n_source))
                for child in positives:
                    by_parent[int(item["labels"][child])].append(int(child))
                divisions = [children for children in by_parent.values() if len(children) >= 2]
                for key, value in item["teacher_stats"].items():
                    teacher[key] += value
                for key, value in item["set_teacher_stats"].items():
                    set_teacher[key] += value
                for cost in edge_costs:
                    choice, decode_stats = decode_daughter_sets(
                        window_logits,
                        item["daughter_sets"],
                        item["candidate_ids_src"],
                        item["candidate_ids_tgt"],
                        edge_cost=cost,
                        maximum_decode_sets_per_mother=maximum_decode_sets_per_mother,
                        time_limit_seconds=time_limit_seconds,
                        relative_gap=relative_gap,
                    )
                    counts = counts_by_cost[cost]
                    for key, value in selection_counts(choice, item["labels"], n_source).items():
                        counts[key] += value
                    for key, value in decode_stats.items():
                        counts[key] += value
                    counts["known_division_parents"] += len(divisions)
                    counts["recovered_division_parents"] += sum(
                        all(choice[child] == item["labels"][child] for child in children)
                        for children in divisions
                    )
    reports = []
    for cost in edge_costs:
        counts = counts_by_cost[cost]
        report = dict(counts)
        for key in (
            "known_edges",
            "known_null",
            "true_positive_edges",
            "observable_false_edges",
            "selected_edges",
            "correct_null",
            "known_division_parents",
            "recovered_division_parents",
        ):
            report.setdefault(key, 0)
        report["teacher"] = dict(teacher)
        report["set_teacher"] = dict(set_teacher)
        report["known_edge_recall"] = counts["true_positive_edges"] / max(counts["known_edges"], 1)
        report["observable_false_edge_rate"] = counts["observable_false_edges"] / max(
            counts["known_edges"] + counts["known_null"], 1
        )
        report["known_division_recall"] = counts["recovered_division_parents"] / max(
            counts["known_division_parents"], 1
        )
        report["selection_accuracy"] = (
            counts["true_positive_edges"] + counts["correct_null"]
        ) / max(counts["known_edges"] + counts["known_null"], 1)
        report["edge_cost"] = cost
        report["structural_violations"] = 0
        reports.append(report)
    return reports


def evaluate_selection(
    model: Any,
    loader: Any,
    device: Any,
    edge_cost: float,
    maximum_decode_sets_per_mother: int,
    time_limit_seconds: float,
    relative_gap: float,
) -> dict[str, Any]:
    return evaluate_costs(
        model,
        loader,
        device,
        [edge_cost],
        maximum_decode_sets_per_mother,
        time_limit_seconds,
        relative_gap,
    )[0]


def choose_edge_cost(
    model: Any,
    loader: Any,
    device: Any,
    options: list[float],
    maximum_decode_sets_per_mother: int,
    time_limit_seconds: float,
    relative_gap: float,
) -> tuple[float, dict[str, Any]]:
    reports = evaluate_costs(
        model,
        loader,
        device,
        options,
        maximum_decode_sets_per_mother,
        time_limit_seconds,
        relative_gap,
    )
    best = max(
        reports,
        key=lambda row: (
            row["selection_accuracy"],
            -row["observable_false_edge_rate"],
            -row["edge_cost"],
        ),
    )
    return float(best["edge_cost"]), {"selected": best, "grid": reports}


def control_window_counts(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    matrix = np.asarray(probabilities, dtype=bool)
    n_source, n_target = matrix.shape
    if len(labels) != n_target:
        raise ValueError("control predictions and labels differ")
    known_parent = (labels >= 0) & (labels < n_source)
    known_null = labels == n_source
    true_positive = sum(
        bool(matrix[int(labels[child]), child]) for child in np.flatnonzero(known_parent)
    )
    false_edges = 0
    for child in np.flatnonzero(known_parent | known_null):
        false_edges += int(matrix[:, child].sum())
        if known_parent[child]:
            false_edges -= int(bool(matrix[int(labels[child]), child]))
    by_parent: dict[int, list[int]] = defaultdict(list)
    for child in np.flatnonzero(known_parent):
        by_parent[int(labels[child])].append(int(child))
    divisions = [children for children in by_parent.values() if len(children) >= 2]
    recovered = sum(
        all(bool(matrix[int(labels[child]), child]) for child in children) for children in divisions
    )
    return {
        "known_edges": int(known_parent.sum()),
        "known_null": int(known_null.sum()),
        "true_positive_edges": int(true_positive),
        "observable_false_edges": int(false_edges),
        "selected_edges": int(matrix.sum()),
        "known_division_parents": len(divisions),
        "recovered_division_parents": int(recovered),
    }


def resolve_control_models(config: dict[str, Any], device: Any) -> dict[int, Any]:
    import torch

    control_cfg = config["model"]["control"]
    slug = str(control_cfg["source_kernel"]).split("/", 1)[-1]
    manifest_path = resolve_unique(
        [
            path
            for path in Path("/kaggle/input/notebooks").rglob("model_manifest.json")
            if slug in path.as_posix()
        ],
        "exp016 train model manifest",
    )
    if file_sha256(manifest_path) != control_cfg["source_manifest_sha256"]:
        raise RuntimeError("saved exp016 manifest SHA changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "exp016_frozen_image_encoder":
        raise RuntimeError("wrong saved control experiment")
    if manifest.get("cache_identity_sha256") != config["data"]["cache"]["identity_sha256"]:
        raise RuntimeError("control cache identity changed")
    model_source_name = "repo/src/biohub_tracking/models/simple_node_transformer.py"
    model_source = resolve_unique(
        [
            path
            for path in Path("/kaggle/input").rglob("simple_node_transformer.py")
            if model_source_name in path.as_posix()
            and "biohub-tracking-support-pack-50ep-v1" in path.as_posix()
        ],
        "public model source",
    )
    if file_sha256(model_source) != control_cfg["model_source_sha256"]:
        raise RuntimeError("public control model source SHA changed")
    sys.path.insert(0, str(model_source.parents[2]))
    transformer_class = importlib.import_module("biohub_tracking.models").SimpleNodeTransformer
    controls = {}
    for record in manifest["models"]:
        fold = int(record["fold"])
        path = manifest_path.parent / str(record["path"])
        if file_sha256(path) != record["file_sha256"]:
            raise RuntimeError({"control_model_sha_mismatch": fold})
        control = transformer_class(
            feat_dim=64,
            hidden_dim=128,
            n_heads=4,
            n_blocks=4,
            dropout=0.3,
            pair_chunk_size=32,
        ).to(device)
        saved = torch.load(path, map_location=device, weights_only=True)
        if (
            saved.get("experiment") != "exp016_frozen_image_encoder"
            or int(saved.get("fold", -1)) != fold
        ):
            raise RuntimeError({"wrong_control_checkpoint": fold})
        control.load_state_dict(saved["state_dict"], strict=True)
        control.eval()
        controls[fold] = control
    if set(controls) != {0, 1}:
        raise RuntimeError("expected two saved control models")
    return controls


def evaluate_control(model: Any, loader: Any, device: Any, threshold: float) -> dict[str, Any]:
    import torch

    counts = defaultdict(int)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            gpu_batch = move_to_device(batch, device)
            model_source_coords = torch.zeros_like(gpu_batch["coords_src"])
            model_target_coords = torch.zeros_like(gpu_batch["coords_tgt"])
            for row, item in enumerate(batch["metadata"]):
                for side, destination in (
                    ("src", model_source_coords),
                    ("tgt", model_target_coords),
                ):
                    grid = np.asarray(item[f"coords_{side}_grid"], dtype=np.float32)
                    voxel = grid * np.array([1, 4, 4], dtype=np.float32)
                    destination[row, : len(grid)] = torch.from_numpy(voxel).to(device)
            logits = model(
                gpu_batch["features_src"][..., :64],
                gpu_batch["features_tgt"][..., :64],
                model_source_coords,
                model_target_coords,
                gpu_batch["source_mask"],
                gpu_batch["target_mask"],
            )
            for row, item in enumerate(batch["metadata"]):
                n_source = len(item["candidate_ids_src"])
                n_target = len(item["candidate_ids_tgt"])
                probs = torch.softmax(logits[row, :n_source, :n_target].float(), dim=0)
                selected = (probs > threshold).cpu().numpy()
                for key, value in control_window_counts(selected, item["labels"]).items():
                    counts[key] += value
    report = dict(counts)
    for key in (
        "known_edges",
        "known_null",
        "true_positive_edges",
        "observable_false_edges",
        "selected_edges",
        "known_division_parents",
        "recovered_division_parents",
    ):
        report.setdefault(key, 0)
    report["known_edge_recall"] = counts["true_positive_edges"] / max(counts["known_edges"], 1)
    report["observable_false_edge_rate"] = counts["observable_false_edges"] / max(
        counts["known_edges"] + counts["known_null"], 1
    )
    report["known_division_recall"] = counts["recovered_division_parents"] / max(
        counts["known_division_parents"], 1
    )
    report["edge_threshold"] = threshold
    report["stage"] = "exp016_saved_primary_tracker_before_ilp_and_repair"
    return report


def assess_early_gate(config: dict[str, Any], folds: list[dict[str, Any]]) -> dict[str, Any]:
    gate_cfg = config["validation"]["early_gate"]
    expected_folds = {
        (int(row["fold"]), str(row["evaluation_embryo"]))
        for row in config["validation"].get("outer_folds", [])
    }
    actual_folds = {(int(row["fold"]), str(row["evaluation_embryo"])) for row in folds}
    if expected_folds and actual_folds != expected_folds:
        raise RuntimeError("two-frame gate fold coverage mismatch")
    if not folds or len(actual_folds) != len(folds):
        raise RuntimeError("two-frame gate needs distinct folds")
    rows = []
    for fold in folds:
        direct = fold["outer"]
        control = fold["control_two_frame"]
        checks = {
            "same_supervised_children": (
                direct["known_edges"] == control["known_edges"]
                and direct["known_null"] == control["known_null"]
                and direct["known_division_parents"] == control["known_division_parents"]
            )
            if gate_cfg.get("require_same_unit_control", True)
            else True,
            "structural_violations": direct["structural_violations"]
            <= gate_cfg["maximum_structural_violations"],
            "known_division_evidence": direct["known_division_parents"]
            >= gate_cfg["minimum_known_division_parents"],
            "known_edge_recall": direct["known_edge_recall"]
            >= control["known_edge_recall"] - gate_cfg["maximum_known_edge_recall_drop"],
            "observable_false_edge_rate": direct["observable_false_edge_rate"]
            <= control["observable_false_edge_rate"]
            + gate_cfg["maximum_observable_false_edge_rate_increase"],
            "known_division_recall": direct["known_division_recall"]
            >= control["known_division_recall"] + gate_cfg["minimum_division_recall_delta"],
        }
        rows.append(
            {
                "fold": fold["fold"],
                "evaluation_embryo": fold["evaluation_embryo"],
                "checks": checks,
                "pass": all(checks.values()),
                "direct": direct,
                "control": control,
            }
        )
    return {
        "experiment": EXPERIMENT,
        "same_cache_and_teacher": True,
        "control_stage": "saved_exp016_primary_tracker_before_ilp_and_repair",
        "folds": rows,
        "pass": all(row["pass"] for row in rows),
        "official_graph_score_measured": False,
    }


def audit_candidate_sets(
    paths: list[Path],
    annotations: dict[str, dict[str, Any]],
    checkpoint_sha: str,
    matching_um: float,
    candidate_cfg: dict[str, Any],
) -> dict[str, Any]:
    by_embryo: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for index, path in enumerate(paths):
        item = make_example(
            path, annotations[path.parent.name], checkpoint_sha, matching_um, candidate_cfg
        )
        embryo = path.parent.name.split("_", 1)[0]
        row = by_embryo[embryo]
        row["windows"] += 1
        row["maximum_sets_in_window"] = max(
            row["maximum_sets_in_window"], item["set_teacher_stats"]["candidate_sets"]
        )
        row["maximum_sets_per_mother"] = max(
            row["maximum_sets_per_mother"], max(map(len, item["daughter_sets"]))
        )
        for key, value in item["teacher_stats"].items():
            row[key] += int(value)
        for key, value in item["set_teacher_stats"].items():
            row[key] += int(value)
        if (index + 1) % 2000 == 0:
            print(f"CANDIDATE_AUDIT_WINDOWS {index + 1}/{len(paths)}", flush=True)
    report = {
        "experiment": EXPERIMENT,
        "candidate_config": candidate_cfg,
        "by_embryo": {
            key: {name: int(count) for name, count in value.items()}
            for key, value in by_embryo.items()
        },
        "window_count": len(paths),
    }
    for embryo, row in report["by_embryo"].items():
        if row.get("supervised_positive_parents", 0) == 0:
            raise RuntimeError({"no_safe_positive_set_teacher": embryo})
        if row.get("known_two_daughter_set_recovered", 0) == 0:
            raise RuntimeError({"no_known_two_daughter_candidate_set": embryo})
    return report


def preflight_model_and_decoder(config: dict[str, Any]) -> None:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config["model"]).to(device)
    batch = {
        "features_src": torch.zeros((1, 1, 65), device=device),
        "features_tgt": torch.zeros((1, 2, 65), device=device),
        "coords_src": torch.zeros((1, 1, 3), device=device),
        "coords_tgt": torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device=device),
        "source_mask": torch.ones((1, 1), dtype=torch.bool, device=device),
        "target_mask": torch.ones((1, 2), dtype=torch.bool, device=device),
        "set_children": torch.tensor([[[[-1, -1], [0, -1], [1, -1], [0, 1]]]], device=device),
        "set_mask": torch.ones((1, 1, 4), dtype=torch.bool, device=device),
    }
    logits = model(batch)
    allowed = torch.tensor([[[False, True, False, True]]], device=device)
    supervised = torch.tensor([[True]], device=device)
    loss = partial_set_loss(logits, allowed, supervised)
    if logits.shape != (1, 1, 4) or not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise RuntimeError("daughter-set model preflight failed")
    loss.backward()
    if not all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    ):
        raise RuntimeError("daughter-set gradient preflight failed")
    chosen, _ = decode_daughter_sets(
        logits[0].detach().cpu().numpy(),
        [[(), (0,), (1,), (0, 1)]],
        np.array([0]),
        np.array([1, 2]),
        edge_cost=0.0,
        maximum_decode_sets_per_mother=4,
        time_limit_seconds=float(config["model"]["decoding"]["time_limit_seconds_per_window"]),
        relative_gap=float(config["model"]["decoding"]["relative_gap"]),
    )
    if len(chosen) != 2:
        raise RuntimeError("daughter-set decoder preflight failed")
    print("MODEL_AND_DECODER_PREFLIGHT_PASS", flush=True)


# %% [markdown]
# ## 6. Kaggle setup, resource gate, and full training


# %%
def verified_resume_file(root: Path, relative_path: str, expected_sha256: str) -> Path:
    source = (root / relative_path).resolve()
    if not source.is_relative_to(root.resolve()) or not source.is_file():
        raise FileNotFoundError(f"resume file unavailable: {relative_path}")
    actual_sha256 = file_sha256(source)
    if actual_sha256 != expected_sha256:
        raise RuntimeError({"resume_sha256_mismatch": relative_path, "actual": actual_sha256})
    return source


def project_remaining_runtime(
    *,
    benchmark_seconds: float,
    benchmark_windows: int,
    gradient_windows: int,
    epochs: int,
    decode_benchmark_seconds: float,
    decode_benchmark_windows: int,
    internal_windows: int,
    outer_windows: int,
    edge_cost_count: int,
    remaining_folds: int,
    multiplier: float,
) -> dict[str, float]:
    if (
        min(
            benchmark_windows,
            gradient_windows,
            epochs,
            decode_benchmark_windows,
            internal_windows,
            outer_windows,
            edge_cost_count,
            remaining_folds,
        )
        <= 0
        or multiplier < 1
    ):
        raise ValueError("invalid runtime projection inputs")
    train_seconds = (
        benchmark_seconds / benchmark_windows * gradient_windows * epochs * remaining_folds
    )
    decode_windows = internal_windows * epochs * edge_cost_count + outer_windows
    decode_seconds = (
        decode_benchmark_seconds / decode_benchmark_windows * decode_windows * remaining_folds
    )
    return {
        "train_seconds": train_seconds,
        "decode_seconds": decode_seconds,
        "total_seconds": (train_seconds + decode_seconds) * multiplier,
    }


def load_resumed_fold(
    root: Path,
    model_dir: Path,
    resume: dict[str, Any],
    outer_folds: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_path = verified_resume_file(
        root, resume["fold_summary_file"], resume["fold_summary_sha256"]
    )
    model_path = verified_resume_file(root, resume["model_file"], resume["model_sha256"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fold = int(resume["fold"])
    matching = [row for row in outer_folds if int(row["fold"]) == fold]
    if (
        len(matching) != 1
        or summary["fold"] != fold
        or summary["evaluation_embryo"] != matching[0]["evaluation_embryo"]
        or summary["model_sha256"] != resume["model_sha256"]
        or summary["model_file"] != model_path.name
    ):
        raise RuntimeError("resume fold summary does not match the configuration")
    target = model_dir / summary["model_file"]
    shutil.copy2(model_path, target)
    if file_sha256(target) != resume["model_sha256"]:
        raise RuntimeError("copied resume model hash mismatch")
    return summary


def restore_resume_files_from_dataset(
    root: Path,
    resume: dict[str, Any],
    input_root: Path = Path("/kaggle/input"),
) -> None:
    dataset_source = str(resume["dataset_source"])
    if dataset_source.count("/") != 1:
        raise ValueError("invalid resume dataset source")
    owner, slug = dataset_source.split("/")
    dataset_root = resolve_unique(
        [input_root / "datasets" / owner / slug, input_root / slug],
        "resume dataset",
    )
    for path_key, hash_key in (
        ("audit_file", "audit_sha256"),
        ("fold_summary_file", "fold_summary_sha256"),
        ("model_file", "model_sha256"),
    ):
        relative = Path(str(resume[path_key]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe resume path")
        source_file = dataset_root / relative.name
        if not source_file.is_file() or file_sha256(source_file) != resume[hash_key]:
            raise RuntimeError({"resume_dataset_file_mismatch": source_file.name})
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
        if file_sha256(target) != resume[hash_key]:
            raise RuntimeError({"copied_resume_file_mismatch": target.name})
    print("RESUME_DATASET_FILES_VERIFIED", dataset_source, flush=True)


def main() -> None:
    import torch
    from scipy.optimize import milp

    if not callable(milp):
        raise RuntimeError("scipy.optimize.milp is unavailable")
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("The authoritative full run must execute on Kaggle")
    root = Path.cwd()
    started = time.perf_counter()
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    if config["experiment"]["name"] != EXPERIMENT:
        raise RuntimeError("wrong experiment config")
    if config["model"]["training"]["active_variants"] != ["mother_daughter_set_selection"]:
        raise RuntimeError("unexpected active variants")
    if config["model"]["training"]["control_retrain"] is not False:
        raise RuntimeError("control retraining is prohibited")
    preflight_model_and_decoder(config)
    ensure_geff_runtime_dependencies()
    cache_root, cache_summary = resolve_cache_root(config)
    train_dir = resolve_train_dir()
    paths = sorted(cache_root.glob("*/*.npz"))
    samples = sorted({path.parent.name for path in paths})
    voxel_scale = tuple(
        float(value) for value in config["data"]["annotation"]["voxel_scale_zyx_um"]
    )
    annotations = {
        sample: load_annotation(train_dir / f"{sample}.geff", voxel_scale) for sample in samples
    }
    paths = [
        path
        for path in paths
        if all(
            frame in annotations[path.parent.name]["frames"]
            and annotations[path.parent.name]["frames"][frame]
            for frame in map(int, path.stem.split("_"))
        )
    ]
    if not paths:
        raise RuntimeError("no windows with annotation in both frames")
    checkpoint_sha = str(config["data"]["cache"]["primary_checkpoint_sha256"])
    matching_um = float(config["data"]["teacher"]["max_matching_distance_um"])
    train_cfg = config["model"]["training"]
    resume = train_cfg.get("resume")
    if resume and resume.get("dataset_source"):
        restore_resume_files_from_dataset(root, resume)
    audit_path = root / "candidate_teacher_audit.json"
    if resume:
        previous_audit = verified_resume_file(root, resume["audit_file"], resume["audit_sha256"])
        candidate_audit = json.loads(previous_audit.read_text(encoding="utf-8"))
        if (
            candidate_audit["window_count"] != len(paths)
            or candidate_audit["candidate_config"] != config["model"]["candidate_sets"]
        ):
            raise RuntimeError("resumed candidate audit does not match the input")
        shutil.copy2(previous_audit, audit_path)
        print("CANDIDATE_TEACHER_AUDIT_REUSED", resume["audit_sha256"], flush=True)
    else:
        candidate_audit = audit_candidate_sets(
            paths, annotations, checkpoint_sha, matching_um, config["model"]["candidate_sets"]
        )
        audit_path.write_text(
            json.dumps(candidate_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print("CANDIDATE_TEACHER_AUDIT", json.dumps(candidate_audit, sort_keys=True), flush=True)
    print("CANDIDATE_TEACHER_AUDIT_SHA256", file_sha256(audit_path), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU training is required")
    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)
    folds = []
    outer_folds = config["validation"]["outer_folds"]
    if resume:
        resumed_summary = load_resumed_fold(root, model_dir, resume, outer_folds)
        folds.append(resumed_summary)
        print(
            "FOLD_RESUMED",
            json.dumps(
                {
                    "fold": resumed_summary["fold"],
                    "summary_sha256": resume["fold_summary_sha256"],
                    "model_sha256": resume["model_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    control_models = resolve_control_models(config, device)
    for fold_cfg in outer_folds:
        fold = int(fold_cfg["fold"])
        if any(int(row["fold"]) == fold for row in folds):
            continue
        fold_seed = int(config["validation"]["seed"]) + fold
        split = split_samples(
            samples,
            fold_cfg["train_embryo"],
            fold_cfg["evaluation_embryo"],
            int(config["validation"]["internal_split_seed"]),
            float(config["validation"]["internal_selection_fraction"]),
        )
        by_split = {
            name: [path for path in paths if path.parent.name in set(names)]
            for name, names in split.items()
        }
        if not all(by_split.values()):
            raise RuntimeError("empty split")
        seed_everything(fold_seed)
        model = make_model(config["model"]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        benchmark_paths = by_split["gradient"][: int(train_cfg["benchmark_windows_per_fold"])]
        benchmark_loader = make_loader(
            benchmark_paths,
            annotations,
            checkpoint_sha,
            matching_um,
            int(train_cfg["batch_size"]),
            True,
            fold_seed,
            int(config["runtime"]["num_workers"]),
            config["model"]["candidate_sets"],
        )
        benchmark_start = time.perf_counter()
        benchmark = run_epoch(model, benchmark_loader, optimizer, device)
        benchmark_seconds = time.perf_counter() - benchmark_start
        decode_benchmark_paths = sorted(
            by_split["gradient"], key=lambda path: path.stat().st_size, reverse=True
        )[:8]
        decode_benchmark_loader = make_loader(
            decode_benchmark_paths,
            annotations,
            checkpoint_sha,
            matching_um,
            int(train_cfg["batch_size"]),
            False,
            fold_seed,
            int(config["runtime"]["num_workers"]),
            config["model"]["candidate_sets"],
        )
        decode_benchmark_start = time.perf_counter()
        decode_benchmark = evaluate_costs(
            model,
            decode_benchmark_loader,
            device,
            [0.0],
            int(config["model"]["candidate_sets"]["maximum_decode_sets_per_mother"]),
            float(config["model"]["decoding"]["time_limit_seconds_per_window"]),
            float(config["model"]["decoding"]["relative_gap"]),
        )[0]
        decode_benchmark_seconds = time.perf_counter() - decode_benchmark_start
        runtime_projection = project_remaining_runtime(
            benchmark_seconds=benchmark_seconds,
            benchmark_windows=len(benchmark_paths),
            gradient_windows=len(by_split["gradient"]),
            epochs=int(train_cfg["epochs"]),
            decode_benchmark_seconds=decode_benchmark_seconds,
            decode_benchmark_windows=len(decode_benchmark_paths),
            internal_windows=len(by_split["internal"]),
            outer_windows=len(by_split["outer"]),
            edge_cost_count=len(config["model"]["decoding"]["edge_cost_grid"]),
            remaining_folds=len(outer_folds) - len(folds),
            multiplier=float(train_cfg["runtime_projection_multiplier"]),
        )
        projection = runtime_projection["total_seconds"]
        print(
            "RUNTIME_PROJECTION",
            json.dumps(
                {
                    "fold": fold,
                    **runtime_projection,
                    "remaining_folds": len(outer_folds) - len(folds),
                    "benchmark_candidate_sets_total": decode_benchmark["candidate_sets"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if (
            time.perf_counter() - started + projection
            > float(train_cfg["runtime_gate_hours"]) * 3600
        ):
            raise RuntimeError({"training_runtime_gate_seconds": projection})
        seed_everything(fold_seed)
        model = make_model(config["model"]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        gradient_loader = make_loader(
            by_split["gradient"],
            annotations,
            checkpoint_sha,
            matching_um,
            int(train_cfg["batch_size"]),
            True,
            fold_seed,
            int(config["runtime"]["num_workers"]),
            config["model"]["candidate_sets"],
        )
        internal_loader = make_loader(
            by_split["internal"],
            annotations,
            checkpoint_sha,
            matching_um,
            int(train_cfg["batch_size"]),
            False,
            fold_seed,
            int(config["runtime"]["num_workers"]),
            config["model"]["candidate_sets"],
        )
        best_score = -float("inf")
        best_path = model_dir / f"fold_{fold}_best.pth"
        history = []
        for epoch in range(int(train_cfg["epochs"])):
            train_report = run_epoch(model, gradient_loader, optimizer, device)
            edge_cost, internal_report = choose_edge_cost(
                model,
                internal_loader,
                device,
                [float(value) for value in config["model"]["decoding"]["edge_cost_grid"]],
                int(config["model"]["candidate_sets"]["maximum_decode_sets_per_mother"]),
                float(config["model"]["decoding"]["time_limit_seconds_per_window"]),
                float(config["model"]["decoding"]["relative_gap"]),
            )
            score = float(internal_report["selected"]["selection_accuracy"])
            history.append(
                {
                    "epoch": epoch + 1,
                    "train": train_report,
                    "internal": internal_report,
                    "edge_cost": edge_cost,
                }
            )
            print(
                f"FOLD {fold} EPOCH {epoch + 1} loss={train_report['loss']:.6f} "
                f"internal_accuracy={score:.6f} edge_cost={edge_cost:.2f}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), best_path)
                selected_epoch = epoch + 1
                selected_edge_cost = edge_cost
            if time.perf_counter() - started > float(train_cfg["runtime_gate_hours"]) * 3600:
                raise RuntimeError("full training runtime gate exceeded")
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        outer_loader = make_loader(
            by_split["outer"],
            annotations,
            checkpoint_sha,
            matching_um,
            int(train_cfg["batch_size"]),
            False,
            fold_seed,
            int(config["runtime"]["num_workers"]),
            config["model"]["candidate_sets"],
        )
        outer_report = evaluate_selection(
            model,
            outer_loader,
            device,
            selected_edge_cost,
            int(config["model"]["candidate_sets"]["maximum_decode_sets_per_mother"]),
            float(config["model"]["decoding"]["time_limit_seconds_per_window"]),
            float(config["model"]["decoding"]["relative_gap"]),
        )
        control_report = evaluate_control(
            control_models[fold],
            outer_loader,
            device,
            float(config["model"]["control"]["edge_threshold"]),
        )
        fold_summary = {
            "fold": fold,
            "evaluation_embryo": fold_cfg["evaluation_embryo"],
            "selected_epoch": selected_epoch,
            "edge_cost": selected_edge_cost,
            "model_sha256": file_sha256(best_path),
            "model_file": best_path.name,
            "benchmark_seconds": benchmark_seconds,
            "benchmark": benchmark,
            "decode_benchmark_seconds": decode_benchmark_seconds,
            "decode_benchmark": decode_benchmark,
            "history": history,
            "outer": outer_report,
            "control_two_frame": control_report,
            "split_samples": {key: len(value) for key, value in split.items()},
            "split_windows": {key: len(value) for key, value in by_split.items()},
        }
        (root / f"fold_{fold}_summary.json").write_text(
            json.dumps(fold_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        folds.append(fold_summary)
        print(f"FOLD {fold} OUTER {json.dumps(outer_report, sort_keys=True)}", flush=True)
        print(f"FOLD {fold} CONTROL {json.dumps(control_report, sort_keys=True)}", flush=True)
    early_gate = assess_early_gate(config, folds)
    early_gate_path = root / "early_gate.json"
    early_gate_path.write_text(
        json.dumps(early_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"EARLY_GATE {json.dumps(early_gate, sort_keys=True)}", flush=True)
    print(f"EARLY_GATE_SHA256 {file_sha256(early_gate_path)}", flush=True)
    manifest = {
        "experiment": EXPERIMENT,
        "cache_summary_sha256": cache_summary["summary_sha256"],
        "cache_identity_sha256": cache_summary["cache_identity_sha256"],
        "candidate_teacher_audit_sha256": file_sha256(audit_path),
        "checkpoint_sha256": checkpoint_sha,
        "early_gate_sha256": file_sha256(early_gate_path),
        "early_gate_passed": bool(early_gate["pass"]),
        "folds": [
            {
                key: row[key]
                for key in (
                    "fold",
                    "evaluation_embryo",
                    "selected_epoch",
                    "edge_cost",
                    "model_file",
                    "model_sha256",
                )
            }
            for row in folds
        ],
        "notebook_runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    manifest_path = root / "model_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["status"] = "running"
    metrics.setdefault("evidence", {})["train_stage"] = {
        "cache_summary_sha256": cache_summary["summary_sha256"],
        "cache_identity_sha256": cache_summary["cache_identity_sha256"],
        "candidate_teacher_audit_sha256": file_sha256(audit_path),
        "model_manifest_sha256": file_sha256(manifest_path),
        "early_gate_sha256": file_sha256(early_gate_path),
        "early_gate_passed": bool(early_gate["pass"]),
        "model_count": len(folds),
        "notebook_runtime_seconds": manifest["notebook_runtime_seconds"],
        "peak_gpu_memory_bytes": manifest["peak_gpu_memory_bytes"],
        "folds": [
            {
                "fold": row["fold"],
                "evaluation_embryo": row["evaluation_embryo"],
                "model_sha256": row["model_sha256"],
                "outer": row["outer"],
                "control_two_frame": row["control_two_frame"],
            }
            for row in folds
        ],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"MODEL_MANIFEST_SHA256 {file_sha256(manifest_path)}", flush=True)
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
