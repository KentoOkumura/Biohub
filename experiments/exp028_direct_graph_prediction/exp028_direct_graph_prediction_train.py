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
# # Direct parent selection from fixed detections
#
# This notebook reads the saved two-frame image features and organizer GEFF
# annotations. Each daughter chooses a parent candidate or no candidate parent.
# Only daughters with an observed incoming annotation contribute to the loss.
# The run trains two embryo-held-out models and records two-frame diagnostics.

# %% [markdown]
# ## 1. Imports and deterministic helpers

# %%
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

EXPERIMENT = "exp028_direct_graph_prediction"
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
def make_model(config: dict[str, Any]) -> Any:
    import torch
    from torch import nn

    class DirectParentSelectionTransformer(nn.Module):
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
            self.query = nn.Linear(width, width)
            self.key = nn.Linear(width, width)
            self.null = nn.Linear(width, 1)
            self.relative = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, 1))
            self.width = width

        def forward(self, batch: dict[str, Any]) -> Any:
            source = batch["features_src"]
            target = batch["features_tgt"]
            source_mask = batch["source_mask"]
            target_mask = batch["target_mask"]
            source_coord = batch["coords_src"] / 20.0
            target_coord = batch["coords_tgt"] / 20.0
            source_tokens = (
                self.feature(source) + self.position(source_coord) + self.frame.weight[0]
            )
            target_tokens = (
                self.feature(target) + self.position(target_coord) + self.frame.weight[1]
            )
            tokens = torch.cat([source_tokens, target_tokens], dim=1)
            present = torch.cat([source_mask, target_mask], dim=1)
            encoded = self.encoder(tokens, src_key_padding_mask=~present)
            n_source = source.shape[1]
            parent = encoded[:, :n_source]
            child = encoded[:, n_source:]
            pair = torch.einsum("bth,bsh->bts", self.query(child), self.key(parent)) / math.sqrt(
                self.width
            )
            relative = (target_coord[:, :, None, :] - source_coord[:, None, :, :]).clamp(-10, 10)
            pair = pair + self.relative(relative).squeeze(-1)
            pair = pair.masked_fill(~source_mask[:, None, :], -1e9)
            return torch.cat([pair, self.null(child)], dim=2)

    return DirectParentSelectionTransformer()


def masked_selection_loss(logits: Any, labels: Any) -> Any:
    import torch.nn.functional as functional

    known = labels >= 0
    if not bool(known.any()):
        return logits.sum() * 0.0
    return functional.cross_entropy(logits[known], labels[known])


def decode_parent_choices(
    logits: np.ndarray,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(logits, dtype=np.float64)
    source = np.asarray(source_ids, dtype=np.int64)
    target = np.asarray(target_ids, dtype=np.int64)
    if scores.shape != (len(target), len(source) + 1):
        raise ValueError("parent-selection score shape mismatch")
    if len(set(source.tolist())) != len(source) or len(set(target.tolist())) != len(target):
        raise ValueError("duplicate candidate ids")
    if len(source) == 0:
        return np.full(len(target), -1, dtype=np.int64), np.zeros(len(target))
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    confidence = probabilities[:, : len(source)].max(axis=1)
    null_prob = probabilities[:, len(source)]
    margin = confidence - null_prob
    order = sorted(range(len(target)), key=lambda j: (-margin[j], int(target[j])))
    chosen = np.full(len(target), -1, dtype=np.int64)
    chosen_probability = np.zeros(len(target), dtype=np.float64)
    capacity = np.zeros(len(source), dtype=np.int64)
    for child in order:
        parents = sorted(
            range(len(source)), key=lambda i: (-probabilities[child, i], int(source[i]))
        )
        for parent in parents:
            probability = float(probabilities[child, parent])
            if probability < threshold or probability <= null_prob[child]:
                break
            if capacity[parent] < 2:
                chosen[child] = parent
                chosen_probability[child] = probability
                capacity[parent] += 1
                break
    if np.any(capacity > 2):
        raise RuntimeError("parent capacity violated")
    return chosen, chosen_probability


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
    ) -> None:
        self.paths = paths
        self.annotations = annotations
        self.checkpoint_sha = checkpoint_sha
        self.max_distance_um = max_distance_um

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        return make_example(
            path, self.annotations[path.parent.name], self.checkpoint_sha, self.max_distance_um
        )


def collate_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch_size = len(examples)
    source_count = max(len(item["candidate_ids_src"]) for item in examples)
    target_count = max(len(item["candidate_ids_tgt"]) for item in examples)
    result = {
        "features_src": torch.zeros(batch_size, source_count, 65),
        "features_tgt": torch.zeros(batch_size, target_count, 65),
        "coords_src": torch.zeros(batch_size, source_count, 3),
        "coords_tgt": torch.zeros(batch_size, target_count, 3),
        "source_mask": torch.zeros(batch_size, source_count, dtype=torch.bool),
        "target_mask": torch.zeros(batch_size, target_count, dtype=torch.bool),
        "labels": torch.full((batch_size, target_count), -1, dtype=torch.long),
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
) -> Any:
    import torch
    from torch.utils.data import DataLoader

    dataset = WindowDataset(paths, annotations, checkpoint_sha, matching_um)
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
    for batch in loader:
        batch = move_to_device(batch, device)
        labels = batch["labels"]
        with torch.set_grad_enabled(training):
            logits = model(batch)
            loss = masked_selection_loss(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        count = int((labels >= 0).sum().item())
        loss_sum += float(loss.detach().item()) * count
        supervised += count
    return {"loss": loss_sum / max(supervised, 1), "supervised_children": supervised}


def evaluate_thresholds(
    model: Any, loader: Any, device: Any, thresholds: list[float]
) -> list[dict[str, Any]]:
    import torch

    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("threshold grid must be nonempty and unique")
    model.eval()
    counts_by_threshold = {threshold: defaultdict(int) for threshold in thresholds}
    teacher = defaultdict(int)
    with torch.no_grad():
        for batch in loader:
            gpu_batch = move_to_device(batch, device)
            logits = model(gpu_batch).cpu().numpy()
            for batch_index, item in enumerate(batch["metadata"]):
                n_source = len(item["candidate_ids_src"])
                n_target = len(item["candidate_ids_tgt"])
                window_logits = np.concatenate(
                    [
                        logits[batch_index, :n_target, :n_source],
                        logits[batch_index, :n_target, batch["source_mask"].shape[1] :][:, :1],
                    ],
                    axis=1,
                )
                by_parent: dict[int, list[int]] = defaultdict(list)
                positives = np.flatnonzero((item["labels"] >= 0) & (item["labels"] < n_source))
                for child in positives:
                    by_parent[int(item["labels"][child])].append(int(child))
                divisions = [children for children in by_parent.values() if len(children) >= 2]
                for key, value in item["teacher_stats"].items():
                    teacher[key] += value
                for threshold in thresholds:
                    choice, _ = decode_parent_choices(
                        window_logits,
                        item["candidate_ids_src"],
                        item["candidate_ids_tgt"],
                        threshold=threshold,
                    )
                    counts = counts_by_threshold[threshold]
                    for key, value in selection_counts(choice, item["labels"], n_source).items():
                        counts[key] += value
                    counts["known_division_parents"] += len(divisions)
                    counts["recovered_division_parents"] += sum(
                        all(choice[child] == item["labels"][child] for child in children)
                        for children in divisions
                    )
    reports = []
    for threshold in thresholds:
        counts = counts_by_threshold[threshold]
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
        report["threshold"] = threshold
        report["structural_violations"] = 0
        reports.append(report)
    return reports


def evaluate_selection(model: Any, loader: Any, device: Any, threshold: float) -> dict[str, Any]:
    return evaluate_thresholds(model, loader, device, [threshold])[0]


def choose_threshold(
    model: Any, loader: Any, device: Any, options: list[float]
) -> tuple[float, dict[str, Any]]:
    reports = evaluate_thresholds(model, loader, device, options)
    best = max(
        reports,
        key=lambda row: (
            row["selection_accuracy"],
            -row["observable_false_edge_rate"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    return float(best["threshold"]), {"selected": best, "grid": reports}


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


# %% [markdown]
# ## 6. Kaggle setup, resource gate, and full training


# %%
def main() -> None:
    import torch

    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("The authoritative full run must execute on Kaggle")
    root = Path.cwd()
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    if config["experiment"]["name"] != EXPERIMENT:
        raise RuntimeError("wrong experiment config")
    if config["model"]["training"]["active_variants"] != ["direct_parent_selection"]:
        raise RuntimeError("unexpected active variants")
    if config["model"]["training"]["control_retrain"] is not False:
        raise RuntimeError("control retraining is prohibited")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU training is required")
    started = time.perf_counter()
    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)
    folds = []
    control_models = resolve_control_models(config, device)
    for fold_cfg in config["validation"]["outer_folds"]:
        fold = int(fold_cfg["fold"])
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
        )
        benchmark_start = time.perf_counter()
        benchmark = run_epoch(model, benchmark_loader, optimizer, device)
        benchmark_seconds = time.perf_counter() - benchmark_start
        projection = benchmark_seconds / len(benchmark_paths) * len(by_split["gradient"])
        projection *= int(train_cfg["epochs"]) * len(config["validation"]["outer_folds"])
        projection *= float(train_cfg["runtime_projection_multiplier"])
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
        )
        best_score = -float("inf")
        best_path = model_dir / f"fold_{fold}_best.pth"
        history = []
        for epoch in range(int(train_cfg["epochs"])):
            train_report = run_epoch(model, gradient_loader, optimizer, device)
            threshold, internal_report = choose_threshold(
                model,
                internal_loader,
                device,
                [float(value) for value in config["model"]["decoding"]["threshold_grid"]],
            )
            score = float(internal_report["selected"]["selection_accuracy"])
            history.append(
                {
                    "epoch": epoch + 1,
                    "train": train_report,
                    "internal": internal_report,
                    "threshold": threshold,
                }
            )
            print(
                f"FOLD {fold} EPOCH {epoch + 1} loss={train_report['loss']:.6f} "
                f"internal_accuracy={score:.6f} threshold={threshold:.2f}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), best_path)
                selected_epoch = epoch + 1
                selected_threshold = threshold
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
        )
        outer_report = evaluate_selection(model, outer_loader, device, selected_threshold)
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
            "threshold": selected_threshold,
            "model_sha256": file_sha256(best_path),
            "model_file": best_path.name,
            "benchmark_seconds": benchmark_seconds,
            "benchmark": benchmark,
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
                    "threshold",
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
