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
# # exp026: equal-budget division triplet candidates
#
# Compare distance ranking against frozen image and Kalman ranking.
# Both variants keep identical observed centers and the same triplet budget,
# using only predictions available before ground-truth evaluation.
# Ground truth is used only for training-side budget selection and evaluation.
#
# ## Contents
# 1. Imports and pinned inputs
# 2. Frozen image scores and saved motion states
# 3. Geometry, equal-budget ranking and GT-only evaluation
# 4. Kaggle execution and reproducible candidate files

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
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

EXPERIMENT = "exp026_division_candidate_budget"


# %% [markdown]
# ## 2. Frozen image scores and saved motion states

# %%


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
# ### State projection from exp025


# %%
@dataclass
class State:
    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    history: int


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


# %% [markdown]
# ### Verified fixed window cache

# %%
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


# %% [markdown]
# ### Frozen tracker checkpoint and all-pair scores


# %%
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


# %% [markdown]
# ### Cache identity and Kaggle input discovery


# %%
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


# %% [markdown]
# ### Public model source and checkpoint resolution


# %%
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
# ## 3. Geometry, equal-budget ranking and GT-only evaluation

# %%


# %% [markdown]
# ### Organizer annotation for evaluation only


# %%
def load_gt(
    path: Path, scale_zyx: np.ndarray
) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, tuple[int, ...]], str, list[str]]:
    import tracksdata as td

    graph = td.graph.IndexedRXGraph.from_geff(path)
    if isinstance(graph, tuple):
        graph = graph[0]
    frame_nodes: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    raw_nodes = []
    node_times = {}
    node_columns = list(graph.node_attrs().columns)
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        frame = int(row["t"])
        raw = np.asarray([row["z"], row["y"], row["x"]], dtype=np.float64)
        if node_id in node_times or not np.isfinite(raw).all():
            raise ValueError(f"duplicate or invalid GT node: {path} {node_id}")
        node_times[node_id] = frame
        frame_nodes[frame][node_id] = raw * scale_zyx
        raw_nodes.append([node_id, frame, *map(float, raw)])
    outgoing: dict[int, list[int]] = defaultdict(list)
    raw_edges = []
    for row in graph.edge_attrs().iter_rows(named=True):
        source, target = int(row["source_id"]), int(row["target_id"])
        if source not in node_times or target not in node_times:
            raise ValueError(f"dangling GT edge: {path}")
        outgoing[source].append(target)
        raw_edges.append([source, target])
    canonical = {
        "nodes": sorted(raw_nodes, key=lambda row: row[0]),
        "edges": sorted(raw_edges),
    }
    return (
        frame_nodes,
        {source: tuple(sorted(targets)) for source, targets in outgoing.items()},
        json_sha(canonical),
        node_columns,
    )


def match_frame(
    gt_nodes: dict[int, np.ndarray],
    candidate_ids: np.ndarray,
    candidate_coords: np.ndarray,
    radius: float,
) -> dict[int, int]:
    if not gt_nodes or len(candidate_ids) == 0:
        return {}
    gt_ids = sorted(gt_nodes)
    gt_coords = np.asarray([gt_nodes[node_id] for node_id in gt_ids], dtype=np.float64)
    distances = np.linalg.norm(gt_coords[:, None, :] - candidate_coords[None, :, :], axis=2)
    feasible = distances <= radius
    gated = np.where(feasible, distances, 1.0e9)
    rows, columns = linear_sum_assignment(gated)
    return {
        gt_ids[int(row)]: int(candidate_ids[int(column)])
        for row, column in zip(rows, columns, strict=True)
        if distances[row, column] <= radius
    }


def true_divisions(
    frame: int, gt_by_frame: dict[int, dict[int, np.ndarray]], outgoing: dict[int, tuple[int, ...]]
) -> list[tuple[int, int, int]]:
    current = gt_by_frame.get(frame, {})
    following = gt_by_frame.get(frame + 1, {})
    return [
        (source, targets[0], targets[1])
        for source, targets in outgoing.items()
        if source in current
        and len(targets) == 2
        and all(target in following for target in targets)
    ]


# %% [markdown]
# ### Unordered triplets and equal-window budgets


# %%
def enumerate_triplets(
    source_ids: np.ndarray,
    source_xyz: np.ndarray,
    target_ids: np.ndarray,
    target_xyz: np.ndarray,
    parent_radius: float,
    sister_radius: float,
    parent_guard: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate unordered daughter pairs without consulting annotation."""
    if not len(source_ids) or len(target_ids) < 2:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float32)
    nearby = cKDTree(target_xyz).query_ball_point(source_xyz, parent_radius)
    records: list[tuple[int, int, int]] = []
    distances: list[float] = []
    for source_index in np.argsort(source_ids, kind="stable"):
        target_indices = sorted(nearby[source_index], key=lambda i: int(target_ids[i]))
        count = 0
        for first, second in combinations(target_indices, 2):
            if np.linalg.norm(target_xyz[first] - target_xyz[second]) > sister_radius:
                continue
            records.append(
                (int(source_ids[source_index]), int(target_ids[first]), int(target_ids[second]))
            )
            distances.append(
                float(
                    np.linalg.norm(source_xyz[source_index] - target_xyz[first])
                    + np.linalg.norm(source_xyz[source_index] - target_xyz[second])
                )
            )
            count += 1
        if count > parent_guard:
            raise ValueError(f"triplet guard exceeded for mother {int(source_ids[source_index])}")
    return np.asarray(records, dtype=np.int64).reshape(-1, 3), np.asarray(
        distances, dtype=np.float32
    )


def budget_indices(triplets: np.ndarray, costs: np.ndarray, fraction: float) -> np.ndarray:
    """Take the same ceiling of each window's candidate count for both ranks."""
    if not 0 < fraction <= 1 or costs.shape != (len(triplets),):
        raise ValueError("invalid budget or score length")
    if not np.isfinite(costs).all() or triplets.shape != (len(triplets), 3):
        raise ValueError("nonfinite costs or invalid triplets")
    count = math.ceil(fraction * len(triplets))
    order = np.lexsort((triplets[:, 2], triplets[:, 1], triplets[:, 0], costs))
    return order[:count]


def triplet_costs(
    arrays: dict[str, np.ndarray],
    triplets: np.ndarray,
    logits: Any,
    saved_states: dict[int, State],
    noise: dict,
    motion_weight: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    import torch

    source_ids = np.asarray(arrays["candidate_ids_src"], dtype=np.int64)
    target_ids = np.asarray(arrays["candidate_ids_tgt"], dtype=np.int64)
    probabilities = torch.softmax(logits[0].float(), dim=0).detach().cpu().numpy()
    if probabilities.shape != (len(source_ids), len(target_ids)):
        raise ValueError("frozen tracker score matrix shape mismatch")
    if not np.isfinite(probabilities).all():
        raise ValueError("nonfinite frozen tracker scores")
    source_index = {int(value): i for i, value in enumerate(source_ids)}
    target_index = {int(value): i for i, value in enumerate(target_ids)}
    missing_states = set(map(int, source_ids)) - set(saved_states)
    if missing_states:
        raise ValueError(f"exp025 motion states miss {len(missing_states)} source candidates")
    scale = float(noise["nll_scale"])
    if scale <= 0 or not math.isfinite(scale) or motion_weight < 0:
        raise ValueError("invalid exp025 motion calibration")
    edge_image = -np.log(np.clip(probabilities, 1e-8, 1.0))
    image = np.empty(len(triplets), dtype=np.float32)
    motion = np.zeros(len(triplets), dtype=np.float32)
    targets_by_mother: dict[int, set[int]] = defaultdict(set)
    for mother, first, second in triplets:
        targets_by_mother[int(mother)].update((int(first), int(second)))
    edge_motion = {}
    target_xyz = np.asarray(arrays["coords_tgt_physical"], dtype=np.float64)
    for mother, daughters in targets_by_mother.items():
        state = saved_states[mother]
        if state.history <= 1:
            continue
        forecast = predict(state, noise)
        for daughter in daughters:
            nll = innovation_cost(forecast, target_xyz[target_index[daughter]], noise)
            edge_motion[mother, daughter] = (nll - float(noise["nll_center"])) / scale
    historyless = 0
    for row, (mother, first, second) in enumerate(triplets):
        a, b, c = int(mother), int(first), int(second)
        i, j, k = source_index[a], target_index[b], target_index[c]
        image[row] = edge_image[i, j] + edge_image[i, k]
        if saved_states[a].history <= 1:
            historyless += 1
        else:
            motion[row] = motion_weight * (edge_motion[a, b] + edge_motion[a, c])
    if not np.isfinite(image).all() or not np.isfinite(motion).all():
        raise ValueError("nonfinite triplet scores")
    return image, motion, historyless


def read_exp025_output(input_root: Path, cfg: dict) -> tuple[Path, dict, dict, dict]:
    slug = cfg["data"]["motion"]["kernel_source"].split("/", 1)[-1]
    candidates = [
        path
        for path in artifact_paths(input_root, "execution_receipt.json")
        if slug in path.as_posix()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one completed exp025 output, found {candidates}")
    root = candidates[0].parent
    expected_receipt_sha = cfg["data"]["motion"].get("execution_receipt_sha256")
    if not expected_receipt_sha or file_sha(candidates[0]) != expected_receipt_sha:
        raise ValueError("exp025 completed execution receipt is not SHA-pinned")
    receipt = json.loads(candidates[0].read_text())
    if (
        receipt.get("experiment") != "exp025_kalman_hungarian_links"
        or receipt.get("sample_count") != 199
    ):
        raise ValueError("exp025 execution receipt is incomplete")
    predictions = json.loads((root / "predictions.json").read_text())
    if json_sha(predictions) != receipt.get("oof_prediction_sha"):
        raise ValueError("exp025 prediction manifest SHA mismatch")
    by_sample = {}
    for item in predictions:
        if item["variant"] != "kalman":
            continue
        sample = item["sample"]
        if sample in by_sample:
            raise ValueError("duplicate exp025 sample")
        relative = Path(item["states"]).relative_to("/kaggle/working/diagnostic")
        path = require_sha(root / relative, item["states_sha256"])
        graph_relative = Path(item["graph"]).relative_to("/kaggle/working/diagnostic")
        graph_path = require_sha(root / graph_relative, item["graph_sha256"])
        by_sample[sample] = {
            "path": path,
            "sha256": item["states_sha256"],
            "graph_path": graph_path,
            "graph_sha256": item["graph_sha256"],
        }
    if len(by_sample) != 199:
        raise ValueError("exp025 motion state coverage mismatch")
    calibrations = {}
    for fold in (0, 1):
        path = root / f"fold_{fold}" / "calibration.json"
        expected_calibration_sha = cfg["data"]["motion"]["calibration_sha256"].get(str(fold))
        if not expected_calibration_sha or file_sha(path) != expected_calibration_sha:
            raise ValueError(f"exp025 fold {fold} calibration is not SHA-pinned")
        item = json.loads(path.read_text())
        noise = json.loads((root / f"fold_{fold}" / "noise.json").read_text())
        if item["noise_sha256"] != json_sha(noise) or item["fold"] != fold:
            raise ValueError("exp025 calibration/noise mismatch")
        calibrations[fold] = {
            "selected": item["selected"],
            "noise": noise,
            "calibration_sha256": file_sha(path),
        }
    return root, receipt, by_sample, calibrations


def load_motion_states(path: Path) -> dict[int, State]:
    with np.load(path, allow_pickle=False) as saved:
        ids = np.asarray(saved["node_ids"], dtype=np.int64)
        means = np.asarray(saved["means"], dtype=np.float64)
        covariance = np.asarray(saved["covariances"], dtype=np.float64)
        tracks = np.asarray(saved["track_ids"], dtype=np.int64)
        histories = np.asarray(saved["histories"], dtype=np.int64)
    if (
        len(set(map(int, ids))) != len(ids)
        or means.shape != (len(ids), 6)
        or covariance.shape != (len(ids), 6, 6)
        or not np.isfinite(means).all()
        or not np.isfinite(covariance).all()
        or np.any(histories < 1)
        or np.any(tracks < 0)
    ):
        raise ValueError("invalid exp025 motion states")
    return {
        int(ids[i]): State(int(tracks[i]), means[i], covariance[i], int(histories[i]))
        for i in range(len(ids))
    }


def choose_training_budget(
    rows: list[dict], fractions: list[float], threshold: float, train_embryo: str
) -> float:
    def recovered(fraction: float) -> int:
        return sum(
            int(row["recovered"])
            for row in rows
            if row["embryo"] == train_embryo
            and row["method"] == "distance"
            and row["fraction"] == fraction
        )

    full = recovered(1.0)
    if full == 0:
        raise ValueError("no training-side known divisions in full geometry")
    return next(
        (fraction for fraction in fractions if recovered(fraction) >= threshold * full), 1.0
    )


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


# %% [markdown]
# ## 4. Kaggle execution and reproducible candidate files
#
# Select the smallest fraction retaining the configured share of full-geometry
# known divisions in the *training* embryo using the distance control alone.
# Apply that locked fraction to both ranks on the opposite embryo. The full
# ratio grid is saved as a diagnostic and is never selected using outer labels.


# %%
def run_diagnostic(cfg: dict) -> dict:
    import torch

    if not is_kaggle_runtime():
        raise RuntimeError("the first full diagnostic must run on Kaggle")
    started = time.monotonic()
    runtime, validation, data = cfg["runtime"], cfg["validation"], cfg["data"]
    deadline = started + float(runtime["time_limit_seconds"])
    input_root = Path("/kaggle/input")
    output_root = Path("/kaggle/working") / EXPERIMENT
    output_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(runtime["torch_threads"]))
    _, public_repo = support_runtime(input_root, cfg["official_metric_source"], output_root)
    require_sha(
        public_repo / "src/biohub_tracking/models/simple_node_transformer.py",
        cfg["model"]["tracker_source_sha256"],
    )
    _, motion_receipt, motion_files, calibrations = read_exp025_output(input_root, cfg)
    split_path = require_sha(Path.cwd() / validation["split_file"], validation["split_sha256"])
    splits = json.loads(split_path.read_text())
    if len(splits) != 2 or {item["fold"] for item in splits} != {0, 1}:
        raise ValueError("expected two fixed embryo splits")
    cache_matches = []
    for path in artifact_paths(input_root, "window_cache_summary.json"):
        value = json.loads(path.read_text())
        unsigned = {key: item for key, item in value.items() if key != "summary_sha256"}
        if value.get("summary_sha256") == data["cache"]["summary_sha256"] == json_sha(unsigned):
            cache_matches.append(path)
    if len(cache_matches) != 1:
        raise ValueError("expected one SHA-pinned exp015 cache")
    cache_root = cache_matches[0].parent / "window_cache"
    paths = sorted(cache_root.glob("*/*.npz"))
    if (
        len(paths) != int(validation["expected_window_count"])
        or recompute_cache_identity_sha256(paths) != data["cache"]["identity_sha256"]
    ):
        raise ValueError("exp015 cache coverage or identity changed")
    samples = sorted({path.parent.name for path in paths})
    if (
        len(samples) != int(validation["expected_sample_count"])
        or dict(Counter(s.split("_", 1)[0] for s in samples))
        != validation["expected_embryo_counts"]
    ):
        raise ValueError("sample or embryo coverage mismatch")
    sample_split = {}
    for split in splits:
        validate_split(split, samples)
        for sample in split["outer_evaluation"]:
            if sample in sample_split:
                raise ValueError("overlapping outer sample")
            sample_split[sample] = split
    if sorted(sample_split) != samples or sorted(motion_files) != samples:
        raise ValueError("outer split or exp025 state coverage mismatch")
    train_root = resolve_pinned(
        input_root, "model_manifest.json", data["model_manifest_sha256"]
    ).parent
    secondary_checkpoint = resolve_checkpoint(
        input_root, output_root, cfg["model"]["secondary_sha256"]
    )
    device = str(runtime["device"])
    secondary = _load_tracker(secondary_checkpoint, device, cfg["model"]["tracker_params"])
    scale = np.asarray(validation["voxel_scale_zyx_um"], dtype=np.float64)
    fractions = list(map(float, validation["budget_fractions"]))
    if fractions != sorted(set(fractions)) or fractions[-1] != 1.0:
        raise ValueError("budget fractions must be unique, sorted and end at one")
    train_dir = Path(f"/kaggle/input/competitions/{data['competition_slug']}/train")
    if not train_dir.is_dir():
        train_dir = Path(f"/kaggle/input/{data['competition_slug']}/train")
    if not train_dir.is_dir():
        raise FileNotFoundError("competition train GEFF directory")
    rows: list[dict] = []
    ranking_receipts: list[dict] = []
    gt_receipts: list[list[str]] = []
    max_window_triplets = 0
    max_historyless = 0
    for split in splits:
        primary = _load_tracker(
            fold_tracker_checkpoint(train_root, split, cfg), device, cfg["model"]["tracker_params"]
        )
        noise = calibrations[split["fold"]]["noise"]
        motion_weight = float(calibrations[split["fold"]]["selected"]["motion_weight"])
        for sample in sorted(split["outer_evaluation"]):
            check_deadline(deadline)
            sample_paths = sorted((cache_root / sample).glob("*.npz"))
            if len(sample_paths) != int(validation["expected_frames_per_sample"]) - 1:
                raise ValueError(f"incomplete windows: {sample}")
            gt_frames, outgoing, gt_sha, _ = load_gt(train_dir / f"{sample}.geff", scale)
            gt_receipts.append([sample, gt_sha])
            states = load_motion_states(motion_files[sample]["path"])
            with np.load(motion_files[sample]["graph_path"], allow_pickle=False) as saved_graph:
                graph_ids = np.asarray(saved_graph["node_ids"], dtype=np.int64)
                graph_tzyx = np.asarray(saved_graph["node_tzyx"], dtype=np.float64)
            if graph_tzyx.shape != (len(graph_ids), 4) or len(set(map(int, graph_ids))) != len(
                graph_ids
            ):
                raise ValueError("invalid exp025 reference graph nodes")
            graph_coordinates = {
                int(node_id): (int(tzyx[0]), tzyx[1:] * scale)
                for node_id, tzyx in zip(graph_ids, graph_tzyx, strict=True)
            }
            frame_registry: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            frame_matches: dict[int, dict[int, int]] = {}
            counts = {
                (method, fraction): Counter()
                for method in ("distance", "image_motion")
                for fraction in fractions
            }
            triplet_parts, distance_parts, image_parts, motion_parts = [], [], [], []
            offsets = [0]
            for path in sample_paths:
                check_deadline(deadline)
                source_frame, target_frame = map(int, path.stem.split("_"))
                if target_frame != source_frame + 1:
                    raise ValueError("nonadjacent cache window")
                arrays, _ = _read_cache_window(path)
                for frame, side in ((source_frame, "src"), (target_frame, "tgt")):
                    ids = np.asarray(arrays[f"candidate_ids_{side}"], dtype=np.int64)
                    xyz = np.asarray(arrays[f"coords_{side}_physical"], dtype=np.float64)
                    if any(int(node_id) not in graph_coordinates for node_id in ids):
                        raise ValueError("exp025 graph misses fixed cache candidates")
                    if any(graph_coordinates[int(node_id)][0] != frame for node_id in ids):
                        raise ValueError("exp025 graph candidate frame mismatch")
                    graph_xyz = np.asarray([graph_coordinates[int(node_id)][1] for node_id in ids])
                    if not np.allclose(xyz, graph_xyz, atol=1e-4, rtol=0):
                        raise ValueError("exp025 graph coordinates differ from fixed cache")
                    if frame in frame_registry:
                        before_ids, before_xyz = frame_registry[frame]
                        if not np.array_equal(ids, before_ids) or not np.array_equal(
                            xyz, before_xyz
                        ):
                            raise ValueError(f"overlapping cache frame changed: {sample}/{frame}")
                    else:
                        frame_registry[frame] = (ids, xyz)
                        frame_matches[frame] = match_frame(
                            gt_frames.get(frame, {}),
                            ids,
                            xyz,
                            float(validation["matching_radius_um"]),
                        )
                triplets, distance = enumerate_triplets(
                    np.asarray(arrays["candidate_ids_src"], dtype=np.int64),
                    np.asarray(arrays["coords_src_physical"], dtype=np.float64),
                    np.asarray(arrays["candidate_ids_tgt"], dtype=np.int64),
                    np.asarray(arrays["coords_tgt_physical"], dtype=np.float64),
                    float(validation["parent_daughter_max_um"]),
                    float(validation["sister_max_um"]),
                    int(validation["max_triplets_per_parent_guard"]),
                )
                max_window_triplets = max(max_window_triplets, len(triplets))
                logits = fuse_cached_edge_logits(
                    primary,
                    secondary,
                    arrays,
                    device=device,
                    downsample_zyx=tuple(cfg["model"]["replay"]["downsample_zyx"]),
                    bidirectional_weight=float(cfg["model"]["replay"]["bidirectional_weight"]),
                    secondary_edge_weight=float(cfg["model"]["replay"]["secondary_edge_weight"]),
                    secondary_low_margin_max=float(
                        cfg["model"]["replay"]["secondary_low_margin_max"]
                    ),
                    secondary_mix_temperature=float(
                        cfg["model"]["replay"]["secondary_mix_temperature"]
                    ),
                )
                image, motion, historyless = triplet_costs(
                    arrays, triplets, logits, states, noise, motion_weight
                )
                max_historyless += historyless
                full = set(map(tuple, triplets.tolist()))
                known = true_divisions(source_frame, gt_frames, outgoing)
                positive = set()
                for mother, first, second in known:
                    src_match, tgt_match = frame_matches[source_frame], frame_matches[target_frame]
                    if mother in src_match and first in tgt_match and second in tgt_match:
                        pair = tuple(sorted((tgt_match[first], tgt_match[second])))
                        candidate = (src_match[mother], *pair)
                        if candidate in full:
                            positive.add(candidate)
                for method, cost in (("distance", distance), ("image_motion", image + motion)):
                    for fraction in fractions:
                        selected = budget_indices(triplets, cost, fraction)
                        selected_set = {tuple(triplets[i]) for i in selected}
                        row = counts[method, fraction]
                        row["windows"] += 1
                        row["available"] += len(triplets)
                        row["selected"] += len(selected)
                        row["known_divisions"] += len(known)
                        row["full_recovered"] += len(positive)
                        row["recovered"] += len(positive & selected_set)
                triplet_parts.append(triplets)
                distance_parts.append(distance)
                image_parts.append(image)
                motion_parts.append(motion)
                offsets.append(offsets[-1] + len(triplets))
            if len(frame_registry) != int(validation["expected_frames_per_sample"]):
                raise ValueError("incomplete candidate frames")
            ranking_path = output_root / "rankings" / f"{sample}.npz"
            ranking_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                ranking_path,
                triplets=np.concatenate(triplet_parts),
                distance_cost=np.concatenate(distance_parts),
                image_cost=np.concatenate(image_parts),
                motion_cost=np.concatenate(motion_parts),
                window_offsets=np.asarray(offsets, dtype=np.int64),
            )
            ranking_receipts.append(
                {
                    "sample": sample,
                    "sha256": file_sha(ranking_path),
                    "triplets": offsets[-1],
                    "exp025_states_sha256": motion_files[sample]["sha256"],
                    "exp025_graph_sha256": motion_files[sample]["graph_sha256"],
                }
            )
            for (method, fraction), count in counts.items():
                rows.append(
                    {
                        "sample": sample,
                        "embryo": split["evaluation_embryo"],
                        "method": method,
                        "fraction": fraction,
                        **dict(count),
                    }
                )
            print(
                f"RANKED sample={sample} windows={len(sample_paths)} triplets={offsets[-1]} "
                f"elapsed={time.monotonic() - started:.1f}",
                flush=True,
            )
            if len(ranking_receipts) == int(runtime["projection_after_samples"]):
                projected = (time.monotonic() - started) * len(samples) / len(ranking_receipts)
                projected *= float(runtime["projection_multiplier"])
                projected += float(runtime["artifact_reserve_seconds"])
                if projected >= float(runtime["time_limit_seconds"]):
                    raise TimeoutError(f"12-hour runtime projection failed: {projected:.1f}s")
        del primary
    del secondary
    if sum(
        row["known_divisions"]
        for row in rows
        if row["method"] == "distance" and row["fraction"] == 1.0
    ) != int(validation["expected_known_divisions"]) or sum(
        row["full_recovered"]
        for row in rows
        if row["method"] == "distance" and row["fraction"] == 1.0
    ) != int(validation["expected_full_recovered"]):
        raise ValueError("exp020 known-division or geometry baseline changed")
    gt_bundle_sha = json_sha(sorted(gt_receipts))
    if gt_bundle_sha != validation["expected_gt_bundle_sha256"]:
        raise ValueError("organizer GT content changed from exp021")
    chosen = {
        split["evaluation_embryo"]: choose_training_budget(
            rows, fractions, float(validation["training_recall_floor"]), split["train_embryo"]
        )
        for split in splits
    }
    selected_receipts = []
    for item in ranking_receipts:
        sample = item["sample"]
        fraction = chosen[sample_split[sample]["evaluation_embryo"]]
        with np.load(output_root / "rankings" / f"{sample}.npz", allow_pickle=False) as saved:
            triplets = saved["triplets"]
            distance = saved["distance_cost"]
            joint = saved["image_cost"] + saved["motion_cost"]
            offsets = saved["window_offsets"]
        selected = {method: [] for method in ("distance", "image_motion")}
        selected_offsets = {method: [0] for method in selected}
        for left, right in zip(offsets[:-1], offsets[1:], strict=True):
            for method, cost in (("distance", distance), ("image_motion", joint)):
                indices = budget_indices(triplets[left:right], cost[left:right], fraction)
                selected[method].append(triplets[left:right][indices])
                selected_offsets[method].append(selected_offsets[method][-1] + len(indices))
        if selected_offsets["distance"] != selected_offsets["image_motion"]:
            raise ValueError("methods have unequal window budgets")
        path = output_root / "selected_candidates" / f"{sample}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            distance_triplets=np.concatenate(selected["distance"]),
            image_motion_triplets=np.concatenate(selected["image_motion"]),
            window_offsets=np.asarray(selected_offsets["distance"], dtype=np.int64),
            budget_fraction=np.asarray(fraction, dtype=np.float32),
        )
        selected_receipts.append(
            {
                "sample": sample,
                "sha256": file_sha(path),
                "triplets_per_method": selected_offsets["distance"][-1],
                "fraction": fraction,
            }
        )
    selected_rows = [row for row in rows if row["fraction"] == chosen[row["embryo"]]]
    summary = {
        "experiment": EXPERIMENT,
        "status": "initial_candidate_recovery_only",
        "selected_budget_by_evaluation_embryo": chosen,
        "selected_outer": {
            method: {
                embryo: {
                    key: sum(
                        int(row[key])
                        for row in selected_rows
                        if row["method"] == method and row["embryo"] == embryo
                    )
                    for key in (
                        "available",
                        "selected",
                        "known_divisions",
                        "full_recovered",
                        "recovered",
                    )
                }
                for embryo in sorted(validation["expected_embryo_counts"])
            }
            for method in ("distance", "image_motion")
        },
        "sample_count": len(samples),
        "window_count": len(paths),
        "max_window_triplets": max_window_triplets,
        "historyless_triplet_scores": max_historyless,
        "cache_identity_sha256": data["cache"]["identity_sha256"],
        "gt_content_bundle_sha256": gt_bundle_sha,
        "exp025_receipt_sha256": json_sha(motion_receipt),
        "exp025_calibration_sha256": {
            str(fold): value["calibration_sha256"] for fold, value in calibrations.items()
        },
        "ranking_manifest_sha256": json_sha(ranking_receipts),
        "selected_manifest_sha256": json_sha(selected_receipts),
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else None,
        "fold_model_shas": {
            fold: row["sha256"] for fold, row in cfg["model"]["fold_checkpoints"].items()
        },
        "secondary_model_sha256": cfg["model"]["secondary_sha256"],
        "official_score_measured": False,
    }
    write_json(output_root / "summary.json", summary)
    write_json(output_root / "per_sample_budget.json", rows)
    write_json(output_root / "rankings_manifest.json", ranking_receipts)
    write_json(output_root / "selected_manifest.json", selected_receipts)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


# %%
if __name__ == "__main__":
    configuration = yaml.safe_load((Path.cwd() / "config.yaml").read_text())
    run_diagnostic(configuration)
