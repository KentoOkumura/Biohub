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
# # exp024: adjacent-frame tracker diagnostic
#
# Compare the saved two- and six-epoch primary tracker weights on the same
# embryo-held-out, adjacent-frame cache windows. Only targets with a known
# parent are scored. This does not build a graph, select checkpoints, or
# estimate the official competition score.

# %% [markdown]
# ## Contents
# 1. Configuration and fixed comparison
# 2. Offline GEFF dependencies
# 3. Verify cache, annotations, and checkpoint files
# 4. Evaluate both checkpoints on each held-out embryo
# 5. Save per-embryo and per-sample counts

# %% [markdown]
# ## 1. Configuration and fixed comparison

# %%
from __future__ import annotations

# ruff: noqa: E402
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT = "exp024_tracker_six_epochs"
COMPETITION = "biohub-cell-tracking-during-development"
WORKING_ROOT = Path.cwd()
STARTED = time.perf_counter()
if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
    raise RuntimeError("The full diagnostic must run on Kaggle with the fixed input cache")

config = yaml.safe_load((WORKING_ROOT / "config.yaml").read_text(encoding="utf-8"))
cache_cfg = config["data"]["cache"]
public_cfg = config["model"]["public_source"]
teacher_cfg = config["model"]["teacher"]
params_cfg = config["model"]["params"]
diagnostic_cfg = config["runtime"]["kaggle"]["diagnostic"]
epochs = [int(value) for value in diagnostic_cfg["checkpoint_epochs"]]
if epochs != [2, 6]:
    raise RuntimeError("The fixed comparison must use epoch 2 and epoch 6")
threshold = float(diagnostic_cfg["edge_threshold"])
if threshold != float(config["model"]["inference"]["replay"]["edge_threshold"]):
    raise RuntimeError("The diagnostic threshold differs from candidate-edge inference")
print("Diagnostic:", EXPERIMENT, "epochs", epochs, "threshold", threshold, flush=True)

# %% [markdown]
# ## 2. Offline GEFF dependencies
#
# The sparse annotation reader uses the same offline wheels as training.

# %%
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


ensure_geff_runtime_dependencies()
import torch
from frozen_tracker import (
    FrozenFeatureWindowDataset,
    build_embryo_splits,
    collate_window_examples,
    discover_cache_paths,
    file_sha256,
    filter_nonempty_gt_window_paths,
    load_annotation_graph,
    move_batch_to_device,
    paths_for_samples,
    recompute_cache_identity_sha256,
    tracker_logits,
    validate_cache_summary,
)
from graph_inference import _load_tracker
from torch.utils.data import DataLoader

from src.tracker_pair_metrics import (
    COUNT_KEYS,
    aggregate_known_parent_links,
    count_known_parent_links,
    summarize_known_parent_links,
)

# %% [markdown]
# ## 3. Verify cache, annotations, and checkpoint files


# %%
def unique_existing(paths: list[Path], label: str) -> Path:
    matches: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved.exists() and resolved not in seen:
            matches.append(resolved)
            seen.add(resolved)
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {matches}")
    return matches[0]


def resolve_cache_output_root() -> Path:
    kernel_slug = str(cache_cfg["kernel_source"]).split("/", 1)[-1]
    candidates = [
        path.parent
        for path in Path("/kaggle/input/notebooks").rglob(str(cache_cfg["summary_file"]))
        if kernel_slug in path.as_posix()
    ]
    return unique_existing(candidates, "exp015 cache output root")


def resolve_public_artifact_root() -> Path:
    slug = str(public_cfg["dataset_ref"]).split("/", 1)[-1]
    candidates = [
        Path(f"/kaggle/input/datasets/pilkwang/{slug}"),
        Path(f"/kaggle/input/{slug}"),
        Path(f"/kaggle/input/{slug}/{slug}"),
    ]
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        candidates.extend(
            path.parents[4]
            for path in input_root.rglob(str(public_cfg["model_source"]))
            if len(path.parents) >= 5
        )
    valid = [
        root
        for root in candidates
        if (root / str(public_cfg["train_script"])).is_file()
        and (root / str(public_cfg["model_source"])).is_file()
        and (root / str(public_cfg["checkpoint"])).is_file()
    ]
    return unique_existing(valid, "public tracker artifact root")


def resolve_train_dir() -> Path:
    return unique_existing(
        [
            Path(f"/kaggle/input/competitions/{COMPETITION}/train"),
            Path(f"/kaggle/input/{COMPETITION}/train"),
        ],
        "competition train directory",
    )


def resolve_train_output_root() -> Path:
    expected_hash = str(diagnostic_cfg["expected_train_manifest_sha256"])
    matches = [
        path.parent
        for path in Path("/kaggle/input/notebooks").rglob("model_manifest.json")
        if file_sha256(path) == expected_hash
    ]
    return unique_existing(matches, "completed exp024 train output")


cache_output_root = resolve_cache_output_root()
cache_summary = validate_cache_summary(
    cache_output_root / str(cache_cfg["summary_file"]), cache_cfg
)
cache_root = cache_output_root / str(cache_cfg["directory"])
cache_paths = discover_cache_paths(cache_root, cache_cfg)
cache_identity = recompute_cache_identity_sha256(cache_paths)
if cache_identity != str(cache_cfg["identity_sha256"]):
    raise RuntimeError("fixed cache identity mismatch")
public_root = resolve_public_artifact_root()
model_source = public_root / str(public_cfg["model_source"])
if file_sha256(model_source) != str(public_cfg["model_source_sha256"]):
    raise RuntimeError("public model source checksum mismatch")
sys.path.insert(0, str(public_root / "repo" / "src"))
train_dir = resolve_train_dir()
train_output_root = resolve_train_output_root()
train_manifest = json.loads((train_output_root / "model_manifest.json").read_text(encoding="utf-8"))
if (
    train_manifest.get("experiment") != EXPERIMENT
    or train_manifest.get("cache_identity_sha256") != cache_identity
    or train_manifest.get("training_epochs") != 6
):
    raise RuntimeError("training output manifest differs from the fixed comparison")

sample_names = sorted({path.parent.name for path in cache_paths})
scale_zyx_um = tuple(float(value) for value in config["data"]["annotation"]["voxel_scale_zyx_um"])
annotations = {
    sample: load_annotation_graph(train_dir / f"{sample}.geff", scale_zyx_um)
    for sample in sample_names
}
annotation_manifest = {
    sample: annotation.content_sha256 for sample, annotation in sorted(annotations.items())
}
from frozen_tracker import json_sha256

if json_sha256(annotation_manifest) != train_manifest["annotation_content_sha256"]:
    raise RuntimeError("annotation content changed after training")
cache_paths, filter_audit = filter_nonempty_gt_window_paths(cache_paths, annotations)
splits = build_embryo_splits(
    sample_names,
    config["validation"]["outer_folds"],
    int(config["validation"]["internal_split_seed"]),
)
print("Cache windows after fixed GT filter:", len(cache_paths), flush=True)
print("Train output:", train_output_root, flush=True)

checkpoint_records: dict[int, dict[int, dict[str, Any]]] = {}
for fold in (0, 1):
    fold_summary_path = train_output_root / f"fold_{fold}_summary.json"
    summary = json.loads(fold_summary_path.read_text(encoding="utf-8"))
    checkpoint_records[fold] = {}
    for epoch in epochs:
        record = summary["epochs"][epoch - 1]
        if int(record["epoch"]) != epoch - 1:
            raise RuntimeError("checkpoint epoch index mismatch")
        checkpoint_path = train_output_root / str(record["model_file"])
        if file_sha256(checkpoint_path) != record["model_file_sha256"]:
            raise RuntimeError(f"checkpoint file checksum mismatch: {checkpoint_path}")
        checkpoint_records[fold][epoch] = {
            "path": checkpoint_path,
            "file_sha256": record["model_file_sha256"],
            "state_sha256": record["model_state_sha256"],
        }

# The selected exp016 three-epoch baseline was the second epoch in both folds.
EXP016_BASELINE_STATE_SHA256 = {
    0: "b080c8dc374454f0f97403ec1e27388ef9a5a76c4c658b0ccaeb247ab136ab6c",
    1: "41b03b82d30e9b22d73df6924a70d56ca2b292bb3c41b2678acb5d33990fb3ea",
}
for fold in (0, 1):
    if checkpoint_records[fold][2]["state_sha256"] != EXP016_BASELINE_STATE_SHA256[fold]:
        raise RuntimeError("epoch 2 is not the saved exp016 selected baseline")

# %% [markdown]
# ## 4. Evaluate both checkpoints on each held-out embryo
#
# The same teacher matching, source-axis softmax, and threshold are applied to
# both checkpoints. Unknown target columns are excluded from negative counts.

# %%
device = torch.device("cuda")
if not torch.cuda.is_available():
    raise RuntimeError("the full diagnostic requires the configured Kaggle GPU")
all_results: dict[str, Any] = {}
for split in splits:
    fold = int(split["fold"])
    embryo = str(split["evaluation_embryo"])
    paths = paths_for_samples(cache_paths, split["outer_evaluation"])
    dataset = FrozenFeatureWindowDataset(
        paths,
        annotations,
        feature_channels=int(cache_cfg["feature_channels"]),
        expected_primary_checkpoint_sha256=str(public_cfg["checkpoint_sha256"]),
        max_matching_distance_um=float(teacher_cfg["max_matching_distance_um"]),
        downsample_zyx=tuple(float(value) for value in params_cfg["downsample_zyx"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["runtime"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["runtime"]["num_workers"]),
        collate_fn=collate_window_examples,
    )
    models = {
        epoch: _load_tracker(checkpoint_records[fold][epoch]["path"], device, params_cfg)
        for epoch in epochs
    }
    per_sample = {
        epoch: {sample: dict.fromkeys(COUNT_KEYS, 0) for sample in split["outer_evaluation"]}
        for epoch in epochs
    }
    fold_started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            for epoch, model in models.items():
                logits = tracker_logits(model, batch).float()
                for index in range(logits.shape[0]):
                    n_source = int(batch["source_mask"][index].sum().item())
                    n_target = int(batch["target_mask"][index].sum().item())
                    probs = (
                        torch.softmax(logits[index, :n_source, :n_target], dim=0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    target = batch["target"][index, :n_source, :n_target].detach().cpu().numpy()
                    counts = count_known_parent_links(probs, target, threshold=threshold)
                    sample = str(batch["metadata"][index]["sample"])
                    for key in COUNT_KEYS:
                        per_sample[epoch][sample][key] += counts[key]
            if (batch_index + 1) % 500 == 0:
                print(
                    f"PAIR_DIAGNOSTIC fold={fold} batches={batch_index + 1} "
                    f"elapsed_seconds={time.perf_counter() - fold_started:.1f}",
                    flush=True,
                )
    fold_results = {}
    for epoch in epochs:
        totals = aggregate_known_parent_links(per_sample[epoch].values())
        fold_results[str(epoch)] = {
            "checkpoint": {
                "file_sha256": checkpoint_records[fold][epoch]["file_sha256"],
                "state_sha256": checkpoint_records[fold][epoch]["state_sha256"],
            },
            "counts": totals,
            "rates": summarize_known_parent_links(totals),
            "per_sample_counts": per_sample[epoch],
        }
        print(
            f"PAIR_DIAGNOSTIC fold={fold} embryo={embryo} epoch={epoch} "
            f"counts={json.dumps(totals, sort_keys=True)} "
            f"rates={json.dumps(fold_results[str(epoch)]['rates'], sort_keys=True)}",
            flush=True,
        )
    all_results[str(fold)] = {
        "evaluation_embryo": embryo,
        "sample_count": len(split["outer_evaluation"]),
        "window_count": len(paths),
        "elapsed_seconds": time.perf_counter() - fold_started,
        "epochs": fold_results,
    }
    del models, loader, dataset
    torch.cuda.empty_cache()

# %% [markdown]
# ## 5. Save per-embryo and per-sample counts
#
# This is a primary-tracker diagnostic before bidirectional and secondary
# fusion, ILP, graph repair, and official scoring. No checkpoint rule changes.

# %%
output = {
    "experiment": EXPERIMENT,
    "created_at": datetime.now(UTC).isoformat(),
    "metric_scope": "raw_primary_tracker_adjacent_frame_known_parent_targets",
    "cache_identity_sha256": cache_identity,
    "train_manifest_sha256": str(diagnostic_cfg["expected_train_manifest_sha256"]),
    "annotation_content_sha256": train_manifest["annotation_content_sha256"],
    "edge_threshold": threshold,
    "checkpoint_epochs": epochs,
    "folds": all_results,
    "not_official_score": True,
    "checkpoint_selection_unchanged": True,
    "elapsed_seconds": time.perf_counter() - STARTED,
}
output_path = WORKING_ROOT / "tracker_pair_diagnostic.json"
output_path.write_text(
    json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("PAIR_DIAGNOSTIC output:", output_path, flush=True)
print("PAIR_DIAGNOSTIC elapsed_seconds:", output["elapsed_seconds"], flush=True)
