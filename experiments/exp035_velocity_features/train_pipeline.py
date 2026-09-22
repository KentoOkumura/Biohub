from __future__ import annotations

import copy
import importlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT = "exp035_velocity_features"
COMPETITION = "biohub-cell-tracking-during-development"
WORKING_ROOT = Path.cwd()
CONFIG_PATH = WORKING_ROOT / "config.yaml"
METRICS_PATH = WORKING_ROOT / "metrics.json"
OUTPUT_MODELS = WORKING_ROOT / "models"
OUTPUT_HISTORY_MODELS = WORKING_ROOT / "history_models"
NOTEBOOK_STARTED = time.perf_counter()

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


def resolve_cache_output_root(cache_cfg: dict[str, Any]) -> Path:
    kernel_slug = str(cache_cfg["kernel_source"]).split("/", 1)[-1]
    candidates = [
        path.parent
        for path in Path("/kaggle/input/notebooks").rglob(str(cache_cfg["summary_file"]))
        if kernel_slug in path.as_posix()
    ]
    return unique_existing(candidates, "exp015 cache output root")


def resolve_control_output_root(control_cfg: dict[str, Any]) -> Path:
    kernel_slug = str(control_cfg["kernel_source"]).split("/", 1)[-1]
    candidates = [
        path.parent
        for path in Path("/kaggle/input/notebooks").rglob(str(control_cfg["manifest_file"]))
        if kernel_slug in path.as_posix()
    ]
    return unique_existing(candidates, "exp016 control output root")


def resolve_public_artifact_root(public_cfg: dict[str, Any]) -> Path:
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


def split_two_folds(samples: list[str], seed: int) -> list[list[str]]:
    shuffled = sorted(samples)
    random.Random(seed).shuffle(shuffled)
    midpoint = (len(shuffled) + 1) // 2
    folds = [sorted(shuffled[:midpoint]), sorted(shuffled[midpoint:])]
    if not all(folds) or set(folds[0]) & set(folds[1]):
        raise ValueError("history cross-fit folds must be nonempty and disjoint")
    if set(folds[0]) | set(folds[1]) != set(samples):
        raise ValueError("history cross-fit folds do not cover the training embryo")
    return folds


def split_selection_samples(
    samples: list[str],
    *,
    seed: int,
    validation_fraction: float,
) -> tuple[list[str], list[str]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("selection validation fraction must be in (0, 1)")
    shuffled = sorted(samples)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(len(shuffled) * validation_fraction))
    if validation_count >= len(shuffled):
        validation_count = len(shuffled) - 1
    validation = sorted(shuffled[:validation_count])
    gradient = sorted(shuffled[validation_count:])
    if not gradient or not validation:
        raise ValueError("history model split requires gradient and validation samples")
    return gradient, validation


def main() -> None:
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("The authoritative full run must execute on Kaggle.")
    ensure_geff_runtime_dependencies()

    import torch
    from frozen_tracker import (
        FrozenFeatureWindowDataset,
        PredictionWindowDataset,
        aggregate_teacher_stats,
        build_embryo_splits,
        canonical_state_sha256,
        collate_prediction_examples,
        collate_window_examples,
        dataloader_worker_init,
        discover_cache_paths,
        evaluate_tracker,
        evaluate_tracker_diagnostic,
        extract_public_tracker_state,
        file_sha256,
        filter_nonempty_gt_window_paths,
        generate_fixed_first_pass_history,
        json_sha256,
        load_annotation_graph,
        paths_for_samples,
        recompute_cache_identity_sha256,
        seed_everything,
        train_one_epoch,
        validate_cache_summary,
    )
    from settings import update_metrics
    from torch.utils.data import DataLoader
    from velocity_history import HistoryByFrame
    from velocity_tracker import VELOCITY_PAIR_FEATURE_NAMES, VelocityAugmentedTracker

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    validation_cfg = config["validation"]
    cache_cfg = config["data"]["cache"]
    model_cfg = config["model"]
    train_cfg = model_cfg["training"]
    history_cfg = model_cfg["history"]
    control_cfg = model_cfg["control"]
    teacher_cfg = model_cfg["teacher"]
    loss_cfg = model_cfg["loss"]
    params_cfg = model_cfg["params"]

    expected_trainable = ["primary_SimpleNodeTransformer", "velocity_pair_branch"]
    if model_cfg["trainable_components"] != expected_trainable:
        raise RuntimeError("velocity tracker trainable-component contract changed")
    if train_cfg["active_variants"] != ["velocity_features"]:
        raise RuntimeError("the train stage requires only the velocity_features variant")
    if control_cfg["retrain"] is not False:
        raise RuntimeError("the exp016 control must not be retrained")
    if int(model_cfg["output"]["model_count"]) != 2:
        raise RuntimeError("the experiment requires two outer velocity models")
    if int(model_cfg["output"]["history_model_count"]) != 4:
        raise RuntimeError("the experiment requires four cross-fit history models")
    if int(history_cfg["inner_generator_model_count"]) != 4:
        raise RuntimeError("history config requires four inner generator models")
    if int(params_cfg["velocity_pair_feature_dim"]) != len(VELOCITY_PAIR_FEATURE_NAMES):
        raise RuntimeError("velocity feature width differs from the declared schema")

    cache_output_root = resolve_cache_output_root(cache_cfg)
    cache_summary_path = cache_output_root / str(cache_cfg["summary_file"])
    cache_summary = validate_cache_summary(cache_summary_path, cache_cfg)
    cache_root = cache_output_root / str(cache_cfg["directory"])
    all_cache_paths = discover_cache_paths(cache_root, cache_cfg)
    observed_cache_identity_sha256 = recompute_cache_identity_sha256(all_cache_paths)
    if observed_cache_identity_sha256 != str(cache_cfg["identity_sha256"]):
        raise RuntimeError("exp015 cache identity differs from config")

    public_root = resolve_public_artifact_root(model_cfg["public_source"])
    train_dir = resolve_train_dir()
    control_root = resolve_control_output_root(control_cfg)
    control_manifest_path = control_root / str(control_cfg["manifest_file"])
    if file_sha256(control_manifest_path) != str(control_cfg["manifest_sha256"]):
        raise RuntimeError("exp016 model manifest SHA differs from config")
    control_manifest = json.loads(control_manifest_path.read_text(encoding="utf-8"))
    if control_manifest.get("experiment") != str(control_cfg["experiment"]):
        raise RuntimeError("exp016 model manifest identifies a different experiment")

    public_cfg = model_cfg["public_source"]
    source_paths = {
        "train_script": public_root / str(public_cfg["train_script"]),
        "model_source": public_root / str(public_cfg["model_source"]),
        "checkpoint": public_root / str(public_cfg["checkpoint"]),
    }
    expected_source_hashes = {
        "train_script": str(public_cfg["train_script_sha256"]),
        "model_source": str(public_cfg["model_source_sha256"]),
        "checkpoint": str(public_cfg["checkpoint_sha256"]),
    }
    observed_source_hashes = {name: file_sha256(path) for name, path in source_paths.items()}
    if observed_source_hashes != expected_source_hashes:
        raise RuntimeError("public tracker source or checkpoint SHA differs from config")

    sample_names = sorted({path.parent.name for path in all_cache_paths})
    observed_embryo_counts = {
        embryo: sum(name.startswith(f"{embryo}_") for name in sample_names)
        for embryo in config["data"]["expected_embryo_counts"]
    }
    if observed_embryo_counts != config["data"]["expected_embryo_counts"]:
        raise RuntimeError({"embryo_count_mismatch": observed_embryo_counts})
    missing_geff = [name for name in sample_names if not (train_dir / f"{name}.geff").is_dir()]
    if missing_geff:
        raise FileNotFoundError({"missing_train_geff": missing_geff[:20]})

    splits = build_embryo_splits(
        sample_names,
        validation_cfg["outer_folds"],
        int(validation_cfg["internal_split_seed"]),
    )
    scale_zyx_um = tuple(
        float(value) for value in config["data"]["annotation"]["voxel_scale_zyx_um"]
    )
    annotations = {
        sample: load_annotation_graph(train_dir / f"{sample}.geff", scale_zyx_um)
        for sample in sample_names
    }
    eligible_cache_paths, gt_window_filter_audit = filter_nonempty_gt_window_paths(
        all_cache_paths,
        annotations,
    )
    gt_window_filter_path = WORKING_ROOT / "gt_window_filter_audit.json"
    gt_window_filter_path.write_text(
        json.dumps(gt_window_filter_audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gt_window_filter_audit_sha256 = file_sha256(gt_window_filter_path)
    gt_window_filter_summary = {
        key: value for key, value in gt_window_filter_audit.items() if key != "skipped_windows"
    }

    split_manifest = [
        {
            **record,
            "window_counts": {
                key: len(paths_for_samples(eligible_cache_paths, record[key]))
                for key in ("gradient_update", "internal_validation", "outer_evaluation")
            },
        }
        for record in splits
    ]
    (WORKING_ROOT / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    annotation_manifest = {
        sample: annotation.content_sha256 for sample, annotation in sorted(annotations.items())
    }

    public_src = public_root / "repo" / "src"
    sys.path.insert(0, str(public_src))
    SimpleNodeTransformer = importlib.import_module("biohub_tracking.models").SimpleNodeTransformer
    full_public_state = torch.load(
        source_paths["checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    public_tracker_state = extract_public_tracker_state(full_public_state)
    public_tracker_state_sha256 = canonical_state_sha256(public_tracker_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("the authoritative train run requires a Kaggle GPU")
    OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUTPUT_HISTORY_MODELS.mkdir(parents=True, exist_ok=True)

    feature_channels = int(cache_cfg["feature_channels"])
    max_matching_distance_um = float(teacher_cfg["max_matching_distance_um"])
    downsample_zyx = tuple(float(value) for value in params_cfg["downsample_zyx"])
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"])
    global_seed = int(train_cfg["seed"])
    use_amp = bool(train_cfg["mixed_precision"])
    gamma = float(loss_cfg["focal_gamma"])
    threshold = float(history_cfg["edge_probability_threshold"])

    def new_base_tracker() -> Any:
        tracker = SimpleNodeTransformer(
            feat_dim=int(params_cfg["feature_dim"]),
            hidden_dim=int(params_cfg["hidden_dim"]),
            n_heads=int(params_cfg["n_heads"]),
            n_blocks=int(params_cfg["n_blocks"]),
            dropout=float(params_cfg["dropout"]),
            pair_chunk_size=int(params_cfg["pair_chunk_size"]),
        )
        tracker.load_state_dict(copy.deepcopy(public_tracker_state), strict=True)
        return tracker.to(device)

    def new_velocity_tracker() -> Any:
        return VelocityAugmentedTracker(
            new_base_tracker(),
            pair_feature_dim=int(params_cfg["velocity_pair_feature_dim"]),
            branch_hidden_dim=int(params_cfg["velocity_branch_hidden_dim"]),
            branch_dropout=float(params_cfg["velocity_branch_dropout"]),
            pair_chunk_size=int(params_cfg["pair_chunk_size"]),
            vector_scale=float(history_cfg["vector_scale_um_per_frame"]),
            vector_clip_abs=float(history_cfg["vector_clip_abs"]),
            history_length_cap=int(history_cfg["history_length_cap"]),
        ).to(device)

    control_records = {int(item["fold"]): item for item in control_cfg["models"]}
    manifest_control_records = {int(item["fold"]): item for item in control_manifest["models"]}
    if set(control_records) != {0, 1} or set(manifest_control_records) != {0, 1}:
        raise RuntimeError("exp016 control manifest must contain folds 0 and 1")

    def load_control_tracker(fold: int) -> Any:
        expected = control_records[fold]
        observed = manifest_control_records[fold]
        for key in (
            "fold",
            "evaluation_embryo",
            "path",
            "file_sha256",
            "canonical_state_sha256",
        ):
            if observed.get(key) != expected.get(key):
                raise RuntimeError({"exp016_manifest_mismatch": {"fold": fold, "field": key}})
        model_path = control_root / str(expected["path"])
        if file_sha256(model_path) != str(expected["file_sha256"]):
            raise RuntimeError(f"exp016 fold {fold} model file SHA differs from config")
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        state = payload["state_dict"]
        if canonical_state_sha256(state) != str(expected["canonical_state_sha256"]):
            raise RuntimeError(f"exp016 fold {fold} state SHA differs from config")
        tracker = new_base_tracker()
        tracker.load_state_dict(state, strict=True)
        return tracker

    def make_loader(
        paths: list[Path],
        *,
        shuffle: bool,
        seed: int,
        history_by_frame: HistoryByFrame | None = None,
    ) -> Any:
        dataset = FrozenFeatureWindowDataset(
            paths,
            annotations,
            feature_channels=feature_channels,
            expected_primary_checkpoint_sha256=str(public_cfg["checkpoint_sha256"]),
            max_matching_distance_um=max_matching_distance_um,
            downsample_zyx=downsample_zyx,
            history_by_frame=history_by_frame,
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_window_examples,
            generator=generator,
            worker_init_fn=dataloader_worker_init,
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None,
            pin_memory=True,
        )

    def make_prediction_loader(paths: list[Path]) -> Any:
        dataset = PredictionWindowDataset(
            paths,
            feature_channels=feature_channels,
            expected_primary_checkpoint_sha256=str(public_cfg["checkpoint_sha256"]),
            downsample_zyx=downsample_zyx,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_prediction_examples,
            worker_init_fn=dataloader_worker_init,
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None,
            pin_memory=True,
        )

    def new_optimizer(model: Any) -> Any:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        model_parameter_ids = {id(parameter) for parameter in model.parameters()}
        optimizer_parameter_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        if optimizer_parameter_ids != model_parameter_ids:
            raise RuntimeError("optimizer parameters differ from model parameters")
        return optimizer

    def new_scaler() -> Any:
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            return torch.cuda.amp.GradScaler(enabled=use_amp)

    def fit_model(
        model: Any,
        *,
        train_paths: list[Path],
        validation_paths: list[Path],
        seed: int,
        epochs: int,
        history_by_frame: HistoryByFrame | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        optimizer = new_optimizer(model)
        scaler = new_scaler()
        train_loader = make_loader(
            train_paths,
            shuffle=True,
            seed=seed,
            history_by_frame=history_by_frame,
        )
        validation_loader = make_loader(
            validation_paths,
            shuffle=False,
            seed=seed,
            history_by_frame=history_by_frame,
        )
        best_score = float("-inf")
        best_epoch = -1
        best_state: dict[str, Any] | None = None
        epoch_records: list[dict[str, Any]] = []
        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                gamma=gamma,
                gradient_clip_norm=float(train_cfg["gradient_clip_norm"]),
                use_amp=use_amp,
            )
            validation_metrics = evaluate_tracker(
                model,
                validation_loader,
                device,
                gamma=gamma,
                use_amp=use_amp,
            )
            score = float(validation_metrics["selection_score"])
            selected = score >= best_score
            if selected:
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            epoch_record = {
                "epoch": epoch,
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "train": train_metrics,
                "internal_validation": validation_metrics,
                "selected": selected,
            }
            epoch_records.append(epoch_record)
            print(json.dumps(epoch_record, default=float))
        if best_state is None:
            raise RuntimeError("model training did not select a checkpoint")
        model.load_state_dict(best_state, strict=True)
        del validation_loader, train_loader, scaler, optimizer
        return best_state, {
            "best_epoch": best_epoch,
            "best_internal_selection_score": best_score,
            "epochs": epoch_records,
        }

    def generate_history(model: Any, paths: list[Path]) -> tuple[HistoryByFrame, dict[str, Any]]:
        loader = make_prediction_loader(paths)
        history, audit = generate_fixed_first_pass_history(
            model,
            loader,
            device,
            edge_probability_threshold=threshold,
            use_amp=use_amp,
        )
        del loader
        return history, audit

    benchmark_seed = global_seed + 9000
    benchmark_count = int(train_cfg["benchmark_windows_per_fold"])
    size_ordered_paths = sorted(
        eligible_cache_paths,
        key=lambda path: (path.stat().st_size, path.as_posix()),
    )
    benchmark_indices = [
        round(index * (len(size_ordered_paths) - 1) / (benchmark_count - 1))
        for index in range(benchmark_count)
    ]
    benchmark_paths = [size_ordered_paths[index] for index in benchmark_indices]
    seed_everything(benchmark_seed)
    benchmark_model = new_velocity_tracker()
    benchmark_optimizer = new_optimizer(benchmark_model)
    benchmark_scaler = new_scaler()
    benchmark_loader = make_loader(
        benchmark_paths,
        shuffle=False,
        seed=benchmark_seed,
    )
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    training_benchmark_started = time.perf_counter()
    benchmark_train_metrics = train_one_epoch(
        benchmark_model,
        benchmark_loader,
        benchmark_optimizer,
        benchmark_scaler,
        device,
        gamma=gamma,
        gradient_clip_norm=float(train_cfg["gradient_clip_norm"]),
        use_amp=use_amp,
    )
    torch.cuda.synchronize(device)
    training_benchmark_elapsed = time.perf_counter() - training_benchmark_started

    torch.cuda.synchronize(device)
    diagnostic_benchmark_started = time.perf_counter()
    benchmark_diagnostic_metrics = evaluate_tracker_diagnostic(
        benchmark_model,
        benchmark_loader,
        device,
        gamma=gamma,
        use_amp=use_amp,
        edge_probability_threshold=threshold,
    )
    torch.cuda.synchronize(device)
    diagnostic_benchmark_elapsed = time.perf_counter() - diagnostic_benchmark_started

    history_benchmark_sample = sample_names[0]
    history_benchmark_paths = paths_for_samples(
        all_cache_paths,
        [history_benchmark_sample],
    )[:benchmark_count]
    history_benchmark_model = new_base_tracker()
    torch.cuda.synchronize(device)
    history_benchmark_started = time.perf_counter()
    _, history_benchmark_audit = generate_history(
        history_benchmark_model,
        history_benchmark_paths,
    )
    torch.cuda.synchronize(device)
    history_benchmark_elapsed = time.perf_counter() - history_benchmark_started

    epochs = int(train_cfg["epochs"])
    history_epochs = int(history_cfg["inner_generator_epochs"])
    history_training_windows = 0
    history_validation_windows = 0
    for record in splits:
        outer_train_samples = sorted(
            set(record["gradient_update"]) | set(record["internal_validation"])
        )
        inner_target_folds = split_two_folds(
            outer_train_samples,
            int(validation_cfg["history_crossfit"]["split_seed"]) + int(record["fold"]),
        )
        for inner_fold, target_samples in enumerate(inner_target_folds):
            generator_pool = sorted(set(outer_train_samples) - set(target_samples))
            generator_gradient, generator_validation = split_selection_samples(
                generator_pool,
                seed=int(validation_cfg["history_crossfit"]["split_seed"])
                + int(record["fold"]) * 100
                + inner_fold,
                validation_fraction=float(
                    validation_cfg["history_crossfit"]["internal_selection_fraction"]
                ),
            )
            history_training_windows += history_epochs * len(
                paths_for_samples(eligible_cache_paths, generator_gradient)
            )
            history_validation_windows += history_epochs * len(
                paths_for_samples(eligible_cache_paths, generator_validation)
            )
    velocity_training_windows = epochs * sum(
        len(paths_for_samples(eligible_cache_paths, record["gradient_update"])) for record in splits
    )
    velocity_validation_windows = epochs * sum(
        len(paths_for_samples(eligible_cache_paths, record["internal_validation"]))
        for record in splits
    )
    training_work_windows = history_training_windows + velocity_training_windows
    diagnostic_work_windows = (
        history_validation_windows + velocity_validation_windows + 2 * len(eligible_cache_paths)
    )
    history_generation_work_windows = 2 * len(all_cache_paths)
    training_rate = training_benchmark_elapsed / len(benchmark_paths)
    diagnostic_rate = diagnostic_benchmark_elapsed / len(benchmark_paths)
    history_rate = history_benchmark_elapsed / len(history_benchmark_paths)
    projection_multiplier = float(train_cfg["runtime_projection_multiplier"])
    projection_components = {
        "training": {
            "window_count": training_work_windows,
            "seconds_per_window": training_rate,
            "projected_seconds": training_rate * training_work_windows,
        },
        "diagnostic_forward": {
            "window_count": diagnostic_work_windows,
            "seconds_per_window": diagnostic_rate,
            "projected_seconds": diagnostic_rate * diagnostic_work_windows,
        },
        "history_forward": {
            "window_count": history_generation_work_windows,
            "seconds_per_window": history_rate,
            "projected_seconds": history_rate * history_generation_work_windows,
        },
    }
    unreserved_projected_seconds = sum(
        component["projected_seconds"] for component in projection_components.values()
    )
    projected_seconds = unreserved_projected_seconds * projection_multiplier
    benchmark_summary = {
        "sampling": "64_size_quantiles_across_all_eligible_windows",
        "window_count_per_stage": benchmark_count,
        "training_elapsed_seconds": training_benchmark_elapsed,
        "diagnostic_forward_elapsed_seconds": diagnostic_benchmark_elapsed,
        "history_forward_elapsed_seconds": history_benchmark_elapsed,
        "train_metrics": benchmark_train_metrics,
        "diagnostic_metrics": benchmark_diagnostic_metrics,
        "history_audit": history_benchmark_audit,
        "projection_components": projection_components,
        "projection_scope": (
            "four history models, two velocity models, cross-fit and outer history "
            "generation, exp016 and velocity outer diagnostics"
        ),
        "unreserved_projected_seconds_total": unreserved_projected_seconds,
        "conservative_projected_seconds_total": projected_seconds,
        "runtime_gate_seconds": float(train_cfg["runtime_gate_hours"]) * 3600.0,
        "projection_multiplier": projection_multiplier,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    (WORKING_ROOT / "benchmark_summary.json").write_text(
        json.dumps(benchmark_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del benchmark_loader, benchmark_scaler, benchmark_optimizer, benchmark_model
    del history_benchmark_model
    torch.cuda.empty_cache()
    print(json.dumps(benchmark_summary, indent=2))
    if projected_seconds > benchmark_summary["runtime_gate_seconds"]:
        update_metrics(
            METRICS_PATH,
            {
                "status": "failed",
                "notes": "Stopped before full training because the runtime gate failed.",
                "train_stage": {"benchmark": benchmark_summary},
            },
        )
        raise RuntimeError("conservative train projection exceeds the runtime gate")

    fold_summaries: list[dict[str, Any]] = []
    velocity_model_records: list[dict[str, Any]] = []
    history_model_records: list[dict[str, Any]] = []
    teacher_audit_records: list[dict[str, Any]] = []
    training_started = time.perf_counter()

    for record in splits:
        fold = int(record["fold"])
        fold_seed = global_seed + fold
        outer_train_samples = sorted(
            set(record["gradient_update"]) | set(record["internal_validation"])
        )
        inner_target_folds = split_two_folds(
            outer_train_samples,
            int(validation_cfg["history_crossfit"]["split_seed"]) + fold,
        )
        outer_train_history: HistoryByFrame = {}
        outer_history_records: list[dict[str, Any]] = []

        for inner_fold, target_samples in enumerate(inner_target_folds):
            generator_pool = sorted(set(outer_train_samples) - set(target_samples))
            generator_gradient, generator_validation = split_selection_samples(
                generator_pool,
                seed=int(validation_cfg["history_crossfit"]["split_seed"])
                + fold * 100
                + inner_fold,
                validation_fraction=float(
                    validation_cfg["history_crossfit"]["internal_selection_fraction"]
                ),
            )
            if set(target_samples) & (set(generator_gradient) | set(generator_validation)):
                raise RuntimeError("target history samples leaked into generator fitting")
            history_seed = global_seed + 1000 + fold * 10 + inner_fold
            seed_everything(history_seed)
            history_model = new_base_tracker()
            history_state, history_fit = fit_model(
                history_model,
                train_paths=paths_for_samples(
                    eligible_cache_paths,
                    generator_gradient,
                ),
                validation_paths=paths_for_samples(
                    eligible_cache_paths,
                    generator_validation,
                ),
                seed=history_seed,
                epochs=int(history_cfg["inner_generator_epochs"]),
                history_by_frame=None,
            )
            inner_dir = OUTPUT_HISTORY_MODELS / f"outer_{fold}" / f"inner_{inner_fold}"
            inner_dir.mkdir(parents=True, exist_ok=True)
            history_model_path = inner_dir / "history_tracker.pth"
            torch.save(
                {
                    "experiment": EXPERIMENT,
                    "outer_fold": fold,
                    "inner_fold": inner_fold,
                    "target_samples": target_samples,
                    "gradient_samples": generator_gradient,
                    "validation_samples": generator_validation,
                    "state_dict": history_state,
                    **history_fit,
                },
                history_model_path,
            )
            target_paths = paths_for_samples(all_cache_paths, target_samples)
            target_history, history_audit = generate_history(history_model, target_paths)
            overlap = set(outer_train_history) & set(target_history)
            if overlap:
                raise RuntimeError(f"cross-fit history overlap: {sorted(overlap)[:5]}")
            outer_train_history.update(target_history)
            history_record = {
                "outer_fold": fold,
                "inner_fold": inner_fold,
                "target_sample_count": len(target_samples),
                "gradient_sample_count": len(generator_gradient),
                "validation_sample_count": len(generator_validation),
                "target_sample_excluded": True,
                "target_samples_sha256": json_sha256(target_samples),
                "model_path": history_model_path.relative_to(WORKING_ROOT).as_posix(),
                "model_file_sha256": file_sha256(history_model_path),
                "model_state_sha256": canonical_state_sha256(history_state),
                "fit": history_fit,
                "history_audit": history_audit,
            }
            history_model_records.append(history_record)
            outer_history_records.append(history_record)
            del history_model
            torch.cuda.empty_cache()

        expected_outer_history_frames = len(paths_for_samples(all_cache_paths, outer_train_samples))
        if len(outer_train_history) != expected_outer_history_frames:
            raise RuntimeError(
                {
                    "outer_train_history_coverage": {
                        "fold": fold,
                        "expected": expected_outer_history_frames,
                        "actual": len(outer_train_history),
                    }
                }
            )

        control_tracker = load_control_tracker(fold)
        evaluation_all_paths = paths_for_samples(
            all_cache_paths,
            record["outer_evaluation"],
        )
        evaluation_history, evaluation_history_audit = generate_history(
            control_tracker,
            evaluation_all_paths,
        )
        train_paths = paths_for_samples(
            eligible_cache_paths,
            record["gradient_update"],
        )
        validation_paths = paths_for_samples(
            eligible_cache_paths,
            record["internal_validation"],
        )
        evaluation_paths = paths_for_samples(
            eligible_cache_paths,
            record["outer_evaluation"],
        )
        evaluation_loader = make_loader(
            evaluation_paths,
            shuffle=False,
            seed=fold_seed,
            history_by_frame=evaluation_history,
        )
        baseline_outer = evaluate_tracker_diagnostic(
            control_tracker,
            evaluation_loader,
            device,
            gamma=gamma,
            use_amp=use_amp,
            edge_probability_threshold=threshold,
        )
        teacher_audit_records.append(baseline_outer["teacher"])

        seed_everything(fold_seed)
        velocity_tracker = new_velocity_tracker()
        velocity_state, velocity_fit = fit_model(
            velocity_tracker,
            train_paths=train_paths,
            validation_paths=validation_paths,
            seed=fold_seed,
            epochs=epochs,
            history_by_frame=outer_train_history,
        )
        velocity_outer = evaluate_tracker_diagnostic(
            velocity_tracker,
            evaluation_loader,
            device,
            gamma=gamma,
            use_amp=use_amp,
            edge_probability_threshold=threshold,
        )

        fold_dir = OUTPUT_MODELS / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model_path = fold_dir / "velocity_tracker_best.pth"
        torch.save(
            {
                "experiment": EXPERIMENT,
                "fold": fold,
                "train_embryo": record["train_embryo"],
                "evaluation_embryo": record["evaluation_embryo"],
                "state_dict": velocity_state,
                **velocity_fit,
            },
            model_path,
        )
        base_fp = int(baseline_outer["false_positive_active_pair_count"])
        velocity_fp = int(velocity_outer["false_positive_active_pair_count"])
        if base_fp == 0:
            fp_relative_increase = 0.0 if velocity_fp == 0 else None
            fp_ok = velocity_fp == 0
        else:
            fp_relative_increase = (velocity_fp - base_fp) / base_fp
            fp_ok = fp_relative_increase <= float(
                validation_cfg["graph_progression_gate"][
                    "max_false_positive_active_pair_relative_increase"
                ]
            )
        gate_checks = {
            "positive_edge_recall_improved": (
                float(velocity_outer["positive_edge_recall"])
                > float(baseline_outer["positive_edge_recall"])
            ),
            "false_positive_active_pair_increase_within_limit": fp_ok,
            "division_recovered_count_non_decrease": (
                int(velocity_outer["recovered_division_parent_count"])
                >= int(baseline_outer["recovered_division_parent_count"])
            ),
        }
        gate_checks["passed"] = all(gate_checks.values())
        comparison_keys = (
            "legacy_mask_loss",
            "edge_accuracy",
            "positive_edge_recall",
            "false_positive_active_pair_rate",
            "correct_parent_top1_accuracy",
            "division_parent_recall",
            "selection_score",
        )
        fold_summary = {
            "fold": fold,
            "train_embryo": record["train_embryo"],
            "evaluation_embryo": record["evaluation_embryo"],
            "sample_counts": {
                "gradient_update": len(record["gradient_update"]),
                "internal_validation": len(record["internal_validation"]),
                "outer_evaluation": len(record["outer_evaluation"]),
            },
            "window_counts": {
                "gradient_update": len(train_paths),
                "internal_validation": len(validation_paths),
                "outer_evaluation": len(evaluation_paths),
            },
            "history_crossfit": outer_history_records,
            "outer_evaluation_history": evaluation_history_audit,
            "exp016_saved_control": baseline_outer,
            "velocity_fit": velocity_fit,
            "velocity_outer_evaluation": velocity_outer,
            "outer_metric_delta": {
                key: float(velocity_outer[key]) - float(baseline_outer[key])
                for key in comparison_keys
            },
            "false_positive_active_pair_relative_increase": fp_relative_increase,
            "graph_progression_gate": gate_checks,
            "model_file": model_path.relative_to(WORKING_ROOT).as_posix(),
            "model_file_sha256": file_sha256(model_path),
            "model_state_sha256": canonical_state_sha256(velocity_state),
        }
        fold_summary_path = WORKING_ROOT / f"fold_{fold}_summary.json"
        fold_summary_path.write_text(
            json.dumps(fold_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        fold_summaries.append(fold_summary)
        velocity_model_records.append(
            {
                "fold": fold,
                "path": fold_summary["model_file"],
                "file_sha256": fold_summary["model_file_sha256"],
                "canonical_state_sha256": fold_summary["model_state_sha256"],
                "best_epoch": velocity_fit["best_epoch"],
                "train_embryo": record["train_embryo"],
                "evaluation_embryo": record["evaluation_embryo"],
            }
        )
        del evaluation_loader, velocity_tracker, control_tracker
        del outer_train_history, evaluation_history
        torch.cuda.empty_cache()

    if len(history_model_records) != int(model_cfg["output"]["history_model_count"]):
        raise RuntimeError("history model count differs from config")
    if len(velocity_model_records) != int(model_cfg["output"]["model_count"]):
        raise RuntimeError("velocity model count differs from config")

    graph_gate_passed = all(
        bool(fold["graph_progression_gate"]["passed"]) for fold in fold_summaries
    )
    graph_gate = {
        "passed": graph_gate_passed,
        "threshold": threshold,
        "required_fold_count": 2,
        "passed_fold_count": sum(
            bool(fold["graph_progression_gate"]["passed"]) for fold in fold_summaries
        ),
        "folds": {str(fold["fold"]): fold["graph_progression_gate"] for fold in fold_summaries},
        "next_action": (
            "implement_full_graph_inference"
            if graph_gate_passed
            else "stop_before_full_graph_inference"
        ),
        "official_metric_computed": False,
    }
    graph_gate_path = WORKING_ROOT / "graph_progression_gate.json"
    graph_gate_path.write_text(
        json.dumps(graph_gate, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    teacher_audit = aggregate_teacher_stats(teacher_audit_records)
    teacher_audit.update(
        {
            "matching": teacher_cfg["matching"],
            "max_matching_distance_um": max_matching_distance_um,
            "loss_mask": loss_cfg["mask"],
            "outer_evaluation_coverage": "each of 199 samples exactly once across two folds",
        }
    )
    (WORKING_ROOT / "teacher_audit.json").write_text(
        json.dumps(teacher_audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    history_manifest = {
        "experiment": EXPERIMENT,
        "created_at": datetime.now(UTC).isoformat(),
        "association": history_cfg["association"],
        "edge_probability_threshold": threshold,
        "division_policy": history_cfg["division_policy"],
        "target_sample_excluded_from_history_model_training": True,
        "models": history_model_records,
    }
    history_manifest_path = WORKING_ROOT / "history_manifest.json"
    history_manifest_path.write_text(
        json.dumps(history_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    model_manifest = {
        "experiment": EXPERIMENT,
        "created_at": datetime.now(UTC).isoformat(),
        "model_class": "VelocityAugmentedTracker",
        "trainable_components": model_cfg["trainable_components"],
        "frozen_components": model_cfg["frozen_components"],
        "public_checkpoint_file_sha256": observed_source_hashes["checkpoint"],
        "public_tracker_initial_state_sha256": public_tracker_state_sha256,
        "exp016_control_manifest_sha256": file_sha256(control_manifest_path),
        "cache_summary_sha256": cache_summary["summary_sha256"],
        "cache_identity_sha256": cache_summary["cache_identity_sha256"],
        "gt_window_filter_audit_sha256": gt_window_filter_audit_sha256,
        "annotation_content_sha256": json_sha256(annotation_manifest),
        "history_manifest_sha256": file_sha256(history_manifest_path),
        "feature_schema_contract_sha256": json_sha256(
            {
                "cache_schema_version": cache_cfg["schema_version"],
                "arrays": cache_cfg["arrays"],
                "primary_feature_channels": cache_cfg["feature_channels"],
                "tracker_feature_dim": params_cfg["feature_dim"],
                "velocity_pair_features": list(VELOCITY_PAIR_FEATURE_NAMES),
                "history": history_cfg,
            }
        ),
        "models": velocity_model_records,
        "history_models": [
            {
                key: record[key]
                for key in (
                    "outer_fold",
                    "inner_fold",
                    "model_path",
                    "model_file_sha256",
                    "model_state_sha256",
                )
            }
            for record in history_model_records
        ],
    }
    model_manifest_path = WORKING_ROOT / str(model_cfg["output"]["model_manifest"])
    model_manifest_path.write_text(
        json.dumps(model_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model_manifest_sha = file_sha256(model_manifest_path)

    training_summary = {
        "experiment": EXPERIMENT,
        "status": "pair_diagnostic_completed",
        "official_metric_computed": False,
        "submission_created": False,
        "conditional_validation": validation_cfg["conditional_validation"],
        "elapsed_seconds": time.perf_counter() - training_started,
        "notebook_elapsed_seconds": time.perf_counter() - NOTEBOOK_STARTED,
        "benchmark": benchmark_summary,
        "gt_window_filter": {
            **gt_window_filter_summary,
            "audit_sha256": gt_window_filter_audit_sha256,
        },
        "teacher_audit": teacher_audit,
        "folds": fold_summaries,
        "graph_progression_gate": graph_gate,
        "history_manifest_sha256": file_sha256(history_manifest_path),
        "model_manifest_sha256": model_manifest_sha,
    }
    training_summary_path = WORKING_ROOT / str(model_cfg["output"]["training_summary"])
    training_summary_path.write_text(
        json.dumps(training_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    update_metrics(
        METRICS_PATH,
        {
            "status": "running",
            "metric": "fixed_candidate_velocity_pair_diagnostics",
            "evidence": {
                "kaggle": {
                    "kernel_source_ids": [
                        str(cache_cfg["kernel_source"]),
                        str(control_cfg["kernel_source"]),
                        str(public_cfg["dataset_ref"]),
                    ],
                    "resource": torch.cuda.get_device_name(device),
                    "notebook_runtime_seconds": training_summary["notebook_elapsed_seconds"],
                    "internet_enabled": False,
                },
                "artifacts": {
                    "input_file_sha": model_manifest["annotation_content_sha256"],
                    "cache_file_sha": cache_summary["summary_sha256"],
                    "feature_schema_sha": model_manifest["feature_schema_contract_sha256"],
                    "feature_content_sha": cache_summary["cache_identity_sha256"],
                    "gt_window_filter_audit_sha": gt_window_filter_audit_sha256,
                    "history_manifest_sha": file_sha256(history_manifest_path),
                    "model_manifest_sha": model_manifest_sha,
                    "row_count": len(eligible_cache_paths),
                    "group_count": len(sample_names),
                    "feature_count": int(params_cfg["feature_dim"])
                    + int(params_cfg["velocity_pair_feature_dim"]),
                    "model_count": len(velocity_model_records),
                    "history_model_count": len(history_model_records),
                    "model_shas": {
                        str(record["fold"]): record["file_sha256"]
                        for record in velocity_model_records
                    },
                    "history_model_shas": {
                        f"{record['outer_fold']}/{record['inner_fold']}": record[
                            "model_file_sha256"
                        ]
                        for record in history_model_records
                    },
                },
            },
            "train_stage": training_summary,
            "notes": (
                "Velocity pair diagnostic completed against saved exp016 fold models. "
                + (
                    "The pair gate passed; full graph inference remains pending."
                    if graph_gate_passed
                    else "The pair gate failed; full graph inference was not started."
                )
            ),
        },
    )
    print(json.dumps(training_summary, indent=2, default=float))
    print("Model manifest:", model_manifest_path, model_manifest_sha)
    print("Graph progression gate:", json.dumps(graph_gate, indent=2))
    print("No submission was created.")
