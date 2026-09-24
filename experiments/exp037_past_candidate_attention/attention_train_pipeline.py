from __future__ import annotations

import copy
import importlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import runtime_helpers as runtime
import yaml

EXPERIMENT = "exp037_past_candidate_attention"
WORKING_ROOT = Path.cwd()
CONFIG_PATH = WORKING_ROOT / "config.yaml"
METRICS_PATH = WORKING_ROOT / "metrics.json"
OUTPUT_MODELS = WORKING_ROOT / "models"
NOTEBOOK_STARTED = time.perf_counter()


def main() -> None:
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("The authoritative full run must execute on Kaggle.")
    runtime.ensure_geff_runtime_dependencies()

    import torch
    from frozen_tracker import (
        aggregate_teacher_stats,
        build_embryo_splits,
        canonical_state_sha256,
        dataloader_worker_init,
        discover_cache_paths,
        extract_public_tracker_state,
        file_sha256,
        filter_nonempty_gt_window_paths,
        json_sha256,
        load_annotation_graph,
        paths_for_samples,
        recompute_cache_identity_sha256,
        seed_everything,
        validate_cache_summary,
        validate_window_cache,
    )
    from past_candidate_attention import (
        PAST_CANDIDATE_FEATURE_NAMES,
        PastCandidateAttentionTracker,
    )
    from past_candidate_data import (
        PastCandidateWindowDataset,
        collate_past_candidate_examples,
        evaluate_tracker,
        evaluate_tracker_diagnostic,
        train_one_epoch,
    )
    from settings import update_metrics
    from torch.utils.data import DataLoader

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    validation_cfg = config["validation"]
    cache_cfg = config["data"]["cache"]
    model_cfg = config["model"]
    train_cfg = model_cfg["training"]
    attention_cfg = model_cfg["past_candidate_attention"]
    control_cfg = model_cfg["control"]
    teacher_cfg = model_cfg["teacher"]
    loss_cfg = model_cfg["loss"]
    params_cfg = model_cfg["params"]

    expected_trainable = ["primary_SimpleNodeTransformer", "past_candidate_attention_branch"]
    if model_cfg["trainable_components"] != expected_trainable:
        raise RuntimeError("past-candidate tracker trainable-component contract changed")
    if train_cfg["active_variants"] != ["past_candidate_attention"]:
        raise RuntimeError("the train stage requires only past_candidate_attention")
    if control_cfg["retrain"] is not False:
        raise RuntimeError("the exp016 control must not be retrained")
    if int(model_cfg["output"]["model_count"]) != 2:
        raise RuntimeError("the experiment requires exactly two outer-fold models")
    if int(attention_cfg["feature_dim"]) != len(PAST_CANDIDATE_FEATURE_NAMES):
        raise RuntimeError("past-candidate feature width differs from the declared schema")
    if bool(attention_cfg["clip_features"]):
        raise RuntimeError("the agreed feature contract does not clip displacements")

    cache_output_root = runtime.resolve_cache_output_root(cache_cfg)
    cache_summary_path = cache_output_root / str(cache_cfg["summary_file"])
    cache_summary = validate_cache_summary(cache_summary_path, cache_cfg)
    cache_root = cache_output_root / str(cache_cfg["directory"])
    all_cache_paths = discover_cache_paths(cache_root, cache_cfg)
    observed_cache_identity_sha256 = recompute_cache_identity_sha256(all_cache_paths)
    if observed_cache_identity_sha256 != str(cache_cfg["identity_sha256"]):
        raise RuntimeError("exp015 cache identity differs from config")

    public_root = runtime.resolve_public_artifact_root(model_cfg["public_source"])
    train_dir = runtime.resolve_train_dir()
    control_root = runtime.resolve_control_output_root(control_cfg)
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
        all_cache_paths, annotations
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
    split_manifest_path = WORKING_ROOT / "split_manifest.json"
    split_manifest_path.write_text(
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
        source_paths["checkpoint"], map_location="cpu", weights_only=True
    )
    public_tracker_state = extract_public_tracker_state(full_public_state)
    public_tracker_state_sha256 = canonical_state_sha256(public_tracker_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("the authoritative train run requires a Kaggle GPU")
    OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)

    feature_channels = int(cache_cfg["feature_channels"])
    max_matching_distance_um = float(teacher_cfg["max_matching_distance_um"])
    downsample_zyx = tuple(float(value) for value in params_cfg["downsample_zyx"])
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"])
    global_seed = int(train_cfg["seed"])
    use_amp = bool(train_cfg["mixed_precision"])
    gamma = float(loss_cfg["focal_gamma"])
    threshold = float(validation_cfg["graph_progression_gate"]["threshold"])

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

    def new_attention_tracker() -> Any:
        base_tracker = new_base_tracker()
        base_parameter_count = sum(parameter.numel() for parameter in base_tracker.parameters())
        tracker = PastCandidateAttentionTracker(
            base_tracker,
            feature_dim=int(attention_cfg["feature_dim"]),
            hidden_dim=int(attention_cfg["hidden_dim"]),
            source_chunk_size=int(attention_cfg["source_chunk_size"]),
            target_chunk_size=int(attention_cfg["target_chunk_size"]),
            past_candidate_chunk_size=int(attention_cfg["past_candidate_chunk_size"]),
            vector_scale_um=float(attention_cfg["vector_scale_um"]),
            cosine_epsilon=float(attention_cfg["cosine_epsilon"]),
            gradient_checkpointing=bool(attention_cfg["gradient_checkpointing"]),
        ).to(device)
        added_parameter_count = sum(parameter.numel() for parameter in tracker.parameters()) - (
            base_parameter_count
        )
        if added_parameter_count != int(attention_cfg["expected_added_parameter_count"]):
            raise RuntimeError({"past_candidate_parameter_count": added_parameter_count})
        return tracker

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

    def make_loader(paths: list[Path], *, shuffle: bool, seed: int) -> Any:
        dataset = PastCandidateWindowDataset(
            paths,
            annotations,
            feature_channels=feature_channels,
            expected_primary_checkpoint_sha256=str(public_cfg["checkpoint_sha256"]),
            max_matching_distance_um=max_matching_distance_um,
            downsample_zyx=downsample_zyx,
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_past_candidate_examples,
            generator=generator,
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
        model_ids = {id(parameter) for parameter in model.parameters()}
        optimizer_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        if optimizer_ids != model_ids:
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
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        optimizer = new_optimizer(model)
        scaler = new_scaler()
        train_loader = make_loader(train_paths, shuffle=True, seed=seed)
        validation_loader = make_loader(validation_paths, shuffle=False, seed=seed)
        best_score = float("-inf")
        best_epoch = -1
        best_state: dict[str, Any] | None = None
        epoch_records: list[dict[str, Any]] = []
        for epoch in range(int(train_cfg["epochs"])):
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
                model, validation_loader, device, gamma=gamma, use_amp=use_amp
            )
            score = float(validation_metrics["selection_score"])
            selected = score >= best_score
            if selected:
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            record = {
                "epoch": epoch,
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "train": train_metrics,
                "internal_validation": validation_metrics,
                "selected": selected,
            }
            epoch_records.append(record)
            print(json.dumps(record, default=float))
        if best_state is None:
            raise RuntimeError("model training did not select a checkpoint")
        model.load_state_dict(best_state, strict=True)
        del validation_loader, train_loader, scaler, optimizer
        return best_state, {
            "best_epoch": best_epoch,
            "best_internal_selection_score": best_score,
            "epochs": epoch_records,
        }

    def candidate_work(path: Path) -> int:
        arrays, metadata = validate_window_cache(
            path,
            feature_channels=feature_channels,
            expected_primary_checkpoint_sha256=str(public_cfg["checkpoint_sha256"]),
        )
        source_frame = int(metadata["window_frames"][0])
        if source_frame == 0:
            previous_count = 0
        else:
            prior_path = path.with_name(f"{source_frame - 1:06d}_{source_frame:06d}.npz")
            prior_arrays, _ = validate_window_cache(
                prior_path,
                feature_channels=feature_channels,
                expected_primary_checkpoint_sha256=str(public_cfg["checkpoint_sha256"]),
            )
            previous_count = len(prior_arrays["candidate_ids_src"])
        return previous_count * len(arrays["candidate_ids_src"]) * len(arrays["candidate_ids_tgt"])

    benchmark_count = int(train_cfg["benchmark_windows_per_fold"])
    size_ordered_paths = sorted(
        eligible_cache_paths, key=lambda path: (path.stat().st_size, path.as_posix())
    )
    benchmark_indices = [
        round(index * (len(size_ordered_paths) - 1) / (benchmark_count - 1))
        for index in range(benchmark_count)
    ]
    benchmark_paths = [size_ordered_paths[index] for index in benchmark_indices]
    largest_probe_count = int(train_cfg["benchmark_largest_windows"])
    largest_probe_pool = size_ordered_paths[-max(largest_probe_count * 8, largest_probe_count) :]
    largest_paths = sorted(largest_probe_pool, key=candidate_work, reverse=True)[
        :largest_probe_count
    ]
    benchmark_paths = list(dict.fromkeys([*benchmark_paths, *largest_paths]))
    benchmark_work = {path.as_posix(): candidate_work(path) for path in benchmark_paths}

    benchmark_seed = global_seed + 9000
    seed_everything(benchmark_seed)
    benchmark_model = new_attention_tracker()
    benchmark_optimizer = new_optimizer(benchmark_model)
    benchmark_scaler = new_scaler()
    benchmark_loader = make_loader(benchmark_paths, shuffle=False, seed=benchmark_seed)
    torch.cuda.reset_peak_memory_stats(device)
    benchmark_train_started = time.perf_counter()
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
    benchmark_train_seconds = time.perf_counter() - benchmark_train_started
    benchmark_eval_started = time.perf_counter()
    benchmark_eval_metrics = evaluate_tracker_diagnostic(
        benchmark_model,
        benchmark_loader,
        device,
        gamma=gamma,
        use_amp=use_amp,
        edge_probability_threshold=threshold,
    )
    benchmark_eval_seconds = time.perf_counter() - benchmark_eval_started
    peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated(device))

    epochs = int(train_cfg["epochs"])
    train_windows = epochs * sum(
        len(paths_for_samples(eligible_cache_paths, record["gradient_update"])) for record in splits
    )
    evaluation_windows = sum(
        epochs * len(paths_for_samples(eligible_cache_paths, record["internal_validation"]))
        + 2 * len(paths_for_samples(eligible_cache_paths, record["outer_evaluation"]))
        for record in splits
    )
    train_rate = benchmark_train_seconds / len(benchmark_paths)
    eval_rate = benchmark_eval_seconds / len(benchmark_paths)
    raw_projected_seconds = train_rate * train_windows + eval_rate * evaluation_windows
    projected_seconds = raw_projected_seconds * float(train_cfg["runtime_projection_multiplier"])
    runtime_gate_seconds = float(train_cfg["runtime_gate_hours"]) * 3600.0
    benchmark_summary = {
        "window_count": len(benchmark_paths),
        "largest_window_count": largest_probe_count,
        "candidate_product_by_window": benchmark_work,
        "max_candidate_product": max(benchmark_work.values()),
        "train_elapsed_seconds": benchmark_train_seconds,
        "evaluation_elapsed_seconds": benchmark_eval_seconds,
        "train_seconds_per_window": train_rate,
        "evaluation_seconds_per_window": eval_rate,
        "projected_seconds_before_multiplier": raw_projected_seconds,
        "projection_multiplier": float(train_cfg["runtime_projection_multiplier"]),
        "conservative_projected_seconds": projected_seconds,
        "runtime_gate_seconds": runtime_gate_seconds,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "train_metrics": benchmark_train_metrics,
        "evaluation_metrics": benchmark_eval_metrics,
    }
    (WORKING_ROOT / "runtime_benchmark.json").write_text(
        json.dumps(benchmark_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del benchmark_loader, benchmark_model, benchmark_optimizer, benchmark_scaler
    torch.cuda.empty_cache()
    if projected_seconds > runtime_gate_seconds:
        update_metrics(
            METRICS_PATH,
            {
                "status": "failed",
                "evidence": {"benchmark": benchmark_summary},
                "notes": "Runtime gate failed before full two-fold training.",
            },
        )
        raise RuntimeError("projected runtime exceeds the 12-hour gate")

    training_started = time.perf_counter()
    fold_summaries: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    teacher_audit_records: list[dict[str, Any]] = []
    for split in splits:
        fold = int(split["fold"])
        fold_seed = global_seed + fold
        train_paths = paths_for_samples(eligible_cache_paths, split["gradient_update"])
        validation_paths = paths_for_samples(eligible_cache_paths, split["internal_validation"])
        evaluation_paths = paths_for_samples(eligible_cache_paths, split["outer_evaluation"])
        evaluation_loader = make_loader(evaluation_paths, shuffle=False, seed=fold_seed)
        control_tracker = load_control_tracker(fold)
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
        attention_tracker = new_attention_tracker()
        attention_state, attention_fit = fit_model(
            attention_tracker,
            train_paths=train_paths,
            validation_paths=validation_paths,
            seed=fold_seed,
        )
        attention_outer = evaluate_tracker_diagnostic(
            attention_tracker,
            evaluation_loader,
            device,
            gamma=gamma,
            use_amp=use_amp,
            edge_probability_threshold=threshold,
        )
        fold_dir = OUTPUT_MODELS / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model_path = fold_dir / "past_candidate_attention_tracker_best.pth"
        torch.save(
            {
                "experiment": EXPERIMENT,
                "fold": fold,
                "train_embryo": split["train_embryo"],
                "evaluation_embryo": split["evaluation_embryo"],
                "state_dict": attention_state,
                **attention_fit,
            },
            model_path,
        )
        base_fp = int(baseline_outer["false_positive_active_pair_count"])
        attention_fp = int(attention_outer["false_positive_active_pair_count"])
        if base_fp == 0:
            fp_relative_increase = 0.0 if attention_fp == 0 else None
            fp_ok = attention_fp == 0
        else:
            fp_relative_increase = (attention_fp - base_fp) / base_fp
            fp_ok = fp_relative_increase <= float(
                validation_cfg["graph_progression_gate"][
                    "max_false_positive_active_pair_relative_increase"
                ]
            )
        gate_checks = {
            "positive_edge_recall_improved": float(attention_outer["positive_edge_recall"])
            > float(baseline_outer["positive_edge_recall"]),
            "false_positive_active_pair_increase_within_limit": fp_ok,
            "division_recovered_count_non_decrease": int(
                attention_outer["recovered_division_parent_count"]
            )
            >= int(baseline_outer["recovered_division_parent_count"]),
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
            "train_embryo": split["train_embryo"],
            "evaluation_embryo": split["evaluation_embryo"],
            "sample_counts": {
                key: len(split[key])
                for key in ("gradient_update", "internal_validation", "outer_evaluation")
            },
            "window_counts": {
                "gradient_update": len(train_paths),
                "internal_validation": len(validation_paths),
                "outer_evaluation": len(evaluation_paths),
            },
            "exp016_saved_control": baseline_outer,
            "attention_fit": attention_fit,
            "attention_outer_evaluation": attention_outer,
            "outer_metric_delta": {
                key: float(attention_outer[key]) - float(baseline_outer[key])
                for key in comparison_keys
            },
            "false_positive_active_pair_relative_increase": fp_relative_increase,
            "graph_progression_gate": gate_checks,
            "model_file": model_path.relative_to(WORKING_ROOT).as_posix(),
            "model_file_sha256": file_sha256(model_path),
            "model_state_sha256": canonical_state_sha256(attention_state),
        }
        fold_summary_path = WORKING_ROOT / f"fold_{fold}_summary.json"
        fold_summary_path.write_text(
            json.dumps(fold_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        fold_summaries.append(fold_summary)
        model_records.append(
            {
                "fold": fold,
                "path": fold_summary["model_file"],
                "file_sha256": fold_summary["model_file_sha256"],
                "canonical_state_sha256": fold_summary["model_state_sha256"],
                "best_epoch": attention_fit["best_epoch"],
                "train_embryo": split["train_embryo"],
                "evaluation_embryo": split["evaluation_embryo"],
            }
        )
        del evaluation_loader, attention_tracker, control_tracker
        torch.cuda.empty_cache()

    if len(model_records) != int(model_cfg["output"]["model_count"]):
        raise RuntimeError("trained model count differs from config")
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
            "request_user_approval_for_full_graph_inference"
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
    teacher_audit_path = WORKING_ROOT / "teacher_audit.json"
    teacher_audit_path.write_text(
        json.dumps(teacher_audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    feature_schema = {
        "cache_schema_version": cache_cfg["schema_version"],
        "arrays": cache_cfg["arrays"],
        "primary_feature_channels": cache_cfg["feature_channels"],
        "tracker_feature_dim": params_cfg["feature_dim"],
        "past_candidate_features": list(PAST_CANDIDATE_FEATURE_NAMES),
        "past_candidate_attention": attention_cfg,
    }
    model_manifest = {
        "experiment": EXPERIMENT,
        "created_at": datetime.now(UTC).isoformat(),
        "model_class": "PastCandidateAttentionTracker",
        "trainable_components": model_cfg["trainable_components"],
        "frozen_components": model_cfg["frozen_components"],
        "public_checkpoint_file_sha256": observed_source_hashes["checkpoint"],
        "public_tracker_initial_state_sha256": public_tracker_state_sha256,
        "exp016_control_manifest_sha256": file_sha256(control_manifest_path),
        "cache_summary_sha256": cache_summary["summary_sha256"],
        "cache_identity_sha256": cache_summary["cache_identity_sha256"],
        "gt_window_filter_audit_sha256": gt_window_filter_audit_sha256,
        "annotation_content_sha256": json_sha256(annotation_manifest),
        "feature_schema_contract_sha256": json_sha256(feature_schema),
        "models": model_records,
    }
    model_manifest_path = WORKING_ROOT / str(model_cfg["output"]["model_manifest"])
    model_manifest_path.write_text(
        json.dumps(model_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model_manifest_sha = file_sha256(model_manifest_path)
    training_summary = {
        "experiment": EXPERIMENT,
        "status": "debug_completed",
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
            "status": "debug_completed",
            "metric": "fixed_candidate_past_candidate_attention_pair_diagnostics",
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
                    "model_manifest_sha": model_manifest_sha,
                    "row_count": len(eligible_cache_paths),
                    "group_count": len(sample_names),
                    "feature_count": int(params_cfg["feature_dim"])
                    + int(attention_cfg["feature_dim"]),
                    "model_count": len(model_records),
                    "model_shas": {
                        str(record["fold"]): record["file_sha256"] for record in model_records
                    },
                },
            },
            "train_stage": training_summary,
            "notes": (
                "Past-candidate attention pair diagnostic completed against saved exp016 "
                "fold models. "
                + (
                    "The pair gate passed; full graph inference requires user approval."
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


if __name__ == "__main__":
    main()
