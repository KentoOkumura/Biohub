"""Kaggle pair-cache audit and matched exp048 primary training modes."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import frozen_tracker as labels
from primary_pair_attention import PrimaryPairFeatureAttention
from x138_data import X138PairDataset, known_parent_retention, read_capture
from x138_tracking import X138PairFeatureTracker, x138_fuse_logits


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=float) + "\n")


def _as_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _model_logits(model: X138PairFeatureTracker, batch: dict[str, Any]) -> torch.Tensor:
    return model(
        batch["features_src"],
        batch["features_tgt"],
        batch["coords_src"],
        batch["coords_tgt"],
        batch["features_prev"],
        batch["coords_prev_physical"],
        batch["coords_src_physical"],
        batch["coords_tgt_physical"],
        batch["prev_mask"],
        batch["candidate_ids_prev"],
        batch["secondary_logits"],
    )


def _metrics(
    dataset: X138PairDataset,
    model: X138PairFeatureTracker | None,
    device: torch.device,
    threshold: float,
    diagnostic_thresholds: tuple[float, ...],
    density_radius_um: float,
    *,
    mask_history: bool = False,
) -> dict[str, Any]:
    counts = {
        key: 0
        for key in (
            "windows",
            "active_pairs",
            "active_pair_errors",
            "known_edges",
            "recovered_known_edges",
            "false_edges_on_active_pairs",
            "division_parents",
            "recovered_division_parents",
        )
    }
    threshold_recovered = {str(value): 0 for value in diagnostic_thresholds}
    density = {
        name: {"known_edges": 0, "recovered_known_edges": 0}
        for name in ("isolated", "one_or_two_neighbors", "three_plus_neighbors")
    }
    rank_sum = margin_sum = top1_edges = margin_count = 0.0
    if model is not None:
        model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
            batch = _as_device(batch, device)
            if mask_history:
                batch["prev_mask"] = torch.zeros_like(batch["prev_mask"])
            logits = batch["x138_fused_logits"] if model is None else _model_logits(model, batch)
            truth = batch["target"][0].bool()
            probabilities = torch.softmax(logits[0].float(), dim=0)
            predicted = probabilities > threshold
            for value in diagnostic_thresholds:
                threshold_recovered[str(value)] += int(((probabilities > value) & truth).sum())
            source_coords = batch["coords_src_physical"][0].float()
            neighbor_count = (torch.cdist(source_coords, source_coords) <= density_radius_um).sum(
                dim=1
            ) - 1
            for source, target in truth.nonzero(as_tuple=False).tolist():
                scores = logits[0, :, target].float()
                true_score = scores[source]
                rank = 1 + int((scores > true_score).sum())
                rank_sum += rank
                top1_edges += int(rank == 1)
                if len(scores) > 1:
                    others = torch.cat((scores[:source], scores[source + 1 :]))
                    margin_sum += float(true_score - others.max())
                    margin_count += 1
                neighbors = int(neighbor_count[source])
                key = (
                    "isolated"
                    if neighbors == 0
                    else "one_or_two_neighbors"
                    if neighbors <= 2
                    else "three_plus_neighbors"
                )
                density[key]["known_edges"] += 1
                density[key]["recovered_known_edges"] += int(predicted[source, target])
            active_rows = truth.any(dim=1)
            active_cols = truth.any(dim=0)
            active = active_rows[:, None] | active_cols[None, :]
            division = truth.sum(dim=1) > 1
            counts["windows"] += 1
            counts["active_pairs"] += int(active.sum())
            counts["active_pair_errors"] += int(((predicted != truth) & active).sum())
            counts["known_edges"] += int(truth.sum())
            counts["recovered_known_edges"] += int((predicted & truth).sum())
            counts["false_edges_on_active_pairs"] += int((predicted & ~truth & active).sum())
            counts["division_parents"] += int(division.sum())
            counts["recovered_division_parents"] += int(
                ((predicted | ~truth).all(dim=1) & division).sum()
            )
    counts["known_edge_recall"] = labels.safe_ratio(
        counts["recovered_known_edges"], counts["known_edges"]
    )
    counts["active_pair_error_rate"] = labels.safe_ratio(
        counts["active_pair_errors"], counts["active_pairs"]
    )
    counts["division_parent_recall"] = labels.safe_ratio(
        counts["recovered_division_parents"], counts["division_parents"]
    )
    counts["teacher_masked_negative_predictions"] = counts["false_edges_on_active_pairs"]
    counts["mean_known_parent_rank"] = labels.safe_ratio(rank_sum, counts["known_edges"])
    counts["top1_known_parent_edges"] = int(top1_edges)
    counts["mean_known_edge_margin"] = labels.safe_ratio(margin_sum, margin_count)
    counts["threshold_recovered_known_edges"] = threshold_recovered
    counts["density"] = density
    return counts


def _forecast_bucket_specs(
    workloads: dict[Path, int],
    fit_paths: list[Path],
    largest_path: Path,
    quantiles: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Choose the largest observed pair in each fixed workload quantile."""
    if not fit_paths or len(quantiles) < 3 or quantiles[0] != 0 or quantiles[-1] != 1:
        raise ValueError("Invalid fit paths or forecast quantiles")
    if any(left >= right for left, right in zip(quantiles, quantiles[1:], strict=False)):
        raise ValueError("Forecast quantiles must increase strictly")
    ordered = sorted(fit_paths, key=lambda path: (workloads[path], str(path)))
    result = []
    start = 0
    for lower, upper in zip(quantiles, quantiles[1:], strict=False):
        stop = min(len(ordered), math.ceil(upper * len(ordered)))
        if stop <= start:
            continue
        representative = largest_path if upper == 1 else ordered[stop - 1]
        result.append(
            {
                "lower_quantile": lower,
                "upper_quantile": upper,
                "representative_path": str(representative),
                "maximum_workload": workloads[representative],
                "fit_batches": stop - start,
            }
        )
        start = stop
    if start != len(ordered) or sum(row["fit_batches"] for row in result) != len(fit_paths):
        raise RuntimeError("Forecast buckets lost fit windows")
    return result


def train_from_capture(
    *,
    config: dict[str, Any],
    cache_root: Path,
    train_dir: Path,
    public_repo: Path,
    working_dir: Path,
    groups: dict[str, list[str]],
    capture_seconds: float,
) -> None:
    if not Path("/kaggle/input").is_dir() or not torch.cuda.is_available():
        raise RuntimeError("The authoritative train run requires Kaggle GPU")
    training = config["model"]["training"]
    attention = config["model"]["attention"]
    validation = config["validation"]
    stems = sorted(stem for group in groups.values() for stem in group)
    paths_by_stem = {stem: sorted((cache_root / stem).glob("*.npz")) for stem in stems}
    if any(not paths for paths in paths_by_stem.values()):
        raise RuntimeError("One or more selected videos have no captured pairs")
    for stem, paths in paths_by_stem.items():
        frames = [read_capture(path)["window_frames"].tolist() for path in paths]
        if frames[0][0] != 0 or any(
            previous[1] != current[0] for previous, current in zip(frames, frames[1:], strict=False)
        ):
            raise RuntimeError(f"Incomplete or unordered captured pair windows: {stem}")
    capture_digest = hashlib.sha256()
    for stem in stems:
        for path in paths_by_stem[stem]:
            capture_digest.update(stem.encode())
            capture_digest.update(path.name.encode())
            capture_digest.update(labels.file_sha256(path).encode())
    capture_sha256 = capture_digest.hexdigest()
    feature_keys = (
        "primary_features_src",
        "position_features_src",
        "primary_features_tgt",
        "position_features_tgt",
    )
    feature_schema = {key: {"dtype": "float32", "width": 32} for key in feature_keys}
    feature_schema_sha256 = hashlib.sha256(
        json.dumps(feature_schema, sort_keys=True).encode()
    ).hexdigest()
    feature_digest = hashlib.sha256()
    for stem in stems:
        for path in paths_by_stem[stem]:
            arrays = read_capture(path)
            feature_digest.update(f"{stem}/{path.name}".encode())
            for key in feature_keys:
                value = np.ascontiguousarray(arrays[key])
                if value.dtype != np.float32 or value.shape[1] != 32:
                    raise RuntimeError(f"Feature schema differs: {path} {key}")
                feature_digest.update(key.encode())
                feature_digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
                feature_digest.update(value.tobytes())
    feature_content_sha256 = feature_digest.hexdigest()
    annotations = {
        stem: labels.load_annotation_graph(
            train_dir / f"{stem}.geff",
            tuple(float(x) for x in config["data"]["native_spacing_zyx_um"]),
        )
        for stem in stems
    }
    eligible = {
        stem: [
            path
            for path in paths_by_stem[stem]
            if all(
                int(frame) in annotations[stem].frames
                for frame in read_capture(path)["window_frames"]
            )
        ]
        for stem in stems
    }
    if any(not paths for paths in eligible.values()):
        raise RuntimeError("No annotated pair windows for one or more videos")
    retention = {}
    for embryo, group in groups.items():
        recovered = total = 0
        for stem in group:
            for path in eligible[stem]:
                found, n = known_parent_retention(
                    read_capture(path),
                    annotations[stem],
                    float(config["data"]["candidate_match_radius_um"]),
                    int(attention["max_past_candidates"]),
                )
                recovered += found
                total += n
        retention[embryo] = {
            "recovered": recovered,
            "eligible": total,
            "recall": labels.safe_ratio(recovered, total),
        }
    _write_json(working_dir / "candidate_retention.json", retention)
    if any(
        not row["eligible"] or row["recall"] < float(validation["candidate_retention_minimum"])
        for row in retention.values()
    ):
        raise RuntimeError("K=8 known-parent retention failed before training")

    source_path = public_repo / "src"
    sys.path.insert(0, str(source_path))
    Transformer = importlib.import_module(
        "biohub_tracking.models.simple_node_transformer"
    ).SimpleNodeTransformer
    checkpoint = public_repo / "weights/unet_transformer/split_0/edge_predictor_best.pth"
    actual_sha = labels.file_sha256(checkpoint)
    if actual_sha != config["model"]["primary_checkpoint_sha256"]:
        raise RuntimeError("Public primary checkpoint SHA differs")
    state = labels.extract_public_tracker_state(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    device = torch.device("cuda:0")
    seed = int(training["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    def new_model(mode: str) -> X138PairFeatureTracker:
        params = config["model"]["primary_params"]
        base = Transformer(**params)
        base.load_state_dict(state, strict=True)
        if mode == "primary_with_past_feature_attention":
            primary = PrimaryPairFeatureAttention(
                base,
                max_past_candidates=int(attention["max_past_candidates"]),
                hidden_dim=int(attention["hidden_dim"]),
                n_heads=int(attention["n_heads"]),
                source_chunk_size=int(attention["source_chunk_size"]),
                target_chunk_size=int(attention["target_chunk_size"]),
                vector_scale_um=float(attention["vector_scale_um"]),
                cosine_epsilon=float(attention["cosine_epsilon"]),
            )
        elif mode == "primary_only":
            primary = base
        else:
            raise ValueError(f"Unexpected train mode: {mode}")
        fusion = {
            key: float(value)
            for key, value in config["model"]["fusion"].items()
            if key != "secondary_link_mode"
        }
        if config["model"]["fusion"]["secondary_link_mode"] != "low_margin_consensus":
            raise RuntimeError("x138 secondary fusion mode changed")
        return X138PairFeatureTracker(primary, mode=mode, fusion_kwargs=fusion).to(device)

    reference = new_model("primary_with_past_feature_attention").eval()
    parity = {
        key: 0.0
        for key in (
            "primary_forward_max_abs",
            "primary_reverse_max_abs",
            "fusion_replay_max_abs",
            "zero_adapter_full_max_abs",
            "all_past_mask_max_abs",
        )
    }
    parity["paths"] = []
    for group in groups.values():
        stem = group[0]
        choices = eligible[stem]
        for path in dict.fromkeys((choices[0], choices[len(choices) // 2], choices[-1])):
            sample = X138PairDataset([path], annotations)
            batch = _as_device(next(iter(DataLoader(sample, batch_size=1))), device)
            with torch.no_grad():
                forward, reverse = reference.primary(
                    batch["features_src"],
                    batch["features_tgt"],
                    batch["coords_src"],
                    batch["coords_tgt"],
                    features_previous=batch["features_prev"],
                    coordinates_previous_physical=batch["coords_prev_physical"],
                    ids_previous=batch["candidate_ids_prev"],
                    mask_previous=batch["prev_mask"],
                    coordinates_parent_physical=batch["coords_src_physical"],
                    coordinates_child_physical=batch["coords_tgt_physical"],
                )
                fused = _model_logits(reference, batch)
                replay = x138_fuse_logits(
                    batch["primary_forward_logits"],
                    batch["primary_reverse_logits"],
                    batch["secondary_logits"],
                    **reference.fusion_kwargs,
                )
                no_history_batch = dict(batch)
                no_history_batch["prev_mask"] = torch.zeros_like(batch["prev_mask"])
                masked = _model_logits(reference, no_history_batch)
            observed = {
                "primary_forward_max_abs": float(
                    (forward - batch["primary_forward_logits"]).abs().max()
                ),
                "primary_reverse_max_abs": float(
                    (reverse - batch["primary_reverse_logits"]).abs().max()
                ),
                "fusion_replay_max_abs": float((replay - batch["x138_fused_logits"]).abs().max()),
                "zero_adapter_full_max_abs": float(
                    (fused - batch["x138_fused_logits"]).abs().max()
                ),
                "all_past_mask_max_abs": float((fused - masked).abs().max()),
            }
            for key, value in observed.items():
                parity[key] = max(parity[key], value)
            parity["paths"].append(str(path))
    _write_json(working_dir / "baseline_parity.json", parity)
    if max(value for key, value in parity.items() if key.endswith("max_abs")) > float(
        validation["baseline_logit_max_abs_error"]
    ):
        raise RuntimeError("Exp043 pair-logit replay failed before training")

    # Benchmark the maximum pair for memory and the upper pair in each size bucket
    # for wall time. Exp048 v1 multiplied the maximum-pair time by every fit window,
    # although the median pair has much less work; that rejected the run before fitting.
    workload = {}
    for path in (path for subset in eligible.values() for path in subset):
        arrays = read_capture(path)
        sources = len(arrays["candidate_ids_src"])
        targets = len(arrays["candidate_ids_tgt"])
        workload[path] = sources * sources + targets * targets + sources * targets
    largest_path = max(workload, key=lambda path: (workload[path], str(path)))
    internal_count = int(training["internal_validation_videos_per_embryo"])
    fold_paths = {}
    for held_out in groups:
        train_stems = sorted(set(stems) - set(groups[held_out]))
        internal_stems = train_stems[:internal_count]
        fit_stems = train_stems[internal_count:]
        fold_paths[held_out] = {
            "fit": [path for stem in fit_stems for path in eligible[stem]],
            "internal": [path for stem in internal_stems for path in eligible[stem]],
            "held": [path for stem in groups[held_out] for path in eligible[stem]],
        }
    fit_paths = [path for fold in fold_paths.values() for path in fold["fit"]]
    buckets = _forecast_bucket_specs(
        workload,
        fit_paths,
        largest_path,
        tuple(float(x) for x in config["runtime"]["forecast_size_quantiles"]),
    )
    threshold = float(validation["edge_probability_threshold"])
    diagnostic_thresholds = tuple(float(x) for x in validation["diagnostic_thresholds"])
    density_radius_um = float(validation["density_radius_um"])
    peak_memory_bytes = {mode: 0 for mode in training["active_modes"]}
    for bucket in buckets:
        path = Path(bucket["representative_path"])
        sample = X138PairDataset([path], annotations)
        bucket["workload"] = workload[path]
        bucket["batch_seconds"] = {}
        bucket["evaluation_seconds"] = {}
        for mode in training["active_modes"]:
            torch.manual_seed(seed)
            bench_model = new_model(mode).train()
            optimizer = torch.optim.AdamW(
                bench_model.parameters(),
                lr=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
            )
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            bench_started = time.monotonic()
            bench_batch = _as_device(next(iter(DataLoader(sample, batch_size=1))), device)
            bench_logits = _model_logits(bench_model, bench_batch)
            bench_loss = labels.legacy_focal_bce(
                bench_logits[0].float(),
                bench_batch["target"][0].float(),
                gamma=float(training["focal_gamma"]),
            )
            bench_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                bench_model.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()
            torch.cuda.synchronize(device)
            bucket["batch_seconds"][mode] = time.monotonic() - bench_started
            peak_memory_bytes[mode] = max(
                peak_memory_bytes[mode], torch.cuda.max_memory_allocated(device)
            )
            bench_model.eval()
            torch.cuda.synchronize(device)
            eval_started = time.monotonic()
            _metrics(
                sample,
                bench_model,
                device,
                threshold,
                diagnostic_thresholds,
                density_radius_um,
            )
            torch.cuda.synchronize(device)
            bucket["evaluation_seconds"][mode] = time.monotonic() - eval_started
            del bench_model, optimizer, bench_batch, bench_logits, bench_loss
            torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        baseline_started = time.monotonic()
        _metrics(sample, None, device, threshold, diagnostic_thresholds, density_radius_um)
        torch.cuda.synchronize(device)
        bucket["baseline_evaluation_seconds"] = time.monotonic() - baseline_started
    bounds = [row["maximum_workload"] for row in buckets]

    def bucket_for(path: Path) -> int:
        size = workload[path]
        for index, bound in enumerate(bounds):
            if size <= bound:
                return index
        raise RuntimeError(f"Pair workload exceeds benchmark maximum: {path}")

    forecast_fit_seconds = 0.0
    forecast_evaluation_seconds = 0.0
    for fold in fold_paths.values():
        for mode in training["active_modes"]:
            forecast_fit_seconds += int(training["epochs"]) * sum(
                buckets[bucket_for(path)]["batch_seconds"][mode] for path in fold["fit"]
            )
            forecast_evaluation_seconds += int(training["epochs"]) * sum(
                buckets[bucket_for(path)]["evaluation_seconds"][mode] for path in fold["internal"]
            )
            forecast_evaluation_seconds += sum(
                buckets[bucket_for(path)]["evaluation_seconds"][mode] for path in fold["held"]
            )
        forecast_evaluation_seconds += sum(
            buckets[bucket_for(path)]["evaluation_seconds"]["primary_with_past_feature_attention"]
            + buckets[bucket_for(path)]["baseline_evaluation_seconds"]
            for path in fold["held"]
        )
    forecast_training_seconds = forecast_fit_seconds + forecast_evaluation_seconds
    safety_factor = float(config["runtime"]["forecast_safety_factor"])
    conservative_total = safety_factor * (capture_seconds + forecast_training_seconds)
    forecast = {
        "method": "measured_upper_pair_per_fixed_workload_quantile",
        "workload_formula": "sources^2 + targets^2 + sources*targets",
        "largest_path": str(largest_path),
        "largest_pair_counts": {
            "sources": int(read_capture(largest_path)["candidate_ids_src"].shape[0]),
            "targets": int(read_capture(largest_path)["candidate_ids_tgt"].shape[0]),
            "previous": int(read_capture(largest_path)["candidate_ids_prev"].shape[0]),
        },
        "buckets": buckets,
        "peak_memory_bytes": peak_memory_bytes,
        "fit_batches_per_mode_across_folds": len(fit_paths),
        "epochs": int(training["epochs"]),
        "mode_count": len(training["active_modes"]),
        "capture_seconds": capture_seconds,
        "forecast_fit_seconds": forecast_fit_seconds,
        "forecast_evaluation_seconds": forecast_evaluation_seconds,
        "forecast_training_seconds": forecast_training_seconds,
        "conservative_total_seconds": conservative_total,
    }
    _write_json(working_dir / "runtime_forecast.json", forecast)
    print(
        "Exp048 runtime forecast:",
        json.dumps(
            {
                "capture_seconds": capture_seconds,
                "forecast_fit_seconds": forecast_fit_seconds,
                "forecast_evaluation_seconds": forecast_evaluation_seconds,
                "conservative_total_seconds": conservative_total,
                "peak_memory_bytes": peak_memory_bytes,
            }
        ),
        flush=True,
    )
    if (
        safety_factor * forecast_training_seconds > float(training["budget_minutes"]) * 60
        or conservative_total > float(config["runtime"]["maximum_notebook_minutes"]) * 60
    ):
        raise RuntimeError("Exp048 control-inclusive runtime forecast exceeds budget")

    started = time.monotonic()
    deadline = started + 60 * float(training["budget_minutes"])
    records = {}
    output_models = working_dir / "models"
    output_models.mkdir(exist_ok=True)
    active_modes = list(training["active_modes"])
    if active_modes != ["primary_only", "primary_with_past_feature_attention"]:
        raise RuntimeError("Matched control and history modes must both be active")

    def evaluate(dataset, model, *, mask_history=False):
        return _metrics(
            dataset,
            model,
            device,
            threshold,
            diagnostic_thresholds,
            density_radius_um,
            mask_history=mask_history,
        )

    for fold, held_out in enumerate(groups):
        train_stems = sorted(set(stems) - set(groups[held_out]))
        internal_count = int(training["internal_validation_videos_per_embryo"])
        internal_stems = train_stems[:internal_count]
        fit_stems = train_stems[internal_count:]
        if len(internal_stems) != internal_count or not fit_stems:
            raise RuntimeError("Invalid internal video split")

        def make_dataset(subset: list[str]) -> X138PairDataset:
            return X138PairDataset(
                [path for stem in subset for path in eligible[stem]],
                annotations,
                match_radius_um=float(config["data"]["candidate_match_radius_um"]),
            )

        fit_data = make_dataset(fit_stems)
        internal_data = make_dataset(internal_stems)
        held_data = make_dataset(groups[held_out])
        records[held_out] = {
            "fold": fold,
            "fit_videos": fit_stems,
            "internal_videos": internal_stems,
            "outer_videos": groups[held_out],
            "public_primary": evaluate(held_data, None),
            "modes": {},
        }
        for mode in active_modes:
            torch.manual_seed(seed + fold * 1000)
            np.random.seed(seed + fold * 1000)
            random.seed(seed + fold * 1000)
            model = new_model(mode)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
            )
            best_key: tuple[float, int, int] | None = None
            best_state: dict[str, torch.Tensor] | None = None
            epoch_records = []
            for epoch in range(int(training["epochs"])):
                model.train()
                losses = []
                generator = torch.Generator().manual_seed(seed + fold * 100 + epoch)
                loader = DataLoader(
                    fit_data,
                    batch_size=1,
                    shuffle=True,
                    generator=generator,
                    num_workers=0,
                )
                for raw_batch in loader:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Tracker training budget exceeded")
                    item = _as_device(raw_batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = _model_logits(model, item)
                    loss = labels.legacy_focal_bce(
                        logits[0].float(),
                        item["target"][0].float(),
                        gamma=float(training["focal_gamma"]),
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite tracker loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(training["gradient_clip_norm"])
                    )
                    optimizer.step()
                    losses.append(float(loss.detach()))
                internal = evaluate(internal_data, model)
                selection_key = (
                    -float(internal["active_pair_error_rate"]),
                    int(internal["recovered_known_edges"]),
                    epoch,
                )
                if best_key is None or selection_key > best_key:
                    best_key = selection_key
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                epoch_records.append(
                    {
                        "epoch": epoch,
                        "mean_train_loss": float(np.mean(losses)),
                        "internal": internal,
                    }
                )
                _write_json(working_dir / f"fold_{fold}_{mode}_progress.json", epoch_records)
            if best_state is None:
                raise RuntimeError(f"No checkpoint selected for {mode}")
            model.load_state_dict(best_state)
            model_path = output_models / f"fold_{fold}_{mode}.pt"
            torch.save(
                {
                    "state_dict": best_state,
                    "mode": mode,
                    "held_out_embryo": held_out,
                    "primary_checkpoint_sha256": actual_sha,
                },
                model_path,
            )
            mode_record = {
                "outer": evaluate(held_data, model),
                "checkpoint_sha256": labels.file_sha256(model_path),
                "epochs": epoch_records,
                "selected_epoch": int(best_key[2]),
            }
            if mode == "primary_with_past_feature_attention":
                mode_record["outer_all_past_masked"] = evaluate(held_data, model, mask_history=True)
                inspected = None
                for raw in DataLoader(held_data, batch_size=1, shuffle=False):
                    example = _as_device(raw, device)
                    positive = example["target"][0].nonzero(as_tuple=False)
                    if positive.numel() == 0 or not example["prev_mask"].any():
                        continue
                    source, target = (int(value) for value in positive[0].tolist())
                    inspected = {
                        "path": example["path"][0],
                        "source_index": source,
                        "target_index": target,
                        "attention": model.primary.inspect_pair(
                            example["features_src"],
                            example["features_tgt"],
                            example["coords_src"],
                            example["coords_tgt"],
                            features_previous=example["features_prev"],
                            coordinates_previous_physical=example["coords_prev_physical"],
                            ids_previous=example["candidate_ids_prev"],
                            mask_previous=example["prev_mask"],
                            coordinates_parent_physical=example["coords_src_physical"],
                            coordinates_child_physical=example["coords_tgt_physical"],
                            parent_index=source,
                            child_index=target,
                        ),
                    }
                    break
                mode_record["annotated_pair_attention_example"] = inspected
            records[held_out]["modes"][mode] = mode_record
            _write_json(working_dir / "pair_metrics.json", records)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    model_manifest = {
        "source_primary_checkpoint_sha256": actual_sha,
        "checkpoints": {
            f"fold_{record['fold']}_{mode}": {
                "path": f"models/fold_{record['fold']}_{mode}.pt",
                "sha256": mode_record["checkpoint_sha256"],
                "held_out_embryo": embryo,
                "mode": mode,
            }
            for embryo, record in records.items()
            for mode, mode_record in record["modes"].items()
        },
    }
    _write_json(working_dir / "model_manifest.json", model_manifest)
    progression = {}
    for embryo, record in records.items():
        baseline = record["public_primary"]
        control = record["modes"]["primary_only"]["outer"]
        history = record["modes"]["primary_with_past_feature_attention"]["outer"]
        same_denominators = all(
            baseline[key] == control[key] == history[key]
            for key in ("known_edges", "active_pairs", "division_parents")
        )
        passed = (
            same_denominators
            and history["division_parents"] > 0
            and history["recovered_known_edges"]
            > max(baseline["recovered_known_edges"], control["recovered_known_edges"])
            and history["active_pair_errors"]
            <= min(baseline["active_pair_errors"], control["active_pair_errors"])
            and history["recovered_division_parents"]
            >= max(
                baseline["recovered_division_parents"],
                control["recovered_division_parents"],
            )
        )
        progression[embryo] = {
            "passed": bool(passed),
            "same_denominators": bool(same_denominators),
            "baseline": baseline,
            "primary_only": control,
            "pair_history_attention": history,
        }
    progression["all_embryos_passed"] = all(row["passed"] for row in progression.values())
    _write_json(working_dir / "graph_progression_gate.json", progression)
    receipt = {
        "experiment": config["experiment"]["name"],
        "capture_seconds": capture_seconds,
        "capture_sha256": capture_sha256,
        "feature_schema_sha256": feature_schema_sha256,
        "feature_content_sha256": feature_content_sha256,
        "model_manifest_sha256": labels.file_sha256(working_dir / "model_manifest.json"),
        "pair_metrics_sha256": labels.file_sha256(working_dir / "pair_metrics.json"),
        "training_seconds": time.monotonic() - started,
        "videos": stems,
        "window_counts": {stem: len(eligible[stem]) for stem in stems},
        "primary_checkpoint_sha256": actual_sha,
        "coordinate_head_sha256": config["model"]["coordinate_head_sha256"],
        "candidate_retention": retention,
        "baseline_parity": parity,
        "pair_metrics": records,
        "graph_progression_gate": progression,
        "official_score": None,
        "graph_inference_performed": False,
    }
    _write_json(working_dir / "train_receipt.json", receipt)
