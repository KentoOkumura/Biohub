"""Input, checkpoint, pair-shard, and summary helpers for the diagnostic notebook."""

from __future__ import annotations

import importlib.util
import inspect
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from diagnostic_metrics import (
    bucket_readout,
    fixed_threshold_counts,
    nearest_neighbor_distance_um,
    precision_recall_by_score,
    quantile_boundaries,
)
from frozen_tracker import (
    build_embryo_splits,
    build_window_example,
    canonical_state_sha256,
    discover_cache_paths,
    file_sha256,
    filter_nonempty_gt_window_paths,
    greedy_match_candidates,
    load_annotation_graph,
    recompute_cache_identity_sha256,
    validate_cache_summary,
    validate_window_cache,
)

VARIANTS = ("current", "model_a", "model_b", "model_b_identity_init")
SPLITS = ("internal_validation", "outer_evaluation")


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return file_sha256(path)


def resolve_inputs(config: dict[str, Any], input_root: Path) -> dict[str, Any]:
    cache_cfg = config["data"]["cache"]
    summaries = [
        path
        for path in input_root.rglob(str(cache_cfg["summary_file"]))
        if str(cache_cfg["kernel_source"]).split("/")[-1] in path.as_posix()
    ]
    if len(summaries) != 1:
        raise RuntimeError(f"expected one exp015 cache summary, found {summaries}")
    summary = validate_cache_summary(summaries[0], cache_cfg)
    cache_paths = discover_cache_paths(summaries[0].parent / cache_cfg["directory"], cache_cfg)
    observed_identity = recompute_cache_identity_sha256(cache_paths)
    if observed_identity != cache_cfg["identity_sha256"]:
        raise RuntimeError("exp015 cache identity SHA mismatch")

    competition = "biohub-cell-tracking-during-development"
    train_dirs = [
        path
        for path in (
            input_root / "competitions" / competition / "train",
            input_root / competition / "train",
        )
        if path.is_dir()
    ]
    if len(train_dirs) != 1:
        raise RuntimeError(f"expected one competition train directory, found {train_dirs}")
    train_dir = train_dirs[0]
    sample_names = sorted({path.parent.name for path in cache_paths})
    observed_counts = {
        embryo: sum(name.startswith(f"{embryo}_") for name in sample_names)
        for embryo in config["data"]["expected_embryo_counts"]
    }
    if observed_counts != config["data"]["expected_embryo_counts"]:
        raise RuntimeError(f"embryo sample counts differ: {observed_counts}")
    missing = [sample for sample in sample_names if not (train_dir / f"{sample}.geff").is_dir()]
    if missing:
        raise FileNotFoundError(f"missing GEFF samples: {missing[:8]}")
    splits = build_embryo_splits(
        sample_names,
        config["validation"]["outer_folds"],
        int(config["validation"]["internal_split_seed"]),
    )
    scale = tuple(float(x) for x in config["data"]["annotation"]["voxel_scale_zyx_um"])
    annotations = {
        sample: load_annotation_graph(train_dir / f"{sample}.geff", scale)
        for sample in sample_names
    }
    cache_paths, filter_audit = filter_nonempty_gt_window_paths(cache_paths, annotations)
    return {
        "cache_summary": summary,
        "cache_identity_sha256": observed_identity,
        "cache_paths": cache_paths,
        "annotations": annotations,
        "splits": splits,
        "filter_audit": filter_audit,
        "annotation_sha256": {
            sample: graph.content_sha256 for sample, graph in sorted(annotations.items())
        },
    }


def resolve_models(
    config: dict[str, Any], input_root: Path, fold_spec: dict[str, Any], device: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    manifest_paths = list(input_root.rglob("model_manifest.json"))
    source_paths = list(input_root.rglob("simple_node_transformer.py"))
    models: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    fold = int(fold_spec["fold"])
    for variant in VARIANTS:
        spec = config["model"]["checkpoints"][variant]
        slug = str(spec["kernel_source"]).split("/")[-1]
        matches = [
            path
            for path in manifest_paths
            if slug in path.as_posix() and file_sha256(path) == spec["manifest_sha256"]
        ]
        if len(matches) != 1:
            raise RuntimeError(f"{variant}: expected one verified manifest, found {matches}")
        manifest_path = matches[0]
        manifest = json.loads(manifest_path.read_text())
        observed_source_sha = manifest.get(
            "model_source_sha256", manifest.get("public_model_source_sha256")
        )
        if observed_source_sha != spec["model_source_sha256"]:
            raise RuntimeError(f"{variant}: manifest model source SHA mismatch")
        source_matches = [path for path in source_paths if file_sha256(path) == observed_source_sha]
        if not source_matches:
            raise RuntimeError(f"{variant}: exact model source is not mounted")
        source_path = sorted(source_matches, key=lambda path: slug not in path.as_posix())[0]
        module_spec = importlib.util.spec_from_file_location(
            f"diagnostic_tracker_{variant}", source_path
        )
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError(f"{variant}: unable to load model source")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        model_class = module.SimpleNodeTransformer

        entries = [item for item in manifest["models"] if int(item["fold"]) == fold]
        if len(entries) != 1:
            raise RuntimeError(f"{variant}: fold {fold} missing or duplicated")
        entry = entries[0]
        if (
            entry["train_embryo"] != fold_spec["train_embryo"]
            or entry["evaluation_embryo"] != fold_spec["evaluation_embryo"]
        ):
            raise RuntimeError(f"{variant}: fold embryo metadata mismatch")
        model_path = manifest_path.parent / entry["path"]
        expected_file_sha = spec["file_sha256_by_fold"][str(fold)]
        if (
            entry["file_sha256"] != expected_file_sha
            or file_sha256(model_path) != expected_file_sha
        ):
            raise RuntimeError(f"{variant}: checkpoint file SHA mismatch")
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        for key in ("fold", "train_embryo", "evaluation_embryo"):
            if payload[key] != (fold if key == "fold" else fold_spec[key]):
                raise RuntimeError(f"{variant}: checkpoint {key} mismatch")
        if variant != "current" and payload["variant"] != variant:
            raise RuntimeError(f"{variant}: checkpoint variant mismatch")
        state = payload["state_dict"]
        state_sha = canonical_state_sha256(state)
        if state_sha != entry["canonical_state_sha256"]:
            raise RuntimeError(f"{variant}: checkpoint state SHA mismatch")
        parameters = config["model"]["params"]
        options = {
            "feat_dim": int(parameters["feature_dim"]),
            "hidden_dim": int(parameters["hidden_dim"]),
            "n_heads": int(parameters["n_heads"]),
            "n_blocks": int(parameters["n_blocks"]),
            "mlp_ratio": float(parameters["mlp_ratio"]),
            "dropout": float(parameters["dropout"]),
            "pair_chunk_size": int(parameters["pair_chunk_size"]),
            "n_self_blocks": int(parameters["n_self_blocks"]),
        }
        architecture = manifest.get("architectures", {}).get(variant, {})
        options.update(architecture)
        supported = inspect.signature(model_class).parameters
        tracker = model_class(**{key: value for key, value in options.items() if key in supported})
        tracker.load_state_dict(state, strict=True)
        tracker = tracker.to(device).eval()
        models[variant] = tracker
        audit[variant] = {
            "manifest_sha256": spec["manifest_sha256"],
            "model_source_sha256": observed_source_sha,
            "model_source_path": str(source_path),
            "checkpoint_sha256": expected_file_sha,
            "state_sha256": state_sha,
            "fold": fold,
        }
    return models, audit


def read_candidate_ids(
    path: Path, source_count: int, target_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read IDs omitted by the fixed-cache validator's feature-only return value."""
    with np.load(path, allow_pickle=False) as saved:
        source_ids = np.asarray(saved["candidate_ids_src"], dtype=np.int64)
        target_ids = np.asarray(saved["candidate_ids_tgt"], dtype=np.int64)
    if source_ids.shape != (source_count,) or target_ids.shape != (target_count,):
        raise ValueError(f"candidate ID shape mismatch: {path}")
    return source_ids, target_ids


def load_window(
    path: Path, annotation: Any, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cache = config["data"]["cache"]
    maximum_um = float(config["model"]["teacher"]["max_matching_distance_um"])
    arrays, _ = validate_window_cache(
        path,
        feature_channels=int(cache["feature_channels"]),
        expected_primary_checkpoint_sha256=str(cache["public_checkpoint_sha256"]),
    )
    example = build_window_example(
        path,
        annotation,
        feature_channels=int(cache["feature_channels"]),
        expected_primary_checkpoint_sha256=str(cache["public_checkpoint_sha256"]),
        max_matching_distance_um=maximum_um,
        downsample_zyx=tuple(float(x) for x in config["model"]["params"]["downsample_zyx"]),
    )
    source_frame, target_frame = example["window_frames"]
    source_matches, _ = greedy_match_candidates(
        arrays["coords_src_physical"],
        annotation.frames[source_frame].node_ids,
        annotation.frames[source_frame].coords_physical,
        maximum_um,
    )
    target_matches, _ = greedy_match_candidates(
        arrays["coords_tgt_physical"],
        annotation.frames[target_frame].node_ids,
        annotation.frames[target_frame].coords_physical,
        maximum_um,
    )
    if int(example["target"].sum()) != example["teacher_stats"]["positive_edges"]:
        raise RuntimeError("teacher target count changed")
    source_ids, target_ids = read_candidate_ids(
        path, len(arrays["coords_src_physical"]), len(arrays["coords_tgt_physical"])
    )
    extra = {
        "source_ids": source_ids,
        "target_ids": target_ids,
        "source_coords": np.asarray(arrays["coords_src_physical"], dtype=np.float32),
        "target_coords": np.asarray(arrays["coords_tgt_physical"], dtype=np.float32),
        "source_matched": source_matches >= 0,
        "target_matched": target_matches >= 0,
    }
    extra["target_nearest_um"] = nearest_neighbor_distance_um(extra["target_coords"])
    return example, extra


def infer_probabilities(
    example: dict[str, Any], models: dict[str, Any], device: Any
) -> dict[str, np.ndarray]:
    import torch

    inputs = [
        torch.from_numpy(example[name]).unsqueeze(0).to(device)
        for name in ("features_src", "features_tgt", "coords_src", "coords_tgt")
    ]
    masks = [
        torch.ones((1, len(example[name])), dtype=torch.bool, device=device)
        for name in ("features_src", "features_tgt")
    ]
    output: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for variant, model in models.items():
            logits = model(*inputs, *masks)
            logits = logits[0] if logits.ndim == 3 else logits
            if logits.shape != example["target"].shape:
                raise RuntimeError(f"{variant}: pair-logit shape mismatch")
            if logits.numel() == 0:
                probabilities = np.empty(logits.shape, dtype=np.float32)
            else:
                probabilities = torch.softmax(logits.float(), dim=0).cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise RuntimeError(f"{variant}: non-finite pair probabilities")
            output[variant] = probabilities
    return output


def fit_training_boundaries(
    paths: list[Path], annotations: dict[str, Any], config: dict[str, Any]
) -> dict[str, list[float]]:
    counts: list[int] = []
    nearest: list[np.ndarray] = []
    movement: list[np.ndarray] = []
    for path in paths:
        example, extra = load_window(path, annotations[path.parent.name], config)
        counts.append(len(extra["target_ids"]))
        nearest.append(extra["target_nearest_um"])
        rows, cols = np.nonzero(example["target"] > 0.5)
        if len(rows):
            distance = np.linalg.norm(
                extra["source_coords"][rows] - extra["target_coords"][cols], axis=1
            )
            movement.append(distance)
    if not movement:
        raise RuntimeError("training-side windows contain no known positive pairs")
    quantiles = [float(x) for x in config["validation"]["quantile_points"]]
    return {
        "target_candidate_count": quantile_boundaries(np.asarray(counts), quantiles),
        "target_nearest_um": quantile_boundaries(np.concatenate(nearest), quantiles),
        "pair_displacement_um": quantile_boundaries(np.concatenate(movement), quantiles),
    }


def benchmark_inference(
    paths: list[Path],
    annotations: dict[str, Any],
    config: dict[str, Any],
    models: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    import torch

    count = min(int(config["runtime"]["benchmark_windows_per_fold"]), len(paths))
    selected = sorted(paths, key=lambda path: path.stat().st_size, reverse=True)[:count]
    if count == 0:
        raise RuntimeError("no windows available for runtime benchmark")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for path in selected:
        example, _ = load_window(path, annotations[path.parent.name], config)
        infer_probabilities(example, models, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "window_count": count,
        "elapsed_seconds": elapsed,
        "seconds_per_window": elapsed / count,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }


def write_sample_shard(
    sample_paths: list[Path],
    annotations: dict[str, Any],
    config: dict[str, Any],
    models: dict[str, Any],
    device: Any,
    output_path: Path,
) -> dict[str, Any]:
    """Keep candidate tables once per window and only pair indices per active pair."""
    pair_fields: dict[str, list[np.ndarray]] = {
        name: [] for name in ("window", "source_row", "target_col", "label", "unknown", *VARIANTS)
    }
    candidate_fields: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "source_ids",
            "target_ids",
            "source_coords",
            "target_coords",
            "target_nearest_um",
            "source_matched",
            "target_matched",
            "division_rows",
        )
    }
    source_offsets = [0]
    target_offsets = [0]
    frames = []
    totals = {
        variant: {
            "window_count": 0,
            "division_parent_count": 0,
            "recovered_division_parent_count": 0,
        }
        for variant in VARIANTS
    }
    for window_index, path in enumerate(sorted(sample_paths)):
        example, extra = load_window(path, annotations[path.parent.name], config)
        probabilities = infer_probabilities(example, models, device)
        target = example["target"] > 0.5
        active = target.any(axis=1)[:, None] | target.any(axis=0)[None, :]
        rows, cols = np.nonzero(active)
        unknown = ~(extra["source_matched"][rows] & extra["target_matched"][cols])
        pair_fields["window"].append(np.full(len(rows), window_index, dtype=np.uint16))
        pair_fields["source_row"].append(rows.astype(np.uint16))
        pair_fields["target_col"].append(cols.astype(np.uint16))
        pair_fields["label"].append(target[rows, cols])
        pair_fields["unknown"].append(unknown)
        candidate_fields["division_rows"].append(target.sum(axis=1) > 1)
        for name in extra:
            candidate_fields[name].append(extra[name])
        source_offsets.append(source_offsets[-1] + len(extra["source_ids"]))
        target_offsets.append(target_offsets[-1] + len(extra["target_ids"]))
        frames.append(example["window_frames"])
        division_rows = target.sum(axis=1) > 1
        for variant in VARIANTS:
            probability = probabilities[variant]
            pair_fields[variant].append(probability[rows, cols])
            totals[variant]["window_count"] += 1
            totals[variant]["division_parent_count"] += int(division_rows.sum())
            totals[variant]["recovered_division_parent_count"] += int(
                np.count_nonzero(
                    [
                        (probability[row, target[row]] > 0.5).all()
                        for row in np.flatnonzero(division_rows)
                    ]
                )
            )
    payload = {
        name: np.concatenate(chunks) for name, chunks in {**pair_fields, **candidate_fields}.items()
    }
    payload["source_offsets"] = np.asarray(source_offsets, dtype=np.int32)
    payload["target_offsets"] = np.asarray(target_offsets, dtype=np.int32)
    payload["frames"] = np.asarray(frames, dtype=np.int16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return {
        "sample": sample_paths[0].parent.name,
        "window_count": len(sample_paths),
        "active_pair_count": len(payload["label"]),
        "unknown_endpoint_pair_count": int(payload["unknown"].sum()),
        "path": str(output_path),
        "sha256": file_sha256(output_path),
        "bytes": output_path.stat().st_size,
        "fixed_totals": totals,
    }


def _combine_bucket_rows(
    rows: list[list[dict[str, Any]]], edges: list[float], minimum: int
) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for sample_rows in rows:
        for row in sample_rows:
            bucket = int(row["bucket"])
            target = merged.setdefault(
                bucket,
                {
                    "bucket": bucket,
                    "lower": row["lower"],
                    "upper": row["upper"],
                    "missing_value": row["missing_value"],
                    "positive_edge_count": 0,
                    "true_positive_edge_count": 0,
                    "false_positive_pair_count": 0,
                    "active_pair_count": 0,
                },
            )
            for key in (
                "positive_edge_count",
                "true_positive_edge_count",
                "false_positive_pair_count",
                "active_pair_count",
            ):
                target[key] += int(row[key])
    output = []
    for bucket in sorted(merged):
        row = merged[bucket]
        positives = row["positive_edge_count"]
        tp = row["true_positive_edge_count"]
        fp = row["false_positive_pair_count"]
        row["positive_edge_recall"] = tp / positives if positives >= minimum else None
        row["positive_pair_precision"] = (
            tp / (tp + fp) if positives >= minimum and tp + fp else None
        )
        output.append(row)
    return output


def summarize_group(
    shard_records: list[dict[str, Any]],
    variant: str,
    edges: dict[str, list[float]],
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    known_labels: list[np.ndarray] = []
    known_scores: list[np.ndarray] = []
    bucket_rows: dict[str, list[list[dict[str, Any]]]] = {
        name: [] for name in ("target_candidate_count", "target_nearest_um", "pair_displacement_um")
    }
    division_pair_rows: list[list[dict[str, Any]]] = []
    fixed = {
        "window_count": 0,
        "active_pair_count": 0,
        "positive_edge_count": 0,
        "true_positive_edge_count": 0,
        "false_positive_pair_count": 0,
        "unknown_endpoint_pair_count": 0,
        "division_parent_count": 0,
        "recovered_division_parent_count": 0,
    }
    for record in shard_records:
        if file_sha256(Path(record["path"])) != record["sha256"]:
            raise RuntimeError(f"pair shard changed: {record['path']}")
        with np.load(record["path"], allow_pickle=False) as saved:
            label = saved["label"].astype(np.bool_)
            score = saved[variant].astype(np.float32)
            unknown = saved["unknown"].astype(np.bool_)
            if not (len(label) == len(score) == len(unknown)):
                raise RuntimeError("pair shard arrays do not align")
            counts = fixed_threshold_counts(score, label, np.ones_like(label))
            for key in (
                "active_pair_count",
                "positive_edge_count",
                "true_positive_edge_count",
                "false_positive_pair_count",
            ):
                fixed[key] += int(counts[key])
            fixed["unknown_endpoint_pair_count"] += int(unknown.sum())
            totals = record["fixed_totals"][variant]
            for key in (
                "window_count",
                "division_parent_count",
                "recovered_division_parent_count",
            ):
                fixed[key] += int(totals[key])
            labels.append(label)
            scores.append(score)
            known_labels.append(label[~unknown])
            known_scores.append(score[~unknown])
            window = saved["window"].astype(np.int32)
            source_row = saved["source_row"].astype(np.int32)
            target_col = saved["target_col"].astype(np.int32)
            source_offset = saved["source_offsets"][window].astype(np.int32)
            target_offset = saved["target_offsets"][window].astype(np.int32)
            source_index = source_offset + source_row
            target_index = target_offset + target_col
            source_coords = saved["source_coords"][source_index]
            target_coords = saved["target_coords"][target_index]
            displacement = np.linalg.norm(source_coords - target_coords, axis=1)
            candidate_counts = saved["target_offsets"][window + 1] - saved["target_offsets"][window]
            values = {
                "target_candidate_count": candidate_counts,
                "target_nearest_um": saved["target_nearest_um"][target_index],
                "pair_displacement_um": displacement,
            }
            for name, value in values.items():
                bucket_rows[name].append(
                    bucket_readout(
                        value,
                        score,
                        label,
                        edges[name],
                        min_positive_edges=0,
                    )
                )
            division_flag = saved["division_rows"][source_index].astype(np.int32)
            division_pair_rows.append(
                bucket_readout(
                    division_flag,
                    score,
                    label,
                    [0.5],
                    min_positive_edges=0,
                )
            )
    fixed["positive_edge_recall"] = (
        fixed["true_positive_edge_count"] / fixed["positive_edge_count"]
        if fixed["positive_edge_count"]
        else None
    )
    selected = fixed["true_positive_edge_count"] + fixed["false_positive_pair_count"]
    fixed["positive_pair_precision"] = (
        fixed["true_positive_edge_count"] / selected if selected else None
    )
    fixed["division_parent_recall"] = (
        fixed["recovered_division_parent_count"] / fixed["division_parent_count"]
        if fixed["division_parent_count"]
        >= int(config["validation"]["min_division_parents_for_ratio"])
        else None
    )
    score_all = np.concatenate(scores)
    label_all = np.concatenate(labels)
    score_known = np.concatenate(known_scores)
    label_known = np.concatenate(known_labels)
    target_precision = float(config["validation"]["target_precision"])
    pr = precision_recall_by_score(score_all, label_all, target_precision)
    pr_known = precision_recall_by_score(score_known, label_known, target_precision)
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_paths = {}
    for name, curve in (("active_mask", pr), ("matched_endpoints", pr_known)):
        curve_path = output_dir / f"{variant}_{name}_pr.npz"
        np.savez_compressed(
            curve_path,
            threshold=np.asarray(curve["threshold"], dtype=np.float32),
            precision=np.asarray(curve["precision"], dtype=np.float64),
            recall=np.asarray(curve["recall"], dtype=np.float64),
        )
        curve_paths[name] = {
            "path": str(curve_path),
            "sha256": file_sha256(curve_path),
            "point_count": len(curve["threshold"]),
            "positive_count": curve["positive_count"],
            "pair_count": curve["pair_count"],
            "recall_at_target_precision": curve["recall_at_target_precision"],
            "threshold_at_target_precision": curve["threshold_at_target_precision"],
            "target_reachable": curve["target_reachable"],
        }
    minimum = int(config["validation"]["min_positive_edges_per_bucket"])
    return {
        "fixed_0_5": fixed,
        "pr_curves": curve_paths,
        "buckets": {
            name: _combine_bucket_rows(rows, edges[name], minimum)
            for name, rows in bucket_rows.items()
        },
        "division_parent_pair_buckets": _combine_bucket_rows(division_pair_rows, [0.5], minimum),
    }


def verify_outer_reproduction(
    observed: dict[str, Any], reference: dict[str, Any], embryo: str, variant: str
) -> None:
    expected = reference[embryo]["control" if variant == "current" else variant]
    actual = observed["fixed_0_5"]
    for key in (
        "window_count",
        "active_pair_count",
        "positive_edge_count",
        "division_parent_count",
        "true_positive_edge_count",
        "false_positive_pair_count",
    ):
        if int(actual[key]) != int(expected[key]):
            raise RuntimeError(
                f"{embryo}/{variant}: 0.5 reproduction failed for {key}: "
                f"{actual[key]} != {expected[key]}"
            )
    for key in ("positive_edge_recall", "positive_pair_precision"):
        if abs(float(actual[key]) - float(expected[key])) > 1e-12:
            raise RuntimeError(f"{embryo}/{variant}: 0.5 reproduction failed for {key}")
    expected_division = expected["division_parent_recall"]
    actual_division = (
        actual["recovered_division_parent_count"] / actual["division_parent_count"]
        if actual["division_parent_count"]
        else None
    )
    if abs(float(actual_division) - float(expected_division)) > 1e-12:
        raise RuntimeError(f"{embryo}/{variant}: division-parent reproduction failed")


def reproduce_fixed_counts(shard_records: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    totals = {
        "window_count": 0,
        "active_pair_count": 0,
        "positive_edge_count": 0,
        "true_positive_edge_count": 0,
        "false_positive_pair_count": 0,
        "division_parent_count": 0,
        "recovered_division_parent_count": 0,
    }
    for record in shard_records:
        with np.load(record["path"], allow_pickle=False) as saved:
            scores = saved[variant]
            labels = saved["label"]
            counts = fixed_threshold_counts(scores, labels, np.ones_like(labels))
            for key in (
                "active_pair_count",
                "positive_edge_count",
                "true_positive_edge_count",
                "false_positive_pair_count",
            ):
                totals[key] += int(counts[key])
        for key in ("window_count", "division_parent_count", "recovered_division_parent_count"):
            totals[key] += int(record["fixed_totals"][variant][key])
    positives = totals["positive_edge_count"]
    tp = totals["true_positive_edge_count"]
    fp = totals["false_positive_pair_count"]
    totals["positive_edge_recall"] = tp / positives if positives else None
    totals["positive_pair_precision"] = tp / (tp + fp) if tp + fp else None
    totals["division_parent_recall"] = (
        totals["recovered_division_parent_count"] / totals["division_parent_count"]
        if totals["division_parent_count"]
        else None
    )
    return {"fixed_0_5": totals}
