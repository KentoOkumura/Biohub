from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import frozen_tracker as base
import numpy as np
import torch
from past_candidate_attention import select_past_candidates


def write_evidence(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"evidence": str(path), "sha256": base.file_sha256(path)}), flush=True)


def source_diagnostic_masks(batch: dict[str, Any], index: int, n_source: int) -> dict[str, Any]:
    metadata = batch["metadata"][index]
    context = metadata.get("diagnostic_context")
    if context is None:
        return {}
    cfg = context["config"]
    source = batch["coords_src_physical"][index : index + 1, :n_source]
    previous = batch["coords_prev_physical"][index : index + 1]
    ids = batch["candidate_ids_prev"][index : index + 1]
    valid = batch["prev_mask"][index : index + 1]
    _, selected_valid, selected = select_past_candidates(
        previous, source, ids, valid, int(cfg["k"])
    )
    distance2 = ((source[:, :, None] - previous[:, None]) ** 2).sum(-1)
    density = ((distance2 <= float(cfg["density_radius_um"]) ** 2) & valid[:, None]).sum(-1)[0]
    parents = torch.as_tensor(context["known_parent_candidate_ids"], device=source.device)
    retained = ((selected[0] == parents[:, None]) & selected_valid[0]).any(-1)
    low, high = cfg["density_upper_bounds"]
    return {
        "frame_start": torch.full(
            (n_source,), metadata["window_frames"][0] == 0, device=source.device
        ),
        "density_0": density <= low,
        "density_1": (density > low) & (density <= high),
        "density_2": density > high,
        "past_retained": (parents >= 0) & retained,
        "past_excluded": (parents >= 0) & ~retained,
        "past_unknown": parents < 0,
    }


class RuntimeBudget:
    def __init__(self, started: float, config: dict[str, Any], allocated_gpus: int) -> None:
        self.started = started
        remaining = config["quota_remaining_hours_at_push"]
        used = config["quota_used_hours_at_push"]
        if remaining is None or used is None or not config["quota_checked_at"]:
            raise RuntimeError("fresh quota and user budget evidence are required")
        self.allocated_gpus = max(1, allocated_gpus)
        hours = min(float(remaining), float(config["weekly_gpu_budget_hours"]) - float(used))
        self.limit_seconds = min(
            float(config["runtime_gate_hours"]) * 3600, hours * 3600 / self.allocated_gpus
        )
        self.multiplier = float(config["runtime_projection_multiplier"])
        self.remaining_work = 0.0
        self.planned_done = 0.0
        self.observed_done = 0.0
        self.check()

    def check(self) -> None:
        elapsed = time.perf_counter() - self.started
        speed_factor = max(1.0, self.observed_done / max(self.planned_done, 1e-9))
        projected = elapsed + self.multiplier * self.remaining_work * speed_factor
        if not np.isfinite(projected) or projected >= self.limit_seconds:
            raise RuntimeError(
                f"runtime/user GPU budget gate: projected={projected:.1f}s "
                f"limit={self.limit_seconds:.1f}s elapsed={elapsed:.1f}s"
            )

    def observe(self, elapsed: float, planned: float) -> None:
        self.observed_done += elapsed
        self.planned_done += planned
        self.remaining_work = max(0.0, self.remaining_work - planned)
        self.check()


class BudgetedLoader:
    """Account for reading, forward, backward and evaluation between yielded batches."""

    def __init__(self, loader: Any, budget: RuntimeBudget, per_window: float) -> None:
        self.loader = loader
        self.budget = budget
        self.per_window = per_window

    def __iter__(self) -> Any:
        self.budget.check()
        started = time.perf_counter()
        for batch in self.loader:
            yield batch
            self.budget.observe(
                time.perf_counter() - started, self.per_window * len(batch["metadata"])
            )
            started = time.perf_counter()


def candidate_inventory(paths: list[Path], k: int, budget: RuntimeBudget) -> list[dict[str, Any]]:
    records = {path: base.cache_identity_record(path) for path in paths}
    inventory = []
    for path, record in records.items():
        frame = record["window_frames"][0]
        prior = path.with_name(f"{frame - 1:06d}_{frame:06d}.npz")
        if frame and prior not in records:
            raise FileNotFoundError(prior)
        previous = records[prior]["candidate_count_src"] if frame else 0
        pairs = record["candidate_count_src"] * record["candidate_count_tgt"]
        inventory.append(
            {
                **record,
                "path": path.as_posix(),
                "previous_count": previous,
                "pair_count": pairs,
                "selected_triples": pairs * min(k, previous),
            }
        )
    budget.check()
    return inventory


def stratified_plan(
    rows: list[dict[str, Any]], strata: int, per_stratum: int, seed: int
) -> dict[str, Any]:
    if not rows or strata < 1 or per_stratum < 1:
        raise ValueError("nonempty rows and positive stratification settings required")
    values = np.asarray([row["selected_triples"] for row in rows], dtype=np.float64)
    boundaries = np.quantile(values, np.arange(1, strata) / strata)
    labels = np.searchsorted(boundaries, values, side="right")
    rng = np.random.default_rng(seed)
    sampled = {}
    for label in range(strata):
        indices = np.flatnonzero(labels == label)
        chosen = rng.choice(indices, min(per_stratum, len(indices)), replace=False)
        sampled[str(label)] = sorted(rows[int(index)]["path"] for index in chosen)
    return {"boundaries": boundaries.tolist(), "sampled_paths": sampled}


def stratum_counts(rows: list[dict[str, Any]], boundaries: list[float]) -> list[int]:
    labels = np.searchsorted(boundaries, [row["selected_triples"] for row in rows], side="right")
    return np.bincount(labels, minlength=len(boundaries) + 1).tolist()


def weighted_runtime(counts: list[int], rates: dict[str, float]) -> float:
    measured = [float(value) for value in rates.values()]
    if not measured or not all(np.isfinite(measured)):
        raise ValueError("finite measured rates are required")
    # A stratum absent from learning-side sampling uses the slowest measured rate.
    return sum(
        count * float(rates.get(str(index), max(measured))) for index, count in enumerate(counts)
    )


def retention_window(
    previous: np.ndarray,
    source: np.ndarray,
    previous_ids: np.ndarray,
    annotation: base.AnnotationGraph,
    source_frame: int,
    selected_ids: np.ndarray,
    matching_um: float,
    cfg: dict[str, Any],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    counts: Counter[str] = Counter(
        known_edges=0,
        matched_edges=0,
        retained_edges=0,
        missing_endpoints=0,
        boundary_windows=0,
        missing_annotation_windows=0,
    )
    buckets: dict[str, Counter[str]] = {}
    if source_frame == 0:
        counts["boundary_windows"] += 1
        return dict(counts), {}
    if source_frame - 1 not in annotation.frames or source_frame not in annotation.frames:
        counts["missing_annotation_windows"] += 1
        return dict(counts), {}
    past_gt = annotation.frames[source_frame - 1]
    source_gt = annotation.frames[source_frame]
    past_matches, _ = base.greedy_match_candidates(
        previous, past_gt.node_ids, past_gt.coords_physical, matching_um
    )
    source_matches, _ = base.greedy_match_candidates(
        source, source_gt.node_ids, source_gt.coords_physical, matching_um
    )
    past_map = {int(node): i for i, node in enumerate(past_matches) if node >= 0}
    source_map = {int(node): i for i, node in enumerate(source_matches) if node >= 0}
    source_gt_ids = set(map(int, source_gt.node_ids))
    outgoing = annotation.outgoing_edges
    if outgoing is None:
        outgoing = {}
        for p, a in annotation.edges:
            outgoing.setdefault(p, []).append(a)
    for p in past_gt.node_ids:
        children = [int(a) for a in outgoing.get(int(p), ()) if int(a) in source_gt_ids]
        for a in children:
            counts["known_edges"] += 1
            if int(p) not in past_map or a not in source_map:
                counts["missing_endpoints"] += 1
                continue
            pi, ai = past_map[int(p)], source_map[a]
            kept = int(previous_ids[pi] in selected_ids[ai])
            counts["matched_edges"] += 1
            counts["retained_edges"] += kept
            displacement = float(np.linalg.norm(source[ai] - previous[pi]))
            density = int(
                (np.linalg.norm(previous - source[ai], axis=1) <= cfg["density_radius_um"]).sum()
            )
            displacement_bin = np.searchsorted(
                cfg["displacement_upper_bounds_um"], displacement, side="left"
            )
            labels = (
                f"density_{np.searchsorted(cfg['density_upper_bounds'], density, side='left')}",
                f"displacement_{displacement_bin}",
                f"division_{int(len(children) > 1)}",
            )
            for label in labels:
                bucket = buckets.setdefault(label, Counter(matched_edges=0, retained_edges=0))
                bucket["matched_edges"] += 1
                bucket["retained_edges"] += kept
    return dict(counts), {key: dict(value) for key, value in buckets.items()}


def audit_candidate_retention(
    paths: list[Path],
    annotations: dict[str, base.AnnotationGraph],
    splits: list[dict[str, Any]],
    config: dict[str, Any],
    device: Any,
    budget: RuntimeBudget,
) -> dict[str, Any]:
    model_cfg = config["model"]
    cfg = model_cfg["candidate_retention"]
    k = int(model_cfg["past_candidate_attention"]["max_past_candidates"])
    sample_fold = {
        sample: int(split["fold"]) for split in splits for sample in split["gradient_update"]
    }
    totals = {str(split["fold"]): Counter() for split in splits}
    buckets: dict[str, dict[str, Counter[str]]] = {key: {} for key in totals}
    digests = {key: hashlib.sha256() for key in totals}
    # Keep only two adjacent windows resident. Cache coordinates are fixed, not GT-derived.
    last_path = None
    last_arrays = None
    read_kwargs = {
        "feature_channels": int(config["data"]["cache"]["feature_channels"]),
        "expected_primary_checkpoint_sha256": model_cfg["public_source"]["checkpoint_sha256"],
    }
    processed = 0
    for path in sorted(paths):
        if path.parent.name not in sample_fold:
            continue
        arrays, metadata = base.validate_window_cache(path, **read_kwargs)
        frame = int(metadata["window_frames"][0])
        if frame:
            prior_path = path.with_name(f"{frame - 1:06d}_{frame:06d}.npz")
            prior = (
                last_arrays
                if prior_path == last_path
                else base.validate_window_cache(prior_path, **read_kwargs)[0]
            )
            for key in ("candidate_ids", "coords_grid", "coords_physical"):
                src_key = (
                    "candidate_ids_src"
                    if key == "candidate_ids"
                    else key.replace("coords_", "coords_src_")
                )
                tgt_key = src_key.replace("_src", "_tgt")
                if not np.array_equal(arrays[src_key], prior[tgt_key]):
                    raise ValueError(f"past/current identity mismatch: {path} {key}")
            previous = prior["coords_src_physical"]
            previous_ids = prior["candidate_ids_src"]
        else:
            previous = np.zeros((0, 3), dtype=np.float32)
            previous_ids = np.zeros(0, dtype=np.int64)
        source = arrays["coords_src_physical"]
        with torch.no_grad():
            _, _, selected = select_past_candidates(
                torch.as_tensor(previous, device=device)[None],
                torch.as_tensor(source, device=device)[None],
                torch.as_tensor(previous_ids, device=device)[None],
                torch.ones((1, len(previous)), dtype=torch.bool, device=device),
                k,
            )
        selected_ids = selected[0].cpu().numpy()
        fold = str(sample_fold[path.parent.name])
        counts, window_buckets = retention_window(
            previous,
            source,
            previous_ids,
            annotations[path.parent.name],
            frame,
            selected_ids,
            float(model_cfg["teacher"]["max_matching_distance_um"]),
            cfg,
        )
        totals[fold].update(counts)
        totals[fold]["window_count"] += 1
        for key, value in window_buckets.items():
            buckets[fold].setdefault(key, Counter()).update(value)
        digests[fold].update(
            base.json_sha256(
                {
                    "sample": path.parent.name,
                    "frame": frame,
                    "source_ids": arrays["candidate_ids_src"].tolist(),
                    "selected_ids": selected_ids.tolist(),
                }
            ).encode()
        )
        last_path, last_arrays = path, arrays
        processed += 1
        if processed % int(model_cfg["training"]["preparation_progress_interval"]) == 0:
            budget.check()
            print(json.dumps({"retention_windows": processed, "fold_counts": totals}), flush=True)
    folds = {}
    for fold, counts in totals.items():
        denominator = counts["matched_edges"]
        rate = counts["retained_edges"] / denominator if denominator else None
        folds[fold] = {
            **dict(counts),
            "conditional_retention": rate,
            "selected_candidate_content_sha256": digests[fold].hexdigest(),
            "buckets": {key: dict(value) for key, value in buckets[fold].items()},
            "passed": rate is not None and rate >= float(cfg["min_retention"]),
        }
    budget.check()
    return {"folds": folds, "passed": all(row["passed"] for row in folds.values()), "config": cfg}
