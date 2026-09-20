from __future__ import annotations

from typing import Any

import numpy as np


def diagnose_pair(
    logits: np.ndarray,
    probabilities: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    threshold_grid: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Describe known edges separately from sparse-teacher negative predictions."""
    logits = np.asarray(logits)
    probabilities = np.asarray(probabilities)
    target = np.asarray(target)
    if logits.shape != target.shape or probabilities.shape != target.shape or target.ndim != 2:
        raise ValueError("logit/probability/target shapes differ")
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise ValueError("nonfinite predictions")
    if not np.allclose(probabilities.sum(axis=0), 1.0, atol=1e-5):
        raise ValueError("probabilities must normalize over source parents")
    positive = target > 0.5
    active = positive.any(axis=1)[:, None] | positive.any(axis=0)[None, :]
    predicted = probabilities > threshold
    best_parent = np.argmax(logits, axis=0)
    division_rows = positive.sum(axis=1) > 1
    edges = []
    for source, child in zip(*np.nonzero(positive), strict=True):
        score = logits[source, child]
        ties = logits[:, child] == score
        rank = 1 + np.count_nonzero(logits[:, child] > score)
        rank += np.count_nonzero(ties[:source])
        competitor = np.arange(logits.shape[0]) != source
        best_other_logit = float(logits[competitor, child].max()) if competitor.any() else None
        best_other_probability = (
            float(probabilities[competitor, child].max()) if competitor.any() else None
        )
        edges.append(
            {
                "source_index": int(source),
                "target_index": int(child),
                "true_parent_rank": int(rank),
                "top1_correct": bool(best_parent[child] == source),
                "tied_parent_count": int(ties.sum()),
                "predicted_parent_index": int(best_parent[child]),
                "true_parent_probability": float(probabilities[source, child]),
                "true_parent_logit": float(score),
                "best_other_probability": best_other_probability,
                "logit_margin": None
                if best_other_logit is None
                else float(score - best_other_logit),
                "probability_margin": None
                if best_other_probability is None
                else float(probabilities[source, child] - best_other_probability),
                "recovered": bool(predicted[source, child]),
                "division_parent": bool(division_rows[source]),
            }
        )
    summary = {
        "positive_edge_count": int(positive.sum()),
        "true_positive_count": int((predicted & positive).sum()),
        "top1_correct_count": sum(row["top1_correct"] for row in edges),
        "false_positive_active_pair_count": int((predicted & ~positive & active).sum()),
        "active_pair_count": int(active.sum()),
        "correct_active_pair_count": int(((predicted == positive) & active).sum()),
        "division_parent_count": int(division_rows.sum()),
        "recovered_division_parent_count": sum(
            bool(predicted[row, positive[row]].all()) for row in np.flatnonzero(division_rows)
        ),
        "threshold_counts": [
            {
                "threshold": float(value),
                "true_positive_count": int(((probabilities > value) & positive).sum()),
                "false_positive_active_pair_count": int(
                    ((probabilities > value) & ~positive & active).sum()
                ),
            }
            for value in threshold_grid
        ],
    }
    return summary, edges


def aggregate_diagnostics(
    windows: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, Any]:
    count_keys = (
        "positive_edge_count",
        "true_positive_count",
        "top1_correct_count",
        "false_positive_active_pair_count",
        "active_pair_count",
        "correct_active_pair_count",
        "division_parent_count",
        "recovered_division_parent_count",
    )
    totals = {key: sum(int(row[key]) for row in windows) for key in count_keys}
    for name, numerator, denominator in (
        ("positive_edge_recall", "true_positive_count", "positive_edge_count"),
        ("top1_accuracy", "top1_correct_count", "positive_edge_count"),
        ("edge_accuracy", "correct_active_pair_count", "active_pair_count"),
        ("division_parent_recall", "recovered_division_parent_count", "division_parent_count"),
    ):
        totals[name] = totals[numerator] / totals[denominator] if totals[denominator] else 0.0
    totals["window_count"] = len(windows)
    totals["legacy_mask_loss"] = float(np.mean([row["legacy_mask_loss"] for row in windows]))
    totals["mean_true_parent_probability"] = (
        float(np.mean([row["true_parent_probability"] for row in edges])) if edges else None
    )
    totals["mean_reciprocal_rank"] = (
        float(np.mean([1.0 / row["true_parent_rank"] for row in edges])) if edges else None
    )
    totals["threshold_curve"] = []
    for index, point in enumerate(windows[0]["threshold_counts"]):
        tp = sum(row["threshold_counts"][index]["true_positive_count"] for row in windows)
        fp = sum(
            row["threshold_counts"][index]["false_positive_active_pair_count"] for row in windows
        )
        totals["threshold_curve"].append(
            {
                "threshold": point["threshold"],
                "true_positive_count": tp,
                "positive_edge_recall": tp / totals["positive_edge_count"]
                if totals["positive_edge_count"]
                else 0.0,
                "false_positive_active_pair_count": fp,
            }
        )
    return totals


def edge_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["sample"],
        row["source_frame"],
        row["target_frame"],
        row["source_id"],
        row["target_id"],
    )


def compare_known_edges(
    reference: list[dict[str, Any]],
    changed: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    before = {edge_identity(row): row for row in reference}
    after = {edge_identity(row): row for row in changed}
    if len(before) != len(reference) or len(after) != len(changed) or before.keys() != after.keys():
        raise ValueError("paired edge identities are duplicated or differ")
    rows = []
    for key in sorted(before):
        a, b = before[key], after[key]
        rows.append(
            {
                "sample": key[0],
                "source_frame": key[1],
                "target_frame": key[2],
                "source_id": key[3],
                "target_id": key[4],
                "reference_rank": a["true_parent_rank"],
                "changed_rank": b["true_parent_rank"],
                "reference_probability": a["true_parent_probability"],
                "changed_probability": b["true_parent_probability"],
                "probability_delta": b["true_parent_probability"] - a["true_parent_probability"],
                "rescued_threshold": not a["recovered"] and b["recovered"],
                "harmed_threshold": a["recovered"] and not b["recovered"],
                "rescued_top1": not a["top1_correct"] and b["top1_correct"],
                "harmed_top1": a["top1_correct"] and not b["top1_correct"],
                "both_top1": a["top1_correct"] and b["top1_correct"],
                "reference_top1": a["top1_correct"],
                "changed_top1": b["top1_correct"],
                "reference_recovered": a["recovered"],
                "changed_recovered": b["recovered"],
                "reference_predicted_parent_id": a["predicted_parent_id"],
                "changed_predicted_parent_id": b["predicted_parent_id"],
                "division_parent": a["division_parent"],
                "displacement_um": a.get("displacement_um"),
                "previous_neighbor_count": a.get("previous_neighbor_count"),
            }
        )
    summary = {
        "positive_edge_count": len(rows),
        **{
            key: sum(row[key] for row in rows)
            for key in (
                "rescued_threshold",
                "harmed_threshold",
                "rescued_top1",
                "harmed_top1",
                "both_top1",
            )
        },
        "threshold_rescues_with_both_top1": sum(
            row["rescued_threshold"] and row["both_top1"] for row in rows
        ),
        "threshold_harms_with_both_top1": sum(
            row["harmed_threshold"] and row["both_top1"] for row in rows
        ),
        "mean_probability_delta": float(np.mean([row["probability_delta"] for row in rows]))
        if rows
        else None,
    }
    summary["recall_delta"] = (
        (summary["rescued_threshold"] - summary["harmed_threshold"]) / len(rows) if rows else None
    )
    summary["top1_delta"] = (
        (summary["rescued_top1"] - summary["harmed_top1"]) / len(rows) if rows else None
    )
    return summary, rows


def paired_video_bootstrap(
    paired_edges: list[dict[str, Any]],
    samples: list[str],
    *,
    repeats: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    if len(set(samples)) != len(samples) or not samples:
        raise ValueError("sample list must be nonempty and unique")
    index = {sample: i for i, sample in enumerate(samples)}
    counts = np.zeros((len(samples), 3), dtype=np.float64)
    for row in paired_edges:
        i = index[row["sample"]]
        counts[i] += [
            1,
            int(row["changed_recovered"]) - int(row["reference_recovered"]),
            int(row["changed_top1"]) - int(row["reference_top1"]),
        ]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(samples), size=(repeats, len(samples)))
    totals = counts[draws].sum(axis=1)
    totals = totals[totals[:, 0] > 0]
    bounds = [(1 - confidence) / 2, 1 - (1 - confidence) / 2]
    return {
        "unit": "video",
        "video_count": len(samples),
        "repeats": repeats,
        "confidence": confidence,
        "nonempty_repeats": len(totals),
        "recall_delta_interval": np.quantile(totals[:, 1] / totals[:, 0], bounds).tolist()
        if len(totals)
        else None,
        "top1_delta_interval": np.quantile(totals[:, 2] / totals[:, 0], bounds).tolist()
        if len(totals)
        else None,
    }
