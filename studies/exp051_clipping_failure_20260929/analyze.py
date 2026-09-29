"""Describe exp051's saved outer evaluation; never rerun or retune inference."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/exp051_x138_ilp_score_clipping"
SAVED = EXP / "artifacts/kaggle_evaluate/resume_v4_summary/exp051_clipping"
OUT = Path(__file__).resolve().parent
CONTROL = "expanded_optuna_cost"
CLIPPED = CONTROL + "_clipped"
ARMS = (CONTROL, CLIPPED)
STAGE_ORDER = (
    "candidate",
    "ilp",
    "edge_filter",
    "relink",
    "gap1",
    "gap2",
    "low_detection",
    "division",
    "short_track",
    "final",
)


def load(path):
    return json.loads(path.read_text())


def summary(rows):
    totals = {
        key: sum(r[key] for r in rows)
        for key in (
            "edge_tp",
            "edge_fp",
            "edge_fn",
            "division_tp",
            "division_fp",
            "division_fn",
            "num_pred_nodes",
        )
    }
    union = totals["edge_tp"] + totals["edge_fp"] + totals["edge_fn"]
    raw = totals["edge_tp"] / union
    adjusted = (
        sum((r["edge_tp"] + r["edge_fp"] + r["edge_fn"]) * r["adj_edge_jaccard"] for r in rows)
        / union
    )
    div_union = sum(totals[k] for k in ("division_tp", "division_fp", "division_fn"))
    div = totals["division_tp"] / div_union if div_union else 0.0
    return {
        **totals,
        "n": len(rows),
        "edge_union": union,
        "edge_jaccard": raw,
        "adj_edge_jaccard": adjusted,
        "node_adjustment_contribution": adjusted - raw,
        "division_jaccard": div,
        "score": adjusted + 0.1 * div,
    }


def graph_stage_counts(stems):
    counts = {}
    sources = {}
    for stem in stems:
        root = (
            SAVED
            if stem == "6bba_fe670320"
            else EXP / "artifacts/kaggle_evaluate/v2_output/exp051_clipping"
        )
        counts[stem] = {}
        for arm in ARMS:
            directory = root / "stages" / arm / stem
            with np.load(directory / "candidate.npz") as saved:
                original_ids = saved["nodes"][:, 0].astype(np.int64)
            counts[stem][arm] = {}
            for stage in STAGE_ORDER:
                path = directory / f"{stage}.npz"
                sources[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
                with np.load(path) as saved:
                    nodes, edges = saved["nodes"], saved["edges"]
                ids = nodes[:, 0].astype(np.int64)
                _, outdegree = np.unique(edges[:, 0], return_counts=True)
                retained = int(np.isin(ids, original_ids).sum())
                counts[stem][arm][stage] = {
                    "nodes": len(nodes),
                    "edges": len(edges),
                    "retained_original_nodes": retained,
                    "added_nodes": len(nodes) - retained,
                    "branch_nodes": int((outdegree >= 2).sum()),
                }
    return counts, sources


def main():
    official = load(SAVED / "official_metric.json")
    receipt = load(SAVED / "evaluation_receipt.json")
    stages = load(SAVED / "stage_diagnostic.json")
    changes = load(SAVED / "edge_changes.json")
    rows = {arm: {r["dataset"]: r for r in official[arm]["rows"]} for arm in ARMS}
    stems = sorted(rows[CONTROL])
    graph_counts, graph_sources = graph_stage_counts(stems)
    timeout = set(receipt["time_limited_incumbents"][CLIPPED])
    groups = {
        "all": stems,
        "44b6": [s for s in stems if s.startswith("44b6_")],
        "6bba": [s for s in stems if s.startswith("6bba_")],
        "without_two_timelimit_videos": [s for s in stems if s not in timeout],
        "two_timelimit_videos": sorted(timeout),
        "6bba_without_two_timelimit_videos": [
            s for s in stems if s.startswith("6bba_") and s not in timeout
        ],
    }
    group_summary = {}
    for name, selected in groups.items():
        arms = {arm: summary([rows[arm][s] for s in selected]) for arm in ARMS}
        delta = {key: arms[CLIPPED][key] - arms[CONTROL][key] for key in arms[CONTROL]}
        stage_totals = {
            stage: {
                arm: {
                    key: sum(stages[s][arm][stage][key] for s in selected)
                    for key in stages[selected[0]][arm][stage]
                }
                for arm in ARMS
            }
            for stage in STAGE_ORDER
            if stage in stages[selected[0]][CONTROL]
        }
        graph_totals = {
            stage: {
                arm: {
                    key: sum(graph_counts[s][arm][stage][key] for s in selected)
                    for key in graph_counts[selected[0]][arm][stage]
                }
                for arm in ARMS
            }
            for stage in STAGE_ORDER
        }
        group_summary[name] = {
            "arms": arms,
            "clipped_minus_control": delta,
            "stages": stage_totals,
            "graph_stages": graph_totals,
            "edge_changes": {
                key: sum(changes[s][key] for s in selected) for key in changes[selected[0]]
            },
        }
    for arm in ARMS:
        assert (
            abs(group_summary["all"]["arms"][arm]["score"] - official[arm]["overall"]["score"])
            < 1e-12
        )
        assert (
            group_summary["all"]["graph_stages"]["final"][arm]["nodes"]
            == group_summary["all"]["arms"][arm]["num_pred_nodes"]
        )
    video_rows = []
    total_w = {arm: group_summary["all"]["arms"][arm]["edge_union"] for arm in ARMS}
    for stem in stems:
        a, b = (rows[arm][stem] for arm in ARMS)
        weight = {
            arm: sum(rows[arm][stem][k] for k in ("edge_tp", "edge_fp", "edge_fn")) for arm in ARMS
        }
        video_rows.append(
            {
                "video": stem,
                "timelimit": stem in timeout,
                "control_adjusted_jaccard": a["adj_edge_jaccard"],
                "clipped_adjusted_jaccard": b["adj_edge_jaccard"],
                "delta_adjusted_jaccard": b["adj_edge_jaccard"] - a["adj_edge_jaccard"],
                "delta_edge_tp": b["edge_tp"] - a["edge_tp"],
                "delta_edge_fp": b["edge_fp"] - a["edge_fp"],
                "delta_edge_fn": b["edge_fn"] - a["edge_fn"],
                "delta_nodes": b["num_pred_nodes"] - a["num_pred_nodes"],
                "control_node_ratio": a["total_node_ratio"],
                "clipped_node_ratio": b["total_node_ratio"],
                "contribution_to_total_score_delta": (
                    weight[CLIPPED] * b["adj_edge_jaccard"] / total_w[CLIPPED]
                    - weight[CONTROL] * a["adj_edge_jaccard"] / total_w[CONTROL]
                ),
                **changes[stem],
            }
        )
    assert (
        abs(
            sum(v["contribution_to_total_score_delta"] for v in video_rows)
            - group_summary["all"]["clipped_minus_control"]["score"]
        )
        < 1e-12
    )
    tuning = {}
    tune_sources = {}
    for fold in (0, 1):
        for suffix, arm in (("control", CONTROL), ("clipped", CLIPPED)):
            version = 2 if fold == 1 and suffix == "clipped" else 1
            path = (
                EXP
                / f"artifacts/kaggle_tuning/tune{fold}_{suffix}_v{version}"
                / "exp051_clipping/tune_receipt.json"
            )
            tune_sources[str(path.relative_to(REPO))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            tune = load(path)["arms"][arm]
            tuning[f"{fold}/{arm}"] = {
                key: tune[key]
                for key in (
                    "selected_trial",
                    "selected_cap",
                    "selected_costs",
                    "selected_summary",
                    "elapsed_seconds",
                )
            }
            tuning[f"{fold}/{arm}"]["trials"] = [
                {key: t.get(key) for key in ("trial", "status", "costs", "cap", "summary")}
                for t in tune["trials"]
            ]
    report = {
        "sources": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SAVED.glob("*.json")
        },
        "groups": group_summary,
        "videos": sorted(video_rows, key=lambda r: r["contribution_to_total_score_delta"]),
        "tuning": tuning,
        "tuning_receipt_source_sha256": tune_sources,
        "graph_stage_counts": graph_counts,
        "graph_stage_source_sha256": graph_sources,
        "scope": "Arithmetic analysis of existing Kaggle results; no retuning or new official run",
    }
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (OUT / "videos.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(video_rows[0]))
        writer.writeheader()
        writer.writerows(report["videos"])
    stage_rows = []
    for group in ("all", "44b6", "6bba"):
        for stage in STAGE_ORDER:
            for arm in ARMS:
                stage_rows.append(
                    {
                        "group": group,
                        "stage": stage,
                        "arm": arm,
                        **group_summary[group]["graph_stages"][stage][arm],
                        **group_summary[group]["stages"].get(stage, {}).get(arm, {}),
                    }
                )
    with (OUT / "stages.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stage_rows[0]))
        writer.writeheader()
        writer.writerows(stage_rows)
    tune_rows = []
    for name, tuned in tuning.items():
        for trial in tuned["trials"]:
            if not trial["summary"]:
                continue
            tune_rows.append(
                {
                    "fold_arm": name,
                    "trial": trial["trial"],
                    "selected": trial["trial"] == tuned["selected_trial"],
                    "cap": trial["cap"],
                    "costs": str(trial["costs"]),
                    **{
                        k: trial["summary"][k]
                        for k in (
                            "score",
                            "adj_edge_jaccard",
                            "division_jaccard",
                            "division_tp",
                            "division_fp",
                            "division_fn",
                        )
                    },
                }
            )
    with (OUT / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tune_rows[0]))
        writer.writeheader()
        writer.writerows(tune_rows)
    print(
        json.dumps(
            {
                "groups": {k: v["clipped_minus_control"] for k, v in group_summary.items()},
                "selected_trials": [r for r in tune_rows if r["selected"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
