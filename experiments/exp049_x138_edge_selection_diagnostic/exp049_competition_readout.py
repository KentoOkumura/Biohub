"""Read out score and ILP choices for known events added by exp047's candidate expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
EVIDENCE = EXP / "artifacts/evidence/diagnostic_v6"
CACHE = EXP.parent / "exp047_x138_edge_candidates/artifacts/evidence/inference_v1/candidate_cache"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_key(row: dict) -> tuple[str, str, tuple[tuple[int, int], ...]]:
    return row["stem"], row["type"], tuple(sorted(tuple(edge) for edge in row["event_edges"]))


def cache_scores(path: Path, expected_sha: str) -> dict[tuple[int, int], float]:
    if sha256(path) != expected_sha:
        raise ValueError(f"cache SHA differs from input preflight: {path}")
    with np.load(path, allow_pickle=False) as payload:
        scores = {
            (int(source), int(target)): float(probability)
            for source, target, probability in zip(
                payload["edge_src"], payload["edge_tgt"], payload["edge_prob"], strict=True
            )
        }
        if len(scores) != len(payload["edge_src"]):
            raise ValueError(f"duplicate score edge: {path}")
        return scores


def selected_parent(row: dict, target: int, scores: dict) -> dict | None:
    if row["type"] == "edge":
        parents = row["ilp_selected_parents_for_daughter"]
    else:
        parents = row["ilp_selected_parents_for_daughters"][str(target)]
    if len(parents) > 1:
        raise ValueError(f"ILP indegree exceeds one: {row['stem']}/{target}")
    if not parents:
        return None
    parent = int(parents[0])
    return {"parent": parent, "score": scores[parent, target]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVIDENCE / "edge_events.json")
    parser.add_argument("--input-checks", type=Path, default=EVIDENCE / "input_checks.json")
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=EVIDENCE / "competition_readout.json")
    args = parser.parse_args()

    metrics = json.loads((EXP / "metrics.json").read_text())
    expected = metrics["evidence"]["full_diagnostic"]["artifact_sha256"]
    for path in (args.events, args.input_checks):
        if sha256(path) != expected[path.name]:
            raise ValueError(f"diagnostic artifact SHA differs from metrics: {path}")
    checks = json.loads(args.input_checks.read_text())["videos"]
    rows = json.loads(args.events.read_text())
    by_arm = {arm: {} for arm in ("baseline", "expanded")}
    for row in rows:
        key = event_key(row)
        arm = row["arm"]
        if key in by_arm[arm]:
            raise ValueError(f"duplicate event: {arm}/{key}")
        by_arm[arm][key] = row
    if set(by_arm["baseline"]) != set(by_arm["expanded"]):
        raise ValueError("baseline and expanded event IDs differ")
    if {key[0] for key in by_arm["expanded"]} != set(checks):
        raise ValueError("event videos differ from input preflight")

    added = defaultdict(list)
    for key, expanded in by_arm["expanded"].items():
        if expanded["stages"]["candidate"] and not by_arm["baseline"][key]["stages"]["candidate"]:
            added[expanded["stem"]].append(expanded)
    edge_details, division_details = [], []
    for stem in sorted(checks):
        scores = cache_scores(args.cache_dir / f"{stem}.npz", checks[stem]["cache_sha256"])
        for row in added[stem]:
            if row["type"] == "edge":
                parent, target = row["event_edges"][0]
                score = scores[parent, target]
                if not np.isclose(score, row["score"], rtol=0, atol=1e-7):
                    raise ValueError(
                        f"known edge score differs from event: {stem}/{parent}/{target}"
                    )
                competitor = selected_parent(row, target, scores)
                if row["stages"]["ilp"]:
                    decision = "selected"
                elif competitor is None:
                    decision = "no_selected_parent_edge"
                elif competitor["score"] > score:
                    decision = "higher_score_parent_selected"
                elif competitor["score"] < score:
                    decision = "lower_score_parent_selected"
                else:
                    decision = "equal_score_parent_selected"
                edge_details.append(
                    {
                        "stem": stem,
                        "embryo": row["embryo"],
                        "edge": [parent, target],
                        "score": score,
                        "rank": row["rank"],
                        "ilp_selected": row["stages"]["ilp"],
                        "final_selected": row["stages"]["final"],
                        "selected_parent_for_daughter": competitor,
                        "selected_child_count_for_true_parent": len(
                            row["ilp_selected_children_for_parent"]
                        ),
                        "decision": decision,
                    }
                )
            elif row["type"] == "division":
                edges = [tuple(edge) for edge in row["event_edges"]]
                division_details.append(
                    {
                        "stem": stem,
                        "embryo": row["embryo"],
                        "edges": [list(edge) for edge in edges],
                        "edge_scores": [scores[edge] for edge in edges],
                        "selected_parent_for_daughter": [
                            selected_parent(row, edge[1], scores) for edge in edges
                        ],
                        "selected_children_for_true_parent": row[
                            "ilp_selected_children_for_parent"
                        ],
                        "true_edges_selected_by_ilp": sum(
                            edge[1] in row["ilp_selected_children_for_parent"] for edge in edges
                        ),
                        "both_true_edges_selected_by_ilp": row["stages"]["ilp"],
                        "both_true_edges_selected_final": row["stages"]["final"],
                    }
                )
            else:
                raise ValueError(f"unexpected event type: {row['type']}")
    by_embryo = {}
    for embryo in ("44b6", "6bba"):
        edges = [row for row in edge_details if row["embryo"] == embryo]
        divisions = [row for row in division_details if row["embryo"] == embryo]
        by_embryo[embryo] = {
            "new_known_edges_candidate": len(edges),
            "new_known_edges_ilp": sum(row["ilp_selected"] for row in edges),
            "new_known_edges_final": sum(row["final_selected"] for row in edges),
            "ilp_decision": dict(Counter(row["decision"] for row in edges)),
            "ilp_rejected_rank1": sum(
                row["rank"] == 1 and not row["ilp_selected"] for row in edges
            ),
            "ilp_rejected_true_parent_selected_child_count": dict(
                Counter(
                    str(row["selected_child_count_for_true_parent"])
                    for row in edges
                    if not row["ilp_selected"]
                )
            ),
            "new_known_divisions_candidate": len(divisions),
            "new_known_divisions_ilp": sum(
                row["both_true_edges_selected_by_ilp"] for row in divisions
            ),
            "division_true_edges_selected_by_ilp": dict(
                Counter(str(row["true_edges_selected_by_ilp"]) for row in divisions)
            ),
            "division_true_parent_selected_child_count": dict(
                Counter(str(len(row["selected_children_for_true_parent"])) for row in divisions)
            ),
        }
    output = {
        "source_edge_events_sha256": expected[args.events.name],
        "source_input_checks_sha256": expected[args.input_checks.name],
        "by_embryo": by_embryo,
        "new_edge_details": edge_details,
        "new_division_details": division_details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(by_embryo, ensure_ascii=False, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
