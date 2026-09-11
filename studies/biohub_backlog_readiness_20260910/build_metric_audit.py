from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/exp003_official_metric_audit"
cfg = yaml.safe_load((EXP / "config.yaml").read_text())
public_path = Path("/tmp/biohub-baselines-20260910/dctta-source.txt")
public_text = public_path.read_text()
names = {"match_nodes_bipartite", "compute_edge_confusion", "edge_jaccard", "adjusted_jaccard", "weakly_connected_components", "compute_division_confusion"}
public_tree = ast.parse(public_text)
public_functions = [node for node in public_tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(public_functions) == len(names)
public_code = "\n\n".join(ast.get_source_segment(public_text, node) for node in public_functions)
(EXP / "assets/public_proxy.py").write_text(public_code + "\n")
manifest_file = EXP / "assets/source_manifest.json"
manifest = json.loads(manifest_file.read_text())
manifest["public_proxy"] = {"kernel": "sjlee101/biohub-lf-dctta-v020", "source_sha256": hashlib.sha256(public_path.read_bytes()).hexdigest(), "functions": sorted(names)}
manifest["files"]["public_proxy.py"] = hashlib.sha256((EXP / "assets/public_proxy.py").read_bytes()).hexdigest()
manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")

nodes = {"0": [0, 0, 0, 0], "1": [1, 0, 0, -30], "2": [1, 0, 0, 30], "3": [2, 0, 0, -32], "4": [2, 0, 0, 32], "5": [3, 0, 0, -34], "6": [3, 0, 0, 34]}
edges = [[0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
fixtures = []
def fixture(name, pred_nodes, pred_edges, gt_nodes=nodes, gt_edges=edges, expected=None):
    fixtures.append({"name": name, "pred_nodes": pred_nodes, "pred_edges": pred_edges, "gt_nodes": gt_nodes, "gt_edges": gt_edges, "estimated_total_nodes": len(gt_nodes), "expected_official": expected or {}})
fixture("correct_division", nodes, edges, expected={"edge_tp": 6, "edge_fp": 0, "edge_fn": 0, "division_tp": 1, "division_fp": 0, "division_fn": 0})
fixture("distant_division", {"0": nodes["0"], "10": [1, 0, 0, 500], "11": [2, 0, 0, 500], "5": nodes["5"], "6": nodes["6"]}, [[0, 10], [10, 11], [11, 5], [11, 6]], expected={"division_tp": 0, "division_fn": 1})
fixture("duplicate_edge", nodes, edges + [[0, 1]], expected={"edge_tp": 6, "edge_fn": 0})
fixture("nonconsecutive_edge", nodes, edges + [[0, 3]], expected={"edge_tp": 6, "edge_fn": 0})
fixture("merged_daughter", nodes, edges + [[1, 4]])
fixture("three_children", {**nodes, "7": [1, 0, 0, 90]}, edges + [[0, 7]])
fixture("no_edges", nodes, [], expected={"edge_tp": 0, "edge_fp": 0, "edge_fn": 6, "division_tp": 0, "division_fn": 1})
fixture("unannotated_component", {**nodes, "8": [2, 0, 0, 999], "9": [3, 0, 0, 999]}, edges + [[8, 9]], expected={"edge_tp": 6, "edge_fp": 0, "edge_fn": 0})
linear_nodes = {"0": [0, 0, 0, 0], "1": [1, 0, 0, 0]}
fixture("no_divisions", linear_nodes, [[0, 1]], linear_nodes, [[0, 1]], {"edge_tp": 1, "edge_fp": 0, "edge_fn": 0, "division_tp": 0, "division_fn": 0})
(EXP / "assets/fixtures.json").write_text(json.dumps(fixtures, indent=2) + "\n")
cfg["audit"]["fixture_count"] = len(fixtures)
(EXP / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

start = '''# %% [markdown]
# # exp003 Official metric audit
#
# Compare the complete pinned official evaluator with the archived public
# notebook evaluator. Fixed synthetic graphs and four existing predictions
# are diagnostic inputs; these results are not independent-embryo CV.
#
# ## Contents
# 1. Imports, configuration, and offline dependencies
# 2. Complete official division and edge evaluation
# 3. Archived public evaluator
# 4. Graph conversion, input contracts, and result helpers
# 5. Synthetic graph checks
# 6. Fixed predictions against official annotations
# 7. Aggregation, evidence, and artifact persistence

# %% [markdown]
# ## 1. Imports, configuration, and offline dependencies

# %%
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, NamedTuple

import yaml

WORKING = Path("/kaggle/working")
if not Path("/kaggle/input").is_dir():
    raise RuntimeError("The authoritative audit must execute on Kaggle.")
CONFIG = yaml.safe_load((WORKING / "config.yaml").read_text())
AUDIT = CONFIG["audit"]
ARTIFACTS = WORKING / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
ASSETS = WORKING / "assets"
STARTED = time.monotonic()
wheels = sorted(p for p in Path("/kaggle/input").rglob("wheels")
                if p.is_dir() and any(p.glob("tracksdata-*.whl")))
if len(wheels) != 1:
    raise RuntimeError(f"expected one offline wheel directory, found {wheels}")
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                "--find-links", str(wheels[0]), "--no-deps",
                *AUDIT["offline_packages"]], check=True)

import numpy as np
import polars as pl
import tracksdata as td
from geff import GeffMetadata
from scipy.optimize import linear_sum_assignment

print("AUDIT_CONFIG", json.dumps(AUDIT), flush=True)
SOURCE_MANIFEST = json.loads((ASSETS / "source_manifest.json").read_text())
for filename, expected in SOURCE_MANIFEST["files"].items():
    observed = hashlib.sha256((ASSETS / filename).read_bytes()).hexdigest()
    if observed != expected:
        raise RuntimeError(f"source SHA mismatch: {filename}")
assert SOURCE_MANIFEST["commit"] == AUDIT["source_commit"]
print("SOURCE_VERIFIED", SOURCE_MANIFEST["commit"], flush=True)
'''
division = (EXP / "assets/division_metrics.py").read_text()
metrics = (EXP / "assets/metrics.py").read_text()
# Only a package-relative import is removed; the referenced function is already
# present in the self-contained notebook. Tests compare every definition's AST.
metrics = metrics.replace("    from .division_metrics import evaluate_divisions\n", "")
parts = [start, "# %% [markdown]\n# ## 2. Complete official division and edge evaluation\n\n# %%\n" + division + "\n\n# %%\n" + metrics, "# %% [markdown]\n# ## 3. Archived public evaluator\n\n# %%\n" + public_code]
parts.append('''# %% [markdown]
# ## 4. Graph conversion, input contracts, and result helpers

# %%
assert ADJUSTMENT_ALPHA == AUDIT["official_adjustment_alpha"]
assert SCORE_DIVISION_WEIGHT == AUDIT["official_division_weight"]
VOXEL_SCALE_UM = tuple(AUDIT["scale_zyx_um"])


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def save_json(path, payload):
    path.write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\\n")


def build_audit_graph(nodes, edges):
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    original_ids = list(nodes)
    new_ids = graph.bulk_add_nodes([
        {"t": int(nodes[i][0]), "z": float(nodes[i][1]),
         "y": float(nodes[i][2]), "x": float(nodes[i][3])}
        for i in original_ids
    ])
    mapping = dict(zip(original_ids, new_ids, strict=True))
    if edges:
        graph.bulk_add_edges([
            {"source_id": mapping[source], "target_id": mapping[target]}
            for source, target in edges
        ])
    return graph, mapping


def graph_arrays(graph):
    nodes = {
        int(row["node_id"]): [int(row["t"]), float(row["z"]),
                             float(row["y"]), float(row["x"])]
        for row in graph.node_attrs().iter_rows(named=True)
    }
    edges = [(int(row["source_id"]), int(row["target_id"]))
             for row in graph.edge_attrs().iter_rows(named=True)]
    return nodes, edges


def read_csv_graphs(path):
    grouped = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            entry = grouped.setdefault(row["dataset"], {"nodes": {}, "edges": []})
            if row["row_type"] == "node":
                node_id = int(row["node_id"])
                if node_id in entry["nodes"]:
                    raise ValueError(f"duplicate node ID: {row['dataset']} {node_id}")
                entry["nodes"][node_id] = [
                    int(row["t"]), float(row["z"]), float(row["y"]), float(row["x"])
                ]
            elif row["row_type"] == "edge":
                entry["edges"].append((int(row["source_id"]), int(row["target_id"])))
            else:
                raise ValueError(f"unexpected row type: {row['row_type']}")
    for name, entry in grouped.items():
        for source, target in entry["edges"]:
            if source not in entry["nodes"] or target not in entry["nodes"]:
                raise ValueError(f"edge references absent node in {name}")
    return grouped


def evaluate_both(name, pred_nodes, pred_edges, gt_nodes, gt_edges, n_total, scale):
    global VOXEL_SCALE_UM
    VOXEL_SCALE_UM = tuple(scale)
    pred_graph, pred_ids = build_audit_graph(pred_nodes, pred_edges)
    gt_graph, gt_ids = build_audit_graph(gt_nodes, gt_edges)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        official = evaluate(pred_graph, gt_graph, scale=tuple(scale),
                            max_distance=AUDIT["max_distance_um"])
        recall = node_recall(pred_graph, gt_graph) if pred_graph.num_edges() else 0.0
        official_row = per_sample_metrics(official, n_total, recall)
        official_summary = summarise([official_row])
    pred_to_gt, gt_to_pred = match_nodes_bipartite(
        pred_nodes, gt_nodes, max_dist=AUDIT["max_distance_um"])
    edge_counts = compute_edge_confusion(pred_edges, gt_edges, pred_to_gt, gt_to_pred)
    div_counts = compute_division_confusion(
        pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred)
    edge_score = edge_jaccard(*edge_counts)
    proxy_score = adjusted_jaccard(
        edge_score, len(pred_nodes), n_total, a=ADJUSTMENT_ALPHA)
    proxy_score += SCORE_DIVISION_WEIGHT * edge_jaccard(*div_counts)
    inverse_pred, inverse_gt = ({v: k for k, v in mapping.items()}
                               for mapping in (pred_ids, gt_ids))
    official_matches = set()
    if td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID in pred_graph.node_attr_keys():
        for row in pred_graph.node_attrs().iter_rows(named=True):
            matched = row.get(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID)
            if matched is not None and matched != -1:
                official_matches.add((inverse_pred[int(row["node_id"])],
                                      inverse_gt[int(matched)]))
    proxy_matches = set(pred_to_gt.items())
    return {
        "name": name, "official_counts": official._asdict(),
        "official_row": official_row, "official_summary": official_summary,
        "proxy_edge_tp_fp_fn": edge_counts,
        "proxy_division_tp_fp_fn": div_counts,
        "proxy_score": proxy_score,
        "score_delta_official_minus_proxy": official_summary["score"] - proxy_score,
        "node_matching_symmetric_difference_count":
            len(official_matches.symmetric_difference(proxy_matches)),
        "warnings": [str(item.message) for item in caught],
        "pred_nodes": len(pred_nodes), "pred_edges": len(pred_edges),
        "gt_nodes": len(gt_nodes), "gt_edges": len(gt_edges),
    }


# %% [markdown]
# ## 5. Synthetic graph checks

# %%
fixture_definitions = json.loads((ASSETS / "fixtures.json").read_text())
assert len(fixture_definitions) == AUDIT["fixture_count"]
fixture_results = []
failures = []
for fixture in fixture_definitions:
    try:
        result = evaluate_both(
            fixture["name"],
            {int(k): v for k, v in fixture["pred_nodes"].items()},
            [tuple(e) for e in fixture["pred_edges"]],
            {int(k): v for k, v in fixture["gt_nodes"].items()},
            [tuple(e) for e in fixture["gt_edges"]],
            fixture["estimated_total_nodes"], AUDIT["fixture_scale_zyx_um"])
        for field, expected in fixture["expected_official"].items():
            if result["official_counts"][field] != expected:
                raise AssertionError(
                    f"{fixture['name']} {field}: "
                    f"{result['official_counts'][field]} != {expected}")
        fixture_results.append(result)
        print("FIXTURE", json.dumps(clean_json(result)), flush=True)
    except Exception as error:
        failures.append({"name": fixture["name"], "error": repr(error)})
        print("FIXTURE_FAILED", failures[-1], flush=True)
save_json(ARTIFACTS / "fixture_results.json", fixture_results)
save_json(ARTIFACTS / "failures.json", failures)
if failures:
    raise RuntimeError(f"{len(failures)} synthetic checks failed; see saved evidence.")


# %% [markdown]
# ## 6. Fixed predictions against official annotations
#
# These four public samples overlap the competition's training data. They are
# used only to compare evaluators. No CV or leaderboard claim is made.

# %%
csv_candidates = sorted(Path("/kaggle/input").rglob("submission.csv"))
prediction_paths = [p for p in csv_candidates
                    if sha256_file(p) == AUDIT["prediction_sha256"]]
if not prediction_paths:
    raise FileNotFoundError("fixed prediction CSV with expected SHA is absent")
prediction_path = prediction_paths[0]
prediction_graphs = read_csv_graphs(prediction_path)
if sorted(prediction_graphs) != sorted(AUDIT["sample_ids"]):
    raise RuntimeError(f"unexpected prediction sample IDs: {sorted(prediction_graphs)}")
train_dirs = [p for p in Path("/kaggle/input").rglob("train")
              if p.is_dir() and all((p / f"{sample}.geff").is_dir()
                                   for sample in AUDIT["sample_ids"])]
if len(train_dirs) != 1:
    raise RuntimeError(f"expected one matching train GEFF directory: {train_dirs}")
real_results = []
for sample in AUDIT["sample_ids"]:
    try:
        gt_path = train_dirs[0] / f"{sample}.geff"
        loaded = td.graph.IndexedRXGraph.from_geff(gt_path)
        gt_graph = loaded[0] if isinstance(loaded, tuple) else loaded
        gt_nodes, gt_edges = graph_arrays(gt_graph)
        metadata = GeffMetadata.read(gt_path)
        n_total = (metadata.extra or {}).get("estimated_number_of_nodes")
        if n_total is None or float(n_total) <= 0:
            raise ValueError(f"missing estimated total node count: {sample}")
        prediction = prediction_graphs[sample]
        result = evaluate_both(sample, prediction["nodes"], prediction["edges"],
                               gt_nodes, gt_edges, float(n_total),
                               AUDIT["scale_zyx_um"])
        result["estimated_total_nodes_for_scoring_only"] = float(n_total)
        real_results.append(result)
        save_json(ARTIFACTS / "sample_results.json", real_results)
        print("SAMPLE", json.dumps(clean_json(result)), flush=True)
    except Exception as error:
        failures.append({"name": sample, "error": repr(error)})
        print("SAMPLE_FAILED", failures[-1], flush=True)
save_json(ARTIFACTS / "failures.json", failures)
if failures or len(real_results) != len(AUDIT["sample_ids"]):
    raise RuntimeError("real sample audit incomplete; inspect saved evidence")


# %% [markdown]
# ## 7. Aggregation, evidence, and artifact persistence

# %%
official_summary = summarise([row["official_row"] for row in real_results])
official_order = sorted(real_results,
                        key=lambda row: row["official_summary"]["score"], reverse=True)
proxy_order = sorted(real_results, key=lambda row: row["proxy_score"], reverse=True)
summary = {
    "diagnostic_only": True,
    "comparison": "fixed predictions; evaluator comparison, not independent CV",
    "official_summary": official_summary,
    "sample_order_by_official_score": [r["name"] for r in official_order],
    "sample_order_by_proxy_score": [r["name"] for r in proxy_order],
    "fixture_count": len(fixture_results), "sample_count": len(real_results),
    "failed_count": len(failures), "expected_sample_ids": AUDIT["sample_ids"],
    "max_absolute_sample_score_difference":
        max(abs(row["score_delta_official_minus_proxy"]) for row in real_results),
    "notebook_runtime_seconds": time.monotonic() - STARTED,
}
save_json(ARTIFACTS / "audit_summary.json", summary)
with (ARTIFACTS / "sample_comparison.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        "sample_id", "official_score", "proxy_score", "difference",
        "official_edge_tp", "official_edge_fp", "official_edge_fn",
        "official_division_tp", "official_division_fp", "official_division_fn",
        "node_matching_difference_count"])
    writer.writeheader()
    for row in real_results:
        writer.writerow({
            "sample_id": row["name"],
            "official_score": row["official_summary"]["score"],
            "proxy_score": row["proxy_score"],
            "difference": row["score_delta_official_minus_proxy"],
            **{"official_" + key: value for key, value in row["official_counts"].items()
               if key != "num_pred_nodes"},
            "node_matching_difference_count":
                row["node_matching_symmetric_difference_count"],
        })
environment = {
    "python": platform.python_version(),
    "packages": {name: importlib.metadata.version(name)
                 for name in ["numpy", "scipy", "polars", "tracksdata", "geff"]},
    "source_manifest_sha256": sha256_file(ASSETS / "source_manifest.json"),
    "fixture_sha256": sha256_file(ASSETS / "fixtures.json"),
    "prediction_sha256": sha256_file(prediction_path),
    "resource": "CPU", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
}
save_json(ARTIFACTS / "environment.json", environment)
artifact_manifest = {
    p.name: sha256_file(p) for p in sorted(ARTIFACTS.iterdir()) if p.is_file()
}
save_json(ARTIFACTS / "artifact_manifest.json", artifact_manifest)
metrics_path = WORKING / "metrics.json"
metric_record = json.loads(metrics_path.read_text())
metric_record.update(status="debug_completed",
                     updated_at=datetime.now(timezone.utc).isoformat(),
                     notes="Full CPU audit executed; experiment completion awaits user judgement.")
metric_record["diagnostic_validation"] = {"official_metric_audit": summary}
metric_record["evidence"]["artifacts"].update(
    input_file_sha=environment["prediction_sha256"],
    row_count=len(real_results), group_count=len(AUDIT["sample_ids"]),
    model_count=0, source_manifest_sha=environment["source_manifest_sha256"],
    audit_manifest_sha=sha256_file(ARTIFACTS / "artifact_manifest.json"))
metric_record["evidence"]["kaggle"].update(
    notebook_runtime_seconds=summary["notebook_runtime_seconds"],
    internet_enabled=False, resource={"cpu_only": True})
save_json(metrics_path, metric_record)
print("AUDIT_SUMMARY", json.dumps(clean_json(summary)), flush=True)
''')
notebook_source = "\n\n".join(parts) + "\n"
(EXP / "exp003_official_metric_audit_audit.py").write_text(notebook_source)
# Keep unmodified vendor files out of style rewriting; AST identity is tested.
(EXP / "ruff.toml").write_text('extend = "../../pyproject.toml"\nextend-exclude = ["assets"]\n[lint.per-file-ignores]\n"exp003_official_metric_audit_audit.py" = ["E402", "E501", "I001", "UP", "B905"]\n')
print(f"built self-contained source: {len(notebook_source.splitlines())} lines, {len(fixtures)} fixtures")
