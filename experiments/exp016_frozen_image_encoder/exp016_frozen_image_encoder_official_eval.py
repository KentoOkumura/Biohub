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
# # exp016 frozen image encoder: recovered-graph official evaluation
#
# This Kaggle Notebook evaluates the 199 checksum-verified repaired graphs
# recovered from Kaggle official-evaluation version 5. That run completed the
# unchanged exp015 motion, gap, division, DeepCenter, short-track, and smoothing
# stages before its CPU-only evaluator tried to load an unused image tensor.
# This recovery run skips both tracker inference and graph repair, then evaluates
# the saved graphs with the pinned public evaluator. It does not train a model,
# rerun the image encoder, or create a competition submission.
#
# ## Contents
#
# 1. Runtime and pinned inputs
# 2. Fixed environment setup and repaired-graph recovery
# 3. Official evaluator preflight
# 4. Official metrics
# 5. Evidence receipt

# %%
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import graph_inference as gi
import numpy as np
import official_evaluation as oe
import torch

EXPERIMENT = "exp016_frozen_image_encoder"
EXECUTION_ENVIRONMENT = "kaggle"
BENCHMARK_ROLE = "official_evaluation_from_verified_kaggle_v5_repaired_graphs"
WORKING_ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
INPUT_ROOT = Path("/kaggle/input")
INPUT_NOTEBOOK_ROOT = Path("/kaggle/input/notebooks")
BASE_SOURCE_PATH = Path("exp015_inference_base.py")
BASE_SOURCE_SHA256 = "d9a7f46dda21a01d03ffa0072e4e4396dd2542ec4d4996b934bec5b116f49e04"
PUBLIC_EVALUATOR_SHA256 = "614813cc51c3581c6ccda4bb20725a19da8ecac4a27620654bfca58319cffa3c"
RECOVERED_FULL_SUMMARY_SHA256 = "49a3a90758bc64f169f9e2950559d2b0827e864257c5c6d1e98fb1ace2368245"
RECOVERED_REPAIRED_MANIFEST_FILE_SHA256 = (
    "90ba3a345b000b7a0b347eaca5018835725b55ccd326f5f437688c691aa62a88"
)
RECOVERED_REPAIRED_CONTENT_SHA256 = (
    "22af437949d27d1d0afb6c85c76f1fb3ab4a9bfc4198f570515d6c05bb75aafb"
)
OFFICIAL_EVAL_RUNTIME_GATE_SECONDS = 12 * 60 * 60
METRICS_PATH = WORKING_ROOT / "metrics.json"
OFFICIAL_SUMMARY_PATH = WORKING_ROOT / "official_graph_evaluation.json"
PER_SAMPLE_PATH = WORKING_ROOT / "official_graph_per_sample.json"
EXECUTION_RECEIPT_PATH = WORKING_ROOT / "official_evaluation_receipt.json"
OFFICIAL_PREFLIGHT_PATH = WORKING_ROOT / "official_evaluator_preflight.json"
started = time.monotonic()

if not INPUT_NOTEBOOK_ROOT.exists():
    raise RuntimeError("official graph evaluation must run in a Kaggle Notebook")
if gi.sha256_file(BASE_SOURCE_PATH) != BASE_SOURCE_SHA256:
    raise RuntimeError("the embedded exp015 fixed-pipeline source SHA changed")


def resolve_recovered_repaired_root() -> Path:
    candidates = [
        INPUT_ROOT / "datasets" / "kentookumura" / "exp016-repaired-graphs-v5",
        INPUT_ROOT / "exp016-repaired-graphs-v5",
    ]
    candidates.extend(
        path.parent
        for path in INPUT_ROOT.rglob("repaired_graph_manifest.json")
        if "exp016-repaired-graphs-v5" in str(path)
    )
    matches: list[Path] = []
    for candidate in candidates:
        if candidate in matches or not candidate.is_dir():
            continue
        manifest_path = candidate / "repaired_graph_manifest.json"
        if (
            manifest_path.is_file()
            and gi.sha256_file(manifest_path) == RECOVERED_REPAIRED_MANIFEST_FILE_SHA256
        ):
            matches.append(candidate)
    if len(matches) != 1:
        raise RuntimeError(f"expected one verified repaired-graph dataset, found {matches}")
    return matches[0]


RECOVERED_REPAIRED_ROOT = resolve_recovered_repaired_root()
print(f"Benchmark role: {BENCHMARK_ROLE} | execution environment: {EXECUTION_ENVIRONMENT}")
print("Recovered repaired-graph input:", RECOVERED_REPAIRED_ROOT)
print("Visible GPU count:", torch.cuda.device_count())
print("Additional training: 0 | image encoder forwards: 0 | tracker forwards: 0")
print("Kaggle competition submission created: False")

# %% [markdown]
# ## 2. Fixed environment setup and repaired-graph recovery
#
# The SHA-pinned exp015 source still performs its dependency, source,
# checkpoint, train-image, and configuration checks. Prediction and graph repair
# are skipped because their 199 compact outputs were recovered from version 5.
# Every saved graph and both repair evidence files are checksum-verified here.

# %%
base_source = BASE_SOURCE_PATH.read_text(encoding="utf-8")
patched_source = oe.patch_exp015_source_for_repaired_graph_evaluation(base_source)
runtime_namespace = {
    **globals(),
    "started": started,
}
exec(
    compile(patched_source, "exp016_repaired_graph_evaluation_setup.py", "exec"),
    runtime_namespace,
)
repo_dir = Path(runtime_namespace["REPO_DIR"])
train_data_dir = Path(runtime_namespace["TEST_DIR"])
repaired_compact_root = WORKING_ROOT / "oracle_final_graphs"
recovered_input_receipt_path = WORKING_ROOT / "recovered_ilp_input_receipt.json"
run_stats_path = WORKING_ROOT / "run_stats.csv"
repaired_manifest_path = RECOVERED_REPAIRED_ROOT / "repaired_graph_manifest.json"
repaired_manifest = json.loads(repaired_manifest_path.read_text(encoding="utf-8"))
manifest_payload = dict(repaired_manifest)
manifest_sha256 = str(manifest_payload.pop("manifest_sha256"))
if gi.json_sha256(manifest_payload) != manifest_sha256:
    raise RuntimeError("repaired-graph manifest payload checksum changed")
if (
    repaired_manifest.get("source_kernel_version") != 5
    or repaired_manifest.get("graph_count") != 199
    or repaired_manifest.get("graph_repair_complete") is not True
    or repaired_manifest.get("compact_graph_content_sha256") != RECOVERED_REPAIRED_CONTENT_SHA256
    or repaired_manifest.get("graph_names") != sorted(runtime_namespace["test_stems"])
):
    raise RuntimeError("repaired-graph manifest contract changed")
expected_graph_records = repaired_manifest["graph_files_sha256"]
observed_graph_records = [
    [path.name, gi.sha256_file(path)]
    for path in sorted((RECOVERED_REPAIRED_ROOT / "oracle_final_graphs").glob("*.npz"))
]
if observed_graph_records != expected_graph_records:
    raise RuntimeError("recovered repaired-graph file checksums changed")
if repaired_compact_root.exists():
    shutil.rmtree(repaired_compact_root)
shutil.copytree(RECOVERED_REPAIRED_ROOT / "oracle_final_graphs", repaired_compact_root)
shutil.copy2(RECOVERED_REPAIRED_ROOT / "run_stats.csv", run_stats_path)
shutil.copy2(
    RECOVERED_REPAIRED_ROOT / "recovered_ilp_input_receipt.json",
    recovered_input_receipt_path,
)
if gi.compact_graph_content_sha256(repaired_compact_root) != RECOVERED_REPAIRED_CONTENT_SHA256:
    raise RuntimeError("materialized repaired-graph content checksum changed")
if gi.sha256_file(run_stats_path) != repaired_manifest["run_stats_sha256"]:
    raise RuntimeError("recovered graph-repair statistics checksum changed")
if (
    gi.sha256_file(recovered_input_receipt_path)
    != repaired_manifest["recovered_ilp_input_receipt_sha256"]
):
    raise RuntimeError("recovered ILP input receipt checksum changed")
print("Recovered and verified all 199 repaired compact graphs from Kaggle version 5.")

# %% [markdown]
# ## 3. Official evaluator preflight
#
# Before scoring all graphs, one saved exp015 control graph is converted to
# GEFF and evaluated end to end. The evaluator source, entrypoint, and ground
# truth directory are pinned.

# %%
import polars as pl  # noqa: E402
import tracksdata as td  # noqa: E402

sys.path.insert(0, str(repo_dir / "scripts"))
sys.path.insert(0, str(repo_dir / "src"))
public_evaluator_path = repo_dir / "scripts" / "evaluate.py"
if gi.sha256_file(public_evaluator_path) != PUBLIC_EVALUATOR_SHA256:
    raise RuntimeError("the public support-pack evaluator source SHA changed")
import evaluate as public_evaluator  # noqa: E402
from biohub_tracking.io import save_graph  # noqa: E402
from biohub_tracking.metrics import summarise  # noqa: E402

if Path(public_evaluator.DATA_DIR).resolve() != train_data_dir.resolve():
    raise RuntimeError("the public evaluator GT directory does not match the fixed pipeline")
oe.force_public_evaluator_metadata_only(public_evaluator)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_compact_graph_as_geff(source: Path, destination: Path) -> None:
    nodes, indexed_edges = gi.load_compact_graph_arrays(source)
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    node_ids = [int(node_id) for node_id in graph.bulk_add_nodes(nodes)]
    if indexed_edges:
        graph.bulk_add_edges(
            [
                {"source_id": node_ids[source], "target_id": node_ids[target]}
                for source, target in indexed_edges
            ]
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_graph(graph, destination)


baseline_root = gi.resolve_verified_exp015_output(INPUT_NOTEBOOK_ROOT)
preflight_compact = sorted((baseline_root / "oracle_final_graphs").glob("*.npz"))[0]
preflight_root = WORKING_ROOT / "_official_evaluator_preflight"
if preflight_root.exists():
    shutil.rmtree(preflight_root)
preflight_geff = preflight_root / f"{preflight_compact.stem}.geff"
write_compact_graph_as_geff(preflight_compact, preflight_geff)
preflight_run = {
    "username": "exp016",
    "method": "official_evaluator_preflight",
    "split": "split_0",
    "dir": preflight_root,
    "geffs": [preflight_geff],
}
preflight_rows = public_evaluator.evaluate_run(preflight_run, max_distance=7.0)
preflight_matches = [
    row for row in preflight_rows if str(row.get("dataset")) == preflight_compact.stem
]
if len(preflight_matches) != 1 or not math.isfinite(
    float(preflight_matches[0].get("edge_tp", float("nan")))
):
    raise RuntimeError("official evaluator failed its saved-control end-to-end preflight")
preflight_summary = summarise(preflight_matches)
if not math.isfinite(float(preflight_summary.get("score", float("nan")))):
    raise RuntimeError("official evaluator preflight produced a non-finite score")
OFFICIAL_PREFLIGHT_PATH.write_text(
    json.dumps(
        json_safe(
            {
                "sample": preflight_compact.stem,
                "evaluator_source_sha256": PUBLIC_EVALUATOR_SHA256,
                "entrypoint": "evaluate_run",
                "dataset_loader": "pinned_open_dataset_with_load_image_false",
                "row": preflight_matches[0],
                "summary": preflight_summary,
                "passed": True,
            }
        ),
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
shutil.rmtree(preflight_root)
print("Official evaluator preflight passed.")

# %% [markdown]
# ## 4. Official metrics
#
# The saved exp015 public-control graphs and the repaired retrained-tracker
# graphs are each scored twice. All 199 sample rows and both aggregate summaries
# must be identical across recomputation.


# %%
def evaluate_compact_variant(
    label: str, compact_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if time.monotonic() - started >= OFFICIAL_EVAL_RUNTIME_GATE_SECONDS:
        raise TimeoutError("exp016 official evaluation exceeded the 12-hour runtime gate")
    temporary_root = WORKING_ROOT / "_official_graph_eval" / label
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    temporary_root.mkdir(parents=True)
    compact_paths = sorted(compact_root.glob("*.npz"))
    names = [path.stem for path in compact_paths]
    if len(names) != 199 or len(set(names)) != 199:
        raise RuntimeError(f"{label}: expected 199 unique compact graphs")
    for path in compact_paths:
        write_compact_graph_as_geff(path, temporary_root / f"{path.stem}.geff")

    public_run = {
        "username": "exp016",
        "method": label,
        "split": "split_0",
        "dir": temporary_root,
        "geffs": sorted(temporary_root.glob("*.geff")),
    }
    rows = public_evaluator.evaluate_run(public_run, max_distance=7.0)
    if time.monotonic() - started >= OFFICIAL_EVAL_RUNTIME_GATE_SECONDS:
        raise TimeoutError("exp016 official evaluation exceeded the 12-hour runtime gate")
    evaluated_names = [str(row.get("dataset")) for row in rows]
    invalid_rows = [
        name
        for name, row in zip(evaluated_names, rows, strict=True)
        if not math.isfinite(float(row.get("edge_tp", float("nan"))))
    ]
    if evaluated_names != names or invalid_rows or len(rows) != 199:
        raise RuntimeError(
            {
                "label": label,
                "evaluated": len(rows),
                "expected_names_match": evaluated_names == names,
                "invalid_rows": invalid_rows,
            }
        )
    named_rows = [
        {"sample": name, "embryo": name.split("_", 1)[0], **row}
        for name, row in zip(names, rows, strict=True)
    ]
    overall = summarise(rows)
    by_embryo = {
        embryo: summarise(
            [
                {key: value for key, value in row.items() if key not in {"sample", "embryo"}}
                for row in named_rows
                if row["embryo"] == embryo
            ]
        )
        for embryo in ("44b6", "6bba")
    }
    recheck_rows = public_evaluator.evaluate_run(public_run, max_distance=7.0)
    if time.monotonic() - started >= OFFICIAL_EVAL_RUNTIME_GATE_SECONDS:
        raise TimeoutError("exp016 official evaluation exceeded the 12-hour runtime gate")
    recheck = summarise(recheck_rows)
    if json_safe(recheck) != json_safe(overall):
        raise RuntimeError(f"{label}: official metric recomputation changed")
    return {
        "label": label,
        "overall": overall,
        "by_embryo": by_embryo,
        "evaluated_count": len(rows),
        "skipped": [],
        "recomputed_from_saved_graphs": True,
        "recomputed_summary_matches": True,
        "compact_graph_content_sha256": gi.compact_graph_content_sha256(compact_root),
    }, named_rows


control_summary, control_rows = evaluate_compact_variant(
    "fixed_public_control", baseline_root / "oracle_final_graphs"
)
retrained_summary, retrained_rows = evaluate_compact_variant(
    "fold_specific_retrained_primary_tracker", repaired_compact_root
)
metric_keys = (
    "score",
    "adj_edge_jaccard",
    "edge_jaccard",
    "division_jaccard",
    "node_recall",
)
delta = {
    "overall": {
        key: float(retrained_summary["overall"][key]) - float(control_summary["overall"][key])
        for key in metric_keys
    },
    "by_embryo": {
        embryo: {
            key: float(retrained_summary["by_embryo"][embryo][key])
            - float(control_summary["by_embryo"][embryo][key])
            for key in metric_keys
        }
        for embryo in ("44b6", "6bba")
    },
}
official_evaluation = {
    "experiment": EXPERIMENT,
    "benchmark_role": BENCHMARK_ROLE,
    "execution_environment": EXECUTION_ENVIRONMENT,
    "metric": "official_adjusted_edge_jaccard_plus_0.1_division_jaccard",
    "max_matching_distance_um": 7.0,
    "conditional_validation": True,
    "source_ilp_runtime": "colab_t4_cache_replay",
    "source_full_replay_summary_sha256": RECOVERED_FULL_SUMMARY_SHA256,
    "control": control_summary,
    "retrained": retrained_summary,
    "delta_retrained_minus_control": delta,
    "fixed_candidate_coordinates_exact": True,
    "public_control_candidate_graphs_exact": True,
    "post_retention_candidates_reused_from_exp015": True,
    "main_image_encoder_forward_count": 0,
    "tracker_forward_count_in_this_stage": 0,
    "fold_model_by_evaluation_embryo": {"6bba": 0, "44b6": 1},
    "graph_repair_complete": True,
    "graph_repair_reused_from_kernel_version": 5,
    "evaluator_dataset_loader": "pinned_open_dataset_with_load_image_false",
    "official_metric_computed": True,
    "submission_created": False,
}
OFFICIAL_SUMMARY_PATH.write_text(
    json.dumps(json_safe(official_evaluation), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
PER_SAMPLE_PATH.write_text(
    json.dumps(
        json_safe({"control": control_rows, "retrained": retrained_rows}),
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

# %% [markdown]
# ## 5. Evidence receipt
#
# The output keeps the repaired compact graphs, aggregate and per-sample
# official metrics, repair statistics, and checksums. User judgment remains
# separate from this execution state.

# %%
elapsed_seconds = time.monotonic() - started
recovered_input_receipt = json.loads(recovered_input_receipt_path.read_text(encoding="utf-8"))
receipt = {
    "experiment": EXPERIMENT,
    "benchmark_role": BENCHMARK_ROLE,
    "execution_environment": EXECUTION_ENVIRONMENT,
    "stage": "recovered_repaired_graph_official_evaluation",
    "elapsed_seconds": elapsed_seconds,
    "visible_gpu_count": torch.cuda.device_count(),
    "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    "source_full_replay_summary_sha256": RECOVERED_FULL_SUMMARY_SHA256,
    "recovered_repaired_manifest_file_sha256": RECOVERED_REPAIRED_MANIFEST_FILE_SHA256,
    "recovered_repaired_manifest_payload_sha256": manifest_sha256,
    "recovered_ilp_input_receipt_sha256": gi.sha256_file(recovered_input_receipt_path),
    "recovered_ilp_archive_manifest_sha256": recovered_input_receipt["archive_manifest_sha256"],
    "control_compact_graph_content_sha256": control_summary["compact_graph_content_sha256"],
    "retrained_compact_graph_content_sha256": retrained_summary["compact_graph_content_sha256"],
    "run_stats_sha256": gi.sha256_file(run_stats_path),
    "official_evaluation_sha256": gi.sha256_file(OFFICIAL_SUMMARY_PATH),
    "official_evaluator_preflight_sha256": gi.sha256_file(OFFICIAL_PREFLIGHT_PATH),
    "per_sample_metrics_sha256": gi.sha256_file(PER_SAMPLE_PATH),
    "fixed_candidate_coordinates_exact": True,
    "public_control_candidate_graphs_exact": True,
    "post_retention_candidates_reused_from_exp015": True,
    "main_image_encoder_forward_count": 0,
    "tracker_forward_count_in_this_stage": 0,
    "additional_training": False,
    "graph_repair_complete": True,
    "graph_repair_reused_from_kernel_version": 5,
    "evaluator_dataset_loader": "pinned_open_dataset_with_load_image_false",
    "official_metric_computed": True,
    "submission_created": False,
}
receipt["receipt_sha256"] = gi.json_sha256(receipt)
EXECUTION_RECEIPT_PATH.write_text(
    json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)

metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
metrics = gi.deep_merge(
    metrics,
    {
        "status": "debug_completed",
        "cv": None,
        "metric": "official_adjusted_edge_jaccard_plus_0.1_division_jaccard",
        "updated_at": datetime.now(UTC).isoformat(),
        "notes": (
            "Official evaluation completed on Kaggle from checksum-verified repaired "
            "graphs recovered from kernel version 5. User completion and adoption judgment "
            "remain pending. No submission was created."
        ),
        "official_graph_evaluation": official_evaluation,
        "evidence": {
            "artifacts": {
                "selected_mode": "fold_specific_primary_tracker_cached_graph_replay",
                "selected_model": "two_held_out_embryo_primary_trackers",
                "oof_prediction_sha": retrained_summary["compact_graph_content_sha256"],
                "test_prediction_content_sha": None,
                "submission_sha": None,
            },
            "official_evaluation_stage": receipt,
        },
    },
)
METRICS_PATH.write_text(
    json.dumps(json_safe(metrics), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)

for temporary in (
    WORKING_ROOT / "tracking_repo",
    WORKING_ROOT / "_official_graph_eval",
):
    if temporary.exists():
        shutil.rmtree(temporary)
temporary_rows = WORKING_ROOT / "final_graph_rows.csv"
if temporary_rows.is_file():
    temporary_rows.unlink()

print(json.dumps(json_safe(official_evaluation), indent=2, ensure_ascii=False, sort_keys=True))
print("Official graph evaluation:", OFFICIAL_SUMMARY_PATH)
print("Execution receipt:", EXECUTION_RECEIPT_PATH)
print("Kaggle competition submission created: False")
