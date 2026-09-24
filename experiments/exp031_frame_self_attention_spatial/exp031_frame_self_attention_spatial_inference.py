# %% [markdown]
# # exp031 frame self attention spatial: Kaggle graph evaluation
#
# This notebook replays the exact exp015 candidate cache for the legacy,
# self-only, and self-then-cross trackers. Fold 0 is used for 6bba and fold 1
# for 44b6. The secondary tracker, edge threshold, ILP, graph repair, and
# official evaluator are fixed. Each variant produces 199 repaired graphs.
#
# The saved exp015 public-checkpoint graphs remain a reference; the main
# comparison uses the legacy model retrained in this experiment. This notebook
# evaluates train graphs only and creates no Kaggle submission.

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
import torch

EXPERIMENT = "exp031_frame_self_attention_spatial"
EXECUTION_ENVIRONMENT = "kaggle"
BENCHMARK_ROLE = "canonical"
WORKING_ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
INPUT_NOTEBOOK_ROOT = Path("/kaggle/input/notebooks")
BASE_SOURCE_PATH = Path("exp015_inference_base.py")
BASE_SOURCE_SHA256 = "d9a7f46dda21a01d03ffa0072e4e4396dd2542ec4d4996b934bec5b116f49e04"
PUBLIC_EVALUATOR_SHA256 = "614813cc51c3581c6ccda4bb20725a19da8ecac4a27620654bfca58319cffa3c"
METRICS_PATH = WORKING_ROOT / "metrics.json"
OFFICIAL_SUMMARY_PATH = WORKING_ROOT / "official_graph_evaluation.json"
PER_SAMPLE_PATH = WORKING_ROOT / "official_graph_per_sample.json"
EXECUTION_RECEIPT_PATH = WORKING_ROOT / "graph_inference_receipt.json"
OFFICIAL_PREFLIGHT_PATH = WORKING_ROOT / "official_evaluator_preflight.json"
started = time.monotonic()

if not INPUT_NOTEBOOK_ROOT.exists():
    raise RuntimeError("authoritative graph inference must run in a Kaggle Notebook")
if gi.sha256_file(BASE_SOURCE_PATH) != BASE_SOURCE_SHA256:
    raise RuntimeError("the embedded exp015 fixed-pipeline source SHA changed")
if torch.cuda.device_count() < 2:
    raise RuntimeError("the authoritative graph inference requires two visible T4 GPUs")
print("GPU devices:", [torch.cuda.get_device_name(index) for index in range(2)])
print(f"Benchmark role: {BENCHMARK_ROLE} | execution environment: {EXECUTION_ENVIRONMENT}")
print("Active evaluation variants: legacy, self_only, self_cross; public reference")
print("Additional training: 0 | loaded fold models: 6 | booster models: 0")
print("Main image encoder forward passes: 0")

# %% [markdown]
# ## Prepare the fixed pipeline and validate the official evaluator
#
# The SHA-pinned exp015 source prepares the same support repository, checkpoint
# set, graph settings, and repair functions. Execution is paused before the long
# stage so the public evaluator API and GT directory are checked before any
# cached tracker replay begins.

# %%
base_source = BASE_SOURCE_PATH.read_text(encoding="utf-8")
patched_source = gi.patch_exp015_source(base_source)
cache_replay_marker = "# EXP016_CACHE_REPLAY_START"
if patched_source.count(cache_replay_marker) != 1:
    raise RuntimeError("cache replay execution marker changed")
setup_source, replay_source = patched_source.split(cache_replay_marker, 1)
runtime_namespace = {
    **globals(),
    "build_hybrid_checkpoints": gi.build_hybrid_checkpoints,
    "resolve_verified_train_output": gi.resolve_verified_train_output,
    "resolve_verified_exp015_output": gi.resolve_verified_exp015_output,
    "run_cached_graph_replay": gi.run_cached_graph_replay,
}
exec(compile(setup_source, "exp016_fixed_pipeline_setup.py", "exec"), runtime_namespace)
repo_dir = runtime_namespace["REPO_DIR"]
train_data_dir = runtime_namespace["TEST_DIR"]
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
    "username": "exp031",
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
print("Official evaluator passed an end-to-end saved-control preflight before cache replay.")

# %% [markdown]
# ## Replay tracker and graph stages from the fixed candidate cache
#
# Two GPU workers read all 19,701 exp015 windows for each configuration.
# The public tracker must reproduce the saved pre-ILP candidate graph exactly
# on every pass. The fold-specific graph then uses the fixed ILP and repair.
# No main image encoder or detection head is executed.

# %%
VARIANTS = ("legacy", "self_only", "self_cross")
variant_graph_roots: dict[str, Path] = {}
variant_replays: dict[str, dict[str, Any]] = {}
prediction_seconds_by_variant: dict[str, float] = {}
for variant in VARIANTS:
    runtime_namespace["EXP031_VARIANT"] = variant
    exec(
        compile(
            cache_replay_marker + replay_source, f"exp031_{variant}_cached_graph_replay.py", "exec"
        ),
        runtime_namespace,
    )
    prediction_seconds_by_variant[variant] = float(runtime_namespace["predict_seconds"])
    cache_replay_path = WORKING_ROOT / "cache_graph_replay_summary.json"
    if not cache_replay_path.is_file():
        raise FileNotFoundError(cache_replay_path)
    cache_replay = json.loads(cache_replay_path.read_text(encoding="utf-8"))
    required_replay = {
        "sample_count": 199,
        "window_count": 19_701,
        "main_image_encoder_forward_count": 0,
        "public_control_candidate_graphs_exact": True,
        "variant": variant,
    }
    observed_replay = {key: cache_replay.get(key) for key in required_replay}
    if observed_replay != required_replay:
        raise RuntimeError(
            {
                "cache_replay_contract_changed": {
                    "expected": required_replay,
                    "actual": observed_replay,
                }
            }
        )
    if variant != "legacy":
        reference = variant_replays["legacy"]
        for key in (
            "cache_identity_sha256",
            "public_control_candidate_graph_content_sha256",
            "candidate_coordinate_content_sha256",
        ):
            if cache_replay[key] != reference[key]:
                raise RuntimeError(f"{variant}: fixed input or candidate graph changed: {key}")
    variant_replays[variant] = cache_replay
    destination = WORKING_ROOT / "variant_graphs" / variant
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(WORKING_ROOT / "oracle_final_graphs", destination)
    variant_graph_roots[variant] = destination
    shutil.copy2(cache_replay_path, WORKING_ROOT / f"{variant}_cache_graph_replay_summary.json")
    print(f"{variant}: all fixed candidates reused and 199 repaired graphs saved")

# %% [markdown]
# ## Convert compact final graphs and run the official evaluator
#
# exp015 stores repaired final graphs as compact NPZ files. Coordinates are
# rounded and clamped exactly as in its row writer, then converted to temporary
# GEFF graphs. Both variants are evaluated twice from those saved graphs; the
# summaries must be identical across recomputation.


# %%
def evaluate_compact_variant(
    label: str, compact_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        "username": "exp031",
        "method": label,
        "split": "split_0",
        "dir": temporary_root,
        "geffs": sorted(temporary_root.glob("*.geff")),
    }
    rows = public_evaluator.evaluate_run(public_run, max_distance=7.0)
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


control_compact_root = baseline_root / "oracle_final_graphs"
control_summary, control_rows = evaluate_compact_variant(
    "fixed_public_control", control_compact_root
)
variant_summaries: dict[str, dict[str, Any]] = {}
variant_rows: dict[str, list[dict[str, Any]]] = {}
for variant in VARIANTS:
    variant_summaries[variant], variant_rows[variant] = evaluate_compact_variant(
        variant, variant_graph_roots[variant]
    )
metric_keys = ("score", "adj_edge_jaccard", "edge_jaccard", "division_jaccard", "node_recall")
legacy_summary = variant_summaries["legacy"]
deltas = {
    variant: {
        "overall": {
            key: float(summary["overall"][key]) - float(legacy_summary["overall"][key])
            for key in metric_keys
        },
        "by_embryo": {
            embryo: {
                key: float(summary["by_embryo"][embryo][key])
                - float(legacy_summary["by_embryo"][embryo][key])
                for key in metric_keys
            }
            for embryo in ("44b6", "6bba")
        },
    }
    for variant, summary in variant_summaries.items()
    if variant != "legacy"
}
official_evaluation = {
    "experiment": EXPERIMENT,
    "benchmark_role": BENCHMARK_ROLE,
    "execution_environment": EXECUTION_ENVIRONMENT,
    "metric": "official_adjusted_edge_jaccard_plus_0.1_division_jaccard",
    "max_matching_distance_um": 7.0,
    "conditional_validation": True,
    "public_checkpoint_reference": control_summary,
    "variants": variant_summaries,
    "delta_vs_same_run_legacy": deltas,
    "fixed_candidate_coordinates_exact": True,
    "public_control_candidate_graphs_exact": True,
    "post_retention_candidates_reused_from_exp015": True,
    "main_image_encoder_forward_count": 0,
    "fold_model_by_evaluation_embryo": {"6bba": 0, "44b6": 1},
    "submission_created": False,
}
OFFICIAL_SUMMARY_PATH.write_text(
    json.dumps(json_safe(official_evaluation), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
PER_SAMPLE_PATH.write_text(
    json.dumps(
        json_safe({"public_checkpoint_reference": control_rows, "variants": variant_rows}),
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

# %% [markdown]
# ## Evidence receipt and output pruning
#
# The exp015 feature cache is read directly and never copied. The retrained
# candidate graphs, compact repaired graphs, official metric rows, fold
# assignment, and integrity receipts remain as persisted evidence.

# %%
elapsed_seconds = time.monotonic() - started
variant_manifest_paths = {
    variant: WORKING_ROOT / f"{variant}_fold_tracker_inference_manifest.json"
    for variant in VARIANTS
}
receipt = {
    "experiment": EXPERIMENT,
    "benchmark_role": BENCHMARK_ROLE,
    "execution_environment": EXECUTION_ENVIRONMENT,
    "stage": "official_graph_evaluation",
    "elapsed_seconds": elapsed_seconds,
    "prediction_seconds_by_variant": prediction_seconds_by_variant,
    "visible_gpu_count": torch.cuda.device_count(),
    "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    "variant_model_manifest_sha256": {
        variant: gi.sha256_file(path) for variant, path in variant_manifest_paths.items()
    },
    "control_compact_graph_content_sha256": control_summary["compact_graph_content_sha256"],
    "variant_graph_content_sha256": {
        variant: summary["compact_graph_content_sha256"]
        for variant, summary in variant_summaries.items()
    },
    "variant_replay_summary_sha256": {
        variant: gi.sha256_file(WORKING_ROOT / f"{variant}_cache_graph_replay_summary.json")
        for variant in VARIANTS
    },
    "cache_identity_sha256": variant_replays["legacy"]["cache_identity_sha256"],
    "cache_window_count": variant_replays["legacy"]["window_count"],
    "public_control_candidate_graph_content_sha256": variant_replays["legacy"][
        "public_control_candidate_graph_content_sha256"
    ],
    "candidate_coordinate_content_sha256": variant_replays["legacy"][
        "candidate_coordinate_content_sha256"
    ],
    "official_evaluation_sha256": gi.sha256_file(OFFICIAL_SUMMARY_PATH),
    "official_evaluator_preflight_sha256": gi.sha256_file(OFFICIAL_PREFLIGHT_PATH),
    "per_sample_metrics_sha256": gi.sha256_file(PER_SAMPLE_PATH),
    "fixed_candidate_coordinates_exact": True,
    "public_control_candidate_graphs_exact": True,
    "post_retention_candidates_reused_from_exp015": True,
    "main_image_encoder_forward_count": 0,
    "model_count_loaded": 6,
    "additional_training": False,
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
        "status": "running",
        "cv": None,
        "metric": "official_adjusted_edge_jaccard_plus_0.1_division_jaccard",
        "updated_at": datetime.now(UTC).isoformat(),
        "notes": (
            "Official graph evaluation completed conditionally on the public image encoder; "
            "user completion and adoption judgment remain pending. No submission was created."
        ),
        "official_graph_evaluation": official_evaluation,
        "evidence": {
            "artifacts": {
                "selected_mode": "three_variant_fold_tracker_cached_graph_replay",
                "selected_model": "six_held_out_embryo_primary_trackers",
                "oof_prediction_sha": variant_summaries["self_cross"][
                    "compact_graph_content_sha256"
                ],
                "variant_graph_shas": receipt["variant_graph_content_sha256"],
                "test_prediction_content_sha": None,
                "submission_sha": None,
            },
            "inference_stage": receipt,
        },
    },
)
METRICS_PATH.write_text(
    json.dumps(json_safe(metrics), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)

for temporary in (
    WORKING_ROOT / "tracking_repo",
    WORKING_ROOT / "hybrid_primary_models",
    WORKING_ROOT / "_official_graph_eval",
    WORKING_ROOT / "cache_replay_workers",
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
