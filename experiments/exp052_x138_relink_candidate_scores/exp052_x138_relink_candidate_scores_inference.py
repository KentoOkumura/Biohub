# %%
# ruff: noqa
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp052: saved candidate scores for motion relinking

# %% [markdown]
# ## Contents
# 1. Fixed x138 configuration and artifacts
# 2. Video split and input checks
# 3. Frozen tracker inference
# 4. Existing graph postprocessing
# 5. Paired relink replay
# 6. Stage diagnostics
# 7. Official evaluation
# 8. Runtime receipt

# %%
EXP052_PHASE = "inference"  # "diagnostic" or "inference"
EXP052_INTERNAL_GATE_RECEIPT_SHA = ""  # filled only after diagnostic evidence passes

# %% [markdown]
# ## 1. Public x138 configuration and fixed artifacts

# %%
"""Biohub Harmonic Fusion

Production 3D lineage reconstruction with dual temporal models,
dual edge-feature TTA, and geometry-validated divisions.

Record edition."""

import os

BIOHUB_PRESET = "harmonic_v3_division_wide"
BIOHUB_SCORE_AXIS = "public 0.939 base + holdout-selected post-process configuration"

os.environ["BIOHUB_OUTPUT_FILTER_SHORT_TRACKS"] = "1"
os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"
os.environ["BIOHUB_MOTION_RELINK_LEARNED_BONUS"] = "1.0"
os.environ["BIOHUB_ILP_APPEARANCE_WEIGHT"] = "0.0"
os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "2"
os.environ["BIOHUB_GAP_CLOSE_MAX_GAP"] = "2"
os.environ["BIOHUB_GAP_CLOSE_UM"] = "5.0"
os.environ["BIOHUB_GAP_DENSITY_ADAPTIVE"] = "1"
os.environ["BIOHUB_GAP_DENSITY_REFERENCE_UM"] = "6.5"
os.environ["BIOHUB_GAP_DENSITY_GAIN"] = "0.040"
os.environ["BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM"] = "0.125"
os.environ["BIOHUB_GAP_DENSITY_NEIGHBORS"] = "3"
os.environ["BIOHUB_OUTPUT_MIN_TRACK_LEN"] = "6"
os.environ["BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS"] = "1"
os.environ["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "1"
os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "9.0"


os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "14.0"


os.environ["BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU"] = "0.6"
os.environ["BIOHUB_SAFE_DIV_DIVERGE_UM"] = "2.25"


os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "10.0"
os.environ["BIOHUB_SAFE_DIV_FRAME_FRAC_CAP"] = "0.0076"
os.environ["BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP"] = "0.00375"

os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "1.2"
os.environ["BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE"] = "1"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN"] = "4"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB"] = "0.88"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM"] = "3.0"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC"] = "0.012"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS"] = "120"
os.environ["BIOHUB_USE_DEEPCENTER_VETO"] = "1"
os.environ["BIOHUB_REQUIRE_DEEPCENTER_VETO"] = "1"
os.environ["BIOHUB_DEEPCENTER_EXPECTED_EPOCH"] = "2"
os.environ["BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM"] = "8.5"
os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = (
    "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt"
)
os.environ["BIOHUB_DEEPCENTER_GAP_VETO"] = "1"
os.environ["BIOHUB_DEEPCENTER_GAP_THRESHOLD"] = "0.25"
os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "1"
os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"
os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.15"
os.environ["BIOHUB_BIDIRECTIONAL_FUSION_MODE"] = "harmonic_probability"
os.environ["BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION"] = "0.90"
os.environ["BIOHUB_DIAGNOSTIC_ARM"] = "harmonic_association_production"
os.environ["BIOHUB_VALIDATOR_N_PER_TYPE"] = "4"
os.environ["BIOHUB_PPSWEEP_SELECT_MARGIN"] = "0.001"
os.environ["BIOHUB_PPSWEEP_MAX_ADJ_LOSS"] = "0.0005"

os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.25"
os.environ["BIOHUB_DEEPCENTER_TTA"] = "1"

# Runtime hardening. None of these change the output on the visible test.
import time as _t0_time

os.environ["BIOHUB_KERNEL_START_TS"] = str(_t0_time.time())
os.environ["BIOHUB_VALIDATOR_ENABLE"] = "0"  # in-sample train proxy, ~11 min of GPU per run
os.environ["BIOHUB_ILP_TIMEOUT_S"] = "1200"  # per dataset; SCIP keeps its incumbent at the limit
os.environ["BIOHUB_REPAIR_DEADLINE_S"] = (
    "27000"  # 7.5 h after kernel start the repair loop degrades
)
os.environ["BIOHUB_FRAME_CACHE_MAX_FRAMES"] = "48"
os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = (
    "5.5"  # ppsweep_selected.json of the public run (tight55)
)
# Neighbourhood-flow motion prior for the motion re-link (agent/frontier943_flow).
os.environ["BIOHUB_MOTION_RELINK_FLOW_MODE"] = "seed"
os.environ["BIOHUB_MOTION_RELINK_FLOW_K"] = "12"
os.environ["BIOHUB_MOTION_RELINK_FLOW_RADIUS_UM"] = "40.0"
os.environ["BIOHUB_MOTION_RELINK_FLOW_EXCLUDE_UM"] = "1.5"
os.environ["BIOHUB_MOTION_RELINK_FLOW_MIN_SAMPLES"] = "4"
os.environ["BIOHUB_MOTION_RELINK_FLOW_GATE"] = "1"
# Association geometry knobs (agent/frontier943_flow2).
os.environ["BIOHUB_MOTION_RELINK_FLOW_ITER"] = "1"
os.environ["BIOHUB_MOTION_RELINK_FLOW_SEED_GATE_UM"] = "0"
os.environ["BIOHUB_MOTION_RELINK_FLOW_RAW_ADMIT"] = "1"
os.environ["BIOHUB_MOTION_RELINK_FLOW_Z_WEIGHT"] = "1.0"
os.environ["BIOHUB_MOTION_RELINK_FLOW_RAW_COST"] = "0"
os.environ["BIOHUB_MOTION_RELINK_FLOW_TIGHT_UM"] = "7.0"
os.environ["BIOHUB_MOTION_RELINK_FLOW_RELAXED_UM"] = "0"
# Discarded detections re-admitted before the re-link (agent/frontier947_readmit).
os.environ["BIOHUB_READMIT_RADIUS_UM"] = "4"
os.environ["BIOHUB_READMIT_MIN_SCORE"] = "0.965"
# Gap filler on the detector's sub-threshold peaks (agent/frontier947_gapfill).
os.environ["BIOHUB_GAPFILL_MAX_GAP"] = "3"
os.environ["BIOHUB_GAPFILL_MIN_SCORE"] = "0.5"
os.environ["BIOHUB_GAPFILL_STEP_UM"] = "5.0"
os.environ["BIOHUB_GAPFILL_PEAK_RADIUS_UM"] = "3.5"
os.environ["BIOHUB_GAPFILL_EXCLUDE_UM"] = "2.0"
os.environ["BIOHUB_GAPFILL_ALLOW_SYNTHETIC"] = "0"
os.environ["BIOHUB_GAPFILL_CONTEXT"] = "1"
os.environ["BIOHUB_GAPFILL_MAX_ADDED_FRAC"] = "0.03"
os.environ["BIOHUB_CACHE_DIR"] = "/kaggle/working/exp045_audit/initial_cache"
os.environ["BIOHUB_CACHE_EDGE_THRESHOLD"] = "1.0e-8"
os.environ["BIOHUB_LOWDET_THRESHOLD"] = "0.3"
print("BIOHUB_PRESET:", BIOHUB_PRESET)
print("BIOHUB_SCORE_AXIS:", BIOHUB_SCORE_AXIS)

# %%

import json as _guard_json
import math as _guard_math
import os as _guard_os

_EXPECTED_NUMERIC = {
    "BIOHUB_DET_THRESHOLD": 0.965,
    "BIOHUB_ILP_APPEARANCE_WEIGHT": 0.0,
    "BIOHUB_ILP_DISAPPEARANCE_WEIGHT": 2,
    "BIOHUB_GAP_CLOSE_UM": 5.0,
    "BIOHUB_OUTPUT_MIN_TRACK_LEN": 6.0,
    "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": 0.15,
}

_EXPECTED_TEXT = {
    "BIOHUB_BIDIRECTIONAL_FUSION_MODE": "harmonic_probability",
    "BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION": "0.90",
}

_drift = {}
for _key, _want in _EXPECTED_NUMERIC.items():
    _raw = _guard_os.environ.get(_key)
    if _raw is None:
        _drift[_key] = "missing"
        continue
    _got = float(_raw)
    if not _guard_math.isclose(_got, _want, rel_tol=0.0, abs_tol=1e-12):
        _drift[_key] = {"expected": _want, "actual": _got}

for _key, _want in _EXPECTED_TEXT.items():
    _got = _guard_os.environ.get(_key)
    if _got != _want:
        _drift[_key] = {"expected": _want, "actual": _got}

if _drift:
    raise RuntimeError("Configuration drift detected: " + _guard_json.dumps(_drift, sort_keys=True))

print("Configuration guard: PASS")
print("Baseline: fixed-90 dual-seed clean pipeline (public LB 0.913)")
print("Single model-level change: harmonic mutual-support association fusion")
print("Reverse-time association weight: 0.200")

# %%

import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import tempfile
import zipfile
import sys
import time
from pathlib import Path

import pandas as pd
from IPython.display import display

COMPETITION = "biohub-cell-tracking-during-development"
COMP_DIR_CANDIDATES = [
    Path(f"/kaggle/input/competitions/{COMPETITION}"),
    Path(f"/kaggle/input/{COMPETITION}"),
]
COMP_DIR = next((path for path in COMP_DIR_CANDIDATES if path.exists()), COMP_DIR_CANDIDATES[0])

TEST_DIR = COMP_DIR / "test"

WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
REPO_DIR = WORKING_DIR / "tracking_repo"
SUBMISSION_PATH = WORKING_DIR / "submission.csv"
RUN_STATS_PATH = WORKING_DIR / "run_stats.csv"

METHOD = "unet_transformer"
WEIGHTS_RELATIVE = f"weights/{METHOD}/split_0/edge_predictor_best.pth"
EXPERIMENT_TAG = "selected_101_dual_seed_near_balanced_center_confirmed_synthetic_gap"
TARGET_ARTIFACT_SLUG = os.environ.get(
    "BIOHUB_TARGET_ARTIFACT_SLUG", "biohub-tracking-support-pack-50ep-v1"
)
PRIMARY_ARTIFACT_MANIFEST = Path(
    os.environ.get(
        "BIOHUB_PRIMARY_ARTIFACT_MANIFEST",
        "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/ARTIFACT_MANIFEST.json",
    )
)
ALLOW_ARTIFACT_FALLBACK = os.environ.get("BIOHUB_ALLOW_ARTIFACT_FALLBACK", "0") != "0"

DET_THRESHOLD = float(os.environ.get("BIOHUB_DET_THRESHOLD", "0.99"))
UNET_BATCH_SIZE = int(os.environ.get("BIOHUB_UNET_BATCH_SIZE", "4"))
USE_ILP = os.environ.get("BIOHUB_USE_ILP", "1") != "0"
ILP_EDGE_WEIGHT = float(os.environ.get("BIOHUB_ILP_EDGE_WEIGHT", "-1.0"))
ILP_APPEARANCE_WEIGHT = float(os.environ.get("BIOHUB_ILP_APPEARANCE_WEIGHT", "0.1"))
ILP_DISAPPEARANCE_WEIGHT = float(os.environ.get("BIOHUB_ILP_DISAPPEARANCE_WEIGHT", "0.1"))
ILP_DIVISION_WEIGHT = float(os.environ.get("BIOHUB_ILP_DIVISION_WEIGHT", "1.0"))


SLICE = ""


ALLOW_PIP_INSTALL = os.environ.get("BIOHUB_ALLOW_PIP_INSTALL", "0") != "0"
RUN_OUTPUT_DIAGNOSTICS = os.environ.get("BIOHUB_RUN_OUTPUT_DIAGNOSTICS", "1") != "0"


OUTPUT_EDGE_MAX_UM = float(os.environ.get("BIOHUB_OUTPUT_EDGE_MAX_UM", "14.0"))
OUTPUT_ENFORCE_NEXT_FRAME = os.environ.get("BIOHUB_OUTPUT_ENFORCE_NEXT_FRAME", "1") != "0"
OUTPUT_SINGLE_PARENT_REPAIR = os.environ.get("BIOHUB_OUTPUT_SINGLE_PARENT_REPAIR", "1") != "0"
OUTPUT_SINGLE_CHILD_REPAIR = os.environ.get("BIOHUB_OUTPUT_SINGLE_CHILD_REPAIR", "0") != "0"
OUTPUT_PRUNE_ISOLATED = os.environ.get("BIOHUB_OUTPUT_PRUNE_ISOLATED", "1") != "0"
OUTPUT_MOTION_RELINK = os.environ.get("BIOHUB_OUTPUT_MOTION_RELINK", "1") != "0"
MOTION_RELINK_TIGHT_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_TIGHT_UM", "6.0"))
MOTION_RELINK_RELAXED_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_RELAXED_UM", "10.0"))
MOTION_RELINK_VELOCITY_WEIGHT = float(os.environ.get("BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT", "0.5"))
MOTION_RELINK_FLOW_MODE = os.environ.get("BIOHUB_MOTION_RELINK_FLOW_MODE", "off").strip().lower()
MOTION_RELINK_FLOW_K = int(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_K", "8"))
MOTION_RELINK_FLOW_RADIUS_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_RADIUS_UM", "25.0"))
MOTION_RELINK_FLOW_EXCLUDE_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_EXCLUDE_UM", "1.5"))
MOTION_RELINK_FLOW_MIN_SAMPLES = int(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_MIN_SAMPLES", "4"))
MOTION_RELINK_FLOW_GATE = os.environ.get("BIOHUB_MOTION_RELINK_FLOW_GATE", "0").strip() == "1"
MOTION_RELINK_FLOW_ITER = int(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_ITER", "1"))
MOTION_RELINK_FLOW_SEED_GATE_UM = float(
    os.environ.get("BIOHUB_MOTION_RELINK_FLOW_SEED_GATE_UM", "0")
)
MOTION_RELINK_FLOW_RAW_ADMIT = (
    os.environ.get("BIOHUB_MOTION_RELINK_FLOW_RAW_ADMIT", "1").strip() != "0"
)
MOTION_RELINK_FLOW_Z_WEIGHT = float(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_Z_WEIGHT", "1.0"))
MOTION_RELINK_FLOW_RAW_COST = float(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_RAW_COST", "0.05"))
MOTION_RELINK_FLOW_TIGHT_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_TIGHT_UM", "0"))
MOTION_RELINK_FLOW_RELAXED_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_RELAXED_UM", "0"))
READMIT_RADIUS_UM = float(os.environ.get("BIOHUB_READMIT_RADIUS_UM", "0"))
READMIT_MIN_SCORE = float(os.environ.get("BIOHUB_READMIT_MIN_SCORE", "0.965"))
GAPFILL_MAX_GAP = int(os.environ.get("BIOHUB_GAPFILL_MAX_GAP", "0"))
GAPFILL_MIN_SCORE = float(os.environ.get("BIOHUB_GAPFILL_MIN_SCORE", "0.5"))
GAPFILL_STEP_UM = float(os.environ.get("BIOHUB_GAPFILL_STEP_UM", "5.0"))
GAPFILL_PEAK_RADIUS_UM = float(os.environ.get("BIOHUB_GAPFILL_PEAK_RADIUS_UM", "3.5"))
GAPFILL_EXCLUDE_UM = float(os.environ.get("BIOHUB_GAPFILL_EXCLUDE_UM", "2.0"))
GAPFILL_ALLOW_SYNTHETIC = int(os.environ.get("BIOHUB_GAPFILL_ALLOW_SYNTHETIC", "0"))
GAPFILL_CONTEXT = os.environ.get("BIOHUB_GAPFILL_CONTEXT", "1") != "0"
GAPFILL_MAX_ADDED_FRAC = float(os.environ.get("BIOHUB_GAPFILL_MAX_ADDED_FRAC", "0.03"))
MOTION_RELINK_LEARNED_BONUS = float(os.environ.get("BIOHUB_MOTION_RELINK_LEARNED_BONUS", "0.75"))
MOTION_RELINK_MAX_FRAME_NODES = int(os.environ.get("BIOHUB_MOTION_RELINK_MAX_FRAME_NODES", "2600"))

OUTPUT_DIVISION_GEOMETRY_FILTER = (
    os.environ.get("BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER", "0") != "0"
)
DIV_PARENT_MAX_UM = float(os.environ.get("BIOHUB_DIV_PARENT_MAX_UM", "10.5"))
DIV_SISTER_MAX_UM = float(os.environ.get("BIOHUB_DIV_SISTER_MAX_UM", "8.0"))
DIV_DROP_TO_SINGLE_IF_BAD = os.environ.get("BIOHUB_DIV_DROP_TO_SINGLE_IF_BAD", "1") != "0"
OUTPUT_GAP_CLOSE = os.environ.get("BIOHUB_OUTPUT_GAP_CLOSE", "1") != "0"
GAP_CLOSE_MAX_GAP = int(os.environ.get("BIOHUB_GAP_CLOSE_MAX_GAP", "1"))
GAP_CLOSE_UM = float(os.environ.get("BIOHUB_GAP_CLOSE_UM", "6.0"))
GAP_DENSITY_ADAPTIVE = os.environ.get("BIOHUB_GAP_DENSITY_ADAPTIVE", "0") != "0"
GAP_DENSITY_REFERENCE_UM = float(os.environ.get("BIOHUB_GAP_DENSITY_REFERENCE_UM", "6.5"))
GAP_DENSITY_GAIN = float(os.environ.get("BIOHUB_GAP_DENSITY_GAIN", "0.040"))
GAP_DENSITY_MAX_STEP_DELTA_UM = float(
    os.environ.get("BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM", "0.125")
)
GAP_DENSITY_NEIGHBORS = int(os.environ.get("BIOHUB_GAP_DENSITY_NEIGHBORS", "3"))
GAP_CLOSE_REUSE_EXISTING = os.environ.get("BIOHUB_GAP_CLOSE_REUSE_EXISTING", "1") != "0"
GAP_CLOSE_REUSE_UM = float(os.environ.get("BIOHUB_GAP_CLOSE_REUSE_UM", "3.2"))
GAP_CLOSE_MAX_ADDED_FRAC = float(os.environ.get("BIOHUB_GAP_CLOSE_MAX_ADDED_FRAC", "0.05"))
GAP_CLOSE_MAX_ADDED_ABS = int(os.environ.get("BIOHUB_GAP_CLOSE_MAX_ADDED_ABS", "2000"))
GAP_REFINE_SYNTHETIC = os.environ.get("BIOHUB_GAP_REFINE_SYNTHETIC", "1") != "0"
GAP_REFINE_WIN_Z = int(os.environ.get("BIOHUB_GAP_REFINE_WIN_Z", "1"))
GAP_REFINE_WIN_YX = int(os.environ.get("BIOHUB_GAP_REFINE_WIN_YX", "3"))
GAP_REFINE_MAX_SHIFT_UM = float(os.environ.get("BIOHUB_GAP_REFINE_MAX_SHIFT_UM", "3.2"))

OUTPUT_FILTER_SHORT_TRACKS = os.environ.get("BIOHUB_OUTPUT_FILTER_SHORT_TRACKS", "1") != "0"
OUTPUT_MIN_TRACK_LEN = int(os.environ.get("BIOHUB_OUTPUT_MIN_TRACK_LEN", "6"))
OUTPUT_KEEP_DIVISION_COMPONENTS = (
    os.environ.get("BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS", "1") != "0"
)
ADAPTIVE_SHORT_TRACK_RESCUE = os.environ.get("BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE", "0") != "0"
SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC = float(
    os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC", "0.10")
)
SHORT_TRACK_RESCUE_MIN_LEN = int(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN", "4"))
SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB = float(
    os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB", "0.82")
)
SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM = float(
    os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM", "3.25")
)
SHORT_TRACK_RESCUE_MAX_NODES_FRAC = float(
    os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC", "0.018")
)
SHORT_TRACK_RESCUE_MAX_NODES_ABS = int(
    os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS", "180")
)

OUTPUT_LINEFIT_SMOOTH = os.environ.get("BIOHUB_OUTPUT_LINEFIT_SMOOTH", "1") != "0"
OUTPUT_LINEFIT_WEIGHT = float(os.environ.get("BIOHUB_OUTPUT_LINEFIT_WEIGHT", "0.8"))
OUTPUT_LINEFIT_WINDOW = int(os.environ.get("BIOHUB_OUTPUT_LINEFIT_WINDOW", "2"))

OUTPUT_GAP2_RECOVERY = os.environ.get("BIOHUB_OUTPUT_GAP2_RECOVERY", "0") != "0"
GAP2_MAX_TOTAL_UM = float(os.environ.get("BIOHUB_GAP2_MAX_TOTAL_UM", "10.2"))
GAP2_MAX_STEP_UM = float(os.environ.get("BIOHUB_GAP2_MAX_STEP_UM", "4.4"))
GAP2_MAX_LINKS_FRAC = float(os.environ.get("BIOHUB_GAP2_MAX_LINKS_FRAC", "0.0045"))
GAP2_MAX_LINKS_ABS = int(os.environ.get("BIOHUB_GAP2_MAX_LINKS_ABS", "180"))
GAP2_REQUIRE_CONTEXT = os.environ.get("BIOHUB_GAP2_REQUIRE_CONTEXT", "1") != "0"
GAP2_FRAME_FRAC_CAP = float(os.environ.get("BIOHUB_GAP2_FRAME_FRAC_CAP", "0.006"))

OUTPUT_SAFE_DIVISIONS = os.environ.get("BIOHUB_OUTPUT_SAFE_DIVISIONS", "1") != "0"
SAFE_DIV_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_MAX_UM", "4.7"))
SAFE_DIV_SISTER_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_SISTER_MAX_UM", "7.2"))
SAFE_DIV_SISTER_SYMMETRY_TAU = float(os.environ.get("BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU", "0.0"))
SAFE_DIV_EXISTING_CHILD_MAX_UM = float(
    os.environ.get("BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM", "7.8")
)
SAFE_DIV_FRAME_FRAC_CAP = float(os.environ.get("BIOHUB_SAFE_DIV_FRAME_FRAC_CAP", "0.008"))
SAFE_DIV_GLOBAL_FRAC_CAP = float(os.environ.get("BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP", "0.004"))


SAFE_DIV_DIVERGE_UM = float(os.environ.get("BIOHUB_SAFE_DIV_DIVERGE_UM", "2.25"))
SAFE_DIV_REQUIRE_DIVERGENCE = os.environ.get("BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE", "1") != "0"
SAFE_DIV_REQUIRE_MUTUAL_NN = os.environ.get("BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN", "1") != "0"


USE_DEEPCENTER_VETO = os.environ.get("BIOHUB_USE_DEEPCENTER_VETO", "1") != "0"
REQUIRE_DEEPCENTER_VETO = os.environ.get("BIOHUB_REQUIRE_DEEPCENTER_VETO", "1") != "0"
DEEPCENTER_MANIFEST_DEFAULT = os.environ.get(
    "BIOHUB_DEEPCENTER_MANIFEST_DEFAULT",
    "/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/ARTIFACT_MANIFEST.json",
)
DEEPCENTER_CHECKPOINT_DEFAULT = os.environ.get(
    "BIOHUB_DEEPCENTER_CHECKPOINT_DEFAULT",
    "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt",
)
DEEPCENTER_RELATIVE = os.environ.get(
    "BIOHUB_DEEPCENTER_RELATIVE", "weights/full_frame_center/best.pt"
)
DEEPCENTER_GAP_VETO = os.environ.get("BIOHUB_DEEPCENTER_GAP_VETO", "1") != "0"
DEEPCENTER_SAFE_DIV_VETO = os.environ.get("BIOHUB_DEEPCENTER_SAFE_DIV_VETO", "1") != "0"
DEEPCENTER_GAP_THRESHOLD = float(os.environ.get("BIOHUB_DEEPCENTER_GAP_THRESHOLD", "0.10"))
DEEPCENTER_EXPECTED_EPOCH = int(os.environ.get("BIOHUB_DEEPCENTER_EXPECTED_EPOCH", "0"))
DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM = float(
    os.environ.get("BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM", "0")
)
DEEPCENTER_SAFE_DIV_THRESHOLD = float(
    os.environ.get("BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD", "0.12")
)
DEEPCENTER_SCORE_WIN_Z = int(os.environ.get("BIOHUB_DEEPCENTER_SCORE_WIN_Z", "1"))
DEEPCENTER_SCORE_WIN_YX = int(os.environ.get("BIOHUB_DEEPCENTER_SCORE_WIN_YX", "2"))
DEEPCENTER_SCORE_CACHE_MAX_FRAMES = int(
    os.environ.get("BIOHUB_DEEPCENTER_SCORE_CACHE_MAX_FRAMES", "8")
)

CONFIG_DISPLAY = {
    "experiment_tag": EXPERIMENT_TAG,
    "method": METHOD,
    "weights": WEIGHTS_RELATIVE,
    "target_artifact_slug": TARGET_ARTIFACT_SLUG,
    "primary_artifact_manifest": str(PRIMARY_ARTIFACT_MANIFEST),
    "allow_artifact_fallback": ALLOW_ARTIFACT_FALLBACK,
    "det_threshold": DET_THRESHOLD,
    "unet_batch_size": UNET_BATCH_SIZE,
    "use_ilp": USE_ILP,
    "ilp_edge_weight": ILP_EDGE_WEIGHT,
    "ilp_appearance_weight": ILP_APPEARANCE_WEIGHT,
    "ilp_disappearance_weight": ILP_DISAPPEARANCE_WEIGHT,
    "ilp_division_weight": ILP_DIVISION_WEIGHT,
    "slice": SLICE,
    "allow_pip_install": ALLOW_PIP_INSTALL,
    "output_edge_max_um": OUTPUT_EDGE_MAX_UM,
    "output_enforce_next_frame": OUTPUT_ENFORCE_NEXT_FRAME,
    "output_single_parent_repair": OUTPUT_SINGLE_PARENT_REPAIR,
    "output_single_child_repair": OUTPUT_SINGLE_CHILD_REPAIR,
    "output_prune_isolated": OUTPUT_PRUNE_ISOLATED,
    "output_motion_relink": OUTPUT_MOTION_RELINK,
    "motion_relink_tight_um": MOTION_RELINK_TIGHT_UM,
    "motion_relink_relaxed_um": MOTION_RELINK_RELAXED_UM,
    "motion_relink_velocity_weight": MOTION_RELINK_VELOCITY_WEIGHT,
    "motion_relink_learned_bonus": MOTION_RELINK_LEARNED_BONUS,
    "motion_relink_max_frame_nodes": MOTION_RELINK_MAX_FRAME_NODES,
    "output_division_geometry_filter": OUTPUT_DIVISION_GEOMETRY_FILTER,
    "div_parent_max_um": DIV_PARENT_MAX_UM,
    "div_sister_max_um": DIV_SISTER_MAX_UM,
    "div_drop_to_single_if_bad": DIV_DROP_TO_SINGLE_IF_BAD,
    "output_gap_close": OUTPUT_GAP_CLOSE,
    "gap_close_max_gap": GAP_CLOSE_MAX_GAP,
    "gap_close_effective_max_gap": min(GAP_CLOSE_MAX_GAP, 1),
    "gap_close_um": GAP_CLOSE_UM,
    "gap_density_adaptive": GAP_DENSITY_ADAPTIVE,
    "gap_density_reference_um": GAP_DENSITY_REFERENCE_UM,
    "gap_density_gain": GAP_DENSITY_GAIN,
    "gap_density_max_step_delta_um": GAP_DENSITY_MAX_STEP_DELTA_UM,
    "gap_density_neighbors": GAP_DENSITY_NEIGHBORS,
    "gap_close_reuse_existing": GAP_CLOSE_REUSE_EXISTING,
    "gap_close_reuse_um": GAP_CLOSE_REUSE_UM,
    "gap_close_max_added_frac": GAP_CLOSE_MAX_ADDED_FRAC,
    "gap_close_max_added_abs": GAP_CLOSE_MAX_ADDED_ABS,
    "gap_refine_synthetic": GAP_REFINE_SYNTHETIC,
    "gap_refine_win_z": GAP_REFINE_WIN_Z,
    "gap_refine_win_yx": GAP_REFINE_WIN_YX,
    "gap_refine_max_shift_um": GAP_REFINE_MAX_SHIFT_UM,
    "output_filter_short_tracks": OUTPUT_FILTER_SHORT_TRACKS,
    "output_min_track_len": OUTPUT_MIN_TRACK_LEN,
    "output_keep_division_components": OUTPUT_KEEP_DIVISION_COMPONENTS,
    "adaptive_short_track_rescue": ADAPTIVE_SHORT_TRACK_RESCUE,
    "short_track_rescue_trigger_removed_frac": SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC,
    "short_track_rescue_min_len": SHORT_TRACK_RESCUE_MIN_LEN,
    "short_track_rescue_min_mean_edge_prob": SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB,
    "short_track_rescue_max_mean_edge_dist_um": SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM,
    "short_track_rescue_max_nodes_frac": SHORT_TRACK_RESCUE_MAX_NODES_FRAC,
    "short_track_rescue_max_nodes_abs": SHORT_TRACK_RESCUE_MAX_NODES_ABS,
    "output_linefit_smooth": OUTPUT_LINEFIT_SMOOTH,
    "output_linefit_weight": OUTPUT_LINEFIT_WEIGHT,
    "output_linefit_window": OUTPUT_LINEFIT_WINDOW,
    "output_gap2_recovery": OUTPUT_GAP2_RECOVERY,
    "gap2_max_total_um": GAP2_MAX_TOTAL_UM,
    "gap2_max_step_um": GAP2_MAX_STEP_UM,
    "gap2_max_links_frac": GAP2_MAX_LINKS_FRAC,
    "gap2_max_links_abs": GAP2_MAX_LINKS_ABS,
    "gap2_require_context": GAP2_REQUIRE_CONTEXT,
    "gap2_frame_frac_cap": GAP2_FRAME_FRAC_CAP,
    "output_safe_divisions": OUTPUT_SAFE_DIVISIONS,
    "safe_div_max_um": SAFE_DIV_MAX_UM,
    "safe_div_sister_max_um": SAFE_DIV_SISTER_MAX_UM,
    "safe_div_existing_child_max_um": SAFE_DIV_EXISTING_CHILD_MAX_UM,
    "safe_div_frame_frac_cap": SAFE_DIV_FRAME_FRAC_CAP,
    "safe_div_global_frac_cap": SAFE_DIV_GLOBAL_FRAC_CAP,
    "use_deepcenter_add_only_gate": USE_DEEPCENTER_VETO,
    "deepcenter_gap_add_gate": DEEPCENTER_GAP_VETO,
    "deepcenter_safe_div_add_gate": DEEPCENTER_SAFE_DIV_VETO,
    "deepcenter_gap_threshold": DEEPCENTER_GAP_THRESHOLD,
    "deepcenter_expected_epoch": DEEPCENTER_EXPECTED_EPOCH,
    "deepcenter_gap_confirm_min_span_um": DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM,
    "deepcenter_safe_div_threshold": DEEPCENTER_SAFE_DIV_THRESHOLD,
    "deepcenter_checkpoint_default": DEEPCENTER_CHECKPOINT_DEFAULT,
}

print("Biohub learned UNet + node-transformer + ILP submission")
print("COMP_DIR:", COMP_DIR, "exists:", COMP_DIR.exists())
print("TEST_DIR:", TEST_DIR, "exists:", TEST_DIR.exists())
print(json.dumps(CONFIG_DISPLAY, indent=2, sort_keys=True))

# %%
import re

os.environ.setdefault("POLARS_PREFER_PKG", "32")

PACKAGE_SPECS = {
    "tracksdata": ("tracksdata", "tracksdata"),
    "zarr": ("zarr", "zarr>=3.0.10,<4"),
    "pyscipopt": ("pyscipopt", "pyscipopt"),
    "geff": ("geff", "geff>=1.1.3.1.1"),
    "geff_spec": ("geff_spec", "geff-spec<1.2"),
    "ilpy": ("ilpy", "ilpy>=0.5.1"),
    "polars": ("polars", "polars>=1.36"),
    "blosc2": ("blosc2", "blosc2"),
    "dask": ("dask", "dask"),
    "imagecodecs": ("imagecodecs", "imagecodecs"),
    "skimage": ("skimage", "scikit-image>=0.24"),
    "pyarrow": ("pyarrow", "pyarrow"),
    "rustworkx": ("rustworkx", "rustworkx>=0.17.1"),
    "sqlalchemy": ("sqlalchemy", "sqlalchemy>=2"),
    "numcodecs": ("numcodecs", "numcodecs>=0.13,<0.16"),
    "donfig": ("donfig", "donfig>=0.8"),
    "google_crc32c": ("google_crc32c", "google-crc32c>=1.5"),
    "bidict": ("bidict", "bidict>=0.23.1"),
    "psygnal": ("psygnal", "psygnal>=0.14"),
    "rich": ("rich", "rich"),
    "networkx": ("networkx", "networkx>=3.2.1"),
    "pydantic": ("pydantic", "pydantic>=2.11"),
    "pydantic_core": ("pydantic_core", "pydantic-core"),
    "annotated_types": ("annotated_types", "annotated-types"),
    "typing_extensions": ("typing_extensions", "typing-extensions>=4.13"),
    "typing_inspection": ("typing_inspection", "typing-inspection"),
    "markdown_it": ("markdown_it", "markdown-it-py"),
    "pygments": ("pygments", "pygments"),
    "click": ("click", "click"),
    "cloudpickle": ("cloudpickle", "cloudpickle"),
    "fsspec": ("fsspec", "fsspec"),
    "partd": ("partd", "partd"),
    "locket": ("locket", "locket"),
    "toolz": ("toolz", "toolz"),
    "yaml": ("yaml", "pyyaml"),
    "ndindex": ("ndindex", "ndindex"),
    "msgpack": ("msgpack", "msgpack"),
    "numexpr": ("numexpr", "numexpr"),
    "deprecated": ("deprecated", "deprecated"),
    "wrapt": ("wrapt", "wrapt"),
    "imageio": ("imageio", "imageio"),
    "PIL": ("PIL", "pillow"),
    "tifffile": ("tifffile", "tifffile"),
    "lazy_loader": ("lazy_loader", "lazy-loader"),
    "tqdm": ("tqdm", "tqdm"),
}
EXTRA_SPECS_BY_NAME = {
    "tracksdata": ["bidict>=0.23.1", "psygnal>=0.14", "rich"],
    "zarr": ["donfig>=0.8", "google-crc32c>=1.5", "numcodecs>=0.13,<0.16"],
    "geff": ["geff-spec<1.2", "networkx>=3.2.1", "pydantic>=2.11", "numcodecs>=0.13,<0.16"],
    "geff_spec": ["pydantic>=2.11", "annotated-types", "pydantic-core", "typing-inspection"],
    "polars": ["polars-runtime-32"],
    "dask": ["click", "cloudpickle", "fsspec", "partd", "pyyaml", "toolz"],
    "partd": ["locket"],
    "blosc2": ["ndindex", "msgpack", "numexpr"],
    "numcodecs": ["deprecated", "msgpack", "wrapt"],
    "rich": ["markdown-it-py", "pygments"],
    "pydantic": [
        "annotated-types",
        "pydantic-core",
        "typing-extensions>=4.13",
        "typing-inspection",
    ],
    "skimage": ["imageio", "pillow", "tifffile", "lazy-loader", "networkx"],
}
PIP_DEPENDENCIES = [spec for _, spec in PACKAGE_SPECS.values()]
REQUIRED_MODULES = {name: module for name, (module, _) in PACKAGE_SPECS.items() if module}
FALLBACK_ARTIFACT_SLUGS = ["biohub-tracking-support-pack-v1"]


ALLOW_PIP_INSTALL = os.environ.get("BIOHUB_ALLOW_PIP_INSTALL", "0") != "0"


def module_missing(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is None


def has_model_artifact(path: Path) -> bool:
    has_repo_dir = (path / "repo").exists()
    has_weights_dir = (path / "weights" / METHOD / "split_0" / "edge_predictor_best.pth").exists()
    has_repo_zip = (path / "repo.zip").exists()
    has_weights_zip = (path / "weights.zip").exists()
    return (has_repo_dir and has_weights_dir) or (has_repo_zip and has_weights_zip)


def artifact_manifest(path: Path) -> dict:
    manifest = path / "ARTIFACT_MANIFEST.json"
    if not manifest.exists():
        return {}
    try:
        return json.loads(manifest.read_text())
    except Exception:
        return {}


def artifact_matches_target(path: Path) -> bool:
    if ALLOW_ARTIFACT_FALLBACK:
        return True
    manifest = artifact_manifest(path)
    artifact_name = str(manifest.get("artifact_name", ""))
    path_text = str(path)
    return TARGET_ARTIFACT_SLUG in {artifact_name, path.name} or TARGET_ARTIFACT_SLUG in path_text


def candidate_roots_for_slug(slug: str) -> list[Path]:
    return [
        Path(f"/kaggle/input/datasets/pilkwang/{slug}"),
        Path(f"/kaggle/input/{slug}"),
        Path(f"/kaggle/input/{slug}/{slug}"),
        Path(f"PublicNotebook/{slug}"),
    ]


def find_artifacts_root() -> Path:
    candidates: list[Path] = []
    for env_name in ["BIOHUB_MODEL_ARTIFACTS", "BIOHUB_ARTIFACTS"]:
        explicit = os.environ.get(env_name, "").strip()
        if explicit:
            candidates.append(Path(explicit))

    candidates.append(PRIMARY_ARTIFACT_MANIFEST.parent)
    candidates.extend(candidate_roots_for_slug(TARGET_ARTIFACT_SLUG))

    if ALLOW_ARTIFACT_FALLBACK:
        for slug in FALLBACK_ARTIFACT_SLUGS:
            candidates.extend(candidate_roots_for_slug(slug))

    input_root = Path("/kaggle/input")
    if input_root.exists():
        for child in input_root.iterdir():
            if not child.is_dir():
                continue
            child_text = str(child)
            if TARGET_ARTIFACT_SLUG in child_text or ALLOW_ARTIFACT_FALLBACK:
                candidates.append(child)
                candidates.append(child / child.name)
                for grandchild in child.iterdir():
                    if grandchild.is_dir():
                        candidates.append(grandchild)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if has_model_artifact(candidate) and artifact_matches_target(candidate):
            return candidate
    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(
        "Could not find the required model artifact. "
        f"Expected slug: {TARGET_ARTIFACT_SLUG}\n"
        "Attach the newly uploaded support dataset, or set BIOHUB_MODEL_ARTIFACTS.\n"
        "To debug with an older artifact, set BIOHUB_ALLOW_ARTIFACT_FALLBACK=1.\n"
        "Checked:\n" + checked
    )


def _has_package_file(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    patterns = ("*.whl", "*.tar.gz", "*.zip")
    return any(any(path.glob(pattern)) for pattern in patterns)


def find_offline_package_dirs(artifacts: Path) -> list[Path]:
    candidates: list[Path] = [
        artifacts / "wheels",
        artifacts,
        Path("/kaggle/working"),
        Path("/kaggle/working/wheels"),
    ]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for child in input_root.iterdir():
            if child.is_dir():
                candidates.extend([child / "wheels", child])
                for grandchild in child.iterdir():
                    if grandchild.is_dir():
                        candidates.extend([grandchild / "wheels", grandchild])

    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _has_package_file(candidate):
            out.append(candidate)
    return out


def purge_imported_modules(package_names: list[str]) -> None:
    roots = {"tracksdata"}
    for name in package_names:
        if name in PACKAGE_SPECS:
            module = PACKAGE_SPECS[name][0]
            roots.add(module.split(".")[0])
        if name == "polars":
            roots.add("polars")
    for root in roots:
        for module_name in list(sys.modules):
            if module_name == root or module_name.startswith(root + "."):
                sys.modules.pop(module_name, None)


def polars_runtime_ready() -> bool:
    try:
        import polars as _pl
        from polars._plr import PySeries as _PySeries

        _ = _PySeries
        return (
            hasattr(_pl, "Float16")
            and _pl.Series([-999999.0], dtype=_pl.Float64).dtype == _pl.Float64
        )
    except Exception:
        return False


def packages_requiring_refresh() -> list[str]:
    refresh: list[str] = []
    if not module_missing("polars") and not polars_runtime_ready():
        refresh.append("polars")

    if not module_missing("zarr"):
        try:
            import zarr as _zarr

            version_text = str(getattr(_zarr, "__version__", "0"))
            major = int(version_text.split(".", 1)[0])
            if major < 3:
                refresh.append("zarr")
        except Exception:
            refresh.append("zarr")
    return refresh


def dependency_specs_for(missing: list[str]) -> list[str]:
    specs: list[str] = []
    seen: set[str] = set()

    def add(spec: str) -> None:
        key = spec.lower()
        if key not in seen:
            seen.add(key)
            specs.append(spec)

    for name in missing:
        if name in PACKAGE_SPECS:
            add(PACKAGE_SPECS[name][1])
        for spec in EXTRA_SPECS_BY_NAME.get(name, []):
            add(spec)
    return specs


def import_failures() -> dict[str, str]:
    failures: dict[str, str] = {}
    for name, module_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    return failures


def missing_names_from_failures(failures: dict[str, str]) -> list[str]:
    names: list[str] = []
    module_to_name = {module: name for name, module in REQUIRED_MODULES.items()}
    for message in failures.values():
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", message)
        if match:
            module = match.group(1).split(".")[0]
        else:
            match = re.search(r"module ['\"]([^'\"]+)['\"] has no attribute", message)
            if not match:
                continue
            module = match.group(1).split(".")[0]
        name = module_to_name.get(module)
        if name and name not in names:
            names.append(name)
    return names


def install_missing_dependencies(missing: list[str], artifacts: Path) -> None:
    specs = dependency_specs_for(missing)
    force_reinstall = bool({"polars", "zarr"} & set(missing))
    if not specs:
        return

    package_dirs = find_offline_package_dirs(artifacts)
    if package_dirs:
        offline_cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
        if force_reinstall:
            offline_cmd.append("--force-reinstall")
        for package_dir in package_dirs:
            offline_cmd.extend(["--find-links", str(package_dir)])
        offline_cmd.extend(specs)
        print("Installing missing packages from offline package dirs:", missing)
        print(
            "Dependency resolver is disabled with --no-deps to avoid replacing Kaggle numpy/scipy in a live kernel."
        )
        print("Offline package dirs:", [str(path) for path in package_dirs])
        result = subprocess.run(offline_cmd, text=True, capture_output=True)
        if result.returncode == 0:
            purge_imported_modules(missing)
            print("Offline dependency install succeeded.")
            return
        print("Offline dependency install failed. Last pip output:")
        print((result.stdout or "")[-2000:])
        print((result.stderr or "")[-2000:])

    if ALLOW_PIP_INSTALL:
        online_cmd = [sys.executable, "-m", "pip", "install", "--no-deps"]
        if force_reinstall:
            online_cmd.append("--force-reinstall")
        online_cmd.extend(specs)
        print("Installing missing packages from PyPI:", missing)
        result = subprocess.run(online_cmd, text=True, capture_output=True)
        if result.returncode == 0:
            purge_imported_modules(missing)
            print("PyPI dependency install succeeded.")
            return
        print("PyPI dependency install failed. Last pip output:")
        print((result.stdout or "")[-2000:])
        print((result.stderr or "")[-2000:])

    command = "pip install tracksdata zarr>=3.0.10,<4 pyscipopt geff geff-spec ilpy polars blosc2 dask imagecodecs pyarrow rustworkx sqlalchemy donfig numcodecs"
    raise ImportError(
        "Missing required packages or dependency wheels: " + ", ".join(missing) + "\n"
        "Attach the support dataset with offline wheels. If supplying Kaggle dependency input instead, use:\n"
        + command
        + "\n"
        "Do not quote zarr>=3.0.10,<4 in Kaggle dependency input."
    )


def ensure_dependencies(artifacts: Path) -> None:
    for _ in range(5):
        refresh = packages_requiring_refresh()
        if refresh:
            install_missing_dependencies(refresh, artifacts)
            continue

        missing = [pkg for pkg, module in REQUIRED_MODULES.items() if module_missing(module)]
        if missing:
            install_missing_dependencies(missing, artifacts)
            continue

        failures = import_failures()
        if not failures:
            print("Required graph/Zarr/ILP packages import successfully.")
            return

        missing_from_import = missing_names_from_failures(failures)
        if missing_from_import:
            install_missing_dependencies(missing_from_import, artifacts)
            continue

        raise ImportError(
            "Required packages are present but failed to import. "
            "This may indicate a binary dependency mismatch in the live notebook kernel. "
            "Keep Kaggle dependency input empty and attach the wheels artifact.\n"
            + json.dumps(failures, indent=2)
        )

    failures = import_failures()
    raise ImportError(
        "Dependency recovery did not converge after repeated offline installs. "
        "The attached support artifact may be missing wheels.\n" + json.dumps(failures, indent=2)
    )


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def copy_or_extract_tree(src_dir: Path, src_zip: Path, dst: Path) -> None:
    remove_path(dst)
    if src_dir.exists() and src_dir.is_dir():
        shutil.copytree(src_dir, dst)
        return
    if src_zip.exists() and src_zip.is_file():
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src_zip) as zf:
            zf.extractall(dst)
        return
    raise FileNotFoundError(f"Missing source tree or zip: {src_dir} / {src_zip}")


def link_or_copy_tree(src: Path, dst: Path) -> None:
    remove_path(dst)
    try:
        os.symlink(src, dst, target_is_directory=True)
    except Exception:
        shutil.copytree(src, dst)


def materialize_inference_repo(artifacts: Path) -> None:
    copy_or_extract_tree(artifacts / "repo", artifacts / "repo.zip", REPO_DIR)

    weights_src = artifacts / "weights"
    weights_zip = artifacts / "weights.zip"
    weights_dst = REPO_DIR / "weights"
    if weights_src.exists() and weights_src.is_dir():
        link_or_copy_tree(weights_src, weights_dst)
    elif weights_zip.exists() and weights_zip.is_file():
        remove_path(weights_dst)
        weights_dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(weights_zip) as zf:
            zf.extractall(weights_dst)
    else:
        raise FileNotFoundError(f"Missing weights tree or zip under {artifacts}")

    required = [
        REPO_DIR / "scripts" / "predict_unet_transformer.py",
        REPO_DIR / WEIGHTS_RELATIVE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Materialized inference repo is incomplete:\n" + "\n".join(missing))
    print("Inference repo:", REPO_DIR)
    print("Weights:", REPO_DIR / WEIGHTS_RELATIVE)


ARTIFACTS = find_artifacts_root()
print("ARTIFACTS:", ARTIFACTS)
print("Has offline wheels:", (ARTIFACTS / "wheels").exists())
manifest_info = artifact_manifest(ARTIFACTS)
if manifest_info:
    print("Artifact name:", manifest_info.get("artifact_name"))
    print("Weight sha256:", manifest_info.get("model", {}).get("weight_sha256"))
    print("Weight path:", manifest_info.get("model", {}).get("weight_path"))
    _expected_primary_sha256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
    _actual_primary_sha256 = str(manifest_info.get("model", {}).get("weight_sha256", ""))
    if _actual_primary_sha256 != _expected_primary_sha256:
        raise RuntimeError(
            "Primary model checksum mismatch: "
            f"expected {_expected_primary_sha256}, got {_actual_primary_sha256 or 'missing'}"
        )

ensure_dependencies(ARTIFACTS)
materialize_inference_repo(ARTIFACTS)


import hashlib as _integrity_hashlib

_support_expected_sha256 = {
    "scripts/augmentations.py": "13db09817bf492f8d0f710a0a4d09776320b262060167055090a303fc6057f4e",
    "scripts/dataspec.py": "e69bf952fb985477ac50ff8598a35020c95d20a035a09b81ab4056e655dd311f",
    "scripts/evaluate.py": "614813cc51c3581c6ccda4bb20725a19da8ecac4a27620654bfca58319cffa3c",
    "scripts/predict_unet_transformer.py": "c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9",
    "scripts/train_unet_transformer.py": "c4f6317736bb3bb1ec8f3f6e9a6d935a463e3f0f1f685481b2d13218d35dc9ea",
    "src/biohub_tracking/__init__.py": "26a18d8da84e40da73281a48ebc3017d847a2e57431ab63e8629d2109e6e8571",
    "src/biohub_tracking/division_metrics.py": "d1cf1e0a43009d02174f1699ce2aa28458a2220ac4b521731d3bcf31cf8c76be",
    "src/biohub_tracking/img_proc.py": "00e8ef0adc8b39f1aaaa547ea6197b906bf9e8c009e339d3e95f8f8dbf31be3f",
    "src/biohub_tracking/io.py": "efae135b088cecaab463d889f16c885ef6da3ad27b0747327d8ddc28d866b7bd",
    "src/biohub_tracking/metrics.py": "31baf45b54c78f68bab4f65dd8f4b38bca702abb644171c6df7c46cdeef55d83",
    "src/biohub_tracking/models/__init__.py": "ab7587ef79856bae50d24b62e5805092d0459ee1c586522b763f9ef70c093e1d",
    "src/biohub_tracking/models/simple_node_transformer.py": "b97209edeb03840e80d903e3e2a8c81c520641c8ef343f6ca2904d0f80db064e",
    "src/biohub_tracking/models/temporal_unet.py": "d809c35d42f504161074ddeaaa7aee5b407e5bca7f9b4e1d5f9b2ff345666cac",
}
_support_expected_manifest_sha256 = (
    "978b626d1fd1e7397435a437dfe68691defe1572fc3c20e61012d7c9b52ed029"
)
_primary_expected_sha256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
_deepcenter_expected_sha256 = "8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0"


def _integrity_sha256_file(path: Path) -> str:
    digest = _integrity_hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_support_materialized_paths = {
    path.relative_to(REPO_DIR).as_posix(): path for path in REPO_DIR.rglob("*.py")
}
_support_actual_names = set(_support_materialized_paths)
_support_expected_names = set(_support_expected_sha256)
if _support_actual_names != _support_expected_names:
    raise RuntimeError(
        {
            "support_repo_python_files_missing": sorted(
                _support_expected_names - _support_actual_names
            ),
            "support_repo_python_files_extra": sorted(
                _support_actual_names - _support_expected_names
            ),
        }
    )
_support_actual_sha256 = {
    relative: _integrity_sha256_file(_support_materialized_paths[relative])
    for relative in sorted(_support_materialized_paths)
}
if _support_actual_sha256 != _support_expected_sha256:
    raise RuntimeError(
        {
            "support_repo_python_checksum_mismatch": {
                relative: {
                    "expected": _support_expected_sha256[relative],
                    "actual": _support_actual_sha256[relative],
                }
                for relative in sorted(_support_expected_sha256)
                if _support_actual_sha256[relative] != _support_expected_sha256[relative]
            }
        }
    )
_support_manifest_bytes = "".join(
    f"{_support_actual_sha256[relative]}  {relative}\n"
    for relative in sorted(_support_actual_sha256)
).encode("utf-8")
_support_actual_manifest_sha256 = _integrity_hashlib.sha256(_support_manifest_bytes).hexdigest()
if _support_actual_manifest_sha256 != _support_expected_manifest_sha256:
    raise RuntimeError(
        "Support repo manifest checksum mismatch: "
        f"expected {_support_expected_manifest_sha256}, "
        f"got {_support_actual_manifest_sha256}"
    )

_primary_materialized_path = REPO_DIR / WEIGHTS_RELATIVE
_primary_actual_sha256 = _integrity_sha256_file(_primary_materialized_path)
if _primary_actual_sha256 != _primary_expected_sha256:
    raise RuntimeError(
        "Materialized primary model checksum mismatch: "
        f"expected {_primary_expected_sha256}, got {_primary_actual_sha256}"
    )

_deepcenter_candidate_strings = [
    os.environ.get("BIOHUB_DEEPCENTER_CHECKPOINT", "").strip(),
    "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt",
    "/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/"
    "weights/full_frame_center/best.pt",
]
_deepcenter_candidates = []
for _candidate_string in _deepcenter_candidate_strings:
    if not _candidate_string:
        continue
    _candidate_path = Path(_candidate_string)
    if _candidate_path not in _deepcenter_candidates:
        _deepcenter_candidates.append(_candidate_path)
_deepcenter_materialized_path = next(
    (path for path in _deepcenter_candidates if path.is_file()),
    None,
)
if _deepcenter_materialized_path is None:
    raise FileNotFoundError(
        {"missing_deepcenter_checkpoint": [str(path) for path in _deepcenter_candidates]}
    )
_deepcenter_actual_sha256 = _integrity_sha256_file(_deepcenter_materialized_path)
if _deepcenter_actual_sha256 != _deepcenter_expected_sha256:
    raise RuntimeError(
        "DeepCenter checkpoint checksum mismatch: "
        f"expected {_deepcenter_expected_sha256}, "
        f"got {_deepcenter_actual_sha256}"
    )
os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(_deepcenter_materialized_path)

print("Support repo Python manifest SHA256:", _support_actual_manifest_sha256)
print("Primary materialized SHA256:", _primary_actual_sha256)
print("DeepCenter materialized SHA256:", _deepcenter_actual_sha256)


import hashlib as _hashlib

_secondary_manifest_explicit = Path(
    os.environ.get(
        "BIOHUB_SECONDARY_ARTIFACT_MANIFEST",
        "/kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1/ARTIFACT_MANIFEST.json",
    )
)
_secondary_expected_sha256 = "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
_secondary_slug = "biohub-temporal-unet3d-seed314159-v1"


import itertools as _itertools


def _find_secondary_artifact_root() -> tuple[Path, dict]:
    candidates = [
        _secondary_manifest_explicit,
        Path(f"/kaggle/input/{_secondary_slug}/ARTIFACT_MANIFEST.json"),
        Path(f"/kaggle/input/datasets/pilkwang/{_secondary_slug}/ARTIFACT_MANIFEST.json"),
    ]
    input_root = Path("/kaggle/input")

    def _walk_input_root():
        # Walking every file under /kaggle/input (the competition's train tree
        # included) cost ~210 s on the visible run. Only pay for it when the
        # explicit paths above miss.
        if input_root.exists():
            yield from input_root.rglob("ARTIFACT_MANIFEST.json")

    seen = set()
    for manifest_path in _itertools.chain(candidates, _walk_input_root()):
        manifest_path = manifest_path.expanduser()
        if manifest_path in seen or not manifest_path.is_file():
            continue
        seen.add(manifest_path)
        try:
            info = json.loads(manifest_path.read_text())
        except Exception:
            continue
        sha256 = str(info.get("model", {}).get("weight_sha256", ""))
        if sha256 == _secondary_expected_sha256:
            return manifest_path.parent, info
    raise FileNotFoundError(
        "Could not find the independent-seed artifact with weight SHA256 "
        + _secondary_expected_sha256
    )


SECONDARY_ARTIFACTS, secondary_manifest_info = _find_secondary_artifact_root()
SECONDARY_WEIGHTS_ROOT = WORKING_DIR / "secondary_seed_weights"
copy_or_extract_tree(
    SECONDARY_ARTIFACTS / "weights",
    SECONDARY_ARTIFACTS / "weights.zip",
    SECONDARY_WEIGHTS_ROOT,
)
SECONDARY_WEIGHTS_PATH = (
    SECONDARY_WEIGHTS_ROOT / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
)
SECONDARY_CONFIG_PATH = SECONDARY_WEIGHTS_PATH.parent / "config.json"
for _required_secondary_path in (SECONDARY_WEIGHTS_PATH, SECONDARY_CONFIG_PATH):
    if not _required_secondary_path.is_file():
        raise FileNotFoundError(f"Missing secondary model file: {_required_secondary_path}")


def _sha256_file(path: Path) -> str:
    digest = _hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_secondary_actual_sha256 = _sha256_file(SECONDARY_WEIGHTS_PATH)
if _secondary_actual_sha256 != _secondary_expected_sha256:
    raise RuntimeError(
        "Secondary model checksum mismatch: "
        f"expected {_secondary_expected_sha256}, got {_secondary_actual_sha256}"
    )

os.environ["BIOHUB_SECONDARY_WEIGHTS"] = str(SECONDARY_WEIGHTS_PATH)
os.environ["BIOHUB_SECONDARY_EDGE_WEIGHT"] = "0.15"
print("Secondary artifact:", SECONDARY_ARTIFACTS)
print("Secondary weight:", SECONDARY_WEIGHTS_PATH)
print("Secondary SHA256:", _secondary_actual_sha256)
print("Secondary edge-logit weight:", os.environ["BIOHUB_SECONDARY_EDGE_WEIGHT"])

os.environ["BIOHUB_SECONDARY_DETECTION_WEIGHT"] = "0.80"
os.environ["BIOHUB_SECONDARY_LINK_MODE"] = "low_margin_consensus"
os.environ["BIOHUB_SECONDARY_MIX_TEMPERATURE"] = "1"
os.environ["BIOHUB_SECONDARY_LOW_MARGIN_MAX"] = "0.35"
os.environ["BIOHUB_DUAL_SEED_EDGE_THRESHOLD"] = "0.48"

_runtime_integrity_receipt = {
    "status": "complete_label_free_runtime_integrity",
    "verified_before_dynamic_source_patch": True,
    "support_repo_python_file_count": len(_support_actual_sha256),
    "support_repo_python_sha256": _support_actual_sha256,
    "support_repo_python_manifest_sha256": _support_actual_manifest_sha256,
    "checkpoint_sha256": {
        "primary": _primary_actual_sha256,
        "secondary": _secondary_actual_sha256,
        "deepcenter": _deepcenter_actual_sha256,
    },
    "materialized_paths": {
        "primary": str(_primary_materialized_path),
        "secondary": str(SECONDARY_WEIGHTS_PATH),
        "deepcenter": str(_deepcenter_materialized_path),
    },
    "ground_truth_accessed": False,
}
_runtime_integrity_receipt_path = WORKING_DIR / "bidirectional_production_runtime_integrity.json"
_runtime_integrity_receipt_path.write_text(
    json.dumps(_runtime_integrity_receipt, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("Runtime integrity receipt:", _runtime_integrity_receipt_path)

# %% [markdown]
# ## Audit selection and fixed-ID diagnostic helpers

# %%
"""Pure selection and fixed-ID readouts for the exp045 coordinate audit."""

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

GRID_SPACING_UM = np.array([1.625, 1.625, 1.625], dtype=np.float64)
NATIVE_SPACING_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)


def video_rank(stem: str, seed: int = 42) -> tuple[str, str]:
    return hashlib.sha256(f"{seed}:{stem}".encode()).hexdigest(), stem


def select_videos(
    eligible: Iterable[str],
    trained: Iterable[str],
    *,
    embryos: tuple[str, str] = ("44b6", "6bba"),
    per_embryo: int = 10,
    seed: int = 42,
    rank_start: int = 11,
) -> dict[str, list[str]]:
    """Take a fixed ten-video rank block after the head-training videos."""
    eligible_list = list(eligible)
    trained_list = list(trained)
    eligible_set = set(eligible_list)
    trained_set = set(trained_list)
    if len(eligible_set) != len(eligible_list) or len(trained_set) != len(trained_list):
        raise ValueError("duplicate video stems")
    if not trained_set <= eligible_set:
        raise ValueError("head training video absent from eligible train images")
    selection: dict[str, list[str]] = {}
    for embryo in embryos:
        ranked = sorted(
            (stem for stem in eligible_set if stem.split("_", 1)[0] == embryo),
            key=lambda stem: video_rank(stem, seed),
        )
        rank_end = rank_start + per_embryo - 1
        if len(ranked) < rank_end:
            raise ValueError(f"{embryo}: fewer than {rank_end} eligible videos")
        if set(ranked[:per_embryo]) != trained_set.intersection(ranked):
            raise ValueError(f"{embryo}: head manifest differs from original hash selection")
        selection[embryo] = ranked[rank_start - 1 : rank_end]
    if trained_set.intersection(stem for group in selection.values() for stem in group):
        raise ValueError("selected a head training video")
    if len(trained_set) != len(embryos) * per_embryo:
        raise ValueError("unexpected head training video count")
    return selection


@dataclass(frozen=True)
class Match:
    gt_to_candidate: dict[int, int]
    candidate_to_gt: dict[int, int]
    matched_distance_um: dict[int, float]
    ambiguous_gt: int
    ambiguous_candidate: int


def match_known_centers(
    candidate_rows: np.ndarray,
    gt_rows: np.ndarray,
    *,
    radius_um: float = 7.0,
) -> Match:
    """Rows are [id, t, z, y, x] on the native Zarr/GEFF voxel grid."""
    candidate_rows = np.asarray(candidate_rows, dtype=np.float64)
    gt_rows = np.asarray(gt_rows, dtype=np.float64)
    if candidate_rows.ndim != 2 or candidate_rows.shape[1] != 5:
        raise ValueError("candidate rows must have 5 columns")
    if gt_rows.ndim != 2 or gt_rows.shape[1] != 5:
        raise ValueError("GT rows must have 5 columns")
    if len(set(candidate_rows[:, 0])) != len(candidate_rows):
        raise ValueError("duplicate candidate ID")
    if len(set(gt_rows[:, 0])) != len(gt_rows):
        raise ValueError("duplicate GT node ID")
    gt_to_candidate: dict[int, int] = {}
    candidate_to_gt: dict[int, int] = {}
    matched_distance_um: dict[int, float] = {}
    ambiguous_gt = 0
    ambiguous_candidate = 0
    for t in sorted(set(candidate_rows[:, 1]) | set(gt_rows[:, 1])):
        candidates = candidate_rows[candidate_rows[:, 1] == t]
        known = gt_rows[gt_rows[:, 1] == t]
        if len(candidates) == 0 or len(known) == 0:
            continue
        distances = cdist(
            candidates[:, 2:5] * NATIVE_SPACING_UM,
            known[:, 2:5] * NATIVE_SPACING_UM,
        )
        in_radius = distances <= radius_um
        ambiguous_gt += int((in_radius.sum(axis=0) > 1).sum())
        ambiguous_candidate += int((in_radius.sum(axis=1) > 1).sum())
        rows, cols = linear_sum_assignment(np.where(in_radius, distances, 1e9))
        for row, col in zip(rows, cols, strict=True):
            if not in_radius[row, col]:
                continue
            candidate_id = int(candidates[row, 0])
            gt_id = int(known[col, 0])
            gt_to_candidate[gt_id] = candidate_id
            candidate_to_gt[candidate_id] = gt_id
            matched_distance_um[gt_id] = float(distances[row, col])
    return Match(
        gt_to_candidate,
        candidate_to_gt,
        matched_distance_um,
        ambiguous_gt,
        ambiguous_candidate,
    )


def known_edges(
    gt_edges: Iterable[tuple[int, int]], match: Match
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    all_edges = {(int(source), int(target)) for source, target in gt_edges}
    mapped = {
        (match.gt_to_candidate[source], match.gt_to_candidate[target])
        for source, target in all_edges
        if source in match.gt_to_candidate and target in match.gt_to_candidate
    }
    return all_edges, mapped


def readout_stages(
    gt_edges: Iterable[tuple[int, int]],
    match: Match,
    stages: Mapping[str, Iterable[tuple[int, int]]],
) -> dict[str, dict[str, int]]:
    """Count known connections and complete annotated daughter sets by stage."""
    all_gt, mapped = known_edges(gt_edges, match)
    daughters: dict[int, set[int]] = defaultdict(set)
    for source, target in all_gt:
        daughters[source].add(target)
    divisions = {
        source: {
            (match.gt_to_candidate[source], match.gt_to_candidate[target]) for target in targets
        }
        for source, targets in daughters.items()
        if len(targets) >= 2
        and source in match.gt_to_candidate
        and all(target in match.gt_to_candidate for target in targets)
    }
    annotated_parent = {target: source for source, target in all_gt}
    output: dict[str, dict[str, int]] = {}
    for name, stage_edges in stages.items():
        selected = {(int(source), int(target)) for source, target in stage_edges}
        contradictory = 0
        for source, target in selected:
            gt_source = match.candidate_to_gt.get(source)
            gt_target = match.candidate_to_gt.get(target)
            if gt_source is not None and gt_target is not None:
                known_parent = annotated_parent.get(gt_target)
                contradictory += int(known_parent is not None and known_parent != gt_source)
        output[name] = {
            "known_edge_denominator": len(all_gt),
            "both_endpoints_detected": len(mapped),
            "known_edges_selected": len(mapped & selected),
            "known_division_denominator": sum(len(targets) >= 2 for targets in daughters.values()),
            "all_division_nodes_detected": len(divisions),
            "known_divisions_selected": sum(edges <= selected for edges in divisions.values()),
            "annotated_parent_contradictions": contradictory,
            "selected_edges": len(selected),
        }
    return output


def scored_known_edge_ranks(
    scores: Iterable[tuple[int, int, float]],
    matched_known_edges: set[tuple[int, int]],
) -> dict[tuple[int, int], tuple[float | None, int | None]]:
    known_sources = {int(source) for source, _ in matched_known_edges}
    by_source: dict[int, dict[int, float]] = defaultdict(dict)
    for source, target, score in scores:
        source = int(source)
        if source not in known_sources:
            continue
        target = int(target)
        value = float(score)
        if not np.isfinite(value):
            raise ValueError("nonfinite edge score")
        by_source[source][target] = max(by_source[source].get(target, -np.inf), value)
    result = {}
    for source, target in matched_known_edges:
        ranked = sorted(by_source.get(source, {}).items(), key=lambda item: (-item[1], item[0]))
        found = next(
            ((value, rank) for rank, (other, value) in enumerate(ranked, 1) if other == target),
            None,
        )
        result[source, target] = found if found is not None else (None, None)
    return result


def validate_initial_graph_ids(candidate_coords: np.ndarray, graph_rows: np.ndarray) -> set[int]:
    """Graph rows [id,t,z,y,x] may be a sparse subset of candidate [t,z,y,x]."""
    candidate_coords = np.asarray(candidate_coords, dtype=np.float64)
    graph_rows = np.asarray(graph_rows, dtype=np.float64)
    if candidate_coords.ndim != 2 or candidate_coords.shape[1] != 4:
        raise ValueError("candidate coordinates must have 4 columns")
    if graph_rows.ndim != 2 or graph_rows.shape[1] != 5:
        raise ValueError("graph rows must have 5 columns")
    ids = graph_rows[:, 0]
    if (
        len(ids) == 0
        or not np.isfinite(graph_rows).all()
        or not np.array_equal(ids, np.floor(ids))
        or np.any(ids < 0)
        or np.any(ids >= len(candidate_coords))
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("invalid or empty initial graph IDs")
    if not np.allclose(graph_rows[:, 1:5], candidate_coords[ids.astype(int)], rtol=0, atol=1e-4):
        raise ValueError("initial graph IDs do not map to cached candidate coordinates")
    return set(ids.astype(int))


def trace_original_edges(
    edges: Iterable[tuple[int, int]], original_ids: set[int]
) -> set[tuple[int, int]]:
    """Exclude nodes created after the initial ILP graph, even if IDs are reused."""
    return {
        (int(source), int(target))
        for source, target in edges
        if int(source) in original_ids and int(target) in original_ids
    }


# %% [markdown]
# ## 2. Verify the self-trained head input

# %%

import hashlib as _self_hashlib
import json as _self_json
from pathlib import Path as _SelfPath

_SELF_EXPECTED_SHA = "32d6c62f738c3dfe4862e3df5272850312e81b622e57d9ba45209ebb382824dc"
_SELF_HEAD_MATCHES = sorted(_SelfPath("/kaggle/input").rglob("self_trained_v1284_head.pt"))
if len(_SELF_HEAD_MATCHES) != 1:
    raise RuntimeError({"self_trained_head_matches": [str(p) for p in _SELF_HEAD_MATCHES]})
_SELF_HEAD_PATH = _SELF_HEAD_MATCHES[0]
_SELF_HEAD_SHA = _self_hashlib.sha256(_SELF_HEAD_PATH.read_bytes()).hexdigest()
if _SELF_HEAD_SHA != _SELF_EXPECTED_SHA:
    raise RuntimeError({"expected_head_sha": _SELF_EXPECTED_SHA, "actual": _SELF_HEAD_SHA})
print("Self-trained coordinate head:", _SELF_HEAD_PATH, _SELF_HEAD_SHA, flush=True)

# %% [markdown]
# ## 3. Run x138 with the self-trained coordinate head

# %%

import torch as _torch

if not _torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for this notebook. Enable a Kaggle GPU accelerator and commit again."
    )
print("CUDA device:", _torch.cuda.get_device_name(0))


_ps = REPO_DIR / "scripts" / "predict_unet_transformer.py"
_s = _ps.read_text()
_old = """        if cfg.det_tta:
            tta_flips = [(-1,), (-2,), (-2, -1)]
            for dims in tta_flips:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
            for f in range(W):
                det_logits[f] = det_logits[f] / 4"""
_new = """        if cfg.det_tta:
            _nv = 1
            for dims in [(-1,), (-2,), (-2, -1)]:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
                _nv += 1
            for _k in (1, 3):
                imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))
                _, det_rot = model.encode(imgs_rot)
                for f in range(W):
                    det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims=(-2, -1))
                del imgs_rot, det_rot
                _nv += 1
            imgs_t = imgs.transpose(-1, -2)
            _, det_t = model.encode(imgs_t)
            for f in range(W):
                det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)
            del imgs_t, det_t
            _nv += 1
            imgs_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)
            _, det_at = model.encode(imgs_at)
            for f in range(W):
                det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))
            del imgs_at, det_at
            _nv += 1
            for f in range(W):
                det_logits[f] = det_logits[f] / _nv"""
if _old in _s:
    _ps.write_text(_s.replace(_old, _new))
    print("TTA patch applied (400ep spatial D4-style)")
else:
    print("TTA WARNING: block not found - using default 4-way")


_s = _ps.read_text()
_ensemble_replacements = [
    (
        "    downsample: tuple[int, ...] = (1, 4, 4),\n) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:",
        '    downsample: tuple[int, ...] = (1, 4, 4),\n    secondary_model: UNetNodeTransformer | None = None,\n    secondary_edge_weight: float = 0.0,\n    secondary_detection_weight: float = 0.0,\n    secondary_link_mode: str = "raw",\n    secondary_mix_temperature: float = 1.0,\n    secondary_low_margin_max: float = 0.2,\n) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:',
    ),
    (
        "            for f in range(W):\n                det_logits[f] = det_logits[f] / _nv\n\n        del imgs",
        "            for f in range(W):\n                det_logits[f] = det_logits[f] / _nv\n\n        secondary_unet_out = None\n        if secondary_model is not None:\n            secondary_unet_out, secondary_det_logits = secondary_model.encode(imgs)\n\n            if secondary_detection_weight > 0.0:\n                if cfg.det_tta:\n                    _secondary_nv = 1\n                    for dims in [(-1,), (-2,), (-2, -1)]:\n                        secondary_imgs_flip = imgs.flip(dims)\n                        _, secondary_det_flip = secondary_model.encode(secondary_imgs_flip)\n                        for f in range(W):\n                            secondary_det_logits[f] = (\n                                secondary_det_logits[f] + secondary_det_flip[f].flip(dims)\n                            )\n                        del secondary_imgs_flip, secondary_det_flip\n                        _secondary_nv += 1\n                    for _k in (1, 3):\n                        secondary_imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))\n                        _, secondary_det_rot = secondary_model.encode(secondary_imgs_rot)\n                        for f in range(W):\n                            secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                                secondary_det_rot[f], -_k, dims=(-2, -1)\n                            )\n                        del secondary_imgs_rot, secondary_det_rot\n                        _secondary_nv += 1\n                    secondary_imgs_t = imgs.transpose(-1, -2)\n                    _, secondary_det_t = secondary_model.encode(secondary_imgs_t)\n                    for f in range(W):\n                        secondary_det_logits[f] = (\n                            secondary_det_logits[f] + secondary_det_t[f].transpose(-1, -2)\n                        )\n                    del secondary_imgs_t, secondary_det_t\n                    _secondary_nv += 1\n                    secondary_imgs_at = torch.rot90(\n                        imgs, 1, dims=(-2, -1)\n                    ).transpose(-1, -2)\n                    _, secondary_det_at = secondary_model.encode(secondary_imgs_at)\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                            secondary_det_at[f].transpose(-1, -2),\n                            -1,\n                            dims=(-2, -1),\n                        )\n                    del secondary_imgs_at, secondary_det_at\n                    _secondary_nv += 1\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] / _secondary_nv\n\n                for f in range(W):\n                    primary_det = det_logits[f]\n                    secondary_det = secondary_det_logits[f]\n                    primary_mean = primary_det.mean()\n                    secondary_mean = secondary_det.mean()\n                    primary_scale = primary_det.float().std(unbiased=False).clamp_min(1e-4)\n                    secondary_scale = secondary_det.float().std(unbiased=False).clamp_min(1e-4)\n                    scale_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)\n                    secondary_det_aligned = (\n                        (secondary_det - secondary_mean) * scale_ratio + primary_mean\n                    )\n                    det_logits[f] = (\n                        (1.0 - secondary_detection_weight) * primary_det\n                        + secondary_detection_weight * secondary_det_aligned\n                    )\n\n            del secondary_det_logits\n\n        del imgs",
    ),
    (
        "            edge_logits_pair = model.predict_edges(\n                unet_feat_src, unet_feat_tgt,\n                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                p_pos_src, p_pos_tgt,\n                p_mask_src, p_mask_tgt,\n            )  # (1, n_src, n_tgt)\n\n            raw = edge_logits_pair[0]",
        '            edge_logits_pair = model.predict_edges(\n                unet_feat_src, unet_feat_tgt,\n                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                p_pos_src, p_pos_tgt,\n                p_mask_src, p_mask_tgt,\n            )  # (1, n_src, n_tgt)\n\n            if secondary_model is not None:\n                if secondary_unet_out is None:\n                    raise RuntimeError("Secondary model is loaded but its feature map is missing")\n                secondary_feat_src = secondary_model._index_features(\n                    secondary_unet_out[:, f_idx], p_coords_src, p_mask_src,\n                )\n                secondary_feat_tgt = secondary_model._index_features(\n                    secondary_unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt,\n                )\n                secondary_logits_pair = secondary_model.predict_edges(\n                    secondary_feat_src, secondary_feat_tgt,\n                    p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                    p_pos_src, p_pos_tgt,\n                    p_mask_src, p_mask_tgt,\n                )\n\n                if secondary_link_mode == "raw":\n                    secondary_for_mix = secondary_logits_pair\n                    blend_weight = secondary_edge_weight\n                elif secondary_link_mode in {\n                    "calibrated", "adaptive", "low_margin_consensus"\n                }:\n                    primary_center = edge_logits_pair.mean(dim=1, keepdim=True)\n                    primary_scale = edge_logits_pair.float().std(\n                        dim=1, keepdim=True, unbiased=False\n                    ).clamp_min(1e-4)\n                    secondary_center = secondary_logits_pair.mean(dim=1, keepdim=True)\n                    secondary_scale = secondary_logits_pair.float().std(\n                        dim=1, keepdim=True, unbiased=False\n                    ).clamp_min(1e-4)\n                    secondary_scale_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)\n                    secondary_for_mix = (\n                        (secondary_logits_pair - secondary_center) * secondary_scale_ratio\n                        + primary_center\n                    )\n                    if secondary_link_mode == "calibrated":\n                        blend_weight = secondary_edge_weight\n                    elif secondary_link_mode == "adaptive":\n                        if n_src >= 2:\n                            primary_probs = torch.softmax(edge_logits_pair[0], dim=0)\n                            secondary_probs = torch.softmax(secondary_for_mix[0], dim=0)\n                            primary_top2 = torch.topk(primary_probs, k=2, dim=0)\n                            secondary_top2 = torch.topk(secondary_probs, k=2, dim=0)\n                            primary_margin = primary_top2.values[0] - primary_top2.values[1]\n                            secondary_margin = secondary_top2.values[0] - secondary_top2.values[1]\n                            local_weight = (\n                                secondary_edge_weight + secondary_margin - primary_margin\n                            ).clamp(0.15, 0.75)\n                            same_parent = primary_top2.indices[0].eq(\n                                secondary_top2.indices[0]\n                            )\n                            local_weight = torch.where(\n                                same_parent,\n                                torch.maximum(\n                                    local_weight,\n                                    torch.full_like(local_weight, secondary_edge_weight),\n                                ),\n                                local_weight,\n                            )\n                            blend_weight = local_weight.view(1, 1, -1)\n                        else:\n                            blend_weight = secondary_edge_weight\n                    else:\n                        if n_src >= 2:\n                            primary_probs = torch.softmax(edge_logits_pair[0], dim=0)\n                            secondary_probs = torch.softmax(secondary_for_mix[0], dim=0)\n                            primary_top2 = torch.topk(primary_probs, k=2, dim=0)\n                            secondary_top2 = torch.topk(secondary_probs, k=2, dim=0)\n                            primary_margin = primary_top2.values[0] - primary_top2.values[1]\n                            same_parent = primary_top2.indices[0].eq(\n                                secondary_top2.indices[0]\n                            )\n                            uncertainty = (\n                                (secondary_low_margin_max - primary_margin)\n                                / secondary_low_margin_max\n                            ).clamp(0.0, 1.0)\n                            local_weight = secondary_edge_weight * uncertainty\n                            local_weight = torch.where(\n                                same_parent,\n                                local_weight,\n                                torch.zeros_like(local_weight),\n                            )\n                            blend_weight = local_weight.view(1, 1, -1)\n                        else:\n                            blend_weight = 0.0\n                else:\n                    raise ValueError(f"Unsupported secondary link mode: {secondary_link_mode}")\n\n                edge_logits_pair = (\n                    (1.0 - blend_weight) * edge_logits_pair\n                    + blend_weight * secondary_for_mix\n                )\n                if secondary_mix_temperature != 1.0:\n                    mixed_center = edge_logits_pair.mean(dim=1, keepdim=True)\n                    edge_logits_pair = mixed_center + (\n                        edge_logits_pair - mixed_center\n                    ) / secondary_mix_temperature\n\n            raw = edge_logits_pair[0]',
    ),
    (
        "        del unet_out\n",
        "        del unet_out\n        if secondary_unet_out is not None:\n            del secondary_unet_out\n",
    ),
    (
        "    model, window_size, downsample = load_model(weights_path, device)\n    print(",
        '    model, window_size, downsample = load_model(weights_path, device)\n\n    secondary_model = None\n    secondary_weights_text = os.environ.get("BIOHUB_SECONDARY_WEIGHTS", "").strip()\n    secondary_edge_weight = float(os.environ.get("BIOHUB_SECONDARY_EDGE_WEIGHT", "0"))\n    secondary_detection_weight = float(\n        os.environ.get("BIOHUB_SECONDARY_DETECTION_WEIGHT", "0")\n    )\n    secondary_link_mode = os.environ.get("BIOHUB_SECONDARY_LINK_MODE", "raw").strip()\n    secondary_mix_temperature = float(\n        os.environ.get("BIOHUB_SECONDARY_MIX_TEMPERATURE", "1")\n    )\n    secondary_low_margin_max = float(\n        os.environ.get("BIOHUB_SECONDARY_LOW_MARGIN_MAX", "0.2")\n    )\n    edge_candidate_threshold = float(\n        os.environ.get("BIOHUB_DUAL_SEED_EDGE_THRESHOLD", str(cfg.threshold))\n    )\n    if secondary_weights_text:\n        if not 0.0 < secondary_edge_weight < 1.0:\n            raise ValueError("BIOHUB_SECONDARY_EDGE_WEIGHT must be strictly between 0 and 1")\n        if not 0.0 <= secondary_detection_weight < 1.0:\n            raise ValueError(\n                "BIOHUB_SECONDARY_DETECTION_WEIGHT must be in the half-open interval [0, 1)"\n            )\n        if secondary_link_mode not in {\n            "raw", "calibrated", "adaptive", "low_margin_consensus"\n        }:\n            raise ValueError(\n                "BIOHUB_SECONDARY_LINK_MODE must be raw, calibrated, adaptive, "\n                "or low_margin_consensus"\n            )\n        if not 0.5 <= secondary_mix_temperature <= 2.0:\n            raise ValueError("BIOHUB_SECONDARY_MIX_TEMPERATURE must be in [0.5, 2.0]")\n        if not 0.0 < edge_candidate_threshold < 1.0:\n            raise ValueError("BIOHUB_DUAL_SEED_EDGE_THRESHOLD must be strictly between 0 and 1")\n        if not 0.0 < secondary_low_margin_max <= 1.0:\n            raise ValueError("BIOHUB_SECONDARY_LOW_MARGIN_MAX must be in (0, 1]")\n        secondary_model, secondary_window_size, secondary_downsample = load_model(\n            Path(secondary_weights_text), device,\n        )\n        if secondary_window_size != window_size or secondary_downsample != downsample:\n            raise ValueError(\n                "Primary and secondary models have incompatible inference grids: "\n                f"primary=(window={window_size}, downsample={downsample}), "\n                f"secondary=(window={secondary_window_size}, downsample={secondary_downsample})"\n            )\n        cfg.threshold = edge_candidate_threshold\n        print(\n            f"Secondary model: {secondary_weights_text} | "\n            f"edge weight={secondary_edge_weight:.3f} | "\n            f"detection weight={secondary_detection_weight:.3f} | "\n            f"link mode={secondary_link_mode} | "\n            f"temperature={secondary_mix_temperature:.3f} | "\n            f"low-margin max={secondary_low_margin_max:.3f} | "\n            f"edge threshold={cfg.threshold:.3f}",\n            flush=True,\n        )\n\n    print(',
    ),
    (
        "                unet_batch_size=unet_batch_size,\n                downsample=downsample,\n            )",
        "                unet_batch_size=unet_batch_size,\n                downsample=downsample,\n                secondary_model=secondary_model,\n                secondary_edge_weight=secondary_edge_weight,\n                secondary_detection_weight=secondary_detection_weight,\n                secondary_link_mode=secondary_link_mode,\n                secondary_mix_temperature=secondary_mix_temperature,\n                secondary_low_margin_max=secondary_low_margin_max,\n            )",
    ),
]
for _patch_index, (_ensemble_old, _ensemble_new) in enumerate(_ensemble_replacements, start=1):
    _ensemble_count = _s.count(_ensemble_old)
    if _ensemble_count != 1:
        raise RuntimeError(
            f"Calibrated dual-seed patch {_patch_index} expected one match, found {_ensemble_count}"
        )
    _s = _s.replace(_ensemble_old, _ensemble_new, 1)
compile(_s, str(_ps), "exec")
_ps.write_text(_s)
print("Calibrated dual-seed runtime patch applied")


os.environ["BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION"] = "0.90"
for _guard_old_log in WORKING_DIR.glob("retention_guard_*.jsonl"):
    _guard_old_log.unlink()

_s = _ps.read_text()
_guard_old = """                    det_logits[f] = (
                        (1.0 - secondary_detection_weight) * primary_det
                        + secondary_detection_weight * secondary_det_aligned
                    )"""
_guard_new = """                    blended_det = (
                        (1.0 - secondary_detection_weight) * primary_det
                        + secondary_detection_weight * secondary_det_aligned
                    )
                    primary_candidates = len(_detect_cells_pooled(
                        primary_det[0],
                        int(frame_indices[f]),
                        cfg.det_threshold,
                        pool_k,
                    ))
                    blended_candidates = len(_detect_cells_pooled(
                        blended_det[0],
                        int(frame_indices[f]),
                        cfg.det_threshold,
                        pool_k,
                    ))
                    minimum_retention = float(os.environ.get(
                        "BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION",
                        "0.90",
                    ))
                    candidate_retention = (
                        blended_candidates / primary_candidates
                        if primary_candidates
                        else 1.0
                    )
                    use_primary_detection = bool(
                        primary_candidates > 0
                        and candidate_retention < minimum_retention
                    )
                    det_logits[f] = (
                        primary_det if use_primary_detection else blended_det
                    )
                    if int(frame_indices[f]) not in seen_frames:
                        shard = os.environ.get(
                            "BIOHUB_GPU_SHARD", "single"
                        ).replace("/", "_")
                        guard_log = (
                            Path("/kaggle/working")
                            / f"retention_guard_{shard}.jsonl"
                        )
                        guard_record = {
                            "dataset": ds_path.stem,
                            "frame": int(frame_indices[f]),
                            "primary_candidates": int(primary_candidates),
                            "blended_candidates": int(blended_candidates),
                            "retention": float(candidate_retention),
                            "minimum_retention": float(minimum_retention),
                            "use_primary": bool(use_primary_detection),
                        }
                        with guard_log.open("a") as guard_handle:
                            guard_handle.write(
                                json.dumps(guard_record, sort_keys=True)
                                + "\\n"
                            )
                        if use_primary_detection:
                            print(
                                "BIOHUB_RETENTION_GUARD "
                                + json.dumps(guard_record, sort_keys=True),
                                flush=True,
                            )"""
_guard_matches = _s.count(_guard_old)
if _guard_matches != 1:
    raise RuntimeError(f"Retention guard expected one blend block, found {_guard_matches}")
_s = _s.replace(_guard_old, _guard_new, 1)
compile(_s, str(_ps), "exec")
_ps.write_text(_s)
print(
    "Frozen frame retention guard applied at "
    + os.environ["BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION"]
)


import math as _bidirectional_math

_bidirectional_weight_guard = float(os.environ.get("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT", "0"))
if not _bidirectional_math.isclose(_bidirectional_weight_guard, 0.15, rel_tol=0.0, abs_tol=1e-12):
    raise ValueError(
        {
            "expected_bidirectional_weight": 0.15,
            "actual_bidirectional_weight": _bidirectional_weight_guard,
        }
    )

_s = _ps.read_text()
_bi_old = "            edge_logits_pair = model.predict_edges(\n                unet_feat_src, unet_feat_tgt,\n                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                p_pos_src, p_pos_tgt,\n                p_mask_src, p_mask_tgt,\n            )  # (1, n_src, n_tgt)\n\n            if secondary_model is not None:\n"
_bi_new = '            edge_logits_pair = model.predict_edges(\n                unet_feat_src, unet_feat_tgt,\n                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                p_pos_src, p_pos_tgt,\n                p_mask_src, p_mask_tgt,\n            )  # (1, n_src, n_tgt)\n\n            _bidirectional_weight = float(\n                os.environ.get("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT", "0")\n            )\n            if _bidirectional_weight > 0.0:\n                reverse_logits_native = model.predict_edges(\n                    unet_feat_tgt, unet_feat_src,\n                    p_coords_tgt * ds_arr_t, p_coords_src * ds_arr_t,\n                    p_pos_tgt, p_pos_src,\n                    p_mask_tgt, p_mask_src,\n                )  # (1, n_tgt, n_src)\n                reverse_logits_pair = reverse_logits_native.transpose(1, 2)\n\n                forward_center = edge_logits_pair.mean(dim=1, keepdim=True)\n                forward_scale = edge_logits_pair.float().std(\n                    dim=1, keepdim=True, unbiased=False\n                ).clamp_min(1e-4)\n                reverse_center = reverse_logits_pair.mean(dim=1, keepdim=True)\n                reverse_scale = reverse_logits_pair.float().std(\n                    dim=1, keepdim=True, unbiased=False\n                ).clamp_min(1e-4)\n                reverse_scale_ratio = (forward_scale / reverse_scale).clamp(0.5, 2.0)\n                reverse_scale_ratio = reverse_scale_ratio.to(reverse_logits_pair.dtype)\n                reverse_aligned = (\n                    (reverse_logits_pair - reverse_center) * reverse_scale_ratio\n                    + forward_center\n                )\n                # Biohub 145: require mutual forward/reverse support in probability space.\n                # The harmonic mean penalizes a candidate when either temporal direction\n                # assigns it very low probability, while calibration preserves the forward\n                # logit scale used by the unchanged downstream candidate threshold and ILP.\n                forward_prob = torch.softmax(edge_logits_pair.float(), dim=1).clamp_min(1e-8)\n                reverse_prob = torch.softmax(reverse_aligned.float(), dim=1).clamp_min(1e-8)\n                harmonic_prob = 1.0 / (\n                    (1.0 - _bidirectional_weight) / forward_prob\n                    + _bidirectional_weight / reverse_prob\n                )\n                harmonic_prob = harmonic_prob / harmonic_prob.sum(\n                    dim=1, keepdim=True\n                ).clamp_min(1e-8)\n                harmonic_logits = torch.log(harmonic_prob.clamp_min(1e-8))\n                harmonic_center = harmonic_logits.mean(dim=1, keepdim=True)\n                harmonic_scale = harmonic_logits.std(\n                    dim=1, keepdim=True, unbiased=False\n                ).clamp_min(1e-4)\n                harmonic_scale_ratio = (forward_scale / harmonic_scale).clamp(0.5, 2.0)\n                edge_logits_pair = (\n                    (harmonic_logits - harmonic_center) * harmonic_scale_ratio\n                    + forward_center\n                ).to(reverse_aligned.dtype)\n                del (\n                    reverse_logits_native,\n                    reverse_logits_pair,\n                    reverse_aligned,\n                    forward_prob,\n                    reverse_prob,\n                    harmonic_prob,\n                    harmonic_logits,\n                )\n            if secondary_model is not None:\n'
_bi_count = _s.count(_bi_old)
if _bi_count != 1:
    raise RuntimeError(
        f"Bidirectional edge patch expected one transformed block, found {_bi_count}"
    )
_s = _s.replace(_bi_old, _bi_new, 1)

_coordinate_manifest_old = "    coords = coords.astype(np.int16)\n    return coords, all_edges"
_coordinate_manifest_new = '    coords = coords.astype(np.int16)\n\n    # Label-free, pre-ILP detector-coordinate manifest. This executes inside\n    # predict_video, before build_graph and the ILP call in predict().\n    _coordinate_manifest_arm = os.environ.get(\n        "BIOHUB_DIAGNOSTIC_ARM", ""\n    ).strip()\n    if _coordinate_manifest_arm:\n        import hashlib as _coordinate_hashlib\n\n        _coordinate_shard = os.environ.get(\n            "BIOHUB_GPU_SHARD", "single"\n        ).replace("/", "_")\n        _coordinate_array = np.ascontiguousarray(\n            coords.astype("<i2", copy=False)\n        )\n        _coordinate_frame_counts = [\n            [int(_coordinate_t), int((_coordinate_array[:, 0] == _coordinate_t).sum())]\n            for _coordinate_t in np.unique(_coordinate_array[:, 0])\n        ]\n        _coordinate_record = {\n            "columns": ["t", "z", "y", "x"],\n            "coordinate_sha256": _coordinate_hashlib.sha256(\n                _coordinate_array.tobytes(order="C")\n            ).hexdigest(),\n            "dataset": ds_path.stem,\n            "dtype": "<i2",\n            "frame_counts": _coordinate_frame_counts,\n            "rows": int(len(_coordinate_array)),\n            "stage": "post_detection_pre_graph_pre_ilp",\n        }\n        _coordinate_manifest_path = (\n            Path("/kaggle/working")\n            / f"detector_coordinates_{_coordinate_manifest_arm}_"\n            f"{_coordinate_shard}.jsonl"\n        )\n        with _coordinate_manifest_path.open("a") as _coordinate_handle:\n            _coordinate_handle.write(\n                json.dumps(_coordinate_record, sort_keys=True) + "\\n"\n            )\n\n    return coords, all_edges'
_coordinate_manifest_count = _s.count(_coordinate_manifest_old)
if _coordinate_manifest_count != 1:
    raise RuntimeError(
        "Coordinate-manifest patch expected one pre-return block, found "
        f"{_coordinate_manifest_count}"
    )
_s = _s.replace(_coordinate_manifest_old, _coordinate_manifest_new, 1)

# Runtime profile and bound for the ILP. The support pack solves each dataset's
# whole graph in one SCIP model with no time limit; print how long that takes
# and cap it with BIOHUB_ILP_TIMEOUT_S (SCIP returns its incumbent at the limit,
# and tracksdata only logs a warning when the status is not OPTIMAL).
_ilp_old = (
    "            solver = td.solvers.ILPSolver(\n"
    '                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),\n'
    "                appearance_weight=cfg.ilp_appearance_weight,\n"
    "                disappearance_weight=cfg.ilp_disappearance_weight,\n"
    "                division_weight=cfg.ilp_division_weight,\n"
    "            )\n"
    "            with suppress_output():\n"
    "                graph = solver.solve(graph)\n"
)
_ilp_new = (
    "            import time as _ilp_time\n"
    '            _ilp_timeout = float(os.environ.get("BIOHUB_ILP_TIMEOUT_S", "0") or 0.0)\n'
    "            solver = td.solvers.ILPSolver(\n"
    '                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),\n'
    "                appearance_weight=cfg.ilp_appearance_weight,\n"
    "                disappearance_weight=cfg.ilp_disappearance_weight,\n"
    "                division_weight=cfg.ilp_division_weight,\n"
    "                timeout=_ilp_timeout if _ilp_timeout > 0 else None,\n"
    "            )\n"
    "            _ilp_t0 = _ilp_time.time()\n"
    "            _ilp_nodes, _ilp_edges = graph.num_nodes(), graph.num_edges()\n"
    "            with suppress_output():\n"
    "                graph = solver.solve(graph)\n"
    "            print(\n"
    '                f"[{name}] ILP {_ilp_time.time() - _ilp_t0:.1f}s"\n'
    '                f" | candidate nodes={_ilp_nodes} edges={_ilp_edges}"\n'
    '                f" | timeout={_ilp_timeout if _ilp_timeout > 0 else None}",\n'
    "                flush=True,\n"
    "            )\n"
)
_ilp_count = _s.count(_ilp_old)
if _ilp_count != 1:
    raise RuntimeError(f"ILP timeout patch expected one solver block, found {_ilp_count}")
_s = _s.replace(_ilp_old, _ilp_new, 1)
print("ILP timeout patch applied |", os.environ.get("BIOHUB_ILP_TIMEOUT_S", "0"), "s per dataset")

# Per-dataset detection timing. Diagnostic only, so a missing anchor is a
# warning rather than a failed kernel.
_pv_old = "        coords, edges = predict_video(\n"
_bg_old = "        graph = build_graph(coords, edges)\n"
if _s.count(_pv_old) == 1 and _s.count(_bg_old) == 1:
    _s = _s.replace(
        _pv_old,
        "        import time as _pv_time\n        _pv_t0 = _pv_time.time()\n" + _pv_old,
        1,
    )
    _s = _s.replace(
        _bg_old,
        _bg_old
        + '        print(f"[{name}] detection+edges {_pv_time.time() - _pv_t0:.1f}s'
        + ' | detections={len(coords)}", flush=True)\n',
        1,
    )
    print("Per-dataset detection timing patch applied")
else:
    print(
        "WARNING: detection timing anchors not unique "
        f"({_s.count(_pv_old)}, {_s.count(_bg_old)}); timing prints skipped"
    )
compile(_s, str(_ps), "exec")
_ps.write_text(_s)
print(
    "Bidirectional harmonic-probability association fusion applied | weight=",
    _bidirectional_weight_guard,
)
print("Pre-ILP detector-coordinate manifest hook applied")


_et_s = _ps.read_text()
_et_old = "        if cfg.det_tta:\n            _nv = 1\n            for dims in [(-1,), (-2,), (-2, -1)]:\n                imgs_flip = imgs.flip(dims)\n                _, det_flip = model.encode(imgs_flip)\n                for f in range(W):\n                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)\n                del imgs_flip, det_flip\n                _nv += 1\n            for _k in (1, 3):\n                imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))\n                _, det_rot = model.encode(imgs_rot)\n                for f in range(W):\n                    det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims=(-2, -1))\n                del imgs_rot, det_rot\n                _nv += 1\n            imgs_t = imgs.transpose(-1, -2)\n            _, det_t = model.encode(imgs_t)\n            for f in range(W):\n                det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)\n            del imgs_t, det_t\n            _nv += 1\n            imgs_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)\n            _, det_at = model.encode(imgs_at)\n            for f in range(W):\n                det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))\n            del imgs_at, det_at\n            _nv += 1\n            for f in range(W):\n                det_logits[f] = det_logits[f] / _nv\n"
_et_new = "        if cfg.det_tta:\n            _edge_tta = os.environ.get('BIOHUB_EDGE_FEATURE_TTA', '0') != '0'\n            _unet_acc = unet_out.clone() if _edge_tta else None\n            _nv = 1\n            for dims in [(-1,), (-2,), (-2, -1)]:\n                imgs_flip = imgs.flip(dims)\n                _u_flip, det_flip = model.encode(imgs_flip)\n                for f in range(W):\n                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)\n                if _edge_tta:\n                    _unet_acc = _unet_acc + _u_flip.flip(dims)\n                del imgs_flip, det_flip, _u_flip\n                _nv += 1\n            for _k in (1, 3):\n                imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))\n                _u_rot, det_rot = model.encode(imgs_rot)\n                for f in range(W):\n                    det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims=(-2, -1))\n                if _edge_tta:\n                    _unet_acc = _unet_acc + torch.rot90(_u_rot, -_k, dims=(-2, -1))\n                del imgs_rot, det_rot, _u_rot\n                _nv += 1\n            imgs_t = imgs.transpose(-1, -2)\n            _u_t, det_t = model.encode(imgs_t)\n            for f in range(W):\n                det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)\n            if _edge_tta:\n                _unet_acc = _unet_acc + _u_t.transpose(-1, -2)\n            del imgs_t, det_t, _u_t\n            _nv += 1\n            imgs_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)\n            _u_at, det_at = model.encode(imgs_at)\n            for f in range(W):\n                det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))\n            if _edge_tta:\n                _unet_acc = _unet_acc + torch.rot90(_u_at.transpose(-1, -2), -1, dims=(-2, -1))\n            del imgs_at, det_at, _u_at\n            _nv += 1\n            for f in range(W):\n                det_logits[f] = det_logits[f] / _nv\n            if _edge_tta:\n                if _unet_acc.shape != unet_out.shape:\n                    raise RuntimeError('EDGE-TTA SHAPE MISMATCH: %s vs %s'\n                                       % (tuple(_unet_acc.shape), tuple(unet_out.shape)))\n                _delta = float((_unet_acc / _nv - unet_out).abs().mean())\n                if _delta == 0.0:\n                    raise RuntimeError('EDGE-TTA NO-OP: averaged features bit-identical to the '\n                                       'single-pass features, so the augmented encodes '\n                                       'contributed nothing and this arm would read as a '\n                                       'false null')\n                unet_out = _unet_acc / _nv\n                print('EDGE_TTA_ACTIVE views=', _nv, 'mean_abs_feat_delta=', round(_delta, 6), flush=True)\n                del _unet_acc\n"
if _et_s.count(_et_old) != 1:
    raise RuntimeError("edge-TTA anchor block not unique: %d" % _et_s.count(_et_old))
_et_s = _et_s.replace(_et_old, _et_new, 1)
compile(_et_s, str(_ps), "exec")
_ps.write_text(_et_s)
if "EDGE_TTA_ACTIVE" not in _ps.read_text():
    raise RuntimeError("EDGE-TTA PATCH DID NOT PERSIST")
os.environ["BIOHUB_EDGE_FEATURE_TTA"] = "1"
print("EDGE_TTA patch installed and enabled in", _ps)

_secondary_tta_source = _ps.read_text()
_secondary_tta_old = "        secondary_unet_out, secondary_det_logits = secondary_model.encode(imgs)\n\n            if secondary_detection_weight > 0.0:\n                if cfg.det_tta:\n                    _secondary_nv = 1\n                    for dims in [(-1,), (-2,), (-2, -1)]:\n                        secondary_imgs_flip = imgs.flip(dims)\n                        _, secondary_det_flip = secondary_model.encode(secondary_imgs_flip)\n                        for f in range(W):\n                            secondary_det_logits[f] = (\n                                secondary_det_logits[f] + secondary_det_flip[f].flip(dims)\n                            )\n                        del secondary_imgs_flip, secondary_det_flip\n                        _secondary_nv += 1\n                    for _k in (1, 3):\n                        secondary_imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))\n                        _, secondary_det_rot = secondary_model.encode(secondary_imgs_rot)\n                        for f in range(W):\n                            secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                                secondary_det_rot[f], -_k, dims=(-2, -1)\n                            )\n                        del secondary_imgs_rot, secondary_det_rot\n                        _secondary_nv += 1\n                    secondary_imgs_t = imgs.transpose(-1, -2)\n                    _, secondary_det_t = secondary_model.encode(secondary_imgs_t)\n                    for f in range(W):\n                        secondary_det_logits[f] = (\n                            secondary_det_logits[f] + secondary_det_t[f].transpose(-1, -2)\n                        )\n                    del secondary_imgs_t, secondary_det_t\n                    _secondary_nv += 1\n                    secondary_imgs_at = torch.rot90(\n                        imgs, 1, dims=(-2, -1)\n                    ).transpose(-1, -2)\n                    _, secondary_det_at = secondary_model.encode(secondary_imgs_at)\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                            secondary_det_at[f].transpose(-1, -2),\n                            -1,\n                            dims=(-2, -1),\n                        )\n                    del secondary_imgs_at, secondary_det_at\n                    _secondary_nv += 1\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] / _secondary_nv\n\n                for f in range(W):"
_secondary_tta_new = '        secondary_unet_out, secondary_det_logits = secondary_model.encode(imgs)\n            _secondary_edge_tta = os.environ.get(\n                "BIOHUB_SECONDARY_EDGE_FEATURE_TTA", "0"\n            ) != "0"\n            _secondary_unet_acc = (\n                secondary_unet_out.clone() if _secondary_edge_tta else None\n            )\n\n            if secondary_detection_weight > 0.0:\n                if cfg.det_tta:\n                    _secondary_nv = 1\n                    for dims in [(-1,), (-2,), (-2, -1)]:\n                        secondary_imgs_flip = imgs.flip(dims)\n                        _secondary_u_flip, secondary_det_flip = secondary_model.encode(\n                            secondary_imgs_flip\n                        )\n                        for f in range(W):\n                            secondary_det_logits[f] = (\n                                secondary_det_logits[f] + secondary_det_flip[f].flip(dims)\n                            )\n                        if _secondary_edge_tta:\n                            _secondary_unet_acc = _secondary_unet_acc + _secondary_u_flip.flip(dims)\n                        del secondary_imgs_flip, secondary_det_flip, _secondary_u_flip\n                        _secondary_nv += 1\n                    for _k in (1, 3):\n                        secondary_imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))\n                        _secondary_u_rot, secondary_det_rot = secondary_model.encode(\n                            secondary_imgs_rot\n                        )\n                        for f in range(W):\n                            secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                                secondary_det_rot[f], -_k, dims=(-2, -1)\n                            )\n                        if _secondary_edge_tta:\n                            _secondary_unet_acc = _secondary_unet_acc + torch.rot90(\n                                _secondary_u_rot, -_k, dims=(-2, -1)\n                            )\n                        del secondary_imgs_rot, secondary_det_rot, _secondary_u_rot\n                        _secondary_nv += 1\n                    secondary_imgs_t = imgs.transpose(-1, -2)\n                    _secondary_u_t, secondary_det_t = secondary_model.encode(secondary_imgs_t)\n                    for f in range(W):\n                        secondary_det_logits[f] = (\n                            secondary_det_logits[f] + secondary_det_t[f].transpose(-1, -2)\n                        )\n                    if _secondary_edge_tta:\n                        _secondary_unet_acc = _secondary_unet_acc + _secondary_u_t.transpose(-1, -2)\n                    del secondary_imgs_t, secondary_det_t, _secondary_u_t\n                    _secondary_nv += 1\n                    secondary_imgs_at = torch.rot90(\n                        imgs, 1, dims=(-2, -1)\n                    ).transpose(-1, -2)\n                    _secondary_u_at, secondary_det_at = secondary_model.encode(\n                        secondary_imgs_at\n                    )\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                            secondary_det_at[f].transpose(-1, -2),\n                            -1,\n                            dims=(-2, -1),\n                        )\n                    if _secondary_edge_tta:\n                        _secondary_unet_acc = _secondary_unet_acc + torch.rot90(\n                            _secondary_u_at.transpose(-1, -2), -1, dims=(-2, -1)\n                        )\n                    del secondary_imgs_at, secondary_det_at, _secondary_u_at\n                    _secondary_nv += 1\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] / _secondary_nv\n                    if _secondary_edge_tta:\n                        if _secondary_unet_acc.shape != secondary_unet_out.shape:\n                            raise RuntimeError("SECONDARY_EDGE_TTA_SHAPE_MISMATCH")\n                        _secondary_delta = float(\n                            (_secondary_unet_acc / _secondary_nv - secondary_unet_out).abs().mean()\n                        )\n                        if _secondary_delta == 0.0:\n                            raise RuntimeError("SECONDARY_EDGE_TTA_NO_OP")\n                        _secondary_edge_tta_weight = float(os.environ.get(\n                            "BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT", "1.0"\n                        ))\n                        if not 0.0 < _secondary_edge_tta_weight <= 1.0:\n                            raise RuntimeError("SECONDARY_EDGE_TTA_BAD_WEIGHT")\n                        _secondary_tta_mean = _secondary_unet_acc / _secondary_nv\n                        secondary_unet_out = (\n                            (1.0 - _secondary_edge_tta_weight) * secondary_unet_out\n                            + _secondary_edge_tta_weight * _secondary_tta_mean\n                        )\n                        print(\n                            "SECONDARY_EDGE_TTA_ACTIVE views=",\n                            _secondary_nv,\n                            "weight=",\n                            _secondary_edge_tta_weight,\n                            "mean_abs_feat_delta=",\n                            round(_secondary_delta, 6),\n                            flush=True,\n                        )\n                        del _secondary_unet_acc\n\n                for f in range(W):'
_secondary_tta_count = _secondary_tta_source.count(_secondary_tta_old)
if _secondary_tta_count != 1:
    raise RuntimeError(
        "secondary edge-TTA anchor expected one match, found " + str(_secondary_tta_count)
    )
_secondary_tta_source = _secondary_tta_source.replace(_secondary_tta_old, _secondary_tta_new, 1)
compile(_secondary_tta_source, str(_ps), "exec")
_ps.write_text(_secondary_tta_source)
if "SECONDARY_EDGE_TTA_ACTIVE" not in _ps.read_text():
    raise RuntimeError("secondary edge-TTA patch did not persist")
os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA"] = "1"
os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT"] = "0.75"
print("secondary edge-feature TTA patch installed and enabled", flush=True)


def list_test_stems() -> list[str]:
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory does not exist: {TEST_DIR}")
    stems = sorted(path.name[:-5] for path in TEST_DIR.iterdir() if path.name.endswith(".zarr"))
    if not stems:
        raise FileNotFoundError(f"No test .zarr files found in {TEST_DIR}")
    return stems


# Label-blind split: diagnostic ranks 11-20, inference ranks 21-30.
from settings import load_config as _exp052_load_config

_EXP052_CONFIG = _exp052_load_config()
if EXP052_PHASE not in {"diagnostic", "inference"}:
    raise RuntimeError(f"unsupported phase: {EXP052_PHASE}")
_EXP052_DATA = _EXP052_CONFIG["data"]
_EXP052_RANK_START = int(
    _EXP052_DATA[
        "internal_rank_start_per_embryo"
        if EXP052_PHASE == "diagnostic"
        else "evaluation_rank_start_per_embryo"
    ]
)
_EXP052_RANK_END = int(
    _EXP052_DATA[
        "internal_rank_end_per_embryo"
        if EXP052_PHASE == "diagnostic"
        else "evaluation_rank_end_per_embryo"
    ]
)
if _EXP052_RANK_END - _EXP052_RANK_START != 9:
    raise RuntimeError("expected ten videos per embryo")
TRAIN_DIR = COMP_DIR / "train"
_manifest_matches = sorted(Path("/kaggle/input").rglob("self_trained_head_manifest.json"))
if len(_manifest_matches) != 1:
    raise RuntimeError({"head_manifest_matches": [str(p) for p in _manifest_matches]})
_head_manifest_path = _manifest_matches[0]
_expected_manifest_sha = "c597e64c78e5faf400df9b15b29dc6cccfc0b7dc9afc010fd520a5263d1eedd6"
if hashlib.sha256(_head_manifest_path.read_bytes()).hexdigest() != _expected_manifest_sha:
    raise RuntimeError("exp043 head manifest SHA mismatch")
_head_manifest = json.loads(_head_manifest_path.read_text())
if _head_manifest["checkpoint_sha256"] != _SELF_HEAD_SHA:
    raise RuntimeError("head and manifest SHA mismatch")
_training_stems = _head_manifest["training_videos"]
_eligible_stems = sorted(
    path.stem for path in TRAIN_DIR.glob("*.zarr") if (TRAIN_DIR / f"{path.stem}.geff").is_dir()
)
_selection = select_videos(_eligible_stems, _training_stems, rank_start=_EXP052_RANK_START)
test_stems = sorted(stem for stems in _selection.values() for stem in stems)
if len(test_stems) != 20:
    raise RuntimeError("expected 20 head-unseen videos")
_selection_dir = WORKING_DIR / "exp052_selected_train"
_selection_dir.mkdir(exist_ok=True)
for _stem in test_stems:
    _source = TRAIN_DIR / f"{_stem}.zarr"
    _target = _selection_dir / _source.name
    if _target.exists() or _target.is_symlink():
        if _target.resolve() != _source.resolve():
            raise RuntimeError(f"selected input link changed: {_target}")
    else:
        _target.symlink_to(_source, target_is_directory=True)
TEST_DIR = _selection_dir
_selection_receipt = {
    "trained": sorted(_training_stems),
    "selected": _selection,
    "eligible_count": len(_eligible_stems),
    "seed": 42,
    "head_sha256": _SELF_HEAD_SHA,
    "head_manifest_sha256": _expected_manifest_sha,
}
(WORKING_DIR / "exp052_selection.json").write_text(
    json.dumps(_selection_receipt, indent=2, sort_keys=True) + "\n"
)
print("Selected head-unseen videos:", _selection, flush=True)

print(f"Found {len(test_stems)} test videos")
print(test_stems[:10])

splits_path = REPO_DIR / "kaggle_test_splits_50ep.json"
splits_path.parent.mkdir(parents=True, exist_ok=True)
splits_path.write_text(json.dumps([{"split": 0, "train": [], "test": test_stems}], indent=2))

predict_cmd = [
    sys.executable,
    "scripts/predict_unet_transformer.py",
    "--data-dir",
    str(TEST_DIR),
    "--splits",
    str(splits_path.name),
    "--split",
    "0",
    "--weights",
    WEIGHTS_RELATIVE,
    "--unet-batch-size",
    str(UNET_BATCH_SIZE),
    "--det-threshold",
    str(DET_THRESHOLD),
    "--ilp-edge-weight",
    str(ILP_EDGE_WEIGHT),
    "--ilp-appearance-weight",
    str(ILP_APPEARANCE_WEIGHT),
    "--ilp-disappearance-weight",
    str(ILP_DISAPPEARANCE_WEIGHT),
    "--ilp-division-weight",
    str(ILP_DIVISION_WEIGHT),
]
if USE_ILP:
    predict_cmd.append("--use-ilp")
if SLICE:
    predict_cmd.extend(["--slice", SLICE])


def _visible_cuda_tokens(count: int) -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw and raw != "-1":
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        if len(tokens) < count:
            raise RuntimeError(
                f"torch reports {count} CUDA devices but CUDA_VISIBLE_DEVICES={raw!r}"
            )
        return tokens[:count]
    return [str(index) for index in range(count)]


def _prediction_dir_for_method(method: str) -> Path:
    matches = sorted((REPO_DIR / "predictions").glob(f"*/{method}/split_0"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one prediction directory for {method!r}, found {matches}"
        )
    return matches[0]


def _wait_for_prediction_shards(
    processes: dict[int, subprocess.Popen],
    commands: dict[int, list[str]],
) -> None:
    while processes:
        failed: tuple[int, int] | None = None
        for shard_index, process in list(processes.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            del processes[shard_index]
            if return_code != 0:
                failed = (shard_index, return_code)
                break
        if failed is None:
            if processes:
                time.sleep(1.0)
            continue

        failed_index, failed_code = failed
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise subprocess.CalledProcessError(failed_code, commands[failed_index])


def _merge_prediction_shards(worker_count: int) -> Path:
    shard_dirs: list[Path] = []
    seen: set[str] = set()
    expected_all = set(test_stems)

    for shard_index in range(worker_count):
        shard_method = f"{METHOD}_gpu{shard_index}"
        shard_dir = _prediction_dir_for_method(shard_method)
        expected = set(test_stems[shard_index::worker_count])
        shard_paths = sorted(shard_dir.glob("*.geff"))
        found = {path.stem for path in shard_paths}
        if found != expected:
            raise RuntimeError(
                f"GPU shard {shard_index} output mismatch: "
                f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
            )
        overlap = seen & found
        if overlap:
            raise RuntimeError(f"Duplicate datasets across GPU shards: {sorted(overlap)}")
        seen.update(found)
        shard_dirs.append(shard_dir)

    if seen != expected_all:
        raise RuntimeError(
            f"Merged GPU shards do not cover the test set: "
            f"missing={sorted(expected_all - seen)}, extra={sorted(seen - expected_all)}"
        )

    username_roots = {shard_dir.parents[1] for shard_dir in shard_dirs}
    if len(username_roots) != 1:
        raise RuntimeError(f"GPU shards used inconsistent prediction roots: {username_roots}")
    import shutil as _shutil

    final_root = next(iter(username_roots)) / METHOD
    final_dir = final_root / "split_0"
    staging_dir = final_root / "split_0_dual_gpu_staging"
    if staging_dir.exists():
        if staging_dir.is_dir():
            _shutil.rmtree(staging_dir)
        else:
            staging_dir.unlink()
    staging_dir.mkdir(parents=True, exist_ok=False)

    for shard_dir in shard_dirs:
        for source in sorted(shard_dir.glob("*.geff")):
            destination = staging_dir / source.name
            if destination.exists():
                raise RuntimeError(f"Refusing to overwrite duplicate merged output: {destination}")
            _shutil.move(str(source), str(destination))

    merged = {path.stem for path in staging_dir.glob("*.geff")}
    if merged != expected_all:
        raise RuntimeError(
            f"Staged prediction directory failed verification: "
            f"missing={sorted(expected_all - merged)}, extra={sorted(merged - expected_all)}"
        )

    if final_dir.exists():
        if final_dir.is_dir():
            _shutil.rmtree(final_dir)
        else:
            final_dir.unlink()
    staging_dir.rename(final_dir)
    for shard_dir in shard_dirs:
        _shutil.rmtree(shard_dir.parent)
    print(f"Merged {len(merged)} prediction graphs into {final_dir}")
    return final_dir


# Dump the detection registry and every sub-threshold detection peak next to
# the prediction graphs (agent/frontier947_gapfill): the candidate-cache
# replacements agent/cvcache2 applies, then the low-detection replacements on
# top. Non-fatal: without the dump the gap filler is idle and the output is
# the flow2 kernel's.
_cache_dir_env = os.environ.get("BIOHUB_CACHE_DIR", "").strip()
if _cache_dir_env:
    try:
        _cs = _ps.read_text()
        for _label, _old, _new in [
            (
                "predict_video globals",
                "@torch.no_grad()\ndef predict_video(",
                "_CACHE_EDGES: list = []\n_CACHE_THRESHOLD = float(os.environ.get('BIOHUB_CACHE_EDGE_THRESHOLD', '0') or 0)\n_CACHE_DIR = os.environ.get('BIOHUB_CACHE_DIR', '').strip()\n\n\n@torch.no_grad()\ndef predict_video(",
            ),
            (
                "candidate dump",
                "            candidates = sorted(",
                "            if _CACHE_THRESHOLD > 0:\n                _ci, _cj = np.nonzero(probs > _CACHE_THRESHOLD)\n                if _ci.size:\n                    _CACHE_EDGES.append((\n                        np.asarray(idx_src)[_ci].astype(np.int32),\n                        np.asarray(idx_tgt)[_cj].astype(np.int32),\n                        probs[_ci, _cj].astype(np.float32),\n                    ))\n            candidates = sorted(",
            ),
            (
                "cache write",
                "        graph = build_graph(coords, edges)",
                "        if _CACHE_DIR:\n            import pathlib\n            _cd = pathlib.Path(_CACHE_DIR)\n            _cd.mkdir(parents=True, exist_ok=True)\n            if _CACHE_EDGES:\n                _es = np.concatenate([e[0] for e in _CACHE_EDGES])\n                _et = np.concatenate([e[1] for e in _CACHE_EDGES])\n                _ep = np.concatenate([e[2] for e in _CACHE_EDGES])\n            else:\n                _es = np.empty(0, np.int32); _et = np.empty(0, np.int32)\n                _ep = np.empty(0, np.float32)\n            np.savez_compressed(\n                _cd / f'{name}.npz', coords=coords,\n                edge_src=_es, edge_tgt=_et, edge_prob=_ep,\n                admitted=np.asarray(edges, dtype=np.float64),\n            )\n            print(f'CACHE {name}: {len(coords)} nodes, {_es.size} edges', flush=True)\n            _CACHE_EDGES.clear()\n        graph = build_graph(coords, edges)",
            ),
            (
                "lowdet globals",
                "_CACHE_DIR = os.environ.get('BIOHUB_CACHE_DIR', '').strip()\n",
                "_CACHE_DIR = os.environ.get('BIOHUB_CACHE_DIR', '').strip()\n_LOWDET_THRESHOLD = float(os.environ.get('BIOHUB_LOWDET_THRESHOLD', '0') or 0)\n_LOWDET: list = []\n",
            ),
            (
                "lowdet peaks",
                "                arr = _detect_cells_pooled(\n                    det_logits[f_idx][0], t, cfg.det_threshold, pool_k,\n                )\n                coord_offset[t] = (global_node_count, global_node_count + len(arr))\n",
                "                arr = _detect_cells_pooled(\n                    det_logits[f_idx][0], t, cfg.det_threshold, pool_k,\n                )\n                if _LOWDET_THRESHOLD > 0:\n                    _low = _detect_cells_pooled(det_logits[f_idx][0], t, _LOWDET_THRESHOLD, pool_k)\n                    if len(_low):\n                        _lg = det_logits[f_idx][0][0]\n                        _lz = torch.as_tensor(_low[:, 1:].astype(np.int64), device=_lg.device)\n                        _lsc = torch.sigmoid(_lg[_lz[:, 0], _lz[:, 1], _lz[:, 2]]).float().cpu().numpy()\n                        _LOWDET.append((_low.astype(np.float32), _lsc.astype(np.float32)))\n                coord_offset[t] = (global_node_count, global_node_count + len(arr))\n",
            ),
            (
                "lowdet write",
                "            np.savez_compressed(\n                _cd / f'{name}.npz', coords=coords,\n",
                "            if _LOWDET:\n                _lc = np.concatenate([e[0] for e in _LOWDET]).astype(np.float32)\n                _lc[:, 1:] *= np.array(downsample, dtype=np.float32)\n                _lc = _lc.astype(np.int16)\n                _lsc = np.concatenate([e[1] for e in _LOWDET]).astype(np.float32)\n            else:\n                _lc = np.empty((0, 4), np.int16)\n                _lsc = np.empty(0, np.float32)\n            _LOWDET.clear()\n            print(f'LOWDET {name}: {len(_lc)} peaks above {_LOWDET_THRESHOLD}', flush=True)\n            np.savez_compressed(\n                _cd / f'{name}.npz', coords=coords, low_coords=_lc, low_score=_lsc,\n",
            ),
        ]:
            if _cs.count(_old) != 1:
                raise RuntimeError(f"{_label}: anchor count={_cs.count(_old)}")
            _cs = _cs.replace(_old, _new, 1)
        compile(_cs, str(_ps), "exec")
        _ps.write_text(_cs)
        print(
            "Low-detection dump applied | dir=",
            _cache_dir_env,
            "| lowdet threshold=",
            os.environ.get("BIOHUB_LOWDET_THRESHOLD"),
        )
    except Exception as _dump_exc:
        print("LOW-DETECTION DUMP SKIPPED (non-fatal):", type(_dump_exc).__name__, _dump_exc)
else:
    print("Low-detection dump disabled (BIOHUB_CACHE_DIR unset)")


# --------------------------------- V1284 head, AFTER readmit's low-detection dump patch
# x122 v1 FAILED SILENTLY: my V1284 patch inserts a line before
#   coord_offset[t] = (global_node_count, global_node_count + len(arr))
# and readmit's 'lowdet peaks' patch ANCHORS ON THAT EXACT LINE. Mine ran first, so readmit's
# anchor count went to 0, its try/except swallowed it as non-fatal, the low-detection dump was
# never written, and every movie logged 'gap filler idle'. Confirmed in the log:
#   LOW-DETECTION DUMP SKIPPED (non-fatal): RuntimeError lowdet peaks: anchor count=0
# Two patches, one anchor. Ordering mine AFTER theirs gives each a pristine anchor; readmit's
# replacement text re-emits the coord_offset line, so my anchor still resolves to exactly 1.
(_ps.parent / "v1284_coordinate_refinement.py").write_text(
    "\"\"\"Frozen-feature coordinate regression at first-seen fused detector centers.\"\"\"\nimport os\nfrom pathlib import Path\nimport numpy as np\nimport torch\n\nSPACING = np.array([1.625, 1.625, 1.625], dtype=np.float32)\nOFFSETS = ((0,0,0), (-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1))\n_CACHE = None\n\n\ndef make_head():\n    head = torch.nn.Sequential(torch.nn.Linear(224, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))\n    torch.nn.init.zeros_(head[-1].weight)\n    torch.nn.init.zeros_(head[-1].bias)\n    return head\n\n\ndef bounded(head, x):\n    delta = head(x)\n    return 2.0 * delta / (1.0 + torch.linalg.vector_norm(delta, dim=-1, keepdim=True))\n\n\ndef sample_features(feature, arr):\n    xyz = torch.as_tensor(arr[:, 1:], device=feature.device, dtype=torch.long)\n    blocks = []\n    for offset in OFFSETS:\n        loc = xyz + torch.tensor(offset, device=feature.device)\n        for axis, size in enumerate(feature.shape[-3:]):\n            loc[:, axis].clamp_(0, size-1)\n        blocks.append(feature[0, :, loc[:,0], loc[:,1], loc[:,2]].T)\n    # Directional differences plus the central representation.\n    return torch.cat([blocks[0]] + [b - blocks[0] for b in blocks[1:]], dim=1)\n\n\ndef index_features(self, maps, coords, mask):\n    \"\"\"Trilinear lookup; integer coordinates reproduce native gather exactly.\"\"\"\n    out = torch.zeros((*coords.shape[:2], maps.shape[1]), device=maps.device, dtype=maps.dtype)\n    for batch in range(len(maps)):\n        n = int(mask[batch].sum())\n        if not n:\n            continue\n        q = coords[batch, :n].clone()\n        for axis, size in enumerate(maps.shape[-3:]):\n            q[:,axis].clamp_(0, size-1)\n        low = q.floor().long()\n        frac = q-low\n        for z in (0,1):\n            for y in (0,1):\n                for x in (0,1):\n                    shift = torch.tensor([z,y,x], device=maps.device)\n                    loc = low+shift\n                    for axis, size in enumerate(maps.shape[-3:]):\n                        loc[:,axis].clamp_(0,size-1)\n                    weight = torch.where(shift.bool(), frac, 1-frac).prod(dim=1)\n                    out[batch,:n] += maps[batch,:,loc[:,0],loc[:,1],loc[:,2]].T * weight[:,None]\n    return out\n\n\ndef refine(ds_path, t, arr, feature):\n    global _CACHE\n    mode = os.environ['V1284_MODE']\n    if not len(arr):\n        return arr\n    if mode == 'zero':\n        return arr.astype(np.float32)\n    x = sample_features(feature, arr).float()\n    if mode == 'capture':\n        folder = Path(os.environ['V1284_CAPTURE']) / ds_path.stem\n        folder.mkdir(parents=True, exist_ok=True)\n        np.savez_compressed(folder/f'{int(t):04d}.npz', coords=arr, features=x.cpu().numpy())\n        return arr\n    if _CACHE is None:\n        saved = torch.load(os.environ['V1284_HEAD'], map_location='cpu', weights_only=True)\n        head = make_head().to(feature.device)\n        head.load_state_dict(saved['state_dict']); head.eval()\n        _CACHE = (head, saved['mean'].to(feature.device), saved['scale'].to(feature.device))\n    head, mean, scale = _CACHE\n    shift = bounded(head, (x-mean)/scale).cpu().numpy() / SPACING\n    result = arr.astype(np.float32).copy()\n    result[:,1:] += shift\n    result[:,1:] = np.clip(result[:,1:], 0, np.asarray(feature.shape[-3:])-1)\n    if not np.isfinite(result).all() or np.max(np.linalg.norm((result[:,1:]-arr[:,1:])*SPACING,axis=1)) > 2.00001:\n        raise RuntimeError('invalid V1284 displacement')\n    return result\n"
)
os.environ["V1284_MODE"] = "candidate"
os.environ["V1284_HEAD"] = str(_SELF_HEAD_PATH)

_trial_source = _ps.read_text()
if _trial_source.count("import tracksdata as td\n") != 1:
    raise RuntimeError("V1284 patch anchor mismatch")
_trial_source = _trial_source.replace(
    "import tracksdata as td\n",
    "import tracksdata as td\nfrom v1284_coordinate_refinement import refine as _v1284_refine, index_features as _v1284_index\n",
)
if (
    _trial_source.count(
        "                coord_offset[t] = (global_node_count, global_node_count + len(arr))"
    )
    != 1
):
    raise RuntimeError("V1284 patch anchor mismatch")
_trial_source = _trial_source.replace(
    "                coord_offset[t] = (global_node_count, global_node_count + len(arr))",
    "                arr = _v1284_refine(ds_path, t, arr, unet_out[:, f_idx])\n                coord_offset[t] = (global_node_count, global_node_count + len(arr))",
)
if _trial_source.count("    coords = coords.astype(np.int16)\n") != 1:
    raise RuntimeError("V1284 patch anchor mismatch")
_trial_source = _trial_source.replace(
    "    coords = coords.astype(np.int16)\n",
    "    # Preserve refined geometry through association and graph output.\n",
)
if (
    _trial_source.count("    model, window_size, downsample = load_model(weights_path, device)")
    != 1
):
    raise RuntimeError("V1284 patch anchor mismatch")
_trial_source = _trial_source.replace(
    "    model, window_size, downsample = load_model(weights_path, device)",
    "    model, window_size, downsample = load_model(weights_path, device)\n    UNetNodeTransformer._index_features = _v1284_index",
)

_ps.write_text(_trial_source)
print("V1284 head patched AFTER the readmit dump patch; mode =", os.environ["V1284_MODE"])


# Capture the detector's uncorrected candidate order before either arm applies
# a coordinate offset. The public tracker and detector code are otherwise fixed.
_audit_mod_path = _ps.parent / "v1284_coordinate_refinement.py"
_audit_mod = _audit_mod_path.read_text()
_audit_refine_anchor = "    if mode == 'zero':\n"
_audit_refine_capture = (
    "    raw_root = os.environ.get('BIOHUB_AUDIT_RAW_DIR', '').strip()\n"
    "    if raw_root:\n"
    "        raw_dir = Path(raw_root) / ds_path.stem\n"
    "        raw_dir.mkdir(parents=True, exist_ok=True)\n"
    "        np.savez_compressed(raw_dir / f'{int(t):04d}.npz', coords=np.asarray(arr))\n"
)
if _audit_mod.count(_audit_refine_anchor) != 1:
    raise RuntimeError("raw coordinate capture anchor changed")
_audit_mod = _audit_mod.replace(
    _audit_refine_anchor, _audit_refine_capture + _audit_refine_anchor, 1
)
_audit_feature_anchor = "    return out\n\n\ndef refine"
_audit_feature_capture = (
    "    log_path = os.environ.get('BIOHUB_AUDIT_FEATURE_LOG', '').strip()\n"
    "    if log_path:\n"
    "        import hashlib, json\n"
    "        values = np.ascontiguousarray(out.detach().float().cpu().numpy())\n"
    "        with Path(log_path).open('a') as handle:\n"
    "            handle.write(json.dumps({'shape': list(values.shape), 'sha256': hashlib.sha256(values.tobytes()).hexdigest()}) + '\\n')\n"
    "    return out\n\n\ndef refine"
)
if _audit_mod.count(_audit_feature_anchor) != 1:
    raise RuntimeError("feature capture anchor changed")
_audit_mod = _audit_mod.replace(_audit_feature_anchor, _audit_feature_capture, 1)
compile(_audit_mod, str(_audit_mod_path), "exec")
_audit_mod_path.write_text(_audit_mod)
if "_CACHE_EDGES" not in _ps.read_text() or "_LOWDET" not in _ps.read_text():
    raise RuntimeError("required candidate cache patch did not apply")

# %% [markdown]
# ## 4. Original graph postprocessing and output checks

# %%
import tracksdata as td
import numpy as np
import blosc2
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

SUBMISSION_COLUMNS = [
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]
CSV_COLUMNS = ["id", *SUBMISSION_COLUMNS]
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)

import time as _time
import traceback as _traceback

# Runtime guard. Kaggle reruns this notebook on a hidden test set larger than
# the visible one, so the repair loop watches the wall clock: once the kernel
# has run for REPAIR_DEADLINE_S the remaining datasets are processed with the
# optional repair stages switched off, and a dataset whose repair raises is
# written from its ILP graph instead of failing the whole kernel.
KERNEL_START_TS = float(os.environ.get("BIOHUB_KERNEL_START_TS", str(_time.time())))
REPAIR_DEADLINE_S = float(os.environ.get("BIOHUB_REPAIR_DEADLINE_S", "27000"))
FRAME_CACHE_MAX_FRAMES = int(os.environ.get("BIOHUB_FRAME_CACHE_MAX_FRAMES", "48"))
_deadline_degraded = False


def _frame_cache_trim(frame_cache: dict[int, np.ndarray]) -> None:
    while len(frame_cache) > max(1, FRAME_CACHE_MAX_FRAMES):
        frame_cache.pop(next(iter(frame_cache)))


def _deadline_degrade() -> None:
    global _deadline_degraded, OUTPUT_MOTION_RELINK, OUTPUT_GAP_CLOSE, OUTPUT_GAP2_RECOVERY
    global OUTPUT_SAFE_DIVISIONS, OUTPUT_LINEFIT_SMOOTH
    _deadline_degraded = True
    OUTPUT_MOTION_RELINK = False
    OUTPUT_GAP_CLOSE = False
    OUTPUT_GAP2_RECOVERY = False
    OUTPUT_SAFE_DIVISIONS = False
    OUTPUT_LINEFIT_SMOOTH = False
    print(
        f"DEADLINE: {_time.time() - KERNEL_START_TS:.0f}s since kernel start exceeds"
        f" {REPAIR_DEADLINE_S:.0f}s; remaining datasets get edge filtering and"
        " short-track filtering only",
        flush=True,
    )


def fallback_output_graph(
    nodes_by_id: dict[int, dict[str, object]],
    raw_edges: list[dict[str, object]],
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]], dict[str, int]]:
    """Cheapest valid graph from the ILP output: consecutive-frame edges within
    OUTPUT_EDGE_MAX_UM, one parent per node, at most two children per node."""
    stats: dict[str, int] = {"raw_edges": len(raw_edges), "repair_fallback": 1}
    edges: list[dict[str, object]] = []
    for edge in raw_edges:
        source = nodes_by_id.get(int(edge["source_id"]))
        target = nodes_by_id.get(int(edge["target_id"]))
        if source is None or target is None:
            continue
        if int(target["t"]) != int(source["t"]) + 1:
            continue
        edge["distance_um"] = edge_distance_um(source, target)
        if OUTPUT_EDGE_MAX_UM > 0 and float(edge["distance_um"]) > OUTPUT_EDGE_MAX_UM:
            continue
        edges.append(edge)
    best_by_target: dict[int, dict[str, object]] = {}
    for edge in edges:
        target_id = int(edge["target_id"])
        prev = best_by_target.get(target_id)
        if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
            best_by_target[target_id] = edge
    by_source: dict[int, list[dict[str, object]]] = {}
    for edge in best_by_target.values():
        by_source.setdefault(int(edge["source_id"]), []).append(edge)
    edges = []
    for source_edges in by_source.values():
        edges.extend(sorted(source_edges, key=edge_sort_key, reverse=True)[:2])
    incident = {int(e["source_id"]) for e in edges} | {int(e["target_id"]) for e in edges}
    kept = {node_id: node for node_id, node in nodes_by_id.items() if node_id in incident}
    return (kept or nodes_by_id), edges, stats


def graph_from_geff(path: Path):
    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def edge_distance_um(source: dict[str, object], target: dict[str, object]) -> float:
    dz = (float(source["z"]) - float(target["z"])) * VOXEL_SCALE_UM[0]
    dy = (float(source["y"]) - float(target["y"])) * VOXEL_SCALE_UM[1]
    dx = (float(source["x"]) - float(target["x"])) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def point_distance_um(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dz = (a[0] - b[0]) * VOXEL_SCALE_UM[0]
    dy = (a[1] - b[1]) * VOXEL_SCALE_UM[1]
    dx = (a[2] - b[2]) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def node_point(node: dict[str, object]) -> tuple[float, float, float]:
    return (float(node["z"]), float(node["y"]), float(node["x"]))


def edge_sort_key(edge: dict[str, object]) -> tuple[float, float]:
    prob = edge.get("edge_prob")
    prob_value = float(prob) if prob is not None else 0.0
    return prob_value, -float(edge["distance_um"])


def _next_node_id(nodes_by_id: dict[int, dict[str, object]]) -> int:
    return max(nodes_by_id) + 1 if nodes_by_id else 1


def read_test_frame(dataset: str, t: int, frame_cache: dict[int, np.ndarray]) -> np.ndarray:
    if t in frame_cache:
        return frame_cache[t]
    zarr_path = TEST_DIR / f"{dataset}.zarr"
    meta = json.loads((zarr_path / "0" / "zarr.json").read_text())
    shape = tuple(int(v) for v in meta["shape"])
    dtype = np.dtype(meta["data_type"])
    frame_shape = shape[1:]
    chunk_path = zarr_path / "0" / "c" / str(t) / "0" / "0" / "0"
    try:
        raw = chunk_path.read_bytes()
        arr = np.frombuffer(blosc2.decompress(raw), dtype=dtype)
        if arr.size == int(np.prod(frame_shape)):
            frame = arr.reshape(frame_shape).copy()
            frame_cache[t] = frame
            _frame_cache_trim(frame_cache)
            return frame
    except Exception:
        pass
    import zarr

    frame = np.asarray(zarr.open(zarr_path / "0", mode="r")[t])
    frame_cache[t] = frame
    _frame_cache_trim(frame_cache)
    return frame


def refine_synthetic_midpoint(
    dataset: str | None,
    t: int,
    midpoint: tuple[float, float, float],
    frame_cache: dict[int, np.ndarray],
    stats: dict[str, int],
) -> tuple[float, float, float]:
    if not GAP_REFINE_SYNTHETIC or dataset is None:
        return midpoint
    try:
        frame = read_test_frame(dataset, t, frame_cache)
        z, y, x = [int(round(v)) for v in midpoint]
        z0 = max(0, z - GAP_REFINE_WIN_Z)
        z1 = min(frame.shape[0], z + GAP_REFINE_WIN_Z + 1)
        y0 = max(0, y - GAP_REFINE_WIN_YX)
        y1 = min(frame.shape[1], y + GAP_REFINE_WIN_YX + 1)
        x0 = max(0, x - GAP_REFINE_WIN_YX)
        x1 = min(frame.shape[2], x + GAP_REFINE_WIN_YX + 1)
        patch = frame[z0:z1, y0:y1, x0:x1].astype(np.float64)
        if patch.size == 0:
            stats["gap_refine_failed"] += 1
            return midpoint
        baseline = float(np.percentile(patch, 20.0))
        weights = np.maximum(patch - baseline, 0.0)
        total = float(weights.sum())
        if total <= 0:
            stats["gap_refine_failed"] += 1
            return midpoint
        zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
        yy = np.arange(y0, y1, dtype=np.float64)[None, :, None]
        xx = np.arange(x0, x1, dtype=np.float64)[None, None, :]
        refined = (
            float((weights * zz).sum() / total),
            float((weights * yy).sum() / total),
            float((weights * xx).sum() / total),
        )
        if point_distance_um(refined, midpoint) > GAP_REFINE_MAX_SHIFT_UM:
            stats["gap_refine_rejected_shift"] += 1
            return midpoint
        stats["gap_refined_synthetic"] += 1
        return refined
    except Exception:
        stats["gap_refine_failed"] += 1
        return midpoint


def _dc_pool_frame_xy(volume: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return volume.astype(np.float32, copy=False)
    z, y, x = volume.shape
    y2 = (y // factor) * factor
    x2 = (x // factor) * factor
    cropped = volume[:, :y2, :x2].astype(np.float32, copy=False)
    return cropped.reshape(z, y2 // factor, factor, x2 // factor, factor).mean(axis=(2, 4))


def _dc_normalize_dynamic_range(volume: np.ndarray, cfg: object) -> np.ndarray:
    vol = np.asarray(volume, dtype=np.float32)
    lo = float(np.percentile(vol, float(getattr(cfg, "norm_lo_pct", 50.0))))
    hi = float(np.percentile(vol, float(getattr(cfg, "norm_hi_pct", 99.5))))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    ratio = (vol - lo) / (hi - lo)
    return np.clip(
        ratio,
        float(getattr(cfg, "norm_clip_lo", -0.5)),
        float(getattr(cfg, "norm_clip_hi", 6.0)),
    ).astype(np.float32)


def _dc_manifest_weight_paths(manifest_path: Path) -> list[Path]:
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        print("Could not read DeepCenter manifest:", manifest_path, type(exc).__name__, exc)
        return []
    root = manifest_path.parent
    sections: list[dict[str, object]] = []
    for section in [
        manifest.get("model", {}),
        manifest.get("models", {}).get("full_frame_center", {})
        if isinstance(manifest.get("models", {}), dict)
        else {},
        manifest.get("full_frame_center", {}),
    ]:
        if isinstance(section, dict):
            sections.append(section)
    candidates: list[Path] = []
    for section in sections:
        for key in ("weight_path", "path"):
            rel = section.get(key)
            if isinstance(rel, str) and rel:
                candidates.append(root / rel)
        for key in ("last_checkpoint", "best_checkpoint"):
            item = section.get(key)
            if isinstance(item, dict):
                rel = item.get("path")
                if isinstance(rel, str) and rel:
                    candidates.append(root / rel)
    for name in ("checkpoint_last.pt", "best.pt", "last.pt"):
        candidates.append(root / "weights" / "full_frame_center" / name)
        candidates.append(root / name)
    candidates.append(root / DEEPCENTER_RELATIVE)
    return candidates


def _dc_checkpoint_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get("BIOHUB_DEEPCENTER_CHECKPOINT", DEEPCENTER_CHECKPOINT_DEFAULT).strip()
    if explicit:
        candidates.append(Path(explicit))
    manifest_explicit = os.environ.get(
        "BIOHUB_DEEPCENTER_MANIFEST", DEEPCENTER_MANIFEST_DEFAULT
    ).strip()
    if manifest_explicit:
        candidates.extend(_dc_manifest_weight_paths(Path(manifest_explicit)))

    input_root = Path("/kaggle/input")
    preferred_dirs = [
        Path("/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1"),
        Path("/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1"),
    ]
    for directory in preferred_dirs:
        candidates.extend(_dc_manifest_weight_paths(directory / "ARTIFACT_MANIFEST.json"))
        for name in ("checkpoint_last.pt", "best.pt", "last.pt"):
            candidates.append(directory / "weights" / "full_frame_center" / name)
            candidates.append(directory / name)
    if input_root.exists() and not any(path.is_file() for path in candidates):
        # Three recursive walks of /kaggle/input cost ~370 s on the visible
        # run. The explicit checkpoint path is checksum-pinned by the setup
        # cell, so only walk when none of the known paths exist.
        for name in ("checkpoint_last.pt", "best.pt", "last.pt"):
            candidates.extend(sorted(input_root.glob(f"**/full_frame_center/**/{name}")))

    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        path = path.expanduser()
        try:
            key = path.resolve() if path.exists() else path
        except Exception:
            key = path
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


try:
    import torch
except Exception as _dc_torch_error:
    torch = None


if torch is not None:

    class _DCConvBlock3d(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            groups = min(8, out_channels)
            self.block = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
                torch.nn.GroupNorm(groups, out_channels),
                torch.nn.SiLU(inplace=True),
                torch.nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
                torch.nn.GroupNorm(groups, out_channels),
                torch.nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class _DCDeepCenterUNet3D(torch.nn.Module):
        def __init__(self, in_channels: int = 1, base_channels: int = 24) -> None:
            super().__init__()
            c = int(base_channels)
            self.enc1 = _DCConvBlock3d(in_channels, c)
            self.down1 = torch.nn.MaxPool3d(2, 2)
            self.enc2 = _DCConvBlock3d(c, c * 2)
            self.down2 = torch.nn.MaxPool3d(2, 2)
            self.enc3 = _DCConvBlock3d(c * 2, c * 4)
            self.down3 = torch.nn.MaxPool3d(2, 2)
            self.bottleneck = _DCConvBlock3d(c * 4, c * 8)
            self.up3 = torch.nn.ConvTranspose3d(c * 8, c * 4, 2, 2)
            self.dec3 = _DCConvBlock3d(c * 8, c * 4)
            self.up2 = torch.nn.ConvTranspose3d(c * 4, c * 2, 2, 2)
            self.dec2 = _DCConvBlock3d(c * 4, c * 2)
            self.up1 = torch.nn.ConvTranspose3d(c * 2, c, 2, 2)
            self.dec1 = _DCConvBlock3d(c * 2, c)
            self.head = torch.nn.Conv3d(c, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.down1(e1))
            e3 = self.enc3(self.down2(e2))
            b = self.bottleneck(self.down3(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)
else:
    _DCConvBlock3d = None
    _DCDeepCenterUNet3D = None


def load_deepcenter_veto_detector() -> dict[str, object] | None:
    if not USE_DEEPCENTER_VETO:
        print("DeepCenter add-only repair gate disabled by configuration.")
        return None
    if torch is None:
        if REQUIRE_DEEPCENTER_VETO:
            raise ImportError("torch is required for DeepCenter add-only repair gate")
        print("DeepCenter add-only repair gate skipped because torch is unavailable.")
        return None
    from types import SimpleNamespace

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_errors: list[str] = []
    for checkpoint_path in _dc_checkpoint_candidates():
        if not checkpoint_path.exists():
            continue
        try:
            print("Trying DeepCenter add-only gate checkpoint:", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
                raise ValueError("checkpoint has no model_state")
            checkpoint_epoch = int(checkpoint.get("epoch", -1))
            if DEEPCENTER_EXPECTED_EPOCH > 0 and checkpoint_epoch != DEEPCENTER_EXPECTED_EPOCH:
                raise ValueError(
                    f"expected DeepCenter epoch {DEEPCENTER_EXPECTED_EPOCH}, got {checkpoint_epoch}"
                )
            cfg = SimpleNamespace(**checkpoint.get("config", {}))
            model = _DCDeepCenterUNet3D(base_channels=int(getattr(cfg, "base_channels", 24)))
            model.load_state_dict(checkpoint["model_state"])
            model.to(device)
            model.eval()
            print("Loaded DeepCenter add-only gate checkpoint:", checkpoint_path)
            print(
                "DeepCenter checkpoint epoch:",
                checkpoint.get("epoch"),
                "best_score:",
                checkpoint.get("best_score"),
            )
            return {
                "model": model,
                "cfg": cfg,
                "device": device,
                "path": checkpoint_path,
                "torch": torch,
            }
        except Exception as exc:
            load_errors.append(f"{checkpoint_path}: {type(exc).__name__}: {exc}")
            print(
                "Skipping incompatible DeepCenter checkpoint:",
                checkpoint_path,
                "|",
                type(exc).__name__,
                exc,
            )
    message = "No usable DeepCenter checkpoint found for add-only repair gate."
    if REQUIRE_DEEPCENTER_VETO:
        checked = "\n".join(str(p) for p in _dc_checkpoint_candidates()[:80])
        errors = "\n".join(load_errors[-20:])
        raise FileNotFoundError(
            message + "\nChecked:\n" + checked + ("\nLoad errors:\n" + errors if errors else "")
        )
    print(message)
    return None


def _dc_cache_trim(cache: dict[tuple[str, int], np.ndarray]) -> None:
    limit = max(1, int(DEEPCENTER_SCORE_CACHE_MAX_FRAMES))
    while len(cache) > limit:
        cache.pop(next(iter(cache)))


def deepcenter_heatmap_for_frame(
    dataset: str,
    t: int,
    detector_bundle: dict[str, object] | None,
    frame_cache: dict[int, np.ndarray],
    heatmap_cache: dict[tuple[str, int], np.ndarray],
) -> np.ndarray | None:
    if detector_bundle is None:
        return None
    key = (dataset, int(t))
    cached = heatmap_cache.get(key)
    if cached is not None:
        return cached
    model = detector_bundle["model"]
    cfg = detector_bundle["cfg"]
    device = detector_bundle["device"]
    torch_mod = detector_bundle["torch"]
    pool_factor = int(getattr(cfg, "pool_factor", 4))
    volume = read_test_frame(dataset, int(t), frame_cache)
    pooled = _dc_pool_frame_xy(volume, pool_factor)
    image = _dc_normalize_dynamic_range(pooled, cfg)
    with torch_mod.no_grad():
        tensor = torch_mod.from_numpy(image[None, None, ...]).to(
            device=device, dtype=torch_mod.float32
        )
        logits = model(tensor)

        if os.environ.get("BIOHUB_DEEPCENTER_TTA", "0") != "0":
            acc = logits.clone()
            nv = 1
            for dims in [(-1,), (-2,), (-2, -1)]:
                acc = acc + model(tensor.flip(dims)).flip(dims)
                nv += 1
            if tensor.shape[-1] == tensor.shape[-2]:
                for k in (1, 3):
                    acc = acc + torch_mod.rot90(
                        model(torch_mod.rot90(tensor, k, dims=(-2, -1))), -k, dims=(-2, -1)
                    )
                    nv += 1
                acc = acc + model(tensor.transpose(-1, -2)).transpose(-1, -2)
                nv += 1
                at = torch_mod.rot90(tensor, 1, dims=(-2, -1)).transpose(-1, -2)
                acc = acc + torch_mod.rot90(model(at).transpose(-1, -2), -1, dims=(-2, -1))
                nv += 1
            delta = float((acc / nv - logits).abs().mean())
            if delta == 0.0:
                raise RuntimeError(
                    "DEEPCENTER_TTA_NO_OP: averaged veto logits identical to the single view"
                )
            if not getattr(deepcenter_heatmap_for_frame, "_tta_announced", False):
                print(
                    "DEEPCENTER_TTA_ACTIVE views=",
                    nv,
                    "mean_abs_logit_delta=",
                    round(delta, 6),
                    flush=True,
                )
                deepcenter_heatmap_for_frame._tta_announced = True
            logits = acc / nv
        heatmap = (
            torch_mod.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        )
    heatmap_cache[key] = heatmap
    _dc_cache_trim(heatmap_cache)
    return heatmap


def deepcenter_score_point(
    dataset: str | None,
    t: int,
    point: tuple[float, float, float],
    detector_bundle: dict[str, object] | None,
    frame_cache: dict[int, np.ndarray],
    heatmap_cache: dict[tuple[str, int], np.ndarray],
) -> float | None:
    if not USE_DEEPCENTER_VETO or detector_bundle is None or dataset is None:
        return None
    heatmap = deepcenter_heatmap_for_frame(
        dataset, int(t), detector_bundle, frame_cache, heatmap_cache
    )
    if heatmap is None or heatmap.size == 0:
        return None
    cfg = detector_bundle["cfg"]
    pool_factor = int(getattr(cfg, "pool_factor", 4))
    z = int(round(float(point[0])))
    y = int(round(float(point[1]) / max(pool_factor, 1)))
    x = int(round(float(point[2]) / max(pool_factor, 1)))
    z0, z1 = (
        max(0, z - DEEPCENTER_SCORE_WIN_Z),
        min(heatmap.shape[0], z + DEEPCENTER_SCORE_WIN_Z + 1),
    )
    y0, y1 = (
        max(0, y - DEEPCENTER_SCORE_WIN_YX),
        min(heatmap.shape[1], y + DEEPCENTER_SCORE_WIN_YX + 1),
    )
    x0, x1 = (
        max(0, x - DEEPCENTER_SCORE_WIN_YX),
        min(heatmap.shape[2], x + DEEPCENTER_SCORE_WIN_YX + 1),
    )
    patch = heatmap[z0:z1, y0:y1, x0:x1]
    if patch.size == 0:
        return None
    score = float(np.max(patch))
    return score if np.isfinite(score) else None


def deepcenter_accept_repair_point(
    dataset: str | None,
    t: int,
    point: tuple[float, float, float],
    detector_bundle: dict[str, object] | None,
    frame_cache: dict[int, np.ndarray],
    heatmap_cache: dict[tuple[str, int], np.ndarray],
    stats: dict[str, int],
    prefix: str,
    threshold: float,
) -> bool:
    if not USE_DEEPCENTER_VETO:
        return True
    if detector_bundle is None or dataset is None:
        stats[f"deepcenter_{prefix}_missing"] += 1
        return True
    stats[f"deepcenter_{prefix}_checked"] += 1
    score = deepcenter_score_point(
        dataset, int(t), point, detector_bundle, frame_cache, heatmap_cache
    )
    if score is None:
        stats[f"deepcenter_{prefix}_missing"] += 1
        return True
    if score < float(threshold):
        stats[f"deepcenter_{prefix}_rejected"] += 1
        return False
    stats[f"deepcenter_{prefix}_accepted"] += 1
    return True


def _position_um(node: dict[str, object]) -> np.ndarray:
    return np.array(
        [
            float(node["z"]) * VOXEL_SCALE_UM[0],
            float(node["y"]) * VOXEL_SCALE_UM[1],
            float(node["x"]) * VOXEL_SCALE_UM[2],
        ],
        dtype=np.float64,
    )


def motion_relink_edges(
    nodes_by_id: dict[int, dict[str, object]],
    stats: dict[str, int],
    learned_edge_probs: dict[tuple[int, int], float] | None = None,
    considered_pairs: set[tuple[int, int]] | None = None,
) -> list[dict[str, object]]:
    if not OUTPUT_MOTION_RELINK or not nodes_by_id:
        return []

    learned_edge_probs = learned_edge_probs or {}

    def learned_prob(source_id: int, target_id: int) -> float:
        value = learned_edge_probs.get((source_id, target_id), 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        if value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))
        return float(np.clip(value, 0.0, 1.0))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)
    for ids in ids_by_t.values():
        ids.sort()

    frame_sizes = [len(ids) for ids in ids_by_t.values()]
    if frame_sizes and max(frame_sizes) > MOTION_RELINK_MAX_FRAME_NODES:
        stats["motion_relink_skipped_large_frame"] = 1
        return []

    position_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}
    predecessor_position_um: dict[int, np.ndarray] = {}
    selected_edges: list[dict[str, object]] = []

    def _flow_predictor(flow_src, flow_disp):
        """Median displacement of the nearest flow samples, or None.

        Cells move with their neighbours, so a displacement field sampled
        from confident links predicts a source's next position better than
        its own last step (experiments/motion_model_audit.py).
        """
        if len(flow_src) < MOTION_RELINK_FLOW_MIN_SAMPLES:
            return None
        flow_src = np.asarray(flow_src, dtype=np.float64)
        flow_disp = np.asarray(flow_disp, dtype=np.float64)
        tree = cKDTree(flow_src)
        k_query = min(MOTION_RELINK_FLOW_K + 1, len(flow_src))

        def predict(source_pos, exclude_um):
            dist, idx = tree.query(
                source_pos, k=k_query, distance_upper_bound=MOTION_RELINK_FLOW_RADIUS_UM
            )
            dist = np.atleast_1d(dist)
            idx = np.atleast_1d(idx)
            keep = np.isfinite(dist) & (dist >= exclude_um)
            idx = idx[keep][:MOTION_RELINK_FLOW_K]
            if idx.size == 0:
                return None
            return np.median(flow_disp[idx], axis=0)

        return predict

    # The flow residual is weighted along z when asked; the raw distance never is.
    flow_weight = np.array([MOTION_RELINK_FLOW_Z_WEIGHT, 1.0, 1.0], dtype=np.float64)
    flow_anisotropic = MOTION_RELINK_FLOW_Z_WEIGHT != 1.0

    def assign_pass(
        source_ids: list[int],
        target_ids: list[int],
        gate_um: float,
        flow=None,
        flow_exclude_um: float = 0.0,
    ) -> list[tuple[int, int, float, float, float]]:
        if not source_ids or not target_ids:
            return []
        big = gate_um * 1000.0 + 1.0
        cost = np.full((len(source_ids), len(target_ids)), big, dtype=np.float64)
        raw_dist = np.full_like(cost, np.inf)
        motion_dist = np.full_like(cost, np.inf)
        prob_matrix = np.zeros_like(cost)
        target_arr = np.stack([position_um[target_id] for target_id in target_ids])
        source_arr = np.stack([position_um[source_id] for source_id in source_ids])
        predicted_arr = np.empty_like(source_arr)
        flow_hit = np.zeros(len(source_ids), dtype=bool)
        for i, source_id in enumerate(source_ids):
            source_pos = position_um[source_id]
            prev_pos = predecessor_position_um.get(source_id)
            flow_step = flow(source_pos, flow_exclude_um) if flow is not None else None
            if flow_step is not None:
                predicted_arr[i] = source_pos + flow_step
                flow_hit[i] = True
                stats["motion_relink_flow_predicted"] = (
                    stats.get("motion_relink_flow_predicted", 0) + 1
                )
            elif prev_pos is None:
                predicted_arr[i] = source_pos
            else:
                predicted_arr[i] = source_pos + MOTION_RELINK_VELOCITY_WEIGHT * (
                    source_pos - prev_pos
                )
        target_tree = cKDTree(target_arr)
        gate_radius = gate_um * (1.0 + 1e-9) + 1e-9
        gate_neighbours = target_tree.query_ball_point(source_arr, r=gate_radius)
        # With a flow prior, a pair is also admitted when the target sits
        # within the gate of the *predicted* position: a cell moving 8 um with
        # its neighbours can then compete in the tight pass instead of waiting
        # for the relaxed one where slower cells have already taken its target.
        # With RAW_ADMIT off, that is the only admission for flow-predicted
        # sources; sources without a field sample keep the raw gate.
        flow_gated = flow is not None and MOTION_RELINK_FLOW_GATE
        if flow_gated:
            predicted_radius = (
                gate_radius / MOTION_RELINK_FLOW_Z_WEIGHT if flow_anisotropic else gate_radius
            )
            around_predicted = target_tree.query_ball_point(predicted_arr, r=predicted_radius)
            gate_neighbours = [
                sorted(set(near_source) | set(near_predicted))
                for near_source, near_predicted in zip(gate_neighbours, around_predicted)
            ]
        for i, source_id in enumerate(source_ids):
            source_pos = position_um[source_id]
            predicted = predicted_arr[i]
            raw_admits = MOTION_RELINK_FLOW_RAW_ADMIT or not (flow_gated and flow_hit[i])
            for j in sorted(gate_neighbours[i]):
                target_id = target_ids[j]
                target_pos = position_um[target_id]
                raw = float(np.linalg.norm(target_pos - source_pos))
                if flow_anisotropic:
                    motion = float(np.linalg.norm((target_pos - predicted) * flow_weight))
                else:
                    motion = float(np.linalg.norm(target_pos - predicted))
                if not ((raw_admits and raw <= gate_um) or (flow_gated and motion <= gate_um)):
                    continue
                if considered_pairs is not None:
                    considered_pairs.add((source_id, target_id))
                prob = learned_prob(source_id, target_id)
                raw_dist[i, j] = raw
                motion_dist[i, j] = motion
                prob_matrix[i, j] = prob
                cost[i, j] = (
                    motion + MOTION_RELINK_FLOW_RAW_COST * raw - MOTION_RELINK_LEARNED_BONUS * prob
                )
        row_ind, col_ind = linear_sum_assignment(cost)
        matches: list[tuple[int, int, float, float, float]] = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= big:
                continue
            matches.append(
                (
                    source_ids[int(r)],
                    target_ids[int(c)],
                    float(raw_dist[r, c]),
                    float(motion_dist[r, c]),
                    float(prob_matrix[r, c]),
                )
            )
        return matches

    def _field_from(matches):
        return _flow_predictor(
            [position_um[m[0]] for m in matches],
            [position_um[m[1]] - position_um[m[0]] for m in matches],
        )

    seed_gate_um = (
        MOTION_RELINK_FLOW_SEED_GATE_UM
        if MOTION_RELINK_FLOW_SEED_GATE_UM > 0
        else MOTION_RELINK_TIGHT_UM
    )
    flow_tight_um = (
        MOTION_RELINK_FLOW_TIGHT_UM if MOTION_RELINK_FLOW_TIGHT_UM > 0 else MOTION_RELINK_TIGHT_UM
    )
    flow_relaxed_um = (
        MOTION_RELINK_FLOW_RELAXED_UM
        if MOTION_RELINK_FLOW_RELAXED_UM > 0
        else MOTION_RELINK_RELAXED_UM
    )

    times = sorted(ids_by_t)
    previous_flow = None
    for t in times:
        source_ids = ids_by_t.get(t, [])
        target_ids = ids_by_t.get(t + 1, [])
        if not source_ids or not target_ids:
            continue
        flow = None
        flow_exclude_um = 0.0
        if MOTION_RELINK_FLOW_MODE == "prev":
            flow = previous_flow
        elif MOTION_RELINK_FLOW_MODE == "seed":
            # Confident tight matches first, then everything is assigned
            # against the field they define. A source's own seed match is
            # excluded from its sample so a wrong seed cannot vote for itself.
            seed = assign_pass(source_ids, target_ids, seed_gate_um, previous_flow)
            flow = _field_from(seed)
            flow_exclude_um = MOTION_RELINK_FLOW_EXCLUDE_UM
            if flow is not None:
                stats["motion_relink_flow_frames"] = stats.get("motion_relink_flow_frames", 0) + 1
        rounds = MOTION_RELINK_FLOW_ITER if MOTION_RELINK_FLOW_MODE != "off" else 1
        frame_matches: list[tuple[int, int, float, float, str, float]] = []
        for round_index in range(max(1, rounds)):
            if round_index > 0:
                # Refine: the field from the previous round's final matches,
                # then assign every source again against it.
                refined = _field_from([(s, g) for s, g, _r, _m, _n, _p in frame_matches])
                if refined is None:
                    break
                flow = refined
                flow_exclude_um = MOTION_RELINK_FLOW_EXCLUDE_UM
            unmatched_sources = set(source_ids)
            unmatched_targets = set(target_ids)
            frame_matches = []
            passes = (
                (("tight", flow_tight_um), ("relaxed", flow_relaxed_um))
                if flow is not None
                else (("tight", MOTION_RELINK_TIGHT_UM), ("relaxed", MOTION_RELINK_RELAXED_UM))
            )
            for pass_name, gate_um in passes:
                pass_sources = [node_id for node_id in source_ids if node_id in unmatched_sources]
                pass_targets = [node_id for node_id in target_ids if node_id in unmatched_targets]
                matches = assign_pass(pass_sources, pass_targets, gate_um, flow, flow_exclude_um)
                for source_id, target_id, raw, motion, prob in matches:
                    if source_id not in unmatched_sources or target_id not in unmatched_targets:
                        continue
                    unmatched_sources.remove(source_id)
                    unmatched_targets.remove(target_id)
                    frame_matches.append((source_id, target_id, raw, motion, pass_name, prob))
        for source_id, target_id, raw, motion, pass_name, prob in frame_matches:
            if pass_name == "tight":
                stats["motion_relink_tight_edges"] += 1
            else:
                stats["motion_relink_relaxed_edges"] += 1
            selected_edges.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "edge_prob": prob,
                    "distance_um": raw,
                    "motion_distance_um": motion,
                    "motion_relinked": 1,
                    "motion_pass": pass_name,
                }
            )
            predecessor_position_um[target_id] = position_um[source_id]
        if MOTION_RELINK_FLOW_MODE != "off":
            previous_flow = _field_from([(s, g) for s, g, _r, _m, _n, _p in frame_matches])
        stats["motion_relink_frames"] += 1

    stats["motion_relink_edges"] = len(selected_edges)
    return selected_edges


def close_single_frame_gaps(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
    deepcenter_bundle: dict[str, object] | None = None,
    frame_cache: dict[int, np.ndarray] | None = None,
    deepcenter_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_GAP_CLOSE or GAP_CLOSE_MAX_GAP < 1 or not edges:
        return nodes_by_id, edges

    outgoing = {int(edge["source_id"]) for edge in edges}
    incoming = {int(edge["target_id"]) for edge in edges}
    incident = outgoing | incoming

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    isolated_by_t: dict[int, list[int]] = {}
    all_ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        all_ids_by_t.setdefault(t, []).append(node_id)
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)
        if node_id not in incident:
            isolated_by_t.setdefault(t, []).append(node_id)

    max_synthetic = min(
        GAP_CLOSE_MAX_ADDED_ABS,
        max(1, int(round(len(nodes_by_id) * GAP_CLOSE_MAX_ADDED_FRAC)))
        if GAP_CLOSE_MAX_ADDED_FRAC > 0
        else 0,
    )
    next_id = _next_node_id(nodes_by_id)
    frame_cache = frame_cache if frame_cache is not None else {}
    deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}
    used_starts: set[int] = set()
    used_isolated: set[int] = set()
    synthetic_added = 0
    new_edges: list[dict[str, object]] = []

    density_cache: dict[int, dict[int, float]] = {}

    def frame_local_spacing(t: int) -> dict[int, float]:
        cached = density_cache.get(t)
        if cached is not None:
            return cached

        frame_ids = all_ids_by_t.get(t, [])
        if len(frame_ids) <= 1:
            result = {node_id: GAP_DENSITY_REFERENCE_UM for node_id in frame_ids}
            density_cache[t] = result
            return result

        positions = np.stack([_position_um(nodes_by_id[node_id]) for node_id in frame_ids])
        tree = cKDTree(positions)
        query_k = min(
            len(frame_ids),
            max(2, GAP_DENSITY_NEIGHBORS + 1),
        )
        distances, _ = tree.query(positions, k=query_k)
        if distances.ndim == 1:
            distances = distances[:, None]

        result: dict[int, float] = {}
        for idx, node_id in enumerate(frame_ids):
            neighbour_distances = distances[idx, 1:]
            neighbour_distances = neighbour_distances[np.isfinite(neighbour_distances)]
            spacing = (
                float(np.median(neighbour_distances))
                if neighbour_distances.size
                else GAP_DENSITY_REFERENCE_UM
            )
            result[node_id] = spacing

        density_cache[t] = result
        stats["gap_density_nodes_scored"] += len(result)
        return result

    effective_gap_max = min(GAP_CLOSE_MAX_GAP, 1)
    stats["gap_close_effective_max_gap"] = effective_gap_max
    for gap in range(1, effective_gap_max + 1):
        for t, end_ids in sorted(ends_by_t.items()):
            start_ids = [sid for sid in starts_by_t.get(t + gap + 1, []) if sid not in used_starts]
            if not end_ids or not start_ids:
                continue

            end_points = [node_point(nodes_by_id[eid]) for eid in end_ids]
            start_points = [node_point(nodes_by_id[sid]) for sid in start_ids]
            threshold_um = GAP_CLOSE_UM * (gap + 1)
            d = np.zeros(
                (len(end_ids), len(start_ids)),
                dtype=np.float64,
            )
            adaptive_threshold = np.full_like(d, threshold_um)

            source_spacing = frame_local_spacing(t)
            target_spacing = frame_local_spacing(t + gap + 1)

            for i, ep in enumerate(end_points):
                for j, sp in enumerate(start_points):
                    d[i, j] = point_distance_um(ep, sp)

                    if GAP_DENSITY_ADAPTIVE:
                        local_spacing = 0.5 * (
                            source_spacing.get(
                                end_ids[i],
                                GAP_DENSITY_REFERENCE_UM,
                            )
                            + target_spacing.get(
                                start_ids[j],
                                GAP_DENSITY_REFERENCE_UM,
                            )
                        )
                        step_delta = float(
                            np.clip(
                                GAP_DENSITY_GAIN * (local_spacing - GAP_DENSITY_REFERENCE_UM),
                                -GAP_DENSITY_MAX_STEP_DELTA_UM,
                                GAP_DENSITY_MAX_STEP_DELTA_UM,
                            )
                        )
                        adaptive_threshold[i, j] = threshold_um + step_delta * (gap + 1)
                        stats["gap_density_step_delta_milli_sum"] += int(round(1000.0 * step_delta))

            base_allowed = d <= threshold_um
            adaptive_allowed = d <= adaptive_threshold

            stats["gap_density_candidates_expanded"] += int(
                (adaptive_allowed & ~base_allowed).sum()
            )
            stats["gap_density_candidates_restricted"] += int(
                (base_allowed & ~adaptive_allowed).sum()
            )
            stats["gap_candidates"] += int(adaptive_allowed.sum())

            if not np.isfinite(d).any():
                continue

            max_threshold = float(np.max(adaptive_threshold))
            big = max_threshold * 1000.0 + 1.0
            cost = np.where(adaptive_allowed, d, big)
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                if not adaptive_allowed[r, c]:
                    continue
                if not base_allowed[r, c]:
                    stats["gap_density_selected_outside_base"] += 1
                source_id = end_ids[int(r)]
                target_id = start_ids[int(c)]
                if source_id in outgoing or target_id in used_starts:
                    continue

                source = nodes_by_id[source_id]
                target = nodes_by_id[target_id]
                mid_t = int(source["t"]) + gap
                mid_point = (
                    (float(source["z"]) + float(target["z"])) / 2.0,
                    (float(source["y"]) + float(target["y"])) / 2.0,
                    (float(source["x"]) + float(target["x"])) / 2.0,
                )

                middle_id: int | None = None
                middle_reused = False
                if GAP_CLOSE_REUSE_EXISTING:
                    candidates = [
                        nid for nid in isolated_by_t.get(mid_t, []) if nid not in used_isolated
                    ]
                    if candidates:
                        distances = [
                            point_distance_um(node_point(nodes_by_id[nid]), mid_point)
                            for nid in candidates
                        ]
                        best_idx = int(np.argmin(distances))
                        if distances[best_idx] <= GAP_CLOSE_REUSE_UM:
                            middle_id = candidates[best_idx]
                            middle_reused = True

                if middle_id is None:
                    if synthetic_added >= max_synthetic:
                        stats["gap_skipped_node_cap"] += 1
                        continue
                    middle_id = next_id
                    next_id += 1
                    refined_point = refine_synthetic_midpoint(
                        dataset, mid_t, mid_point, frame_cache, stats
                    )
                    nodes_by_id[middle_id] = {
                        "node_id": middle_id,
                        "t": mid_t,
                        "z": refined_point[0],
                        "y": refined_point[1],
                        "x": refined_point[2],
                        "gap_synthetic": 1,
                    }
                    synthetic_added += 1
                    stats["gap_inserted_synthetic"] += 1

                middle = nodes_by_id[middle_id]
                gap_span_um = float(d[r, c])
                marginal_gap = gap_span_um >= DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM
                synthetic_middle = int(middle.get("gap_synthetic", 0)) == 1
                requires_center_confirmation = (
                    DEEPCENTER_GAP_VETO and marginal_gap and synthetic_middle
                )
                if DEEPCENTER_GAP_VETO and not marginal_gap:
                    stats["deepcenter_gap_bypassed_strong_motion"] += 1
                elif DEEPCENTER_GAP_VETO and not synthetic_middle:
                    stats["deepcenter_gap_bypassed_observed_node"] += 1
                if requires_center_confirmation and not deepcenter_accept_repair_point(
                    dataset,
                    mid_t,
                    node_point(middle),
                    deepcenter_bundle,
                    frame_cache,
                    deepcenter_cache,
                    stats,
                    "gap",
                    DEEPCENTER_GAP_THRESHOLD,
                ):
                    if int(middle.get("gap_synthetic", 0)) == 1:
                        nodes_by_id.pop(middle_id, None)
                        synthetic_added = max(0, synthetic_added - 1)
                        stats["gap_inserted_synthetic"] = max(
                            0, stats["gap_inserted_synthetic"] - 1
                        )
                    continue
                if middle_reused:
                    used_isolated.add(middle_id)
                    stats["gap_reused_existing"] += 1

                e1 = {
                    "source_id": source_id,
                    "target_id": middle_id,
                    "edge_prob": None,
                    "distance_um": edge_distance_um(source, middle),
                    "gap_closed": 1,
                }
                e2 = {
                    "source_id": middle_id,
                    "target_id": target_id,
                    "edge_prob": None,
                    "distance_um": edge_distance_um(middle, target),
                    "gap_closed": 1,
                }
                new_edges.extend([e1, e2])
                outgoing.add(source_id)
                incoming.add(middle_id)
                outgoing.add(middle_id)
                incoming.add(target_id)
                used_starts.add(target_id)
                stats["gap_pairs_selected"] += 1
                stats["gap_added_edges"] += 2

    if new_edges:
        edges = [*edges, *new_edges]
    stats["gap_added_nodes"] = stats["gap_inserted_synthetic"]
    return nodes_by_id, edges


def _single_successor_map(edges: list[dict[str, object]]) -> dict[int, int]:
    by_source: dict[int, list[int]] = {}
    for edge in edges:
        by_source.setdefault(int(edge["source_id"]), []).append(int(edge["target_id"]))
    return {source: targets[0] for source, targets in by_source.items() if len(targets) == 1}


def _single_predecessor_map(edges: list[dict[str, object]]) -> dict[int, int]:
    by_target: dict[int, list[int]] = {}
    for edge in edges:
        by_target.setdefault(int(edge["target_id"]), []).append(int(edge["source_id"]))
    return {target: sources[0] for target, sources in by_target.items() if len(sources) == 1}


def recover_strict_gap2(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_GAP2_RECOVERY or not edges or not nodes_by_id:
        return nodes_by_id, edges

    outgoing = {int(edge["source_id"]) for edge in edges}
    incoming = {int(edge["target_id"]) for edge in edges}
    predecessor = _single_predecessor_map(edges)
    successor = _single_successor_map(edges)

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)

    cap = min(GAP2_MAX_LINKS_ABS, max(1, int(round(len(edges) * GAP2_MAX_LINKS_FRAC))))
    proposals: list[tuple[float, int, int, int, float]] = []

    def pos_um(node_id: int) -> np.ndarray:
        node = nodes_by_id[node_id]
        return np.array(
            [float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64
        ) * np.array(VOXEL_SCALE_UM)

    for t, end_ids in sorted(ends_by_t.items()):
        start_ids = starts_by_t.get(t + 3, [])
        if not end_ids or not start_ids:
            continue
        for end_id in end_ids:
            end_pos = pos_um(end_id)
            for start_id in start_ids:
                start_pos = pos_um(start_id)
                dist = float(np.linalg.norm(start_pos - end_pos))
                if dist > GAP2_MAX_TOTAL_UM or dist / 3.0 > GAP2_MAX_STEP_UM:
                    continue
                step = (start_pos - end_pos) / 3.0
                context_penalty = 0.0
                if GAP2_REQUIRE_CONTEXT:
                    ok_context = False
                    prev_id = predecessor.get(end_id)
                    if prev_id is not None:
                        prev_step = end_pos - pos_um(prev_id)
                        prev_norm = float(np.linalg.norm(prev_step))
                        step_norm = float(np.linalg.norm(step))
                        if prev_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(prev_step, step) / (prev_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(prev_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    next_id = successor.get(start_id)
                    if next_id is not None:
                        next_step = pos_um(next_id) - start_pos
                        next_norm = float(np.linalg.norm(next_step))
                        step_norm = float(np.linalg.norm(step))
                        if next_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(next_step, step) / (next_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(next_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    if not ok_context:
                        continue
                proposals.append((dist + 2.0 * context_penalty, end_id, start_id, t, dist))

    proposals.sort(key=lambda item: item[0])
    stats["gap2_candidates"] = len(proposals)
    if not proposals:
        return nodes_by_id, edges

    selected: list[tuple[float, int, int, int, float]] = []
    used_ends: set[int] = set()
    used_starts: set[int] = set()
    per_frame_count: dict[int, int] = {}
    for proposal in proposals:
        if len(selected) >= cap:
            stats["gap2_skipped_cap"] += 1
            break
        _, end_id, start_id, t, _ = proposal
        if end_id in used_ends or start_id in used_starts:
            continue
        frame_cap = max(1, int(round(len(ends_by_t.get(t, [])) * GAP2_FRAME_FRAC_CAP)))
        if per_frame_count.get(t, 0) >= frame_cap:
            continue
        selected.append(proposal)
        used_ends.add(end_id)
        used_starts.add(start_id)
        per_frame_count[t] = per_frame_count.get(t, 0) + 1

    if not selected:
        return nodes_by_id, edges

    next_node_id = _next_node_id(nodes_by_id)
    frame_cache: dict[int, np.ndarray] = {}
    new_edges: list[dict[str, object]] = []
    for _, end_id, start_id, t, _ in selected:
        source = nodes_by_id[end_id]
        target = nodes_by_id[start_id]
        previous_id = end_id
        inserted_ids: list[int] = []
        for k in (1, 2):
            frac = k / 3.0
            mid_t = int(source["t"]) + k
            midpoint = (
                float(source["z"]) + (float(target["z"]) - float(source["z"])) * frac,
                float(source["y"]) + (float(target["y"]) - float(source["y"])) * frac,
                float(source["x"]) + (float(target["x"]) - float(source["x"])) * frac,
            )
            refined_point = refine_synthetic_midpoint(dataset, mid_t, midpoint, frame_cache, stats)
            node_id = next_node_id
            next_node_id += 1
            nodes_by_id[node_id] = {
                "node_id": node_id,
                "t": mid_t,
                "z": refined_point[0],
                "y": refined_point[1],
                "x": refined_point[2],
            }
            inserted_ids.append(node_id)
            current = nodes_by_id[node_id]
            new_edges.append(
                {
                    "source_id": previous_id,
                    "target_id": node_id,
                    "edge_prob": None,
                    "distance_um": edge_distance_um(nodes_by_id[previous_id], current),
                    "gap2_recovered": 1,
                }
            )
            previous_id = node_id
        new_edges.append(
            {
                "source_id": previous_id,
                "target_id": start_id,
                "edge_prob": None,
                "distance_um": edge_distance_um(nodes_by_id[previous_id], target),
                "gap2_recovered": 1,
            }
        )
        stats["gap2_pairs_selected"] += 1
        stats["gap2_added_nodes"] += len(inserted_ids)
        stats["gap2_added_edges"] += 3

    return nodes_by_id, [*edges, *new_edges]


def _gapfill_bump(stats: dict[str, int], key: str, n: int = 1) -> None:
    stats[key] = int(stats.get(key, 0)) + n


def load_low_detections(
    nodes_by_id: dict[int, dict[str, object]],
    dataset: str | None,
    stats: dict[str, int],
) -> dict[int, dict[str, np.ndarray]] | None:
    """Per-frame pool of the detector's sub-threshold peaks from the prediction cell's dump.

    Peaks below GAPFILL_MIN_SCORE and peaks within GAPFILL_EXCLUDE_UM of a node
    of their frame (the node set's own peaks among them) are dropped. None when
    the dump is missing or unreadable, and the filler then does nothing.
    """
    cache_dir = os.environ.get("BIOHUB_CACHE_DIR", "").strip()
    if GAPFILL_MAX_GAP < 1 or not cache_dir or not dataset:
        return None
    cache_path = Path(cache_dir) / f"{dataset}.npz"
    if not cache_path.exists():
        print(f"  [{dataset}] no low-detection dump at {cache_path}; gap filler idle")
        return None
    try:
        with np.load(cache_path) as cz:
            if "low_coords" not in cz.files:
                print(f"  [{dataset}] dump has no low_coords; gap filler idle")
                return None
            low = np.asarray(cz["low_coords"], dtype=np.float64).reshape(-1, 4)
            score = np.asarray(cz["low_score"], dtype=np.float64).reshape(-1)
    except Exception as exc:
        print(
            f"  [{dataset}] low-detection dump unreadable ({type(exc).__name__}: {exc}); gap filler idle"
        )
        return None
    return build_low_detection_pool(nodes_by_id, low, score, stats, dataset)


def build_low_detection_pool(
    nodes_by_id: dict[int, dict[str, object]],
    low: np.ndarray,
    score: np.ndarray,
    stats: dict[str, int],
    dataset: str | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    keep = score >= GAPFILL_MIN_SCORE
    low, score = low[keep], score[keep]
    scale = np.array(VOXEL_SCALE_UM, dtype=np.float64)
    node_um_by_t: dict[int, list] = {}
    for node in nodes_by_id.values():
        node_um_by_t.setdefault(int(node["t"]), []).append(
            np.array(node_point(node), dtype=np.float64) * scale
        )
    pool: dict[int, dict[str, np.ndarray]] = {}
    excluded = 0
    for t in np.unique(low[:, 0]).astype(int).tolist() if len(low) else []:
        sel = low[:, 0] == t
        vox = low[sel, 1:]
        um = vox * scale
        sc = score[sel]
        existing = node_um_by_t.get(t)
        if existing:
            d, _ = cKDTree(np.stack(existing)).query(um, k=1)
            free = d > GAPFILL_EXCLUDE_UM
            excluded += int((~free).sum())
            vox, um, sc = vox[free], um[free], sc[free]
        if len(vox):
            pool[t] = {"vox": vox, "um": um, "score": sc}
    n_free = int(sum(len(p["vox"]) for p in pool.values()))
    _gapfill_bump(stats, "gapfill_pool_peaks", n_free)
    _gapfill_bump(stats, "gapfill_pool_excluded", excluded)
    if dataset is not None:
        print(
            f"  [{dataset}] low-detection pool: {len(low)} peaks >= {GAPFILL_MIN_SCORE}, "
            f"{excluded} on existing nodes, {n_free} free"
        )
    return pool


def readmit_discarded_detections(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
) -> dict[int, dict[str, object]]:
    """Add detector peaks the ILP discarded that sit next to an open track end or start.

    A free peak (load_low_detections: not within GAPFILL_EXCLUDE_UM of a node)
    scoring at least READMIT_MIN_SCORE comes back when a node at t-1 with no
    outgoing edge, or one at t+1 with no incoming edge, is within
    READMIT_RADIUS_UM. ``edges`` are the first re-link's; the caller re-links
    afterwards, which places the new nodes or leaves them isolated for the
    prune. Non-fatal: any failure leaves the node set as it was.
    """
    if READMIT_RADIUS_UM <= 0:
        return nodes_by_id
    try:
        pool = load_low_detections(nodes_by_id, dataset, stats)
        if not pool:
            return nodes_by_id
        has_out = {int(e["source_id"]) for e in edges}
        has_in = {int(e["target_id"]) for e in edges}
        scale = np.array(VOXEL_SCALE_UM, dtype=np.float64)
        anchors: dict[int, list] = {}
        for nid, node in nodes_by_id.items():
            t = int(node["t"])
            um = np.array(node_point(node), dtype=np.float64) * scale
            if nid not in has_out:
                anchors.setdefault(t + 1, []).append(um)
            if nid not in has_in:
                anchors.setdefault(t - 1, []).append(um)
        next_id = _next_node_id(nodes_by_id)
        added = 0
        for t in sorted(pool):
            near = anchors.get(int(t))
            if not near:
                continue
            peaks = pool[t]
            d, _ = cKDTree(np.stack(near)).query(peaks["um"], k=1)
            keep = (d <= READMIT_RADIUS_UM) & (peaks["score"] >= READMIT_MIN_SCORE)
            for vox in peaks["vox"][keep]:
                nodes_by_id[next_id] = {
                    "node_id": next_id,
                    "t": int(t),
                    "z": float(vox[0]),
                    "y": float(vox[1]),
                    "x": float(vox[2]),
                    "readmitted": 1,
                }
                next_id += 1
                added += 1
        _gapfill_bump(stats, "readmitted_nodes", added)
        if dataset is not None:
            print(
                f"  [{dataset}] readmitted {added} discarded detections within {READMIT_RADIUS_UM} um of an open end/start"
            )
    except Exception as exc:
        print(f"  [{dataset}] readmit skipped (non-fatal): {type(exc).__name__}: {exc}")
    return nodes_by_id


def fill_gaps_from_low_detections(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
    frame_cache: dict[int, np.ndarray] | None = None,
    pool: dict[int, dict[str, np.ndarray]] | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    """Bridge a track end at t to a track start at t+g+1 through sub-threshold peaks.

    Runs after the single-frame closer and gap2, on what they left open. For
    each candidate pair (span within GAPFILL_STEP_UM per frame, gap2's
    direction test when a neighbour exists) the straight line from end to
    start is sampled at each missing frame and the nearest free peak within
    GAPFILL_PEAK_RADIUS_UM of the sample is taken; a bridge needs a peak at
    every frame except at most GAPFILL_ALLOW_SYNTHETIC of them, which get an
    interpolated node refined against the frame like the closer's midpoints.
    The pairs of one (t, g) are assigned by Hungarian on span/(g+1) plus the
    mean peak deviation, shorter gaps first. Added nodes are capped at
    GAPFILL_MAX_ADDED_FRAC of the node set.
    """
    if GAPFILL_MAX_GAP < 1 or not edges or not nodes_by_id:
        return nodes_by_id, edges
    if pool is None:
        pool = load_low_detections(nodes_by_id, dataset, stats)
    if not pool:
        return nodes_by_id, edges
    scale = np.array(VOXEL_SCALE_UM, dtype=np.float64)
    outgoing: dict[int, list[int]] = {}
    incoming: dict[int, list[int]] = {}
    for edge in edges:
        outgoing.setdefault(int(edge["source_id"]), []).append(int(edge["target_id"]))
        incoming.setdefault(int(edge["target_id"]), []).append(int(edge["source_id"]))
    pos = {
        nid: np.array(node_point(node), dtype=np.float64) * scale
        for nid, node in nodes_by_id.items()
    }
    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    for nid, node in nodes_by_id.items():
        t = int(node["t"])
        if nid not in outgoing:
            ends_by_t.setdefault(t, []).append(nid)
        if nid not in incoming:
            starts_by_t.setdefault(t, []).append(nid)
    trees = {t: cKDTree(p["um"]) for t, p in pool.items()}
    used_peak = {t: np.zeros(len(p["um"]), dtype=bool) for t, p in pool.items()}
    budget = int(round(len(nodes_by_id) * GAPFILL_MAX_ADDED_FRAC))
    frame_cache = frame_cache if frame_cache is not None else {}
    next_id = _next_node_id(nodes_by_id)
    used_end: set[int] = set()
    used_start: set[int] = set()
    added_nodes = 0
    new_edges: list[dict[str, object]] = []

    def context_ok(end_id: int, start_id: int) -> bool:
        if not GAPFILL_CONTEXT:
            return True
        step = pos[start_id] - pos[end_id]
        sn = float(np.linalg.norm(step))
        if sn <= 0.01:
            return True
        prev = incoming.get(end_id)
        if prev:
            other = pos[end_id] - pos[prev[0]]
            on = float(np.linalg.norm(other))
            if on > 0.01 and float(np.dot(other, step)) / (on * sn) <= -0.25:
                return False
        nxt = outgoing.get(start_id)
        if nxt:
            other = pos[nxt[0]] - pos[start_id]
            on = float(np.linalg.norm(other))
            if on > 0.01 and float(np.dot(other, step)) / (on * sn) <= -0.25:
                return False
        return True

    def chain_for(end_id: int, start_id: int, g: int):
        span = pos[start_id] - pos[end_id]
        t0 = int(nodes_by_id[end_id]["t"])
        items, dev, synthetic = [], [], 0
        for k in range(1, g + 1):
            q = pos[end_id] + span * (k / (g + 1))
            tk = t0 + k
            tree = trees.get(tk)
            j = None
            if tree is not None:
                cand = [
                    c
                    for c in tree.query_ball_point(q, r=GAPFILL_PEAK_RADIUS_UM)
                    if not used_peak[tk][c]
                ]
                if cand:
                    dists = np.linalg.norm(pool[tk]["um"][cand] - q, axis=1)
                    best = int(np.argmin(dists))
                    j = cand[best]
                    dev.append(float(dists[best]))
            if j is None:
                synthetic += 1
                if synthetic > GAPFILL_ALLOW_SYNTHETIC:
                    return None
                dev.append(GAPFILL_PEAK_RADIUS_UM)
            items.append((tk, j, q))
        cost = float(np.linalg.norm(span)) / (g + 1) + (sum(dev) / len(dev) if dev else 0.0)
        return cost, items

    for g in range(1, GAPFILL_MAX_GAP + 1):
        gate = GAPFILL_STEP_UM * (g + 1)
        for t in sorted(ends_by_t):
            if added_nodes + g > budget:
                _gapfill_bump(stats, "gapfill_budget_hit")
                break
            ends = [e for e in ends_by_t[t] if e not in used_end]
            starts = [s for s in starts_by_t.get(t + g + 1, []) if s not in used_start]
            if not ends or not starts:
                continue
            end_pts = np.stack([pos[e] for e in ends])
            start_pts = np.stack([pos[s] for s in starts])
            start_tree = cKDTree(start_pts)
            cost = np.full((len(ends), len(starts)), np.inf)
            n_chains = 0
            for i, js in enumerate(start_tree.query_ball_point(end_pts, r=gate)):
                for j in js:
                    if not context_ok(ends[i], starts[j]):
                        continue
                    chain = chain_for(ends[i], starts[j], g)
                    if chain is None:
                        continue
                    cost[i, j] = chain[0]
                    n_chains += 1
            if n_chains == 0:
                continue
            _gapfill_bump(stats, "gapfill_candidates", n_chains)
            finite = np.isfinite(cost)
            big = float(np.max(cost[finite])) * 1000.0 + 1.0
            row_ind, col_ind = linear_sum_assignment(np.where(finite, cost, big))
            picks = sorted(
                (float(cost[i, j]), int(i), int(j))
                for i, j in zip(row_ind, col_ind)
                if finite[i, j]
            )
            for _, i, j in picks:
                if added_nodes + g > budget:
                    _gapfill_bump(stats, "gapfill_budget_hit")
                    break
                chain = chain_for(ends[i], starts[j], g)  # an earlier pick may have taken a peak
                if chain is None:
                    continue
                prev = ends[i]
                for tk, pk, q in chain[1]:
                    nid = next_id
                    next_id += 1
                    if pk is not None:
                        vox = pool[tk]["vox"][pk]
                        used_peak[tk][pk] = True
                        node = {
                            "node_id": nid,
                            "t": tk,
                            "z": float(vox[0]),
                            "y": float(vox[1]),
                            "x": float(vox[2]),
                            "gapfill_peak": 1,
                        }
                        _gapfill_bump(stats, "gapfill_peak_nodes")
                    else:
                        p = q / scale
                        refined = refine_synthetic_midpoint(
                            dataset, tk, (float(p[0]), float(p[1]), float(p[2])), frame_cache, stats
                        )
                        node = {
                            "node_id": nid,
                            "t": tk,
                            "z": float(refined[0]),
                            "y": float(refined[1]),
                            "x": float(refined[2]),
                            "gap_synthetic": 1,
                        }
                        _gapfill_bump(stats, "gapfill_synthetic_nodes")
                    nodes_by_id[nid] = node
                    new_edges.append(
                        {
                            "source_id": prev,
                            "target_id": nid,
                            "edge_prob": None,
                            "distance_um": edge_distance_um(nodes_by_id[prev], node),
                            "gap_filled": 1,
                        }
                    )
                    prev = nid
                    added_nodes += 1
                new_edges.append(
                    {
                        "source_id": prev,
                        "target_id": starts[j],
                        "edge_prob": None,
                        "distance_um": edge_distance_um(nodes_by_id[prev], nodes_by_id[starts[j]]),
                        "gap_filled": 1,
                    }
                )
                used_end.add(ends[i])
                used_start.add(starts[j])
                _gapfill_bump(stats, f"gapfill_pairs_g{g}")
    _gapfill_bump(stats, "gapfill_added_nodes", added_nodes)
    _gapfill_bump(stats, "gapfill_added_edges", len(new_edges))
    if dataset is not None and new_edges:
        print(f"  [{dataset}] gap filler: +{added_nodes} nodes, +{len(new_edges)} edges")
    return nodes_by_id, [*edges, *new_edges]


def add_safe_divisions_postlink(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
    deepcenter_bundle: dict[str, object] | None = None,
    frame_cache: dict[int, np.ndarray] | None = None,
    deepcenter_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> list[dict[str, object]]:
    if not OUTPUT_SAFE_DIVISIONS or not edges or not nodes_by_id:
        return edges
    frame_cache = frame_cache if frame_cache is not None else {}
    deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}

    out_by_source: dict[int, list[dict[str, object]]] = {}
    incoming: set[int] = set()
    for edge in edges:
        out_by_source.setdefault(int(edge["source_id"]), []).append(edge)
        incoming.add(int(edge["target_id"]))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)

    existing_edges = {(int(edge["source_id"]), int(edge["target_id"])) for edge in edges}
    global_cap = max(1, int(round(max(1, len(edges)) * SAFE_DIV_GLOBAL_FRAC_CAP)))
    added: list[dict[str, object]] = []
    used_targets: set[int] = set()
    used_sources: set[int] = set()

    for t in sorted(ids_by_t):
        child_frame_ids = ids_by_t.get(t + 1, [])
        if not child_frame_ids:
            continue
        source_ids = [
            node_id for node_id in ids_by_t[t] if len(out_by_source.get(node_id, [])) == 1
        ]
        candidate_ids = [
            node_id
            for node_id in child_frame_ids
            if node_id not in incoming and node_id not in used_targets
        ]
        if not source_ids or not candidate_ids:
            continue

        candidate_tree = None
        if SAFE_DIV_REQUIRE_MUTUAL_NN:
            candidate_positions = np.stack(
                [_position_um(nodes_by_id[cid]) for cid in candidate_ids]
            )
            candidate_tree = cKDTree(candidate_positions)

        frame_cap = max(1, int(round(len(source_ids) * SAFE_DIV_FRAME_FRAC_CAP)))
        proposals: list[tuple[float, int, int, float, float]] = []
        for source_id in source_ids:
            source = nodes_by_id[source_id]
            existing_child_edge = out_by_source[source_id][0]
            existing_child_id = int(existing_child_edge["target_id"])
            existing_child = nodes_by_id.get(existing_child_id)
            if existing_child is None or int(existing_child["t"]) != t + 1:
                continue
            child_dist = edge_distance_um(source, existing_child)
            if child_dist > SAFE_DIV_EXISTING_CHILD_MAX_UM:
                continue

            mutual_nn_id = None
            if candidate_tree is not None:
                _, nn_idx = candidate_tree.query(_position_um(existing_child))
                mutual_nn_id = candidate_ids[int(nn_idx)]

            for candidate_id in candidate_ids:
                if (source_id, candidate_id) in existing_edges:
                    continue
                candidate = nodes_by_id[candidate_id]
                parent_dist = edge_distance_um(source, candidate)
                if parent_dist > SAFE_DIV_MAX_UM:
                    continue
                sister_dist = edge_distance_um(existing_child, candidate)
                if sister_dist > SAFE_DIV_SISTER_MAX_UM:
                    continue

                if SAFE_DIV_REQUIRE_MUTUAL_NN and candidate_id != mutual_nn_id:
                    stats["safe_division_mutual_nn_rejected"] += 1
                    continue

                if SAFE_DIV_REQUIRE_DIVERGENCE:
                    c1_succ = out_by_source.get(existing_child_id, [])
                    q_succ = out_by_source.get(candidate_id, [])
                    if len(c1_succ) != 1 or len(q_succ) != 1:
                        stats["safe_division_divergence_rejected"] += 1
                        continue
                    c1_grandchild = nodes_by_id.get(int(c1_succ[0]["target_id"]))
                    q_grandchild = nodes_by_id.get(int(q_succ[0]["target_id"]))
                    if (
                        c1_grandchild is None
                        or q_grandchild is None
                        or int(c1_grandchild["t"]) != t + 2
                        or int(q_grandchild["t"]) != t + 2
                    ):
                        stats["safe_division_divergence_rejected"] += 1
                        continue
                    grandchild_dist = edge_distance_um(c1_grandchild, q_grandchild)
                    if grandchild_dist - sister_dist < SAFE_DIV_DIVERGE_UM:
                        stats["safe_division_divergence_rejected"] += 1
                        continue

                stats["safe_division_geometric_candidates"] += 1
                if DEEPCENTER_SAFE_DIV_VETO and not deepcenter_accept_repair_point(
                    dataset,
                    int(candidate["t"]),
                    node_point(candidate),
                    deepcenter_bundle,
                    frame_cache,
                    deepcenter_cache,
                    stats,
                    "safe_div",
                    DEEPCENTER_SAFE_DIV_THRESHOLD,
                ):
                    continue

                if SAFE_DIV_SISTER_SYMMETRY_TAU > 0.0:
                    _sym_denom = max((child_dist + parent_dist) / 2.0, 1e-6)
                    if abs(child_dist - parent_dist) / _sym_denom > SAFE_DIV_SISTER_SYMMETRY_TAU:
                        stats["safe_division_symmetry_rejected"] += 1
                        continue
                score = parent_dist + 0.15 * sister_dist
                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))

        stats["safe_division_candidates"] += len(proposals)
        if not proposals:
            continue
        proposals.sort(key=lambda item: item[0])
        added_this_frame = 0
        for _, source_id, candidate_id, parent_dist, _ in proposals:
            if len(added) >= global_cap:
                stats["safe_division_skipped_cap"] += 1
                break
            if added_this_frame >= frame_cap:
                break
            if candidate_id in used_targets or candidate_id in incoming:
                continue
            if source_id in used_sources:
                continue
            candidate = nodes_by_id[candidate_id]
            added.append(
                {
                    "source_id": source_id,
                    "target_id": candidate_id,
                    "edge_prob": None,
                    "distance_um": parent_dist,
                    "safe_division": 1,
                }
            )
            used_targets.add(candidate_id)
            used_sources.add(source_id)
            added_this_frame += 1

    if added:
        stats["safe_divisions_added"] = len(added)
        return [*edges, *added]
    return edges


def filter_short_track_components(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_FILTER_SHORT_TRACKS or OUTPUT_MIN_TRACK_LEN <= 1 or not edges:
        return nodes_by_id, edges

    parent = {node_id: node_id for node_id in nodes_by_id}

    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(a: int, b: int) -> None:
        if a not in parent or b not in parent:
            return
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[ra] = rb

    out_count: dict[int, int] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        union(source_id, target_id)
        out_count[source_id] = out_count.get(source_id, 0) + 1

    components: dict[int, list[int]] = {}
    for node_id in nodes_by_id:
        components.setdefault(find(node_id), []).append(node_id)

    component_edges: dict[int, list[dict[str, object]]] = {root: [] for root in components}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        if source_id in parent and target_id in parent:
            component_edges.setdefault(find(source_id), []).append(edge)

    keep: set[int] = set()
    for root, members in components.items():
        has_division = any(out_count.get(node_id, 0) >= 2 for node_id in members)
        if len(members) >= OUTPUT_MIN_TRACK_LEN or (
            OUTPUT_KEEP_DIVISION_COMPONENTS and has_division
        ):
            keep.update(members)

    if not keep:
        stats["short_track_filter_skipped_all"] += 1
        return nodes_by_id, edges

    removed_before_rescue = len(nodes_by_id) - len(keep)
    if removed_before_rescue <= 0:
        return nodes_by_id, edges

    if ADAPTIVE_SHORT_TRACK_RESCUE:
        removed_frac = removed_before_rescue / max(len(nodes_by_id), 1)
        if removed_frac >= SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC:
            budget = min(
                SHORT_TRACK_RESCUE_MAX_NODES_ABS,
                max(0, int(round(len(nodes_by_id) * SHORT_TRACK_RESCUE_MAX_NODES_FRAC))),
            )
            stats["short_track_rescue_triggered"] = 1
            stats["short_track_rescue_budget"] = budget
            proposals: list[tuple[float, int, float, int, list[int]]] = []
            for root, members in components.items():
                if set(members) & keep:
                    continue
                if (
                    len(members) < SHORT_TRACK_RESCUE_MIN_LEN
                    or len(members) >= OUTPUT_MIN_TRACK_LEN
                ):
                    continue
                c_edges = component_edges.get(root, [])
                if not c_edges:
                    continue
                probs: list[float] = []
                dists: list[float] = []
                for edge in c_edges:
                    try:
                        prob = float(edge.get("edge_prob", 0.0))
                    except (TypeError, ValueError):
                        prob = 0.0
                    if np.isfinite(prob):
                        probs.append(prob)
                    try:
                        dist = float(edge.get("distance_um", np.nan))
                    except (TypeError, ValueError):
                        dist = np.nan
                    if np.isfinite(dist):
                        dists.append(dist)
                mean_prob = float(np.mean(probs)) if probs else 0.0
                mean_dist = float(np.mean(dists)) if dists else float("inf")
                if mean_prob < SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB:
                    continue
                if mean_dist > SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM:
                    continue
                score = mean_prob - 0.02 * mean_dist + 0.004 * len(members)
                proposals.append((score, len(members), mean_prob, root, members))
            proposals.sort(reverse=True)
            rescued_nodes = 0
            rescued_components = 0
            for _, size, _, _, members in proposals:
                if budget <= 0 or rescued_nodes + size > budget:
                    continue
                keep.update(members)
                rescued_nodes += size
                rescued_components += 1
            stats["short_track_rescue_components"] = rescued_components
            stats["short_track_rescue_nodes"] = rescued_nodes

    removed_nodes = len(nodes_by_id) - len(keep)
    if removed_nodes <= 0:
        return nodes_by_id, edges

    kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id in keep}
    kept_edges = [
        edge
        for edge in edges
        if int(edge["source_id"]) in kept_nodes and int(edge["target_id"]) in kept_nodes
    ]
    stats["short_track_components_removed"] = sum(
        1 for members in components.values() if not (set(members) & keep)
    )
    stats["short_track_nodes_removed"] = removed_nodes
    stats["short_track_edges_removed"] = len(edges) - len(kept_edges)
    return kept_nodes, kept_edges


def linefit_smooth_output_graph(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> dict[int, dict[str, object]]:
    """Smooth linear track interiors without changing graph topology."""
    if (
        not OUTPUT_LINEFIT_SMOOTH
        or OUTPUT_LINEFIT_WEIGHT <= 0
        or OUTPUT_LINEFIT_WINDOW <= 0
        or not edges
    ):
        return nodes_by_id

    predecessor: dict[int, list[int]] = {}
    successor: dict[int, list[int]] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        if source is None or target is None:
            continue
        if int(target["t"]) != int(source["t"]) + 1:
            continue
        successor.setdefault(source_id, []).append(target_id)
        predecessor.setdefault(target_id, []).append(source_id)

    original_pos = {
        node_id: np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64)
        for node_id, node in nodes_by_id.items()
    }
    updated_pos: dict[int, np.ndarray] = {}
    weight = float(np.clip(OUTPUT_LINEFIT_WEIGHT, 0.0, 1.0))

    for node_id in sorted(nodes_by_id):
        neighbourhood: list[tuple[int, int]] = [(0, node_id)]

        current = node_id
        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            prev_ids = predecessor.get(current, [])
            if len(prev_ids) != 1:
                break
            current = prev_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((-step, current))

        current = node_id
        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            next_ids = successor.get(current, [])
            if len(next_ids) != 1:
                break
            current = next_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((step, current))

        if len(neighbourhood) < 3:
            stats["linefit_skipped_nodes"] += 1
            continue

        dts = np.array([delta for delta, _ in neighbourhood], dtype=np.float64)
        coords = np.stack([original_pos[nid] for _, nid in neighbourhood])
        fitted = np.array(
            [np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0) for axis in range(3)],
            dtype=np.float64,
        )
        if not np.isfinite(fitted).all():
            stats["linefit_skipped_nodes"] += 1
            continue
        updated_pos[node_id] = (1.0 - weight) * original_pos[node_id] + weight * fitted

    for node_id, pos in updated_pos.items():
        nodes_by_id[node_id]["z"] = float(pos[0])
        nodes_by_id[node_id]["y"] = float(pos[1])
        nodes_by_id[node_id]["x"] = float(pos[2])

    stats["linefit_smoothed_nodes"] = len(updated_pos)
    return nodes_by_id


AUDIT_STAGE_ROOT = WORKING_DIR / f"exp052_{EXP052_PHASE}" / "stages"
AUDIT_ARM = None


def _audit_capture_stage(dataset, stage, nodes_by_id, edges):
    if AUDIT_ARM is None:
        return
    destination = AUDIT_STAGE_ROOT / AUDIT_ARM / dataset
    destination.mkdir(parents=True, exist_ok=True)
    ids = sorted(nodes_by_id)
    node_rows = np.asarray(
        [
            [
                int(node_id),
                int(nodes_by_id[node_id]["t"]),
                float(nodes_by_id[node_id]["z"]),
                float(nodes_by_id[node_id]["y"]),
                float(nodes_by_id[node_id]["x"]),
            ]
            for node_id in ids
        ],
        dtype=np.float64,
    ).reshape(-1, 5)
    edge_rows = np.asarray(
        [
            [
                int(edge["source_id"]),
                int(edge["target_id"]),
                np.nan if edge.get("edge_prob") is None else float(edge["edge_prob"]),
            ]
            for edge in edges
        ],
        dtype=np.float64,
    ).reshape(-1, 3)
    np.savez_compressed(destination / f"{stage}.npz", nodes=node_rows, edges=edge_rows)


def filter_output_graph(
    nodes_by_id: dict[int, dict[str, object]],
    raw_edges: list[dict[str, object]],
    dataset: str | None = None,
    deepcenter_bundle: dict[str, object] | None = None,
    extra_edge_probs: dict[tuple[int, int], float] | None = None,
    fixed_readmitted_nodes: dict[int, dict[str, object]] | None = None,
    readmit_record: dict[int, dict[str, object]] | None = None,
    considered_pairs: set[tuple[int, int]] | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]], dict[str, int]]:
    _stage_t0 = _time.time()
    stats = {
        "raw_edges": len(raw_edges),
        "dropped_nonconsecutive_edges": 0,
        "dropped_long_edges": 0,
        "dropped_multi_parent_edges": 0,
        "dropped_multi_child_edges": 0,
        "dropped_division_edges": 0,
        "gap_candidates": 0,
        "gap_pairs_selected": 0,
        "gap_reused_existing": 0,
        "gap_inserted_synthetic": 0,
        "gap_added_nodes": 0,
        "gap_added_edges": 0,
        "gap_skipped_node_cap": 0,
        "gap_density_nodes_scored": 0,
        "gap_density_candidates_expanded": 0,
        "gap_density_candidates_restricted": 0,
        "gap_density_selected_outside_base": 0,
        "gap_density_step_delta_milli_sum": 0,
        "gap_refined_synthetic": 0,
        "gap_refine_failed": 0,
        "gap_refine_rejected_shift": 0,
        "pruned_isolated_nodes": 0,
        "motion_relink_edges": 0,
        "motion_relink_tight_edges": 0,
        "motion_relink_relaxed_edges": 0,
        "motion_relink_frames": 0,
        "motion_relink_replaced_raw_edges": 0,
        "motion_relink_fallback_raw": 0,
        "motion_relink_skipped_large_frame": 0,
        "gap2_candidates": 0,
        "gap2_pairs_selected": 0,
        "gap2_added_nodes": 0,
        "gap2_added_edges": 0,
        "gap2_skipped_cap": 0,
        "safe_division_candidates": 0,
        "safe_division_geometric_candidates": 0,
        "safe_divisions_added": 0,
        "safe_division_skipped_cap": 0,
        "safe_division_mutual_nn_rejected": 0,
        "safe_division_divergence_rejected": 0,
        "safe_division_symmetry_rejected": 0,
        "deepcenter_gap_checked": 0,
        "deepcenter_gap_bypassed_strong_motion": 0,
        "deepcenter_gap_bypassed_observed_node": 0,
        "deepcenter_gap_accepted": 0,
        "deepcenter_gap_rejected": 0,
        "deepcenter_gap_missing": 0,
        "deepcenter_safe_div_checked": 0,
        "deepcenter_safe_div_accepted": 0,
        "deepcenter_safe_div_rejected": 0,
        "deepcenter_safe_div_missing": 0,
        "short_track_components_removed": 0,
        "short_track_nodes_removed": 0,
        "short_track_edges_removed": 0,
        "short_track_filter_skipped_all": 0,
        "short_track_rescue_triggered": 0,
        "short_track_rescue_components": 0,
        "short_track_rescue_nodes": 0,
        "short_track_rescue_budget": 0,
        "linefit_smoothed_nodes": 0,
        "linefit_skipped_nodes": 0,
    }

    edges: list[dict[str, object]] = []
    for edge in raw_edges:
        source = nodes_by_id.get(int(edge["source_id"]))
        target = nodes_by_id.get(int(edge["target_id"]))
        if source is None or target is None:
            continue
        if OUTPUT_ENFORCE_NEXT_FRAME and int(target["t"]) != int(source["t"]) + 1:
            stats["dropped_nonconsecutive_edges"] += 1
            continue
        distance_um = edge_distance_um(source, target)
        edge["distance_um"] = distance_um
        if OUTPUT_EDGE_MAX_UM > 0 and distance_um > OUTPUT_EDGE_MAX_UM:
            stats["dropped_long_edges"] += 1
            continue
        edges.append(edge)

    _audit_capture_stage(dataset, "edge_filter", nodes_by_id, edges)

    if OUTPUT_MOTION_RELINK:
        learned_edge_probs: dict[tuple[int, int], float] = {}
        for edge in edges:
            prob = edge.get("edge_prob")
            if prob is None:
                continue
            try:
                prob = float(prob)
            except (TypeError, ValueError):
                continue
            if np.isfinite(prob):
                key = (int(edge["source_id"]), int(edge["target_id"]))
                learned_edge_probs[key] = max(learned_edge_probs.get(key, float("-inf")), prob)
        if extra_edge_probs:
            overlap = set(learned_edge_probs) & set(extra_edge_probs)
            if overlap:
                raise RuntimeError(f"extra score map includes {len(overlap)} selected edges")
            learned_edge_probs.update(extra_edge_probs)
        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs, considered_pairs)
        if READMIT_RADIUS_UM > 0 and (motion_edges or fixed_readmitted_nodes is not None):
            if fixed_readmitted_nodes is None:
                original_ids = set(nodes_by_id)
                nodes_by_id = readmit_discarded_detections(
                    nodes_by_id, motion_edges, stats, dataset=dataset
                )
                new_ids = set(nodes_by_id) - original_ids
                if readmit_record is not None:
                    readmit_record.update(
                        {node_id: dict(nodes_by_id[node_id]) for node_id in new_ids}
                    )
            else:
                if set(fixed_readmitted_nodes) & set(nodes_by_id):
                    raise RuntimeError("fixed readmitted node ID collides with initial graph")
                nodes_by_id.update(
                    {node_id: dict(node) for node_id, node in fixed_readmitted_nodes.items()}
                )
                new_ids = set(fixed_readmitted_nodes)
                stats["readmitted_nodes"] = len(new_ids)
            if new_ids:
                motion_edges = (
                    motion_relink_edges(nodes_by_id, stats, learned_edge_probs, considered_pairs)
                    or motion_edges
                )
        if motion_edges:
            stats["motion_relink_replaced_raw_edges"] = len(edges)
            edges = motion_edges
        else:
            stats["motion_relink_fallback_raw"] = 1

    if OUTPUT_SINGLE_PARENT_REPAIR and edges:
        best_by_target: dict[int, dict[str, object]] = {}
        for edge in edges:
            target_id = int(edge["target_id"])
            prev = best_by_target.get(target_id)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_target[target_id] = edge
        kept_ids = {id(edge) for edge in best_by_target.values()}
        stats["dropped_multi_parent_edges"] = sum(1 for edge in edges if id(edge) not in kept_ids)
        edges = [edge for edge in edges if id(edge) in kept_ids]

    if OUTPUT_SINGLE_CHILD_REPAIR and edges:
        best_by_source: dict[int, dict[str, object]] = {}
        for edge in edges:
            source_id = int(edge["source_id"])
            prev = best_by_source.get(source_id)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_source[source_id] = edge
        kept_ids = {id(edge) for edge in best_by_source.values()}
        stats["dropped_multi_child_edges"] = sum(1 for edge in edges if id(edge) not in kept_ids)
        edges = [edge for edge in edges if id(edge) in kept_ids]

    _audit_capture_stage(dataset, "relink", nodes_by_id, edges)
    print(
        f"  [{dataset}] after edge-filter+motion-relink: {len(nodes_by_id)} nodes, {len(edges)} edges | {_time.time() - _stage_t0:.1f}s"
    )
    repair_frame_cache: dict[int, np.ndarray] = {}
    deepcenter_heatmap_cache: dict[tuple[str, int], np.ndarray] = {}
    nodes_by_id, edges = close_single_frame_gaps(
        nodes_by_id,
        edges,
        stats,
        dataset=dataset,
        deepcenter_bundle=deepcenter_bundle,
        frame_cache=repair_frame_cache,
        deepcenter_cache=deepcenter_heatmap_cache,
    )
    _audit_capture_stage(dataset, "gap1", nodes_by_id, edges)
    nodes_by_id, edges = recover_strict_gap2(nodes_by_id, edges, stats, dataset=dataset)
    _audit_capture_stage(dataset, "gap2", nodes_by_id, edges)
    nodes_by_id, edges = fill_gaps_from_low_detections(
        nodes_by_id,
        edges,
        stats,
        dataset=dataset,
        frame_cache=repair_frame_cache,
    )
    _audit_capture_stage(dataset, "low_detection", nodes_by_id, edges)
    print(
        f"  [{dataset}] after gap-closing (single-frame + gap2 + low-detection filler): {len(nodes_by_id)} nodes, {len(edges)} edges | {_time.time() - _stage_t0:.1f}s"
    )
    edges = add_safe_divisions_postlink(
        nodes_by_id,
        edges,
        stats,
        dataset=dataset,
        deepcenter_bundle=deepcenter_bundle,
        frame_cache=repair_frame_cache,
        deepcenter_cache=deepcenter_heatmap_cache,
    )

    _audit_capture_stage(dataset, "division", nodes_by_id, edges)
    _geo_cands = stats["safe_division_geometric_candidates"]
    _post_veto_cands = stats["safe_division_candidates"]
    _rejected_by_dc = _geo_cands - _post_veto_cands
    print(
        f"  [{dataset}] after safe-division repair: {len(nodes_by_id)} nodes, {len(edges)} edges | {_time.time() - _stage_t0:.1f}s"
        f" (geometric_candidates={_geo_cands}, deepcenter_rejected={_rejected_by_dc},"
        f" post_veto_candidates={_post_veto_cands}, added={stats['safe_divisions_added']},"
        f" cap_skipped={stats['safe_division_skipped_cap']},"
        f" mutual_nn_rejected={stats['safe_division_mutual_nn_rejected']},"
        f" divergence_rejected={stats['safe_division_divergence_rejected']})"
    )
    if OUTPUT_DIVISION_GEOMETRY_FILTER and edges:
        by_source: dict[int, list[dict[str, object]]] = {}
        for edge in edges:
            by_source.setdefault(int(edge["source_id"]), []).append(edge)

        filtered: list[dict[str, object]] = []
        for source_id, source_edges in by_source.items():
            if len(source_edges) <= 1:
                filtered.extend(source_edges)
                continue

            ranked = sorted(source_edges, key=edge_sort_key, reverse=True)
            source = nodes_by_id[source_id]
            top1 = ranked[0]
            top2 = ranked[1]
            d1 = float(top1["distance_um"])
            d2 = float(top2["distance_um"])
            sister = edge_distance_um(
                nodes_by_id[int(top1["target_id"])], nodes_by_id[int(top2["target_id"])]
            )
            valid_division = (
                max(d1, d2) <= DIV_PARENT_MAX_UM
                and sister <= DIV_SISTER_MAX_UM
                and int(nodes_by_id[int(top1["target_id"])]["t"]) == int(source["t"]) + 1
                and int(nodes_by_id[int(top2["target_id"])]["t"]) == int(source["t"]) + 1
            )
            if valid_division:
                filtered.extend([top1, top2])
                stats["dropped_division_edges"] += max(0, len(ranked) - 2)
            elif DIV_DROP_TO_SINGLE_IF_BAD:
                filtered.append(top1)
                stats["dropped_division_edges"] += len(ranked) - 1
            else:
                filtered.extend(ranked)
        edges = filtered

    if OUTPUT_PRUNE_ISOLATED:
        incident = {int(edge["source_id"]) for edge in edges} | {
            int(edge["target_id"]) for edge in edges
        }
        if incident:
            kept_nodes = {
                node_id: node for node_id, node in nodes_by_id.items() if node_id in incident
            }
            stats["pruned_isolated_nodes"] = len(nodes_by_id) - len(kept_nodes)
            nodes_by_id = kept_nodes
            edges = [
                edge
                for edge in edges
                if int(edge["source_id"]) in nodes_by_id and int(edge["target_id"]) in nodes_by_id
            ]

    print(
        f"  [{dataset}] after division-geometry-filter+prune-isolated: {len(nodes_by_id)} nodes, {len(edges)} edges | {_time.time() - _stage_t0:.1f}s"
    )
    nodes_by_id, edges = filter_short_track_components(nodes_by_id, edges, stats)
    print(
        f"  [{dataset}] after short-track filtering: {len(nodes_by_id)} nodes, {len(edges)} edges"
        f" (components_removed={stats['short_track_components_removed']})"
    )
    _audit_capture_stage(dataset, "short_track", nodes_by_id, edges)
    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)
    _audit_capture_stage(dataset, "final", nodes_by_id, edges)
    print(
        f"  [{dataset}] FINAL: {len(nodes_by_id)} nodes, {len(edges)} edges | {_time.time() - _stage_t0:.1f}s"
    )

    return nodes_by_id, edges, stats


DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()


# %% [markdown]
# ## 5. Score once, retain candidate edges, and solve paired graphs

# %%
import hashlib as _audit_hashlib
import math as _audit_math
import shutil as _audit_shutil
from collections import defaultdict as _audit_defaultdict

sys.path.insert(0, str(REPO_DIR / "src"))
from biohub_tracking.io import save_graph as _audit_save_graph
from biohub_tracking.metrics import summarise as _audit_summarise
import polars as _audit_pl

AUDIT_ROOT = WORKING_DIR / f"exp052_{EXP052_PHASE}"
AUDIT_ROOT.mkdir(exist_ok=True)
AUDIT_ARMS = ("baseline", "candidate_scores")
AUDIT_EXPECTED_EVALUATOR_SHA = "614813cc51c3581c6ccda4bb20725a19da8ecac4a27620654bfca58319cffa3c"
AUDIT_START = time.time()
# A partial algorithm after the parent's deadline fallback is not comparable.
REPAIR_DEADLINE_S = float("inf")


def _audit_sha256(path: Path) -> str:
    return _audit_hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _audit_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_audit_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _audit_jsonable(value.item())
    if isinstance(value, float) and not _audit_math.isfinite(value):
        return None
    return value


def _audit_run_prediction(arm: str, label: str, stems: list[str]) -> float:
    if arm not in AUDIT_ARMS or not stems or len(stems) != len(set(stems)):
        raise ValueError((arm, label, stems))
    split = REPO_DIR / f"exp052_{arm}_{label}_split.json"
    split.write_text(json.dumps([{"split": 0, "train": [], "test": sorted(stems)}], indent=2))
    command = list(predict_cmd)
    split_arg = command.index("--splits") + 1
    command[split_arg] = split.name
    method = f"exp052_{arm}_{label}"
    command.extend(["--method", method])
    arm_root = AUDIT_ROOT / arm
    cache_root = arm_root / "candidate_cache"
    raw_root = arm_root / "raw_detector_coordinates"
    cache_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "PYTHONPATH": "src",
        "V1284_MODE": "candidate",
        "V1284_HEAD": str(_SELF_HEAD_PATH),
        "BIOHUB_CACHE_DIR": str(cache_root),
        "BIOHUB_CACHE_EDGE_THRESHOLD": "1.0e-8",
        "BIOHUB_AUDIT_RAW_DIR": str(raw_root),
        "BIOHUB_AUDIT_FEATURE_LOG": str(arm_root / "interpolated_feature_hashes.jsonl"),
        "BIOHUB_DIAGNOSTIC_ARM": f"exp052_{arm}_{label}",
    }
    start = time.time()
    subprocess.run(command, cwd=REPO_DIR, env=environment, check=True)
    elapsed = time.time() - start
    raw_graph_dir = _prediction_dir_for_method(method)
    output_dir = arm_root / "raw_ilp_graphs"
    output_dir.mkdir(exist_ok=True)
    observed = {path.stem for path in raw_graph_dir.glob("*.geff")}
    if observed != set(stems):
        raise RuntimeError(
            {"arm": arm, "batch": label, "observed": sorted(observed), "expected": sorted(stems)}
        )
    for source in sorted(raw_graph_dir.glob("*.geff")):
        destination = output_dir / source.name
        if destination.exists():
            raise RuntimeError(f"duplicate raw graph: {destination}")
        _audit_shutil.copytree(source, destination)
        _cache_file = cache_root / f"{source.stem}.npz"
        if not _cache_file.is_file():
            raise RuntimeError(f"candidate-score cache missing: {source.stem}")
        with np.load(_cache_file, allow_pickle=False) as _cache_check:
            _required = {
                "coords",
                "edge_src",
                "edge_tgt",
                "edge_prob",
                "admitted",
                "low_coords",
                "low_score",
            }
            if not _required <= set(_cache_check.files):
                raise RuntimeError(
                    {
                        "incomplete_prediction_cache": source.stem,
                        "missing": sorted(_required - set(_cache_check.files)),
                    }
                )
            if len(_cache_check["low_coords"]) != len(_cache_check["low_score"]):
                raise RuntimeError(f"{source.stem}: low-detection cache length mismatch")
    print(f"{arm} {label}: predicted {len(stems)} videos in {elapsed:.1f}s", flush=True)
    return elapsed


def _audit_save_official_graph(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    destination: Path,
) -> None:
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, _audit_pl.Float64, -999999.0)
    old_ids = sorted(nodes_by_id)
    nodes = [
        {
            "t": int(nodes_by_id[old]["t"]),
            "z": max(0, int(round(float(nodes_by_id[old]["z"])))),
            "y": max(0, int(round(float(nodes_by_id[old]["y"])))),
            "x": max(0, int(round(float(nodes_by_id[old]["x"])))),
        }
        for old in old_ids
    ]
    new_ids = [int(value) for value in graph.bulk_add_nodes(nodes)]
    old_to_new = dict(zip(old_ids, new_ids, strict=True))
    if edges:
        graph.bulk_add_edges(
            [
                {
                    "source_id": old_to_new[int(edge["source_id"])],
                    "target_id": old_to_new[int(edge["target_id"])],
                }
                for edge in edges
            ]
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _audit_save_graph(graph, destination)


_EXP052_READMIT_BY_STEM: dict[str, dict[int, dict[str, object]]] = {}
_EXP052_SCORE_AUDITS: dict[str, dict[str, object]] = {}
_EXP052_INITIAL_IDS: dict[str, set[int]] = {}


def _audit_postprocess(arm: str, stems: list[str]) -> float:
    global AUDIT_ARM
    AUDIT_ARM = arm
    os.environ["BIOHUB_CACHE_DIR"] = str(AUDIT_ROOT / "baseline" / "candidate_cache")
    start = time.time()
    for stem in stems:
        source = AUDIT_ROOT / "baseline" / "raw_ilp_graphs" / f"{stem}.geff"
        graph = graph_from_geff(source)
        nodes_by_id = {
            int(row["node_id"]): {
                "node_id": int(row["node_id"]),
                "t": int(row["t"]),
                "z": float(row["z"]),
                "y": float(row["y"]),
                "x": float(row["x"]),
            }
            for row in graph.node_attrs().iter_rows(named=True)
        }
        raw_edges = [
            {
                "source_id": int(row["source_id"]),
                "target_id": int(row["target_id"]),
                "edge_prob": None if row.get("edge_prob") is None else float(row["edge_prob"]),
            }
            for row in graph.edge_attrs().iter_rows(named=True)
        ]
        _EXP052_INITIAL_IDS[stem] = set(nodes_by_id)
        _audit_capture_stage(stem, "ilp", nodes_by_id, raw_edges)
        score_map = None
        fixed_readmit = None
        readmit_record: dict[int, dict[str, object]] = {}
        if arm == "candidate_scores":
            fixed_readmit = _EXP052_READMIT_BY_STEM[stem]
            score_map, audit = _exp052_score_map(stem, nodes_by_id, raw_edges)
            _EXP052_SCORE_AUDITS[stem] = audit
        considered: set[tuple[int, int]] = set()
        final_nodes, final_edges, stats = filter_output_graph(
            nodes_by_id,
            raw_edges,
            dataset=stem,
            deepcenter_bundle=DEEPCENTER_VETO_DETECTOR,
            extra_edge_probs=score_map,
            fixed_readmitted_nodes=fixed_readmit,
            readmit_record=readmit_record if arm == "baseline" else None,
            considered_pairs=considered,
        )
        if arm == "baseline":
            _EXP052_READMIT_BY_STEM[stem] = readmit_record
        else:
            if stats.get("readmitted_nodes", 0) != len(fixed_readmit):
                raise RuntimeError(f"{stem}: treatment did not preserve readmitted nodes")
            audit["new_scores_considered"] = len(considered & set(score_map))
            audit["readmitted_pairs_considered"] = sum(
                source_id in fixed_readmit or target_id in fixed_readmit
                for source_id, target_id in considered
            )
            audit["missing_score_reason_unknown_on_initial_nodes"] = sum(
                source_id in _EXP052_INITIAL_IDS[stem]
                and target_id in _EXP052_INITIAL_IDS[stem]
                and (source_id, target_id) not in audit["all_saved_pairs"]
                for source_id, target_id in considered
            )
            audit.pop("all_saved_pairs")
        if _deadline_degraded or not final_nodes:
            raise RuntimeError(f"{stem}: fixed postprocessing degraded or produced no nodes")
        destination = AUDIT_ROOT / arm / "final_graphs" / f"{stem}.geff"
        _audit_save_official_graph(final_nodes, final_edges, destination)
        (AUDIT_ROOT / arm / "stages" / stem).mkdir(parents=True, exist_ok=True)
        (AUDIT_ROOT / arm / "stages" / stem / "repair_stats.json").write_text(
            json.dumps(_audit_jsonable(stats), indent=2, sort_keys=True) + "\n"
        )
    return time.time() - start


def _audit_load_stage_edges(
    arm: str, stem: str, stage: str, original_ids: set[int] | None = None
) -> set[tuple[int, int]]:
    path = AUDIT_STAGE_ROOT / arm / stem / f"{stage}.npz"
    with np.load(path, allow_pickle=False) as payload:
        edges = np.asarray(payload["edges"], dtype=np.float64).reshape(-1, 3)
    pairs = {(int(row[0]), int(row[1])) for row in edges}
    return pairs if original_ids is None else trace_original_edges(pairs, original_ids)


def _audit_gt_graph(stem: str) -> tuple[np.ndarray, set[tuple[int, int]]]:
    graph = graph_from_geff(TRAIN_DIR / f"{stem}.geff")
    node_rows = np.asarray(
        [
            [int(row["node_id"]), int(row["t"]), float(row["z"]), float(row["y"]), float(row["x"])]
            for row in graph.node_attrs().iter_rows(named=True)
        ],
        dtype=np.float64,
    ).reshape(-1, 5)
    edge_set = {
        (int(row["source_id"]), int(row["target_id"]))
        for row in graph.edge_attrs().iter_rows(named=True)
    }
    return node_rows, edge_set


# %% [markdown]
# ## 5. Paired relink replay: verify the original cache and extend only the score map

# %%
from settings import load_config as _exp052_load_config

_EXP052_PARAMS = _exp052_load_config()["model"]["params"]
if (
    float(_EXP052_PARAMS["ilp_edge_probability_strictly_greater_than"]) != 0.48
    or float(_EXP052_PARAMS["score_cache_probability_strictly_greater_than"]) != 1.0e-8
    or float(_EXP052_PARAMS["missing_score_fallback"]) != 0.0
    or not _EXP052_PARAMS["fix_readmitted_nodes_from_baseline"]
    or int(_EXP052_PARAMS["ilp_timeout_seconds_per_video"]) != 1200
):
    raise RuntimeError("relink score policy and config.yaml differ")
if EXP052_PHASE == "inference" and not EXP052_INTERNAL_GATE_RECEIPT_SHA:
    raise RuntimeError("internal diagnostic gate receipt SHA must be recorded before inference")


def _exp052_sha_tree(path: Path) -> str:
    digest = _audit_hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(path)).encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


def _exp052_score_map(
    stem: str,
    nodes_by_id: dict[int, dict[str, object]],
    raw_edges: list[dict[str, object]],
) -> tuple[dict[tuple[int, int], float], dict[str, object]]:
    path = AUDIT_ROOT / "baseline" / "candidate_cache" / f"{stem}.npz"
    with np.load(path, allow_pickle=False) as cache:
        coords = np.asarray(cache["coords"], dtype=np.float64)
        sources = np.asarray(cache["edge_src"], dtype=np.int64)
        targets = np.asarray(cache["edge_tgt"], dtype=np.int64)
        probs = np.asarray(cache["edge_prob"], dtype=np.float64)
        admitted = np.asarray(cache["admitted"], dtype=np.float64).reshape(-1, 4)
    node_rows = np.array(
        [
            [node_id, node["t"], node["z"], node["y"], node["x"]]
            for node_id, node in sorted(nodes_by_id.items())
        ],
        dtype=np.float64,
    )
    original_ids = validate_initial_graph_ids(coords, node_rows)
    if not (len(sources) == len(targets) == len(probs)):
        raise RuntimeError(f"{stem}: cache score array lengths differ")
    if (
        np.any(sources < 0)
        or np.any(targets < 0)
        or np.any(sources >= len(coords))
        or np.any(targets >= len(coords))
        or not np.isfinite(probs).all()
        or np.any(probs <= 1.0e-8)
        or np.any(probs > 1.0)
    ):
        raise RuntimeError(f"{stem}: invalid score cache")
    if np.any(coords[sources, 0] + 1 != coords[targets, 0]):
        raise RuntimeError(f"{stem}: nonadjacent score cache pair")
    cached = {
        (int(src), int(tgt)): float(prob)
        for src, tgt, prob in zip(sources, targets, probs, strict=True)
    }
    if len(cached) != len(probs):
        raise RuntimeError(f"{stem}: duplicate score pair")
    selected = {}
    for edge in raw_edges:
        pair = (int(edge["source_id"]), int(edge["target_id"]))
        if pair in selected or edge.get("edge_prob") is None:
            raise RuntimeError(f"{stem}: invalid selected edge")
        selected[pair] = float(edge["edge_prob"])
    admitted_pairs = {(int(row[0]), int(row[1])) for row in admitted}
    if not set(selected) <= admitted_pairs:
        raise RuntimeError(f"{stem}: ILP-selected edge absent from cache candidates")
    for row in admitted:
        pair = (int(row[0]), int(row[1]))
        saved = cached.get(pair)
        if saved is None or not np.isclose(saved, float(row[2]), rtol=2e-6, atol=1e-8):
            raise RuntimeError(f"{stem}: original candidate score differs from cache")
    for pair, value in selected.items():
        saved = cached.get(pair)
        if saved is None or not np.isclose(saved, value, rtol=2e-6, atol=1e-8):
            raise RuntimeError(f"{stem}: selected edge score differs from original cache")
    extra = {
        pair: prob
        for pair, prob in cached.items()
        if pair not in selected and pair[0] in original_ids and pair[1] in original_ids
    }
    return extra, {
        "score_cache_sha256": _audit_sha256(path),
        "score_cache_threshold_strict": 1.0e-8,
        "all_saved_pairs": set(cached),
        "saved_score_count": len(cached),
        "selected_edge_count": len(selected),
        "new_scores_available_on_initial_nodes": len(extra),
        "original_ilp_node_count": len(original_ids),
    }


# The baseline is scored once. Both arms read the exact same ILP graph and score cache.
_PILOT = sorted(stems[0] for stems in _selection.values())
_PILOT_PREDICTION_SECONDS = _audit_run_prediction("baseline", "inference_pilot", _PILOT)
_PREDICTION_ESTIMATE_SECONDS = _PILOT_PREDICTION_SECONDS * len(test_stems) / len(_PILOT) * 1.5
if _PREDICTION_ESTIMATE_SECONDS >= 8 * 3600:
    raise RuntimeError({"new_video_prediction_budget_gate_failed": _PREDICTION_ESTIMATE_SECONDS})
_REMAINING = [stem for stem in test_stems if stem not in _PILOT]
_PREDICTION_SECONDS = _PILOT_PREDICTION_SECONDS + _audit_run_prediction(
    "baseline", "inference_remaining", _REMAINING
)
_PROCESSING = {}
_ORDER = [*_PILOT, *_REMAINING]
for _index, _stem in enumerate(_ORDER):
    baseline_seconds = _audit_postprocess("baseline", [_stem])
    candidate_seconds = _audit_postprocess("candidate_scores", [_stem])
    initial_ids = _EXP052_INITIAL_IDS[_stem]
    baseline_ilp = _audit_load_stage_edges("baseline", _stem, "ilp", initial_ids)
    candidate_ilp = _audit_load_stage_edges("candidate_scores", _stem, "ilp", initial_ids)
    if baseline_ilp != candidate_ilp:
        raise RuntimeError(f"{_stem}: ILP graph changed between arms")
    baseline_relink = _audit_load_stage_edges("baseline", _stem, "relink")
    candidate_relink = _audit_load_stage_edges("candidate_scores", _stem, "relink")
    changed_edges = len(baseline_relink ^ candidate_relink)
    _PROCESSING[_stem] = {
        "baseline_repair_seconds": baseline_seconds,
        "candidate_repair_seconds": candidate_seconds,
        "relink_symmetric_difference": changed_edges,
        "score_audit": _EXP052_SCORE_AUDITS[_stem],
        "baseline_readmitted_nodes": len(_EXP052_READMIT_BY_STEM[_stem]),
    }
    print("PROCESSED", _stem, _PROCESSING[_stem], flush=True)
    if _index == 1:
        elapsed = time.time() - AUDIT_START
        pair_cost = (elapsed - _PREDICTION_SECONDS) / 2
        estimate = elapsed + (len(test_stems) - 2) * pair_cost * 1.25
        if estimate >= 12 * 3600:
            raise RuntimeError({"notebook_runtime_gate_failed": estimate})

# %% [markdown]
# ## 6. Compare the saved scores, relink edges, and known annotated edges

# %%
_POSTPROCESS_STAGES = (
    "edge_filter",
    "relink",
    "gap1",
    "gap2",
    "low_detection",
    "division",
    "short_track",
    "final",
)
_DIAGNOSTIC_STAGES = ("saved_score", "ilp", *_POSTPROCESS_STAGES)
_DIAGNOSTIC = {}
for _stem in test_stems:
    with np.load(
        AUDIT_ROOT / "baseline" / "candidate_cache" / f"{_stem}.npz",
        allow_pickle=False,
    ) as cache:
        coords = np.asarray(cache["coords"], dtype=np.float32)
        scored_pairs = set(
            zip(
                cache["edge_src"].astype(int).tolist(),
                cache["edge_tgt"].astype(int).tolist(),
                strict=True,
            )
        )
    gt_nodes, gt_edges = _audit_gt_graph(_stem)
    match = match_known_centers(
        np.column_stack((np.arange(len(coords)), coords)),
        gt_nodes,
        radius_um=7.0,
    )
    stage_by_arm = {}
    for arm in AUDIT_ARMS:
        stages = {
            "saved_score": scored_pairs,
            "ilp": _audit_load_stage_edges(arm, _stem, "ilp", _EXP052_INITIAL_IDS[_stem]),
        }
        stages.update(
            {
                stage: _audit_load_stage_edges(arm, _stem, stage, _EXP052_INITIAL_IDS[_stem])
                for stage in _POSTPROCESS_STAGES
            }
        )
        stage_by_arm[arm] = readout_stages(gt_edges, match, stages)
    _DIAGNOSTIC[_stem] = {
        "embryo": _stem.split("_", 1)[0],
        "matched_gt_nodes": len(match.gt_to_candidate),
        "ambiguous_gt": match.ambiguous_gt,
        "arms": stage_by_arm,
        "relink_symmetric_difference": _PROCESSING[_stem]["relink_symmetric_difference"],
        "new_scores_considered": _PROCESSING[_stem]["score_audit"]["new_scores_considered"],
    }
(AUDIT_ROOT / "fixed_id_diagnostic.json").write_text(
    json.dumps(_audit_jsonable(_DIAGNOSTIC), indent=2, sort_keys=True) + "\n"
)
_BY_EMBRYO = {}
for embryo in ("44b6", "6bba"):
    group = [row for row in _DIAGNOSTIC.values() if row["embryo"] == embryo]
    if len(group) != 10:
        raise RuntimeError(f"{embryo}: expected ten videos")
    _BY_EMBRYO[embryo] = {
        "relink_symmetric_difference": sum(row["relink_symmetric_difference"] for row in group),
        "new_scores_considered": sum(row["new_scores_considered"] for row in group),
        "stages": {
            arm: {
                stage: {
                    key: sum(row["arms"][arm][stage][key] for row in group)
                    for key in group[0]["arms"][arm][stage]
                }
                for stage in _DIAGNOSTIC_STAGES
            }
            for arm in AUDIT_ARMS
        },
    }
(AUDIT_ROOT / "fixed_id_by_embryo.json").write_text(
    json.dumps(_audit_jsonable(_BY_EMBRYO), indent=2, sort_keys=True) + "\n"
)

# %% [markdown]
# ## 7. Run the public evaluator on both complete graphs

# %%
sys.path.insert(0, str(REPO_DIR / "scripts"))
sys.path.insert(0, str(REPO_DIR / "src"))
_evaluator_path = REPO_DIR / "scripts" / "evaluate.py"
if _audit_sha256(_evaluator_path) != AUDIT_EXPECTED_EVALUATOR_SHA:
    raise RuntimeError("public evaluator source SHA changed")
import evaluate as _audit_evaluator

if Path(_audit_evaluator.DATA_DIR).resolve() != TRAIN_DIR.resolve():
    raise RuntimeError("public evaluator points to a different GT directory")


def _exp052_evaluate(arm):
    geffs = sorted((AUDIT_ROOT / arm / "final_graphs").glob("*.geff"))
    if [path.stem for path in geffs] != sorted(test_stems):
        raise RuntimeError(f"{arm}: incomplete final graph coverage")
    run = {
        "username": "exp052",
        "method": arm,
        "split": "split_0",
        "dir": AUDIT_ROOT / arm / "final_graphs",
        "geffs": geffs,
    }
    rows = _audit_evaluator.evaluate_run(run, max_distance=7.0)
    if len(rows) != 20 or {str(row["dataset"]) for row in rows} != set(test_stems):
        raise RuntimeError("public evaluator returned incomplete rows")
    overall = _audit_summarise(rows)
    by_embryo = {
        embryo: _audit_summarise(
            [row for row in rows if str(row["dataset"]).startswith(embryo + "_")]
        )
        for embryo in ("44b6", "6bba")
    }
    return {"overall": overall, "by_embryo": by_embryo, "rows": rows}


_OFFICIAL = {arm: _exp052_evaluate(arm) for arm in AUDIT_ARMS}
(AUDIT_ROOT / "official_metric.json").write_text(
    json.dumps(_audit_jsonable(_OFFICIAL), indent=2, sort_keys=True) + "\n"
)
print("OFFICIAL", {arm: _OFFICIAL[arm]["overall"] for arm in AUDIT_ARMS}, flush=True)

# %% [markdown]
# ## 8. Verify the internal gate and save runtime evidence

# %%
_EXPECTED_EXP047_BASELINE_HASHES = {
    "44b6_1d530831": "e58ac68100366900df6d9a8147635b1bfc1dd28209535780ba7e6f0cce182934",
    "44b6_2a2eff9f": "0a22ece9c0b47acdcf3fdf28edfb6ee357294ff4684fd45189cfd063d8d1fa44",
    "44b6_668e0cc7": "0806d9a91494854e728aa4b25fcc07cb3c63c4947c18f2995615a4322696735f",
    "44b6_706092f0": "5a3818e181c37f1c8dbd3c87542f5d6ec68fa7e252a1ac7a84b50156593b5f82",
    "44b6_7a302da0": "f9c2afceb03f8d94a07ce0fa182e10bc4bca1caeca3a304536ad0f70736c3608",
    "44b6_87bba6c4": "c42fefbdef9c0469aaaa21956b911ef93c40c060cbcbb538f19ad66297150226",
    "44b6_8f9ecab4": "44e3d3f65823e26db8f786292deb7cdb16c7653f19ad67f8716d90a8d914650f",
    "44b6_97725144": "ea68913692c3535e0a5b0a919f93db98190da5ea14738f519a67cfeb1764441f",
    "44b6_996155de": "d3b4254653af18bdb014ab0ff36f4ca353713aa8ff359b218d42ecc5cd97c149",
    "44b6_cf8fed6b": "90715b4d15b86e24159f1b7d737437336f9c3a30a401989c16db580b60b4dcda",
    "6bba_07e24132": "2a073174dace3bc5f4051a4d891156f6481d02732e06578ad36acbe091eadd69",
    "6bba_2819ca14": "613f8f5a96ee1c7963e99068c056ae0279dfea8ffa5cf75fef00e7936f632910",
    "6bba_474be664": "7483a0a66c3df1ae8655200783e5d568c4bb074dd6b1e4a35bc97c5912d5d4b3",
    "6bba_57b7cc1e": "49b01a5f9744f6e9da1c5e257656ef26316baf72ba0d7da9db88d9f2ecd39216",
    "6bba_5dfe9ad1": "814cb5ca758003da09d193ee8975e9c22fe4dbd3ea64dca3dc66d904dad9a608",
    "6bba_67ebd073": "746c73a122f5142de027510d70a4e8632733b15cc885317146b35ef33fe5824c",
    "6bba_705ec2c9": "08e4946d4fc9ceaa161fd9dfba15804e2e1c382ba12cb5dbf5a47ef291831de3",
    "6bba_af149c94": "117dc5375c86e33d7e68698e42b7a0608378a3f85107f96196f3d63bf7474634",
    "6bba_d2b9fc0c": "730d878c090ffe02d95402d296c0d18abd5d80926ce05dbcd168bf2ea568891b",
    "6bba_fc5f39dc": "1abd54d829dd7cbd403651d79b67a9df5553a8bb59afe59deafebd409c4700a7",
}
if EXP052_PHASE == "diagnostic":
    baseline_hash_match = {
        stem: _exp052_sha_tree(AUDIT_ROOT / "baseline" / "final_graphs" / f"{stem}.geff")
        == _EXPECTED_EXP047_BASELINE_HASHES[stem]
        for stem in test_stems
    }
    gate_passed = all(baseline_hash_match.values()) and all(
        _BY_EMBRYO[embryo]["new_scores_considered"] > 0
        and _BY_EMBRYO[embryo]["relink_symmetric_difference"] > 0
        for embryo in ("44b6", "6bba")
    )
    gate = {
        "phase": "diagnostic",
        "passed": bool(gate_passed),
        "baseline_graph_hash_match": baseline_hash_match,
        "by_embryo": {
            embryo: {
                "new_scores_considered": _BY_EMBRYO[embryo]["new_scores_considered"],
                "relink_symmetric_difference": _BY_EMBRYO[embryo]["relink_symmetric_difference"],
            }
            for embryo in ("44b6", "6bba")
        },
    }
else:
    gate = {
        "phase": "inference",
        "internal_gate_receipt_sha256": EXP052_INTERNAL_GATE_RECEIPT_SHA,
    }
(AUDIT_ROOT / "gate.json").write_text(
    json.dumps(_audit_jsonable(gate), indent=2, sort_keys=True) + "\n"
)
memory_peak_path = Path("/sys/fs/cgroup/memory.peak")
memory_limit_path = Path("/sys/fs/cgroup/memory.max")
memory_peak = int(memory_peak_path.read_text()) if memory_peak_path.exists() else None
memory_limit_text = memory_limit_path.read_text().strip() if memory_limit_path.exists() else ""
memory_limit = int(memory_limit_text) if memory_limit_text.isdigit() else None
if memory_peak is not None and memory_limit is not None and memory_peak >= 0.80 * memory_limit:
    raise RuntimeError({"memory_limit_gate_failed": {"peak": memory_peak, "limit": memory_limit}})
elapsed = time.time() - AUDIT_START
if elapsed >= 12 * 3600:
    raise RuntimeError({"notebook_limit_failed": elapsed})
_RECEIPT = {
    "status": "evaluated",
    "phase": EXP052_PHASE,
    "selected_videos": _selection,
    "head_sha256": _SELF_HEAD_SHA,
    "public_evaluator_sha256": AUDIT_EXPECTED_EVALUATOR_SHA,
    "prediction_seconds": _PREDICTION_SECONDS,
    "pilot_prediction_seconds": _PILOT_PREDICTION_SECONDS,
    "prediction_estimate_seconds": _PREDICTION_ESTIMATE_SECONDS,
    "elapsed_seconds": elapsed,
    "memory_peak_bytes": memory_peak,
    "memory_limit_bytes": memory_limit,
    "processing": _PROCESSING,
    "gate": gate,
    "official_metric_sha256": _audit_sha256(AUDIT_ROOT / "official_metric.json"),
    "fixed_id_diagnostic_sha256": _audit_sha256(AUDIT_ROOT / "fixed_id_diagnostic.json"),
    "final_graph_sha256": {
        arm: {
            stem: _exp052_sha_tree(AUDIT_ROOT / arm / "final_graphs" / f"{stem}.geff")
            for stem in test_stems
        }
        for arm in AUDIT_ARMS
    },
}
(AUDIT_ROOT / "receipt.json").write_text(
    json.dumps(_audit_jsonable(_RECEIPT), indent=2, sort_keys=True) + "\n"
)
print("EXP052", EXP052_PHASE, "COMPLETE", flush=True)
