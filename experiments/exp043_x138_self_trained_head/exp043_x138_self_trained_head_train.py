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
# # exp043 x138 self-trained coordinate head
#
# Fixed public image models and tracker; train the coordinate head from GEFF.

# %% [markdown]
# ## 1. Public x138 configuration and fixed artifacts

# %%
'''Biohub Harmonic Fusion

Production 3D lineage reconstruction with dual temporal models,
dual edge-feature TTA, and geometry-validated divisions.

Record edition.'''

import os
BIOHUB_PRESET = 'harmonic_v3_division_wide'
BIOHUB_SCORE_AXIS = 'public 0.939 base + holdout-selected post-process configuration'

os.environ["BIOHUB_OUTPUT_FILTER_SHORT_TRACKS"] = "1"
os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"
os.environ["BIOHUB_MOTION_RELINK_LEARNED_BONUS"] = '1.0'
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
os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt"
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
os.environ["BIOHUB_VALIDATOR_ENABLE"] = "0"          # in-sample train proxy, ~11 min of GPU per run
os.environ["BIOHUB_ILP_TIMEOUT_S"] = "1200"           # per dataset; SCIP keeps its incumbent at the limit
os.environ["BIOHUB_REPAIR_DEADLINE_S"] = "27000"      # 7.5 h after kernel start the repair loop degrades
os.environ["BIOHUB_FRAME_CACHE_MAX_FRAMES"] = "48"
os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"   # ppsweep_selected.json of the public run (tight55)
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
os.environ["BIOHUB_CACHE_DIR"] = "/kaggle/working/edge_cache"
os.environ["BIOHUB_CACHE_EDGE_THRESHOLD"] = "1.0"
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
    raise RuntimeError(
        "Configuration drift detected: " + _guard_json.dumps(_drift, sort_keys=True)
    )

print("Configuration guard: PASS")
print("Baseline: fixed-90 dual-seed clean pipeline (public LB 0.913)")
print("Single model-level change: harmonic mutual-support association fusion")
print("Reverse-time association weight: 0.200")

# %%
from __future__ import annotations

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
TARGET_ARTIFACT_SLUG = os.environ.get("BIOHUB_TARGET_ARTIFACT_SLUG", "biohub-tracking-support-pack-50ep-v1")
PRIMARY_ARTIFACT_MANIFEST = Path(os.environ.get(
    "BIOHUB_PRIMARY_ARTIFACT_MANIFEST",
    "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/ARTIFACT_MANIFEST.json",
))
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
MOTION_RELINK_FLOW_SEED_GATE_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_SEED_GATE_UM", "0"))
MOTION_RELINK_FLOW_RAW_ADMIT = os.environ.get("BIOHUB_MOTION_RELINK_FLOW_RAW_ADMIT", "1").strip() != "0"
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

OUTPUT_DIVISION_GEOMETRY_FILTER = os.environ.get("BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER", "0") != "0"
DIV_PARENT_MAX_UM = float(os.environ.get("BIOHUB_DIV_PARENT_MAX_UM", "10.5"))
DIV_SISTER_MAX_UM = float(os.environ.get("BIOHUB_DIV_SISTER_MAX_UM", "8.0"))
DIV_DROP_TO_SINGLE_IF_BAD = os.environ.get("BIOHUB_DIV_DROP_TO_SINGLE_IF_BAD", "1") != "0"
OUTPUT_GAP_CLOSE = os.environ.get("BIOHUB_OUTPUT_GAP_CLOSE", "1") != "0"
GAP_CLOSE_MAX_GAP = int(os.environ.get("BIOHUB_GAP_CLOSE_MAX_GAP", "1"))
GAP_CLOSE_UM = float(os.environ.get("BIOHUB_GAP_CLOSE_UM", "6.0"))
GAP_DENSITY_ADAPTIVE = os.environ.get("BIOHUB_GAP_DENSITY_ADAPTIVE", "0") != "0"
GAP_DENSITY_REFERENCE_UM = float(os.environ.get("BIOHUB_GAP_DENSITY_REFERENCE_UM", "6.5"))
GAP_DENSITY_GAIN = float(os.environ.get("BIOHUB_GAP_DENSITY_GAIN", "0.040"))
GAP_DENSITY_MAX_STEP_DELTA_UM = float(os.environ.get("BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM", "0.125"))
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
OUTPUT_KEEP_DIVISION_COMPONENTS = os.environ.get("BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS", "1") != "0"
ADAPTIVE_SHORT_TRACK_RESCUE = os.environ.get("BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE", "0") != "0"
SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC = float(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC", "0.10"))
SHORT_TRACK_RESCUE_MIN_LEN = int(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN", "4"))
SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB = float(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB", "0.82"))
SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM = float(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM", "3.25"))
SHORT_TRACK_RESCUE_MAX_NODES_FRAC = float(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC", "0.018"))
SHORT_TRACK_RESCUE_MAX_NODES_ABS = int(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS", "180"))

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
SAFE_DIV_EXISTING_CHILD_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM", "7.8"))
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
DEEPCENTER_RELATIVE = os.environ.get("BIOHUB_DEEPCENTER_RELATIVE", "weights/full_frame_center/best.pt")
DEEPCENTER_GAP_VETO = os.environ.get("BIOHUB_DEEPCENTER_GAP_VETO", "1") != "0"
DEEPCENTER_SAFE_DIV_VETO = os.environ.get("BIOHUB_DEEPCENTER_SAFE_DIV_VETO", "1") != "0"
DEEPCENTER_GAP_THRESHOLD = float(os.environ.get("BIOHUB_DEEPCENTER_GAP_THRESHOLD", "0.10"))
DEEPCENTER_EXPECTED_EPOCH = int(os.environ.get("BIOHUB_DEEPCENTER_EXPECTED_EPOCH", "0"))
DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM = float(os.environ.get("BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM", "0"))
DEEPCENTER_SAFE_DIV_THRESHOLD = float(os.environ.get("BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD", "0.12"))
DEEPCENTER_SCORE_WIN_Z = int(os.environ.get("BIOHUB_DEEPCENTER_SCORE_WIN_Z", "1"))
DEEPCENTER_SCORE_WIN_YX = int(os.environ.get("BIOHUB_DEEPCENTER_SCORE_WIN_YX", "2"))
DEEPCENTER_SCORE_CACHE_MAX_FRAMES = int(os.environ.get("BIOHUB_DEEPCENTER_SCORE_CACHE_MAX_FRAMES", "8"))

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
    "pydantic": ["annotated-types", "pydantic-core", "typing-extensions>=4.13", "typing-inspection"],
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
        return hasattr(_pl, "Float16") and _pl.Series([-999999.0], dtype=_pl.Float64).dtype == _pl.Float64
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
        print("Dependency resolver is disabled with --no-deps to avoid replacing Kaggle numpy/scipy in a live kernel.")
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
        + command + "\n"
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
        "The attached support artifact may be missing wheels.\n"
        + json.dumps(failures, indent=2)
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
    "src/biohub_tracking/models/temporal_unet.py": "d809c35d42f504161074ddeaaa7aee5b407e5bca7f9b4e1d5f9b2ff345666cac"
}
_support_expected_manifest_sha256 = "978b626d1fd1e7397435a437dfe68691defe1572fc3c20e61012d7c9b52ed029"
_primary_expected_sha256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
_deepcenter_expected_sha256 = "8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0"  


def _integrity_sha256_file(path: Path) -> str:
    digest = _integrity_hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_support_materialized_paths = {
    path.relative_to(REPO_DIR).as_posix(): path
    for path in REPO_DIR.rglob("*.py")
}
_support_actual_names = set(_support_materialized_paths)
_support_expected_names = set(_support_expected_sha256)
if _support_actual_names != _support_expected_names:
    raise RuntimeError({
        "support_repo_python_files_missing": sorted(
            _support_expected_names - _support_actual_names
        ),
        "support_repo_python_files_extra": sorted(
            _support_actual_names - _support_expected_names
        ),
    })
_support_actual_sha256 = {
    relative: _integrity_sha256_file(_support_materialized_paths[relative])
    for relative in sorted(_support_materialized_paths)
}
if _support_actual_sha256 != _support_expected_sha256:
    raise RuntimeError({
        "support_repo_python_checksum_mismatch": {
            relative: {
                "expected": _support_expected_sha256[relative],
                "actual": _support_actual_sha256[relative],
            }
            for relative in sorted(_support_expected_sha256)
            if _support_actual_sha256[relative]
            != _support_expected_sha256[relative]
        }
    })
_support_manifest_bytes = "".join(
    f"{_support_actual_sha256[relative]}  {relative}\n"
    for relative in sorted(_support_actual_sha256)
).encode("utf-8")
_support_actual_manifest_sha256 = _integrity_hashlib.sha256(
    _support_manifest_bytes
).hexdigest()
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
    "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/"
    "full_frame_center/best.pt",
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
    raise FileNotFoundError({
        "missing_deepcenter_checkpoint": [str(path) for path in _deepcenter_candidates]
    })
_deepcenter_actual_sha256 = _integrity_sha256_file(
    _deepcenter_materialized_path
)
if _deepcenter_actual_sha256 != _deepcenter_expected_sha256:
    raise RuntimeError(
        "DeepCenter checkpoint checksum mismatch: "
        f"expected {_deepcenter_expected_sha256}, "
        f"got {_deepcenter_actual_sha256}"
    )
os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(
    _deepcenter_materialized_path
)

print("Support repo Python manifest SHA256:", _support_actual_manifest_sha256)
print("Primary materialized SHA256:", _primary_actual_sha256)
print("DeepCenter materialized SHA256:", _deepcenter_actual_sha256)



import hashlib as _hashlib

_secondary_manifest_explicit = Path(os.environ.get(
    "BIOHUB_SECONDARY_ARTIFACT_MANIFEST",
    "/kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1/ARTIFACT_MANIFEST.json",
))
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
    SECONDARY_WEIGHTS_ROOT
    / "unet_transformer"
    / "split_0"
    / "edge_predictor_best.pth"
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
_runtime_integrity_receipt_path = (
    WORKING_DIR / "bidirectional_production_runtime_integrity.json"
)
_runtime_integrity_receipt_path.write_text(
    json.dumps(_runtime_integrity_receipt, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("Runtime integrity receipt:", _runtime_integrity_receipt_path)

# %% [markdown]
# ## 2. Train input and feature capture

# %%

import hashlib as _self_hashlib
import json as _self_json
import yaml as _self_yaml
from pathlib import Path as _SelfPath

SELF_CONFIG = _self_yaml.safe_load(_SelfPath('config.yaml').read_text())
TRAIN_IMAGE_DIR = COMP_DIR / 'train'
PUBLIC_TEST_DIR = COMP_DIR / 'test'
if not TRAIN_IMAGE_DIR.is_dir() or not PUBLIC_TEST_DIR.is_dir():
    raise FileNotFoundError((TRAIN_IMAGE_DIR, PUBLIC_TEST_DIR))
if _self_hashlib.sha256(_SelfPath('config.yaml').read_bytes()).hexdigest() == '':
    raise RuntimeError('unreachable config guard')
TEST_DIR = TRAIN_IMAGE_DIR
print('Training images and GEFF:', TRAIN_IMAGE_DIR)

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
    ('    downsample: tuple[int, ...] = (1, 4, 4),\n) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:', '    downsample: tuple[int, ...] = (1, 4, 4),\n    secondary_model: UNetNodeTransformer | None = None,\n    secondary_edge_weight: float = 0.0,\n    secondary_detection_weight: float = 0.0,\n    secondary_link_mode: str = "raw",\n    secondary_mix_temperature: float = 1.0,\n    secondary_low_margin_max: float = 0.2,\n) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:'),
    ('            for f in range(W):\n                det_logits[f] = det_logits[f] / _nv\n\n        del imgs', '            for f in range(W):\n                det_logits[f] = det_logits[f] / _nv\n\n        secondary_unet_out = None\n        if secondary_model is not None:\n            secondary_unet_out, secondary_det_logits = secondary_model.encode(imgs)\n\n            if secondary_detection_weight > 0.0:\n                if cfg.det_tta:\n                    _secondary_nv = 1\n                    for dims in [(-1,), (-2,), (-2, -1)]:\n                        secondary_imgs_flip = imgs.flip(dims)\n                        _, secondary_det_flip = secondary_model.encode(secondary_imgs_flip)\n                        for f in range(W):\n                            secondary_det_logits[f] = (\n                                secondary_det_logits[f] + secondary_det_flip[f].flip(dims)\n                            )\n                        del secondary_imgs_flip, secondary_det_flip\n                        _secondary_nv += 1\n                    for _k in (1, 3):\n                        secondary_imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))\n                        _, secondary_det_rot = secondary_model.encode(secondary_imgs_rot)\n                        for f in range(W):\n                            secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                                secondary_det_rot[f], -_k, dims=(-2, -1)\n                            )\n                        del secondary_imgs_rot, secondary_det_rot\n                        _secondary_nv += 1\n                    secondary_imgs_t = imgs.transpose(-1, -2)\n                    _, secondary_det_t = secondary_model.encode(secondary_imgs_t)\n                    for f in range(W):\n                        secondary_det_logits[f] = (\n                            secondary_det_logits[f] + secondary_det_t[f].transpose(-1, -2)\n                        )\n                    del secondary_imgs_t, secondary_det_t\n                    _secondary_nv += 1\n                    secondary_imgs_at = torch.rot90(\n                        imgs, 1, dims=(-2, -1)\n                    ).transpose(-1, -2)\n                    _, secondary_det_at = secondary_model.encode(secondary_imgs_at)\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                            secondary_det_at[f].transpose(-1, -2),\n                            -1,\n                            dims=(-2, -1),\n                        )\n                    del secondary_imgs_at, secondary_det_at\n                    _secondary_nv += 1\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] / _secondary_nv\n\n                for f in range(W):\n                    primary_det = det_logits[f]\n                    secondary_det = secondary_det_logits[f]\n                    primary_mean = primary_det.mean()\n                    secondary_mean = secondary_det.mean()\n                    primary_scale = primary_det.float().std(unbiased=False).clamp_min(1e-4)\n                    secondary_scale = secondary_det.float().std(unbiased=False).clamp_min(1e-4)\n                    scale_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)\n                    secondary_det_aligned = (\n                        (secondary_det - secondary_mean) * scale_ratio + primary_mean\n                    )\n                    det_logits[f] = (\n                        (1.0 - secondary_detection_weight) * primary_det\n                        + secondary_detection_weight * secondary_det_aligned\n                    )\n\n            del secondary_det_logits\n\n        del imgs'),
    ('            edge_logits_pair = model.predict_edges(\n                unet_feat_src, unet_feat_tgt,\n                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                p_pos_src, p_pos_tgt,\n                p_mask_src, p_mask_tgt,\n            )  # (1, n_src, n_tgt)\n\n            raw = edge_logits_pair[0]', '            edge_logits_pair = model.predict_edges(\n                unet_feat_src, unet_feat_tgt,\n                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                p_pos_src, p_pos_tgt,\n                p_mask_src, p_mask_tgt,\n            )  # (1, n_src, n_tgt)\n\n            if secondary_model is not None:\n                if secondary_unet_out is None:\n                    raise RuntimeError("Secondary model is loaded but its feature map is missing")\n                secondary_feat_src = secondary_model._index_features(\n                    secondary_unet_out[:, f_idx], p_coords_src, p_mask_src,\n                )\n                secondary_feat_tgt = secondary_model._index_features(\n                    secondary_unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt,\n                )\n                secondary_logits_pair = secondary_model.predict_edges(\n                    secondary_feat_src, secondary_feat_tgt,\n                    p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                    p_pos_src, p_pos_tgt,\n                    p_mask_src, p_mask_tgt,\n                )\n\n                if secondary_link_mode == "raw":\n                    secondary_for_mix = secondary_logits_pair\n                    blend_weight = secondary_edge_weight\n                elif secondary_link_mode in {\n                    "calibrated", "adaptive", "low_margin_consensus"\n                }:\n                    primary_center = edge_logits_pair.mean(dim=1, keepdim=True)\n                    primary_scale = edge_logits_pair.float().std(\n                        dim=1, keepdim=True, unbiased=False\n                    ).clamp_min(1e-4)\n                    secondary_center = secondary_logits_pair.mean(dim=1, keepdim=True)\n                    secondary_scale = secondary_logits_pair.float().std(\n                        dim=1, keepdim=True, unbiased=False\n                    ).clamp_min(1e-4)\n                    secondary_scale_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)\n                    secondary_for_mix = (\n                        (secondary_logits_pair - secondary_center) * secondary_scale_ratio\n                        + primary_center\n                    )\n                    if secondary_link_mode == "calibrated":\n                        blend_weight = secondary_edge_weight\n                    elif secondary_link_mode == "adaptive":\n                        if n_src >= 2:\n                            primary_probs = torch.softmax(edge_logits_pair[0], dim=0)\n                            secondary_probs = torch.softmax(secondary_for_mix[0], dim=0)\n                            primary_top2 = torch.topk(primary_probs, k=2, dim=0)\n                            secondary_top2 = torch.topk(secondary_probs, k=2, dim=0)\n                            primary_margin = primary_top2.values[0] - primary_top2.values[1]\n                            secondary_margin = secondary_top2.values[0] - secondary_top2.values[1]\n                            local_weight = (\n                                secondary_edge_weight + secondary_margin - primary_margin\n                            ).clamp(0.15, 0.75)\n                            same_parent = primary_top2.indices[0].eq(\n                                secondary_top2.indices[0]\n                            )\n                            local_weight = torch.where(\n                                same_parent,\n                                torch.maximum(\n                                    local_weight,\n                                    torch.full_like(local_weight, secondary_edge_weight),\n                                ),\n                                local_weight,\n                            )\n                            blend_weight = local_weight.view(1, 1, -1)\n                        else:\n                            blend_weight = secondary_edge_weight\n                    else:\n                        if n_src >= 2:\n                            primary_probs = torch.softmax(edge_logits_pair[0], dim=0)\n                            secondary_probs = torch.softmax(secondary_for_mix[0], dim=0)\n                            primary_top2 = torch.topk(primary_probs, k=2, dim=0)\n                            secondary_top2 = torch.topk(secondary_probs, k=2, dim=0)\n                            primary_margin = primary_top2.values[0] - primary_top2.values[1]\n                            same_parent = primary_top2.indices[0].eq(\n                                secondary_top2.indices[0]\n                            )\n                            uncertainty = (\n                                (secondary_low_margin_max - primary_margin)\n                                / secondary_low_margin_max\n                            ).clamp(0.0, 1.0)\n                            local_weight = secondary_edge_weight * uncertainty\n                            local_weight = torch.where(\n                                same_parent,\n                                local_weight,\n                                torch.zeros_like(local_weight),\n                            )\n                            blend_weight = local_weight.view(1, 1, -1)\n                        else:\n                            blend_weight = 0.0\n                else:\n                    raise ValueError(f"Unsupported secondary link mode: {secondary_link_mode}")\n\n                edge_logits_pair = (\n                    (1.0 - blend_weight) * edge_logits_pair\n                    + blend_weight * secondary_for_mix\n                )\n                if secondary_mix_temperature != 1.0:\n                    mixed_center = edge_logits_pair.mean(dim=1, keepdim=True)\n                    edge_logits_pair = mixed_center + (\n                        edge_logits_pair - mixed_center\n                    ) / secondary_mix_temperature\n\n            raw = edge_logits_pair[0]'),
    ('        del unet_out\n', '        del unet_out\n        if secondary_unet_out is not None:\n            del secondary_unet_out\n'),
    ('    model, window_size, downsample = load_model(weights_path, device)\n    print(', '    model, window_size, downsample = load_model(weights_path, device)\n\n    secondary_model = None\n    secondary_weights_text = os.environ.get("BIOHUB_SECONDARY_WEIGHTS", "").strip()\n    secondary_edge_weight = float(os.environ.get("BIOHUB_SECONDARY_EDGE_WEIGHT", "0"))\n    secondary_detection_weight = float(\n        os.environ.get("BIOHUB_SECONDARY_DETECTION_WEIGHT", "0")\n    )\n    secondary_link_mode = os.environ.get("BIOHUB_SECONDARY_LINK_MODE", "raw").strip()\n    secondary_mix_temperature = float(\n        os.environ.get("BIOHUB_SECONDARY_MIX_TEMPERATURE", "1")\n    )\n    secondary_low_margin_max = float(\n        os.environ.get("BIOHUB_SECONDARY_LOW_MARGIN_MAX", "0.2")\n    )\n    edge_candidate_threshold = float(\n        os.environ.get("BIOHUB_DUAL_SEED_EDGE_THRESHOLD", str(cfg.threshold))\n    )\n    if secondary_weights_text:\n        if not 0.0 < secondary_edge_weight < 1.0:\n            raise ValueError("BIOHUB_SECONDARY_EDGE_WEIGHT must be strictly between 0 and 1")\n        if not 0.0 <= secondary_detection_weight < 1.0:\n            raise ValueError(\n                "BIOHUB_SECONDARY_DETECTION_WEIGHT must be in the half-open interval [0, 1)"\n            )\n        if secondary_link_mode not in {\n            "raw", "calibrated", "adaptive", "low_margin_consensus"\n        }:\n            raise ValueError(\n                "BIOHUB_SECONDARY_LINK_MODE must be raw, calibrated, adaptive, "\n                "or low_margin_consensus"\n            )\n        if not 0.5 <= secondary_mix_temperature <= 2.0:\n            raise ValueError("BIOHUB_SECONDARY_MIX_TEMPERATURE must be in [0.5, 2.0]")\n        if not 0.0 < edge_candidate_threshold < 1.0:\n            raise ValueError("BIOHUB_DUAL_SEED_EDGE_THRESHOLD must be strictly between 0 and 1")\n        if not 0.0 < secondary_low_margin_max <= 1.0:\n            raise ValueError("BIOHUB_SECONDARY_LOW_MARGIN_MAX must be in (0, 1]")\n        secondary_model, secondary_window_size, secondary_downsample = load_model(\n            Path(secondary_weights_text), device,\n        )\n        if secondary_window_size != window_size or secondary_downsample != downsample:\n            raise ValueError(\n                "Primary and secondary models have incompatible inference grids: "\n                f"primary=(window={window_size}, downsample={downsample}), "\n                f"secondary=(window={secondary_window_size}, downsample={secondary_downsample})"\n            )\n        cfg.threshold = edge_candidate_threshold\n        print(\n            f"Secondary model: {secondary_weights_text} | "\n            f"edge weight={secondary_edge_weight:.3f} | "\n            f"detection weight={secondary_detection_weight:.3f} | "\n            f"link mode={secondary_link_mode} | "\n            f"temperature={secondary_mix_temperature:.3f} | "\n            f"low-margin max={secondary_low_margin_max:.3f} | "\n            f"edge threshold={cfg.threshold:.3f}",\n            flush=True,\n        )\n\n    print('),
    ('                unet_batch_size=unet_batch_size,\n                downsample=downsample,\n            )', '                unet_batch_size=unet_batch_size,\n                downsample=downsample,\n                secondary_model=secondary_model,\n                secondary_edge_weight=secondary_edge_weight,\n                secondary_detection_weight=secondary_detection_weight,\n                secondary_link_mode=secondary_link_mode,\n                secondary_mix_temperature=secondary_mix_temperature,\n                secondary_low_margin_max=secondary_low_margin_max,\n            )'),
]
for _patch_index, (_ensemble_old, _ensemble_new) in enumerate(
    _ensemble_replacements, start=1
):
    _ensemble_count = _s.count(_ensemble_old)
    if _ensemble_count != 1:
        raise RuntimeError(
            f'Calibrated dual-seed patch {_patch_index} expected one match, '
            f'found {_ensemble_count}'
        )
    _s = _s.replace(_ensemble_old, _ensemble_new, 1)
compile(_s, str(_ps), 'exec')
_ps.write_text(_s)
print('Calibrated dual-seed runtime patch applied')



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
    raise RuntimeError(
        f"Retention guard expected one blend block, found {_guard_matches}"
    )
_s = _s.replace(_guard_old, _guard_new, 1)
compile(_s, str(_ps), "exec")
_ps.write_text(_s)
print(
    "Frozen frame retention guard applied at "
    + os.environ["BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION"]
)



import math as _bidirectional_math

_bidirectional_weight_guard = float(
    os.environ.get("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT", "0")
)
if not _bidirectional_math.isclose(
    _bidirectional_weight_guard, 0.15, rel_tol=0.0, abs_tol=1e-12
):
    raise ValueError({
        "expected_bidirectional_weight": 0.15,
        "actual_bidirectional_weight": _bidirectional_weight_guard,
    })

_s = _ps.read_text()
_bi_old = '            edge_logits_pair = model.predict_edges(\n                unet_feat_src, unet_feat_tgt,\n                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                p_pos_src, p_pos_tgt,\n                p_mask_src, p_mask_tgt,\n            )  # (1, n_src, n_tgt)\n\n            if secondary_model is not None:\n'
_bi_new = '            edge_logits_pair = model.predict_edges(\n                unet_feat_src, unet_feat_tgt,\n                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                p_pos_src, p_pos_tgt,\n                p_mask_src, p_mask_tgt,\n            )  # (1, n_src, n_tgt)\n\n            _bidirectional_weight = float(\n                os.environ.get("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT", "0")\n            )\n            if _bidirectional_weight > 0.0:\n                reverse_logits_native = model.predict_edges(\n                    unet_feat_tgt, unet_feat_src,\n                    p_coords_tgt * ds_arr_t, p_coords_src * ds_arr_t,\n                    p_pos_tgt, p_pos_src,\n                    p_mask_tgt, p_mask_src,\n                )  # (1, n_tgt, n_src)\n                reverse_logits_pair = reverse_logits_native.transpose(1, 2)\n\n                forward_center = edge_logits_pair.mean(dim=1, keepdim=True)\n                forward_scale = edge_logits_pair.float().std(\n                    dim=1, keepdim=True, unbiased=False\n                ).clamp_min(1e-4)\n                reverse_center = reverse_logits_pair.mean(dim=1, keepdim=True)\n                reverse_scale = reverse_logits_pair.float().std(\n                    dim=1, keepdim=True, unbiased=False\n                ).clamp_min(1e-4)\n                reverse_scale_ratio = (forward_scale / reverse_scale).clamp(0.5, 2.0)\n                reverse_scale_ratio = reverse_scale_ratio.to(reverse_logits_pair.dtype)\n                reverse_aligned = (\n                    (reverse_logits_pair - reverse_center) * reverse_scale_ratio\n                    + forward_center\n                )\n                # Biohub 145: require mutual forward/reverse support in probability space.\n                # The harmonic mean penalizes a candidate when either temporal direction\n                # assigns it very low probability, while calibration preserves the forward\n                # logit scale used by the unchanged downstream candidate threshold and ILP.\n                forward_prob = torch.softmax(edge_logits_pair.float(), dim=1).clamp_min(1e-8)\n                reverse_prob = torch.softmax(reverse_aligned.float(), dim=1).clamp_min(1e-8)\n                harmonic_prob = 1.0 / (\n                    (1.0 - _bidirectional_weight) / forward_prob\n                    + _bidirectional_weight / reverse_prob\n                )\n                harmonic_prob = harmonic_prob / harmonic_prob.sum(\n                    dim=1, keepdim=True\n                ).clamp_min(1e-8)\n                harmonic_logits = torch.log(harmonic_prob.clamp_min(1e-8))\n                harmonic_center = harmonic_logits.mean(dim=1, keepdim=True)\n                harmonic_scale = harmonic_logits.std(\n                    dim=1, keepdim=True, unbiased=False\n                ).clamp_min(1e-4)\n                harmonic_scale_ratio = (forward_scale / harmonic_scale).clamp(0.5, 2.0)\n                edge_logits_pair = (\n                    (harmonic_logits - harmonic_center) * harmonic_scale_ratio\n                    + forward_center\n                ).to(reverse_aligned.dtype)\n                del (\n                    reverse_logits_native,\n                    reverse_logits_pair,\n                    reverse_aligned,\n                    forward_prob,\n                    reverse_prob,\n                    harmonic_prob,\n                    harmonic_logits,\n                )\n            if secondary_model is not None:\n'
_bi_count = _s.count(_bi_old)
if _bi_count != 1:
    raise RuntimeError(
        f"Bidirectional edge patch expected one transformed block, found {_bi_count}"
    )
_s = _s.replace(_bi_old, _bi_new, 1)

_coordinate_manifest_old = '    coords = coords.astype(np.int16)\n    return coords, all_edges'
_coordinate_manifest_new = '    coords = coords.astype(np.int16)\n\n    # Label-free, pre-ILP detector-coordinate manifest. This executes inside\n    # predict_video, before build_graph and the ILP call in predict().\n    _coordinate_manifest_arm = os.environ.get(\n        "BIOHUB_DIAGNOSTIC_ARM", ""\n    ).strip()\n    if _coordinate_manifest_arm:\n        import hashlib as _coordinate_hashlib\n\n        _coordinate_shard = os.environ.get(\n            "BIOHUB_GPU_SHARD", "single"\n        ).replace("/", "_")\n        _coordinate_array = np.ascontiguousarray(\n            coords.astype("<i2", copy=False)\n        )\n        _coordinate_frame_counts = [\n            [int(_coordinate_t), int((_coordinate_array[:, 0] == _coordinate_t).sum())]\n            for _coordinate_t in np.unique(_coordinate_array[:, 0])\n        ]\n        _coordinate_record = {\n            "columns": ["t", "z", "y", "x"],\n            "coordinate_sha256": _coordinate_hashlib.sha256(\n                _coordinate_array.tobytes(order="C")\n            ).hexdigest(),\n            "dataset": ds_path.stem,\n            "dtype": "<i2",\n            "frame_counts": _coordinate_frame_counts,\n            "rows": int(len(_coordinate_array)),\n            "stage": "post_detection_pre_graph_pre_ilp",\n        }\n        _coordinate_manifest_path = (\n            Path("/kaggle/working")\n            / f"detector_coordinates_{_coordinate_manifest_arm}_"\n            f"{_coordinate_shard}.jsonl"\n        )\n        with _coordinate_manifest_path.open("a") as _coordinate_handle:\n            _coordinate_handle.write(\n                json.dumps(_coordinate_record, sort_keys=True) + "\\n"\n            )\n\n    return coords, all_edges'
_coordinate_manifest_count = _s.count(_coordinate_manifest_old)
if _coordinate_manifest_count != 1:
    raise RuntimeError(
        "Coordinate-manifest patch expected one pre-return block, found "
        f"{_coordinate_manifest_count}"
    )
_s = _s.replace(
    _coordinate_manifest_old, _coordinate_manifest_new, 1
)

# Runtime profile and bound for the ILP. The support pack solves each dataset's
# whole graph in one SCIP model with no time limit; print how long that takes
# and cap it with BIOHUB_ILP_TIMEOUT_S (SCIP returns its incumbent at the limit,
# and tracksdata only logs a warning when the status is not OPTIMAL).
_ilp_old = (
    '            solver = td.solvers.ILPSolver(\n'
    '                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),\n'
    '                appearance_weight=cfg.ilp_appearance_weight,\n'
    '                disappearance_weight=cfg.ilp_disappearance_weight,\n'
    '                division_weight=cfg.ilp_division_weight,\n'
    '            )\n'
    '            with suppress_output():\n'
    '                graph = solver.solve(graph)\n'
)
_ilp_new = (
    '            import time as _ilp_time\n'
    '            _ilp_timeout = float(os.environ.get("BIOHUB_ILP_TIMEOUT_S", "0") or 0.0)\n'
    '            solver = td.solvers.ILPSolver(\n'
    '                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),\n'
    '                appearance_weight=cfg.ilp_appearance_weight,\n'
    '                disappearance_weight=cfg.ilp_disappearance_weight,\n'
    '                division_weight=cfg.ilp_division_weight,\n'
    '                timeout=_ilp_timeout if _ilp_timeout > 0 else None,\n'
    '            )\n'
    '            _ilp_t0 = _ilp_time.time()\n'
    '            _ilp_nodes, _ilp_edges = graph.num_nodes(), graph.num_edges()\n'
    '            with suppress_output():\n'
    '                graph = solver.solve(graph)\n'
    '            print(\n'
    '                f"[{name}] ILP {_ilp_time.time() - _ilp_t0:.1f}s"\n'
    '                f" | candidate nodes={_ilp_nodes} edges={_ilp_edges}"\n'
    '                f" | timeout={_ilp_timeout if _ilp_timeout > 0 else None}",\n'
    '                flush=True,\n'
    '            )\n'
)
_ilp_count = _s.count(_ilp_old)
if _ilp_count != 1:
    raise RuntimeError(f"ILP timeout patch expected one solver block, found {_ilp_count}")
_s = _s.replace(_ilp_old, _ilp_new, 1)
print("ILP timeout patch applied |", os.environ.get("BIOHUB_ILP_TIMEOUT_S", "0"), "s per dataset")

# Per-dataset detection timing. Diagnostic only, so a missing anchor is a
# warning rather than a failed kernel.
_pv_old = '        coords, edges = predict_video(\n'
_bg_old = '        graph = build_graph(coords, edges)\n'
if _s.count(_pv_old) == 1 and _s.count(_bg_old) == 1:
    _s = _s.replace(
        _pv_old,
        '        import time as _pv_time\n        _pv_t0 = _pv_time.time()\n' + _pv_old,
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
_et_old = '        if cfg.det_tta:\n            _nv = 1\n            for dims in [(-1,), (-2,), (-2, -1)]:\n                imgs_flip = imgs.flip(dims)\n                _, det_flip = model.encode(imgs_flip)\n                for f in range(W):\n                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)\n                del imgs_flip, det_flip\n                _nv += 1\n            for _k in (1, 3):\n                imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))\n                _, det_rot = model.encode(imgs_rot)\n                for f in range(W):\n                    det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims=(-2, -1))\n                del imgs_rot, det_rot\n                _nv += 1\n            imgs_t = imgs.transpose(-1, -2)\n            _, det_t = model.encode(imgs_t)\n            for f in range(W):\n                det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)\n            del imgs_t, det_t\n            _nv += 1\n            imgs_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)\n            _, det_at = model.encode(imgs_at)\n            for f in range(W):\n                det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))\n            del imgs_at, det_at\n            _nv += 1\n            for f in range(W):\n                det_logits[f] = det_logits[f] / _nv\n'
_et_new = "        if cfg.det_tta:\n            _edge_tta = os.environ.get('BIOHUB_EDGE_FEATURE_TTA', '0') != '0'\n            _unet_acc = unet_out.clone() if _edge_tta else None\n            _nv = 1\n            for dims in [(-1,), (-2,), (-2, -1)]:\n                imgs_flip = imgs.flip(dims)\n                _u_flip, det_flip = model.encode(imgs_flip)\n                for f in range(W):\n                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)\n                if _edge_tta:\n                    _unet_acc = _unet_acc + _u_flip.flip(dims)\n                del imgs_flip, det_flip, _u_flip\n                _nv += 1\n            for _k in (1, 3):\n                imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))\n                _u_rot, det_rot = model.encode(imgs_rot)\n                for f in range(W):\n                    det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims=(-2, -1))\n                if _edge_tta:\n                    _unet_acc = _unet_acc + torch.rot90(_u_rot, -_k, dims=(-2, -1))\n                del imgs_rot, det_rot, _u_rot\n                _nv += 1\n            imgs_t = imgs.transpose(-1, -2)\n            _u_t, det_t = model.encode(imgs_t)\n            for f in range(W):\n                det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)\n            if _edge_tta:\n                _unet_acc = _unet_acc + _u_t.transpose(-1, -2)\n            del imgs_t, det_t, _u_t\n            _nv += 1\n            imgs_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)\n            _u_at, det_at = model.encode(imgs_at)\n            for f in range(W):\n                det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))\n            if _edge_tta:\n                _unet_acc = _unet_acc + torch.rot90(_u_at.transpose(-1, -2), -1, dims=(-2, -1))\n            del imgs_at, det_at, _u_at\n            _nv += 1\n            for f in range(W):\n                det_logits[f] = det_logits[f] / _nv\n            if _edge_tta:\n                if _unet_acc.shape != unet_out.shape:\n                    raise RuntimeError('EDGE-TTA SHAPE MISMATCH: %s vs %s'\n                                       % (tuple(_unet_acc.shape), tuple(unet_out.shape)))\n                _delta = float((_unet_acc / _nv - unet_out).abs().mean())\n                if _delta == 0.0:\n                    raise RuntimeError('EDGE-TTA NO-OP: averaged features bit-identical to the '\n                                       'single-pass features, so the augmented encodes '\n                                       'contributed nothing and this arm would read as a '\n                                       'false null')\n                unet_out = _unet_acc / _nv\n                print('EDGE_TTA_ACTIVE views=', _nv, 'mean_abs_feat_delta=', round(_delta, 6), flush=True)\n                del _unet_acc\n"
if _et_s.count(_et_old) != 1:
    raise RuntimeError('edge-TTA anchor block not unique: %d' % _et_s.count(_et_old))
_et_s = _et_s.replace(_et_old, _et_new, 1)
compile(_et_s, str(_ps), 'exec')
_ps.write_text(_et_s)
if 'EDGE_TTA_ACTIVE' not in _ps.read_text():
    raise RuntimeError('EDGE-TTA PATCH DID NOT PERSIST')
os.environ['BIOHUB_EDGE_FEATURE_TTA'] = '1'
print('EDGE_TTA patch installed and enabled in', _ps)

_secondary_tta_source = _ps.read_text()
_secondary_tta_old = '        secondary_unet_out, secondary_det_logits = secondary_model.encode(imgs)\n\n            if secondary_detection_weight > 0.0:\n                if cfg.det_tta:\n                    _secondary_nv = 1\n                    for dims in [(-1,), (-2,), (-2, -1)]:\n                        secondary_imgs_flip = imgs.flip(dims)\n                        _, secondary_det_flip = secondary_model.encode(secondary_imgs_flip)\n                        for f in range(W):\n                            secondary_det_logits[f] = (\n                                secondary_det_logits[f] + secondary_det_flip[f].flip(dims)\n                            )\n                        del secondary_imgs_flip, secondary_det_flip\n                        _secondary_nv += 1\n                    for _k in (1, 3):\n                        secondary_imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))\n                        _, secondary_det_rot = secondary_model.encode(secondary_imgs_rot)\n                        for f in range(W):\n                            secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                                secondary_det_rot[f], -_k, dims=(-2, -1)\n                            )\n                        del secondary_imgs_rot, secondary_det_rot\n                        _secondary_nv += 1\n                    secondary_imgs_t = imgs.transpose(-1, -2)\n                    _, secondary_det_t = secondary_model.encode(secondary_imgs_t)\n                    for f in range(W):\n                        secondary_det_logits[f] = (\n                            secondary_det_logits[f] + secondary_det_t[f].transpose(-1, -2)\n                        )\n                    del secondary_imgs_t, secondary_det_t\n                    _secondary_nv += 1\n                    secondary_imgs_at = torch.rot90(\n                        imgs, 1, dims=(-2, -1)\n                    ).transpose(-1, -2)\n                    _, secondary_det_at = secondary_model.encode(secondary_imgs_at)\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                            secondary_det_at[f].transpose(-1, -2),\n                            -1,\n                            dims=(-2, -1),\n                        )\n                    del secondary_imgs_at, secondary_det_at\n                    _secondary_nv += 1\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] / _secondary_nv\n\n                for f in range(W):'
_secondary_tta_new = '        secondary_unet_out, secondary_det_logits = secondary_model.encode(imgs)\n            _secondary_edge_tta = os.environ.get(\n                "BIOHUB_SECONDARY_EDGE_FEATURE_TTA", "0"\n            ) != "0"\n            _secondary_unet_acc = (\n                secondary_unet_out.clone() if _secondary_edge_tta else None\n            )\n\n            if secondary_detection_weight > 0.0:\n                if cfg.det_tta:\n                    _secondary_nv = 1\n                    for dims in [(-1,), (-2,), (-2, -1)]:\n                        secondary_imgs_flip = imgs.flip(dims)\n                        _secondary_u_flip, secondary_det_flip = secondary_model.encode(\n                            secondary_imgs_flip\n                        )\n                        for f in range(W):\n                            secondary_det_logits[f] = (\n                                secondary_det_logits[f] + secondary_det_flip[f].flip(dims)\n                            )\n                        if _secondary_edge_tta:\n                            _secondary_unet_acc = _secondary_unet_acc + _secondary_u_flip.flip(dims)\n                        del secondary_imgs_flip, secondary_det_flip, _secondary_u_flip\n                        _secondary_nv += 1\n                    for _k in (1, 3):\n                        secondary_imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))\n                        _secondary_u_rot, secondary_det_rot = secondary_model.encode(\n                            secondary_imgs_rot\n                        )\n                        for f in range(W):\n                            secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                                secondary_det_rot[f], -_k, dims=(-2, -1)\n                            )\n                        if _secondary_edge_tta:\n                            _secondary_unet_acc = _secondary_unet_acc + torch.rot90(\n                                _secondary_u_rot, -_k, dims=(-2, -1)\n                            )\n                        del secondary_imgs_rot, secondary_det_rot, _secondary_u_rot\n                        _secondary_nv += 1\n                    secondary_imgs_t = imgs.transpose(-1, -2)\n                    _secondary_u_t, secondary_det_t = secondary_model.encode(secondary_imgs_t)\n                    for f in range(W):\n                        secondary_det_logits[f] = (\n                            secondary_det_logits[f] + secondary_det_t[f].transpose(-1, -2)\n                        )\n                    if _secondary_edge_tta:\n                        _secondary_unet_acc = _secondary_unet_acc + _secondary_u_t.transpose(-1, -2)\n                    del secondary_imgs_t, secondary_det_t, _secondary_u_t\n                    _secondary_nv += 1\n                    secondary_imgs_at = torch.rot90(\n                        imgs, 1, dims=(-2, -1)\n                    ).transpose(-1, -2)\n                    _secondary_u_at, secondary_det_at = secondary_model.encode(\n                        secondary_imgs_at\n                    )\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                            secondary_det_at[f].transpose(-1, -2),\n                            -1,\n                            dims=(-2, -1),\n                        )\n                    if _secondary_edge_tta:\n                        _secondary_unet_acc = _secondary_unet_acc + torch.rot90(\n                            _secondary_u_at.transpose(-1, -2), -1, dims=(-2, -1)\n                        )\n                    del secondary_imgs_at, secondary_det_at, _secondary_u_at\n                    _secondary_nv += 1\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] / _secondary_nv\n                    if _secondary_edge_tta:\n                        if _secondary_unet_acc.shape != secondary_unet_out.shape:\n                            raise RuntimeError("SECONDARY_EDGE_TTA_SHAPE_MISMATCH")\n                        _secondary_delta = float(\n                            (_secondary_unet_acc / _secondary_nv - secondary_unet_out).abs().mean()\n                        )\n                        if _secondary_delta == 0.0:\n                            raise RuntimeError("SECONDARY_EDGE_TTA_NO_OP")\n                        _secondary_edge_tta_weight = float(os.environ.get(\n                            "BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT", "1.0"\n                        ))\n                        if not 0.0 < _secondary_edge_tta_weight <= 1.0:\n                            raise RuntimeError("SECONDARY_EDGE_TTA_BAD_WEIGHT")\n                        _secondary_tta_mean = _secondary_unet_acc / _secondary_nv\n                        secondary_unet_out = (\n                            (1.0 - _secondary_edge_tta_weight) * secondary_unet_out\n                            + _secondary_edge_tta_weight * _secondary_tta_mean\n                        )\n                        print(\n                            "SECONDARY_EDGE_TTA_ACTIVE views=",\n                            _secondary_nv,\n                            "weight=",\n                            _secondary_edge_tta_weight,\n                            "mean_abs_feat_delta=",\n                            round(_secondary_delta, 6),\n                            flush=True,\n                        )\n                        del _secondary_unet_acc\n\n                for f in range(W):'
_secondary_tta_count = _secondary_tta_source.count(_secondary_tta_old)
if _secondary_tta_count != 1:
    raise RuntimeError(
        "secondary edge-TTA anchor expected one match, found "
        + str(_secondary_tta_count)
    )
_secondary_tta_source = _secondary_tta_source.replace(
    _secondary_tta_old, _secondary_tta_new, 1
)
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



def _stable_video_rank(stem: str) -> str:
    seed = SELF_CONFIG['data']['selection_seed']
    return _self_hashlib.sha256(f'{seed}:{stem}'.encode()).hexdigest()

_all_train_stems = sorted(path.stem for path in TRAIN_IMAGE_DIR.glob('*.zarr'))
_selected_by_embryo = {}
for _embryo in SELF_CONFIG['data']['expected_embryos']:
    _eligible = [stem for stem in _all_train_stems
                 if stem.split('_', 1)[0] == str(_embryo)
                 and (TRAIN_IMAGE_DIR / f'{stem}.geff').is_dir()]
    _eligible.sort(key=lambda stem: (_stable_video_rank(stem), stem))
    _required = int(SELF_CONFIG['data']['videos_per_embryo'])
    if len(_eligible) < _required:
        raise RuntimeError({'embryo': _embryo, 'eligible': len(_eligible), 'required': _required})
    _selected_by_embryo[str(_embryo)] = _eligible[:_required]
test_stems = sorted(stem for group in _selected_by_embryo.values() for stem in group)
if len(test_stems) != 2 * int(SELF_CONFIG['data']['videos_per_embryo']):
    raise RuntimeError('Unexpected train video count')
(WORKING_DIR / 'selected_train_videos.json').write_text(
    _self_json.dumps(_selected_by_embryo, indent=2, sort_keys=True) + '\n')
print('Selected train videos:', _selected_by_embryo, flush=True)

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
        for _label, _old, _new in [('predict_video globals', '@torch.no_grad()\ndef predict_video(', "_CACHE_EDGES: list = []\n_CACHE_THRESHOLD = float(os.environ.get('BIOHUB_CACHE_EDGE_THRESHOLD', '0') or 0)\n_CACHE_DIR = os.environ.get('BIOHUB_CACHE_DIR', '').strip()\n\n\n@torch.no_grad()\ndef predict_video("), ('candidate dump', '            candidates = sorted(', '            if _CACHE_THRESHOLD > 0:\n                _ci, _cj = np.nonzero(probs > _CACHE_THRESHOLD)\n                if _ci.size:\n                    _CACHE_EDGES.append((\n                        np.asarray(idx_src)[_ci].astype(np.int32),\n                        np.asarray(idx_tgt)[_cj].astype(np.int32),\n                        probs[_ci, _cj].astype(np.float32),\n                    ))\n            candidates = sorted('), ('cache write', '        graph = build_graph(coords, edges)', "        if _CACHE_DIR:\n            import pathlib\n            _cd = pathlib.Path(_CACHE_DIR)\n            _cd.mkdir(parents=True, exist_ok=True)\n            if _CACHE_EDGES:\n                _es = np.concatenate([e[0] for e in _CACHE_EDGES])\n                _et = np.concatenate([e[1] for e in _CACHE_EDGES])\n                _ep = np.concatenate([e[2] for e in _CACHE_EDGES])\n            else:\n                _es = np.empty(0, np.int32); _et = np.empty(0, np.int32)\n                _ep = np.empty(0, np.float32)\n            np.savez_compressed(\n                _cd / f'{name}.npz', coords=coords,\n                edge_src=_es, edge_tgt=_et, edge_prob=_ep,\n                admitted=np.asarray(edges, dtype=np.float64),\n            )\n            print(f'CACHE {name}: {len(coords)} nodes, {_es.size} edges', flush=True)\n            _CACHE_EDGES.clear()\n        graph = build_graph(coords, edges)"), ('lowdet globals', "_CACHE_DIR = os.environ.get('BIOHUB_CACHE_DIR', '').strip()\n", "_CACHE_DIR = os.environ.get('BIOHUB_CACHE_DIR', '').strip()\n_LOWDET_THRESHOLD = float(os.environ.get('BIOHUB_LOWDET_THRESHOLD', '0') or 0)\n_LOWDET: list = []\n"), ('lowdet peaks', '                arr = _detect_cells_pooled(\n                    det_logits[f_idx][0], t, cfg.det_threshold, pool_k,\n                )\n                coord_offset[t] = (global_node_count, global_node_count + len(arr))\n', '                arr = _detect_cells_pooled(\n                    det_logits[f_idx][0], t, cfg.det_threshold, pool_k,\n                )\n                if _LOWDET_THRESHOLD > 0:\n                    _low = _detect_cells_pooled(det_logits[f_idx][0], t, _LOWDET_THRESHOLD, pool_k)\n                    if len(_low):\n                        _lg = det_logits[f_idx][0][0]\n                        _lz = torch.as_tensor(_low[:, 1:].astype(np.int64), device=_lg.device)\n                        _lsc = torch.sigmoid(_lg[_lz[:, 0], _lz[:, 1], _lz[:, 2]]).float().cpu().numpy()\n                        _LOWDET.append((_low.astype(np.float32), _lsc.astype(np.float32)))\n                coord_offset[t] = (global_node_count, global_node_count + len(arr))\n'), ('lowdet write', "            np.savez_compressed(\n                _cd / f'{name}.npz', coords=coords,\n", "            if _LOWDET:\n                _lc = np.concatenate([e[0] for e in _LOWDET]).astype(np.float32)\n                _lc[:, 1:] *= np.array(downsample, dtype=np.float32)\n                _lc = _lc.astype(np.int16)\n                _lsc = np.concatenate([e[1] for e in _LOWDET]).astype(np.float32)\n            else:\n                _lc = np.empty((0, 4), np.int16)\n                _lsc = np.empty(0, np.float32)\n            _LOWDET.clear()\n            print(f'LOWDET {name}: {len(_lc)} peaks above {_LOWDET_THRESHOLD}', flush=True)\n            np.savez_compressed(\n                _cd / f'{name}.npz', coords=coords, low_coords=_lc, low_score=_lsc,\n")]:
            if _cs.count(_old) != 1:
                raise RuntimeError(f"{_label}: anchor count={_cs.count(_old)}")
            _cs = _cs.replace(_old, _new, 1)
        compile(_cs, str(_ps), "exec")
        _ps.write_text(_cs)
        print("Low-detection dump applied | dir=", _cache_dir_env,
              "| lowdet threshold=", os.environ.get("BIOHUB_LOWDET_THRESHOLD"))
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
(_ps.parent/'v1284_coordinate_refinement.py').write_text('"""Frozen-feature coordinate regression at first-seen fused detector centers."""\nimport os\nfrom pathlib import Path\nimport numpy as np\nimport torch\n\nSPACING = np.array([1.625, 1.625, 1.625], dtype=np.float32)\nOFFSETS = ((0,0,0), (-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1))\n_CACHE = None\n\n\ndef make_head():\n    head = torch.nn.Sequential(torch.nn.Linear(224, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))\n    torch.nn.init.zeros_(head[-1].weight)\n    torch.nn.init.zeros_(head[-1].bias)\n    return head\n\n\ndef bounded(head, x):\n    delta = head(x)\n    return 2.0 * delta / (1.0 + torch.linalg.vector_norm(delta, dim=-1, keepdim=True))\n\n\ndef sample_features(feature, arr):\n    xyz = torch.as_tensor(arr[:, 1:], device=feature.device, dtype=torch.long)\n    blocks = []\n    for offset in OFFSETS:\n        loc = xyz + torch.tensor(offset, device=feature.device)\n        for axis, size in enumerate(feature.shape[-3:]):\n            loc[:, axis].clamp_(0, size-1)\n        blocks.append(feature[0, :, loc[:,0], loc[:,1], loc[:,2]].T)\n    # Directional differences plus the central representation.\n    return torch.cat([blocks[0]] + [b - blocks[0] for b in blocks[1:]], dim=1)\n\n\ndef index_features(self, maps, coords, mask):\n    """Trilinear lookup; integer coordinates reproduce native gather exactly."""\n    out = torch.zeros((*coords.shape[:2], maps.shape[1]), device=maps.device, dtype=maps.dtype)\n    for batch in range(len(maps)):\n        n = int(mask[batch].sum())\n        if not n:\n            continue\n        q = coords[batch, :n].clone()\n        for axis, size in enumerate(maps.shape[-3:]):\n            q[:,axis].clamp_(0, size-1)\n        low = q.floor().long()\n        frac = q-low\n        for z in (0,1):\n            for y in (0,1):\n                for x in (0,1):\n                    shift = torch.tensor([z,y,x], device=maps.device)\n                    loc = low+shift\n                    for axis, size in enumerate(maps.shape[-3:]):\n                        loc[:,axis].clamp_(0,size-1)\n                    weight = torch.where(shift.bool(), frac, 1-frac).prod(dim=1)\n                    out[batch,:n] += maps[batch,:,loc[:,0],loc[:,1],loc[:,2]].T * weight[:,None]\n    return out\n\n\ndef refine(ds_path, t, arr, feature):\n    global _CACHE\n    mode = os.environ[\'V1284_MODE\']\n    if not len(arr):\n        return arr\n    if mode == \'zero\':\n        return arr.astype(np.float32)\n    x = sample_features(feature, arr).float()\n    if mode == \'capture\':\n        folder = Path(os.environ[\'V1284_CAPTURE\']) / ds_path.stem\n        folder.mkdir(parents=True, exist_ok=True)\n        np.savez_compressed(folder/f\'{int(t):04d}.npz\', coords=arr, features=x.cpu().numpy())\n        return arr\n    if _CACHE is None:\n        saved = torch.load(os.environ[\'V1284_HEAD\'], map_location=\'cpu\', weights_only=True)\n        head = make_head().to(feature.device)\n        head.load_state_dict(saved[\'state_dict\']); head.eval()\n        _CACHE = (head, saved[\'mean\'].to(feature.device), saved[\'scale\'].to(feature.device))\n    head, mean, scale = _CACHE\n    shift = bounded(head, (x-mean)/scale).cpu().numpy() / SPACING\n    result = arr.astype(np.float32).copy()\n    result[:,1:] += shift\n    result[:,1:] = np.clip(result[:,1:], 0, np.asarray(feature.shape[-3:])-1)\n    if not np.isfinite(result).all() or np.max(np.linalg.norm((result[:,1:]-arr[:,1:])*SPACING,axis=1)) > 2.00001:\n        raise RuntimeError(\'invalid V1284 displacement\')\n    return result\n')
os.environ['V1284_MODE']='capture'
os.environ['V1284_CAPTURE']=str(WORKING_DIR / 'v1284_capture')

_trial_source = _ps.read_text()
if _trial_source.count('import tracksdata as td\n') != 1: raise RuntimeError("V1284 patch anchor mismatch")
_trial_source = _trial_source.replace('import tracksdata as td\n', 'import tracksdata as td\nfrom v1284_coordinate_refinement import refine as _v1284_refine, index_features as _v1284_index\n')
if _trial_source.count('                coord_offset[t] = (global_node_count, global_node_count + len(arr))') != 1: raise RuntimeError("V1284 patch anchor mismatch")
_trial_source = _trial_source.replace('                coord_offset[t] = (global_node_count, global_node_count + len(arr))', '                arr = _v1284_refine(ds_path, t, arr, unet_out[:, f_idx])\n                coord_offset[t] = (global_node_count, global_node_count + len(arr))')
if _trial_source.count('    coords = coords.astype(np.int16)\n') != 1: raise RuntimeError("V1284 patch anchor mismatch")
_trial_source = _trial_source.replace('    coords = coords.astype(np.int16)\n', '    # Preserve refined geometry through association and graph output.\n')
if _trial_source.count('    model, window_size, downsample = load_model(weights_path, device)') != 1: raise RuntimeError("V1284 patch anchor mismatch")
_trial_source = _trial_source.replace('    model, window_size, downsample = load_model(weights_path, device)', '    model, window_size, downsample = load_model(weights_path, device)\n    UNetNodeTransformer._index_features = _v1284_index')

_ps.write_text(_trial_source)
print('V1284 head patched AFTER the readmit dump patch; mode =', os.environ['V1284_MODE'])

# %%

# The original x138 source has already been materialized and patched. The only
# change to the V1284 module is its supported capture mode instead of a private head.
_CAPTURE_ROOT = WORKING_DIR / 'v1284_capture'
_CAPTURE_ROOT.mkdir(exist_ok=True)
os.environ['V1284_MODE'] = 'capture'
os.environ['V1284_CAPTURE'] = str(_CAPTURE_ROOT)


def _run_capture_subset(stems: list[str], label: str) -> float:
    splits_path.write_text(json.dumps([{'split': 0, 'train': [], 'test': stems}], indent=2))
    worker_count = min(2, _torch.cuda.device_count(), len(stems))
    if worker_count != 2:
        raise RuntimeError(f'Expected two T4 GPUs, got {_torch.cuda.device_count()}')
    tokens = _visible_cuda_tokens(worker_count)
    processes = {}
    commands = {}
    started = time.time()
    for shard in range(worker_count):
        command = [*predict_cmd, '--method', f'{METHOD}_capture_{label}_gpu{shard}',
                   '--slice', f'{shard}::{worker_count}']
        child_env = {**os.environ, 'PYTHONPATH': 'src',
                     'CUDA_VISIBLE_DEVICES': tokens[shard],
                     'BIOHUB_GPU_SHARD': f'{shard}/{worker_count}'}
        commands[shard] = command
        processes[shard] = subprocess.Popen(command, cwd=REPO_DIR, env=child_env)
    _wait_for_prediction_shards(processes, commands)
    elapsed = time.time() - started
    for stem in stems:
        folder = _CAPTURE_ROOT / stem
        if not folder.is_dir() or not any(folder.glob('*.npz')):
            raise RuntimeError(f'No coordinate features captured for {stem}')
    print(f'Captured {label}: {len(stems)} videos, {elapsed / 60:.2f} min', flush=True)
    return elapsed

_pilot = sorted(group[0] for group in _selected_by_embryo.values())
_pilot_seconds = _run_capture_subset(_pilot, 'pilot')
_expected_capture_minutes = (_pilot_seconds / 60) * len(test_stems) / len(_pilot)
_budget_minutes = float(SELF_CONFIG['runtime']['capture_budget_minutes'])
print('Projected capture minutes:', round(_expected_capture_minutes, 2),
      'budget:', _budget_minutes, flush=True)
if _expected_capture_minutes > _budget_minutes:
    raise RuntimeError('Pilot projection exceeds capture budget; remaining videos were not started')
_remaining = [stem for stem in test_stems if stem not in set(_pilot)]
_remaining_seconds = _run_capture_subset(_remaining, 'remaining')
_capture_seconds = _pilot_seconds + _remaining_seconds
if _capture_seconds / 60 > _budget_minutes:
    raise RuntimeError('Actual capture exceeded configured budget')

# %% [markdown]
# ## 3. Match known centers and assemble supervised examples

# %%

import numpy as np
import torch
import tracksdata as td
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

_GRID_SPACING = np.asarray(SELF_CONFIG['data']['feature_grid_spacing_zyx_um'], dtype=np.float32)
_NATIVE_SPACING = np.asarray(SELF_CONFIG['data']['native_spacing_zyx_um'], dtype=np.float32)
_MATCH_RADIUS = float(SELF_CONFIG['data']['candidate_match_radius_um'])
_FEATURE_WIDTH = int(SELF_CONFIG['data']['feature_width'])
_movie_arrays = {}
_capture_digest = _self_hashlib.sha256()
for stem in test_stems:
    graph = td.graph.IndexedRXGraph.from_geff(TRAIN_IMAGE_DIR / f'{stem}.geff')
    if isinstance(graph, tuple):
        graph = graph[0]
    gt_by_t = {}
    for row in graph.node_attrs().iter_rows(named=True):
        gt_by_t.setdefault(int(row['t']), []).append(
            np.asarray([row['z'], row['y'], row['x']], dtype=np.float32) * _NATIVE_SPACING)
    x_parts, y_parts, before_parts, frame_parts = [], [], [], []
    for path in sorted((_CAPTURE_ROOT / stem).glob('*.npz')):
        t = int(path.stem)
        with np.load(path, allow_pickle=False) as payload:
            coords = np.asarray(payload['coords'], dtype=np.float32)
            features = np.asarray(payload['features'], dtype=np.float32)
        valid_shape = coords.ndim == 2 and coords.shape[1] == 4
        valid_shape &= features.shape == (len(coords), _FEATURE_WIDTH)
        if not valid_shape:
            raise RuntimeError({'file': str(path),
                                'coords': coords.shape, 'features': features.shape})
        if not np.all(coords[:, 0] == t) or not np.isfinite(features).all():
            raise RuntimeError(f'Bad frame/features: {path}')
        _capture_digest.update(stem.encode())
        _capture_digest.update(path.name.encode())
        _capture_digest.update(np.ascontiguousarray(coords).tobytes())
        _capture_digest.update(np.ascontiguousarray(features).tobytes())
        gt = np.asarray(gt_by_t.get(t, []), dtype=np.float32).reshape(-1, 3)
        if not len(gt) or not len(coords):
            continue
        candidate_um = coords[:, 1:] * _GRID_SPACING
        cost = cdist(candidate_um, gt)
        rows, cols = linear_sum_assignment(np.where(cost <= _MATCH_RADIUS, cost, 1e6))
        valid = cost[rows, cols] <= _MATCH_RADIUS
        rows, cols = rows[valid], cols[valid]
        if len(rows):
            x_parts.append(features[rows])
            y_parts.append(gt[cols] - candidate_um[rows])
            before_parts.append(cost[rows, cols].astype(np.float32))
            frame_parts.append(np.full(len(rows), t, dtype=np.int16))
    if not x_parts:
        raise RuntimeError(f'No matched known centers for {stem}')
    _movie_arrays[stem] = {
        'x': np.concatenate(x_parts), 'y': np.concatenate(y_parts),
        'before': np.concatenate(before_parts), 't': np.concatenate(frame_parts),
        'embryo': stem.split('_', 1)[0],
    }
    print('Matched', stem, len(_movie_arrays[stem]['x']), flush=True)
if sum(len(row['x']) for row in _movie_arrays.values()) < 100:
    raise RuntimeError('Too few matched known centers')
_CAPTURE_SHA = _capture_digest.hexdigest()
print('Capture content SHA256:', _CAPTURE_SHA, flush=True)

# %% [markdown]
# ## 4. Video-grouped evaluation and final head training

# %%

import copy
import random
import torch.nn.functional as F

_PARAMS = SELF_CONFIG['model']['params']
_SEED = int(_PARAMS['seed'])
_FOLDS = int(_PARAMS['validation_folds'])
if _FOLDS != 5 or len(test_stems) != 20:
    raise RuntimeError('Five-fold 20-video contract changed')
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
torch.set_num_threads(2)


def _head():
    model = torch.nn.Sequential(torch.nn.Linear(_FEATURE_WIDTH, 32),
                                torch.nn.SiLU(), torch.nn.Linear(32, 3))
    torch.nn.init.zeros_(model[-1].weight)
    torch.nn.init.zeros_(model[-1].bias)
    return model


def _bounded(model, x):
    delta = model(x)
    norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    return float(_PARAMS['max_shift_um']) * delta / (1 + norm)


def _fit_head(train_stems: list[str], seed: int):
    torch.manual_seed(seed)
    x = np.concatenate([_movie_arrays[s]['x'] for s in train_stems]).astype(np.float32)
    y = np.concatenate([_movie_arrays[s]['y'] for s in train_stems]).astype(np.float32)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    xx = torch.from_numpy((x - mean) / scale)
    yy = torch.from_numpy(y)
    model = _head()
    opt = torch.optim.AdamW(model.parameters(), lr=float(_PARAMS['learning_rate']),
                            weight_decay=float(_PARAMS['weight_decay']))
    batch_size = int(_PARAMS['batch_size'])
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(int(_PARAMS['epochs'])):
        for batch in torch.randperm(len(xx), generator=generator).split(batch_size):
            prediction = _bounded(model, xx[batch])
            loss = F.huber_loss(prediction, yy[batch], delta=float(_PARAMS['huber_delta_um']))
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite train loss at epoch {epoch}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model, mean, scale


def _predict(model, mean, scale, stems):
    with torch.no_grad():
        return {stem: _bounded(model, torch.from_numpy(
            (_movie_arrays[stem]['x'] - mean) / scale)).numpy() for stem in stems}

_fold_stems = {fold: [] for fold in range(_FOLDS)}
for embryo, stems in _selected_by_embryo.items():
    ordered = sorted(stems, key=lambda s: (_stable_video_rank(s), s))
    for i, stem in enumerate(ordered):
        _fold_stems[i % _FOLDS].append(stem)
_oof_shifts = {}
for fold in range(_FOLDS):
    valid_stems = sorted(_fold_stems[fold])
    train_stems = sorted(set(test_stems) - set(valid_stems))
    model, mean, scale = _fit_head(train_stems, _SEED + fold)
    _oof_shifts.update(_predict(model, mean, scale, valid_stems))
    print('Fold', fold, 'train', len(train_stems), 'valid', valid_stems, flush=True)
if set(_oof_shifts) != set(test_stems):
    raise RuntimeError('Video-fold predictions incomplete')

_per_movie = []
_oof_digest = _self_hashlib.sha256()
for stem in test_stems:
    row = _movie_arrays[stem]
    after = np.linalg.vector_norm(row['y'] - _oof_shifts[stem], axis=1)
    before = row['before']
    _oof_digest.update(stem.encode())
    _oof_digest.update(np.ascontiguousarray(_oof_shifts[stem], dtype=np.float32).tobytes())
    _per_movie.append({'video': stem, 'embryo': row['embryo'], 'pairs': len(before),
                       'before_mean_um': float(before.mean()),
                       'after_mean_um': float(after.mean()),
                       'improved': bool(after.mean() < before.mean())})

_readout = {
    'experiment': 'exp043_x138_self_trained_head',
    'evaluation': 'video-grouped, conditional on pretrained public image models',
    'capture_sha256': _CAPTURE_SHA,
    'oof_prediction_sha256': _oof_digest.hexdigest(),
    'videos': len(_per_movie),
    'pairs': int(sum(row['pairs'] for row in _per_movie)),
    'per_movie': _per_movie,
    'by_embryo': {},
}
for embryo in sorted(_selected_by_embryo):
    stems = _selected_by_embryo[embryo]
    base = np.concatenate([_movie_arrays[s]['before'] for s in stems])
    refined = np.concatenate([np.linalg.vector_norm(
        _movie_arrays[s]['y'] - _oof_shifts[s], axis=1) for s in stems])
    _readout['by_embryo'][embryo] = {
        'videos': len(stems), 'pairs': len(base),
        'before_mean_um': float(base.mean()), 'after_mean_um': float(refined.mean()),
        'improved_videos': sum(row['improved'] for row in _per_movie if row['embryo'] == embryo),
    }
(WORKING_DIR / 'coordinate_validation.json').write_text(_self_json.dumps(_readout, indent=2) + '\n')
print(_self_json.dumps(_readout['by_embryo'], indent=2), flush=True)

_final_model, _final_mean, _final_scale = _fit_head(test_stems, _SEED + _FOLDS)
_HEAD_PATH = WORKING_DIR / 'self_trained_v1284_head.pt'
torch.save({'state_dict': {k: v.cpu() for k, v in _final_model.state_dict().items()},
            'mean': torch.from_numpy(_final_mean),
            'scale': torch.from_numpy(_final_scale)}, _HEAD_PATH)
_HEAD_SHA = _self_hashlib.sha256(_HEAD_PATH.read_bytes()).hexdigest()
_manifest = {
    'checkpoint': _HEAD_PATH.name, 'checkpoint_sha256': _HEAD_SHA,
    'reference_notebook_sha256': SELF_CONFIG['source']['reference_notebook_sha256'],
    'capture_content_sha256': _CAPTURE_SHA,
    'oof_prediction_sha256': _readout['oof_prediction_sha256'],
    'training_videos': test_stems, 'pairs': _readout['pairs'],
    'feature_width': _FEATURE_WIDTH, 'feature_grid_spacing_um': _GRID_SPACING.tolist(),
    'model': 'Linear(224,32)-SiLU-Linear(32,3)',
    'shift_bound_um': float(_PARAMS['max_shift_um']),
}
(WORKING_DIR / 'self_trained_head_manifest.json').write_text(
    _self_json.dumps(_manifest, indent=2, sort_keys=True) + '\n')
print('Final self-trained head:', _HEAD_PATH, _HEAD_SHA, flush=True)

# %% [markdown]
# ## 5. Verify saved training artifacts

# %%

_validation_path = WORKING_DIR / 'coordinate_validation.json'
_manifest_path = WORKING_DIR / 'self_trained_head_manifest.json'
for _required_path in (_validation_path, _manifest_path, _HEAD_PATH):
    if not _required_path.is_file():
        raise FileNotFoundError(_required_path)
_saved_validation = _self_json.loads(_validation_path.read_text())
_saved_manifest = _self_json.loads(_manifest_path.read_text())
if _saved_validation['capture_sha256'] != _CAPTURE_SHA:
    raise RuntimeError('Saved validation capture SHA differs from the current run')
_actual_head_sha = _self_hashlib.sha256(_HEAD_PATH.read_bytes()).hexdigest()
if _saved_manifest['checkpoint_sha256'] != _actual_head_sha:
    raise RuntimeError('Saved checkpoint SHA differs from the manifest')
if (_saved_validation['pairs'] != _saved_manifest['pairs']
        or _saved_validation['videos'] != len(test_stems)):
    raise RuntimeError('Saved training coverage differs from the manifest')
_receipt = {
    'experiment': 'exp043_x138_self_trained_head',
    'training_complete': True,
    'head_sha256': _saved_manifest['checkpoint_sha256'],
    'capture_sha256': _CAPTURE_SHA,
    'oof_prediction_sha256': _saved_validation['oof_prediction_sha256'],
    'training_videos': len(test_stems),
    'matched_pairs': _saved_validation['pairs'],
    'capture_seconds': _capture_seconds,
    'official_score': None,
    'competition_submission_created': False,
}
(WORKING_DIR / 'self_trained_head_train_receipt.json').write_text(
    _self_json.dumps(_receipt, indent=2, sort_keys=True) + '\n')
print(_self_json.dumps(_receipt, indent=2, sort_keys=True), flush=True)
