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
# # exp045: official evaluation from the saved version 1 final graphs
#
# Version 1 completed all 40 graphs and failed at the first GT read in the
# fixed-ID diagnostic. This CPU notebook runs the public evaluator on those
# immutable graphs. The larger fixed-ID diagnostic is run from saved files
# separately; neither step reruns prediction.

# %%
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

AUDIT_START = time.time()
PREDICTION_KERNEL = "kentookumura/exp045-x138-coordinate-effect-audit-inference"
PREDICTION_VERSION = 1
RECOVERY_DATASET = "kentookumura/exp045-x138-coordinate-audit-v1-artifacts"
EXPECTED_ARCHIVE_SHA = "8315c5b394c524ffee283561effc231a3b88aa1cac8f7954ec3a71837115f2d1"
EXPECTED_DATASET_FILE_MANIFEST_SHA = "868282ef3065bfb7d30249a38d80234d3681856d9284559d82f2af5ab3b788a2"
EXPECTED_SELECTION_SHA = "ba7eb44348be9ee18a0a15180276dbb92399a5c5995e26ae3b3d667b8cc30daa"
EXPECTED_RUNTIME_GATE_SHA = "b5b53aacfae8c67a81bb75be00aaeda6c1809ca2faceaf80f1a3bc75b7ebd85c"
EXPECTED_HEAD_SHA = "32d6c62f738c3dfe4862e3df5272850312e81b622e57d9ba45209ebb382824dc"
EXPECTED_EVALUATOR_SHA = "614813cc51c3581c6ccda4bb20725a19da8ecac4a27620654bfca58319cffa3c"
COMPETITION = "biohub-cell-tracking-during-development"
ARMS = ("zero", "refined")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


input_root = Path("/kaggle/input")
slug = RECOVERY_DATASET.split("/", 1)[1]
dataset_candidates = (
    input_root / "datasets" / "kentookumura" / slug,
    input_root / slug,
)
dataset_root = next((p for p in dataset_candidates if (p / "exp045_selection.json").is_file()), None)
if dataset_root is None:
    raise FileNotFoundError(f"private graph Dataset is not mounted: {dataset_candidates}")
# Kaggle expands uploaded tar files into Dataset members. Compare every mounted
# file against the manifest calculated from the version 1 output before upload.
dataset_files = sorted(p for p in dataset_root.rglob("*") if p.is_file())
dataset_manifest = [
    {
        "path": p.relative_to(dataset_root).as_posix(),
        "bytes": p.stat().st_size,
        "sha256": sha256(p),
    }
    for p in dataset_files
]
dataset_manifest_sha = hashlib.sha256(
    (json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n").encode()
).hexdigest()
if len(dataset_files) != 842 or dataset_manifest_sha != EXPECTED_DATASET_FILE_MANIFEST_SHA:
    raise RuntimeError("mounted Dataset differs from the saved graph manifest")
audit_root = dataset_root / "exp045_audit"
selection_path = dataset_root / "exp045_selection.json"
gate_path = audit_root / "runtime_gate.json"
if sha256(selection_path) != EXPECTED_SELECTION_SHA:
    raise RuntimeError("video selection manifest SHA changed")
if sha256(gate_path) != EXPECTED_RUNTIME_GATE_SHA:
    raise RuntimeError("prediction runtime gate SHA changed")
selection_receipt = json.loads(selection_path.read_text())
selection = selection_receipt["selected"]
runtime_gate = json.loads(gate_path.read_text())
test_stems = sorted(stem for group in selection.values() for stem in group)
if (
    set(selection) != {"44b6", "6bba"}
    or any(len(group) != 10 for group in selection.values())
    or len(set(test_stems)) != 20
    or set(test_stems) & set(selection_receipt["trained"])
    or selection_receipt["head_sha256"] != EXPECTED_HEAD_SHA
    or runtime_gate["pilot"] != sorted(group[0] for group in selection.values())
):
    raise RuntimeError("saved selection, head, and runtime gate disagree")
for arm in ARMS:
    geffs = sorted((audit_root / arm / "final_graphs").glob("*.geff"))
    if {p.stem for p in geffs} != set(test_stems):
        raise RuntimeError(f"{arm}: final graph coverage differs from the 20 selected videos")
    if any(not any(p.rglob("*")) for p in geffs):
        raise RuntimeError(f"{arm}: empty final graph directory")
comp_dir = next(
    (p for p in (
        input_root / "competitions" / COMPETITION,
        input_root / COMPETITION,
    ) if p.exists()),
    None,
)
if comp_dir is None:
    raise FileNotFoundError("competition train input is not mounted")
train_dir = comp_dir / "train"
if any(not (train_dir / f"{stem}.geff").is_dir() for stem in test_stems):
    raise RuntimeError("selected GT GEFF is missing")
audit_out = Path("/kaggle/working/exp045_official_resume")
audit_out.mkdir(parents=True, exist_ok=True)
print("Selected 20 videos and 40 saved final graphs: PASS", flush=True)

# %% [markdown]
# ## Load the same public evaluator used in the pilot

# %%
_support = input_root / "datasets" / "pilkwang" / "biohub-tracking-support-pack-50ep-v1"
_wheels = _support / "wheels"
if not _wheels.is_dir():
    raise FileNotFoundError("public support pack offline wheels are missing")
# Kaggle's preinstalled Polars can import but lacks Float16, which the pinned
# tracksdata wheel needs at import time. Match the successful GPU notebook.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
     "--force-reinstall", "--find-links", str(_wheels),
     "polars", "polars-runtime-32"],
    check=True,
)
_required_imports = (
    "polars", "tracksdata", "zarr", "pyscipopt", "geff", "geff_spec",
    "ilpy", "imagecodecs", "rustworkx", "numcodecs", "donfig", "bidict",
)
_missing = [name for name in _required_imports if importlib.util.find_spec(name) is None]
if _missing:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
         "--find-links", str(_wheels), *_missing],
        check=True,
    )
repo_dir = Path("/kaggle/working/tracking_repo")
if repo_dir.exists():
    raise RuntimeError("unexpected preexisting public evaluator directory")
if (_support / "repo").is_dir():
    shutil.copytree(_support / "repo", repo_dir)
elif (_support / "repo.zip").is_file():
    repo_dir.mkdir(parents=True)
    with zipfile.ZipFile(_support / "repo.zip") as archive:
        archive.extractall(repo_dir)
else:
    raise FileNotFoundError("public support pack source repo is missing")
sys.path.insert(0, str(repo_dir / "src"))
sys.path.insert(0, str(repo_dir / "scripts"))
from biohub_tracking.metrics import summarise
import evaluate

evaluator_path = repo_dir / "scripts" / "evaluate.py"
if sha256(evaluator_path) != EXPECTED_EVALUATOR_SHA:
    raise RuntimeError("public evaluator SHA changed")
if Path(evaluate.DATA_DIR).resolve() != train_dir.resolve():
    raise RuntimeError("public evaluator points to a different GT directory")
# evaluate_run uses only ds.tracks and ds.scale. The default loader also loads
# image data and calls CUDA-only pinned memory; skip that unused image data.
_original_open_dataset = evaluate.open_dataset


def _open_gt_without_image(path, **kwargs):
    if "load_image" in kwargs:
        raise RuntimeError("unexpected evaluator image-loader override")
    return _original_open_dataset(path, load_image=False, **kwargs)


evaluate.open_dataset = _open_gt_without_image

# %% [markdown]
# ## Run the official evaluator on both complete sets of final graphs
#
# These train subset scores are conditional on the fixed public image models,
# which were trained with competition train images. They are not Public LB.

# %%
def evaluate_arm(arm: str) -> dict[str, object]:
    geffs = sorted((audit_root / arm / "final_graphs").glob("*.geff"))
    run = {
        "username": "exp045", "method": arm, "split": "split_0",
        "dir": audit_root / arm / "final_graphs", "geffs": geffs,
    }
    rows = evaluate.evaluate_run(run, max_distance=7.0)
    names = [str(row["dataset"]) for row in rows]
    if len(rows) != 20 or set(names) != set(test_stems):
        raise RuntimeError(f"{arm}: evaluator returned incomplete rows")
    if any(not math.isfinite(float(row.get("edge_tp", float("nan")))) for row in rows):
        raise RuntimeError(f"{arm}: nonfinite edge counts")
    overall = summarise(rows)
    by_embryo = {
        embryo: summarise([
            row for row in rows if str(row["dataset"]).split("_", 1)[0] == embryo
        ])
        for embryo in ("44b6", "6bba")
    }
    recheck = summarise(evaluate.evaluate_run(run, max_distance=7.0))
    if jsonable(recheck) != jsonable(overall):
        raise RuntimeError(f"{arm}: official metric recomputation differs")
    return {
        "overall": overall, "by_embryo": by_embryo, "rows": rows,
        "recomputed_summary_matches": True,
    }


official = {arm: evaluate_arm(arm) for arm in ARMS}
(audit_out / "official_metric.json").write_text(
    json.dumps(jsonable(official), indent=2, sort_keys=True) + "\n"
)
for arm in ARMS:
    print(arm, jsonable(official[arm]["overall"]), flush=True)

# %% [markdown]
# ## Save source graph hashes and receipt

# %%
source_files = sorted(p for p in audit_root.rglob("*") if p.is_file())
source_manifest = [
    {
        "path": p.relative_to(audit_root).as_posix(),
        "bytes": p.stat().st_size,
        "sha256": sha256(p),
    }
    for p in source_files
]
manifest_path = audit_out / "source_file_manifest.json"
manifest_path.write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n")
receipt = {
    "experiment": "exp045_x138_coordinate_effect_audit",
    "prediction_kernel_id": PREDICTION_KERNEL,
    "prediction_kernel_version": PREDICTION_VERSION,
    "prediction_run_failed_after_all_final_graphs": True,
    "resume_mode": "official_evaluator_from_saved_final_graphs_cpu",
    "recovery_dataset": RECOVERY_DATASET,
    "recovery_upload_archive_sha256": EXPECTED_ARCHIVE_SHA,
    "recovery_dataset_file_manifest_sha256": dataset_manifest_sha,
    "selection_manifest_sha256": EXPECTED_SELECTION_SHA,
    "runtime_gate_sha256": EXPECTED_RUNTIME_GATE_SHA,
    "head_sha256": EXPECTED_HEAD_SHA,
    "official_evaluator_sha256": EXPECTED_EVALUATOR_SHA,
    "evaluator_dataset_loader": "public_open_dataset_load_image_false",
    "selected_videos": test_stems,
    "selected_by_embryo": selection,
    "arms": list(ARMS),
    "official_scores": {
        arm: {
            "overall": official[arm]["overall"],
            "by_embryo": official[arm]["by_embryo"],
        }
        for arm in ARMS
    },
    "source_file_count": len(source_manifest),
    "source_total_bytes": sum(row["bytes"] for row in source_manifest),
    "source_file_manifest_sha256": sha256(manifest_path),
    "diagnostic_runtime_seconds": time.time() - AUDIT_START,
    "prediction_rerun": False,
    "competition_submission_created": False,
}
(audit_out / "exp045_official_receipt.json").write_text(
    json.dumps(jsonable(receipt), indent=2, sort_keys=True) + "\n"
)
print("Official evaluation receipt:", audit_out / "exp045_official_receipt.json", flush=True)
