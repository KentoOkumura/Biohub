# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# # exp016 frozen image encoder: Colab cache replay
#
# This notebook runs the GPU-dependent tracker and ILP part of exp016 on one
# Colab GPU. It does not retrain any model and never runs the main image encoder.
# A two-sample benchmark must reproduce the saved exp015 public candidate graph
# exactly before the full 199-sample replay starts. Outputs and logs are written
# to Google Drive in resumable batches.
#
# Before running:
#
# 1. Upload `exp016_colab_bundle.zip` to `BUNDLE_ON_DRIVE`.
# 2. In Colab Secrets, create `KAGGLE_API_TOKEN` and allow this notebook to use it.
# 3. Select a GPU runtime. One T4 is sufficient for this stage.
#
# `INPUT_MODE="drive"` is also supported when the Kaggle outputs and public
# support files have already been placed under `DRIVE_INPUT_ROOT`.

# %%
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# %% [markdown]
# ## User settings

# %%
RUN_MODE = "benchmark_then_full"  # benchmark, full, or benchmark_then_full
INPUT_MODE = "kaggle"  # kaggle or drive
DRIVE_MOUNT = Path("/content/drive")
DRIVE_PROJECT_ROOT = DRIVE_MOUNT / "MyDrive/Kaggle/Biohub/exp016_colab"
BUNDLE_ON_DRIVE = DRIVE_PROJECT_ROOT / "exp016_colab_bundle.zip"
DRIVE_INPUT_ROOT = DRIVE_PROJECT_ROOT / "inputs"
DRIVE_RUNS_ROOT = DRIVE_PROJECT_ROOT / "artifacts/colab_runs"
FULL_RUN_ID = "cache_replay_v1"
BENCHMARK_SAMPLES = ["44b6_0113de3b", "6bba_05b6850b"]
RUNTIME_GATE_HOURS = 12.0
FULL_BATCH_SIZE = 5

if RUN_MODE not in {"benchmark", "full", "benchmark_then_full"}:
    raise ValueError(f"unsupported RUN_MODE: {RUN_MODE}")
if INPUT_MODE not in {"kaggle", "drive"}:
    raise ValueError(f"unsupported INPUT_MODE: {INPUT_MODE}")


# %% [markdown]
# ## Mount Drive and initialize persistent logging

# %%
from google.colab import drive  # noqa: E402

drive.mount(str(DRIVE_MOUNT))
DRIVE_PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
DRIVE_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
SESSION_ROOT = DRIVE_RUNS_ROOT / FULL_RUN_ID
SESSION_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH = SESSION_ROOT / "colab_cache_replay.log"
LOCAL_ROOT = Path("/content/exp016_colab")
LOCAL_INPUT_ROOT = LOCAL_ROOT / "inputs"
LOCAL_BUNDLE_ROOT = LOCAL_ROOT / "bundle"
LOCAL_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
LOCAL_BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)


def append_log(message: str) -> None:
    stamp = datetime.now(UTC).isoformat()
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


append_log(
    json.dumps(
        {
            "event": "session_start",
            "run_mode": RUN_MODE,
            "input_mode": INPUT_MODE,
            "python": sys.version,
            "runtime_gate_hours": RUNTIME_GATE_HOURS,
        },
        sort_keys=True,
    )
)


# %% [markdown]
# ## GPU, disk, and dependency preflight

# %%
import torch  # noqa: E402

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError(f"Select a one-GPU Colab runtime; visible GPUs: {torch.cuda.device_count()}")
disk_total, disk_used, disk_free = shutil.disk_usage("/content")
preflight = {
    "gpu_name": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "python": sys.version,
    "disk_total_bytes": disk_total,
    "disk_free_bytes": disk_free,
}
(SESSION_ROOT / "runtime_preflight.json").write_text(
    json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
append_log(json.dumps({"event": "runtime_preflight", **preflight}, sort_keys=True))

DEPENDENCIES = [
    "kaggle==2.2.4",
    "blosc2>=3.7,<5",
    "geff>=1.2,<2",
    "geff-spec>=1.1,<2",
    "ilpy>=0.6,<1",
    "polars>=1.42,<2",
    "pyarrow>=20,<25",
    "pyscipopt>=6.0,<7",
    "rustworkx>=0.17,<1",
    "tracksdata>=0.1.0rc6,<0.2",
    "zarr>=3.0.10,<4",
]
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        *DEPENDENCIES,
    ],
    check=True,
)


# %% [markdown]
# ## Verify and extract the small code/model bundle


# %%
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


if not BUNDLE_ON_DRIVE.is_file():
    raise FileNotFoundError(f"Upload exp016_colab_bundle.zip to {BUNDLE_ON_DRIVE} before running")
with zipfile.ZipFile(BUNDLE_ON_DRIVE) as archive:
    names = archive.namelist()
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise RuntimeError("unsafe path found in Colab bundle")
    archive.extractall(LOCAL_BUNDLE_ROOT)
bundle_manifest = json.loads(
    (LOCAL_BUNDLE_ROOT / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8")
)
if bundle_manifest.get("experiment") != "exp016_frozen_image_encoder":
    raise RuntimeError("wrong experiment in Colab bundle")
if bundle_manifest.get("contains_credentials") is not False:
    raise RuntimeError("Colab bundle credential declaration changed")
for item in bundle_manifest["files"]:
    path = LOCAL_BUNDLE_ROOT / item["path"]
    if not path.is_file() or sha256_file(path) != item["sha256"]:
        raise RuntimeError(f"Colab bundle checksum failed: {item['path']}")
append_log(
    json.dumps(
        {
            "event": "bundle_verified",
            "bundle_sha256": sha256_file(BUNDLE_ON_DRIVE),
            "manifest_sha256": bundle_manifest["manifest_sha256"],
        },
        sort_keys=True,
    )
)


# %% [markdown]
# ## Input acquisition
#
# Kaggle mode reads `KAGGLE_API_TOKEN` from Colab Secrets. The value is never
# printed or written to Drive. Drive mode copies previously downloaded inputs
# into `/content` before GPU work begins.


# %%
def run_logged(command: list[str], *, env: dict[str, str] | None = None) -> None:
    append_log(json.dumps({"event": "command_start", "command": command}, sort_keys=True))
    with LOG_PATH.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    append_log(json.dumps({"event": "command_complete", "command": command}, sort_keys=True))


KAGGLE_ENV: dict[str, str] | None = None
if INPUT_MODE == "kaggle":
    from google.colab import userdata

    try:
        kaggle_token = userdata.get("KAGGLE_API_TOKEN")
    except Exception as exc:
        raise RuntimeError(
            "Create the KAGGLE_API_TOKEN Colab Secret and grant notebook access"
        ) from exc
    if not kaggle_token:
        raise RuntimeError("KAGGLE_API_TOKEN Colab Secret is empty")
    KAGGLE_ENV = {**os.environ, "KAGGLE_API_TOKEN": kaggle_token}
    del kaggle_token


def prepare_support_inputs() -> tuple[Path, Path, Path]:
    primary_root = LOCAL_INPUT_ROOT / "primary_support"
    secondary_root = LOCAL_INPUT_ROOT / "secondary_support"
    if INPUT_MODE == "kaggle":
        if not (primary_root / "repo/scripts/predict_unet_transformer.py").is_file():
            primary_root.mkdir(parents=True, exist_ok=True)
            run_logged(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "pilkwang/biohub-tracking-support-pack-50ep-v1",
                    "--path",
                    str(primary_root),
                    "--unzip",
                    "--quiet",
                ],
                env=KAGGLE_ENV,
            )
        secondary_weight = secondary_root / "edge_predictor_best.pth"
        if not secondary_weight.is_file():
            secondary_root.mkdir(parents=True, exist_ok=True)
            run_logged(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
                    "--file",
                    "weights/unet_transformer/split_0/edge_predictor_best.pth",
                    "--path",
                    str(secondary_root),
                    "--unzip",
                    "--quiet",
                ],
                env=KAGGLE_ENV,
            )
    else:
        source_primary = DRIVE_INPUT_ROOT / "primary_support"
        source_secondary = DRIVE_INPUT_ROOT / "secondary_support"
        if not primary_root.exists():
            shutil.copytree(source_primary, primary_root)
        if not secondary_root.exists():
            shutil.copytree(source_secondary, secondary_root)
    repo_dir = primary_root / "repo"
    public_weight = primary_root / "weights/unet_transformer/split_0/edge_predictor_best.pth"
    secondary_weight = secondary_root / "edge_predictor_best.pth"
    for path in (repo_dir / "scripts/predict_unet_transformer.py", public_weight, secondary_weight):
        if not path.is_file():
            raise FileNotFoundError(f"support input is missing: {path}")
    return repo_dir, public_weight, secondary_weight


def prepare_exp015_inputs(mode: str) -> Path:
    target = LOCAL_INPUT_ROOT / f"exp015_{mode}"
    if mode == "benchmark":
        sample_pattern = "|".join(BENCHMARK_SAMPLES)
        pattern = (
            rf"^(window_cache/({sample_pattern})/.*|"
            rf"oracle_candidate_graphs/({sample_pattern})\.geff/.*|"
            r"oracle_inference_manifest\.json)$"
        )
    elif mode == "full":
        pattern = (
            r"^(window_cache/.*|oracle_candidate_graphs/.*|"
            r"window_cache_summary\.json|oracle_inference_manifest\.json)$"
        )
    else:
        raise ValueError(mode)
    if INPUT_MODE == "kaggle":
        target.mkdir(parents=True, exist_ok=True)
        run_logged(
            [
                "kaggle",
                "kernels",
                "output",
                "kentookumura/exp015-oracle-stage-limits-inference",
                "--path",
                str(target),
                "--file-pattern",
                pattern,
                "--page-size",
                "200",
                "--quiet",
            ],
            env=KAGGLE_ENV,
        )
    else:
        source = DRIVE_INPUT_ROOT / f"exp015_{mode}"
        if not target.exists():
            shutil.copytree(source, target)
    return target


REPO_DIR, PUBLIC_WEIGHT, SECONDARY_WEIGHT = prepare_support_inputs()


# %% [markdown]
# ## Resumable one-GPU cache replay


# %%
def run_replay(mode: str) -> dict[str, Any]:
    baseline_root = prepare_exp015_inputs(mode)
    output_root = SESSION_ROOT / mode
    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(LOCAL_BUNDLE_ROOT / "code/colab_graph_replay.py"),
        "--code-root",
        str(LOCAL_BUNDLE_ROOT / "code"),
        "--baseline-root",
        str(baseline_root),
        "--repo-dir",
        str(REPO_DIR),
        "--public-weights",
        str(PUBLIC_WEIGHT),
        "--secondary-weights",
        str(SECONDARY_WEIGHT),
        "--fold-0-weights",
        str(LOCAL_BUNDLE_ROOT / "models/fold_0/primary_tracker_best.pth"),
        "--fold-1-weights",
        str(LOCAL_BUNDLE_ROOT / "models/fold_1/primary_tracker_best.pth"),
        "--output-root",
        str(output_root),
        "--scratch-root",
        str(LOCAL_ROOT / "scratch" / mode),
        "--batch-size",
        "1" if mode == "benchmark" else str(FULL_BATCH_SIZE),
        "--runtime-gate-hours",
        str(RUNTIME_GATE_HOURS),
        "--resume",
    ]
    if mode == "benchmark":
        for sample in BENCHMARK_SAMPLES:
            command.extend(["--sample", sample])
    else:
        command.append("--require-full")
    started = time.monotonic()
    try:
        run_logged(command)
    except Exception as exc:
        failure = {
            "stage": f"colab_cache_replay_{mode}",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.monotonic() - started,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        (output_root / "failure.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    summary_path = output_root / "colab_cache_replay_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("public_control_candidate_graphs_exact"):
        raise RuntimeError("public control candidate graph exactness gate failed")
    completion = {
        "stage": f"colab_cache_replay_{mode}",
        "status": "complete",
        "summary_sha256": sha256_file(summary_path),
        "sample_count": summary["sample_count"],
        "window_count": summary["window_count"],
        "main_image_encoder_forward_count": 0,
        "official_graph_evaluation_complete": False,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    (output_root / "cache_replay_complete.json").write_text(
        json.dumps(completion, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    append_log(json.dumps({"event": "replay_complete", "mode": mode, **completion}, sort_keys=True))
    return summary


benchmark_summary: dict[str, Any] | None = None
full_summary: dict[str, Any] | None = None
if RUN_MODE in {"benchmark", "benchmark_then_full"}:
    benchmark_summary = run_replay("benchmark")
    projected_seconds = (
        float(benchmark_summary["elapsed_seconds"])
        / int(benchmark_summary["sample_count"])
        * 199
        * 1.25
    )
    benchmark_gate = {
        "projected_full_seconds_with_25pct_reserve": projected_seconds,
        "runtime_gate_seconds": RUNTIME_GATE_HOURS * 60 * 60,
        "within_gate": projected_seconds < RUNTIME_GATE_HOURS * 60 * 60,
        "public_control_candidate_graphs_exact": benchmark_summary[
            "public_control_candidate_graphs_exact"
        ],
    }
    (SESSION_ROOT / "benchmark_gate.json").write_text(
        json.dumps(benchmark_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    append_log(json.dumps({"event": "benchmark_gate", **benchmark_gate}, sort_keys=True))
    if not benchmark_gate["within_gate"]:
        raise TimeoutError("benchmark projection exceeds the 12-hour runtime gate")
if RUN_MODE in {"full", "benchmark_then_full"}:
    full_summary = run_replay("full")


# %% [markdown]
# ## Stage result
#
# A successful full run means the cached tracker and ILP graphs were generated
# for all 199 annotated train movies and the public tracker reproduced every
# saved exp015 candidate graph exactly. Graph repair and the official metric are
# intentionally not claimed here; they are the next Colab stage.

# %%
display_payload = {
    "benchmark": None
    if benchmark_summary is None
    else {
        "sample_count": benchmark_summary["sample_count"],
        "elapsed_seconds": benchmark_summary["elapsed_seconds"],
        "exact": benchmark_summary["public_control_candidate_graphs_exact"],
    },
    "full": None
    if full_summary is None
    else {
        "sample_count": full_summary["sample_count"],
        "elapsed_seconds": full_summary["elapsed_seconds"],
        "exact": full_summary["public_control_candidate_graphs_exact"],
        "summary_sha256": full_summary["summary_sha256"],
    },
    "persistent_root": str(SESSION_ROOT),
    "official_graph_evaluation_complete": False,
}
print(json.dumps(display_payload, indent=2, ensure_ascii=False, sort_keys=True))
