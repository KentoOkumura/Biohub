from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

INPUT_ARCHIVE = Path("/content/exp016_colab_smoke_inputs.tar.gz")
INPUT_ROOT = Path("/content/exp016_smoke")
OUTPUT_ROOT = Path("/content/exp016_smoke_output")
SCRATCH_ROOT = Path("/content/exp016_smoke_scratch")
LOG_PATH = Path("/content/exp016_smoke.log")
COMPLETION_PATH = Path("/content/exp016_smoke_completion.json")
RESULT_ARCHIVE = Path("/content/exp016_smoke_results.zip")
SAMPLE = "44b6_0113de3b"

DEPENDENCIES = [
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def log(message: str) -> None:
    stamp = datetime.now(UTC).isoformat()
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def extract_inputs() -> None:
    if not INPUT_ARCHIVE.is_file():
        raise FileNotFoundError(INPUT_ARCHIVE)
    if INPUT_ROOT.exists():
        shutil.rmtree(INPUT_ROOT)
    INPUT_ROOT.mkdir(parents=True)
    with tarfile.open(INPUT_ARCHIVE, "r:gz") as archive:
        for member in archive.getmembers():
            target = (INPUT_ROOT / member.name).resolve()
            if not target.is_relative_to(INPUT_ROOT.resolve()):
                raise RuntimeError(f"unsafe archive member: {member.name}")
        archive.extractall(INPUT_ROOT, filter="data")


def run_streaming(command: list[str]) -> None:
    log(json.dumps({"event": "command_start", "command": command}, sort_keys=True))
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def create_result_archive() -> str:
    if RESULT_ARCHIVE.exists():
        RESULT_ARCHIVE.unlink()
    with zipfile.ZipFile(RESULT_ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT_ROOT.rglob("*")):
            if path.is_file():
                archive.write(path, Path("exp016_smoke_output") / path.relative_to(OUTPUT_ROOT))
    return sha256_file(RESULT_ARCHIVE)


def main() -> None:
    started = time.monotonic()
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    if COMPLETION_PATH.exists():
        COMPLETION_PATH.unlink()
    log(
        json.dumps(
            {
                "event": "smoke_start",
                "input_archive_sha256": sha256_file(INPUT_ARCHIVE),
                "python": sys.version,
                "sample": SAMPLE,
            },
            sort_keys=True,
        )
    )
    run_streaming(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            *DEPENDENCIES,
        ]
    )
    extract_inputs()
    baseline_root = INPUT_ROOT / "exp016_colab_smoke_exp015"
    support_root = INPUT_ROOT / "exp016_support_smoke"
    command = [
        sys.executable,
        "/content/colab_graph_replay.py",
        "--code-root",
        "/content",
        "--baseline-root",
        str(baseline_root),
        "--repo-dir",
        str(support_root / "repo"),
        "--public-weights",
        str(support_root / "repo/weights/unet_transformer/split_0/edge_predictor_best.pth"),
        "--secondary-weights",
        str(support_root / "secondary_weights/edge_predictor_best.pth"),
        "--fold-0-weights",
        "/content/fold_0_primary_tracker_best.pth",
        "--fold-1-weights",
        "/content/fold_1_primary_tracker_best.pth",
        "--output-root",
        str(OUTPUT_ROOT),
        "--scratch-root",
        str(SCRATCH_ROOT),
        "--sample",
        SAMPLE,
        "--batch-size",
        "1",
        "--runtime-gate-hours",
        "12",
        "--resume",
    ]
    run_streaming(command)

    summary_path = OUTPUT_ROOT / "colab_cache_replay_summary.json"
    receipt_path = OUTPUT_ROOT / "batch_receipts/batch_000.json"
    archive_path = OUTPUT_ROOT / "batch_archives/batch_000.zip"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checks = {
        "summary_status_complete": summary.get("status") == "complete",
        "sample_count_one": summary.get("sample_count") == 1,
        "window_count_99": summary.get("window_count") == 99,
        "public_control_exact": summary.get("public_control_candidate_graphs_exact") is True,
        "main_image_encoder_not_run": summary.get("main_image_encoder_forward_count") == 0,
        "tracker_forward_count_five": summary.get("tracker_forward_count_per_window") == 5,
        "persistent_receipt_exact": receipt.get("public_control_candidate_graphs_exact") is True,
        "batch_archive_present": archive_path.is_file(),
        "batch_archive_sha_matches": archive_path.is_file()
        and sha256_file(archive_path) == receipt.get("archive_sha256"),
    }
    if not all(checks.values()):
        raise RuntimeError({"smoke_checks_failed": checks})
    completion = {
        "status": "complete",
        "sample": SAMPLE,
        "checks": checks,
        "elapsed_seconds": time.monotonic() - started,
        "summary_sha256": sha256_file(summary_path),
        "receipt_sha256": sha256_file(receipt_path),
        "batch_archive_sha256": sha256_file(archive_path),
        "result_archive": str(RESULT_ARCHIVE),
    }
    COMPLETION_PATH.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    completion["result_archive_sha256"] = create_result_archive()
    COMPLETION_PATH.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log(json.dumps({"event": "smoke_complete", **completion}, sort_keys=True))


if __name__ == "__main__":
    main()
