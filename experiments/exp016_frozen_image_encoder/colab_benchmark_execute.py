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

INPUT_ARCHIVE = Path("/content/exp016_colab_benchmark_inputs_v1.tar.gz")
INPUT_ROOT = Path("/content/exp016_benchmark")
OUTPUT_ROOT = Path("/content/exp016_benchmark_output")
SCRATCH_ROOT = Path("/content/exp016_benchmark_scratch")
LOG_PATH = Path("/content/exp016_benchmark.log")
COMPLETION_PATH = Path("/content/exp016_benchmark_completion.json")
RESULT_ARCHIVE = Path("/content/exp016_benchmark_results.zip")
SAMPLES = ["44b6_0113de3b", "6bba_05b6850b"]
RUNTIME_GATE_SECONDS = 12 * 60 * 60
PROJECTION_RESERVE = 1.25

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
                archive.write(path, Path("exp016_benchmark_output") / path.relative_to(OUTPUT_ROOT))
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
                "event": "benchmark_start",
                "input_archive_sha256": sha256_file(INPUT_ARCHIVE),
                "python": sys.version,
                "samples": SAMPLES,
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
    baseline_root = INPUT_ROOT / "exp016_colab_benchmark_exp015_v1"
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
        "--batch-size",
        "1",
        "--runtime-gate-hours",
        "12",
        "--resume",
    ]
    for sample in SAMPLES:
        command.extend(["--sample", sample])
    run_streaming(command)

    summary_path = OUTPUT_ROOT / "colab_cache_replay_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = {str(row["sample"]): row for row in summary.get("samples", [])}
    projected_seconds = (
        float(summary["elapsed_seconds"]) / int(summary["sample_count"]) * 199 * PROJECTION_RESERVE
    )
    checks = {
        "summary_status_complete": summary.get("status") == "complete",
        "sample_count_two": summary.get("sample_count") == 2,
        "window_count_198": summary.get("window_count") == 198,
        "public_control_exact": summary.get("public_control_candidate_graphs_exact") is True,
        "main_image_encoder_not_run": summary.get("main_image_encoder_forward_count") == 0,
        "tracker_forward_count_five": summary.get("tracker_forward_count_per_window") == 5,
        "both_samples_present": sorted(rows) == sorted(SAMPLES),
        "both_samples_exact": all(rows.get(sample, {}).get("exact") is True for sample in SAMPLES),
        "both_samples_have_99_windows": all(
            rows.get(sample, {}).get("window_count") == 99 for sample in SAMPLES
        ),
        "fold_mapping_exact": rows.get("44b6_0113de3b", {}).get("fold") == 1
        and rows.get("6bba_05b6850b", {}).get("fold") == 0,
        "two_batch_archives_present": all(
            (OUTPUT_ROOT / f"batch_archives/batch_{index:03d}.zip").is_file() for index in range(2)
        ),
        "within_12_hour_gate": projected_seconds < RUNTIME_GATE_SECONDS,
    }
    if not all(checks.values()):
        raise RuntimeError({"benchmark_checks_failed": checks})

    completion = {
        "status": "complete",
        "stage": "two_sample_colab_cache_replay_benchmark",
        "samples": SAMPLES,
        "checks": checks,
        "replay_elapsed_seconds": summary["elapsed_seconds"],
        "end_to_end_elapsed_seconds": time.monotonic() - started,
        "projected_full_seconds_with_25pct_reserve": projected_seconds,
        "runtime_gate_seconds": RUNTIME_GATE_SECONDS,
        "summary_sha256": sha256_file(summary_path),
        "result_archive": str(RESULT_ARCHIVE),
    }
    COMPLETION_PATH.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    completion["result_archive_sha256"] = create_result_archive()
    COMPLETION_PATH.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log(json.dumps({"event": "benchmark_complete", **completion}, sort_keys=True))


if __name__ == "__main__":
    main()
