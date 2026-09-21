from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

EXPERIMENT = "exp032_three_frame_ten_epoch_training"
STAGE_PATH = Path("/exp032_stage.json")
BUNDLE_ROOT = Path("/content/exp032_cli_bundle")
INPUT_ROOT = Path("/content") / EXPERIMENT / "inputs"
RUN_ROOT = Path("/content") / EXPERIMENT / "run"
OUTPUT_CHUNK_BYTES = 16 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise RuntimeError(f"Unsafe archive or manifest path: {value!r}")
    return path


def assemble(parts: list[dict], output: Path) -> None:
    if not parts:
        raise RuntimeError("No transfer parts declared")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".tmp")
    with temporary.open("wb") as destination:
        for row in parts:
            source = Path("/") / safe_relative(row["name"])
            if not source.is_file() or source.stat().st_size != row["bytes"]:
                raise RuntimeError(f"Missing transfer part: {row['name']}")
            if sha256(source) != row["sha256"]:
                raise RuntimeError(f"Transfer part checksum mismatch: {row['name']}")
            with source.open("rb") as handle:
                shutil.copyfileobj(handle, destination)
    temporary.replace(output)


def extract_bundle(archive_path: Path, expected_sha256: str) -> None:
    if sha256(archive_path) != expected_sha256:
        raise RuntimeError("CLI bundle SHA mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
        if (
            manifest.get("experiment") != EXPERIMENT
            or manifest.get("contains_credentials") is not False
        ):
            raise RuntimeError("Unexpected CLI bundle manifest")
        names = {row["path"] for row in manifest["files"]}
        if set(archive.namelist()) != names | {"BUNDLE_MANIFEST.json"}:
            raise RuntimeError("CLI bundle file list differs from manifest")
        for row in manifest["files"]:
            relative = safe_relative(row["path"])
            target = BUNDLE_ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(row["path"]) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            if target.stat().st_size != row["bytes"] or sha256(target) != row["sha256"]:
                raise RuntimeError(f"CLI bundle checksum failed: {relative}")
    if not (BUNDLE_ROOT / "project.yml").is_file():
        raise RuntimeError("CLI bundle has no project.yml")


def extract_seed(archive_path: Path, expected_sha256: str) -> None:
    if sha256(archive_path) != expected_sha256:
        raise RuntimeError("Resume seed SHA mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            relative = safe_relative(name)
            if not name.startswith("run/") or name.endswith("/"):
                raise RuntimeError(f"Unexpected resume member: {name}")
            target = RUN_ROOT / relative.relative_to("run")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def load_manifest(path: Path, expected_source: str) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != EXPERIMENT or payload.get("source") != expected_source:
        raise RuntimeError(f"Unexpected signed URL manifest source: {path.name}")
    rows = payload.get("files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Empty signed URL manifest: {path.name}")
    names = [str(row["path"]) for row in rows]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Duplicate signed URL manifest paths: {path.name}")
    for row in rows:
        safe_relative(str(row["path"]))
        parsed = urlparse(str(row["url"]))
        if parsed.scheme != "https" or not parsed.netloc:
            raise RuntimeError(f"Non-HTTPS signed URL for {row['path']}")
    return rows


def fetch_one(row: dict, destination_root: Path) -> int:
    relative = safe_relative(str(row["path"]))
    target = destination_root / relative
    expected_bytes = row.get("size_bytes")
    if target.is_file() and (
        (expected_bytes is None or target.stat().st_size == expected_bytes)
        and (row.get("sha256") is None or sha256(target) == row["sha256"])
    ):
        return target.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("." + target.name + ".tmp")
    for attempt in range(5):
        try:
            request = urllib.request.Request(row["url"], headers={"User-Agent": "exp032-colab-cli"})
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as output,
            ):
                shutil.copyfileobj(response, output, length=1024 * 1024)
            actual = temporary.stat().st_size
            if expected_bytes is not None and actual != expected_bytes:
                raise RuntimeError("download byte count differs from Kaggle listing")
            expected_sha = row.get("sha256")
            if expected_sha is not None and sha256(temporary) != expected_sha:
                raise RuntimeError("download SHA differs from saved prediction manifest")
            temporary.replace(target)
            return actual
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            if attempt == 4:
                code = getattr(exc, "code", None)
                raise RuntimeError(
                    f"Kaggle download failed for {relative}: {type(exc).__name__}, status={code}"
                ) from None
            time.sleep(min(30, 2 ** (attempt + 1)))
    raise AssertionError("unreachable")


def extract_geff(archive_path: Path) -> None:
    split_path = BUNDLE_ROOT / "data/external/exp027_train_v4/split_manifest.json"
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    expected_samples = {
        name
        for fold in splits
        for key in ("gradient_update", "internal_validation", "outer_evaluation")
        for name in fold[key]
    }
    if len(expected_samples) != 199:
        raise RuntimeError("Parent split does not list 199 GEFF samples")
    target_root = BUNDLE_ROOT / "data/raw/biohub-cell-tracking-during-development/train"
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("GEFF_MANIFEST.json"))
        rows = manifest.get("files")
        if (
            manifest.get("competition") != "biohub-cell-tracking-during-development"
            or manifest.get("samples") != 199
            or not isinstance(rows, list)
            or len(rows) != 4179
        ):
            raise RuntimeError("Kaggle GEFF archive has an unexpected manifest")
        declared = {row["path"]: row for row in rows}
        if len(declared) != 4179 or set(archive.namelist()) != set(declared) | {
            "GEFF_MANIFEST.json"
        }:
            raise RuntimeError("Kaggle GEFF archive member list differs from manifest")
        observed_samples = set()
        for row in rows:
            relative = safe_relative(str(row["path"]))
            if (
                len(relative.parts) < 3
                or relative.parts[0] != "train"
                or not relative.parts[1].endswith(".geff")
            ):
                raise RuntimeError(f"Unexpected GEFF archive path: {relative}")
            observed_samples.add(relative.parts[1].removesuffix(".geff"))
            target = target_root / Path(*relative.parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(row["path"]) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            if target.stat().st_size != row["bytes"] or sha256(target) != row["sha256"]:
                raise RuntimeError(f"Kaggle GEFF checksum mismatch: {relative}")
        if observed_samples != expected_samples:
            raise RuntimeError("Kaggle GEFF samples differ from parent split")
    archive_path.unlink()
    print(json.dumps({"event": "geff_ready", "samples": 199, "files": 4179}), flush=True)


def download_inputs(stage: str) -> None:
    sources = [
        (
            "exp032_geff_manifest.json",
            "exp032_geff_export_v1",
            INPUT_ROOT / "geff_archive",
        ),
    ]
    if stage == "diagnostic":
        sources.append(
            (
                "exp032_prediction_manifest.json",
                "exp027_saved_predictions_v1",
                BUNDLE_ROOT / "data/external/exp027_diagnostic_v1",
            )
        )
    sources.append(("exp032_cache_manifest.json", "exp015_kernel_output_v1", INPUT_ROOT / "exp015"))
    for manifest_name, source, target in sources:
        manifest_path = Path("/") / manifest_name
        rows = load_manifest(manifest_path, source)
        target.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps({"event": "download_start", "source": source, "files": len(rows)}),
            flush=True,
        )
        started = time.monotonic()
        total = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
            for index, size in enumerate(
                pool.map(lambda row, target=target: fetch_one(row, target), rows), 1
            ):
                total += size
                if index % 1000 == 0 or index == len(rows):
                    print(
                        json.dumps(
                            {
                                "event": "download_progress",
                                "source": source,
                                "done": index,
                                "total": len(rows),
                            }
                        ),
                        flush=True,
                    )
        manifest_path.unlink()
        print(
            json.dumps(
                {
                    "event": "download_complete",
                    "source": source,
                    "bytes": total,
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    if not (INPUT_ROOT / "exp015/window_cache_summary.json").is_file():
        raise RuntimeError("Kaggle cache summary is missing")
    archive = INPUT_ROOT / "geff_archive/exp032_train_geff.zip"
    if not archive.is_file():
        raise RuntimeError("Kaggle GEFF export ZIP is missing")
    extract_geff(archive)
    if stage == "diagnostic":
        prediction_root = BUNDLE_ROOT / "data/external/exp027_diagnostic_v1/context_diagnostic"
        manifest = json.loads(
            (prediction_root / "prediction_manifest.json").read_text(encoding="utf-8")
        )
        if len(manifest) != 128 or any(
            not (prediction_root / safe_relative(row["path"])).is_file()
            or sha256(prediction_root / safe_relative(row["path"])) != row["file_sha256"]
            for row in manifest
        ):
            raise RuntimeError("Saved exp027 predictions failed local SHA validation")


def run_stage(stage: str) -> None:
    source = BUNDLE_ROOT / "experiments" / EXPERIMENT / f"{EXPERIMENT}_colab_{stage}.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    env = {**os.environ, "EXP032_CLI_MODE": "1", "PYTHONUNBUFFERED": "1"}
    log_path = RUN_ROOT / f"cli_{stage}.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, str(source)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"CLI {stage} stage exited with status {return_code}")
    marker = RUN_ROOT / ("train_complete.json" if stage == "train" else "diagnostic_complete.json")
    if not marker.is_file():
        raise RuntimeError(f"CLI {stage} finished without a completion marker")


def final_files(stage: str) -> list[Path]:
    if stage == "train":
        files = [
            path
            for path in RUN_ROOT.rglob("*")
            if path.is_file()
            and not path.name.startswith("resume_state")
            and (
                not path.name.startswith("primary_tracker_epoch_")
                or path.name == "primary_tracker_epoch_01.pth"
            )
            and not path.name.endswith(".tmp")
            and "ten_epoch_diagnostic" not in path.parts
        ]
    else:
        files = [
            path
            for path in RUN_ROOT.rglob("*")
            if path.is_file()
            and "pair_logits" not in path.parts
            and (
                "ten_epoch_diagnostic" in path.parts
                or path.name in {"diagnostic_complete.json", "metrics.json", "cli_diagnostic.log"}
            )
        ]
    return sorted(files)


def publish_final(stage: str) -> None:
    archive_path = Path("/content") / f"exp032_{stage}_final.zip"
    files = final_files(stage)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in files:
            archive.write(path, "run/" + path.relative_to(RUN_ROOT).as_posix())
    parts = []
    with archive_path.open("rb") as source:
        for index in range(10000):
            block = source.read(OUTPUT_CHUNK_BYTES)
            if not block:
                break
            part = Path("/") / f"exp032_{stage}_final_{index:03d}.part"
            part.write_bytes(block)
            parts.append({"name": part.name, "bytes": len(block), "sha256": sha256(part)})
    receipt = {
        "experiment": EXPERIMENT,
        "stage": stage,
        "run_id": "ten_epoch_v1",
        "archive_sha256": sha256(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "files": len(files),
        "parts": parts,
    }
    write_json(Path("/exp032_final_receipt.json"), receipt)
    print(
        json.dumps(
            {
                "event": "final_ready",
                "stage": stage,
                "parts": len(parts),
                "bytes": receipt["archive_bytes"],
            }
        ),
        flush=True,
    )
    ack = Path("/exp032_final_ack.txt")
    for _ in range(360):
        if ack.is_file() and ack.read_text(encoding="utf-8").strip() == receipt["archive_sha256"]:
            print(json.dumps({"event": "final_acknowledged", "stage": stage}), flush=True)
            return
        time.sleep(5)
    raise RuntimeError("CLI result archive was not acknowledged within 30 minutes")


def main() -> None:
    descriptor = json.loads(STAGE_PATH.read_text(encoding="utf-8"))
    if descriptor.get("experiment") != EXPERIMENT or descriptor.get("stage") not in {
        "train",
        "diagnostic",
    }:
        raise RuntimeError("Unexpected CLI stage descriptor")
    stage = str(descriptor["stage"])
    bundle_zip = Path("/content/exp032_bundle.zip")
    assemble(descriptor["bundle_parts"], bundle_zip)
    extract_bundle(bundle_zip, descriptor["bundle_sha256"])
    bundle_zip.unlink()
    if descriptor.get("seed_parts"):
        seed_zip = Path("/content/exp032_seed.zip")
        assemble(descriptor["seed_parts"], seed_zip)
        extract_seed(seed_zip, descriptor["seed_sha256"])
        seed_zip.unlink()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    download_inputs(stage)
    run_stage(stage)
    publish_final(stage)


if __name__ == "__main__":
    main()
