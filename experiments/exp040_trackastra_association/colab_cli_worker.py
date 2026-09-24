"""Colab-side exp040 CLI worker; no Kaggle or Google credentials are uploaded."""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

EXPERIMENT = "exp040_trackastra_association"
BASE = Path("/content") / EXPERIMENT
WORK = BASE / "run"
INPUTS = BASE / "inputs"
BUNDLE = BASE / "bundle"
CHUNK_BYTES = 16 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True) + "\n")
    temporary.replace(path)


def assemble(parts: list[dict], output: Path, expected_sha: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as target:
        for row in parts:
            part = Path("/") / safe_relative(row["name"])
            if part.stat().st_size != row["bytes"] or sha256(part) != row["sha256"]:
                raise ValueError(f"transfer part differs: {part.name}")
            with part.open("rb") as source:
                shutil.copyfileobj(source, target)
    if sha256(output) != expected_sha:
        raise ValueError("assembled archive SHA differs")


def extract_bundle(archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
        rows = manifest["files"]
        if (
            manifest.get("experiment") != EXPERIMENT
            or manifest.get("contains_credentials") is not False
        ):
            raise ValueError("unexpected bundle manifest")
        if set(archive.namelist()) != {row["path"] for row in rows} | {"BUNDLE_MANIFEST.json"}:
            raise ValueError("bundle member list differs")
        for row in rows:
            path = BUNDLE / safe_relative(row["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(row["path"]) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target)
            if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
                raise ValueError(f"bundle member differs: {row['path']}")
    shutil.copytree(BUNDLE / "work", WORK, dirs_exist_ok=True)


def extract_seed(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            relative = safe_relative(name)
            if not name.startswith("run/models/") or name.endswith("/"):
                raise ValueError(f"unexpected resume member: {name}")
            target = WORK / relative.relative_to("run")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def load_urls(path: Path, source: str) -> list[dict]:
    value = json.loads(path.read_text())
    if value.get("experiment") != EXPERIMENT or value.get("source") != source:
        raise ValueError("signed URL manifest identity differs")
    rows = value["files"]
    if not rows or len(rows) != len({row["path"] for row in rows}):
        raise ValueError("signed URL manifest missing files or duplicated names")
    for row in rows:
        safe_relative(row["path"])
        parsed = urlparse(row["url"])
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("signed URL must use HTTPS")
    return rows


def fetch_one(row: dict, root: Path) -> int:
    target = root / safe_relative(row["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("." + target.name + ".tmp")
    for attempt in range(5):
        try:
            request = urllib.request.Request(row["url"], headers={"User-Agent": "exp040-colab-cli"})
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as output,
            ):
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if row.get("size_bytes") is not None and temporary.stat().st_size != row["size_bytes"]:
                raise ValueError("download byte count differs")
            if row.get("sha256") is not None and sha256(temporary) != row["sha256"]:
                raise ValueError("download SHA differs")
            temporary.replace(target)
            return target.stat().st_size
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            if attempt == 4:
                code = getattr(exc, "code", None)
                raise RuntimeError(
                    f"Kaggle download failed for {row['path']}: {type(exc).__name__}, status={code}"
                ) from None
            time.sleep(min(30, 2 ** (attempt + 1)))
    raise AssertionError("unreachable")


def fetch_inputs() -> None:
    sources = (
        ("exp040_geff_urls.json", "exp032_geff_export_v1", INPUTS / "geff_archive"),
        ("exp040_cache_urls.json", "exp015_kernel_output_v1", INPUTS / "exp015"),
    )
    for filename, source, target in sources:
        manifest_path = Path("/") / filename
        rows = load_urls(manifest_path, source)
        started = time.monotonic()
        total = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
            for index, size in enumerate(
                pool.map(lambda row, destination=target: fetch_one(row, destination), rows), 1
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
    archive_path = INPUTS / "geff_archive/exp032_train_geff.zip"
    train_root = INPUTS / "geff/train"
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("GEFF_MANIFEST.json"))
        rows = manifest["files"]
        if (
            manifest.get("competition") != "biohub-cell-tracking-during-development"
            or manifest.get("samples") != 199
            or len(rows) != 4179
        ):
            raise ValueError("GEFF archive manifest differs")
        if set(archive.namelist()) != {row["path"] for row in rows} | {"GEFF_MANIFEST.json"}:
            raise ValueError("GEFF archive members differ")
        for row in rows:
            relative = safe_relative(row["path"])
            if (
                len(relative.parts) < 3
                or relative.parts[0] != "train"
                or not relative.parts[1].endswith(".geff")
            ):
                raise ValueError(f"unexpected GEFF member: {relative}")
            target = train_root / Path(*relative.parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(row["path"]) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            if target.stat().st_size != row["bytes"] or sha256(target) != row["sha256"]:
                raise ValueError(f"GEFF member differs: {relative}")
    if len(list(train_root.glob("*.geff"))) != 199:
        raise ValueError("GEFF sample count differs")
    archive_path.unlink()
    print(json.dumps({"event": "geff_ready", "samples": 199}), flush=True)


def run_training() -> None:
    packages = (
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
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            *packages,
        ],
        check=True,
    )
    importlib.invalidate_caches()
    source = WORK / f"{EXPERIMENT}_train.py"
    env = {
        **os.environ,
        "EXP040_COLAB_MODE": "1",
        "EXP040_WORK": str(WORK),
        "EXP040_CACHE_OUTPUT": str(INPUTS / "exp015"),
        "EXP040_CONTROL_OUTPUT": str(BUNDLE / "inputs/control"),
        "EXP040_PUBLIC_ROOT": str(BUNDLE / "inputs/public"),
        "EXP040_TRAIN_DIR": str(INPUTS / "geff/train"),
        "PYTHONUNBUFFERED": "1",
    }
    with (WORK / "colab_train.log").open("a", encoding="utf-8") as log:
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
        code = process.wait()
    if code:
        raise RuntimeError(f"exp040 training exited with code {code}")
    for name in ("model_manifest.json", "training_summary.json", "metrics.json"):
        if not (WORK / name).is_file():
            raise FileNotFoundError(f"training ended without {name}")
    summary = json.loads((WORK / "training_summary.json").read_text())
    if len(summary["fold_evaluation"]) != 2:
        raise ValueError("training did not evaluate both held-out embryos")
    marker = {
        "experiment": EXPERIMENT,
        "run_id": json.loads((WORK / "metrics.json").read_text())["evidence"]["colab"]["run_id"],
        "model_manifest_sha256": sha256(WORK / "model_manifest.json"),
        "training_summary_sha256": sha256(WORK / "training_summary.json"),
        "metrics_sha256": sha256(WORK / "metrics.json"),
    }
    write_json(WORK / "train_complete.json", marker)


def publish_final() -> None:
    archive_path = BASE / "train_final.zip"
    selected = [
        path
        for path in WORK.rglob("*")
        if path.is_file()
        and not path.name.startswith("resume_state_epoch_")
        and not path.name.endswith(".tmp")
    ]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, "run/" + path.relative_to(WORK).as_posix())
    parts = []
    with archive_path.open("rb") as source:
        for index in range(1000):
            chunk = source.read(CHUNK_BYTES)
            if not chunk:
                break
            part = Path("/") / f"exp040_final_{index:03d}.part"
            part.write_bytes(chunk)
            parts.append({"name": part.name, "bytes": len(chunk), "sha256": sha256(part)})
    receipt = {
        "experiment": EXPERIMENT,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "parts": parts,
    }
    write_json(Path("/exp040_final_receipt.json"), receipt)
    print(
        json.dumps(
            {"event": "final_ready", "bytes": receipt["archive_bytes"], "parts": len(parts)}
        ),
        flush=True,
    )
    ack = Path("/exp040_final_ack.txt")
    for _ in range(360):
        if ack.is_file() and ack.read_text().strip() == receipt["archive_sha256"]:
            return
        time.sleep(5)
    raise RuntimeError("final result archive was not acknowledged")


def main() -> None:
    descriptor = json.loads(Path("/exp040_stage.json").read_text())
    if descriptor.get("experiment") != EXPERIMENT:
        raise ValueError("unexpected stage descriptor")
    BASE.mkdir(parents=True, exist_ok=True)
    bundle_zip = BASE / "bundle.zip"
    assemble(descriptor["bundle_parts"], bundle_zip, descriptor["bundle_sha256"])
    extract_bundle(bundle_zip)
    bundle_zip.unlink()
    if descriptor.get("seed_parts"):
        seed_zip = BASE / "seed.zip"
        assemble(descriptor["seed_parts"], seed_zip, descriptor["seed_sha256"])
        extract_seed(seed_zip)
        seed_zip.unlink()
    fetch_inputs()
    run_training()
    publish_final()


if __name__ == "__main__":
    main()
