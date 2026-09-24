"""Run exp040 on a named Colab T4 session and preserve epoch checkpoints locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import yaml
from build_colab_bundle import OUTPUT as BUNDLE_PATH
from build_colab_bundle import collect_files, seed_metrics_bytes, sha256
from colab_cli_manifest import prepare as prepare_manifests

EXPERIMENT = "exp040_trackastra_association"
ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / EXPERIMENT
CONFIG_PATH = EXP / "config.yaml"
CHUNK_BYTES = 16 * 1024 * 1024


def safe_relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"unsafe artifact path: {value!r}")
    return path


def cli(*args: str, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["colab", *args], text=True, capture_output=True, check=False)
    if check and result.returncode:
        raise RuntimeError(f"colab {args[0]} failed: {(result.stderr or result.stdout)[-900:]}")
    if not quiet and result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    return result


def verify_bundle() -> None:
    expected = collect_files()
    with zipfile.ZipFile(BUNDLE_PATH) as archive:
        manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
        rows = {row["path"]: row for row in manifest["files"]}
        if (
            manifest.get("experiment") != EXPERIMENT
            or manifest.get("contains_credentials") is not False
            or set(rows) != set(expected)
        ):
            raise ValueError("Colab bundle is stale or incomplete")
        if set(archive.namelist()) != set(rows) | {"BUNDLE_MANIFEST.json"}:
            raise ValueError("Colab bundle members differ")
        for name, path in expected.items():
            payload = path.read_bytes()
            if name == "work/metrics.json":
                payload = seed_metrics_bytes(payload)
            if (
                len(payload) != rows[name]["bytes"]
                or hashlib.sha256(payload).hexdigest() != rows[name]["sha256"]
                or archive.read(name) != payload
            ):
                raise ValueError(f"Colab bundle member is stale: {name}")


def split_file(path: Path, directory: Path, prefix: str) -> list[dict]:
    result = []
    with path.open("rb") as source:
        for index in range(1000):
            chunk = source.read(CHUNK_BYTES)
            if not chunk:
                break
            part = directory / f"{prefix}_{index:03d}.part"
            part.write_bytes(chunk)
            result.append(
                {"name": part.name, "local": str(part), "bytes": len(chunk), "sha256": sha256(part)}
            )
    if not result:
        raise ValueError("empty transfer archive")
    return result


def create_seed(run_root: Path, output: Path) -> bool:
    mapped = {}
    for fold in (0, 1):
        receipts = sorted((run_root / f"models/fold_{fold}").glob("epoch_*_receipt.json"))
        if not receipts:
            continue
        path = receipts[-1]
        receipt = json.loads(path.read_text())
        if (
            receipt.get("experiment") != EXPERIMENT
            or receipt.get("fold") != fold
            or receipt.get("config_sha256") != sha256(CONFIG_PATH)
            or receipt.get("training_source_sha256") != sha256(EXP / f"{EXPERIMENT}_train.py")
        ):
            raise ValueError("resume receipt identity differs, including training source SHA")
        state_relative = safe_relative(receipt["state_path"])
        state_path = run_root / state_relative
        if (
            state_path.stat().st_size != receipt["state_bytes"]
            or sha256(state_path) != receipt["state_sha256"]
        ):
            raise ValueError("local resume state differs from receipt")
        mapped["run/" + state_relative.as_posix()] = state_path
        mapped["run/" + path.relative_to(run_root).as_posix()] = path
    if not mapped:
        return False
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, path in sorted(mapped.items()):
            archive.write(path, name)
    return True


LAST_PROXY_PROBE = 0.0


def refresh_proxy(session: str) -> None:
    launcher = Path(shutil.which("colab") or "")
    if not launcher.is_file():
        raise RuntimeError("Colab CLI launcher is unavailable")
    shebang = launcher.read_text().splitlines()[0]
    if not shebang.startswith("#!/"):
        raise RuntimeError("Colab CLI launcher has no Python interpreter")
    interpreter = shebang[2:]
    result = subprocess.run(
        [interpreter, str(EXP / "colab_cli_refresh.py"), session],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Colab proxy refresh failed: {(result.stderr or result.stdout)[-500:]}")
    print(result.stdout.strip(), flush=True)


def probe_proxy_on_download_failure(session: str) -> bool:
    global LAST_PROXY_PROBE
    now = time.monotonic()
    if now - LAST_PROXY_PROBE < 300:
        return False
    LAST_PROXY_PROBE = now
    if cli("ls", "-s", session, ".", check=False, quiet=True).returncode == 0:
        return False
    refresh_proxy(session)
    return True


def download_file(session: str, remote: str, target: Path, expected: dict | None = None) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("." + target.name + ".tmp")
    result = cli("download", "-s", session, remote, str(temporary), check=False, quiet=True)
    if result.returncode and probe_proxy_on_download_failure(session):
        result = cli("download", "-s", session, remote, str(temporary), check=False, quiet=True)
    if result.returncode:
        temporary.unlink(missing_ok=True)
        return False
    if expected and (
        temporary.stat().st_size != expected["bytes"] or sha256(temporary) != expected["sha256"]
    ):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Colab artifact SHA differs: {remote}")
    temporary.replace(target)
    return True


def collect_epochs(session: str, run_root: Path) -> None:
    for fold in (0, 1):
        base = f"models/fold_{fold}"
        for epoch in range(1, 11):
            relative = f"{base}/epoch_{epoch:02d}_receipt.json"
            receipt_path = run_root / relative
            if not receipt_path.is_file() and not download_file(
                session, f"content/{EXPERIMENT}/run/{relative}", receipt_path
            ):
                break
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt.get("experiment") != EXPERIMENT
                or receipt.get("fold") != fold
                or receipt.get("epoch") != epoch
                or receipt.get("config_sha256") != sha256(CONFIG_PATH)
            ):
                raise ValueError(f"unexpected epoch receipt: {relative}")
            state_relative = safe_relative(receipt["state_path"])
            state_path = run_root / state_relative
            if not state_path.is_file():
                newer = sorted((run_root / base).glob("resume_state_epoch_*.pth"))
                if newer and int(newer[-1].stem.removeprefix("resume_state_epoch_")) > epoch:
                    continue
                expected = {"bytes": receipt["state_bytes"], "sha256": receipt["state_sha256"]}
                if not download_file(
                    session,
                    f"content/{EXPERIMENT}/run/{state_relative.as_posix()}",
                    state_path,
                    expected,
                ):
                    raise RuntimeError(f"epoch state unavailable: {state_relative}")
                print(f"Collected fold {fold} epoch {epoch}/10", flush=True)
            elif (
                state_path.stat().st_size != receipt["state_bytes"]
                or sha256(state_path) != receipt["state_sha256"]
            ):
                raise ValueError(f"local epoch state differs: {state_path}")
            if epoch > 1:
                (run_root / base / f"resume_state_epoch_{epoch - 1:02d}.pth").unlink(
                    missing_ok=True
                )


def extract_final_archive(archive: Path, run_root: Path) -> None:
    """Extract large diagnostics without holding a ZIP member in memory."""
    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        if len(members) != len({item.filename for item in members}):
            raise ValueError("duplicate final archive member")
        for item in members:
            name = item.filename
            relative = safe_relative(name)
            if not name.startswith("run/") or item.is_dir():
                raise ValueError(f"unexpected final member: {name}")
            target = run_root / relative.relative_to("run")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name("." + target.name + ".tmp")
            digest = hashlib.sha256()
            copied = 0
            try:
                with package.open(item) as source, temporary.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                        digest.update(block)
                        copied += len(block)
                if copied != item.file_size:
                    raise ValueError(f"final archive member size differs: {name}")
                if (
                    target.is_file()
                    and sha256(target) != digest.hexdigest()
                    and target.name != "metrics.json"
                ):
                    raise ValueError(f"final archive conflicts with local artifact: {target}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def collect_final(session: str, run_root: Path, temporary_root: Path) -> bool:
    receipt_path = temporary_root / "exp040_final_receipt.json"
    if not download_file(session, "exp040_final_receipt.json", receipt_path):
        return False
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("experiment") != EXPERIMENT:
        raise ValueError("final receipt identifies another experiment")
    archive = temporary_root / "final.zip"
    with archive.open("wb") as target:
        for item in receipt["parts"]:
            part = temporary_root / safe_relative(item["name"])
            if not download_file(session, item["name"], part, item):
                raise RuntimeError(f"final archive part unavailable: {item['name']}")
            with part.open("rb") as source:
                shutil.copyfileobj(source, target)
            part.unlink()
    if (
        archive.stat().st_size != receipt["archive_bytes"]
        or sha256(archive) != receipt["archive_sha256"]
    ):
        raise ValueError("final archive differs from receipt")
    extract_final_archive(archive, run_root)
    marker = json.loads((run_root / "train_complete.json").read_text())
    if marker.get("experiment") != EXPERIMENT:
        raise ValueError("completion marker identifies another experiment")
    for name, key in (
        ("model_manifest.json", "model_manifest_sha256"),
        ("training_summary.json", "training_summary_sha256"),
        ("metrics.json", "metrics_sha256"),
    ):
        if sha256(run_root / name) != marker[key]:
            raise ValueError(f"completion marker SHA differs: {name}")
    manifest = json.loads((run_root / "model_manifest.json").read_text())
    if len(manifest["models"]) != 2:
        raise ValueError("expected two trained fold models")
    for row in manifest["models"]:
        if sha256(run_root / safe_relative(row["path"])) != row["file_sha256"]:
            raise ValueError(f"fold model SHA differs: {row['path']}")
    for fold in (0, 1):
        for epoch in range(1, 11):
            if not (run_root / f"models/fold_{fold}/epoch_{epoch:02d}_receipt.json").is_file():
                raise ValueError(f"missing fold {fold} epoch {epoch} receipt")
    final_path = run_root / "train_final.zip"
    archive.replace(final_path)
    ack = temporary_root / "exp040_final_ack.txt"
    ack.write_text(receipt["archive_sha256"] + "\n")
    cli("upload", "-s", session, str(ack), ack.name, quiet=True)
    print(f"Collected and verified final Colab result: {run_root}", flush=True)
    return True


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    colab = config["runtime"]["colab"]
    if (
        config["model"]["training"]["epochs"] != 10
        or config["model"]["training"]["windows_per_epoch"] != 4096
    ):
        raise ValueError("Colab runner requires the fixed ten-epoch, 4096-window design")
    if colab["run_id"] != "trackastra_logbce_v2":
        raise ValueError("unexpected Colab run ID")
    run_root = EXP / "artifacts/colab_runs" / colab["run_id"]
    run_root.mkdir(parents=True, exist_ok=True)
    verify_bundle()
    with tempfile.TemporaryDirectory(prefix="exp040_colab_") as directory:
        temporary_root = Path(directory)
        seed = temporary_root / "seed.zip"
        has_seed = create_seed(run_root, seed)
        cache_urls = temporary_root / "exp040_cache_urls.json"
        geff_urls = temporary_root / "exp040_geff_urls.json"
        prepare_manifests(cache_urls, geff_urls)
        bundle_parts = split_file(BUNDLE_PATH, temporary_root, "exp040_bundle")
        seed_parts = split_file(seed, temporary_root, "exp040_seed") if has_seed else []
        descriptor = {
            "experiment": EXPERIMENT,
            "bundle_sha256": sha256(BUNDLE_PATH),
            "bundle_parts": [
                {key: row[key] for key in ("name", "bytes", "sha256")} for row in bundle_parts
            ],
            "seed_sha256": sha256(seed) if has_seed else None,
            "seed_parts": [
                {key: row[key] for key in ("name", "bytes", "sha256")} for row in seed_parts
            ],
        }
        descriptor_path = temporary_root / "exp040_stage.json"
        descriptor_path.write_text(json.dumps(descriptor, sort_keys=True) + "\n")
        session = args.session or f"exp040-trackastra-{int(time.time())}"
        process = None
        try:
            cli("new", "-s", session, "--gpu", args.gpu or colab["gpu"])
            for row in (*bundle_parts, *seed_parts):
                cli("upload", "-s", session, row["local"], row["name"], quiet=True)
            for path in (cache_urls, geff_urls, descriptor_path):
                cli("upload", "-s", session, str(path), path.name, quiet=True)
            log_path = run_root / f"session_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.log"
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    [
                        "colab",
                        "exec",
                        "-s",
                        session,
                        "-f",
                        str(EXP / "colab_cli_worker.py"),
                        "--timeout",
                        "43000",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                while True:
                    collect_epochs(session, run_root)
                    if collect_final(session, run_root, temporary_root):
                        break
                    if process.poll() is not None:
                        collect_epochs(session, run_root)
                        raise RuntimeError(
                            f"Colab worker exited before final receipt; see {log_path}"
                        )
                    time.sleep(30)
                if process.wait(timeout=180):
                    raise RuntimeError(f"Colab CLI exited after final collection; see {log_path}")
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
            cli("stop", "-s", session, check=False, quiet=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session")
    parser.add_argument("--gpu", choices=("T4", "L4", "G4", "H100", "A100"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
