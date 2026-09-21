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
from typing import Any

import yaml
from build_colab_bundle import collect_files
from colab_cli_manifest import prepare as prepare_manifests

EXPERIMENT = "exp032_three_frame_ten_epoch_training"
ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiments" / EXPERIMENT
CONFIG_PATH = EXPERIMENT_ROOT / "config.yaml"
BUNDLE_PATH = EXPERIMENT_ROOT / "artifacts/colab_bundle" / f"{EXPERIMENT}_colab_bundle.zip"
CHUNK_BYTES = 16 * 1024 * 1024
POLL_SECONDS = 25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle() -> None:
    expected = collect_files()
    with zipfile.ZipFile(BUNDLE_PATH) as archive:
        manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
        declared = {row["path"]: row for row in manifest["files"]}
        if (
            manifest.get("experiment") != EXPERIMENT
            or manifest.get("contains_credentials") is not False
            or set(declared) != set(expected)
            or set(archive.namelist()) != set(expected) | {"BUNDLE_MANIFEST.json"}
        ):
            raise RuntimeError("CLI ZIP is stale or incomplete; rebuild it")
        for name, path in expected.items():
            row = declared[name]
            if row["bytes"] != path.stat().st_size or row["sha256"] != sha256(path):
                raise RuntimeError(f"CLI ZIP member is stale: {name}")


def safe_relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise RuntimeError(f"Unsafe artifact path: {value!r}")
    return path


def cli(
    *arguments: str, check: bool = True, quiet: bool = False
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["colab", *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"colab {arguments[0]} failed: {(result.stderr or result.stdout)[-600:]}"
        )
    if not quiet and result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    return result


def split_archive(path: Path, temporary_root: Path, prefix: str) -> list[dict[str, Any]]:
    parts = []
    with path.open("rb") as source:
        for index in range(10000):
            block = source.read(CHUNK_BYTES)
            if not block:
                break
            output = temporary_root / f"{prefix}_{index:03d}.part"
            output.write_bytes(block)
            parts.append(
                {
                    "name": output.name,
                    "local": str(output),
                    "bytes": len(block),
                    "sha256": sha256(output),
                }
            )
    if not parts:
        raise RuntimeError(f"Empty input archive: {path}")
    return parts


def create_train_seed(run_root: Path, output: Path) -> bool:
    receipts = sorted(run_root.glob("models/three_frame_local/fold_*/epoch_*_receipt.json"))
    if not receipts:
        return False
    mapped: dict[str, Path] = {}
    latest: dict[int, tuple[int, dict[str, Any]]] = {}
    for receipt_path in receipts:
        row = json.loads(receipt_path.read_text(encoding="utf-8"))
        fold, epoch = int(row["fold"]), int(row["epoch"])
        if (
            row["experiment"] != EXPERIMENT
            or row["run_id"] != "ten_epoch_v1"
            or row["config_sha256"] != sha256(CONFIG_PATH)
        ):
            raise RuntimeError(f"Unrelated or stale resume receipt: {receipt_path}")
        if fold not in latest or epoch > latest[fold][0]:
            latest[fold] = (epoch, row)
        if epoch == 1:
            model_item = next(
                item
                for item in row["files"]
                if Path(item["path"]).name == "primary_tracker_epoch_01.pth"
            )
            relative = safe_relative(model_item["path"])
            path = run_root / relative
            if (
                not path.is_file()
                or path.stat().st_size != model_item["bytes"]
                or sha256(path) != model_item["sha256"]
            ):
                raise RuntimeError(f"Corrupt first-epoch checkpoint: {path}")
            mapped["run/" + relative.as_posix()] = path
        mapped["run/" + receipt_path.relative_to(run_root).as_posix()] = receipt_path
    for fold, (_, row) in latest.items():
        state_item = next(
            item
            for item in row["files"]
            if Path(item["path"]).name.startswith("resume_state_epoch_")
        )
        relative = safe_relative(state_item["path"])
        path = run_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != state_item["bytes"]
            or sha256(path) != state_item["sha256"]
        ):
            raise RuntimeError(f"Corrupt latest resume checkpoint: {path}")
        mapped[f"run/models/three_frame_local/fold_{fold}/resume_state.pth"] = path
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, path in sorted(mapped.items()):
            archive.write(path, name)
    return True


def prepare_seed(stage: str, run_root: Path, temporary_root: Path) -> Path | None:
    if stage == "diagnostic":
        archive = run_root / "train_final.zip"
        if not archive.is_file() or not (run_root / "train_complete.json").is_file():
            raise RuntimeError("Complete and collect CLI training before diagnostic")
        return archive
    seed = temporary_root / "exp032_train_resume.zip"
    return seed if create_train_seed(run_root, seed) else None


def upload_parts(session: str, parts: list[dict[str, Any]]) -> None:
    for index, row in enumerate(parts, 1):
        cli("upload", "-s", session, row["local"], row["name"], quiet=True)
        if index % 4 == 0 or index == len(parts):
            print(f"Uploaded input parts: {index}/{len(parts)}", flush=True)


def download_file(session: str, remote: str, target: Path, expected: dict | None = None) -> bool:
    temporary = target.with_name("." + target.name + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    result = cli("download", "-s", session, remote, str(temporary), check=False, quiet=True)
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        return False
    if expected and (
        temporary.stat().st_size != expected["bytes"] or sha256(temporary) != expected["sha256"]
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded artifact checksum mismatch: {remote}")
    temporary.replace(target)
    return True


def collect_epoch(session: str, run_root: Path, fold: int, epoch: int) -> bool:
    base = f"models/three_frame_local/fold_{fold}"
    relative = f"{base}/epoch_{epoch:02d}_receipt.json"
    receipt_path = run_root / relative
    if not receipt_path.is_file() and not download_file(
        session, f"content/{EXPERIMENT}/run/{relative}", receipt_path
    ):
        return False
    row = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        row.get("experiment") != EXPERIMENT
        or row.get("run_id") != "ten_epoch_v1"
        or row.get("config_sha256") != sha256(CONFIG_PATH)
        or row.get("fold") != fold
        or row.get("epoch") != epoch
    ):
        raise RuntimeError(f"Unexpected epoch receipt: {relative}")
    state_item = next(
        item
        for item in row["files"]
        if Path(item["path"]).name == f"resume_state_epoch_{epoch:02d}.pth"
    )
    state_relative = safe_relative(state_item["path"])
    state_path = run_root / state_relative
    if not state_path.is_file():
        newer = sorted((run_root / base).glob("resume_state_epoch_*.pth"))
        if newer and int(newer[-1].stem.removeprefix("resume_state_epoch_")) > epoch:
            return True
        if not download_file(
            session,
            f"content/{EXPERIMENT}/run/{state_relative.as_posix()}",
            state_path,
            state_item,
        ):
            raise RuntimeError(f"Epoch resume state disappeared: {state_relative}")
    elif (
        state_path.stat().st_size != state_item["bytes"]
        or sha256(state_path) != state_item["sha256"]
    ):
        raise RuntimeError(f"Corrupt local resume state: {state_path}")
    if epoch == 1:
        model_item = next(
            item
            for item in row["files"]
            if Path(item["path"]).name == "primary_tracker_epoch_01.pth"
        )
        model_relative = safe_relative(model_item["path"])
        model_path = run_root / model_relative
        if not model_path.is_file() and not download_file(
            session,
            f"content/{EXPERIMENT}/run/{model_relative.as_posix()}",
            model_path,
            model_item,
        ):
            raise RuntimeError(f"First-epoch model disappeared: {model_relative}")
        if (
            model_path.stat().st_size != model_item["bytes"]
            or sha256(model_path) != model_item["sha256"]
        ):
            raise RuntimeError(f"Corrupt first-epoch model: {model_path}")
    if epoch > 1:
        previous = run_root / base / f"resume_state_epoch_{epoch - 1:02d}.pth"
        previous.unlink(missing_ok=True)
    print(f"Collected fold {fold} epoch {epoch}/10 resume state", flush=True)
    return True


def collect_final(session: str, stage: str, run_root: Path, temporary_root: Path) -> bool:
    receipt_path = temporary_root / "exp032_final_receipt.json"
    if not download_file(session, "exp032_final_receipt.json", receipt_path):
        return False
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("experiment") != EXPERIMENT or receipt.get("stage") != stage:
        raise RuntimeError("Final receipt identifies another run")
    if stage == "diagnostic" and receipt["archive_bytes"] > 20_000_000:
        raise RuntimeError("Diagnostic archive exceeds the 20 MB local result limit")
    archive_path = run_root / f"{stage}_final.zip"
    temporary_archive = temporary_root / f"exp032_{stage}_final.zip"
    with temporary_archive.open("wb") as destination:
        for item in receipt["parts"]:
            part_path = temporary_root / item["name"]
            if not download_file(session, item["name"], part_path, item):
                raise RuntimeError(f"Final archive part unavailable: {item['name']}")
            with part_path.open("rb") as source:
                shutil.copyfileobj(source, destination)
            part_path.unlink()
    if (
        temporary_archive.stat().st_size != receipt["archive_bytes"]
        or sha256(temporary_archive) != receipt["archive_sha256"]
    ):
        raise RuntimeError("Final archive SHA differs from receipt")
    with zipfile.ZipFile(temporary_archive) as archive:
        for name in archive.namelist():
            relative = safe_relative(name)
            if not name.startswith("run/") or name.endswith("/"):
                raise RuntimeError(f"Unexpected final archive member: {name}")
            target = run_root / relative.relative_to("run")
            content = archive.read(name)
            if target.is_file():
                if hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(content).digest():
                    continue
                if stage != "diagnostic":
                    raise RuntimeError(
                        f"Final archive conflicts with a collected artifact: {target}"
                    )
                if relative.as_posix() == "run/metrics.json":
                    previous = json.loads(target.read_text(encoding="utf-8"))
                    updated = json.loads(content)
                    if (
                        previous.get("train_stage") != updated.get("train_stage")
                        or updated.get("status") != "debug_completed"
                    ):
                        raise RuntimeError("Diagnostic metrics changed training evidence")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    marker = run_root / ("train_complete.json" if stage == "train" else "diagnostic_complete.json")
    if not marker.is_file():
        raise RuntimeError(f"Final archive has no {stage} completion marker")
    if stage == "train":
        complete = json.loads(marker.read_text(encoding="utf-8"))
        manifest = run_root / "model_manifest.json"
        if sha256(manifest) != complete["model_manifest_sha256"]:
            raise RuntimeError("Training model manifest SHA mismatch")
        for fold in (0, 1):
            for epoch in range(1, 11):
                if not (
                    run_root
                    / f"models/three_frame_local/fold_{fold}/epoch_{epoch:02d}_receipt.json"
                ).is_file():
                    raise RuntimeError(f"Missing local fold {fold} epoch {epoch} receipt")
    else:
        complete = json.loads(marker.read_text(encoding="utf-8"))
        summary = run_root / "ten_epoch_diagnostic/diagnostic_summary.json"
        if sha256(summary) != complete["summary_sha256"]:
            raise RuntimeError("Diagnostic summary SHA mismatch")
    temporary_archive.replace(archive_path)
    ack = temporary_root / "exp032_final_ack.txt"
    ack.write_text(receipt["archive_sha256"] + "\n", encoding="utf-8")
    cli("upload", "-s", session, str(ack), ack.name, quiet=True)
    print(f"Collected and verified {stage} final archive", flush=True)
    return True


def validate_permissions(args: argparse.Namespace, seed: Path | None, stage: str) -> None:
    missing = []
    if not args.approve_bundle_transfer:
        missing.append(
            "--approve-bundle-transfer (code, public support checkpoint, saved controls)"
        )
    if not args.approve_signed_manifest_transfer:
        missing.append("--approve-signed-manifest-transfer (temporary Kaggle cache URLs)")
    if not args.approve_geff_manifest_transfer:
        missing.append("--approve-geff-manifest-transfer (temporary Kaggle GEFF ZIP URL)")
    if stage == "diagnostic" and not args.approve_prediction_manifest_transfer:
        missing.append("--approve-prediction-manifest-transfer (temporary saved prediction URLs)")
    if not args.approve_result_transfer:
        missing.append(
            "--approve-result-transfer (generated model states and diagnostics to local disk)"
        )
    if seed and not args.approve_resume_transfer:
        missing.append("--approve-resume-transfer (saved checkpoints back to Colab)")
    if missing:
        raise RuntimeError("Explicit transfer approvals required: " + "; ".join(missing))


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["model"]["training"]["epochs"] != 10:
        raise RuntimeError("CLI runner requires the ten-epoch experiment")
    colab = config["runtime"]["colab"]
    if colab["run_id"] != "ten_epoch_v1":
        raise RuntimeError("CLI runner currently expects run_id ten_epoch_v1")
    run_root = EXPERIMENT_ROOT / "artifacts/colab_runs" / (colab["run_id"] + "_cli")
    run_root.mkdir(parents=True, exist_ok=True)
    if not BUNDLE_PATH.is_file():
        raise FileNotFoundError(f"Build the CLI ZIP before running: {BUNDLE_PATH}")
    verify_bundle()
    with tempfile.TemporaryDirectory(prefix="exp032_cli_") as directory:
        temporary_root = Path(directory)
        seed = prepare_seed(args.stage, run_root, temporary_root)
        validate_permissions(args, seed, args.stage)
        cache_manifest = temporary_root / "exp032_cache_manifest.json"
        geff_manifest = temporary_root / "exp032_geff_manifest.json"
        prediction_manifest = (
            temporary_root / "exp032_prediction_manifest.json"
            if args.stage == "diagnostic"
            else None
        )
        prepare_manifests(args.stage, cache_manifest, geff_manifest, prediction_manifest)
        bundle_parts = split_archive(BUNDLE_PATH, temporary_root, "exp032_bundle")
        seed_parts = split_archive(seed, temporary_root, "exp032_seed") if seed else []
        descriptor = {
            "experiment": EXPERIMENT,
            "stage": args.stage,
            "bundle_sha256": sha256(BUNDLE_PATH),
            "bundle_parts": [
                {key: row[key] for key in ("name", "bytes", "sha256")} for row in bundle_parts
            ],
            "seed_sha256": sha256(seed) if seed else None,
            "seed_parts": [
                {key: row[key] for key in ("name", "bytes", "sha256")} for row in seed_parts
            ],
        }
        descriptor_path = temporary_root / "exp032_stage.json"
        descriptor_path.write_text(json.dumps(descriptor, sort_keys=True) + "\n", encoding="utf-8")
        session = args.session or f"{colab['cli']['session_name']}-{int(time.time())}"
        exec_process: subprocess.Popen[str] | None = None
        try:
            cli("new", "-s", session, "--gpu", args.gpu or colab["cli"]["gpu"])
            upload_parts(session, bundle_parts)
            upload_parts(session, seed_parts)
            for manifest in (cache_manifest, geff_manifest, prediction_manifest):
                if manifest is not None:
                    cli("upload", "-s", session, str(manifest), manifest.name, quiet=True)
            cli("upload", "-s", session, str(descriptor_path), descriptor_path.name, quiet=True)
            # The manifests contain temporary signed URLs. They remain only in this temp dir.
            log_path = (
                run_root / f"cli_{args.stage}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.log"
            )
            with log_path.open("w", encoding="utf-8") as log:
                exec_process = subprocess.Popen(
                    [
                        "colab",
                        "exec",
                        "-s",
                        session,
                        "-f",
                        str(EXPERIMENT_ROOT / "colab_cli_worker.py"),
                        "--timeout",
                        "43000",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                final_collected = False
                collected_epochs: set[tuple[int, int]] = set()
                while not final_collected:
                    if args.stage == "train":
                        for fold in (0, 1):
                            for epoch in range(1, 11):
                                if (fold, epoch) in collected_epochs:
                                    continue
                                if collect_epoch(session, run_root, fold, epoch):
                                    collected_epochs.add((fold, epoch))
                                else:
                                    break
                            if len(collected_epochs) < (fold + 1) * 10:
                                break
                    final_collected = collect_final(session, args.stage, run_root, temporary_root)
                    if final_collected:
                        break
                    if exec_process.poll() is not None:
                        raise RuntimeError(
                            f"Colab worker exited before final receipt; see {log_path}"
                        )
                    time.sleep(POLL_SECONDS)
                try:
                    return_code = exec_process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        f"Colab worker did not acknowledge final archive; see {log_path}"
                    ) from None
                if return_code != 0:
                    raise RuntimeError(
                        f"Colab CLI exec failed after final collection; see {log_path}"
                    )
            print(f"CLI {args.stage} complete: {run_root}", flush=True)
        finally:
            if exec_process is not None and exec_process.poll() is None:
                exec_process.terminate()
            cli("stop", "-s", session, check=False, quiet=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "diagnostic"), required=True)
    parser.add_argument("--session")
    parser.add_argument("--gpu", choices=("T4", "L4", "G4", "H100", "A100"))
    parser.add_argument("--approve-bundle-transfer", action="store_true")
    parser.add_argument("--approve-signed-manifest-transfer", action="store_true")
    parser.add_argument("--approve-geff-manifest-transfer", action="store_true")
    parser.add_argument("--approve-prediction-manifest-transfer", action="store_true")
    parser.add_argument("--approve-result-transfer", action="store_true")
    parser.add_argument("--approve-resume-transfer", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
