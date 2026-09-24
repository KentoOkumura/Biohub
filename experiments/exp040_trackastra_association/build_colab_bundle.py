"""Package the small, SHA-checked exp040 inputs for Colab CLI."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

EXPERIMENT = "exp040_trackastra_association"
ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / EXPERIMENT
SUPPORT = (
    ROOT / "experiments/exp032_three_frame_ten_epoch_training/artifacts/cli_inputs/public_support"
)
CONTROL = ROOT / "experiments/exp016_frozen_image_encoder/artifacts/train_v3"
OUTPUT = EXP / "artifacts/colab_bundle" / f"{EXPERIMENT}_colab_bundle.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for name in (
        "config.yaml",
        "metrics.json",
        "exp040_trackastra_association_train.py",
        "vendor/LICENSE",
        "assets/runtime_budget_cap4096.json",
    ):
        files[f"work/{name}"] = EXP / name
    files["inputs/control/model_manifest.json"] = CONTROL / "model_manifest.json"
    for fold in (0, 1):
        name = f"models/fold_{fold}/primary_tracker_best.pth"
        files[f"inputs/control/{name}"] = CONTROL / name
    for path in sorted((SUPPORT / "repo/src/biohub_tracking").rglob("*.py")):
        files["inputs/public/" + path.relative_to(SUPPORT).as_posix()] = path
    if not files or any(not path.is_file() or path.is_symlink() for path in files.values()):
        raise FileNotFoundError("Colab bundle inputs are missing or symbolic links")
    return files


def seed_metrics_bytes(payload: bytes) -> bytes:
    """Package an unscored metrics seed without prior Colab results."""
    metrics = json.loads(payload)
    metrics["status"] = "planned"
    metrics["notes"] = "Colab再実行用の初期状態。学習・評価結果は未取得。"
    metrics.pop("train_stage", None)
    metrics.pop("post_run_audit", None)
    evidence = metrics["evidence"]
    evidence.pop("colab", None)
    artifacts = evidence["artifacts"]
    for key in ("model_manifest_sha", "model_count", "oof_prediction_sha"):
        artifacts[key] = None
    artifacts["model_shas"] = {}
    return (json.dumps(metrics, ensure_ascii=False, indent=2) + "\n").encode()


def main() -> None:
    files = collect_files()
    payloads = {
        name: seed_metrics_bytes(path.read_bytes())
        if name == "work/metrics.json"
        else path.read_bytes()
        for name, path in files.items()
    }
    manifest = {
        "experiment": EXPERIMENT,
        "contains_credentials": False,
        "files": [
            {
                "path": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in sorted(payloads.items())
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(payloads.items()):
            archive.writestr(name, payload)
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, sort_keys=True) + "\n")
    with zipfile.ZipFile(temporary) as archive:
        if set(archive.namelist()) != set(files) | {"BUNDLE_MANIFEST.json"}:
            raise ValueError("Colab bundle member list differs from manifest")
        for row in manifest["files"]:
            data = archive.read(row["path"])
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"Colab bundle verification failed: {row['path']}")
    temporary.replace(OUTPUT)
    print(
        json.dumps({"path": str(OUTPUT), "bytes": OUTPUT.stat().st_size, "sha256": sha256(OUTPUT)})
    )


if __name__ == "__main__":
    main()
