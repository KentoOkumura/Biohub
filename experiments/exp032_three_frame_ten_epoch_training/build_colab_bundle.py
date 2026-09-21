from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

EXPERIMENT = "exp032_three_frame_ten_epoch_training"
ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiments" / EXPERIMENT
OUTPUT = EXPERIMENT_ROOT / "artifacts/colab_bundle" / f"{EXPERIMENT}_colab_bundle.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_file(files: dict[str, Path], source: Path, archive_path: str) -> None:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    if archive_path in files:
        raise ValueError(f"Duplicate archive path: {archive_path}")
    files[archive_path] = source


def collect_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    add_file(files, ROOT / "project.yml", "project.yml")
    for name in (
        "config.yaml",
        "metrics.json",
        "README.md",
        "requirements.md",
        "SESSION_NOTES.md",
        "result.md",
        "settings.py",
        "build_colab_bundle.py",
        "colab_cli_manifest.py",
        "colab_cli_run.py",
        "colab_cli_worker.py",
        f"{EXPERIMENT}_colab_train.py",
        f"{EXPERIMENT}_colab_train.ipynb",
        f"{EXPERIMENT}_colab_diagnostic.py",
        f"{EXPERIMENT}_colab_diagnostic.ipynb",
    ):
        add_file(
            files,
            EXPERIMENT_ROOT / name,
            f"experiments/{EXPERIMENT}/{name}",
        )

    for name in ("exp032_train_geff_export.py", "kernel-metadata.json"):
        add_file(
            files,
            EXPERIMENT_ROOT / "geff_export" / name,
            f"experiments/{EXPERIMENT}/geff_export/{name}",
        )

    controls = (
        (
            ROOT / "experiments/exp016_frozen_image_encoder/artifacts/train_v3",
            "data/external/exp016_train_v3",
            ("model_manifest.json", "split_manifest.json"),
        ),
        (
            ROOT / "experiments/exp027_multi_frame_tracker/artifacts/kaggle_train_v4",
            "data/external/exp027_train_v4",
            ("model_manifest.json", "split_manifest.json", "pilot_outer_window_selection.json"),
        ),
    )
    for source_root, archive_root, metadata_names in controls:
        for name in metadata_names:
            add_file(files, source_root / name, f"{archive_root}/{name}")
        weights = sorted((source_root / "models").rglob("*.pth"))
        if not weights:
            raise RuntimeError(f"Missing control weights: {source_root}")
        for weight in weights:
            relative = weight.relative_to(source_root).as_posix()
            add_file(files, weight, f"{archive_root}/{relative}")

    diagnostic_root = (
        ROOT
        / "experiments/exp027_multi_frame_tracker/artifacts/kaggle_diagnostic_v1/context_diagnostic"
    )
    prediction_manifest = diagnostic_root / "prediction_manifest.json"
    add_file(
        files,
        prediction_manifest,
        "data/external/exp027_diagnostic_v1/context_diagnostic/prediction_manifest.json",
    )
    predictions = json.loads(prediction_manifest.read_text(encoding="utf-8"))
    if len(predictions) != 128:
        raise RuntimeError(f"Expected 128 saved control predictions, found {len(predictions)}")
    for row in predictions:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "pair_logits":
            raise RuntimeError(f"Unsafe saved prediction path: {relative}")
        path = diagnostic_root / relative
        if sha256(path) != row["file_sha256"]:
            raise RuntimeError(f"Saved prediction checksum mismatch: {path}")
    validation_root = (
        ROOT / "experiments/exp027_multi_frame_tracker/artifacts/"
        "kaggle_validation_diagnostic_v1/validation_diagnostic"
    )
    add_file(
        files,
        validation_root / "selected_thresholds.json",
        "data/external/exp027_validation_v1/validation_diagnostic/selected_thresholds.json",
    )
    support_root = EXPERIMENT_ROOT / "artifacts/cli_inputs/public_support"
    public_root = "data/external/biohub-tracking-support-pack-50ep-v1"
    for name in (
        "repo/scripts/train_unet_transformer.py",
        "repo/src/biohub_tracking/__init__.py",
        "repo/src/biohub_tracking/models/__init__.py",
        "repo/src/biohub_tracking/models/simple_node_transformer.py",
        "weights/unet_transformer/split_0/edge_predictor_best.pth",
    ):
        add_file(files, support_root / name, f"{public_root}/{name}")
    for path in sorted((support_root / "repo/src/biohub_tracking").rglob("*.py")):
        relative = path.relative_to(support_root).as_posix()
        if f"{public_root}/{relative}" not in files:
            add_file(files, path, f"{public_root}/{relative}")
    return files


def main() -> None:
    files = collect_files()
    manifest = {
        "experiment": EXPERIMENT,
        "contains_credentials": False,
        "files": [
            {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in sorted(files.items())
        ],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".zip.tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3
    ) as archive:
        for name, path in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(
                info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=3
            )
        info = zipfile.ZipInfo("BUNDLE_MANIFEST.json", date_time=(2026, 9, 20, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, manifest_bytes, compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(temporary) as archive:
        if set(archive.namelist()) != set(files) | {"BUNDLE_MANIFEST.json"}:
            raise RuntimeError("ZIP member list differs from manifest")
        for row in manifest["files"]:
            payload = archive.read(row["path"])
            if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise RuntimeError(f"ZIP verification failed: {row['path']}")
    temporary.replace(OUTPUT)
    print(
        json.dumps(
            {
                "zip": str(OUTPUT),
                "files": len(files),
                "bytes": OUTPUT.stat().st_size,
                "sha256": sha256(OUTPUT),
            }
        )
    )


if __name__ == "__main__":
    main()
