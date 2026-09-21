from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

import colab_cli_manifest as manifests  # noqa: E402
import colab_cli_run as host  # noqa: E402
import colab_cli_worker as worker  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_cli_bundle_extraction_verifies_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = "project.yml"
    payload = b"competition: biohub\n"
    archive_path = tmp_path / "bundle.zip"
    manifest = {
        "experiment": worker.EXPERIMENT,
        "contains_credentials": False,
        "files": [{"path": relative, "bytes": len(payload), "sha256": digest(payload)}],
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(relative, payload)
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest))
    monkeypatch.setattr(worker, "BUNDLE_ROOT", tmp_path / "extracted")
    worker.extract_bundle(archive_path, host.sha256(archive_path))
    assert (worker.BUNDLE_ROOT / relative).read_bytes() == payload
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        worker.extract_bundle(archive_path, "0" * 64)


def test_resume_seed_preserves_latest_state_and_first_epoch_model(tmp_path: Path) -> None:
    run = tmp_path / "run"
    model_dir = run / "models/three_frame_local/fold_0"
    model_dir.mkdir(parents=True)
    for epoch in (1, 2):
        files = []
        for stem, data in (
            (f"primary_tracker_epoch_{epoch:02d}.pth", f"model-{epoch}".encode()),
            (f"resume_state_epoch_{epoch:02d}.pth", f"state-{epoch}".encode()),
        ):
            path = model_dir / stem
            path.write_bytes(data)
            files.append(
                {
                    "path": path.relative_to(run).as_posix(),
                    "bytes": len(data),
                    "sha256": digest(data),
                }
            )
        receipt = {
            "experiment": host.EXPERIMENT,
            "run_id": "ten_epoch_v1",
            "config_sha256": host.sha256(host.CONFIG_PATH),
            "fold": 0,
            "epoch": epoch,
            "files": files,
        }
        (model_dir / f"epoch_{epoch:02d}_receipt.json").write_text(json.dumps(receipt))
    seed = tmp_path / "seed.zip"
    assert host.create_train_seed(run, seed)
    with zipfile.ZipFile(seed) as archive:
        assert archive.read("run/models/three_frame_local/fold_0/resume_state.pth") == b"state-2"
        assert (
            archive.read("run/models/three_frame_local/fold_0/primary_tracker_epoch_01.pth")
            == b"model-1"
        )
        assert "run/models/three_frame_local/fold_0/primary_tracker_epoch_02.pth" not in (
            archive.namelist()
        )


def test_transfer_requires_separate_explicit_approvals() -> None:
    args = Namespace(
        approve_bundle_transfer=True,
        approve_signed_manifest_transfer=False,
        approve_geff_manifest_transfer=False,
        approve_prediction_manifest_transfer=False,
        approve_result_transfer=True,
        approve_resume_transfer=False,
    )
    with pytest.raises(RuntimeError, match="signed-manifest-transfer"):
        host.validate_permissions(args, None, "train")
    args.approve_signed_manifest_transfer = True
    with pytest.raises(RuntimeError, match="geff-manifest-transfer"):
        host.validate_permissions(args, None, "train")
    args.approve_geff_manifest_transfer = True
    with pytest.raises(RuntimeError, match="prediction-manifest-transfer"):
        host.validate_permissions(args, None, "diagnostic")
    args.approve_prediction_manifest_transfer = True
    with pytest.raises(RuntimeError, match="resume-transfer"):
        host.validate_permissions(args, Path("seed.zip"), "diagnostic")


def test_cli_preflight_rejects_stale_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "project.yml"
    source.write_bytes(b"current")
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("project.yml", b"old")
        archive.writestr(
            "BUNDLE_MANIFEST.json",
            json.dumps(
                {
                    "experiment": host.EXPERIMENT,
                    "contains_credentials": False,
                    "files": [
                        {
                            "path": "project.yml",
                            "bytes": 3,
                            "sha256": digest(b"old"),
                        }
                    ],
                }
            ),
        )
    monkeypatch.setattr(host, "BUNDLE_PATH", archive_path)
    monkeypatch.setattr(host, "collect_files", lambda: {"project.yml": source})
    with pytest.raises(RuntimeError, match="stale"):
        host.verify_bundle()


def test_colab_extracts_kaggle_geff_archive_with_full_sample_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = [f"sample_{index:03d}" for index in range(199)]
    bundle_root = tmp_path / "bundle"
    split_path = bundle_root / "data/external/exp027_train_v4/split_manifest.json"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(
        json.dumps(
            [
                {
                    "gradient_update": names,
                    "internal_validation": [],
                    "outer_evaluation": [],
                }
            ]
        )
    )
    rows = [
        {
            "path": f"train/{name}.geff/chunk_{chunk:02d}",
            "bytes": 1,
            "sha256": digest(b"x"),
        }
        for name in names
        for chunk in range(21)
    ]
    archive_path = tmp_path / "geff.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for row in rows:
            archive.writestr(row["path"], b"x")
        archive.writestr(
            "GEFF_MANIFEST.json",
            json.dumps(
                {
                    "competition": "biohub-cell-tracking-during-development",
                    "samples": 199,
                    "files": rows,
                }
            ),
        )
    monkeypatch.setattr(worker, "BUNDLE_ROOT", bundle_root)
    worker.extract_geff(archive_path)
    target = (
        bundle_root
        / "data/raw/biohub-cell-tracking-during-development/train/sample_198.geff/chunk_20"
    )
    assert target.read_bytes() == b"x"
    assert not archive_path.exists()


def test_collected_resume_state_rotates_after_sha_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    model_dir = run / "models/three_frame_local/fold_0"
    model_dir.mkdir(parents=True)
    first = model_dir / "resume_state_epoch_01.pth"
    first.write_bytes(b"old")
    second = model_dir / "resume_state_epoch_02.pth"
    second.write_bytes(b"new")
    receipt = {
        "experiment": host.EXPERIMENT,
        "run_id": "ten_epoch_v1",
        "config_sha256": host.sha256(host.CONFIG_PATH),
        "fold": 0,
        "epoch": 2,
        "files": [
            {
                "path": second.relative_to(run).as_posix(),
                "bytes": 3,
                "sha256": digest(b"new"),
            }
        ],
    }
    (model_dir / "epoch_02_receipt.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(
        host,
        "download_file",
        lambda *_args, **_kwargs: pytest.fail("No remote download expected"),
    )
    assert host.collect_epoch("synthetic", run, 0, 2)
    assert second.read_bytes() == b"new"
    assert not first.exists()


def test_final_archive_keeps_only_comparison_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "models/three_frame_local/fold_0"
    model_dir.mkdir(parents=True)
    for name in (
        "primary_tracker_epoch_01.pth",
        "primary_tracker_epoch_02.pth",
        "primary_tracker_best.pth",
        "resume_state_epoch_02.pth",
    ):
        (model_dir / name).write_bytes(b"weight")
    monkeypatch.setattr(worker, "RUN_ROOT", tmp_path)
    names = {path.name for path in worker.final_files("train")}
    assert "primary_tracker_epoch_01.pth" in names
    assert "primary_tracker_best.pth" in names
    assert "primary_tracker_epoch_02.pth" not in names
    assert "resume_state_epoch_02.pth" not in names


@pytest.mark.parametrize("stage", ["train", "diagnostic"])
def test_signed_manifests_include_cache_and_geff_for_each_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    class LocalApi:
        def authenticate(self) -> None:
            pass

    monkeypatch.setattr(manifests, "KaggleApi", LocalApi)
    row = {"path": "input.bin", "url": "https://example.test/input", "size_bytes": 1}
    monkeypatch.setattr(manifests, "cache_files", lambda _api: [row])
    monkeypatch.setattr(manifests, "geff_files", lambda _api: [row])
    monkeypatch.setattr(manifests, "prediction_files", lambda _api: [row])
    cache = tmp_path / "cache.json"
    geff = tmp_path / "geff.json"
    predictions = tmp_path / "predictions.json"
    manifests.prepare(stage, cache, geff, predictions if stage == "diagnostic" else None)
    assert json.loads(cache.read_text())["source"] == "exp015_kernel_output_v1"
    assert json.loads(geff.read_text())["source"] == "exp032_geff_export_v1"
    assert predictions.is_file() == (stage == "diagnostic")


def test_diagnostic_archive_excludes_prediction_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "ten_epoch_diagnostic"
    predictions = output / "pair_logits/epoch_01/sample.npz"
    predictions.parent.mkdir(parents=True)
    predictions.write_bytes(b"large prediction")
    (output / "diagnostic_summary.json").write_text("{}")
    (tmp_path / "metrics.json").write_text("{}")
    monkeypatch.setattr(worker, "RUN_ROOT", tmp_path)
    names = {path.relative_to(tmp_path).as_posix() for path in worker.final_files("diagnostic")}
    assert "ten_epoch_diagnostic/diagnostic_summary.json" in names
    assert "metrics.json" in names
    assert not any("pair_logits" in name for name in names)


def test_diagnostic_collection_updates_metrics_without_changing_train_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    previous = {"status": "running", "train_stage": {"model_manifest_sha256": "abc"}}
    updated = {"status": "debug_completed", "train_stage": previous["train_stage"]}
    (run / "metrics.json").write_text(json.dumps(previous))
    summary = b'{"done": true}'
    marker = json.dumps({"summary_sha256": digest(summary)}).encode()
    archive_path = tmp_path / "remote.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("run/diagnostic_complete.json", marker)
        archive.writestr("run/metrics.json", json.dumps(updated))
        archive.writestr("run/ten_epoch_diagnostic/diagnostic_summary.json", summary)
    part = archive_path.read_bytes()
    receipt = {
        "experiment": host.EXPERIMENT,
        "stage": "diagnostic",
        "archive_bytes": len(part),
        "archive_sha256": digest(part),
        "parts": [{"name": "result.part", "bytes": len(part), "sha256": digest(part)}],
    }
    remote = {
        "exp032_final_receipt.json": json.dumps(receipt).encode(),
        "result.part": part,
    }

    def fake_download(_session: str, remote_path: str, target: Path, _expected=None) -> bool:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(remote[remote_path])
        return True

    monkeypatch.setattr(host, "download_file", fake_download)
    monkeypatch.setattr(host, "cli", lambda *_args, **_kwargs: None)
    assert host.collect_final("synthetic", "diagnostic", run, tmp_path / "temporary")
    assert json.loads((run / "metrics.json").read_text()) == updated
    assert (run / "diagnostic_final.zip").read_bytes() == part
