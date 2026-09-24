from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
import build_colab_bundle as bundle  # noqa: E402
import colab_cli_run as runner  # noqa: E402
import colab_cli_worker as worker  # noqa: E402


def test_bundle_metrics_seed_excludes_previous_training_result() -> None:
    prior = {
        "status": "debug_completed",
        "notes": "old result",
        "train_stage": {"fold_evaluation": {"0": {"known_edges": 1}}},
        "post_run_audit": {"loss_numerical_fix_implemented": True},
        "evidence": {
            "colab": {"run_id": "previous"},
            "artifacts": {
                "model_manifest_sha": "old",
                "model_count": 2,
                "model_shas": {"0": "old"},
                "oof_prediction_sha": "old",
                "cache_identity_sha": "fixed",
            },
        },
    }
    seed = json.loads(bundle.seed_metrics_bytes(json.dumps(prior).encode()))
    assert seed["status"] == "planned"
    assert "train_stage" not in seed and "post_run_audit" not in seed
    assert "colab" not in seed["evidence"]
    assert seed["evidence"]["artifacts"] == {
        "model_manifest_sha": None,
        "model_count": None,
        "model_shas": {},
        "oof_prediction_sha": None,
        "cache_identity_sha": "fixed",
    }


def test_bundle_rejects_corrupt_member_and_unsafe_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(worker, "BUNDLE", tmp_path / "unpacked")
    archive = tmp_path / "bundle.zip"
    good = b"model-code"
    manifest = {
        "experiment": worker.EXPERIMENT,
        "contains_credentials": False,
        "files": [
            {
                "path": "work/source.py",
                "bytes": len(good),
                "sha256": hashlib.sha256(good).hexdigest(),
            }
        ],
    }
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("work/source.py", b"tampered")
        package.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest))
    with pytest.raises(ValueError, match="bundle member differs"):
        worker.extract_bundle(archive)
    with pytest.raises(ValueError, match="unsafe"):
        worker.safe_relative("../credentials.json")
    with pytest.raises(ValueError, match="unsafe"):
        runner.safe_relative("/etc/passwd")


def test_resume_seed_verifies_receipt_and_state(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("fixed: true\n")
    monkeypatch.setattr(runner, "CONFIG_PATH", config)
    monkeypatch.setattr(runner, "EXP", tmp_path)
    source = tmp_path / f"{runner.EXPERIMENT}_train.py"
    source.write_text("source version 1\n")
    run = tmp_path / "run"
    folder = run / "models/fold_0"
    folder.mkdir(parents=True)
    state = folder / "resume_state_epoch_02.pth"
    state.write_bytes(b"verified-state")
    receipt = {
        "experiment": runner.EXPERIMENT,
        "fold": 0,
        "epoch": 2,
        "config_sha256": runner.sha256(config),
        "training_source_sha256": runner.sha256(source),
        "state_path": "models/fold_0/resume_state_epoch_02.pth",
        "state_bytes": state.stat().st_size,
        "state_sha256": runner.sha256(state),
    }
    (folder / "epoch_02_receipt.json").write_text(json.dumps(receipt))
    seed = tmp_path / "seed.zip"
    assert runner.create_seed(run, seed)
    with zipfile.ZipFile(seed) as package:
        assert set(package.namelist()) == {
            "run/models/fold_0/epoch_02_receipt.json",
            "run/models/fold_0/resume_state_epoch_02.pth",
        }
    source.write_text("source version 2\n")
    with pytest.raises(ValueError, match="training source SHA"):
        runner.create_seed(run, seed)
    source.write_text("source version 1\n")
    state.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="resume state differs"):
        runner.create_seed(run, seed)


def test_notebook_epoch_resume_restores_model_and_rejects_tampering(tmp_path: Path) -> None:
    import ast
    import random

    import numpy as np
    import torch

    source = (EXP / f"{runner.EXPERIMENT}_train.py").read_text()
    tree = ast.parse(source)
    names = {"resume_identity", "save_epoch_resume", "load_epoch_resume"}
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *functions,
        ],
        type_ignores=[],
    )
    work = tmp_path / "run"
    work.mkdir()
    (work / "config.yaml").write_text("fixed: true\n")
    source_path = work / f"{runner.EXPERIMENT}_train.py"
    source_path.write_text("source version 1\n")
    (work / "models").mkdir()
    namespace = {
        "COLAB_MODE": True,
        "WORK": work,
        "MODELS": work / "models",
        "CONFIG": {"runtime": {"colab": {"run_id": "test_v1"}}},
        "EXPERIMENT": runner.EXPERIMENT,
        "SCHEDULE_SHA": {0: "schedule-sha"},
        "file_sha256": runner.sha256,
        "torch": torch,
        "np": np,
        "random": random,
        "json": json,
    }
    exec(
        compile(
            ast.fix_missing_locations(module), str(EXP / f"{runner.EXPERIMENT}_train.py"), "exec"
        ),
        namespace,
    )
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.ones((1, 2))).sum()
    loss.backward()
    optimizer.step()
    best = {key: value.detach().clone() for key, value in model.state_dict().items()}
    namespace["save_epoch_resume"](0, 1, model, optimizer, best, 0.2, 1, [{"epoch": 1}])
    with torch.no_grad():
        model.weight.zero_()
    epoch, best_loss, best_epoch, restored_best, history = namespace["load_epoch_resume"](
        0, model, optimizer
    )
    assert (epoch, best_loss, best_epoch, history) == (1, 0.2, 1, [{"epoch": 1}])
    torch.testing.assert_close(model.weight, best["weight"])
    torch.testing.assert_close(restored_best["weight"], best["weight"])
    source_path.write_text("source version 2\n")
    with pytest.raises(ValueError, match="training_source_sha256"):
        namespace["load_epoch_resume"](0, model, optimizer)
    source_path.write_text("source version 1\n")
    state_path = work / "models/fold_0/resume_state_epoch_01.pth"
    state_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="resume state SHA"):
        namespace["load_epoch_resume"](0, model, optimizer)


def test_proxy_probe_refreshes_only_when_root_is_unreachable(monkeypatch) -> None:
    import subprocess

    calls = []
    monkeypatch.setattr(runner, "LAST_PROXY_PROBE", -10_000.0)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(
        runner,
        "cli",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1),
    )
    monkeypatch.setattr(runner, "refresh_proxy", calls.append)
    assert runner.probe_proxy_on_download_failure("session")
    assert calls == ["session"]
    assert not runner.probe_proxy_on_download_failure("session")


def test_final_archive_streams_large_member_and_rejects_conflict(
    tmp_path: Path, monkeypatch
) -> None:
    archive = tmp_path / "final.zip"
    payload = b"score,row\n" * 350_000
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr("run/fold_0_outer_labeled_scores.jsonl", payload)

    def reject_whole_member_read(*_args, **_kwargs):
        raise AssertionError("whole ZIP member read would exhaust memory")

    monkeypatch.setattr(zipfile.ZipFile, "read", reject_whole_member_read)
    run = tmp_path / "run"
    runner.extract_final_archive(archive, run)
    target = run / "fold_0_outer_labeled_scores.jsonl"
    assert target.read_bytes() == payload
    target.write_bytes(b"different local artifact")
    with pytest.raises(ValueError, match="conflicts"):
        runner.extract_final_archive(archive, run)
