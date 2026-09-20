from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

EXP = Path(__file__).resolve().parents[1]
PARENT = EXP.parent / "exp016_frozen_image_encoder"
sys.path.insert(0, str(EXP))
import graph_inference as gi  # noqa: E402


def test_six_epoch_config_changes_only_training_length() -> None:
    parent = yaml.safe_load((PARENT / "config.yaml").read_text(encoding="utf-8"))
    current = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    assert current["lineage"]["parent"] == "exp016_frozen_image_encoder"
    assert current["lineage"]["hypothesis_id"] == "HYP-20260920-01"
    assert current["lineage"]["backlog_candidate"] == "tracker_six_epochs"
    assert current["data"] == parent["data"]
    assert current["validation"] == parent["validation"]
    for key in (
        "public_source",
        "frozen_components",
        "trainable_components",
        "params",
        "teacher",
        "loss",
    ):
        assert current["model"][key] == parent["model"][key]
    parent_training = parent["model"]["training"]
    current_training = current["model"]["training"]
    assert current_training["epochs"] == 6
    assert parent_training["epochs"] == 3
    assert current_training["late_checkpoint_min_epoch_index"] == 3
    for key, value in parent_training.items():
        if key != "epochs":
            assert current_training[key] == value
    assert current["model"]["control"]["retrain"] is False
    assert current["runtime"]["kaggle"]["inference"]["kernel_sources"][-1] == (
        "kentookumura/exp024-tracker-six-epochs-train"
    )


def make_train_output(root: Path, best_epochs: tuple[int, int]) -> Path:
    output = root / "train-output"
    output.mkdir()
    models = []
    for fold, best_epoch in enumerate(best_epochs):
        model = output / f"fold_{fold}.pth"
        model.write_bytes(f"fold {fold}".encode())
        models.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "path": model.name,
                "file_sha256": gi.sha256_file(model),
            }
        )
    manifest = {
        "experiment": "exp024_tracker_six_epochs",
        "loss_mask": "pair_touches_source_or_target_with_positive_annotated_edge",
        "training_epochs": 6,
        "late_checkpoint_min_epoch_index": 3,
        "cache_summary_sha256": gi.EXPECTED_EXP015_CACHE_SUMMARY_SHA256,
        "cache_identity_sha256": gi.EXPECTED_EXP015_CACHE_IDENTITY_SHA256,
        "public_checkpoint_file_sha256": gi.EXPECTED_PUBLIC_PRIMARY_CHECKPOINT_SHA256,
        "models": models,
    }
    (output / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output / "training_summary.json").write_text(
        json.dumps(
            {
                "continuation_gate": {
                    "both_folds_selected_after_three_epochs": all(
                        epoch >= 3 for epoch in best_epochs
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    return output


def test_graph_inference_requires_late_checkpoints_in_both_folds(tmp_path: Path) -> None:
    output = make_train_output(tmp_path, (3, 4))
    assert gi.resolve_verified_train_output(tmp_path) == output
    manifest_path = output / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["models"][1]["best_epoch"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected one verified exp024 train output"):
        gi.resolve_verified_train_output(tmp_path)


def test_graph_inference_rejects_tampered_checkpoint(tmp_path: Path) -> None:
    output = make_train_output(tmp_path, (3, 5))
    (output / "fold_0.pth").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="expected one verified exp024 train output"):
        gi.resolve_verified_train_output(tmp_path)


def test_fixed_pipeline_patch_includes_replay_and_injection() -> None:
    source = (
        EXP.parent / "exp015_oracle_stage_limits" / "exp015_oracle_stage_limits_inference.py"
    ).read_text(encoding="utf-8")
    patched = gi.patch_exp015_source(source)
    assert patched.count("EXP024_FOLD_TRACKER_INJECTION_START") == 1
    assert patched.count("EXP024_CACHE_REPLAY_START") == 1
