from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

from frozen_tracker import (  # noqa: E402
    build_legacy_edge_target,
    greedy_match_candidates,
    legacy_active_pair_mask,
    partial_edge_pair_mask,
)
from graph_inference import (  # noqa: E402
    patch_exp015_source,
    resolve_verified_train_output,
    sha256_file,
)


def test_only_positive_child_columns_are_direct_loss_terms() -> None:
    source_matches = np.array([1, 4, -1])
    target_matches = np.array([2, -1, 5])
    target = build_legacy_edge_target(
        source_matches,
        target_matches,
        frozenset({(1, 2), (1, 3), (4, 5)}),
    )
    assert target.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    assert legacy_active_pair_mask(target).tolist() == [
        [True, True, True],
        [True, True, True],
        [True, False, True],
    ]
    assert partial_edge_pair_mask(target).tolist() == [
        [True, False, True],
        [True, False, True],
        [True, False, True],
    ]
    assert not partial_edge_pair_mask(np.zeros((3, 2), dtype=np.float32)).any()


def test_duplicate_candidate_does_not_create_a_second_positive_parent() -> None:
    candidates = np.array([[0.2, 0, 0], [0.1, 0, 0], [10, 0, 0]], dtype=float)
    ids = np.array([1, 4])
    gt = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
    diagnostics: dict[str, int] = {}
    matches, _ = greedy_match_candidates(candidates, ids, gt, 5.0, diagnostics=diagnostics)
    assert matches.tolist() == [-1, 1, 4]
    assert diagnostics["duplicate_near_gt_candidate_count"] == 1
    target = build_legacy_edge_target(
        matches,
        np.array([2, 5]),
        frozenset({(1, 2), (4, 5)}),
    )
    assert target.tolist() == [[0, 0], [1, 0], [0, 1]]


def test_softmax_denominator_retains_unknown_parent_gradient() -> None:
    torch = pytest.importorskip("torch")
    from frozen_tracker import partial_edge_focal_bce

    logits = torch.tensor(
        [[0.3, -0.4], [0.2, 0.7], [-0.1, 1.1]],
        requires_grad=True,
    )
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    loss = partial_edge_focal_bce(logits, target)
    loss.backward()
    gradient = logits.grad.detach().numpy()
    assert np.isfinite(gradient).all()
    assert gradient[1, 0] != 0.0  # unknown parent in the observed child's denominator
    assert np.array_equal(gradient[:, 1], np.zeros(3))  # unobserved child column
    empty = torch.ones((3, 2), requires_grad=True)
    zero = partial_edge_focal_bce(empty, torch.zeros_like(empty))
    zero.backward()
    assert np.array_equal(empty.grad.detach().numpy(), np.zeros((3, 2)))


def test_config_fixes_only_the_loss_mask_and_reuses_exp016() -> None:
    config = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    assert config["lineage"]["parent"] == "exp016_frozen_image_encoder"
    assert config["lineage"]["hypothesis_id"] == "HYP-20260910-01"
    assert config["lineage"]["backlog_candidate"] == "partial_edge_mask"
    assert config["model"]["loss"]["mask"] == "target_column_with_positive_annotated_edge"
    assert config["model"]["training"]["active_variants"] == ["partial_edge_mask"]
    assert config["model"]["control"]["source"] == "exp016_frozen_image_encoder"
    assert config["model"]["control"]["retrain"] is False
    assert config["validation"]["n_folds"] == 2
    assert config["model"]["trainable_components"] == ["primary_SimpleNodeTransformer"]
    assert config["experiment"]["notebooks"] == ["train", "inference"]


def test_inference_requires_exp019_manifest_and_verified_fold_files(tmp_path: Path) -> None:
    from graph_inference import (
        EXPECTED_EXP015_CACHE_IDENTITY_SHA256,
        EXPECTED_EXP015_CACHE_SUMMARY_SHA256,
        EXPECTED_PUBLIC_PRIMARY_CHECKPOINT_SHA256,
    )

    root = tmp_path / "notebooks" / "train"
    root.mkdir(parents=True)
    models = []
    for fold, embryo in ((0, "6bba"), (1, "44b6")):
        path = root / f"models/fold_{fold}/primary_tracker_best.pth"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"fold-{fold}".encode())
        models.append(
            {
                "fold": fold,
                "evaluation_embryo": embryo,
                "path": path.relative_to(root).as_posix(),
                "file_sha256": sha256_file(path),
                "canonical_state_sha256": "placeholder-for-this-path-test",
            }
        )
    manifest = {
        "experiment": "exp019_partial_edge_mask",
        "loss_mask": "target_column_with_positive_annotated_edge",
        "cache_summary_sha256": EXPECTED_EXP015_CACHE_SUMMARY_SHA256,
        "cache_identity_sha256": EXPECTED_EXP015_CACHE_IDENTITY_SHA256,
        "public_checkpoint_file_sha256": EXPECTED_PUBLIC_PRIMARY_CHECKPOINT_SHA256,
        "models": models,
    }
    path = root / "model_manifest.json"
    path.write_text(json.dumps(manifest))
    assert resolve_verified_train_output(tmp_path) == root
    manifest["loss_mask"] = "pair_touches_source_or_target_with_positive_annotated_edge"
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="expected one verified"):
        resolve_verified_train_output(tmp_path)


def test_fixed_graph_replay_patch_keeps_both_injections() -> None:
    source = (
        EXP.parent / "exp015_oracle_stage_limits" / "exp015_oracle_stage_limits_inference.py"
    ).read_text(encoding="utf-8")
    patched = patch_exp015_source(source)
    assert patched.count("EXP019_FOLD_TRACKER_INJECTION_START") == 1
    assert patched.count("EXP019_CACHE_REPLAY_START") == 1
