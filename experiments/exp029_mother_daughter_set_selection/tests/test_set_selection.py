from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "exp029_mother_daughter_set_selection_train.py"
SPEC = importlib.util.spec_from_file_location("exp029_set_train", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_unordered_sets_and_known_daughter_coverage() -> None:
    catalog = MODULE.enumerate_daughter_sets(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        np.array([10]),
        np.array([3, 2, 1]),
        maximum_distance_um=10.0,
        maximum_daughters_per_mother=2,
    )
    assert catalog == [[(), (0,), (1,), (1, 0)]]
    allowed, supervised, stats = MODULE.make_set_supervision(np.array([0, 0, -1]), catalog, 1)
    assert allowed[0].tolist() == [False, False, False, True]
    assert supervised.tolist() == [True]
    assert stats["known_two_daughter_set_recovered"] == 1
    assert json.loads(json.dumps(stats))["unknown_candidate_daughters"] == 0


def test_partial_annotation_does_not_make_unknown_daughter_negative() -> None:
    catalog = [[(), (0,), (1,), (0, 1)]]
    allowed, supervised, _ = MODULE.make_set_supervision(np.array([0, -1]), catalog, 1)
    assert supervised.tolist() == [True]
    assert allowed[0].tolist() == [False, True, False, True]


def test_known_other_parent_is_negative_and_missing_positive_is_skipped() -> None:
    catalog = [[(), (0,), (1,), (0, 1)], [(), (0,), (1,), (0, 1)]]
    allowed, supervised, _ = MODULE.make_set_supervision(np.array([0, 1]), catalog, 2)
    assert allowed[0].tolist() == [False, True, False, False]
    assert allowed[1].tolist() == [False, False, True, False]
    assert supervised.all()
    missing_catalog = [[(), (1,)]]
    _, supervised, stats = MODULE.make_set_supervision(np.array([0, -1]), missing_catalog, 1)
    assert not supervised.any()
    assert stats["parent_with_missing_known_daughter"] == 1


def test_global_set_decoding_prevents_duplicate_daughters() -> None:
    catalog = [[(), (0,), (1,), (0, 1)], [(), (0,), (1,), (0, 1)]]
    logits = np.array([[0.0, 10.0, 0.0, 0.0], [0.0, 9.0, 8.0, 0.0]])
    chosen, stats = MODULE.decode_daughter_sets(
        logits,
        catalog,
        np.array([10, 20]),
        np.array([30, 40]),
        edge_cost=0.0,
        maximum_decode_sets_per_mother=4,
        time_limit_seconds=10.0,
        relative_gap=0.0,
    )
    assert chosen.tolist() == [0, 1]
    assert stats["candidate_sets"] == 8


def test_partial_set_loss_marginalizes_allowed_sets() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]], requires_grad=True)
    allowed = torch.tensor([[[False, True, False, True]]])
    supervised = torch.tensor([[True]])
    loss = MODULE.partial_set_loss(logits, allowed, supervised)
    expected = torch.logsumexp(logits[0, 0], 0) - torch.logsumexp(logits[0, 0, [1, 3]], 0)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None


def test_empty_and_tied_catalogs_select_no_edges() -> None:
    empty = MODULE.enumerate_daughter_sets(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[50.0, 0.0, 0.0]]),
        np.array([10]),
        np.array([20]),
        maximum_distance_um=14.0,
        maximum_daughters_per_mother=8,
    )
    assert empty == [[()]]
    chosen, _ = MODULE.decode_daughter_sets(
        np.array([[0.0]]),
        empty,
        np.array([10]),
        np.array([20]),
        edge_cost=0.0,
        maximum_decode_sets_per_mother=16,
        time_limit_seconds=10.0,
        relative_gap=0.0,
    )
    assert chosen.tolist() == [-1]
    chosen, _ = MODULE.decode_daughter_sets(
        np.array([[0.0, 0.0]]),
        [[(), (0,)]],
        np.array([10]),
        np.array([20]),
        edge_cost=0.0,
        maximum_decode_sets_per_mother=16,
        time_limit_seconds=10.0,
        relative_gap=0.0,
    )
    assert chosen.tolist() == [-1]


def test_full_audit_report_is_json_serializable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_example(*_args: object) -> dict[str, object]:
        return {
            "daughter_sets": [[(), (0,)]],
            "teacher_stats": {"known_parent": np.int64(1)},
            "set_teacher_stats": {
                "candidate_sets": np.int64(2),
                "supervised_positive_parents": np.int64(1),
                "known_two_daughter_set_recovered": np.int64(1),
            },
        }

    monkeypatch.setattr(MODULE, "make_example", fake_example)
    paths = [Path("6bba_sample/000000_000001.npz"), Path("44b6_sample/000000_000001.npz")]
    report = MODULE.audit_candidate_sets(
        paths,
        {"6bba_sample": {}, "44b6_sample": {}},
        "checkpoint",
        5.0,
        {"maximum_distance_um": 14.0, "maximum_daughters_per_mother": 8},
    )
    assert json.loads(json.dumps(report))["by_embryo"]["6bba"]["known_parent"] == 1


def test_runtime_projection_only_counts_remaining_folds() -> None:
    kwargs = {
        "benchmark_seconds": 10.0,
        "benchmark_windows": 5,
        "gradient_windows": 10,
        "epochs": 2,
        "decode_benchmark_seconds": 8.0,
        "decode_benchmark_windows": 4,
        "internal_windows": 3,
        "outer_windows": 5,
        "edge_cost_count": 2,
        "multiplier": 1.5,
    }
    remaining_one = MODULE.project_remaining_runtime(**kwargs, remaining_folds=1)
    remaining_two = MODULE.project_remaining_runtime(**kwargs, remaining_folds=2)
    assert remaining_one == {
        "train_seconds": 40.0,
        "decode_seconds": 34.0,
        "total_seconds": 111.0,
    }
    assert remaining_two["total_seconds"] == 222.0


def test_resume_fold_checks_saved_hashes(tmp_path: Path) -> None:
    import hashlib

    resume_dir = tmp_path / "resume"
    model_dir = tmp_path / "models"
    resume_dir.mkdir()
    model_dir.mkdir()
    model_path = resume_dir / "fold_0_best.pth"
    model_path.write_bytes(b"saved model")
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    summary = {
        "fold": 0,
        "evaluation_embryo": "6bba",
        "model_file": "fold_0_best.pth",
        "model_sha256": model_sha,
    }
    summary_path = resume_dir / "fold_0_summary.json"
    summary_path.write_text(json.dumps(summary))
    summary_sha = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    resume = {
        "fold": 0,
        "fold_summary_file": "resume/fold_0_summary.json",
        "fold_summary_sha256": summary_sha,
        "model_file": "resume/fold_0_best.pth",
        "model_sha256": model_sha,
    }
    loaded = MODULE.load_resumed_fold(
        tmp_path, model_dir, resume, [{"fold": 0, "evaluation_embryo": "6bba"}]
    )
    assert loaded == summary
    assert (model_dir / "fold_0_best.pth").read_bytes() == b"saved model"
    model_path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="resume_sha256_mismatch"):
        MODULE.load_resumed_fold(
            tmp_path, model_dir, resume, [{"fold": 0, "evaluation_embryo": "6bba"}]
        )


def test_restore_resume_files_from_private_dataset(tmp_path: Path) -> None:
    import hashlib

    input_root = tmp_path / "input"
    dataset = input_root / "datasets/kentookumura/exp029-resume-test"
    dataset.mkdir(parents=True)
    work = tmp_path / "working"
    work.mkdir()
    resume = {"dataset_source": "kentookumura/exp029-resume-test"}
    for key, hash_key, name in (
        ("audit_file", "audit_sha256", "candidate_teacher_audit.json"),
        ("fold_summary_file", "fold_summary_sha256", "fold_0_summary.json"),
        ("model_file", "model_sha256", "fold_0_best.pth"),
    ):
        payload = name.encode()
        (dataset / name).write_bytes(payload)
        resume[key] = "resume/" + name
        resume[hash_key] = hashlib.sha256(payload).hexdigest()
    MODULE.restore_resume_files_from_dataset(work, resume, input_root)
    for name in ("candidate_teacher_audit.json", "fold_0_summary.json", "fold_0_best.pth"):
        assert (work / "resume" / name).read_bytes() == name.encode()
    (dataset / "fold_0_best.pth").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="resume_dataset_file_mismatch"):
        MODULE.restore_resume_files_from_dataset(work, resume, input_root)
