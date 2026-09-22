from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

from train_pipeline import split_selection_samples, split_two_folds  # noqa: E402
from velocity_history import (  # noqa: E402
    align_history_frame,
    empty_history_frame,
    history_content_sha256,
    propagate_history_frame,
)


def load_config() -> dict:
    return yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))


def test_contract_uses_only_saved_exp016_and_two_velocity_models() -> None:
    config = load_config()
    assert config["experiment"]["notebooks"] == ["train"]
    assert config["lineage"]["parent"] == "exp016_frozen_image_encoder"
    assert config["lineage"]["hypothesis_id"] == "HYP-20260910-10"
    assert config["lineage"]["backlog_candidate"] == "velocity_features"
    model = config["model"]
    assert model["trainable_components"] == [
        "primary_SimpleNodeTransformer",
        "velocity_pair_branch",
    ]
    assert model["training"]["active_variants"] == ["velocity_features"]
    assert model["training"]["epochs"] == 3
    assert model["history"]["inner_generator_model_count"] == 4
    assert model["output"]["history_model_count"] == 4
    assert model["output"]["model_count"] == 2
    assert model["control"]["retrain"] is False
    assert len(model["control"]["models"]) == 2
    assert model["inference"]["implemented"] is False
    assert model["inference"]["active_variants"] == []


def test_history_crossfit_splits_exclude_each_target_sample() -> None:
    samples = [f"44b6_{index:03d}" for index in range(11)]
    target_folds = split_two_folds(samples, seed=17)
    assert set(target_folds[0]).isdisjoint(target_folds[1])
    assert set(target_folds[0]) | set(target_folds[1]) == set(samples)
    for inner_fold, target in enumerate(target_folds):
        pool = sorted(set(samples) - set(target))
        gradient, validation = split_selection_samples(
            pool,
            seed=17 + inner_fold,
            validation_fraction=0.1,
        )
        assert gradient
        assert validation
        assert not set(target) & (set(gradient) | set(validation))
        assert set(gradient) | set(validation) == set(pool)


def test_history_propagation_records_velocity_and_resets_predicted_division() -> None:
    source_ids = np.asarray([10, 20], dtype=np.int64)
    source_coords = np.asarray([[0, 0, 0], [10, 0, 0]], dtype=np.float32)
    source = empty_history_frame(source_ids, source_coords)
    target, audit = propagate_history_frame(
        source,
        candidate_ids_src=source_ids,
        candidate_ids_tgt=np.asarray([30, 40], dtype=np.int64),
        coords_src_physical=source_coords,
        coords_tgt_physical=np.asarray([[0, 2, 0], [11, 0, 0]], dtype=np.float32),
        source_axis_probabilities=np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
        edge_probability_threshold=0.5,
    )
    assert target["present"].tolist() == [True, True]
    assert target["velocity"].tolist() == [[0.0, 2.0, 0.0], [1.0, 0.0, 0.0]]
    assert target["length"].tolist() == [1, 1]
    assert audit["selected_edge_count"] == 2

    division, division_audit = propagate_history_frame(
        source,
        candidate_ids_src=source_ids,
        candidate_ids_tgt=np.asarray([31, 32], dtype=np.int64),
        coords_src_physical=source_coords,
        coords_tgt_physical=np.asarray([[0, 1, 0], [0, -1, 0]], dtype=np.float32),
        source_axis_probabilities=np.asarray([[0.9, 0.8], [0.1, 0.2]], dtype=np.float32),
        edge_probability_threshold=0.5,
    )
    assert division["present"].tolist() == [False, False]
    assert division_audit == {
        "selected_edge_count": 2,
        "predicted_division_source_count": 1,
        "division_reset_target_count": 2,
        "history_present_target_count": 0,
    }


def test_history_alignment_uses_candidate_ids_and_checks_coordinates() -> None:
    history = empty_history_frame(
        np.asarray([10, 20]),
        np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
    )
    history["present"][:] = [True, False]
    history["velocity"][:] = [[1, 0, 0], [2, 0, 0]]
    aligned = align_history_frame(
        history,
        np.asarray([20, 10]),
        np.asarray([[4, 5, 6], [1, 2, 3]], dtype=np.float32),
    )
    assert aligned["velocity"].tolist() == [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert history_content_sha256({("sample", 0): history})
    with pytest.raises(ValueError, match="coordinates differ"):
        align_history_frame(
            history,
            np.asarray([20, 10]),
            np.asarray([[4, 5, 7], [1, 2, 3]], dtype=np.float32),
        )


def test_train_pipeline_has_six_new_models_gate_and_no_submission() -> None:
    source = (EXP / "train_pipeline.py").read_text(encoding="utf-8")
    for marker in (
        "inner_generator_model_count",
        "history_model_count",
        "model_count",
        "generate_fixed_first_pass_history",
        "load_control_tracker",
        "evaluate_tracker_diagnostic",
        "graph_progression_gate",
        "stop_before_full_graph_inference",
        "No submission was created.",
    ):
        assert marker in source
    assert "kaggle competitions submit" not in source
    assert "neutral control" not in source.lower()
