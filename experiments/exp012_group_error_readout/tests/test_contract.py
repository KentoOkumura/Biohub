from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import yaml

from tests.test_support import require_saved_files

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SOURCE = EXP / "exp012_group_error_readout_diagnostic.py"


def load_diagnostic_module():
    spec = importlib.util.spec_from_file_location("exp012_group_error_readout_diagnostic", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_row(**updates):
    row = {
        "edge_tp": 10,
        "edge_fp": 2,
        "edge_fn": 3,
        "division_tp": 1,
        "division_fp": 0,
        "division_fn": 1,
        "num_pred_nodes": 20,
        "node_recall": 0.8,
        "adj_edge_jaccard": 0.5,
        "evaluation_failed": False,
    }
    row.update(updates)
    return row


def test_cpu_only_diagnostic_contract_and_lineage():
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    assert config["experiment"]["notebooks"] == ["diagnostic"]
    assert config["lineage"] == {
        "parent": "exp011_public_detector_selection",
        "hypothesis_id": "HYP-20260910-14",
        "backlog_candidate": "group_error_readout",
        "diff_summary": config["lineage"]["diff_summary"],
    }
    assert config["validation"]["expected_sample_count"] == 199
    assert config["validation"]["expected_embryo_counts"] == {"44b6": 71, "6bba": 128}
    assert config["model"]["name"] == "none_saved_prediction_diagnostic"
    assert config["runtime"]["kaggle"]["enable_gpu"] is False
    assert config["runtime"]["kaggle"]["enable_internet"] is False
    assert config["runtime"]["kaggle"]["diagnostic"]["enable_gpu"] is False
    assert config["diagnostic"]["buckets"]["threshold_source"] == ("baseline_route_opposite_embryo")
    assert set(config["diagnostic"]["buckets"]["features"]) == {
        "brightness",
        "candidate_density",
        "boundary_distance",
    }
    assert len(config["diagnostic"]["prediction_sources"]) == 2
    source = SOURCE.read_text()
    assert "sample_submission" not in source
    assert "submission.csv" not in source
    assert "__file__" not in source


def test_parent_public_detector_manifest_is_pinned_and_adopted():
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    manifest_cfg = config["diagnostic"]["parent_public_detector_manifest"]
    path = (EXP / manifest_cfg["path"]).resolve()
    payload = json.loads(path.read_text())
    observed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert observed_sha == manifest_cfg["sha256"]
    assert payload["selection_state"] == manifest_cfg["selection_state"] == "adopted"


def test_official_group_summary_uses_micro_counts_and_adjusted_weights():
    diagnostic = load_diagnostic_module()
    rows = [
        metric_row(),
        metric_row(
            edge_tp=2,
            edge_fp=1,
            edge_fn=1,
            division_tp=0,
            division_fp=1,
            division_fn=0,
            num_pred_nodes=5,
            node_recall=0.6,
            adj_edge_jaccard=0.75,
        ),
    ]
    summary = diagnostic.summarise_official(rows, division_weight=0.1)
    expected_adjusted = (15 * 0.5 + 4 * 0.75) / 19
    assert math.isclose(summary["edge_jaccard"], 12 / 19)
    assert math.isclose(summary["division_jaccard"], 1 / 3)
    assert math.isclose(summary["adj_edge_jaccard"], expected_adjusted)
    assert math.isclose(summary["score"], expected_adjusted + 0.1 / 3)
    assert math.isclose(summary["node_recall"], 0.7)
    assert summary["n_total"] == summary["n_valid"] == summary["n_adj"] == 2
    assert summary["failure_count"] == summary["nan_count"] == 0


def test_failed_rows_remain_visible_but_do_not_change_valid_aggregate():
    diagnostic = load_diagnostic_module()
    failed = metric_row(
        edge_tp=float("nan"),
        edge_fp=float("nan"),
        edge_fn=float("nan"),
        adj_edge_jaccard=float("nan"),
        evaluation_failed=True,
    )
    summary = diagnostic.summarise_official([metric_row(), failed])
    assert summary["n_total"] == 2
    assert summary["n_valid"] == 1
    assert summary["failure_count"] == 1
    assert summary["coverage"] == 0.5
    assert math.isclose(summary["adj_edge_jaccard"], 0.5)


def test_candidate_boundary_distance_uses_physical_nearest_image_face():
    diagnostic = load_diagnostic_module()
    coords = np.asarray([[0, 0, 0, 0], [0, 2, 2, 2]], dtype=np.int16)
    summary = diagnostic.candidate_boundary_statistics(
        coords,
        spatial_shape_zyx=(5, 5, 5),
        scale_zyx_um=(2.0, 1.0, 0.5),
        near_distance_um=0.0,
    )
    assert math.isclose(summary["candidate_boundary_distance_um_p50"], 0.5)
    assert math.isclose(summary["candidate_near_boundary_fraction"], 0.5)


def test_bucket_boundaries_come_from_opposite_embryo_baseline_only():
    diagnostic = load_diagnostic_module()
    rows = []
    for embryo, values in {"44b6": [1, 2, 3, 4], "6bba": [10, 20, 30, 40]}.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "route": "baseline",
                    "sample": f"{embryo}_{index}",
                    "embryo": embryo,
                    "brightness": value,
                    "density": value,
                    "boundary": value,
                }
            )
    thresholds = diagnostic.derive_bucket_thresholds(
        rows,
        baseline_route="baseline",
        feature_columns={
            "brightness": "brightness",
            "candidate_density": "density",
            "boundary_distance": "boundary",
        },
        quantiles=[0.25, 0.5, 0.75],
    )
    assert thresholds["44b6"]["brightness"] == [17.5, 25.0, 32.5]
    assert thresholds["6bba"]["brightness"] == [1.75, 2.5, 3.25]


def test_saved_exp005_rows_reproduce_fixed_official_overall_when_available():
    diagnostic = load_diagnostic_module()
    per_sample_path = (
        ROOT
        / "experiments/exp005_embryo_holdout_batch8/artifacts/inference_v1/per_sample_metrics.json"
    )
    official_path = (
        ROOT
        / "experiments"
        / "exp005_embryo_holdout_batch8"
        / "artifacts/inference_v1/official_metric_summary.json"
    )
    require_saved_files(per_sample_path, official_path)
    rows = json.loads(per_sample_path.read_text())["rows"]
    for row in rows:
        row["evaluation_failed"] = False
    observed = diagnostic.summarise_official(rows)
    expected = json.loads(official_path.read_text())["overall"]
    for key in (
        "edge_jaccard",
        "division_jaccard",
        "adj_edge_jaccard",
        "node_recall",
        "score",
    ):
        assert math.isclose(observed[key], expected[key], rel_tol=0.0, abs_tol=1e-12)


def test_pinned_saved_route_inputs_share_the_image_feature_sample_set_when_available():
    diagnostic = load_diagnostic_module()
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    image_cfg = config["diagnostic"]["image_feature_source"]
    image_path = (EXP / image_cfg["local_path"]).resolve()
    route_paths = [
        (EXP / source["local_artifact_dir"]).resolve()
        for source in config["diagnostic"]["prediction_sources"]
    ]
    require_saved_files(
        image_path,
        *(path / "prediction_manifest.json" for path in route_paths),
        *(path / "per_sample_metrics.json" for path in route_paths),
    )
    image_rows = diagnostic.read_feature_rows(image_path, image_cfg)
    assert len(image_rows) == config["validation"]["expected_sample_count"]
    for source_cfg, source_dir in zip(
        config["diagnostic"]["prediction_sources"], route_paths, strict=True
    ):
        predictions, metrics, _ = diagnostic.load_prediction_rows(
            source_dir,
            source_dir / "prediction_manifest.json",
            source_dir / "per_sample_metrics.json",
            source_cfg,
        )
        assert set(predictions) == set(metrics) == set(image_rows)
