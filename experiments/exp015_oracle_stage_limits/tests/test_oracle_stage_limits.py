from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from tests.test_support import require_saved_files

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SOURCE = EXP / "exp015_oracle_stage_limits_diagnostic.py"
INFERENCE_SOURCE = EXP / "exp015_oracle_stage_limits_inference.py"

sys.path.insert(0, str(EXP))
from window_cache import (  # noqa: E402
    assert_exact_arrays,
    read_window_cache,
    write_window_cache,
)


def load_diagnostic_module():
    spec = importlib.util.spec_from_file_location("exp015_oracle_stage_limits_diagnostic", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def conditions(embryo: str = "44b6"):
    return {
        "embryo": embryo,
        "brightness_bucket": "Q4_high",
        "candidate_density_bucket": "Q3",
        "boundary_distance_bucket": "Q2",
        "known_division": True,
    }


def test_contract_lineage_and_runtime_resources():
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    assert config["experiment"]["notebooks"] == ["inference", "diagnostic"]
    assert config["lineage"]["parent"] == "exp014_exact_window_cache"
    assert config["lineage"]["hypothesis_id"] == "HYP-20260910-14"
    assert config["lineage"]["backlog_candidate"] == "oracle_stage_limits"
    assert config["validation"]["expected_sample_count"] == 199
    assert config["validation"]["expected_embryo_counts"] == {"44b6": 71, "6bba": 128}
    assert config["validation"]["matching"]["max_distance_um"] == 7.0
    assert config["model"]["params"]["change_prediction_parameters"] is False
    assert config["runtime"]["kaggle"]["inference"]["enable_gpu"] is True
    assert config["runtime"]["kaggle"]["diagnostic"]["enable_gpu"] is False
    assert config["runtime"]["kaggle"]["diagnostic"]["dataset_sources"] == [
        "pilkwang/biohub-tracking-support-pack-50ep-v1"
    ]
    assert "do_not_submit" in config["oracle"]["prohibitions"]
    cache = config["cache"]["train_window"]
    assert cache["expected_dataset_count"] == 199
    assert cache["expected_window_count"] == 19701
    assert cache["feature_channels"] == 32
    assert cache["max_bytes_per_gpu_shard"] == 6_000_000_000
    assert cache["max_total_cache_bytes"] == 12_000_000_000
    assert cache["notebook_output_soft_limit_bytes"] == 18_000_000_000
    assert cache["kaggle_output_limit_bytes"] == 20_000_000_000


def test_inference_uses_train_and_saves_pre_ilp_and_final_graphs():
    source = INFERENCE_SOURCE.read_text()
    assert "TEST_DIR = COMP_DIR / 'train'" in source
    assert "Expected 199 training videos" in source
    assert "Pre-ILP oracle candidate graph hook installed" in source
    assert "ORACLE_CANDIDATE_GRAPH_DIR" in source
    assert "ORACLE_FINAL_GRAPH_DIR" in source
    assert "oracle_inference_manifest.json" in source
    assert "ground_truth_accessed': False" in source
    for marker in (
        "EXP015_TRAIN_WINDOW_CACHE_ADDITION_START",
        "write_window_cache",
        "read_window_cache",
        "assert_exact_arrays",
        "expected_train_window_count",
        "window_cache_summary.json",
        "edge_logits_recomputed_for_cache",
        "prediction_uses_in_memory_features",
        "WINDOW_CACHE_MAX_BYTES_PER_GPU_SHARD",
        "NOTEBOOK_OUTPUT_SOFT_LIMIT_BYTES",
    ):
        assert marker in source
    assert "_exp015_cached_primary_logits" not in source
    assert "_exp015_cached_secondary_logits" not in source
    assert "np.savez_compressed" not in (EXP / "window_cache.py").read_text()
    assert "kaggle competitions submit" not in source


def test_train_window_cache_round_trip_preserves_dtype_shape_and_values(tmp_path):
    arrays = {
        "candidate_ids_src": np.array([10, 11], dtype=np.int64),
        "coords_src_physical": np.array([[1.625, 2.0, 3.0]], dtype=np.float32),
        "primary_features_src": np.arange(64, dtype=np.float16).reshape(2, 32),
        "secondary_features_src": np.arange(64, dtype=np.float32).reshape(2, 32),
    }
    metadata = {
        "experiment": "exp015_oracle_stage_limits",
        "dataset": "movie",
        "window_frames": [3, 4],
    }
    path = tmp_path / "movie__000003__000004.npz"
    written = write_window_cache(path, metadata=metadata, arrays=arrays)
    loaded, read = read_window_cache(path, expected_metadata=metadata)
    assert_exact_arrays(arrays, loaded)
    assert written["content_sha256"] == read["content_sha256"]
    assert written["schema_sha256"] == read["schema_sha256"]
    assert written["bytes"] == path.stat().st_size


def test_train_window_cache_rejects_wrong_window_identity(tmp_path):
    path = tmp_path / "cache.npz"
    write_window_cache(
        path,
        metadata={"dataset": "movie", "window_frames": [0, 1]},
        arrays={"features": np.zeros((1, 32), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="cache_metadata_mismatch"):
        read_window_cache(
            path,
            expected_metadata={"dataset": "movie", "window_frames": [1, 2]},
        )


def test_notebook_safe_diagnostic_source():
    source = SOURCE.read_text()
    assert "__file__" not in source
    assert "submission.csv" not in source
    assert "linear_sum_assignment" in source
    assert "ensure_geff_runtime_dependencies" in source
    assert "--no-deps" in source
    assert "--force-reinstall" in source
    assert "polars_runtime_ready" in source
    assert "DIAGNOSTIC_RECEIPT" in source
    assert "scope_summaries" in source
    assert 'if __name__ == "__main__":' in source


def test_per_frame_matching_is_one_to_one_and_reports_ambiguity():
    diagnostic = load_diagnostic_module()
    gt_nodes = {
        1: (0, 0.0, 0.0, 0.0),
        2: (0, 0.0, 1.0, 0.0),
        3: (1, 0.0, 0.0, 0.0),
    }
    candidate_nodes = {
        10: (0, 0.0, 0.1, 0.0),
        11: (0, 0.0, 0.9, 0.0),
        12: (1, 0.0, 0.1, 0.0),
    }
    mapping, readout = diagnostic.match_nodes_per_frame(
        gt_nodes,
        candidate_nodes,
        scale_zyx_um=(1.0, 1.0, 1.0),
        max_distance_um=2.0,
    )
    assert mapping == {1: 10, 2: 11, 3: 12}
    assert len(set(mapping.values())) == len(mapping)
    assert readout["ambiguous_gt_node_count"] == 2
    assert readout["ambiguous_candidate_node_count"] == 2
    assert math.isclose(readout["matching_distance_um_max"], 0.1)


def test_stage_limits_are_cumulative_and_division_requires_two_daughters():
    diagnostic = load_diagnostic_module()
    gt_nodes = {
        1: (0, 0.0, 0.0, 0.0),
        2: (1, 0.0, 1.0, 0.0),
        3: (1, 0.0, -1.0, 0.0),
        4: (2, 0.0, 2.0, 0.0),
    }
    gt_edges = {(1, 2), (1, 3), (2, 4)}
    candidate_nodes = {
        10: (0, 0.0, 0.0, 0.0),
        20: (1, 0.0, 1.0, 0.0),
        30: (1, 0.0, -1.0, 0.0),
        40: (2, 0.0, 2.0, 0.0),
    }
    candidate_edges = {(10, 20), (10, 30), (20, 40)}
    final_node_ids = {10, 20, 30, 40}
    final_edges = {(10, 20), (20, 40)}
    row = diagnostic.evaluate_sample_stages(
        "44b6_example",
        gt_nodes,
        gt_edges,
        candidate_nodes,
        candidate_edges,
        final_node_ids,
        final_edges,
        conditions(),
        scale_zyx_um=(1.0, 1.0, 1.0),
        max_distance_um=0.5,
    )
    assert row["known_nodes_candidate_matched"] == 4
    assert row["known_edges_endpoints_matched"] == 3
    assert row["known_edges_candidate_present"] == 3
    assert row["known_edges_final_selected"] == 2
    assert row["known_divisions_total"] == 1
    assert row["known_divisions_candidate_triplet"] == 1
    assert row["known_divisions_final_selected"] == 0
    assert row["candidate_edges_not_selected_final"] == 1
    assert math.isclose(row["candidate_edge_recall"], 1.0)
    assert math.isclose(row["final_selected_edge_recall"], 2 / 3)


def test_final_graph_npz_preserves_original_ids_and_edges(tmp_path):
    diagnostic = load_diagnostic_module()
    path = tmp_path / "sample.npz"
    np.savez_compressed(
        path,
        node_ids=np.asarray([10, 20], dtype=np.int64),
        node_tzyx=np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.float64),
        edges=np.asarray([[10, 20]], dtype=np.int64),
    )
    node_ids, nodes, edges, content_sha = diagnostic.load_final_graph(path)
    assert node_ids == {10, 20}
    assert nodes[10] == (0, 1.0, 2.0, 3.0)
    assert edges == {(10, 20)}
    assert len(content_sha) == 64


def test_condition_summaries_include_full_degraded_and_embryo_scopes():
    diagnostic = load_diagnostic_module()
    base = {
        **conditions(),
        "evaluation_failed": False,
        **{column: 1 for column in diagnostic.COUNT_COLUMNS},
    }
    rows = [
        {"sample": "44b6_a", **base},
        {"sample": "44b6_b", **base},
        {"sample": "6bba_c", **{**base, "embryo": "6bba"}},
    ]
    summaries = diagnostic.make_condition_summaries(
        rows,
        degraded_samples={"44b6_a"},
        degraded_embryo="44b6",
        condition_columns=["embryo", "brightness_bucket"],
    )
    scopes = {row["scope"] for row in summaries}
    assert scopes == {"all_199", "degraded_16", "all_44b6", "degraded_16_44b6"}
    overall = next(
        row for row in summaries if row["scope"] == "all_199" and row["group_type"] == "all"
    )
    assert overall["sample_count"] == 3
    assert overall["valid_sample_count"] == 3


def test_fixed_exp012_sha_and_degraded_cohort_when_saved():
    diagnostic = load_diagnostic_module()
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    root = ROOT / "experiments" / "exp012_group_error_readout" / "artifacts" / "readout_v1"
    per_sample = root / "per_sample_readout.csv"
    paired = root / "paired_route_comparison.csv"
    require_saved_files(per_sample, paired)
    conditions_by_sample, degraded, evidence = diagnostic.load_fixed_conditions(
        root,
        config["data"]["exp012_readout"],
    )
    assert len(conditions_by_sample) == 199
    assert len(degraded) == 16
    assert {row["embryo"] for row in degraded} == {"44b6"}
    assert (
        evidence["per_sample_readout_sha256"]
        == (config["data"]["exp012_readout"]["per_sample_readout_sha256"])
    )
