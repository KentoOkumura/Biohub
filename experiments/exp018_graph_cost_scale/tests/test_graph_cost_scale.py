from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

EXP = Path(__file__).resolve().parents[1]
SOURCE = EXP / "exp018_graph_cost_scale_diagnostic.py"


def load_diagnostic_module():
    module_name = "exp018_graph_cost_scale_diagnostic"
    spec = importlib.util.spec_from_file_location(module_name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_summaries():
    scores = {
        "44b6": {0.25: 0.45, 0.5: 0.70, 1.0: 0.50, 2.0: 0.60, 4.0: 0.40},
        "6bba": {0.25: 0.45, 0.5: 0.60, 1.0: 0.50, 2.0: 0.70, 4.0: 0.40},
    }
    counts = {"44b6": 71, "6bba": 128}
    return [
        {
            "embryo": embryo,
            "alpha": alpha,
            "sample_count": counts[embryo],
            "valid_sample_count": counts[embryo],
            "failure_count": 0,
            "score": score,
            "division_jaccard": 0.4,
        }
        for embryo, values in scores.items()
        for alpha, score in values.items()
    ]


def test_cpu_only_fixed_candidate_contract():
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    params = config["model"]["params"]
    assert config["experiment"]["notebooks"] == [
        "alpha_025",
        "alpha_05",
        "alpha_1",
        "alpha_2",
        "alpha_4",
        "aggregate",
    ]
    assert config["lineage"] == {
        "parent": "exp015_oracle_stage_limits",
        "hypothesis_id": "HYP-20260910-11",
        "backlog_candidate": "graph_cost_scale",
        "diff_summary": config["lineage"]["diff_summary"],
    }
    assert params["alpha_grid"] == [0.25, 0.5, 1.0, 2.0, 4.0]
    assert params["baseline_alpha"] == 1.0
    assert params["edge_cost_expression"] == "-alpha * edge_prob"
    assert params["appearance_weight"] == 0.0
    assert params["disappearance_weight"] == 2.0
    assert params["division_weight"] == 1.2
    assert params["trained_model_count"] == 0
    assert params["booster_count"] == 0
    assert params["control_retrained"] is False
    assert config["runtime"]["kaggle"]["enable_gpu"] is False
    assert config["runtime"]["kaggle"]["enable_internet"] is False
    assert config["stages"]["initial_ilp_only"]["repair_enabled"] is False
    shards = config["stages"]["initial_ilp_only"]["alpha_shards"]
    assert [float(shards[name]["alpha"]) for name in shards] == [0.25, 0.5, 1.0, 2.0, 4.0]
    assert config["stages"]["initial_ilp_only"]["execution_mode"] == (
        "independent_sequential_alpha_shards_then_aggregate"
    )
    assert len(config["runtime"]["kaggle"]["aggregate"]["kernel_sources"]) == 5
    assert config["stages"]["fixed_full_repair"]["implemented"] is False
    assert config["stages"]["fixed_full_repair"]["experiment"] == "exp018_graph_cost_scale"


def test_source_preserves_public_ilp_call_and_forbids_repair_and_submit():
    source = SOURCE.read_text()
    assert 'edge_weight=-float(alpha) * td.EdgeAttr("edge_prob")' in source
    assert 'appearance_weight=float(solver_cfg["appearance_weight"])' in source
    assert 'disappearance_weight=float(solver_cfg["disappearance_weight"])' in source
    assert 'division_weight=float(solver_cfg["division_weight"])' in source
    assert 'repair_executed": False' in source
    assert "Official metric smoke evaluation passed." in source
    assert "official metric smoke evaluation failed" in source
    assert "motion_relink" not in source
    assert "competitions submit" not in source
    assert "__file__" not in source


def test_support_repo_root_removes_the_full_evaluator_relative_path(tmp_path):
    diagnostic = load_diagnostic_module()
    evaluator = tmp_path / "tracking_repo" / "repo" / "scripts" / "evaluate.py"
    evaluator.parent.mkdir(parents=True)
    evaluator.touch()
    root = diagnostic.root_containing_relative_file(evaluator, "repo/scripts/evaluate.py")
    assert root == tmp_path / "tracking_repo"


def test_official_metric_import_root_is_derived_from_pinned_metrics_path():
    source = SOURCE.read_text()
    assert "source_root = metrics_path.parent.parent" in source
    assert 'source_root = repo_root / "src"' not in source


def test_kaggle_kernel_inputs_support_direct_and_notebooks_mount_layouts():
    source = SOURCE.read_text()
    assert source.count('Path("/kaggle/input").rglob(') >= 3
    assert 'Path("/kaggle/input/notebooks").rglob(' not in source


def test_official_metric_content_receipt_ignores_mount_path_but_not_content():
    diagnostic = load_diagnostic_module()
    hashes = {
        "evaluator_sha256": "a" * 64,
        "metrics_sha256": "b" * 64,
        "division_metrics_sha256": "c" * 64,
    }
    nested_mount = {
        **hashes,
        "support_repo": "/kaggle/input/datasets/owner/support-pack",
    }
    direct_mount = {
        **hashes,
        "support_repo": "/kaggle/input/support-pack",
    }
    assert diagnostic.official_metric_content_receipt(nested_mount) == (
        diagnostic.official_metric_content_receipt(direct_mount)
    )

    changed = {**direct_mount, "metrics_sha256": "d" * 64}
    assert diagnostic.official_metric_content_receipt(changed) != (
        diagnostic.official_metric_content_receipt(direct_mount)
    )


def test_solver_graph_view_is_detached_before_metric_use(monkeypatch, tmp_path):
    diagnostic = load_diagnostic_module()
    candidate = object()
    detached = object()

    class FakeView:
        def detach(self):
            return detached

    class FakeSolver:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def solve(self, graph):
            assert graph is candidate
            return FakeView()

    class FakeEdgeAttr:
        def __init__(self, name):
            self.name = name

        def __rmul__(self, value):
            return (value, self.name)

    fake_td = SimpleNamespace(
        EdgeAttr=FakeEdgeAttr,
        solvers=SimpleNamespace(ILPSolver=FakeSolver),
    )
    monkeypatch.setitem(sys.modules, "tracksdata", fake_td)
    monkeypatch.setattr(diagnostic, "graph_from_geff", lambda path: candidate)
    monkeypatch.setattr(diagnostic, "candidate_graph_receipt", lambda graph: {"candidate": 1})
    monkeypatch.setattr(
        diagnostic,
        "solution_topology_receipt",
        lambda graph: {"detached": graph is detached},
    )

    solution, candidate_receipt, solution_receipt = diagnostic.solve_candidate_graph(
        tmp_path / "candidate.geff",
        0.5,
        {
            "appearance_weight": 0.0,
            "disappearance_weight": 2.0,
            "division_weight": 1.2,
        },
    )
    assert solution is detached
    assert candidate_receipt == {"candidate": 1}
    assert solution_receipt == {"detached": True}


def test_alpha_metric_csv_row_parser_restores_numeric_types():
    diagnostic = load_diagnostic_module()
    row = {
        "sample": "44b6_sample",
        "embryo": "44b6",
        "alpha": "0.5",
        "candidate_graph_content_sha256": "a" * 64,
        "candidate_node_count": "10",
        "candidate_edge_count": "9",
        "edge_probability_min": "0.1",
        "edge_probability_max": "0.9",
        "solution_topology_sha256": "b" * 64,
        "solution_node_count": "10",
        "solution_edge_count": "8",
        "edge_tp": "7",
        "edge_fp": "1",
        "edge_fn": "2",
        "division_tp": "1",
        "division_fp": "0",
        "division_fn": "1",
        "num_pred_nodes": "10",
        "node_recall": "0.8",
        "total_node_ratio": "0.0",
        "edge_jaccard": "0.7",
        "adj_edge_jaccard": "0.7",
        "failure": "",
    }
    parsed = diagnostic.parse_alpha_metric_row(row)
    assert parsed["alpha"] == 0.5
    assert parsed["edge_tp"] == 7
    assert parsed["adj_edge_jaccard"] == 0.7


def test_score_tie_prefers_baseline_then_nearest_lower_alpha():
    diagnostic = load_diagnostic_module()
    rows = make_summaries()
    for row in rows:
        if row["embryo"] == "44b6":
            row["score"] = 0.5
    selected = diagnostic.select_alpha(
        rows,
        "44b6",
        [0.25, 0.5, 1.0, 2.0, 4.0],
        baseline_alpha=1.0,
        score_tolerance=1e-12,
    )
    assert selected["alpha"] == 1.0

    rows = [row for row in rows if row["alpha"] != 1.0]
    selected = diagnostic.select_alpha(
        rows,
        "44b6",
        [0.25, 0.5, 2.0, 4.0],
        baseline_alpha=1.0,
        score_tolerance=1e-12,
    )
    assert selected["alpha"] == 0.5


def test_two_direction_gate_uses_calibration_only_and_requires_both_improvements():
    diagnostic = load_diagnostic_module()
    rows = make_summaries()
    directions = [
        {"calibration_embryo": "44b6", "heldout_embryo": "6bba"},
        {"calibration_embryo": "6bba", "heldout_embryo": "44b6"},
    ]
    decisions, passed = diagnostic.build_cross_embryo_decisions(
        rows,
        directions,
        [0.25, 0.5, 1.0, 2.0, 4.0],
        {"44b6": 71, "6bba": 128},
        baseline_alpha=1.0,
        minimum_score_delta=0.0,
        score_tolerance=1e-12,
        require_division_nonregression=True,
    )
    assert [row["selected_alpha"] for row in decisions] == [0.5, 2.0]
    assert passed is True

    # 6bba selects alpha=2 before the 44b6 held-out score is read.
    heldout = next(row for row in rows if row["embryo"] == "44b6" and row["alpha"] == 2.0)
    heldout["score"] = 0.49
    decisions, passed = diagnostic.build_cross_embryo_decisions(
        rows,
        directions,
        [0.25, 0.5, 1.0, 2.0, 4.0],
        {"44b6": 71, "6bba": 128},
        baseline_alpha=1.0,
        minimum_score_delta=0.0,
        score_tolerance=1e-12,
        require_division_nonregression=True,
    )
    assert decisions[1]["selected_alpha"] == 2.0
    assert decisions[1]["combined_score_strictly_improved"] is False
    assert passed is False


def test_division_regression_and_incomplete_coverage_stop_promotion():
    diagnostic = load_diagnostic_module()
    rows = make_summaries()
    directions = [
        {"calibration_embryo": "44b6", "heldout_embryo": "6bba"},
        {"calibration_embryo": "6bba", "heldout_embryo": "44b6"},
    ]
    selected_heldout = next(row for row in rows if row["embryo"] == "6bba" and row["alpha"] == 0.5)
    selected_heldout["division_jaccard"] = 0.39
    decisions, passed = diagnostic.build_cross_embryo_decisions(
        rows,
        directions,
        [0.25, 0.5, 1.0, 2.0, 4.0],
        {"44b6": 71, "6bba": 128},
        baseline_alpha=1.0,
        minimum_score_delta=0.0,
        score_tolerance=1e-12,
        require_division_nonregression=True,
    )
    assert decisions[0]["division_jaccard_not_worse"] is False
    assert passed is False

    selected_heldout["division_jaccard"] = 0.4
    selected_heldout["valid_sample_count"] = 127
    decisions, passed = diagnostic.build_cross_embryo_decisions(
        rows,
        directions,
        [0.25, 0.5, 1.0, 2.0, 4.0],
        {"44b6": 71, "6bba": 128},
        baseline_alpha=1.0,
        minimum_score_delta=0.0,
        score_tolerance=1e-12,
        require_division_nonregression=True,
    )
    assert all(row["complete_coverage"] is False for row in decisions)
    assert passed is False
