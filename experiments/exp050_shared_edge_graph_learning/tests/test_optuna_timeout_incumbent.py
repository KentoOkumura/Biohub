"""Contract checks for the LB-only Optuna SCIP incumbent path."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from experiments.exp050_shared_edge_graph_learning import exp050_optuna_solver_worker as worker

SOURCE = (
    Path(__file__).resolve().parents[1] / "exp050_shared_edge_graph_learning_submission_optuna.py"
)


class FakeGraph:
    def add_node_attr_key(self, *_args):
        pass

    def bulk_add_nodes(self, rows):
        return list(range(len(rows)))

    def add_edge_attr_key(self, *_args):
        pass

    def bulk_add_edges(self, _rows):
        pass


class FakeRows:
    def __init__(self, rows):
        self.rows = rows

    def iter_rows(self, *, named):
        assert named
        return iter(self.rows)


class FakeSolution:
    def __init__(self, edges=None):
        self.edges = (
            edges if edges is not None else [{"source_id": 0, "target_id": 1, "edge_prob": 0.8}]
        )

    def node_attrs(self):
        return FakeRows(
            [
                {"node_id": 0, "t": 0, "z": 0, "y": 0, "x": 0},
                {"node_id": 1, "t": 1, "z": 0, "y": 1, "x": 1},
            ]
        )

    def edge_attrs(self):
        return FakeRows(self.edges)


def run_solver(warnings, solution):
    log = SimpleNamespace(handler=None)
    log.addHandler = lambda handler: setattr(log, "handler", handler)
    log.removeHandler = lambda _handler: None

    class WarningCollector:
        def __init__(self):
            self.messages = []

    class FakeSolver:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] == 1200

        def solve(self, _graph):
            log.handler.messages.extend(warnings)
            return solution

    ticks = iter([100.0, 1359.0])
    td = SimpleNamespace(
        graph=SimpleNamespace(InMemoryGraph=FakeGraph),
        solvers=SimpleNamespace(ILPSolver=FakeSolver),
        EdgeAttr=lambda _name: 1.0,
    )
    coords = [(0, 0, 0, 0), (1, 0, 1, 1)]
    edges = [(0, 1, 0.8, 1.0)]
    with (
        patch.object(
            worker,
            "_solver_dependencies",
            return_value=(td, SimpleNamespace(Float64=float), log),
        ),
        patch.object(worker, "SolverWarnings", WarningCollector),
        patch.object(worker, "time", SimpleNamespace(monotonic=lambda: next(ticks))),
    ):
        return worker.legacy_solve(coords, edges, (0.5, 2.0, -0.2), "video", 1200)


def test_timeout_incumbent_is_accepted_and_marked_nonoptimal():
    _, selected, info = run_solver(
        ["Solver did not converge to an optimal solution, returned status SolverStatus.TIMELIMIT."],
        FakeSolution(),
    )
    assert selected == [{"source_id": 0, "target_id": 1, "edge_prob": 0.8}]
    assert info["solver_status"] == "TIMELIMIT_FEASIBLE"
    assert info["optimality_proven"] is False
    assert info["solver_limit_seconds"] == 1200


def test_optimal_solution_remains_distinct_from_timeout_incumbent():
    _, _, info = run_solver([], FakeSolution())
    assert info["solver_status"] == "OPTIMAL"
    assert info["optimality_proven"] is True


@pytest.mark.parametrize(
    ("warnings", "solution"),
    [
        (["Solver did not converge, returned status SolverStatus.INFEASIBLE."], FakeSolution()),
        (["Solver did not converge, returned status SolverStatus.TIMELIMIT."], None),
        (["Trivial solution"], FakeSolution()),
        (
            ["Solver did not converge, returned status SolverStatus.TIMELIMIT."],
            FakeSolution([{"source_id": 1, "target_id": 0, "edge_prob": 0.8}]),
        ),
    ],
)
def test_unusable_incumbent_is_rejected(warnings, solution):
    with pytest.raises(RuntimeError):
        run_solver(warnings, solution)


def selected_cost_function():
    module = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "equal_weight_selected_costs"
    )
    namespace = {"np": __import__("numpy")}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["equal_weight_selected_costs"]


def test_equal_weight_costs_from_frozen_trials():
    receipt = {
        "folds": {
            "0": {
                "selected_trial": 2,
                "selected_costs": [0.5986584841970366, 0.46805592132730955, -0.5320164389913921],
            },
            "1": {
                "selected_trial": 4,
                "selected_costs": [0.9697115002140642, 2.9957539309421573, 1.913067542545972],
            },
        }
    }
    assert selected_cost_function()(receipt) == pytest.approx(
        (0.7841849922055504, 1.7319049261347334, 0.69052555177729)
    )


def test_equal_weight_costs_reject_changed_trial():
    receipt = {
        "folds": {
            "0": {"selected_trial": 3, "selected_costs": [0.5, 1.0, 0.0]},
            "1": {"selected_trial": 4, "selected_costs": [0.5, 1.0, 0.0]},
        }
    }
    with pytest.raises(RuntimeError, match="trial selection changed"):
        selected_cost_function()(receipt)
