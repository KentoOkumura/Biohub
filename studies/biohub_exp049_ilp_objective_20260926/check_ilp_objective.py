"""Check the exp049 ILP on a synthetic division; no competition predictions."""

import ast
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp


def main():
    root = Path(__file__).resolve().parents[2]
    exp = root / "experiments/exp049_x138_edge_selection_diagnostic"
    source = exp / "exp049_x138_edge_selection_diagnostic_diagnostic.py"
    tree = ast.parse(source.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "solve_ilp"
    )
    params = yaml.safe_load((exp / "config.yaml").read_text())["model"]["params"]
    costs = {
        name: params[f"ilp_{name}_weight"]
        for name in ("edge", "appearance", "disappearance", "division")
    }
    namespace = dict(globals(), ILP_COSTS=costs, ILP_TIMEOUT=10)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    solve = namespace["solve_ilp"]
    rows = []
    for probability in (0.1, 0.5, 1.0):
        edges = np.array(
            [[0, 1, 1, 0], [0, 2, probability, 0]]
            + [[s, t, 1, 0] for s, t in ((1, 3), (3, 5), (5, 7), (2, 4), (4, 6), (6, 8))],
            dtype=float,
        )
        normal = solve(9, edges)
        forced = solve(9, edges, {(0, 1), (0, 2)})
        assert normal["optimal"] and forced["optimal"]
        assert max(Counter(s for s, _ in normal["selected_edges"]).values()) == 1
        expected = costs["division"] - costs["appearance"] + costs["edge"] * probability
        actual = forced["objective"] - normal["objective"]
        assert np.isclose(actual, expected)
        rows.append({"second_daughter_probability": probability, "normal_objective": normal["objective"],
                     "forced_division_objective": forced["objective"], "extra_cost": actual})
    result = {"kind": "synthetic_objective_check_not_competition_evaluation", "costs": costs, "cases": rows}
    output = Path(__file__).with_name("objective_check.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
