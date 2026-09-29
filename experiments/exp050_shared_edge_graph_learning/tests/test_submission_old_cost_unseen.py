"""The fixed-cost submission must accept the competition's unseen embryos."""

from __future__ import annotations

import ast
import json
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[1] / "exp050_shared_edge_graph_learning_submission_old_cost.py"
)


def test_test_split_includes_unseen_embryo(tmp_path):
    module = ast.parse(SOURCE.read_text())
    start = next(
        index
        for index, node in enumerate(module.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "TRAIN_DIR" for target in node.targets
        )
    )
    end = next(
        index
        for index in range(start, len(module.body))
        if isinstance(module.body[index], ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "predict_cmd"
            for target in module.body[index].targets
        )
    )
    list_test_stems = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "list_test_stems"
    )
    code = compile(
        ast.Module(body=[list_test_stems, *module.body[start:end]], type_ignores=[]),
        str(SOURCE),
        "exec",
    )
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    for name in ("44b6_public.zarr", "new_embryo_hidden.zarr"):
        (test_dir / name).mkdir()
    namespace = {
        "COMP_DIR": tmp_path,
        "REPO_DIR": tmp_path,
        "TEST_DIR": test_dir,
        "json": json,
    }
    exec(code, namespace)

    assert namespace["test_stems"] == ["44b6_public", "new_embryo_hidden"]
    split = json.loads((tmp_path / "kaggle_test_splits_50ep.json").read_text())
    assert split[0]["test"] == namespace["test_stems"]
