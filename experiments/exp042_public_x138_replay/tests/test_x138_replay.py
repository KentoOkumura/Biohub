from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "assets/reference_notebook/biohub-x138.ipynb"
INFERENCE = ROOT / "exp042_public_x138_replay_inference.ipynb"
EXPECTED_SHA = "6b655e39bbfd2d3d6c762badea69847d3f00f5b548f385cb01b07ee2600fde6d"

sys.path.insert(0, str(ROOT))
from compare_replays import compare_run_dirs, json_sha256  # noqa: E402


def source(cell: dict) -> str:
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def original_and_generated() -> tuple[list[str], list[str]]:
    reference = json.loads(REFERENCE.read_text())
    generated = json.loads(INFERENCE.read_text())
    original_code = [source(cell) for cell in reference["cells"] if cell["cell_type"] == "code"]
    generated_code = [
        source(cell)
        for cell in generated["cells"]
        if cell["cell_type"] == "code" and "EXP042_ADDITION_START" not in source(cell)
    ]
    return original_code, generated_code


def test_public_source_is_pinned_and_prediction_cells_match() -> None:
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == EXPECTED_SHA
    original, generated = original_and_generated()
    assert len(original) == 12
    assert len(generated) == 13  # Public cell 4 is split around read-only instrumentation.
    generated[4:6] = [generated[4] + "\n" + generated[5]]
    for index, (public, implementation) in enumerate(zip(original, generated, strict=True)):
        assert ast.dump(ast.parse(implementation), include_attributes=False) == ast.dump(
            ast.parse(public), include_attributes=False
        ), index


def test_checkpoint_guard_and_coordinate_diagnostics_are_present() -> None:
    notebook = json.loads(INFERENCE.read_text())
    code = "\n".join(source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code")
    head = yaml.safe_load((ROOT / "config.yaml").read_text())["data"]["v1284_head"]
    assert f"_X138_HEAD_SHA256 = {(head['sha256'] or '')!r}" in code
    assert f"_X138_HEAD_DATASET_VERSION_ID = {head['version_id'] or ''!r}" in code
    assert "V1284 checkpoint dataset ref and SHA256 must be pinned" in code
    assert "V1284_MODE') != 'candidate'" in code
    assert "v1284_coordinates_" in code
    assert "pre_refinement_coordinates_sha256" in code
    assert "post_refinement_coordinates_sha256" in code
    assert "graph_topology_sha256" in code
    assert "submission_sha256" in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(source(cell))

    if not head["sha256"]:
        with pytest.raises(RuntimeError, match="must be pinned"):
            exec(source(notebook["cells"][1]), {})


def test_coordinate_diagnostic_preserves_refine_output(tmp_path: Path) -> None:
    import numpy as np

    notebook = json.loads(INFERENCE.read_text())
    diagnostic = next(
        source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "V1284 pre/post-coordinate diagnostics installed" in source(cell)
    )
    tree = ast.parse(diagnostic)
    module_source = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_x138_diagnostic_source"
            for target in node.targets
        )
    )
    module_source = module_source.replace("Path('/kaggle/working')", f"Path({str(tmp_path)!r})")

    def reference_refine(ds_path, t, arr, feature):
        return arr.astype(np.float32) + np.float32(0.5)

    namespace = {"np": np, "os": os, "Path": Path, "refine": reference_refine}
    exec(module_source, namespace)
    coords = np.array([[1, 2, 3, 4]], dtype=np.int16)
    result = namespace["refine"](Path("movie.zarr"), 1, coords, None)
    assert np.array_equal(result, reference_refine(None, None, coords, None))
    records = (tmp_path / "v1284_coordinates_single.jsonl").read_text().splitlines()
    assert len(records) == 1
    record = json.loads(records[0])
    assert record["dataset"] == "movie"
    assert record["changed_rows"] == 1
    assert record["before_sha256"] != record["after_sha256"]


def write_run(root: Path, *, post_sha: str = "post-a", runtime: str = "10") -> None:
    root.mkdir()
    submission = b"id,dataset,row_type\n0,movie,node\n"
    (root / "submission.csv").write_bytes(submission)
    input_manifest = {
        "schema_version": 1,
        "v1284_head": {"sha256": "head-a"},
        "test_movies": {"movie": {"files": 1}},
    }
    input_sha = json_sha256(input_manifest)
    (root / "replay_input_manifest.json").write_text(
        json.dumps({**input_manifest, "input_manifest_sha256": input_sha})
    )
    receipt = {
        "input_manifest_sha256": input_sha,
        "v1284_head_sha256": "head-a",
        "pre_refinement_coordinates_sha256": "pre-a",
        "post_refinement_coordinates_sha256": post_sha,
        "graph_topology_sha256": "graph-a",
        "submission_sha256": hashlib.sha256(submission).hexdigest(),
        "prediction_seconds_observed": runtime,
        "comparison_fields": [
            "input_manifest_sha256",
            "v1284_head_sha256",
            "pre_refinement_coordinates_sha256",
            "post_refinement_coordinates_sha256",
            "graph_topology_sha256",
            "submission_sha256",
        ],
    }
    (root / "replay_receipt.json").write_text(
        json.dumps({**receipt, "receipt_sha256": json_sha256(receipt)})
    )


def test_compare_replays_requires_post_refinement_match_but_ignores_runtime(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    write_run(first)
    write_run(second, runtime="12")
    assert compare_run_dirs(first, second)["byte_identical_to_reference"]
    write_run_third = tmp_path / "third"
    write_run(write_run_third, post_sha="post-b")
    report = compare_run_dirs(first, write_run_third)
    assert report["mismatches"] == ["post_refinement_coordinates_sha256"]


def test_compare_replays_rejects_tampered_receipt(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    write_run(first)
    write_run(second)
    path = second / "replay_receipt.json"
    receipt = json.loads(path.read_text())
    receipt["v1284_head_sha256"] = "different"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt_sha_mismatch"):
        compare_run_dirs(first, second)
