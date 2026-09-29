from __future__ import annotations

import ast
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ROOT / f"exp051_x138_ilp_score_clipping_{kind}.py"
    for kind in ("tune0_control", "tune0_clipped", "tune1_control", "tune1_clipped", "evaluate")
]
RESUME_SOURCE = ROOT / "exp051_x138_ilp_score_clipping_evaluate_resume.py"
FROZEN_EVALUATION_SHA = "0c8ee638421b5b45b55a0ab4942efe29cde56321bcd99139cd68506ebe1469ea"


def extracted_function(path: Path, name: str, namespace: dict) -> object:
    tree = ast.parse(path.read_text())
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(functions) == 1
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def test_every_notebook_embeds_current_config_and_same_clipping_solver() -> None:
    package_digest = hashlib.sha256((ROOT / "config.yaml").read_bytes()).hexdigest()
    tuning_digest = "578649cd79b3e6977d0fd59f2a866fdfbc4662fe837da5ea5ee6e5b9efdf3639"
    functions = []
    ordered_solvers = []
    for path in SOURCES:
        source = path.read_text()
        expected_digest = FROZEN_EVALUATION_SHA if path == SOURCES[-1] else tuning_digest
        assert f'CONFIG_SHA256 = "{expected_digest}"' in source
        assert source.index('sys.path.insert(0, str(REPO_DIR / "src"))') < source.index(
            "from biohub_tracking.io import save_graph"
        )
        assert 'edge_weight=-1.0 * td.EdgeAttr("ilp_score")' in source
        tree = ast.parse(source)
        solve = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "legacy_solve"
        )
        functions.append(ast.dump(solve, include_attributes=False))
        ordered = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "OrderedFlowILPSolver"
        )
        ordered_solvers.append(ast.dump(ordered, include_attributes=False))
        assert source.count("maintain_order=True") == 2
        assert "edge_ids = sorted(group[DEFAULT_ATTR_KEYS.EDGE_ID].to_list())" in source
    assert len(set(functions[:-1])) == 1
    assert functions[-1] != functions[0]  # evaluation has a longer, separate ILP limit
    assert len(set(ordered_solvers[:-1])) == 1
    assert ordered_solvers[-1] != ordered_solvers[0]  # TIMELIMIT incumbent audit
    resume_source = RESUME_SOURCE.read_text()
    assert f'CONFIG_SHA256 = "{FROZEN_EVALUATION_SHA}"' in resume_source
    assert f'RESUME_PACKAGE_CONFIG_SHA256 = "{package_digest}"' in resume_source
    assert 'REUSED_DATASET_ID = "kentookumura/exp051-x138-clipping-v2-graphs"' in resume_source
    resume_solver = next(
        node
        for node in ast.parse(resume_source).body
        if isinstance(node, ast.FunctionDef) and node.name == "legacy_solve"
    )
    assert ast.dump(resume_solver, include_attributes=False) == functions[-1]
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert cfg["experiment"]["notebooks"] == [
        "tune0_control",
        "tune0_clipped",
        "tune1_control",
        "tune1_clipped",
        "evaluate",
        "evaluate_resume",
    ]
    assert cfg["optuna"]["max_elapsed_hours_per_fold_arm"] == 6
    assert cfg["optuna"]["min_valid_trials_per_fold_arm"] == 4
    assert cfg["clipping"]["search_range"] == [0.2, 0.5]
    assert cfg["model"]["decoding"]["num_threads"] == 1
    assert cfg["model"]["decoding"]["time_limit_seconds_per_video"] == 1200
    assert cfg["runtime"]["evaluate_ilp_time_limit_seconds_per_video"] == 3600
    resume_runtime = cfg["runtime"]["kaggle"]["evaluate_resume"]
    assert "kentookumura/exp051-x138-clipping-v2-graphs" in resume_runtime["dataset_sources"]
    assert (
        "kentookumura/exp051-x138-ilp-score-clipping-evaluate"
        not in resume_runtime["kernel_sources"]
    )
    assert cfg["validation"]["outer_accept_checked_incumbent_on_timelimit"] is True
    assert f'TUNING_CONFIG_SHA256 = "{tuning_digest}"' in SOURCES[-1].read_text()


def test_resume_notebook_contains_current_dataset_loader() -> None:
    source = RESUME_SOURCE.read_text()
    notebook = json.loads(RESUME_SOURCE.with_suffix(".ipynb").read_text())
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    loader = source.split("def find_reused_graph_root", 1)[1].split("\n\nmanifest_path", 1)[0]
    assert loader.strip() in code
    assert "reused_graphs.zip" not in code


def test_resume_finds_kaggle_dataset_mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_root = tmp_path / "input"
    dataset_root = input_root / "datasets/kentookumura/exp051-x138-clipping-v2-graphs"
    (dataset_root / "exp051_clipping").mkdir(parents=True)
    (tmp_path / "assets").mkdir()
    manifest = (ROOT / "assets/reused_graph_manifest.json").read_bytes()
    (dataset_root / "reused_graph_manifest.json").write_bytes(manifest)
    (tmp_path / "assets/reused_graph_manifest.json").write_bytes(manifest)
    monkeypatch.chdir(tmp_path)
    loader = extracted_function(RESUME_SOURCE, "find_reused_graph_root", {"Path": Path})
    assert loader(input_root) == dataset_root / "exp051_clipping"


def test_reused_selection_file_and_content_hashes_are_distinct() -> None:
    selection = {"outer": {"44b6": ["video_b"]}, "seed": 42}
    selection_bytes = (json.dumps(selection, indent=2, sort_keys=True) + "\n").encode()
    file_sha = hashlib.sha256(selection_bytes).hexdigest()
    content_sha = hashlib.sha256(
        json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert file_sha != content_sha
    config_bytes = b"frozen-evaluation-config"
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    manifest = {
        "source_kernel_id": "kentookumura/exp051-x138-ilp-score-clipping-evaluate",
        "source_kernel_version": 2,
        "selection_sha256": file_sha,
        "source_config_sha256": config_sha,
    }
    validator = extracted_function(
        RESUME_SOURCE,
        "validate_reused_source_provenance",
        {"hashlib": hashlib, "json": json},
    )
    result = validator(manifest, selection_bytes, config_bytes, content_sha, config_sha)
    assert result["selection_file_sha256"] == file_sha
    assert result["selection_content_sha256"] == content_sha
    with pytest.raises(RuntimeError, match="selection_content_sha"):
        validator(manifest, selection_bytes, config_bytes, file_sha, config_sha)


def test_ilp_gets_clipped_attribute_but_preserves_raw_score_and_id_order() -> None:
    class FakeRows:
        def __init__(self, rows: list[dict]):
            self.rows = rows

        def iter_rows(self, *, named: bool):
            assert named
            return iter(self.rows)

    class FakeGraph:
        def __init__(self):
            self.nodes: list[dict] = []
            self.edges: list[dict] = []

        def add_node_attr_key(self, *_args):
            pass

        def add_edge_attr_key(self, *_args):
            pass

        def bulk_add_nodes(self, rows):
            self.nodes = [dict(node_id=i, **row) for i, row in enumerate(rows)]
            return list(range(len(rows)))

        def bulk_add_edges(self, rows):
            self.edges = rows

        def node_attrs(self):
            return FakeRows(self.nodes)

        def edge_attrs(self):
            return FakeRows(self.edges)

    class FakeAttr:
        def __init__(self, name):
            self.name = name

        def __rmul__(self, factor):
            return factor, self.name

    class FakeSolver:
        last_graph = None
        last_options = None

        def __init__(self, **options):
            self.__class__.last_options = options
            self.num_threads = options["num_threads"]

        def solve(self, graph):
            self.__class__.last_graph = graph
            self.last_status = status_holder["status"]
            self.last_objective = -1.0
            self.last_incumbent_audit = (
                {"checked_constraints": 3} if self.last_status == timelimit_status else None
            )
            return graph

    class FakeLog:
        def addHandler(self, _handler):
            pass

        def removeHandler(self, _handler):
            pass

    class FakeWarnings:
        def __init__(self):
            self.messages = []

    optimal_status = SimpleNamespace(name="OPTIMAL")
    timelimit_status = SimpleNamespace(name="TIMELIMIT")
    status_holder = {"status": optimal_status}
    namespace = {
        "np": np,
        "CONFIG": {
            "model": {"decoding": {"num_threads": 1}},
            "runtime": {"evaluate_ilp_time_limit_seconds_per_video": 3600},
            "validation": {"outer_accept_checked_incumbent_on_timelimit": True},
        },
        "OrderedFlowILPSolver": FakeSolver,
        "_SolverStatus": SimpleNamespace(OPTIMAL=optimal_status, TIMELIMIT=timelimit_status),
        "td": SimpleNamespace(
            graph=SimpleNamespace(InMemoryGraph=FakeGraph),
            solvers=SimpleNamespace(ILPSolver=FakeSolver),
            EdgeAttr=FakeAttr,
        ),
        "_audit_pl": SimpleNamespace(Float64=float),
        "LOG": FakeLog(),
        "SolverWarnings": FakeWarnings,
        "time": time,
    }
    solve = extracted_function(SOURCES[0], "legacy_solve", namespace)
    coords = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=float)
    edges = np.array([[1, 2, 0.8, 1], [0, 2, 0.7, 1]], dtype=float)
    before = edges.copy()
    _nodes, selected, info = solve(coords, edges, (0, 2, 1.2), "sample", 0.4)
    assert np.array_equal(edges, before)
    assert [(e["source_id"], e["target_id"]) for e in FakeSolver.last_graph.edges] == [
        (0, 2),
        (1, 2),
    ]
    assert [e["edge_prob"] for e in FakeSolver.last_graph.edges] == [0.7, 0.8]
    assert [e["ilp_score"] for e in FakeSolver.last_graph.edges] == [0.4, 0.4]
    assert [e["edge_prob"] for e in selected] == [0.7, 0.8]
    assert FakeSolver.last_options["edge_weight"] == (-1.0, "ilp_score")
    assert FakeSolver.last_options["num_threads"] == 1
    assert info["solver_status"] == "OPTIMAL"
    assert info["objective_value"] == -1.0
    assert info["clipped_edges"] == 2
    solve(coords, edges, (0, 2, 1.2), "sample", 1.0)
    assert [e["ilp_score"] for e in FakeSolver.last_graph.edges] == [0.7, 0.8]
    evaluation_solve = extracted_function(SOURCES[-1], "legacy_solve", namespace)
    evaluation_solve(coords, edges, (0, 2, 1.2), "sample", 1.0)
    assert FakeSolver.last_options["timeout"] == 3600
    status_holder["status"] = timelimit_status
    _nodes, _selected, fallback_info = evaluation_solve(coords, edges, (0, 2, 1.2), "sample", 0.4)
    assert fallback_info["solver_status"] == "TIMELIMIT"
    assert fallback_info["is_proven_optimal"] is False
    assert fallback_info["incumbent_audit"] == {"checked_constraints": 3}


def test_time_limited_incumbent_requires_binary_feasible_solution() -> None:
    relations = SimpleNamespace(Equal="equal", LessEqual="less", GreaterEqual="greater")
    namespace = {"np": np, "_Relation": relations}
    audit = extracted_function(SOURCES[-1], "audit_feasible_incumbent", namespace)

    class Constraint:
        def __init__(self, coefficients, relation, rhs):
            self.coefficients, self.relation, self.rhs = coefficients, relation, rhs

        def get_coefficients(self):
            return self.coefficients

        def get_relation(self):
            return self.relation

        def get_value(self):
            return self.rhs

    constraints = [
        Constraint({0: 1, 1: 1}, relations.Equal, 1),
        Constraint({1: 1, 2: 1}, relations.LessEqual, 1),
        Constraint({0: 1, 2: 1}, relations.GreaterEqual, 2),
    ]
    objective = SimpleNamespace(
        get_coefficients=lambda: [-1, 2, -2],
        get_constant=lambda: 0,
    )
    solution = SimpleNamespace(variable_values=[1, 0, 1], objective_value=-3)
    report = audit(solution, constraints, objective, 3)
    assert report["checked_constraints"] == 3
    assert report["max_constraint_error"] == 0
    assert report["objective_error"] == 0

    solution.variable_values = [1, 0, 0]
    with pytest.raises(RuntimeError, match="violates ILP constraints"):
        audit(solution, constraints, objective, 3)
    solution.variable_values = [1, 0.1, 1]
    with pytest.raises(RuntimeError, match="binary variable bounds"):
        audit(solution, constraints, objective, 3)
    solution.variable_values = [0, 0, 0]
    with pytest.raises(RuntimeError, match="no nontrivial incumbent"):
        audit(solution, constraints, objective, 3)
    solution.variable_values = [1, 0, 1]
    solution.objective_value = -2
    with pytest.raises(RuntimeError, match="objective mismatch"):
        audit(solution, constraints, objective, 3)


def test_expanded_candidates_use_raw_probabilities_before_clipping() -> None:
    select = extracted_function(
        SOURCES[0], "select_expanded_edges", {"np": np, "defaultdict": defaultdict}
    )
    coords = np.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0], [1, 0, 0, 0]], dtype=float
    )
    src = np.array([0, 1, 2, 3])
    tgt = np.array([4, 4, 4, 4])
    score = np.array([0.9, 0.6, 0.3, 0.11])
    admitted = np.array([[0, 4, 0.9, 0], [1, 4, 0.6, 1]], dtype=float)
    selected = select(coords, src, tgt, score, admitted)
    assert [(int(s), int(t)) for s, t in selected[:, :2]] == [(0, 4), (1, 4), (2, 4)]
    assert selected[:, 2].tolist() == [0.9, 0.6, 0.3]


def test_capture_shards_are_verified_and_only_selected_caches_are_copied(tmp_path: Path) -> None:
    import json
    import shutil

    blocks = []
    for path in SOURCES:
        source = path.read_text()
        start = source.index("_capture_receipts = sorted(")
        end = source.index("\n# %% [markdown]\n# ## 6.", start)
        blocks.append(source[start:end].replace('Path("/kaggle/input")', "Path(INPUT_ROOT)"))
    assert len(set(blocks)) == 1
    input_root = tmp_path / "input"
    input_root.mkdir()
    video_stems = {f"video_{i}" for i in range(5)}
    for shard in range(5):
        folder = input_root / f"capture_{shard}"
        cache = folder / "cache"
        cache.mkdir(parents=True)
        stem = f"video_{shard}"
        content = f"fixed x138 score {shard}".encode()
        (cache / f"{stem}.npz").write_bytes(content)
        receipt = {
            "status": "capture_finished",
            "shard": shard,
            "shards": 5,
            "selection_sha256": "selection",
            "config_sha256": "upstream",
            "videos": {stem: hashlib.sha256(content).hexdigest()},
        }
        (folder / "capture_receipt.json").write_text(json.dumps(receipt))
    namespace = {
        "Path": Path,
        "INPUT_ROOT": input_root,
        "EXP051_ROOT": tmp_path / "experiment",
        "_selection_sha": "selection",
        "UPSTREAM_CONFIG_SHA256": "upstream",
        "_eligible_stems": video_stems,
        "test_stems": ["video_1", "video_3"],
        "hashlib": hashlib,
        "json": json,
        "shutil": shutil,
    }
    namespace["EXP051_ROOT"].mkdir()
    exec(compile(blocks[0], "<capture verification>", "exec"), namespace)
    assert {p.stem for p in namespace["CACHE_ROOT"].glob("*.npz")} == {"video_1", "video_3"}
    assert len(namespace["_capture_receipts_sha256"]) == 64
    (input_root / "capture_1" / "cache" / "video_1.npz").write_bytes(b"tampered")
    try:
        exec(compile(blocks[0], "<capture verification>", "exec"), namespace)
    except RuntimeError as exc:
        assert "capture cache SHA mismatch" in str(exc)
    else:
        raise AssertionError("tampered selected cache was accepted")


def test_four_arm_receipts_are_required_and_share_fold_inputs(tmp_path: Path) -> None:
    arms = ("expanded_optuna_cost", "expanded_optuna_cost_clipped")
    namespace = {
        "Path": Path,
        "json": json,
        "hashlib": hashlib,
        "np": np,
        "TUNING_CONFIG_SHA256": "current-config",
        "CONFIG": {
            "validation": {"arms": list(arms)},
            "optuna": {"min_valid_trials_per_fold_arm": 4, "max_trials_per_fold_arm": 12},
        },
    }
    loader = extracted_function(SOURCES[-1], "load_tune_receipts", namespace)
    paths = []
    for fold in (0, 1):
        for arm in arms:
            folder = tmp_path / f"fold_{fold}_{arm}"
            folder.mkdir()
            path = folder / "tune_receipt.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "tuning_finished",
                        "fold": fold,
                        "fit_embryo": "44b6" if fold == 0 else "6bba",
                        "inner_stems": [f"video_{fold}"],
                        "selection_sha256": "selection",
                        "config_sha256": "current-config",
                        "upstream_capture_receipts_sha256": "capture",
                        "active_arm": arm,
                        "pilot": {"raw_edges_sha256": f"candidate_{fold}"},
                        "arms": {
                            arm: {
                                "trials": [
                                    {"trial": i, "status": "valid", "summary": {"score": 0.5}}
                                    for i in range(4)
                                ],
                                "selected_trial": 0,
                            }
                        },
                    }
                )
            )
            paths.append(path)
    receipts, shas = loader(tmp_path, "selection", "capture")
    assert all(set(receipts[fold]) == set(arms) for fold in (0, 1))
    assert len(shas) == 4
    broken = json.loads(paths[0].read_text())
    broken["arms"][arms[0]]["trials"][0]["summary"]["score"] = float("nan")
    paths[0].write_text(json.dumps(broken))
    with pytest.raises(RuntimeError, match="non-finite trial score"):
        loader(tmp_path, "selection", "capture")
    broken["arms"][arms[0]]["trials"][0]["summary"]["score"] = 0.5
    paths[0].write_text(json.dumps(broken))
    broken = json.loads(paths[0].read_text())
    broken["arms"][arms[0]]["trials"] = broken["arms"][arms[0]]["trials"][:3]
    paths[0].write_text(json.dumps(broken))
    with pytest.raises(RuntimeError, match="trial count invalid"):
        loader(tmp_path, "selection", "capture")
    paths[0].unlink()
    with pytest.raises(RuntimeError, match="missing tuning arms"):
        loader(tmp_path, "selection", "capture")


def test_official_evaluator_must_return_finite_metrics_for_every_video(tmp_path: Path) -> None:
    for stem in ("video_a", "video_b"):
        (tmp_path / f"{stem}.geff").touch()
    rows = [
        {"dataset": "video_a", "edge_tp": 1, "adj_edge_jaccard": 0.7},
        {"dataset": "video_b", "edge_tp": 1, "adj_edge_jaccard": 0.8},
    ]
    evaluator = SimpleNamespace(DATA_DIR=tmp_path, evaluate_run=lambda *_args, **_kwargs: rows)
    namespace = {
        "official_evaluator": evaluator,
        "_audit_summarise": lambda _rows: {"score": 0.75},
        "np": np,
    }
    evaluate_folder = extracted_function(SOURCES[0], "evaluate_folder", namespace)
    score, returned = evaluate_folder(tmp_path, ["video_a", "video_b"])
    assert score["score"] == 0.75 and returned is rows
    rows[1]["edge_tp"] = float("nan")
    with pytest.raises(RuntimeError, match="official evaluator failed"):
        evaluate_folder(tmp_path, ["video_a", "video_b"])
