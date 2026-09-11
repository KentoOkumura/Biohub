from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
NOTEBOOK_SOURCE = EXP / "exp003_official_metric_audit_audit.py"


def definitions(source):
    tree = ast.parse(source)
    return {
        node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


class NormalizeDocumentation(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.level == 1 and node.module == "division_metrics":
            return None
        return node


def semantic_ast(node):
    return ast.dump(NormalizeDocumentation().visit(node), include_attributes=False)


def test_complete_official_definitions_preserved():
    actual = definitions(NOTEBOOK_SOURCE.read_text())
    for name in ["metrics.py", "division_metrics.py"]:
        expected = definitions((EXP / "assets" / name).read_text())
        assert expected
        for function_name, definition in expected.items():
            assert function_name in actual, function_name
            assert semantic_ast(definition) == semantic_ast(actual[function_name]), function_name


def test_public_proxy_not_silently_fixed():
    actual = definitions(NOTEBOOK_SOURCE.read_text())
    expected = definitions((EXP / "assets/public_proxy.py").read_text())
    assert len(expected) == 6
    for name, definition in expected.items():
        assert semantic_ast(definition) == semantic_ast(actual[name]), name


def test_source_snapshot_content():
    manifest = json.loads((EXP / "assets/source_manifest.json").read_text())
    assert manifest["commit"] == "075fc5f5a52d11077f9dc2b074644618f26939e2"
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((EXP / "assets" / name).read_bytes()).hexdigest() == expected


def test_fixture_references_and_required_failure_modes():
    fixtures = json.loads((EXP / "assets/fixtures.json").read_text())
    assert len({f["name"] for f in fixtures}) == len(fixtures) == 9
    required = {
        "correct_division",
        "distant_division",
        "duplicate_edge",
        "nonconsecutive_edge",
        "merged_daughter",
        "three_children",
        "no_edges",
        "unannotated_component",
        "no_divisions",
    }
    assert {f["name"] for f in fixtures} == required
    for fixture in fixtures:
        for side in ["pred", "gt"]:
            nodes = {int(key) for key in fixture[f"{side}_nodes"]}
            assert nodes
            assert all(len(value) == 4 for value in fixture[f"{side}_nodes"].values())
            assert all(a in nodes and b in nodes for a, b in fixture[f"{side}_edges"])


def test_cpu_only_fixed_prediction_audit_contract():
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    assert config["lineage"]["backlog_candidate"] == "official_metric_check"
    assert config["lineage"]["hypothesis_id"] == "HYP-20260910-14"
    assert config["runtime"]["kaggle"]["enable_gpu"] is False
    assert config["runtime"]["kaggle"]["audit"]["enable_gpu"] is False
    assert config["runtime"]["kaggle"]["enable_internet"] is False
    assert config["validation"]["diagnostic_only"] is True
    assert config["model"]["name"] == "none_metric_audit"
    assert config["audit"]["max_distance_um"] == 7.0
    assert config["audit"]["prediction_sha256"] == (
        "b039062114962347aca22f1268a64196893eaabb62c1046ae0c90acbad29f223"
    )
    assert len(config["audit"]["sample_ids"]) == 4
    assert "__file__" not in NOTEBOOK_SOURCE.read_text()
