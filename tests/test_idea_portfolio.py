from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".agents/skills/kaggle-idea-forge"
SCRIPT = SKILL / "scripts/validate_portfolio.py"
spec = importlib.util.spec_from_file_location("validate_idea_portfolio", SCRIPT)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


@pytest.fixture
def portfolio() -> dict:
    # The documented example is also a usable minimal record.
    schema = (SKILL / "references/portfolio-schema.md").read_text()
    return json.loads(schema.split("```json\n", 1)[1].split("```", 1)[0])


def test_selected_idea_and_lightweight_alternative_need_no_quotas_or_estimates(portfolio):
    alternative = copy.deepcopy(portfolio["idea_cards"][0])
    alternative["id"] = "I02"
    alternative["title"] = "Alternative supervision"
    portfolio["idea_cards"].append(alternative)
    assert validator.validate(portfolio) == []
    assert "experiment" not in alternative
    assert "compute_estimate" not in json.dumps(portfolio)


@pytest.mark.parametrize("count", [1, 3, 16])
def test_card_and_selection_counts_are_not_fixed(portfolio, count):
    card = portfolio["idea_cards"][0]
    selection = portfolio["portfolio"][0]
    portfolio["idea_cards"] = [dict(card, id=f"I{i}") for i in range(count)]
    portfolio["portfolio"] = [dict(selection, idea_id=f"I{i}") for i in range(count)]
    assert validator.validate(portfolio) == []


def test_unselected_hypotheses_can_be_saved_without_experiment_contracts(portfolio):
    portfolio["portfolio"] = []
    portfolio["allowed_sources"] = []
    portfolio["selection_notes"] = "Validation split is undecided; retain hypotheses."
    portfolio["data_exploration"] = "No raw data was provided; assumptions only."
    assert validator.validate(portfolio) == []


@pytest.mark.parametrize(
    "field",
    ["comparison", "preserved_mechanism", "evaluation_limitations", "decision_rules"],
)
def test_selected_experiment_must_explain_what_the_comparison_tests(portfolio, field):
    del portfolio["portfolio"][0]["experiment"][field]
    assert any(field in error for error in validator.validate(portfolio))


def test_diagnostic_requires_decision_and_scope_but_diagnostics_are_optional(portfolio):
    experiment = portfolio["portfolio"][0]["experiment"]
    experiment["diagnostics"] = []
    experiment["dependencies"] = []
    experiment["resource_limits"] = "One GPU session available; duration unmeasured."
    assert validator.validate(portfolio) == []
    diagnostic = {
        "check": "Check that input IDs align with labels.",
        "decision_if_pass": "Proceed to the planned comparison.",
        "decision_if_fail": "Repair the join before training.",
        "cannot_refute": "This does not measure the proposed representation's accuracy.",
    }
    experiment["diagnostics"] = [diagnostic]
    assert validator.validate(portfolio) == []
    del diagnostic["cannot_refute"]
    assert any("cannot_refute" in error for error in validator.validate(portfolio))


@pytest.mark.parametrize("location", ["idea_cards", "portfolio"])
def test_duplicate_ids_are_rejected(portfolio, location):
    portfolio[location].append(copy.deepcopy(portfolio[location][0]))
    assert any("unique" in error for error in validator.validate(portfolio))


def test_unknown_selected_id_is_rejected(portfolio):
    portfolio["portfolio"][0]["idea_id"] = "missing"
    assert any("not found" in error for error in validator.validate(portfolio))


@pytest.mark.parametrize("value", [None, [], {}, " ", 123])
def test_invalid_ids_report_validation_errors_without_crashing(portfolio, value):
    portfolio["idea_cards"][0]["id"] = value
    portfolio["portfolio"][0]["idea_id"] = value
    assert validator.validate(portfolio)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("idea_cards",), []),
        (("idea_cards",), {}),
        (("idea_cards", 0), None),
        (("portfolio",), None),
        (("portfolio", 0), []),
        (("portfolio", 0, "experiment"), None),
        (("portfolio", 0, "experiment", "diagnostics"), {}),
        (("portfolio", 0, "experiment", "diagnostics"), [None]),
        (("portfolio", 0, "experiment", "dependencies"), [""]),
        (("data_exploration",), ""),
        (("selection_notes",), ""),
        (("allowed_sources",), [123]),
        (("assumptions",), {}),
        (("rejected",), ["invalid"]),
        (("schema_version",), []),
    ],
)
def test_malformed_records_return_errors(portfolio, path, value):
    record = portfolio
    for key in path[:-1]:
        record = record[key]
    record[path[-1]] = value
    assert validator.validate(portfolio)


def legacy_portfolio(version: str) -> dict:
    # Frozen field names rather than current validator constants protect old records.
    fields = (
        "id title hypothesis changed_mechanism nearest_prior_attempt exact_difference "
        "counterevidence cheap_test full_test kill_criterion reopen_criterion coverage_test "
        "selectability_test hidden_inference_contract compute_estimate"
    ).split()
    families = [
        "representation", "information", "data_generation", "fusion_uncertainty", "validation",
    ]
    cards = []
    for index in range(10):
        card = dict.fromkeys(fields, "Recorded legacy detail")
        card.update(
            id=f"I{index}", mechanism_family=families[index % 5],
            roles=["target"], evidence_ids=["historical evidence"],
            preserved_invariants=["group identity"], is_parameter_only=False,
            novelty_level="representation_change", confidence="B",
        )
        if version == "2":
            card.update(
                origin_pass="task_first", information_sources=["historical input"],
                input_target_decode="Recorded mechanism",
                deployment_error_simulated="not applicable: no intermediate model",
            )
        cards.append(card)
    payload = {
        "task_summary": "Legacy task", "evidence_cutoff": "historical cutoff",
        "allowed_sources": ["historical packet"], "assumptions": [],
        "closure_ledger": [], "rejected": [], "idea_cards": cards,
        "portfolio": [
            {"idea_id": f"I{i}", "slot": "exploration", "why": "Recorded selection"}
            for i in range(5)
        ],
    }
    if version != "implicit":
        payload["schema_version"] = version
    return payload


@pytest.mark.parametrize("version", ["implicit", "1", "2"])
def test_legacy_records_keep_their_original_validation_rules(version):
    payload = legacy_portfolio(version)
    original = copy.deepcopy(payload)
    assert validator.validate(payload) == []
    assert payload == original
    payload["portfolio"].pop()
    assert any("exactly 5" in error for error in validator.validate(payload))


def test_legacy_v2_still_checks_required_categories():
    payload = legacy_portfolio("2")
    payload["idea_cards"][2]["mechanism_family"] = "other"
    assert any("data_generation" in error for error in validator.validate(payload))


def test_cli_validates_new_format_without_family_fields(portfolio, tmp_path):
    path = tmp_path / "portfolio.json"
    path.write_text(json.dumps(portfolio))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "schema v3, 1 cards, 1 portfolio entries" in result.stdout
    path.write_text("{")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "FAIL" in result.stdout
