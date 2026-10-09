#!/usr/bin/env python3
"""Validate structural requirements of a kaggle-idea-forge portfolio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FAMILIES = {
    "representation",
    "information",
    "candidate_generation",
    "fusion_uncertainty",
    "data_generation",
    "validation",
    "compute_enabler",
    "other",
}
SLOTS = {"safe", "exploration", "orthogonal", "compute_enabler"}
CONFIDENCE = {"A", "B", "C"}
NOVELTY = {"incremental", "role_change", "representation_change"}
ORIGINS = {"task_first", "evidence_inversion", "cross_pollination"}
CARD_FIELDS_V1 = {
    "id",
    "title",
    "mechanism_family",
    "roles",
    "hypothesis",
    "evidence_ids",
    "changed_mechanism",
    "preserved_invariants",
    "nearest_prior_attempt",
    "exact_difference",
    "counterevidence",
    "cheap_test",
    "full_test",
    "kill_criterion",
    "reopen_criterion",
    "coverage_test",
    "selectability_test",
    "hidden_inference_contract",
    "compute_estimate",
    "is_parameter_only",
    "novelty_level",
    "confidence",
}
CARD_FIELDS_V2 = CARD_FIELDS_V1 | {
    "origin_pass",
    "information_sources",
    "input_target_decode",
    "deployment_error_simulated",
}


def nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_legacy(payload: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["root must be a JSON object"]

    version = payload.get("schema_version", "1")
    if version not in {"1", "2"}:
        errors.append(f"unsupported schema_version: {version!r}")

    root_fields = [
        "task_summary",
        "evidence_cutoff",
        "allowed_sources",
        "assumptions",
        "closure_ledger",
        "idea_cards",
        "portfolio",
        "rejected",
    ]
    if version == "2":
        root_fields.append("schema_version")
    for key in root_fields:
        if key not in payload:
            errors.append(f"missing root field: {key}")

    for key in ("task_summary", "evidence_cutoff"):
        if key in payload and not nonempty_string(payload[key]):
            errors.append(f"{key} must be a non-empty string")

    for key in ("allowed_sources", "assumptions", "closure_ledger", "rejected"):
        if key in payload and not isinstance(payload[key], list):
            errors.append(f"{key} must be a list")

    cards = payload.get("idea_cards", [])
    if not isinstance(cards, list):
        return errors + ["idea_cards must be a list"]
    if not 10 <= len(cards) <= 14:
        errors.append(f"idea_cards must contain 10-14 cards, got {len(cards)}")

    ids: list[str] = []
    families: set[str] = set()
    parameter_only = 0
    card_fields = CARD_FIELDS_V2 if version == "2" else CARD_FIELDS_V1
    for index, card in enumerate(cards):
        prefix = f"idea_cards[{index}]"
        if not isinstance(card, dict):
            errors.append(f"{prefix} must be an object")
            continue
        missing = card_fields - set(card)
        if missing:
            errors.append(f"{prefix} missing fields: {', '.join(sorted(missing))}")
        list_fields = {"roles", "evidence_ids", "preserved_invariants"}
        if version == "2":
            list_fields.add("information_sources")
        for field in card_fields - list_fields - {"is_parameter_only"}:
            if field in card and not nonempty_string(card[field]):
                errors.append(f"{prefix}.{field} must be a non-empty string")
        for field in list_fields:
            value = card.get(field)
            if (
                not isinstance(value, list)
                or not value
                or not all(nonempty_string(x) for x in value)
            ):
                errors.append(f"{prefix}.{field} must be a non-empty string list")
        idea_id = card.get("id")
        if isinstance(idea_id, str):
            ids.append(idea_id)
        family = card.get("mechanism_family")
        if family not in FAMILIES:
            errors.append(f"{prefix}.mechanism_family is invalid: {family!r}")
        else:
            families.add(family)
        if card.get("confidence") not in CONFIDENCE:
            errors.append(f"{prefix}.confidence is invalid")
        if card.get("novelty_level") not in NOVELTY:
            errors.append(f"{prefix}.novelty_level is invalid")
        if version == "2" and card.get("origin_pass") not in ORIGINS:
            errors.append(f"{prefix}.origin_pass is invalid")
        if not isinstance(card.get("is_parameter_only"), bool):
            errors.append(f"{prefix}.is_parameter_only must be boolean")
        elif card["is_parameter_only"]:
            parameter_only += 1

    if len(ids) != len(set(ids)):
        errors.append("idea card ids must be unique")
    if len(families) < 4:
        errors.append(f"idea_cards must cover at least 4 families, got {len(families)}")
    if parameter_only > 2:
        errors.append(f"at most 2 parameter-only cards are allowed, got {parameter_only}")

    portfolio = payload.get("portfolio", [])
    if not isinstance(portfolio, list):
        errors.append("portfolio must be a list")
        portfolio = []
    if len(portfolio) != 5:
        errors.append(f"portfolio must contain exactly 5 entries, got {len(portfolio)}")

    card_by_id = {card.get("id"): card for card in cards if isinstance(card, dict)}
    portfolio_ids: list[str] = []
    portfolio_families: set[str] = set()
    portfolio_parameter_only = 0
    for index, entry in enumerate(portfolio):
        prefix = f"portfolio[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix} must be an object")
            continue
        idea_id = entry.get("idea_id")
        portfolio_ids.append(idea_id)
        if idea_id not in card_by_id:
            errors.append(f"{prefix}.idea_id not found in cards: {idea_id!r}")
        else:
            card = card_by_id[idea_id]
            portfolio_families.add(card.get("mechanism_family"))
            portfolio_parameter_only += int(card.get("is_parameter_only") is True)
        if entry.get("slot") not in SLOTS:
            errors.append(f"{prefix}.slot is invalid")
        if not nonempty_string(entry.get("why")):
            errors.append(f"{prefix}.why must be a non-empty string")

    if len(portfolio_ids) != len(set(portfolio_ids)):
        errors.append("portfolio idea ids must be unique")
    if len(portfolio_families) < 3:
        errors.append(f"portfolio must cover at least 3 families, got {len(portfolio_families)}")
    if portfolio_parameter_only > 1:
        errors.append("portfolio may contain at most 1 parameter-only idea")

    if version == "2":
        portfolio_cards = [card_by_id[i] for i in portfolio_ids if i in card_by_id]
        portfolio_family_set = {card.get("mechanism_family") for card in portfolio_cards}
        required_families = {"representation", "information", "data_generation"}
        missing_families = required_families - portfolio_family_set
        if missing_families:
            errors.append(
                "portfolio missing required families: " + ", ".join(sorted(missing_families))
            )
        if not portfolio_family_set & {"candidate_generation", "fusion_uncertainty"}:
            errors.append("portfolio needs candidate_generation or fusion_uncertainty")
        if not portfolio_family_set & {"validation", "compute_enabler"}:
            errors.append("portfolio needs validation or compute_enabler")
        if not any(card.get("origin_pass") == "task_first" for card in portfolio_cards):
            errors.append("portfolio needs at least one task_first idea")
        if not any(
            card.get("novelty_level") == "representation_change" for card in portfolio_cards
        ):
            errors.append("portfolio needs at least one representation_change idea")

    return errors


V3_CARD_FIELDS = (
    "id", "title", "hypothesis", "basis", "changed_mechanism", "key_unknown",
)
V3_EXPERIMENT_FIELDS = (
    "input_target_output_loss_decode", "comparison", "preserved_mechanism",
    "evaluation_limitations", "decision_rules", "inference_contract",
)


def require_strings(record: dict, fields: tuple[str, ...], prefix: str) -> list[str]:
    return [
        f"{prefix}.{field} must be a non-empty string"
        for field in fields if not nonempty_string(record.get(field))
    ]


def string_list(value: object) -> bool:
    return isinstance(value, list) and all(nonempty_string(item) for item in value)


def validate_v3(payload: dict) -> list[str]:
    errors = require_strings(
        payload,
        ("task_summary", "evidence_cutoff", "data_exploration", "selection_notes"),
        "root",
    )
    if not string_list(payload.get("allowed_sources")):
        errors.append("allowed_sources must be a string list")
    if "assumptions" in payload and not string_list(payload["assumptions"]):
        errors.append("assumptions must be a string list")
    for field in ("closure_ledger", "rejected"):
        if field in payload and (
            not isinstance(payload[field], list)
            or not all(isinstance(item, dict) for item in payload[field])
        ):
            errors.append(f"{field} must be an object list")

    cards = payload.get("idea_cards")
    if not isinstance(cards, list) or not cards:
        errors.append("idea_cards must be a non-empty list")
        cards = []
    ids: set[str] = set()
    for index, card in enumerate(cards):
        prefix = f"idea_cards[{index}]"
        if not isinstance(card, dict):
            errors.append(f"{prefix} must be an object")
            continue
        errors.extend(require_strings(card, V3_CARD_FIELDS, prefix))
        idea_id = card.get("id")
        if nonempty_string(idea_id):
            if idea_id in ids:
                errors.append("idea card ids must be unique")
            ids.add(idea_id)

    portfolio = payload.get("portfolio")
    if not isinstance(portfolio, list):
        errors.append("portfolio must be a list")
        portfolio = []
    selected: set[str] = set()
    for index, entry in enumerate(portfolio):
        prefix = f"portfolio[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix} must be an object")
            continue
        errors.extend(require_strings(entry, ("idea_id", "why"), prefix))
        idea_id = entry.get("idea_id")
        if nonempty_string(idea_id):
            if idea_id not in ids:
                errors.append(f"{prefix}.idea_id not found in cards: {idea_id!r}")
            if idea_id in selected:
                errors.append("portfolio idea ids must be unique")
            selected.add(idea_id)
        experiment = entry.get("experiment")
        if not isinstance(experiment, dict):
            errors.append(f"{prefix}.experiment must be an object")
            continue
        exp_prefix = f"{prefix}.experiment"
        errors.extend(require_strings(experiment, V3_EXPERIMENT_FIELDS, exp_prefix))
        if "resource_limits" in experiment:
            errors.extend(require_strings(experiment, ("resource_limits",), exp_prefix))
        if "dependencies" in experiment and not string_list(experiment["dependencies"]):
            errors.append(f"{exp_prefix}.dependencies must be a string list")
        diagnostics = experiment.get("diagnostics", [])
        if not isinstance(diagnostics, list):
            errors.append(f"{exp_prefix}.diagnostics must be a list")
            continue
        for diag_index, diagnostic in enumerate(diagnostics):
            diag_prefix = f"{exp_prefix}.diagnostics[{diag_index}]"
            if not isinstance(diagnostic, dict):
                errors.append(f"{diag_prefix} must be an object")
                continue
            errors.extend(require_strings(
                diagnostic,
                ("check", "decision_if_pass", "decision_if_fail", "cannot_refute"),
                diag_prefix,
            ))
    return errors


def validate(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return ["root must be a JSON object"]
    version = payload.get("schema_version", "1")
    if version == "3":
        return validate_v3(payload)
    if isinstance(version, str) and version in {"1", "2"}:
        return validate_legacy(payload)
    return [f"unsupported schema_version: {version!r}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("portfolio", type=Path)
    args = parser.parse_args()
    try:
        payload = json.loads(args.portfolio.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}")
        return 1

    errors = validate(payload)
    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "PASS: "
        f"schema v{payload.get('schema_version', '1')}, "
        f"{len(payload['idea_cards'])} cards, "
        f"{len(payload['portfolio'])} portfolio entries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
