"""Inventory training, split, metric, and schema checks in archived public notebooks."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = (
    REPOSITORY_ROOT
    / "docs"
    / "notebooks"
    / "biohub-cell-tracking-during-development"
)
OUTPUT_PATH = Path(__file__).with_name("public_notebook_code_inventory.json")

CODE_PATTERNS = {
    "training_loop": re.compile(
        r"\.backward\s*\(|optimizer\s*=|model\.train\s*\(|for\s+epoch\b|train_loader",
        re.IGNORECASE,
    ),
    "fit_call": re.compile(r"\.fit\s*\(", re.IGNORECASE),
    "fold_api": re.compile(
        r"\b(?:GroupKFold|StratifiedKFold|KFold|LeaveOneGroupOut|cross_val_score)\b",
        re.IGNORECASE,
    ),
    "fixed_cv_datasets": re.compile(
        r"CV_FIXED_DATASETS|fixed[-_ ]?8|resolve_cv_stems", re.IGNORECASE
    ),
    "split_zero_checkpoint": re.compile(r"split_0", re.IGNORECASE),
    "local_metric": re.compile(
        r"official_spec_evaluate|edge_jaccard|division_jaccard|adjusted.?edge",
        re.IGNORECASE,
    ),
    "submission_schema_check": re.compile(
        r"schema validation|submission guard|unexpected submission columns|CSV_COLUMNS",
        re.IGNORECASE,
    ),
}


def cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def evidence_lines(text: str, pattern: re.Pattern[str], limit: int = 8) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines():
        if pattern.search(line):
            compact = line.strip()
            if compact and compact not in lines:
                lines.append(compact[:500])
        if len(lines) >= limit:
            break
    return lines


def listing_metadata() -> dict[str, dict[str, str]]:
    listing_path = NOTEBOOK_ROOT / "kernel_listing.csv"
    with listing_path.open(newline="", encoding="utf-8") as file:
        return {row["ref"]: row for row in csv.DictReader(file)}


def main() -> None:
    listing = listing_metadata()
    records: list[dict[str, Any]] = []
    for notebook_path in sorted(NOTEBOOK_ROOT.glob("*/*.ipynb")):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            cell_source(cell)
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        markdown = "\n".join(
            cell_source(cell)
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "markdown"
        )
        metadata_path = notebook_path.with_name("kernel-metadata.json")
        kernel_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        kernel_ref = str(kernel_metadata.get("id", ""))
        pattern_evidence = {
            name: evidence_lines(code, pattern)
            for name, pattern in CODE_PATTERNS.items()
        }
        records.append(
            {
                "kernel_ref": kernel_ref,
                "title": listing.get(kernel_ref, {}).get(
                    "title", str(kernel_metadata.get("title", ""))
                ),
                "total_votes": int(listing.get(kernel_ref, {}).get("totalVotes", 0)),
                "last_run_time": listing.get(kernel_ref, {}).get("lastRunTime"),
                "notebook_path": str(notebook_path.relative_to(REPOSITORY_ROOT)),
                "code_cell_count": sum(
                    cell.get("cell_type") == "code"
                    for cell in notebook.get("cells", [])
                ),
                "markdown_mentions_training": bool(
                    re.search(r"\btrain(?:ing)?\b", markdown, re.IGNORECASE)
                ),
                "evidence": pattern_evidence,
                "flags": {
                    name: bool(lines) for name, lines in pattern_evidence.items()
                },
            }
        )

    flag_counts = {
        name: sum(record["flags"][name] for record in records)
        for name in CODE_PATTERNS
    }
    output = {
        "competition_slug": "biohub-cell-tracking-during-development",
        "notebook_count": len(records),
        "scope": "top public notebooks sorted by vote count and archived on 2026-08-14",
        "flag_counts": flag_counts,
        "records": records,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
