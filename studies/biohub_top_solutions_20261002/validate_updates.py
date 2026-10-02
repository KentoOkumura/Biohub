"""Check this survey update's saved sources, local links, ranks and figures."""

from __future__ import annotations

import json
import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_markdown_links import LINK_PATTERN, normalized_target

REPORT = ROOT / "docs/surveys/biohub-final-top-solutions-comparison_20260930.md"
RANKS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 16, 17, 18]


def main():
    old = json.loads((ROOT / "studies/biohub_top_solutions_20260930/archive_inventory.json").read_text())
    new = json.loads((OUT / "archive_inventory.json").read_text())
    assert len(old) == 10 and len(new) == 22
    files = [REPORT, ROOT / "docs/surveys/README.md"] + [ROOT / row["archive"] for row in old + new]
    failures = []
    for path in files:
        for match in LINK_PATTERN.finditer(path.read_text()):
            target = normalized_target(match.group(1))
            if target and not (path.parent / target).resolve().exists():
                failures.append(f"{path.relative_to(ROOT)}: {target}")
    assert not failures, "\n".join(failures)

    report = REPORT.read_text()
    chapters = [int(match) for match in re.findall(r"^## (\d+)位:", report, re.M)]
    assert chapters == RANKS, chapters
    assert report.count("### モデルアーキテクチャ") == len(RANKS)
    assert len(re.findall(r"^!\[.*\]\(../images/", report, re.M)) == len(RANKS)
    assert "status: final" in report and "TODO" not in report

    snapshot = json.loads((OUT / "final_leaderboard.json").read_text())
    rows = snapshot["rows"]
    assert len({row["teamId"] for row in rows}) == len(rows)
    assert all(float(a["score"]) >= float(b["score"]) for a, b in zip(rows, rows[1:]))
    table = report.split("## 取得した上位解法\n", 1)[1].split("## こちらの解法", 1)[0]
    found = re.findall(r"^\| (\d+) \| ([^|]+) \| ([\d.]+) \|", table, re.M)
    assert [int(rank) for rank, _, _ in found] == RANKS
    for rank, name, score in found:
        row = rows[int(rank) - 1]
        assert name.strip() == row["teamName"] and score == row["score"], (rank, name, score)
    own = next((rank, row) for rank, row in enumerate(rows, 1) if row["teamId"] == 16717777)
    assert own[0] == 409 and own[1]["score"] == "0.91963"
    assert "430位／4,020チーム" in report and "409位、0.91963" in report
    old_snapshot = json.loads((ROOT / "studies/biohub_top_solutions_20260930/final_leaderboard.json").read_text())
    assert old_snapshot["rows"][3]["teamId"] == rows[3]["teamId"] == 16494792

    figures = []
    for folder in ["biohub_top_solutions_20260930", "biohub_top_solutions_20261002"]:
        manifest = json.loads((ROOT / "docs/images" / folder / "manifest.json").read_text())
        figures += manifest["figures"] if isinstance(manifest, dict) else manifest
    assert sorted(item["rank"] for item in figures) == RANKS
    for record in figures:
        png = ROOT / record["png"]
        svg = ROOT / record["svg"]
        assert png.exists() and svg.exists()
        header = png.read_bytes()[:24]
        assert header[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", header[16:24])
        assert width >= 2000 and height >= 1400
        ET.parse(svg)
        assert record["text_bounds"] == "passed"
        assert "synthetic" in record["example_data"]
    for row in new:
        raw = json.loads((OUT / f"topic_{row['id']}_full.json").read_text())
        assert raw["id"] == row["id"] and raw["content"] and raw["retrieved_at_utc"]
        assert f"discussion/{row['id']}" in (ROOT / row["archive"]).read_text()
    result = {
        "snapshot_checked_at_utc": snapshot["checked_at_utc"],
        "local_link_documents": len(files), "archived_discussions": len(old) + len(new),
        "main_solution_ranks": RANKS, "model_architecture_tables": len(RANKS),
        "figures_with_png_svg_source_and_bounds": len(figures),
        "leaderboard_unique_rows": len(rows),
        "own_rank": own[0], "own_private": own[1]["score"],
        "figure_visual_review": "performed manually; separate from automated bounds checks",
        "upstream_models_rerun": False,
    }
    (OUT / "validation_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
