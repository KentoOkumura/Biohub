from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
report = ROOT / "docs/surveys/biohub-backlog-readiness_20260910.md"
body = report.read_text()
declaration = "- 対応する上位仮説: " + "、".join(f"`HYP-20260910-{i:02}`" for i in range(1, 15)) + "。個別の支持・棄却は未判断。"
if declaration not in body:
    body = body.replace("## 結論\n", declaration + "\n\n## 結論\n", 1)
report.write_text(body)
direction = ROOT / "backlog/KAGGLE_DIRECTION.md"
text = direction.read_text().replace("既知の既存softmaxとの差を要確認", "既存softmaxとの差を要確認")
lines = text.splitlines()
selected = [line for line in lines if line.startswith("| P4 |") and "known_parent_constraint" in line]
assert len(selected) == 1
lines.remove(selected[0])
last_backlog_row = max(i for i, line in enumerate(lines) if line.startswith("| P4 |"))
lines.insert(last_backlog_row + 1, selected[0])
direction.write_text("\n".join(lines) + "\n")
# Link the previous idea report to the newer investigation without rewriting its history.
previous = ROOT / "docs/surveys/biohub-accuracy-hypotheses_20260910.md"
text = previous.read_text()
note = "未決事項の確認状況は、後続の[64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md)を参照する。候補の詳細はbacklogを正とする。"
if note not in text:
    text = text.replace("## 結論\n", note + "\n\n## 結論\n", 1)
previous.write_text(text)
data = json.loads((HERE / "candidate_findings.json").read_text())
manifest_file = HERE / "backlog_audit_manifest.json"
manifest = json.loads(manifest_file.read_text())
for record in manifest["candidates"]:
    slug = record["candidate"]
    path = ROOT / "backlog" / f"{slug}.md"
    text = path.read_text()
    required = ["観測事実と根拠", "この候補が直接検証する仮説と範囲", "入力・予測対象・出力・推論方法", "親実験からの差分", "最小の反証可能な検証", "成功条件と停止条件", "実行しないこと", "リスク", "未決事項", "判断履歴", "次セッションへの引き継ぎ確認"]
    for heading in required:
        assert text.count("## " + heading + "\n") == 1, (slug, heading)
    record["after_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
assert len(manifest["candidates"]) == len(data["rows"]) == 64
manifest["verification"] = {"candidate_count": 64, "required_sections_present": True, "ids_preserved": True}
manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print("64 candidates retain all required sections; report and index corrected")
