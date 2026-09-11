"""Apply a reviewed 64-candidate audit without changing experiment records."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPORT = ROOT / "docs/surveys/biohub-backlog-readiness_20260910.md"
REPORT_REL = "../docs/surveys/biohub-backlog-readiness_20260910.md"


def replace_section(text, title, body):
    pattern = rf"(?ms)^## {re.escape(title)}\n.*?(?=^## |\Z)"
    result, count = re.subn(pattern, f"## {title}\n\n{body.strip()}\n\n", text)
    assert count == 1, (title, count)
    return result


def main():
    findings = json.loads((HERE / "candidate_findings.json").read_text())
    specs = json.loads((ROOT / "studies/biohub_accuracy_ideas_20260910/backlog_specs.json").read_text())
    original = {r[0]: (g, r) for g in specs["groups"] for r in g["rows"]}
    assert len(findings["rows"]) == len(original) == 64
    audit_record = []
    for slug, fact, proposal, measurement, conditional in findings["rows"]:
        path = ROOT / "backlog" / f"{slug}.md"
        before = path.read_text()
        if "## 2026-09-10の未決事項調査" in before:
            raise ValueError(f"already audited: {path}")
        group, spec = original[slug]
        user = "D1（実験対象とこの比較案）、D2（予算）"
        if slug not in {"official_metric_check", "group_error_readout", "oracle_stage_limits", "exact_window_cache"}:
            user += "、D3（改善の判定方針）"
        if conditional:
            user += "、" + "・".join(conditional.split(",")) + "（この案を選ぶ場合のみ）"
        detail = (
            f"## 2026-09-10の未決事項調査\n\n"
            f"- 確認できたこと: {fact}\n"
            f"- 最初の比較案（提案・未承認）: {proposal}\n"
            f"- 実データ・生成物で測ること: {measurement}\n"
            f"- ユーザーが決めること: {user}。各項目の選択肢と推奨は[判断一覧]({REPORT_REL}#ユーザーに決めてもらう事項)を参照する。\n"
            f"- 調査根拠: [確認済みの事実と検証範囲]({REPORT_REL}#確認済みの事実)と[本候補の対応表]({REPORT_REL}#{slug})。\n"
            f"- 扱い: 数値の未実測はユーザー判断待ちと区別する。比較案・予算の合意前なので状態は検討メモのままとする。\n\n"
        )
        text = before.replace("## 観測事実と根拠\n", detail + "## 観測事実と根拠\n", 1)
        text = text.replace("比較checkpoint・学習から除いた胚の予測・正式な親実験は未確定。", "構成の参照版は確定。主評価には学習から除いた胚の新しい予測が必要であり、現在の両胚を使う重みを独立予測の代用にはしない。")
        text = text.replace("比較checkpoint・独立予測・候補cacheの実験生成物とSHAは未確定。", "精度比較に使える独立予測・候補cache・重みは未取得。これらのSHAは生成・取得後に記録する作業であり、ユーザーが値を選ぶ事項ではない。人工例と静的コード調査には重みを要しない。")
        text = text.replace("最小改善幅と許容悪化は未決で、設計時に合意する。", "改善の判定方針は調査レポートのD3で選ぶ。測定前に改善幅や誤差幅を推定せず、両方向の差と費用を記録する。")
        text = text.replace("具体的な窓長・領域は未決。", "最初の処理単位は上記の比較案で示し、データに依存する上限は学習側の回収率・時間から設定する。")
        text = text.replace("具体的な基準版と値は未取得のため、この記述だけで実装を開始しない。", "参照sourceと既定値はexp002のconfigで確定している。比較の学習重み・教師の来歴は実験化時に別途固定する。")
        text = text.replace("exp002の参照設定と公式sourceを確認して選ぶ。外部重み、候補cache、独立予測の再利用可否は未確定。", "exp002のconfigと公式sourceの版は照合済み。未知の重み・cache・独立予測を存在すると仮定せず、取得後に学習来歴から再利用可否を確認する。")
        text = text.replace("実測見積、追加GPU予算、最大入力形状は未確定。", "GPU残量の確認値と見積の限界は調査レポートを参照する。追加の利用予算はD2、最大入力での費用は実測事項。")
        text = text.replace("model config数・学習量は未決。", "比較する方法は上記の案に限定し、学習量と実行数はD1・D2で確定する。")
        text = text.replace("本候補自体が出力構造を変える場合の具体的な復号は未決事項を解決して固定する。", "本候補が出力構造を変える場合は上記の比較案を確認して具体的な復号を固定する。")
        text = text.replace("実装区分: 未決定。外部手法を使う場合は教師・構造・省略点を確定後、", "実装区分: 実験化時に比較案の教師・構造・省略点から、")
        text = replace_section(text, "未決事項", (
            f"- ユーザー判断: {user}。[選択肢・推奨案]({REPORT_REL}#ユーザーに決めてもらう事項)。\n"
            f"- 測定・生成物の待ち: {measurement}\n"
            f"- 比較する具体的な案: {proposal}\n"
            "- 既に解消: 公式評価の参照版、胚を分ける主評価、物理単位、提出入力の制約、参照config。重みのSHAは取得後に記録するもので、ユーザー判断ではない。\n"
            "- 実装契約: 選択した比較案を実験化するとき、上記測定から設定値・教師mask・停止条件を記録する。データから決められない設計変更が新たに生じた場合だけ追加確認する。"
        ))
        text = text.replace("## 判断履歴\n\n", "## 判断履歴\n\n- 2026-09-10: 未決事項の先行調査依頼に基づき、確認済みの仕様・コードと測定待ち、ユーザー判断を分離した。上記の比較案は提案であり、実験化や採用の承認は含まない。\n", 1)
        text = replace_section(text, "次セッションへの引き継ぎ確認", (
            "- 固定するもの: 評価方針、物理単位、参照source/configは確認済み。選択した比較の重み・対象一覧は生成時に固定する。\n"
            "- 変更するもの: 上記の『最初の比較案』を候補の具体案とする。ユーザーが選択するまでは提案である。\n"
            "- 最小検証と停止条件: 元の仮説と禁止事項を維持し、調査済み部分を繰り返さず未測定部分を検証する。\n"
            "- 未決事項: ユーザー判断と測定待ちを分けて引き継ぐ。実験化の指示があれば既存の承認範囲を確認し、同じ承認を再要求しない。"
        ))
        if slug == "known_parent_constraint":
            text = text.replace("- 優先度: P3", "- 優先度: P4")
            text = re.sub(r"(?m)^- 優先度の理由:.*$", "- 優先度の理由: 親候補softmaxとの具体的な差が定義されていないため、追加目的の根拠が見つかるまで保留。上位仮説の棄却や実験の不採用ではない。", text)
        if slug == "official_metric_check":
            text = re.sub(r"(?m)^- 親実験 / 比較対象:.*$", "- 親実験 / 比較対象: 公式commit `075fc5f5a52d11077f9dc2b074644618f26939e2`と保存したDCTTA公開proxy。人工例は重み不要。同一の実予測graphを追加照合する場合はその来歴とSHAを記録する。", text)
            text = re.sub(r"(?m)^- 処理単位:.*$", "- 処理単位: 小さな有向graph1件、および保存された1動画の予測graph。新しい画像窓長を選ぶ候補ではない。", text)
        path.write_text(text)
        audit_record.append({"candidate": slug, "original_unresolved": spec[7], "before_sha256": hashlib.sha256(before.encode()).hexdigest(), "after_sha256": hashlib.sha256(text.encode()).hexdigest(), "conditional_decisions": conditional.split(",") if conditional else []})

    direction_path = ROOT / "backlog/KAGGLE_DIRECTION.md"
    direction = direction_path.read_text()
    old = "全件は未決事項を持つ検討メモであり、実験化の承認ではない。"
    new = "[未決事項の先行調査](../docs/surveys/biohub-backlog-readiness_20260910.md)で全64件を確認し、各詳細の仕様確認・測定待ち・ユーザー判断を分けた。状態は比較案と予算の合意前を表す検討メモであり、64件すべてがユーザーの技術判断待ちという意味ではない。"
    assert old in direction
    direction = direction.replace(old, new)
    direction = direction.replace("| P3 | `HYP-20260910-11` | [`known_parent_constraint`]", "| P4 | `HYP-20260910-11` | [`known_parent_constraint`]")
    direction = direction.replace("親の排他性を学習 | partial_edge_mask", "既存softmaxとの差を要確認 | 独立した追加目的の根拠")
    direction = direction.replace("| 評価器と比較graphの版 | `検討メモ・設計不可` |", "| 局所差は検証済み・全評価器と実graph比較が残る | `検討メモ・設計不可` |", 1)
    direction_path.write_text(direction)
    (HERE / "backlog_audit_manifest.json").write_text(json.dumps({"date": "2026-09-10", "candidates": audit_record}, ensure_ascii=False, indent=2) + "\n")

    metadata_text, body = REPORT.read_text().split("---", 2)[1:]
    metadata = yaml.safe_load(metadata_text)
    metadata.update({"status": "final", "hypotheses": [f"HYP-20260910-{n:02}" for n in range(1, 15)], "experiments": ["exp001", "exp002"], "summary": "64候補の仕様・測定待ち・ユーザー判断を分離。公式分裂評価と公開proxyの局所差、未知候補への勾配、checkpoint保存と既存正規化を確認。"})
    body = body.strip() + "\n"
    body += "\n## 全64候補の調査結果と最初の比較案\n\n"
    body += "以下の比較案は提案であり、実験化や方式選択の承認ではない。D1は選ぶ候補と比較内容、D2は費用、D3は改善判定の共通判断。条件付き判断だけを末尾に示す。候補固有の詳細はbacklogを正とする。\n\n"
    for group in specs["groups"]:
        body += f"### {group['id']} / HYP-20260910-{int(group['id'][1:]):02}\n\n"
        for slug, fact, proposal, measurement, conditional in findings["rows"]:
            if original[slug][0]["id"] != group["id"]:
                continue
            body += f"<a id=\"{slug}\"></a>\n\n"
            body += f"**[{slug}](../../backlog/{slug}.md)** — {original[slug][1][1]}\n\n"
            body += f"- 確認結果: {fact}\n- 最初の比較案: {proposal}\n- 測定・資料待ち: {measurement}\n- 条件付きのユーザー判断: {conditional or 'なし（共通判断のみ）'}。\n\n"
    REPORT.write_text("---\n" + yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False).strip() + "\n---\n\n" + body)
    print("updated 64 candidate files, direction index, report, and audit manifest")


if __name__ == "__main__":
    main()
