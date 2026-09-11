from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / "studies/biohub_accuracy_ideas_20260910"
PORTFOLIO_PATH = AREA / "idea_portfolio.json"
SPEC_PATH = AREA / "backlog_specs.json"
DIRECTION_PATH = ROOT / "backlog/KAGGLE_DIRECTION.md"
REPORT_PATH = ROOT / "docs/surveys/biohub-accuracy-hypotheses_20260910.md"
REPORT_LINK = "../docs/surveys/biohub-accuracy-hypotheses_20260910.md"
PORTFOLIO_LINK = "../studies/biohub_accuracy_ideas_20260910/idea_portfolio.json"
STATE = "検討メモ・設計不可"
DATE = "2026-09-10"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer the reviewed Biohub ideas into individual backlog memos.")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    portfolio = json.loads(PORTFOLIO_PATH.read_text())
    spec = json.loads(SPEC_PATH.read_text())
    cards = {card["id"]: card for card in portfolio["idea_cards"]}
    direction_before = DIRECTION_PATH.read_text()
    report_before = REPORT_PATH.read_text()
    groups = spec["groups"]
    assert len(groups) == 14
    assert sum(len(group["rows"]) for group in groups) == 64
    names = [row[0] for group in groups for row in group["rows"]]
    assert len(set(names)) == len(names)
    assert all(re.fullmatch(r"[a-z0-9_]+", name) for name in names)
    existing_details = [
        path for path in (ROOT / "backlog").glob("*.md")
        if path.name not in {"_TEMPLATE.md", "KAGGLE_DIRECTION.md", "README.md"}
    ]
    if existing_details:
        raise SystemExit("Backlog changed since review; inspect existing details before registering.")
    new_ids = [f"HYP-20260910-{int(group['id'][1:]):02d}" for group in groups]
    if any(hypothesis in direction_before for hypothesis in new_ids):
        raise SystemExit("Hypothesis IDs already exist; do not overwrite.")
    files: dict[Path, str] = {}
    candidates = []
    hypothesis_rows = []
    report_mapping = []
    item_number = 0
    for group in groups:
        card = cards[group["id"]]
        hypothesis_id = f"HYP-20260910-{int(group['id'][1:]):02d}"
        assert len(group["rows"]) == len(card["subhypotheses"])
        links = []
        sibling_names = [row[0] for row in group["rows"]]
        for branch, row in enumerate(group["rows"], start=1):
            item_number += 1
            name, title, priority, input_text, output, loss, cheap, unresolved, dependency = row
            original_idea = card["subhypotheses"][branch - 1]
            assert not dependency or dependency in names
            if dependency:
                dependency_text = f"[`{dependency}`]({dependency}.md)の成立条件。先行候補の実験化や成功を前提にしない。"
            else:
                dependency_text = "保存済みの公式評価コードと比較graph。小さな人工graphでの照合は先に検討できる。"
            priority_reason = {
                "P1": "比較の前提またはコード上の教師定義に直接根拠があり、次に内容を具体化する価値が高い。進行中の実験を中止する指示ではない。",
                "P2": "公式評価と基準予測、または先行候補の診断が成立した後に、重点仮説を直接比較する候補。",
                "P3": "効果を支える実測と先行候補の結果を確認してから扱う後続候補。",
                "P4": "必要な教師・注釈・識別可能な目的が現時点では揃っておらず、下記の再開条件が成立するまで保留する。",
            }[priority]
            evidence_lines = []
            for evidence_id in card["evidence_ids"]:
                evidence = portfolio["evidence_registry"][evidence_id]
                evidence_lines.append(
                    f"  - [{evidence_id}]({REPORT_LINK}#{evidence_id.lower()}): {evidence['fact']}"
                )
            siblings = [
                f"[`{sibling}`]({sibling}.md)"
                for sibling in sibling_names if sibling != name
            ]
            remaining = group["remaining"] + "。別候補 " + "、".join(siblings) + " の検証が残る。"
            is_diagnostic = group["id"] == "I14" and name != "graph_checkpoint"
            if is_diagnostic:
                primary = "本候補で照合する公式評価の成分値・対象件数・差分。モデルの精度向上と診断の完了を区別する。"
                success = "上記の最小検証を同一対象・同一版で再計算でき、一致点、不一致点、回復可能な誤りと未確認部分が数値と証拠で分かれること。差がない結果もそのまま記録する。"
                loss_text = "なし"
                control = "評価・分析のみの段階ではなし。独立予測や学習checkpointが不足する場合、その作成は別途実験契約と実行承認が必要。"
            else:
                primary = "現行公式のcombined score。成分の接続・分裂指標、胚別結果、実行失敗と有効件数を併記する。"
                success = "本候補の変更だけで期待する誤りが減り、同じ候補数・学習量・時間など必要な対照を揃えた公式指標へ改善が残ること。最小改善幅と許容悪化は未決で、設計時に合意する。"
                loss_text = loss
                control = "学習を変える比較では同じ分割・画像露出・初期化方針のcontrolを要する。保存済み同条件controlの有無と再学習量は未確定。推論・選択だけの比較では重みを固定する。"
            parent = (
                "[exp002の設定](../experiments/exp002_unet3d_expandable_segments/config.yaml)"
                "と[requirements](../experiments/exp002_unet3d_expandable_segments/requirements.md)"
                "を構成の参照先とする。sample holdoutは診断用であり主評価へ流用しない。"
                "比較checkpoint・学習から除いた胚の予測・正式な親実験は未確定。"
            )
            stop = card["kill_criterion"].split(" 各段階を通過できなければ")[0]
            if is_diagnostic:
                stop = "評価器・対象動画・予測の学習来歴を揃えられない場合は、その比較の結論を保留する。診断で差がないことを他の手法全体の棄却へ広げない。"
            implementation = (
                "未決定。外部手法を使う場合は教師・構造・省略点を確定後、"
                "[用語集](../docs/glossary.md)のfaithful / staged-faithful / proxyから記録する。"
                "今回はコードを作成せず、この候補の中核処理を別手法へ置き換えない。"
            )
            target = (
                f"「{title}」だけを変更し、この候補の最小検証で"
                "元の条件との差を測る。"
                + ("予測対象は変更せず評価・診断の差を記録する。" if is_diagnostic
                   else f"期待する予測上の方向は「{card['hypothesis']}」。同じ主仮説の他変更は含めない。")
            )
            decode = (
                "同一graphと対象一覧へ評価処理を適用する。正解を使う置換は診断出力に限定し、提出予測と混ぜない。"
                if is_diagnostic else
                f"本候補の出力「{output}」を使う箇所だけを変更する。"
                "その他の中心選択・隣接時刻への接続・分裂の制約は比較対象と揃える。"
                "この候補自体が出力構造を変える場合の具体的な復号は未決事項を解決して固定する。"
            )
            text = f"""# {name}

- 候補名: `{name}`
- 説明: {title}
- 状態: `{STATE}`
- 対応する上位仮説: `{hypothesis_id}`
- 関連する上位仮説: なし。依存は先行検証の条件であり、主仮説を複数にしない。
- 作成日: {DATE}
- 最終更新日: {DATE}
- 依頼原文: 「この結果も踏まえて精度向上の仮説をできるだけ考えてください」「backlog/に記載するんではないですか？」「続きを実行してください」
- 期待する成果: {title}の成立条件と反証可能な一変更の比較を具体化する。
- 親実験 / 比較対象: {parent}
- 優先度: {priority}
- 優先度の理由: {priority_reason}
- `backlog/KAGGLE_DIRECTION.md` の対応箇所: [検証中の仮説と未着手索引](KAGGLE_DIRECTION.md#検証中の仮説)
- 元の調査項目: [14仮説・64候補の調査]({REPORT_LINK})の{group["id"]}、検証候補{item_number}（同節の{branch}項目目）。
- 先行条件 / 依存: {dependency_text}

## 観測事実と根拠

- 実測済みの事実: 本候補の改善値は未取得。根拠は次の既存集計・静的コード確認・参加者報告であり、効果の実証ではない。
{chr(10).join(evidence_lines)}
- 根拠ファイル / 一次資料: 上記出典と[統合仮説の原記録]({PORTFOLIO_LINK})の{group["id"]}。実験の数値は[metrics](../experiments/exp002_unet3d_expandable_segments/metrics.json)を参照する。
- 利用する保存済み生成物とSHA: 比較checkpoint・独立予測・候補cacheの実験生成物とSHAは未確定。今回の調査入力のSHAは[引き継ぎ記録](../studies/biohub_accuracy_ideas_20260910/backlog_handoff.json)に保存する。
- 仮定: Assumption: {card["hypothesis"]} この候補で実現できるかは未検証。画像由来の推論入力だけを使い、未知の注釈や完全maskを存在すると仮定しない。

## この候補が直接検証する仮説と範囲

- 上位仮説のうちこの候補が検証する範囲: {original_idea}
- この候補の具体的な仮説: {target}
- 仮説が正しい場合に期待する観測: {success}
- 仮説を棄却する観測: {stop}
- この候補だけで上位仮説を判断できるか: いいえ
- 上位仮説の判断に残る検証: {remaining}

## 入力・予測対象・出力・推論方法

- input: {input_text}
- target / objective: {target}
- output: {output}
- loss: {loss_text}
- decode / 推論方法: {decode}
- 処理単位: 1動画内の局所3D領域・時間窓・候補集合を対象処理に合わせて固定し、評価の独立性は胚単位で扱う。具体的な窓長・領域は未決。
- 実装区分: {implementation}

## 親実験からの差分

- 変更するもの: {original_idea} この候補名が表す処理だけを比較する。
- 固定するもの: 本候補以外の教師・候補・モデル・分割・前処理・復号を対照と揃える。具体的な基準版と値は未取得のため、この記述だけで実装を開始しない。
- 再利用するコード / config / 生成物: exp002の参照設定と公式sourceを確認して選ぶ。外部重み、候補cache、独立予測の再利用可否は未確定。
- 新しく作るもの: 最小検証に必要な本候補の処理・診断・差分記録。実験化後に確定し、今回は候補文書だけを作成する。

## 最小の反証可能な検証

- 検証方法: {cheap} 成立した場合だけ、学習側で方法を固定して胚を入れ替える2方向の比較へ進む。
- variant / config / fold / booster数: 条件は上記の対照に限定する。model config数・学習量は未決。外側は2胚を入れ替える2方向、選択は学習側内部だけ。booster数は0。
- control再学習: {control}
- 想定runtime / resource: {card["compute_estimate"]} 実測見積、追加GPU予算、最大入力形状は未確定。
- 候補の回収と実選別: 正解を診断だけに使う候補上限と、推論画像だけでの選別・補正を別々に記録する。評価・集計だけの候補ではモデル改善と診断結果を区別する。

## 成功条件と停止条件

- primary指標: {primary}
- 成功条件: {success}
- 必須guard: 未注釈を誤検出・負例にしない。教師・誤差分布・選別規則を検証胚の正解で作らない。評価対象動画、NaN、失敗、有効件数を対照と揃えて報告する。
- 成功時の次段階: 検証結果と未決事項をユーザーへ示し、実験化・採否を確認する。成功を理由に別候補の実行、submission、完了statusへの変更を自動で行わない。
- 失敗時の停止範囲: {stop} 他の兄弟候補や上位仮説全体は自動で閉じない。
- 再開条件: {unresolved}を具体化し、対照と診断を再現できる証拠を揃える。

## 実行しないこと

- 禁止する代替実装、proxy、同一OOF上の救済探索: hiddenのGEFF・正解由来の個数を読む、未対応候補を一律負例にする、外側胚で重みや条件を選ぶ、正解を使う候補上限を実際の改善と呼ぶ、を行わない。本候補を閾値調整だけの比較へ置き換えない。
- 壁打ちで採らなかった案と理由: 同じ主仮説の他変更は原因を分けるためこの候補へ同時投入しない。{card["counterevidence"]} [調査の採らない前提]({REPORT_LINK}#採らない前提と再検討条件)も参照する。

## リスク

- leakage / validation: 学習から除いた胚の予測が未取得。2胚の結果を繰り返し見て選ぶと探索結果になる。教師・内部予測・校正を含めた全工程の分割を記録する。
- hidden test: 推論は画像と利用可能なmetadataのみ。動画間の位置共有や絶対時刻を仮定せず、隣接時刻の辺へ戻せない欠測は直接skip edgeとして提出しない。
- runtime / memory: 非公開テスト全件のT4 2基・12時間内完了は未確認。学習費用と推論費用を分け、最悪例と上限を設計時に測る。
- 再現性: source、重み、分割、教師、候補、設定、評価器の版とSHAを実験化時に固定する。現在の参照設定だけから同一結果を保証しない。

## 未決事項

- 本候補固有: {unresolved}。
- 比較元のcheckpoint・独立胚予測・候補cacheとSHA。exp002の診断用splitを主評価へ読み替えない。
- 具体的な損失・モデル・復号、固定するconfig、選択用内部split、最小改善幅、許容悪化、実行予算。
- 上記の先行条件が成立しているか。今回はバックログ記載の承認であり、実験化・実装・実行・採否は未承認。

## 判断履歴

- {DATE}: ユーザーのバックログ記載と続行の指示に基づき、調査の{group["id"]}第{branch}項目を1候補1ファイルへ移した。前提不足を推測で埋めず、{STATE}として登録した。
- {DATE}: {priority}は着手条件に基づく検討順。実験の採用・不採用・完了の判断ではない。

## 次セッションへの引き継ぎ確認

- 固定するものを一意に説明できる: いいえ。基準のcheckpoint・分割・設定値が未確定。
- 変更するものを一意に説明できる: 処理の範囲は上記の1項目。実装方法の未決事項が残る。
- 最小検証と停止条件を一意に説明できる: 比較する処理と反証の方向は記録済み。数値条件と実行契約は未確定。
- 実行しないことを一意に説明できる: はい。上記の禁止事項と実験化未承認の範囲を維持する。
- 未決事項が明示されている: はい。未決事項節を解決する前に設計・実装へ進まない。
"""
            path = ROOT / "backlog" / f"{name}.md"
            assert not path.exists()
            assert "TODO" not in text and "TBD" not in text
            files[path] = text
            links.append(f"[`{name}`]({name}.md)")
            candidates.append({
                "name": name, "title": title, "priority": priority,
                "hypothesis_id": hypothesis_id, "origin_idea": group["id"],
                "origin_branch": branch, "survey_item_number": item_number,
                "original_subhypothesis": original_idea,
                "path": str(path.relative_to(ROOT)), "dependency": dependency,
                "state": STATE,
            })
        hypothesis_rows.append(
            f"| `{hypothesis_id}` | {group['statement']} | {'<br>'.join(links)} | - | {group['remaining']} |"
        )
        report_mapping.append(
            f"| {group['id']} | `{hypothesis_id}` | {len(group['rows'])} | {group['statement']} |"
        )

    # Preserve all unrelated direction content and the existing experiment lineage.
    anchor = "### 未着手バックログ"
    assert direction_before.count(anchor) == 1
    prefix, tail = direction_before.split(anchor, maxsplit=1)
    direction = prefix.rstrip() + "\n" + "\n".join(hypothesis_rows) + "\n\n" + anchor + tail
    registration_note = (
        "\n\n2026-09-10の[調査](../docs/surveys/biohub-accuracy-hypotheses_20260910.md)"
        "から14仮説・64候補を登録した。全件は未決事項を持つ検討メモであり、"
        "実験化の承認ではない。P1から順に内容を具体化し、依存が成立した候補だけを次へ進める。"
        "追加注釈・教師・識別可能な目的を要するP4は条件が揃うまで保留する。"
    )
    direction = direction.replace(anchor, anchor + registration_note, 1)
    direction = direction.replace(
        "未着手候補が0件の状態は正常であり、検証用のダミー候補は追加しない。\n\n", ""
    )
    assert direction.rstrip().endswith("| --- | --- | --- | --- | --- | --- |")
    def sort_key(candidate: dict) -> tuple:
        return (
            candidate["priority"], 0 if candidate["origin_idea"] == "I14" else 1,
            candidate["origin_idea"], candidate["origin_branch"],
        )
    index_rows = []
    for candidate in sorted(candidates, key=sort_key):
        name = candidate["name"]
        dependency = candidate["dependency"] or "評価器と比較graphの版"
        index_rows.append(
            f"| {candidate['priority']} | `{candidate['hypothesis_id']}` | "
            f"[`{name}`]({name}.md) | {candidate['title']} | {dependency} | `{STATE}` |"
        )
    direction = direction.rstrip() + "\n" + "\n".join(index_rows) + "\n"
    old_basis = re.search(
        r"(?s)(### 現行の比較基準\n\n)(.*?)(\n## アイデアバックログ)", direction
    )
    assert old_basis
    basis = (
        "`exp001_temporal_unet3d_baseline`はbatch size 16のsmoke backwardでT4がOOMとなった。"
        "`exp002_unet3d_expandable_segments`のversion 1はメモリ割当変更でsmokeを通過したが、"
        "時間予測が11時間gateを超えてfull trainingへ進まなかった。"
        "ユーザー承認後のversion 2は12時間gateでfull trainingを開始し、参照したローカル記録はrunningである。"
        "CV・Public LB・比較用checkpointは未取得。"
        "数値とstatusは[metrics](../experiments/exp002_unet3d_expandable_segments/metrics.json)、"
        "version別の進行は[SESSION_NOTES](../experiments/exp002_unet3d_expandable_segments/SESSION_NOTES.md)を正とする。"
        "現在のsample holdoutは診断用であり、下記の主評価用候補は学習から除いた胚の予測を別途要する。\n"
    )
    direction = direction[:old_basis.start(2)] + basis + direction[old_basis.end(2):]
    direction = direction.replace(
        "11時間gate内で3 epochsを完走できる学習条件",
        "承認済み12時間gateでの3 epochs完走と学習条件"
    )
    assert len(direction.encode()) <= 50_000, len(direction.encode())
    assert len(direction.splitlines()) <= 220
    assert max(map(len, direction.splitlines())) <= 800
    files[DIRECTION_PATH] = direction

    assert report_before.startswith("---\n") and "hypotheses: []" in report_before
    report = report_before.replace(
        "hypotheses: []", "hypotheses:\n" + "\n".join("- " + h for h in new_ids), 1
    )
    report = report.replace(
        "- 対応する上位仮説: なし",
        "- 対応する上位仮説: " + "、".join(f"`{h}`" for h in new_ids)
        + "。登録時の対応を末尾に記載。", 1
    )
    report = report.replace(
        "I01～I14はこのレポート内の参照番号であり、バックログの上位仮説IDではない。",
        "I01～I14はこのレポート内の参照番号であり、正式なバックログ上位仮説IDとの対応は末尾に記載する。"
    )
    report = report.replace(
        "本レポートとJSONの保存は、バックログ登録、実験作成、学習実行、提出、実験の採否を伴わない。",
        "初回の調査保存後、ユーザーのバックログ記載指示により以下の引き継ぎを行った。実験作成、学習実行、提出、実験の採否は行っていない。"
    )
    report += (
        "\n## バックログへの引き継ぎ\n\n"
        "2026-09-10、ユーザーの「backlog/に記載するんではないですか？」と続行指示を受け、"
        "64項目を[未着手バックログ](../../backlog/KAGGLE_DIRECTION.md#未着手バックログ)へ1候補1ファイルで登録した。"
        "以後、未着手候補の現在の優先度・状態・実験境界はbacklogの詳細ファイルを正とし、"
        "このレポートと元のJSONは調査時の根拠として残す。全候補は未決事項のある検討メモであり、実験化は未承認である。\n\n"
        "| 調査内番号 | 上位仮説ID | 候補数 | 仮説 |\n"
        "| --- | --- | ---: | --- |\n"
        + "\n".join(report_mapping)
        + "\n\n[64項目と候補ファイルの一対一対応・入力SHA](../../studies/biohub_accuracy_ideas_20260910/backlog_handoff.json)も保存した。\n"
    )
    files[REPORT_PATH] = report

    # Validate all dependencies before writes.
    by_name = {candidate["name"]: candidate for candidate in candidates}
    def visit(name: str, active: set[str]) -> None:
        if name in active:
            raise ValueError(f"Dependency cycle at {name}")
        dependency = by_name[name]["dependency"]
        if dependency:
            visit(dependency, active | {name})
    for name in by_name:
        visit(name, set())
    handoff = {
        "registered_at": DATE,
        "scope": spec["scope"],
        "source_sha256": {
            "portfolio": sha(PORTFOLIO_PATH), "candidate_specs": sha(SPEC_PATH),
            "survey_before_handoff": sha(REPORT_PATH),
            "direction_before_handoff": sha(DIRECTION_PATH),
        },
        "counts": {
            "new_hypotheses": len(new_ids), "candidates": len(candidates),
            "priorities": dict(sorted(Counter(c["priority"] for c in candidates).items())),
        },
        "hypotheses": [
            {"source_id": group["id"], "hypothesis_id": hid, "statement": group["statement"]}
            for group, hid in zip(groups, new_ids, strict=True)
        ],
        "candidates": candidates,
        "note": "Current candidate state and priority belong to backlog details; this is the registration snapshot.",
    }
    files[AREA / "backlog_handoff.json"] = json.dumps(handoff, ensure_ascii=False, indent=2) + "\n"
    for path, text in files.items():
        for target in re.findall(r"\]\(([^)]+)\)", text):
            target = target.split("#", 1)[0]
            if not target or "://" in target:
                continue
            linked = (path.parent / target).resolve()
            if linked not in files and not linked.exists():
                raise ValueError(f"Missing link in {path.relative_to(ROOT)}: {target}")
    print(json.dumps({
        "mode": "write" if args.write else "preview",
        **handoff["counts"],
        "direction_bytes": len(direction.encode()),
        "direction_lines": len(direction.splitlines()),
        "max_direction_line": max(map(len, direction.splitlines())),
        "files": len(files),
        "dependency_cycles": 0,
    }, ensure_ascii=False))
    if args.write:
        for path, text in files.items():
            path.write_text(text)
        print("Backlog handoff written; run check-strategy-docs and update the survey index.")

if __name__ == "__main__":
    main()
