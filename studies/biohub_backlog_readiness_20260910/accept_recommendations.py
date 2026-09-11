from __future__ import annotations
import hashlib
import json
import re
import shutil
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/exp003_official_metric_audit"
SOURCE = ROOT / "experiments/exp002_unet3d_expandable_segments/official_source"
old_candidate = ROOT / "backlog/official_metric_check.md"
old = old_candidate.read_text()
approval = "2026-09-11: ユーザー『すべて推奨でいいです』。推奨の順序、無料枠内・課金なし、両胚改善、外部重みは選んだ追加候補のみ、人手注釈は当面なしを承認。FOCUSの取得同意は今回の対象外。"
record = """## 承認済みの進め方

- """ + approval + """
- 実施順: 公式評価の照合、胚を分けた基準予測、誤差と段階別回収上限の分析、未注釈子に対する接続損失の比較。
- 計算費用: Kaggle無料枠内。課金しない。実行前の残量と小規模実測を確認し、後続の推論確認に必要な枠を残す。残量不足なら承認済み方針のまま再開可能なところまで準備し、残量を超えるrunを開始しない。
- 精度判断: 公式指標の両胚での改善、実行失敗の増加なし、提出推論12時間以内。採否・完了は結果提示後のユーザー判断を維持する。
- 外部重みと人手: 当面の基準予測は自前学習。外部重みは選択した追加候補に限定し、人手注釈は当面行わない。FOCUSの条件同意は必要になった時点で本人が確認する。
- 64候補全部を一括実行する承認ではない。選択済みの推奨順序を進め、後続候補の実験化はその依存証拠を確認する。共通方針や選択済み比較の同じ承認を再度求めない。
"""
requirements = """# exp003_official_metric_audit 要件と実装方法

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: 2026-09-11、ユーザー「すべて推奨でいいです」。推奨された公式評価の照合を最初に実施する。
- 移行元backlog: `official_metric_check`。元の契約と判断履歴を末尾に保存し、現在の実行条件をこの節で確定する。
- 対応する上位仮説: `HYP-20260910-14`。
- 検証範囲: 最新確認済み公式のnode対応・edge・division・集計を、省略せず公開DCTTAのproxyと同じ入力で比較する。
- この実験だけで上位仮説を判断できるか: いいえ。独立胚でのモデル順位、checkpoint選択、誤差と段階別上限の分析が残る。
- 親実験: exp002_unet3d_expandable_segments。モデルの学習変更はない。
- 一次資料: 公式commit `075fc5f5a52d11077f9dc2b074644618f26939e2`、[先行調査](../../docs/surveys/biohub-backlog-readiness_20260910.md)、保存したDCTTA公開Notebook。
- 固定するもの: config記載の公式版、公開proxy関数、入力fixture、物理scale、7µm対応、既存4動画の予測CSVとSHA。予測CSVを採点のために修正しない。
- 変更するもの: 同じgraphに適用する評価関数だけ。モデル・候補生成・学習は変更しない。
- 最小検証: 人工graphで完全な公式evaluate/per_sample_metrics/summariseを実行する。既存予測CSVに対応する4動画のGTを読み、公式値とproxy値・成分・順位を記録する。
- 成功条件: 全fixtureと4動画を処理し、入力・source SHA、component counts、NaN、失敗、対応差が再計算可能な形で残る。差があること自体を成功条件にしない。
- 停止条件: sourceまたは入力SHA不一致、期待する4動画の不足、意図しない失敗や集計からの脱落。結果を保存して失敗終了する。
- 実行しないこと: 学習、予測の再生成、GPU利用、実submission、実画像や重みのローカル取得、proxyによる公式の置換。
- 未決事項: なし。依存環境とgraphは以下の実装と入力で確定した。実データの採点値は実行後に得る値であり設計上の未決ではない。
- backlogからの具体化: 局所関数だけの先行調査を完全な公式評価へ拡張。学習済み予測が新しく得られたため、その4動画を固定入力として使用する。胚分離の性能推定とは呼ばない。

## 判断履歴

- 2026-09-10: 先行調査とbacklog登録。元の根拠・禁止事項・判断は末尾に保持する。
- """ + approval + """

## 手法契約

- 依頼原文: 「すべて推奨でいいです」。
- input: JSONの人工graph、exp002 inference version 1のsubmission.csv、同じ4 sample IDの公式train GEFF。画像本体は読まない。
- target / objective: 同じ入力に対する評価差・仕様一致と、不一致の原因の特定。
- output: fixtureごとの採点表、4動画の公式/proxy成分表、run集計、順位差、環境・SHA・失敗一覧。
- loss: なし。
- decode: なし。CSVのnode IDを一対一でtracksdataのIDへ写像し辺を再構成する。入力の重複辺・非隣接辺をこちらで落とさず、公式とproxy各々の規則に渡す。
- context unit: 1人工graphまたは1動画。公式summariseによる4動画の集計。
- 実装区分: faithful（このリポジトリ内の管理用語）。固定公式の全関連関数と公開proxyをNotebookに展開し、相対importだけを除く。
- 省略する機構: なし。推論・学習はこの評価実験の対象に含まれない。
- この実験で判断できないこと: 独立胚への一般化、モデル改善、非公開LB、未注釈の全細胞への完全な精度。4動画の順位は動画ごとの得点順位であり、モデル順位ではない。

## 実装方法

- Notebook: `exp003_official_metric_audit_audit.py`からJupytextで変換する。imports、config、offline依存、公式関数、proxy、入力確認、fixture、実予測採点、集計と保存をセルへ展開する。
- source: 元のmetrics.pyとdivision_metrics.pyをassetsへ固定し、同じASTの関数をNotebookへ持ち込む。公式evaluate内の相対importは既に展開した関数を使うため取り除く。
- 入力: metadataから4 sample IDを固定。Kaggle kernel source内のCSVをSHAで特定する。GEFFの推定総数は評価器のpenalty計算だけに使い、予測へ渡さない。
- 公開proxy: 保存したDCTTAのnode matching、edge confusion、division confusion、Jaccardと調整関数を同じ形で使用する。公式側は完全なtracksdata matchingを含む。
- fixture: 正しい局所分裂、遠い分裂、重複辺、非隣接辺、合流、過剰な娘、分裂のない継続、辺なし、注釈外node。期待する不変条件をconfigへ置く。
- テスト: sourceとNotebook関数のAST一致、fixtureの参照整合、入力SHAの固定、GPU無効・再学習なし。全公式採点の実行はKaggleを正とする。
- 出力: artifacts/内のJSON・CSV・環境記録。JSONの非有限値はnullへ変換し、欠測理由を保持する。

## 探索幅とpivot判定

- 変更class: mechanism（評価方法の照合でありモデル改善の探索ではない）。
- 学習variant/model/config/fold/booster: 0/0/1/0/0。control再学習なし、CPUのみ。
- 比較する方式: 固定公式と固定公開proxyの2方式だけ。係数探索は行わない。
- 表現を変える候補: division_tripletsを後続案として保存済み。本実験はその評価の前提確認。
- kaggle-idea-forge: 今回は不要。64候補の発想と評価前提の調査が済んでいる。

## 再現性・リスク

- seed policy: 42固定。乱数を使わない固定fixtureと保存済み予測を処理する。
- 並列処理: 逐次CPU評価。GPU、augmentation、baggingなし。
- SHA: 入力CSV、fixture、公式とproxy source、Notebook、結果表、kernel versionを記録する。
- bootstrap: configとassetsを埋め込み、正のNotebookと生成packageをpush直前に検証する。
- リーク: 4動画には学習に使った胚の予測が含まれる。評価器照合専用でありCV/LBに記録しない。
- NaN・失敗: 有効件数と全対象を併記し、失敗を黙って集計から除かない。
- 実行時間: CPUの4動画評価のみ。12時間を上限とし、費用のかかる学習や推論へ切り替えない。
- 忠実性: source AST一致テストで関数の改変を検出する。公開proxyの欠点を修正してから比較しない。

## 受け入れ基準

- 固定した全入力と全fixtureを処理し、差・件数・失敗・sourceとinputのSHAが記録される。
- 新しいモデル精度や独立胚のCVと混同しない。
- strict実験検証、Ruff、対象実験テスト、Jupytext変換、Kaggle package検証が通る。
- 実行結果をmetrics、解釈をresult、時系列をSESSION_NOTESに記録する。
- ユーザーの実験完了・採否判断は結果提示後まで未判断のままにする。

## 移行元候補の記録

以下は2026-09-10時点の検討記録の保存。過去の未承認・未取得という記載は履歴であり、現在の契約は上記を正とする。

"""
def link_rewrite(match):
    target = match.group(1)
    if target.startswith(("#", "http:", "https:")):
        return match.group(0)
    full = (old_candidate.parent / target.split("#")[0]).resolve()
    suffix = "#" + target.split("#", 1)[1] if "#" in target else ""
    import os
    return "](" + os.path.relpath(full, EXP) + suffix + ")"
archive = re.sub(r"\]\(([^)]+)\)", link_rewrite, old).replace("# official_metric_check\n", "### official_metric_check（移行元）\n", 1)
# Demote the source document headings to distinguish historical sections.
archive = re.sub(r"(?m)^## ", "### ", archive)
(EXP / "requirements.md").write_text(requirements + archive)
source_files = ["src/tracking_cellmot/metrics.py", "src/tracking_cellmot/division_metrics.py", "LICENSE"]
(EXP / "assets").mkdir(exist_ok=True)
manifest = {"commit": "075fc5f5a52d11077f9dc2b074644618f26939e2", "files": {}}
for source in source_files:
    destination = EXP / "assets" / Path(source).name
    shutil.copyfile(SOURCE / source, destination)
    manifest["files"][destination.name] = hashlib.sha256(destination.read_bytes()).hexdigest()
(EXP / "assets/source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
config = yaml.safe_load((EXP / "config.yaml").read_text())
config["experiment"].update(description="Compare complete official scoring and archived public proxy on synthetic and fixed predicted graphs.", route="official_metric_audit")
config["lineage"] = {"parent": "exp002_unet3d_expandable_segments", "hypothesis_id": "HYP-20260910-14", "backlog_candidate": "official_metric_check", "diff_summary": "CPU-only full official scorer/proxy comparison; no model change."}
config["validation"] = {"strategy": "fixed_graph_metric_audit", "metric": "metric_component_agreement", "seed": 42, "n_folds": 0, "diagnostic_only": True}
config["model"] = {"name": "none_metric_audit", "params": {}}
packages = ["bidict==0.23.1", "donfig==0.8.1.post1", "geff==1.2.0.1.1", "geff-spec==1.1.1", "ilpy==0.6.0", "numcodecs==0.15.1", "polars==1.42.0", "polars-runtime-32==1.42.0", "pyscipopt==6.2.1", "rustworkx==0.18.0", "tracksdata==0.1.0rc6.dev3+g980c2d30a", "zarr==3.2.1"]
config["audit"] = {"max_distance_um": 7.0, "scale_zyx_um": [1.625, 0.40625, 0.40625], "fixture_scale_zyx_um": [1., 1., 1.], "source_commit": manifest["commit"], "prediction_sha256": "b039062114962347aca22f1268a64196893eaabb62c1046ae0c90acbad29f223", "sample_ids": ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"], "official_adjustment_alpha": 0.1, "official_division_weight": 0.1, "offline_packages": packages}
config["runtime"] = {"use_amp": False, "num_workers": 1, "batch_size": 1, "kaggle": {"enable_gpu": False, "enable_internet": False, "time_limit_hours": 12, "dataset_sources": ["thibautgoldsborough/cellmot-baseline-artifacts"], "audit": {"enable_gpu": False, "enable_internet": False, "kernel_sources": ["kentookumura/exp002-unet3d-expandable-segments-inference"]}, "bootstrap_files": ["assets/" + name for name in [*manifest["files"], "source_manifest.json", "public_proxy.py", "fixtures.json"]]}}
config["reproducibility"].update(seed_policy="fixed_fixture_no_rng", seed=42, parallel_rng_policy="sequential_no_rng", stochastic_components=[], notes=["No model score claim; fixed CSV SHA and source SHA; CPU audit."])
config["notes"] = [approval, "No GPU, model training, paid resources, raw-image download, or submission."]
(EXP / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
(EXP / "README.md").write_text("""# exp003_official_metric_audit

## 概要

固定した人工graphと既存予測へ、完全な公式評価器と公開proxyを適用して差を記録する。モデルの改善や独立胚のCVを測る実験ではない。次はCPUの監査NotebookをKaggleで実行する。

## 正の記録

- [要件と承認](requirements.md)
- [設定・系譜](config.yaml)
- [数値と実行証拠](metrics.json)
- [解釈とユーザー判断](result.md)
- [実行中の記録](SESSION_NOTES.md)

## 実行入口

- 監査Notebook: `exp003_official_metric_audit_audit.ipynb`
- Kaggleでの実行手順と進捗: [SESSION_NOTES](SESSION_NOTES.md)
""")
(EXP / "result.md").write_text("""# exp003_official_metric_audit 結果

## 仮説

公開proxyと公式の採点差を、完全な公式評価器と固定入力で再現できる。

## 実行証拠

設定は[config](config.yaml)、実行値は[metrics](metrics.json)、時系列は[SESSION_NOTES](SESSION_NOTES.md)を参照する。現在は実行前である。

## 解釈

先行調査は局所関数の差だけを確認した。本実験で全公式採点と固定4動画へ照合範囲を広げる。独立胚への精度と解釈しない。

## ユーザー判断

実験化は2026-09-11「すべて推奨でいいです」で承認。実験の完了・採否は未判断。

## 次

CPU監査を実装・検証し、Kaggleで実行する。
""")
(EXP / "SESSION_NOTES.md").write_text("""# exp003_official_metric_audit SESSION_NOTES

## 現在の作業

承認済みの公式評価照合を実装する。GPU使用なし、学習variant 0、model 0、config 1、fold 0、booster 0、control再学習なし。

## 時系列

- 2026-09-11: """ + approval + """
- 2026-09-11: exp002の最新記録とログから3 epochs完走・4動画推論済みと確認。公式評価照合へ既存CSVを再利用し、独立胚評価とは分ける。
- 2026-09-11: GPU quotaを読み取り、21.40h残と確認。CPU監査では消費しない。
- 2026-09-11: make new-exp EXP=exp003_official_metric_auditを実行。移行元候補をrequirementsへ保存し、現在の契約を具体化した。

## 実行予定

Jupytext変換、strict検証、Ruff、対象テスト、CPU package準備、Kaggle push、結果回収と記録。

## 残る事項

Kaggle実行値。ユーザー判断の未決事項はなし。
""")
metrics = json.loads((EXP / "metrics.json").read_text())
metrics.update(experiment=EXP.name, metric="metric_component_agreement", key_idea="Full official scorer versus fixed public proxy; CPU audit.", notes="Authorized recommendations; prepared for CPU audit.", status="planned")
(EXP / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
# Remove only untouched scaffold notebooks: this experiment has no train/inference.
for kind in ["train", "inference"]:
    path = EXP / f"{EXP.name}_{kind}.ipynb"
    if path.exists():
        path.unlink()
# kaggle-strategy handoff after the complete source contract has been saved.
direction_path = ROOT / "backlog/KAGGLE_DIRECTION.md"
direction = direction_path.read_text()
direction = direction.replace("## 現在の重点\n", record + "\n## 現在の重点\n", 1)
lines = []
for line in direction.splitlines():
    if line.startswith("| P1 |") and "[`official_metric_check`]" in line:
        continue
    if line.startswith("| `HYP-20260910-14` |"):
        cells = line.split("|")
        cells[3] = cells[3].replace("[`official_metric_check`](official_metric_check.md)<br>", "")
        cells[4] = " [`exp003_official_metric_audit`](../experiments/exp003_official_metric_audit/) "
        line = "|".join(cells)
    lines.append(line)
direction_path.write_text("\n".join(lines) + "\n")
old_candidate.unlink()
for path in (ROOT / "backlog").glob("*.md"):
    if path.name.startswith("_") or path.name == "KAGGLE_DIRECTION.md":
        continue
    text = path.read_text()
    text = text.replace("](official_metric_check.md)", "](../experiments/exp003_official_metric_audit/)")
    note = "## 2026-09-11の承認確認\n\n- 共通の推奨方針はユーザーの「すべて推奨でいいです」で承認済み。[承認済みの進め方](KAGGLE_DIRECTION.md#承認済みの進め方)を参照する。下記D2・D3等を同じ内容で再質問しない。\n- この候補の実行は承認済みの順序と依存する証拠に従う。選ばれていない64候補の一括実行は意味しない。\n\n"
    text = text.replace("## 2026-09-10の未決事項調査\n", note + "## 2026-09-10の未決事項調査\n", 1)
    path.write_text(text)
for path in (ROOT / "docs/surveys").glob("*.md"):
    text = path.read_text()
    text = text.replace("](../../backlog/official_metric_check.md)", "](../../experiments/exp003_official_metric_audit/)")
    if path.name == "biohub-backlog-readiness_20260910.md":
        text = text.replace("## ユーザーに決めてもらう事項\n", "## ユーザーに決めてもらう事項\n\n2026-09-11追記: ユーザー「すべて推奨でいいです」により以下の推奨方針は承認済み。実験化の対応は[backlogの現在方針](../../backlog/KAGGLE_DIRECTION.md#承認済みの進め方)を参照する。\n", 1)
    path.write_text(text)
print("approval recorded, full contract migrated, exp003 CPU audit configured")
