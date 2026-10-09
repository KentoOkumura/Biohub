---
name: kaggle-submit-check
description: "Kaggle 提出前に、提出物と notebook metadata を検証し、問題を切り分けて記録する。`submission.csv`、zip された予測ファイル、Kaggle Notebook output、`kernel-metadata.json`、`sample_submission` 互換性、hidden test 対応、行数、重複 ID、欠損、internet/GPU metadata、オフライン実行前提を確認する。Kaggle submit 直前、notebook output の検証、提出失敗時に使う。"
---

# Kaggle 提出チェック

ローカルで、アップロードしない事前チェックを実行する。このスキルはsubmit、push、uploadを行わない。

## 手順

1. 対象を特定する。
   - CSV ファイル、zip ファイル、notebook フォルダ、またはプロジェクトルート。
   - 指定がなければ、`submission.csv`、`*.zip`、`kernel-metadata.json`、`sample_submission.csv` などの候補を探す。

2. リポジトリでは正のCSV validatorを実行する。`EXP`を指定した場合は、PASS/FAIL、行数、重複ID数、欠損数、infinite value数、target統計、submission SHAが同じ実験の`metrics.json`へ自動保存される。対象実験を特定できない場合は`EXP`を省略し、別実験へ推測で保存しない。

```bash
task submit-check EXP=expXXX_title SUBMISSION=/tmp/kaggle-output/expXXX_title/inference/submission.csv
```

`task`が利用できない場合は、同じ引数で`make submit-check`を使う。CSV以外のzip、notebook folder、`kernel-metadata.json`もまとめて調べる必要がある場合だけ、補助checkerを追加実行する。補助checkerはCSV部分を正のCSV validator、Notebook packageを`scripts/validate_kaggle_metadata.py`へ委譲する。Notebook検証の対象は`prepare-kaggle-notebooks`が生成したpackage全体であり、`project.yml`、bootstrapを含むNotebook、同梱sourceを一緒に指定する。metadataだけを切り出したコピーはPASSにしない。複数CSVから提出ファイルを選べない場合や検査対象がない場合はFAILとし、対象ファイルを明示して再実行する。

```bash
uv run python .agents/skills/kaggle-submit-check/scripts/check_submission.py PATH --sample sample_submission.csv
```

`--sample`はsample fileが存在する場合だけ使う。リポジトリ外へcheckerだけをコピーして使わず、正のvalidatorと`project.yml`を含むリポジトリルートから実行する。sampleとの行数・ID内容・行順の一致は固定行数の予測コンペに限る。Biohubの可変行数tracking graphではsampleは列schemaの照合に使い、testの全dataset、連続した行ID、node/edge参照など[コンペ固有の提出仕様](../../../docs/01_competition.md#提出)を検証する。

3. スクリプトだけでは証明できない warning を手で確認する。
   - CV は test で想定される grouping/time split と一致しているか。
   - 出力はコンペ固有の行単位とID規則を満たしているか。固定行数の予測コンペでは `sample_submission.csv` の行順を保っているか。
   - Kaggle のオフライン環境で依存関係を利用できるか。
   - ルールで許可されていない限り、internet が無効になっているか。
   - Kaggle Notebook の実行時間とメモリ制限に収まる推論になっているか。
   - code competition では公開 `test/` と `sample_submission.csv` が hidden test 用に差し替えられても動作するか。公開 test 固有の ID、行数、ファイル名、SHA、予測値に依存していないか。
   - hidden test の入力と保存済み model manifest / model 生成物だけで推論が完結するか。固定行数の予測コンペでは、実行時の sample submission と ID で 1 対 1 に整列できるか。

4. 結果を報告する。
   - `PASS`: チェックした範囲からは提出してよい。
   - `WARN`: ユーザーがリスクを受け入れる場合のみ提出してよい。
   - `FAIL`: 提出しない。正確な修正方法を示す。

## Code competitionの入力差し替え検証

提出経路を新規実装、または入力処理を変更したときは、公式の入力契約に照らして、公開例から変わり得るID、順序、件数、入力配置などを選んで確認する。任意の変化をすべて許容するのではなく、そのコンペで保証される構造と可変部分を区別する。

- 保存済みの小規模入力または契約を満たすテスト用の入力を使い、実際のNotebookの推論入口から入力列挙、モデル選択、予測、出力保存、コンペ固有の形式検証まで通す。補助関数だけのテストやsourceの文字列検査で代用しない。
- 高価なモデル呼出しは既知の小規模出力を返すstubへ置き換えられるが、検証したい入力処理・モデル選択・出力処理は本番コードを使う。未知ID、複数単位、並べ替えなど該当するケースで欠落・重複・参照切れがないことを確かめ、固定行数の提出では実行時sampleとの整列も確認する。
- 既存テストが同じ契約を検証していれば再利用する。検証した入力条件、置き換えた処理、未検証範囲を記録し、stubによる成功をモデル精度・hidden testの完走・時間制限内実行の保証にしない。local smokeの実行条件とテスト配置は`AGENTS.md`に従う。

提出前は対応テストの結果を確認する。既存のCSV・metadata validatorのPASSは、この入力差し替え検証を実行したことを意味しない。

## リポジトリ内の提出フロー

Kaggle 実験リポジトリ内で作業する場合:

1. `project.yml` に提出設定があることを確認する。sample file、id column、target columns を含む。
2. inference notebook の生成、push、Kaggle 実行は `kaggle-review-exp` と `kaggle-platform` に委譲する。このスキルでは、生成済み notebook と `kernel-metadata.json` を検証する。slug は 50 文字以内、`id` と `title` 由来 slug は一致、accelerator / internet / competition source は意図どおりであることを確認する。

3. `kaggle-platform` の手順で取得された Kaggle output がある場合、Kaggle 上で生成された `submission.csv` を手順2のコマンドでsample submissionに対して検証する。ローカル実験ディレクトリに提出CSVを常設しない。

4. PASS / WARN / FAIL と`metrics.json`へ保存した証拠を報告する。code competitionでは、Kaggle outputに存在する提出ファイル名（通常`submission.csv`）も引き渡し情報として明記する。実際のsubmitは`AGENTS.md`の承認条件と`kaggle-platform`の操作手順へ委譲する。submit後の監視は`kaggle-submit-monitor`に委譲する。

5. submit が行われてスコアが分かったら、リポジトリに記録する。

```bash
task record-exp EXP=expXXX SUBMISSION_REF=12345678 CV=0.1234 PUBLIC_LB=0.1200 EXTRA_ARGS="--submission-status complete"
task record-submission EXP=expXXX SUBMISSION=/path/to/submission.csv SUBMISSION_REF=12345678 EXTRA_ARGS="--notes baseline"
```

`record-exp`を先に実行して`metrics.json`の当該submission refの値を更新する。実験全体の代表CV/LBは変更しない。`record-submission`はCV/LBの引数を受け取らず、同じrefの値を読み、`SUBMISSIONS.md`へ記録する。同じrefを再指定すると既存行を更新するため、Private LB判明後も新しい行を追加しない。同じスコアを複数のコマンドへ手入力しない。

code competitionでKaggle outputをローカル取得していない場合は、`kaggle kernels files`またはUIで対象kernel versionの`submission.csv`を確認したうえで、ローカルファイル証拠を未取得として記録する。

```bash
task record-submission EXP=expXXX SUBMISSION=submission.csv SUBMISSION_REF=12345678 EXTRA_ARGS="--allow-missing-file --notes 'Kaggle output not downloaded'"
```

記録先の役割は`AGENTS.md`を正とする。`EXP`付きの`submit-check`は検証結果とsubmission SHAを対象実験の構造化された証拠へ保存する。提出コマンド、scoring経過、結果の解釈は同じ実験の正本へ分担して追記し、ここでは別の分担規則を定義しない。

確認項目:
- 固定行数の予測コンペでは、行数が実行時の `sample_submission.csv` と一致している。可変行数ではコンペ固有の出力単位と入力datasetの網羅を検証する。
- 必須列が存在し、想定外の追加列がある場合は意図的である。
- IDがコンペ固有の規則を満たしている。固定行数の予測コンペでは、IDの順序と内容が実行時sampleと一致している。
- missing、NaN、infinite values がない。
- offline/notebook competition では、ルールで許可されていない限り `enable_internet` が false。
- code competition では、公開 test 固有値に依存せず hidden test に差し替え可能である。
- hidden test入力と保存済み model manifest / model 生成物だけで推論できる。sampleへのID完全整列は固定行数の予測コンペに限る。

## 提出物の種類

- CSV: header、行数、重複 ID、空値/NaN/Inf、sample 互換性を検証する。
- Zip: member、hidden file、nested path、含まれる CSV があればその内容を検証する。
- Kaggle Notebook: 正のpackage validatorで`kernel-metadata.json`、Notebookの存在、bootstrapとsourceの整合性、internet/GPU flagsを検証する。期待するoutput fileの存在・内容はmetadataだけでは証明できないため、Kaggle実行後のfiles一覧・UIまたは取得済みoutputで別に確認する。

このスキルは提出前検証に集中する。アップロード済みの提出監視、またはコンペ終了・Private LB公開後の結果照合は[kaggle-submit-monitor](../kaggle-submit-monitor/SKILL.md)を使う。
