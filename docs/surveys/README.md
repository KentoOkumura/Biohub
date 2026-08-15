# 調査レポート

完了した調査結果を探すときは、最初にこのファイルを参照します。対象は、実験構成・モデル説明、OOF／結果EDA、特徴量・failure mode、複数実験比較、論文・公開Notebook調査です。

## 保存ルール

- 人間が読む完了レポートは`docs/surveys/*.md`を正とします。
- 上位仮説を直接扱うレポートはfront matterの`hypotheses`に`HYP-YYYYMMDD-NN`を記録し、本文にも判断対象と結論を記載します。
- 調査コードと生の表・図は`studies/`、実験実装・実行記録・公式結果は`experiments/`に残し、レポートからリンクします。
- 同じテーマの追調査は原則として既存レポートを更新し、新しい問い・証拠範囲・結論になる場合だけ新しいレポートを作ります。

## レポートの状態

- `draft`: 調査中。placeholderを許可する。
- `final`: 現在参照する完了レポート。
- `superseded`: 後継レポートへ置き換えられた履歴。`superseded_by`に後継ファイル名を記録する。

## 作成・完了手順

```bash
task new-survey-report SURVEY_TITLE="調査タイトル" SURVEY_SLUG="report-slug" EXTRA_ARGS="--type survey --topic topic"
task update-survey-index
task validate-surveys
```

<!-- BEGIN AUTO SURVEY INDEX -->
## レポート一覧

| 日付 | レポート | 種類 | 上位仮説 | 実験 | トピック | 状態 | 後継 | 一行要約 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-15 | [Biohub 公式アノテーション分布調査](biohub-official-annotation-distribution_20260815.md) | `survey` | - | - | `data`, `annotation` | `final` | - | 公式GEFFは全体で推定細胞nodeの2.82%を収録し、胚別annotation率とsample間分布に大きな差がある。 |
| 2026-08-14 | [Biohub リポジトリ設定・validation調査](biohub-repository-setup-validation_20260814.md) | `survey` | - | - | `validation`, `data`, `public-notebooks` | `final` | - | 入力metadata、主催者baseline、vote順上位15公開Notebookを確認し、暫定5-foldを撤回してvalidatorとmetric evaluatorの出所を整理した。 |

## 上位仮説別

| キー | レポート |
| --- | --- |
| - | - |

## 実験番号別

| キー | レポート |
| --- | --- |
| - | - |

## 種類別

| キー | レポート |
| --- | --- |
| `survey` | [Biohub 公式アノテーション分布調査](biohub-official-annotation-distribution_20260815.md)<br>[Biohub リポジトリ設定・validation調査](biohub-repository-setup-validation_20260814.md) |

## トピック別

| キー | レポート |
| --- | --- |
| `annotation` | [Biohub 公式アノテーション分布調査](biohub-official-annotation-distribution_20260815.md) |
| `data` | [Biohub 公式アノテーション分布調査](biohub-official-annotation-distribution_20260815.md)<br>[Biohub リポジトリ設定・validation調査](biohub-repository-setup-validation_20260814.md) |
| `public-notebooks` | [Biohub リポジトリ設定・validation調査](biohub-repository-setup-validation_20260814.md) |
| `validation` | [Biohub リポジトリ設定・validation調査](biohub-repository-setup-validation_20260814.md) |
<!-- END AUTO SURVEY INDEX -->
