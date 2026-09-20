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
| 2026-09-20 | [Biohub SimpleNodeTransformerへのフレーム内Self-Attention追加設計](biohub-node-self-attention-design_20260920.md) | `survey` | `HYP-20260920-02` | `exp016` | `architecture`, `tracking` | `final` | - | 現行の入出力・lossを維持し、共有Encoderによるフレーム内Self-AttentionとCross-Attentionを切り替える設計。Model Bの逐次更新はユーザー確認済み。実装・学習は未実施。 |
| 2026-09-15 | [Biohub・トラッキングの6技術を図で理解する](biohub-tracking-techniques-illustrated_20260915.md) | `survey` | - | - | `tracking`, `algorithms`, `tutorial` | `final` | - | ILP、ハンガリアン法、SORT、DoG、HOG、対照学習を8枚の図と具体例で解説。コンペで確認した用途と一般的な追跡手法を区別する。 |
| 2026-09-13 | [Biohub Cell Tracking: 0.946 LB Notebook 解説](biohub-cell-tracking-0946-notebook-explanation_20260913.md) | `survey` | `HYP-20260910-12` | `exp011` | `public-notebooks`, `baseline`, `architecture` | `final` | - | 採用した公開0.946 Notebookについて、入力、2つのTemporalUNet3D、D4 TTA、候補点検出、Node Transformer、ILP、軌跡修復、出力検証をコードに沿って解説する。 |
| 2026-09-12 | [Biohub 公開検出器・トラッカー選定](biohub-public-detector-selection_20260912.md) | `survey` | `HYP-20260910-12` | `exp011` | `public-notebooks`, `baseline`, `model-selection` | `final` | - | 公開0.946 Notebookの推論構成を参照し、Pilkwang公開dataset 3件の現行版とcheckpoint SHAを固定。2026-09-12にユーザー採用済み。 |
| 2026-09-12 | [Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md) | `survey` | `HYP-20260910-01`, `HYP-20260910-02`, `HYP-20260910-03`, `HYP-20260910-04`, `HYP-20260910-05`, `HYP-20260910-06`, `HYP-20260910-07`, `HYP-20260910-08`, `HYP-20260910-09`, `HYP-20260910-10`, `HYP-20260910-11`, `HYP-20260910-12`, `HYP-20260910-13`, `HYP-20260910-14`, `HYP-20260911-01` | - | `backlog`, `readiness` | `final` | - | 全64候補の状態を監査し、測定・先行成果物待ちを設計不可としていた9件を訂正。設計可能10件・設計判断が残る54件とし、全件の理由を保存。 |
| 2026-09-10 | [Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md) | `survey` | `HYP-20260910-01`, `HYP-20260910-02`, `HYP-20260910-03`, `HYP-20260910-04`, `HYP-20260910-05`, `HYP-20260910-06`, `HYP-20260910-07`, `HYP-20260910-08`, `HYP-20260910-09`, `HYP-20260910-10`, `HYP-20260910-11`, `HYP-20260910-12`, `HYP-20260910-13`, `HYP-20260910-14` | - | `hypotheses`, `baseline`, `validation` | `final` | - | 最新公開資料と公式学習コードから14の主仮説・64の検証候補を整理。疎注釈の損失、実予測誤差を使う補正、分裂組、異なる候補の統合を重点候補とする。 |
| 2026-09-10 | [Biohub 最新公開ベースライン調査](biohub-public-baselines_20260910.md) | `survey` | - | `exp001`, `exp002` | `public-notebooks`, `baseline`, `validation` | `final` | - | 公開比較対象はTemporalUNet3D・Node Transformerに複数モデルとTTA・graph補正を加えた0.94前後。0.946報告もあるが、最新スコアと独自proxyの扱いに注意が必要。 |
| 2026-09-10 | [Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) | `survey` | `HYP-20260910-01`, `HYP-20260910-02`, `HYP-20260910-03`, `HYP-20260910-04`, `HYP-20260910-05`, `HYP-20260910-06`, `HYP-20260910-07`, `HYP-20260910-08`, `HYP-20260910-09`, `HYP-20260910-10`, `HYP-20260910-11`, `HYP-20260910-12`, `HYP-20260910-13`, `HYP-20260910-14` | `exp001`, `exp002`, `exp003` | `backlog`, `validation` | `final` | - | 64候補の仕様・測定待ち・ユーザー判断を分離。公式分裂評価と公開proxyの局所差、未知候補への勾配、checkpoint保存と既存正規化を確認。 |
| 2026-08-15 | [Biohub 公式アノテーション分布調査](biohub-official-annotation-distribution_20260815.md) | `survey` | - | - | `data`, `annotation` | `final` | - | 公式GEFFは全体で推定細胞nodeの2.82%を収録し、胚別annotation率とsample間分布に大きな差がある。 |
| 2026-08-14 | [Biohub リポジトリ設定・validation調査](biohub-repository-setup-validation_20260814.md) | `survey` | - | - | `validation`, `data`, `public-notebooks` | `final` | - | 入力metadata、主催者baseline、vote順上位15公開Notebookを確認し、暫定5-foldを撤回してvalidatorとmetric evaluatorの出所を整理した。 |

## 上位仮説別

| キー | レポート |
| --- | --- |
| `HYP-20260910-01` | [Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md)<br>[Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) |
| `HYP-20260910-02` | [Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md)<br>[Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) |
| `HYP-20260910-03` | [Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md)<br>[Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) |
| `HYP-20260910-04` | [Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md)<br>[Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) |
| `HYP-20260910-05` | [Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md)<br>[Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) |
| `HYP-20260910-06` | [Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md)<br>[Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) |
| `HYP-20260910-07` | [Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md)<br>[Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) |
| `HYP-20260910-08` | [Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md)<br>[Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) |
| `HYP-20260910-09` | [Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md)<br>[Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) |
| `HYP-20260910-10` | [Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md)<br>[Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) |
| `HYP-20260910-11` | [Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md)<br>[Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) |
| `HYP-20260910-12` | [Biohub Cell Tracking: 0.946 LB Notebook 解説](biohub-cell-tracking-0946-notebook-explanation_20260913.md)<br>[Biohub 公開検出器・トラッカー選定](biohub-public-detector-selection_20260912.md)<br>[Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md)<br>[Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) |
| `HYP-20260910-13` | [Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md)<br>[Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) |
| `HYP-20260910-14` | [Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md)<br>[Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) |
| `HYP-20260911-01` | [Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md) |
| `HYP-20260920-02` | [Biohub SimpleNodeTransformerへのフレーム内Self-Attention追加設計](biohub-node-self-attention-design_20260920.md) |

## 実験番号別

| キー | レポート |
| --- | --- |
| `exp001` | [Biohub 最新公開ベースライン調査](biohub-public-baselines_20260910.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) |
| `exp002` | [Biohub 最新公開ベースライン調査](biohub-public-baselines_20260910.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) |
| `exp003` | [Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) |
| `exp011` | [Biohub Cell Tracking: 0.946 LB Notebook 解説](biohub-cell-tracking-0946-notebook-explanation_20260913.md)<br>[Biohub 公開検出器・トラッカー選定](biohub-public-detector-selection_20260912.md) |
| `exp016` | [Biohub SimpleNodeTransformerへのフレーム内Self-Attention追加設計](biohub-node-self-attention-design_20260920.md) |

## 種類別

| キー | レポート |
| --- | --- |
| `survey` | [Biohub SimpleNodeTransformerへのフレーム内Self-Attention追加設計](biohub-node-self-attention-design_20260920.md)<br>[Biohub・トラッキングの6技術を図で理解する](biohub-tracking-techniques-illustrated_20260915.md)<br>[Biohub Cell Tracking: 0.946 LB Notebook 解説](biohub-cell-tracking-0946-notebook-explanation_20260913.md)<br>[Biohub 公開検出器・トラッカー選定](biohub-public-detector-selection_20260912.md)<br>[Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md)<br>[Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md)<br>[Biohub 最新公開ベースライン調査](biohub-public-baselines_20260910.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md)<br>[Biohub 公式アノテーション分布調査](biohub-official-annotation-distribution_20260815.md)<br>[Biohub リポジトリ設定・validation調査](biohub-repository-setup-validation_20260814.md) |

## トピック別

| キー | レポート |
| --- | --- |
| `algorithms` | [Biohub・トラッキングの6技術を図で理解する](biohub-tracking-techniques-illustrated_20260915.md) |
| `annotation` | [Biohub 公式アノテーション分布調査](biohub-official-annotation-distribution_20260815.md) |
| `architecture` | [Biohub SimpleNodeTransformerへのフレーム内Self-Attention追加設計](biohub-node-self-attention-design_20260920.md)<br>[Biohub Cell Tracking: 0.946 LB Notebook 解説](biohub-cell-tracking-0946-notebook-explanation_20260913.md) |
| `backlog` | [Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md) |
| `baseline` | [Biohub Cell Tracking: 0.946 LB Notebook 解説](biohub-cell-tracking-0946-notebook-explanation_20260913.md)<br>[Biohub 公開検出器・トラッカー選定](biohub-public-detector-selection_20260912.md)<br>[Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md)<br>[Biohub 最新公開ベースライン調査](biohub-public-baselines_20260910.md) |
| `data` | [Biohub 公式アノテーション分布調査](biohub-official-annotation-distribution_20260815.md)<br>[Biohub リポジトリ設定・validation調査](biohub-repository-setup-validation_20260814.md) |
| `hypotheses` | [Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md) |
| `model-selection` | [Biohub 公開検出器・トラッカー選定](biohub-public-detector-selection_20260912.md) |
| `public-notebooks` | [Biohub Cell Tracking: 0.946 LB Notebook 解説](biohub-cell-tracking-0946-notebook-explanation_20260913.md)<br>[Biohub 公開検出器・トラッカー選定](biohub-public-detector-selection_20260912.md)<br>[Biohub 最新公開ベースライン調査](biohub-public-baselines_20260910.md)<br>[Biohub リポジトリ設定・validation調査](biohub-repository-setup-validation_20260814.md) |
| `readiness` | [Biohub バックログ64候補の状態分類監査](biohub-backlog-status-audit_20260912.md) |
| `tracking` | [Biohub SimpleNodeTransformerへのフレーム内Self-Attention追加設計](biohub-node-self-attention-design_20260920.md)<br>[Biohub・トラッキングの6技術を図で理解する](biohub-tracking-techniques-illustrated_20260915.md) |
| `tutorial` | [Biohub・トラッキングの6技術を図で理解する](biohub-tracking-techniques-illustrated_20260915.md) |
| `validation` | [Biohub 精度向上の仮説と最小検証](biohub-accuracy-hypotheses_20260910.md)<br>[Biohub 最新公開ベースライン調査](biohub-public-baselines_20260910.md)<br>[Biohub 64候補の未決事項調査とユーザー判断](biohub-backlog-readiness_20260910.md)<br>[Biohub リポジトリ設定・validation調査](biohub-repository-setup-validation_20260814.md) |
<!-- END AUTO SURVEY INDEX -->
