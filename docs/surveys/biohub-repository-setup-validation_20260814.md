---
title: Biohub リポジトリ設定・validation調査
date: '2026-08-14'
types:
- survey
hypotheses: []
experiments: []
topics:
- validation
- data
- public-notebooks
status: final
summary: 入力metadata、主催者baseline、vote順上位15公開Notebookを確認し、暫定5-foldを撤回してvalidatorとmetric evaluatorの出所を整理した。
---

# Biohub リポジトリ設定・validation調査

作成日: 2026-08-14

## 結論

- 以前「validator」と呼んだものは、このリポジトリ独自の[`scripts/validate_submission.py`](../../scripts/validate_submission.py)であり、コンペ主催者の提供物ではない。調査時点ではsample submissionと同じ行数・同じ`id`列を要求していたが、その後、本コンペでは可変行数のtracking graphを検証するよう修正した。
- 主催者が公開しているのは、評価仕様[`metrics.md`](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)と、`tracking_cellmot.metrics` / `scripts/evaluate.py`を含むbaseline repositoryである。Kaggle hidden scorer、主催者の公開参照実装、参加者の再実装、当リポジトリの提出形式検証は別物である。
- vote順上位15件の公開Notebookには学習ループも`KFold` / `GroupKFold`もなかった。11件は学習済み`split_0` checkpointを読む推論Notebookであり、公開Notebookだけからcheckpoint作成時の全fold構成は確定できない。
- `Clean Approach + Lightweight Local CV`の“fixed-8”は8-foldではない。2胚から4 sampleずつ選んだ固定8 sampleを1個の`split_0` checkpointで推論し、参加者が再実装したmetricで評価する1回のholdoutである。
- trainは2胚しかないため、embryo-disjointの5-foldは不可能である。ユーザー判断により、primary validationは2方向のleave-one-embryo-out、secondary validationは公開Notebookと同じ固定8 sampleのholdoutに確定した。

## 調査目的

ローカルへcompetition raw dataをダウンロードせず、リポジトリ設定に必要なsample数、胚数、入力shape、提出schemaをKaggle上で確認する。また、主催者baselineと公開Notebookの学習、fold、ローカル評価の実装を確認し、設定値と用語の出所を明確にする。

## 対象と証拠範囲

- 対応する上位仮説: なし
- 対象実験: なし。リポジトリ設定調査であり、`experiments/`の対象ではない。
- 入力データ・生成物: Kaggle上のcompetition input metadata、sample submission、vote順上位15公開Notebook、主催者公開repository。画像chunkと正解node/edge array本体は読み込んでいない。
- 評価方法: Notebook sourceを対象に、学習API、fold API、checkpoint path、固定validation sample、metric実装、submission schema checkを検索し、該当箇所を手作業で確認した。
- このレポートからは判断できないこと: hidden testの胚数、学習済み公開checkpointの完全な作成履歴、Kaggle hidden scorerと参加者再実装の全edge caseでの一致、2方向leave-one-embryo-outの実測分散。

## 主催者baselineの学習方法

- 主催者公開repositoryは、`TemporalUNet3D`で連続frameの3D画像からvoxel featureとcell center detection mapを作り、検出nodeのfeatureと位置embeddingを`SimpleNodeTransformer`へ渡して隣接timepoint間のedgeをscoreする。
- detection lossとedge lossを同時に最適化する。正解tracking graphが疎なため、edge lossはannotationがあるrowまたはcolumnを対象にし、未annotation cellを通常のnegativeとして全面的には扱わない。
- 公開training scriptの既定は50 epochs、AdamW、window size 2、空間downsample `(1,4,4)`である。best checkpointはvalidationのedge分類accuracyとnode recallの積で選ばれ、コンペのcombined metricを直接checkpoint selectionに使ってはいない。
- 推論ではcell候補とedge scoreからgraphを構築し、公開Notebookの多くはILPとgraph後処理を加える。
- 出典: [主催者baseline README](https://github.com/royerlab/kaggle-cell-tracking-competition)、[training script](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/scripts/train_unet_transformer.py)。

## Foldと公開Notebookのローカル評価

- 主催者training scriptはprecomputed JSONの`train` / `test`を読み、split index 0～4または`all`を受け付ける。splitの内容はJSON次第で、script自体は`embryo_id`の分離を検査しない。split JSONがなければ、seed 0でsampleをshuffleして90% train / 10% validationの単一split 0だけを生成する。
- 保存した上位15件では、学習ループ0件、`.fit()` 0件、`KFold` / `GroupKFold`等のfold API 0件だった。11件が`weights/unet_transformer/split_0/edge_predictor_best.pth`または同等のpathを参照した。
- 公開Notebookの推論用`kaggle_test_splits_50ep.json`は、hidden testの全sampleを`test`へ入れて予測scriptに渡すだけで、training foldを作るJSONではない。
- `Clean Approach + Lightweight Local CV`だけが固定8 sampleのローカル評価を含む。対象は`44b6` 4 sampleと`6bba` 4 sampleで、foldを反復しない。コードは`split_0` checkpointがこのholdoutに対応すると仮定するが、同じ胚の他sampleまで学習から除いたことはNotebook単体では確認できない。
- 同Notebookのmetric codeは主催者repositoryとcommitを記録する一方、`OFFICIAL_SPEC_EXACT_SOURCE_COPY = False`としている。主催者仕様を参考にした参加者実装であり、Kaggle hidden scorerそのものではない。

## 入力metadataの確認結果

- trainは199 image / 199 graphで1対1に対応する。胚IDは`44b6`と`6bba`の2個で、sample数は71と128。
- 全train imageは`(T,Z,Y,X)=(100,64,256,256)`、chunkは`(1,64,256,256)`、dtypeは`uint16`。
- 疎な正解は合計133,318 node、128,883 edge。`estimated_number_of_nodes`の合計は4,725,117。
- 公開testは4 sample。sample submissionは10列20行の小さな形式例で、本番submissionの行数を指定しない。
- 監査JSONのSHA256は`2a3d677afeab43320eb3098e628a583260afa94d334799cc2cdf5a5dd28f68fc`。competition raw image dataはローカルへ保存していない。

## 解釈

- 支持されたこと: 胚をまたがないvalidationをprimary候補にするなら2方向leave-one-embryo-outしか作れない。公開Notebookのfixed-8は、公開解法とのA/B比較用の補助diagnosticとしてのみ位置付けるのが妥当である。
- 否定されたこと: 「5-foldは胚数未確認の暫定値」という説明、および当リポジトリのCSV validatorを主催者提供物と受け取れる説明。胚数は確認済みで、5-foldはprimary候補として成立しない。
- ユーザー判断: primaryを2方向leave-one-embryo-out、公開Notebook互換の固定sample holdoutをsecondary validationとする。両者のscoreは混ぜずに記録する。

## 関連ファイル

- 実験の`result.md` / `metrics.json`: なし。
- 入力調査コード: [`input_metadata_audit.py`](../../studies/biohub_repository_setup/input_metadata_audit.py)、[`input_metadata_audit.ipynb`](../../studies/biohub_repository_setup/input_metadata_audit.ipynb)。
- 公開Notebook調査コード: [`inspect_public_notebooks.py`](../../studies/biohub_repository_setup/inspect_public_notebooks.py)。
- 生の出力: [`input_metadata_audit.json`](../../studies/biohub_repository_setup/input_metadata_audit.json)、[`public_notebook_code_inventory.json`](../../studies/biohub_repository_setup/public_notebook_code_inventory.json)。
- 保存した公開Notebook: [`docs/notebooks/biohub-cell-tracking-during-development`](../notebooks/biohub-cell-tracking-during-development/)。
- 公開Notebook: [Clean Approach + Lightweight Local CV](https://www.kaggle.com/code/yusuketogashi/clean-approach-lightweight-local-cv-no-hack)。

## 次のアクション

1. 対応済み: primary validationを2方向leave-one-embryo-outとし、各方向のscoreとTP/FP/FNを別々に保存する。
2. 対応済み: secondary validationとして公開Notebookと同じ固定8 sampleを使う。これはembryo-disjointではない可能性を明記し、primary scoreとは混ぜない。
3. 対応済み: 当リポジトリのCSV validatorを、可変行数、node/edge行の`-1`規則、連続`id`、edge参照整合性を確認する本コンペ用検証へ置き換えた。test Zarr metadataが利用可能な環境ではdataset網羅と座標範囲も確認する。
