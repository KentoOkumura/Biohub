---
title: Biohub 公開検出器・トラッカー選定
date: '2026-09-12'
types:
- survey
hypotheses:
- HYP-20260910-12
experiments:
- exp011
topics:
- public-notebooks
- baseline
- model-selection
status: final
summary: 公開0.946 Notebookの推論構成を参照し、Pilkwang公開dataset 3件の現行版とcheckpoint SHAを固定。2026-09-12にユーザー採用済み。
---

# Biohub 公開検出器・トラッカー選定

作成日: 2026-09-12

## 結論

初回の固定検出器・画像特徴抽出器として、[Biohub Cell Tracking: 0.946 LB](https://www.kaggle.com/code/reyhanksatria/biohub-cell-tracking-0-946-lb)の2026-09-12取得版を処理構成の参照にすることを推奨する。取得したNotebookはkernel id `133199516`、content SHA `ae8e01a262211045161984e469e8be23e3386bab9140fe12df503dc6a1e010e6`で識別する。

checkpointはNotebook内のmirror名ではなく、同じ期待SHAを持つPilkwangの元dataset 3件から取得する。primary `TemporalUNet3D + SimpleNodeTransformer`、secondary `TemporalUNet3D + SimpleNodeTransformer`、repair用DeepCenterを一組として固定する。初回の下流学習ではprimary checkpointの`SimpleNodeTransformer`だけを学習対象とし、primary image encoderとdetect head、secondary branch全体、DeepCenter、candidate detection、ILP、graph repairを固定する。

これは2026-09-12にユーザーが採用した構成である。0.946は公開pipeline全体の報告で、detector単体の精度ではない。現行Public Scoreの独立取得、full inference、hidden test runtime測定、original dataset mountでの動作確認は実施していない。

## 調査目的

公開検出器と画像特徴抽出器を更新せずtrackerだけを学習する方針に対し、使用するNotebook、checkpoint、時間窓、前処理、候補点特徴、既存tracker初期値、学習来歴、利用条件を一意に指定する。全公開モデルの再学習や全件推論を選定の前提にはしない。

## 対象と証拠範囲

- 対応する上位仮説: `HYP-20260910-12`
- 対象実験: [`exp011_public_detector_selection`](../../experiments/exp011_public_detector_selection/)
- 入力: Kaggle CLIで2026-09-12に取得した公開Notebook 4件、公開dataset 3件のmetadata・artifact manifest、実際にdownloadしたcheckpoint 3件。
- 評価方法: Notebook sourceからmodel、前処理、特徴抽出、association、decode、repairを比較し、checkpoint SHAをNotebook内の期待値とdownloadした実ファイルで照合した。
- このレポートからは判断できないこと: detector単体精度、0.946 Public LBの独立再現、候補数・特徴保存量、hidden test runtime、固定特徴で再学習したtrackerの精度。

## 比較した公開Notebook

| Notebook | 取得時content SHA | 初回controlでの扱い |
| --- | --- | --- |
| `reyhanksatria/biohub-cell-tracking-0-946-lb` | `ae8e01a262211045161984e469e8be23e3386bab9140fe12df503dc6a1e010e6` | 推奨。0.946構成の処理と期待checkpoint SHAを固定する参照。 |
| `sjlee101/biohub-lf-dctta-v020` | `7129370737acb810fc8aa1dc0b515e782571ad61f9cc0b689d777f2c257fb012` | secondary association-feature TTAとDeepCenter TTAを含むため、後続のcontrolled variantへ残す。 |
| `flexonafft/biohub-lineage-forge-precision-tracking` | `f1c7d777e5c41a937ce4b679f65f2e0d5795aec411e1cccda99487a86b06dbb2` | 有用な参照実装だが、取得時に現行Public Scoreを確認できず、0.946 Notebookの方が後続差分を特定しやすい。 |
| `redoctopusk/biohub-942tta` | `521cb97f0f457643379a51b60c4f71e3f4cc7d1823fd98cbb97633ffaa515ec4` | 近い構成だが、0.942は上流の報告であり、0.946取得版を初回controlにする。 |

Kaggle CLIのpull metadataにはkernel version番号が含まれなかった。そのため、Notebookはkernel id、一覧取得時のlast run time、取得したcontent SHAで固定する。題名とNotebook本文の`verified_public_lb_0946`記録は0.946の根拠になるが、Kaggleの現行Public Score欄は独立取得できていない。

## 固定する公開artifact

| role | datasetと現行version | checkpoint | SHA | 確認できた来歴 |
| --- | --- | --- | --- | --- |
| primary detector・image encoder・tracker初期値 | [`pilkwang/biohub-tracking-support-pack-50ep-v1`](https://www.kaggle.com/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1) v10 | `weights/unet_transformer/split_0/edge_predictor_best.pth` | `12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771` | dataset version noteは400 epoch snapshot。best epoch、split、seed、独立validationはmanifestにない。 |
| secondary detector・image encoder・固定tracker branch | [`pilkwang/biohub-temporal-unet3d-seed314159-v1`](https://www.kaggle.com/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1) v2 | `edge_predictor_best.pth` | `9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f` | seed 314159、snapshot epoch 400、best epoch 381、internal best score 0.9779747766406395。199 train datasets、40 validation datasets、独立validationではなく、deterministic trainingでもない。 |
| conservative repair gate | [`pilkwang/biohub-deepcenter-unet3d-center-prior-v1`](https://www.kaggle.com/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1) v5 | `weights/full_frame_center/best.pt` | `8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0` | best checkpointはepoch 2。epoch 500のlast checkpointは選ばない。sparse center heatmapに対するpositive-unlabeled weighted BCEで学習されたrepair gate。 |

3 datasetの表示licenseはCC0-1.0である。実際にdownloadした3 checkpointのSHAは、0.946 Notebook内の期待値とすべて一致した。artifact manifest SHAやbyte sizeを含む完全な値は[`public_detector_selection.json`](../../experiments/exp011_public_detector_selection/assets/public_detector_selection.json)を正とする。

Notebookそのもののcode licenseはpullしたmetadataとsourceから確認できなかった。したがってNotebookはKaggle上の参照または処理構成の記録として扱い、Notebook独自patchをリポジトリへコピーする場合はlicense確認を先に行う。

## 入力・特徴・trackerの対応

primaryとsecondaryの`TemporalUNet3D`は同じmovieの連続2 frameを入力にし、z/y/x方向を1/4/4 strideで読む。movie全体のZarr `image_statistics.quantiles`にある0.001と0.999 quantileでnormalizeし、0未満をclampする。両branchのimage encoder出力は32 channelsである。

primary branchは8個のD4 viewについてdetection logitsと32-channel feature mapを元の向きへ戻して平均する。候補点座標で32-channel featureをinteger indexingする。secondary branchはdetection logitsを8-view平均するが、0.946取得版ではassociation featureを8-view平均せずoriginal spatial viewから取り出す。

各branchのtracker inputは32-channel image featureと32-dimensional window-relative positional encodingの連結である。`SimpleNodeTransformer`はsource-target間のrelative z/y/x displacementも使ってassociationをscoreする。公開学習sourceではsource軸softmax後のfocal-weighted binary cross entropyと、annotated sourceまたはtargetに触れるpairのmaskを使う。後続の`frozen_image_encoder`ではsparse labelに対するteacher matchingとmaskを学習前にauditする必要がある。

0.946構成はprimary・secondaryの2 tracker branch、forward/reverseのharmonic mutual support、primaryのlow-margin agreementに限定したsecondary contribution、ILP、motion relinking、gap・division repair、short-track filtering、DeepCenter vetoを使う。初回比較ではこれらを変更しない。

## 資源・validation・leakage

- 選定調査はGPUを使わず、trackerやdetectorを学習していない。
- 取得時metadataのNotebook acceleratorはNvidia Tesla T4。保存済み公開調査には関連ページの30–40分程度の表示runtimeがあるが、selected hidden test全件の実測ではない。
- primary checkpointの詳細なsplit・seed・独立validationは不明である。secondary checkpointも独立validationではない。両者を独立評価済みdetectorと扱わない。
- 公開0.946はdetector、association、複数model、ILP、repairを含む全体値であり、detector単体の性能へ帰属しない。
- original Pilkwang datasetをmountした小規模Kaggle runで、3 SHA、import、候補feature shape、candidate数、feature storage、実行時間を確認する余地がある。この確認は選定調査の完了条件にはせず、採用後の実行判断へ渡す。

## 解釈

- 支持されたこと: 取得可能な公開checkpointと公開sourceから、画像側を更新せずcandidate featureを取得し、既存`SimpleNodeTransformer`を初期値にする具体的な後続実験を定義できる。
- 否定されたこと: Notebook題名のscoreだけで選ぶこと、公開checkpointを自前重みで代用すること、全公開modelの再学習を選定の前提にすること。
- 未解決: Notebook独自patchのcode license、original dataset mountでの動作、candidate-feature保存量、hidden test runtime、tracker再学習の精度。

## 関連ファイル

- 実験の記録: [`result.md`](../../experiments/exp011_public_detector_selection/result.md)、[`metrics.json`](../../experiments/exp011_public_detector_selection/metrics.json)
- 機械可読な選定内容: [`public_detector_selection.json`](../../experiments/exp011_public_detector_selection/assets/public_detector_selection.json)
- 選定Notebookの処理解説: [`biohub-cell-tracking-0946-notebook-explanation_20260913.md`](biohub-cell-tracking-0946-notebook-explanation_20260913.md)
- 調査コード: なし。Kaggle CLIで取得し、Notebook source・metadata・manifestを静的に照合した。
- 生の取得物: `/tmp/public_detector_selection_20260912/`。一時調査物でありGitには保存しない。

## 次のアクション

採用済みmanifestを`exact_window_cache`、公開基準のdiagnostic、`frozen_image_encoder`へ渡す。まずoriginal Pilkwang dataset mountで小規模動作確認を行い、tracker学習やsubmissionは別途承認された実験で扱う。
