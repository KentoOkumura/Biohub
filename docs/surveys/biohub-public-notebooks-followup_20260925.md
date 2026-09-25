---
title: Biohub 公開上位Notebookの再調査とx138重みの取得
date: '2026-09-25'
types:
- survey
hypotheses:
- HYP-20260910-12
experiments:
- exp043
topics:
- public-notebooks
- tracking
- validation
status: final
summary: 新たな上位Notebookはx138と同じ0.953で、コード構文・入力Dataset版・公開出力SHAが一致。前回403だった作者の座標補正重みを今回は取得できたため、exp043の自前重みとの比較が可能になった。
---

# Biohub 公開上位Notebookの再調査とx138重みの取得

確認日: 2026-09-25、19:15〜19:20 JST。前回の[x138と従来構成の比較](biohub-public-0953-differences_20260923.md)に続き、今回は新たな上位Notebookの独自差分と、作者の重みの取得可否を調べた。前回レポートの手法説明は引き続き参照できる。

## 結論

1. 新たにスコア順の先頭に出た[Biohub Cell Tracking / Kunal Desale](https://www.kaggle.com/code/kunaldesale2408/biohub-cell-tracking)のBest Public Scoreは **0.953**。前回のx138と同点で、0.953を上回ったNotebookではなかった。
2. Kunal版V10とx138 V1は、空白と空セルを除いた全12コードセルの構文、入力Datasetの4バージョン、GPU・internet設定が一致する。公開4動画の出力もログ記録のSHA256まで一致した。新モデル、再学習した重み、トラッカーの変更、新しい後処理は確認されなかった。
3. **前回403だった[x138作者の座標補正モデル](https://www.kaggle.com/datasets/anvithpothula/biohub-v1284-head-s075)を今回は実際にダウンロードできた。** Datasetはpublic、V1。これが今回の実験上の新しい材料である。
4. 現行ベースライン[exp043](../../experiments/exp043_x138_self_trained_head/result.md)は、自前で学習した座標補正重みを使ってPublic LB 0.950。作者の重みと同じ条件で比較できるようになったが、作者の0.953とこちらの0.950との差を重み単独の効果と確定する対照実験はまだない。

## 対象と証拠範囲

- 対応する上位仮説: `HYP-20260910-12`。公開構成と学習済み重みの再現性に関する証拠を追加する。上位仮説や実験の採否判断は変更しない。
- CLIでコンペに紐づく公開Notebookを`scoreDescending`順に30件取得し、上位5件のNotebookとmetadataを日付別に保存した。
- Kaggleの公開API `kernels.KernelsService/GetKernel`でBest Public Score、`GetKernelVersion`で現在の公開バージョン・完了実行・入力Dataset版を取得した。
- NotebookはJSONとPythonの構文木として静的に比較した。Notebookコードの実行、Kaggleでの新規推論、学習、submissionは行っていない。
- 重みのファイル取得と、CPU上の`torch.load(..., weights_only=True)`による内容検査を行った。保存先はGit対象外の`data/external/`。
- Web検索と[関連Discussion](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742266)も確認した。コメント中の高得点の言及やNotebook本文の自己申告を、公開実装と紐づいたスコアとして採用しなかった。
- この調査ではhidden testでの再現性、Private LB、追加モデル単独のスコア寄与は測っていない。

## 確認した上位Notebook

| Notebook | Best Public Score | 現在の公開版 | 確認結果 |
| --- | --- | --- | --- |
| [Biohub Cell Tracking / Kunal Desale](https://www.kaggle.com/code/kunaldesale2408/biohub-cell-tracking) | **0.953** | V10、9月24日 | 新たな上位掲載。x138と同じ処理・入力・公開出力 |
| [biohub x138 / Anvith Pothula](https://www.kaggle.com/code/anvithpothula/biohub-x138) | **0.953** | V1、9月21日 | 前回保存版とNotebook全体のSHA256も一致 |
| [Biohub Geometric Fusion / Aman](https://www.kaggle.com/code/amanatar/biohub-geometric-fusion) | 0.948 | V3、9月21日 | 上位スコアの更新なし |
| [Biohub 0.947 LB PROXY_SCORE=0.9490 / Evgen Dvorkin](https://www.kaggle.com/code/evgendvorkin/biohub-0-947-lb-proxy-score-0-9490) | 0.947 | V31 | Public Scoreは0.947。タイトルの0.9490は独自proxy |
| [Biohub: Harmonic Fusion V3 / Raunak Dey](https://www.kaggle.com/code/raunakdey07/biohub-harmonic-fusion-v3) | 0.947 | V5、9月25日 | 最新本文では0.953と記載するが、APIのBest Public Scoreは0.947。コードはx138の説明・コメント・表示を主に編集した構成 |

Best Public ScoreはNotebook全体の過去最高値であり、現在の公開版に対応する個別submissionの得点を必ず表すわけではない。特にRaunak版は本文と確認できるスコアを区別する。Evgen版はcurrentRunIdとmostRecentRunIdが異なるため、取得できた公開版を表に記載した。

スコア順一覧の掲載位置だけでは同点かどうか分からない。今回確認した上位5件では、前回の0.953を上回る公開スコアは確認できなかった。

## 新しいKunal版とx138の照合

[機械比較の結果](../../studies/biohub_public_notebooks_20260925/source_comparison.json)と[テキスト差分](../../studies/biohub_public_notebooks_20260925/x138_to_kunal.diff)を保存した。

| 比較対象 | 結果 |
| --- | --- |
| 非空のコードセル | 両方12セル。全セルのPython構文木が完全一致 |
| 実装上の差 | 空行の削除と末尾の空セル追加のみ |
| モデル構成 | 2つのTemporalUNet3D・対応予測トラッカー、DeepCenter、V1284座標補正モデルが同じ |
| primary / secondary / DeepCenter | 公開実行のreceiptに記録された3 checkpointのSHA256が一致 |
| V1284 | 同じDataset ID `12103746`、Dataset version source ID `19822532`を参照。旧receiptにはheadのSHAがないため、3重みのSHA一致と混同しない |
| トラッカー入力 | 同じV1284座標補正と、補正座標での三線形補間による画像特徴取得 |
| 接続選択・後処理 | 同じ整数線形計画法（ILP）、ハンガリアン法による再接続、検出点再追加、欠落補完・分裂修復 |
| 公開4動画の出力 | ログ記録では両方238,260行、同じCSV SHA256 |
| 公開4動画の統計 | `run_stats.csv`の差は予測時間・修復時間・経過時間の3項目だけ |
| 公開Notebookの実行時間 | x138: 1,228.45秒、Kunal版: 1,414.46秒。いずれも公開4動画を含む実行であり、submission採点所要時間ではない |

両ログのCSV SHA256:

```text
d52a5da2ae5cb0d1b22499f6ca51a00838a6c32ae9a1986ec756ea72e7909e03
```

ログのSHAを照合したものであり、今回はCSV本体を再取得してハッシュ計算していない。コード・入力・出力証拠が揃って一致するため、Kunal版を独自の新手法として移植する理由は見つからなかった。

Raunak版V5も同じ4つのDataset版を使う。x138との差分は説明セルの追加、コメント・docstring・ログ表示の編集、`time` importの移動が中心で、予測処理・後処理の新しい変更は見つからなかった。本文に書かれた自動後処理探索も、実設定では`BIOHUB_VALIDATOR_ENABLE=0`で無効である。[差分](../../studies/biohub_public_notebooks_20260925/x138_to_raunak.diff)を保存した。

## 作者の座標補正重みを取得できた

前回の[取得記録](../../studies/biohub_public_notebooks_20260923/x138_v1284_attachment.json)では、2026-09-23にDatasetのfiles/downloadが403だった。今回はfiles、metadata、実ファイルdownloadがすべて成功し、公開APIも`isPrivate=false`を返した。公開範囲が変わった正確な時刻は確認していない。Datasetの作成日時を公開切替日時とは扱わない。

| 項目 | 取得結果 |
| --- | --- |
| Dataset | `anvithpothula/biohub-v1284-head-s075` |
| 現行版 | V1、履歴もV1のみ。上位Notebookが参照するDatasetと一致 |
| ファイル | `v1284_head.pt`、33,913 bytes |
| 保存先 | `data/external/biohub-v1284-head-s075/v1284_head.pt`、Git対象外 |
| SHA256 | `625a0d9340f48193f2ec294fc2d81c5bb3c03087eab78ef0ae998a9c4c7da00c` |
| checkpointの内容 | `state_dict`、`mean`、`scale` |
| 学習可能パラメータ | 7,299個 |
| 層の形状 | 224入力→32中間→3出力。Notebookの定義では中間にSiLU活性化 |
| 特徴の正規化 | 平均・標準化用scaleとも224次元。全tensor有限、scaleは正 |
| ライセンス表示 | CC0 / Public Domain |

[重み検査結果](../../studies/biohub_public_notebooks_20260925/v1284_checkpoint_inspection.json)、[Datasetの版・公開状態](../../studies/biohub_public_notebooks_20260925/v1284_dataset_version.json)、[metadata](../../studies/biohub_public_notebooks_20260925/v1284_metadata/dataset-metadata.json)を保存した。

これは前回発見した追加モデルそのもの。中心と周辺6点の固定画像特徴から中心の移動量3成分を予測し、最大2 µm未満の補正を加えてからトラッカーへ渡す。新しい検出器や新しいトラッカーを追加したものではない。checkpointには学習コード、損失関数、動画一覧、fold情報は収録されておらず、それらの再現条件は引き続き未確認である。

## 現行exp043への意味と次の比較

[exp043の結果](../../experiments/exp043_x138_self_trained_head/result.md)と[metrics](../../experiments/exp043_x138_self_trained_head/metrics.json)では、作者のheadが取得できなかったため同じ構造のheadを自前学習し、Public LB 0.950を確認した。今回取得した作者のheadと、自前headのSHAは異なる。

- 自前head: `32d6c62f738c3dfe4862e3df5272850312e81b622e57d9ba45209ebb382824dc`
- 作者head: `625a0d9340f48193f2ec294fc2d81c5bb3c03087eab78ef0ae998a9c4c7da00c`

次に比較する価値があるのは、exp043の入力・トラッカー・後処理を固定し、headとそれに付属する特徴の平均・scaleだけを作者版へ替えた場合である。既知中心への位置誤差だけでなく、補正後の候補と画像特徴、接続得点、最終接続・分裂まで確認する。作者headの学習動画が不明なため、train上の比較を独立した交差検証と呼ばない。

この提案は現在のベースラインを自動的に置き換える判断ではない。同じNotebook構成でもhidden test再実行の差や実行条件の影響を分離していないため、0.953を保証したり、0.003の差をhead単独に帰属させたりできない。今回の依頼では調査と重み取得までを行い、バックログや実験契約は変更していない。

## 保存した証拠

- [上位30件の一覧](../../studies/biohub_public_notebooks_20260925/listing_score.json)
- [日付別Notebook保存先](../notebooks/biohub-cell-tracking-during-development/20260925/)
- [NotebookのSHAとmetadata](../../studies/biohub_public_notebooks_20260925/notebook_inventory.json)
- [スコア・公開版・入力版・完了状態](../../studies/biohub_public_notebooks_20260925/public_scores.json)
- [Kunal版の公開実行ログ](../../studies/biohub_public_notebooks_20260925/kunal_run.log)
- [3 checkpointの実行receipt](../../studies/biohub_public_notebooks_20260925/kunal_output/bidirectional_production_runtime_integrity.json)
- [静的抽出コード](../../studies/biohub_public_notebooks_20260925/inspect_notebooks.py)、[API取得コード](../../studies/biohub_public_notebooks_20260925/collect_public_metadata.py)、[構文比較コード](../../studies/biohub_public_notebooks_20260925/compare_sources.py)、[出力差分確認コード](../../studies/biohub_public_notebooks_20260925/inspect_output_difference.py)、[重み検査コード](../../studies/biohub_public_notebooks_20260925/inspect_head.py)
