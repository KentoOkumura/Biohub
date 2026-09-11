---
title: Biohub 最新公開ベースライン調査
date: '2026-09-10'
types:
- survey
hypotheses: []
experiments:
- exp001
- exp002
topics:
- public-notebooks
- baseline
- validation
status: final
summary: 公開比較対象はTemporalUNet3D・Node Transformerに複数モデルとTTA・graph補正を加えた0.94前後。0.946報告もあるが、最新スコアと独自proxyの扱いに注意が必要。
---

# Biohub 最新公開ベースライン調査

確認日: 2026-09-10。Kaggle APIによる更新一覧・コード確認と、検索エンジンが保持するKaggleページのスコア表示を区別する。

## 結論

1. 公開手法との比較には、TemporalUNet3DとNode Transformerに、2モデルの予測統合、test-time augmentation（TTA: 推論時に画像を反転・回転して予測を平均する処理）、時間の正逆方向の対応予測、整数線形計画法（ILP）による接続選択、軌跡・分裂の後処理を加えた構成が有用。確認できたページ表示はPublic LB（公開リーダーボードのスコア）0.941、更新Notebookのタイトルには0.942、参照先の結果として0.946の報告もある。
2. 学習から再現するための基本構成は引き続き主催者のTemporalUNet3DとSimpleNodeTransformer。ただし、それだけで上記の公開推論Notebookと同じスコアになる証拠はない。公開重みと後処理を含めた全体が比較対象になる。
3. 新しい学習方法として、FOCUS-3Dで密な疑似ラベルを作り、小さな検出モデルと対応予測モデルを学習する議論が進んでいる。これは有望な研究方向であり、検証済みの最良ベースラインとはまだ言えない。
4. 今回確認した5件の公開Notebookの独自PROXY_SCOREを、そのまま現在の公式指標や独立した交差検証（CV）として使うべきではない。分裂判定の実装と、検証データを学習から除いた保証に問題が残る。

- 対応する上位仮説: なし

この調査は既存仮説の支持・棄却や実験の採否を決めない。

## 公開Notebookとスコアの証拠

| 情報源 | 更新・確認範囲 | スコアと確度 | 位置付け |
| --- | --- | --- | --- |
| [Biohub LB : 941](https://www.kaggle.com/code/analyticaobscura/biohub-lb-941) | 9月10日の検索で、前日クロールのKaggleページを取得 | Public Score 0.941、Best Score 0.941 V1。表示実行時間36分40秒、T4 x2 | 0.94前後の公開結果を確認できる例。今回このNotebookのコードは取得していない |
| [Biohub 0.942 LB PROXY_SCORE=0.9417](https://www.kaggle.com/code/evgendvorkin/biohub-0-942-lb-proxy-score-0-9417) | APIの最終実行日時9月10日11:35 UTC、コード取得済み | 0.942は最新タイトル・作者の記録。現在のPublic Score欄は未確認 | 複数モデル、正逆方向の対応統合、DeepCenter、分裂後処理を説明した参照先 |
| [biohub-942tta](https://www.kaggle.com/code/redoctopusk/biohub-942tta) | APIの最終実行日時9月6日23:07 UTC、コード取得済み | 上のevgendvorkin Notebook本文で0.946と紹介。現在のPublic Score欄は未確認 | 対応予測に渡す特徴にもTTAを適用し、後処理を検証値で選択 |
| [Biohub Harmonic Fusion](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion) | APIの最終実行日時9月7日12:28 UTC、コード取得済み | 検索キャッシュはPublic 0.932 / Best 0.933 V23だが、最新コードのスコアとして使わない | 時間の正逆方向の確率を調和平均で統合する実装 |
| [Biohub Lineage Forge — Precision Tracking](https://www.kaggle.com/code/flexonafft/biohub-lineage-forge-precision-tracking) | APIの最終実行日時9月8日11:06 UTC、コード取得済み | 最新Public Scoreは未確認 | 2個目のモデルでも対応用特徴にTTAを適用。元の特徴との混合率0.75 |
| [biohub-lf-dctta-v020](https://www.kaggle.com/code/sjlee101/biohub-lf-dctta-v020) | APIの最終実行日時9月9日20:31 UTC、コード取得済み | 最新Public Scoreは未確認 | 上記にDeepCenterのTTAと、分裂候補を除く閾値0.20を追加 |

これらは全参加者の最良スコアを確定した一覧ではない。CLIのスコア降順には7月のmetric-hack Notebookが上位に残っていた。8月22日の[Public Notebook Rankings Need a Metric Refresh](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/736937)でも、指標修正後の再採点がNotebookの古い表示値へ反映されていない問題が報告されている。タイトル、古いBest Score、現在のPublic Score、作者の実験報告は同一視しない。

## 現在の公開構成

今回コードを確認した5件は、主催者のunet_transformerを基礎に以下を組み合わせている。

- 入力は連続時点の3D画像。TemporalUNet3Dが細胞中心の検出値とvoxel特徴を出力し、Node Transformerが隣接時点の細胞候補間の接続を予測する。
- 主モデルはpilkwangのBiohub Tracking Support Pack、2個目はBiohub TemporalUNet3D Seed 314159 V1の学習済み重みを読む。入力dataset名や50epというファイル名だけから、実際の学習epoch数・全fold履歴を確定しない。
- 画像の検出値だけでなく、対応予測に渡す特徴も8通りの反転・回転から平均する。Lineage Forgeとdctta版は2個目のモデルにもこの処理を加える。
- 細胞間の対応を時間の順方向と逆方向で予測し、片方だけが高い候補を抑えるよう確率を調和平均でまとめる。
- ILPでgraphを選び、運動方向を使った再接続、1～2フレームの途切れの補間、短い軌跡の除去・救済、娘細胞候補の距離・対称性による分裂補正を行う。
- 参加者提供のDeepCenterUNet3Dで、補間した細胞や追加した分裂の位置を確認し、不支持の候補を除く。dctta版はこの確認用モデルにもTTAを適用する。

この5件は主に学習済み重みを使う推論コードであり、同じ条件の学習からPublic LBまで今回再実行したものではない。公開ページの約30～40分という時間も、そのページに表示されたNotebook実行時間であり、hidden test全体の実行時間を保証しない。

## 最新discussionから分かること

| 投稿 | 日付 | 内容 | 解釈 |
| --- | --- | --- | --- |
| [Post-Processing Plateau?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/740103) | 9月8日 | 投稿者が後処理を試しても0.942で停滞と報告 | 0.942付近が比較対象になっている一例。全員の性能上限ではない |
| [magic or overfitting?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/740145) | 9月8日、返信9日 | hengck23が位置補正後のnode recall 99.45%等を報告。改善した検出は密な学習用サンプル作成に利用すると説明 | 検出位置の改善と対応候補の取りこぼし削減に注目。ただし公式LBの実証とは別 |
| [focus3d : one of the best 3d cell segmentation](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738217) | 8月30日、最新返信9月10日 | FOCUS-3Dの密な疑似ラベル、3D検出器への学習、密な軌跡の生成を議論。直接実行の時間超過や主催者座標とのずれも報告 | 事前にラベルを作って学習する用途に価値がありそう。高いLBを再現した標準構成とは未確認 |
| [How are your local CVs like?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739352) | 9月3日、返信4～6日 | CV 0.96 / LB 0.93、localで0.02～0.04改善してLB改善なし等の報告 | 合算スコアだけでなく検出、接続、分裂の誤りを分けて比較すべき |
| [Division steps are not long steps](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/740573) | 9月10日 | 73ファイル・36分裂の正解ラベルを分析。分裂時の移動距離は通常移動とも重なる。予測座標から計測すると位置誤差の影響を受ける | 分裂判定を距離だけで改善するには限界がある。小規模で選択に偏りがあるラベル集合の分析として読む |

FOCUS-3D、Harmonic Fusion、Lineage Forge、DeepCenterなどの名称は、各公開手法・Notebook・datasetの名称である。

## PROXY_SCOREと公式指標の差

9月10日に取得した[主催者metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)では、分裂評価は分裂前後の局所的な有向構造、異なる2本の娘細胞系列、予測分裂と正解分裂の一対一の対応、枝の合流の排除などを要求する。単に同じ弱連結成分に入るだけでは不十分と明記されている。

一方、今回の5件のcompute_division_confusionは、予測graphの弱連結成分を作り、正解の各娘細胞から全子孫を探索し、親側と両娘系列の対応nodeおよび分岐が同じ成分へ入るかで判定する。これは現在の主催者の局所判定とは異なる。Notebook本文にofficialとあっても、PROXY_SCOREを公式値とは扱えない。この調査は静的な実装差の確認であり、同じ予測graphで差分を数値測定したものではない。

また、検証コードはtrainからtestと同名の動画を除き、分裂を含む動画を優先して胚ごとに4本選ぶが、それらが読み込んだcheckpointの学習集合から除外されているか照合していない。held-outという表示だけで学習から独立したCVだと判断しない。

8月31日の[0.94以上の検出器に関する投稿](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738276)には「LBはdense labels」とする返信もあるが、根拠を問う返信があり、公式確認を得ていないため本調査の前提には採用しない。

## このリポジトリの状況

9月10日の参照時点で、数値の正本であるmetrics.jsonでは以下の状態だった。

- [exp001](../../experiments/exp001_temporal_unet3d_baseline/metrics.json): 学習時のGPUメモリ不足によりfailed。CV・Public LBは未取得。
- [exp002](../../experiments/exp002_unet3d_expandable_segments/metrics.json): running。メモリ割当設定を変更し、version 1のsmokeは通過。ユーザー承認によりversion 2では12時間を上限にbatch size 16、3 epochsの学習を実行中と記録。CV・Public LBは未取得。

exp002のresult本文や方針索引にはversion 1の11時間gate停止の説明も残っていたが、それだけで現在停止中とは判断しない。最新metricsの記録を優先した。今回Kaggle上のprivate実験のログは取得していないため、ここでいう実行中はローカル記録の状態である。

したがって、このリポジトリではまだ実測スコア付きのベースラインは成立していない。現在の学習実験は主催者基本構成の再現に当たり、公開0.94前後の推論構成に含まれる複数モデルや追加後処理の比較は今後の検証になる。

## 証拠範囲と再利用

- Kaggle API: recent順discussionの先頭2ページ40件、公開NotebookのscoreDescending順30件、dateRun順30件、pilkwangの最新一覧。
- 関連投稿の本文と返信を追加確認。JSON表示では最初の投稿本文が省略されるため、通常のtopics show表示も確認した。
- 上記5件のNotebookを一時領域へpullし、コードを実行せずに読み取った。
- [NotebookのSHA・入力・設定の一覧](../../studies/biohub_public_baselines_20260910/notebook_inventory.json)、[取得一覧と取得上の制約](../../studies/biohub_public_baselines_20260910/retrieval_evidence.json)、[静的抽出コード](../../studies/biohub_public_baselines_20260910/inspect_notebooks.py)。
- 新しいNotebookの現在スコア欄は取得できなかった。CLI metadataにもスコアフィールドがなく、通常の公開ページAPIも403だったため、タイトルや並び順から補完しない。
- 公式指標の定義、投稿者の自己報告、コードからの推論、Kaggleページのキャッシュ上の表示値を区別した。
- 実験・バックログの設定変更、Kaggle学習・提出、実験の採否決定は行っていない。
