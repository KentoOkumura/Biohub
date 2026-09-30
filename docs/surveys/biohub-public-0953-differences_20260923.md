---
title: Biohub 公開0.953 Notebookと従来0.946構成の差分
date: '2026-09-23'
types:
- survey
hypotheses:
- HYP-20260910-12
experiments:
- exp013
- exp016
topics:
- public-notebooks
- tracking
- validation
status: final
summary: 公開x138のBest Public Scoreは0.953。既存3重みに加えて学習済み座標補正モデルを使い、トラッカー入力も変更する。近傍移動量による再接続・検出点の再追加・欠落補完も追加。Geometric Fusionは0.948で後処理探索が中心。
---

# Biohub 公開0.953 Notebookと従来0.946構成の差分

確認日: 2026-09-23。Kaggle公開APIのスコア、取得コード、作者の公開実行ログを照合した。

訂正（同日）: 初回調査では学習済み座標補正モデルを見落とし、x138の差分を後処理中心と説明した。保存済みコードと実行ログの再確認で、追加モデルがトラッカーより前で有効になっていることを確認した。既存3 checkpointの一致だけでは、使用する全モデルやトラッカー出力の一致を示せない。

## 結論

1. [biohub x138](https://www.kaggle.com/code/anvithpothula/biohub-x138)の **Best Public Scoreは0.953**。現行公開版はV1、9月21日更新。従来の公開0.947群より0.006、こちらの[exp013の実提出0.944](../../experiments/exp013_public_notebook_replay/metrics.json)より0.009高い。ただし今回こちらで再提出した結果ではない。
2. 主要な変更は、学習済みの小さなニューラルネットワークで検出座標を補正してからトラッカーへ渡すこと、近くの細胞の移動から次位置を予測して接続を選び直すこと、整数線形計画法（ILP）が落とした検出点を再追加すること、検出得点が低い候補を使って最大3フレームの欠落を埋めること。既存3 checkpointはexp011とSHA256まで一致するが、それらとは別に座標補正用の学習済み重みを使用する。
3. [Biohub Geometric Fusion](https://www.kaggle.com/code/amanatar/biohub-geometric-fusion)はBest Public Score 0.948。現行V3では、弱い接続で終わる点の削除、分裂の距離・検出得点条件の緩和、移動量の外挿係数の変更を、8動画の独自指標で選んでいる。
4. 元の公開検出器と画像encoderの重みは固定されているが、追加の学習済み座標補正モデル、補正座標での特徴取得、候補回収を含む。固定候補・固定画像特徴からトラッカーだけを学習する現在の実験条件と同一ではない。採用案を作る場合は、追加モデルと入力変更を明示する必要がある。

## 調査目的と証拠範囲

- 対応する上位仮説: `HYP-20260910-12`。調査後のユーザー依頼で、公開構成の再現と保存済みtrackerの比較を同仮説の候補へ引き継いだ。
- 対象: 新しい公開Notebookと、公開構成を再現したexp013、固定画像特徴からprimaryトラッカーを学習するexp016の相違。
- 取得範囲: CLIのscoreDescending順30件を確認し、先頭7件のNotebookとmetadataを保存。最近のDiscussion一覧と関連2投稿を確認した。
- スコア: Kaggle Webページが使用する読み取りAPI `kernels.KernelsService/GetKernel` の `bestPublicScore`。CLI一覧はスコア値を返さないため、順位やタイトルから推測していない。
- 版・実行時間: `GetKernelVersion` の公開run情報。上位2件の保存済みstdoutを `kaggle kernels logs -f` で取得した。取得Notebookのoutputsは空だったため、コードだけで実行成功を判定していない。
- 未確認: hidden testの予測内訳、Private LB、各変更の単独の効果、こちらの環境での再現性。

| 公開Notebook | Best Public Score | コード確認版・日付 | 位置付け |
| --- | --- | --- | --- |
| [biohub x138](https://www.kaggle.com/code/anvithpothula/biohub-x138) | 0.953 | V1 / 9月21日 | 学習済み座標補正と、近傍移動量・候補回収・欠落補完を追加 |
| [Biohub Geometric Fusion](https://www.kaggle.com/code/amanatar/biohub-geometric-fusion) | 0.948 | V3 / 9月21日 | 後処理の候補を拡張して選択 |
| [Biohub 0.947 LB PROXY_SCORE=0.9490](https://www.kaggle.com/code/evgendvorkin/biohub-0-947-lb-proxy-score-0-9490) | 0.947 | 9月23日取得 | 従来構成 |
| [Biohub: Harmonic Fusion V3](https://www.kaggle.com/code/raunakdey07/biohub-harmonic-fusion-v3) | 0.947 | 9月23日取得 | 従来構成 |
| [Biohub Harmonic Fusion](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion) | 0.947 | 9月23日取得 | 従来構成 |
| [Biohub Cell Tracking: 0.947 LB](https://www.kaggle.com/code/reyhanksatria/biohub-cell-tracking-0-947-lb) | 0.947 | 9月13日更新 / 9月23日取得 | 採用した0.946 Notebookの後続版 |
| [Biohub Lineage Forge — Precision Tracking](https://www.kaggle.com/code/flexonafft/biohub-lineage-forge-precision-tracking) | 0.947 | 9月23日取得 | 従来構成 |

Best Public ScoreはNotebook全体の最良値。複数版のあるGeometric Fusion等では、今回確認した現行コードの単一実行の提出値と同一だとは断定しない。x138は現行V1。確認時刻とAPI項目は[public_scores.json](../../studies/biohub_public_notebooks_20260923/public_scores.json)を参照。

## 共通部分と0.946から0.947への差

入力は隣接2フレームの3D画像。2つのTemporalUNet3Dが検出値と画像特徴を出し、SimpleNodeTransformerが細胞候補間の対応を予測する。時間の正逆方向の対応確率を統合し、ILP、運動に基づく接続の組み直し、途切れ・分裂の修復、短い軌跡の処理を行う。DeepCenterUNet3Dは追加候補の画像上の裏付けを確認する。

exp013で固定した旧0.946 sourceではprimaryの対応用画像特徴だけにTTA（test-time augmentation、画像を反転・回転して推論し平均する処理）を適用する。現在の0.947群および上位2件は、secondaryの対応用特徴にも8方向のTTAを追加し、TTA特徴75%と元特徴25%を混合する。DeepCenterにも8方向のTTAを追加する。この差は9月10日の調査時に一部派生版で確認済みであり、x138固有の変更ではない。

[従来構成の解説](biohub-cell-tracking-0946-notebook-explanation_20260913.md)、[公開重み選定記録](../../experiments/exp011_public_detector_selection/assets/public_detector_selection.json)、[今回取得した0.947 source](../notebooks/biohub-cell-tracking-during-development/reyhanksatria__biohub-cell-tracking-0-947-lb/biohub-cell-tracking-0-947-lb.ipynb)を照合した。

## x138の変更

### トラッカーより前の学習済み座標補正

x138には、作者が `V1284` と呼ぶ座標補正用のモデルが追加されている。検出した点の中心と周囲6点の固定画像特徴から224次元の入力を作り、`Linear(224,32) → SiLU → Linear(32,3)` で3軸の座標補正量を予測する。補正量は物理距離2 µm未満に制限し、補正後の座標をトラッカーへ渡す。同時に特徴取得を3次元の線形補間へ置き換えているため、接続予測に使う座標と画像特徴の両方が変わる。既存トラッカーの構造と重みが同じでも、その出力を同じとは扱えない。

読み込む重みは `biohub-v1284-head-s075/v1284_head.pt`。コードは `V1284_MODE='candidate'` を指定し、保存済み公開ログにも有効化と補間コード実行の記録がある。モデル重み自体のSHAと学習コードは今回未取得。追記: 公開実行の生metadataから入力dataset `anvithpothula/biohub-v1284-head-s075` とDataset Version ID `19822532` を確認したが、取得APIは403で重み本体を取得できていない。[入力metadataの保存記録](../../studies/biohub_public_notebooks_20260923/x138_v1284_attachment.json)を参照。20 train動画の4,136対応点で学習したという来歴は作者のコードコメントによる説明であり、独立検証した事実ではない。座標補正単独のLB寄与も未確認である。

証拠: [Notebookから静的抽出した補正モジュール](../../studies/biohub_public_notebooks_20260923/x138_v1284_coordinate_refinement.py)、[組み込みコード](../../studies/biohub_public_notebooks_20260923/anvithpothula__biohub-x138.code.py)、[公開実行ログ](../../studies/biohub_public_notebooks_20260923/x138_run.log)。

### トラッカー後の処理と実行制御

| 処理 | 従来の処理 | x138で追加された処理 |
| --- | --- | --- |
| 次位置の予測 | 自分の直前の移動量を使って外挿 | まず近距離の仮接続を作り、40 µm以内の最大12近傍の移動量の中央値を利用。自身の仮接続が自己支持しないよう近すぎる点を除外 |
| 接続候補の選択 | 元の位置からの距離を主な候補条件にする | 元位置の周辺に加え、近傍移動量から予測した位置の周辺も候補に含め、ハンガリアン法で一対一の対応を選ぶ |
| 落ちた検出点の回収 | ILP後に残った点を主に使う | 途切れた軌跡の前後4 µm以内にある検出得点0.965以上の未使用点を戻し、接続を再計算 |
| 欠落フレームの補完 | 従来の1～2フレームの補間・画像による位置補正 | 検出得点0.3以上の極大値を保存し、そのうち0.5以上の点で最大3フレームを補完。予測経路から3.5 µm以内、既存点から2 µm超、追加数は処理時点の点数の3%以内 |
| 新しい補完で画像候補がない場合 | 従来の補完では補間点を作れる | 新しく追加した補完では全欠落フレームに検出候補を要求。画像候補なしの合成点を許可する設定は0。既存の補間処理は残る |
| 実行時間の制御 | train上の独自指標と設定探索も実行 | train上の独自指標を無効化。ILPは動画ごとに1200秒上限、7.5時間経過後は重い修復の一部を省略 |

この表の後処理の入口は `motion_relink_edges`、`readmit_discarded_detections`、`fill_gaps_from_low_detections`、`filter_output_graph`。これらはモデルの推論結果を使うルールによる処理であり、前述の学習済み座標補正とは区別する。近傍が不足すると自身の過去の移動量などへ戻る。

## Geometric Fusion V3の変更

24個の後処理候補と改善候補の組合せを、train内8動画の独自 `PROXY_SCORE` で比較する。選択には全体改善0.001以上、adjusted edge Jaccardの悪化0.0005以内、各胚の独自指標の悪化0.001以内という条件を使う。

公開ログで実際に選ばれた変更は次の7設定だった。

- 接続の近距離条件: 6.0 → 5.5 µm。
- 弱い終端点の削除: incoming edgeの得点0.3未満で、後続点がなく、最終フレームでもない点を1回だけ削除する。接続得点のない追加分裂枝などはこの条件では削除しない。
- 母から新しい娘への距離上限: 9 → 11 µm。
- 娘同士の距離上限: 14 → 16 µm。
- 母から既存の娘への距離上限: 10 → 12 µm。
- 分裂追加時のDeepCenter得点下限: 0.25 → 0.15。
- 自身の過去の移動量を外挿する係数: 0.5 → 0.25。

分裂を取り逃した原因を「娘の検出不足」「点はあるが未接続」等へ分ける診断も追加している。x138の近傍移動量推定、未使用検出点の回収、低得点候補での欠落補完は含まない。名称にGeometricとあるが、時間の正逆方向の対応確率は引き続き調和平均で統合する。

## 作者の実行で確認できたこと

x138の公開実行で、既存3 checkpointのSHA256を[exp011の固定値](../../experiments/exp011_public_detector_selection/assets/public_detector_selection.json)と機械照合し、完全一致を確認した。証拠は[実行時の重み記録](../../studies/biohub_public_notebooks_20260923/x138_output/bidirectional_production_runtime_integrity.json)。この記録は追加の座標補正モデルを対象に含まないため、全モデルの重みが従来と同じだという証拠にはならない。

公開test 4動画のログと `run_stats.csv` には、新しい候補回収処理が実際に動いた記録がある。

| 公開test動画 | 未使用検出点の再追加 | 低得点候補による欠落補完の追加点 |
| --- | ---: | ---: |
| 44b6_0113de3b | 94 | 3 |
| 44b6_0b24845f | 662 | 67 |
| 6bba_05b6850b | 84 | 3 |
| 6bba_05db0fb1 | 497 | 37 |
| 合計 | 1,337 | 110 |

これらは各処理の追加数であり、最終的に残った正しい細胞数ではない。その後の短い軌跡除去等で再度消える点もある。最終出力は238,260行。公開実行全体は1,228.45秒、画像・対応予測区間はログ上9.38分だった。hidden test全体の所要時間ではない。

Geometric Fusion V3では独自 `PROXY_SCORE` が0.9490 → 0.9530、adjusted edge Jaccardが0.9260 → 0.9300。division Jaccardは0.2308、正検出・誤検出・見逃しは3 / 1 / 9で変化しなかった。分裂条件を緩和した設定が選ばれていても、この8動画で分裂指標が改善した証拠はない。APIのBest Public Scoreは0.948であり、独自指標0.9530と区別する。公開実行全体は14,216.86秒、約3時間57分で、大部分がtrainの診断と設定探索に使われている。

## 検証上の制限

1. 独自 `PROXY_SCORE` の分裂判定は、前回調査と同じく弱連結成分と正解娘の子孫探索を使っており、現在の公式の局所構造判定とは異なる。[前回の実装照合](biohub-public-baselines_20260910.md#proxy_scoreと公式指標の差)を参照。
2. secondary公開重みは全199 train動画で学習済み。primaryの独立検証来歴も保証されない。8動画を選んでも独立した交差検証（CV）にはならない。今回の[Discussion 742064](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742064)も同じ問題と、train内の改善がLBで悪化した例を報告している。これは既にexp011で確認した前提を補強する。
3. [Discussion 742266](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742266)では、train内の接続指標が上がってもLBで悪化した10回の変更を参加者が報告している。全手法に一般化する実験ではないが、局所指標だけで後処理を採用できない根拠になる。
4. x138の0.953は複数変更をまとめた構成の値であり、どの変更が何点を稼いだかを示す独立比較は公開資料から確認できなかった。

## このリポジトリへの含意

- 既存公開重みに、学習済み座標補正と候補回収・接続の組み直しを加えたNotebookが0.953を記録している。後処理だけによる改善と断定できず、各変更の寄与は切り分けていない。
- 従来・今回とも、ILPの接続は後段の運動に基づくハンガリアン割当で置き換えられる。学習モデルの接続確率は費用の補助項として使われ、今回の実装でも全候補の学習済み接続得点を最終段へ渡すように変えたわけではない。
- exp016等で隣接2-frameの対応学習を改善しても、最終の接続組み直しによって効果が小さくなる可能性がある。これは今回の実装からの推論であり、既存学習実験の不採用理由を確定するものではない。
- 未解決なのは、x138の再現性、hidden testにおける分裂と通常接続への寄与、誤った近傍移動量による悪化、追加候補による過検出、処理時間上限に達した際の性能。

今後比較するなら、まずx138全体の再現を確認し、学習済み座標補正、近傍移動量、未使用点の回収、低得点候補での欠落補完を個別に外して寄与を測る価値がある。公開detectorの重みは固定できるが、座標補正を含める場合は追加モデルの使用を明示する。

ただし低得点候補の回収には、従来の高閾値候補だけのcacheでは不足する。補正後の座標で取得する特徴とsecondaryの対応特徴TTAも旧cacheと同じ特徴定義ではない。既存実験の入力を黙って差し替えず、追加モデル・候補・特徴・復号の変更を実験契約で明示する必要がある。調査後の2026-09-23、ユーザー依頼で全体再現と公開/exp016 tracker重みの比較を`HYP-20260910-12`の候補にした。全体再現は[`exp042_public_x138_replay`](../../experiments/exp042_public_x138_replay/)へ実験化してNotebookを実装した。追加重み未取得のため実行・提出は行っていない。重み比較は[`public_x138_tracker_comparison`](../../backlog/public_x138_tracker_comparison.md)として未着手で、追加学習は含めない。

## 関連ファイル

- [exp013の結果](../../experiments/exp013_public_notebook_replay/result.md)、[数値](../../experiments/exp013_public_notebook_replay/metrics.json)、[固定した旧Notebook](../../experiments/exp013_public_notebook_replay/assets/reference_notebook/biohub-cell-tracking-0-947-lb.ipynb)。ファイル名は0.947だが、固定SHAの内容と契約は旧0.946構成を指す。
- [exp016の結果](../../experiments/exp016_frozen_image_encoder/result.md)、[現在の学習方針](../../backlog/KAGGLE_DIRECTION.md#今後の学習方針)。
- [取得Notebook一覧とSHA](../../studies/biohub_public_notebooks_20260923/notebook_inventory.json)、[公式Web APIのスコアと版](../../studies/biohub_public_notebooks_20260923/public_scores.json)。
- [x138の抽出source](../../studies/biohub_public_notebooks_20260923/anvithpothula__biohub-x138.code.py)、[公開ログ](../../studies/biohub_public_notebooks_20260923/x138_run.log)、[統計CSV](../../studies/biohub_public_notebooks_20260923/x138_output/run_stats.csv)。
- [Geometric Fusionの抽出source](../../studies/biohub_public_notebooks_20260923/amanatar__biohub-geometric-fusion.code.py)、[公開ログ](../../studies/biohub_public_notebooks_20260923/geometric_fusion_run.log)。
- [静的抽出スクリプト](../../studies/biohub_public_notebooks_20260923/inspect_notebooks.py)、[公開metadata取得と重み照合](../../studies/biohub_public_notebooks_20260923/collect_public_metadata.py)。取得Notebookのコードは実行していない。
