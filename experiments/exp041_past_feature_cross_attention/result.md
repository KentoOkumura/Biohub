# exp041_past_feature_cross_attention 結果

## 仮説

過去3時点の未対応候補の固定画像特徴・座標・時刻を対象点ごとに参照すると、保存済みexp016より両胚の既知接続回収が改善し、教師負例予測と分裂回収を許容範囲に保てるかを検証した。exp015の固定候補と特徴、exp016の教師・分割・接続損失を使い、2fold・各3 epochでprimary trackerと追加attentionを学習した。公開画像重みの学習来歴にtrain動画が含まれるため、外側胚分割を独立CVとは呼ばない。

## 実行証拠

- Kaggle [学習Notebook version 2](https://www.kaggle.com/code/kentookumura/exp041-past-feature-cross-attention-train): `COMPLETE`。学習と隣接2-frameの単体診断を完走し、提出は行っていない。Notebook実行時間は16,245.96秒（約4.51時間）。2026-09-23 12:41 UTCのKaggle quotaはGPU使用40.52時間・残4.48時間だった。
- 回収した正本: [`artifacts/kaggle_v2/training_summary.json`](artifacts/kaggle_v2/training_summary.json)、[`graph_progression_gate.json`](artifacts/kaggle_v2/graph_progression_gate.json)、[`model_manifest.json`](artifacts/kaggle_v2/model_manifest.json)、[`runtime_benchmark.json`](artifacts/kaggle_v2/runtime_benchmark.json)、[`実行ログ`](artifacts/kaggle_v2/exp041-past-feature-cross-attention-train.log)。モデル2件とmanifest、GT窓監査、両foldの教師一致、指標の有限値をローカルで照合した。SHAと構造化された数値は[`metrics.json`](metrics.json)を正とする。
- version 1は過去窓のcache SHAを必須配列だけから再計算した実装不備で、最初の学習batch前に失敗した。version 2では全保存配列を照合するよう修正し、前回の停止箇所を通過した。version 1の失敗証拠は[`artifacts/kaggle_v1/`](artifacts/kaggle_v1/)に保存した。
- 対象窓はexp016と同じGT条件の18,707/19,701窓。199動画の対象が外側2foldで一度ずつ評価された。保存済みexp016モデルを同じ対象窓・教師で再評価し、対照は再学習していない。

## 両胚別の単体比較

接続判定は対象ペアの確率が0.5を超える条件で行った。教師負例予測数は正例に接するactive pair内の数であり、疎い注釈のため真の誤接続総数ではない。

| 評価胚 | 既知接続回収（exp016 → exp041） | 教師負例予測（exp016 → exp041） | 分裂親の全娘回収（exp016 → exp041） | 進行条件 |
| --- | --- | --- | --- | --- |
| 6bba | 100,230/103,393 → 100,235/103,393（+5本） | 4,355 → 4,459（+2.39%） | 48/108 → 46/108 | 未達 |
| 44b6 | 17,959/18,949 → 17,863/18,949（-96本） | 1,341 → 1,252（-6.64%） | 6/22 → 3/22 | 未達 |

6bbaでは既知接続回収がわずかに増えたが分裂回収が減った。44b6では既知接続と分裂回収の両方が減った。両胚の教師負例予測は事前の増加5%以内という条件を満たしたが、両胚で既知接続を改善し分裂を非悪化に保つ条件は満たさなかった。両胚のgraph進行判定は0/2で、全graph推論・公式指標評価は実行していない。公式scoreとLBは未計測である。

## 解釈

原因の考察と学習履歴の分析は[調査レポート](../../docs/surveys/biohub-exp041-past-feature-analysis_20260923.md)に保存した。44b6では親順位正解が9本増える一方、正解が1位でも確率0.5以下の接続が105本増えた。回収減96本はこの差に対応する。過去入力の寄与、密集度、checkpoint選択の影響と未測定事項を分けて整理した。

同じ学習済み重みで過去を全maskすると、通常入力との既知接続回収差は6bbaで+20本、44b6で-16本だった。過去3時点のframe番号だけを逆順にした場合との差は、両胚とも通常入力が+1本だった。過去入力が既知接続を一貫して改善した証拠にはならない。15 µm内の他候補が8個以上の密集領域では、通常入力の対照との差は6bbaで+22本、44b6で-71本であり、後者の悪化は密集領域に多い。過去候補を利用できるsource点の割合は両胚で約99%だったため、全体の未達を履歴欠落だけでは説明できない。

本結果が否定するのは、この固定特徴・全候補・1層128次元4 headのattention・exp016の教師と損失・2fold各3 epochという構成で、事前の両胚別進行条件を満たすという仮説である。過去情報を使う別の表現や学習方法、上位仮説全体の採否までは判断しない。公開画像重みの学習来歴と部分注釈に伴う評価限界は残る。

費用ベンチマークは129窓、最大のattention組数6,369,518、ピークGPU割当312,432,128 bytesを記録した。学習・評価の保守的な壁時計見積もりは約8.14時間、実際のNotebook時間は約4.51時間だった。version 2のruntime gateは壁時計10時間であり、T4×2のGPU quota消費へ直接換算した停止条件ではなかった。この設定を次回の実行予算判断へそのまま使わない。

## ユーザー判断

- ユーザーからKaggle実行完了の連絡を受けた。実験を完了扱いにするか、採用・不採用とするかの判断は未確認であり、`metrics.json`のstatusは実行状態の`debug_completed`とする。
- 推奨: この設定の全graph評価は行わず、exp016を接続トラッカーの比較基準として維持する。exp041を今回の単体検証で終了するか、追加調査を保留するかはユーザー判断を待つ。Kaggle submissionは行わない。
