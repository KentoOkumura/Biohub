# exp038_past_candidate_knn_attention 結果

## 仮説

接続元の物理近傍にある過去候補だけをattentionへ入力すると、全中央接続候補を維持したまま計算費用を抑え、保存済みexp016より両胚の接続判断を改善できる。

## 実行証拠

- 契約・設定: [requirements](requirements.md)、[config](config.yaml)。
- 比較対象: exp016保存済みfoldモデル。親exp037には本学習済みモデルがない。
- 実行状態・測定値・生成物SHA: [metrics](metrics.json)。
- 作業・実行履歴: [SESSION_NOTES](SESSION_NOTES.md)。
- Kaggle version 2の実測出力: [候補保持率](artifacts/kaggle_v2/candidate_retention.json)、[費用診断](artifacts/kaggle_v2/runtime_benchmark.json)、[Kaggle側metrics](artifacts/kaggle_v2/metrics.json)。モデルとtraining summaryは生成されていない。
- Kaggle version 1は設定参照の不一致により学習前に停止。出力metricsを回収し、設定修正と起動コードの回帰テストを追加した。実行履歴はmetricsのrerunsとSESSION_NOTESを参照。
- version 2は保持率・費用診断まで実行し、時間条件で本学習前に停止した。Notebookの記録時間は876.50秒。

## 解釈

学習側の既知edgeで、両端が固定候補に対応したものに限ると、過去候補8点の保持率はfold 0が16552/16554（99.9879%）、fold 1が93628/93681（99.9434%）。事前条件の両fold 99%以上を満たした。これは疎い注釈に対する条件付き保持率であり、未検出edgeを含む全細胞の保持率や予測精度ではない。

実測からの保守的な全工程所要時間は39,350.16秒（10.93時間）で、今回のGPU残枠を2台で按分した23,454秒（6.515時間）の上限を超えた。GPUメモリは408,843,776 / 12,508,830,105 bytesで上限内。停止条件に従い本学習、exp016との両胚の単体比較、全graph、公式score、Public LBは未実行。

## ユーザー判断

- 判断: 未判断。
- 承認範囲: 2026-09-23に実装・Kaggle実行、今回限り週45 GPU時間までの無料枠利用を承認された。
- 実験の採用・不採用・完了判断は結果提示後に受ける。

## 次

同じ実験条件で本学習へ進むには、十分なGPU枠のある時点で費用診断から再実行する必要がある。2026-09-23 13:26 UTC時点のaccount全体のGPU残枠は4.48時間、週次resetは2026-09-26 00:00 UTC。実験の採否はユーザーの判断を待つ。
