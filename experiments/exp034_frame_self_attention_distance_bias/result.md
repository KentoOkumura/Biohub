# exp034_frame_self_attention_distance_bias 結果

## 仮説

exp025の恒等初期化Model Bへ学習可能な物理距離biasを追加すると、固定候補・固定decodeのholdout最終graph scoreが保存済み公開controlより改善する。

## 実行証拠

Kaggle train version 1は入力監査と64-window benchmarkまで完了したが、pair診断を含む保守見積りが12時間gateを超えたため、full training前に停止した。

- fold 0: 64 window、16.3834秒、旧見積り23,708.41秒
- fold 1: 64 window、17.0530秒、旧見積り26,921.25秒
- 合計旧見積り: 50,629.66秒
- checkpoint、最終graph、公式score、submission: 未生成
- version 1の証拠とbenchmarkはmetrics.jsonのevidence.rerunsへ保存

## 解釈

version 1の停止は距離biasモデルの失敗を示さない。最大窓のbackprop速度をbucket計算と重複outer評価にも適用し、1.5倍した見積りが停止条件を超えた。

ユーザー方針によりpair診断、bucket計算、重複outer評価を採否条件から削除した。改訂版は2fold×3epochの学習後、固定ILP・graph repairを通したholdout最終graphを公式指標で直接評価する。

評価対象は次のとおり。

- control: exp015の保存済み公開tracker最終graph
- candidate: fold 0を6bba、fold 1を44b6へ適用した距離bias tracker最終graph
- metric: adjusted edge Jaccard + 0.1 × division Jaccard
- report: overall、胚別、sample別、controlとの差分

改訂版のKaggle train version 2は2fold×3epochを完走した。

- 実測学習時間: 3,439.39秒
- 保守実行時間見積り: 21,493.31秒（12時間gate内）
- fold 0: best epoch 0、internal selection score 0.9859505746、checkpoint SHA 83dd2421921c80137dbe806f48584cad8d883269c494c868b7c5620ee864c845
- fold 1: best epoch 2、internal selection score 0.9782032196、checkpoint SHA 0a39707a63b30afcacf59b6e5b3b2cb4c2c6125df2a80271237717438c3f54c1
- model manifest SHA: df7a94d5a04651ac8922f3dd8fe5e672a99b91f99ab956a677b95616939c12e2
- submission: 未作成

checkpointとmanifestのファイルSHAは取得後に再計算し、manifest記載値と一致した。

trackerの内部検証値はfold 0でepoch 0が最良となり、fold 1のepoch 0からepoch 2への差も約+0.00000345に留まった。指定した距離bias設定には、週30時間のGPU上限を超えて全199動画のgraph推論へ進むだけの改善根拠がないと判断した。最終graphと公式scoreは未計測であるため、距離biasによる公式指標改善の仮説自体は確認できていない。

## ユーザー判断

- 2026-09-22: 物理距離bias 1設定、2fold×3epochの実行を承認。
- 2026-09-22: version 1停止後、「普通に最終的な予測結果を評価すればいい」と指示。
- 反映内容: pair gateを廃止し、最終holdout graphの公式直接評価を正とする。
- submissionは作成しない。
- 2026-09-22: tracker学習結果を確認し、最終graph評価を実行せず本実験を完了するよう指示。
- 完了時の結論: 指定した距離bias設定は追加計算へ進めない。空間Self-Attention全般の不採用判断には使わない。
