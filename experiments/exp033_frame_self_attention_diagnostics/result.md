# exp033_frame_self_attention_diagnostics 結果

## 仮説

保存済み4構成の既知edge pair順位と条件別誤りを同じ隣接2-frame windowで比較し、確率0.5での差に加えて、順位と条件ごとの違いを確認する。

## 実行証拠

- [Kaggle診断Notebook version 2](https://www.kaggle.com/code/kentookumura/exp033-frame-self-attention-diagnostics-diagnostic) が非公開T4で完走した。Notebook実行時間は3,330.12秒（約55.5分）。trackerの再学習、graph復元、submissionは行っていない。公式graph scoreは未計測。
- exp015の固定cacheは18,707 window、199サンプルで、identity SHAは `440891c4550adf8540e3c47f9784e68b6ca1b64bbe56dbb93a25eb87c46bc1ee`。4構成×2foldのcheckpoint、manifest、sourceを照合した。
- 既存の確率0.5指標は8通りすべてで件数と値が一致した。218件のpair shard（977,678,779 bytes）と32本のPR曲線を回収し、全ファイルのSHAを照合した。pair shard manifest SHAは `2e34f8fd0b7408bcd0ee1a14cbdc335ebe56327f0b600a02bf64038f5226073d`、集計JSON SHAは `c5c93911407cbbe75a4570a02736b7a7a373e42b16a659d3d60ff1fef4fd9ebd`。
- 胚別・構成別の正確な数値、内部検証、bucket件数・境界、PR曲線SHAは [metrics.json](metrics.json) を正とする。ローカルに回収した曲線とpair shardは `artifacts/kaggle_diagnostic_v2/diagnostic/` にある。実行経過は [SESSION_NOTES.md](SESSION_NOTES.md) に記録した。

## 外側胚のpair順位

precision 0.95以上を満たす点での最大recall。同点のfloat32 scoreは一括処理した。表の差は現行モデルとの差（percentage point）。

| 評価胚 | 現行 | Model A | Model B | 恒等初期化B |
| --- | ---: | ---: | ---: | ---: |
| 6bba | 0.9762 | 0.9218（-5.44） | 0.9683（-0.79） | 0.9730（-0.32） |
| 44b6 | 0.9102 | 0.7171（-19.30） | 0.8987（-1.15） | 0.9137（+0.35） |

- 44b6の恒等初期化Bだけが、このPR基準では現行を0.35ポイント上回る。6bbaでは下回り、両胚で一貫した改善は観測されなかった。
- 確率0.5での既知edge recallとfalse-positive pairは、6bbaで現行0.9694・4,355件、A 0.9487・7,192件、B 0.9675・5,176件、恒等初期化B 0.9665・4,431件。44b6では現行0.9478・1,341件、A 0.9169・1,830件、B 0.9521・1,495件、恒等初期化B 0.9384・1,164件。
- 両endpointがGEFF注釈に対応したpairだけでは、precision 0.95時のrecallは全構成で0.996以上だった。主母集団のactive pairには未知endpointを含むpairが6bbaで41,165,763 / 42,351,260件、44b6で13,623,411 / 13,721,808件ある。主PRの差を確定的な生物学的誤接続差とは解釈できない。

## 条件別の観測

bucket境界は各foldの学習側windowのみで固定し、外側評価胚では調整していない。以下は確率0.5の既知edge recall。

- 子候補の最近傍距離が最小のbucketでは、6bbaで現行0.8573、A 0.8063、B 0.8606、恒等初期化B 0.8433。44b6では現行0.8968、A 0.8593、B 0.9126、恒等初期化B 0.8793。Bにはこの条件で小さなrecall増があるが、同bucketのfalse-positive pairも両胚で増えた。
- 移動距離が最大のbucketでは、6bbaで現行0.9200、A 0.8589、B 0.9119、恒等初期化B 0.9135。44b6では現行0.7895、A 0.6552、B 0.7935、恒等初期化B 0.7607。Aの低下が目立つ。
- 6bbaの分裂親回収は現行48/108、A 45/108、B 56/108、恒等初期化B 50/108。44b6は現行6/22、A 5/22、B 4/22、恒等初期化B 4/22。Bの増加は6bbaだけで、pair順位や44b6の分裂親回収とは一致しない。

## 解釈

0.5での差が閾値だけによるものならprecisionを固定した比較で解消するはずだが、Model Aは両胚で大きく低下し、Model Bも両胚で現行を下回った。恒等初期化Bは44b6だけでわずかに上回る。Self-Attention構成の一貫したpair順位改善を示す証拠は得られていない。固定公開画像モデルの学習来歴とGEFFの部分注釈により、これは独立CVでも公式graph scoreでもない。

## 次

この診断とexp025の対照を確認し、後続の距離bias・近傍制限の検討を進めるかユーザーが判断する。

## ユーザー判断

2026-09-21、ユーザーがexp033を完了と判断した。Self-Attention構成の採否は未判断。
