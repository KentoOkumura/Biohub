# exp031_frame_self_attention_spatial 結果

注意: この実験は旧候補 `frame_self_attention`（距離biasなしのSelf-Attention追加）を実装した。コミット `7237a65` の別候補 `frame_self_attention_spatial` が指定するフレーム内Attentionへの距離biasまたは近傍制限は、ここでは実装・評価していない。

## 仮説

固定公開特徴・既存教師・復号の下で、フレーム内の細胞間Self-Attentionを加えたSelf後Cross構成が、同じ3 epochで再学習した現行構成より両胚の公式graph接続指標を改善するか検証する。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- 実験status、全foldの数値、model SHA、Kaggle実行証拠: [`metrics.json`](metrics.json)
- 実行コマンドと進行判断: [`SESSION_NOTES.md`](SESSION_NOTES.md)
- Kaggle train Notebook: [version 1](https://www.kaggle.com/code/kentookumura/exp031-frame-self-attention-spatial-train)

## 実行証拠

- Kaggle train version 1 は `COMPLETE`。3構成×2fold×3 epochの6モデルを学習。Notebook実行時間 11662.80秒（約 3.24時間）。model manifest SHA256は `f9906ac3d747a2905c0a89923c5a6ac54d97e9a045068083c9a0a8bbc4d0f905`。6つのmodel file SHAをmanifestと照合した。
- PyTorch動的preflightで旧checkpointのstrict読み込みとlogits一致、3構成のpadding・順序・空集合・勾配・checkpoint復元を確認。
- 同一runの再学習legacyを対照に、同じouter windowの隣接2フレーム指標を両胚別に比較した。

| 評価胚 | 構成 | 既知edge回収 | 教師負例pairへの予測 | 既知分裂母回収 |
| --- | --- | ---: | ---: | ---: |
| 44b6 | legacy | 17,959/18,949 (94.775%) | 1,341 | 6/22 |
| 44b6 | self_only | 17,214/18,949 (90.844%) | 1,674 | 4/22 |
| 44b6 | self_cross | 18,074/18,949 (95.382%) | 1,528 | 6/22 |
| 6bba | legacy | 100,230/103,393 (96.941%) | 4,355 | 48/108 |
| 6bba | self_only | 98,232/103,393 (95.008%) | 7,202 | 39/108 |
| 6bba | self_cross | 99,874/103,393 (96.596%) | 5,197 | 53/108 |

## 解釈

`self_only` は両胚で既知edge回収が減り、教師負例pairへの予測が増えた。`self_cross` は44b6の既知edge回収が増え、6bbaの既知分裂母回収が増えたが、教師負例pairへの予測は両胚で現行対照より多い。`requirements.md` のgraph自動進行条件を満たす構成はないため、全graph推論を保留した。**公式graph scoreは未計測**。

評価pairのうち未知endpointを含む割合は6bbaで97.20%、44b6で99.28%。疎い注釈の教師負例pair数は真の誤接続全体を表さない。また、隣接2フレームの閾値付き予測ではsecondary tracker、ILP、graph repair後の効果は測れない。公開画像encoderと初期trackerの学習来歴にも評価胚が含まれるため、結果は固定公開モデル下の条件付き評価である。

## ユーザー判断

- 判断: 未判断。実験の採用・不採用・完了は確定していない。
- 未解決事項: 自動進行条件が不成立でも全199動画の公式graph評価を実行するか。直近のquota確認では週30 GPU時間の上限まで7.93時間。

## 次

ユーザーが全graph実行を明示した場合は、前段指標と条件不成立を記録したうえで残GPU時間と推論費用を再確認して実行する。competition submissionは別途明示依頼まで行わない。
