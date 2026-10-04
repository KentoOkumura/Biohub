# exp031_frame_self_attention_spatial

注意: この実験は旧候補 `frame_self_attention`（距離biasなしのSelf-Attention追加）を実装した。コミット `7237a65` の別候補 `frame_self_attention_spatial` が指定するフレーム内Attentionへの距離biasまたは近傍制限は、ここでは実装・評価していない。

## 概要

固定した公開画像特徴から隣接時刻の接続を学ぶトラッカーに、各フレーム内の細胞間Self-Attentionを追加する。現行のCross-Attentionのみ、Selfのみ、Self後Crossの3構成を比較する。

- 変更点: 共有2層Encoderと3構成の学習・graph評価。画像encoder、教師、loss、候補、復号は固定。
- リスク: 追加Attentionの時間とメモリ、公開画像モデルの学習来歴による条件付き評価。
- 次: 動的テスト、Kaggle学習、隣接2フレームの評価は終了した。[結果](result.md)のgraph進行条件を満たさなかったため全graph推論を保留し、実験の完了・採否と追加評価のユーザー判断を待つ。

## 正の記録

- 実装契約: [`requirements.md`](requirements.md)
- 設定と系譜: [`config.yaml`](config.yaml)
- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- 実行ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)
- 結果の解釈とユーザー判断: [`result.md`](result.md)

## 実行入口

- 学習Notebook: [`exp031_frame_self_attention_spatial_train.ipynb`](exp031_frame_self_attention_spatial_train.ipynb)
- 推論Notebook: [`exp031_frame_self_attention_spatial_inference.ipynb`](exp031_frame_self_attention_spatial_inference.ipynb)
- 編集用Jupytext sourceは同名の `.py`。最初のフル実行と公式評価はKaggle上で行う。
