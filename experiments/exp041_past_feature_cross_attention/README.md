# exp041_past_feature_cross_attention

## 概要

exp015の固定画像特徴から、過去3時点の全検出候補を1層のcross-attentionで参照して対象2時点の接続を予測する。exp016の保存済みfoldモデルを同じ外側窓で評価し、両胚別に比較する。

- 差分: 過去3時点の画像特徴・物理座標・実frame番号を使い、既存4 blockの前で対象点の特徴を更新する。
- リスク: 全候補attentionのGPU時間・メモリ、部分注釈に基づく負例評価、公開画像重みの学習来歴。
- 次: Kaggleの学習・単体診断は完走し、両胚の全graph進行条件は未達。結果を確認して実験の完了判断を受ける。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- 学習・単体診断Notebook: [`exp041_past_feature_cross_attention_train.ipynb`](exp041_past_feature_cross_attention_train.ipynb)
- 作業コマンドと実行境界: [`SESSION_NOTES.md`](SESSION_NOTES.md)
