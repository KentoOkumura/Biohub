# exp027_multi_frame_tracker

## 概要

固定した公開画像特徴と検出候補を使い、前フレームの近傍検出点を参照する局所attentionで中央ペアの接続を予測する。同じ構造で前フレームを隠した2時点条件と比較する。


## 正の記録

- 実装前の契約と判定条件: [requirements.md](requirements.md)
- 設定と系譜: [config.yaml](config.yaml)
- 実行経緯: [SESSION_NOTES.md](SESSION_NOTES.md)
- 数値・SHA・実験status: [metrics.json](metrics.json)
- 結果と解釈: [result.md](result.md)

## 実行入口

- train Notebook: [exp027_multi_frame_tracker_train.ipynb](exp027_multi_frame_tracker_train.ipynb)
- 保存重みの過去入力比較: [exp027_multi_frame_tracker_diagnostic.ipynb](exp027_multi_frame_tracker_diagnostic.ipynb)
- 内部validationでの閾値診断: [exp027_multi_frame_tracker_validation_diagnostic.ipynb](exp027_multi_frame_tracker_validation_diagnostic.ipynb)

主なリスクは、画像特徴が元の2時点窓に依存すること、密な領域の近傍数、疎い注釈と公開画像重みの来歴である。Kaggleのruntime benchmarkは12時間gateを超えた。ユーザーが承認した1epoch・各胚64窓の予備実験はKaggleで終了した。胚別の結果と費用は結果記録を参照する。保存重みの過去入力比較も終了し、recall差に対する親順位と確率閾値の影響を結果記録へ整理した。内部validationで閾値を固定して外側へ適用する診断も終了し、その結果を考察へ追記した。全graph推論の進行条件、ユーザー判断、予測履歴を使う候補への引き継ぎは[result.md](result.md)を参照する。
