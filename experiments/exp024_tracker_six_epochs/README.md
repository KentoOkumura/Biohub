# exp024_tracker_six_epochs

## 概要

- 仮説要約: 固定画像特徴から学ぶprimary trackerの学習を最大6エポックへ延ばすと、3エポック基準より両胚の公式graph指標が改善するか。
- 変更点要約: exp016の公開初期重み・教師・損失・分割・復号を維持し、エポック上限だけを3から6へ変更する。両foldで後半重みが選ばれた場合に公式評価へ進む。
- リスク: 公開初期モデルの学習来歴にtrain動画が含まれる可能性があり、胚別の結果は条件付き評価。6エポックの実測費用は記録済みだが、再実行時のGPU費用には変動がある。
- 判断: ユーザーはこの6エポック条件を不採用として完了とした。公式graph指標は未計測で、保存済み2エポック基準を維持する。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- 学習 notebook: `exp024_tracker_six_epochs_train.ipynb`
- 推論 notebook（続行条件不成立のため未実行）: `exp024_tracker_six_epochs_inference.ipynb`
- 2時点の接続診断 notebook: `exp024_tracker_six_epochs_diagnostic.ipynb`
- Kaggle実行と停止条件の判定は`SESSION_NOTES.md`、結果の解釈は`result.md`に記録する。
