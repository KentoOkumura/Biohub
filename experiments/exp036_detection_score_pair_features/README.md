# exp036_detection_score_pair_features

## 概要

- 仮説要約: 固定候補の検出得点をcandidate pair単位で接続判定へ追加すると、画像特徴と位置だけを使う同構造対照より、既知接続と分裂親の回収を改善できる。
- 変更点要約: fold内のgradient-update候補だけで検出得点をclipped logit標準化し、source、target、小さい方、絶対差の4特徴から4→16→1の残差をprimary接続logitへ加える。
- 比較: 残差headを含む同じモデル構造で、標準化後の得点を0に置換する対照と、実際の得点を使うvariantを外側2foldで再学習する。
- リスク: 検出得点が既存画像特徴と冗長な可能性、部分注釈下の誤接続指標の限界、条件付き外側評価で独立CVではない点。
- 現在地: Kaggle version 2で4 model学習と外側2fold評価が完了。両foldでpositive edge recallが同構造対照を下回り、進行条件は未達。公式graph評価は未実行。ユーザー判断により不採用で終了した。
- 次: この実験の追加実行は行わない。上位仮説にはDoG・HOGの未検証候補が残る。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- 学習Notebook: `exp036_detection_score_pair_features_train.ipynb`
- Jupytext source: `exp036_detection_score_pair_features_train.py`
- score pair head: `detection_score_tracker.py`
- cache検査、正規化、教師、loss、診断: `frozen_tracker.py`
- 初段は学習だけを実行し、inference Notebook、公式評価Notebook、submissionを作らない。
- 最初のフル実行はKaggleを正とする。
