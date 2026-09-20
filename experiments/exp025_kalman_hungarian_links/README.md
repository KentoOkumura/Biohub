# exp025_kalman_hungarian_links

## 概要

固定した画像得点と予測分裂の下で、自己予測履歴に基づく3次元カルマン費用が通常接続を改善するかを調べる。画像費用のみと、画像費用にカルマンの予測観測尤度を加える条件を比較する。

公開検出器・画像特徴・tracker重みを固定し、対応なしを含む一対一割当を行う。分裂娘は母速度と増やした不確実性を引き継ぐ。共通の出力制約により分裂予約と中心座標を保持する。未変更の親実験との差には後処理の変更も含まれるため、主比較とは分けて解釈する。

部分注釈による校正の偏りと、誤接続による履歴の誤りが主なリスク。Kaggleの公式評価後、ユーザー判断で不採用として完了した。結果の範囲と根拠は[result.md](result.md)に記録する。

## 正の記録

- 数値、実験status、構造化された実行証拠: [metrics.json](metrics.json)
- route、設定、系譜: [config.yaml](config.yaml)
- 実装契約、承認、受け入れ条件: [requirements.md](requirements.md)
- 結果の解釈とユーザー判断: [result.md](result.md)
- 実行中のコマンドと判断: [SESSION_NOTES.md](SESSION_NOTES.md)

## 実行入口

- [診断Notebook](exp025_kalman_hungarian_links_diagnostic.ipynb)
- [Jupytext source](exp025_kalman_hungarian_links_diagnostic.py)
- [保存済みgraphの短区間smoke](smoke_saved_graphs.py)

最初のフル実行と公式評価はKaggleで行う。診断専用であり、submission生成・提出の経路は持たない。
