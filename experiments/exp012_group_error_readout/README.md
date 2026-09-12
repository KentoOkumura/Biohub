# exp012_group_error_readout

## 概要

- 仮説要約: 固定した予測を胚・画像輝度・候補密度・候補の境界距離・既知分裂で分けると、全体値だけでは見えない改善と悪化を特定できる。
- 変更点要約: exp005 と exp006 の同じ 199 動画を、exp003 と同じ公式集計式で条件別に再集計する。モデル、予測、復号は変更しない。
- リスク: 2 胚だけの説明的診断であり、条件から補正規則を選ぶと外側評価が開発集合になる。
- 次: Kaggle CPU diagnostic を実行し、固定 candidate cache を含む全条件表と SHA を取得する。

## 正の記録

- 数値、実験 status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- diagnostic Notebook: [`exp012_group_error_readout_diagnostic.ipynb`](exp012_group_error_readout_diagnostic.ipynb)
- Jupytext source: [`exp012_group_error_readout_diagnostic.py`](exp012_group_error_readout_diagnostic.py)
- Kaggle 準備: `make prepare-kaggle-notebooks EXP=exp012_group_error_readout EXTRA_ARGS="--notebook diagnostic --run-on-push"`

Kaggle Notebook 実行を正とする。ローカル実行は固定 candidate cache がローカルにないため、関数と保存済み小規模証拠の test に限定する。
