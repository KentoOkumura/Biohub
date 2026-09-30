# exp046_x138_author_head_comparison

exp043の固定推論構成で、自前の座標補正headとx138作者の公開headを交換し、同じ20動画の中心・接続・分裂・最終graphを比較する。

## 概要

- 差分: checkpointのstate_dictと、そのmean・scaleの一式だけを交換する。
- リスク: 作者headの学習動画は不明で、competition train動画上の評価は独立した交差検証ではない。
- 次: [結果](result.md)を確認し、作者headの採否と実験完了をユーザーが判断する。

## 正の記録

要件と受け入れ条件は[requirements.md](requirements.md)、設定と系譜は[config.yaml](config.yaml)、実行証拠は[metrics.json](metrics.json)、経過は[SESSION_NOTES.md](SESSION_NOTES.md)、解釈とユーザー判断は[result.md](result.md)に記録する。

## 実行入口

[予備確認Notebook](exp046_x138_author_head_comparison_pilot.ipynb)で両headを照合し、対照と一致した場合は[作者head全件Notebook](exp046_x138_author_head_comparison_author_only.ipynb)を実行する。[両head全件Notebook](exp046_x138_author_head_comparison_inference.ipynb)は対照不一致時に使う。学習Notebookとcompetition submissionはこの比較に含めない。
