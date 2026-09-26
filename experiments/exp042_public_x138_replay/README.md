# exp042_public_x138_replay

## 概要

公開 biohub x138 V1 の全推論を、追加の V1284 座標補正モデルを含めて再実行する。旧公開構成の基準は exp013。

- 差分: x138 の補正座標、特徴補間、接続の組み直し、未使用検出点の再追加、欠落補完を元の順序で保持し、入力・中間座標・graph・提出ファイルの記録を追加した。
- リスク: 公開版のV1284 checkpointを取得してSHAを固定した。残るリスクはCUDA・SCIP・2 GPU分割と時間依存の処理縮小による出力差。
- 次: Public LB 0.953を確認済み。hidden testの動画別ILPログとPrivate LBは未取得。

## 正の記録

- 契約・実装方法: [requirements.md](requirements.md)
- route・系譜・固定設定: [config.yaml](config.yaml)
- 実行状況と証拠: [SESSION_NOTES.md](SESSION_NOTES.md)、[metrics.json](metrics.json)
- 結果の解釈とユーザー判断: [result.md](result.md)
- 参照Notebookと実行Notebook: [assets/reference_notebook/biohub-x138.ipynb](assets/reference_notebook/biohub-x138.ipynb)、[exp042_public_x138_replay_inference.ipynb](exp042_public_x138_replay_inference.ipynb)

## 実行入口

- SHAを固定した [inference notebook](exp042_public_x138_replay_inference.ipynb) をKaggleへ準備・pushする。手順と停止条件は [SESSION_NOTES.md](SESSION_NOTES.md) を参照。
