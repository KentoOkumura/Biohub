# exp042_public_x138_replay

## 概要

公開 biohub x138 V1 の全推論を、追加の V1284 座標補正モデルを含めて再実行する。旧公開構成の基準は exp013。

- 差分: x138 の補正座標、特徴補間、接続の組み直し、未使用検出点の再追加、欠落補完を元の順序で保持し、入力・中間座標・graph・提出ファイルの記録を追加した。
- リスク: 公開実行からdataset refとversion IDは特定したが、Kaggle APIが403を返す。checkpoint本体とSHAは未取得で、入力guardは実行を拒否する。
- 次: 追加checkpointを固定し、公開test全件のcleanな2実行と提出前検証を行う。

## 正の記録

- 契約・実装方法: [requirements.md](requirements.md)
- route・系譜・固定設定: [config.yaml](config.yaml)
- 実行状況と証拠: [SESSION_NOTES.md](SESSION_NOTES.md)、[metrics.json](metrics.json)
- 結果の解釈とユーザー判断: [result.md](result.md)
- 参照Notebookと実行Notebook: [assets/reference_notebook/biohub-x138.ipynb](assets/reference_notebook/biohub-x138.ipynb)、[exp042_public_x138_replay_inference.ipynb](exp042_public_x138_replay_inference.ipynb)

## 実行入口

- 追加checkpointへのアクセスとSHAを確認してから [inference notebook](exp042_public_x138_replay_inference.ipynb) をKaggleへ準備・pushする。手順と停止条件は [SESSION_NOTES.md](SESSION_NOTES.md) を参照。
