# exp050_shared_edge_graph_learning

## 概要

固定したexp043入力にGNNの共有接続得点と母ごとの集合選択を加え、費用調整対照を含む4条件で全graphを比較する。主なリスクは疎な教師、2娘候補の不足、集合ILPの計算量、公開画像モデル由来の条件付き評価。

## 正の記録

- 契約: [requirements.md](requirements.md)
- 設定と系譜: [config.yaml](config.yaml)
- 実行の時系列: [SESSION_NOTES.md](SESSION_NOTES.md)
- 数値と状態: [metrics.json](metrics.json)
- 証拠と解釈: [result.md](result.md)

## 実行入口

実装と実行証拠を保管する。実施した比較、残る検証範囲、ユーザー判断は[result.md](result.md)を参照する。追加実行の予定はなく、再検討する場合は保存済み設定・試行履歴・提出版を起点に対象を定める。
