# exp042_public_x138_replay 結果

## 仮説

公開 biohub x138 V1 の全checkpointと推論順序を固定すれば、公開test全件の候補座標・graph・raw提出ファイルが同一条件の2回実行で一致する。

## 実行証拠

- 契約と参照source: [requirements.md](requirements.md)、[assets/reference_notebook/biohub-x138.ipynb](assets/reference_notebook/biohub-x138.ipynb)
- 実行状態と数値: [metrics.json](metrics.json)
- 作業・取得確認: [SESSION_NOTES.md](SESSION_NOTES.md)
- 比較対象: [exp013の結果](../exp013_public_notebook_replay/result.md)。同実験のPublic LBは0.944。公開作者の0.953は私たちの実測値ではない。
- Kaggle実行・再実行比較: 未実施。追加V1284 checkpointのdataset ref・version IDは判明したが、取得APIが403でSHA未確認。

## 解釈

sourceを固定して推論Notebookと比較経路を実装した。追加モデルなしの3重みだけで実行してもx138再現とはならないため、入力guardで停止する。現在の静的テスト成功は、公開testでの再現一致、実行時間、Public LBを示さない。CVもない。

## ユーザー判断

- 完了・採用・不採用は未判断。実行証拠を得た後に判断を仰ぐ。

## 次

追加checkpointの正確なdataset ref、version、SHAを固定し、同じ条件で公開test全件を2回実行する。submissionは別途承認後に行う。
