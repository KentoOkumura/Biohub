# exp042_public_x138_replay 結果

## 仮説

公開 biohub x138 V1 の全checkpointと推論順序を固定すれば、公開test全件の候補座標・graph・raw提出ファイルが同一条件の2回実行で一致する。

## 実行証拠

- 契約と参照source: [requirements.md](requirements.md)、[assets/reference_notebook/biohub-x138.ipynb](assets/reference_notebook/biohub-x138.ipynb)
- 実行状態と数値: [metrics.json](metrics.json)
- 作業・取得確認: [SESSION_NOTES.md](SESSION_NOTES.md)
- 比較対象: [exp013の結果](../exp013_public_notebook_replay/result.md)のPublic LB 0.944、[exp043の結果](../exp043_x138_self_trained_head/result.md)のPublic LB 0.950。公開作者の報告値0.953に対し、今回の作者版再現もref `56569806`で0.953を実測した。
- Kaggle公開test 4動画をprivate Notebook V1・V2で2回実行し、両方正常完了。追加V1284 checkpoint SHA256は`625a0d9340f48193f2ec294fc2d81c5bb3c03087eab78ef0ae998a9c4c7da00c`。提出前検証もPASS（238,260行、重複ID・欠損・無限大0）。[比較report](artifacts/replay_comparison.json)の15項目はすべて一致し、raw CSV SHA256 `d52a5da2ae5cb0d1b22499f6ca51a00838a6c32ae9a1986ec756ea72e7909e03`は保存済み公開x138 V1の出力SHAとも一致した。

## 解釈

sourceと全checkpointを固定し、公開testの2回実行で補正前後の座標、graph topology、決定的なrun統計、raw提出CSVのSHAが一致した。Notebook経過時間はV1が約1201秒、V2が約1270秒で、ILPの最長時間はそれぞれ77.5秒、73.5秒。両実行とも`repair_fallback=0`、`deadline_degraded=0`である。公開testでの再実行一致と、作者版のPublic LB 0.953を確認した。これはexp043の自前版0.950を0.003上回り、保存済み公開作者の報告値0.953と一致する。提出から初回の採点完了確認までは最大約6時間54分なので、hidden実行が提出後に始まったなら、Notebook経過7.5時間による後処理縮小条件には届かない。ただし採点時のhidden test出力と動画別ILPログは未取得であり、動画ごとの1200秒上限への到達、headだけの寄与、hidden test再実行の一致は検証できていない。CUDA・ILPと時間依存の処理分岐が残るため、コード上でhidden testまでの決定性を保証せず、`config.yaml`の`reproducibility.deterministic_anchor`は`false`のままにする。CVもない。

## ユーザー判断

- 2026-09-26、ユーザーがこの比較実験の完了を明示した。実験statusは`completed`。作者版の採用・不採用は未判断。

## 次

作者版V2のsubmission ref `56569806`は`COMPLETE`、Public LB 0.953、Private LB未表示。採点時のhidden test出力と動画別ILPログは未取得のまま。
