# exp045_x138_coordinate_effect_audit

## 概要

exp043の採用済み座標headの学習動画を除く両胚各10本で、補正なしと補正ありを同じ公開画像モデル・tracker・後処理に適用し、既知接続、分裂、最終graphへの作用を調べる。新規学習とコンペ提出は行わない。

- 差分: 座標headの適用有無と、それぞれの位置での三線形補間。
- リスク: 公開画像モデルは対象train画像を学習済みであり、結果は固定公開モデル下の条件付き評価となる。20動画2条件の全graph処理はGPUと出力容量を要する。
- 次: 今回の条件付き20動画評価でのscore改善を記録し、未知動画またはPublic LBでの効果は別実験で検証する。

## 正の記録

- 実験の契約: [`requirements.md`](requirements.md)
- 設定と系譜: [`config.yaml`](config.yaml)
- 実行時系列: [`SESSION_NOTES.md`](SESSION_NOTES.md)
- 数値と実行証拠: [`metrics.json`](metrics.json)
- 解釈とユーザー判断: [`result.md`](result.md)

## 実行入口

- [`exp045_x138_coordinate_effect_audit_pilot.ipynb`](exp045_x138_coordinate_effect_audit_pilot.ipynb): 両胚各1本の2条件と公式評価で時間を実測する。
- [`exp045_x138_coordinate_effect_audit_inference.ipynb`](exp045_x138_coordinate_effect_audit_inference.ipynb): 20動画の2条件推論とgraph生成。version 1は全graph生成後のGEFF読込で停止し、sourceは修正済み。
- [`exp045_x138_coordinate_effect_audit_official.ipynb`](exp045_x138_coordinate_effect_audit_official.ipynb): 保存済み40最終graphをKaggle CPUで公開評価器にかける。
- [`run_local_fixed_id.py`](run_local_fixed_id.py): 保存済み候補cacheと段階保存物を使い、同じIDの既知edgeをローカルで追跡する。
- [同じNotebookのJupytextソース](exp045_x138_coordinate_effect_audit_inference.py)
- [診断の純粋関数](audit_core.py): 動画選択、物理距離での対応、既知edge・分裂の集計。
