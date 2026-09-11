# 実験サマリー

このファイルは`task update-summary`で生成します。数値、status、構造化された実行証拠は各実験の`metrics.json`、証拠への参照、解釈、採否判断は`result.md`、時系列の作業履歴は`SESSION_NOTES.md`を正とします。手作業の戦略メモや変更履歴はここへ追記しません。

<!-- BEGIN AUTO EXPERIMENT SUMMARY -->

## 実験比較

| 実験 | ルート | 親 | 状態 | CV | Public LB | Private LB | 更新日 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp001_temporal_unet3d_baseline | temporal_unet3d_node_transformer | N/A | 失敗 | - | - | - | 2026-09-09 |
| exp002_unet3d_expandable_segments | temporal_unet3d_node_transformer | exp001_temporal_unet3d_baseline | 利用可 | - | 0.453 | - | 2026-09-11 |
| exp003_official_metric_audit | official_metric_audit | exp002_unet3d_expandable_segments | 完了 | - | - | - | 2026-09-11 |

<!-- END AUTO EXPERIMENT SUMMARY -->
