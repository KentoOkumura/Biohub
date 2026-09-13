# 実験サマリー

このファイルは`task update-summary`で生成します。数値、status、構造化された実行証拠は各実験の`metrics.json`、証拠への参照、解釈、採否判断は`result.md`、時系列の作業履歴は`SESSION_NOTES.md`を正とします。手作業の戦略メモや変更履歴はここへ追記しません。

<!-- BEGIN AUTO EXPERIMENT SUMMARY -->

## 実験比較

| 実験 | ルート | 親 | 状態 | CV | Public LB | Private LB | 更新日 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp001_temporal_unet3d_baseline | temporal_unet3d_node_transformer | N/A | 失敗 | - | - | - | 2026-09-09 |
| exp002_unet3d_expandable_segments | temporal_unet3d_node_transformer | exp001_temporal_unet3d_baseline | 利用可 | - | 0.453 | - | 2026-09-11 |
| exp003_official_metric_audit | official_metric_audit | exp002_unet3d_expandable_segments | 完了 | - | - | - | 2026-09-11 |
| exp004_embryo_holdout_baseline | temporal_unet3d_node_transformer | exp002_unet3d_expandable_segments | 失敗 | - | - | - | 2026-09-11 |
| exp005_embryo_holdout_batch8 | temporal_unet3d_node_transformer | exp004_embryo_holdout_baseline | 完了 | 0.12490551617521294 | - | - | 2026-09-12 |
| exp006_embryo_holdout_seed314159 | temporal_unet3d_node_transformer | exp005_embryo_holdout_batch8 | デバッグ完了 | 0.5533789422289814 | - | - | 2026-09-12 |
| exp007_graph_checkpoint_selection | temporal_unet3d_node_transformer | exp005_embryo_holdout_batch8 | 破棄 | - | - | - | 2026-09-12 |
| exp008_deformation_pair_aux | temporal_unet3d_node_transformer | exp005_embryo_holdout_batch8 | 計画中 | - | - | - | 2026-09-11 |
| exp009_exp006_twofold_ensemble | temporal_unet3d_twofold_probability_ensemble | exp006_embryo_holdout_seed314159 | デバッグ完了 | - | 0.693 | - | 2026-09-12 |
| exp010_exp006_deterministic_replay | temporal_unet3d_node_transformer | exp006_embryo_holdout_seed314159 | 破棄 | 0.4981248954847012 | - | - | 2026-09-13 |
| exp011_public_detector_selection | public_detector_selection | N/A | 利用可 | - | - | - | 2026-09-12 |
| exp012_group_error_readout | group_error_diagnostic | exp011_public_detector_selection | 完了 | - | - | - | 2026-09-12 |
| exp013_public_notebook_replay | public_notebook_replay | exp011_public_detector_selection | デバッグ完了 | - | - | - | 2026-09-13 |
| exp014_exact_window_cache | public_notebook_replay | exp013_public_notebook_replay | 利用可 | - | - | - | 2026-09-13 |

<!-- END AUTO EXPERIMENT SUMMARY -->
