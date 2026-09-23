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
| exp013_public_notebook_replay | public_notebook_replay | exp011_public_detector_selection | 利用可 | - | 0.944 | - | 2026-09-13 |
| exp014_exact_window_cache | public_notebook_replay | exp013_public_notebook_replay | 利用可 | - | - | - | 2026-09-13 |
| exp015_oracle_stage_limits | oracle_stage_diagnostic | exp014_exact_window_cache | 完了 | - | - | - | 2026-09-13 |
| exp016_frozen_image_encoder | frozen_feature_primary_tracker | exp011_public_detector_selection | 完了 | - | - | - | 2026-09-18 |
| exp017_cross_crop_registration_audit | cross_crop_registration_audit | N/A | 完了 | - | - | - | 2026-09-14 |
| exp018_graph_cost_scale | fixed_candidate_graph_cost_scale | exp015_oracle_stage_limits | 破棄 | - | - | - | 2026-09-15 |
| exp019_partial_edge_mask | frozen_feature_primary_tracker | exp016_frozen_image_encoder | 破棄 | - | - | - | 2026-09-20 |
| exp020_division_triplet_candidates | division_triplet_candidate_diagnostic | exp015_oracle_stage_limits | 完了 | - | - | - | 2026-09-19 |
| exp021_division_teacher_audit | division_triplet_teacher_diagnostic | exp020_division_triplet_candidates | 完了 | - | - | - | 2026-09-19 |
| exp022_synthetic_division_teacher_audit | synthetic_division_teacher_diagnostic | exp021_division_teacher_audit | 完了 | - | - | - | 2026-09-20 |
| exp023_synthetic_detector_teacher_audit | synthetic_detector_teacher_diagnostic | exp022_synthetic_division_teacher_audit | 完了 | - | - | - | 2026-09-20 |
| exp024_tracker_six_epochs | frozen_feature_primary_tracker | exp016_frozen_image_encoder | 破棄 | - | - | - | 2026-09-20 |
| exp025_frame_self_attention | frozen_feature_frame_self_attention_tracker | exp016_frozen_image_encoder | 実行中 | - | - | - | 2026-09-20 |
| exp025_kalman_hungarian_links | kalman_hungarian_fixed_divisions | exp016_frozen_image_encoder | 破棄 | 0.8733317200940277 | - | - | 2026-09-20 |
| exp027_multi_frame_tracker | frozen_feature_local_three_frame_tracker | exp016_frozen_image_encoder | 完了 | - | - | - | 2026-09-20 |
| exp028_direct_graph_prediction | direct_parent_selection | exp016_frozen_image_encoder | デバッグ完了 | - | - | - | 2026-09-20 |
| exp029_mother_daughter_set_selection | mother_daughter_set_selection | exp028_direct_graph_prediction | デバッグ完了 | - | - | - | 2026-09-20 |
| exp030_frame_self_attention_diagnostics | frozen_feature_primary_tracker_diagnostic | exp016_frozen_image_encoder | 完了 | - | - | - | 2026-09-20 |
| exp032_three_frame_ten_epoch_training | frozen_feature_local_three_frame_tracker | exp027_multi_frame_tracker | デバッグ完了 | - | - | - | 2026-09-20 |
| exp033_frame_self_attention_diagnostics | frozen_feature_primary_tracker_diagnostic | exp025_frame_self_attention | 完了 | - | - | - | 2026-09-21 |
| exp034_frame_self_attention_distance_bias | frozen_feature_frame_self_attention_distance_bias_tracker | exp025_frame_self_attention | 完了 | - | - | - | 2026-09-22 |
| exp035_velocity_features | frozen_feature_velocity_tracker | exp016_frozen_image_encoder | 実行中 | - | - | - | 2026-09-22 |
| exp036_detection_score_pair_features | detection_score_pair_tracker | exp016_frozen_image_encoder | 破棄 | - | - | - | 2026-09-23 |
| exp039_dog_features | frozen_detector_dog_node_tracker | exp016_frozen_image_encoder | 破棄 | - | - | - | 2026-09-23 |

<!-- END AUTO EXPERIMENT SUMMARY -->
