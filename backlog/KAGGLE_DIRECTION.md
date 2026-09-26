# Kaggle 方針

## 参照する前提

- 機械可読なコンペ設定、データパス、validation、submission、Kaggle runtimeは[`project.yml`](../project.yml)を正とする。
- 公式情報は[`docs/01_competition.md`](../docs/01_competition.md)、評価指標は[`docs/02_metric.md`](../docs/02_metric.md)、評価分割とリーク確認は[`docs/03_validation.md`](../docs/03_validation.md)、データ仕様は[`docs/04_data.md`](../docs/04_data.md)を参照する。
- 保存場所・記録責務・ユーザー判断・用語の規則は[`AGENTS.md`](../AGENTS.md)、略語とリポジトリ内の管理用語は[`docs/glossary.md`](../docs/glossary.md)を正とする。
- このファイルを現在の学習方針・重点・比較基準・検証中の仮説・未着手候補の索引の正とする。個別候補の設計は各詳細ファイルに置く。

## 今後の学習方針

- **公開検出器を固定し、トラッカーを学習する。** 2026-09-12のユーザー依頼により、今後の提案・候補設計・新規実験の標準とする。検出器と画像特徴抽出器の重み・正規化の統計を固定し、接続・分裂を予測する下流部分を学習する。公開モデルが検出と追跡を同時学習した重みであっても、この段階で同時学習し直すことを必須にしない。
- **計算環境はKaggle Notebookのみ、GPUは週45時間以内、課金なし。** Colabや外部GPUを前提にしない。Colab例外はユーザー承認済みのexp032と、2026-09-23にGPU不足で切替指示を受けたexp040の学習・診断に限る。初回の検出・特徴抽出、トラッカー学習、検証、提出用推論を予算に含め、実行前の残量と小規模実測で実行数を決める。保存済み予測の集計・graphの再選択は可能ならCPU Notebookで行う。Notebook経過時間と実際のGPU割当消費は区別し、未実測の所要時間を保証しない。
- 最初に使用する公開Notebook・モデル版・checkpoint・ライセンス等の利用条件・学習来歴は[`exp011_public_detector_selection`](../experiments/exp011_public_detector_selection/)で選定し、2026-09-12にユーザーが採用した。過去の自前重みを「公開検出器」と読み替えない。公開の既存トラッカーを基準とし、同じ固定検出器でトラッカーを学習した結果を比較する。後続案も同じ固定検出器下のトラッカー同士で比較し、自前検出器の再学習や全層更新の対照を必須にしない。
- 検出候補・物理座標・検出得点・必要な画像特徴を保存して再利用する。時間を扱う画像モデルの特徴は入力窓に依存するため、元の全時間窓・前処理・crop・変形・padding・精度・抽出座標・重みの版を対応付ける。初回は候補点特徴を中心に等価性と容量を測り、全voxel特徴の一括保存やframe単独の特徴への置換を前提にしない。画像・窓・候補座標等を変える比較は必要な特徴を再抽出し、費用に含める。
- トラッカーの入力は固定検出器の予測と画像特徴、正解の根拠は主催者のGEFFに記録された中心・接続・分裂とする。予測候補を既知注釈に対応付けて教師を作り、検出予測自体を正解とはみなさない。疎い注釈の未知部分を真の負例と断定せず、既存lossの教師maskとその変更を明示的に比較する。未承認の人手注釈や未取得の密な領域教師をあるものとして設計しない。
- [`exp028_direct_graph_prediction`](../experiments/exp028_direct_graph_prediction/)は主催者の既知接続から娘ごとに母または対応なしを学ぶ。ILP擬似教師と未知の負例化は使わない。詳細は同実験の要件を参照。
- 初回の学習比較では検出候補生成・座標・前処理・復号を対照と揃える。候補回収、中心補正、局所再推論は後続の別比較として明示し、公開検出器の重み更新を含めない。検出不足が判明しても検出器の学習を自動で再開せず、固定候補の上限と追加候補回収の条件を記録する。
- 評価は現行公式指標とその成分を使い、両胚別の改善・悪化、失敗と有効件数を示す。公開重みが評価胚を学習した、または来歴不明の場合は、固定公開モデル下の条件付きの比較と明記し、独立した交差検証（CV）とは呼ばない。トラッカーだけの分割では画像モデル由来の学習内評価は解消しない。既存の胚を分けた自前予測は補助診断として区別し、教師・設定の選択は学習側内部に限定する。
- 検出器の再学習、画像特徴抽出器の更新・事前学習、新規画像モデルの学習を要する原案はP4で保留し、再開条件を各詳細に記す。方針変更の明示承認なしに全層更新へ戻さない。固定特徴で別の比較に組み直す場合も、入力・学習対象・対照の変更を詳細へ明記する。

## 承認済みの進め方

- 2026-09-11の「すべて推奨でいいです」による無料枠・課金なし、両胚の改善、人手注釈を当面行わない方針は継続する。FOCUSの取得同意は含まれない。
- 2026-09-12の「公開検出器を固定し、トラッカーを学習する」前提への変更により、従来の「自前検出器を基準とし、外部重みは追加候補のみ」と、検出器学習を先に行う未着手候補の順序を更新した。同じ学習方針・予算の承認を再質問しない。
- 2026-09-12の方針更新時点では新規実験を開始しなかった。2026-09-13に`public_notebook_replay`の実験化・実装・Kaggle submissionが順に承認され、`exp013_public_notebook_replay`へ移行した。submission ref `56199738`のPublic LBは`0.944`。既承認実験を新方針の比較対象として使う場合も、実際の学習来歴を示す。
- 精度と計算費用の実証を提示し、採否・完了はユーザーが判断する。提出推論12時間以内という既存条件を維持する。

## 現在の重点

1. 採用済み[exp043](../experiments/exp043_x138_self_trained_head/result.md)のPublic LB 0.950を提出全体の基準とする。旧exp013は0.944。差を座標head単独やtracker学習の改善とは解釈しない。
2. [exp045の座標・接続診断](../experiments/exp045_x138_coordinate_effect_audit/result.md)は完了。採用済みheadの学習動画を除く両胚各10動画で、補正ありのtrain内scoreが0.879239→0.890348へ改善した。固定公開画像モデル下の条件付き結果であり、Public LBでのhead単独効果は未確認。
3. exp049後のP1は[母と2娘の共有得点](shared_edge_graph_learning.md)。現行exp043・拡張候補と旧費用・同候補とOptuna調整費用・同候補と学習得点の4条件を同一実験で比較する。費用は学習側内部で選び、学習方式全体が調整済み費用を上回るか確認する。P2は[局所graph選択](x138_local_graph_choice.md)、[近傍運動](neighbor_dynamics.md)、[検出回収](x138_detection_recovery.md)。後続は残る誤りで選び、GNNであること自体を優先理由にしない。
4. Transformer内部の特徴・attention追加、履歴延長、対照損失などは再開条件付きP4へ下げる。[exp040修正版](../experiments/exp040_trackastra_association/result.md)・[exp041](../experiments/exp041_past_feature_cross_attention/result.md)の単体未達は根拠だが、手法全体の無効や未測定の公式score低下は主張しない。
5. exp043入力のtrain対照は必要な改善実験内で取得する。exp015/016の旧cache・数値は直接代用しない。公開モデルと座標headの学習来歴を含む条件付き比較であり、独立CVとPublic LBの整合は未判定。

### 現行の比較基準

- 公式評価器は[exp003の照合結果](../experiments/exp003_official_metric_audit/result.md)を参照する。実験横断の最新結果は[`experiment_summary.md`](../experiment_summary.md)、提出履歴は[`SUBMISSIONS.md`](../SUBMISSIONS.md)、数値と実行状態は各実験の`metrics.json`を正とする。
- 自前学習の補助基準は[exp005の結果](../experiments/exp005_embryo_holdout_batch8/result.md)と[exp006の結果](../experiments/exp006_embryo_holdout_seed314159/result.md)。既存の予測・評価集計を再利用し、今回の方針変更のために検出器を再学習しない。Kaggle上の候補生成物とローカルの集計は所在を区別する。
- 公開重みはexp011で選定。今後の改善案は[exp043の採用構成](../experiments/exp043_x138_self_trained_head/result.md)を基準とし、exp016は旧構成の評価手順の参考に留める。Public LB 0.950とtrain診断は別指標で、独立CVとは呼ばない。

## アイデアバックログ

この節と`backlog/`の作成・更新・削除は`kaggle-strategy`が担当する。上位仮説は既存の系譜を維持し、候補の保留を仮説の棄却や実験の不採用とは扱わない。

### 検証中の仮説

| 仮説ID | 仮説 | 対応する未着手候補 | 対応する実験 | 残っている問い |
| --- | --- | --- | --- | --- |
| `HYP-20260909-01` | 主催者公開コードの `TemporalUNet3D` と `SimpleNodeTransformer` をrandom initializationから公式公開checkpointと同じ3 epochs学習し、そのcheckpointでhidden testを推論すれば、学習から再現可能な3D U-NetのPublic LB比較基準を確立できる。 | — | [`exp001_temporal_unet3d_baseline`](../experiments/exp001_temporal_unet3d_baseline/)<br>[`exp002_unet3d_expandable_segments`](../experiments/exp002_unet3d_expandable_segments/) | 胚を分けた主評価用CVとPublic LBの整合、診断用holdoutとPublic LBの関係 |
| `HYP-20260910-01` | 未注釈の中心・接続を未知として扱う教師と損失にすると、実在する細胞や第2娘を抑える学習が減り、公式指標が改善する。 | [`sample_loss_balance`](sample_loss_balance.md)<br>[`sparse_det_mask`](sparse_det_mask.md)<br>[`dense_region_labels`](dense_region_labels.md) | [`exp019_partial_edge_mask`](../experiments/exp019_partial_edge_mask/) | 負例の保証、過検出への退化、胚ごとの効果 |
| `HYP-20260910-02` | 中心位置を固定せず画像と接続の情報で補正・選択すると、位置誤差に由来する一対一対応の失敗と誤接続を減らせる。 | [`x138_author_head_comparison`](x138_author_head_comparison.md)<br>[`anisotropic_position`](anisotropic_position.md)<br>[`match_radius_point`](match_radius_point.md)<br>[`joint_position_edges`](joint_position_edges.md)<br>[`image_label_centers`](image_label_centers.md) | - | exp043の直接実験で位置補正は実装済み。接続への単独寄与は座標・接続診断へ引継ぎ。分布・共同選択は未検証。 |
| `HYP-20260910-03` | 母細胞・2娘・前後画像を一つの分裂事象として扱えば、独立した接続や距離だけの判定より分裂と通常継続を区別できる。 | [`division_triplets`](division_triplets.md)<br>[`division_local_ilp`](division_local_ilp.md)<br>[`division_time_dist`](division_time_dist.md)<br>[`division_state_model`](division_state_model.md)<br>[`division_search_gate`](division_search_gate.md)<br>[`division_context`](division_context.md) | [`exp026_division_candidate_budget`](../experiments/exp026_division_candidate_budget/) | 同予算での組回収、合成教師の実画像への転移、局所再割当。exp023の教師接続を学習効果とは扱わない |
| `HYP-20260910-04` | 同じ動画の画像から運動・見た目・相対配置の基準を作ると、動画共通の基準だけでは誤る接続を改善できる。 | [`video_calibration`](video_calibration.md)<br>[`motion_reference`](motion_reference.md)<br>[`appearance_reference`](appearance_reference.md)<br>[`relative_neighbors`](relative_neighbors.md) | - | 予測参照の汚染、利用できる範囲、参照不足時の処理。appearance_referenceで履歴の追加と対照損失を別比較する |
| `HYP-20260910-05` | 密な検出・領域・移動・軌跡の不完全な予測を信頼度付き教師として使うと、疎い公式注釈だけの学習より細胞と接続を回収できる。 | [`focus_center_teacher`](focus_center_teacher.md)<br>[`teacher_agreement`](teacher_agreement.md)<br>[`track_teacher`](track_teacher.md)<br>[`flow_feature_teacher`](flow_feature_teacher.md)<br>[`shape_aux_teacher`](shape_aux_teacher.md)<br>[`teacher_review_labels`](teacher_review_labels.md) | - | 教師の来歴と誤り、fold分離、教師から生徒への改善移行。対応候補は現方針では保留し、詳細の再開条件を確認する。 |
| `HYP-20260910-06` | 推論で現れる中間候補の誤差を再現した学習と実予測への調整により、現行の検出候補学習より画像からの補正能力を高められる。 | [`correlated_jitter`](correlated_jitter.md)<br>[`missing_frame_noise`](missing_frame_noise.md)<br>[`real_error_finetune`](real_error_finetune.md)<br>[`duplicate_swap_noise`](duplicate_swap_noise.md)<br>[`contrastive_context_noise`](contrastive_context_noise.md)<br>[`x138_history_error_training`](x138_history_error_training.md) | - | 測れる誤差の範囲、時間相関、誤修正、実予測への移行 |
| `HYP-20260910-07` | 関係が既知の画像変形・合成分裂・未注釈画像の予測を使うと、少ない実注釈だけの学習より実画像の対応と分裂を学べる。 | [`synthetic_divisions`](synthetic_divisions.md)<br>[`division_lookalikes`](division_lookalikes.md)<br>[`masked_video_pretrain`](masked_video_pretrain.md) | [`exp008_deformation_pair_aux`](../experiments/exp008_deformation_pair_aux/) | 実画像への移行、合成の識別可能性、変形と系譜の整合 |
| `HYP-20260910-08` | 細胞の集合・領域・軌跡を画像から直接推定する表現なら、既存の極大値検出で失う細胞や接続を別の誤り方で回収できる。 | [`voxel_time_affinity`](voxel_time_affinity.md)<br>[`trajectory_set`](trajectory_set.md)<br>[`inverse_cell_image`](inverse_cell_image.md)<br>[`slice_3d_instances`](slice_3d_instances.md)<br>[`direct_center_set`](direct_center_set.md)<br>[`nnunet_instance_segmentation`](nnunet_instance_segmentation.md) | - | 部分注釈からの個数識別、領域教師の品質、接触細胞の分離と中心への変換、開発・推論費用。対応候補は現方針では保留し、詳細の再開条件を確認する。 |
| `HYP-20260910-09` | 異なる観測・表現・接続方法の候補と不確実性を残し画像から選ぶと、同系統の平均では直らない誤りを回収できる。 | [`x138_detection_recovery`](x138_detection_recovery.md)<br>[`x138_local_graph_choice`](x138_local_graph_choice.md)<br>[`one_two_cells`](one_two_cells.md)<br>[`graph_support_fusion`](graph_support_fusion.md)<br>[`image_condition_gate`](image_condition_gate.md)<br>[`whole_graph_choice`](whole_graph_choice.md)<br>[`contrastive_edge_candidates`](contrastive_edge_candidates.md)<br>[`candidate_union`](candidate_union.md)<br>[`position_mixture`](position_mixture.md) | [`exp047_x138_edge_candidates`](../experiments/exp047_x138_edge_candidates/)<br>[`exp049_x138_edge_selection_diagnostic`](../experiments/exp049_x138_edge_selection_diagnostic/) | exp047で候補回収は増えたが両胚の最終scoreは低下。exp049で追加した既知辺が通常ILPで失われることを確認。得点・費用・制約の変更、検出回収、別候補源、画像に応じた選択が残る。 |
| `HYP-20260910-10` | 多時点情報で対照より接続・分裂と最終graphを改善できる。 | [`neighbor_dynamics`](neighbor_dynamics.md)<br>[`tracklet_join`](tracklet_join.md)<br>[`latent_missing_nodes`](latent_missing_nodes.md)<br>[`x138_window_features`](x138_window_features.md)<br>[`position_history_tracker`](position_history_tracker.md)<br>[`multi_time_past_candidate_attention`](multi_time_past_candidate_attention.md)<br>[`x138_two_step_links`](x138_two_step_links.md)<br>[`long_window_links`](long_window_links.md) | [`exp025_kalman_hungarian_links`](../experiments/exp025_kalman_hungarian_links/)<br>[`exp027_multi_frame_tracker`](../experiments/exp027_multi_frame_tracker/)<br>[`exp032_three_frame_ten_epoch_training`](../experiments/exp032_three_frame_ten_epoch_training/)<br>[`exp035_velocity_features`](../experiments/exp035_velocity_features/)<br>[`exp037_past_candidate_attention`](../experiments/exp037_past_candidate_attention/)<br>[`exp038_past_candidate_knn_attention`](../experiments/exp038_past_candidate_knn_attention/)<br>[`exp040_trackastra_association`](../experiments/exp040_trackastra_association/)<br>[`exp041_past_feature_cross_attention`](../experiments/exp041_past_feature_cross_attention/)<br>[`exp048_x138_primary_past_feature_attention`](../experiments/exp048_x138_primary_past_feature_attention/) | 単体と最終出力を区別。exp048の直前候補画像特徴をprimary内部で使う初回設計は両胚のpair進行条件に届かず、ユーザーが不採用と判断。ほかの多時点案と最終graphへの効果は未検証。 |
| `HYP-20260910-11` | 対応なしと観測可能な構造制約を学習・復号へ明示すると、誤接続を抑えながら正しい継続と分裂を保持できる。 | [`shared_edge_graph_learning`](shared_edge_graph_learning.md)<br>[`final_graph_edit`](final_graph_edit.md)<br>[`neural_graph_selection`](neural_graph_selection.md)<br>[`explicit_no_match`](explicit_no_match.md)<br>[`image_count_prior`](image_count_prior.md)<br>[`known_parent_constraint`](known_parent_constraint.md) | [`exp018_graph_cost_scale`](../experiments/exp018_graph_cost_scale/)<br>[`exp028_direct_graph_prediction`](../experiments/exp028_direct_graph_prediction/)<br>[`exp029_mother_daughter_set_selection`](../experiments/exp029_mother_daughter_set_selection/) | exp049後は同一評価対象で候補拡張、学習側内部でのOptuna費用調整、共有得点学習の追加効果を順に比較。部分教師と最終出力への効果は未検証。 |
| `HYP-20260910-12` | 等価な処理の再利用や局所的な計算配分により、高解像度・多時点・密な候補の手法を予算内で比較し最終精度を改善できる。 | [`uncertain_highres`](uncertain_highres.md)<br>[`sparse_motion_graph`](sparse_motion_graph.md)<br>[`public_x138_tracker_comparison`](public_x138_tracker_comparison.md)<br>[`distill_reinvest`](distill_reinvest.md) | [`exp011_public_detector_selection`](../experiments/exp011_public_detector_selection/)<br>[`exp013_public_notebook_replay`](../experiments/exp013_public_notebook_replay/)<br>[`exp014_exact_window_cache`](../experiments/exp014_exact_window_cache/)<br>[`exp016_frozen_image_encoder`](../experiments/exp016_frozen_image_encoder/)<br>[`exp042_public_x138_replay`](../experiments/exp042_public_x138_replay/) | 追加座標補正を含むx138の再現性・費用、公開/exp016重みの順位。 |
| `HYP-20260910-13` | 位置と系譜を保って撮像条件や境界・密度を変える学習により、別の胚の見え方に対する性能低下を抑えられる。 | [`anisotropic_blur`](anisotropic_blur.md)<br>[`photometric_shift`](photometric_shift.md)<br>[`lineage_density_aug`](lineage_density_aug.md)<br>[`crop_boundary_aug`](crop_boundary_aug.md) | - | 実際の胚差との対応、ラベル整合、片側胚の悪化。対応候補は現方針では保留し、詳細の再開条件を確認する。 |
| `HYP-20260910-14` | 現行公式指標・胚を分けた評価・段階別の上限検査を用いると、独自proxyや学習内指標では見えない候補の順位差と失敗箇所を識別できる。 | - | [`exp003_official_metric_audit`](../experiments/exp003_official_metric_audit/)<br>[`exp004_embryo_holdout_baseline`](../experiments/exp004_embryo_holdout_baseline/)<br>[`exp005_embryo_holdout_batch8`](../experiments/exp005_embryo_holdout_batch8/)<br>[`exp007_graph_checkpoint_selection`](../experiments/exp007_graph_checkpoint_selection/)<br>[`exp012_group_error_readout`](../experiments/exp012_group_error_readout/)<br>[`exp015_oracle_stage_limits`](../experiments/exp015_oracle_stage_limits/)<br>[`exp045_x138_coordinate_effect_audit`](../experiments/exp045_x138_coordinate_effect_audit/) | 位置・採点・候補保持・最終graphの作用段階。対照学習固有の診断はcontrastive_parent_child内へ統合し関連仮説として維持。 |
| `HYP-20260911-01` | nnU-Netのデータに応じた前処理・構造・学習設定を中心マップの予測へ適応すると、疎注釈を適切に扱う条件で現行検出器より細胞を回収でき、両胚の公式接続・分裂指標が改善する。 | [`nnunet_center_detection`](nnunet_center_detection.md) | - | 中心教師とnnU-Net設定の寄与、背景と未知領域の識別、時間入力と接続特徴、学習・推論費用。対応候補は現方針では保留し、詳細の再開条件を確認する。 |
| `HYP-20260915-01` | 固定画像特徴から学ぶtrackerへ、正しい親子対応を近づけ確定した誤親から離す対照損失を加えると、接続分類損失だけの場合より近傍の取り違えと両胚の公式graph誤りを減らせる。 | [`contrastive_parent_child`](contrastive_parent_child.md) | - | 共有tracker特徴への寄与、疎い教師と分裂の整合、固定候補内の順位から公式指標への移行。履歴・欠落への頑健性・候補回収は関連仮説で別比較する。 |
| `HYP-20260920-01` | 公開画像特徴を固定して既存のprimary trackerを追加学習する際、学習を3エポックより長くすると、同じ教師・損失・復号でも両胚の公式graph精度を改善できる。 | — | [`exp024_tracker_six_epochs`](../experiments/exp024_tracker_six_epochs/) | 6エポックで後半の重みが内部検証から選ばれるか、両胚の公式scoreへ改善が残るか。異なる学習率や長さでも成立するかは別検証。 |
| `HYP-20260920-02` | 固定した公開画像特徴から隣接時刻の接続を学ぶ際、各時刻のcell間を先にSelf-Attentionで文脈化してから既存のCross-Attentionを行うと、現行のCross-Attentionのみより親候補の取り違えが減り、両胚の公式graph指標が改善する。 | — | [`exp025_frame_self_attention`](../experiments/exp025_frame_self_attention/)<br>[`exp030_frame_self_attention_diagnostics`](../experiments/exp030_frame_self_attention_diagnostics/)<br>[`exp031_frame_self_attention_spatial`](../experiments/exp031_frame_self_attention_spatial/)<br>[`exp033_frame_self_attention_diagnostics`](../experiments/exp033_frame_self_attention_diagnostics/)<br>[`exp034_frame_self_attention_distance_bias`](../experiments/exp034_frame_self_attention_distance_bias/) | exp033では一貫したpair改善はなかった。exp034は物理距離の学習可能な負の二乗biasを1設定×2fold×3epochで完走したが、内部検証の改善が極小だったため全graph推論へ進まず完了した。公式graph改善は未検証。 |
| `HYP-20260920-03` | 固定候補の検出得点・DoG・HOGを個別入力すると、同構造対照より両胚の公式指標が改善する。 | [`hog_features`](hog_features.md) | [`exp036_detection_score_pair_features`](../experiments/exp036_detection_score_pair_features/)<br>[`exp039_dog_features`](../experiments/exp039_dog_features/) | DoGはexp039の隣接ペア進行条件未達で不採用。HOGの入力方式と効果、特徴間の相補性が未検証。 |

2026-09-26: 母と2娘の共有得点を近傍運動より先に設計する。短い動作確認後、学習・両胚の全graph公式評価を一続きで行う方針は維持。独立予備実験は増やさず、実験化は未承認。

### 未着手バックログ

2026-09-24の[統合・分割と全候補再評価](../docs/surveys/biohub-backlog-reorganization_20260924.md)を反映。優先度は着手順であり、採否・実験化承認ではない。P4は各詳細の再開条件が成立するまで保留する。2026-09-26のexp047とexp049の結果を反映し、共有得点をP1、検出回収をP2へ変更した。他候補の優先度は維持する。

2026-09-25: 作者headの取得を受け、exp043のhead交換比較をP2へ追加。既存候補の優先度は維持する。

教師・特徴診断は対照学習候補へ、旧座標補正の未測定効果はexp045の座標・接続診断へ移行。位置変更は局所graph選択から既存の共同位置選択候補へ分離した。候補名が似ていても入力・出力・比較が異なる案は維持する。

| 優先度 | 対応仮説 | アイデア | 短い要約 | 主な先行条件 / 依存 | 状態 |
| --- | --- | --- | --- | --- | --- |
| P1 | `HYP-20260910-11` | [`shared_edge_graph_learning`](shared_edge_graph_learning.md) | Optuna費用調整を対照に母と2娘の共有得点を比較 | 現行・拡張候補と旧費用・Optuna調整費用・学習得点の4条件。損失・探索範囲と予算・新評価対象を確定し、費用調整と学習から全graphまで進める。 | `検討メモ・設計不可` |
| P2 | `HYP-20260910-02` | [`x138_author_head_comparison`](x138_author_head_comparison.md) | exp043の座標補正重みだけを作者版へ交換 | 作者head取得済み。同じ20動画のexp045自前head対照を条件一致時に再利用し、両胚の公式graph指標を比較する。 | `設計可能・実験化未承認` |
| P2 | `HYP-20260910-09` | [`x138_local_graph_choice`](x138_local_graph_choice.md) | 位置固定で複数局所graphを保持・選択 | 正しい局所得点が改善しても全体競合で失う場合の後続。正解なしの候補生成・選別と境界整合を確認する。 | `検討メモ・設計不可` |
| P2 | `HYP-20260910-10` | [`neighbor_dynamics`](neighbor_dynamics.md) | 近傍運動を共有するGNNで再接続 | 通常接続の運動の取り違えが残る場合の後続。同じ近傍情報の現行ルールを対照とし、参照不足時の処理を固定する。 | `検討メモ・設計不可` |
| P2 | `HYP-20260910-09` | [`x138_detection_recovery`](x138_detection_recovery.md) | 得点・画像・接続から回収候補を選ぶ | 共有得点の設計に続く別の誤りの候補。既知接続の候補内回収と教師を確認し、出力・損失・競合解消を決める。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-03` | [`division_triplets`](division_triplets.md) | 同じ分裂組モデルで合成事前学習の有無を比較 | 公開合成データと実画像の固定候補・特徴・教師対応を揃え、同じ組モデル・実教師・復号で合成事前学習の有無を比較できること。exp043へ移す場合の再抽出を旧cacheで代用しない。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-06` | [`correlated_jitter`](correlated_jitter.md) | 時間相関のある位置誤差を学ぶ | [`exp012_group_error_readout`](../experiments/exp012_group_error_readout/)。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-06` | [`missing_frame_noise`](missing_frame_noise.md) | 連続する見逃しから補正を学ぶ | [`exp012_group_error_readout`](../experiments/exp012_group_error_readout/)。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-06` | [`real_error_finetune`](real_error_finetune.md) | 合成誤差の補正器を実予測へ調整 | correlated_jitter。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `設計可能・実験化未承認` |
| P3 | `HYP-20260910-12` | [`uncertain_highres`](uncertain_highres.md) | 曖昧な領域だけ高解像度化 | [`exp012_group_error_readout`](../experiments/exp012_group_error_readout/)。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-02` | [`anisotropic_position`](anisotropic_position.md) | zとxyの位置不確実性を分ける | [`exp015_oracle_stage_limits`](../experiments/exp015_oracle_stage_limits/)。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `設計可能・実験化未承認` |
| P3 | `HYP-20260910-02` | [`match_radius_point`](match_radius_point.md) | 7µm対応を意識して点を選ぶ | anisotropic_position。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `設計可能・実験化未承認` |
| P3 | `HYP-20260910-03` | [`division_local_ilp`](division_local_ilp.md) | 同じ得点で専用分裂判定と局所ILPを比較 | exp025、exp026、division_triplets。通常割当を選び直せる局所範囲・境界・校正方式を確定する。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-04` | [`video_calibration`](video_calibration.md) | 動画内の見え方で調整する | [`exp012_group_error_readout`](../experiments/exp012_group_error_readout/)。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-06` | [`duplicate_swap_noise`](duplicate_swap_noise.md) | 重複・取り違えから補正を学ぶ | [`exp012_group_error_readout`](../experiments/exp012_group_error_readout/)。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-09` | [`one_two_cells`](one_two_cells.md) | 1細胞か2細胞かを前後で選ぶ | [`exp015_oracle_stage_limits`](../experiments/exp015_oracle_stage_limits/)、同一の固定公開検出器の抑制前候補と前後の画像支持。candidate_unionは不要。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-10` | [`tracklet_join`](tracklet_join.md) | 短い軌跡の端同士を接続 | [`exp015_oracle_stage_limits`](../experiments/exp015_oracle_stage_limits/)。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-10` | [`latent_missing_nodes`](latent_missing_nodes.md) | 見逃した時点を画像から回収 | missing_frame_noise。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-11` | [`final_graph_edit`](final_graph_edit.md) | 最終graphの修正 | exp043。修復上限・教師を確認。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-10` | [`x138_window_features`](x138_window_features.md) | 前後窓の固定画像特徴で接続を学ぶ | 同じx138入力の小規模診断で前後窓が競合親を識別し、既存の後処理でも改善が残り得る証拠を得る。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-12` | [`sparse_motion_graph`](sparse_motion_graph.md) | 近傍候補で多時点予測を可能に | 精度上の根拠がある先行案で候補間計算が律速と実測され、同予算の候補保持と両娘回収を失わない比較が必要になった場合に再開する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260915-01` | [`contrastive_parent_child`](contrastive_parent_child.md) | 教師・特徴診断を含め親子特徴を対照学習 | 統合した診断で確定負例と競合親の識別が成立し、exp043で得点差が最終接続へ残る使用先を特定する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-10` | [`position_history_tracker`](position_history_tracker.md) | 予測対応した位置列を入力して接続を学習 | 予測履歴と正解履歴を区別して残る誤りを診断し、実予測履歴の追加で改善可能な接続を特定する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-10` | [`multi_time_past_candidate_attention`](multi_time_past_candidate_attention.md) | 過去座標の複数時点集約 | 1時点版または作業中のexp044に、同入力で両胚の単体条件を満たす証拠と最終出力への作用を得て、追加時点の費用を見積もる。exp044の結果は現時点で未取得。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-01` | [`sample_loss_balance`](sample_loss_balance.md) | トラッカー損失の動画間の寄与を揃える | 動画ごとの有効窓数・勾配寄与と失敗の関係を確認し、同じ露出・教師maskで集約だけを変える比較にする。設計可能の状態は維持する。 | `設計可能・実験化未承認` |
| P4 | `HYP-20260910-03` | [`division_time_dist`](division_time_dist.md) | 分裂時刻の複数候補を残す | 誤りが時刻のずれに由来すると確認でき、固定候補・得点の時間分布だけを変える比較が定まる。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-03` | [`division_state_model`](division_state_model.md) | 継続・分裂・観測不能を選ぶ | 観測不能の状態を識別できる教師・出力・損失が定まり、共有得点案では測れない効果を明示する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-03` | [`division_search_gate`](division_search_gate.md) | 画像変化で分裂探索箇所を選ぶ | 同予算の距離候補が失う分裂を画像の時間変化で回収できることと、特徴取得費用・未知教師の扱いを確認する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-04` | [`motion_reference`](motion_reference.md) | 組織の移動を差し引く | 画像運動が現行近傍移動と異なる誤りを補う範囲と抽出費用を確認する。neighbor_dynamicsへ同じ方法とみなして統合しない。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-04` | [`appearance_reference`](appearance_reference.md) | 予測履歴と次の細胞を対照的に照合 | 実予測で対応付けた履歴に、現在特徴だけでは区別できない競合相手を分ける情報があり、誤履歴時の悪化を抑えられると示す。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-04` | [`relative_neighbors`](relative_neighbors.md) | 周囲の細胞との相対配置を使う | 静的な相対配置の差に有効な情報が残ると示す。再接続を変えるneighbor_dynamicsとは目的・対照を分ける。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-07` | [`synthetic_divisions`](synthetic_divisions.md) | ぼけ・雑音を含む分裂を合成 | 既存合成データに不足する形状・ぼけ・雑音・系譜を学習側で特定し、生成器変更だけの比較を設計する。division_tripletsへ異なる情報源を黙って統合しない。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-07` | [`division_lookalikes`](division_lookalikes.md) | 分裂に似た通常事象を合成 | 公開合成教師の通常継続例だけでは解消しない実画像の誤分裂を特定し、同じ正例・総学習量で追加教師の有無を比較する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-09` | [`graph_support_fusion`](graph_support_fusion.md) | 同一検出器の複数トラッカーの支持を融合 | 同入力の保存済み複数出力で、既知の正解を補い合う量と未知を負例にしない選別条件を確認する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-09` | [`image_condition_gate`](image_condition_gate.md) | 画像条件で複数トラッカーの出力を重み付け | graph_support_fusionの入力と補完性を確認し、固定重みと画像条件付き重みを学習側だけで比較できること。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-09` | [`whole_graph_choice`](whole_graph_choice.md) | 同一検出器の候補graphを1つ選ぶ | 同入力の複数完成graphで、動画単位の選択余地と画像条件による選別可能性を確認する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260920-03` | [`hog_features`](hog_features.md) | HOG特徴・類似度 | 補正後の候補または回収候補で、勾配方向の情報が既存画像特徴と異なる識別力を持つと小規模診断で示す。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-11` | [`neural_graph_selection`](neural_graph_selection.md) | 接続を直接選択 | 教師に整合した合法解、小例での分裂適合、両胚での誤接続・分裂保持、費用が成立し、既存ILPを置換する必要性を示す。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-11` | [`explicit_no_match`](explicit_no_match.md) | 対応なしを明示的な出力に | 対応なしを保証できる教師と、exp028の構造変更を混ぜない出力有無の比較を確定する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-06` | [`contrastive_context_noise`](contrastive_context_noise.md) | 候補欠落に強いtracker特徴を学ぶ | 基本の対照損失と教師を固定し、実際の文脈欠落が誤りを増やす証拠を学習側で得る。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-09` | [`contrastive_edge_candidates`](contrastive_edge_candidates.md) | 特徴の類似度で接続候補を追加 | contrastive_parent_childから同入力の照合特徴を取得し、同じ追加edge数で距離方式より候補回収が増えると確認する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-06` | [`x138_history_error_training`](x138_history_error_training.md) | 実誤履歴で学習し実予測へ調整 | 同じ履歴モデルで実予測入力の誤差が損失原因と確認された場合に再開。exp049のILPでの欠落だけを再開根拠にしない。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-10` | [`x138_two_step_links`](x138_two_step_links.md) | 3時点GNNと2時点先対応で経路選択 | GNN案内で後続。先行2案の結果と残り時間・計算枠から再開判断。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-12` | [`public_x138_tracker_comparison`](public_x138_tracker_comparison.md) | 公開/exp016重みを比較 | exp042の忠実再現・exp016の2fold重み。作者headは取得済みだがtracker転用比較の再開目的を要する。 | `設計可能・実験化未承認` |
| P4 | `HYP-20260910-01` | [`sparse_det_mask`](sparse_det_mask.md) | 検出の未知領域を負例から外す | 検出損失と検出器の更新を必要とするため現方針では保留。検出器更新への方針変更が明示承認され、未知領域と背景の教師設計が成立した場合に再検討する。検出不足の診断だけでは再開しない。 | `検討メモ・設計不可` |
| P4 | `HYP-20260911-01` | [`nnunet_center_detection`](nnunet_center_detection.md) | nnU-Netで中心マップを学習して追跡へ渡す | nnU-Netによる検出器学習は現方針の対象外。検出器更新への方針変更が明示承認され、中心教師と学習費用の条件が成立した場合に再検討する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-09` | [`candidate_union`](candidate_union.md) | 異なる検出法の候補を統合 | 原案の領域教師由来の中心が未成立。追加の固定公開情報源と取得条件・候補数を揃える対照・費用が確定するまで保留。単一検出器の閾値変更へ原案を置き換えない。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-02` | [`joint_position_edges`](joint_position_edges.md) | 中心位置と接続を同時に選ぶ | 原案の複数検出、またはx138元位置・補正位置の排他選択。再採点と点別利益を確認。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-03` | [`division_context`](division_context.md) | 分裂前後の形状変化を使う | 5時点の画像特徴を得る方法が固定公開モデルで成立するか未確認。必要な窓の対応と費用を確認して再設計するまで保留。2時点特徴の寄せ集めを元の5時点画像特徴と同等とはみなさない。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-05` | [`focus_center_teacher`](focus_center_teacher.md) | FOCUSの中心を教師に使う | FOCUSの教師で検出器を学習する原案は現方針の対象外。取得条件の本人確認、教師品質と来歴、検出器更新への明示承認が揃った場合に再検討する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-05` | [`teacher_agreement`](teacher_agreement.md) | 複数教師の一致で重み付け | FOCUS等の複数教師と検出生徒の学習を要する。focus_center_teacherの再開条件と教師間一致の品質確認が揃った場合に再検討する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-05` | [`track_teacher`](track_teacher.md) | 短い軌跡を対応学習の教師に | 原案のTrackastra入力となる領域分割が未成立。条件を満たす固定領域予測と軌跡教師の来歴・品質が確定するまで保留。点検出だけを同じ入力の代用にはしない。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-05` | [`flow_feature_teacher`](flow_feature_teacher.md) | 移動場を特徴の補助教師に | 画像特徴抽出器を補助損失で更新する原案は固定方針と両立しない。画像特徴の更新への方針変更が承認された場合に再検討する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-05` | [`shape_aux_teacher`](shape_aux_teacher.md) | 領域の形状を補助的に学習 | 領域教師による画像特徴の学習を要する。教師の取得・品質確認と画像特徴の更新への方針変更が成立するまで保留する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-07` | [`masked_video_pretrain`](masked_video_pretrain.md) | 隠した画像を予測して事前学習 | 画像モデルの事前学習を要するため保留。画像特徴抽出器の更新への方針変更と計算予算が成立した場合に再検討する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-09` | [`position_mixture`](position_mixture.md) | 位置分布を点にする前に融合 | 候補統合に必要な複数の位置分布とその校正が未成立。candidate_unionの再開条件と、固定重みの下で追加位置分布を作る方法が揃った場合に再検討する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-10` | [`long_window_links`](long_window_links.md) | 複数時点の特徴で接続を予測 | 2時点と5時点の画像モデルを比較する原案は保留。選定した固定公開モデルが必要な時間窓を扱える証拠、または画像モデル変更への明示承認が必要。複数の2時点特徴の連結を元の5時点画像モデルと同じ実装とは扱わない。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-12` | [`distill_reinvest`](distill_reinvest.md) | 複数予測を小モデルへ蒸留 | 小さい画像モデルへの蒸留は新しい検出器の学習を要する。検出器更新への方針変更と複数教師・費用の成立まで保留する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-13` | [`anisotropic_blur`](anisotropic_blur.md) | zとxyのぼけ・雑音を変える | 検出・接続を画像変形で学習する原案は保留。検出器更新を認める方針変更、または画像特徴を固定したトラッカー専用の別設計と特徴再抽出費用が具体化した場合に再検討する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-13` | [`photometric_shift`](photometric_shift.md) | 輝度・背景変化に対応する | 画像拡張による検出・接続の学習を前提とした原案は保留。検出器更新を認める方針変更、または固定画像モデルを使うトラッカー専用の別設計と再抽出費用が具体化した場合に再検討する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-13` | [`lineage_density_aug`](lineage_density_aug.md) | 系譜を保って密度を変える | 合成画像による検出・接続の学習を前提とした原案は保留。固定検出器を使うトラッカー専用の教師設計・実画像との差・再抽出費用を具体化するか、検出器更新への方針変更が成立した場合に再検討する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-13` | [`crop_boundary_aug`](crop_boundary_aug.md) | 境界での出入りを学習する | 画像cropに伴う検出・接続の学習を前提とした原案は保留。固定画像モデルから境界の対応だけを学習する別設計と教師の未知状態を具体化するか、画像側更新への方針変更が成立した場合に再検討する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-01` | [`dense_region_labels`](dense_region_labels.md) | 少数の完全注釈領域を追加する | 人手注釈を行わない既存方針により保留。追加注釈の明示承認と完全注釈領域の品質確認が必要。検出器も更新する場合は学習方針の変更承認を要する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-02` | [`image_label_centers`](image_label_centers.md) | 画像中心と注釈位置を分ける | 画像中心と注釈位置を区別する独立した教師が未成立。追加教師の取得方法と固定検出器の下流だけを学習する設計が成立した場合に再検討する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-05` | [`teacher_review_labels`](teacher_review_labels.md) | 教師が異なる領域を追加注釈 | 人手の追加注釈と複数教師を要する。追加注釈の明示承認とteacher_agreementの先行条件が揃うまで保留する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-08` | [`voxel_time_affinity`](voxel_time_affinity.md) | voxelの時空間関係から追跡 | 画像から検出・領域・軌跡を推定する新規モデルの学習を要するため保留。画像側の学習への方針変更が明示承認され、原案の教師・計算費用が成立した場合に再検討する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-08` | [`trajectory_set`](trajectory_set.md) | 画像から短い軌跡集合を出す | 画像から検出・領域・軌跡を推定する新規モデルの学習を要するため保留。画像側の学習への方針変更が明示承認され、原案の教師・計算費用が成立した場合に再検討する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-08` | [`inverse_cell_image`](inverse_cell_image.md) | 細胞像で画像を説明して追跡 | 画像から検出・領域・軌跡を推定する新規モデルの学習を要するため保留。画像側の学習への方針変更が明示承認され、原案の教師・計算費用が成立した場合に再検討する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-08` | [`slice_3d_instances`](slice_3d_instances.md) | 2D断面の領域を3Dへ統合 | 画像から検出・領域・軌跡を推定する新規モデルの学習を要するため保留。画像側の学習への方針変更が明示承認され、原案の教師・計算費用が成立した場合に再検討する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-08` | [`direct_center_set`](direct_center_set.md) | 画像から可変個数の中心集合 | 画像から検出・領域・軌跡を推定する新規モデルの学習を要するため保留。画像側の学習への方針変更が明示承認され、原案の教師・計算費用が成立した場合に再検討する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-08` | [`nnunet_instance_segmentation`](nnunet_instance_segmentation.md) | nnU-Netの3D領域から細胞を分離して追跡 | 新規領域モデルの学習と領域教師を必要とする。検出器更新への方針変更と、来歴・品質を確認した領域教師が揃うまで保留する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-11` | [`image_count_prior`](image_count_prior.md) | 画像からの細胞数を補助情報に | 画像から細胞数を学ぶ追加モデルと個数教師を要する。固定特徴だけで学習できる具体案と教師の識別可能性、または画像モデル更新への方針変更が成立した場合に再検討する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-11` | [`known_parent_constraint`](known_parent_constraint.md) | 既存softmaxとの差を要確認 | 既存softmaxと異なる学習目的をまだ特定できていない。追加する制約の非重複性と、固定検出器下での反証可能な比較が成立するまで保留する。 | `検討メモ・設計不可` |
