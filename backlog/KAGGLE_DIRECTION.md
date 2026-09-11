# Kaggle 方針

## 参照する前提

- 機械可読なコンペ設定、データパス、validation、submission、Kaggle runtimeは[`project.yml`](../project.yml)を正とする。
- コンペの公式情報と制約は[`docs/01_competition.md`](../docs/01_competition.md)、評価指標は[`docs/02_metric.md`](../docs/02_metric.md)、CV設計とリーク確認は[`docs/03_validation.md`](../docs/03_validation.md)、データ仕様は[`docs/04_data.md`](../docs/04_data.md)を参照する。
- 保存場所、記録責務、ユーザー判断、用語の規則は[`AGENTS.md`](../AGENTS.md)、コンペ固有の略語とリポジトリ内の管理用語は[`docs/glossary.md`](../docs/glossary.md)を正とする。
- このファイルは、現在の重点、比較基準、検証中の上位仮説、未着手候補の索引だけを管理する。

## 承認済みの進め方

- 2026-09-11: ユーザー『すべて推奨でいいです』。推奨の順序、無料枠内・課金なし、両胚改善、外部重みは選んだ追加候補のみ、人手注釈は当面なしを承認。FOCUSの取得同意は今回の対象外。
- 2026-09-11: batch size 16の[`exp004_embryo_holdout_baseline`](../experiments/exp004_embryo_holdout_baseline/)がfold 1のsmokeでOOMとなった後、ユーザー『その方針で進めてください』により、batch sizeだけを8へ変更する[`exp005_embryo_holdout_batch8`](../experiments/exp005_embryo_holdout_batch8/)の実験化、実装、Kaggle Notebook実行を承認。competition submissionの承認ではない。
- 実施順: [公式評価の照合](../experiments/exp003_official_metric_audit/result.md)、batch size 16で失敗した[胚を分けた基準予測](../experiments/exp004_embryo_holdout_baseline/)、batch size 8で再試行する[`exp005_embryo_holdout_batch8`](../experiments/exp005_embryo_holdout_batch8/)、[誤差分析](group_error_readout.md)と[段階別回収上限の分析](oracle_stage_limits.md)。段階別診断で検出が主な制約と確認できた場合は[検出の未知領域を負例から外す比較](sparse_det_mask.md)を先に行い、その後に[未注釈子に対する接続損失の比較](partial_edge_mask.md)へ進む。検出が主な制約でなければ、診断結果に対応する既存候補を選ぶ。
- 計算費用: Kaggle無料枠内。課金しない。実行前の残量と小規模実測を確認し、後続の推論確認に必要な枠を残す。残量不足なら承認済み方針のまま再開可能なところまで準備し、残量を超えるrunを開始しない。
- 精度判断: 公式指標の両胚での改善、実行失敗の増加なし、提出推論12時間以内。採否・完了は結果提示後のユーザー判断を維持する。
- 外部重みと人手: 当面の基準予測は自前学習。外部重みは選択した追加候補に限定し、人手注釈は当面行わない。FOCUSの条件同意は必要になった時点で本人が確認する。
- 64候補全部を一括実行する承認ではない。選択済みの推奨順序を進め、後続候補の実験化はその依存証拠を確認する。共通方針や選択済み比較の同じ承認を再度求めない。

## 現在の重点

対象コンペを設定した後、公式評価指標とデータ仕様を確認し、再現可能な最初のbaselineとvalidationを確立する。

### 現行の比較基準

公式評価の前提確認は[exp003の結果](../experiments/exp003_official_metric_audit/result.md)を参照する。人工例で公開実装との差を再現し、固定した実予測では一致した。実行証拠を回収し、2026-09-11にユーザーが本監査の完了を承認。[`exp004_embryo_holdout_baseline`](../experiments/exp004_embryo_holdout_baseline/)はKaggle train versions 1-3を実行したが、固定batch size 16のfold 1 smokeがCUDA OOMとなり、full trainingと胚を分けた基準予測は未成立である。直接承認された[`exp005_embryo_holdout_batch8`](../experiments/exp005_embryo_holdout_batch8/)はbatch sizeだけを8へ変更した再試行として実装済みで、`metrics.json`のstatusは`planned`、CVと実行証拠は未取得である。

`exp001_temporal_unet3d_baseline`はbatch size 16のsmoke backwardでT4がOOMとなった。`exp002_unet3d_expandable_segments`のversion 1はメモリ割当変更でsmokeを通過したが、時間予測が11時間gateを超えてfull trainingへ進まなかった。ユーザー承認後のversion 2は12時間gateを通過し、同じbatch size 16で3 epochsを28,488.802秒（約7時間54分49秒）で完走してcheckpointを取得した。inference version 1は実行時に列挙した4件の公開test datasetから26,580行を788.633秒で生成し、提出前検証を通過した。code submission ref `56153451`は固定監視の開始から206分後に完了し、Public LBは`0.453`だった。診断用holdoutのbest selection scoreはepoch 0の0.9016で、胚を分けた主評価用CVは未取得。数値とstatusは[metrics](../experiments/exp002_unet3d_expandable_segments/metrics.json)、version別の進行は[SESSION_NOTES](../experiments/exp002_unet3d_expandable_segments/SESSION_NOTES.md)を正とする。現在のsample holdoutは診断用であり、下記の主評価用候補は学習から除いた胚の予測を別途要する。

exp002の公開test 4動画を固定公式評価器で診断した値はscore `0.3588721993`、node recall `0.3634941118`、division Jaccard `0.0`だった。対象はtrainにも存在し、学習から除いた胚ではないためCVとして使わない。この観測と公開上位Notebookの検出閾値`0.965`に対し、exp002・exp005は`0.99`を使う差は、[`oracle_stage_limits`](oracle_stage_limits.md)で固定checkpointを用いた候補回収の診断対象にする。外側評価胚の正解を使った閾値選択や、この診断だけを実際の改善とすることは行わない。

## アイデアバックログ

この節と`backlog/`の作成・更新・削除は`kaggle-strategy`の手順で行う。上位仮説は複数の未着手候補・実験を束ね、各`backlog/<candidate>.md`は1回の実験として実施できる具体的な設計を保持する。

### 検証中の仮説

| 仮説ID | 仮説 | 対応する未着手候補 | 対応する実験 | 残っている問い |
| --- | --- | --- | --- | --- |
| `HYP-20260909-01` | 主催者公開コードの `TemporalUNet3D` と `SimpleNodeTransformer` をrandom initializationから公式公開checkpointと同じ3 epochs学習し、そのcheckpointでhidden testを推論すれば、学習から再現可能な3D U-NetのPublic LB比較基準を確立できる。 | — | [`exp001_temporal_unet3d_baseline`](../experiments/exp001_temporal_unet3d_baseline/)<br>[`exp002_unet3d_expandable_segments`](../experiments/exp002_unet3d_expandable_segments/) | 胚を分けた主評価用CVとPublic LBの整合、診断用holdoutとPublic LBの関係 |
| `HYP-20260910-01` | 未注釈の中心・接続を未知として扱う教師と損失にすると、実在する細胞や第2娘を抑える学習が減り、公式指標が改善する。 | [`sparse_det_mask`](sparse_det_mask.md)<br>[`partial_edge_mask`](partial_edge_mask.md)<br>[`sample_loss_balance`](sample_loss_balance.md)<br>[`dense_region_labels`](dense_region_labels.md) | - | 負例の保証、過検出への退化、胚ごとの効果 |
| `HYP-20260910-02` | 中心位置を固定せず画像と接続の情報で補正・選択すると、位置誤差に由来する一対一対応の失敗と誤接続を減らせる。 | [`subvoxel_offset`](subvoxel_offset.md)<br>[`image_label_centers`](image_label_centers.md)<br>[`anisotropic_position`](anisotropic_position.md)<br>[`joint_position_edges`](joint_position_edges.md)<br>[`match_radius_point`](match_radius_point.md) | - | 位置誤差が接続を制限する度合い、分布の校正、近接細胞の混同 |
| `HYP-20260910-03` | 母細胞・2娘・前後画像を一つの分裂事象として扱えば、独立した接続や距離だけの判定より分裂と通常継続を区別できる。 | [`division_triplets`](division_triplets.md)<br>[`division_context`](division_context.md)<br>[`division_time_dist`](division_time_dist.md)<br>[`division_state_model`](division_state_model.md)<br>[`division_search_gate`](division_search_gate.md) | - | 3中心の同時回収、境界の未知状態、通常接続との両立 |
| `HYP-20260910-04` | 同じ動画の画像から運動・見た目・相対配置の基準を作ると、動画共通の基準だけでは誤る接続を改善できる。 | [`motion_reference`](motion_reference.md)<br>[`appearance_reference`](appearance_reference.md)<br>[`relative_neighbors`](relative_neighbors.md)<br>[`video_calibration`](video_calibration.md) | - | 予測参照の汚染、利用できる範囲、参照不足時の処理 |
| `HYP-20260910-05` | 密な検出・領域・移動・軌跡の不完全な予測を信頼度付き教師として使うと、疎い公式注釈だけの学習より細胞と接続を回収できる。 | [`focus_center_teacher`](focus_center_teacher.md)<br>[`teacher_agreement`](teacher_agreement.md)<br>[`track_teacher`](track_teacher.md)<br>[`teacher_review_labels`](teacher_review_labels.md)<br>[`flow_feature_teacher`](flow_feature_teacher.md)<br>[`shape_aux_teacher`](shape_aux_teacher.md) | - | 教師の来歴と誤り、fold分離、教師から生徒への改善移行 |
| `HYP-20260910-06` | 推論で現れる中間候補の誤差を再現した学習と実予測への調整により、現行の検出候補学習より画像からの補正能力を高められる。 | [`correlated_jitter`](correlated_jitter.md)<br>[`missing_frame_noise`](missing_frame_noise.md)<br>[`duplicate_swap_noise`](duplicate_swap_noise.md)<br>[`real_error_finetune`](real_error_finetune.md) | - | 測れる誤差の範囲、時間相関、誤修正、実予測への移行 |
| `HYP-20260910-07` | 関係が既知の画像変形・合成分裂・未注釈画像の予測を使うと、少ない実注釈だけの学習より実画像の対応と分裂を学べる。 | [`deformation_pairs`](deformation_pairs.md)<br>[`synthetic_divisions`](synthetic_divisions.md)<br>[`division_lookalikes`](division_lookalikes.md)<br>[`masked_video_pretrain`](masked_video_pretrain.md) | - | 実画像への移行、合成の識別可能性、変形と系譜の整合 |
| `HYP-20260910-08` | 細胞の集合・領域・軌跡を画像から直接推定する表現なら、既存の極大値検出で失う細胞や接続を別の誤り方で回収できる。 | [`voxel_time_affinity`](voxel_time_affinity.md)<br>[`trajectory_set`](trajectory_set.md)<br>[`inverse_cell_image`](inverse_cell_image.md)<br>[`slice_3d_instances`](slice_3d_instances.md)<br>[`direct_center_set`](direct_center_set.md)<br>[`nnunet_instance_segmentation`](nnunet_instance_segmentation.md) | - | 部分注釈からの個数識別、領域教師の品質、接触細胞の分離と中心への変換、開発・推論費用 |
| `HYP-20260910-09` | 異なる観測・表現・接続方法の候補と不確実性を残し画像から選ぶと、同系統の平均では直らない誤りを回収できる。 | [`candidate_union`](candidate_union.md)<br>[`position_mixture`](position_mixture.md)<br>[`graph_support_fusion`](graph_support_fusion.md)<br>[`image_condition_gate`](image_condition_gate.md)<br>[`one_two_cells`](one_two_cells.md)<br>[`whole_graph_choice`](whole_graph_choice.md) | - | 候補数を揃えた追加回収、正解なしの選別、相関した誤り |
| `HYP-20260910-10` | 複数時点の画像と軌跡を使って接続を選べば、短い観測だけで起きる取り違えや一時的な見逃しを修正できる。 | [`long_window_links`](long_window_links.md)<br>[`tracklet_join`](tracklet_join.md)<br>[`latent_missing_nodes`](latent_missing_nodes.md)<br>[`neighbor_dynamics`](neighbor_dynamics.md) | - | 長窓で増える情報、実欠測での回収、境界と隣接辺の整合 |
| `HYP-20260910-11` | 対応なしと観測可能な構造制約を学習・復号へ明示すると、誤接続を抑えながら正しい継続と分裂を保持できる。 | [`explicit_no_match`](explicit_no_match.md)<br>[`known_parent_constraint`](known_parent_constraint.md)<br>[`graph_cost_scale`](graph_cost_scale.md)<br>[`image_count_prior`](image_count_prior.md) | - | 棄権の教師、既存softmaxとの差、費用校正と細胞数の誤差 |
| `HYP-20260910-12` | 等価な処理の再利用や局所的な計算配分により、高解像度・多時点・密な候補の手法を予算内で比較し最終精度を改善できる。 | [`exact_window_cache`](exact_window_cache.md)<br>[`sparse_motion_graph`](sparse_motion_graph.md)<br>[`uncertain_highres`](uncertain_highres.md)<br>[`distill_reinvest`](distill_reinvest.md)<br>[`frozen_image_encoder`](frozen_image_encoder.md) | - | 等価性、最悪時の費用、解禁した処理の最終精度 |
| `HYP-20260910-13` | 位置と系譜を保って撮像条件や境界・密度を変える学習により、別の胚の見え方に対する性能低下を抑えられる。 | [`anisotropic_blur`](anisotropic_blur.md)<br>[`photometric_shift`](photometric_shift.md)<br>[`lineage_density_aug`](lineage_density_aug.md)<br>[`crop_boundary_aug`](crop_boundary_aug.md) | - | 実際の胚差との対応、ラベル整合、片側胚の悪化 |
| `HYP-20260910-14` | 現行公式指標・胚を分けた評価・段階別の上限検査を用いると、独自proxyや学習内指標では見えない候補の順位差と失敗箇所を識別できる。 | [`graph_checkpoint`](graph_checkpoint.md)<br>[`group_error_readout`](group_error_readout.md)<br>[`oracle_stage_limits`](oracle_stage_limits.md) | [`exp003_official_metric_audit`](../experiments/exp003_official_metric_audit/)<br>[`exp004_embryo_holdout_baseline`](../experiments/exp004_embryo_holdout_baseline/)<br>[`exp005_embryo_holdout_batch8`](../experiments/exp005_embryo_holdout_batch8/) | 固定公式と公開実装の差はexp003で確認。exp004は固定batch size 16のfold 1 smokeがCUDA OOMで、基準予測は未成立。exp005のbatch size 8による再試行、モデル順位差、候補上限は未検証 |
| `HYP-20260911-01` | nnU-Netのデータに応じた前処理・構造・学習設定を中心マップの予測へ適応すると、疎注釈を適切に扱う条件で現行検出器より細胞を回収でき、両胚の公式接続・分裂指標が改善する。 | [`nnunet_center_detection`](nnunet_center_detection.md) | - | 中心教師とnnU-Net設定の寄与、背景と未知領域の識別、時間入力と接続特徴、学習・推論費用 |

### 未着手バックログ

2026-09-10の[調査](../docs/surveys/biohub-accuracy-hypotheses_20260910.md)から14仮説・64候補を登録した。[未決事項の先行調査](../docs/surveys/biohub-backlog-readiness_20260910.md)で当時の全64件を確認した。2026-09-11に共通方針と推奨順序を承認済み。公式評価照合をexp003へ、胚を分けた基準予測をexp004へ移した後の未着手63件へ、同日の依頼でnnU-Netの2候補を追加し、現在は65件。exp005は直接承認された派生実験であり、未着手候補を重複追加しない。exp005の実行証拠を得た後、依存が成立したP2候補から進める。追加注釈・教師・識別可能な目的を要するP4は条件が揃うまで保留する。

nnU-Netの[中心検出](nnunet_center_detection.md)は検出不足の診断と教師設計を先行条件とするP2、[3D領域からの検出](nnunet_instance_segmentation.md)は領域教師の準備待ちでP4とする。追加時に既存63件の優先度・依存を見直し、新しい実行証拠がないため既存の優先度と承認済み順序を維持した。2案とも未決事項を持つ`検討メモ・設計不可`として保存し、今回の反映依頼を実験化・Kaggle実行の承認とは扱わない。

| 優先度 | 対応仮説 | アイデア | 短い要約 | 主な先行条件 / 依存 | 状態 |
| --- | --- | --- | --- | --- | --- |
| P2 | `HYP-20260910-14` | [`group_error_readout`](group_error_readout.md) | 胚・画像条件ごとの誤りを測る | [exp005の胚を分けた基準予測](../experiments/exp005_embryo_holdout_batch8/)とexp003の固定公式評価器 | `検討メモ・設計不可` |
| P2 | `HYP-20260910-14` | [`oracle_stage_limits`](oracle_stage_limits.md) | 検出・接続・分裂の上限を分ける | [exp005の基準予測と候補cache](../experiments/exp005_embryo_holdout_batch8/)。誤差分析と併せ、検出閾値`0.99`と`0.965`の候補回収差も診断 | `検討メモ・設計不可` |
| P2 | `HYP-20260910-01` | [`sparse_det_mask`](sparse_det_mask.md) | 検出の未知領域を負例から外す | exp005の基準予測とoracle_stage_limitsで検出が主な制約と確認できること | `検討メモ・設計不可` |
| P2 | `HYP-20260910-01` | [`partial_edge_mask`](partial_edge_mask.md) | 未記録の第2娘を負例にしない | [exp005の基準予測](../experiments/exp005_embryo_holdout_batch8/)、[誤差分析](group_error_readout.md)、[段階別上限](oracle_stage_limits.md)。検出が主な制約ならsparse_det_maskを先行 | `検討メモ・設計不可` |
| P2 | `HYP-20260911-01` | [`nnunet_center_detection`](nnunet_center_detection.md) | nnU-Netで中心マップを学習して追跡へ渡す | exp005、oracle_stage_limits、sparse_det_maskの教師設計、時間入力と接続特徴・費用の確定 | `検討メモ・設計不可` |
| P2 | `HYP-20260910-14` | [`graph_checkpoint`](graph_checkpoint.md) | 最終graph指標で重みを選ぶ | [公式評価照合の実行結果](../experiments/exp003_official_metric_audit/) | `検討メモ・設計不可` |
| P2 | `HYP-20260910-02` | [`subvoxel_offset`](subvoxel_offset.md) | voxel未満の中心位置を補正 | oracle_stage_limits | `検討メモ・設計不可` |
| P2 | `HYP-20260910-03` | [`division_triplets`](division_triplets.md) | 母と2娘の組を採点する | oracle_stage_limits | `検討メモ・設計不可` |
| P2 | `HYP-20260910-06` | [`correlated_jitter`](correlated_jitter.md) | 時間相関のある位置誤差を学ぶ | group_error_readout | `検討メモ・設計不可` |
| P2 | `HYP-20260910-06` | [`missing_frame_noise`](missing_frame_noise.md) | 連続する見逃しから補正を学ぶ | group_error_readout | `検討メモ・設計不可` |
| P2 | `HYP-20260910-06` | [`real_error_finetune`](real_error_finetune.md) | 合成誤差の補正器を実予測へ調整 | correlated_jitter | `検討メモ・設計不可` |
| P2 | `HYP-20260910-09` | [`candidate_union`](candidate_union.md) | 異なる検出法の候補を統合 | oracle_stage_limits | `検討メモ・設計不可` |
| P2 | `HYP-20260910-11` | [`graph_cost_scale`](graph_cost_scale.md) | 接続と出現・分裂の費用を整合 | oracle_stage_limits | `検討メモ・設計不可` |
| P2 | `HYP-20260910-12` | [`exact_window_cache`](exact_window_cache.md) | 同一条件の特徴を再利用 | [公式評価照合の実行結果](../experiments/exp003_official_metric_audit/) | `検討メモ・設計不可` |
| P2 | `HYP-20260910-12` | [`sparse_motion_graph`](sparse_motion_graph.md) | 近傍候補で多時点予測を可能に | oracle_stage_limits | `検討メモ・設計不可` |
| P2 | `HYP-20260910-12` | [`uncertain_highres`](uncertain_highres.md) | 曖昧な領域だけ高解像度化 | group_error_readout | `検討メモ・設計不可` |
| P2 | `HYP-20260910-12` | [`frozen_image_encoder`](frozen_image_encoder.md) | 画像モデルを固定し接続を学習 | exact_window_cache | `検討メモ・設計不可` |
| P3 | `HYP-20260910-01` | [`sample_loss_balance`](sample_loss_balance.md) | 注釈密度に応じた損失集約 | [公式評価照合の実行結果](../experiments/exp003_official_metric_audit/) | `検討メモ・設計不可` |
| P3 | `HYP-20260910-02` | [`anisotropic_position`](anisotropic_position.md) | zとxyの位置不確実性を分ける | oracle_stage_limits | `検討メモ・設計不可` |
| P3 | `HYP-20260910-02` | [`joint_position_edges`](joint_position_edges.md) | 中心位置と接続を同時に選ぶ | candidate_union | `検討メモ・設計不可` |
| P3 | `HYP-20260910-02` | [`match_radius_point`](match_radius_point.md) | 7µm対応を意識して点を選ぶ | anisotropic_position | `検討メモ・設計不可` |
| P3 | `HYP-20260910-03` | [`division_context`](division_context.md) | 分裂前後の形状変化を使う | division_triplets | `検討メモ・設計不可` |
| P3 | `HYP-20260910-03` | [`division_time_dist`](division_time_dist.md) | 分裂時刻の複数候補を残す | division_triplets | `検討メモ・設計不可` |
| P3 | `HYP-20260910-03` | [`division_state_model`](division_state_model.md) | 継続・分裂・観測不能を選ぶ | division_triplets | `検討メモ・設計不可` |
| P3 | `HYP-20260910-03` | [`division_search_gate`](division_search_gate.md) | 画像変化で分裂探索箇所を選ぶ | oracle_stage_limits | `検討メモ・設計不可` |
| P3 | `HYP-20260910-04` | [`motion_reference`](motion_reference.md) | 組織の移動を差し引く | group_error_readout | `検討メモ・設計不可` |
| P3 | `HYP-20260910-04` | [`appearance_reference`](appearance_reference.md) | 同じ細胞の過去の像を参照 | group_error_readout | `検討メモ・設計不可` |
| P3 | `HYP-20260910-04` | [`relative_neighbors`](relative_neighbors.md) | 周囲の細胞との相対配置を使う | group_error_readout | `検討メモ・設計不可` |
| P3 | `HYP-20260910-04` | [`video_calibration`](video_calibration.md) | 動画内の見え方で調整する | group_error_readout | `検討メモ・設計不可` |
| P3 | `HYP-20260910-05` | [`focus_center_teacher`](focus_center_teacher.md) | FOCUSの中心を教師に使う | [公式評価照合の実行結果](../experiments/exp003_official_metric_audit/) | `検討メモ・設計不可` |
| P3 | `HYP-20260910-05` | [`teacher_agreement`](teacher_agreement.md) | 複数教師の一致で重み付け | focus_center_teacher | `検討メモ・設計不可` |
| P3 | `HYP-20260910-05` | [`track_teacher`](track_teacher.md) | 短い軌跡を対応学習の教師に | focus_center_teacher | `検討メモ・設計不可` |
| P3 | `HYP-20260910-05` | [`flow_feature_teacher`](flow_feature_teacher.md) | 移動場を特徴の補助教師に | [公式評価照合の実行結果](../experiments/exp003_official_metric_audit/) | `検討メモ・設計不可` |
| P3 | `HYP-20260910-05` | [`shape_aux_teacher`](shape_aux_teacher.md) | 領域の形状を補助的に学習 | focus_center_teacher | `検討メモ・設計不可` |
| P3 | `HYP-20260910-06` | [`duplicate_swap_noise`](duplicate_swap_noise.md) | 重複・取り違えから補正を学ぶ | group_error_readout | `検討メモ・設計不可` |
| P3 | `HYP-20260910-07` | [`deformation_pairs`](deformation_pairs.md) | 既知の3D変形で対応教師を作る | [公式評価照合の実行結果](../experiments/exp003_official_metric_audit/) | `検討メモ・設計不可` |
| P3 | `HYP-20260910-07` | [`synthetic_divisions`](synthetic_divisions.md) | ぼけ・雑音を含む分裂を合成 | division_triplets | `検討メモ・設計不可` |
| P3 | `HYP-20260910-07` | [`division_lookalikes`](division_lookalikes.md) | 分裂に似た通常事象を合成 | synthetic_divisions | `検討メモ・設計不可` |
| P3 | `HYP-20260910-07` | [`masked_video_pretrain`](masked_video_pretrain.md) | 隠した画像を予測して事前学習 | [公式評価照合の実行結果](../experiments/exp003_official_metric_audit/) | `検討メモ・設計不可` |
| P3 | `HYP-20260910-09` | [`position_mixture`](position_mixture.md) | 位置分布を点にする前に融合 | candidate_union | `検討メモ・設計不可` |
| P3 | `HYP-20260910-09` | [`graph_support_fusion`](graph_support_fusion.md) | 複数graphの局所支持を融合 | candidate_union | `検討メモ・設計不可` |
| P3 | `HYP-20260910-09` | [`image_condition_gate`](image_condition_gate.md) | 画像条件で情報源を選ぶ | candidate_union | `検討メモ・設計不可` |
| P3 | `HYP-20260910-09` | [`one_two_cells`](one_two_cells.md) | 1細胞か2細胞かを前後で選ぶ | candidate_union | `検討メモ・設計不可` |
| P3 | `HYP-20260910-09` | [`whole_graph_choice`](whole_graph_choice.md) | 候補graphを1つ選ぶ方法を比較 | candidate_union | `検討メモ・設計不可` |
| P3 | `HYP-20260910-10` | [`long_window_links`](long_window_links.md) | 複数時点の特徴で接続を予測 | sparse_motion_graph | `検討メモ・設計不可` |
| P3 | `HYP-20260910-10` | [`tracklet_join`](tracklet_join.md) | 短い軌跡の端同士を接続 | oracle_stage_limits | `検討メモ・設計不可` |
| P3 | `HYP-20260910-10` | [`latent_missing_nodes`](latent_missing_nodes.md) | 見逃した時点を画像から回収 | missing_frame_noise | `検討メモ・設計不可` |
| P3 | `HYP-20260910-10` | [`neighbor_dynamics`](neighbor_dynamics.md) | 周囲の動きを接続特徴へ追加 | relative_neighbors | `検討メモ・設計不可` |
| P3 | `HYP-20260910-11` | [`explicit_no_match`](explicit_no_match.md) | 対応なしを明示的な出力に | partial_edge_mask | `検討メモ・設計不可` |
| P3 | `HYP-20260910-12` | [`distill_reinvest`](distill_reinvest.md) | 複数予測を小モデルへ蒸留 | candidate_union | `検討メモ・設計不可` |
| P3 | `HYP-20260910-13` | [`anisotropic_blur`](anisotropic_blur.md) | zとxyのぼけ・雑音を変える | group_error_readout | `検討メモ・設計不可` |
| P3 | `HYP-20260910-13` | [`photometric_shift`](photometric_shift.md) | 輝度・背景変化に対応する | group_error_readout | `検討メモ・設計不可` |
| P3 | `HYP-20260910-13` | [`lineage_density_aug`](lineage_density_aug.md) | 系譜を保って密度を変える | group_error_readout | `検討メモ・設計不可` |
| P3 | `HYP-20260910-13` | [`crop_boundary_aug`](crop_boundary_aug.md) | 境界での出入りを学習する | group_error_readout | `検討メモ・設計不可` |
| P4 | `HYP-20260910-01` | [`dense_region_labels`](dense_region_labels.md) | 少数の完全注釈領域を追加する | sparse_det_mask | `検討メモ・設計不可` |
| P4 | `HYP-20260910-02` | [`image_label_centers`](image_label_centers.md) | 画像中心と注釈位置を分ける | oracle_stage_limits | `検討メモ・設計不可` |
| P4 | `HYP-20260910-05` | [`teacher_review_labels`](teacher_review_labels.md) | 教師が異なる領域を追加注釈 | teacher_agreement | `検討メモ・設計不可` |
| P4 | `HYP-20260910-08` | [`voxel_time_affinity`](voxel_time_affinity.md) | voxelの時空間関係から追跡 | [公式評価照合の実行結果](../experiments/exp003_official_metric_audit/) | `検討メモ・設計不可` |
| P4 | `HYP-20260910-08` | [`trajectory_set`](trajectory_set.md) | 画像から短い軌跡集合を出す | [公式評価照合の実行結果](../experiments/exp003_official_metric_audit/) | `検討メモ・設計不可` |
| P4 | `HYP-20260910-08` | [`inverse_cell_image`](inverse_cell_image.md) | 細胞像で画像を説明して追跡 | [公式評価照合の実行結果](../experiments/exp003_official_metric_audit/) | `検討メモ・設計不可` |
| P4 | `HYP-20260910-08` | [`slice_3d_instances`](slice_3d_instances.md) | 2D断面の領域を3Dへ統合 | focus_center_teacher | `検討メモ・設計不可` |
| P4 | `HYP-20260910-08` | [`direct_center_set`](direct_center_set.md) | 画像から可変個数の中心集合 | [公式評価照合の実行結果](../experiments/exp003_official_metric_audit/) | `検討メモ・設計不可` |
| P4 | `HYP-20260910-08` | [`nnunet_instance_segmentation`](nnunet_instance_segmentation.md) | nnU-Netの3D領域から細胞を分離して追跡 | 来歴と品質を確認した領域教師、分離・中心変換、胚別比較、費用の確定 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-11` | [`image_count_prior`](image_count_prior.md) | 画像からの細胞数を補助情報に | [公式評価照合の実行結果](../experiments/exp003_official_metric_audit/) | `検討メモ・設計不可` |
| P4 | `HYP-20260910-11` | [`known_parent_constraint`](known_parent_constraint.md) | 既存softmaxとの差を要確認 | 独立した追加目的の根拠 | `検討メモ・設計不可` |
