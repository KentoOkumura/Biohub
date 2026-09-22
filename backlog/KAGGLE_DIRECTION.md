# Kaggle 方針

## 参照する前提

- 機械可読なコンペ設定、データパス、validation、submission、Kaggle runtimeは[`project.yml`](../project.yml)を正とする。
- 公式情報は[`docs/01_competition.md`](../docs/01_competition.md)、評価指標は[`docs/02_metric.md`](../docs/02_metric.md)、評価分割とリーク確認は[`docs/03_validation.md`](../docs/03_validation.md)、データ仕様は[`docs/04_data.md`](../docs/04_data.md)を参照する。
- 保存場所・記録責務・ユーザー判断・用語の規則は[`AGENTS.md`](../AGENTS.md)、略語とリポジトリ内の管理用語は[`docs/glossary.md`](../docs/glossary.md)を正とする。
- このファイルを現在の学習方針・重点・比較基準・検証中の仮説・未着手候補の索引の正とする。個別候補の設計は各詳細ファイルに置く。

## 今後の学習方針

- **公開検出器を固定し、トラッカーを学習する。** 2026-09-12のユーザー依頼により、今後の提案・候補設計・新規実験の標準とする。検出器と画像特徴抽出器の重み・正規化の統計を固定し、接続・分裂を予測する下流部分を学習する。公開モデルが検出と追跡を同時学習した重みであっても、この段階で同時学習し直すことを必須にしない。
- **計算環境はKaggle Notebookのみ、GPUは週30時間以内、課金なし。** Colabや外部GPUを前提にしない。ただし2026-09-20にユーザーが承認した `exp032_three_frame_ten_epoch_training` の学習と共通窓診断だけはColabを使う。初回の検出・特徴抽出、トラッカー学習、検証、提出用推論を予算に含め、実行前の残量と小規模実測で実行数を決める。保存済み予測の集計・graphの再選択は可能ならCPU Notebookで行う。Notebook経過時間と実際のGPU割当消費は区別し、未実測の所要時間を保証しない。
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

1. [`exp013_public_notebook_replay`](../experiments/exp013_public_notebook_replay/)は、同一Kaggle T4環境の2回実行で公開test全件の候補座標・graph・241,282行の`submission.csv`が一致し、code submission ref `56199738`でPublic LB `0.944`を得た。ユーザー判断により、後続比較の基準として`usable`に確定した。
2. [`exp015_oracle_stage_limits`](../experiments/exp015_oracle_stage_limits/)はKaggleで完走し、199動画・19,701 windowのtrain feature cacheを4,151,337,848 bytesで生成した。array round-tripは完全一致し、cache・candidate graph・final graphはKaggle outputに保存したまま診断Notebookから直接参照できる。GPU予測区間は23,994.20秒、peak GPU memoryは688,422,400 bytesで、working outputはmanifest作成前4,813,320,625 bytesだった。
3. exp015では既知中心の候補対応99.286%に対し、既知edgeは両端対応98.843%、candidate graph 94.829%、final graph 91.602%だった。[`exp016_frozen_image_encoder`](../experiments/exp016_frozen_image_encoder/)で、公開sourceの教師と既存lossを保ち、固定画像特徴からprimary `SimpleNodeTransformer`だけを学習・公式評価する比較基準を確立した。[`exp019_partial_edge_mask`](../experiments/exp019_partial_edge_mask/)は正例子列だけを損失に含めて再学習したが、公式scoreはexp016の0.912055から0.908982へ低下し、ユーザー判断で不採用となった。固定候補・固定復号での比較基準はexp016を維持する。
4. 分裂組はexp015で既知151件中candidate graphに61件、final graphに15件。[exp020](../experiments/exp020_division_triplet_candidates/)の固定中心9/14 µmでは100件を回収したが、[exp021](../experiments/exp021_division_teacher_audit/)で同じ正例母の正誤組比較は最大2/100件だった。[exp023](../experiments/exp023_synthetic_detector_teacher_audit/result.md)は合成32時系列で固定検出後の正例1,035/2,266件、同母比較445母を確認して完了したが、実画像への学習効果は未測定。次は候補制限の契約を受け、合成教師の分割・mask・損失と共通復号を定めてから、合成事前学習の有無を同条件で比較する。候補制限・分裂学習・通常接続との局所再選択は分離する。[exp018](../experiments/exp018_graph_cost_scale/result.md)の費用倍率だけの変更は両胚の改善条件を満たさず不採用となった。
5. CV/LBの整合はまだ判定できない。自前routeではexp006の条件付きCV 0.5533789422、exp009のPublic LB 0.693、公開routeではexp013のPublic LB 0.944だが、同じ評価分割を共有していない。exp004のCUDA OOM、exp002の11時間gate、exp010の再実行CV 0.4981248955を失敗・不安定性の証拠として、全画像model再学習を再開せずcache再利用を優先する。

### 現行の比較基準

- 公式評価器は[exp003の照合結果](../experiments/exp003_official_metric_audit/result.md)を参照する。実験横断の最新結果は[`experiment_summary.md`](../experiment_summary.md)、提出履歴は[`SUBMISSIONS.md`](../SUBMISSIONS.md)、数値と実行状態は各実験の`metrics.json`を正とする。
- 自前学習の補助基準は[exp005の結果](../experiments/exp005_embryo_holdout_batch8/result.md)と[exp006の結果](../experiments/exp006_embryo_holdout_seed314159/result.md)。既存の予測・評価集計を再利用し、今回の方針変更のために検出器を再学習しない。Kaggle上の候補生成物とローカルの集計は所在を区別する。
- 新方針の公開重み・公開トラッカーはexp011で選定し、exp013で公開test全件の2回再実行一致、予測部分の実測時間、Public LB `0.944`を確認した。現行の自前提出ベスト`0.693`より`0.251`高いが、CVはなく、公開重みの学習来歴とleaderboard feedback利用を含むため独立validationとは扱わない。参照資料は[公開Notebook調査](../docs/surveys/biohub-public-baselines_20260910.md)、[候補の先行調査](../docs/surveys/biohub-backlog-readiness_20260910.md)、[exp013の結果](../experiments/exp013_public_notebook_replay/result.md)。

## アイデアバックログ

この節と`backlog/`の作成・更新・削除は`kaggle-strategy`が担当する。上位仮説は既存の系譜を維持し、候補の保留を仮説の棄却や実験の不採用とは扱わない。

### 検証中の仮説

| 仮説ID | 仮説 | 対応する未着手候補 | 対応する実験 | 残っている問い |
| --- | --- | --- | --- | --- |
| `HYP-20260909-01` | 主催者公開コードの `TemporalUNet3D` と `SimpleNodeTransformer` をrandom initializationから公式公開checkpointと同じ3 epochs学習し、そのcheckpointでhidden testを推論すれば、学習から再現可能な3D U-NetのPublic LB比較基準を確立できる。 | — | [`exp001_temporal_unet3d_baseline`](../experiments/exp001_temporal_unet3d_baseline/)<br>[`exp002_unet3d_expandable_segments`](../experiments/exp002_unet3d_expandable_segments/) | 胚を分けた主評価用CVとPublic LBの整合、診断用holdoutとPublic LBの関係 |
| `HYP-20260910-01` | 未注釈の中心・接続を未知として扱う教師と損失にすると、実在する細胞や第2娘を抑える学習が減り、公式指標が改善する。 | [`sparse_det_mask`](sparse_det_mask.md)<br>[`sample_loss_balance`](sample_loss_balance.md)<br>[`dense_region_labels`](dense_region_labels.md) | [`exp019_partial_edge_mask`](../experiments/exp019_partial_edge_mask/) | 負例の保証、過検出への退化、胚ごとの効果 |
| `HYP-20260910-02` | 中心位置を固定せず画像と接続の情報で補正・選択すると、位置誤差に由来する一対一対応の失敗と誤接続を減らせる。 | [`subvoxel_offset`](subvoxel_offset.md)<br>[`image_label_centers`](image_label_centers.md)<br>[`anisotropic_position`](anisotropic_position.md)<br>[`joint_position_edges`](joint_position_edges.md)<br>[`match_radius_point`](match_radius_point.md) | - | 位置誤差が接続を制限する度合い、分布の校正、近接細胞の混同 |
| `HYP-20260910-03` | 母細胞・2娘・前後画像を一つの分裂事象として扱えば、独立した接続や距離だけの判定より分裂と通常継続を区別できる。 | [`division_candidate_budget`](division_candidate_budget.md)<br>[`division_triplets`](division_triplets.md)<br>[`division_local_ilp`](division_local_ilp.md)<br>[`division_context`](division_context.md)<br>[`division_time_dist`](division_time_dist.md)<br>[`division_state_model`](division_state_model.md)<br>[`division_search_gate`](division_search_gate.md) | - | 同予算での組回収、合成教師の実画像への転移、局所再割当。exp023の教師接続を学習効果とは扱わない |
| `HYP-20260910-04` | 同じ動画の画像から運動・見た目・相対配置の基準を作ると、動画共通の基準だけでは誤る接続を改善できる。 | [`motion_reference`](motion_reference.md)<br>[`appearance_reference`](appearance_reference.md)<br>[`relative_neighbors`](relative_neighbors.md)<br>[`video_calibration`](video_calibration.md) | - | 予測参照の汚染、利用できる範囲、参照不足時の処理。appearance_referenceで履歴の追加と対照損失を別比較する |
| `HYP-20260910-05` | 密な検出・領域・移動・軌跡の不完全な予測を信頼度付き教師として使うと、疎い公式注釈だけの学習より細胞と接続を回収できる。 | [`focus_center_teacher`](focus_center_teacher.md)<br>[`teacher_agreement`](teacher_agreement.md)<br>[`track_teacher`](track_teacher.md)<br>[`teacher_review_labels`](teacher_review_labels.md)<br>[`flow_feature_teacher`](flow_feature_teacher.md)<br>[`shape_aux_teacher`](shape_aux_teacher.md) | - | 教師の来歴と誤り、fold分離、教師から生徒への改善移行。対応候補は現方針では保留し、詳細の再開条件を確認する。 |
| `HYP-20260910-06` | 推論で現れる中間候補の誤差を再現した学習と実予測への調整により、現行の検出候補学習より画像からの補正能力を高められる。 | [`correlated_jitter`](correlated_jitter.md)<br>[`missing_frame_noise`](missing_frame_noise.md)<br>[`duplicate_swap_noise`](duplicate_swap_noise.md)<br>[`real_error_finetune`](real_error_finetune.md)<br>[`contrastive_context_noise`](contrastive_context_noise.md) | - | 測れる誤差の範囲、時間相関、誤修正、実予測への移行 |
| `HYP-20260910-07` | 関係が既知の画像変形・合成分裂・未注釈画像の予測を使うと、少ない実注釈だけの学習より実画像の対応と分裂を学べる。 | [`synthetic_divisions`](synthetic_divisions.md)<br>[`division_lookalikes`](division_lookalikes.md)<br>[`masked_video_pretrain`](masked_video_pretrain.md) | [`exp008_deformation_pair_aux`](../experiments/exp008_deformation_pair_aux/) | 実画像への移行、合成の識別可能性、変形と系譜の整合 |
| `HYP-20260910-08` | 細胞の集合・領域・軌跡を画像から直接推定する表現なら、既存の極大値検出で失う細胞や接続を別の誤り方で回収できる。 | [`voxel_time_affinity`](voxel_time_affinity.md)<br>[`trajectory_set`](trajectory_set.md)<br>[`inverse_cell_image`](inverse_cell_image.md)<br>[`slice_3d_instances`](slice_3d_instances.md)<br>[`direct_center_set`](direct_center_set.md)<br>[`nnunet_instance_segmentation`](nnunet_instance_segmentation.md) | - | 部分注釈からの個数識別、領域教師の品質、接触細胞の分離と中心への変換、開発・推論費用。対応候補は現方針では保留し、詳細の再開条件を確認する。 |
| `HYP-20260910-09` | 異なる観測・表現・接続方法の候補と不確実性を残し画像から選ぶと、同系統の平均では直らない誤りを回収できる。 | [`candidate_union`](candidate_union.md)<br>[`position_mixture`](position_mixture.md)<br>[`graph_support_fusion`](graph_support_fusion.md)<br>[`image_condition_gate`](image_condition_gate.md)<br>[`one_two_cells`](one_two_cells.md)<br>[`whole_graph_choice`](whole_graph_choice.md)<br>[`contrastive_edge_candidates`](contrastive_edge_candidates.md) | - | 候補数を揃えた追加回収、正解なしの選別、相関した誤り |
| `HYP-20260910-10` | 多時点で接続誤りを減らせる。 | [`trackastra_association`](trackastra_association.md)<br>[`multi_time_past_candidate_attention`](multi_time_past_candidate_attention.md)<br>[`past_feature_cross_attention`](past_feature_cross_attention.md)<br>[`past_candidate_attention`](past_candidate_attention.md)<br>[`position_history_tracker`](position_history_tracker.md)<br>[`long_window_links`](long_window_links.md)<br>[`tracklet_join`](tracklet_join.md)<br>[`latent_missing_nodes`](latent_missing_nodes.md)<br>[`neighbor_dynamics`](neighbor_dynamics.md) | [`exp025_kalman_hungarian_links`](../experiments/exp025_kalman_hungarian_links/)<br>[`exp027_multi_frame_tracker`](../experiments/exp027_multi_frame_tracker/)<br>[`exp032_three_frame_ten_epoch_training`](../experiments/exp032_three_frame_ten_epoch_training/)<br>[`exp035_velocity_features`](../experiments/exp035_velocity_features/) | 座標・画像特徴・窓内対応の効果。 |
| `HYP-20260910-11` | 対応なしと観測可能な構造制約を学習・復号へ明示すると、誤接続を抑えながら正しい継続と分裂を保持できる。 | [`explicit_no_match`](explicit_no_match.md)<br>[`known_parent_constraint`](known_parent_constraint.md)<br>[`image_count_prior`](image_count_prior.md)<br>[`shared_edge_graph_learning`](shared_edge_graph_learning.md) | [`exp018_graph_cost_scale`](../experiments/exp018_graph_cost_scale/)<br>[`exp028_direct_graph_prediction`](../experiments/exp028_direct_graph_prediction/)<br>[`exp029_mother_daughter_set_selection`](../experiments/exp029_mother_daughter_set_selection/) | 棄権の教師、費用校正、母ごとの部分注釈損失と競合、接続得点共有とgraph比較による分裂学習の成立 |
| `HYP-20260910-12` | 等価な処理の再利用や局所的な計算配分により、高解像度・多時点・密な候補の手法を予算内で比較し最終精度を改善できる。 | [`sparse_motion_graph`](sparse_motion_graph.md)<br>[`uncertain_highres`](uncertain_highres.md)<br>[`distill_reinvest`](distill_reinvest.md) | [`exp011_public_detector_selection`](../experiments/exp011_public_detector_selection/)<br>[`exp013_public_notebook_replay`](../experiments/exp013_public_notebook_replay/)<br>[`exp014_exact_window_cache`](../experiments/exp014_exact_window_cache/)<br>[`exp016_frozen_image_encoder`](../experiments/exp016_frozen_image_encoder/) | exp013のPublic LBは`0.944`。exp015でtrain 199動画・19,701 windowの4.15GB cacheを生成した。exp016で固定特徴からprimary trackerだけを学習し、199動画の公式graph評価まで完了した。hidden test全件の費用と精度は未確認 |
| `HYP-20260910-13` | 位置と系譜を保って撮像条件や境界・密度を変える学習により、別の胚の見え方に対する性能低下を抑えられる。 | [`anisotropic_blur`](anisotropic_blur.md)<br>[`photometric_shift`](photometric_shift.md)<br>[`lineage_density_aug`](lineage_density_aug.md)<br>[`crop_boundary_aug`](crop_boundary_aug.md) | - | 実際の胚差との対応、ラベル整合、片側胚の悪化。対応候補は現方針では保留し、詳細の再開条件を確認する。 |
| `HYP-20260910-14` | 現行公式指標・胚を分けた評価・段階別の上限検査を用いると、独自proxyや学習内指標では見えない候補の順位差と失敗箇所を識別できる。 | [`contrastive_feature_audit`](contrastive_feature_audit.md) | [`exp003_official_metric_audit`](../experiments/exp003_official_metric_audit/)<br>[`exp004_embryo_holdout_baseline`](../experiments/exp004_embryo_holdout_baseline/)<br>[`exp005_embryo_holdout_batch8`](../experiments/exp005_embryo_holdout_batch8/)<br>[`exp007_graph_checkpoint_selection`](../experiments/exp007_graph_checkpoint_selection/)<br>[`exp012_group_error_readout`](../experiments/exp012_group_error_readout/)<br>[`exp015_oracle_stage_limits`](../experiments/exp015_oracle_stage_limits/) | 固定公開モデル下で対照学習の教師と特徴の識別力を監査し、候補順位の差を公式graph指標へ引き継げるか |
| `HYP-20260911-01` | nnU-Netのデータに応じた前処理・構造・学習設定を中心マップの予測へ適応すると、疎注釈を適切に扱う条件で現行検出器より細胞を回収でき、両胚の公式接続・分裂指標が改善する。 | [`nnunet_center_detection`](nnunet_center_detection.md) | - | 中心教師とnnU-Net設定の寄与、背景と未知領域の識別、時間入力と接続特徴、学習・推論費用。対応候補は現方針では保留し、詳細の再開条件を確認する。 |
| `HYP-20260915-01` | 固定画像特徴から学ぶtrackerへ、正しい親子対応を近づけ確定した誤親から離す対照損失を加えると、接続分類損失だけの場合より近傍の取り違えと両胚の公式graph誤りを減らせる。 | [`contrastive_parent_child`](contrastive_parent_child.md) | - | 共有tracker特徴への寄与、疎い教師と分裂の整合、固定候補内の順位から公式指標への移行。履歴・欠落への頑健性・候補回収は関連仮説で別比較する。 |
| `HYP-20260920-01` | 公開画像特徴を固定して既存のprimary trackerを追加学習する際、学習を3エポックより長くすると、同じ教師・損失・復号でも両胚の公式graph精度を改善できる。 | - | [`exp024_tracker_six_epochs`](../experiments/exp024_tracker_six_epochs/) | 6エポックで後半の重みが内部検証から選ばれるか、両胚の公式scoreへ改善が残るか。異なる学習率や長さでも成立するかは別検証。 |
| `HYP-20260920-02` | 固定した公開画像特徴から隣接時刻の接続を学ぶ際、各時刻のcell間を先にSelf-Attentionで文脈化してから既存のCross-Attentionを行うと、現行のCross-Attentionのみより親候補の取り違えが減り、両胚の公式graph指標が改善する。 | — | [`exp025_frame_self_attention`](../experiments/exp025_frame_self_attention/)<br>[`exp030_frame_self_attention_diagnostics`](../experiments/exp030_frame_self_attention_diagnostics/)<br>[`exp033_frame_self_attention_diagnostics`](../experiments/exp033_frame_self_attention_diagnostics/)<br>[`exp034_frame_self_attention_distance_bias`](../experiments/exp034_frame_self_attention_distance_bias/) | exp033では一貫したpair改善はなかった。exp034は物理距離の学習可能な負の二乗biasを1設定×2fold×3epochで完走したが、内部検証の改善が極小だったため全graph推論へ進まず完了した。公式graph改善は未検証。 |
| `HYP-20260920-03` | 固定候補の検出得点・DoG・HOGを個別入力すると、同構造対照より両胚の公式指標が改善する。 | [`detection_score_features`](detection_score_features.md)<br>[`dog_features`](dog_features.md)<br>[`hog_features`](hog_features.md) | - | 寄与・冗長性・入力・費用。 |

### 未着手バックログ

未着手候補・状態は下表を正とする。分類の根拠は[状態監査](../docs/surveys/biohub-backlog-status-audit_20260912.md)、実験化済みの候補は仮説表と各実験のrequirementsを参照する。先行成果物待ちは実行の依存とし、優先度と設計状態を分ける。

公開重みとトラッカー初期値は[`exp011_public_detector_selection`](../experiments/exp011_public_detector_selection/)で選定し、固定するSHA・特徴取得方法を採用済みである。[`exp013_public_notebook_replay`](../experiments/exp013_public_notebook_replay/)は公開test全件を同じKaggle T4環境で2回実行し、候補座標、graph topology、決定的なrun統計、raw `submission.csv`の一致とPublic LB `0.944`を確認した。

[exp014](../experiments/exp014_exact_window_cache/)で公開testのcache再生一致、[exp015](../experiments/exp015_oracle_stage_limits/)でtrain 199動画の特徴保存と段階別診断を確認した。数値は上の「現在の重点」と各metricsを参照する。候補形成まで変える比較は`division_triplets`に分離し、固定特徴trackerの基準と同時に変更しない。

`partial_edge_mask`から`sparse_det_mask`への依存を外し、同一検出器で複数トラッカーを比較できる融合案も`candidate_union`への依存を外した。検出・画像モデルの学習や未取得の領域教師が必要な原案は、代替処理へ読み替えずP4に保存する。


2026-09-21: `exp025_frame_self_attention` と `exp025_kalman_hungarian_links` は別実験であり、採番が重複する。実行証拠は元の正式名で参照する。


2026-09-20、`mother_daughter_set_selection`をP2に追加。集合損失と競合復号は未決。exp028の判断は保留。

2026-09-20、ユーザーが母ごとの娘集合logit、部分注釈の周辺化損失、娘の重複を防ぐ全体最適化を選び、同候補を[exp029](../experiments/exp029_mother_daughter_set_selection/)へ移行した。exp028の採否・完了は引き続き未判断。

| 優先度 | 対応仮説 | アイデア | 短い要約 | 主な先行条件 / 依存 | 状態 |
| --- | --- | --- | --- | --- | --- |
| P2 | `HYP-20260920-03` | [`detection_score_features`](detection_score_features.md) | 検出得点を入力 | exp015得点・exp016。 | `検討メモ・設計不可` |
| P2 | `HYP-20260920-03` | [`dog_features`](dog_features.md) | 固定点のDoG応答 | 元画像・画素間隔・exp015/016。 | `検討メモ・設計不可` |
| P2 | `HYP-20260910-10` | [`past_candidate_attention`](past_candidate_attention.md) | 接続候補ごとに全過去候補の3点座標特徴を集約 | exp015固定候補・exp016保存済み対照。13→32→32のMLPとattention、履歴を使わない選択肢。予測履歴生成は不要。全候補の時間・メモリを確認。 | `設計可能・実験化未承認` |
| P2 | `HYP-20260910-10` | [`position_history_tracker`](position_history_tracker.md) | 予測対応した位置列を入力して接続を学習 | exp015/016の分割外履歴。履歴生成・集約・信頼度・分裂後処理を確定する。 | `検討メモ・設計不可` |
| P2 | `HYP-20260910-10` | [`trackastra_association`](trackastra_association.md) | Trackastraの窓内対応学習 | exp015/016。教師・構造を確定。 | `検討メモ・設計不可` |
| P2 | `HYP-20260910-10` | [`past_feature_cross_attention`](past_feature_cross_attention.md) | 過去特徴へのcross-attention | exp015/016。特徴元窓・構造を確定。 | `検討メモ・設計不可` |
| P2 | `HYP-20260910-03` | [`division_candidate_budget`](division_candidate_budget.md) | 同じ組数で距離のみと画像・運動の候補を比較 | exp020/021、exp016得点、exp025の不採用結果。初段で候補契約を固定し、分裂学習後に同じ得点で実選別を確認する。 | `検討メモ・設計不可` |
| P2 | `HYP-20260910-03` | [`division_triplets`](division_triplets.md) | 同じ分裂組モデルで合成事前学習の有無を比較 | 完了したexp023の検出後教師（32時系列・正例1,035件）とdivision_candidate_budgetの初段契約。合成分割・sampling、実画像の未知組を除外する教師mask、損失・共通復号を確定する。 | `検討メモ・設計不可` |
| P2 | `HYP-20260910-11` | [`shared_edge_graph_learning`](shared_edge_graph_learning.md) | 接続得点を継続・分裂で共有しgraphを比較学習 | exp029追加診断・exp015/016。損失、対照条件、checkpoint規則を定め、小例適合を先行。 | `検討メモ・設計不可` |
| P2 | `HYP-20260910-12` | [`sparse_motion_graph`](sparse_motion_graph.md) | 近傍候補で多時点予測を可能に | exp015のGPU予測区間6時間39分54秒、peak GPU memory 688MB、candidate edge recall 94.829%。初回tracker学習で候補間計算が時間・メモリを支配すると実測された場合だけ、学習前に候補制限を診断する。 | `検討メモ・設計不可` |
| P2 | `HYP-20260910-14` | [`contrastive_feature_audit`](contrastive_feature_audit.md) | 対照学習の教師と固定特徴を診断 | exp015 cacheと固定tracker基準。確定負例・重複対応を監査し、距離のみと小型特徴モデルの親候補順位を比較する。 | `検討メモ・設計不可` |
| P2 | `HYP-20260915-01` | [`contrastive_parent_child`](contrastive_parent_child.md) | 正しい親子対応の特徴を対照学習 | contrastive_feature_audit、固定tracker基準、partial_edge_maskとの教師mask分離。既存接続損失を残し対照損失の有無だけを比較する。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-10` | [`multi_time_past_candidate_attention`](multi_time_past_candidate_attention.md) | 過去座標の複数時点集約 | 1時点版の証拠。時間差・集約を確定。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-02` | [`subvoxel_offset`](subvoxel_offset.md) | voxel未満の中心位置を補正 | [`exp015_oracle_stage_limits`](../experiments/exp015_oracle_stage_limits/)。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `設計可能・実験化未承認` |
| P3 | `HYP-20260910-06` | [`correlated_jitter`](correlated_jitter.md) | 時間相関のある位置誤差を学ぶ | [`exp012_group_error_readout`](../experiments/exp012_group_error_readout/)。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-06` | [`missing_frame_noise`](missing_frame_noise.md) | 連続する見逃しから補正を学ぶ | [`exp012_group_error_readout`](../experiments/exp012_group_error_readout/)。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-06` | [`real_error_finetune`](real_error_finetune.md) | 合成誤差の補正器を実予測へ調整 | correlated_jitter。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `設計可能・実験化未承認` |
| P3 | `HYP-20260910-12` | [`uncertain_highres`](uncertain_highres.md) | 曖昧な領域だけ高解像度化 | [`exp012_group_error_readout`](../experiments/exp012_group_error_readout/)。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-01` | [`sample_loss_balance`](sample_loss_balance.md) | トラッカー損失の動画間の寄与を揃える | 固定公開検出器のトラッカー学習基準と[`exp012_group_error_readout`](../experiments/exp012_group_error_readout/)、動画ごとの有効窓数と接続損失の寄与。 | `設計可能・実験化未承認` |
| P3 | `HYP-20260910-02` | [`anisotropic_position`](anisotropic_position.md) | zとxyの位置不確実性を分ける | [`exp015_oracle_stage_limits`](../experiments/exp015_oracle_stage_limits/)。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `設計可能・実験化未承認` |
| P3 | `HYP-20260910-02` | [`match_radius_point`](match_radius_point.md) | 7µm対応を意識して点を選ぶ | anisotropic_position。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `設計可能・実験化未承認` |
| P3 | `HYP-20260910-03` | [`division_local_ilp`](division_local_ilp.md) | 同じ得点で専用分裂判定と局所ILPを比較 | exp025の不採用結果、division_candidate_budget、division_triplets。通常割当を選び直せる局所範囲・境界・校正方式を確定する。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-03` | [`division_time_dist`](division_time_dist.md) | 分裂時刻の複数候補を残す | division_triplets。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-03` | [`division_state_model`](division_state_model.md) | 継続・分裂・観測不能を選ぶ | division_triplets。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-03` | [`division_search_gate`](division_search_gate.md) | 画像変化で分裂探索箇所を選ぶ | [`exp015_oracle_stage_limits`](../experiments/exp015_oracle_stage_limits/)。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-04` | [`motion_reference`](motion_reference.md) | 組織の移動を差し引く | [`exp012_group_error_readout`](../experiments/exp012_group_error_readout/)。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-04` | [`appearance_reference`](appearance_reference.md) | 予測履歴と次の細胞を対照的に照合 | 固定tracker基準とcontrastive_feature_audit。履歴の有無と対照損失の有無を分け、短い・不確かな履歴では既存特徴へ戻す。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-04` | [`relative_neighbors`](relative_neighbors.md) | 周囲の細胞との相対配置を使う | [`exp012_group_error_readout`](../experiments/exp012_group_error_readout/)。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-04` | [`video_calibration`](video_calibration.md) | 動画内の見え方で調整する | [`exp012_group_error_readout`](../experiments/exp012_group_error_readout/)。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-06` | [`duplicate_swap_noise`](duplicate_swap_noise.md) | 重複・取り違えから補正を学ぶ | [`exp012_group_error_readout`](../experiments/exp012_group_error_readout/)。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-07` | [`synthetic_divisions`](synthetic_divisions.md) | ぼけ・雑音を含む分裂を合成 | division_triplets。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-07` | [`division_lookalikes`](division_lookalikes.md) | 分裂に似た通常事象を合成 | synthetic_divisions。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-09` | [`graph_support_fusion`](graph_support_fusion.md) | 同一検出器の複数トラッカーの支持を融合 | 固定公開検出器で作る複数トラッカーの候補得点と誤りの差、[`exp015_oracle_stage_limits`](../experiments/exp015_oracle_stage_limits/)。candidate_unionは不要。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-09` | [`image_condition_gate`](image_condition_gate.md) | 画像条件で複数トラッカーの出力を重み付け | graph_support_fusionで用意する同一検出器由来の複数トラッカー出力と学習側の画像条件。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-09` | [`one_two_cells`](one_two_cells.md) | 1細胞か2細胞かを前後で選ぶ | [`exp015_oracle_stage_limits`](../experiments/exp015_oracle_stage_limits/)、同一の固定公開検出器の抑制前候補と前後の画像支持。candidate_unionは不要。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-09` | [`whole_graph_choice`](whole_graph_choice.md) | 同一検出器の候補graphを1つ選ぶ | graph_support_fusionと同じ固定候補・複数トラッカー出力、学習側の選択指標。 | `検討メモ・設計不可` |
| P3 | `HYP-20260920-03` | [`hog_features`](hog_features.md) | HOG特徴・類似度 | 元画像・exp015/016。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-10` | [`tracklet_join`](tracklet_join.md) | 短い軌跡の端同士を接続 | [`exp015_oracle_stage_limits`](../experiments/exp015_oracle_stage_limits/)。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-10` | [`latent_missing_nodes`](latent_missing_nodes.md) | 見逃した時点を画像から回収 | missing_frame_noise。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-10` | [`neighbor_dynamics`](neighbor_dynamics.md) | 周囲の動きを接続特徴へ追加 | relative_neighbors。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-11` | [`explicit_no_match`](explicit_no_match.md) | 対応なしを明示的な出力に | partial_edge_mask。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-06` | [`contrastive_context_noise`](contrastive_context_noise.md) | 候補欠落に強いtracker特徴を学ぶ | contrastive_parent_childと学習側の実誤差。合成文脈で学習後に実予測で調整する。検出点補正のreal_error_finetuneとは別案。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-09` | [`contrastive_edge_candidates`](contrastive_edge_candidates.md) | 特徴の類似度で接続候補を追加 | contrastive_parent_childの照合特徴。固定中心間の追加edge数を距離方式と揃え、候補回収と最終graphの選別を分ける。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-01` | [`sparse_det_mask`](sparse_det_mask.md) | 検出の未知領域を負例から外す | 検出損失と検出器の更新を必要とするため現方針では保留。検出器更新への方針変更が明示承認され、未知領域と背景の教師設計が成立した場合に再検討する。検出不足の診断だけでは再開しない。 | `検討メモ・設計不可` |
| P4 | `HYP-20260911-01` | [`nnunet_center_detection`](nnunet_center_detection.md) | nnU-Netで中心マップを学習して追跡へ渡す | nnU-Netによる検出器学習は現方針の対象外。検出器更新への方針変更が明示承認され、中心教師と学習費用の条件が成立した場合に再検討する。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-09` | [`candidate_union`](candidate_union.md) | 異なる検出法の候補を統合 | 原案の領域教師由来の中心が未成立。追加の固定公開情報源と取得条件・候補数を揃える対照・費用が確定するまで保留。単一検出器の閾値変更へ原案を置き換えない。 | `検討メモ・設計不可` |
| P4 | `HYP-20260910-02` | [`joint_position_edges`](joint_position_edges.md) | 中心位置と接続を同時に選ぶ | 原案の複数検出・位置候補を得る先行条件が未成立。candidate_unionの再開条件、または固定検出器由来の位置候補だけを使う比較の具体化後に再検討する。 | `検討メモ・設計不可` |
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
