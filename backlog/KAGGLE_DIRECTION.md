# Kaggle 方針

## 参照する前提

- 機械可読なコンペ設定、データパス、validation、submission、Kaggle runtimeは[`project.yml`](../project.yml)を正とする。
- 公式情報は[`docs/01_competition.md`](../docs/01_competition.md)、評価指標は[`docs/02_metric.md`](../docs/02_metric.md)、評価分割とリーク確認は[`docs/03_validation.md`](../docs/03_validation.md)、データ仕様は[`docs/04_data.md`](../docs/04_data.md)を参照する。
- 保存場所・記録責務・ユーザー判断・用語の規則は[`AGENTS.md`](../AGENTS.md)、略語とリポジトリ内の管理用語は[`docs/glossary.md`](../docs/glossary.md)を正とする。
- このファイルを現在の学習方針・重点・比較基準・検証中の仮説・未着手候補の索引の正とする。個別候補の設計は各詳細ファイルに置く。

## 今後の学習方針

- **公開検出器を固定し、トラッカーを学習する。** 2026-09-12のユーザー依頼により、今後の提案・候補設計・新規実験の標準とする。検出器と画像特徴抽出器の重み・正規化の統計を固定し、接続・分裂を予測する下流部分を学習する。公開モデルが検出と追跡を同時学習した重みであっても、この段階で同時学習し直すことを必須にしない。
- **計算環境はKaggle Notebookのみ、GPUは週30時間以内、課金なし。** Colabや外部GPUを前提にしない。初回の検出・特徴抽出、トラッカー学習、検証、提出用推論を予算に含め、実行前の残量と小規模実測で実行数を決める。保存済み予測の集計・graphの再選択は可能ならCPU Notebookで行う。Notebook経過時間と実際のGPU割当消費は区別し、未実測の所要時間を保証しない。
- 最初に使用する公開Notebook・モデル版・checkpoint・ライセンス等の利用条件・学習来歴を特定する。**具体的な公開重みとトラッカー初期値は未選定**で、[`public_detector_selection`](public_detector_selection.md)で扱う。過去の自前重みを「公開検出器」と読み替えない。公開の既存トラッカーを基準とし、同じ固定検出器でトラッカーを学習した結果を比較する。後続案も同じ固定検出器下のトラッカー同士で比較し、自前検出器の再学習や全層更新の対照を必須にしない。
- 検出候補・物理座標・検出得点・必要な画像特徴を保存して再利用する。時間を扱う画像モデルの特徴は入力窓に依存するため、元の全時間窓・前処理・crop・変形・padding・精度・抽出座標・重みの版を対応付ける。初回は候補点特徴を中心に等価性と容量を測り、全voxel特徴の一括保存やframe単独の特徴への置換を前提にしない。画像・窓・候補座標等を変える比較は必要な特徴を再抽出し、費用に含める。
- トラッカーの入力は固定検出器の予測と画像特徴、正解の根拠は主催者のGEFFに記録された中心・接続・分裂とする。予測候補を既知注釈に対応付けて教師を作り、検出予測自体を正解とはみなさない。疎い注釈の未知部分を真の負例と断定せず、既存lossの教師maskとその変更を明示的に比較する。未承認の人手注釈や未取得の密な領域教師をあるものとして設計しない。
- 初回の学習比較では検出候補生成・座標・前処理・復号を対照と揃える。候補回収、中心補正、局所再推論は後続の別比較として明示し、公開検出器の重み更新を含めない。検出不足が判明しても検出器の学習を自動で再開せず、固定候補の上限と追加候補回収の条件を記録する。
- 評価は現行公式指標とその成分を使い、両胚別の改善・悪化、失敗と有効件数を示す。公開重みが評価胚を学習した、または来歴不明の場合は、固定公開モデル下の条件付きの比較と明記し、独立した交差検証（CV）とは呼ばない。トラッカーだけの分割では画像モデル由来の学習内評価は解消しない。既存の胚を分けた自前予測は補助診断として区別し、教師・設定の選択は学習側内部に限定する。
- 検出器の再学習、画像特徴抽出器の更新・事前学習、新規画像モデルの学習を要する原案はP4で保留し、再開条件を各詳細に記す。方針変更の明示承認なしに全層更新へ戻さない。固定特徴で別の比較に組み直す場合も、入力・学習対象・対照の変更を詳細へ明記する。

## 承認済みの進め方

- 2026-09-11の「すべて推奨でいいです」による無料枠・課金なし、両胚の改善、人手注釈を当面行わない方針は継続する。FOCUSの取得同意は含まれない。
- 2026-09-12の「公開検出器を固定し、トラッカーを学習する」前提への変更により、従来の「自前検出器を基準とし、外部重みは追加候補のみ」と、検出器学習を先に行う未着手候補の順序を更新した。同じ学習方針・予算の承認を再質問しない。
- 今回の依頼はバックログと方針文書の更新。新規実験の実装・Kaggle実行・submissionは開始しない。既存実験の契約・実行記録・採否はこの文書更新で変更せず、既承認実験を新方針の比較対象として使う場合も実際の学習来歴を示す。今後の追加・再実行案は本方針との整合を確認する。
- 精度と計算費用の実証を提示し、採否・完了はユーザーが判断する。提出推論12時間以内という既存条件を維持する。

## 現在の重点

1. [`public_detector_selection`](public_detector_selection.md)でベースとなる公開Notebook・重みの版と特徴取得方法を選定する。保存済み自前予測の誤差分析は並行して進められる。選定後に公開構成の基準推論から同じ診断を行う。
2. 固定検出候補と窓ごとの特徴の保存を小規模に検証し、同じ公開検出器でトラッカーを学習する基準を作る。候補数が費用を支配する場合は親候補の制限を先に診断する。
3. 基準成立後は`partial_edge_mask`で接続の教師maskを比較し、母と2娘の候補回収が十分なら`division_triplets`で分裂の組採点を学習する。`graph_cost_scale`は保存済み得点で行う軽い比較として併行検討できるが、トラッカー学習の代わりにはしない。
4. 観測した誤りに応じて近傍・運動・見た目の接続特徴、短い軌跡の接続、合成分裂へ進む。画像モデルの再学習はこの順序に含めない。新しいGPU実行数と時間配分は実測後に決める。

### 現行の比較基準

- 公式評価器は[exp003の照合結果](../experiments/exp003_official_metric_audit/result.md)を参照する。実験横断の最新結果は[`experiment_summary.md`](../experiment_summary.md)、提出履歴は[`SUBMISSIONS.md`](../SUBMISSIONS.md)、数値と実行状態は各実験の`metrics.json`を正とする。
- 自前学習の補助基準は[exp005の結果](../experiments/exp005_embryo_holdout_batch8/result.md)と[exp006の結果](../experiments/exp006_embryo_holdout_seed314159/result.md)。既存の予測・評価集計を再利用し、今回の方針変更のために検出器を再学習しない。Kaggle上の候補生成物とローカルの集計は所在を区別する。
- 新方針の公開重み・公開トラッカーによる基準は未選定・未測定。公開Notebookの報告スコアを、同じ対象・同じ評価分割での自前実験との優劣の証拠にはしない。参照資料は[公開Notebook調査](../docs/surveys/biohub-public-baselines_20260910.md)と[候補の先行調査](../docs/surveys/biohub-backlog-readiness_20260910.md)。

## アイデアバックログ

この節と`backlog/`の作成・更新・削除は`kaggle-strategy`が担当する。上位仮説は既存の系譜を維持し、候補の保留を仮説の棄却や実験の不採用とは扱わない。

### 検証中の仮説

| 仮説ID | 仮説 | 対応する未着手候補 | 対応する実験 | 残っている問い |
| --- | --- | --- | --- | --- |
| `HYP-20260909-01` | 主催者公開コードの `TemporalUNet3D` と `SimpleNodeTransformer` をrandom initializationから公式公開checkpointと同じ3 epochs学習し、そのcheckpointでhidden testを推論すれば、学習から再現可能な3D U-NetのPublic LB比較基準を確立できる。 | — | [`exp001_temporal_unet3d_baseline`](../experiments/exp001_temporal_unet3d_baseline/)<br>[`exp002_unet3d_expandable_segments`](../experiments/exp002_unet3d_expandable_segments/) | 胚を分けた主評価用CVとPublic LBの整合、診断用holdoutとPublic LBの関係 |
| `HYP-20260910-01` | 未注釈の中心・接続を未知として扱う教師と損失にすると、実在する細胞や第2娘を抑える学習が減り、公式指標が改善する。 | [`sparse_det_mask`](sparse_det_mask.md)<br>[`partial_edge_mask`](partial_edge_mask.md)<br>[`sample_loss_balance`](sample_loss_balance.md)<br>[`dense_region_labels`](dense_region_labels.md) | - | 負例の保証、過検出への退化、胚ごとの効果 |
| `HYP-20260910-02` | 中心位置を固定せず画像と接続の情報で補正・選択すると、位置誤差に由来する一対一対応の失敗と誤接続を減らせる。 | [`subvoxel_offset`](subvoxel_offset.md)<br>[`image_label_centers`](image_label_centers.md)<br>[`anisotropic_position`](anisotropic_position.md)<br>[`joint_position_edges`](joint_position_edges.md)<br>[`match_radius_point`](match_radius_point.md) | - | 位置誤差が接続を制限する度合い、分布の校正、近接細胞の混同 |
| `HYP-20260910-03` | 母細胞・2娘・前後画像を一つの分裂事象として扱えば、独立した接続や距離だけの判定より分裂と通常継続を区別できる。 | [`division_triplets`](division_triplets.md)<br>[`division_context`](division_context.md)<br>[`division_time_dist`](division_time_dist.md)<br>[`division_state_model`](division_state_model.md)<br>[`division_search_gate`](division_search_gate.md) | - | 3中心の同時回収、境界の未知状態、通常接続との両立 |
| `HYP-20260910-04` | 同じ動画の画像から運動・見た目・相対配置の基準を作ると、動画共通の基準だけでは誤る接続を改善できる。 | [`motion_reference`](motion_reference.md)<br>[`appearance_reference`](appearance_reference.md)<br>[`relative_neighbors`](relative_neighbors.md)<br>[`video_calibration`](video_calibration.md) | - | 予測参照の汚染、利用できる範囲、参照不足時の処理 |
| `HYP-20260910-05` | 密な検出・領域・移動・軌跡の不完全な予測を信頼度付き教師として使うと、疎い公式注釈だけの学習より細胞と接続を回収できる。 | [`focus_center_teacher`](focus_center_teacher.md)<br>[`teacher_agreement`](teacher_agreement.md)<br>[`track_teacher`](track_teacher.md)<br>[`teacher_review_labels`](teacher_review_labels.md)<br>[`flow_feature_teacher`](flow_feature_teacher.md)<br>[`shape_aux_teacher`](shape_aux_teacher.md) | - | 教師の来歴と誤り、fold分離、教師から生徒への改善移行。対応候補は現方針では保留し、詳細の再開条件を確認する。 |
| `HYP-20260910-06` | 推論で現れる中間候補の誤差を再現した学習と実予測への調整により、現行の検出候補学習より画像からの補正能力を高められる。 | [`correlated_jitter`](correlated_jitter.md)<br>[`missing_frame_noise`](missing_frame_noise.md)<br>[`duplicate_swap_noise`](duplicate_swap_noise.md)<br>[`real_error_finetune`](real_error_finetune.md) | - | 測れる誤差の範囲、時間相関、誤修正、実予測への移行 |
| `HYP-20260910-07` | 関係が既知の画像変形・合成分裂・未注釈画像の予測を使うと、少ない実注釈だけの学習より実画像の対応と分裂を学べる。 | [`synthetic_divisions`](synthetic_divisions.md)<br>[`division_lookalikes`](division_lookalikes.md)<br>[`masked_video_pretrain`](masked_video_pretrain.md) | [`exp008_deformation_pair_aux`](../experiments/exp008_deformation_pair_aux/) | 実画像への移行、合成の識別可能性、変形と系譜の整合 |
| `HYP-20260910-08` | 細胞の集合・領域・軌跡を画像から直接推定する表現なら、既存の極大値検出で失う細胞や接続を別の誤り方で回収できる。 | [`voxel_time_affinity`](voxel_time_affinity.md)<br>[`trajectory_set`](trajectory_set.md)<br>[`inverse_cell_image`](inverse_cell_image.md)<br>[`slice_3d_instances`](slice_3d_instances.md)<br>[`direct_center_set`](direct_center_set.md)<br>[`nnunet_instance_segmentation`](nnunet_instance_segmentation.md) | - | 部分注釈からの個数識別、領域教師の品質、接触細胞の分離と中心への変換、開発・推論費用。対応候補は現方針では保留し、詳細の再開条件を確認する。 |
| `HYP-20260910-09` | 異なる観測・表現・接続方法の候補と不確実性を残し画像から選ぶと、同系統の平均では直らない誤りを回収できる。 | [`candidate_union`](candidate_union.md)<br>[`position_mixture`](position_mixture.md)<br>[`graph_support_fusion`](graph_support_fusion.md)<br>[`image_condition_gate`](image_condition_gate.md)<br>[`one_two_cells`](one_two_cells.md)<br>[`whole_graph_choice`](whole_graph_choice.md) | - | 候補数を揃えた追加回収、正解なしの選別、相関した誤り |
| `HYP-20260910-10` | 複数時点の画像と軌跡を使って接続を選べば、短い観測だけで起きる取り違えや一時的な見逃しを修正できる。 | [`long_window_links`](long_window_links.md)<br>[`tracklet_join`](tracklet_join.md)<br>[`latent_missing_nodes`](latent_missing_nodes.md)<br>[`neighbor_dynamics`](neighbor_dynamics.md) | - | 長窓で増える情報、実欠測での回収、境界と隣接辺の整合 |
| `HYP-20260910-11` | 対応なしと観測可能な構造制約を学習・復号へ明示すると、誤接続を抑えながら正しい継続と分裂を保持できる。 | [`explicit_no_match`](explicit_no_match.md)<br>[`known_parent_constraint`](known_parent_constraint.md)<br>[`graph_cost_scale`](graph_cost_scale.md)<br>[`image_count_prior`](image_count_prior.md) | - | 棄権の教師、既存softmaxとの差、費用校正と細胞数の誤差 |
| `HYP-20260910-12` | 等価な処理の再利用や局所的な計算配分により、高解像度・多時点・密な候補の手法を予算内で比較し最終精度を改善できる。 | [`public_detector_selection`](public_detector_selection.md)<br>[`exact_window_cache`](exact_window_cache.md)<br>[`sparse_motion_graph`](sparse_motion_graph.md)<br>[`uncertain_highres`](uncertain_highres.md)<br>[`distill_reinvest`](distill_reinvest.md)<br>[`frozen_image_encoder`](frozen_image_encoder.md) | - | 公開重みと特徴取得の成立、等価性、最悪時の費用、解禁した処理の最終精度 |
| `HYP-20260910-13` | 位置と系譜を保って撮像条件や境界・密度を変える学習により、別の胚の見え方に対する性能低下を抑えられる。 | [`anisotropic_blur`](anisotropic_blur.md)<br>[`photometric_shift`](photometric_shift.md)<br>[`lineage_density_aug`](lineage_density_aug.md)<br>[`crop_boundary_aug`](crop_boundary_aug.md) | - | 実際の胚差との対応、ラベル整合、片側胚の悪化。対応候補は現方針では保留し、詳細の再開条件を確認する。 |
| `HYP-20260910-14` | 現行公式指標・胚を分けた評価・段階別の上限検査を用いると、独自proxyや学習内指標では見えない候補の順位差と失敗箇所を識別できる。 | [`group_error_readout`](group_error_readout.md)<br>[`oracle_stage_limits`](oracle_stage_limits.md) | [`exp003_official_metric_audit`](../experiments/exp003_official_metric_audit/)<br>[`exp004_embryo_holdout_baseline`](../experiments/exp004_embryo_holdout_baseline/)<br>[`exp005_embryo_holdout_batch8`](../experiments/exp005_embryo_holdout_batch8/)<br>[`exp007_graph_checkpoint_selection`](../experiments/exp007_graph_checkpoint_selection/) | 公式評価はexp003、胚別予測の最新証拠はexp005・exp006を参照。今後は固定公開重みの来歴と段階別の回収上限、公開モデル下での比較の独立性を確認する |
| `HYP-20260911-01` | nnU-Netのデータに応じた前処理・構造・学習設定を中心マップの予測へ適応すると、疎注釈を適切に扱う条件で現行検出器より細胞を回収でき、両胚の公式接続・分裂指標が改善する。 | [`nnunet_center_detection`](nnunet_center_detection.md) | - | 中心教師とnnU-Net設定の寄与、背景と未知領域の識別、時間入力と接続特徴、学習・推論費用。対応候補は現方針では保留し、詳細の再開条件を確認する。 |

### 未着手バックログ

2026-09-12に全63候補を公開検出器固定の方針で見直した後、公開モデルの選定を独立候補として追加し、現在は64件。P1は診断・特徴保存・トラッカー学習基準の準備、P2はその後の優先比較、P3は誤りと費用の証拠を要する後続、P4は再開条件が揃うまで保留とする。公開モデルの選定調査は対象・方法・成果・制約が定義済みのため設計可能とする。重みの版や来歴はその調査で確認する事項。具体的な重みや教師・対照が設計に必要な後続候補は、それぞれの未決事項を解消してから状態を更新する。

公開重み選定は[`public_detector_selection`](public_detector_selection.md)をP1先頭で扱い、具体的な版・特徴取得方法を後続へ渡す。既存トラッカーの基準推論は選定後の診断・学習比較の入力として扱う。診断・cacheの候補はトラッカー再学習の成功を待たない。`partial_edge_mask`から`sparse_det_mask`への依存を外し、同一検出器で複数トラッカーを比較できる融合案も`candidate_union`への依存を外した。検出・画像モデルの学習や未取得の領域教師が必要な原案は、代替処理へ読み替えずP4に保存する。

| 優先度 | 対応仮説 | アイデア | 短い要約 | 主な先行条件 / 依存 | 状態 |
| --- | --- | --- | --- | --- | --- |
| P1 | `HYP-20260910-12` | [`public_detector_selection`](public_detector_selection.md) | ベースの公開Notebook・検出重みを選定 | 保存済み公開調査が入口。公開コード・重みの版と来歴は本候補で確認する。特徴保存・トラッカー学習の完了には依存しない。 | `設計可能・実験化未承認` |
| P1 | `HYP-20260910-14` | [`group_error_readout`](group_error_readout.md) | 胚・画像条件ごとの誤りを測る | [exp005](../experiments/exp005_embryo_holdout_batch8/metrics.json)・[exp006](../experiments/exp006_embryo_holdout_seed314159/metrics.json)の保存済み結果とexp003の公式評価器。公開基準の診断は[`public_detector_selection`](public_detector_selection.md)後に行う。自前予測の集計は選定を待たずに進められる。 | `検討メモ・設計不可` |
| P1 | `HYP-20260910-14` | [`oracle_stage_limits`](oracle_stage_limits.md) | 検出・接続・分裂の上限を分ける | group_error_readoutと同じ対象一覧、固定重みの候補・得点・最終graph。公開基準の診断は[`public_detector_selection`](public_detector_selection.md)後の既存トラッカー推論を使い、再学習前に行う。 | `検討メモ・設計不可` |
| P1 | `HYP-20260910-12` | [`exact_window_cache`](exact_window_cache.md) | 固定公開モデルの候補点特徴を窓ごとに保存 | [`public_detector_selection`](public_detector_selection.md)で公開重み・元の時間窓・抽出層と来歴を特定する。未加工推論との等価性・容量・読み込み費用を本候補で測る。トラッカー再学習は先行条件にしない。 | `検討メモ・設計不可` |
| P1 | `HYP-20260910-12` | [`frozen_image_encoder`](frozen_image_encoder.md) | 公開検出器を固定し接続・分裂を学習 | [`public_detector_selection`](public_detector_selection.md)の選定結果、公開構成の基準予測、exact_window_cacheの等価性と費用、group_error_readoutとoracle_stage_limitsの初期診断。 | `検討メモ・設計不可` |
| P2 | `HYP-20260910-01` | [`partial_edge_mask`](partial_edge_mask.md) | 未記録の第2娘を負例にしない | 固定公開検出器のトラッカー学習基準と教師mask監査、group_error_readout・oracle_stage_limits。sparse_det_maskの実行は不要。 | `検討メモ・設計不可` |
| P2 | `HYP-20260910-03` | [`division_triplets`](division_triplets.md) | 母と2娘の組を採点する | 固定公開検出器のトラッカー学習基準とoracle_stage_limitsでの母・2娘の同時回収、既知分裂教師。 | `検討メモ・設計不可` |
| P2 | `HYP-20260910-12` | [`sparse_motion_graph`](sparse_motion_graph.md) | 近傍候補で多時点予測を可能に | 固定公開検出候補の親候補数・時間・メモリ実測とoracle_stage_limits。学習の費用超過が見えた場合は初回学習の前に候補制限を診断。 | `検討メモ・設計不可` |
| P2 | `HYP-20260910-11` | [`graph_cost_scale`](graph_cost_scale.md) | 接続と出現・分裂の費用を整合 | oracle_stage_limitsと固定公開検出器・トラッカーの学習側の候補得点。初回校正はトラッカー再学習の完了を待たずに可能。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-02` | [`subvoxel_offset`](subvoxel_offset.md) | voxel未満の中心位置を補正 | oracle_stage_limits。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-06` | [`correlated_jitter`](correlated_jitter.md) | 時間相関のある位置誤差を学ぶ | group_error_readout。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-06` | [`missing_frame_noise`](missing_frame_noise.md) | 連続する見逃しから補正を学ぶ | group_error_readout。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-06` | [`real_error_finetune`](real_error_finetune.md) | 合成誤差の補正器を実予測へ調整 | correlated_jitter。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-12` | [`uncertain_highres`](uncertain_highres.md) | 曖昧な領域だけ高解像度化 | group_error_readout。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-01` | [`sample_loss_balance`](sample_loss_balance.md) | トラッカー損失の動画間の寄与を揃える | 固定公開検出器のトラッカー学習基準とgroup_error_readout、動画ごとの有効窓数と接続損失の寄与。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-02` | [`anisotropic_position`](anisotropic_position.md) | zとxyの位置不確実性を分ける | oracle_stage_limits。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-02` | [`match_radius_point`](match_radius_point.md) | 7µm対応を意識して点を選ぶ | anisotropic_position。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-03` | [`division_time_dist`](division_time_dist.md) | 分裂時刻の複数候補を残す | division_triplets。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-03` | [`division_state_model`](division_state_model.md) | 継続・分裂・観測不能を選ぶ | division_triplets。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-03` | [`division_search_gate`](division_search_gate.md) | 画像変化で分裂探索箇所を選ぶ | oracle_stage_limits。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-04` | [`motion_reference`](motion_reference.md) | 組織の移動を差し引く | group_error_readout。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-04` | [`appearance_reference`](appearance_reference.md) | 同じ細胞の過去の像を参照 | group_error_readout。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-04` | [`relative_neighbors`](relative_neighbors.md) | 周囲の細胞との相対配置を使う | group_error_readout。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-04` | [`video_calibration`](video_calibration.md) | 動画内の見え方で調整する | group_error_readout。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-06` | [`duplicate_swap_noise`](duplicate_swap_noise.md) | 重複・取り違えから補正を学ぶ | group_error_readout。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-07` | [`synthetic_divisions`](synthetic_divisions.md) | ぼけ・雑音を含む分裂を合成 | division_triplets。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-07` | [`division_lookalikes`](division_lookalikes.md) | 分裂に似た通常事象を合成 | synthetic_divisions。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-09` | [`graph_support_fusion`](graph_support_fusion.md) | 同一検出器の複数トラッカーの支持を融合 | 固定公開検出器で作る複数トラッカーの候補得点と誤りの差、oracle_stage_limits。candidate_unionは不要。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-09` | [`image_condition_gate`](image_condition_gate.md) | 画像条件で複数トラッカーの出力を重み付け | graph_support_fusionで用意する同一検出器由来の複数トラッカー出力と学習側の画像条件。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-09` | [`one_two_cells`](one_two_cells.md) | 1細胞か2細胞かを前後で選ぶ | oracle_stage_limits、同一の固定公開検出器の抑制前候補と前後の画像支持。candidate_unionは不要。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-09` | [`whole_graph_choice`](whole_graph_choice.md) | 同一検出器の候補graphを1つ選ぶ | graph_support_fusionと同じ固定候補・複数トラッカー出力、学習側の選択指標。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-10` | [`tracklet_join`](tracklet_join.md) | 短い軌跡の端同士を接続 | oracle_stage_limits。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-10` | [`latent_missing_nodes`](latent_missing_nodes.md) | 見逃した時点を画像から回収 | missing_frame_noise。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-10` | [`neighbor_dynamics`](neighbor_dynamics.md) | 周囲の動きを接続特徴へ追加 | relative_neighbors。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
| P3 | `HYP-20260910-11` | [`explicit_no_match`](explicit_no_match.md) | 対応なしを明示的な出力に | partial_edge_mask。公開検出器固定のトラッカー学習基準を共通の先行条件とする。 | `検討メモ・設計不可` |
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
