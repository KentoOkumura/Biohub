---
title: Biohub バックログの統合・分割と優先度再評価
date: '2026-09-24'
types: [survey]
hypotheses: [HYP-20260910-01, HYP-20260910-02, HYP-20260910-03, HYP-20260910-04, HYP-20260910-06, HYP-20260910-07, HYP-20260910-09, HYP-20260910-10, HYP-20260910-11, HYP-20260910-12, HYP-20260910-14, HYP-20260915-01, HYP-20260920-03]
experiments: [exp013, exp016, exp023, exp025, exp028, exp029, exp032, exp035, exp036, exp037, exp038, exp039, exp040, exp041, exp043, exp044]
topics: [backlog, tracking, validation]
status: final
summary: 76候補を74候補へ整理し26件の優先度を変更。教師診断と座標補正の残課題を統合し、位置変更を局所graph選択から分離。tracker内部追加を保留し、作用段階の診断・接続候補保持・検出回収を優先。
---

# Biohub バックログの統合・分割と優先度再評価

## 結論

**Transformer内部への特徴・attention追加を先に増やす根拠は弱い。現行exp043のどの処理で正解接続を失うかを確認し、候補保持・検出回収・最終接続の選択を変える案を優先する。** 手法全体を棄却したり、未測定の公式score低下を断定したりする判断ではない。

未着手76件を74件へ整理し、26件の優先度を変更した。P1は3件、P2は3件へ絞った。既存の検出器・画像特徴抽出器を固定する方針、Kaggleのみ・週45 GPU時間以内・課金なし、実験化とsubmissionの承認条件を維持する。P4は再開条件付き保留であり、不採用の実験statusではない。

本レポートは整理時点の判断記録。現在の優先度・仮説の正本は[KAGGLE_DIRECTION](../../backlog/KAGGLE_DIRECTION.md)、契約と再開条件は各候補詳細とする。

## 対象と根拠

- 対応する上位仮説: `HYP-20260910-01`, `HYP-20260910-02`, `HYP-20260910-03`, `HYP-20260910-04`, `HYP-20260910-06`, `HYP-20260910-07`, `HYP-20260910-09`, `HYP-20260910-10`, `HYP-20260910-11`, `HYP-20260910-12`, `HYP-20260910-14`, `HYP-20260915-01`, `HYP-20260920-03`

依頼は「統合できるバックログがあれば統合」「分割すべきバックログがあれば分割」「最近の実験結果に基づく優先順位の見直し。特にtracker(transformer)に関するアイデアはスコアに紐づかなさそう」。全76候補の入力・出力・損失・比較・依存を照合した。外部の新しい手法探索や学習・推論は行っていない。

[実験サマリー](../../experiment_summary.md)、[提出履歴](../../SUBMISSIONS.md)、[直近のtracker再評価](biohub-exp043-tracker-reassessment_20260924.md)、exp040・041・043・044のresult・metrics・SESSION_NOTESを優先して読んだ。整理開始時のexp044は`metrics.json`がplannedで、結果文書は未記入だった。並行して実装が進む既存実験であり、成功・失敗の根拠に含めない。別作業で進む既存実験を本整理で停止・完了・不採用へ変更していない。

## 現在の比較基準と信頼度

| 比較する範囲 | 基準 | 解釈の限界 |
| --- | --- | --- |
| 採用済み提出全体 | [exp043](../../experiments/exp043_x138_self_trained_head/result.md) Public LB 0.950、ref 56508119 | 当リポジトリの提出済み最高値。旧exp013の0.944から+0.006だが、構成全体の差でありhead単独の効果ではない。作者の0.953の再現とは呼ばない |
| 旧構成の固定特徴tracker | [exp016](../../experiments/exp016_frozen_image_encoder/result.md)の同入力・同教師・同窓の単体／全graph記録 | 旧構成の診断基準。x138の補正座標・補間特徴に対して旧cache・数値を直接流用しない |
| x138で接続・分裂を変える案 | exp043そのものと、必要なら同入力の標準学習対照 | train対照は改善実験内で取得する。既に独立CVがあるとは扱わない |
| 座標補正そのもの | exp043の20動画・10,894対応点、1.82144→1.51618 µm、17/20動画改善 | 公開画像モデルを固定したheadの動画分割評価。接続・分裂の改善量や点別の再選択利益は未測定 |

交差検証（CV）とリーダーボード（LB）の一貫性は未判定。公開モデルと採用headの学習来歴を含む条件付き比較であり、trackerだけ胚を分けても全工程の独立性は得られない。[exp043 metrics](../../experiments/exp043_x138_self_trained_head/metrics.json)のCVは未記録。

## 優先度を下げる実験証拠

| 証拠 | 観測 | 今回の含意 |
| --- | --- | --- |
| [exp040修正版](../../experiments/exp040_trackastra_association/result.md) | 既知接続再現率は6bba 0.617053対対照0.969408、44b6 0.469260対0.947754。分裂親は17対48、2対6。数値不具合修正後も進行条件未達 | Trackastraの学習を増やすことを優先しない。公式graph score・LBは未測定、実験の採否は未判断 |
| [exp041](../../experiments/exp041_past_feature_cross_attention/result.md) | 過去特徴追加で既知接続+5/-96本、分裂48→46・6→3。時刻反転で差が各胚1本 | 過去情報が有効に使われている実証に乏しい。履歴延長・追加窓・誤履歴学習の前に、競合識別と情報の使用先を確認する |
| [exp032](../../experiments/exp032_three_frame_ten_epoch_training/result.md) | 10 epochs後も両胚の接続再現率で対照未満 | 時間文脈の追加に対して学習回数だけを増やす根拠はない |
| [exp035](../../experiments/exp035_velocity_features/result.md)・[exp025 Kalman](../../experiments/exp025_kalman_hungarian_links/result.md) | 自己速度の分裂親48→41・6→3。Kalmanは同条件対照より公式score低下 | 予測位置履歴を長くする案は保留。現行の近傍移動を対照に再接続へ直接作用させる案は別比較として残す |
| [exp036](../../experiments/exp036_detection_score_pair_features/result.md)・[exp039](../../experiments/exp039_dog_features/result.md) | 得点追加は既知接続−11/-8。局所画像特徴は+4/+2だが44b6で判定可能な誤り増加 | 通常の高得点検出点へさらに特徴を加える案の根拠は弱い。回収候補の選択は未検証の別用途 |
| [exp037](../../experiments/exp037_past_candidate_attention/result.md)・[exp038](../../experiments/exp038_past_candidate_knn_attention/result.md) | 学習前に費用条件で停止 | 精度悪化の証拠ではない。過去候補保持率の高さを精度改善に置き換えず、拡張は費用と効果の先行証拠を待つ |
| [exp029](../../experiments/exp029_mother_daughter_set_selection/result.md) | 正しい2娘がある128/128件で1娘・空集合より低得点 | 出力得点の共有・分裂損失の比較には具体的な根拠がある。ただしILP全面置換の成功は示さない |

exp043は整数線形計画法（ILP）の後に近傍移動を使って接続を組み直す。保存された学習確率も全候補ではなく、一定条件を通った接続に限られる。したがって**単体得点の改善が最終出力へ残るかは未確認**で、完全に無関係と断定するのも誤り。[コード確認と処理件数](biohub-exp043-tracker-reassessment_20260924.md#12-入力と最終出力の間にある処理)を根拠とする。

## 統合と分割

| 元の範囲 | 整理後 | 理由・保持したもの |
| --- | --- | --- |
| `contrastive_feature_audit` | [contrastive_parent_childの事前診断](../../backlog/contrastive_parent_child.md#統合した教師資格と固定特徴の診断)へ統合 | 教師の正負・未知・重複、距離／生特徴／小型モデルの3条件、娘同士を競わせない制約、停止条件、probeの未決事項を保存。診断を別実験にする依存を除いた |
| `subvoxel_offset` | [座標・接続診断](../../experiments/exp045_x138_coordinate_effect_audit/requirements.md#旧座標補正候補からの引き継ぎ)へ残課題を統合 | 固定特徴・3軸Huber損失・補正位置の補間はexp043で実装済み。補正有無だけの接続・分裂・近接点衝突・最終出力の比較を残す。直接承認exp043のlineage N/Aを過去の移行に書き換えない |
| `x138_local_graph_choice`に含まれた位置拡張 | [位置固定の局所graph選択](../../backlog/x138_local_graph_choice.md)と既存[joint_position_edges](../../backlog/joint_position_edges.md#x138の元位置と補正位置を使う比較の引き継ぎ)へ分割 | 前者は接続の選択、後者は位置と接続の共同選択。同一IDの位置排他、入出力edge間の位置共有、特徴再取得と再採点を後者へ移し、混在した初回比較を解消。後者はP4のまま |

別の比較を一つにするだけの統合は避けた。具体的には次を維持する。

- `x138_edge_candidates`は固定得点でedgeを残す。`contrastive_edge_candidates`は学習特徴で同数のedgeを探す。候補生成の情報源が違う。
- `x138_local_graph_choice`は同じtrackerの局所複数解、`whole_graph_choice`は複数trackerの完成graph、`final_graph_edit`は完成出力への修正操作。処理単位と対照が違う。
- `neighbor_dynamics`は候補の近傍移動、`motion_reference`は画像から推定する運動、`relative_neighbors`は相対配置。入力を同一視しない。
- `division_triplets`は既存の公開合成教師を使う学習比較、`synthetic_divisions`は独自生成器、`division_lookalikes`は対照的な通常事象の追加。exp023の既存教師を先に活用し、生成器は不足を確認するまでP4。
- `x138_history_error_training`は履歴誤差の学習、`contrastive_context_noise`は対照学習下の文脈欠落。`real_error_finetune`は位置補正器。教師・出力・損失の異なる案を名前だけで統合しない。
- 対照の作成・教師診断・学習・必要なgraph検証は一つの実験内の段階として扱える。実行手順を細かい独立バックログへ分割しない。

## 現在の順序

| 優先度 | 候補 | 進む理由 |
| --- | --- | --- |
| P1 | 座標・接続診断 | 補正と採点の効果が消える処理を特定。最初の改善実験内で共通対照を取得 |
| P1 | 接続候補保持 | 同じ得点と重みで候補を残す比較。新規tracker学習なしで候補損失を調べる |
| P1 | 検出回収の選択 | exp043に実在する回収処理を対象に、候補を通す接続／追加しない選択を学ぶ |
| P2 | 近傍移動による再接続 | 現行の近傍移動ルールを対照に、再接続へ直接得点を渡す。構造がGNNだから優先するわけではない |
| P2 | 共有接続得点で分裂を選ぶ | exp029の2娘採点失敗を対象に、通常接続と分裂の出力を変える |
| P2 | 固定位置で複数局所graphを選ぶ | 正解graphを早く捨てる場合に最終選択の単位を変える。選択器・部分教師のリスクは大きい |

手堅い比較は、共通診断を含めた固定得点の候補保持。大きい改善を狙う比較は回収候補の経路選択、または合法な複数局所graphの選択であり、通常のtracker特徴追加では測れない変更を含む。後者は教師・選択法の未決事項が残る。ここで実験の開始や方式を確定していない。

ローカル[コンペ概要](../01_competition.md)の最終提出予定は2026-09-29 23:59 UTCで、整理日から約5日。現時点の予定を新たに外部確認したものではない。週45時間は上限であり残量ではない。[exp043の実行履歴](../../experiments/exp043_x138_self_trained_head/SESSION_NOTES.md)の残量記録は過去のスナップショットなので、再開前に最新残量と小規模費用を確認する。保存済み集計・再選択は可能ならCPUで行う。未測定の学習の連続実行を前提にしない。

## 全候補の再評価記録

状態は優先度と独立に維持した。未実測だから設計不可へ戻したり、P4だから設計不可にしたりしない。削除した2件を除く全74件の理由を以下へ保存する。再開条件・未決事項の正本はリンク先。

| 候補 | 旧→新 | 理由 |
| --- | --- | --- |
| [`x138_coordinate_effect_audit`](../../experiments/exp045_x138_coordinate_effect_audit/requirements.md) | P1→P1 | 最初の改善実験に共通する診断。座標補正だけの効果と、採点から最終接続まで改善が残る段階を先に確認する。 |
| [`x138_edge_candidates`](../../experiments/exp047_x138_edge_candidates/requirements.md) | P1→P1 | 新しいtrackerを学習せず、固定得点の枝刈りで失う接続・両娘を保持する比較を先に扱う。後処理後の残存まで測る。 |
| [`x138_detection_recovery`](../../backlog/x138_detection_recovery.md) | P1→P1 | 現行exp043で実際に働く回収処理の採否・接続を変える。通常の高得点点への特徴追加より、出力へ作用する箇所と比較対象が明確。 |
| [`neighbor_dynamics`](../../backlog/neighbor_dynamics.md) | P1→P2 | P1の診断の後。再接続に直接使う得点を学ぶ点は残すが、exp035の自己速度とexp043既存の近傍移動ルールを上回る証拠はまだない。GNNという構造名だけで上位にしない。 |
| [`shared_edge_graph_learning`](../../experiments/exp050_shared_edge_graph_learning/requirements.md) | P2→P2 | exp029の正解2娘128/128件が1娘・空集合に負けた失敗に対し、出力得点の共有と分裂教師の寄与を変える案。通常trackerの特徴追加とは分けて残す。 |
| [`x138_local_graph_choice`](../../backlog/x138_local_graph_choice.md) | P3→P2 | 固定位置で合法な複数の接続graphを保持・選択する比較へ範囲を限定する。pairの得点改善が最終選択へ届かない場合に、出力の選び方を変える候補として残す。 |
| [`division_triplets`](../../backlog/division_triplets.md) | P2→P3 | exp023で合成教師は得られたが実画像への改善移行は未検証。exp026の候補契約と教師・校正の設計が残り、x138の出力段階を直接変える案の後に扱う。 |
| [`correlated_jitter`](../../backlog/correlated_jitter.md) | P3→P3 | 固定公開検出器のトラッカー学習基準と優先比較の結果を得た後、対応する誤りと追加費用を確認して扱う後続候補。 |
| [`missing_frame_noise`](../../backlog/missing_frame_noise.md) | P3→P3 | 固定公開検出器のトラッカー学習基準と優先比較の結果を得た後、対応する誤りと追加費用を確認して扱う後続候補。 |
| [`real_error_finetune`](../../backlog/real_error_finetune.md) | P3→P3 | 固定公開検出器のトラッカー学習基準と優先比較の結果を得た後、対応する誤りと追加費用を確認して扱う後続候補。 |
| [`uncertain_highres`](../../backlog/uncertain_highres.md) | P3→P3 | 固定公開検出器のトラッカー学習基準と優先比較の結果を得た後、対応する誤りと追加費用を確認して扱う後続候補。 |
| [`anisotropic_position`](../../backlog/anisotropic_position.md) | P3→P3 | 固定公開検出器のトラッカー学習基準と優先比較の結果を得た後、対応する誤りと追加費用を確認して扱う後続候補。 |
| [`match_radius_point`](../../backlog/match_radius_point.md) | P3→P3 | 固定公開検出器のトラッカー学習基準と優先比較の結果を得た後、対応する誤りと追加費用を確認して扱う後続候補。 |
| [`division_local_ilp`](../../backlog/division_local_ilp.md) | P3→P3 | 運動、候補、分裂学習の効果を先に切り分けた後の統合比較。未成立の分裂得点をあるものとして実装しない。 |
| [`video_calibration`](../../backlog/video_calibration.md) | P3→P3 | 固定公開検出器のトラッカー学習基準と優先比較の結果を得た後、対応する誤りと追加費用を確認して扱う後続候補。 |
| [`duplicate_swap_noise`](../../backlog/duplicate_swap_noise.md) | P3→P3 | 固定公開検出器のトラッカー学習基準と優先比較の結果を得た後、対応する誤りと追加費用を確認して扱う後続候補。 |
| [`one_two_cells`](../../backlog/one_two_cells.md) | P3→P3 | 固定公開検出器のトラッカー学習基準と優先比較の結果を得た後、対応する誤りと追加費用を確認して扱う後続候補。 |
| [`tracklet_join`](../../backlog/tracklet_join.md) | P3→P3 | 固定公開検出器のトラッカー学習基準と優先比較の結果を得た後、対応する誤りと追加費用を確認して扱う後続候補。 |
| [`latent_missing_nodes`](../../backlog/latent_missing_nodes.md) | P3→P3 | 固定公開検出器のトラッカー学習基準と優先比較の結果を得た後、対応する誤りと追加費用を確認して扱う後続候補。 |
| [`final_graph_edit`](../../backlog/final_graph_edit.md) | P3→P3 | 前段の改善が後処理で消える場合や、最後に修復可能な誤りが残る場合に比較する後続案。修正操作での回収上限と教師方式が未確定で、案①の成功・失敗を自動の採否条件にしない。 |
| [`x138_window_features`](../../backlog/x138_window_features.md) | P2→P4 | exp027/032/041で時間情報の追加が両胚の接続・分裂改善に結び付かず、前後窓特徴にも最終出力への追加価値が未確認。 |
| [`sparse_motion_graph`](../../backlog/sparse_motion_graph.md) | P2→P4 | 計算削減のための候補制限は選んだ学習案の費用対策。exp038のK近傍化後も費用条件未達で、多時点学習の精度利益も未確認。 |
| [`contrastive_parent_child`](../../backlog/contrastive_parent_child.md) | P2→P4 | 対照損失による表現変更の改善は未実証。教師・固定特徴の診断を本候補へ統合し、通常trackerの学習を先に増やさない。 |
| [`position_history_tracker`](../../backlog/position_history_tracker.md) | P2→P4 | exp025の等速履歴、exp035の自己速度、exp032の多時点学習に一貫した改善がない。位置列の長さを増やす学習を先行させない。 |
| [`multi_time_past_candidate_attention`](../../backlog/multi_time_past_candidate_attention.md) | P3→P4 | 1時点版の精度改善が未確認のまま参照時点を増やすと、学習費用と仮定だけが増える。exp037/038の学習前停止を精度失敗とは扱わない。 |
| [`sample_loss_balance`](../../backlog/sample_loss_balance.md) | P3→P4 | 動画間の損失集約だけを変える優先度を下げる。現行x138で動画ごとの寄与偏りが誤りの原因だという証拠がない。 |
| [`division_time_dist`](../../backlog/division_time_dist.md) | P3→P4 | 分裂時刻を広げる必要性と部分教師の時間範囲が未確認。現在の候補回収・2娘採点を先に診断する。 |
| [`division_state_model`](../../backlog/division_state_model.md) | P3→P4 | 継続・分裂・観測不能の状態を増やす前に、exp029で2娘が選ばれない問題と部分注釈の教師を整理する。 |
| [`division_search_gate`](../../backlog/division_search_gate.md) | P3→P4 | 分裂候補の追加場所を新しい画像規則で選ぶ前に、exp026の候補予算とx138での候補損失を確認する。 |
| [`motion_reference`](../../backlog/motion_reference.md) | P3→P4 | 画像からの運動推定と近傍細胞の移動中央値は別入力。現行exp043の近傍移動を超える画像運動の情報が未確認。 |
| [`appearance_reference`](../../backlog/appearance_reference.md) | P3→P4 | 過去特徴のexp041では履歴反転による回収差が各胚1本に留まり、参照履歴と対照損失を追加する根拠が弱い。 |
| [`relative_neighbors`](../../backlog/relative_neighbors.md) | P3→P4 | 相対配置を通常trackerへ足すだけの比較は、既存attention診断の混在した結果を上回る根拠がない。 |
| [`synthetic_divisions`](../../backlog/synthetic_divisions.md) | P3→P4 | 自作生成器と公開合成データは別情報源。exp022/023の公開合成教師が既にあるため、新しい生成器の追加はその不足が分かるまで保留。 |
| [`division_lookalikes`](../../backlog/division_lookalikes.md) | P3→P4 | 合成の紛らわしい通常事象の追加は、実分裂への移行が未確認の段階では後続に保留する。 |
| [`graph_support_fusion`](../../backlog/graph_support_fusion.md) | P3→P4 | 同じ検出点に対する複数trackerの誤りの補完性と最終graphの利益が未確認。弱いモデルを増やす前提を置かない。 |
| [`image_condition_gate`](../../backlog/image_condition_gate.md) | P3→P4 | 融合する複数trackerの補完性が未確認であり、その選択器学習を先行させない。 |
| [`whole_graph_choice`](../../backlog/whole_graph_choice.md) | P3→P4 | 複数trackerの完成graph選択は、同じtrackerの局所複数解を選ぶ案とは別。前提となる補完的な完成graph群が未確認。 |
| [`hog_features`](../../backlog/hog_features.md) | P3→P4 | exp039の局所画像特徴追加は既知接続計6本の利点に誤接続・誤分裂増を伴った。HOG自体は未試行だが追加特徴の根拠が不足。 |
| [`neural_graph_selection`](../../backlog/neural_graph_selection.md) | P3→P4 | ILPの全面置換はexp028/029の誤接続・2娘採点の問題を抱える。既存の最終選択を診断し局所的な変更を比べる案の後に保留。 |
| [`explicit_no_match`](../../backlog/explicit_no_match.md) | P3→P4 | exp028で対応なしを含む母選択を扱っており、同機能の再追加より教師と誤接続の交換条件を先に確認する。単独の効果は未検証。 |
| [`contrastive_context_noise`](../../backlog/contrastive_context_noise.md) | P3→P4 | 前提となる対照学習の有効性と候補欠落の学習への影響が未確認であり、複数の学習変更を先行させない。 |
| [`contrastive_edge_candidates`](../../backlog/contrastive_edge_candidates.md) | P3→P4 | 特徴で候補を追加する案は距離候補保持と区別するが、使用する対照学習特徴が未成立。 |
| [`x138_history_error_training`](../../backlog/x138_history_error_training.md) | P3→P4 | 履歴の有効性が未確認の段階で、合成誤履歴と微調整を先に増やさない。exp041の未達だけから誤履歴が原因とは断定できない。 |
| [`x138_two_step_links`](../../backlog/x138_two_step_links.md) | P4→P4 | GNN案内で近傍運動、母と2娘の後に保留。最近の多時点モデルで一貫した改善がなく、追加の教師・経路選択の設計も必要。旧案のP3から時間制約を理由に変更する。 |
| [`public_x138_tracker_comparison`](../../backlog/public_x138_tracker_comparison.md) | P4→P4 | 作者の追加座標補正重みは未取得で、忠実なx138再現を先行条件とする本候補は現在実行できない。[exp043](../../experiments/exp043_x138_self_trained_head/result.md)は自前学習した別のheadを使う採用・完了済み構成であり、今後のtracker学習評価の直接の対照には本候補を使わない。作者重みが取得でき、保存済みexp016重みの転用比較を改めて行う目的が生じた場合に再開する。 |
| [`sparse_det_mask`](../../backlog/sparse_det_mask.md) | P4→P4 | 検出損失と検出器の更新を必要とするため現方針では保留。検出器更新への方針変更が明示承認され、未知領域と背景の教師設計が成立した場合に再検討する。検出不足の診断だけでは再開しない。 |
| [`nnunet_center_detection`](../../backlog/nnunet_center_detection.md) | P4→P4 | nnU-Netによる検出器学習は現方針の対象外。検出器更新への方針変更が明示承認され、中心教師と学習費用の条件が成立した場合に再検討する。 |
| [`candidate_union`](../../backlog/candidate_union.md) | P4→P4 | 原案の領域教師由来の中心が未成立。追加の固定公開情報源と取得条件・候補数を揃える対照・費用が確定するまで保留。単一検出器の閾値変更へ原案を置き換えない。 |
| [`joint_position_edges`](../../backlog/joint_position_edges.md) | P4→P4 | P4を維持。x138の元位置・補正位置を使う範囲を引き継いだが、点単位の追加回収・整合した特徴再取得・位置選択の方式は未確定。 |
| [`division_context`](../../backlog/division_context.md) | P4→P4 | 5時点の画像特徴を得る方法が固定公開モデルで成立するか未確認。必要な窓の対応と費用を確認して再設計するまで保留。2時点特徴の寄せ集めを元の5時点画像特徴と同等とはみなさない。 |
| [`focus_center_teacher`](../../backlog/focus_center_teacher.md) | P4→P4 | FOCUSの教師で検出器を学習する原案は現方針の対象外。取得条件の本人確認、教師品質と来歴、検出器更新への明示承認が揃った場合に再検討する。 |
| [`teacher_agreement`](../../backlog/teacher_agreement.md) | P4→P4 | FOCUS等の複数教師と検出生徒の学習を要する。focus_center_teacherの再開条件と教師間一致の品質確認が揃った場合に再検討する。 |
| [`track_teacher`](../../backlog/track_teacher.md) | P4→P4 | 原案のTrackastra入力となる領域分割が未成立。条件を満たす固定領域予測と軌跡教師の来歴・品質が確定するまで保留。点検出だけを同じ入力の代用にはしない。 |
| [`flow_feature_teacher`](../../backlog/flow_feature_teacher.md) | P4→P4 | 画像特徴抽出器を補助損失で更新する原案は固定方針と両立しない。画像特徴の更新への方針変更が承認された場合に再検討する。 |
| [`shape_aux_teacher`](../../backlog/shape_aux_teacher.md) | P4→P4 | 領域教師による画像特徴の学習を要する。教師の取得・品質確認と画像特徴の更新への方針変更が成立するまで保留する。 |
| [`masked_video_pretrain`](../../backlog/masked_video_pretrain.md) | P4→P4 | 画像モデルの事前学習を要するため保留。画像特徴抽出器の更新への方針変更と計算予算が成立した場合に再検討する。 |
| [`position_mixture`](../../backlog/position_mixture.md) | P4→P4 | 候補統合に必要な複数の位置分布とその校正が未成立。candidate_unionの再開条件と、固定重みの下で追加位置分布を作る方法が揃った場合に再検討する。 |
| [`long_window_links`](../../backlog/long_window_links.md) | P4→P4 | 2時点と5時点の画像モデルを比較する原案は保留。選定した固定公開モデルが必要な時間窓を扱える証拠、または画像モデル変更への明示承認が必要。複数の2時点特徴の連結を元の5時点画像モデルと同じ実装とは扱わない。 |
| [`distill_reinvest`](../../backlog/distill_reinvest.md) | P4→P4 | 小さい画像モデルへの蒸留は新しい検出器の学習を要する。検出器更新への方針変更と複数教師・費用の成立まで保留する。 |
| [`anisotropic_blur`](../../backlog/anisotropic_blur.md) | P4→P4 | 検出・接続を画像変形で学習する原案は保留。検出器更新を認める方針変更、または画像特徴を固定したトラッカー専用の別設計と特徴再抽出費用が具体化した場合に再検討する。 |
| [`photometric_shift`](../../backlog/photometric_shift.md) | P4→P4 | 画像拡張による検出・接続の学習を前提とした原案は保留。検出器更新を認める方針変更、または固定画像モデルを使うトラッカー専用の別設計と再抽出費用が具体化した場合に再検討する。 |
| [`lineage_density_aug`](../../backlog/lineage_density_aug.md) | P4→P4 | 合成画像による検出・接続の学習を前提とした原案は保留。固定検出器を使うトラッカー専用の教師設計・実画像との差・再抽出費用を具体化するか、検出器更新への方針変更が成立した場合に再検討する。 |
| [`crop_boundary_aug`](../../backlog/crop_boundary_aug.md) | P4→P4 | 画像cropに伴う検出・接続の学習を前提とした原案は保留。固定画像モデルから境界の対応だけを学習する別設計と教師の未知状態を具体化するか、画像側更新への方針変更が成立した場合に再検討する。 |
| [`dense_region_labels`](../../backlog/dense_region_labels.md) | P4→P4 | 人手注釈を行わない既存方針により保留。追加注釈の明示承認と完全注釈領域の品質確認が必要。検出器も更新する場合は学習方針の変更承認を要する。 |
| [`image_label_centers`](../../backlog/image_label_centers.md) | P4→P4 | 画像中心と注釈位置を区別する独立した教師が未成立。追加教師の取得方法と固定検出器の下流だけを学習する設計が成立した場合に再検討する。 |
| [`teacher_review_labels`](../../backlog/teacher_review_labels.md) | P4→P4 | 人手の追加注釈と複数教師を要する。追加注釈の明示承認とteacher_agreementの先行条件が揃うまで保留する。 |
| [`voxel_time_affinity`](../../backlog/voxel_time_affinity.md) | P4→P4 | 画像から検出・領域・軌跡を推定する新規モデルの学習を要するため保留。画像側の学習への方針変更が明示承認され、原案の教師・計算費用が成立した場合に再検討する。 |
| [`trajectory_set`](../../backlog/trajectory_set.md) | P4→P4 | 画像から検出・領域・軌跡を推定する新規モデルの学習を要するため保留。画像側の学習への方針変更が明示承認され、原案の教師・計算費用が成立した場合に再検討する。 |
| [`inverse_cell_image`](../../backlog/inverse_cell_image.md) | P4→P4 | 画像から検出・領域・軌跡を推定する新規モデルの学習を要するため保留。画像側の学習への方針変更が明示承認され、原案の教師・計算費用が成立した場合に再検討する。 |
| [`slice_3d_instances`](../../backlog/slice_3d_instances.md) | P4→P4 | 画像から検出・領域・軌跡を推定する新規モデルの学習を要するため保留。画像側の学習への方針変更が明示承認され、原案の教師・計算費用が成立した場合に再検討する。 |
| [`direct_center_set`](../../backlog/direct_center_set.md) | P4→P4 | 画像から検出・領域・軌跡を推定する新規モデルの学習を要するため保留。画像側の学習への方針変更が明示承認され、原案の教師・計算費用が成立した場合に再検討する。 |
| [`nnunet_instance_segmentation`](../../backlog/nnunet_instance_segmentation.md) | P4→P4 | 新規領域モデルの学習と領域教師を必要とする。検出器更新への方針変更と、来歴・品質を確認した領域教師が揃うまで保留する。 |
| [`image_count_prior`](../../backlog/image_count_prior.md) | P4→P4 | 画像から細胞数を学ぶ追加モデルと個数教師を要する。固定特徴だけで学習できる具体案と教師の識別可能性、または画像モデル更新への方針変更が成立した場合に再検討する。 |
| [`known_parent_constraint`](../../backlog/known_parent_constraint.md) | P4→P4 | 既存softmaxと異なる学習目的をまだ特定できていない。追加する制約の非重複性と、固定検出器下での反証可能な比較が成立するまで保留する。 |

## 仮説と判断の扱い

この整理が直接扱った上位仮説は、部分教師`HYP-20260910-01`、位置`HYP-20260910-02`、分裂`HYP-20260910-03`、動画内参照`HYP-20260910-04`、誤入力学習`HYP-20260910-06`、合成教師`HYP-20260910-07`、候補選択`HYP-20260910-09`、多時点`HYP-20260910-10`、構造選択`HYP-20260910-11`、費用`HYP-20260910-12`、診断`HYP-20260910-14`、対照学習`HYP-20260915-01`、追加特徴`HYP-20260920-03`。いずれも支持・棄却・終了へ変更していない。

教師診断の主仮説は統合先では関連仮説として維持し、主仮説を二つ持たせない。実装済み座標処理に関連するexp043は直接承認の系譜を維持する。既存実験のmetrics、result、config、採否判断を本整理で変更しない。

## 検証

- 戦略文書の候補・優先度・状態・仮説・実験系譜の対応は`make check-strategy-docs`で検証済み。
- 調査索引を更新し`make validate-surveys`でメタデータと索引の一致を確認済み。
- 今回変更したMarkdownのローカルリンク、統合元への残存リンク、空白差分を確認済み。数式本文の変更やモデルコードの変更はない。
