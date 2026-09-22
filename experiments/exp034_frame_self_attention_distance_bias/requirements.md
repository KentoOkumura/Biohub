# exp034_frame_self_attention_distance_bias 要件と実装方法

この文書を実験契約、実装方法、受け入れ条件の正とする。実行履歴はSESSION_NOTES.mdへ記録する。

## 実験化の入口・引き継ぎ・承認

- 2026-09-22、ユーザーは「物理距離を使った学習可能な距離biasを1設定だけ、2fold×3epoch」で進める案を承認した。
- 同日、pair診断を含む初回実行が保守的runtime gateで停止した後、ユーザーは「普通に最終的な予測結果を評価すればいい」と方針を変更した。
- 親実験はexp025_frame_self_attention、上位仮説はHYP-20260920-02、移行元候補はframe_self_attention_spatial。
- 目的は、恒等初期化Model Bへ学習可能な物理距離biasを追加し、各holdout胚の最終グラフを公式指標で保存済み公開controlと直接比較すること。
- 未決事項はない。pair ranking、bucket診断、pair gateは採否条件に使わない。

## 判断履歴

- 2026-09-21: exp033のpair診断が完了し、Self-Attentionの一貫した改善は確認できなかった。
- 2026-09-22: ユーザーが物理距離biasの1設定、2fold×3epochを高リスク探索として承認した。
- 2026-09-22: 初回trainはpair診断を含む保守見積りが12時間を超え、full training前に停止した。
- 2026-09-22: ユーザーがpair診断を外し、最終予測結果を通常どおり評価する方針へ変更した。

## 手法契約

- input: exp015 cacheの固定32次元画像特徴、32次元位置特徴、original voxel z/y/x座標、候補mask。
- target / objective: 5 µm以内の貪欲一対一対応で作る隣接frame annotated edge。既存のfocal weighted BCEを最小化する。
- output: 候補pairのraw edge logit。最終的には固定threshold、融合、ILP、graph repairを通したgraph。
- context unit: 隣接2-frame。Self-Attentionは各frame内の全有効cellを共有重みで別々に処理する。
- loss: exp025と同じsource-axis softmax後のfocal weighted BCE、既存mask、division weight 1.0。
- decode: exp015と同じedge threshold、双方向融合、secondary融合、ILP、motion/gap/division repair。
- control: 保存済みexp015 final graph。controlを再学習しない。
- primary metric: adjusted edge Jaccard + 0.1 × division Jaccard。overall、胚別、sample別を報告する。
- 実装区分: 固定候補・固定decode内で距離biasだけを変更するmechanism比較。特定論文の忠実再現は主張しない。

## 実装方法

各Self-Attention layer/headに係数 a_h = softplus(raw_h) を持たせ、attention logitへ次を加える。

    -a_h × squared_physical_distance / fold_distance_scale²

- voxel z/y/xへ [1.625, 0.40625, 0.40625] µmを掛ける。
- fold 0のdistance scaleは8.125 µm、fold 1は8.285906791687012 µm。
- 係数初期値は1.0。
- attention output projectionとfeed-forward最終層をゼロ初期化し、学習開始時の公開tracker logitsを維持する。
- Pair MLPへ渡す既存座標は変更しない。
- 1構成×2fold×3epochで2 trackerを学習する。
- checkpointは各foldの内部検証 edge_accuracy × candidate_node_recall で選ぶ。
- outer holdoutのpair診断、bucket計算、学習前・学習後outer pair評価はtrain Notebookで実行しない。
- trainのruntime gateは学習と内部検証、および2窓のidentity確認だけを対象とする。
- train出力のmanifest SHA、model source SHA、checkpoint SHA、state SHAを検証してfold別trackerを復元する。
- fold 0を6bba、fold 1を44b6だけに使い、全199動画・19,701 windowを固定cacheからreplayする。
- 公開control候補graphがexp015保存値と完全一致することを確認する。
- candidate graphへ固定ILPとgraph repairを適用し、保存済みcontrolと同じ公式評価器で評価する。
- Kaggle competition submissionは作成しない。

## 探索幅とpivot判定

- 変更classはmechanism。距離bias以外のloss、特徴、候補、epoch、decodeを同時に変更しない。
- 探索は指定された1設定だけとし、hard近傍や相対位置ベクトルbiasを同じ実験へ追加しない。
- exp033のpair診断は方式選択の背景として使うが、改訂後の進行条件には使わない。
- 最終graph scoreが改善しない場合、この距離bias設定は追加探索せず終了する。
- ユーザーが方式と探索幅を直接指定したため、追加のidea-forgeは行わない。

## 再現性・リスク

- DataLoader順序はfold別generator、worker seedを固定する。augmentationとseed baggingは使わない。
- cache identity、annotation、feature schema、model source/file/state/manifest、最終graph、公式評価結果のSHAを記録する。
- 初期logit一致失敗、NaN/Inf、入力・checkpoint SHA不一致、public control candidate graph不一致、12時間runtime上限超過では該当stageを停止する。
- 学習とgraph inferenceは独立Notebookに分け、各stageを12時間以内に実行する。
- 公開画像encoderと初期trackerの学習来歴にtrain 199動画が含まれるため、独立CVとは呼ばない。
- 固定候補を使うため、距離biasだけで未検出cellは回収できない。
- この実験で判断できるのは指定した距離bias構成の最終graph改善であり、空間Self-Attention全般の有効性ではない。

## 受け入れ基準

- [ ] 距離bias式、物理単位、head別係数、padding除外、identity初期化、gradientのtestを通す。
- [ ] fold別distance scaleをcheckpointとmanifestからstrict復元できる。
- [x] train Notebookがpair診断なしで2fold×3epochを完走し、2 checkpointとmanifestを保存する。
- [ ] 全199動画の公開control candidate graphがexp015保存値と完全一致する。
- [ ] holdout最終graphを固定ILP・repair後に公式評価し、overall・胚別・sample別とcontrol差分を保存する。
- [ ] 数値とSHAをmetrics.json、result.md、SESSION_NOTES.mdへ記録する。
- [ ] submissionを作成しない。

完了時注記: ユーザー判断により、学習完走・checkpoint検証までで実験を終了した。最終graph評価に関する未チェック項目は未実行範囲として残す。
