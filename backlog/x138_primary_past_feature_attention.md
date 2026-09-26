# x138_primary_past_feature_attention

- 候補名: `x138_primary_past_feature_attention`
- 状態: `検討メモ・設計不可`
- 対応する上位仮説: `HYP-20260910-10`
- 関連する上位仮説: なし
- 作成日: 2026-09-26
- 最終更新日: 2026-09-26
- 依頼原文: 「primaryにも複座なattentionを組み込んだ方がよくないですか？」「過去候補の情報をprimary内部に入れる」への選択「こっちです」、直前frameの候補について座標だけでなく画像特徴も使う案への選択「後者がいいです」、続く「バックログに追加してください」。
- 期待する成果: 過去候補の画像特徴を接続logitの後付け補正ではなくprimary tracker内で現在の候補表現に反映すると、exp043の接続・分裂判断が改善するかを検証する。
- 親実験 / 比較対象: 採用済み[exp043](../experiments/exp043_x138_self_trained_head/)の入力・推論構成と公開primary checkpointを基準とする。[exp044](../experiments/exp044_x138_past_candidate_knn_attention/)の同じpair capture・公開primary対照を初段の比較に使う。exp044の実験完了・採否は未判断。
- 優先度: P4
- 優先度の理由: exp041の過去画像特徴attention、exp044の直前候補attentionは両胚の単体進行条件を満たしていない。新しい入力と学習分離には検証価値があるが、同じ失敗を繰り返す費用を避けるため再開条件付きとする。
- `backlog/KAGGLE_DIRECTION.md` の対応箇所: [検証中の仮説](KAGGLE_DIRECTION.md#検証中の仮説)、[未着手バックログ](KAGGLE_DIRECTION.md#未着手バックログ)

## 観測事実と根拠

- 実測済みの事実: exp044の1823 pair窓では、直前8候補の座標由来deltaとprimaryを同時学習すると既知edge回収が44b6で3162→3144/3302、6bbaで7016→6884/7275となった。追加CPU診断で再学習primaryにattentionを加えた判定変更は各胚1・2 pair、公開primaryから再学習primaryへの置換では72・265 pairだった。再学習による変化との整合はあるが、共同学習の因果分解や本案の効果は未検証。公式scoreは未計測。
- 根拠ファイル / 一次資料: [exp044 result](../experiments/exp044_x138_past_candidate_knn_attention/result.md)、[exp044 metrics](../experiments/exp044_x138_past_candidate_knn_attention/metrics.json)、[exp044 requirements](../experiments/exp044_x138_past_candidate_knn_attention/requirements.md)、[exp041 result](../experiments/exp041_past_feature_cross_attention/result.md)、[exp041 requirements](../experiments/exp041_past_feature_cross_attention/requirements.md)、[exp043 config](../experiments/exp043_x138_self_trained_head/config.yaml)。本案はこれらの結果を受けたユーザーとの設計案であり、改善実証ではない。
- 利用する保存済み生成物とSHA: exp044 `artifacts/kaggle_v1_diagnostic_inputs/`のpair captureとfold checkpointを候補とし、Kaggle receipt・source・checkpoint SHAは[exp044 metrics](../experiments/exp044_x138_past_candidate_knn_attention/metrics.json)を正とする。未追跡生成物の現存とSHAは実験化・実行前に再確認する。別窓で抽出した同一frameの特徴を黙って代用しない。
- 仮定: 直前frameの複数候補の見た目と相対位置をprimary内部の現在候補表現に反映できれば、座標だけのpost-logit deltaでは変わらなかった競合親順位を変えられる可能性がある。exp044の悪化原因がarchitectureだけだったとは仮定しない。

## この候補が直接検証する仮説と範囲

- 上位仮説のうちこの候補が検証する範囲: 直前1時点の未対応候補集合の固定画像特徴・座標を、exp043 primary trackerの接続得点生成より前に取り込む効果。
- この候補の具体的な仮説: 同じ検出候補・座標補正・固定画像特徴・教師・後処理の下で、直前候補へのattentionをprimary内部へ追加すると、公開primary対照および単なるprimary再学習対照より両胚の既知接続と分裂を改善し、判定可能な誤接続を増やさない。
- 仮説が正しい場合に期待する観測: 公開primaryとの初期logit parityを保ったうえで、過去特徴ありの学習後に正解親順位・既知edge回収が両胚で改善し、過去をmaskした同一重みとの差が入力利用を裏づける。最終graphへ進める場合も公式成分を確認する。
- 仮説を棄却する観測: 確定した初回構成で同一pair対照の進行条件に届かない、分裂や片方の胚が悪化する、入力整列または予算が成立しない。本候補の結果を多時点情報全般の棄却とはしない。
- この候補だけで上位仮説を判断できるか: いいえ
- 上位仮説の判断に残る検証: 他の過去時点数、予測履歴、候補生成、全graphでの作用と公式score。

## 入力・予測対象・出力・推論方法

- input: exp043の現在の隣接2-frame候補・補正座標・公開primary画像特徴と位置特徴に、直前frameの近傍候補の固定画像特徴・座標・候補ID・有効maskを追加する。初回の近傍8点はexp044との比較案であり、最終契約では未決。過去の正解対応や確定予測軌跡は使わない。
- target / objective: 現在の隣接2-frameで主催者GEFFに記録された既知接続・分裂。疎い注釈の未知部分を真の負例とはみなさない。
- output: 過去候補を参照して更新したprimary内部表現から、既存と同じ隣接frame候補間の接続logitを出す。exp044のように完成後の接続logitへ別branchのdeltaだけを足す構成にはしない。
- loss: exp044と同じ部分注釈mask・source軸softmax後のfocal BCEを初回比較案とする。primaryの凍結・再学習範囲や損失の厳密な固定は実装前に決める。
- decode / 推論方法: exp043の双方向採点、secondary融合、接続候補選択、ILP、後処理を固定する。直前窓由来の特徴を逐次供給し、動画先頭・欠測では過去なしmaskを使う。逆方向採点の過去特徴対応は未決。
- 処理単位: 同一動画の直前窓と現在の隣接pair。現在のsource候補ごとに直前候補集合を参照し、現在の接続候補を採点する。
- 実装区分: 既存手法の忠実再現ではなく、exp043 primaryへの明示的な構造変更。exp041の過去3時点・全候補attentionやTrackastraの窓内対応学習とは別の比較とする。

## 親実験からの差分

- 変更するもの: 直前候補の固定画像特徴・座標を参照するprimary内部attentionと、その学習対象。ゼロ初期化の残差出力で公開primaryを初期状態として再現する構成を提案するが、挿入位置は未決。
- 固定するもの: 公開検出器・画像encoder・座標補正head・secondaryの重み、候補生成、現在pairの特徴と教師対応、明示差分以外のexp043推論。学習量を変えた対照は別に記録する。
- 再利用するコード / config / 生成物: exp043のprimary構造と推論、exp044のpair capture・baseline parity・胚別評価とexp041の入力整列・ゼロ初期化テストを参照する。ただし旧exp016 cacheをexp043入力と同一視しない。
- 新しく作るもの: 実験化後に内部attention、履歴特徴整列、推論時の逐次履歴保持、同条件対照と実験固有テスト。今回作成するのは候補詳細と索引だけ。

## 最小の反証可能な検証

- 検証方法: exp044 captureで候補ID・座標・元窓・特徴SHAを照合し、先頭窓の過去なしmaskと初期logit parityを確認する。公開primary固定で追加層のみ学ぶ条件を最初の分離比較として提案し、公開primary、過去全mask、同じ入力でprimaryだけを再学習する対照を区別する。両胚の同一pair窓で親順位、閾値別回収、active-pair errors、分裂親と両娘回収を比較する。
- variant / config / fold / booster数: 初回は直前1時点・近傍8点・2胚の評価を比較案とする。学習epoch、fold別モデル数、primary凍結後の共同更新を含めるかは未決。boosterは使わない。
- control再学習: exp044の共同学習済みprimaryを「primaryのみ学習」の対照に流用しない。情報追加と再学習の効果を分けるには同条件の独立対照が必要。既存の公開primaryは固定対照として再利用する。
- 想定runtime / resource: Kaggle Notebookのみ、無料GPU週45時間以内。既存captureでのCPU整列診断を先行し、特徴供給・学習・推論を含む費用を小規模に測る。必要なcapture転送・Kaggle入力の実在は未確認。

## 成功条件と停止条件

- primary指標: 両胚別の既知edge recall、active-pair errors、既知分裂親・両娘回収。全graphへ進める場合の主評価は公式combined scoreと接続・分裂成分。
- 成功条件: 初段では同じpair窓の公開primaryと同条件再学習対照に対し、両胚の既知edge回収を改善し、active-pair errorsと分裂回収を悪化させないことを提案する。最終的な数値閾値・比較順序は実装前に固定し、pair指標を公式scoreの代用にしない。
- 必須guard: 初期logit parity、過去mask対照、候補IDと特徴元窓の一致、近傍候補保持率、閾値別recall、正解親順位、分裂の分母、教師mask、非有限値、動画端、胚別結果と費用。
- 成功時の次段階: 必要なpair条件と実行費用を提示し、全graph・公式評価へ進むかユーザーに確認する。採否・完了・submissionも別判断。
- 失敗時の停止範囲: 両胚の事前条件未達、入力不整合、予算超過なら全graphへ自動進行せず、公式score未計測として報告する。閾値・候補数・epochを外側結果で救済探索しない。

## 実行しないこと

- 禁止する代替実装、proxy、同一OOF上の救済探索: 座標だけのpost-logit delta、過去正解軌跡の入力、検出器・画像encoder・座標headの更新、exp016旧cacheとの無条件比較、外側胚での構造・閾値選択、無断の全graph推論・submission。
- 壁打ちで採らなかった案と理由: 今回のユーザー選択は過去候補の画像特徴をprimary内部で使う案。座標だけを使うexp044の延長やattention head数だけの増加では情報源と作用箇所が変わらない。exp041は過去3時点の全候補・旧入力で内部attentionを検証済みだが、本案はexp043入力・直前近傍候補・primary学習分離を直接比較するため別候補とする。[x138_window_features](x138_window_features.md)は前後窓特徴を広く扱い、こちらは直前候補集合からの内部参照に限定する。

## リスク

- leakage / validation: 公開画像モデルと座標headの学習来歴を明示し、外側胚評価を独立CVとは呼ばない。注釈の有無で過去候補を選ばない。
- hidden test: 同じframeでも画像特徴は入力2-frame窓に依存する。学習と推論で直前窓のsource特徴を同じ規則で使い、初回pairや欠測に対応する。
- runtime / memory: 候補数×近傍数のattentionに加え、前窓特徴を逐次保持する。pair captureのローカル存在だけでKaggle実行時間や入力可用性を保証しない。
- 再現性: 特徴元窓・候補ID・座標単位・モデル/source/capture SHA、初期化、凍結範囲、学習設定、checkpoint選択、maskと閾値を記録する。

## 調査・実行時に確認する事項

- exp044 capture 1980ファイルのうち先頭20件以外の直前窓との候補ID・座標・32次元画像特徴の対応はローカルで事前確認済み。ただし入力の現存・SHA・Kaggle側への供給方法は実験化時に再検証する。追加モデルの精度・費用・公式scoreは未測定。

## 未決事項

- primary内部の挿入位置、query/key/valueへの画像・位置特徴の与え方、残差の初期化と近傍8点の固定可否。
- 最初に公開primaryを凍結して追加層だけ学ぶか、その後にprimary共同更新を別条件として含めるか。教師mask・損失・更新量・checkpoint選択を同条件対照とどう揃えるか。
- 逆方向採点での過去窓の意味、Kaggleでの履歴特徴供給、pairから全graphへの数値進行条件。これらをユーザーと確定するまで実験化しない。

## 判断履歴

- 2026-09-26: ユーザーは過去候補の情報をprimary内部に入れる方針を選び、直前候補の座標だけでなく画像特徴も使う案を選択した。今回の依頼でバックログへ追加。実験化・実装・実行は未承認。

## 次セッションへの引き継ぎ確認

- 固定するものを一意に説明できる: exp043の固定検出器・encoder・座標headと明示差分以外の推論。
- 変更するものを一意に説明できる: 直前候補の画像特徴・座標をprimary内部で参照する。ただし挿入位置と学習範囲は未決。
- 最小検証と停止条件を一意に説明できる: 同一pairの公開・同条件再学習・過去mask対照を提案済み。数値条件は実装前に確定する。
- 実行しないことを一意に説明できる: post-logit deltaへの置換、画像側の更新、外側結果での救済探索、今回の実験化は含まない。
- 未決事項が明示されている: はい。次セッションは本詳細とexp043/041/044の契約・結果を読み、未決方式を確認してから実験化する。
