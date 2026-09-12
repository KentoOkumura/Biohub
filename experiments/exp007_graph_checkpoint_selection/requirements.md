# exp007_graph_checkpoint_selection 要件と実装方法

この文書を、実装前の契約、実装方法、受け入れ条件の正とする。実行中の進捗は`SESSION_NOTES.md`へ記録し、この文書へ重複させない。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: backlog `graph_checkpoint`からの移行。2026-09-11に、epochごとの重みを保存し、学習側の内部dev動画に対する公式graph指標でcheckpointを選ぶ案を提示し、ユーザーが「それぞれ実験化してください。実装はまだです。」と実験化だけを承認した。
- 移行元backlog: `backlog/graph_checkpoint.md`。本書と`config.yaml`への移行確認後、元候補と未着手行を削除する。
- 対応する上位仮説: `HYP-20260910-14`。
- 上位仮説のうちこの実験が検証する範囲: exp005の内部checkpoint選択用動画で、現行のedge accuracyとnode recallの積による選択と、固定公式graph指標による選択が異なるepochを選ぶか、外側の胚holdoutで順位差が残るかを測る。
- この実験だけで上位仮説を判断できるか: いいえ。
- 上位仮説の判断に残る検証: `group_error_readout`、`oracle_stage_limits`、モデル間順位差、同じ外側結果を反復選択に使うことによる探索バイアス。
- 親実験: [`exp005_embryo_holdout_batch8`](../exp005_embryo_holdout_batch8/)。batch size 8で完走した分割、モデル、損失、3 epochs、推論、decode、固定公式評価を継承する。
- 根拠 / 一次資料 / 参照実装: 移行前の`backlog/graph_checkpoint.md`の内容を本書へ移した。[未決事項調査](../../docs/surveys/biohub-backlog-readiness_20260910.md#graph_checkpoint)、[exp005要件](../exp005_embryo_holdout_batch8/requirements.md)、[exp005設定](../exp005_embryo_holdout_batch8/config.yaml)、[exp005 metrics](../exp005_embryo_holdout_batch8/metrics.json)、[exp003で照合した公式評価](../exp003_official_metric_audit/result.md)を継続参照する。
- 固定するもの: exp005の固定source、outer fold、内部split、seed 42、モデル、教師、loss、batch size 8、3 epochs、optimizer、augmentation、推論閾値、整数線形計画の費用、公式評価を固定する。
- 変更するもの: 各foldの全3 epoch checkpointを保存する。同じ内部checkpoint選択用動画で現行proxyと公式graph指標を計算し、両selectorが選んだunique checkpointを外側評価胚へ適用する。
- 最小の反証可能な検証: 1回の2fold学習から各fold 3 checkpointを保存する。内部選択動画で2つのselectorを固定し、outer評価ではselectorごとの全体・胚別・動画別公式指標を比較する。両selectorが同じcheckpointを選ぶ場合は重複推論しない。
- 成功条件: 実行成功には、6 checkpoint、選択表、unique selected model、対応する外側199動画の予測と再採点可能な証拠が揃うことを要求する。公式graph選択を支持するには、outer公式指標が両胚で現行selectorより改善し、実行失敗が増えず、evaluation Notebookが12時間以内であることを要求する。
- 停止条件: outer評価胚をcheckpoint選択へ使う、全epoch重みの学習来歴を追跡できない、2 selectorで評価対象が異なる、保存graphの再採点が一致しない、またはtrain/evaluationの各Notebookが11.5時間gateを超える見込みなら停止する。
- 実行しないこと: outerラベルによるepoch選択、閾値・整数線形計画の費用調整、epoch数追加、seed探索、モデル・loss変更、単純なtrain accuracyでの代替、hidden test推論、competition submissionを行わない。
- 未決事項: なし。D1は上記の同一run内selector比較、D2は無料GPU枠かつ各Notebook 11.5時間gate、D3は両胚改善・失敗増加なし・推論12時間以内の推奨条件として承認済みである。2026-09-11にユーザーが「exp007を実装してください」と明示し、実装が承認された。Kaggle実行とcompetition submissionは今回の依頼に含めない。
- backlog記録から解釈を変更した箇所とユーザー承認: 候補作成時はexp002を構成参照先としていた。2026-09-11の実験案と実験化承認に基づき、同じモデルの胚holdoutをbatch size 8で完走したexp005を親・比較基準へ更新する。候補の仮説、内部選択、外側2方向評価は維持する。

## 判断履歴

- 2026-09-10: 現行学習はproxy最大の1 checkpointだけを上書きし、過去epochを復元できないことを確認してbacklogへ記録した。
- 2026-09-11: ユーザーの「すべて推奨でいいです」により、無料枠、両胚改善、実行失敗増加なし、推論12時間以内という共通方針が承認された。
- 2026-09-11: exp005がbatch size 8で2fold学習を完走したため、同構成で全epochを保存してselectorだけを比較する案を提示した。
- 2026-09-11: ユーザーが本候補の実験化を承認し、実装はまだ行わないよう指定した。
- 2026-09-11: ユーザーの「exp007を実装してください」により、trainと後段の評価Notebook、実験固有testの実装が承認された。Kaggle push、実行、competition submissionの承認とは扱わない。
- 2026-09-12: 後段はsubmission用の推論ではなく、学習で得たcheckpointの選択とvalidationであることを確認した。ユーザーの「それでは次に進んでください」により、Notebook種別を`evaluation`へ改め、保存済みcheckpointを入力にKaggleで実行することが承認された。

## 手法契約

- 依頼原文: 「GPUの使用時間は日本時間の明日9時にリセットされます」「3つ教えてください」「それぞれ実験化してください。実装はまだです。」「exp007を実装してください」
- 期待する成果: 各foldの全epoch checkpoint、内部選択動画に対する2種類の選択score、selectorごとの選択manifest、外側評価の公式指標を得る。
- input: outer学習胚のZarr画像とGEFF注釈、同じ学習胚から勾配更新に使わず分離した内部checkpoint選択用動画。
- target / objective: 学習objectiveはexp005と同じ。学習後、現行の`edge_accuracy_times_node_recall`と公式のcombined graph metricを別々のcheckpoint selectorとして比較する。
- output: fold×epochの6 checkpoint、内部selector score表、selectorごとの選択checkpoint、outer評価胚の予測graph、候補cache、公式評価summary。
- loss: exp005と同じ検出損失と接続損失。checkpoint選択自体に追加lossはない。
- decode: exp005と同じ検出時augmentation、中心抽出、接続softmax、閾値、整数線形計画を全checkpointへ適用する。
- context unit: 学習は2時刻の3D画像窓、checkpoint選択と外側評価は1動画のgraph、validation独立性は1胚。
- 実装区分: `faithful`。候補で定めた「全epochを保持し、最終graph指標で選択する」処理を省略せず実装対象とする。
- 省略する機構と理由: なし。両selectorが同じcheckpointを選んだ場合の重複推論だけを省く。
- proxyで検証できない主張: `N/A`。
- proxyの場合のユーザー承認: `N/A`。
- この実験が支持 / 棄却できる主張: 現行proxyと公式graph指標が異なるcheckpointを選ぶか、その順位差が外側2胚でも同方向に残るか。
- この実験では判断できない主張: 別model familyの順位、hidden testの順位、外側結果を見た追加selector探索の有効性。

## 実装方法

- アプローチ: 実装時にexp005のcompact self-contained train/inference Notebookを参照し、学習loopで各epoch checkpointを別名保存する。内部選択動画を各checkpointで推論し、exp003で固定した公式評価器へ渡す。
- inputの実装箇所と変換: exp005の`artifacts/splits.json`と同じouter・内部splitを再生成してSHAを照合する。内部選択動画は勾配更新へ含めない。
- target / objectiveの構築箇所: exp005の教師作成とlossを変更しない。selectorは学習後の評価処理として分離する。
- outputの生成箇所と表現: `artifacts/models/fold_<n>/epoch_<index>.pth`、`artifacts/checkpoint_selection.json`、selector別predictionとcandidate manifest、公式評価summaryを保存する。
- lossの実装箇所: exp005と同じ固定sourceを呼び、checkpoint保存処理だけを外から追加する。
- decode / postprocessの実装箇所: exp005 inferenceと同じdecodeを内部選択用動画とouter評価動画へ使う。2 selectorが選んだcheckpointをsample単位で取り違えないmanifestを必須にする。
- context unitを保つ処理箇所: outer分割は胚単位、内部選択は同じ学習胚の動画単位、decodeと評価は動画graph単位で行う。
- 変更するファイル / component: train/evaluationのJupytext sourceとNotebook、checkpoint manifest処理、実験固有test、設定、実装状態の記録を変更する。固定`official_source/`はexp005からbyte-identicalにコピーして変更しない。
- 固定事項を保つ確認方法: exp005 configとの差分をcheckpoint保存、selector、出力先、runtime計画だけに限定するtestを作る。
- 参照sourceとの一致を確認するテスト: exp005 source manifest SHA、モデル、loss、decode、公式評価器、外側と内部split件数を照合する。
- 承認済み差分を確認するテスト: foldごとにepoch index 0、1、2の3ファイルが必要であること、2 selectorが同じ内部動画を使うこと、outerラベルをselectorが読まないことを検査する。

## 探索幅とpivot判定

- 変更class: `selector-only`。モデル、loss、decodeを変えず、保存checkpointから選ぶ基準だけを比較する。
- 同じ親 / familyで連続した小改善実験数: 1。checkpoint selectorとしては最初の比較である。
- positiveなoracle headroom / coverage / 誤差非相関性: 未測定。本実験はselectorの順位差を測るが、候補上限は`oracle_stage_limits`へ残す。
- 比較したtarget、output、decode、context unitを変える案: 同時に実験化するexp008は対応特徴の補助targetと中間outputを追加する。
- 小改善の継続またはpivotを選ぶ根拠: exp005の内部選択scoreはfold 0がepoch 1と2で同率、fold 1はepoch 1が最大であり、最終graph指標で順位が変わる余地を1回だけ直接検証する。
- `kaggle-idea-forge`の実行要否と根拠: 追加実行不要。直前の次実験検討でrepresentation変更を含む案を生成し、ユーザーが本selector比較とexp008を併せて選んだ。

## 再現性・リスク

- 実行予定: active training variant 1、model/config 1、outer fold 2、booster 0、保存checkpoint 6。selectorは2種類、outer推論はselectorが選んだunique checkpointだけとする。
- control再学習: 別control runは行わない。同一学習runの保存checkpointへ2 selectorを適用する。exp005はruntimeと既存基準の副参照に使う。
- 追加GPUコスト: train 7〜9時間、outer評価用の予測生成は最大2 selector分で約7.2時間、保守係数込みでも各Notebook 11.5時間以内を条件とする。trainとevaluationは別Notebookである。
- seed policy: exp005と同じglobal seed 42、内部split seed 0を固定する。
- stochastic 処理の有無: random initialization、DataLoader shuffle、brightness/flip augmentation、CUDA kernelがある。
- stochastic feature generation / augmentation / seed bagging の有無: augmentationあり、seed baggingなし。
- 並列処理と乱数の関係: exp005と同じDataLoader generatorを使う。引数なしNumPy RNGの制約によりexp005とのbitwise比較ではなく、同一run内selector比較を主証拠にする。
- CPU/GPU runtime と deterministic flags: Kaggle T4 2基、DataParallel、AMP、internet無効。deterministic anchorとは扱わない。
- train cache / test feature regeneration の SHA 記録方針: checkpointごとの内部prediction graph SHA、outer candidate content SHA、selector manifest SHAを記録する。
- model manifest / prediction / submission SHA 記録方針: 6 checkpoint SHA、selectorごとのselected model、外側prediction SHAを記録する。submissionは作らない。
- Kaggle package bootstrap 確認方針: 実装後に正のNotebook、config、metrics、source manifestの一致をpush前validatorで確認する。
- リークリスク: outer評価胚の正解をselector、threshold、費用選択へ使わない。同じouter結果を見て第3selectorを追加しない。
- CV/LB 不一致リスク: outer胚評価はPublic LBではない。LB順位を推定しない。
- ランタイム/メモリリスク: 6 checkpointの保存容量と最大2倍の推論時間をsmokeで測り、11.5時間を超える場合はselectorや評価動画を縮小せず停止する。
- 再現性リスク: exp005とのrun間差は残るため、主比較は同一run内の2 selectorとする。
- 手法忠実性リスク: 簡略graph proxyへ置換すると仮説を検証できないため、exp003の固定公式評価器とSHAを使う。
- 過度な縮小 / proxy化リスク: 内部動画の一部だけでgraph指標を近似したり、outer片側だけを報告したりしない。

## 受け入れ基準

- [x] backlogの仮説、検証範囲、残る検証、根拠、差分、固定事項、実装方法、成功条件、停止条件、禁止事項、判断履歴を移行した。
- [x] 手法契約の `input / target / output / loss / decode / context unit` が実装前に一意である。
- [x] `config.yaml`の`HYP-20260910-14`と`graph_checkpoint`が本書と一致する。
- [x] train/evaluation Notebookと実験固有testを実装する。
- [x] `validate-exp`、`check-exp`、`test-exp`を実装後に通す。
- [ ] Kaggleで6 checkpoint、2 selector、外側評価予測とSHAを取得する。
- [ ] 実験の完了、採用、不採用は結果提示後にユーザーが判断する。
