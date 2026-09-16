# exp018_graph_cost_scale 要件と実装方法

この文書を、実装前の契約、実装方法、受け入れ条件の正とする。実行中の進捗は`SESSION_NOTES.md`へ記録し、この文書へ重複させない。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: backlog候補`graph_cost_scale`は、校正する費用、有限集合、選択分割が未決だった。2026-09-15にユーザーが、推奨した「edge costのscalarだけを有限比較し、胚を入れ替える2方向で選択と評価を分離する二段階案」で進めることを承認した。
- 移行元backlog: `backlog/graph_cost_scale.md`。
- 対応する上位仮説: `HYP-20260910-11`。
- 上位仮説のうちこの実験が検証する範囲: 固定したILP前候補graphの`edge_prob`と、出現・消失・分裂費用の相対スケールを変えるだけで、現行ILPが正しい接続と分裂をより多く残せるかを検証する。
- この実験だけで上位仮説を判断できるか: いいえ。
- 上位仮説の判断に残る検証: 棄権を明示する教師、既存softmaxとの差、既知親制約、画像由来の細胞数補助情報。同じ仮説の別候補`explicit_no_match`、`known_parent_constraint`、`image_count_prior`が残る。
- 親実験: [`exp015_oracle_stage_limits`](../exp015_oracle_stage_limits/)。固定公開構成の採用は[`exp011_public_detector_selection`](../exp011_public_detector_selection/)、Public LB 0.944の再現は[`exp013_public_notebook_replay`](../exp013_public_notebook_replay/)を参照する。
- 根拠 / 一次資料 / 参照実装: exp015の`requirements.md`・`result.md`・`metrics.json`とKaggle inference output、exp002の`official_source/scripts/predict_unet_transformer.py`にある公開ILP呼出し、support datasetの公式評価sourceを参照する。exp015では既知edge 128,883本のうち候補graphに122,218本、現行ILPとrepair後に118,059本が残った。既知division 151件のうち候補graphに母と2娘が揃った61件に対して、最終graphに残ったのは15件だった。
- 固定するもの: exp015の199個のILP前候補graph、node、edge候補、`edge_prob`、候補生成閾値、ILP制約とsolver、出現費用0、消失費用2、分裂費用1.2、公式評価source、7 µm matching、胚別sample集合。検出器、画像特徴、trackerを再実行または再学習しない。
- 変更するもの: 公開式`-1 * edge_prob`の1だけを`alpha`へ置き換え、`alpha`を0.25、0.5、1、2、4から選ぶ。`alpha=1`を同じNotebook内で再計算する内部対照にする。
- 最小の反証可能な検証: repairを実行しないILP出力を5つの`alpha`で作り、公式combined scoreを胚別に集約する。44b6だけで`alpha`を選んで6bbaで評価する方向と、6bbaだけで選んで44b6で評価する方向を実行する。外側胚の結果を選択へ使わない。
- 成功条件: 2方向とも選択された`alpha`が同じ方向の外側胚における`alpha=1`より公式combined scoreを厳密に改善し、division Jaccardを悪化させず、199件すべてが有効でNaNと失敗がないこと。
- 停止条件: いずれかの方向でcombined scoreが改善しない、division Jaccardが悪化する、sample集合・入力SHA・評価source SHAが一致しない、`edge_prob`が欠ける、NaN、solver/evaluator失敗、有効件数不一致、または12時間内の完走が見込めない場合は、後段repairへ進まない。
- 実行しないこと: 候補graph・edge score・event cost・ILP制約・評価器の変更、alpha grid外の救済探索、外側胚でのalpha選択、個別sampleごとのalpha選択、正解からの候補修正、モデル再学習、motion/gap/safe-division/DeepCenter repairの初段実行、Kaggle submission、Public LB取得。
- 未決事項: なし。初段結果がgateを通った後段だけは条件付きであり、同じ`exp018_graph_cost_scale`内に固定repair比較を追加する。別実験は作らず、結果をユーザーへ示してから実装する。
- backlog記録から解釈を変更した箇所とユーザー承認: 候補段階で未決だった費用はedge cost scalarだけ、有限集合は`[0.25, 0.5, 1, 2, 4]`、選択分割は2胚を入れ替える2方向とした。2026-09-15のユーザー回答「推奨案で進めてください」による。

## 判断履歴

- 2026-09-10: 調査I11の候補として追加し、edge scoreとevent costの対応を未決事項として残した。
- 2026-09-12: 公開検出器を固定し、下流trackerを学習する全体方針を反映した。本候補の初回は保存済み候補を使うCPU比較とした。
- 2026-09-13: exp015でILP前候補graphとedge scoreを保存する依存を追加した。
- 2026-09-14: exp015がtrain全199動画の候補graphと最終graphを保存し、candidateからfinalへのedge回収低下とdivision選択損失を確認した。
- 2026-09-15: ユーザーが、初段の両方向で公式combined scoreが改善した場合だけ後段repair比較へ進むこと、後段も同じ実験内で行う推奨案を承認した。
- 2026-09-16: 単一Notebookのversion 3で公式metric integration failureを確認し、version 4で修正後のmetric smokeを通した。version 4は20/199時点で約3時間を要し、単一Notebookでは12時間上限内の完走が見込めないため、alphaごとに独立したCPU Notebookへ分割し、全shard完了後に軽量aggregate Notebookで同じ両方向gateを判定する実行方式へ変更した。alpha、sample、solver、metric、選択規則は変更しない。

## 手法契約

実装区分は`docs/glossary.md`に定義したこのリポジトリ内の管理用ラベルを使う。

- 依頼原文: 「`graph_cost_scale`を実装してください」「推奨案で進めてください」。
- 期待する成果: 固定候補に対するedge costの単位だけを変えたとき、現行ILPより公式指標が改善する範囲と、固定repairまで進める価値があるかを評価できる実行可能なCPU Notebook群。
- input: exp015のKaggle inference outputにある199個の`oracle_candidate_graphs/*.geff`。各graphのnode、edge候補、`edge_prob`をそのまま使う。評価時だけcompetition trainのGEFFを読む。sampleと胚の対応はSHA固定したexp012のreadoutから読む。
- target / objective: 学習targetとlossはない。ILPの加法目的でedgeを選ぶ項だけを`-alpha * edge_prob`とし、固定event costとの相対スケールを比較する。
- output: 5つのalphaのILP解に対するsample別公式metric、胚・alpha別集約、2方向の選択alphaと外側胚差、gate判定、入力・source・solver・solution SHAを持つmanifest。初段graph自体は一時値として扱いNotebook outputへ995個保存しない。
- loss: なし。モデル学習を行わない。
- decode: `tracksdata.solvers.ILPSolver`へ、`edge_weight=-alpha * EdgeAttr("edge_prob")`、`appearance_weight=0`、`disappearance_weight=2`、`division_weight=1.2`を渡す。その他の引数と制約は公開呼出しと同じ既定値を使う。初段はsolver出力を直ちに公式評価し、repairを適用しない。
- context unit: decodeは1動画の候補graph全体。alpha選択と評価の独立性は胚単位。
- 実装区分: `staged-faithful`。公開ILPをそのまま再生してedge cost係数だけを変更する初段は実装するが、motion、gap、safe-division、DeepCenter repairはgate成立まで実行しない。
- 省略する機構と理由: 初段では固定repairを省略し、cost変更がILP選択へ与える効果を分離する。検出器・tracker・候補生成は保存済み入力を使い、再計算差を混ぜない。
- proxyで検証できない主張: N/A。proxyではないが、初段だけではrepair後の改善、Public LB、hidden test一般化を判断できない。
- proxyの場合のユーザー承認: N/A。
- この実験が支持 / 棄却できる主張: edge cost scalarだけで、胚を入れ替えた2方向のILP-only公式combined scoreを現行alpha=1より改善し、divisionを保持できるか。
- この実験では判断できない主張: repair後の改善、独立CV、Public LB改善、学習したtrackerとの相互作用、候補graphに存在しないedgeやdivisionの回収。

## 実装方法

- アプローチ: 5つのalpha shard Notebookが、それぞれexp015 manifestと199個のcandidate graph、exp012の胚対応、support datasetの公式metric sourceをSHA検査する。各shard内ではsampleを固定順序で逐次処理し、1つのalphaだけを同じsolverで実行する。5 shard完了後、aggregate Notebookが各manifestとtable SHA、入力receipt、候補graph bundle SHA、metric source SHA、199件coverageを照合し、995行を結合して両方向gateを判定する。
- inputの実装箇所と変換: `exp018_graph_cost_scale_diagnostic.py`のinput resolverとmanifest guardがKaggle kernel sourceを解決する。graph digestはnode ID・時刻・座標とedge ID・source・target・`edge_prob`から作る。`edge_prob`の値は変換せず、有限性と0から1の範囲を検査する。
- target / objectiveの構築箇所: 同Notebookの`solve_candidate_graph`が`alpha`を掛けたedge cost式と固定event costを構築する。
- outputの生成箇所と表現: 各alpha shardは`artifacts/alpha_<値>_v1/`へsample別metric、胚別集約、summary、manifestを保存する。aggregate Notebookだけが`artifacts/ilp_only_v1/`へ`per_sample_alpha_metrics.csv`、`embryo_alpha_summary.csv`、`cross_embryo_selection.csv`、`graph_cost_scale_summary.json`、`graph_cost_scale_manifest.json`を保存し、Notebook rootの`metrics.json`を更新する。
- lossの実装箇所: なし。
- decode / postprocessの実装箇所: `solve_candidate_graph`だけが初段decodeを担当する。graph repair関数を呼ばない。
- context unitを保つ処理箇所: candidate graphをsample単位で読み、sample集合と胚件数を実行前に検査する。`select_alpha`へ渡す行をcalibration胚だけに限定し、held-out胚は選択後の評価にだけ使う。
- 変更するファイル / component: exp018のconfig、requirements、alpha shard・aggregate・共通diagnosticのJupytext sourceとNotebook、実験固有tests、README、SESSION_NOTES、result、metrics。
- 固定事項を保つ確認方法: configとtestでalpha grid、baseline、event cost、候補・score・ILP制約の固定、CPU、submission禁止を固定する。runtimeではmanifestと公式source SHA、sample coverage、edge属性を検査する。
- 参照sourceとの一致を確認するテスト: `ILPSolver`呼出しが公開実装と同じ4項目を持ち、edge costだけが`-alpha * edge_prob`であることを静的に検査する。公式evaluatorとmetrics sourceのSHAはruntimeで検査する。
- 承認済み差分を確認するテスト: 合成summaryでcalibration胚だけからalphaを選ぶこと、score tieでalpha=1を優先すること、両方向の厳密改善とdivision非悪化をgateにすること、片方向だけの改善ではrepairを許可しないことを検査する。

## 探索幅とpivot判定

- 変更class: `parameter`。既存のedge cost係数だけを小さな有限集合で比較する。
- 同じ親 / familyで連続した小改善実験数: 0。exp015はheadroom診断であり、費用係数を比較していない。
- positiveなoracle headroom / coverage / 誤差非相関性: exp015でcandidate edge recall 94.829%に対してfinal selected edge recall 91.602%、candidate division triplet recall 40.397%に対してfinal division recall 9.934%だった。選択段階に改善余地はあるが、公式combined score改善を保証しない。
- 比較したtarget、output、decode、context unitを変える案: `explicit_no_match`はtarget/output、`division_triplets`は候補表現、`sparse_motion_graph`は候補と時間contextを変えるため、この実験へ混ぜない。
- 小改善の継続またはpivotを選ぶ根拠: 保存済み候補からCPUだけで短く反証できるため初段を実行する。両方向gateに失敗したらgridを拡張せず停止し、別mechanismを検討する。
- `kaggle-idea-forge` の実行要否と根拠: 不要。候補と比較設計はユーザー承認済みで、今回は新しい発想探索ではない。

## 再現性・リスク

- seed policy: 明示的なrandom samplingなし。各alpha shard内ではsampleをsort済み固定順序で逐次処理する。
- stochastic 処理の有無: SCIP solverの同値解におけるtie resolutionはpackage実装に依存し得る。
- stochastic feature generation / augmentation / seed bagging の有無: すべてなし。
- 並列処理と乱数の関係: 1つのshard内は並列化しない。alpha間は独立Kaggle kernelとして同時実行できるため、単一processでsample外側・alpha内側に逐次実行する順序とは異なる。global RNGは使わないが、SCIPの同値解tie resolutionがprocess履歴へ依存する可能性はsolution topology SHAで可視化する。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU無効、internet無効、12時間上限。solverのbitwise deterministicは主張しない。
- train cache / test feature regeneration の SHA 記録方針: exp015 manifest file/payload/input SHAと、199 candidate graphのcanonical content SHAを入力証拠にする。feature再生成は行わない。
- model manifest / prediction / submission SHA 記録方針: modelは0。各sample・alphaのsolution topology SHAと、全行を束ねたSHAを保存する。submissionは生成しない。
- Kaggle package bootstrap 確認方針: exp015で有効だったoffline wheel bootstrapを再利用し、`tracksdata`、`ilpy`、`pyscipopt`、GEFF、Polarsをimport検査する。support repoのevaluator・metrics・division source SHAを処理前に検査する。
- リークリスク: 公開モデルはtrain 199動画を含む可能性があるため独立CVではない。追加で、各方向の外側胚GTをalpha選択へ使わない。2方向の結果を見てgridやtie-breakを変更しない。
- CV/LB 不一致リスク: train上の条件付き比較でありPublic LBを推定しない。初段はrepairなしなので、現行最終pipelineとの差ではなく同じILP-only alpha=1との差だけを解釈する。
- ランタイム/メモリリスク: 公式division metricはGT divisionごとのmatchingを行うため、単一Notebookの実測では12時間超過が見込まれた。199 graphを1 alphaずつ扱う5 CPU shardへ分割し、各shardを12時間内に収める。graphを全件保持せずsample単位に解放し、995解を保存しない。shardが12時間を超える場合もgridやsampleを削らず停止する。
- 再現性リスク: solver packageやsupport sourceが変わると結果が変わり得る。package version、solver source SHA、input/output SHAを記録する。
- 手法忠実性リスク: exp015の最終graphではなくILP前graphから再solveする必要がある。manifestとcandidate countを必須guardにし、repair済みgraphを誤入力しない。
- 過度な縮小 / proxy化リスク: sample、alpha、公式metric、division guardを減らして完走扱いにしない。失敗行を除外したscoreを成功判定に使わない。

## 後段repairの契約

- gateを通った場合も実験ディレクトリは`exp018_graph_cost_scale`のままとする。
- 2方向それぞれでcalibration胚から選んだalphaを、その方向の外側胚へ適用する。2方向のalphaが異なっても、全データから1つへ再選択しない。
- exp015と同じmotion relink、gap、safe-division、DeepCenter repairを変更せず適用し、同じ固定公開対照を再計算する。
- repair parameterの再探索、repairの一部だけの変更、外側胚によるalpha再選択は行わない。
- 後段の実装・実行前に初段のscore、division、件数、SHA、runtimeをユーザーへ示す。gate失敗時は後段Notebookを追加しない。

## 受け入れ基準

- [x] backlogの上位仮説、検証範囲、残る検証、根拠、差分、固定事項、成功条件、停止条件、実行しないこと、判断履歴を移行した。
- [x] `config.yaml`の`lineage.hypothesis_id`と`lineage.backlog_candidate`が本書と一致する。
- [x] input / target / output / loss / decode / context unitと`staged-faithful`の範囲を記録した。
- [x] alpha 5、model config 0、fold学習0、booster 0、control再学習なしを記録した。
- [x] alpha shard Jupytext sourceとNotebookが入力・source SHA guard、各alphaの199 ILP、公式metric、artifact/manifest保存を実装し、aggregate Notebookが5 shardのSHA・coverage照合、2方向選択、gate、最終artifact/metrics保存を実装する。
- [x] 合成summaryの選択・tie-break・両方向gateとconfig契約testが通る。
- [x] Jupytext round-trip、validate-exp、check-exp、test-expが通る。
- [x] Kaggle CPU Notebookが199件×5 alphaを完走し、入力・solver・評価source・solution SHAと全metricを保存する。
- [x] 初段結果を提示し、両方向gateが成立した場合だけ同じ実験へ固定repair段階を追加する。実測ではgate不成立のため追加していない。
- [x] 2026-09-16のユーザー判断により、実験を完了、手法を不採用（`discarded`）として確定した。
