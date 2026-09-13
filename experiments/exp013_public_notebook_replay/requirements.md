# exp013_public_notebook_replay 要件と実装方法

この文書を、実装前の契約、実装方法、受け入れ条件の正とする。実行中の進捗は`SESSION_NOTES.md`へ記録する。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: backlog候補`public_notebook_replay`は、公開Notebook独自patchのsource保持方法が未決だったため`検討メモ・設計不可`だった。2026-09-13のユーザー依頼「public_notebook_replayを実装してください」で実験化を依頼され、同日の「そのライセンスは気にする必要ありません」で公開Notebook本文を実験ディレクトリへ保持する方針が承認された。
- 移行元backlog: `backlog/public_notebook_replay.md`
- 対応する上位仮説: `HYP-20260910-12`
- 上位仮説のうちこの実験が検証する範囲: 固定公開構成を同じ入力条件で再実行でき、後続の特徴cache・診断・トラッカー学習へ渡す安定した基準予測と実測費用を作れるか。
- この実験だけで上位仮説を判断できるか: いいえ。
- 上位仮説の判断に残る検証: `exact_window_cache`の保存等価性と費用、`frozen_image_encoder`の下流学習成立と精度、後続候補の改善幅。
- 親実験: `exp011_public_detector_selection`。同実験で採用した公開Notebook取得版、3 checkpoint、feature contractを入力とする。
- 根拠 / 一次資料 / 参照実装: `experiments/exp011_public_detector_selection/assets/public_detector_selection.json`、`experiments/exp011_public_detector_selection/result.md`、`docs/surveys/biohub-cell-tracking-0946-notebook-explanation_20260913.md`、kernel id `133199516`の取得版。
- 固定するもの: 参照Notebook content SHA `ae8e01a262211045161984e469e8be23e3386bab9140fe12df503dc6a1e010e6`、13 Python source SHA、3 checkpoint SHA、全推論設定、前処理、候補生成、association、integer linear programming（ILP）、repair、Kaggle docker image、T4 2基、testのsorted order。
- 変更するもの: 作者mirrorからexp011で採用したPilkwangのcanonical dataset 3件へmount先を置き換える。wheel filename・size・SHAを含む入力manifest、run receipt、2 run比較を追加する。予測処理の閾値・モデル・順序は変更しない。
- 最小の反証可能な検証: 同じprivate Kaggle Notebookをclean stateから2 version実行し、公開test全4動画のcandidate coordinate SHA、graph topology、決定的なrun統計、`submission.csv`を照合する。
- 成功条件: 2 runのsource・artifact・dependency・environment manifestが一致し、`submission.csv`がbyte-identicalで、candidate coordinate SHAとgraph topologyも一致する。
- 停止条件: source・artifact・dependency guardの不一致、canonical datasetでの起動失敗、1回目と2回目の不一致、GPU残量不足、または12時間超過で停止する。決定性設定や処理順序を変える場合は別variantとしてユーザー確認を受ける。
- 実行しないこと: 作者outputを自分のrunとして扱う、titleやhard-coded receiptをPublic LB証拠とする、public test固有IDで分岐する、1動画smokeを全件再現と呼ぶ、閾値・TTA・ILP・repairを変更する、CPUや別modelへ置換する、Kaggle submissionを行う。
- 未決事項: なし。Public LB確認を伴うsubmissionは別途の明示承認を必要とし、本実装・2 replayには含めない。
- backlog記録から解釈を変更した箇所とユーザー承認: 2026-09-13にユーザーが、公開Notebookのsource保持に関するライセンス懸念を本作業では考慮不要と明示した。これによりexact sourceを`assets/reference_notebook/`へ保存し、取得版から生成する実行Notebookをリポジトリで管理する。

## 判断履歴

- 2026-09-13: 公開Notebookの実装と再現性を先に確認し、full inference・再実行一致・Public LB確認を独立したP1候補として追加した。
- 2026-09-13: 現行source、metadata、support source、作者outputを監査した。source・checkpoint・設定・output guardは確認できたが、2回目の実行、決定性設定、独立submissionはなく、再現性は未担保と判断した。
- 2026-09-13: ユーザーが`public_notebook_replay`の実装を依頼した。
- 2026-09-13: ユーザーが公開Notebook sourceのライセンス懸念を本作業では考慮不要と明示し、sourceを実験内へ保持する方針を選択した。
- 2026-09-13: ユーザーが「実行してください」と明示し、private Kaggle Notebookの2回実行を承認した。Kaggle submissionの承認とは扱わない。
- 2026-09-13: ユーザーが「提出までしてください」と明示し、再実行一致を確認したprivate Notebook version 2のcode submissionを承認した。

## 手法契約

実装区分は`docs/glossary.md`に定義したこのリポジトリ内の管理用ラベルである。本実験は公開test全件のreplayと比較を先に行い、submissionは別の明示承認後に同じ実験へ追記する。

- 依頼原文: 「公開NotebookをKaggle上でfull inferenceし、再実行一致とPublic LBを確認する独立候補をP1として追加する」「まずはノートブックの実装を確認して再現性が担保されているか確認してください」「public_notebook_replayを実装してください」「そのライセンスは気にする必要ありません」
- 期待する成果: 採用した公開Notebookと同じsource・artifact・Kaggle環境で公開test全件を2回推論し、予測と中間出力の一致を確認できる実行・比較経路を作る。
- input: content SHAを固定した公開Notebook、competitionのtest Zarr、exp011で固定したPilkwang original dataset 3件、全offline wheel、docker image SHA、T4 2基、公開testのsorted dataset一覧。
- target / objective: 同じ固定条件から同じ候補点・接続・修復後graph・submissionを生成できるか確認する。
- output: runごとのenvironment・source・artifact・wheel manifest、integrity receipt、candidate coordinate manifest、retention guard、`run_stats.csv`、`submission.csv`、replay receipt、2 run間diff。
- loss: なし。学習しない。
- decode: 採用sourceの2 model検出、8-view D4 test-time augmentation（TTA）、forward/reverse association、secondary低margin補助、ILP、motion・gap・division repair、DeepCenter veto、short-track filteringを変更しない。
- context unit: 公開test全動画を含むNotebook全体。中間一致は動画・frame・候補・graph単位、最終一致は`submission.csv`全体で確認する。
- 実装区分: `staged-faithful`。第1段階は公開test全件の2回replay、第2段階は別途の明示承認を受けたcode submissionであり、sourceの予測処理を省略しない。
- 省略する機構と理由: 予測機構は省略しない。第1段階ではKaggle submissionとPublic LB取得だけを省略し、再実行一致を先に確認する。
- proxyで検証できない主張: N/A。proxyではないが、公開testの一致だけではhidden testの一致、12時間完走、独立validation、未知胚への一般化を判断できない。
- proxyの場合のユーザー承認: N/A。
- この実験が支持 / 棄却できる主張: 固定したsource・入力・依存・GPU条件で、公開test全件の候補座標、graph、submissionを同じ内容として再生成できるか。
- この実験では判断できない主張: detector単体精度、公開重みの学習来歴、独立validation、hidden testでのbitwise一致、Public LB。Public LBはsubmission承認後だけ取得する。

## 実装方法

- アプローチ: Kaggleから取得したunmodified notebookを`assets/reference_notebook/`へSHA付きで保存する。そこからJupytext percent形式の実行sourceを生成し、canonical path置換とreplay証拠cellだけを追加して`exp013_public_notebook_replay_inference.ipynb`へ変換する。
- inputの実装箇所と変換: `config.yaml`の`runtime.kaggle.inference.dataset_sources`でcanonical dataset 3件をmountし、notebook先頭のruntime guardでGPU、dataset root、全wheel、source/checkpoint manifestを収集する。作者mirrorを指す環境変数だけをcanonical pathへ置換する。
- target / objectiveの構築箇所: notebook末尾で公開test全件の出力ファイルを収集し、`replay_receipt.json`にcontent SHA、row数、dataset一覧、graph topology、決定的なrun統計を保存する。
- outputの生成箇所と表現: 公開source既存の`submission.csv`、`run_stats.csv`、`detector_coordinates_*.jsonl`、`retention_guard_*.jsonl`、integrity receiptに加え、`replay_input_manifest.json`と`replay_receipt.json`を`/kaggle/working`へ保存する。`compare_replays.py`が2つのoutput directoryを比較し、JSON reportを標準出力または指定先へ出す。
- lossの実装箇所: N/A。
- decode / postprocessの実装箇所: 取得source内の既存処理を保持する。replay追加部分は予測前の入力検査と予測後のoutput hashだけで、候補・edge・graphを書き換えない。
- context unitを保つ処理箇所: 公開sourceのsorted test列挙、2 GPU shard、全graph被覆検査を保持し、receiptはdatasetごとの候補座標とtopologyをsorted keyで集約する。
- 変更するファイル / component: exp013の契約・設定・記録、reference source、Jupytext inference source、inference notebook、replay比較helper、実験固有test、移行後の戦略索引。
- 固定事項を保つ確認方法: testでreference SHA、canonical path以外の予測設定差分がないこと、3 checkpoint SHA、13 source SHA、T4×2、dataset source、comparison contractを検査する。
- 参照sourceとの一致を確認するテスト: reference `.ipynb` SHAを固定し、実行`.py`からreplay追加cellとcanonical path差分を除いた公開source本文がreference code cellと一致することを検査する。
- 承認済み差分を確認するテスト: mirror owner文字列が実行sourceのpath設定に残らず、canonical 3 datasetとsource attribution、input/output receiptが存在し、prediction parameterの変更がないことを検査する。

## 探索幅とpivot判定

- 変更class: `add-only`。予測機構を変更せず、canonical mountと再現性検査を追加する。
- 同じ親 / familyで連続した小改善実験数: 0。score改善ではなく、採用済み公開構成の実行再現性を検査する。
- positiveなoracle headroom / coverage / 誤差非相関性: 対象外。公開testの再実行一致を測る。
- 比較したtarget、output、decode、context unitを変える案: 対象外。親実験で固定した出力・decode・Notebook全体の処理単位を変えないことが検証条件である。
- 小改善の継続またはpivotを選ぶ根拠: scoreを改善する実験ではなく、後続比較の基準予測を作る前提監査である。
- `kaggle-idea-forge` の実行要否と根拠: 不要。承認済みの具体的候補を実験へ移行し、予測機構を変えず再現性を確認するため。

## 再現性・リスク

- seed policy: 追加処理は乱数を使わない。公開sourceにも明示的な乱数生成はないが、CUDA・SCIP・dependencyまで含むbitwise一致は2 runの実測で判断する。
- stochastic 処理の有無: 明示的な乱数処理は確認されていない。CUDA kernel、ILP solver、並列processに実装由来の非決定性が残る可能性がある。
- stochastic feature generation / augmentation / seed bagging の有無: D4 TTAは固定8 viewの決定的列挙。学習・augmentation sampling・seed baggingは行わない。
- 並列処理と乱数の関係: 最大2 GPUのdataset shardを使う。dataset順とshard割当はsorted orderで固定され、output集約時もsortする。明示的乱数は使わない。
- CPU/GPU runtime と deterministic flags: T4 2基を必須とする。元sourceに`torch.use_deterministic_algorithms`、cuDNN deterministic設定、`CUBLAS_WORKSPACE_CONFIG`はないため追加せず、実測一致の対象とする。
- train cache / test feature regeneration の SHA 記録方針: train cacheはない。testから再生成するcandidate coordinateはdataset単位のSHA・row数・frame countを記録する。
- model manifest / prediction / submission SHA 記録方針: 3 checkpoint SHA、13 source SHA、reference source SHA、実行notebook SHA、candidate coordinate SHA、graph topology、`test_prediction_content_sha`、`submission_sha`をrunごとに記録する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks`で正のnotebook、config、metrics、比較helperをpackage化し、push前validatorで一致を確認する。生成packageは手編集しない。
- リークリスク: 作者receiptはleaderboard feedbackを設定へ使用したと記録する。公開test再現は独立validationや未知胚への一般化の証拠ではない。ground truth、作者output、保存済みsubmissionを予測生成へ使わない。
- CV/LB 不一致リスク: CVはない。Notebook titleとhard-coded receiptの0.946/0.947を実測LBとして扱わない。
- ランタイム/メモリリスク: 作者runの公開test prediction部分は9.634876分だが、Notebook全体・canonical mount・hidden testの時間へ外挿しない。2 replay前にGPU残量を確認する。
- 再現性リスク: 2 runが一致しても1組のhardware・docker image内の実測であり、全環境でのbitwise再現性とは呼ばない。dataset versionはmetadataだけで固定せず、全input file SHAで照合する。
- 手法忠実性リスク: mirrorからcanonical datasetへのpath変更とreplay guard以外にsource差分が入ると比較対象が変わるため、差分testで拒否する。
- 過度な縮小 / proxy化リスク: 1動画smokeや作者output比較で全件2 replayを代替しない。static validationだけを再実行一致と呼ばない。

## 受け入れ基準

- [x] 手法契約の `input / target / output / loss / decode / context unit` がcodeと一致する。
- [x] 実装区分と実験名が実装した処理を正確に表す。
- [x] `staged-faithful`の第1段階と、submissionを省略する範囲が記録されている。
- [x] backlogの上位仮説ID、検証範囲、残る検証、根拠、差分、成功条件、停止条件、実行しないこと、判断履歴が移行されている。
- [x] `config.yaml`の`lineage.hypothesis_id`と`lineage.backlog_candidate`がこの文書と一致する。
- [x] reference source SHA、source差分、canonical dataset mount、3 checkpoint、13 source、wheel、GPU条件を検査する。
- [x] 実験固有testと`validate-exp`、`check-exp`、`test-exp`が通る。
- [x] 2つのclean Kaggle runを比較し、必要なSHAとkernel versionを`metrics.json`の`evidence.reruns`へ記録する。
- [x] 2 run一致前にdeterministic anchor、実験完了、採用と記録しない。
