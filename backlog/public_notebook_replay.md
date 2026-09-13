# public_notebook_replay

- 候補名: `public_notebook_replay`
- 状態: `検討メモ・設計不可`
- 対応する上位仮説: `HYP-20260910-12`
- 関連する上位仮説: なし
- 作成日: 2026-09-13
- 最終更新日: 2026-09-13
- 依頼原文: 「公開NotebookをKaggle上でfull inferenceし、再実行一致とPublic LBを確認する独立候補をP1として追加する」「バックログに追加してください」「まずはノートブックの実装を確認して再現性が担保されているか確認してください」
- 期待する成果: 採用した公開Notebookと同じsource・artifact・Kaggle環境で公開test全件を2回推論し、予測の一致を確認する。別途の明示的なsubmission承認後にcode submissionを行い、Public LBとhidden test実行時間を作者報告から独立して記録する。
- 親実験 / 比較対象: [exp011_public_detector_selection](../experiments/exp011_public_detector_selection/)で採用した取得版と、2026-09-13に同じkernel id・content SHAから取得した作者の公開run。
- 優先度: P1
- 優先度の理由: 公開構成の予測をcache・診断・トラッカー学習の基準にする前に、同じ入力から同じ予測を生成できることと費用を確認する必要がある。exact_window_cacheより前、または同じ初回抽出で扱う。
- backlog/KAGGLE_DIRECTION.md の対応箇所: [検証中の仮説と未着手索引](KAGGLE_DIRECTION.md#検証中の仮説)

## 観測事実と根拠

- 実測済みの事実: 2026-09-13のKaggle CLI 2.2.4では、採用時のref reyhanksatria/biohub-cell-tracking-0-946-lbは404だった。検索で同じkernel id 133199516がreyhanksatria/biohub-cell-tracking-0-947-lbとして見つかり、Notebook SHA256はexp011のae8e01a262211045161984e469e8be23e3386bab9140fe12df503dc6a1e010e6と一致した。コードを変えずslugとtitleだけが更新されている。
- 実測済みの事実: 現在のmetadataはpublic、GPU有効、internet無効、NvidiaTeslaT4、docker image SHA256は37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461である。一方、dataset_sourcesは空文字3件で、metadataだけでは必要なdatasetを再接続できない。
- 実測済みの事実: 作者の公開outputを取得できた。submission.csvは4 dataset・241282 rows、SHA256は0319ba6d8e864335d3573f6b1a6227c546f17e9247a0c2858fa09b6c2422db3f、run_stats.csvのprediction部分は9.634876分だった。これは作者による1回の公開test実行であり、このリポジトリによる再実行ではない。
- 実測済みの事実: Notebookは主要設定、13 Python source、3 checkpoint、dynamic patch、test全件被覆、GPU shard、submission schema・ID・dataset・座標・lineage次数を検査し、最終submission SHAをreceiptへ保存する。test名とgraphはsorted orderで処理する。
- 実測済みの事実: Notebook本体と固定sourceにはtorch.use_deterministic_algorithms、cuDNN deterministic設定、CUBLAS_WORKSPACE_CONFIGがない。Notebook本体に乱数生成は見つからなかったが、CUDA・SCIP・dependencyを含む同一出力は1回の実行だけでは確認できない。
- 実測済みの事実: receiptのstatus: verified_public_lb_0946はNotebookコードで設定された文字列で、validated_receipt_sha256はnullである。現在のtitleは0.947、同じコード内のreceiptは0.946であり、どちらもこのリポジトリが取得したsubmission結果ではない。
- 実測済みの事実: offline wheelはversion付きfilenameを持つが、Notebookはwheel SHAを検査せず、複数directoryをpip --find-linksへ渡す。別の互換wheelが同時にmountされた場合の選択は固定されていない。
- 根拠ファイル / 一次資料: [exp011 manifest](../experiments/exp011_public_detector_selection/assets/public_detector_selection.json)、[exp011 result](../experiments/exp011_public_detector_selection/result.md)、[公開Notebook解説](../docs/surveys/biohub-cell-tracking-0946-notebook-explanation_20260913.md)、[Kaggle上の現在のNotebook](https://www.kaggle.com/code/reyhanksatria/biohub-cell-tracking-0-947-lb)。
- 利用する保存済み生成物とSHA: 参照Notebookと3 checkpointの正はexp011 manifest。作者outputは監査時に/tmpへ取得しただけで保存していないため、上記SHA・row数・dataset数を再取得時の照合値とする。実験化後は自分の各runのsource、input、dependency、prediction、submission SHAを対応実験へ保存する。
- 仮定: Assumption: exp011で固定したPilkwang original dataset 3件は、作者mirrorと同じsource・checkpoint SHAを持つため、同じdocker imageとGPU数で同じ推論結果を生成できる。mountとoffline dependencyを含む実行等価性は未確認である。

## この候補が直接検証する仮説と範囲

- 上位仮説のうちこの候補が検証する範囲: 固定公開構成を同じ入力条件で再実行でき、後続の特徴cache・診断・トラッカー学習へ渡す安定した基準予測と実測費用を作れるか。
- この候補の具体的な仮説: exact Notebook source、canonical artifact、dependency、docker image、T4 2基、test一覧を固定すれば、2回の公開test全件推論でbyte-identicalなsubmission.csvと同じ中間座標・graph統計を生成できる。明示承認後のcode submissionからPublic LBを取得できる。
- 仮説が正しい場合に期待する観測: 2 runのsource・入力・dependency manifestが一致し、submission.csv、候補座標manifest、graph topology、run statisticsの決定的な列が一致する。submission後にPublic LB、kernel version、hidden test実行時間、submission refを取得できる。
- 仮説を棄却する観測: 同じ固定条件でもsubmissionまたは中間座標・graphが一致しない、canonical datasetで検査を通せない、12時間以内にhidden testを完走できない、または実測Public LBが作者報告と一致しない。
- この候補だけで上位仮説を判断できるか: いいえ
- 上位仮説の判断に残る検証: [exact_window_cache](exact_window_cache.md)の保存等価性と費用、[frozen_image_encoder](frozen_image_encoder.md)の下流学習成立と精度、後続候補の改善幅。

## 入力・予測対象・出力・推論方法

実装区分はこのリポジトリ内の管理用ラベルである。最初に公開test全件でexact replayを確認し、submissionは別の明示承認後に実行する。

- input: content SHAを固定した公開Notebook、competitionのtest Zarr、exp011で固定したPilkwang original dataset 3件、全offline wheel、docker image SHA、T4 2基、公開testのsorted dataset一覧。
- target / objective: 同じ固定条件から同じ候補点・接続・修復後graph・submissionを生成できるかを確認し、作者titleや自己申告receiptではなく自分のKaggle runとsubmissionからPublic LBを取得する。
- output: runごとのenvironment・source・artifact・wheel manifest、integrity receipt、候補座標manifest、retention guard、run_stats.csv、submission.csv、SHAとrun間diff。明示承認後はkernel id/version、submission ref、Public LB、hidden test実行時間を追加する。
- loss: なし。学習しない。
- decode / 推論方法: 採用sourceの2 model検出、8-view D4 TTA、forward/reverse association、secondary低margin補助、ILP、motion・gap・division repair、DeepCenter veto、short-track filteringを変更しない。
- 処理単位: 公開test全4動画を含むNotebook全体。中間一致は動画・frame・候補・graph単位、最終一致はsubmission.csv全体で確認する。
- 実装区分: staged-faithful。第1段階は公開test全件の2回replay、第2段階は別途の明示承認を受けたcode submissionであり、sourceの処理を省略しない。

## 親実験からの差分

- 変更するもの: exp011の静的選定から、canonical datasetをmountしたKaggle実行と2 runの出力比較へ進める。Public LBは自分のsubmission結果として取得する。
- 固定するもの: Notebook content SHA、13 Python source SHA、3 checkpoint SHA、全設定、前処理、候補生成、association、ILP、repair、docker image、GPU数、test順序。
- 再利用するコード / config / 生成物: exp011のmanifestとpipeline contract、公開Notebookのintegrity/output guard、作者公開runのsubmission SHA・topology・runtime統計。
- 新しく作るもの: 利用条件に従うexact source保持方法、canonical dataset mount、wheel SHA manifest、2 run比較、Kaggle実行証拠、明示承認後のsubmission記録。

## 最小の反証可能な検証

- 検証方法: source・docker・GPU・3 dataset・全wheelをmanifest化し、既存integrity guardを通す。同一のprivate Kaggle Notebookをclean stateから2 version実行し、公開test全4動画のcandidate coordinate SHA、graph topology、submission.csv SHAを照合する。2 run一致後、ユーザーがsubmissionを明示承認した場合だけcode submissionを行う。
- variant / config / fold / booster数: 1構成、独立した2 replay。fold・booster・再学習は0。閾値や後処理を変更しない。
- control再学習: なし。既存の3 checkpointをbyte単位で固定して推論再現性だけを確認する。
- 想定runtime / resource: 作者の公開test prediction部分はT4 2基で9.634876分。Notebook全体とcanonical mount、hidden testは未測定。Kaggle GPU週30時間・課金なしを守り、1回目から2回目とsubmission分を見積もる。hidden testは12時間以内を必須とする。

## 成功条件と停止条件

- primary指標: 固定条件で独立実行した2つの公開test submission.csv SHAの一致。
- 成功条件: 2 runのsource・artifact・dependency・environment manifestが一致し、submission.csv SHAがbyte-identicalで、candidate coordinate SHAとgraph topologyも一致する。別途の明示承認後にsubmission scoringが完了し、実測Public LBとhidden test実行時間を記録する。0.946または0.947との一致は、取得日時を持つ作者title・receiptと実測値を並べて判定する。
- 必須guard: source SHA、docker image SHA、T4 2基、test一覧、13 source SHA、3 checkpoint SHA、wheel filename・size・SHAをrun開始前に照合する。既存予測を消してclean runにし、ground truth、公開output、保存済みsubmissionを予測生成へ使わない。
- 成功時の次段階: 結果と限界をユーザーへ示して再現性の採否判断を受ける。一致した基準予測はexact_window_cacheとfrozen_image_encoderへ渡すが、実験化・学習・追加submissionを自動で開始しない。
- 失敗時の停止範囲: 最初に異なる処理段階を記録して停止する。決定性設定や処理順序を変更して一致させる場合は別variantとしてユーザー確認を受ける。12時間超過またはGPU残量不足なら2回目・submissionを開始しない。

## 実行しないこと

- 禁止する代替実装、proxy、同一OOF上の救済探索: 作者outputを自分の再実行結果として扱う、titleやhard-coded receiptをPublic LB証拠とする、public test固有IDで分岐する、1動画smokeを全件再現と呼ぶ、閾値・TTA・ILP・repairを変更して差を隠す、CPUや別modelへ置換する。
- 壁打ちで採らなかった案と理由: exact_window_cacheへ同居させない。同候補は保存前後の特徴等価性と費用が対象で、公開Notebookの再実行一致・Public LB・hidden test runtimeを同時に扱うと原因を分離できない。

## リスク

- leakage / validation: 作者receiptはleaderboard feedbackを設定へ使用したと記録する。再現できても独立validationや未知胚への一般化の証拠にはならず、公開重みの不明な学習来歴も解消しない。
- hidden test: 公開testの4動画でbyte-identicalでも、hidden testでの同一性や12時間完走を保証しない。入力列挙・出力被覆はsubmission runで別に確認する。
- runtime / memory: 2回のreplayとsubmissionはGPU枠を使う。公開test prediction部分9.63分をhidden testやNotebook全体へ外挿しない。
- 再現性: 現行実装はsource・重み・出力構造の検査が強い一方、CUDAとSCIPの決定性を強制せず、wheel SHAも検査しない。2 runの一致は実測で確認し、1組のhardware・imageで一致しても全環境でのbitwise再現性とは呼ばない。

## 未決事項

- Notebook独自patchのcode licenseがpull metadataとsourceに宣言されていない。exact sourceをこのリポジトリの実験Notebookとして保存できるか、Kaggle上のprivate forkだけを実行sourceとして保持するかを実験化前に決める。後者はリポジトリ内に正の実行sourceを保存できないため、Notebook-firstの再現性記録との整合方法も確定する。

## 判断履歴

- 2026-09-13: ユーザーが公開Notebookの実装と再現性を先に確認し、full inference・再実行一致・Public LB確認を独立したP1候補として追加するよう依頼した。
- 2026-09-13: 現行source、metadata、support source、作者outputを監査した。source・checkpoint・設定・output guardは確認できたが、2回目の実行、決定性設定、独立submissionはなく、再現性は未担保と判断した。code licenseとsource保持方法が設計に影響するため、状態を検討メモ・設計不可とした。

## 次セッションへの引き継ぎ確認

- 固定するものを一意に説明できる: はい
- 変更するものを一意に説明できる: はい
- 最小検証と停止条件を一意に説明できる: はい
- 実行しないことを一意に説明できる: はい
- 未決事項が明示されている: はい
