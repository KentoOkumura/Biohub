# exp001_temporal_unet3d_baseline 要件と実装方法

この文書を実装前の契約、実装方法、受け入れ条件の正とする。実行中の進捗は SESSION_NOTES.md に記録し、この文書へ重複させない。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: backlog 候補からの実験化。2026-09-09 のユーザーメッセージ「実験化してください」で承認された。
- 移行元 backlog: backlog/official_temporal_unet3d_train_submission_baseline.md
- 対応する上位仮説: HYP-20260909-01
- 親実験: なし
- 未決事項: なし
- backlog 記録から解釈を変更した箇所とユーザー承認: N/A。公開 artifacts dataset は offline wheels の供給にだけ使い、model source は固定 commit を同梱し、公開 checkpoint は読み込まない。

## 実装上の正本

- 設定と系譜: config.yaml
- 学習の正の編集対象: exp001_temporal_unet3d_baseline_train.ipynb
- 推論の正の編集対象: exp001_temporal_unet3d_baseline_inference.ipynb
- Notebook 生成元:同名の Jupytext percent 形式 .py
- 固定 source: official_source/SOURCE.json に記録した GitHub commit と各ファイル SHA-256
- 学習成果物: artifacts/model/edge_predictor_best.pth、config.json、model_manifest.json、split_0.json、training_log.txt、training_summary.json
- 推論成果物: submission.csv、inference_summary.json、metrics.json

## 移行した候補契約
- 候補名: `official_temporal_unet3d_train_submission_baseline`
- 実験化時の候補状態: `設計可能・実験化未承認`
- 対応する上位仮説: `HYP-20260909-01`
- 関連する上位仮説: なし
- 作成日: 2026-09-09
- 最終更新日: 2026-09-09
- 依頼原文: 「公式ノートブックを提出しベースラインを作成するバックログを作成してください」「対象は変えてください。3dunetが使用されているものにしてください。」「これです。これは3dunetは使用されていないですか https://github.com/royerlab/kaggle-cell-tracking-competition」「学習からやり直したいですが、現実的ではないですか」「それでバックログを更新してください」
- 期待する成果: 主催者公開GitHub repositoryの`TemporalUNet3D`と`SimpleNodeTransformer`をrandom initializationから、公開checkpointの作成に使われたREADME記載の3 epochsで学習する。checkpointを固定して別のKaggle Notebookでhidden testを推論し、提出形式を検証した後、ユーザーの明示承認を得てsubmissionする。学習時間、validation値、checkpoint SHA、Public LB、submission ref、submission SHAを後続実験の比較基準として記録する。
- 親実験 / 比較対象: 親実験なし。採点済み実験はなく、公開checkpointは同一性を要求しない外部参照値としてのみ扱う。
- 優先度: P0
- 優先度の理由: 現在の重点は最初のbaseline確立であり、`experiment_summary.md`と`SUBMISSIONS.md`には比較可能な実験・提出がない。独自のmodel変更前に主催者公開手法を学習から通し、計算時間、checkpoint、推論、Public LBを一続きの再現可能な基準として固定する。
- `backlog/KAGGLE_DIRECTION.md` の対応箇所: 「検証中の仮説」の`HYP-20260909-01`と「未着手バックログ」の`official_temporal_unet3d_train_submission_baseline`行。

## 観測事実と根拠

- 実測済みの事実: 主催者のpublic GitHub repositoryは、temporal attentionを持つ3D U-Netの`TemporalUNet3D`でvoxel featureとcell-center detection mapを生成し、検出中心のfeatureを`SimpleNodeTransformer`へ渡して隣接timepointのnode pairをscoreする。READMEは公開checkpointを3 epochsで学習したと明記し、未収束なので長時間学習に改善余地があるとしている。固定commitのtraining scriptはCLI既定50 epochs、AdamW、learning rate `1e-4`、batch size 16、window size 2、downsample `(1,4,4)`、UNet layers `(32,64,128)`、T4 x2のDataParallel対応である。split fileがなければseed 0でsampleをshuffleした90%/10%のsplit 0を生成する。trainは199 sample、各100 frames、shape `(100,64,256,256)`である。公式設定を使った学習時間、CV、LBはこのリポジトリでは未実測である。
- 根拠ファイル / 一次資料: [主催者GitHub repositoryの固定commit](https://github.com/royerlab/kaggle-cell-tracking-competition/tree/075fc5f5a52d11077f9dc2b074644618f26939e2)、[学習script](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/scripts/train_unet_transformer.py)、[`TemporalUNet3D`実装](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/src/tracking_cellmot/models/temporal_unet.py)、[推論script](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/scripts/predict_unet_transformer.py)、[READMEがリンクするKaggle推論Notebook](https://www.kaggle.com/code/thibautgoldsborough/unet-baseline-inference-submission)、[`docs/temporal_unet3d_explainer.md`](../../docs/temporal_unet3d_explainer.md)、[`docs/surveys/biohub-repository-setup-validation_20260814.md`](../../docs/surveys/biohub-repository-setup-validation_20260814.md)、[`docs/01_competition.md`](../../docs/01_competition.md)、[`project.yml`](../../project.yml)
- 利用する保存済み生成物とSHA: 学習済みmodelは再利用しない。GitHub sourceはcommit `075fc5f5a52d11077f9dc2b074644618f26939e2`へ固定する。学習Notebookが生成したbest checkpoint、model config、split JSON、実行logのSHAを保存し、そのcheckpointだけを推論Notebookへ入力する。
- 仮定: 「学習からやり直す」は、公開checkpointをwarm startに使わず、READMEが公開checkpointの作成条件として示す3 epochsをrandom initializationから実行する意味とする。公開時のrandom seedは記録から確定できないため、当リポジトリ既定のseed 42を明示して再現可能性を確保し、公開checkpointとのbyte同一性は主張しない。

## この候補が直接検証する仮説と範囲

- 上位仮説のうちこの候補が検証する範囲: 主催者公開のmodelとlossを保持した3 epochsのscratch trainingがKaggle制限内で完走し、生成checkpointから形式上有効なhidden test提出を作成して最初のPublic LBを得られるかを検証する。
- この候補の具体的な仮説: T4 x2で公式公開設定の3 epochsを学習Notebook内の12時間制限内に完了し、best checkpointを保存できる。そのcheckpointを別の推論Notebookで使うと、hidden test全datasetの`submission.csv`が生成され、scoringが完了する。
- 仮説が正しい場合に期待する観測: 3 epochsすべてについてtrain/validation loss、edge accuracy、node recall、epoch所要時間が記録され、best checkpointとconfigのSHAを取得できる。推論Notebookが完走し、提出形式検証を通る`submission.csv`、submission ref、Public LB、Notebook実行時間、submission SHAを記録できる。
- 仮説を棄却する観測: T4 x2でOOMまたは12時間超過となる、3 epochsを完了してcheckpointを保存できない、checkpointを推論Notebookへ受け渡せない、hidden test全datasetの有効な提出を生成できない、またはsubmissionが採点不能になる。
- この候補だけで上位仮説を判断できるか: はい
- 上位仮説の判断に残る検証: なし

## 手法契約

## 入力・予測対象・出力・推論方法

実装区分は`docs/glossary.md`に定義したこのリポジトリ内の管理用ラベルを使う。短いsmokeはfull trainingを置き換えず、同じsourceと設定で実行経路と所要時間を確認するためだけに行う。

- input: 学習はKaggle competitionのtrainにある4次元画像 `(T,Z,Y,X)` と疎な正解tracking graphのGEFF。推論は学習Notebookが生成したcheckpointとhidden test画像。
- target / objective: cell-center detection mapと、隣接timepointの検出node間のedgeを同時に学習する。
- output: 学習時はbest checkpoint、model config、split JSON、train/validation metrics、実行時間。推論時はnodeとedgeから成るtracking graphを`submission.csv`へ変換する。
- loss: 正解node voxelに対する重み付きBinary Cross Entropyのdetection lossと、疎なannotationを考慮して正解edgeを持つrowまたはcolumnを対象にするedge lossの和。weightは公開training scriptのCLI設定を固定する。
- decode / 推論方法: 連続frameを`TemporalUNet3D`へ入力し、cell-center detection mapから3D local maximum suppressionでnode候補を抽出する。候補位置のvoxel featureと位置embeddingを`SimpleNodeTransformer`へ入力して隣接timepointのnode pairをscoreし、公開推論処理でtracking graphとCSVへ変換する。
- 処理単位: window size 2の連続frameをtraining batchとして扱い、推論では隣接する2 timepointのnode集合ごとにedgeを予測してdataset単位のgraphを出力する。
- 実装区分: `staged-faithful`。smokeの後に、主催者公開コードと3 epochsのtraining/inferenceを省略せず実行する。seed 42の固定と記録処理だけを再現性のために追加する。

## 親実験からの差分

- 変更するもの: 親実験はない。固定commitの学習・推論コードをKaggle Notebookへ配置し、seed 42、実行時間、metrics、checkpoint/config/split/submissionのSHA記録、提出形式検証を追加する。
- 固定するもの: random initialization、3 epochs、split 0、90%/10%のseed-0 sample split、AdamW、learning rate `1e-4`、batch size 16、window size 2、downsample `(1,4,4)`、UNet output channels 32、layers `(32,64,128)`、公開scriptのaugmentation・loss・checkpoint selection、T4 x2 DataParallel、internet無効。学習後のcheckpointは変更せず推論へ渡す。
- 再利用するコード / config / 生成物: GitHub commit `075fc5f5a52d11077f9dc2b074644618f26939e2`のmodel・学習・推論・CSV変換コード、`project.yml`のcompetition・submission・runtime設定、リポジトリの提出形式validator。公開checkpointは使用しない。
- 新しく作るもの: train NotebookとKaggle package、best checkpointとmodel config、inference Notebookとpackage、`metrics.json`・`SESSION_NOTES.md`・`result.md`、検証済み`submission.csv`。実際のsubmissionは直前にユーザーの明示承認を得てから行う。

## 探索幅とpivot判定

- 変更class: mechanism。親実験なしの最初の3D U-Net baselineを作る。
- 同じ親 / familyで連続した小改善実験数: 0。
- positiveなoracle headroom / coverage / 誤差非相関性: 未測定。baseline確立が先であり、この実験内ではparameter tuningを行わない。
- 比較したtarget、output、decode、context unitを変える案: 2D U-Net、50 epochs、leave-one-embryo-out、window size拡大、detection閾値調整はいずれも未実行。
- 小改善の継続またはpivotを選ぶ根拠: 最初のbaselineなので公開手法を固定して実行し、結果を得るまでpivotしない。
- `kaggle-idea-forge` の実行要否と根拠: 不要。今回は既に承認された具体的な公開手法の再現である。

## 最小の反証可能な検証

- 検証方法: まず同じsource、GPU、shape、batch sizeで`--debug-video`と少数`--max-iters`を使い、forward/backward、checkpoint保存、1 iterationおよびepoch相当の推定時間、peak GPU memoryを確認する。合格時だけsplit 0、3 epochs、全iterationのtrain NotebookをKaggleで実行する。checkpointを取得・SHA固定後、別のinference Notebookをフル実行する。生成物を`task submit-check EXP=<exp> SUBMISSION=<path>`で検証し、明示承認後に1回submissionしてscoring完了まで監視する。
- variant / config / fold / booster数: variantはseed 42の公式3-epoch設定1条件だけ。foldはsplit 0だけ。boosterはなし。
- control再学習: なし。既存の自前checkpointがなく、公開checkpointは比較対象であって学習入力にはしない。
- 想定runtime / resource: trainはKaggle T4 x2、inferenceは元Notebookに合わせたGPU、internet無効。各Notebookは12時間以内。smokeの実測から、3 epochsと各epoch validation、checkpoint保存が11時間以内に収まる見込みがない場合はfull trainingを開始せず停止する。

## 成功条件と停止条件

- primary指標: scratch trainingしたcheckpointによるKaggle Public LB。補助証拠として各epochのvalidation edge accuracy、node recall、両者の積、train/validation loss、所要時間を記録する。
- 成功条件: smokeがOOMなく完了し、3 epochsのfull trainingが11時間以内に完走してbest checkpointを保存する。そのcheckpointだけを使うinferenceが12時間以内にhidden test全datasetの有効な`submission.csv`を生成し、submissionのscoringが完了する。checkpoint SHA、submission ref、Public LB、Notebook実行時間、submission SHAを記録できること。
- 必須guard: 学習開始前にGitHub commitと全hyperparameterを`config.yaml`へ固定する。smokeとfull trainingを区別し、smoke checkpointを提出に使わない。training inputとsplitを記録し、公開checkpointを読み込まないことを検査する。checkpoint/config/split/sourceのSHA、Kaggle kernel id/version、container image、GPU、internet設定を記録する。submission前に形式検証を通し、ユーザーの明示承認を得る。
- 成功時の次段階: 得られたPublic LBを学習から再現した3D U-Net baselineとして固定する。実験の完了判断、50 epochsへの延長、2方向leave-one-embryo-out、temporal context変更へ進むかはユーザーへ確認し、自動実行しない。
- 失敗時の停止範囲: smokeのOOM、3 epochsの予測所要時間超過、full training失敗、checkpoint受け渡し失敗、inferenceまたは提出形式検証失敗、scoring失敗の原因を記録して停止する。同じ実験内でbatch size、model幅、downsample、epoch数、split、loss、decodeを変更して救済しない。

## 実行しないこと

- 禁止する代替実装、proxy、同一OOF上の救済探索: 公開checkpointによるwarm start、2D U-Netや別の3D U-Netへの置換、model縮小、batch size変更、追加fold、50 epochsへの延長、leave-one-embryo-outへの変更、detection閾値調整、edge後処理追加、ensemble、metric exploit、失敗時の別方式によるfallback出力を行わない。変更が必要なら別候補またはユーザー確認後の契約変更として扱う。
- 壁打ちで採らなかった案と理由: 公開checkpointをそのまま提出する案は、ユーザーが学習からの再実行を希望したため対象から外した。50 epochsは公開checkpointの作成条件3 epochsと異なり、12時間制限とGPU quotaの実測がないため初回へ含めない。2方向leave-one-embryo-outは信頼できるCVには必要だが、学習を2回必要とし、まず公式再現の時間と出力を測る目的から分離する。window sizeを増やす案は時間方向のcontextを変える高upside候補だが、baseline確立前には実行しない。

## 再現性・リスク

- leakage / validation: 公式のseed-0 90%/10% splitはsample単位で、同じ`embryo_id`がtrainとvalidationへ混ざり得る。ここで得るvalidation値をprimary CVや汎化性能の証拠にせず、公開学習手順の再現診断としてのみ使う。
- hidden test: Code Competitionでは入力がhidden testへ差し替わる。学習時と異なるembryo、dataset数、shape、Zarr layout、node数により、推論時間やmemoryが公開testから増える可能性がある。
- runtime / memory: 3D convolution、時間方向attention、node pair scoringのforward/backwardはGPU memoryと時間を消費する。T4 x2でも3 epochsが11時間以内に収まることはsmoke実測前には確定しない。
- 再現性: 公開training scriptはCLIからrandom seedを渡していないため、seed 42をNotebook側でPython、NumPy、PyTorch、DataLoaderへ設定する最小追加を行う。GitHub commit、split、source SHA、Kaggle image、GPU、checkpoint SHA、submission SHAを記録する。公開checkpointとのbyte同一性は期待しない。

## 未決事項

- なし

## 判断履歴

- 2026-09-09: 当初の最近傍Getting Started Notebookは3D U-Netを使わないため、ユーザー指示で対象から除外した。
- 2026-09-09: 主催者GitHub repositoryとREADMEがリンクする3D U-Net推論Notebookを対象にしたが、公開checkpointをそのまま使う契約としていた。
- 2026-09-09: ユーザーが学習からの再実行を希望したため、公開checkpointの作成条件と同じ3 epochsをrandom initializationから学習し、そのcheckpointで推論・提出する契約へ更新した。50 epochsとleave-one-embryo-outは後続判断へ分離した。

## 次セッションへの引き継ぎ確認

- 固定するものを一意に説明できる: はい
- 変更するものを一意に説明できる: はい
- 最小検証と停止条件を一意に説明できる: はい
- 実行しないことを一意に説明できる: はい
- 未決事項が明示されている: はい（なし）

## 実装方法

- train Notebook は competition train を動的に解決し、199組の Zarr/GEFF を確認して seed 0 の90%/10% sample splitをJSONへ保存する。
- offline wheelsから tracksdata と zarr をinstallし、同梱sourceの全SHAを SOURCE.json と照合する。dependency dataset内の公開checkpointやsourceは使わない。
- Python、NumPy、PyTorch、DataLoader generatorへseed 42を設定する。固定source内のaugmentationは item ごとに引数なしの NumPy default_rng を作るため、augmentation drawのbyte-level再現性は主張しない。
- 同じarchitecture、batch size、downsample、lossで1 sample・1 epoch・2 training iterationのsmokeを実行し、peak GPU memoryとbatch換算時間からfull 3 epochsを保守的に見積もる。11時間を超える見込みならfull trainingを開始しない。
- runtime gate通過時だけrandom initializationからsplit 0を3 epochs学習し、edge accuracyとnode recallの積が最大のcheckpointを保存する。
- inference Notebookはtrain kernel outputのmodel_manifest.jsonを唯一のcheckpoint入口とし、manifestとcheckpoint/config SHAを検証する。
- hidden testの全 .zarr 名を実行時に列挙し、公開 inference Notebook と同じ detection threshold 0.99、UNet batch size 4、ILP設定でGEFFを生成し、node/edge行のsubmission.csvへ変換する。
- submission schema、dataset coverage、node id重複、edge endpoint参照、欠損をNotebook内で検査し、リポジトリ側のsubmit checkはKaggle output取得後に実行する。
- 実際の Kaggle submission はこの実験化承認に含めず、ユーザーの別途明示承認まで実行しない。

## 探索幅と pivot 判定

- 変更 class: mechanism。採点済み親実験がないため、主催者公開modelを最初の学習baselineとして通す。
- 同じ親またはfamilyで連続した小改善実験数: 0
- positiveなoracle headroom、coverage、誤差非相関性: 未測定。baseline確立前なので探索根拠には使わない。
- target、output、decode、context unitを変える案: 50 epochs、2方向leave-one-embryo-out、長いtemporal contextはすべて後続候補として分離する。
- 小改善の継続またはpivotを選ぶ根拠: 本実験は比較基準の作成であり、成功・失敗の実測後にユーザーが判断する。
- kaggle-idea-forge の実行要否と根拠: 不要。既にユーザーが対象と契約を指定している。

## 受け入れ基準

- [ ] input、target、output、loss、decode、context unitが固定sourceと一致する。
- [ ] config.yaml の hypothesis_id と backlog_candidate が本書と一致する。
- [ ] 同梱sourceの全SHAが SOURCE.json と一致する。
- [ ] 公開checkpointを学習入力にしないguardがある。
- [ ] smokeのGPU、peak memory、所要時間、full runtime予測が記録される。
- [ ] runtime gate通過時はsplit 0の3 epochsが完走し、checkpoint、config、split、log、manifestのSHAが保存される。
- [ ] inferenceは学習済みmanifestからcheckpointを再学習なしで解決する。
- [ ] hidden testの全datasetを動的に列挙し、可変行数のsubmission.csvを生成・検証する。
- [ ] 実験固有test、static check、Notebook変換testが通る。
- [ ] Kaggle kernel id/version、container image、GPU、internet設定、Notebook実行時間がmetrics.jsonへ記録される。
- [ ] submissionはユーザーの別途明示承認後だけ実行される。
