# exp005_embryo_holdout_batch8 要件と実装方法

この文書を、実装前の契約、実装方法、受け入れ条件の正とする。実行中の進捗は`SESSION_NOTES.md`へ記録し、この文書へ重複させない。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: 直接承認。2026-09-11に、exp004のbatch size 16でのOOMを別実験のbatch size 8で再実行する推奨へ、ユーザーの「その方針で進めてください」で実験化、実装、Kaggle Notebook実行が承認された。
- 移行元backlog: `embryo_holdout_baseline`。候補はexp004作成時に移行済みであり、本実験はその同じ系譜を保持する直接承認済みの派生実験である。新しいbacklog候補の移行や削除は行わない。
- 対応する上位仮説: `HYP-20260910-14`。
- 上位仮説のうちこの実験が検証する範囲: 評価する胚が学習、重み選択、推論条件選択に含まれない基準予測を、学習batch size 8の固定構成と固定した公式評価器で全199動画について作れるか。
- この実験だけで上位仮説を判断できるか: いいえ。
- 上位仮説の判断に残る検証: `group_error_readout`、`oracle_stage_limits`、`graph_checkpoint`と後続手法の比較。単一構成の基準予測だけではモデル順位差や未知胚への一般化全体を判断しない。
- 親実験: [`exp004_embryo_holdout_baseline`](../exp004_embryo_holdout_baseline/)。同実験の分割、モデル、学習、内部の重み選択、推論、候補保存、評価処理とfold間のGPU memory解放を継承する。
- 根拠 / 一次資料 / 参照実装: [exp004の実験契約](../exp004_embryo_holdout_baseline/requirements.md)、[exp004の設定](../exp004_embryo_holdout_baseline/config.yaml)、[exp004の学習実装](../exp004_embryo_holdout_baseline/exp004_embryo_holdout_baseline_train.py)、[exp004の結果](../exp004_embryo_holdout_baseline/result.md)、[exp004のmetrics](../exp004_embryo_holdout_baseline/metrics.json)、[exp003の結果](../exp003_official_metric_audit/result.md)。
- 固定source commit: `075fc5f5a52d11077f9dc2b074644618f26939e2`。公式評価器のファイルSHAは[`exp003`のsource manifest](../exp003_official_metric_audit/assets/source_manifest.json)と照合する。
- 固定するもの: exp004のsource、2方向の胚holdout、`TemporalUNet3D_SimpleNodeTransformer`、random initialization、seed 42、3 epochs、AdamW、learning rate 0.0001、workers 8、window size 2、downsample `(1,4,4)`、モデル幅、augmentation、損失、内部の重み選択指標、推論batch size 4、検出と接続の閾値、検出時augmentationによる予測集約、整数線形計画の費用、候補保存、公式評価を維持する。
- 変更するもの: 学習batch sizeだけを16から8へ下げる。これに伴う1 epochあたりのoptimizer step数の増加とminibatch構成の変化は、本実験の承認済み差分として扱う。
- 最小の反証可能な検証: 2方向とも同一構成で学習し、外側の評価胚の全動画を推論する。199動画の重複と欠落、学習・重み選択への評価胚混入、候補cacheと最終graphの対応、保存graphからの公式評価再計算を検査する。
- 成功条件: 全199動画に学習から除いた胚の予測が各1件あり、欠落と未解決の実行失敗が0である。予測、候補、学習来歴を追跡でき、公式の動画別成分、胚別集計、全体集計を保存graphから再計算できる。低い得点だけを失敗条件にはしない。
- 停止条件: 胚の混入、対象動画の欠落、候補と最終graphの対応不能、再採点不一致、契約内で解消できないメモリ不足または12時間制約超過があれば、影響する方向と依存分析を停止する。両方向が成立するまで基準予測全体の完成としない。
- 実行しないこと: gradient accumulation、学習batch size 8以外への自動変更、両胚で学習した既存重みの流用、動画ランダム分割を主評価にすること、評価胚の正解を使うepoch・閾値・整数線形計画の費用選択、簡略評価器への置換、外部重み、追加refit、モデル構造や損失の変更、hidden testへの提出、competition submission、Public LB取得を行わない。
- 未決事項: なし。
- backlog記録から解釈を変更した箇所とユーザー承認: 元候補の分割・出力・評価契約は維持する。exp004の固定batch size 16契約からbatch size 8へ変えることを2026-09-11にユーザーが直接承認した。

## 観測事実と仮定

- 実測済みの事実: metadata上の199動画は胚`44b6`が71動画、`6bba`が128動画。exp002は両胚を含むsample単位splitで3 epochsを完走したため、その重みを独立した胚評価には使えない。
- 実測済みの事実: exp003は人工例9件と保存済み予測4動画で固定公式処理を照合した。実予測4動画では一致したが、学習から除いた胚の予測ではない。
- 実測済みの事実: exp004は学習batch size 16で3回Kaggle実行した。version 3ではfold 0のsmoke完了後にmodel参照を削除しCUDA cacheを空にしても、fold 1の最初のbackwardで2.09 GiBを追加確保できずOOMとなった。checkpointは0個で、学習・推論の公式評価は未実行である。
- 利用する保存済み生成物: モデル重みと予測は再利用しない。固定sourceと固定公式評価器だけをコピーし、コピー後にmanifestのSHAを照合する。
- 新規に記録する生成物: 2個のcheckpoint、分割manifest、モデルmanifest、動画別の予測graph、候補cache、予測manifest、動画別・胚別・全体の公式評価結果。それぞれの内容SHAと実行環境を記録する。
- Assumption: 学習batch sizeを8へ下げれば2foldともT4でbackwardできる。optimizer step数とminibatch構成がexp004から変わるため、精度と総所要時間は未実測である。

## 判断履歴

- 2026-09-11: ユーザーの「すべて推奨でいいです」で、公式評価照合、胚を分けた基準予測、誤差と段階別上限の分析、接続損失の比較という順序と無料枠内の方針が承認された。
- 2026-09-11: exp003の完了後、基準予測作成を`embryo_holdout_baseline`としてbacklogへ追加した。候補段階では採番、実装、Kaggle実行を行わなかった。
- 2026-09-11: ユーザーの「embryo_holdout_baselineを実装してください」で`exp004_embryo_holdout_baseline`への実験化と実装が承認された。competition submissionの承認とは扱わない。
- 2026-09-11: exp004のbatch size 16がfold 1 smokeの最初のbackwardでOOMとなったため、別実験でbatch size 8へ変更する方針を提示した。ユーザーの「その方針で進めてください」で`exp005_embryo_holdout_batch8`の実装とKaggle実行が承認された。competition submissionの承認とは扱わない。

## 手法契約

実装区分は`docs/glossary.md`に定義したこのリポジトリ内の管理用ラベルを使う。処理内容と省略点を先に記録する。

- 依頼原文: exp004の固定条件の失敗を残し、batch size 8へ変更する別実験を作る推奨に対する「その方針で進めてください」。
- 期待する成果: batch size 8で2foldの3 epochs学習を完走し、学習から除いた胚を予測した全199動画の基準予測、検出・接続候補、公式評価結果を揃える。
- input: 外側の学習胚のZarr画像とGEFF注釈。外側の評価胚は推論時に画像と利用可能なmetadataだけを読み、GEFFは予測を固定した後の採点だけに使う。
- target / objective: exp004と同じ細胞中心検出と隣接時刻間の接続予測。新しい教師や損失は導入しない。
- output: 2個の選択済みcheckpoint、分割一覧、199動画の予測graph、検出中心と検出得点、全接続score行列、実在node対のmask、閾値通過mask、最適化へ渡すmask、最終選択mask、動画別・胚別・全体の公式評価結果。
- loss: exp004の重み付き検出損失と接続損失をそのまま使う。検出損失の重み1.0、検出負例の重み0.01を保つ。未注釈の子を扱う損失変更は`partial_edge_mask`へ分ける。
- decode: exp004と同じ検出時augmentationによるlogitの平均、中心抽出、接続のsoftmax、閾値、整数線形計画によるgraph選択を維持する。検出閾値0.99、接続閾値0.5、推論batch size 4、各費用を固定する。
- context unit: モデル入力は2時刻の3D画像窓、推論と保存は1動画、外側の分割は1胚。動画間の座標共有や絶対時刻の共有は仮定しない。
- 実装区分: `faithful`。参照モデルの教師、構造、損失、推論を保つ。batch size 8は承認済みの学習parameter差分であり、元のbatch size 16と同一のoptimization trajectoryは主張しない。
- 省略する機構と理由: なし。候補cacheの追加は予測値の記録であり、decodeへ介入しない。
- proxyで検証できない主張: N/A。
- proxyの場合のユーザー承認: N/A。
- この実験が支持 / 棄却できる主張: 学習batch size 8の固定構成で、胚を分けた全199動画の基準予測と再採点可能な中間候補を作れるか。
- この実験では判断できない主張: 別のモデルや損失の優劣、hidden test全件の時間内完走、Public LB、未知胚一般化全体。

## 実装方法

- アプローチ: exp004のNotebookを継承し、学習batch sizeを8へ変更する。学習Notebookで2方向の分割を作り、各外側学習胚の中だけでseed 0の90%/10%内部分割を作る。両方向をscratchから3 epochs学習し、内部の`edge_accuracy_times_node_recall`で各1個のcheckpointを選ぶ。推論Notebookはcheckpointを胚ごとに固定し、外側評価胚だけを予測して固定公式処理で採点する。
- inputの実装箇所と変換: `exp005_embryo_holdout_batch8_train.py`がtrain directoryのZarr/GEFF組を動的列挙し、sample名の最初の`_`より前を`embryo_id`とする。各学習胚のsample名をsortし、`random.Random(0)`でshuffleした先頭の`max(1, n // 10)`を内部の重み選択へ回す。
- target / objectiveの構築箇所: 固定sourceの`official_source/scripts/train_unet_transformer.py`をそのまま呼び、教師作成と損失を変更しない。
- outputの生成箇所と表現: 学習Notebookは`artifacts/models/fold_<n>/`と`artifacts/splits.json`へ保存する。推論Notebookは`artifacts/predictions/<sample>.geff`、`artifacts/candidates/<sample>.npz`、`artifacts/prediction_manifest.json`、`artifacts/per_sample_metrics.json`、`artifacts/official_metric_summary.json`へ動画単位で保存する。
- lossの実装箇所: 固定sourceの学習関数へexp004と同じ引数を渡し、`batch_size`だけ8とする。外側評価胚のGEFFを学習関数へ渡さない。
- decode / postprocessの実装箇所: 推論Notebook内に固定sourceの`predict_video`と同じ処理を展開し、検出scoreと全接続score行列の保存だけを加える。graph構築と整数線形計画の係数は固定sourceと一致させる。
- context unitを保つ処理箇所: 分割検査は胚単位、モデル処理は2時刻窓、候補とgraphの保存・解放は動画単位で行う。
- 変更するファイル / component: `config.yaml`、`requirements.md`、train/inferenceのJupytext sourceとNotebook、`tests/test_contract.py`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json`。固定`official_source/`は変更しない。
- 固定事項を保つ確認方法: exp004 configとの比較テストで`model.training.batch_size`と`runtime.batch_size`以外が同じこと、固定source manifestのSHA照合、Notebook内の引数、2foldの期待件数、評価胚非混入、submission処理不在を検査する。
- 参照sourceとの一致を確認するテスト: source manifest全件のSHA、学習モデル・損失・decode設定、検出logitの平均、softmaxの軸、厳密な閾値比較、整数線形計画の費用を検査する。
- 承認済み差分を確認するテスト: 学習batch sizeがconfigの2箇所とも8でexp004だけが16であること、`44b6 -> 6bba`と`6bba -> 44b6`の2方向、期待件数64/7/128と116/12/71、全edge scoreと4種類のmask、公式の`evaluate_pairs`と`summarise`の利用、全199動画の一意性を検査する。

## 探索幅とpivot判定

- 変更class: `parameter`。このリポジトリ内では、OOMを避けるため学習batch sizeだけを16から8へ変更する実行parameter差分として管理する。
- 同じ親 / familyで連続した小改善実験数: 1。exp004の実行不能を解消する最初のparameter差分であり、精度改善の連続調整ではない。
- positiveなoracle headroom / coverage / 誤差非相関性: 未測定。この実験の候補cacheを`oracle_stage_limits`へ渡して測る。
- 比較したtarget、output、decode、context unitを変える案: 今回は比較基準を固定するため変更しない。division tripletを直接予測する案は別候補であり、同時導入しない。
- 小改善の継続またはpivotを選ぶ根拠: exp004はbatch size 16のGPU memoryだけで失敗し、分割、model、loss、decodeの妥当性は未評価である。batch sizeを半分にする最小差分で実行可能性を再検証する。
- `kaggle-idea-forge`の実行要否と根拠: 不要。同じfamilyのparameter変更は1件目で、positiveなoracle headroom後にend-to-end改善が得られない状況でもない。

## 最小の反証可能な検証

| 外側の学習胚 | 勾配更新用動画数 | 内部の重み選択用動画数 | 外側の評価胚 | 外側の評価動画数 |
| --- | ---: | ---: | --- | ---: |
| `44b6` | 64 | 7 | `6bba` | 128 |
| `6bba` | 116 | 12 | `44b6` | 71 |

- 分割の検査: 199動画の重複と欠落がなく、各方向で外側の評価胚が勾配更新と重み選択のどちらにも含まれないことを確認する。smokeは各方向の勾配更新用動画から名前順で最初の1件を使う。
- variant / config / fold / booster数: 1構成、外側2fold、各3 epochs、選択済みモデル2個。追加seedとboosterは0。学習胚全件での追加refitは行わない。
- control再学習: batch size 8の胚holdoutモデル2個だけを学習する。exp004のbatch size 16とexp002の両胚混在splitは再学習しない。既存重みは評価胚の独立性を満たさないため使用しない。
- 想定runtime / resource: KaggleのT4 2基、学習batch size 8、internet無効、1Notebook最大12時間。学習と推論を別Notebookにし、学習は各foldの2 iteration smoke、推論は各foldの最初の1動画から保守的に総時間を見積もる。gateを超える場合は縮小せず停止し、動画単位で得られた生成物を保持する。
- 評価処理: 固定sourceの`evaluate`、`per_sample_metrics`、`summarise`を使用し、対応付け距離7 µmと各動画の物理scaleを使う。全199動画の成分をまとめて`summarise`へ渡した値を全体値とし、2胚の得点の単純平均で置き換えない。

## 再現性・リスク

- seed policy: exp002と同じseed 42、DataLoader generator、内部split seed 0を維持する。foldごとに同じseed 42からscratchで初期化する。
- stochastic 処理の有無: random initialization、DataLoader shuffle、brightness/flip augmentation、CUDA kernelがある。
- stochastic feature generation / augmentation / seed bagging の有無: augmentationはある。seed baggingはない。固定sourceがitemごとに引数なしのNumPy `default_rng`を作るため、byte-level deterministicとは扱わない。
- 並列処理と乱数の関係: workers 8と固定generatorを維持するが、augmentationの乱数は完全固定されない。この制約をmodel manifestへ記録する。
- CPU/GPU runtime と deterministic flags: Kaggle T4 2基、DataParallel、internet無効。`PYTORCH_ALLOC_CONF=expandable_segments:True`をTorch import前に設定し、GPU名、kernel version、Notebook実行時間、memoryを記録する。
- train cache / test feature regeneration の SHA 記録方針: candidate NPZのZIP metadataではなく、array key、dtype、shape、連続bytesから動画別content SHAを計算し、全動画のmanifest SHAを集約証拠にする。
- model manifest / prediction / submission SHA 記録方針: 分割、各foldのcheckpoint/config/manifest、各動画のgraph tree、候補content、全体prediction manifestのSHAを記録する。submissionは作らない。
- Kaggle package bootstrap 確認方針: prepare後に正のNotebook、config、metrics、固定sourceとbootstrap ZIPのmanifestが一致することをpush前validatorで確認する。
- リークリスク: 外側評価胚の正解を学習、重み選択、閾値、費用選択に使わない。2胚しかないため、後続で同じ外側結果を繰り返し選択に使うと開発集合になる。
- CV/LB 不一致リスク: この実験はtraining data内の胚を分けたCVでありPublic LBではない。hidden testやLBの値を推定しない。
- ランタイム/メモリリスク: batch size 8でもOOMとなる可能性があり、batch数が増えるため12時間gateを超える可能性もある。全edge scoreは大きくなり得るため、動画ごとに圧縮保存してCPU/GPU memoryを解放し、全特徴tensorを一括保持しない。
- 再現性リスク: augmentationとGPU kernelにより再学習時の重みはbitwise一致しない。固定source、config、split、checkpoint、候補、graph、評価器のSHAを保存し、保存graphからの再採点を検証する。
- 手法忠実性リスク: 候補保存用の推論処理が固定sourceのdecodeからずれる可能性があるため、演算順序、比較演算、座標scale、整数線形計画の入力と費用を静的テストで固定する。
- 過度な縮小 / proxy化リスク: runtime gateまたはmemoryで失敗しても動画数、epoch、解像度、batch size 8を変更しない。別構成が必要なら本実験へ混ぜずユーザーへ確認する。

## 受け入れ基準

- [x] 手法契約の `input / target / output / loss / decode / context unit` が実装前に一意である。
- [x] 実装区分と実験名が実装する処理を正確に表す。
- [x] 直接承認、親実験との差分、固定事項、成功条件、停止条件、実行しないこと、判断履歴を記録した。
- [x] `config.yaml`の`lineage.hypothesis_id`と`lineage.backlog_candidate`が本書と一致する。
- [x] Jupytext round-trip、`validate-exp`、`check-exp`、実験固有testが通る。
- [x] Kaggle T4 2基で2foldのsmokeと3 epochs学習が完走し、checkpoint 2個を保存する。
- [x] 外側評価199動画の予測、候補cache、動画別・胚別・全体の公式評価を保存し、欠落と失敗が0である。
- [x] 保存graphからの再採点が一致し、model、candidate、predictionのSHAとKaggle kernel versionを`metrics.json`へ記録する。
- [x] `HYP-20260910-14`の対応実験へ本実験を追加し、依存候補の基準予測参照を更新する。
- [x] 2026-09-12、結果提示後にユーザーがexp005を閉じるよう明示し、実験statusを`completed`とした。これはモデル採用や今後のvalidation方針の確定を意味しない。

## 2026-09-12 追加診断

- 承認: exp005の低い`6bba`側node recallについて、ユーザーの「それでは進めてください」に基づき、候補検出coverageと胚間の画像分布差を同じ実験内で診断する。
- input: Kaggle inference version 1の固定prediction manifest、0.99通過後の候補cache、動画別公式評価、同じ199件のtrain Zarr/GEFFを使う。modelの再学習と再推論は行わない。
- target / objective: 保存候補が既知GT nodeを7 µm以内で回収する割合、最終graphまでに失うnode recall、候補数・検出score、決定的にsamplingした画像輝度統計を動画別・胚別に測る。
- output: 動画別CSV、胚別・全体summary JSON、比較plot、入力と出力のSHA、`metrics.json.diagnostics.candidate_image_readout`。
- loss: なし。
- decode / 推論方法: 保存済み候補と最終graphを変更せず診断する。GTは回収率計算にだけ使い、閾値、checkpoint、graph費用の選択には使わない。
- context unit: 199動画を1動画ずつ処理し、集計は胚別と全体で行う。画像統計は各動画の固定3時刻と固定空間strideのsampleであり、全voxel統計とは呼ばない。
- 実装区分: 予測を変更しない診断。既存の`faithful`な予測契約は維持し、追加診断を精度改善として扱わない。
- 制約: 既存cacheは検出確率0.99を厳密に超えたlocal maximumだけを含むため、0.965の回収率は測れない。0.965比較は固定checkpointから検出を再実行する別の比較として残す。
- 成功条件: 全199動画の候補cacheとGTを一意に対応させ、候補node recallが最終graph node recall以上であることを検査し、公式の胚別最終node recallを再現する。欠落・重複・NaN件数とsampling範囲を保存する。
- 停止条件: prediction、candidate、GTの対象集合または来歴が一致しない場合は実行を停止し、推測で補完しない。
