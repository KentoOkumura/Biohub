# exp002_unet3d_expandable_segments 要件と実装方法

この文書を、実装前の契約、実装方法、受け入れ条件の正とする。実行中の進捗は`SESSION_NOTES.md`へ記録し、この文書へ重複させない。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: `設計可能・実験化未承認`のbacklog候補からの実験化。2026-09-10のユーザーメッセージ「exp002を実装してください」で承認された。
- 移行元backlog: `backlog/temporal_unet3d_expandable_segments_retry.md`。内容の移行確認後、元ファイルと未着手行を`kaggle-strategy`の手順で削除する。
- 対応する上位仮説: `HYP-20260909-01`。
- 親実験: `experiments/exp001_temporal_unet3d_baseline`。
- 未決事項: なし。
- backlog記録から解釈を変更した箇所とユーザー承認: 実験内容の変更はない。Kaggleの50文字kernel slug制限を満たすため、実験ディレクトリ名だけを候補名より短い`exp002_unet3d_expandable_segments`とした。
- version 2の追加承認: 2026-09-10のユーザーメッセージ「このまま最後まで実行してみてください」により、batch size 16と全学習条件を維持し、実行開始判定のruntime gateだけを11時間からKaggle上限と同じ12時間へ変更してfull 3 epochsを試すことが承認された。

## 移行した候補契約

- 候補名: `temporal_unet3d_expandable_segments_retry`
- 状態: `設計可能・実験化未承認`
- 対応する上位仮説: `HYP-20260909-01`
- 関連する上位仮説: なし
- 作成日: 2026-09-10
- 最終更新日: 2026-09-10
- 依頼原文: 「exp002のバックログを作成してください」。直前に提案した、PyTorchのallocator設定だけを変更してbatch size 16を維持する後続実験を指す。
- 期待する成果: `PYTORCH_ALLOC_CONF=expandable_segments:True`をCUDA利用前に設定し、主催者公開モデル、batch size 16、split 0、3 epochsを変えずにT4 x2で学習を完走してcheckpointを生成する。checkpointが得られた場合だけ別Notebookでhidden testを推論し、提出形式を検証する。
- 親実験 / 比較対象: `experiments/exp001_temporal_unet3d_baseline`。比較対象は同実験のKaggle train kernel version 3で観測したsmoke backwardのCUDA OOM。
- 優先度: P0
- 優先度の理由: checkpoint、CV、LBがまだなく、最初の3D U-Net比較基準の確立が全後続実験の先行条件である。OOM時に1.44 GiBがPyTorchに予約済みだが未割当で、要求量1.84 GiBに対してfreeは1.75 GiBだったため、学習条件を変えないallocator設定を最初に反証する価値がある。
- `backlog/KAGGLE_DIRECTION.md` の対応箇所: 「検証中の仮説」の`HYP-20260909-01`行と「未着手バックログ」の`temporal_unet3d_expandable_segments_retry`行。

## 観測事実と根拠

- 実測済みの事実: exp001 version 3はKaggleのTesla T4 x2、internet無効でoffline依存導入、固定source 14 filesの検証、199 sampleのsplit 0作成まで成功した。2,076,706 parameters、batch size 16、DataParallelで各GPU batch 8としてsmokeを開始したが、最初のbackwardで1.84 GiBを追加確保できずOOMとなった。GPU 0は14.56 GiB中1.75 GiB free、11.17 GiB allocated、1.44 GiB reserved but unallocatedだった。checkpointとmodel manifestは生成されていない。
- 根拠ファイル / 一次資料: [`experiments/exp001_temporal_unet3d_baseline/result.md`](../exp001_temporal_unet3d_baseline/result.md)、[`experiments/exp001_temporal_unet3d_baseline/metrics.json`](../exp001_temporal_unet3d_baseline/metrics.json)、[`experiments/exp001_temporal_unet3d_baseline/SESSION_NOTES.md`](../exp001_temporal_unet3d_baseline/SESSION_NOTES.md)、[Kaggle train Notebook](https://www.kaggle.com/code/kentookumura/exp001-temporal-unet3d-baseline-train)。
- 利用する保存済み生成物とSHA: exp001のsplit SHAは`fca45709656ad0f0900fc5fcbf0b7f4aad3abb7d547fafcf4e022af6de1aa68e`、dataset index SHAは`cd0eab17e481c9a9956b585230d3865fe6fb80ebd6f85472a47404fa9f9c9a8c`、smoke log SHAは`a8f91dde0de366a7b63f60610217e56926d55ec4380f6cb7af966e1fdd5541d4`。checkpointは存在しないため再利用しない。親実験のNotebook、固定source、configをコピー元にする。
- 仮定: `PYTORCH_ALLOC_CONF=expandable_segments:True`によりreserved memoryの断片化が緩和され、モデルやbatch sizeを変えずに1回のbackwardを完了できる可能性がある。OOMが純粋な容量不足である場合は解消しない。

## この候補が直接検証する仮説と範囲

- 上位仮説のうちこの候補が検証する範囲: 現行Kaggle T4 x2環境で、主催者公開モデルと公式batch size 16を保持した3 epochsのscratch trainingを完走し、推論可能なcheckpointを生成できるか。
- この候補の具体的な仮説: CUDA利用前に`PYTORCH_ALLOC_CONF=expandable_segments:True`を設定すると、exp001 version 3で失敗した同一smokeのbackwardがOOMなく完了し、runtime gate通過後にsplit 0の3 epochsを11時間以内で完走できる。
- 仮説が正しい場合に期待する観測: 同一smokeの2 iterationsが完了し、peak GPU memoryとfull run予測時間を記録できる。予測時間が11時間以内なら3 epochsが完走し、best checkpoint、model config、model manifest、validation値、各生成物SHAが得られる。
- 仮説を棄却する観測: allocator設定が反映されない、同一smokeが再度OOMになる、full run予測が11時間を超える、3 epochs中にOOMまたは12時間制限へ到達する、またはcheckpointとmodel manifestを保存できない。
- この候補だけで上位仮説を判断できるか: いいえ
- 上位仮説の判断に残る検証: checkpoint取得後のhidden test推論、提出形式検証、ユーザー承認後のsubmission、Public LB、checkpointとsubmissionのSHA記録。

## 手法契約

実装区分は`docs/glossary.md`に定義したこのリポジトリ内の管理用語を使う。処理内容と省略点を先に具体的に記録する。

- input: Kaggle competition trainの4次元画像`(T,Z,Y,X)`と正解tracking graphのGEFF。推論へ進める場合は、この候補の学習Notebookが生成したcheckpointとhidden test画像。
- target / objective: cell-center detection mapと、隣接timepointの検出node間edgeを同時に学習する。
- output: 学習時はbest checkpoint、model config、model manifest、split JSON、train/validation metrics、実行時間、memory記録。推論時はnodeとedgeから成るtracking graphと`submission.csv`。
- loss: exp001と同じ重み付きdetection lossと、疎なannotationを考慮したedge lossの和。loss weightは変更しない。
- decode / 推論方法: exp001と同じ主催者公開処理を使い、`TemporalUNet3D`のdetection mapからnode候補を抽出し、`SimpleNodeTransformer`で隣接timepointのedgeをscoreしてtracking graphへ変換する。
- 処理単位: window size 2の連続frameをbatchとして学習し、推論はdataset単位でtracking graphを生成する。
- 実装区分: `staged-faithful`。同じsourceと学習条件のsmokeを先に実行し、memoryと時間のgateを通った場合だけ3 epochsと推論へ進む。参照手法からの差分はallocator設定と証拠記録だけ。

## 実装方法

- 設定と系譜の正本は`config.yaml`、学習と推論の正の編集対象は同名のJupytext percent形式`.py`から生成する`.ipynb`とする。
- train Notebookは`config.yaml`の`runtime.environment.PYTORCH_ALLOC_CONF`を読み、明示的な`import torch`より前に環境変数へ設定する。既にTorchがimport済みなら停止し、設定順序違反を隠さない。
- smokeの成功時と失敗時に、allocator設定値とGPUごとのpeak allocated/reserved memoryを`metrics.json`へ保存する。
- inference Notebookはexp002 train kernelの`model_manifest.json`だけをcheckpoint入口にし、exp001のcheckpointや公開checkpointを使わない。
- copied testはsource commit、model、batch size、split、loss、3 epochs、推論条件がexp001と同一であることと、allocator設定だけが追加されたことを検査する。

### Version 2の実行条件変更

- 変更するもの: version 1のsmoke後、full trainingを開始するruntime gateを11時間から12時間へ変更する。
- 固定するもの: batch size 16、3 epochs、モデル、split、seed、optimizer、loss、augmentation、T4 x2、allocator設定を含むその他の条件。
- 解釈: version 1で「11時間以内」という複合仮説は不成立だった事実を変更しない。version 2はKaggleの12時間上限内で実際に完走できるかを直接確認する追加実行である。予測11.706時間に対して起動・保存時間の余裕が小さいため、timeoutは許容した失敗条件とする。

### 親実験からの差分

- 変更するもの: 実験の`config.yaml`にallocator設定を明示し、NotebookのTorch importおよび最初のCUDA allocationより前に`PYTORCH_ALLOC_CONF=expandable_segments:True`を設定して値をlogへ残す。
- 固定するもの: GitHub source commit `075fc5f5a52d11077f9dc2b074644618f26939e2`、random initialization、seed 42、3 epochs、split 0、90%/10%のseed-0 sample split、AdamW、learning rate `1e-4`、batch size 16、num workers 8、window size 2、downsample `(1,4,4)`、UNet output channels 32、layers `(32,64,128)`、gradient checkpointing、augmentation、loss、checkpoint selection、T4 x2 DataParallel、internet無効、offline dependencyの固定package一覧。
- 再利用するコード / config / 生成物: exp001のtrain/inference Jupytext sourceとNotebook、固定source、source manifest、tests、Kaggle package生成手順。exp001にcheckpointはないためmodel artifactは再利用しない。
- 新しく作るもの: 実験化承認後の新しい実験ディレクトリ、allocator設定を含むtrain Notebook、Kaggle train/inference package、実行証拠、成功時のcheckpointと提出候補。

## 探索幅とpivot判定

- 変更class: `parameter`。学習処理やモデルを変えず、CUDA allocatorのruntime設定だけを変更する。
- 同じ親 / familyで連続した小改善実験数: 1。
- positiveなoracle headroom / coverage / 誤差非相関性: 未測定。exp001は最初のbackwardでOOMとなり、checkpoint、CV、LBはない。
- 比較したtarget、output、decode、context unitを変える案: 今回はいずれも変更しない。batch size 8、AMP、model縮小、downsample変更はmemory対策候補だが、allocator設定だけの効果を分離できなくなるため後続候補へ分ける。
- 小改善を継続する根拠: exp001のOOM時には1.44 GiBがreserved but unallocatedで、追加要求1.84 GiBに対してfreeが1.75 GiBだったため、学習条件を変えないallocator設定を1回だけ反証する価値がある。
- `kaggle-idea-forge`の実行要否: 不要。同じfamilyの小変更は本実験が1件目であり、baseline未成立の直接的なruntime blockerを検証する。

## 最小の反証可能な検証

- 検証方法: exp001と同じfirst sorted training sample、1 epoch、最大2 iterationsのsmokeをT4 x2で実行する。allocator設定値、GPU名、peak allocated/reserved memory、成功またはOOMを記録する。smoke完了後の保守的なfull run予測が11時間以内の場合だけ、同じ設定でsplit 0の3 epochsを実行する。checkpoint取得時だけ推論Notebookを実行し、`submission.csv`を提出前検証する。
- variant / config / fold / booster数: allocator有効の1 variant、model config 1、split 0のみ、booster 0。
- control再学習: なし。親実験のallocator無効・batch size 16のsmoke OOMが比較対象であり、同条件を再実行してGPU時間を消費しない。
- 想定runtime / resource: Kaggle T4 x2、internet無効。smokeは数分、full trainingと推論は各12時間以内。full trainingはsmoke投影が11時間以内のときだけ開始する。

## 成功条件と停止条件

- primary指標: allocator設定だけでbatch size 16のsmoke backwardと2 iterationsを完了できるか。full runへ進んだ場合は3 epochs完走とbest checkpoint生成を次の成立条件にする。
- 成功条件: smokeがOOMなく完了し、full run予測が11時間以内であること。続いて3 epochsを完走し、best checkpoint、model config、model manifest、validation値、Notebook実行時間、各SHAを保存できること。
- 必須guard: allocator設定をTorch importとCUDA allocationより前に設定し、logへ値を残す。exp001とsource/hyperparameter/splitが一致することをtestする。公開checkpointを読み込まない。smoke checkpointを推論に使わない。Kaggle push前にGPU quotaを確認する。実際のcompetition submissionは別途ユーザー承認を得る。
- 成功時の次段階: checkpointだけを入力にしたinference Notebookを実行し、提出形式を検証する。competition submissionと実験の完了・採用判断は自動で確定せず、証拠を示してユーザーへ確認する。
- 失敗時の停止範囲: smoke OOM、allocator設定未反映、11時間超の予測、full training失敗のいずれかで停止する。同じ候補内でbatch size、model幅、downsample、epoch数、loss、splitを変更しない。

## 実行しないこと

- 禁止する代替実装、proxy、同一OOF上の救済探索: batch size 8への変更、AMP追加、gradient accumulation追加、model縮小、downsample拡大、公開checkpointの利用、epoch削減、別GPU種別、追加fold、推論閾値調整、fallback出力をこの候補へ混ぜない。
- 壁打ちで採らなかった案と理由: batch size 8はmemory削減の確度が高いがoptimizer更新ごとのsample数を変えるため、allocator設定だけの反証後に別候補として扱う。AMPは公式training loopに未実装で数値計算を変更するため、この最小差分候補には含めない。model幅やdownsampleの変更はarchitectureまたは入力解像度を変え、最初の公開手法比較基準から離れるため採らない。

## 再現性・リスク

- seed policy: exp001と同じseed 42、seed-0 split、seeded DataLoader generatorを維持する。固定source内のaugmentationはitemごとに引数なしのNumPy `default_rng`を作るため、byte-level deterministicとは扱わない。
- GPU runtime: Kaggle T4 x2、internet無効。`PYTORCH_ALLOC_CONF=expandable_segments:True`をTorch import前に固定し、GPU名、kernel version、Notebook実行時間、allocator設定、memoryを記録する。
- 生成物のSHA: dataset index、split、source manifest、checkpoint、model config、model manifest、test prediction content、submissionを対象に、取得できた段階で`metrics.json`へ記録する。
- Kaggle package: 正のNotebookとconfigから再生成し、bootstrap manifestとkernel metadataの一致をpush前に検証する。

- leakage / validation: exp001と同じsample単位splitは同じembryo由来sampleがtrainとvalidationへ混ざる可能性がある。validation値は実行診断であり、信頼できる汎化性能のprimary CVにはしない。
- hidden test: checkpointが得られても、hidden testのdataset数、shape、node数により推論時間とmemoryが増える可能性がある。test名は実行時に動的列挙する。
- runtime / memory: reserved memoryの断片化が主因でなければallocator設定だけではOOMを解消しない。smoke成功後もfull datasetのvalidationや一時tensorでpeak memoryが増える可能性がある。
- 再現性: allocatorの挙動はKaggle container、PyTorch、CUDA driverへ依存する。kernel version、container image、GPU、環境変数、source/split/checkpoint SHA、Notebook実行時間を記録する。augmentationは完全なbyte-level決定性を保証しない。

## 未決事項

- なし

## 判断履歴

- 2026-09-10: exp001 version 3で依存問題を解消後、公式batch size 16の最初のbackwardがT4 x2でOOMになった。
- 2026-09-10: ユーザーが「exp002のバックログを作成してください」と依頼したため、直前に提案したallocator設定だけの再試行を未着手候補として登録した。バックログ段階ではexp番号を採番しない。
- 2026-09-10: batch size 8は学習条件を変えるため、この候補が再度OOMになった場合に別候補として検討する。
- 2026-09-10: ユーザーが「exp002を実装してください」と依頼し、この候補の実験化と実装を承認した。
- 2026-09-10: version 1はsmokeを完了したが3 epochs予測42,143.044秒が11時間gateを超え、full trainingを開始しなかった。
- 2026-09-10: ユーザーが「このまま最後まで実行してみてください」と依頼し、batch size 16などを維持したままversion 2のruntime gateだけを12時間へ変更してfull trainingを試すことを承認した。
- 2026-09-11: ユーザーが「推論と提出に進んでください」と依頼し、学習済みcheckpointによるinference、提出前検証、competition submissionを承認した。

## 受け入れ基準

- 固定するものを一意に説明できる: はい
- 変更するものを一意に説明できる: はい
- 最小検証と停止条件を一意に説明できる: はい
- 実行しないことを一意に説明できる: はい
- 未決事項が明示されている: はい（なし）
- `config.yaml`と`requirements.md`のlineageが一致する: はい
- allocator設定がTorch importより前にあり、成功・OOM時のmemory証拠を記録する: はい（static test済み）
- Jupytext round-trip、strict validation、Ruff、実験固有testが通る: はい
- Kaggle T4 x2で同一smokeが完了する: はい（version 1で2 iterations完了）
- version 2の12時間gate変更にユーザー承認がある: はい（3 epochs完走、checkpoint生成）
- inferenceとcompetition submissionにユーザー承認がある: はい（2026-09-11、実行前）
