# exp001_temporal_unet3d_baseline セッションノート

## 目的

主催者公開 TemporalUNet3D と SimpleNodeTransformer を3 epochs scratch trainingし、そのcheckpointだけでhidden testのtracking graph提出を生成できるか検証する。

## 現在の作業

- 作業内容: private train Notebook version 3まで実行し、固定batch size 16のsmoke OOMを確認した。Kaggle outputを回収し、証拠を記録中。
- ブロック要因: 実験契約上、smoke OOM時は停止し、同じ実験内でbatch size、model幅、downsampleなどを変更しない。
- 次: batch size 8などの変更を別実験として行うか、ユーザー判断を得る。checkpointがないためinferenceは開始しない。

## GPU cost guard

- active variant: seed 42の公式3-epoch設定1条件。
- model config: 1。
- fold: split 0のみ。
- booster: 0。
- control再学習: なし。
- 公開checkpointのwarm start: なし。
- runtime gate: 同じsourceと固定hyperparameterのsmokeからfull runを推定し、11時間超ならfull trainingを開始しない。

## 変更点

- 2026-09-09: make new-exp EXP=exp001_temporal_unet3d_baseline で雛形を作成した。task commandは環境にないため、AGENTS.mdに従ってMakefileを使用した。
- 2026-09-09: backlog/official_temporal_unet3d_train_submission_baseline.md の契約と判断履歴をrequirements.mdへ移行した。
- 2026-09-09: GitHub commit 075fc5f5a52d11077f9dc2b074644618f26939e2 を取得し、学習・推論に必要なBSD-3-Clause sourceとfile SHAをofficial_sourceへ固定した。
- 2026-09-09: apply_patchで同梱したdataspec.pyだけ末尾改行が追加された。実行内容は不変で、SOURCE.jsonにvendored SHA、元commitのbyte SHA、正規化内容を併記した。
- 2026-09-09: 公開Kaggle inference Notebookのmetadataとsourceを確認した。offline wheelsは thibautgoldsborough/cellmot-baseline-artifacts、推論条件はdet threshold 0.99、UNet batch size 4、ILP有効だった。
- 2026-09-09: artifacts dataset内のtraining sourceは固定commitよりsplit fallbackが古かったため、runtime sourceには使わない。datasetはoffline wheelsだけに限定する。
- 2026-09-09: make validate-exp、make check-exp、make test-expを実行し、strict validation、Ruff、実験固有7 testが通過した。
- 2026-09-09: train/inference packageを外部送信なし、run_on_push falseで生成し、T4、internet無効、competition source、dependency dataset、inferenceのtrain kernel sourceを確認した。
- 2026-09-09: run_on_push付きpackage準備は、外部Kaggleへの送信・実行の明示承認不足として環境に拒否された。迂回せず、pushと実行を停止した。
- 2026-09-09: ユーザーが「Kaggleで実行してください」とprivate Notebookのpush・GPU実行を明示承認した。competition submissionの承認とは扱わない。
- 2026-09-09 22:43:23 JST: push前にmake validate-exp、make check-exp、make test-expを再実行し、strict validation、Ruff、実験固有7 testが通過した。
- 2026-09-09 22:43:23 JST: train metadataはenable_gpu=true、enable_tpu=false、machine_shape=NvidiaTeslaT4、internet無効、run_on_push=true。GPU quotaは30.00h残、refreshは2026-09-12T00:00:00。最大11hのruntime gateに対して十分なため、1 variant・1 model config・split 0・booster 0・control再学習なしの学習を実行すると判断した。
- 2026-09-09: 最初のpushは直前に更新したmetrics.jsonと生成packageの差分をvalidatorが検出して停止した。Kaggle実行は開始されなかった。package再生成後のpushでtrain kernel version 1（id_no 133724480）を起動し、Kaggle側metadataでもprivate、internet無効、NvidiaTeslaT4を確認した。
- 2026-09-09: train kernel version 1は学習開始前に失敗した。Notebook冒頭でimport済みのNumPy 2.0.2をoffline wheel導入が2.4.6へ置換したため、同一kernel内でNumPy moduleが混在し、SciPy importが`AttributeError: module 'numpy._core._multiarray_umath' has no attribute '_blas_supports_fpe'`になった。checkpointは生成されていない。
- 2026-09-09: 学習条件を変更せず、train/inference両NotebookでNumPy、SciPy、Torch、Pandasをoffline pip install完了後に初めてimportするよう修正した。依存導入より先に数値stackをimportしない回帰testを追加し、version 2で再実行する。
- 2026-09-09: version 2 push直前のGPU quotaは29.90h残、refreshは2026-09-12T00:00:00。version 1の消費は0.10hで、最大11hのruntime gateに対して十分なため、同じ1条件を再実行すると判断した。
- 2026-09-09: train kernel version 2も学習前に停止した。Papermill/Kaggle kernel自体がNumPy 2.0.2を事前loadしており、Notebook内のimportを後ろへ移してもpip後の観測versionは2.0.2のままだった。disk上だけ2.4.6へ置換されたため、SciPy経由で`ImportError: cannot import name '_center' from 'numpy._core.umath'`が発生した。checkpointは生成されていない。
- 2026-09-09: wheel METADATAを確認するとNumPy更新は`imagecodecs==2026.6.26`の`numpy>=2.1`が原因だった。固定sourceの学習・推論pathにimagecodecs importはなく、Kaggle imageにはその他の数値依存が存在するため、必要な未導入packageだけをversion固定・`--no-deps`で入れ、NumPy/SciPy/imagecodecsを置換しない方式へ修正した。学習条件とsourceは不変。
- 2026-09-09: version 3 push直前のGPU quotaは29.82h残、refreshは2026-09-12T00:00:00。version 1・2の合計消費は0.18hで、最大11hのruntime gateに対して十分なため、同じ1条件を再実行すると判断した。
- 2026-09-09: train kernel version 3でoffline依存を正常導入し、NumPy 2.0.2、SciPy 1.16.3、Tesla T4 x2、固定source 14 files、train/validation 180/19、split SHA `fca45709656ad0f0900fc5fcbf0b7f4aad3abb7d547fafcf4e022af6de1aa68e`を確認した。
- 2026-09-09: version 3のsmokeは1 sampleからtrain/test各50 windowsを読み込み、2,076,706 parameters、DataParallelの各GPU batch 8で開始したが、最初のbackwardでCUDA OOMになった。GPU 0は14.56 GiB中1.75 GiB freeで、追加1.84 GiBを確保できなかった。Notebook log上の終了時刻は起動後367.043秒。full 3 epochs、checkpoint、inference、submission.csv生成は開始していない。
- 2026-09-09: Kaggle outputを`/tmp/kaggle-output/exp001-temporal-unet3d-baseline-train`へ回収した。dataset index SHAは`cd0eab17e481c9a9956b585230d3865fe6fb80ebd6f85472a47404fa9f9c9a8c`、split SHAは`fca45709656ad0f0900fc5fcbf0b7f4aad3abb7d547fafcf4e022af6de1aa68e`、smoke log SHAは`a8f91dde0de366a7b63f60610217e56926d55ec4380f6cb7af966e1fdd5541d4`。model checkpointとmodel_manifestは存在しない。

## コマンドログ

### 実行済み

```bash
make new-exp EXP=exp001_temporal_unet3d_baseline
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle kernels pull thibautgoldsborough/unet-baseline-inference-submission -p /tmp/biohub-official.pVStUp/kaggle-inference -m
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle datasets files thibautgoldsborough/cellmot-baseline-artifacts --page-size 200
```

### static validation実行済み

```bash
make validate-exp EXP=exp001_temporal_unet3d_baseline
make check-exp EXP=exp001_temporal_unet3d_baseline
make test-exp EXP=exp001_temporal_unet3d_baseline
```

実際にpackage生成へ使用したコマンドは、外部実行を開始しない次の2件。

```bash
make prepare-kaggle-notebooks EXP=exp001_temporal_unet3d_baseline EXTRA_ARGS="--notebook train"
make prepare-kaggle-notebooks EXP=exp001_temporal_unet3d_baseline EXTRA_ARGS="--notebook inference"
```

外部実行の明示承認不足により拒否され、packageやKaggle側を変更しなかったコマンドは `make prepare-kaggle-notebooks EXP=exp001_temporal_unet3d_baseline EXTRA_ARGS="--notebook train --run-on-push"`。

### Kaggle実行予定

push直前にGPU quotaを確認し、判断をこのファイルへ記録する。

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle quota --format json
make push-kaggle-train EXP=exp001_temporal_unet3d_baseline
make kaggle-logs KERNEL=kentookumura/exp001-temporal-unet3d-baseline-train
make kaggle-output KERNEL=kentookumura/exp001-temporal-unet3d-baseline-train OUT=/tmp/kaggle-output/exp001-temporal-unet3d-baseline-train
```

train output取得後にcheckpointとmanifestを確認し、inference packageへtrain kernel sourceを接続する。submission commandはユーザーの別途明示承認まで実行しない。

## 次のアクション

1. batch size 8など、公式設定からの変更を別実験として試すかユーザー判断を得る。
2. 別実験を承認された場合は、固定source、3 epochs、split 0を維持し、変更点とoptimizerへの影響を新しい実験契約へ明記する。
3. train checkpointが得られた場合だけinference Notebookをrun-on-pushで生成・実行する。
