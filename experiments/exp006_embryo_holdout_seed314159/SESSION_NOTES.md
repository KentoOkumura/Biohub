# exp006_embryo_holdout_seed314159 セッションノート

## 目的

exp005と同じ2方向の胚holdout・batch size 8構成をseed 314159で再学習し、公式指標と動画別prediction差から1追加seedの変動と誤りの違いを測る。

## 現在の作業

- 作業内容: Kaggle inference version 1をpushし、199動画の推論とexp005とのseed比較を実行中。
- 実行状態: 2026-09-12T08:54:40+09:00にKaggleで`RUNNING`を確認した。
- 次: 完了後にoutputを取得し、公式指標、prediction差、誤差相関、生成物SHAを検証して記録する。

## 実験化時のGPUコスト計画

- active variant: 1
- model/config: 1
- outer fold: 2
- booster: 0
- 選択済みmodel予定数: 2
- control再学習: なし。exp005の保存済みrunを比較対象にする。
- train見積: version 1 smokeの保守的推定28,963.242秒（約8.05時間）。version 2は9時間gateとし、超える場合は縮小せず停止する。
- Kaggle Notebook上限: train/inferenceそれぞれ12時間。push直前にquotaを再確認する。

## コマンドログ

### 2026-09-11 実行済み

```bash
make new-exp EXP=exp006_embryo_holdout_seed314159
```

標準雛形を作り、`requirements.md`、`config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json`を未実装・未実行の契約へ更新した。train/inference Notebookとテストは実装していない。


実験化後の文書・設定検証を実行した。

```bash
make validate-exp EXP=exp006_embryo_holdout_seed314159
make check-exp EXP=exp006_embryo_holdout_seed314159
make test-exp EXP=exp006_embryo_holdout_seed314159
```

初回はREADME必須見出し、requirements見出し名、または雛形`settings.py`の整形で停止した。文書形式とRuff整形だけを修正して再実行し、`validate-exp`と`check-exp`は通過した。`test-exp`は実装前のため「No experiment-specific tests」として終了した。

### 2026-09-11 実装

exp005のself-contained train/inference Jupytext sourceと固定official sourceを移植した。trainはexperiment ID、kernel ID、seed 314159の一貫性検査、model bundleのseed証拠だけを変更し、2fold、batch size 8、3 epochs、loss、decodeを維持した。inferenceはexp006 train kernelとexp005 inference kernelを入力とし、199動画のexact検出座標、最終選択edge、動画別公式score誤差のPearson/Spearman相関を`artifacts/seed_comparison.json`へ保存する。

親との構成比較では、固定official sourceは`diff -qr`で差分なし。Jupytext sourceはtrainが親889行に対して896行で、6章を維持してseed検査とbundle証拠だけを追加した。inferenceは親1245行に対して1514行で、親の8章をすべて維持し、第8章としてseed比較を追加してmetrics契約を第9章へ移した。

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache JUPYTER_DATA_DIR=/tmp/jupyter-data uv run --extra notebook jupytext --to ipynb experiments/exp006_embryo_holdout_seed314159/exp006_embryo_holdout_seed314159_train.py experiments/exp006_embryo_holdout_seed314159/exp006_embryo_holdout_seed314159_inference.py
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache JUPYTER_DATA_DIR=/tmp/jupyter-data uv run --extra notebook jupytext --to ipynb --test experiments/exp006_embryo_holdout_seed314159/exp006_embryo_holdout_seed314159_train.py experiments/exp006_embryo_holdout_seed314159/exp006_embryo_holdout_seed314159_inference.py
make validate-exp EXP=exp006_embryo_holdout_seed314159
make check-exp EXP=exp006_embryo_holdout_seed314159
make test-exp EXP=exp006_embryo_holdout_seed314159
```

結果はJupytext round-trip、F821、`validate-exp`、`check-exp`が通過し、実験固有testは15件通過した。Kaggle prepare、push、実行は行っていない。

### 2026-09-11 Kaggle実行開始

push前に次を再確認した。

- 確認時刻: 2026-09-11 22:57:07 JST
- Kaggle認証: CLI OAuthおよびlegacy username/keyを確認済み。
- GPU quota: 30.00時間中17.69時間使用、12.31時間残り、表示上の更新時刻は2026-09-12 00:00:00。
- resource: `machine_shape=NvidiaTeslaT4`、GPU有効、TPU無効、internet無効。
- 実行対象: active variant 1、model/config 1、outer fold 2、booster 0、control再学習なし。
- コスト判断: exp005 train実測23046.864秒を根拠とするexp006 train見積6.4〜7時間は残りquota内。縮小版へ置換せず、契約どおり2fold・3 epochsを実行する。
- 比較入力: exp005 inferenceはKaggle上で実行中。exp006 trainは独立しているため先行実行し、exp006 inferenceはexp005成果物とexp006 train成果物の完成確認後に開始する。

```bash
make validate-exp EXP=exp006_embryo_holdout_seed314159
make check-exp EXP=exp006_embryo_holdout_seed314159
make test-exp EXP=exp006_embryo_holdout_seed314159
make prepare-kaggle-notebooks EXP=exp006_embryo_holdout_seed314159 EXTRA_ARGS="--notebook train --run-on-push"
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle quota --format json
```

`validate-exp`、`check-exp`は通過し、実験固有testは15件通過した。生成metadataはkernel ID `kentookumura/exp006-embryo-holdout-seed314159-train`、T4、GPU有効、internet無効、run-on-push有効であることを確認した。

trainを2026-09-11 23:00:34 JSTにpushした。

```bash
make push-kaggle-train EXP=exp006_embryo_holdout_seed314159
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle kernels pull kentookumura/exp006-embryo-holdout-seed314159-train -p /tmp/exp006_train_metadata_v1 -m
```

Kaggle kernel version 1が作成され、pullしたmetadataでも同じkernel ID、`machine_shape=NvidiaTeslaT4`、GPU有効、internet無効を確認した。開始直後のGPU quotaは12.21時間残り。

### 2026-09-11 train version 1のruntime gate停止

```bash
make kaggle-logs KERNEL=kentookumura/exp006-embryo-holdout-seed314159-train
make kaggle-output KERNEL=kentookumura/exp006-embryo-holdout-seed314159-train OUT=experiments/exp006_embryo_holdout_seed314159/artifacts/train_v1
```

- Notebook計測時間は76.328秒。Tesla T4を2基使用し、internetは無効。
- fold 0 smokeは42.728秒、peak allocated 7.004 / 6.013 GiB、推定full時間15,649.072秒。
- fold 1 smokeは33.600秒、peak allocated 7.114 / 6.013 GiB、推定full時間13,314.170秒。
- 合計推定は28,963.242秒（約8.05時間）で、7時間gateの25,200秒を超えた。
- Notebookは契約どおり両方のfull three-epoch runを開始せず正常終了した。OOMや例外による失敗ではない。
- `artifacts/train_v1/`へsmoke summary、split、fold別smoke log、kernel logを保存した。full model manifestとfull checkpointはない。
- smoke summary SHAは`6c33c754a9dcc38b18d44c90b02712bba8deb2132ea0e6eacdad80b2bd5d7270`、split SHAは`f73778d5437eef9de9cd82d795c5f310d40bc5453eefb785fc3629e85f6bfb5e`、kernel log SHAは`e6a43ec0369e6d95207bfb552512322bfde94eb245a3a5812135bdd273593676`。
- canonical `metrics.json`へKaggle出力を取り込み、kernel version、resource、SHA、rerun証拠を補完して`status=debug_completed`とした。

### 2026-09-11 train version 2の承認とgate変更

- version 1の結果提示後、9時間gateで同じ2fold・3 epochsをversion 2として再実行する案に対し、ユーザーが「実行してください」と承認した。
- `config.yaml`の`model.smoke.runtime_gate_hours`と`runtime.train_runtime_gate_hours`だけを7から9へ変更した。
- seed 314159、outer fold、batch size 8、3 epochs、モデル、loss、checkpoint選択、control再学習なしは変更しない。
- active variant 1、model/config 1、outer fold 2、booster 0、選択済みmodel予定数2を維持する。
- version 1のsmoke証拠は履歴として`metrics.json.evidence.reruns[0]`と`artifacts/train_v1/`に保持する。

push直前に次を確認した。

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache JUPYTER_DATA_DIR=/tmp/jupyter-data uv run --extra notebook jupytext --to ipynb --test experiments/exp006_embryo_holdout_seed314159/exp006_embryo_holdout_seed314159_train.py
make validate-exp EXP=exp006_embryo_holdout_seed314159
make check-exp EXP=exp006_embryo_holdout_seed314159
make test-exp EXP=exp006_embryo_holdout_seed314159
make prepare-kaggle-notebooks EXP=exp006_embryo_holdout_seed314159 EXTRA_ARGS="--notebook train --run-on-push"
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle quota --format json
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle kernels pull kentookumura/exp006-embryo-holdout-seed314159-train -p /tmp/exp006_train_pre_v2 -m
```

- 確認時刻: 2026-09-11 23:35:10 JST。
- Kaggle CLI認証を確認し、既存canonical kernel `id_no=133968180`をpullできた。
- resourceはT4、GPU有効、TPU無効、internet無効。生成packageにもtrain gate 9時間を確認した。
- GPU quotaは30.00時間中18.52時間使用、11.48時間残り、表示上の更新時刻は2026-09-12 00:00:00。
- 判断: version 1推定8.05時間とversion 2 gate 9時間の双方を11.48時間が上回るためpush可能。

### 2026-09-11 train version 2

```bash
make push-kaggle-train EXP=exp006_embryo_holdout_seed314159
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle kernels pull kentookumura/exp006-embryo-holdout-seed314159-train -p /tmp/exp006_train_metadata_v2 -m
make kaggle-logs KERNEL=kentookumura/exp006-embryo-holdout-seed314159-train
```

- 2026-09-11 23:37:19 JSTに同じcanonical kernelへversion 2をpushした。
- push後metadataは`id_no=133968180`、private、T4、TPU無効、internet無効、docker image SHAはversion 1と一致した。
- `metrics.json.status`を`running`、kernel versionを2へ更新し、version 1のsmoke証拠は保持した。

- 2fold smokeの合計推定は28,291.495秒、9時間gateは32,400秒で`gate_passed=true`。fold 0のfull trainingを開始した。
- live logsではfold 0 epoch 0の56/703 batchまで進行し、OOMや例外はない。
- 成果物取得とinference pushまで将来自動実行するheartbeat案は安全審査で拒否され、作成していない。
- 読み取り専用の15分間隔heartbeat `exp006-train-v2`を作成した。完了・失敗時だけ通知し、push、cancel、output取得、リポジトリ変更は行わない。
- ローカルの`logs -f`接続だけを停止し、Kaggle sessionは継続している。

### 2026-09-12 train version 2完了とinference準備

```bash
make kaggle-output KERNEL=kentookumura/exp006-embryo-holdout-seed314159-train OUT=/tmp/kaggle-output/exp006_embryo_holdout_seed314159/train-v2
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle kernels pull kentookumura/exp006-embryo-holdout-seed314159-train -p /tmp/kaggle-pull/exp006-train-v2-20260912 -m
make validate-exp EXP=exp006_embryo_holdout_seed314159
make check-exp EXP=exp006_embryo_holdout_seed314159
make test-exp EXP=exp006_embryo_holdout_seed314159
make prepare-kaggle-notebooks EXP=exp006_embryo_holdout_seed314159 EXTRA_ARGS="--notebook inference --run-on-push"
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle quota --format json
```

- train version 2はKaggle上で正常終了。Notebook計測時間は22,940.600秒、full trainingは22,866.104秒で、2foldとも3 epochsを完走した。
- fold 0のbest epochは2、内部選択scoreは0.9357。fold 1のbest epochは2、内部選択scoreは0.9310。
- model manifestは2 checkpointを参照し、fold 0 SHAは`e55da03b223b5118eac154dc1b05865b996a78060b3ad2740ba5f3842fd97ba8`、fold 1 SHAは`6e9cf60f956554bd54a96b7afb11089ccb0f14b13e9ff88e1b81265146ba6b52`。取得後の実ファイルSHAと一致した。
- model manifest SHAは`15175596a731e172a88699aae18423986b5f8ef0270486b28609348fb18943d6`、split SHAは`f73778d5437eef9de9cd82d795c5f310d40bc5453eefb785fc3629e85f6bfb5e`、training summary SHAは`95c59353fe4b1423eb2c8cb91c4202cab6f3de14552e3e420c85b7d53ee94f92`、kernel log SHAは`d3725323796cc9cd174a51c1ee79ef9204d604ad3226b96f1c64541e2989dfe7`。
- 検証済み成果物とKaggle metadataをignore済みの`artifacts/train_v2/`へ保存した。metadataは`id_no=133968180`、private、T4、TPU無効、internet無効で、docker image SHAはversion 1と一致した。
- exp005 inference version 1は199動画、欠落0、prediction SHA `0e8ee93962a7ac7dcea8d88d04ccd667237fa50aaafd24254b0bcfcea0fb6d73`で完了済みのため、比較入力として利用可能。
- exp006 inference packageはstrict生成済み。kernel sourceはexp006 trainとexp005 inference、resourceはT4、GPU有効、TPU無効、internet無効である。
- 2026-09-12T08:24:17+09:00のGPU quotaは30.00時間中27.17時間使用、残り2.83時間。exp005 inferenceの実績と余裕込み見積を下回るためpushせず、09:00 JST予定のquota更新を待つ。
- competition submission、hidden test推論、Public LB取得は行っていない。

### 2026-09-12 inference version 1開始

ユーザーからquota更新前の実行開始を明示的に依頼されたため、09:00 JST予定の更新を待たず、生成済みpackageを同じ内容のまま一度だけpushした。

```bash
make push-kaggle-infer EXP=exp006_embryo_holdout_seed314159
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle kernels status kentookumura/exp006-embryo-holdout-seed314159-inference
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle kernels pull kentookumura/exp006-embryo-holdout-seed314159-inference -p /tmp/kaggle-pull/exp006-inference-v1-20260912 -m
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle quota --format json
```

- canonical kernel `kentookumura/exp006-embryo-holdout-seed314159-inference`のversion 1が作成され、2026-09-12T08:54:40+09:00に`RUNNING`を確認した。
- pullしたmetadataは`id_no=134014258`、private、T4、GPU有効、TPU無効、internet無効。
- kernel sourceは`kentookumura/exp006-embryo-holdout-seed314159-train`と`kentookumura/exp005-embryo-holdout-batch8-inference`で、検証済みの2 checkpointとexp005の199動画推論を参照する。
- push直後のGPU quotaは30.00時間中27.49時間使用、残り2.51時間。表示上の更新時刻は2026-09-12T00:00:00Z（09:00 JST）。残量だけではexp005 inferenceの実績約4時間37分を満たさないが、ユーザーは更新前の開始を明示した。
- 開始直後に55秒間live logsへ接続した時点では本文がまだ出力されていなかった。statusは`RUNNING`で、即時失敗は確認されていない。
- competition submission、hidden test推論、Public LB取得は行っていない。

## 変更点

- exp005を親とし、seed 314159だけを変更する契約を固定した。
- 2-seed融合、submission、Public LBは本実験から除外した。
- 親augmentationが完全固定されないため、厳密なseed-only因果比較ではないことを記録した。

## 次のアクション

1. exp006 inference version 1の完了または失敗を確認する。
2. 完了後にoutputを取得し、199動画、欠落、公式指標、prediction差、誤差相関、生成物SHAを検証する。
3. 実行証拠を`metrics.json`と`result.md`へ記録し、ユーザーへ完了・採否判断を求める。

### 2026-09-12 inference version 1完了・結果回収

ユーザーからKaggle実行の完了通知を受け、status、output、metadata、kernel logを取得して実行証拠を検証した。

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle kernels status kentookumura/exp006-embryo-holdout-seed314159-inference
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle kernels files kentookumura/exp006-embryo-holdout-seed314159-inference --page-size 200
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle kernels output kentookumura/exp006-embryo-holdout-seed314159-inference -p /tmp/kaggle-output/exp006-inference-v1-selected --file-pattern 'artifacts/seed_comparison[.]json' --page-size 200
```

- Kaggle statusは`COMPLETE`。metadataは`id_no=134014258`、private、Tesla T4、GPU有効、TPU無効、internet無効で、kernel sourceはexp006 train version 2とexp005 inference version 1。
- Notebook計測時間は19,032.269秒。12時間gateの推定18,520.712秒を通過し、199動画を欠落なく推論した。
- 公式scoreは全体0.5533789422、44b6が0.7045717889、6bbaが0.5267756514。保存graphからの再計算値は一致した。
- exp005との差は全体+0.4284734261、44b6が+0.0797788939、6bbaが+0.5007270983。
- 199動画すべてでgraphが異なり、動画別score誤差相関はPearson 0.6125461826、Spearman 0.7240679729。
- 大きなcandidate cacheとprediction graph全体のローカル取得は中断し、評価、比較、manifest、metadata、kernel logだけを`artifacts/inference_v1/`へ保存した。Kaggle上の成果物は変更していない。
- inference summary SHAは`a1de6e010a40981a2f697cff244069f9ba3da7fcd5fc35952d1b29e543a16335`、official metric summary SHAは`748c0e0cd1d55da31cc0c72de3e971706be3207db93395d0720e35f3c2f4ce68`、prediction manifest SHAは`234619c58e1ea4001b953eefec72cd226119650d4668c51cd523687b3d42f588`、seed comparison SHAは`34a7597b958ad16d5f537777888cb5d60668b8edda5e0909f22f356dae87239c`、kernel log SHAは`e9928a535dd2b36f9734a9717b5591526427a58389f554971c5bd6024738fd27`。
- `metrics.json.status`はユーザーの完了・採否判断前の実行終了状態として`debug_completed`へ更新した。
- competition submission、hidden test推論、Public LB取得は行っていない。

## 次のアクション

1. 検証コマンドとstrict文書レビューを通す。
2. 結果をユーザーへ提示し、実験を`completed`として確定するか判断を求める。
3. ユーザー判断後、この実験に関係する変更だけをcommit・pushする。
