# exp005_embryo_holdout_batch8 セッションノート

## 目的

exp004の学習batch size 16でのKaggle OOMを受け、batch sizeだけ8へ下げた2方向の胚holdout学習と全199動画の推論・公式評価をKaggle上で実行する。

## 現在の作業

- 作業内容: Kaggleのtrain/inference/diagnostic version 1と検証を完了し、ユーザー判断に基づいて実験を閉じた。
- ブロック要因: なし
- 次: exp006とexp007の結果が揃った後に実験方針を更新する。

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 予定

共通確認を先に行います。

```bash
make validate-exp EXP=exp005_embryo_holdout_batch8
make validate-config
make check-exp EXP=exp005_embryo_holdout_batch8
make test-exp EXP=exp005_embryo_holdout_batch8
```

次のうち、実験契約に必要なnotebookだけを予定へ残します。学習を伴わないauditやdiagnosticではtrain、提出を目的としない実験ではinferenceを機械的に実行しません。

trainが必要な場合:

```bash
make prepare-kaggle-notebooks EXP=exp005_embryo_holdout_batch8 EXTRA_ARGS="--notebook train --run-on-push"
make push-kaggle-train EXP=exp005_embryo_holdout_batch8
make kaggle-logs KERNEL=kentookumura/exp005-embryo-holdout-batch8-train
```

inferenceが必要な場合:

```bash
make prepare-kaggle-notebooks EXP=exp005_embryo_holdout_batch8 EXTRA_ARGS="--notebook inference --run-on-push"
make push-kaggle-infer EXP=exp005_embryo_holdout_batch8
make kaggle-logs KERNEL=kentookumura/exp005-embryo-holdout-batch8-inference
```

prepare後に生成された`kernel-metadata.json`からkernel idを取得して`KERNEL`へ指定する。id末尾とtitle由来slugが一致し、50文字以内であることを確認する。自動生成では上限や衝突を解消できない場合だけ、`kaggle-platform`の規則に従って意味のある短縮id/titleを明示する。placeholderのまま実行しない。

## 変更点

- 親実験: `exp004_embryo_holdout_baseline`。
- 学習batch size: 16から8へ変更。`model.training.batch_size`と`runtime.batch_size`を同時に8へ設定した。
- 固定事項: 2方向の胚holdout、3 epochs、model、loss、checkpoint selection、推論batch size 4、decode、候補cache、公式評価、fold間のmodel参照解放とCUDA cache解放。
- GPU学習コスト: active variant 1、model/config 1、outer fold 2、選択済みmodel予定数2、booster 0。exp004 batch 16 controlとexp002 mixed-split controlは再学習しない。
- Kaggle resource予定: T4 2基、internet無効、train/inference各12時間上限。push直前にquotaを再確認する。

### 2026-09-11 実装

```bash
make new-exp EXP=exp005_embryo_holdout_batch8 SOURCE=experiments/exp004_embryo_holdout_baseline EXTRA_ARGS="--copy-tests"
```


trainは889行・6章、inferenceは1245行・8章で、exp004と同じ章立てと記載量を維持した。正規NotebookをJupytext sourceから再生成し、round-tripを確認した。

```bash
make validate-exp EXP=exp005_embryo_holdout_batch8
make check-exp EXP=exp005_embryo_holdout_batch8
make test-exp EXP=exp005_embryo_holdout_batch8
make check-strategy-docs
make update-summary
git diff --check
```

strict validation、Ruff check/format、実験固有13テスト、strategy文書検査、差分検査が成功した。親configとの比較テストにより、学習parameterは`model.training.batch_size`と`runtime.batch_size`の16から8への変更だけであり、inference設定は固定されている。`HYP-20260910-14`へexp005を追加し、後続候補の未生成予測への依存をexp005へ更新した。
親実験のsource、Notebook実装、テストをコピーし、実行記録をplannedへ初期化した。親に残っていたpatch backupはexp005へ引き継がず、正規ファイルだけを実験入力とした。

### 2026-09-11 train push前確認

```bash
make prepare-kaggle-notebooks EXP=exp005_embryo_holdout_batch8 EXTRA_ARGS="--notebook train --run-on-push"
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run python .agents/skills/kaggle-platform/shared/check_all_credentials.py --require cli
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle quota --format json
```

- 確認時刻: 2026-09-11T13:33:50+09:00。
- 対象resource: GPU、`enable_gpu=true`、`enable_tpu=false`、`machine_shape=NvidiaTeslaT4`、internet無効。
- kernel id/title: `kentookumura/exp005-embryo-holdout-batch8-train` / `exp005_embryo_holdout_batch8 train`。id末尾とtitle由来slugは一致し、50文字以内。
- GPU quota: 30.00h中8.93h使用、21.07h残り、refreshは2026-09-12T00:00:00。
- 判断: train Notebookの12時間上限を上回る残量があるためpush可能。inferenceはtrain完了後に残量を再確認し、残りが上限より少ない場合は開始しない。
- GPU学習コスト再確認: active variant 1、model/config 1、outer fold 2、選択済みmodel予定数2、booster 0、control再学習なし。


### 2026-09-11 train version 1

```bash
make push-kaggle-train EXP=exp005_embryo_holdout_batch8
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle kernels pull kentookumura/exp005-embryo-holdout-batch8-train -p /tmp/kaggle-pull-exp005-train.EBg2Ph -m
make kaggle-logs KERNEL=kentookumura/exp005-embryo-holdout-batch8-train
```

- push: version 1、URL `https://www.kaggle.com/code/kentookumura/exp005-embryo-holdout-batch8-train`。
- Kaggle反映metadata: private、T4、internet無効、`id_no=133917265`、docker image SHA `37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`。
- fold 0 smoke: 36.551秒、peak allocated 6.945/6.013 GiB、peak reserved 9.139/7.508 GiB。
- fold 1 smoke: 22.199秒、peak allocated 6.722/6.013 GiB、peak reserved 9.025/7.566 GiB。exp004で失敗した最初のbackwardをbatch size 8で通過した。
- 合計学習見積: 22,182.851秒、12時間gateに対してpass。active variant 1、model/config 1、outer fold 2、booster 0、control再学習なし。
- full training: fold 0を開始し、live logsを切断した時点でepoch 0の3/703 batchまで進行。ローカルの`logs -f`だけを停止し、Kaggle sessionは停止していない。
- 15分間隔で将来のoutput検証・推論まで継続するheartbeat automationは、追加の明示承認が必要として最初の作成を停止した。ユーザーの「承認します」を受け、automation ID `exp005-kaggle`として作成し、statusを`ACTIVE`にした。変化がない間は通知しない。

### 2026-09-11 train version 1完了とinference push前確認

```bash
make kaggle-output KERNEL=kentookumura/exp005-embryo-holdout-batch8-train OUT=/tmp/kaggle-output/exp005_embryo_holdout_batch8/train-v1
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle kernels pull kentookumura/exp005-embryo-holdout-batch8-train -p /tmp/kaggle-pull-exp005-train-final-20260911 -m
make prepare-kaggle-notebooks EXP=exp005_embryo_holdout_batch8 EXTRA_ARGS="--notebook inference --run-on-push"
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle quota --format json
```

- train version 1はKaggle上で正常終了。Notebook計測時間は23,046.864秒、full trainingは22,988.115秒。foldごとに3 epochsを完走した。
- checkpointは2個。fold 0は`02e1e050d406dde11eacec8c06323047fea073031ee40f662eab6ba6b364492c`、fold 1は`86fc541c8a96720bfc5591b76b938219607b1fcff0b7601e4935a02efa8eca09`。
- fold 0のbest epochは1、内部選択scoreは0.9388。fold 1のbest epochは1、内部選択scoreは0.9315。どちらもouter evaluation embryoを選択に使っていない。
- outputを`artifacts/train_v1/`へ保存。model manifest、split、training summary、smoke summary、fold別training log、2 checkpointを取得した。
- model manifest SHAは`1bd9496a5f9d659e6fb9f829a2e86db2db143fb4cb14dab4c5ab83f72f2c7a58`、split SHAは`f73778d5437eef9de9cd82d795c5f310d40bc5453eefb785fc3629e85f6bfb5e`、Kaggle kernel log SHAは`34113fd2928cdc8f8c4c96019686b2827384ca9eddb178b17a0a7e4d1c8ac856`。manifest内SHAと取得後の実ファイルSHAはすべて一致した。
- train metadata再確認: version 1、private、T4、TPU無効、internet無効、`id_no=133917265`、docker image SHAは既記録値と一致。
- inference packageはstrict生成済みで、train version 1を唯一のkernel sourceとして参照する。
- 確認時刻: 2026-09-11T20:33:05+09:00。
- 対象resource: GPU、`enable_gpu=true`、`enable_tpu=false`、`machine_shape=NvidiaTeslaT4`、internet無効。
- kernel id/title: `kentookumura/exp005-embryo-holdout-batch8-inference` / `exp005_embryo_holdout_batch8 inference`。private、`run_on_push=true`で、competition sourceとtrain kernel sourceを保持する。
- GPU quota: 30.00h中15.48h使用、14.52h残り、refreshは2026-09-12T00:00:00。
- 判断: inference Notebookの12時間上限を上回る残量があるため、同じcanonical kernelへversion 1をpush可能。competition submissionは行わない。


### 2026-09-11 inference version 1

```bash
make push-kaggle-infer EXP=exp005_embryo_holdout_batch8
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle kernels pull kentookumura/exp005-embryo-holdout-batch8-inference -p /tmp/kaggle-pull-exp005-inference-v1-20260911 -m
make kaggle-logs KERNEL=kentookumura/exp005-embryo-holdout-batch8-inference
```

- 2026-09-11T20:46:28+09:00確認: canonical kernelへversion 1をpushした。URLは`https://www.kaggle.com/code/kentookumura/exp005-embryo-holdout-batch8-inference`。
- Kaggle反映metadata: private、T4、TPU無効、internet無効、`id_no=133957261`、docker image SHAはtrain version 1と一致し、唯一のkernel sourceは`kentookumura/exp005-embryo-holdout-batch8-train`。
- live logsではsource commitと固定inference設定を確認し、Kaggle sessionは実行を継続している。ローカルの`logs -f`接続だけを停止した。
- 2 checkpoint、model manifest、split、source commitのKaggle上検証後、2動画smokeを137.856秒で完了。全199動画の予測見積は13,011.535秒、余裕込み17,164.419秒で12時間gateにpassし、full inferenceを開始した。
- competition submission、hidden test submission、Public LB取得は行っていない。

### 2026-09-12 inference version 1完了とoutput検証

```bash
make kaggle-logs KERNEL=kentookumura/exp005-embryo-holdout-batch8-inference
make kaggle-output KERNEL=kentookumura/exp005-embryo-holdout-batch8-inference OUT=/tmp/kaggle-output/exp005_embryo_holdout_batch8/inference-v1
.venv/bin/kaggle kernels files kentookumura/exp005-embryo-holdout-batch8-inference --format json --page-size 200
.venv/bin/kaggle kernels output kentookumura/exp005-embryo-holdout-batch8-inference --file-pattern '(^|/)(inference_gate|inference_summary|official_metric_summary|per_sample_metrics|prediction_manifest|metrics)\.json$' --page-size 200
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle kernels pull kentookumura/exp005-embryo-holdout-batch8-inference -p /tmp/kaggle-output/exp005_embryo_holdout_batch8/inference-v1-metadata -m
```

- inference version 1はKaggle上で正常終了。Notebook計測時間は16,625.959秒（約4時間37分06秒）で、smokeからの余裕込み見積17,164.419秒以内だった。
- prediction manifestはexpected/completedとも199、sampleは重複0、胚別件数は`44b6=71`、`6bba=128`。全sampleで評価胚、fold、外側の学習胚、checkpoint SHAの対応が契約どおりだった。
- Kaggle側の6,805 output fileを全ページ列挙し、candidate NPZ 199件とprediction GEFF 199件がmanifestのpath集合と完全一致することを確認した。
- 動画別公式評価は199件、skipped 0。保存graphから再採点し、初回の動画別・胚別・全体集計との一致を確認した。
- 公式評価のoverall scoreは0.1249055162。胚別は`44b6=0.6247928950`、`6bba=0.0260485532`で、division Jaccardは両胚とも0だった。
- inference summaryのprediction SHAは`0e8ee93962a7ac7dcea8d88d04ccd667237fa50aaafd24254b0bcfcea0fb6d73`、candidate content SHAは`4d78e3c2f07aef23173f12c08e573aaf62604cc2dc42290b9a870864b8019752`、prediction manifest SHAは`588f25652a86725367366d574ce76d1d768731301904fff8a51f9b9aa5bc7956`。
- model manifest、split、source manifestのSHAはtrain version 1の取得済み証拠と一致した。kernel log SHAは`727cd87f8e94d19e4292992d4f9aa062310a329ea4ddc2ae1eba6ed17ce68fa6`。
- metadataはversion 1、private、T4、TPU無効、internet無効、`id_no=133957261`、train version 1を唯一のkernel sourceとして再確認した。
- 検証用JSON、kernel log、metadataをignore済みの`artifacts/inference_v1/`へ保存した。候補とgraph本体は大容量のためKaggle version 1を正とし、local full downloadは行わずremote file集合とmanifestで検証した。
- competition submission、hidden test submission、Public LB取得は行っていない。

### 2026-09-12 最終検証と実験レビュー

```bash
make validate-exp EXP=exp005_embryo_holdout_batch8
make check-exp EXP=exp005_embryo_holdout_batch8
make test-exp EXP=exp005_embryo_holdout_batch8
make check-strategy-docs
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run python .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py exp005 --root . --strict
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev ruff check scripts/update_experiment_summary.py tests/test_record_experiment.py
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev --extra notebook pytest -q tests/test_record_experiment.py
make test-common
make update-summary
git diff --check
```

- strict experiment validation、Ruff check/format、実験固有13テスト、strategy文書検査、strict experiment reviewer、差分検査が成功した。
- nested CVのoverall scoreだけを`experiment_summary.md`へ表示するよう共通summary生成を更新し、対象Ruffと回帰テスト4件が成功した。
- `make test-common`は304件中302件が成功し、既存のtemplate専用repository layout前提と、既存`exp001`を「missing experiment」とするsubmission monitor前提の2件だけが失敗した。どちらもexp005またはsummary表示変更が触れた処理ではない。
- `make validate-config`の全体検査は、ローカルに`data/raw/train`、`data/raw/test`、`data/raw/sample_submission.csv`がないため停止した。Kaggle outputとexp005固有のstrict validationは成功している。
- `metrics.json.status`はユーザー判断前の`running`を維持した。

設定と再現性方針は`config.yaml`、kernel情報、Kaggle Notebook実行時間、生成物SHA、rerun比較、実験statusは`metrics.json`へ記録する。このファイルには、それらを得たコマンド、時刻、途中経過、失敗と修正を時系列で残す。提出した場合はsubmission ref・提出日時・submission scoring status・監視開始からscore確定までの所要時間を、詳細な時系列の正として記録し、Notebook実行時間と混同しない。`SUBMISSIONS.md`には横断比較に必要な最終スナップショットだけを`record-submission`で記録し、状態遷移、観測時刻、実行コマンドを転記しない。

## 次のアクション

1. exp005関係の変更だけをcommitし、現在のbranchをpushする。
2. exp006とexp007の結果が揃った後に、動画単位のgroup validationと胚holdoutの役割分担を含む実験方針を更新する。

### 2026-09-12 候補coverage・画像分布の追加診断

- ユーザーの「それでは進めてください」に基づき、exp005の保存済み予測を変更しない追加診断を開始した。新しい実験番号は作らず、model再学習、0.965での再推論、competition submissionは行わない。
- 対象はinference version 1の全199動画。0.99通過後の候補nodeを固定公式評価と同じ7 µm物理距離でGTへ対応付け、最終graph node recallとの差を測る。
- 画像分布は各動画の固定3時刻、空間stride `(4, 8, 8)`でsampleし、Zarr attrsの0.001/0.999 quantileと推論時の正規化後統計を記録する。全voxelの完全な分布とは扱わない。
- 実行予定はCPUのdiagnostic Notebook 1本、variant 1、fold別の追加学習0、booster 0、control再学習なし。入力はcompetition trainと`kentookumura/exp005-embryo-holdout-batch8-inference`の固定outputである。
- 2026-09-12T08:48:39+09:00 push前確認: canonical kernelは`kentookumura/exp005-embryo-holdout-batch8-diagnostic`、private、CPU、TPU無効、internet無効、run-on-push有効。GPUを使わないためquota確認は不要。competition train、support dataset、inference version 1を入力に持ち、CPUで全199動画を逐次処理できる設定としてpush可能と判断した。

```bash
make prepare-kaggle-notebooks EXP=exp005_embryo_holdout_batch8 EXTRA_ARGS="--notebook diagnostic --run-on-push"
make push-kaggle-notebook EXP=exp005_embryo_holdout_batch8 NOTEBOOK=diagnostic
make kaggle-logs KERNEL=kentookumura/exp005-embryo-holdout-batch8-diagnostic
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle kernels pull kentookumura/exp005-embryo-holdout-batch8-diagnostic -p /tmp/kaggle-pull-exp005-diagnostic-v1-20260912 -m
make kaggle-output KERNEL=kentookumura/exp005-embryo-holdout-batch8-diagnostic OUT=/tmp/kaggle-output/exp005_embryo_holdout_batch8/diagnostic-v1
```

- diagnostic version 1はKaggle上で正常終了。Notebook計測時間は804.711秒、version 1、`id_no=134014050`、private、CPU、TPU無効、internet無効である。live logsの初回接続はAPI 500で終了したが、同じkernelのmetadata pullで存在を確認し、再接続後に完了ログとoutputを取得した。
- prediction manifestはSHA `588f25652a86725367366d574ce76d1d768731301904fff8a51f9b9aa5bc7956`、動画別metricsはSHA `f8abd34212aa200d4dd360d7f42fb5a2f27f00210359f871684117f1de523dd1`で、inference version 1の記録と一致した。対象199件、胚別71/128、欠落・重複0を確認した。
- 0.99候補のnode recall平均は`44b6=0.966469`、`6bba=0.945786`。最終graphは`44b6=0.856473`、`6bba=0.034793`で、候補生成後の平均低下は0.109996対0.910993だった。
- prediction manifestの決定的な追加集計では、0.5通過edge / 候補nodeは`44b6=0.615496`、`6bba=0.00513677`。閾値通過edgeとgraph入力edgeは両胚で同数だった。整数線形計画後のedge / 候補nodeは0.595892対0.00412755である。
- 画像の固定samplingでは、0.001–0.999 quantile幅中央値が`44b6=2579.33`、`6bba=1415.00`、推論正規化後のsample平均中央値が0.190322対0.098566、標準偏差中央値が0.160177対0.126912だった。全voxel統計や因果推定とは扱わない。
- CSV、summary、plot、manifestのSHAはKaggle出力manifestと一致した。kernel logとmetadataを含め、ignore済みの`artifacts/diagnostic_v1/`へ保存した。
- 結論: `6bba`方向の支配的な失敗は初期の検出候補coverageではなく、接続scoreが0.5を超えず、最終graphへほとんどnodeが残らない段階にある。整数線形計画は二次的な低下である。

### 2026-09-12 ユーザー判断と終了

- ユーザーが「ひとまずexp005は閉じてgit commitとpushしてください」と明示したため、`metrics.json.status`を`completed`へ変更した。
- 完了判断はexp005の契約と診断作業に対するものであり、モデル採用や今後のvalidation方針の確定ではない。
- exp006とexp007の結果が揃った後に方針を更新する。それまではbacklogと検証中の仮説の優先順位を変更しない。
