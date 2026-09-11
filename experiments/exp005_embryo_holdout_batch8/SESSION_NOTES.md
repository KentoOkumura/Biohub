# exp005_embryo_holdout_batch8 セッションノート

## 目的

exp004の学習batch size 16でのKaggle OOMを受け、batch sizeだけ8へ下げた2方向の胚holdout学習と全199動画の推論・公式評価をKaggle上で実行する。

## 現在の作業

- 作業内容: exp004からexp005を作成し、batch size 8の契約、設定、Notebook source、テスト、実験記録を整備している。
- ブロック要因: なし
- 次: Jupytext変換と静的検証を通し、runtime resourceとGPU quotaを確認してtrain Notebookをpushする。

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 予定

共通確認を先に行います。

```bash
make validate-exp EXP=exp005_embryo_holdout_batch8
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

設定と再現性方針は`config.yaml`、kernel情報、Kaggle Notebook実行時間、生成物SHA、rerun比較、実験statusは`metrics.json`へ記録する。このファイルには、それらを得たコマンド、時刻、途中経過、失敗と修正を時系列で残す。提出した場合はsubmission ref・提出日時・submission scoring status・監視開始からscore確定までの所要時間を、詳細な時系列の正として記録し、Notebook実行時間と混同しない。`SUBMISSIONS.md`には横断比較に必要な最終スナップショットだけを`record-submission`で記録し、状態遷移、観測時刻、実行コマンドを転記しない。

## 次のアクション

1. Jupytext sourceから正規Notebookを再生成し、実験固有のvalidation、Ruff、testを実行する。
2. trainをKaggleへpushしてlive logsを監視し、checkpoint 2個が揃った場合だけinferenceへ進む。
