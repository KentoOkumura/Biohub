# exp014_exact_window_cache セッションノート

## 目的

固定した公開primary/secondary modelの候補点特徴を元の2-frame window単位で可逆保存し、再読込後のedge score・最終graphがexp013と一致すること、および保存量と再利用費用を確認する。

## 現在の作業

- 作業内容: ユーザーが実行結果を確認し、後続tracker比較用cacheとして`usable`と判断した。
- ブロック要因: なし
- 次: exp014と、その成立に必要な親exp013・backlog移行の変更を確認してcommit/pushする。

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 実行済み

- 2026-09-13: ユーザーの「それではこれで進めてください」に基づき、`exact_window_cache`を`exp014_exact_window_cache`として実験化した。
- 2026-09-13: 親を、同じ公開推論の2-run一致を確認済みの`exp013_public_notebook_replay`とした。公開modelとfeature contractはexp011から継承した。
- 2026-09-13: 全public test windowで候補ID、grid/物理座標、検出score、位置特徴、mask、primary/secondary 32-channel特徴を元dtypeのまま非圧縮NPZへ保存する処理を実装した。
- 2026-09-13: 保存直後の再読込配列、primary/secondary edge logits、exp013の候補座標・graph topology・raw submission SHAを完全一致でguardする処理を実装した。
- 2026-09-13: Jupytext同期、strict experiment validation、Ruff check/format、実験固有4 tests、戦略文書検査、`git diff --check`が成功した。
- 2026-09-13 19:00 JST: Kaggle packageを生成し、private、GPU有効、TPU無効、T4、internet無効、`window_cache.py`同梱を確認した。直前の`uv run kaggle quota --format json`ではGPU使用23.84時間、残り6.16時間、refreshは2026-09-19 00:00（Kaggle表示）だった。
- 2026-09-13 19:00 JST: `make push-kaggle-infer EXP=exp014_exact_window_cache`でkernel `kentookumura/exp014-exact-window-cache-inference` version 1を開始した。初期logでは入力manifest、offline package、primary/DeepCenter checkpoint SHAの検証まで成功し、statusは`RUNNING`。
- 2026-09-13: kernel version 1が`COMPLETE`になった。`make kaggle-output KERNEL=kentookumura/exp014-exact-window-cache-inference OUT=experiments/exp014_exact_window_cache/artifacts/kaggle-v1`で全出力を回収した。
- 2026-09-13: public test 4動画の各99 window、合計396 NPZを確認した。GPU shard別manifestは各198行、cache総量は122,239,149 bytesで、`window_cache_summary.json`と一致した。
- 2026-09-13: 保存前後の全配列、primary/secondary edge logitsが完全一致した。exp013との候補座標、graph topology、raw `submission.csv` SHAも一致した。
- 2026-09-13: 集計した特徴抽出776.434秒に対し、cache書込0.599秒、読込1.098秒だった。読込/抽出比は0.001415、最大GPU memoryは684,355,072 bytes、予測処理の観測時間は597.083秒だった。
- 2026-09-13: 実行後のGPU quotaは使用24.23時間、残り5.77時間で、実行前からの差は0.39時間だった。refreshは2026-09-19 00:00（Kaggle表示）のまま。
- 2026-09-13: Kaggle submissionは行っていない。exp013のLBはcache等価性の判定に不要なため、採点完了を待たずに結果整理まで進めた。
- 2026-09-13: ユーザー「はいいいです。最後にcommitとpushしてください。」により、exp014を後続tracker比較用cacheとして`usable`と判断した。

### 再実行入口

共通確認を先に行います。

```bash
make validate-exp EXP=exp014_exact_window_cache
make check-exp EXP=exp014_exact_window_cache
make test-exp EXP=exp014_exact_window_cache
```

```bash
make prepare-kaggle-notebooks EXP=exp014_exact_window_cache EXTRA_ARGS="--notebook inference --run-on-push"
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle quota --format json
make push-kaggle-infer EXP=exp014_exact_window_cache
make kaggle-logs KERNEL=kentookumura/exp014-exact-window-cache-inference
```

prepare後に生成された`kernel-metadata.json`からkernel idを取得して`KERNEL`へ指定する。id末尾とtitle由来slugが一致し、50文字以内であることを確認する。自動生成では上限や衝突を解消できない場合だけ、`kaggle-platform`の規則に従って意味のある短縮id/titleを明示する。placeholderのまま実行しない。

## 変更点

- 予測parameter、候補生成、tracker、ILP、graph repairはexp013から変更しない。
- 各windowの候補点情報と2 modelの特徴を`window_cache.py`でNPZへ保存し、metadataと配列schema/content SHAで混用・破損を検出する。
- 再読込した特徴・座標・位置特徴・maskからedge logitsを計算し、直接計算とbitwise一致しなければ停止する。
- shard別manifestを集約し、抽出・書込・読込時間、容量、peak GPU memory、全window coverage、exp013 baseline一致をreceiptへ記録する。

設定と再現性方針は`config.yaml`、kernel情報、Kaggle Notebook実行時間、生成物SHA、rerun比較、実験statusは`metrics.json`へ記録する。このファイルには、それらを得たコマンド、時刻、途中経過、失敗と修正を時系列で残す。提出した場合はsubmission ref・提出日時・submission scoring status・監視開始からscore確定までの所要時間を、詳細な時系列の正として記録し、Notebook実行時間と混同しない。`SUBMISSIONS.md`には横断比較に必要な最終スナップショットだけを`record-submission`で記録し、状態遷移、観測時刻、実行コマンドを転記しない。

## 次のアクション

1. exp014の成立に必要な親exp013とbacklog移行を含め、関連差分だけをcommit/pushする。
2. 承認済みの次候補へ引き継ぐ。
