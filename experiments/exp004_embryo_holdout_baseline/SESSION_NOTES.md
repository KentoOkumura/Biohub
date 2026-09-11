# exp004_embryo_holdout_baseline セッションノート

## 目的

exp002の学習・推論条件を固定し、`44b6`だけで学習して`6bba`を評価する方向と、その逆方向を作る。外側評価胚を学習とcheckpoint選択から除外した全199動画の予測、候補cache、固定公式評価結果を後続分析へ渡す。

## 現在の作業

- 作業内容: Kaggle train versions 1-3を実行し、fold 1の固定batch size 16がCUDA OOMとなることを確認した。full trainingとinferenceは未実行。
- 実行予定の規模: active variant 1、model config 1、外側2fold、各3 epochs、選択済みmodel 2、booster 0。exp002の両胚混在controlは再学習しない。
- ブロック要因: batch size 16のfold 1 smokeが、fold境界のCUDA解放後も最初のbackwardで再現してOOMとなる。契約を変更せずにはfull trainingへ進めない。
- 次: 計算条件を変更した再実験を行うか、ユーザー判断を確認する。

## コマンドログ

### 2026-09-11 実験化と実装

- `task new-exp EXP=exp004_embryo_holdout_baseline SOURCE=experiments/exp002_unet3d_expandable_segments EXTRA_ARGS="--copy-tests"`を試したが、local environmentに`task`がなかった。
- `make new-exp EXP=exp004_embryo_holdout_baseline SOURCE=experiments/exp002_unet3d_expandable_segments EXTRA_ARGS="--copy-tests"`で親実験をコピーし、実行記録をplannedへ初期化した。
- `backlog/embryo_holdout_baseline.md`の上位仮説、根拠、固定事項、変更事項、成功条件、停止条件、禁止事項、判断履歴を`requirements.md`へ移した。
- `config.yaml`に2方向の胚分割、候補cache schema、train/inferenceのruntime gate、exp004 train kernel依存、SHA記録方針を設定した。
- train sourceを、2foldの分割検査、各foldの2 iteration smoke、合計12時間gate、scratch 3 epochsの2回学習、fold別model manifest保存へ変更した。
- inference sourceを、各fold最初の1動画による推論時間gate、全199動画の動画単位保存、検出score、全接続scoreと4種類のmask、整数線形計画後の選択対応、固定公式評価の再計算へ変更した。
- workspace sandboxの`bubblewrap`が利用できず、標準`apply_patch`も同じhelperで失敗した。編集内容は`/tmp`へstageしてunified diffを生成し、対象を`experiments/exp004_embryo_holdout_baseline/`に限定した`patch`で適用した。

### 2026-09-11 静的検証

- Jupytextのtrainとinferenceの`.py -> .ipynb`変換、および両方のround-trip testが通った。
- `make validate-exp EXP=exp004_embryo_holdout_baseline`: strict validation通過。
- `make check-exp EXP=exp004_embryo_holdout_baseline`: Ruff checkとformat check通過。
- `make test-exp EXP=exp004_embryo_holdout_baseline`: 13 tests通過。
- `make check-strategy-docs`: 候補削除、上位仮説、後続3候補の依存リンクを含む整合性検査通過。
- `make update-summary`: `experiment_summary.md`へplannedのexp004を追加。
- 親実験に`*_compact_selfcontained_*` sourceはない。通常sourceはtrain 683行、inference 490行。exp004はtrain 877行、inference 1246行で、train 6節、inference 8節にsetup、入力・分割検査、実行、候補保存、評価、metricsを展開した。
- Kaggle full runとlocal full runは未実行。CV、checkpoint、予測、候補cache、実行時間は未取得。

### 2026-09-11 Kaggle train実行開始

- 実行承認: ユーザーの「kaggleで実行してください」。Notebookのpushと実行を承認として扱い、competition submissionの承認とは扱わない。
- GPUコスト: active variant 1、model config 1、外側2fold、各3 epochs、選択済みmodel 2、booster 0。exp002の両胚混在controlは再学習しない。
- `make prepare-kaggle-notebooks EXP=exp004_embryo_holdout_baseline EXTRA_ARGS="--notebook train --run-on-push"`: train packageを生成した。
- 2026-09-11T12:16:26+09:00のpush前確認: `enable_gpu=true`、`enable_tpu=false`、`machine_shape=NvidiaTeslaT4`、GPU残量21.40時間 / 30.00時間、refreshは2026-09-12T00:00:00。trainは最大12時間gateなので開始可能と判断した。推論用残量はtrain完了後に再確認する。
- canonical kernel `kentookumura/exp004-embryo-holdout-baseline-train`へversion 1をpushした。pullでid_no 133911105、private、T4、TPU無効、internet無効、docker image SHA `37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`を確認した。
- version 1は199動画の列挙、2foldの件数と非混入、固定source 14 files、T4 x2、allocator `expandable_segments:True`まで確認したが、fold 0の最初のbackwardで1.93 GiBを追加確保できずCUDA OOMとなった。full trainingとcheckpoint生成には進んでいない。
- exp002 version 2は同じsource・debug sample・batch size 16・T4 x2・docker image・allocatorでsmokeを通過しており、固定契約を変えず同じcanonical kernelのversion 2として再実行する。version 1のlog、split、dataset indexは`artifacts/train_v1_failure/`へ保存した。
- version 2のpush直前GPU残量は21.31時間。fold 0 smokeは47.64秒で完走したが、同一processで続けたfold 1の最初のbackwardで1.94 GiBを追加確保できずCUDA OOMとなり、full trainingには進まなかった。version 2のfold別logとkernel logは`artifacts/train_v2_failure/`へ保存した。
- version 2では`run_with_log(train, ...)`が返したfold 0 modelを変数`_`が保持したままfold 1を開始し、fold境界でgarbage collectionとCUDA cache解放を行っていなかった。学習条件を変えず、smokeとfull trainingの各fold終了時に返却modelを削除して`gc.collect()`と`torch.cuda.empty_cache()`を行うよう修正した。
- 修正後はstrict validation、13件の実験固有test、Jupytext round-tripが通過した。Ruffのimport順指摘は自動整形し、正規Notebookへ再同期した。
- version 3のpush直前GPU残量は21.18時間。fold 0 smokeは44.44秒で完走したが、返却model削除、garbage collection、CUDA cache解放後のfold 1も最初のbackwardで2.09 GiBを追加確保できずOOMとなった。GPU 0は14.56 GiB中756.81 MiBだけ空いていた。
- version 3のlog、split、dataset indexは`artifacts/train_v3_failure/`へ保存した。split SHAは全versionで`f73778d5437eef9de9cd82d795c5f310d40bc5453eefb785fc3629e85f6bfb5e`と一致した。
- batch size 16のままfold 1 smokeがversion 2と3で再現して失敗したため、契約の停止条件に従いfull trainingとinferenceを開始しない。再開にはbatch size変更、AMP実装、学習分割ごとの別session化など、結果に影響する方針のユーザー判断が必要。
- versions 1-3の証拠回収後のGPU残量は21.07時間 / 30.00時間。開始前21.40時間から0.33時間を使用した。
- checkpointが0個のためinference Notebookはpushしていない。competition submissionも行っていない。

### 予定

local environmentでは`task`がないため、同名の`make` targetを使う。

```bash
make validate-exp EXP=exp004_embryo_holdout_baseline
make check-exp EXP=exp004_embryo_holdout_baseline
make test-exp EXP=exp004_embryo_holdout_baseline
```

Kaggle実行は別途、push直前に`kaggle-platform`のresource / quota確認を行ってから次を使う。

```bash
make prepare-kaggle-notebooks EXP=exp004_embryo_holdout_baseline EXTRA_ARGS="--notebook train --run-on-push"
make push-kaggle-train EXP=exp004_embryo_holdout_baseline
make kaggle-logs KERNEL=kentookumura/exp004-embryo-holdout-baseline-train
make prepare-kaggle-notebooks EXP=exp004_embryo_holdout_baseline EXTRA_ARGS="--notebook inference --run-on-push"
make push-kaggle-infer EXP=exp004_embryo_holdout_baseline
make kaggle-logs KERNEL=kentookumura/exp004-embryo-holdout-baseline-inference
```

## 変更点

- 外側の評価は胚単位の2foldとし、内部のcheckpoint選択は外側学習胚の中だけで行う。
- 固定sourceへ変更を入れず、train関数へ渡すmodel、loss、augmentation、batch、epoch、checkpoint選択引数をexp002と一致させる。
- 推論Notebook内へ固定sourceの演算順序を展開し、decodeへ影響しない候補記録を加える。
- 最終graphを保存してから正解GEFFを読み、公式の`evaluate_pairs`と`summarise`を2回実行して再計算一致を検査する。
- competition test、submission.csv、Public LBは扱わない。

## 次のアクション

1. Kaggle実行へ進む場合は、`kaggle-platform`の手順でGPU quotaとActive Sessionsを確認する。
2. train Notebookで2foldのsmokeを行い、合計runtime gate通過時だけscratch 3 epochsを2方向で実行する。
3. model 2個が揃った場合だけinference Notebookを実行し、全199動画と固定公式評価の証拠を回収する。
