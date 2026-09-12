# exp009_exp006_twofold_ensemble セッションノート

## 目的

exp006の2fold checkpointをhidden testの全動画へ適用し、検出確率とedge確率を平均してから1回だけgraphを復号し、Kaggleへ提出する。

## 現在の作業

- 作業内容: Kaggle inference version 1をsubmission ref `56182729`として提出し、採点を監視中。
- ブロック要因: なし。採点待ちと並行してexp006の再現性実験へ進む。
- 次: ref `56182729`のscore確定後にLBを記録する。並行して再現性実験を実装・実行する。

## コマンドログ

- 2026-09-12: `make new-exp EXP=exp009_exp006_twofold_ensemble SOURCE=experiments/exp006_embryo_holdout_seed314159`。親sourceをコピーし、実行記録をplannedへ初期化した。
- 2026-09-12: 実行予定はactive variant 1、inference config 1、再学習fold 0、booster 0、再利用model 2。親controlの再学習なし。T4 x2で推論1回だけを実行する。
- 2026-09-12: Jupytext変換とtest、`make validate-exp`、`make check-exp`、`make test-exp`、`git diff --check`が成功。実験固有testは7件成功した。
- 2026-09-12: `make prepare-kaggle-notebooks EXP=exp009_exp006_twofold_ensemble EXTRA_ARGS="--notebook inference --run-on-push"`が成功。生成metadataは`enable_gpu=true`、`enable_tpu=false`、`machine_shape=NvidiaTeslaT4`、`enable_internet=false`、exp006 train kernel source 1件。
- 2026-09-12: push直前の`uv run kaggle quota --format json`でGPU残時間28.18時間、refreshは2026-09-19 00:00（Kaggle表示）。推論1回の12時間上限に足りるためpush可と判断した。
- 2026-09-12: `make push-kaggle-infer EXP=exp009_exp006_twofold_ensemble`で`kentookumura/exp009-exp006-twofold-ensemble-inference` version 1をpushした。pull後のid_noは134059662、T4、internet無効、exp006 train kernel sourceを確認した。
- 2026-09-12: live logでversion 1の完了を確認。4動画、188,460行、submission SHA `90ee74e696747e21a5962fefb4fa9a8d51e20fdec3723275d0388f9ffb42cbb1`、Notebook実行時間915.53444837秒。
- 2026-09-12: `make kaggle-output`でoutputを`/tmp/kaggle-output/exp009_exp006_twofold_ensemble/inference/`へ取得した。
- 2026-09-12: `make submit-check`がPASS。188,460行、duplicate 0、missing 0、infinite 0。Kaggle kernel filesにも`submission.csv`が存在した。
- 2026-09-12: 提出前一覧の既存submission refは56153451。exp009のsubmitコマンドは安全審査で、exp009としての明示承認がないため停止された。submission refは作成されていない。
- 2026-09-12: ユーザーの「提出していいです」でexp009 version 1のcompetition submissionを明示承認された。
- 2026-09-12 09:22:45.757 UTC: `make submit-code`で検証済み`submission.csv`を提出。Kaggle submission refは`56182729`、messageは`exp009 exp006 twofold probability ensemble`、初期statusは`pending`。同refを固定した監視を開始した。

### 予定

```bash
make validate-exp EXP=exp009_exp006_twofold_ensemble
make check-exp EXP=exp009_exp006_twofold_ensemble
make test-exp EXP=exp009_exp006_twofold_ensemble
make prepare-kaggle-notebooks EXP=exp009_exp006_twofold_ensemble EXTRA_ARGS="--notebook inference --run-on-push"
make push-kaggle-infer EXP=exp009_exp006_twofold_ensemble
make kaggle-logs KERNEL=kentookumura/exp009-exp006-twofold-ensemble-inference
make kaggle-output KERNEL=kentookumura/exp009-exp006-twofold-ensemble-inference OUT=/tmp/kaggle-output/exp009_exp006_twofold_ensemble/inference
make submit-check EXP=exp009_exp006_twofold_ensemble SUBMISSION=/tmp/kaggle-output/exp009_exp006_twofold_ensemble/inference/submission.csv
```

提出前後の一覧からsubmission refを特定し、同refを監視する。実行結果、修正、kernel version、提出時系列はこの節へ追記する。

## 変更点

- 完成graphの融合は行わず、両モデルのTTA後検出確率を平均して共通nodeを抽出する。
- 共通node上のsoftmax edge確率を平均し、既存の閾値・最大子親数制約・ILPを1回適用する。
- fold 0は`cuda:0`、fold 1は`cuda:1`を使い、学習済みcheckpointを再利用する。

## 次のアクション

1. submission ref `56182729`の採点を監視し、score確定後に記録する。
2. exp006の決定論的な後継実験を作成・実行する。
