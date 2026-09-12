# exp010_exp006_deterministic_replay セッションノート

## 目的

exp006後継の学習とheld-out評価を決定論化し、同一Kaggle環境の2回のフル実行でmodel、予測、metricのcanonical content SHAとCVが一致するか検証する。

## 現在の作業

- 作業内容: 決定論的な学習・評価・比較処理の実装と静的検証が完了し、train run 1（Kaggle kernel version 3）を実行中。
- ブロック要因: なし。
- 次: train run 1完了後に生成物を保存し、同じcheckpointを使うheld-out inference run 1を実行する。

## コマンドログ

- 2026-09-12: `make new-exp EXP=exp010_exp006_deterministic_replay SOURCE=experiments/exp006_embryo_holdout_seed314159`。親sourceをコピーし、実行記録をplannedへ初期化した。
- 2026-09-12: active variant 1、model config 1、outer fold 2、authoritative rerun 2、booster 0。exp006旧モデルは再利用せず、同じ決定論的構成をscratchから2回実行する契約を記録した。
- 2026-09-12: augmentation RNGをglobal seed・epoch・stable item indexから生成するよう固定sourceを変更し、entropy fallbackを禁止した。`CUBLAS_WORKSPACE_CONFIG=:4096:8`、PyTorch deterministic algorithms、cuDNN deterministic、benchmark無効、TF32無効をtrain/inferenceへ実装した。
- 2026-09-12: checkpoint tensor、候補array、最終graph、per-sample metric、公式summaryのcanonical content SHAと、2 runを比較する`compare_replays.py`を追加した。
- 2026-09-12: Jupytext変換、`make validate-exp`、`make check-exp`、`make test-exp`、Jupytext round-trip、strict experiment review、`git diff --check`が成功。実験固有testは8件成功した。
- 2026-09-12: train run 1 push直前の`uv run kaggle quota --format json`でGPU残時間26.46時間、refreshは2026-09-19 00:00（Kaggle表示）。全4 runの見込み約23.4時間に足りるが余裕は約3時間のため、各runの実測で継続判定する。
- 2026-09-12: `make prepare-kaggle-notebooks ... --notebook train --run-on-push`が成功。kernel ID `kentookumura/exp010-exp006-deterministic-replay-train`、T4 x2、internet無効、run-on-push有効、dataset source 1件、competition source 1件、kernel source 0件を確認した。
- 2026-09-12: `make push-kaggle-train`でtrain kernel version 1をpush。status `RUNNING`と、item別augmentation keyおよび決定論的runtime environmentがlive logへ出力されたことを確認した。
- 2026-09-12: train version 1 smokeは`max_pool3d_with_indices_backward_cuda`に決定論的実装がないため、設定どおりerrorで停止した。警告へ弱めず、UNetの非重複`MaxPool3d(2,2)`を同じ2x2x2領域・flatten順・first-max tie処理のreshape+`torch.max(dim=-1)`へ置換した。source manifestと契約を更新し、次versionのsmokeで検証する。
- 2026-09-12: train version 2の両fold smokeは決定論的algorithm errorなしで成功。全foldの保守的推定32,657.196秒（9.071時間）が9時間gateを257.196秒超えてfull training前に停止した。2fold・3 epochs・batch size・データは変えず、Notebook上限12時間内でgateだけを9.25時間へ最小修正した。
- 2026-09-12: train kernel version 3をauthoritative run 1としてpush。両foldのsmokeが成功し、保守的推定32,291.760秒（8.970時間）が9.25時間gateを通過してfull trainingへ入った。
- 2026-09-12: version 3実行入力のtrain Notebook SHA-256は`853ac1a8ad3e24557925c22cceaaaa22238708cd03d9ee788c9d918f079f3605`、`config.yaml` SHA-256は`4639dd3f449daab1f9fd07d47e3e9fdff0b8beff7c82238009043e4e0a9e34bf`。run 2 push直前に同じ値であることを必須確認する。
- 2026-09-12: full fold 0の初期ウォームアップ後は約5.4秒/batchで推移していることをlive logで確認した。

### 予定

```bash
make validate-exp EXP=exp010_exp006_deterministic_replay
make check-exp EXP=exp010_exp006_deterministic_replay
make test-exp EXP=exp010_exp006_deterministic_replay
make prepare-kaggle-notebooks EXP=exp010_exp006_deterministic_replay EXTRA_ARGS="--notebook train --run-on-push"
make push-kaggle-train EXP=exp010_exp006_deterministic_replay
make kaggle-output KERNEL=kentookumura/exp010-exp006-deterministic-replay-train OUT=<run-1-train-dir>
make prepare-kaggle-notebooks EXP=exp010_exp006_deterministic_replay EXTRA_ARGS="--notebook inference --run-on-push"
make push-kaggle-infer EXP=exp010_exp006_deterministic_replay
make kaggle-output KERNEL=kentookumura/exp010-exp006-deterministic-replay-inference OUT=<run-1-inference-dir>
```

run 1を保存後、Notebookと設定を変えずにtrain/inferenceを再pushし、run 2を保存して比較scriptを実行する。各push直前にGPU quotaを確認する。

## 変更点

- entropy由来だったaugmentation RNGをglobal seed・epoch・item indexのstable keyへ変更する。
- 決定論的PyTorch/CUDA設定を強制し、非決定的operationはerrorで停止する。
- checkpoint、候補予測、最終graph、metricのcanonical content SHAを記録する。

## 次のアクション

1. train/inferenceと比較script、testを実装する。
2. 静的検証後、同一のKaggle train/inferenceを2回ずつ実行する。
3. SHA/CV一致を確認し、証拠と未解決事項をユーザーへ提示する。
