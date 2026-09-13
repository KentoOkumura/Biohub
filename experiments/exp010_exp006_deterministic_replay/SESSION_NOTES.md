# exp010_exp006_deterministic_replay セッションノート

## 目的

exp006後継の学習とheld-out評価を決定論化し、同一Kaggle環境の2回のフル実行でmodel、予測、metricのcanonical content SHAとCVが一致するか検証する。

## 現在の作業

- 作業内容: authoritative run 1のtrain（Kaggle kernel version 3）とheld-out inference（version 1）まで完了・記録した。ユーザー判断によりrun 2を行わず、実験を途中停止した。
- ブロック要因: なし。exp010の追加実行は行わない。
- 次: run 1の証拠を履歴として保持し、固定公開検出器を使う`exact_window_cache`へ進む。run間比較がないため`reproducibility.deterministic_anchor`は`false`のままとする。

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
- 2026-09-13: train kernel version 3が完了した。full trainingは36,627.947秒、Notebook計測は36,712.660秒。fold 0のcheckpoint content SHA-256は`928c1dadb2c58d314cc9f18812999bb500dbe2b2fa5de51da5f859096cff37b3`、fold 1は`a47225407f9643b842098eaeb282682c49178f2cc2b5f3d061ff85420e0e7ef1`。model manifest SHA-256は`fcb5c15685cfde2443c4a9f051a851b22c55a3400d7484922259ca4d64c77d7e`。
- 2026-09-13 08:48 JST: inference run 1 packageのmetadataはGPU有効、TPU無効、T4、internet無効。`uv run kaggle quota --format json`でGPU残時間13.38時間、refreshは2026-09-19 00:00（Kaggle表示）を確認した。inferenceのgate上限12時間には足りるためrun 1を開始するが、実測10.20時間のtrain run 2とinference run 2の両方には不足する見込みであり、次のpush前に再判定する。
- 2026-09-13: `make push-kaggle-infer`でinference kernel version 1をpushした。Kaggle側のmetadataをpullし、T4、GPU有効、TPU無効、internet無効、train kernel sourceを確認した。statusは`RUNNING`。
- 2026-09-13 16:14 JST: inference kernel version 1が`COMPLETE`になったことを確認した。22,557.637秒で199動画を欠落なく推論し、公式metricの再計算も保存graphからの集計と一致した。全体scoreは0.4981248955、44b6は0.5995244749、6bbaは0.4798695306。
- 2026-09-13 16:14 JST: run 1のcandidate content SHAは`065469446e1fdef67b1d1f2265270695171ca57222ebc93a2102d692a2d25fd2`、OOF prediction content SHAは`ef5eed1c0a8834f7e9c80644bd99f9c672e85fe9043546e23d184662bdf48c0e`、per-sample metric content SHAは`1bcbdf166010ee605b5655ecbc56051120ae0cb1623e2d5d9f7575ff1560e86a`、公式summary content SHAは`f7fa91a250a0a32e22baae08b32a94597a966c1d283d828625edb4214f141a41`。
- 2026-09-13 16:14 JST: 大容量の候補cache全体ではなく、run間比較に必要な`inference_summary.json`、公式summary、per-sample metrics、prediction manifest、metricsを`run-1-summary`へ保存した。`compare_replays.py`はtrain/inference双方のsummaryだけを入力とするため、run 2を行う場合のexact比較に必要な証拠は保持できている。
- 2026-09-13 16:14 JST: `uv run kaggle quota --format json`でGPU使用23.84時間、残り6.16時間、refreshは2026-09-19 00:00（Kaggle表示）を確認した。実測10.20時間のtrain run 2に足りないため、新しいrunはpushしなかった。
- 2026-09-13: ユーザーが「それではこれで進めてください」と判断し、run 1の証拠を保存したままexp010を停止して、GPUを現行の固定公開検出器路線へ回す方針を承認した。このリポジトリ内の実験statusを`discarded`へ変更し、exp010の監視automationを削除した。

### 実行済みコマンドと保留中の予定

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

run 1までは実行済み。run 2と比較scriptはquota不足およびユーザー判断待ちのため未実行。続行する場合も、Notebookと設定を変えず、各push直前にGPU quotaを再確認する。

## 変更点

- entropy由来だったaugmentation RNGをglobal seed・epoch・item indexのstable keyへ変更する。
- 決定論的PyTorch/CUDA設定を強制し、非決定的operationはerrorで停止する。
- checkpoint、候補予測、最終graph、metricのcanonical content SHAを記録する。

## 次のアクション

1. run 1のtrain/inferenceとcanonical SHA、CVは記録済み。
2. run 2は実行しない。exp010の完全な再現性は未確認として保持する。
3. 決定論的学習の確認は、現行路線の最初のtracker学習実験で行う。
