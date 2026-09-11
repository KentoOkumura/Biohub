# exp002_unet3d_expandable_segments セッションノート

## 目的

exp001で最初のbackwardがOOMになった公式batch size 16の学習条件を維持し、`PYTORCH_ALLOC_CONF=expandable_segments:True`だけで同じsmokeと3 epochsを実行可能にできるか検証する。

## 現在の作業

- 作業内容: exp002の学習、推論、提出、scoring記録、`usable`へのstatus更新が完了。
- ブロック要因: なし。
- 次: 胚を分けた主評価用CVとの整合を後続実験で確認する。

## GPU cost guard

- active variant: allocator有効の1条件。
- model config: 1。
- fold: split 0のみ。
- booster: 0。
- control再学習: なし。exp001 version 3のallocator無効smoke OOMを比較対象として参照する。
- full run: version 2はユーザー承認によりruntime gateを12時間へ変更し、同一smokeの予測が12時間以内なら3 epochsを開始する。

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 実行済み

```bash
make new-exp EXP=exp002_unet3d_expandable_segments SOURCE=experiments/exp001_temporal_unet3d_baseline EXTRA_ARGS="--copy-tests"
UV_CACHE_DIR=/tmp/uv-cache JUPYTER_DATA_DIR=/tmp/jupyter-data uv run --extra notebook jupytext --to ipynb experiments/exp002_unet3d_expandable_segments/exp002_unet3d_expandable_segments_train.py
UV_CACHE_DIR=/tmp/uv-cache JUPYTER_DATA_DIR=/tmp/jupyter-data uv run --extra notebook jupytext --to ipynb experiments/exp002_unet3d_expandable_segments/exp002_unet3d_expandable_segments_inference.py
make validate-exp EXP=exp002_unet3d_expandable_segments
make check-exp EXP=exp002_unet3d_expandable_segments
make test-exp EXP=exp002_unet3d_expandable_segments
make prepare-kaggle-notebooks EXP=exp002_unet3d_expandable_segments EXTRA_ARGS="--notebook train"
make prepare-kaggle-notebooks EXP=exp002_unet3d_expandable_segments EXTRA_ARGS="--notebook inference"
make check-strategy-docs
```

trainとinference packageは`run_on_push=false`で生成した。外部upload、GPU実行、competition submissionは行っていない。生成metadataのkernel slugは50文字以内で、inferenceのkernel sourceはexp002 train kernelを指す。

Kaggle実行で追加したコマンド:

```bash
make prepare-kaggle-notebooks EXP=exp002_unet3d_expandable_segments EXTRA_ARGS="--notebook train --run-on-push"
make push-kaggle-train EXP=exp002_unet3d_expandable_segments
make kaggle-logs KERNEL=kentookumura/exp002-unet3d-expandable-segments-train
make kaggle-output KERNEL=kentookumura/exp002-unet3d-expandable-segments-train OUT=/tmp/exp002-output.rDZbvE
make kaggle-output KERNEL=kentookumura/exp002-unet3d-expandable-segments-train OUT=/tmp/exp002-v2-output.iFOvZl
make prepare-kaggle-notebooks EXP=exp002_unet3d_expandable_segments EXTRA_ARGS="--notebook inference --run-on-push"
make push-kaggle-infer EXP=exp002_unet3d_expandable_segments
make kaggle-logs KERNEL=kentookumura/exp002-unet3d-expandable-segments-inference
make kaggle-output KERNEL=kentookumura/exp002-unet3d-expandable-segments-inference OUT=/tmp/exp002-inference-v1-output.BpaHgZ
make submit-check EXP=exp002_unet3d_expandable_segments SUBMISSION=experiments/exp002_unet3d_expandable_segments/artifacts/inference_v1/submission.csv
make submit-code COMPETITION=biohub-cell-tracking-during-development KERNEL=kentookumura/exp002-unet3d-expandable-segments-inference KERNEL_VERSION=1 OUTPUT_FILE=submission.csv MESSAGE="exp002 scratch 3ep expandable segments"
make record-exp EXP=exp002_unet3d_expandable_segments STATUS=running PUBLIC_LB=0.453
make record-exp EXP=exp002_unet3d_expandable_segments STATUS=usable
make record-submission EXP=exp002_unet3d_expandable_segments SUBMISSION=experiments/exp002_unet3d_expandable_segments/artifacts/inference_v1/submission.csv SUBMISSION_REF=56153451
```

### 未実行

なし。

## 変更点

- 2026-09-10: `make new-exp EXP=exp002_unet3d_expandable_segments SOURCE=experiments/exp001_temporal_unet3d_baseline EXTRA_ARGS="--copy-tests"`で親実験をコピーし、実行記録をplannedへ初期化した。
- 2026-09-10: backlog候補の契約を`requirements.md`へ移し、`config.yaml`の上位仮説ID、候補名、親実験を一致させた。
- 2026-09-10: `PYTORCH_ALLOC_CONF=expandable_segments:True`をTorch import前に設定するguardと、成功・OOM時のpeak allocated/reserved memory記録をtrain sourceへ追加した。
- 2026-09-10: inference sourceとkernel sourceをexp002 train artifactへ切り替えた。
- 2026-09-10: Jupytext train/inference round-trip、strict experiment validation、Ruff、10件の実験固有test、strategy document検証が通過した。
- 2026-09-10: train/inference Kaggle packageを`run_on_push=false`で生成し、GPU有効、internet無効、T4、competition source、offline dependency dataset、exp002 kernel sourceを確認した。
- 2026-09-10: `make validate-config`はGit管理外の`data/raw/train`、`data/raw/test`、`data/raw/sample_submission.csv`がローカルにないためproject path検証で停止した。exp002 strict validationは別途通過済み。
- 2026-09-10: repository-wide local Markdown math検査は通過した。link検査はexp001 `requirements.md`の既存相対リンク4件だけをmissingとして報告し、exp002とbacklogの変更箇所にはmissing linkがなかった。
- 2026-09-10 19:45:52 JST: push前にstrict validation、Ruff、10 testを再実行して通過した。train metadataはprivate、`run_on_push=true`、GPU有効、TPU無効、`NvidiaTeslaT4`、internet無効。GPU quotaは30.00h中29.72h残、refreshは2026-09-12T00:00:00。allocator有効1 variant・1 model config・split 0・booster 0・control再学習なしで、最大11hのruntime gateに十分なためpushすると判断した。
- 2026-09-10: canonical kernel `kentookumura/exp002-unet3d-expandable-segments-train`をpushし、version 1（id_no 133830163）を起動した。push後のpullでprivate、T4、TPU無効、internet無効、docker image SHA `37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`を確認した。
- 2026-09-10 19:52 JST: `PYTORCH_ALLOC_CONF=expandable_segments:True`、T4 x2、batch size 16でsmoke 2 iterationsがOOMなく完了した。smokeは54.731秒、GPU 0/1のpeak allocatedは12.477/7.866 GiB、peak reservedは14.033/10.963 GiBだった。
- 2026-09-10 19:52 JST: 3 epochsの保守的予測は42,143.044秒で11時間gateの39,600秒を2,543.044秒（約42.38分）超えた。`gate_passed=false`のためfull trainingを開始せず、full checkpoint、model manifest、inference、submission生成は行っていない。
- 2026-09-10 19:56:01 JST: Kaggle outputを`/tmp/exp002-output.rDZbvE`へ回収した。dataset index SHAは`cd0eab17e481c9a9956b585230d3865fe6fb80ebd6f85472a47404fa9f9c9a8c`、split SHAは`fca45709656ad0f0900fc5fcbf0b7f4aad3abb7d547fafcf4e022af6de1aa68e`、smoke summary SHAは`2926b281cfdbddb71b8a92e870e98124527f2181340c898902e53ce3425b54d8`、smoke log SHAは`8fb83b29bb9ce40de784af526fff2d0f75432406daee6d920f62b3b861ef4681`、kernel log SHAは`00cc3d619a9accb4d709c94ec5973b9f36c8f0851741a6db5901ec24abf1d4af`。

- 2026-09-10 20:20:17 JST: version 2 push直前にtrain metadataがprivate、`run_on_push=true`、GPU有効、TPU無効、`NvidiaTeslaT4`、internet無効であることを確認した。GPU quotaは30.00h中29.63h残、refreshは2026-09-12T00:00:00。allocator有効1 variant、model config 1、split 0、booster 0、control再学習なしで、予測11.706hに対してquotaは十分と判断した。Kaggleの12時間実行上限までの余裕が小さいtimeoutリスクは、ユーザー承認済みの追加実行条件として許容する。

- 2026-09-10: 同じcanonical kernel `kentookumura/exp002-unet3d-expandable-segments-train`へversion 2をpushした。pullでid_no 133830163、private、T4、TPU無効、internet無効、docker image SHA `37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`を確認し、local metrics statusを`running`へ更新した。

- 2026-09-10 20:32:44 JST: version 2 smokeは47.241秒でOOMなく完了した。GPU 0/1のpeak allocatedは12.327/7.866 GiB、peak reservedは14.016/10.963 GiB。3 epochsの保守的予測は36,375.657秒（10.104時間）、gateは43,200秒で`gate_passed=true`となり、180 train samples、19 validation samples、batch size 16のfull trainingへ入った。epoch 1の19/1,056 batchesまで正常に進み、直近は約7.2秒/batchだった。
- 2026-09-10 20:32:44 JST: heartbeat `exp002-full-training-monitor`を30分間隔で設定した。正常進行中は通知せず、terminal result時だけoutput回収、実験記録更新、検証、通知を行い、その後停止する。live SSEのローカル接続だけを終了し、Kaggle sessionは継続している。
- 2026-09-11 04:22 JST: version 2は3 epochsを完走した。full trainingは28,488.802秒（約7時間54分49秒）、Notebook全体は28,821.869秒（約8時間00分22秒）。epoch 0/1/2の診断用holdout selection scoreはいずれも0.9016で、best checkpointはepoch 0だった。validation edge accuracyは0.9992、0.9995、0.9997、node recallは0.9023、0.8952、0.8953だった。
- 2026-09-11: Kaggle outputを`/tmp/exp002-v2-output.iFOvZl`へ回収し、実験の`artifacts/`へコピーした。checkpoint、model config、model manifest、training summary、training log、dataset index、split、smoke記録、kernel logのSHAを`metrics.json`の`evidence.artifacts`へ記録した。checkpoint SHAは`f078da5ee21fac59a0ed963831771ab7db7a8d05189bc1017b1ef4807f7c7a4e`で、manifest記載値と一致した。inferenceとcompetition submissionは実行していない。
- 2026-09-11: 回収後にstrict experiment validation、Ruff、10件の実験固有test、strategy document検証、experiment document review、差分の空白検査を実行し、すべて通過した。

- 2026-09-11: ユーザーが「推論と提出に進んでください」と依頼し、exp002 checkpointを使うinference、提出前検証、competition submissionを承認した。
- 2026-09-11 08:07:27 JST: inference push前にstrict validation、Ruff、10件の実験固有test、Jupytext round-tripが通過した。生成metadataはprivate、`run_on_push=true`、GPU有効、TPU無効、`NvidiaTeslaT4`、internet無効、competition sourceとexp002 train kernel sourceを持つ。推論対象は1 checkpoint、1 inference configで再学習なし。GPU quotaは30.00h中21.63h残、refreshは2026-09-12T00:00:00で、12時間上限の1実行に足りると判断した。

- 2026-09-11 08:18:55 JST: canonical kernel `kentookumura/exp002-unet3d-expandable-segments-inference`へversion 1をpushした。pullでid_no 133891118、private、T4、TPU無効、internet無効、docker image SHA `37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`、exp002 train kernel sourceを確認した。live logではoffline依存導入、固定source 14 files、train manifest、checkpoint SHAを検証後、実行時に列挙した4 test datasetsの推論へ入った。ローカルのlive SSE接続だけを終了し、Kaggle sessionは継続している。

- 2026-09-11 08:21 JST: inference version 1は788.633秒で完走した。実行時に列挙した4 test datasetsから15,260 node rowsと11,320 edge rows、合計26,580行を生成し、Notebook内CSV検証を通過した。checkpoint SHAは`f078da5ee21fac59a0ed963831771ab7db7a8d05189bc1017b1ef4807f7c7a4e`、submission SHAは`b039062114962347aca22f1268a64196893eaabb62c1046ae0c90acbad29f223`。
- 2026-09-11: outputを`/tmp/exp002-inference-v1-output.BpaHgZ`へ取得し、Git無視対象の`artifacts/inference_v1/`へ保存した。Kaggle CLIの既定20件取得では多数のGEFF fileにより`submission.csv`が最初のpageへ入らなかったため、`--file-pattern ^submission[.]csv --page-size 200`で同じkernel versionから個別取得した。`kaggle kernels files`でもversion 1の`submission.csv`を確認した。inference kernel log SHAは`0ff5963663cc8822783572f5fec27b033b8b9281ab931f8d963d697327eb672d`。
- 2026-09-11: repository `submit-check`は26,580行、重複ID 0、missing 0、infinite 0でPASSした。push済みversion 1のmetadataはprivate、T4、internet無効、competition sourceとexp002 train kernel sourceを保持する。submit-check後のlocal `metrics.json`更新により生成packageのmetrics bootstrapがstaleと検出されたが、これはpush済みversion 1の再現性検証結果をlocal recordへ追記したためで、Kaggle上のimmutable version 1は変更も再pushもしていない。
- 2026-09-11 08:38:45 JST: submit前のcompetition submission一覧が0件であることを確認後、kernel `kentookumura/exp002-unet3d-expandable-segments-inference` version 1、output `submission.csv`、message `exp002 scratch 3ep expandable segments`でcode submissionを作成した。submit後に唯一追加されたrefは`56153451`、statusは`PENDING`。`artifacts/submission-monitor.log`でこのrefだけを固定監視した。
- 2026-09-11 12:06:55 JST: 固定監視中のsubmission ref `56153451`が`COMPLETE`となり、Public LB `0.453`が確定した。Private LBは未表示。監視開始から確定までの`scoring_elapsed_minutes`は206分で、Kaggle APIでも同じref、status、Public LBを確認した。
- 2026-09-11: `record-exp`でPublic LBを`metrics.json`へ先に記録し、その値を読んだ`record-submission`で同じrefを`SUBMISSIONS.md`の`v001`として保存した。提出履歴のファイル証拠は公開testで実行したimmutable kernel version 1の回収済み`submission.csv`であり、hidden test scoring時の出力ファイル自体はローカル取得していない。
- 2026-09-11: 確定結果の記録後にstrict experiment validation、Ruff、10件の実験固有test、strategy document検証、strict experiment document review、差分の空白検査を実行し、すべて通過した。
- 2026-09-11 12:47 JST: ユーザーがexp002を初期baselineとして`usable`にする推奨を承認し、あわせてexp002関連変更のcommit・pushを依頼した。`metrics.json`のstatusを`usable`へ更新した。

設定と再現性方針は`config.yaml`、kernel情報、Kaggle Notebook実行時間、生成物SHA、rerun比較、実験statusは`metrics.json`へ記録する。このファイルには、それらを得たコマンド、時刻、途中経過、失敗と修正を時系列で残す。提出した場合はsubmission ref・提出日時・submission scoring status・監視開始からscore確定までの所要時間を、詳細な時系列の正として記録し、Notebook実行時間と混同しない。`SUBMISSIONS.md`には横断比較に必要な最終スナップショットだけを`record-submission`で記録し、状態遷移、観測時刻、実行コマンドを転記しない。

## 次のアクション

1. 胚を分けた主評価用CVとの整合を後続実験で確認する。
