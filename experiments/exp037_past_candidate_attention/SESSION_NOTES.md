# exp037_past_candidate_attention セッションノート

## 目的

予測済みの過去対応を入力せず、直前時点の全候補との座標関係を中央pairごとに集約するtrackerを実装し、保存済みexp016対照と同じ外側windowで比較できる状態にする。

## 現在の作業

- 作業内容: backlog契約をexp037へ移行し、座標attention、3時点cache整列、Kaggle学習・単体診断、合成testを実装する。
- ブロック要因: 全組み合わせ実装の保守的runtime予測198.17時間が12時間gateを超える。
- 次: 実測と停止判断をユーザーへ提示し、計算方法または実験契約を見直すか判断を仰ぐ。

## 実装記録

- 2026-09-22: ユーザーの「past_candidate_attentionを実装してください」を実験化・実装承認として受け取った。
- 2026-09-22: 次の番号を `exp037_past_candidate_attention` とし、exp035の保存済みexp016対照に対する単体診断経路を足場にscaffoldした。比較上の親はexp016へ設定した。
- 2026-09-22: 13次元特徴、13→32→32 MLP、候補別score、過去情報を使わない選択肢、候補chunkをまたぐonline softmax、biasなし・zero初期化delta出力を実装した。
- 2026-09-22: 直前windowの全候補を読み、重複時点のcandidate ID・grid座標・物理座標を中央windowと完全比較するdatasetを実装した。
- 2026-09-22: 実行予定はactive variant 1、model 2、fold 2、各3 epoch、booster 0。exp016 controlの再学習なし。履歴生成modelなし。

- 2026-09-22: Jupytext round-tripとF821検査、strictな実験validation、Ruff check・format checkを通した。experiment testは4 passed、1 skipped。ローカル環境にPyTorchがないためtensor依存test一式はskipされ、Kaggle実行前の未検証事項として残す。
- 2026-09-22: requirementsの数式はlocal検査とGitHub Markdown変換APIを通した。browserでの実表示は未確認。
- 2026-09-22: backlog索引から候補を削除し、HYP-20260910-10の対応実験へexp037を追加した。strategy document検査を通した。
- 2026-09-22: ユーザーの「実行してください」をKaggle上の2-fold学習・単体診断の実行承認として受け取った。全graph推論とsubmissionは承認範囲に含めない。
- 2026-09-22 22:41:37 JST: push前検査でmetadataはGPU有効、TPU無効、`NvidiaTeslaT4`、internet無効だった。Kaggle GPU quotaは残り16.18時間（45時間中28.82時間使用、2026-09-26 00:00 refresh）で、12時間runtime gate以内の今回の実行に足りると判断した。active variant 1、model 2、fold 2、各3 epoch、booster 0、exp016 control再学習なしを再確認した。

- 2026-09-22: Kaggle kernel `kentookumura/exp037-past-candidate-attention-train` version 1をT4で実行した。入力解決、18,707 eligible windowsの抽出、split作成までは成功したが、runtime benchmarkの最初のtrain stepでCUDA OOMとなった。14.56 GiB中14.54 GiB使用時に62 MiBを追加確保できず終了した。
- 2026-09-22: OOM原因を、source・過去候補軸だけを分割しtarget軸全体の3候補組MLP activationをbackwardまで保持していた実装と特定した。候補集合、fold、epoch、batch size、lossを変えず、target軸chunk 32とgradient checkpointingを追加した。version 2は同じkernel IDへpushする。
- 2026-09-22 23:07:12 JST: version 2 push前のGPU quotaは残り15.55時間。T4、TPU無効、internet無効、12時間runtime gateを維持し、再実行可能と判断した。

- 2026-09-22: version 2はtarget軸chunkとgradient checkpointingによりpeak GPU memory 1.52 GiBでbenchmarkを完了した。66 windowsのtrainは510.33秒、評価は129.89秒、最大候補積は1,102,004,400だった。全実行の保守的予測713,403.49秒（198.17時間）が43,200秒（12時間）gateを超えたため、本学習前に停止した。model、外側胚pair prediction、CV、graph、submissionは生成していない。
- 2026-09-22: version 2 outputからruntime benchmark、split manifest、GT window filter audit、metricsを取得し、metricsのstatusを`failed`へ更新した。このstatusは実行状態であり、採用・不採用・完了の判断ではない。

## コマンドログ

### 実行済み

```bash
make new-exp EXP=exp037_past_candidate_attention SOURCE=experiments/exp035_velocity_features
```

`task` が環境にないため、AGENTS.mdの規則どおり同名の `make` targetを使った。

### push前再検証

```bash
make validate-exp EXP=exp037_past_candidate_attention
make check-exp EXP=exp037_past_candidate_attention
make test-exp EXP=exp037_past_candidate_attention
make check-markdown-math EXTRA_ARGS='"experiments/exp037_past_candidate_attention/requirements.md"'
```

GitHub APIでの数式変換確認は完了済み。Kaggle prepare・push・実行は2026-09-22に承認された。

2026-09-22の実行承認後、上記3コマンドを再実行し、strict validation、Ruff check・format check、experiment test（4 passed、1 skipped）を通した。skipはローカルPyTorch未導入によるtensor依存testであり、Kaggle実行で確認する。

```bash
make prepare-kaggle-notebooks EXP=exp037_past_candidate_attention EXTRA_ARGS="--notebook train --run-on-push"
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run kaggle quota --format json
```

## 次のアクション

1. runtime gate結果をユーザーへ提示する。
2. 同じ全組み合わせ実装は再実行せず、計算方法または実験契約を変更する場合はユーザー判断を得る。