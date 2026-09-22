# exp035_velocity_features セッションノート

## 目的

exp015の固定cacheとexp016の学習契約を維持し、固定first-pass予測履歴から作る速度pair特徴が保存済みexp016より両胚の接続診断を改善するか確認する。

## 現在の作業

- 作業内容: Kaggle version 3で学習とpair診断が完了し、結果と生成物を回収・検証した。ユーザー判断待ち。
- ブロック要因: なし。
- 次: 今回のvelocity表現を不採用として実験を終了するか、ユーザー判断を確認する。

## コマンドログ

### 2026-09-22 実験化と実装

- ユーザーは保存済みexp016との比較だけを初回目的とし、同構造neutral controlを省く設計を承認した。
- ユーザーは実装とKaggle実行を承認した。submissionは承認されていない。
- `make new-exp EXP=exp035_velocity_features SOURCE=experiments/exp016_frozen_image_encoder EXTRA_ARGS="--copy-tests"` で親実験から作成した。
- backlog候補の契約、根拠、判断履歴、成功条件、停止条件を `requirements.md` へ移した。
- candidate IDと物理座標で整列する固定first-pass履歴、予測分裂後の娘履歴reset、15次元pair特徴、zero-initialized velocity branchを実装した。
- outer学習胚内のsample単位inner 2-foldで履歴model 4個、外側胚分割でvelocity model 2個を学習するpipelineを実装した。
- 保存済みexp016 manifestとfold modelのfile SHA・state SHAをconfigへ固定し、controlを再学習しない。
- 固定0.5で既知edge recall、active pair内の教師負例予測、正解親1位率、分裂回収数、履歴coverage、予測SHAを記録する。
- pair診断gate未達時は全graphを開始せず、公式score未計測として停止する。

### 2026-09-22 ローカル検証

- `make validate-exp EXP=exp035_velocity_features`: 成功。
- `make check-exp EXP=exp035_velocity_features`: 成功。
- `make test-exp EXP=exp035_velocity_features`: 5 passed、1 skipped。skipされたfileはローカル環境にtorchがない場合だけ対象となる速度branchのtensor testで、Kaggle GPU実行前の静的・純粋関数testは成功した。
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile ...`: 履歴、速度model、train pipeline、Notebook sourceで成功。
- Jupytextでtrain sourceとNotebookを同期した。
- exp016からコピーされた未使用のinference、公式評価、Colab replayコードは、pair gate前に誤実行しないようexp035から削除した。

### 2026-09-22 Kaggle push前確認

- `uv run kaggle quota --format json`: GPUは45.00時間中20.57時間残り、refreshは2026-09-26 00:00:00。12時間gateを満たす余裕がある。
- 新規学習はvelocity model 2個、history generator 4個、合計6個。保存済みexp016 controlの再学習は0個。
- `make prepare-kaggle-notebooks EXP=exp035_velocity_features EXTRA_ARGS="--notebook train --run-on-push"`: 成功。
- prepared kernel idは`kentookumura/exp035-velocity-features-train`。T4、internet無効、exp015 cache kernelとexp016 train kernelを入力に含む。metadata検証は成功した。

### 2026-09-22 Kaggle version 1・2の事前停止

- version 1は親exp016の古いcode cellがNotebookへ残っており、trainable component確認で学習前に停止した。thin Jupytext sourceからNotebookを明示再生成し、prepared Notebookに古いassertionがないことを確認した。
- version 2は新pipelineでcache・annotation検証と64 window benchmarkまで完了した。学習は開始していない。
- version 2のbenchmarkは15.680秒、peak GPU memory 256,681,472 bytesだった。全189,058 windowをbackward単価で数えたため保守投影が69,479秒となり、12時間gateで停止した。
- 履歴生成と外側診断はforward-onlyであり、この単価をbackwardと同一にする投影は過大だった。model数、epoch、sample、foldを減らさず、代表64 windowでbackward、診断forward、履歴forwardを別計測するよう修正する。
- version 1・2ともsubmissionは作成していない。

### 2026-09-22 Kaggle version 3の完了と出力回収

- version 3は別計測した64 window benchmarkを通過した。保守投影は16,814.770秒、gateは43,200秒、peak GPU memoryは271,617,536 bytesだった。
- velocity model 2個、対象sampleを除外した履歴生成model 4個を予定どおり各3 epoch学習した。学習・pair診断本体は8,006.298秒、Notebook全体は8,629.124秒だった。
- 6bbaでは、exp016から既知edge recallが96.9408%から96.8837%、正解親1位率が97.9235%から97.8664%、分裂親回収数が48/108から41/108へ低下した。active pair内の教師負例予測数は4,355から4,349へ減った。
- 44b6では、exp016から既知edge recallが94.7754%から94.2794%、分裂親回収数が6/22から3/22へ低下した。正解親1位率は96.8442%から96.8653%へ上がり、active pair内の教師負例予測数は1,341から1,255へ減った。
- 両foldとも既知edge recall改善と分裂回収数非減少を満たさず、pair診断gateは0/2胚で未達だった。事前契約どおり全graph推論と公式評価を開始しなかった。
- `make kaggle-output KERNEL=kentookumura/exp035-velocity-features-train OUT=experiments/exp035_velocity_features/artifacts/train_v3` でversion 3出力を回収した。
- model manifest SHA-256は`20ea0e29df1de080764704e5a3f24433fa22c807e0a681c2d40cbdec33325842`、history manifest SHA-256は`3119186560b6a989b4b00bcb00b963a2c417012f2f9a20b7baad84efc92d8670`。velocity model 2個、履歴model 4個のfile SHAを照合し、すべて一致した。
- 公式scoreは未計測、submissionは未作成。

## 変更点

- exp016のbase logitに、予測履歴の移動、候補変位、予測位置との差、速さ、差の大きさ、方向cosine、履歴有無、履歴長、直前edge確率を入力する小型MLPのdelta logitを加える。
- 履歴は対象sampleをgeneratorのgradient更新とcheckpoint選択から除外して固定生成する。
- outer評価履歴と比較baselineには、評価胚を学習していない保存済みexp016 fold modelを使う。
- 新規modelは履歴生成4個とvelocity 2個。exp016 controlの再学習は0個。

## 次のアクション

1. exp016比で精度改善なし、pair診断gate未達、公式score未計測という証拠をユーザーへ提示する。
2. 今回のvelocity表現を不採用として実験を終了するか、ユーザー判断を確認する。
3. ユーザー判断後にstatus、`result.md`、横断要約を更新し、この実験に関係する変更だけをcommit・pushする。
