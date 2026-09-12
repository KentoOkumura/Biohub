# exp007_graph_checkpoint_selection セッションノート

## 目的

exp005と同じ学習runで全epoch checkpointを保存し、内部選択動画に対する現行proxyと固定公式graph指標が選ぶcheckpointを外側2胚で比較する。

## 現在の作業

- 作業内容: 2026-09-12のユーザー判断により、evaluation version 1の部分結果を残して実験を終了した。
- ブロック要因: なし。外側評価は未完了だが、再実行しない判断が確定した。
- 次: なし。現行方針の`public_detector_selection`を優先する。

## 実験化時のGPUコスト計画

- active training variant: 1
- model/config: 1
- outer fold: 2
- booster: 0
- 保存checkpoint予定数: 6
- selector: 2
- control再学習: なし。同一runのcheckpointへ両selectorを適用する。
- train見積: 7〜9時間、hard gate 11.5時間。
- evaluation見積: exp005の予測・評価時間の最大2倍を上限とし、smokeの保守係数込みで11.5時間を超える場合は停止する。

## コマンドログ

### 2026-09-11 実行済み

```bash
make new-exp EXP=exp007_graph_checkpoint_selection
```

標準雛形を作り、`graph_checkpoint`の契約と`HYP-20260910-14`の系譜を移行した。train/inference Notebookとテストは実装していない。


実験化後の文書・設定検証を実行した。

```bash
make validate-exp EXP=exp007_graph_checkpoint_selection
make check-exp EXP=exp007_graph_checkpoint_selection
make test-exp EXP=exp007_graph_checkpoint_selection
```

初回はREADME必須見出し、requirements見出し名、または雛形`settings.py`の整形で停止した。文書形式とRuff整形だけを修正して再実行し、`validate-exp`と`check-exp`は通過した。`test-exp`は実装前のため「No experiment-specific tests」として終了した。
### 2026-09-11 実装と検証

ユーザーの「exp007を実装してください」を実装承認として、exp005の正規Jupytext sourceを構成参照元にした。親にcompact self-contained版は存在しない。親train 889行・inference 1245行に対し、exp007はtrain 990行・inference 1599行で、親の全章を保ちつつcheckpoint保存とselector比較の章を追加した。固定`official_source/`はbyte-identicalにコピーし、実験ローカル`.ruff.toml`でlint対象外にした。

trainでは固定学習関数の`evaluate`を呼ぶたび、DataParallel prefixを正規化したstate dictを`epoch_0.pth`から`epoch_2.pth`へ保存する。fold manifestとbundleには6 checkpointのSHA、proxy score、同点時latest epochを記録する。固定sourceの学習loop、教師、lossは変更していない。

当初`inference`と呼んでいた後段Notebookでは、各foldの内部選択動画7件・12件を全3 checkpointで推論し、現行proxyと固定公式graph指標を同じ動画で比較する。選択manifestを保存してから、selectorが選んだfold内unique checkpointだけを外側胚へ適用する。外側graphはunique checkpoint単位で1回だけ保存・再採点し、2 selectorの全体・胚別summaryを同じ予測から組み立てる。

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache JUPYTER_DATA_DIR=/tmp/jupyter-data uv run --extra notebook jupytext --to ipynb experiments/exp007_graph_checkpoint_selection/exp007_graph_checkpoint_selection_train.py experiments/exp007_graph_checkpoint_selection/exp007_graph_checkpoint_selection_inference.py
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache JUPYTER_DATA_DIR=/tmp/jupyter-data uv run --extra notebook jupytext --to ipynb --test experiments/exp007_graph_checkpoint_selection/exp007_graph_checkpoint_selection_train.py
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache JUPYTER_DATA_DIR=/tmp/jupyter-data uv run --extra notebook jupytext --to ipynb --test experiments/exp007_graph_checkpoint_selection/exp007_graph_checkpoint_selection_inference.py
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev ruff check experiments/exp007_graph_checkpoint_selection/exp007_graph_checkpoint_selection_train.py experiments/exp007_graph_checkpoint_selection/exp007_graph_checkpoint_selection_inference.py --select F821
make validate-exp EXP=exp007_graph_checkpoint_selection
make check-exp EXP=exp007_graph_checkpoint_selection
make test-exp EXP=exp007_graph_checkpoint_selection
```

Jupytextの2件のround-tripとF821検査は通過した。`validate-exp`と`check-exp`は通過し、実験固有testは13件通過した。Kaggle prepare、push、実行は行っていない。

### 2026-09-12 exp005完了確認とquota待機

- `kentookumura/exp005-embryo-holdout-batch8-inference` version 1がKaggleで`COMPLETE`になったことを確認した。同kernelのmetadataをpullし、T4、GPU有効、internet無効、exp005 trainをkernel sourceとする構成を確認した。完了ログでは199動画すべての予測と公式評価、保存graphからの再採点一致が完了し、competition submissionを作成していない。Kaggle output一覧でも`inference_summary.json`、`official_metric_summary.json`、`per_sample_metrics.json`、`prediction_manifest.json`を確認した。
- exp007はローカルの`metrics.json`が`scaffold_completed`かつkernel情報未記録で、`SESSION_NOTES.md`にも未pushと記録されていた。Kaggle側の`kentookumura/exp007-graph-checkpoint-selection-train`をmetadata pullしたところ403となり、同kernelが未作成であることを確認した。
- push前検証として`make validate-exp`、`make check-exp`、`make test-exp`を再実行し、strict validation、Ruff check/format、実験固有13テストが通過した。
- 学習対象はactive variant 1、model/config 1、outer fold 2、保存checkpoint 6、booster 0で、control再学習なし。train packageを`--notebook train --run-on-push`でprepareし、canonical kernel id、T4、GPU有効、TPU無効、internet無効、正のNotebook/configとbootstrap manifestの一致を検証した。
- 2026-09-12T01:44:31+09:00のpush直前確認ではKaggle CLI認証は利用可能だったが、GPU quotaは30.00時間中22.37時間使用、残り7.63時間だった。11.5時間gateを安全に開始できないためpushしていない。quota refreshは2026-09-12T00:00:00 UTC（09:00 JST）予定。Kaggle CLI 2.2.4ではActive Sessions数を取得できないため事前確認対象にせず、session上限エラーが発生した場合だけ停止判断を行う。quota更新後に同じcanonical kernelが未作成であることを再確認してから、trainを一度だけpushする。

### 2026-09-12 exp007 train push直前確認

- 2026-09-12T08:09:20+09:00に、ユーザーから改めてexp007のpush指示を受けた。Kaggle側のcanonical kernelはmetadata pullが403で未作成、ローカルの`metrics.json`も`scaffold_completed`かつkernel情報未記録だったため、重複実行ではないことを確認した。
- `make validate-exp`、`make check-exp`、`make test-exp`を再実行し、strict validation、Ruff check/format、実験固有13テストが通過した。
- 学習対象はactive variant 1、model/config 1、outer fold 2、保存checkpoint 6、booster 0、control再学習なし。生成packageはtrain Notebookのみをcode fileとし、canonical kernel id、run-on-push、T4 x2想定、GPU有効、TPU無効、internet無効、正のNotebook/configとの一致を確認した。
- Kaggle CLI認証は利用可能。GPU quotaは30.00時間中26.73時間使用、残り3.27時間で、設定済み11.5時間gate未満。途中でquota切れになるリスクを残したまま、直前の明示的なpush指示に従って設定を縮小せずtrainを一度だけpushする。Kaggle CLI 2.2.4ではActive Sessions数を取得できないため事前確認対象にせず、session上限エラー時は既存sessionをcancelせず停止する。
- `make push-kaggle-train EXP=exp007_graph_checkpoint_selection`でcanonical kernelへversion 1を一度だけpushした。Kaggle URLは<https://www.kaggle.com/code/kentookumura/exp007-graph-checkpoint-selection-train>、開始記録時刻は2026-09-12T08:13:14+09:00。
- push後のmetadata pullでkernel id `kentookumura/exp007-graph-checkpoint-selection-train`、id_no `134011413`、`machine_shape: NvidiaTeslaT4`、GPU有効、TPU無効、internet無効を確認した。Kaggle statusは`KernelWorkerStatus.RUNNING`。実行ログで実GPU数を確認するまでは、configの期待値2を実測値として扱わない。

### 2026-09-12 exp007 train version 1完了確認

- 2026-09-12T15:00:31+09:00にKaggle status `KernelWorkerStatus.COMPLETE`を確認した。metadataはkernel id `kentookumura/exp007-graph-checkpoint-selection-train`、id_no `134011413`、`machine_shape: NvidiaTeslaT4`、GPU有効、TPU無効、internet無効のまま一致した。
- 完了ログでsource commit `075fc5f5a52d11077f9dc2b074644618f26939e2`、実測GPU `Tesla T4` 2基、2 fold smokeの合計予測18759.23秒、11.5時間gate通過を確認した。
- full trainingはfold 0が8467.31秒、fold 1が15116.60秒、合計23583.91秒。smoke込みNotebook計測時間は23633.58秒（約6時間33分54秒）だった。
- fold 0とfold 1のepoch 0、1、2をすべて保存し、bundle manifestの`model_count`は6。bundle manifest SHAは`0c1adb728dbc173fc4084abe13354d6e0869324108427e5c9f979d14384835e6`、split SHAは`f73778d5437eef9de9cd82d795c5f310d40bc5453eefb785fc3629e85f6bfb5e`。6 checkpointの個別SHAは`metrics.json`へ記録した。
- train内の既存proxyは両foldともepoch 2を選択した。ただし固定公式graph指標による内部選択と外側胚評価は後段のevaluationで行うため、まだ実行していない。train完了だけから仮説の支持・棄却や実験全体の完了を判断しない。
- 小規模な`metrics.json`、`training_summary.json`、`smoke_summary.json`、`splits.json`、3 manifest、kernel logだけを`/tmp/kaggle-output/exp007_graph_checkpoint_selection/train-evidence/`へ取得した。Kaggle output一覧でも6 checkpointを確認したが、モデル本体はローカルへ取得していない。competition submissionは作成していない。

### 2026-09-12 exp007 evaluation push直前確認

- 正のNotebook種別を`train`と`evaluation`として`config.yaml`へ明示し、Jupytext round-trip、F821、`make validate-exp`、`make check-exp`、`make test-exp`を実行した。strict validation、Ruff check/format、実験固有13テストが通過した。
- evaluation packageはcanonical kernel id `kentookumura/exp007-graph-checkpoint-selection-evaluation`、run-on-push、`machine_shape: NvidiaTeslaT4`、GPU有効、TPU無効、internet無効で、kernel sourceは完了済みの`kentookumura/exp007-graph-checkpoint-selection-train`だけである。
- 2026-09-12T15:42:29+09:00にKaggle CLI認証を確認した。GPU quotaは30.00時間中0.00時間使用、残り30.00時間、refreshは2026-09-19T00:00:00Zで、11.5時間gateを満たす。canonical evaluation kernelのmetadata pullは403で、未作成であることを確認した。
- Kaggle CLI 2.2.4ではActive Sessions数を事前取得できない。pushがsession上限エラーになった場合は既存sessionを停止せず、ユーザーへ判断を求める。

### 2026-09-12 exp007 evaluation version 1実行開始

- `make push-kaggle-notebook EXP=exp007_graph_checkpoint_selection NOTEBOOK=evaluation`でcanonical kernelへversion 1を一度だけpushした。Kaggle URLは<https://www.kaggle.com/code/kentookumura/exp007-graph-checkpoint-selection-evaluation>、開始記録時刻は2026-09-12T15:46:31+09:00。
- push後のmetadata pullでkernel id `kentookumura/exp007-graph-checkpoint-selection-evaluation`、id_no `134049533`、`machine_shape: NvidiaTeslaT4`、GPU有効、TPU無効、internet無効、kernel sourceがexp007 trainのみであることを確認した。
- Kaggle statusは`KernelWorkerStatus.RUNNING`。実行ログで実GPU数を確認するまでは、configの期待値2を実測値として扱わない。competition submissionは作成していない。

### 2026-09-12 exp007 evaluation version 1失敗と修正

- 2026-09-12T21:51:09+09:00にKaggle status `KernelWorkerStatus.ERROR`を確認した。最初の意味のあるtracebackは、Papermillが実行済みNotebookを保存するときの`OSError: [Errno 28] No space left on device`だった。その後の`/bin/sh: echo: I/O error`、`RemovePapermillHeader`のimport失敗、`__notebook__.ipynb`のJSON切断は、ディスク枯渇後の二次障害と判断した。GPU OOM、network、入力path、固定公式評価器の失敗ではない。
- 失敗前に、内部選択用の全6 checkpoint（fold 0は各7動画、fold 1は各12動画）の固定公式評価を完了した。両selectorが両foldでepoch 2を選択し、内部選択は4176.59秒、outer smokeは145.42秒、全体予測は23662.26秒、gate上限41400秒でruntime gateを通過した。full outer評価は開始前だった。
- 原因は、内部選択57動画について、公式評価に必要なGEFF graphに加えて、検出scoreと全edge候補を含むNPZを全checkpoint分保持していたことだった。特に未選択の早期checkpointには4万〜7.5万nodeを出す動画があり、再採点対象でない中間生成物がKaggle working diskを消費した。さらに選択manifest全体をNotebook出力へ表示していたが、これは二次的な増幅要因である。
- 選択指標、内部動画、全6 checkpoint、固定公式評価、内容SHA、外側評価を変えず、内部選択用graphは各fold×epochの公式評価とSHA計算直後に削除し、内部NPZは永続化しないよう修正した。`checkpoint_selection.json`には動画別公式指標、graph SHA、candidate content SHAを残す。外側評価のGEFFと候補NPZは従来どおり保持し、保存graphからの再採点契約も維持する。各内部stage削除後とouter開始前に空き容量を表示し、選択manifestのログ表示はfold×selectorのepochとscoreだけへ縮小した。
- 修正版evaluation sourceからNotebookを再生成した。Jupytext round-tripとF821検査、`make validate-exp`、`make check-exp`、`make test-exp`を実行し、strict validation、Ruff check/format、実験固有14テストが通過した。
- 再実行前のGPU quotaは30.00時間中8.87時間使用、残り21.13時間、refreshは2026-09-19T00:00:00Z。11.5時間gateを満たす。

### 2026-09-12 exp007終了判断

- version 2をpushする前に、両selectorが両foldで同じepoch 2を選んだため、同じcheckpoint予測を共有する外側評価から選択方法の差は得られないことを確認した。
- 今後は公開検出器を固定してトラッカーを学習する方針であり、自前検出器を含むこのrunの外側scoreだけを得る再実行はGPU費用に見合わないと整理した。
- ユーザーの「それでは閉じてください。最後にgit commitとpushしてください。」を`discarded`の明示判断として記録した。修正版evaluationはKaggleへpushせず、外側2胚の公式指標、prediction manifest、生成物SHAは未取得のまま終了する。

## 変更点

- 親をexp005へ更新し、batch size 8の胚holdoutを比較基準にした。
- 各foldのepoch index 0、1、2を保存し、同じ内部選択動画で2 selectorを比較する契約を固定した。
- outerラベルによる選択、閾値調整、追加selector探索を禁止した。

### 2026-09-12 後段Notebookの役割整理

- ユーザーとの確認により、後段はsubmission用inferenceではなく、学習で得た6 checkpointの選択と外側holdout validationを行う処理だと整理した。
- 正のJupytext sourceとNotebook、Kaggle Notebook種別、kernel idを`inference`から`evaluation`へ変更した。予測生成そのものの設定である`model.inference`は意味が正しいため維持した。
- output名は`evaluation_gate.json`と`evaluation_summary.json`へ変更し、hidden test推論とcompetition submissionを行わない契約を維持した。

## 次のアクション

- exp007の再実行は行わない。
- 次の実験作業では、現行方針の`public_detector_selection`を優先する。
