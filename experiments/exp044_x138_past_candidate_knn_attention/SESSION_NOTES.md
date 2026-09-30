# exp044_x138_past_candidate_knn_attention セッションノート

## 2026-09-24 実装

- ユーザー回答: exp043の入力と処理を使用し、公開primary trackerとK=8 attentionを同時学習する。
- `make new-exp EXP=exp044_x138_past_candidate_knn_attention SOURCE=experiments/exp043_x138_self_trained_head` で別実験を作成。exp038の履歴は保持。
- 公開サポートデータセットの `predict_unet_transformer.py`、`simple_node_transformer.py`、`train_unet_transformer.py` を小さなsourceファイルとして取得し、tracker入力が32画像特徴＋32位置特徴、元画像voxel座標であることを確認。
- exp043 inferenceをSHA固定でtrain Notebookの前処理へ再利用。検出候補、self-trained座標head、secondary tracker、双方向harmonicとlow-margin consensusは固定。graph作成前にpair cacheを保存するpatchを追加。
- exp038のK=8・13特徴attentionとGEFF教師lossを新実験へコピーし、exp043融合の再計算と胚別評価を追加。旧exp038 cacheは使用しない。
- `kaggle-strategy` の仮説索引手順を確認。直接承認では移行元backlog候補がなく、現行validatorは仮説IDと候補名の組を要求するため、exp044の系譜は親exp043と同じN/Aを維持した。仮説索引への追加は行わず、既存の未コミット戦略変更は保持した。
- exp043 head学習Notebookのコピーと未変更の推論Notebookコピーは誤実行防止のため新実験から削除・置換した。親実験のファイルは変更していない。
- `make check-exp EXP=exp044_x138_past_candidate_knn_attention`: 通過。
- `make test-exp EXP=exp044_x138_past_candidate_knn_attention`: 5件通過。
- `make validate-exp EXP=exp044_x138_past_candidate_knn_attention`: 通過。
- `make check-strategy-docs`: 既存の未コミット `backlog/KAGGLE_DIRECTION.md` 61行目が1221文字で、規定1200文字を21文字超えるため失敗。exp044からの索引変更は取り消し済みで、この既存行は本実験の実装範囲外として保持。
- `make check-markdown-math EXTRA_ARGS='"experiments/exp044_x138_past_candidate_knn_attention/requirements.md"'`: ローカル検査通過。`--github` でもGitHub API変換を確認し、TeX本文の保持が通過。ブラウザ実表示は未確認。
- Kaggle GPU学習、pair指標、公式score、submissionはいずれも未実行。
- Kaggle Notebook packageは `make prepare-kaggle-notebooks EXP=exp044_x138_past_candidate_knn_attention EXTRA_ARGS='--notebook train --run-on-push'` で生成。metadataはT4 GPU、internet off、4つの固定dataset、canonical id `kentookumura/exp044-x138-past-candidate-knn-attention-train`。
- Push直前の `uv run kaggle quota --format json` でGPUは42.56/45.00時間使用、残り2.44時間、refreshAtは2026-09-26 00:00 UTC。capture上限3時間＋tracker学習上限4時間に不足するため、pushは行っていない。最終source bundleは更新後に再prepareした。
- ユーザーは週次リセット後のKaggle実行を選択。Codexの同タスクheartbeat `exp044-kaggle` を設定し、リセット前にはpushせず、再開時にquota・静的検査・Notebook packageを再確認する。実行後は15分間隔で監視し、結果記録または判断待ちで停止する。

## 2026-09-25 実行判断

- ユーザーは週次リセット待ちを取り消し、現在のquotaでもKaggle実行を明示的に依頼した。GPU quota到達後に既存jobが継続するかは未検証であり、完走は保証しない。
- active variant/configは公開primary tracker＋K=8 attentionの1つ。2 fold、各3 epoch、2 checkpointを学習する。既存exp043対照は同じcaptureの固定logitで再測定し、control再学習はしない。booster方式は使わない（0件）。
- push直前のquotaはGPU 44.39/45.00時間使用、残り0.61時間、refreshAt 2026-09-26 00:00 UTC。T4 2枚、internet off。予算上限はcapture 180分＋tracker学習240分。ユーザー指示により残量不足でも開始を試みる。
- `make check-exp`、`make test-exp`（5件）、`make validate-exp`、`make prepare-kaggle-notebooks EXP=exp044_x138_past_candidate_knn_attention EXTRA_ARGS='--notebook train --run-on-push'` は通過。
- `make push-kaggle-train EXP=exp044_x138_past_candidate_knn_attention` が成功し、canonical notebook `kentookumura/exp044-x138-past-candidate-knn-attention-train` のversion 1を開始。Kaggle URL: https://www.kaggle.com/code/kentookumura/exp044-x138-past-candidate-knn-attention-train
- Kaggleからsource/metadataをpullし、id_no 135800536、T4 GPU、internet offを確認。live logsではbootstrap、weight SHA検証、offline依存導入まで進行を確認した。既存heartbeatを15分間隔の監視専用へ更新し、再pushを禁止した。

- 2026-09-25 10:28 UTC: `uv run kaggle quota --format json` はGPU使用45.68/45.00時間、残り0.00時間を表示。同時点の `uv run kaggle kernels status` はversion 1を `KernelWorkerStatus.RUNNING` と表示した。少なくともこの観測時点ではquota到達後も実行が続いている。

## 次（実行開始後）

1. 同じKaggle notebook version 1のlogs/statusを監視する。実行中の再pushはしない。
2. 2動画pilotがcapture 180分を超えると予測した場合、Notebookは残りのcaptureを開始せず停止する。
3. `candidate_retention.json`、`baseline_parity.json`、`pair_metrics.json`、モデルSHA、kernel version・実行時間を取得し、`metrics.json`と本ノートへ記録する。
4. 両胚のpair結果を示し、全graph推論へ進むかユーザーに判断を求める。submissionは別途承認が必要。

## 2026-09-25 Kaggle train version 1 完了確認

- `uv run kaggle kernels status kentookumura/exp044-x138-past-candidate-knn-attention-train` は `KernelWorkerStatus.COMPLETE`。`kernels files` で `baseline_parity.json`、`candidate_retention.json`、`pair_metrics.json`、`train_receipt.json`、fold別progress、2 checkpointを確認し、必要ファイルだけ `artifacts/kaggle_v1/` に取得した。Kaggle出力側の `metrics.json` は実行前の `planned` のままなので、ローカルの正本を上書きしていない。
- receiptと個別JSONのbaseline parity、candidate retention、pair metricsは一致。20動画、合計1823 pair windows。公開primary trackerのforward/reverse、融合replay、zero-deltaの最大絶対差は確認した6 windowsですべて0で、許容値0.0001以内。
- 直前frame候補8点の既知親回収率は44b6が3275/3275（100%）、6bbaが7181/7183（99.972%）。両方とも学習前停止条件の99%を超えた。
- 胚別pair結果: 44b6の既知edge recallは対照3162/3302（95.760%）からattention 3144/3302（95.215%）へ低下。active-pair errorsは383から364へ改善。6bbaの既知edge recallは7016/7275（96.440%）から6884/7275（94.625%）へ低下し、active-pair errorsは627から658へ悪化。既知分裂親は44b6が1/2から1/2、6bbaが1/11から0/11。
- `sha256sum`でfold_0 checkpoint `13518248d59f7b5684d0b64fad824b3b246c70dd95616aec174f83acf49ff6c2`、fold_1 checkpoint `8fa610c1a76adf5cfc78d6f2bcb06e64eeadd832ba623c30862f0f840d63021f` をreceiptと照合。capture内容digest、head・primary checkpoint SHAは `train_receipt.json`、個別JSON SHAは `metrics.json` に記録した。
- receipt上のcaptureは2195.28秒、学習は3926.72秒。kernel log末尾の相対時刻6533.655秒（Notebook HTML書き出し）から、Notebook実行時間を約6534秒と記録。Kaggle APIによる厳密な終了時刻ではない。version 1、T4 GPU、internet off。
## 2026-09-26 既存captureによる追加CPU診断

- ユーザーが悪化原因の先行調査を依頼。Kaggleへの再pushや新規jobを使わず、version 1の既存pair capture 1980ファイルを取得し、公開／再学習primary × attentionなし／ありの4条件をローカルCPUで比較する方針とした。
- 最初のローカル処理は2026-09-26 00:30 JSTごろのWSL再起動で中断。OOMやPython例外の記録はなく、再起動時刻を確認した。50窓ごとに進捗を保存するよう`diagnose_pair_effects.py`を更新して、00:44 JSTごろ同じ入力・出力で再開した。別のCPU診断プロセスは同時起動していない。
- 使用入力: `artifacts/kaggle_v1_diagnostic_inputs/tracker_capture/`、exp029に保存済みのGEFF、公開`SimpleNodeTransformer` source/checkpoint、`artifacts/kaggle_v1/models/`の2 checkpoint。実行は`--threads 4`で、GPUと全graph推論は使わなかった。
- 04:03 JSTに両胚を処理し、診断プロセス終了を確認。44b6の894窓と6bbaの929窓で計1823窓。captureの内容digestはKaggle receiptと一致し、公開checkpoint・sourceと2つの学習済みcheckpointのSHA-256を照合した。診断JSONのSHA-256、全91閾値のrecall、4条件の数値と判定変更数は`metrics.json`の`evidence.full_pair_diagnostic`に保存した。
- 公開対照と再学習primary＋attentionの各8項目は、両胚ともKaggle版`pair_metrics.json`と完全一致（parity mismatch 0）。閾値0.48で再学習primaryのみの変更active pairは44b6が72、6bbaが265、同primaryへattentionを足した変更は1、2。6bbaでは0.05〜0.95のどの閾値でも公開対照の既知edge回収とactive-pair errorsを同時達成できなかった。
- 公開primary＋attentionは共同学習済みdeltaを流用した反実仮想で、独立に学習したモデルではない。結果の解釈と限界、ユーザー判断の選択肢は`result.md`に記録。公式scoreは未計測で、実験の完了・採否は未判断。Kaggle再実行、全graph推論、submission、Git commit/pushは行っていない。
- 追加記録後の検証: `make check-exp EXP=exp044_x138_past_candidate_knn_attention` 通過、`make test-exp EXP=exp044_x138_past_candidate_knn_attention` 7件通過、`make validate-exp EXP=exp044_x138_past_candidate_knn_attention` 通過。`kaggle-review-exp` の `review_exp_docs.py --strict` でもtarget evidenceの主要区分が揃うことを確認した。
