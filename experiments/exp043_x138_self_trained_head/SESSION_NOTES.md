# exp043_x138_self_trained_head セッションノート

## 目的

公開x138の固定画像特徴から自前の座標補正headを学習し、動画単位で補正前後の中心誤差を比べて、同じ推論構成のpublic testへ組み込む。

## 2026-09-24 実装と実行

- ユーザーの「分かりました進めてください」を自前学習・Kaggle Notebook実行の承認として受けた。competition submissionの承認は含まない。
- `make new-exp EXP=exp043_x138_self_trained_head` で直接承認の実験を作成した。exp042の作者重みを必要とする忠実再現は変更していない。
- x138 V1の固定NotebookからJupytext sourceを生成する`build_notebook.py`を実装した。元の推論セルは保持し、非公開headの参照を`capture` mode、train GEFFでの教師対応と自前head学習に変更した。
- 学習対象はV1284と同じ構造のhead 1件、config 1件、動画単位の評価5分割、最終head 1件。公開画像モデルとトラッカーの再学習は0件、boosterは0件。補正なしは保存座標から計算し、GPU control再学習はしない。
- `make validate-exp`通過。`make check-exp`通過。`make test-exp`は公開推論セルの保持と変更範囲のテスト1件が通過。合成特徴400対で5分割学習、2胚別の改善、checkpointの`state_dict`/`mean`/`scale`形式を確認した。これは実データ精度の証拠ではない。
- `make prepare-kaggle-notebooks EXP=exp043_x138_self_trained_head EXTRA_ARGS="--notebook train --run-on-push"`でprivate Kaggle packageを生成した。metadataはGPU true、TPU false、T4、internet false、公開3 datasetとcompetition trainを入力とする。
- push直前のGPU quota確認: 2026-09-24 00:12 JST時点、週45時間中残り4.48時間、refreshは2026-09-26 09:00 JST。公開x138の4 public動画は作者実行約20分だが、こちらの20 train動画のcapture時間は未測定。2動画の予備確認で全体見込みを計算し、capture 3時間を超える見込みなら残りを開始しない。train 3時間上限とpublic test推論約20分、setup・学習の余裕を含めて残り枠内で開始する判断。

## 初回実行時の予定（履歴）

1. packageをKaggleへpushし、T4設定が反映されたことを確認する。
2. Kaggleログと生成物を取得し、20動画・両胚の評価、head SHA、public test推論を記録する。
3. 推論が完走した場合もcompetition submissionは行わない。結果をユーザーへ示し、採否・完了判断を受ける。

- 2026-09-24 00:14 JST: `make push-kaggle-train EXP=exp043_x138_self_trained_head`でprivate kernel version 1をpushし、実行を開始した。kernel idは`kentookumura/exp043-x138-self-trained-head-train`。
- 同kernelを`kaggle kernels pull -m`で確認。Kaggle側の`id_no=135537982`、GPU true、TPU false、internet false、`machine_shape=NvidiaTeslaT4`、公開3 datasetとcompetition sourceを照合した。
- Kaggle version 1の予備確認: 2動画の特徴取得は4.50分で完了し、各動画の特徴NPZを確認した。20動画への単純外挿は45.01分でcapture上限180分以内。Notebookは残り18動画の取得へ進んだ。
- 選ばれた20 train動画のIDをlive logで確認し、公開testの4動画（`44b6_0113de3b`、`44b6_0b24845f`、`6bba_05b6850b`、`6bba_05db0fb1`）との重複がないことを確認した。検証はそれでも公開画像モデルの学習済みtrain集合に対する条件付き評価である。
- 20動画のcaptureは予備確認4.50分と残り18動画36.24分、合計40.74分で完了。動画単位の5分割評価と最終head保存まで進んだ。live logの条件付き評価では`44b6`が3,367対で平均距離1.56318879→1.45189905 µm、改善7/10動画、`6bba`が7,527対で1.93696129→1.54492807 µm、改善10/10動画。最終headのlive log SHA256は`32d6c62f738c3dfe4862e3df5272850312e81b622e57d9ba45209ebb382824dc`。生成物取得・照合前の速報として記録し、公式scoreとは扱わない。
- 同じNotebookが自前headをx138へ組み込んだpublic test 4動画の推論へ進んだ。competition submissionはしていない。
- train kernel version 1は学習とpublic testの`submission.csv`生成（225,875行）まで進んだが、同じ作業領域に残るtrain動画の`retention_guard_*.jsonl`も後段の公開test診断セルが読んだため、「4動画だけ」という検査で停止した。head学習や画像推論そのものの失敗ではない。
- Kaggle output APIから`coordinate_validation.json`、`self_trained_head_manifest.json`、`self_trained_v1284_head.pt`、`submission.csv`を取得し、`artifacts/train_v1/`に保存。head SHA256はmanifestと実ファイルで一致した。`submission.csv`のSHA256は`3812d2b7a73f47a7ef2b76847ba1f5679100a290a22a13c95c311f5f780814af`だが、後段セルが未完了なので最終検証済み出力とは扱わない。
- 特徴抽出の再実行を避けるため、同じ実験に推論Notebookを追加。固定したhead SHAを持ち、train kernel outputを`kernel_sources`として取り込み、公開testだけをcleanな作業領域でx138の元の後処理と検査まで実行する。
- 推論push直前のquota: 2026-09-24 約01:20 JST時点、週45時間中残り3.56時間。4公開test動画のモデル推論はtrain version 1で9.07分だった。後処理とsetupを含めても残り枠内と判断。
- train kernel version 1は失敗扱いでもAPIで生成物のdownloadは可能だったが、`kernel_sources`として別Notebookへ追加するとKaggleが`not valid kernel sources`と返した。推論kernel version 1は入力が欠けたままpushされたため結果に採用しない。
- 回収したhead 34,133 bytesとmanifest 1,059 bytesをprivate Dataset `kentookumura/exp043-self-trained-coordinate-head`として登録し、Kaggle APIで両ファイルの反映を確認した。配布ライセンスを勝手に宣言しないためmetadataのlicenseは`unknown`とした。checkpoint SHAはNotebook実行時に再検査する。
- 推論kernelの存在を`kaggle kernels pull -m`で確認した後、同じkernel idの入力をprivate Datasetへ変更した。version 2 push直前のquotaは週45時間中残り3.50時間で、推論に十分と判断した。
- 推論kernel version 2は`KernelWorkerStatus.COMPLETE`。T4・internet false・private head DatasetのKaggle側metadataを`kaggle kernels pull -m`で確認した。実行時head SHAはtrain version 1のmanifestと一致。
- `kaggle kernels output`でversion 2の`self_trained_x138_receipt.json`、`run_stats.csv`、`submission.csv`、後処理reportを`artifacts/inference_v2/`へ取得した。元の後処理は`selected=base`、frame-retention reportは`clean_graph_audit_pass_candidate_unverified_quality`。
- receiptと実ファイルの照合で公開test 4動画、225,875行、CSV SHA256 `3812d2b7a73f47a7ef2b76847ba1f5679100a290a22a13c95c311f5f780814af`、モデル予測区間539.08秒を確認。train version 1の後段失敗前CSVと同じSHAだったが、2 clean runとは呼ばない。
- `make submit-check EXP=exp043_x138_self_trained_head SUBMISSION=experiments/exp043_x138_self_trained_head/artifacts/inference_v2/submission.csv`はPASS。重複ID・欠損・無限大はいずれも0。competition submissionは未実行。公式scoreとhidden test全体のruntimeは未計測。

## 2026-09-24 学習Notebookの失敗対応

- train kernel version 1 の失敗は、train動画の診断ログを公開test診断が同じ作業領域から読んだことが原因。推論側 version 2 はすでに正常終了しているが、train側も正常終了するよう、学習Notebookをheadと検証成果物の保存・照合で終了する構成に変更した。公開test推論は別の推論Notebookへ任せる。
- 修正後に `make validate-exp`、`make check-exp`、`make test-exp` を実行し、すべて通過。修正済みtrain packageのmetadataはGPU true、TPU false、T4、internet false、公開3 Datasetとcompetition source。
- 2026-09-24 08:48 JSTのKaggle quotaは週45時間中残り3.15時間、refreshは2026-09-26 09:00 JST。version 1の20動画capture実測40.74分にsetup・学習の余裕を加えても残時間内と判断し、同じkernel idへversion 2をpushする。既存kernelはpush前に`kaggle kernels pull -m`で存在確認済み。

- 2026-09-24 08:49 JST頃: `make push-kaggle-train EXP=exp043_x138_self_trained_head`で同じprivate kernel idへversion 2をpush。Kaggle側へ保存されたmetadataのT4、internet false、公開3 Datasetとcompetition sourceを`kaggle kernels pull -m`で再確認した。
- train kernel version 2は`KernelWorkerStatus.COMPLETE`。20動画のcaptureは2435.32秒で完了し、5分割評価、最終head保存、学習receiptの出力までKaggle保存ログと実ファイルで確認した。manifestのhead SHAは`32d6c62f738c3dfe4862e3df5272850312e81b622e57d9ba45209ebb382824dc`。
- Kaggle output取得時に`www.kaggleusercontent.com`と`api.kaggle.com`のDNS解決が一時失敗した。同じkernel versionのoutput取得を再試行して、評価JSON、manifest、重み、学習receiptを`artifacts/train_v2/`へ保存し、実ファイルSHAを確認した。
- train version 1と2でcapture内容、OOF予測、評価JSON、manifest、checkpointのSHAが一致。train version 2の重みSHAは推論version 2のreceiptに記録された入力重みSHAとも一致し、推論CSVの実ファイルSHAもreceiptと一致。推論Notebookを再実行する必要はない。学習側と推論側の両方がKaggleで正常終了した。competition submissionと公式score測定は未実施。

## 2026-09-24 コンペ提出と採点監視

- ユーザーの「提出に進んでください」を、推論Notebook version 2のcompetition submissionの承認として受けた。
- 提出直前の`make submit-check EXP=exp043_x138_self_trained_head SUBMISSION=experiments/exp043_x138_self_trained_head/artifacts/inference_v2/submission.csv`はPASS。225,875行、重複ID・欠損・無限大はいずれも0。Kaggle側の`kentookumura/exp043-x138-self-trained-head-inference/2`は`KernelWorkerStatus.COMPLETE`で、outputに`submission.csv`を確認した。
- 提出前のcompetition submission一覧にはref `56508119`は存在しなかった。`make submit-code KERNEL=kentookumura/exp043-x138-self-trained-head-inference KERNEL_VERSION=2 OUTPUT_FILE=submission.csv MESSAGE='exp043 self-trained coordinate head x138 inference v2'`を実行し、提出後の一覧で新規ref `56508119`を一意に確認した。Kaggle記録の提出日時は2026-09-24 01:18:41 UTC（10:18:41 JST）、statusは`PENDING`。公式scoreは未確定。
- ref `56508119`を固定して`kaggle-submit-monitor`の監視を開始した。一時ログは`artifacts/submission-monitor.log`に保存し、Gitには含めない。公開4動画のNotebook実行時間と、提出後の採点所要時間は別に記録する。

- Kaggleのsubmission一覧でref `56508119`が`SubmissionStatus.COMPLETE`、Public LB `0.950`、Private LB未表示と確認した。最終pollは2026-09-24 07:44:18 UTC（16:44:18 JST）、直前の07:39:17 UTC（16:39:17 JST）は`PENDING`だった。score確定時刻はこの約5分の区間内。提出時刻から初回`COMPLETE`確認までは6時間25分36秒、監視開始からの記録値`scoring_elapsed_minutes=384`はNotebook実行時間ではない。
- `make record-exp EXP=exp043_x138_self_trained_head PUBLIC_LB=0.950`で`metrics.json`と`experiment_summary.md`を更新した。次に`make record-submission EXP=exp043_x138_self_trained_head SUBMISSION=experiments/exp043_x138_self_trained_head/artifacts/inference_v2/submission.csv SUBMISSION_REF=56508119`で`SUBMISSIONS.md`のv004へ記録した。local CSVは公開4動画の出力であり、採点時のNotebook再実行出力そのものは未取得。

## 2026-09-24 採用判断と比較指標の監査

- ユーザーがexp043を採用すると判断したため、`make record-exp EXP=exp043_x138_self_trained_head STATUS=usable`で実験statusを`usable`へ変更した。
- train version 2の`coordinate_validation.json`に保存された20動画・10,894対応点の動画別件数と平均距離から、全体の対応点加重平均と5-fold別平均を再計算して`metrics.json`へ追記した。fold割当は実行済みNotebookの規則（各胚で`SHA256(42:video)`順、5foldへ順番に配分）を再適用した。GPU再実行や新しい精度測定はしていない。
- 比較に使える証拠は全体・胚別・fold別・動画別の対応点数と補正前後の平均中心距離、Public LB、動画選択と分割規則、checkpoint・OOF内容・公開test CSVのSHA。点ごとのOOF予測値は保存されておらず、SHAだけがある。誤差分布、点ごとの対応比較、接続・分裂指標、採点時の動画別scoreは未取得。
