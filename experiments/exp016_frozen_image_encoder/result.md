# exp016_frozen_image_encoder 結果

## 仮説

公開画像encoder、検出候補、secondary branch、decodeを固定し、primary `SimpleNodeTransformer`だけを再学習すれば、同じ評価条件でtracker変更を比較する基準を作れる。これとは別に、今回の再学習だけで精度が改善したと判断するには、train段階の接続指標だけでなく、後続の固定graph推論で両胚の公式combined scoreが改善することを必要とする。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- CV/LB、実験status、kernel情報、Kaggle Notebook実行時間、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

設定とSHAをこのファイルへ独立した記録として転記しない。CV・Public LB・Private LBは`metrics.json`を数値の正本とし、比較や乖離の説明に必要な場合は、参照キーを添えて同じ値を本文で使用してよい。

## 実行証拠

- 比較対象: 同じ公開tracker初期値と、固定featureから再学習したfold別tracker。画像側とdecodeは比較間で固定する。
- `metrics.json` の参照キー: `status=completed`、`evidence.kaggle`、`evidence.colab_smoke`、`evidence.colab_benchmark`、`evidence.colab_full_replay`、`evidence.runtime_benchmark`、`evidence.kaggle_runs`、`evidence.output_files`、`evidence.official_evaluation_stage`、`official_graph_evaluation`、`train_stage`。Kaggleのprivate T4 kernel `kentookumura/exp016-frozen-image-encoder-train` version 3がtrain段階を正常完了し、Colab T4でcache replayの1動画smoke、2動画benchmark、199動画full replayを完了した。Kaggleのprivate CPU kernel `kentookumura/exp016-frozen-image-encoder-official-eval` version 6で、回収済み199 graphの公式評価を完了した。ユーザーの完了判断により、実験statusを`completed`とした。
- runtime gate: empty-GT除外後の64 window/fold benchmarkによる保守的予測は32,359.34秒（8.99時間）で12時間gate内だった。実際のtrain段階は3,379.48秒、Notebook全体は4,620.86秒（約1時間17分）。
- 入力契約: cache summary SHA、199 sample・19,701 window、公開sourceとcheckpoint、2fold splitの検査は通過した。公開sourceと同じ規則でempty-GTを含む994 windowを除外し、18,707 windowを学習・内部検証・外側評価に使用した。学習対象はprimary `SimpleNodeTransformer`だけである。
- 生成物: fold別tracker model 2件、fold summary、teacher audit、empty-GT監査、split manifest、benchmark summary、training summary、model manifestを取得し、後続inference用に`artifacts/train_v3/`へ保存した。file SHAと文書間の参照を相互検証し、submissionが生成されていないことを確認した。
- 外側胚指標: `train_stage.derived_outer_evaluation_aggregate`では、2fold件数集約の`selection_score`が0.96689044から0.96688577、`positive_edge_recall`が0.96558010から0.96605418、`division_parent_recall`が0.41538462から0.41538462となった。fold別の`positive_edge_recall`差は、6bba評価で+0.001789、44b6評価で-0.006702だった。
- 固定graph inference: version 3はfold別trackerによる199件のpredictionとgraph repairを完了し、候補座標・frame retention判断がexp015と完全一致することを確認した。ただし公開support evaluatorの存在しないentrypointをimportしたため、公式指標の計算前に失敗した。`metrics.json`の`evidence.kaggle.inference.runs`を実行証拠の正とし、失敗実行からgraph出力は回収できなかった。
- cache replay修正: raw画像経路version 3のprediction 23,217.36秒と、失敗時にgraphを回収できなかった構成を受け、後半をexp015の検出後cache 19,701 windowからtracker・ILP・repairだけを再生する実装へ変更した。main画像encoder forwardは0回とし、固定secondary trackerのforwardは公開・再学習primary間で共有する。公開trackerで復元したILP前candidate graphがexp015保存graphと全199件で完全一致することを公式評価前の必須条件にした。公式評価器はSHA・entrypoint・GT directoryだけでなく、保存済みcontrol graph 1件のend-to-end採点まで長時間処理前に検査する。12時間gateはNotebook開始からcache replayとgraph repairを含む。
- Colab smoke: 認証情報を含まない37,297,125 bytesの入力を一時Colab VMへ転送し、Tesla T4、Python 3.13.15、PyTorch 2.11.0+cu128で`44b6_0113de3b`の99 windowを実行した。cache replayは43.31秒、依存導入後の実行全体は50.17秒だった。26,532 nodeに対して公開trackerは25,534 candidate edgeを生成し、exp015保存graphとnode・edge・score・distanceまで完全一致した。fold 1の再学習trackerは25,623 candidate edgeを生成し、ILP graphを保存した。main画像encoder forwardは0回、tracker forwardは1 windowあたり5回である。1動画のreplay時間を199件へ単純外挿し25%を加えると2.99時間だが、正式な12時間gateには両胚2動画のbenchmarkが必要で、graph repairと公式評価時間もこの推定に含まれない。回収archive、batch archive、completion receipt、logのSHAをローカルで再検査し、Colab sessionを終了した。証拠は`metrics.json`の`evidence.colab_smoke`と[`artifacts/colab_runs/smoke_20260918/`](artifacts/colab_runs/smoke_20260918/)を正とする。
- Colab 2動画benchmark: 認証情報を含まない33,358,380 bytesの入力を一時Colab VMへ転送し、`44b6_0113de3b`と`6bba_05b6850b`を各99 window、合計198 window実行した。cache replayは55.66秒、依存導入を含む実行全体は75.73秒だった。公開trackerは44b6で25,534 edge、6bbaで6,440 edgeを生成し、両方ともexp015保存graphと完全一致した。fold 1とfold 0の再学習trackerはそれぞれ25,623 edgeと6,545 edgeを生成し、ILP graphを保存した。199動画への外挿は25% reserve込み6,922.18秒（1.92時間）で、tracker・ILP replayを対象とする12時間gateを通過した。外側・内側archiveのCRCとSHA、receipt、sample・fold対応をローカルで検査し、Colab sessionを終了した。証拠は`metrics.json`の`evidence.colab_benchmark`と[`artifacts/colab_runs/benchmark_20260918/`](artifacts/colab_runs/benchmark_20260918/)を正とする。この予測にはgraph repairと公式評価を含めない。
- Colab full replay: CLIだけでTesla T4へ接続し、Kaggle exp015 version 1から25,076ファイル・4,211,104,823 bytesのcache入力を直接取得した。事前benchmarkは2動画・198 windowを60.22秒で処理し、25% reserve込みの199動画予測7,489.29秒で12時間gateを通過した。最初のfull replayは199動画・19,701 windowを6,145.75秒（約102.4分）で完走し、main画像encoder forward 0回、tracker forward 5回/window、公開trackerのcandidate graph全199件完全一致を確認した。Colab session失効前に13/40バッチだけ回収できたため、その65動画をresume seedとして再利用し、残り134動画・27バッチを3,523.78秒で再計算しながらCLI標準出力経由で逐次回収した。最終的に40個・109,958,234 bytesのcandidate・ILP graph archive、40 receipt、199個の重複なしsample、19,701 window、fold 0の128動画とfold 1の71動画をローカルで照合し、全archive SHAと公開control一致を検証した。ブラウザ操作とGoogle Driveへの手動uploadは不要だった。証拠は`metrics.json`の`evidence.colab_full_replay`と[`artifacts/colab_runs/cache_replay_cli_v2/`](artifacts/colab_runs/cache_replay_cli_v2/)を正とする。このstageはgraph repairと公式評価を含まない。
- Kaggle公式評価: `official_eval` version 5が同じ固定処理で199件のgraph repairを完了し、version 6はそのgraphをSHA検証したprivate recovery Datasetから復元した。公開評価器のsource SHAと`evaluate_run`は維持し、指標に使わない画像tensorだけを`open_dataset(..., load_image=False)`で除外してCPU採点した。controlと再学習後を全199件で各2回採点し、再計算結果が一致、skipは0件だった。Notebook実行時間は1,647.52秒（約27分28秒）で、GPU、tracker forward、追加学習、graph repair再実行、submissionはいずれも0件である。証拠は`metrics.json`の`evidence.official_evaluation_stage`、`official_graph_evaluation`と[`artifacts/official_eval_v6/`](artifacts/official_eval_v6/)を正とする。
- ベンチマークの実行環境: Kaggle向け`exp016_frozen_image_encoder_inference.py/.ipynb`は、全199動画のcache replayから固定graph repairと公式評価までを一続きで再現できる正規実行ファイルとして保持する。今回の比較値は、Colab T4で全199動画のtracker・ILP replayを実行し、公開control candidate graphの完全一致を確認したうえで、Kaggle CPUのversion 5で固定graph repair、version 6で公式評価を行った分割実行から得た。各stageの入力と生成物はSHAで接続している。

## 解釈

固定画像featureからprimary trackerだけを2foldで学習し、公開trackerと同じ候補生成、secondary tracker、ILP、graph repair、公式評価で比較できる再学習基準とmodel 2件を残す目的は達成した。

公式combined scoreはcontrolの0.9117803563から再学習後の0.9120545013へ0.0002741450上昇した。ただし胚別では、6bbaが0.0008675982上昇した一方、44b6は0.0024463507低下した。全体でもadjusted edge Jaccardは0.0006600158、edge Jaccardは0.0003070454、node recallは0.0009862252低下し、combined scoreの微増はdivision Jaccardの0.0093416075上昇による。両胚のcombined score改善を要求した精度判定条件を満たしていないため、今回の再学習だけによる精度改善は確認できない。この判定は、再学習済みtrackerと評価手順を後続のtracker変更の比較基準として使うことを否定しない。

後続実験では、公開trackerを外部の比較対象として残し、この実験のfold別再学習trackerと固定graph評価結果を、学習方法やtrackerの変更を測る基準として使う。今回の結果だけで提出用モデルの選択は行わない。

## ユーザー判断

- 判断: 2026-09-19、ユーザーの「git commitとpushしてください。次に進めるアイデアを教えてください」を、この比較ベンチマーク実験の完了判断として記録した。fold別再学習trackerと固定graph評価は後続のtracker変更の比較基準として使用する。提出用モデルの選択は行っていない。`requirements.md`にあるKaggle正規inference Notebookの単一実行完走は未達であり、今回の判断はSHAで接続したColab replay、Kaggle repair、Kaggle公式評価の分割実行を対象とする。
- 理由: 実装、train、全199動画の固定graph推論、公式評価は完了し、後続のtracker変更に使う比較基準を残す目的は達成した。今回の再学習だけによる両胚での精度改善は確認できなかったが、実験完了の判断とは分ける。

## 次

この実験のfold別再学習tracker、学習手順、固定graph評価結果を後続のtracker改善の比較基準として使用する。次の変更案は別実験で同じ固定条件と公式評価を使って比較する。提出用モデルの選択とsubmissionはこの実験では行わない。
