# exp049_x138_edge_selection_diagnostic 実行記録

- 2026-09-26: ユーザーの「x138_edge_selection_diagnosticを実装してください」により実験化。exp047を親とし、対照1・候補拡張1、新規学習0、fold 0、booster 0、control再学習なし。
- ローカルにはexp047の縮小得点cache、固定ID集計、公式評価JSON、20動画のGEFFがある。exp047のILP・後処理の辺IDを含む段階別NPZは現時点でローカルにない。まず保存入力と候補再構成を確認し、同一性を照合できない段階の原因分類は保留する。
- 2026-09-26 07:53 UTC: Kaggle push前のmetadataを確認。`enable_gpu=false`、`enable_tpu=false`、`enable_internet=false`、competition inputとexp047 inference kernel outputを使用するCPU Notebook。GPU quota確認はCPUのため不要。slug/titleは`kentookumura/exp049-edge-selection-diagnostic` / `exp049 edge selection diagnostic`に揃えた。ローカルsmokeでは両胚各1動画の段階ID・通常ILP辺集合・目的値がexp047と一致した。
- 2026-09-26: Kaggle diagnostic version 1は起動直後に`ModuleNotFoundError: zarr`で停止。予測・正解固定ILP・公式評価には到達していない。exp047と同じsupport packを入力に追加し、オフラインwheelからZarrと依存を入れるセルを追加した。同じkernel slugで再実行する。
- 2026-09-26 07:57 UTC: version 2 push前のmetadataを再確認。CPU、TPUなし、internetなし。exp047 kernel outputとオフラインsupport packを入力に指定。CPUのためGPU quota確認は不要。
- 2026-09-26: Kaggle diagnostic version 2はZarr wheelの探索で同一ディレクトリを二重登録し、1件と判定できず起動前に停止。入力側にwheelが存在することはログで確認した。重複を除く修正を行った。予測・正解固定ILP・公式評価には到達していない。
- 2026-09-26 08:00 UTC: version 3 push前metadataを確認。CPU、TPUなし、internetなし。exp047 kernel outputとsupport packが接続され、GPU quota確認は不要。
- 2026-09-26: Kaggle diagnostic version 3はオフラインZarr導入後、exp047 kernel outputの固定パスが見つからず入力照合前に停止。kernel sourceのマウント位置を`receipt.json`と固定ID証拠から探索するよう修正した。正解固定ILPと公式評価には未到達。
- 2026-09-26 08:03 UTC: version 4 push前metadataを確認。CPU、TPUなし、internetなし。exp047 kernel outputとsupport packを入力に指定し、GPU quota確認は不要。
- 2026-09-26: Kaggle diagnostic version 4で入力ルートが`/kaggle/input/notebooks`配下と確認された。探索範囲からこの階層が漏れていたため、exp047 receiptを見つける前に停止。探索対象へ追加した。
- 2026-09-26 08:05 UTC: version 5 push前metadata確認。CPU、TPUなし、internetなし。exp047 kernel sourceとsupport packを指定。GPU quota確認は不要。
- 2026-09-26: Kaggle diagnostic version 5でexp047 output rootを`/kaggle/input/notebooks/kentookumura/exp047-x138-edge-candidates-inference/exp047_inference`と確認。元出力には縮小cache用`cache_manifest.json`がないため、元実行の`receipt.json`の動画別cache SHA・候補件数を使う分岐を追加した。入力照合前に停止し、正解固定ILPには未到達。
- 2026-09-26 08:08 UTC: version 6 push前metadata確認。CPU、TPUなし、internetなし。exp047 kernel sourceとsupport packを指定。GPU quota確認は不要。
- 2026-09-26: Kaggle diagnostic version 6のlive logsで、両胚各1動画のpilotにおける候補・段階別ID・通常ILP辺集合と目的値の照合通過を確認。残り18動画と正解固定ILPは実行中。
- 2026-09-26: Kaggle diagnostic version 6は`KernelWorkerStatus.COMPLETE`。CPU・internet無効で約1,790秒。exp047の段階別辺IDと件数を固定20動画すべてで照合し、両胚pilotで通常ILPの辺集合と目的値が一致。20動画の既知辺・2娘組を追跡し、正解固定ILPは11件すべて最適終了した。Kaggle outputのJSONを`artifacts/evidence/diagnostic_v6/`へ回収し、SHAを`metrics.json`へ記録した。公式scoreはexp047の保存済み値のみを参照。
- 2026-09-26: ユーザーの「完了しました」を実験の完了判断として記録。採用・不採用の判断、次の実験選択、competition submissionは含まれない。
