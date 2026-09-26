# exp047_x138_edge_candidates セッションノート

## 目的

exp043の融合後の得点と全後処理を固定し、`p > 0.48`の全辺に`p > 0.10`かつ娘ごと上位3親の辺を追加した場合の候補回収と最終graphを比較する。学習側20動画で前段診断と2動画pilotを行い、条件成立後にhead学習外20動画の公式評価へ進む。

## 現在の作業

- 作業内容: 評価結果と実行証拠の記録を完了。2026-09-26にユーザーが実験を閉じた。
- ブロック要因: なし。
- 次: 別候補`x138_edge_selection_diagnostic`で段階別の原因を調べる。

## コマンドログ

- 2026-09-26: ユーザーの「実行してください」を実験化・Notebook実行の承認として受領。competition submissionは対象外。
- 2026-09-26: `make new-exp EXP=exp047_x138_edge_candidates`で実験を作成。元候補の契約、根拠、判断履歴を`requirements.md`へ移し、`config.yaml`へ上位仮説と候補名を設定。
- 2026-09-26: `candidate_policy.py`で対照全辺と各娘上位3親の和集合を実装。`make test-exp EXP=exp047_x138_edge_candidates`: 2 passed。exp045保存済み`44b6_7a302da0`のcacheでは対照40,960辺、追加31,280辺、拡張72,240辺。
- 2026-09-26: exp045の固定head推論・ILP・後処理を引き継ぐpilot Notebookを実装。学習側20動画の候補回収を先に診断し、両胚各1動画のみ対照再現・拡張ILP・後処理を実行する。
- 2026-09-26: `make validate-exp EXP=exp047_x138_edge_candidates`: pass。`make check-exp EXP=exp047_x138_edge_candidates`: pass。`make check-strategy-docs`: pass。
- 2026-09-26: `kaggle-strategy`の引き継ぎ規則に従い、`backlog/x138_edge_candidates.md`と未着手行を削除し、仮説の対応実験へexp047を追加。
- 2026-09-26: Kaggle GPU quotaは残り44.49時間。`make prepare-kaggle-notebooks EXP=exp047_x138_edge_candidates EXTRA_ARGS='--notebook pilot --run-on-push'`でpackageを作り、`make push-kaggle-notebook EXP=exp047_x138_edge_candidates NOTEBOOK=pilot`で`kentookumura/exp047-x138-edge-candidates-pilot` version 1を送信。
- 2026-09-26: 評価側20動画のexp045保存済みcacheを候補に必要な確率0.10超へ縮小し、1.2 GBから24 MBへ変換。元・変換後SHAと辺集合の一致をmanifestで検証。Kaggle datasetへのuploadは自動承認審査により、宛先とpayload共有の明示承認がないという理由で却下された。再試行せず、評価側Notebook内で得点を生成する設計へ変更。
- 2026-09-26: `uv run kaggle kernels pull kentookumura/exp047-x138-edge-candidates-pilot -p /tmp/exp047-pilot-pull -m`で送信先のmetadataを確認。private、T4、internet無効、指定4 datasetsとcompetition sourceが反映されている。
- 2026-09-26: pilot実行中に、solverのGurobi→SCIP fallback警告と非最適終了の区別、注釈済み娘の確定的な誤親判定を修正。version 2のpushは`Maximum batch GPU session count of 2 reached`でKaggleに拒否され、version 1が継続中。枠が空くまで再送信しない。
- 2026-09-26: version 1は20動画の得点取得5095.2秒と候補回収まで完走。既知edge追加は44b6で115本（3162→3277 / 両端対応3302）、6bbaで205本（7016→7221 / 両端対応7275）。`internal_candidate_diagnostic.json` SHA256 `38ce5521e88e1957635a3fb6e86e720b614670db3b05cc7f310efbfa955e316f`を回収し、20動画の選択・胚別件数・算術整合を確認。対照ILP replayで正常なGurobiライセンスfallbackを誤ってfail扱いしたためKaggle status `ERROR`。候補回収ゲート自体は通過。
- 2026-09-26: version 2は検証済みの20動画診断SHAを契約として保持し、固定各胚1動画だけを再推論する形へ修正。`make validate-exp`、`make check-exp`、`make test-exp`（3 passed）を確認。push直前GPU残41.36時間。`make push-kaggle-notebook EXP=exp047_x138_edge_candidates NOTEBOOK=pilot`で同じkernelのversion 2を送信。
- 2026-09-26: pilot version 2は`KernelWorkerStatus.COMPLETE`。`pilot_receipt.json` SHA256 `0885f65171c209056c770e95efe2de3ac00a30dc3cb32792330e80c5b0f261a2`を限定回収し、前段20動画診断SHAとの一致を確認。対照ILP replayは両動画で元graphと同じ選択edge、拡張ILPは両動画で最適終了（44b6 79.0秒、6bba 4.1秒）。確定的な誤親は4→4、0→0。後処理も両arm完走。最大メモリ7,178,678,272/32,212,254,720 bytes（22.3%）、Notebook736.3秒。全進行条件を通過したため評価側20動画へ進む。

- 2026-09-26: `make prepare-kaggle-notebooks EXP=exp047_x138_edge_candidates EXTRA_ARGS='--notebook inference --run-on-push'`で評価Notebookを準備。metadataはprivate T4、internet無効、公開support・temporal・deepcenterと固定exp043 head、competition sourceのみ。GPU quotaはpush直前に残り40.74時間、2026-10-03 00:00:00 refresh。既存cacheのuploadは行わず、Notebook内で固定得点を生成する。
- 2026-09-26: `make push-kaggle-notebook EXP=exp047_x138_edge_candidates NOTEBOOK=inference`で`kentookumura/exp047-x138-edge-candidates-inference` version 1を送信。Kaggle pull metadataを確認し、初期statusは`KernelWorkerStatus.RUNNING`。

- 2026-09-26: `make push-kaggle-notebook EXP=exp047_x138_edge_candidates NOTEBOOK=pilot`で固定2動画の公式評価追加版version 3を送信。送信先はprivate T4、internet無効。pilotと本評価の両kernelがRUNNING。

- 2026-09-26: Kaggle pullしたNotebookのcell typeとsourceを比較し、inference version 1は24 cells・SHA256 `6f96e9a0db6bdc1d76f1b48c315ee6500aa4a8541c4509ab4d9802a0671b9e27`、pilot version 3は18 cells・SHA256 `af05c4cb833c4c8e8d6d7008bc07b244ac63da6cbf6365aee785a05b8daaec64`で送信packageと一致。KaggleがNotebook metadataを書き換えるためファイル全体SHAは一致しない。

- 2026-09-26: pilot補足版version 3は`KernelWorkerStatus.COMPLETE`。限定回収した`pilot_receipt.json` SHA256 `2d3ca4f88e79c952e95164aa66c53eea2712e29c07273221abda27e28d89a388`、`pilot_official_metric.json` SHA256 `83927024e791be2646fc10fc94099c52cff59783f908b22a2ee0e8bcfe90d5f3`を照合。v2とILP候補数・選択辺・確定的誤親は一致し、solve秒数だけ変動。公式combined scoreは固定2動画で対照0.95303855、拡張0.95099798。候補回収増でもこの2動画の最終指標は改善しない。Notebook749.2秒、peak memory11,485,241,344/32,212,254,720 bytes。主判断は学習外20動画で行う。

- 2026-09-26: inference version 1は`KernelWorkerStatus.COMPLETE`。20動画各胚10本、head学習動画との重複0。receipt SHA256 `624f2894de140e817425a360f169e62d1cb5c1e68040a6411edc6befa16267ae`、official metric SHA256 `bce246fa9783db769ece67d9959c1a5939edb74c37f0df9157070219566ed59e`、固定ID診断SHA256 `c4627762f12376295e324991e099e4f0d506beff2d1eb67a584757031d44eb1f`を回収・照合。対照の公式評価20行はexp045 refinedと一致。全20動画の拡張ILPは最適終了、最大814.1秒。Notebook9847.0秒、peak memory22,391,865,344/32,212,254,720 bytes（69.5%）。公式combined scoreは44b6で0.935902→0.929943、6bbaで0.873577→0.859244、全体0.890348→0.878088。両胚の改善条件を満たさない。終了後GPU quota残35.84時間。小さなJSON証拠を`artifacts/evidence/`へ保存し、数値と解釈は`metrics.json`/`result.md`へ記録。採否・実験完了はユーザー判断待ち。

## 実行対象と費用

- active variant: baseline 1、expanded 1。新規学習model 0、fold 0、booster 0。controlの再学習なし。
- pilotはGPU上で固定headの得点を学習側20動画分取得し、ILPは両胚各1動画で対照の再現と拡張を解く。得点と座標は同一cacheから渡す。
- GPU quotaはpush直前に再確認する。課金は行わない。

## 実行中に記録した次のアクション（履歴）

1. head学習外20動画の評価Notebookを準備・実行し、両胚別の公式指標と各段階の既知edgeを回収する。
2. 対照と拡張側の結果・費用をユーザーへ提示し、採否と実験完了の判断を求める。
- 2026-09-26: pilot予備確認2動画の公式評価器による指標保存が契約から漏れていたため、固定2動画・同じ重み/候補/ILP/後処理を変更せず集計を追加。Jupytext同期、`make check-exp`、`make validate-exp`はpass。再push直前のGPU quota残40.59時間。評価側20動画のversion 1はRUNNING。

## 2026-09-26 完了判断

- ユーザーが「先ほど実行した実験は閉じでください。最後にcommitとpushしてください」と依頼。実験の完了判断を受領し、`metrics.json`のstatusを`completed`へ更新した。
- 候補拡張は両胚の事前成功条件を満たさず、現行exp043へ組み込まない。得点・ILP・後処理の切り分けは未着手候補`x138_edge_selection_diagnostic`へ引き継いだ。採否を`discarded`とする明示判断とcompetition submissionの指示は受けていない。
- 完了記録後の検証: `make validate-exp EXP=exp047_x138_edge_candidates` pass、`make check-exp EXP=exp047_x138_edge_candidates` pass、`make test-exp EXP=exp047_x138_edge_candidates` 3 passed、`make check-strategy-docs` pass。未使用の生成済みtrain Notebook雛形は実験に学習がないため除外した。
