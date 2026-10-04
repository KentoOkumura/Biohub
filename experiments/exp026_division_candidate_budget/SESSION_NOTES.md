# exp026_division_candidate_budget セッションノート

## 目的

同じ窓別候補組数で距離順位と画像得点＋運動費用の既知分裂回収を比較する。

## 作業ログ

- 2026-09-20: ユーザーが `division_triplets` より本候補を先に実装すると指定。窓別同件数、距離対照、画像＋exp025運動、履歴なしの画像順位、後段を組モデル完成後に追加する方式を承認した。
- 2026-09-20: `task` が環境にないため `make new-exp EXP=exp026_division_candidate_budget SOURCE=experiments/exp020_division_triplet_candidates` で実験を作成。exp020の幾何・GT対応、exp025の必要な固定得点再生・状態計算をJupytext sourceへ取り込んだ。親コードの無関係な割当・公式評価処理は含めない。
- 2026-09-20: 同窓の比率10/25/50/100%を事前固定し、学習胚の距離対照がfull geometryの既知分裂回収90%以上を維持する最小比率を選ぶ。外側胚の正解は比率選択に使わない。
- 2026-09-20: 窓別順位と組一覧、exp025 state・校正と全候補のSHA確認、5動画の12時間予測gateを実装。母娘の運動費用は辺ごとに一度だけ計算し、同母の複数組で再利用する。
- 2026-09-20: `make check-exp`、`make test-exp`（4件）、`make validate-exp`は成功。Jupytext sourceからNotebookを生成し、round-tripを確認した。実際の199動画と公式指標は未実行。
- 2026-09-20: `backlog/division_candidate_budget.md`の仮説・根拠・判断をrequirementsへ移し、索引をexp026へ更新した。作業ツリーの他候補・実験にある既存変更は対象外。

- 2026-09-20: private・T4・internet無効・`run_on_push=false`のdiagnostic packageを`make prepare-kaggle-notebooks`で生成。exp025 Notebook outputのファイル一覧はまだ空で、完走receiptとfold別校正SHAはconfigへ未設定。未設定ならNotebookが停止する。
- 2026-09-20: exp025保存graphの各候補ID・時刻・座標をcacheと照合するguard、GPU peak memory、exp021のGT bundle SHA照合を追加。`make check-exp`と4件の対象テストは再実行して成功。
- 2026-09-20: `make check-strategy-docs`は別作業で作成中の`exp027_multi_frame_tracker`が索引未反映のためHYP-20260910-10で失敗。本実験のHYP-20260910-03の行は移行済み。別実験の索引はこの作業で変更しない。
- 2026-10-04: リポジトリ監査の指摘に従い、上流出力の待機記録を更新した。exp025のカルマン実験は全199動画を完走し、実行receiptと両foldの校正ファイルは取得・SHA照合済みと上流の`metrics.json`に記録されている。一方、状態のNPZファイルは未回収。本実験で使うKaggle入力との対応・状態内容の照合は未確認のため、configのreceipt・校正SHAは未設定を維持した。追加の取得、Kaggle実行、実験status・採否の変更は行っていない。

## Notebook構成

固定入力確認、画像得点とexp025状態、幾何候補と組順位、学習側の予算選択、Kaggle実行、sample別候補NPZとSHA manifest。新しい学習variant 0、fold別固定重み2件、外側2方向、booster 0、control再学習なし。

## 次の実行

exp025のカルマン実験の状態NPZがKaggle入力として利用可能か確認し、取得済みreceipt・校正のSHA、本実験が読む状態ファイル、候補ID・時刻・座標の対応を照合する。対応を確認後にconfigへreceipt・校正SHAを設定する。入力確認が終わるまでは初回フル実行を開始しない。実行時は現在の学習方針と`kaggle-platform`のquota・resource確認を行い、承認されたGPU予算とNotebook 12時間の制限内でprivate診断Notebookを実行する。submissionは行わない。
