# exp025_kalman_hungarian_links セッションノート

## 目的

承認済みの`kalman_hungarian_links`を実験へ移し、固定分裂・固定画像得点の通常割当でカルマンの有無を比較できるようにする。

## 作業ログ

- 2026-09-20: ユーザーの「それでいいです」により校正、track継承、分裂予約の保持方式が確定。`make new-exp EXP=exp025_kalman_hungarian_links`で雛形作成。環境に`task`がないためMakeを使用した。sandbox launcherにbwrapがなく、shell操作は通常環境で必要な範囲に限定して実行した。
- 2026-09-20: 親の候補archive 40件と各receiptのSHAを照合。全199動画について修復後分裂を候補と照合し、予約専用の保存済み端点が必要なことを確認した。新しい検出・欠測点を生成せず両条件の固定入力とする契約をrequirementsへ追記した。
- 2026-09-20: 内部校正に他foldの保存得点を使うと外側胚を学習した重みが混ざるため、内部検証19動画だけを各foldの固定重みで再生する構成にした。outer 199動画は保存済み得点を読む。active比較2条件、outer方向2、内部gridは画像のみ3＋カルマン3、booster 0、追加学習0、control再学習なし、CPU既定。
- 2026-09-20: 6次元状態、6×6共分散、Joseph更新、log determinant込みの観測費用、連結成分ごとの対応なし付き割当、分裂娘への速度継承を実装。通常候補のhard gateは追加せず、大きすぎる成分は辺を削らず停止する。
- 2026-09-20: 単体テストで分散推定中のdefaultdict変更を検出して修正。公式集計の`score`/`adj_edge_jaccard`と表示名の対応、cache summaryのpayload SHAとfile SHAの違いを既存sourceで確認して修正した。
- 2026-09-20: `make check-exp EXP=exp025_kalman_hungarian_links`と`make test-exp EXP=exp025_kalman_hungarian_links`を実行。具体的な最終件数と証拠はmetricsへ記録する。
- 2026-09-20: 次のコマンドで両胚各8frameの実入力smokeを実施。未校正の数値fixtureを使い、公式評価なし。固定入力、予約、支持集合、有限状態の検査が成功した。

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --offline python experiments/exp025_kalman_hungarian_links/smoke_saved_graphs.py --archive-root experiments/exp016_frozen_image_encoder/artifacts/colab_runs/cache_replay_cli_v2/full/batch_archives --reference-root /tmp/kaggle-output/exp016-official-eval-v5-error/oracle_final_graphs --output experiments/exp025_kalman_hungarian_links/artifacts/local_smoke --frames 8
```

## Notebook構成

親exp016にcompact self-contained版はない。親の正規inferenceは別moduleとexp015固定sourceを組み合わせる構成だった。本実験は必要なcache読み取り・tracker・offline依存関数だけを抽出し、Imports、入力、状態推定、割当、校正、評価、出力を8節にまとめた。setup、校正、outer評価、集計は独立した実行セルに展開した。Notebook内で同じ実験のhelperをimportせず、`__file__`にも依存しない。

## 次の実行

初回のフル実行と公式評価はKaggleで行う。CPU設定と小規模実測の12時間gateを確認し、実行開始時に`kaggle-platform`の手順に従う。

```bash
make prepare-kaggle-notebooks EXP=exp025_kalman_hungarian_links EXTRA_ARGS="--notebook diagnostic --run-on-push"
make push-kaggle-notebook EXP=exp025_kalman_hungarian_links NOTEBOOK=diagnostic
```

この2コマンドは今後の実行用。submission承認を意味しない。内部校正・公式評価が終わったら、比較2条件と未変更exp016、胚別指標、費用、予約検査を整理してユーザーの実験完了・採否判断を求める。

## 最終の検証記録

- 2026-09-20: 入力path最終確認で、主催者annotationが`train/<sample>.geff`であることを親train sourceから確認し、予測graphを拾わない専用resolverへ修正。公開repo/checkpointのzip形式にも対応し、SHAを確認する。
- 2026-09-20: 最終の`make test-exp`は19件成功、両胚各8frame×2条件のsmoke成功。`make check-exp`、strict `validate-exp`、`check-strategy-docs`、Jupytext round-tripを検証。Notebookは8節に加えsetup・校正・outer評価・集計の実行セルを持つ。親compact版は存在せず、親のmodule委譲構成に対して必要な実行経路を自己完結で記載した。
- 2026-09-20: `make prepare-kaggle-notebooks EXP=exp025_kalman_hungarian_links EXTRA_ARGS="--notebook diagnostic"`でprivate CPU・internet無効・`run_on_push=false`のpackageを生成。今回は実装と非実行packageの準備までであり、Kaggle push・公式評価は未実行。

## Kaggle実行

- 2026-09-20T12:14:34+09:00: ユーザーの「実行してください」を受け、初回フル診断を開始する。生成metadataはprivate CPU、enable_gpu=false、enable_tpu=false、machine_shape指定なし、internet=false、run_on_push=true。CPUなのでGPU quota確認は不要。CLIではActive Sessions数を取得できないため事前gateにしない。既存sessionは停止せず、競合エラーが出た場合だけ対応する。認証確認とpackage検証は成功。
- 2026-09-20 12:14 JST: canonical kernel `kentookumura/exp025-kalman-hungarian-links-diagnostic` のversion 1 pushが成功。pullでid_no=135048120、private CPU、GPU/TPU/internet無効を確認。live SSEでbootstrapとoffline依存のsetup開始を確認した。metricsはrecord-expでrunningと実行先を記録した。
- 2026-09-20: v1はsetup中の`checksum-pinned archive missing: batch_000.zip`で停止した。Datasetの既存batchはKaggleが自動展開し、raw fallbackは全batchにはない。全40個の保存archiveを元のSHAで検証し、各候補graphの相対pathとfile SHAの正規化digestをinput_manifestへ追加した。Kaggleではreceipt SHAと候補graph内容SHAを確認して展開済み入力を読む。raw archiveの経路も維持する。画像/GEFFのchunkとcompetition全体をartifact探索から除外し、setupの進捗printを追加した。モデル・費用・評価条件は変更しない。形式一致・改変拒否を含む21 testsとRuffが成功。
- 2026-09-20T12:29:08+09:00: v2 push前にCPU、GPU/TPU/internet無効、machine_shapeなしを再確認。CPU quota確認不要。Kaggle上のbatch_000.jsonを個別downloadし、manifestのreceipt SHAと一致した。v1 pullで同じcanonical kernelの存在は確認済み。
- 2026-09-20: 同じcanonical kernelへv2 push成功。pullでCPU設定と存在を再確認した。live SSEで公式評価sourceと全199候補graphの内容照合成功を確認。ソース・package SHAとpull metadataはartifacts/kaggle_v2に保存し、metricsへ参照値を記録した。
- 2026-09-20: v2は全199候補graph、固定特徴19,701窓のidentity、公式評価source、参照graph、checkpointのsetup照合を通過し、校正cellへ進んだ。追加学習0、image_only/kalmanの2条件、CPUがログへ出力された。公式scoreはまだ未計測。

## Kaggle結果の回収

- 2026-09-20T14:47:00+09:00: ユーザーから実行終了の報告を受け、version 2のliveログとJSON生成物を取得した。全199 OUTERと最終execution receiptを確認。収録された7,357.987280278秒はdiagnostic開始から集計までの計測値であり、submission採点時間ではない。
- `make kaggle-logs KERNEL=kentookumura/exp025-kalman-hungarian-links-diagnostic`を保存し、`uv run kaggle kernels output kentookumura/exp025-kalman-hungarian-links-diagnostic/2 -p experiments/exp025_kalman_hungarian_links/artifacts/kaggle_v2/output --file-pattern '^(diagnostic/(execution_receipt|official_metrics|diagnostics|predictions|reservations|partial_official_rows|candidate_input_layouts)\.json|diagnostic/fold_[01]/[^/]+\.json|metrics\.json|config\.yaml|__notebook__\.ipynb)$' --page-size 200 --quiet`で公式集計、校正、予約、予測receipt、部分集計、metrics、configを取得した。容量の大きいgraph/state NPZ本体は取得していない。
- 回収後の整合検証でconfig/input/split/校正分散/公式集計/予測receiptのSHA、199動画と398予測receipt、両条件の入力同一性、各foldの分離を確認。未変更exp016の全体・胚別4指標の最大絶対差は0。検証結果はartifacts/kaggle_v2/result_verification.jsonへ保存した。
- 両胚でcombined scoreとdivision Jaccardが低下し、成功条件は不成立。内部校正でも正のカルマン係数の全候補が画像費用のみを下回る。内部の分裂予約0件という制約もresultへ記録。公式スコア、時間、SHA、校正、診断集計をmetricsへ反映し、record-expでdebug_completedと条件付き公式CVを記録した。ユーザーの実験終了・採否判断はまだ受けておらず、再探索・submission・commit・pushは実行していない。
- 記録後にstrict validate-exp、実験文書reviewerのstrict検査、git diff --checkが成功。使用した固定tracker重みはfold別2個とsecondary 1個で、追加学習は0。結果照合の再現scriptはartifacts/kaggle_v2/verify_results.pyへ保存した。

## 実験の採否判断

- 2026-09-20T14:55:20+09:00: ユーザーが「不採用として完了としてください。commitとpushしてください」と明示した。公式combined scoreとdivision Jaccardが両胚で低下したため、結果に沿ってstatusを`discarded`へ更新した。変更をexp025関係に限定して現在の作業ブランチへcommit・pushする。
