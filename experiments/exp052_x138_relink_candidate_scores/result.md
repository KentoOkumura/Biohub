# exp052_x138_relink_candidate_scores 結果

## 仮説

同じILP graphと対照由来の再追加nodeで、再接続にILP未選択の保存済みtracker得点も渡すと親割当が改善する。

## 記録の参照先

- 設定、系譜、再現性方針: [config.yaml](config.yaml)
- 数値、実験status、Kaggle実行証拠: [metrics.json](metrics.json)
- 実行コマンドと途中経過: [SESSION_NOTES.md](SESSION_NOTES.md)

## 実行証拠

内部診断Notebook version 1はGPU不使用の訂正を受けて停止・削除した。各胚21位1動画ずつのCPU試行Notebookは準備済みだが、Kaggleのbatch CPU同時実行5件の上限でpushが拒否され、所要時間は未計測。学習動画の公式評価と進行条件も未計測。対照はexp043の固定tracker・ILP・後処理に相当する同一条件とする。

締切前の提出用GPU Notebook [version 3](https://www.kaggle.com/code/kentookumura/exp052-relink-scores-gpu-submit) は公開testの全動画で完走し、`submission.csv`を生成した。実行時間、GPU、行数、SHA、保存済み得点が参照された件数、提出形式チェックの結果は[metrics.json](metrics.json)を正とする。詳細な動画別証拠はローカル取得した`/tmp/kaggle-output/exp052_x138_relink_candidate_scores/submission_gpu_v3/exp052_submission/submission_receipt.json`、実行時系列と注意点は[SESSION_NOTES.md](SESSION_NOTES.md)を参照する。version 3のCSVはversion 1とSHAが一致した。

## 解釈

公開testでは全動画で保存済み得点が再接続の費用に参照され、提出CSVの形式チェックを通過した。初回提出はhidden dataset再実行時の未処理例外で採点できず、Kaggleは具体的な例外行を公開していない。公開testとhidden testの動画数差に対する時間gateが有力な原因である。version 3ではこのgateと時間の外挿を削除し、2基のGPUで動画を並列予測した。公開testの出力は初回と一致し、hidden datasetでの採点が完了した。ref `56675101`のPublic LBは`0.950`で、保存済み対照exp043のPublic LB `0.950`と同値。Private LBは`0.919`だが、exp043のPrivate LBは未記録なので同条件の比較はできない。学習動画の内部診断と割当変更は未判定であり、この範囲は採用判断の根拠に含めない。

## ユーザー判断

- 判断: 2026-09-30にユーザーが採用を明示
- 採否・完了: 採用・完了

## 次

本実験は採用・完了。学習動画の各胚21位のCPU試行と新規20動画評価は未実施であり、必要なら別途扱う。上位仮説`HYP-20260910-09`の残る候補はこの実験の完了だけでは判断しない。
