# exp009_exp006_twofold_ensemble 結果

## 仮説

exp006の2foldモデルについて、検出確率を平均して共通nodeを抽出し、その共通node上のedge確率を平均してから1回だけ復号すれば、両胚由来の学習情報を使うhidden-test提出候補を生成できる。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- 構造化された実行証拠: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行証拠

- Kaggle inference version 1はT4 x2、internet無効で完了した。kernel、version、resource、Notebook実行時間は`evidence.kaggle`を正とする。
- 4件の公開testを動的に列挙して全件処理した。188,460行の`submission.csv`を生成し、106,237 node行、82,223 edge行だった。
- 両checkpoint、model manifest、submissionのSHAは`evidence.artifacts`を正とする。取得済みoutputは`/tmp/kaggle-output/exp009_exp006_twofold_ensemble/inference/`にある。
- repository submit-checkはPASSし、重複ID、欠損値、無限値はいずれも0だった。詳細は`evidence.submission_validation`を正とする。
- Kaggleのversion 1 outputに`submission.csv`が存在することを`kaggle kernels files`で確認した。

## 解釈

確率段階の2fold ensembleと単一ILP decodeは、公開testの4動画でOOMや整合性エラーなく実行できた。これはhidden testへ提出可能な形式と実行経路を確認した証拠であり、Public LB改善やbitwise再現性の証拠ではない。

## ユーザー判断

- 判断: 未判断
- 確認日時 / 依頼メッセージ: 2fold ensembleの実装・実行は2026-09-12に承認済み。同日の「提出していいです」でexp009 version 1のcompetition submissionも明示承認された。
- 理由: 検証済みversion 1をsubmission ref `56182729`として提出済み。採点中のため実験の採否はまだ判断しない。

## 次

submission ref `56182729`のscore確定後にLBと採点所要時間を記録する。並行して、ユーザー指定どおりexp006の再現性を担保する後継実験へ進む。
