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
- submission ref `56182729`は2026-09-12 13:18:05 UTCに採点完了し、Public LBは`0.693`、採点所要時間は231分だった。Private LBは未公表。

## 解釈

確率段階の2fold ensembleと単一ILP decodeは、公開testの4動画でOOMや整合性エラーなく実行できた。Public LB `0.693`はexp002の`0.453`を`0.240`上回り、保存済みexp006の両foldを確率平均して1回だけ復号する推論がhidden testで有効だったことを示す。ただしexp002とは学習条件と推論モデル数が異なるため、改善量をensemble単独の因果効果とは扱わない。bitwise再現性はexp010で別途検証する。

## ユーザー判断

- 判断: 未判断
- 確認日時 / 依頼メッセージ: 2fold ensembleの実装・実行は2026-09-12に承認済み。同日の「提出していいです」でexp009 version 1のcompetition submissionも明示承認された。
- 理由: 検証済みversion 1をsubmission ref `56182729`として提出し、Public LB `0.693`を確認済み。実験の採否はユーザー判断を待つ。

## 次

ユーザー指定どおり、exp006の再現性を担保する後継実験exp010を継続する。exp009の採用・保留・不採用はユーザー判断を待つ。
