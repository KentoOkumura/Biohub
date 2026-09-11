# exp005_embryo_holdout_batch8 結果

## 仮説

exp004のOOMが学習batch size 16のGPU memory要求によるなら、batch size 8へ下げる最小差分で2foldの3 epochs学習と全199動画の胚holdout推論を完走できる。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- CV/LB、実験status、kernel情報、Kaggle Notebook実行時間、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

設定とSHAをこのファイルへ独立した記録として転記しない。CV・Public LB・Private LBは`metrics.json`を数値の正本とし、比較や乖離の説明に必要な場合は、参照キーを添えて同じ値を本文で使用してよい。

## 実行証拠

- 比較対象: `exp004_embryo_holdout_baseline`のbatch size 16でのKaggle OOM。保存済みcontrolを再学習せず、失敗記録を参照する。
- `metrics.json` の参照キー: 実行前は`status=planned`、`cv=null`。Kaggle実行後に`evidence.kaggle`、`evidence.artifacts`、`evidence.reruns`を更新する。
- `SESSION_NOTES.md` の実行記録: 「コマンドログ」に実装、quota確認、push、live logs、output取得を時系列で記録する。
- 参照する生成物パス: 実行前はなし。学習完了時は`artifacts/models/`と`artifacts/model_manifest.json`、推論完了時は`artifacts/predictions/`、`artifacts/candidates/`、各評価JSONを参照する。

## 解釈

未実行。batch size以外の契約を固定したNotebookをKaggleで実行し、OOM回避、runtime gate、2foldのcheckpoint、199動画の予測と公式評価を順に確認する。

## ユーザー判断

- 判断: 未判断
- 確認日時 / 依頼メッセージ: 未判断。2026-09-11の「その方針で進めてください」は実験化とKaggle実行の承認であり、完了・採用判断ではない。
- 理由: Kaggle上の学習・推論結果を確認してから判断する。

## 次

静的検証後にtrain NotebookをKaggle T4で実行する。checkpoint 2個が保存された場合だけinference Notebookへ進み、competition submissionは行わない。
