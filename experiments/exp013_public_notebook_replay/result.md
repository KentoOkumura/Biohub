# exp013_public_notebook_replay 結果

## 仮説

固定した公開Notebookとcanonical artifactを同じKaggle環境で2回clean実行すれば、公開test全件の候補座標、graph topology、決定的なrun統計、`submission.csv`を同じ内容として再生成できる。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- CV/LB、実験status、kernel情報、Kaggle Notebook実行時間、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

設定とSHAをこのファイルへ独立した記録として転記しない。CV・Public LB・Private LBは`metrics.json`を数値の正本とし、比較や乖離の説明に必要な場合は、参照キーを添えて同じ値を本文で使用してよい。

## 実行証拠

- 比較対象: kernel id `133199516`、content SHAを固定した公開Notebook取得版と、canonical path・replay証拠だけを追加したexp013 inference Notebook。
- 実行先: private kernel [`kentookumura/exp013-public-notebook-replay-inference`](https://www.kaggle.com/code/kentookumura/exp013-public-notebook-replay-inference) のversion 1と2。両runともT4 2基、internet無効、同一docker imageで成功した。
- `metrics.json` の参照キー: `evidence.kaggle`、`evidence.artifacts`、`evidence.reruns`、`evidence.replay_comparison`。
- 予測部分の実測時間はversion 1が559.07967877388秒、version 2が546.079577922821秒。Kaggle CLIからNotebook全体の正確な実行秒数は取得できなかったため、`notebook_runtime_seconds`は未取得のままとし、pushからoutput取得までの観測区間約25分と約37分をrunごとの注記に残した。
- `compare_replays.py`で照合した13項目はすべて一致した。241,282行のraw `submission.csv`もbyte-identicalで、両runのSHA-256は`0319ba6d8e864335d3573f6b1a6227c546f17e9247a0c2858fa09b6c2422db3f`。
- `replay_receipt.json`全体のSHAは異なる。receiptに実測の予測時間を含めているためであり、予測時間を除いた比較対象のcandidate coordinate、graph topology、retention guard、決定的なrun統計、入力・wheel・checkpoint・source manifest、dataset一覧、submissionは一致した。
- 比較reportは`artifacts/replay_comparison.json`に保存し、そのSHA-256を`evidence.replay_comparison.report_sha256`へ記録した。
- version 2の`submission.csv`をcode submission ref `56199738`として提出した。現在は採点待ちで、CV、Public LB、Private LBは未取得である。

## 解釈

固定したsource・入力・依存・T4 2基・docker imageの条件では、公開test全件の候補座標、graph topology、決定的なrun統計、最終submissionを同じ内容として再生成できた。これにより、第1段階の再実行一致という成功条件は満たした。ただし確認範囲は同一Kaggle環境の2 runと公開test 4動画に限られ、別hardware、別docker image、hidden testでのbitwise一致や独立validationを保証しない。作者のtitleとhard-coded receiptにある0.946/0.947を本実験のPublic LBとして扱わない。

## ユーザー判断

- 判断: 未判断
- 確認日時 / 依頼メッセージ: 2026-09-13の実行結果を提示後に確認する。
- 理由: 2回のKaggle実行とoutput比較は成功したが、`completed`、`usable`、`discarded`の確定はユーザー判断を待つ。

## 次

submission ref `56199738`の採点完了を監視し、Public LBと採点所要時間を`metrics.json`、`SESSION_NOTES.md`、`SUBMISSIONS.md`へ記録する。その後、第1段階と提出結果を完了とするかユーザーへ判断を依頼する。完了判断後は、この実験に関係する変更だけをcommit・pushし、固定予測を`exact_window_cache`等の後続比較へ渡す。
