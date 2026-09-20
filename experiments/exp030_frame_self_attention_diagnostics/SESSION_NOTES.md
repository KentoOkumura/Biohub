# exp030_frame_self_attention_diagnostics セッションノート

## 2026-09-20 実装

- ユーザーの依頼を受け、追加の確認で「診断だけ先に実装」と範囲を確定。`frame_self_attention`候補そのものは移行しない。
- 予定する構成は現行、Model A、Model Bの3種類、各1 config、学習fold 0、booster 0。公開controlと新規構成の再学習なし。固定画像モデル、secondary tracker、graph復号は実行しない。
- 変更版モデルは旧module keyを保持。Self Encoderを両フレームで共有し、Model BのCross-Attentionは更新済みt側を逆方向に渡す。
- 診断Notebookは固定cacheから代表・最大windowを選び、旧checkpoint互換性、forward/backward時間とpeak GPU memoryを測る。backwardはsynthetic scalarで、精度指標ではない。
- 作業ツリーには他の未コミット変更があるため、exp030以外の変更を避ける。

## 次

1. 契約テストとJupytext同期を検証する。
2. `make validate-exp`、`make check-exp`、`make test-exp`を通す。
3. Kaggle実行前にquotaとActive Sessionsを確認し、診断Notebookをprepare・pushする。
4. 実測値を記録し、後続の学習と公式graph評価の判断材料を提示する。

## 2026-09-20 ローカル検証とKaggle実行の保留

- `make validate-exp EXP=exp030_frame_self_attention_diagnostics`、`make check-exp EXP=exp030_frame_self_attention_diagnostics`、`make test-exp EXP=exp030_frame_self_attention_diagnostics`、Jupytext round-trip、Markdownリンク確認に成功。ローカルPyTorchがないため、モデル動作のpytestは1件skip。Notebook埋め込みsourceと診断設定の契約テスト1件は通過した。PyTorchを使う互換性・mask・勾配の検証はKaggleで実行するセルにも含めた。
- Kaggleへのアップロードと`--run-on-push`を含むprepareがautomatic approval reviewに拒否された。理由は、実装依頼はあるが外部のKaggleへの送信・実行が明示承認されておらず、ローカル準備だけの代替があるため。拒否を迂回せず、pushしていない。
- 送信しないlocal packageは`kentookumura/exp030-self-attention-diagnostic`、private、T4 GPU、TPU・internet無効、`run_on_push: false`で準備した。実行承認があればKaggle quotaを再確認し、`run_on_push: true`でpackageを再生成・検証する必要がある。
- 共有作業ツリーに別の`exp031_frame_self_attention_spatial`が並行作成されている。戦略索引は両実験configの上位仮説対応に一致させ、`make check-strategy-docs`を通した。`frame_self_attention`の未着手候補はこの診断では削除していない。

## 2026-09-20 20:35 JST Kaggle実行前確認

- ユーザーが「実行してください」と明示し、先の外部送信・実行未承認は解消した。診断のみを実行し、学習・公式graph評価・submissionは行わない。
- `make validate-exp`、`make check-exp`、`make test-exp`が成功。後者は静的契約1件pass、ローカルPyTorch不足のモデル動作1件skip。Kaggle Notebook内でモデル動作を確認する。
- Kaggle quota表示はGPU使用15.99時間、残29.01時間、2026-09-26 00:00 UTC更新。内部上限の週30 GPU時間では残14.01時間。診断は現行・Model A・Model Bの3構成、各1 config、学習fold 0、booster 0、既存control再学習なし。cache走査、代表・最大windowのforward/backward数回なので、残量内と判断する。
- `make prepare-kaggle-notebooks`を`--notebook diagnostic --run-on-push`で実行。生成metadataはprivate、T4 GPU、TPU・internet無効、`run_on_push: true`、exp015固定cacheと公開support datasetを入力に指定。Kaggle CLIの自分のNotebook検索で同じslugは`Not found`だった。Active Sessions数はCLIで取得できず、skillの規則どおりpush前gateには使わない。

## 2026-09-20 20:46 JST Kaggle診断完走と証拠回収

- `make push-kaggle-notebook EXP=exp030_frame_self_attention_diagnostics NOTEBOOK=diagnostic`でprivate Notebook `kentookumura/exp030-self-attention-diagnostic` version 1をpush。Kaggleからpullしたmetadataの`id_no`は135097598で、T4、TPU・internet無効、指定入力を確認した。
- `make kaggle-logs`のlive logと`kaggle kernels status`で`KernelWorkerStatus.COMPLETE`を確認。`make kaggle-output`で`artifacts/kaggle_diagnostic_v1/`へJSONとログを回収した。診断JSONのSHA256は`23cec4571be0d8c462558d53bcde8cd44470327c98c58d3dcd516ec9e4fc3d15`で、Notebook stdoutの値と一致。回収したconfig、モデルsource、Jupytext sourceはローカルの正とbyte一致。
- 旧checkpointの`strict=True`読み込み後、現行モードのlogitsは完全一致。A/BのSelf Encoder初期state、保存復元、mask・空集合・順序のruntime guardを通過。代表windowは205×202 cell、最大windowは1,056×1,038 cell。3構成ともforward/backwardが完走し、OOM・非有限値・log内のruntime errorはなかった。
- Notebook本体の計時は494.37秒、うちcache identity走査99.83秒。Kaggle割当全体の経過時間を別途測っていないため、`metrics.json`の`notebook_runtime_seconds`は未取得のままにした。GPU quota表示は前15.99h、後16.32hでアカウント全体の差0.33h。他sessionの影響を含み得るため、このNotebook単体のGPU割当消費時間とはみなさない。週30hの内部上限までの残りは13.68h。
- JSONの必須field、3構成×2window×forward/backward各3回の有限な測定、学習0・公式score未計測・submissionなしを機械確認した。数値は`metrics.json`の`frame_self_attention_diagnostic`、解釈は`result.md`へ記録。実験の完了・採否、後続のフル学習はユーザー判断待ち。
- 記録更新時点で並行作業の`exp031_frame_self_attention_spatial`へ元候補が移行され、`backlog/frame_self_attention.md`は削除済み。診断側の根拠リンクをexp031の`requirements.md`へ更新した。exp031の学習・評価はこの診断には含めない。

## 2026-09-20 ユーザーの完了判断

- ユーザーが「exp030は完了としてgit commitとpushしてください」と明示した。診断の完了を `metrics.json` の `completed` に記録し、`result.md` に判断を反映した。
- Model A/Bの採否と接続精度はこの診断では未判断。学習・両胚別指標・公式graph評価は後続のexp031で扱う。
- exp030に関係するファイルだけをcommitし、現在の作業ブランチをpushする。共有作業ツリーの他実験・backlog・横断集計の未コミット変更は含めない。
