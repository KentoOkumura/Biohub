---
name: kaggle-submit-monitor
description: "Kaggle submission refを固定した採点監視と結果記録、コンペ終了・Private LB公開後の提出結果照合に使う。特に採点の長いcode competitionで、経過時間、LB、実験とsubmission refを対応付けた履歴を記録する。"
---

# Kaggle 提出監視

同梱スクリプトで Kaggle の提出状況を polling する。pollingログは一時生成物であり、Gitへ保存しない。最終的な提出事実は実験記録と`SUBMISSIONS.md`へ残す。

## 手順

1. コンペの slug を特定する。
   - ユーザーが明示した slug を優先する。
   - 明示指定がなければ、リポジトリルートの`project.yml`にある`competition.slug`を使う。未設定なら停止し、Kaggle CLIの既定configから暗黙に補完しない。
2. submit前の提出一覧と、submit後の提出一覧またはsubmit結果を比較し、今回作成されたsubmission refを一意に特定する。message、提出日時、kernel id/versionを照合し、候補が複数ある場合は停止する。単に一覧の先頭を対象にしない。
3. 特定したrefを`--submission-ref`で固定してmonitorを開始する。Kaggle API にアクセスするため、Codex tool では最初から `sandbox_permissions: "require_escalated"` と短い justification を付ける。

```bash
uv run python .agents/skills/kaggle-submit-monitor/scripts/monitor_submission.py EXP_NAME \
  --submission-ref SUBMISSION_REF
```

`project.yml`とは別のコンペを監視する場合だけ`--competition COMPETITION`で上書きする。

4. scoring が長い job では、プロジェクトルートから `nohup` で実行する。実験名が`experiments/`配下のディレクトリ名と一致する場合、既定の監視ログは無視対象の`experiments/<exp>/artifacts/submission-monitor.log`へ保存される。別名で監視するときは`--log-file`で対象実験の`artifacts/`を明示する。

```bash
nohup uv run python .agents/skills/kaggle-submit-monitor/scripts/monitor_submission.py EXP_NAME \
  --submission-ref SUBMISSION_REF \
  --log-file experiments/EXP_NAME/artifacts/submission-monitor.log \
  >/tmp/submission_EXP_NAME.runner.log 2>&1 &
```

5. 一時ログのパスと`tail -f`コマンドを報告する。一時ログをGitへ追加しない。
6. スコアが確定したら、まず`task record-exp EXP=expXXX SUBMISSION_REF=... PUBLIC_LB=... EXTRA_ARGS="--submission-status complete"`（Private LB判明後は`PRIVATE_LB=...`も指定）を実行する。次に`task record-submission EXP=expXXX SUBMISSION=/path/to/submission.csv SUBMISSION_REF=...`を実行し、同じrefは既存行を更新する。code competitionのoutputをローカル取得していない場合は、Kaggle側で対象ファイルを確認してから`SUBMISSION=submission.csv EXTRA_ARGS="--allow-missing-file"`を指定する。submission ref、提出日時、scoring status、score確定までの所要時間の詳細な時系列は`SESSION_NOTES.md`を正とし、`SUBMISSIONS.md`には横断比較に必要な最終スナップショットだけを記録する。スナップショットのキーは`submission_status`と`scoring_elapsed_minutes`を使い、Kaggle Notebook実行時間とsubmission scoring所要時間を混同しない。結果の解釈は`result.md`へ記録する。

採点が失敗・取消で終了した場合も、そのrefと失敗理由を`SESSION_NOTES.md`へ記録する。
`record-exp`へ同じ`SUBMISSION_REF`と`EXTRA_ARGS="--submission-status <取得した失敗状態>"`を渡し、
`PUBLIC_LB=null PRIVATE_LB=null`を指定して失敗状態と未採点の値を保持する。
その後、`record-submission`で同じrefを記録する。別refのスコアを補わない。

## コンペ終了・Private LB公開後の結果照合

終了後の結果整理を依頼された場合や、監視中の対象refにPrivate LBが公開された場合に使う。新しい提出や継続監視の予約は行わない。

1. `kaggle-platform`の読取手順で対象コンペの提出一覧・必要な詳細を取得し、`SUBMISSIONS.md`と各実験の提出別記録をsubmission refで照合する。終了後の全体整理では比較対象と採用候補も含め、未登録refは実験との対応を確認してから記録する。
2. 確認できたrefの最終採点状態とPublic/Private LBを上記手順6と同じ順序で更新する。Private LBだけが新たに判明した場合も同じrefへ`PRIVATE_LB`を渡し、既存の提出行を更新する。空欄・取得失敗を0や別refの値で補わず、既存の取得済み値と矛盾する場合は確認する。
3. 当時選んだ提出・採否と、Private公開後の分析を区別する。観測日と根拠を追記し、最良Privateの提出を当時の選択へ後から置き換えない。代表スコア、実験status、今後の比較対象も自動では変更しない。
4. 同じrefに対応する検証値・Public/Privateの関係を確認し、比較できる範囲と未取得の対照を示す。複数実験を横断した結論を保存する場合は`kaggle-strategy`と`docs/surveys/README.md`へ引き渡す。結果の取得自体を追加チューニングや採否変更の承認とみなさない。

## 出力契約

スクリプトは`--submission-ref`と一致する行だけを監視する。refが見つからなくても最新行へ切り替えず、見つかるまで待つ。実験の`artifacts/`または明示した`--log-file`へ一時ログを書き込み、同じ内容をstdoutにも出す。実験を特定できない場合は`/tmp/kaggle-submission-monitor/`へ保存する。完了してスコア付きの提出が検出された場合、次の形式になる。

```text
[EXP_NAME] scoring-elapsed: X min, submission-status: complete, publicScore: Y, privateScore: Z
```

正常完了では終了コード0を返す。`error`、`failed`、`notebook_unhandled_error`、
`runtime_limit_exceeded`、`cancelled` / `canceled`、または`errorDescription`を検出した場合は、
失敗状態と取得できた理由をログへ残して終了コード3で直ちに終了する。
`pending`などの処理中状態とref未検出では監視を続ける。`--once`で未完了の場合は1、
`--once`でのCLIエラーとrefの曖昧さは2、timeoutは124を返す。
通常監視中のCLIエラーはログへ記録し、timeoutまで再試行する。
`scoring-elapsed`はこの監視の開始からの経過時間であり、記録時の時間の扱いは`AGENTS.md`に従う。

ユーザーが明示的に依頼しない限り、代理で submit しない。このスキルは監視と記録だけを行う。
