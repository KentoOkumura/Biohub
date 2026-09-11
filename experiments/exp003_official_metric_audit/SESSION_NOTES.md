# exp003_official_metric_audit SESSION_NOTES

## 現在の作業

承認済みの公式評価照合は実装・ローカル検証・package準備済み。具体的な送信と実行の承認後にKaggleへのpushが成功し、CPUで監査を実行し結果回収済み。GPU使用なし、学習variant 0、model 0、config 1、fold 0、booster 0、control再学習なし。

## 時系列

- 2026-09-11: ユーザー『すべて推奨でいいです』。推奨の順序、無料枠内・課金なし、両胚改善、外部重みは選んだ追加候補のみ、人手注釈は当面なしを承認。FOCUSの取得同意は今回の対象外。
- 2026-09-11: exp002の最新記録とログから3 epochs完走・4動画推論済みと確認。公式評価照合へ既存CSVを再利用し、独立胚評価とは分ける。
- 2026-09-11: GPU quotaを読み取り、21.40h残と確認。CPU監査では消費しない。
- 2026-09-11: make new-exp EXP=exp003_official_metric_auditを実行。移行元候補をrequirementsへ保存し、現在の契約を具体化した。

## 実行予定

監査の実行・記録とユーザーの完了承認は終了。このセッションの変更をcommit・pushする。

## 残る事項

実験方針と具体的な外部送信・CPU実行は承認済み。結果と解釈はresultに記録済み。ユーザーが2026-09-11に本監査の完了を承認。上位仮説の後続検証は別作業として残る。

- 2026-09-11: 公式・公開関数のAST同一性を含む5 testが通過。既存validatorがtrain/inferenceを無条件要求していたため、experiment.notebooksを明示した監査実験を許容し、未指定の既存動作を維持する限定修正と回帰テストを追加した。

- 2026-09-11T09:21:13.411235+09:00: ローカル検証は対象5 tests、共通validator 21 tests、Ruff、strict validation、Jupytext roundtripすべて通過。
- Notebookは7節・1753行。親の正規self-contained trainは683行で、compact別版はない。監査に必要な公式関数、公開関数、入力検査、実行、集計をNotebook内に展開した。
- prepareの初回要求は自動承認レビューが外部送信と解釈して拒否。prepare実装に通信がなくローカル生成のみであることを確認し、その証拠を示した再実行は成功した。
- 2026-09-11T09:21:13.411235+09:00: push前metadataを確認。宛先 `kentookumura/exp003-official-metric-audit-audit`、private、CPU、TPUなし、internetなし。CPUなので直前のGPU quota再照会は不要。packageは監査Notebook、実験設定・固定fixture・保存した公式/公開評価コードからなり、送信内容のSHAを `artifacts/local_preparation.json` に記録。学習0、model 0、fold 0。

- 2026-09-11T09:22:45.181891+09:00: `make push-kaggle-notebook EXP=exp003_official_metric_audit NOTEBOOK=audit` はツールの自動承認レビューが実行前に拒否。理由は、非公開リポジトリの監査Notebook・設定・評価コードについて具体的なpayloadとKaggle宛先への明示承認がないこと、および宛先所有者を確認できないこと。Kaggleへの送信・実行は未実施。目的と推奨手順の承認は取り消されていない。
- 追加承認の対象は、生成済み `kaggle/audit/exp003_official_metric_audit_audit.ipynb`（固定fixture、公式/公開評価コード、実験設定を同梱）を、設定上のowner `kentookumura` のprivate Notebook `exp003-official-metric-audit-audit` へ送信し、CPUで実行する操作。約152 KiB。認証情報をpackageへ同梱する処理はなく、competition入力と既存予測はKaggle上のinput参照。GPU消費なし。

- 2026-09-11T09:27:44.709746+09:00: ユーザーが目的説明を確認後「実行してください」と明示。準備済みNotebook・設定・評価コードの kentookumura/exp003-official-metric-audit-audit への非公開送信とCPU実行を承認。push直前metadataはprivate、GPU false、TPU false、internet false、学習0、model 0、fold 0。CPUなのでGPU quota再照会は不要。

- 2026-09-11T09:30:25.590107+09:00: version 1 push成功。pullしたmetadataのid_noは133896771、private・CPU・TPUなし・internetなしを確認。live SSEログでbootstrap開始を確認。`artifacts/kaggle_metadata_v1.json`に取得metadataを保存。

- 2026-09-11T09:39:40.821676+09:00: version 1のAUDIT_SUMMARYとNotebook保存ログで実行終了を確認。`make kaggle-output KERNEL=kentookumura/exp003-official-metric-audit-audit/1 OUT=experiments/exp003_official_metric_audit/artifacts/kaggle_v1`で監査成果物を取得。全9人工例・4動画、失敗0。
- 取得後に `inspect_audit_outputs.py kaggle_v1` を実行し、manifest内6ファイルSHA、入力予測SHA、全対象件数を照合。実予測の対応・接続・分裂・得点が一致し、人工例の3スコアと重複辺の件数差を確認。CV/LBはnullを維持した。
- Kaggle内のmetricsと実行IDを突き合わせ、record-expで実行結果を記録。完了・採用・不採用は確定せずdebug_completedとして結果提示する。

- 2026-09-11T09:47:06.389348+09:00: ユーザー「完了してください。また、このセッションでの変更をgit commit, pushしてください」。本監査をcompletedとして記録し、調査・backlog・監査・監査用validator変更をcommit/pushする。別作業のexp002と実行生成物はcommit対象に含めない。

- 2026-09-11T09:51:40.290201+09:00: mainへのcommitは自動承認レビューが対象範囲とdefault branchへの明示承認不足を理由に拒否。対象134ファイルを再確認（backlog 64、調査文書4、調査コード・軽量証拠45、exp003 18、共通validator/test各1、実験summary 1）。exp002・生成物は除外済み。安全な代替としてcodex/biohub-baseline-research-metric-auditブランチを作成し、mainを変更せずcommit/pushする。
