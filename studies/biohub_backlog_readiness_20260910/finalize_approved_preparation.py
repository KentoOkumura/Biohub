from datetime import datetime
from pathlib import Path

exp = Path('experiments/exp003_official_metric_audit')
now = datetime.now().astimezone().isoformat()
notes = exp / 'SESSION_NOTES.md'
text = notes.read_text().replace('- 2026-09-11: 2026-09-11:', '- 2026-09-11:')
text = text.replace('承認済みの公式評価照合を実装する。', '承認済みの公式評価照合は実装・ローカル検証・package準備済み。自動承認レビューがKaggleへの送信を拒否したため実行前で停止している。')
text = text.replace('Jupytext変換、strict検証、Ruff、対象テスト、CPU package準備、Kaggle push、結果回収と記録。', '具体的な送信内容と宛先への追加承認後、同じpackageのmetadata検証、Kaggle push、結果回収と記録。')
text = text.replace('Kaggle実行値。ユーザー判断の未決事項はなし。', '実験の技術方針は承認済み。自動承認レビューにより、下記の具体的な外部送信・CPU実行だけ追加の明示承認が必要となった。Kaggle実行値は未取得。')
text += f'\n- {now}: `make push-kaggle-notebook EXP=exp003_official_metric_audit NOTEBOOK=audit` はツールの自動承認レビューが実行前に拒否。理由は、非公開リポジトリの監査Notebook・設定・評価コードについて具体的なpayloadとKaggle宛先への明示承認がないこと、および宛先所有者を確認できないこと。Kaggleへの送信・実行は未実施。目的と推奨手順の承認は取り消されていない。\n'
text += '- 追加承認の対象は、生成済み `kaggle/audit/exp003_official_metric_audit_audit.ipynb`（固定fixture、公式/公開評価コード、実験設定を同梱）を、設定上のowner `kentookumura` のprivate Notebook `exp003-official-metric-audit-audit` へ送信し、CPUで実行する操作。約152 KiB。認証情報をpackageへ同梱する処理はなく、competition入力と既存予測はKaggle上のinput参照。GPU消費なし。\n'
notes.write_text(text)
result = exp / 'result.md'
text = result.read_text().replace('現在は実行前である。', '実装とローカル検証は終了したが、自動承認レビューが外部送信を拒否したためKaggle実行前である。公式採点の実行値はまだない。')
text = text.replace('CPU監査を実装・検証し、Kaggleで実行する。', '準備済みpackageの具体的な送信内容と宛先への明示承認後、CPU監査をKaggleで実行する。停止理由と対象は[SESSION_NOTES](SESSION_NOTES.md)を参照。')
result.write_text(text)
readme = exp / 'README.md'
text = readme.read_text().replace('次はCPUの監査NotebookをKaggleで実行する。', '実装・検証・package準備済み。自動承認レビューによる送信停止の解消後、CPUの監査NotebookをKaggleで実行する。')
text = text.replace('監査Notebook: `exp003_official_metric_audit_audit.ipynb`', '監査Notebook: [正のNotebook](exp003_official_metric_audit_audit.ipynb) / [送信するNotebook](kaggle/audit/exp003_official_metric_audit_audit.ipynb) / [送信先・公開範囲・resource](kaggle/audit/kernel-metadata.json)')
readme.write_text(text)
direction = Path('backlog/KAGGLE_DIRECTION.md')
text = direction.read_text().replace('| official_metric_check |', '| [公式評価照合の実行結果](../experiments/exp003_official_metric_audit/) |')
text = text.replace('状態は比較案と予算の合意前を表す検討メモであり、64件すべてがユーザーの技術判断待ちという意味ではない。', '2026-09-11に共通方針と推奨順序を承認済み。公式評価照合をexp003へ移し、未着手候補は63件。残る検討メモは測定・候補別条件が未充足のものを含み、すべてがユーザーの技術判断待ちという意味ではない。')
direction.write_text(text)
survey = Path('docs/surveys/biohub-backlog-readiness_20260910.md')
text = survey.read_text().replace('experiments:\n- exp001\n- exp002\n', 'experiments:\n- exp001\n- exp002\n- exp003\n')
survey.write_text(text)
print('Recorded automatic approval rejection; synchronized experiment and strategy references.')
