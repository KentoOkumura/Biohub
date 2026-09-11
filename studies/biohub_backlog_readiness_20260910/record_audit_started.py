import json
from pathlib import Path
from datetime import datetime

exp = Path('experiments/exp003_official_metric_audit')
now = datetime.now().astimezone().isoformat()
pulled = json.loads(Path('/tmp/kaggle-pull/exp003-official-metric-audit-audit-v1/kernel-metadata.json').read_text())
assert pulled['is_private'] and not pulled['enable_gpu'] and not pulled['enable_tpu']
assert not pulled['enable_internet']
(exp / 'artifacts/kaggle_metadata_v1.json').write_text(json.dumps(pulled, indent=2) + '\n')
notes = exp / 'SESSION_NOTES.md'
text = notes.read_text()
text = text.replace('自動承認レビューがKaggleへの送信を拒否したため実行前で停止している。', '具体的な送信と実行の承認後にKaggleへのpushが成功し、CPUで監査中。')
text = text.replace('具体的な送信内容と宛先への追加承認後、同じpackageのmetadata検証、Kaggle push、結果回収と記録。', 'Kaggleの監査ログを確認し、結果回収と記録を行う。')
text = text.replace('実験の技術方針は承認済み。自動承認レビューにより、下記の具体的な外部送信・CPU実行だけ追加の明示承認が必要となった。Kaggle実行値は未取得。', '実験方針と具体的な外部送信・CPU実行は承認済み。Kaggle実行値と実験結果の解釈を確認する。')
text += f'\n- {now}: version 1 push成功。pullしたmetadataのid_noは{pulled["id_no"]}、private・CPU・TPUなし・internetなしを確認。live SSEログでbootstrap開始を確認。`artifacts/kaggle_metadata_v1.json`に取得metadataを保存。\n'
notes.write_text(text)
path = exp / 'README.md'
path.write_text(path.read_text().replace('自動承認レビューによる送信停止の解消後、CPUの監査NotebookをKaggleで実行する。', '追加の実行承認後、非公開Kaggle Notebookへの送信に成功。CPU監査の結果を確認する。'))
path = exp / 'result.md'
text = path.read_text().replace('自動承認レビューが外部送信を拒否したためKaggle実行前である。', '追加の実行承認後にKaggleへの送信とCPU起動を確認した。')
text = text.replace('準備済みpackageの具体的な送信内容と宛先への明示承認後、CPU監査をKaggleで実行する。停止理由と対象は', 'CPU監査の結果を回収して評価差を記録する。実行の経緯は')
path.write_text(text)
print('Kaggle v1 metadata and execution start recorded.')
