from datetime import datetime
from pathlib import Path

exp = Path('experiments/exp003_official_metric_audit')
now = datetime.now().astimezone().isoformat()
p = exp / 'result.md'
s = p.read_text().replace('送信・CPU実行は目的説明後の「実行してください」で明示承認済み。受け入れ基準を満たす実行証拠が揃ったので、この監査実験を完了扱いにすることを推奨する。完了・採否のユーザー判断は未記録であり、実行状態はmetricsを正とする。', '2026-09-11、結果提示後にユーザーが「完了してください。また、このセッションでの変更をgit commit, pushしてください」と明示。9種類の人工例と4動画の実行・証拠回収・解釈という受け入れ基準を満たした本監査実験を完了と判断した。実験statusの正はmetricsとする。これはモデルの改善・採用や上位仮説全体の支持・終了を意味しない。')
p.write_text(s)
p = exp / 'SESSION_NOTES.md'
s = p.read_text().replace('監査の実行と記録は終了。実験の完了判断はユーザーへ結果を提示する。', '監査の実行・記録とユーザーの完了承認は終了。このセッションの変更をcommit・pushする。')
s = s.replace('実験の完了判断はユーザー未回答。', 'ユーザーが2026-09-11に本監査の完了を承認。')
s += f'\n- {now}: ユーザー「完了してください。また、このセッションでの変更をgit commit, pushしてください」。本監査をcompletedとして記録し、調査・backlog・監査・監査用validator変更をcommit/pushする。別作業のexp002と実行生成物はcommit対象に含めない。\n'
p.write_text(s)
p = Path('backlog/KAGGLE_DIRECTION.md')
p.write_text(p.read_text().replace('実行証拠を回収済みで、完了判断は未記録。', '実行証拠を回収し、2026-09-11にユーザーが本監査の完了を承認。'))
print('Recorded user completion decision.')
