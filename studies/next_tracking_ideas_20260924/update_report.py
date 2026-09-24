"""Insert the dated reassessment while preserving the complete prior analysis."""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
report = root / 'docs/surveys/biohub-tracking-next-ideas_20260923.md'
addendum = Path(__file__).with_name('review_addendum.md').read_text()
original = report.read_text()
marker = '## 結論と優先順位\n'
assert original.count(marker) == 1
assert '## 2026-09-24: exp043' not in original
prefix, history = original.split(marker, 1)
old_summary = next(line for line in prefix.splitlines() if line.startswith('summary: '))
new_summary = 'summary: exp043の中心補正とPublic LBを受け12案を再評価。5案は維持し、同じ新入力の公開tracker対照と位置・接続の作用確認を先行。分裂比較学習は条件付き本命。exp040は旧実装の未達と損失修正後未測定を反映。'
prefix = prefix.replace(old_summary, new_summary, 1)
prefix = prefix.replace("date: '2026-09-23'", "date: '2026-09-24'", 1)
prefix = prefix.replace('hypotheses: [HYP-20260910-03,', 'hypotheses: [HYP-20260910-02, HYP-20260910-03,', 1)
prefix = prefix.replace('exp040, exp041]', 'exp040, exp041, exp043]', 1)
prefix = prefix.replace('- 作成日: 2026-09-23', '- 作成日: 2026-09-23\n- 更新日: 2026-09-24', 1)
prefix = prefix.replace('- 対応する上位仮説: `HYP-20260910-03`,', '- 対応する上位仮説: `HYP-20260910-02`, `HYP-20260910-03`,', 1)
# The migrated contract replaces only a dead link; all prior prose is retained.
history = history.replace('../../backlog/public_x138_replay.md', '../../experiments/exp042_public_x138_replay/requirements.md')
summary_history = '前回の要約: ' + old_summary.removeprefix('summary: ') + '\n\n'
updated = prefix + addendum + summary_history + marker + history
assert len(updated) > len(original)
assert updated.endswith(marker + history)
assert history == original.split(marker, 1)[1].replace('../../backlog/public_x138_replay.md', '../../experiments/exp042_public_x138_replay/requirements.md')
report.write_text(updated)
print('Inserted dated reassessment; previous analysis and summary retained.')
