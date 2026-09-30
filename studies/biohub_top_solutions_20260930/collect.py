"""Collect read-only Kaggle evidence without credentials or signed URLs."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from archive_kaggle_discussions import list_topics, write_listing, archive_topic

COMP = 'biohub-cell-tracking-during-development'
TOPICS = {744484, 744549, 744501, 744486, 744531, 744490, 744485, 744491, 744498}

def run(*args):
    return subprocess.run([sys.executable, '-m', 'kaggle', *args], check=True,
                          capture_output=True, text=True).stdout

topics = [row for row in list_topics(COMP, 'recent', 3) if row.get('id', '').isdigit()]
write_listing(topics, OUT / 'topic_listing.csv')
(OUT / 'retrieval.json').write_text(json.dumps({'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'competition': COMP, 'topic_pages': 3, 'selected_ids': sorted(TOPICS)}, indent=2) + '\n')
(OUT / 'leaderboard_top20.csv').write_text(run('competitions', 'leaderboard', COMP, '--show', '--page-size', '20', '-v'))
(OUT / 'own_submissions.csv').write_text(run('competitions', 'submissions', COMP, '-v'))
for topic in topics:
    if int(topic['id']) not in TOPICS:
        continue
    raw = run('competitions', 'topics', 'show', COMP, topic['id'], '--format', 'json', '--page-size', '200')
    (OUT / f"topic_{topic['id']}.json").write_text(raw)
    print(archive_topic(COMP, topic, ROOT / 'docs' / 'discussions', ROOT / '.agents' / 'skills' / 'kaggle-discussion-archive' / 'scripts' / 'html_to_discussion_md.py', False), flush=True)
    obj = json.loads(raw)
    print('json_schema', topic['id'], type(obj).__name__, list(obj)[:20] if isinstance(obj, dict) else len(obj), flush=True)
print('collected', len(TOPICS), 'topics', flush=True)
