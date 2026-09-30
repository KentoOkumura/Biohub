"""Verify source coverage and preserve the initial private-score comparison."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from bs4 import BeautifulSoup

OUT = Path(__file__).resolve().parent
snapshot = json.loads((OUT / 'final_leaderboard.json').read_text())
rows = snapshot['rows']
team_topics = {'yu4u': 744484, 'Tang': 744549, 'Corwin': 744501, 'Vibes & Edges Trade-Off': 744486, 'ymg_aq': 744531}
comparison = []
for rank, row in enumerate(rows, 1):
    if row['teamName'] in team_topics:
        topic_id = team_topics[row['teamName']]
        topic = json.loads((OUT / f'topic_{topic_id}_full.json').read_text())
        assert topic_id == topic['id']
        comparison.append({'rank': rank, 'team': row['teamName'], 'private_lb': float(row['score']), 'topic_id': topic_id, 'topic_title': topic['title']})
with (OUT / 'own_submissions.csv').open() as handle:
    own = list(csv.DictReader(handle))
refs = {x['ref']: x for x in own}
chosen = {'exp013': '56199738', 'exp042': '56569806', 'exp043': '56508119', 'exp050_gnn': '56638386', 'exp050_optuna': '56658424', 'exp050_old_cost': '56662549', 'exp052': '56675101', 'exp053': '56663354'}
ours = [{'experiment': exp, 'ref': ref, 'public_lb': float(refs[ref]['publicScore']), 'private_lb': float(refs[ref]['privateScore'])} for exp, ref in chosen.items()]
notes = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'top_solutions': comparison, 'our_submissions': ours, 'inferences': {'exp053_minus_exp043': {'public': round(refs['56663354'] and float(refs['56663354']['publicScore']) - float(refs['56508119']['publicScore']), 3), 'private': round(float(refs['56663354']['privateScore']) - float(refs['56508119']['privateScore']), 3)}, 'best_private_among_all_scored_submissions': max(float(x['privateScore']) for x in own if x.get('privateScore'))}, 'limits': ['Rank order is a snapshot and can change after final audit.', 'Full downloaded CSV is the public leaderboard, while get_leaderboard returns current final scores.', 'Current report adds API observations; existing experiment metrics and user adoption decisions are unchanged.']}
(OUT / 'comparison_evidence.json').write_text(json.dumps(notes, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(notes, ensure_ascii=False, indent=2))
for item in json.loads((OUT / 'archive_inventory.json').read_text()):
    raw = json.loads((OUT / f"topic_{item['id']}_full.json").read_text())
    md = (OUT.parents[1] / item['archive']).read_text()
    soup = BeautifulSoup(raw['content'], 'html.parser')
    image_count = len(soup.find_all('img', src=True))
    table_count = len(soup.find_all('table'))
    assert 'ARCHIVEBLOCK' not in md
    assert len(md.splitlines()) > 10
    print('source_check', item['id'], 'tables', table_count, 'images', image_count, 'markdown_images', md.count('!['))
