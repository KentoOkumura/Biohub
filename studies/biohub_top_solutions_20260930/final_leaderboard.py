"""Read the final leaderboard, distinct from the public-only CSV download."""
import json
from datetime import datetime, timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetLeaderboardRequest

OUT = Path(__file__).resolve().parent
api = KaggleApi()
api.authenticate()
rows = []
token = None
with api.build_kaggle_client() as client:
    for page in range(10):
        request = ApiGetLeaderboardRequest()
        request.competition_name = 'biohub-cell-tracking-during-development'
        api._set_paging(request, 200, token)
        response = client.competitions.competition_api_client.get_leaderboard(request)
        rows += [x.to_dict() for x in response.submissions or []]
        token = response.next_page_token
        if any(x.get('teamId') == 16717777 for x in rows) or not token:
            break
snapshot = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'pages': page + 1, 'has_more': bool(token), 'scope': 'current final leaderboard as returned by get_leaderboard', 'rows': rows}
(OUT / 'final_leaderboard.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
print('final_top20', json.dumps(rows[:20], ensure_ascii=False))
for rank, row in enumerate(rows, 1):
    if row.get('teamId') == 16717777:
        print('own_final_rank', rank, json.dumps(row, ensure_ascii=False))
print('rows', len(rows), 'pages', page + 1)
