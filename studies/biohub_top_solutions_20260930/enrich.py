"""Retain full public topic HTML, comments, tables, links and image references."""
import csv
import html
import importlib.util
import io
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from archive_kaggle_discussions import slugify

spec = importlib.util.spec_from_file_location('discussion_converter', ROOT / '.agents/skills/kaggle-discussion-archive/scripts/html_to_discussion_md.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)
api = KaggleApi()
api.authenticate()
ids = [744484, 744549, 744501, 744486, 744531, 744490, 744485, 744491, 744498]


def converted(content):
    soup = BeautifulSoup(content, 'html.parser')
    links = []
    for anchor in soup.find_all('a', href=True):
        anchor['href'] = urljoin('https://www.kaggle.com/', anchor['href'])
        links.append(anchor['href'])
    for image in soup.find_all('img', src=True):
        src = urljoin('https://www.kaggle.com/', image['src'])
        image.replace_with(NavigableString(f"\n![{image.get('alt', 'source image')}]({src})\n"))
    for table in soup.find_all('table'):
        rows = [[cell.get_text(' ', strip=True).replace('|', '\\|') for cell in row.find_all(['th', 'td'], recursive=False)] for row in table.find_all('tr')]
        rows = [row for row in rows if row]
        if rows:
            width = max(map(len, rows))
            rows = [row + [''] * (width - len(row)) for row in rows]
            lines = ['| ' + ' | '.join(rows[0]) + ' |', '| ' + ' | '.join(['---'] * width) + ' |']
            lines += ['| ' + ' | '.join(row) + ' |' for row in rows[1:]]
            table.replace_with(NavigableString('\n\n' + '\n'.join(lines) + '\n\n'))
    for heading in soup.find_all(re.compile(r'^h[1-6]$')):
        heading.replace_with(NavigableString('\n\n' + '#' * int(heading.name[1]) + ' ' + heading.get_text(' ', strip=True) + '\n\n'))
    return converter.convert(str(soup)), links


def save_comment(comment):
    return {key: str(getattr(comment, key)) if key == 'post_date' else getattr(comment, key) for key in ['id', 'author_name', 'post_date', 'votes', 'content']} | {'replies': [save_comment(x) for x in (comment.replies or [])]}


def comment_md(comment, depth=0):
    body, _ = converted(comment['content'])
    result = '\n\n' + '#' * min(6, 3 + depth) + f" {comment['author_name']} — {comment['post_date']}\n\nVotes: {comment['votes']}\n\n" + body
    for reply in comment['replies']:
        result += comment_md(reply, depth + 1)
    return result

inventory = []
for topic_id in ids:
    topic, comments, next_page = api.forums_topic_show(topic_id)
    assert not next_page
    raw = {key: str(getattr(topic, key)) if key == 'post_date' else getattr(topic, key) for key in ['id', 'title', 'author_name', 'post_date', 'votes', 'comment_count', 'content']}
    raw['comments'] = [save_comment(x) for x in comments]
    (OUT / f'topic_{topic_id}_full.json').write_text(json.dumps(raw, ensure_ascii=False, indent=2) + '\n')
    body, links = converted(topic.content)
    slug = f'{topic_id}-{slugify(topic.title)[:80]}'
    dest = ROOT / 'docs/discussions' / f'biohub-cell-tracking-during-development-{slug}.md'
    text = f"# {topic.title}\n\n- archived_at: {datetime.now(timezone.utc).isoformat()}\n- source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/{topic_id}\n- author: {topic.author_name}\n- posted_at_utc: {topic.post_date}\n- votes: {topic.votes}\n- comments: {topic.comment_count}\n\n{body}\n"
    if comments:
        text += '\n## Comments\n'
        text += ''.join(comment_md(x) for x in raw['comments']) + '\n'
    dest.write_text(text)
    inventory.append({'id': topic_id, 'title': topic.title, 'author': topic.author_name, 'archive': str(dest.relative_to(ROOT)), 'links': links, 'comment_count_retrieved': len(comments), 'raw_content_chars': len(topic.content)})
    print(json.dumps(inventory[-1], ensure_ascii=False), flush=True)
(OUT / 'archive_inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n')

with zipfile.ZipFile(OUT / 'biohub-cell-tracking-during-development.zip') as z:
    print('leaderboard_members', z.namelist())
    for name in z.namelist():
        if not name.endswith('.csv'):
            continue
        raw = z.read(name).decode('utf-8-sig')
        (OUT / 'leaderboard_full.csv').write_text(raw)
        rows = list(csv.DictReader(io.StringIO(raw)))
        print('leaderboard_columns', list(rows[0]))
        print('leaderboard_top20', json.dumps(rows[:20], ensure_ascii=False))
        print('own_leaderboard', json.dumps([row for row in rows if any('kentookumura' in str(v).lower() or 'kento okumura' in str(v).lower() for v in row.values())], ensure_ascii=False))
        print('total_teams', len(rows))
