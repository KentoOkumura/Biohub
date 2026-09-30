"""Render cached original HTML while retaining tables and public image references."""
import importlib.util
import json
import re
import sys
from pathlib import Path
from urllib.parse import urljoin
from bs4 import BeautifulSoup, NavigableString

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from archive_kaggle_discussions import slugify
spec = importlib.util.spec_from_file_location('discussion_converter', ROOT / '.agents/skills/kaggle-discussion-archive/scripts/html_to_discussion_md.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def converted(content):
    soup = BeautifulSoup(content, 'html.parser')
    blocks = {}
    def block(tag, text):
        token = f'ARCHIVEBLOCK{len(blocks):05d}END'
        blocks[token] = text
        tag.replace_with(NavigableString(token))
    for anchor in soup.find_all('a', href=True):
        anchor['href'] = urljoin('https://www.kaggle.com/', anchor['href'])
    for image in soup.find_all('img', src=True):
        src = urljoin('https://www.kaggle.com/', image['src'])
        block(image, f"\n\n![{image.get('alt') or 'source image'}]({src})\n\n")
    for table in soup.find_all('table'):
        rows = [[cell.get_text(' ', strip=True).replace('|', '\\|') for cell in row.find_all(['th', 'td'], recursive=False)] for row in table.find_all('tr')]
        rows = [row for row in rows if row]
        if rows:
            width = max(map(len, rows))
            rows = [row + [''] * (width - len(row)) for row in rows]
            lines = ['| ' + ' | '.join(rows[0]) + ' |', '| ' + ' | '.join(['---'] * width) + ' |']
            lines += ['| ' + ' | '.join(row) + ' |' for row in rows[1:]]
            block(table, '\n\n' + '\n'.join(lines) + '\n\n')
    for heading in soup.find_all(re.compile(r'^h[1-6]$')):
        block(heading, '\n\n' + '#' * int(heading.name[1]) + ' ' + heading.get_text(' ', strip=True) + '\n\n')
    for li in soup.find_all('li'):
        li.insert(0, NavigableString('- '))
    text = converter.convert(str(soup))
    for token, content in blocks.items():
        text = text.replace(token, content)
    text = re.sub(r'(?m)^[ \t]+$', '', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def render_comment(comment, depth=0):
    result = '\n\n' + '#' * min(6, 3 + depth) + f" {comment['author_name']} — {comment['post_date']}\n\nVotes: {comment['votes']}\n\n" + converted(comment['content'])
    return result + ''.join(render_comment(x, depth + 1) for x in comment['replies'])

inventory = json.loads((OUT / 'archive_inventory.json').read_text())
for item in inventory:
    raw = json.loads((OUT / f"topic_{item['id']}_full.json").read_text())
    dest = ROOT / item['archive']
    archived = dest.read_text().split('- archived_at: ', 1)[1].split('\n', 1)[0]
    text = f"# {raw['title'].rstrip()}\n\n- archived_at: {archived}\n- source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/{raw['id']}\n- author: {raw['author_name'].rstrip()}\n- posted_at_utc: {raw['post_date']}\n- votes: {raw['votes']}\n- comments: {raw['comment_count']}\n\n" + converted(raw['content']) + '\n'
    if raw['comments']:
        text += '\n## Comments\n' + ''.join(render_comment(x) for x in raw['comments']) + '\n'
    dest.write_text(text)
    print('rendered', dest.relative_to(ROOT))

# The shared CLI listing helper includes pagination footers; filter them in this study only.
collect = OUT / 'collect.py'
text = collect.read_text().replace("topics = list_topics(COMP, 'recent', 3)", "topics = [row for row in list_topics(COMP, 'recent', 3) if row.get('id', '').isdigit()]")
collect.write_text(text)
import csv
with (OUT / 'topic_listing.csv').open() as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames
    rows = [row for row in reader if row.get('id', '').isdigit()]
with (OUT / 'topic_listing.csv').open('w') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print('topic_listing_rows', len(rows))
