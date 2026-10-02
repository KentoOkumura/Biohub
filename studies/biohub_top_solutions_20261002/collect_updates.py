"""Retrieve public solution updates and archive their source HTML and links."""

from __future__ import annotations

import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString
from kaggle.api.kaggle_api_extended import ApiGetLeaderboardRequest, KaggleApi

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from archive_kaggle_discussions import list_topics, slugify
sys.path.insert(0, str(ROOT / ".agents/skills/kaggle-discussion-archive/scripts"))
from html_to_discussion_md import convert as html_to_markdown

COMP = "biohub-cell-tracking-during-development"
MAIN = {744801: 1, 744723: 2, 744673: 4, 744937: 7, 744835: 9,
        744980: 10, 745092: 11, 744671: 16, 744930: 17}
RELATED = {744630, 744601, 744785, 744799, 744817, 744912, 744762,
           744842, 744670, 744647, 745086, 744899, 744895}


def convert(content: str) -> str:
    soup = BeautifulSoup(content, "html.parser")
    blocks = {}
    def block(tag, value):
        token = f"ARCHIVEBLOCK{len(blocks):05d}END"
        blocks[token] = value
        tag.replace_with(NavigableString(token))
    for anchor in soup.find_all("a", href=True):
        anchor["href"] = urljoin("https://www.kaggle.com/", anchor["href"])
    for image in soup.find_all("img", src=True):
        src = urljoin("https://www.kaggle.com/", image["src"])
        markdown = f"![{image.get('alt') or 'source image'}]({src})"
        parent = image.parent
        if parent and parent.name == "a" and not parent.get_text(strip=True):
            block(parent, f"\n\n[{markdown}]({parent['href']})\n\n")
        else:
            block(image, f"\n\n{markdown}\n\n")
    for table in soup.find_all("table"):
        rows = [[cell.get_text(" ", strip=True).replace("|", "\\|")
                 for cell in row.find_all(["th", "td"], recursive=False)]
                for row in table.find_all("tr")]
        rows = [row for row in rows if row]
        if not rows:
            continue
        width = max(map(len, rows))
        rows = [row + [""] * (width - len(row)) for row in rows]
        lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
        lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
        block(table, "\n\n" + "\n".join(lines) + "\n\n")
    for heading in soup.find_all(re.compile(r"^h[1-6]$")):
        block(heading, "\n\n" + "#" * int(heading.name[1]) + " " + heading.get_text(" ", strip=True) + "\n\n")
    for li in soup.find_all("li"):
        first_paragraph = li.find("p", recursive=False)
        if first_paragraph:
            first_paragraph.unwrap()
        li.insert(0, NavigableString("- "))
    text = html_to_markdown(str(soup))
    for token, value in blocks.items():
        text = text.replace(token, value)
    text = re.sub(r"(?m)^[ \t]+$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return "\n".join(line.rstrip(" \t") for line in text.splitlines())


def public_comment(comment) -> dict:
    return {"id": comment.id, "author_name": comment.author_name,
            "post_date": str(comment.post_date), "votes": comment.votes,
            "content": comment.content,
            "replies": [public_comment(item) for item in comment.replies or []]}


def comment_markdown(comment: dict, depth: int = 0) -> str:
    title = "#" * min(6, 3 + depth)
    body = f"\n\n{title} {comment['author_name'].rstrip()} — {comment['post_date']}\n\nVotes: {comment['votes']}\n\n"
    return body + convert(comment["content"]) + "".join(comment_markdown(item, depth + 1) for item in comment["replies"])


def fetch_topic(topic_id: int) -> dict:
    api = KaggleApi()
    api.authenticate()
    topic, comments, next_page = api.forums_topic_show(topic_id)
    if next_page:
        raise RuntimeError(f"Topic {topic_id} has additional comment pages; retrieve them explicitly.")
    raw = {"id": topic_id, "title": topic.title, "author_name": topic.author_name,
           "post_date": str(topic.post_date), "votes": topic.votes,
           "comment_count": topic.comment_count, "content": topic.content,
           "comments": [public_comment(item) for item in comments],
           "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / f"topic_{topic_id}_full.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
    slug = f"{COMP}-{topic_id}-{slugify(topic.title)[:80]}"
    dest = ROOT / "docs/discussions" / f"{slug}.md"
    if dest.exists():
        raise RuntimeError(f"Refusing to replace existing archive: {dest}")
    text = f"# {topic.title.rstrip()}\n\n- archived_at: {raw['retrieved_at_utc']}\n- source: https://www.kaggle.com/competitions/{COMP}/discussion/{topic_id}\n- author: {topic.author_name.rstrip()}\n- posted_at_utc: {topic.post_date}\n- votes: {topic.votes}\n- comments: {topic.comment_count}\n\n" + convert(topic.content) + "\n"
    if comments:
        text += "\n## Comments\n" + "".join(comment_markdown(item) for item in raw["comments"]) + "\n"
    dest.write_text(text)
    soup = BeautifulSoup(topic.content, "html.parser")
    entry = {"id": topic_id, "rank_to_verify": MAIN.get(topic_id), "title": topic.title,
             "author": topic.author_name, "archive": str(dest.relative_to(ROOT)),
             "raw_content_chars": len(topic.content),
             "comments_retrieved": len(comments),
             "links": [urljoin("https://www.kaggle.com/", node["href"]) for node in soup.find_all("a", href=True)],
             "images": [urljoin("https://www.kaggle.com/", node["src"]) for node in soup.find_all("img", src=True)]}
    print(topic_id, "rank", MAIN.get(topic_id), topic.title, "by", topic.author_name,
          "body_chars", len(topic.content), "comments", len(comments), flush=True)
    return entry


def main():
    topics = [row for row in list_topics(COMP, "recent", 3) if row.get("id", "").isdigit()]
    with (OUT / "topic_listing.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(topics[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(topics)
    inventory = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for entry in pool.map(fetch_topic, list(MAIN) + sorted(RELATED)):
            inventory.append(entry)
            (OUT / "archive_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    api = KaggleApi()
    api.authenticate()
    rows, token = [], None
    with api.build_kaggle_client() as client:
        for page in range(10):
            request = ApiGetLeaderboardRequest()
            request.competition_name = COMP
            api._set_paging(request, 200, token)
            response = client.competitions.competition_api_client.get_leaderboard(request)
            rows += [item.to_dict() for item in response.submissions or []]
            token = response.next_page_token
            if any(item.get("teamId") == 16717777 for item in rows) or not token:
                break
    snapshot = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "pages": page + 1,
                "has_more": bool(token), "scope": "current final leaderboard returned by get_leaderboard", "rows": rows}
    (OUT / "final_leaderboard.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    for rank, row in enumerate(rows, 1):
        if rank <= 20 or row.get("teamId") == 16717777:
            print("leaderboard", rank, row.get("teamName"), row.get("score"), flush=True)


if __name__ == "__main__":
    main()
