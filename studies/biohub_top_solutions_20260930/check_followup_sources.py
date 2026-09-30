"""Refresh public discussion evidence for the calibration/Transformer follow-up."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from kaggle.api.kaggle_api_extended import KaggleApi


def public_comment(comment):
    return {
        "author": comment.author_name,
        "content": comment.content,
        "replies": [public_comment(item) for item in comment.replies or []],
    }


def show_relevant_comments(comments):
    for comment in comments:
        body = BeautifulSoup(comment["content"], "html.parser").get_text(" ", strip=True)
        if re.search(r"evaluat|calibr|logistic|annotat|transformer|attention", body, re.I):
            print("Comment:", comment["author"], body[:1800])
        show_relevant_comments(comment["replies"])


api = KaggleApi()
api.authenticate()
records = []
for topic_id in (744484, 744549, 744582):
    topic, comments, _ = api.forums_topic_show(topic_id)
    record = {
        "topic_id": topic_id,
        "id": topic_id,
        "title": topic.title,
        "author_name": topic.author_name,
        "post_date": str(topic.post_date),
        "votes": topic.votes,
        "comment_count": topic.comment_count,
        "content": topic.content,
        "comments": [public_comment(item) for item in comments],
    }
    records.append(record)
    print("Topic:", topic_id, topic.title, "comments:", len(comments))
    soup = BeautifulSoup(topic.content, "html.parser")
    if topic_id == 744582:
        print(soup.get_text("\n", strip=True)[:22000])
    for paragraph in soup.find_all("p"):
        body = paragraph.get_text(" ", strip=True)
        if re.search(r"probability of being evaluated|writing these|ordinary links use|self-attention", body, re.I):
            print(body)
    show_relevant_comments(record["comments"])

destination = Path(__file__).with_name("calibration_transformer_followup.json")
destination.write_text(
    json.dumps(
        {"retrieved_at": datetime.now(timezone.utc).isoformat(), "topics": records},
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)

# Add the newly published solution to the existing archive inventory.
out = Path(__file__).resolve().parent
root = out.parents[1]
sixth = next(item for item in records if item["id"] == 744582)
(out / "topic_744582_full.json").write_text(
    json.dumps(sixth, ensure_ascii=False, indent=2) + "\n"
)
archive = root / "docs/discussions/biohub-cell-tracking-during-development-744582-6th-place-six-model-detect-and-link-ensemble-global-ilp.md"
if not archive.exists():
    archive.write_text(f"- archived_at: {datetime.now(timezone.utc).isoformat()}\n")
inventory_path = out / "archive_inventory.json"
inventory = json.loads(inventory_path.read_text())
entry = {
    "id": sixth["id"],
    "title": sixth["title"],
    "author": sixth["author_name"],
    "archive": str(archive.relative_to(root)),
    "links": [],
    "comment_count_retrieved": len(sixth["comments"]),
    "raw_content_chars": len(sixth["content"]),
}
inventory = [item for item in inventory if item["id"] != sixth["id"]] + [entry]
inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
