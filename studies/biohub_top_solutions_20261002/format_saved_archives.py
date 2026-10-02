"""Format the new archives again from their cached public HTML, offline."""

import json
from pathlib import Path

from collect_updates import ROOT, COMP, comment_markdown, convert

OUT = Path(__file__).resolve().parent


def main():
    inventory = json.loads((OUT / "archive_inventory.json").read_text())
    for row in inventory:
        raw = json.loads((OUT / f"topic_{row['id']}_full.json").read_text())
        content = (
            f"# {raw['title'].rstrip()}\n\n"
            f"- archived_at: {raw['retrieved_at_utc']}\n"
            f"- source: https://www.kaggle.com/competitions/{COMP}/discussion/{raw['id']}\n"
            f"- author: {raw['author_name'].rstrip()}\n"
            f"- posted_at_utc: {raw['post_date']}\n"
            f"- votes: {raw['votes']}\n"
            f"- comments: {raw['comment_count']}\n\n"
            + convert(raw["content"]) + "\n"
        )
        if raw["comments"]:
            content += "\n## Comments\n" + "".join(comment_markdown(item) for item in raw["comments"]) + "\n"
        (ROOT / row["archive"]).write_text(content)
    print("Reformatted cached archives:", len(inventory))


if __name__ == "__main__":
    main()
