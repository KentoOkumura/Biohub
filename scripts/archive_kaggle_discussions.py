from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from config_utils import configured_project_path


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "discussion"


def run_csv(cmd: list[str]) -> list[dict[str, str]]:
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return []
    return list(csv.DictReader(lines))


def list_topics(competition: str, sort_by: str, max_pages: int) -> list[dict[str, str]]:
    topics: list[dict[str, str]] = []
    seen: set[str] = set()
    for page in range(1, max_pages + 1):
        rows = run_csv(
            [
                sys.executable,
                "-m",
                "kaggle",
                "competitions",
                "topics",
                "list",
                competition,
                "-s",
                sort_by,
                "-p",
                str(page),
                "-v",
            ]
        )
        new_rows = [row for row in rows if row.get("id") and row["id"] not in seen]
        if not new_rows:
            break
        for row in new_rows:
            seen.add(row["id"])
            row["source_page"] = str(page)
            row["sort_by"] = sort_by
        topics.extend(new_rows)
    return topics


def discussion_api() -> Any:
    # Use the exact client and authentication entry point used by kaggle.cli.
    # This keeps CLI OAuth, access-token and legacy credentials available.
    from kaggle import api

    if not api._authenticated:
        api.authenticate()
    return api


def fetch_topic(topic_id: int, *, api: Any | None = None) -> dict[str, Any]:
    client = discussion_api() if api is None else api
    token: str | None = None
    seen_tokens: set[str] = set()
    pages: list[dict[str, Any]] = []
    first_topic: dict[str, Any] | None = None
    while True:
        topic, comments, next_token = client.forums_topic_show(
            topic_id, page_size=200, page_token=token
        )
        if topic is None or topic.id != topic_id:
            raise ValueError(f"topic {topic_id}: missing or mismatched topic response")
        topic_data = topic.to_dict(ignore_defaults=False)
        if not isinstance(topic_data.get("content"), str):
            raise ValueError(f"topic {topic_id}: response does not contain the topic body")
        if first_topic is None:
            first_topic = topic_data
        pages.append(
            {
                "page_token": token,
                "next_page_token": next_token or None,
                "comments": [comment.to_dict(ignore_defaults=False) for comment in comments or []],
            }
        )
        if not next_token:
            break
        if next_token in seen_tokens:
            raise ValueError(f"topic {topic_id}: repeated comment page token")
        seen_tokens.add(next_token)
        token = next_token
    return {"topic": first_topic, "comment_pages": pages}


def render_topic(snapshot: dict[str, Any]) -> str:
    """Keep API content unchanged, including HTML, Markdown and long comments."""
    topic = snapshot["topic"]
    parts = [
        f"- Author: {topic.get('authorName', '')}",
        f"- Posted: {topic.get('postDate', '')}",
        f"- Votes: {topic.get('votes', '')}",
        "",
        topic["content"],
    ]
    seen_comments: set[int] = set()

    def append_comment(comment: dict[str, Any], parent_id: int | None = None) -> None:
        comment_id = comment["id"]
        if comment_id not in seen_comments:
            seen_comments.add(comment_id)
            parts.extend(
                [
                    "",
                    f"## Comment {comment_id}",
                    "",
                    f"- Author: {comment.get('authorName', '')}",
                    f"- Posted: {comment.get('postDate', '')}",
                    f"- Votes: {comment.get('votes', '')}",
                    *([f"- Reply to: {parent_id}"] if parent_id is not None else []),
                    "",
                    comment.get("content") or "[deleted or unavailable]",
                ]
            )
        for reply in comment.get("replies") or []:
            append_comment(reply, comment_id)

    for page in snapshot["comment_pages"]:
        for comment in page["comments"]:
            append_comment(comment)
    return "\n".join(parts) + "\n"


def source_identity(source: Any) -> tuple[str, str, int] | None:
    if not isinstance(source, str):
        return None
    try:
        parsed = urlsplit(source)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "kaggle.com",
        "www.kaggle.com",
    }:
        return None
    match = re.fullmatch(r"/competitions/([^/]+)/discussion/(\d+)/?", parsed.path)
    if match:
        return "competition", match[1], int(match[2])
    match = re.fullmatch(r"/discussions/([^/]+)/(\d+)/?", parsed.path)
    if match:
        return "forum", match[1], int(match[2])
    return None


def saved_source(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            snapshot = json.loads(text)
            return snapshot.get("source") if isinstance(snapshot, dict) else None
    except (UnicodeError, json.JSONDecodeError):
        return None
    # Only inspect the converter's leading metadata, never URLs quoted in the body.
    lines = iter(text.splitlines())
    if not next(lines, "").startswith("# "):
        return None
    metadata_started = False
    for line in lines:
        if not line and not metadata_started:
            continue
        if not line.startswith("- "):
            break
        metadata_started = True
        if line.startswith("- source: "):
            return line.removeprefix("- source: ").strip()
    return None


def existing_archive(output_dir: Path, source: str) -> Path | None:
    identity = source_identity(source)
    if identity is None:
        raise ValueError(f"unsupported discussion source: {source}")
    sources_by_path: dict[Path, set[tuple[str, str, int]]] = {}
    for path in sorted(output_dir.glob("*")):
        if not path.is_file() or path.suffix not in {".md", ".json"}:
            continue
        saved_identity = source_identity(saved_source(path))
        if saved_identity is not None:
            sources_by_path.setdefault(path.with_suffix(".md"), set()).add(saved_identity)
    matches = [path for path, identities in sources_by_path.items() if identity in identities]
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous existing archives for {source}: " + ", ".join(map(str, matches))
        )
    if matches and sources_by_path[matches[0]] != {identity}:
        raise ValueError(f"conflicting Markdown/JSON sources for {matches[0]}")
    return matches[0] if matches else None


def archive_topic(
    competition: str,
    topic: dict[str, str],
    output_dir: Path,
    converter: Path,
    force: bool,
    *,
    forum: bool = False,
    api: Any | None = None,
) -> str:
    topic_id = str(int(topic["id"]))
    source = (
        f"https://www.kaggle.com/discussions/{competition}/{topic_id}"
        if forum
        else f"https://www.kaggle.com/competitions/{competition}/discussion/{topic_id}"
    )
    out_path = existing_archive(output_dir, source)
    if out_path is not None and out_path.exists() and not force:
        return f"skip existing {topic_id} -> {out_path}"

    snapshot = fetch_topic(int(topic_id), api=api)
    title = snapshot["topic"].get("title") or topic_id
    if out_path is None:
        prefix = f"forum-{competition}" if forum else competition
        slug = slugify(f"{prefix}-{topic_id}-{slugify(title)[:80]}")
        out_path = output_dir / f"{slug}.md"
        if out_path.exists() or out_path.with_suffix(".json").exists():
            raise FileExistsError(
                f"archive path already exists with another or unknown source: {out_path}"
            )
    snapshot.update({"source": source, "retrieved_at": datetime.now(UTC).isoformat()})
    output_dir.mkdir(parents=True, exist_ok=True)
    # Stage both outputs only after pagination succeeds. A failed fetch or
    # conversion must never truncate an existing archive, even with --force.
    with tempfile.TemporaryDirectory(prefix=".discussion-", dir=output_dir) as temporary:
        temporary_dir = Path(temporary)
        tmp_path = temporary_dir / "body.txt"
        tmp_path.write_text(render_topic(snapshot), encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                str(converter),
                str(tmp_path),
                "--input-format",
                "text",
                "--title",
                title,
                "--url",
                source,
                "--slug",
                "discussion",
                "--output-dir",
                str(temporary_dir),
            ],
            check=True,
        )
        snapshot_path = temporary_dir / "discussion.json"
        snapshot_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (temporary_dir / "discussion.md").replace(out_path)
        snapshot_path.replace(out_path.with_suffix(".json"))
    return f"archived {topic_id} -> {out_path}"


def write_listing(topics: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "title",
        "authorName",
        "commentCount",
        "votes",
        "postDate",
        "source_page",
        "sort_by",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(topics)


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--competition")
    source.add_argument("--forum", help="Forum slug; requires --topic-id")
    parser.add_argument("--topic-id", action="append", type=int, default=[])
    parser.add_argument("--sort-by", default="recent")
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=configured_project_path("paths.docs_dir", "docs") / "discussions",
    )
    parser.add_argument("--listing", type=Path, default=None)
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path(".agents/skills/kaggle-discussion-archive/scripts/html_to_discussion_md.py"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.forum and not args.topic_id:
        parser.error("--forum requires at least one --topic-id")
    if any(topic_id <= 0 for topic_id in args.topic_id):
        parser.error("--topic-id must be a positive integer")
    if args.max_pages < 1:
        parser.error("--max-pages must be positive")

    topics = (
        [{"id": str(topic_id)} for topic_id in dict.fromkeys(args.topic_id)]
        if args.topic_id
        else list_topics(args.competition, args.sort_by, args.max_pages)
    )
    source_name = args.competition or args.forum
    listing = args.listing or (
        None if args.topic_id else args.output_dir / f"{source_name}_topics_{args.sort_by}.csv"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for topic in topics:
        print(
            archive_topic(
                source_name,
                topic,
                args.output_dir,
                args.converter,
                args.force,
                forum=bool(args.forum),
            )
        )
    if listing is not None:
        write_listing(topics, listing)
    print(f"total_topics={len(topics)} listing={listing} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
