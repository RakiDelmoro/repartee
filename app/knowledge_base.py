"""File-based knowledge base: one Markdown file per conversation.

The folder is the store. There is no database and no index: listing, searching
and showing all walk the Markdown files. Each file is written atomically and a
complete file is the idempotence marker, so the worker can retry a partial write
without clobbering a finished conversation.

Layout::

    knowledge-base/
      2026-11-28/
        20261128-012000-openyap.md

The frontmatter is YAML; the body is ``# Summary`` and ``# Transcript``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from . import config

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class Entry:
    """One conversation as stored in the knowledge-base folder."""

    id: str
    device: str
    started_at: str | None
    ended_at: str | None
    duration_s: float | None
    language: str | None
    model: str | None
    audio: str | None
    audio_bytes: int | None
    status: str
    created_at: str
    summary: str = ""
    transcript: str = ""


def _date_for(entry: Entry) -> str:
    if entry.started_at and len(entry.started_at) >= 10:
        date = entry.started_at[:10]
        if DATE_RE.match(date):
            return date
    return "unknown"


def path_for(root: Path, entry: Entry) -> Path:
    return Path(root) / _date_for(entry) / f"{entry.id}.md"


def find(root: Path, session_id: str) -> Path | None:
    """Find a conversation file by id, across date folders."""
    matches = sorted(Path(root).glob(f"*/{session_id}.md"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"duplicate conversation id: {session_id}")
    return matches[0]


def exists(root: Path, session_id: str) -> bool:
    return find(root, session_id) is not None


def _render(entry: Entry) -> str:
    meta = {
        "schema": SCHEMA_VERSION,
        "id": entry.id,
        "device": entry.device,
        "started_at": entry.started_at,
        "ended_at": entry.ended_at,
        "duration_s": entry.duration_s,
        "language": entry.language,
        "model": entry.model,
        "audio": entry.audio,
        "audio_bytes": entry.audio_bytes,
        "status": entry.status,
        "created_at": entry.created_at,
    }
    frontmatter = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    body = f"# Summary\n\n{entry.summary.strip()}\n\n# Transcript\n\n{entry.transcript.strip()}\n"
    return f"---\n{frontmatter}---\n\n{body}"


def write(root: Path, entry: Entry, force: bool = False) -> Path:
    """Write one conversation atomically. Existing files are kept unless forced."""
    path = path_for(root, entry)
    if path.exists() and not force:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{entry.id}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_render(entry))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _parse_sections(body: str) -> tuple[str, str]:
    body = body.lstrip("\n")
    summary_marker = "# Summary\n"
    transcript_marker = "# Transcript\n"
    if not body.startswith(summary_marker):
        raise ValueError("missing # Summary section")
    summary, sep, rest = body[len(summary_marker) :].partition(transcript_marker)
    if not sep:
        raise ValueError("missing # Transcript section")
    return summary.strip(), rest.strip()


def read(path: Path) -> Entry:
    text = Path(path).read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing frontmatter")
    frontmatter, sep, body = text[4:].partition("\n---\n")
    if not sep:
        raise ValueError(f"{path}: unclosed frontmatter")
    meta = yaml.safe_load(frontmatter) or {}
    summary, transcript = _parse_sections(body)
    return Entry(
        id=str(meta["id"]),
        device=str(meta.get("device") or "unknown"),
        started_at=meta.get("started_at"),
        ended_at=meta.get("ended_at"),
        duration_s=meta.get("duration_s"),
        language=meta.get("language"),
        model=meta.get("model"),
        audio=meta.get("audio"),
        audio_bytes=meta.get("audio_bytes"),
        status=str(meta.get("status") or "transcribed"),
        created_at=str(meta.get("created_at") or ""),
        summary=summary,
        transcript=transcript,
    )


def _walk(root: Path) -> list[tuple[Path, Entry]]:
    rows: list[tuple[Path, Entry]] = []
    for path in sorted(Path(root).glob("*/*.md")):
        try:
            rows.append((path, read(path)))
        except (OSError, ValueError, KeyError, yaml.YAMLError):
            log.exception("ignoring malformed knowledge-base file %s", path)
    return rows


def _entry_dict(entry: Entry, path: Path, include_text: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": entry.id,
        "device": entry.device,
        "started_at": entry.started_at,
        "ended_at": entry.ended_at,
        "duration_s": entry.duration_s,
        "language": entry.language,
        "model": entry.model,
        "audio": entry.audio,
        "audio_bytes": entry.audio_bytes,
        "status": entry.status,
        "created_at": entry.created_at,
        "path": str(path),
        "summary": entry.summary,
    }
    if include_text:
        row["transcript"] = entry.transcript
    return row


def recent(root: Path, limit: int = 20) -> list[dict[str, Any]]:
    rows = [
        _entry_dict(entry, path)
        for path, entry in _walk(root)
        if entry.started_at is not None
    ]
    rows.sort(key=lambda row: (row["started_at"] or "", row["id"]), reverse=True)
    return rows[:limit]


def _snippet(text: str, query: str, width: int = 60) -> str:
    index = text.lower().find(query.lower())
    if index < 0:
        return ""
    start = max(0, index - width)
    end = min(len(text), index + len(query) + width)
    prefix = "…" if start else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{' '.join(text[start:end].split())}{suffix}"


def search(root: Path, query: str, limit: int = 20) -> list[dict[str, Any]]:
    query = query.strip()
    if not query:
        return []
    rows: list[dict[str, Any]] = []
    for path, entry in _walk(root):
        hit_summary = _snippet(entry.summary, query)
        hit_transcript = _snippet(entry.transcript, query)
        if not hit_summary and not hit_transcript:
            continue
        row = _entry_dict(entry, path)
        row["snippet"] = hit_summary or hit_transcript
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def get(root: Path, session_id: str) -> dict[str, Any] | None:
    path = find(root, session_id)
    if path is None:
        return None
    return _entry_dict(read(path), path, include_text=True)


def count(root: Path) -> int:
    return len(_walk(root))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repartee",
        description="Inspect the file-based conversation knowledge base.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="search summaries and transcripts")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument(
        "--knowledge-base", type=Path, default=config.KNOWLEDGE_BASE_DIR
    )

    listing = sub.add_parser("conversations", help="recent conversations")
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument(
        "--knowledge-base", type=Path, default=config.KNOWLEDGE_BASE_DIR
    )

    show = sub.add_parser("show", help="one conversation with its transcript")
    show.add_argument("session_id")
    show.add_argument(
        "--knowledge-base", type=Path, default=config.KNOWLEDGE_BASE_DIR
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "search":
        rows = search(args.knowledge_base, args.query, args.limit)
    elif args.command == "conversations":
        rows = recent(args.knowledge_base, args.limit)
    else:
        row = get(args.knowledge_base, args.session_id)
        rows = row if row is not None else {"error": "not found", "id": args.session_id}
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
