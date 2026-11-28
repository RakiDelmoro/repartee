"""Knowledge base: one row per conversation, with transcript and summary.

SQLite is the whole store. The transcript and its summary live in the same row,
and FTS5 indexes both, so a conversation cannot be stored without the text that
made it searchable.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    device      TEXT NOT NULL,
    started_at  TEXT,
    ended_at    TEXT,
    duration_s  REAL,
    language    TEXT,
    transcript  TEXT NOT NULL DEFAULT '',
    summary     TEXT NOT NULL DEFAULT '',
    audio_path  TEXT,
    audio_bytes INTEGER,
    model       TEXT,
    created_at  TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'transcribed'
);

CREATE INDEX IF NOT EXISTS conversations_started
    ON conversations(started_at);

CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
    transcript, summary, content='conversations', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS conversations_ai AFTER INSERT ON conversations BEGIN
    INSERT INTO conversations_fts(rowid, transcript, summary)
    VALUES (new.rowid, new.transcript, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS conversations_ad AFTER DELETE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, transcript, summary)
    VALUES ('delete', old.rowid, old.transcript, old.summary);
END;

CREATE TRIGGER IF NOT EXISTS conversations_au AFTER UPDATE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, transcript, summary)
    VALUES ('delete', old.rowid, old.transcript, old.summary);
    INSERT INTO conversations_fts(rowid, transcript, summary)
    VALUES (new.rowid, new.transcript, new.summary);
END;
"""


@dataclass(frozen=True)
class Conversation:
    """One finished recording as the bigger device stores it."""

    id: str
    device: str
    started_at: str | None
    duration_s: float | None
    language: str | None
    transcript: str
    summary: str = ""
    audio_path: str | None = None
    audio_bytes: int | None = None
    model: str | None = None
    ended_at: str | None = None
    status: str = "transcribed"


def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def upsert(conn: sqlite3.Connection, conv: Conversation) -> None:
    conn.execute(
        """INSERT INTO conversations
               (id, device, started_at, ended_at, duration_s, language,
                transcript, summary, audio_path, audio_bytes, model,
                created_at, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               device      = excluded.device,
               started_at  = excluded.started_at,
               ended_at    = excluded.ended_at,
               duration_s  = excluded.duration_s,
               language    = excluded.language,
               transcript  = excluded.transcript,
               summary     = excluded.summary,
               audio_path  = excluded.audio_path,
               audio_bytes = excluded.audio_bytes,
               model       = excluded.model,
               status      = excluded.status""",
        (
            conv.id,
            conv.device,
            conv.started_at,
            conv.ended_at,
            conv.duration_s,
            conv.language,
            conv.transcript,
            conv.summary,
            conv.audio_path,
            conv.audio_bytes,
            conv.model,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            conv.status,
        ),
    )
    conn.commit()


def get(conn: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM conversations WHERE id = ?", (session_id,)
    ).fetchone()
    return dict(row) if row else None


def recent(conn: sqlite3.Connection, limit: int = 20) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT id, device, started_at, duration_s, language, summary,
                  audio_bytes, model, status
           FROM conversations
           ORDER BY started_at DESC, id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def _fts_match(query: str) -> str | None:
    terms = re.findall(r"[A-Za-z0-9']+", query)
    if not terms:
        return None
    return " AND ".join(f'"{term}"' for term in terms)


def search(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[dict[str, Any]]:
    match = _fts_match(query)
    if match is None:
        return []
    rows = conn.execute(
        """SELECT c.id, c.device, c.started_at, c.duration_s, c.language,
                  c.summary, c.status,
                  snippet(conversations_fts, 0, '[', ']', '…', 12) AS transcript_snippet,
                  snippet(conversations_fts, 1, '[', ']', '…', 12) AS summary_snippet
           FROM conversations_fts
           JOIN conversations c ON c.rowid = conversations_fts.rowid
           WHERE conversations_fts MATCH ?
           ORDER BY rank LIMIT ?""",
        (match, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repartee",
        description="Inspect stored conversations.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="full-text search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--db", type=Path, default=config.DB_PATH)

    listing = sub.add_parser("conversations", help="recent conversations")
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument("--db", type=Path, default=config.DB_PATH)

    show = sub.add_parser("show", help="one conversation as JSON")
    show.add_argument("session_id")
    show.add_argument("--db", type=Path, default=config.DB_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    conn = connect(args.db)
    try:
        init(conn)
        if args.command == "search":
            rows = search(conn, args.query, args.limit)
        elif args.command == "conversations":
            rows = recent(conn, args.limit)
        else:
            row = get(conn, args.session_id)
            rows = row if row is not None else {"error": "not found", "id": args.session_id}
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
