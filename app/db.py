"""SQLite store for transcripts, with full-text search."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    device      TEXT NOT NULL,
    started_at  TEXT,
    duration_s  REAL,
    language    TEXT,
    transcript  TEXT NOT NULL,
    audio_bytes INTEGER,
    created_at  TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
    transcript, content='conversations', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS conversations_ai AFTER INSERT ON conversations BEGIN
    INSERT INTO conversations_fts(rowid, transcript) VALUES (new.rowid, new.transcript);
END;

CREATE TRIGGER IF NOT EXISTS conversations_ad AFTER DELETE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, transcript)
    VALUES ('delete', old.rowid, old.transcript);
END;

CREATE TRIGGER IF NOT EXISTS conversations_au AFTER UPDATE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, transcript)
    VALUES ('delete', old.rowid, old.transcript);
    INSERT INTO conversations_fts(rowid, transcript) VALUES (new.rowid, new.transcript);
END;
"""


@dataclass(frozen=True)
class Conversation:
    id: str
    device: str
    started_at: str | None
    duration_s: float | None
    language: str | None
    transcript: str
    audio_bytes: int | None


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def insert(conn: sqlite3.Connection, conv: Conversation) -> bool:
    """Insert a transcript. Returns False when the id already exists."""
    try:
        conn.execute(
            """INSERT INTO conversations
               (id, device, started_at, duration_s, language, transcript,
                audio_bytes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                conv.id,
                conv.device,
                conv.started_at,
                conv.duration_s,
                conv.language,
                conv.transcript,
                conv.audio_bytes,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        conn.rollback()
        return False


def recent(conn: sqlite3.Connection, limit: int = 20) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT id, device, started_at, duration_s, language,
                  length(transcript) AS transcript_chars, created_at
           FROM conversations
           ORDER BY COALESCE(started_at, created_at) DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def search(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT c.id, c.device, c.started_at, c.duration_s, c.language,
                  snippet(conversations_fts, 0, '[', ']', '…', 16) AS excerpt
           FROM conversations_fts
           JOIN conversations c ON c.rowid = conversations_fts.rowid
           WHERE conversations_fts MATCH ?
           ORDER BY rank
           LIMIT ?""",
        (query, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def get(conn: sqlite3.Connection, conversation_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
    ).fetchone()
    return dict(row) if row else None


def count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0])
