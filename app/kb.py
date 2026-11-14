"""Knowledge base: one row per conversation, with its transcript, summary and audio size.

SQLite is the whole store: the dashboard, the search index and the spool worker all
read the same table, so a conversation and its summary can never drift apart.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    device      TEXT NOT NULL,
    started_at  TEXT,
    ended_at    TEXT,
    duration_s  REAL,
    language    TEXT,
    transcript  TEXT NOT NULL DEFAULT '',
    summary     TEXT NOT NULL DEFAULT '',
    audio_bytes INTEGER,
    created_at  TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'receiving'
);

CREATE INDEX IF NOT EXISTS sessions_started ON sessions(started_at);

CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
    transcript, content='sessions', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN
    INSERT INTO sessions_fts(rowid, transcript) VALUES (new.rowid, new.transcript);
END;

CREATE TRIGGER IF NOT EXISTS sessions_ad AFTER DELETE ON sessions BEGIN
    INSERT INTO sessions_fts(sessions_fts, rowid, transcript)
    VALUES ('delete', old.rowid, old.transcript);
END;

CREATE TRIGGER IF NOT EXISTS sessions_au AFTER UPDATE ON sessions BEGIN
    INSERT INTO sessions_fts(sessions_fts, rowid, transcript)
    VALUES ('delete', old.rowid, old.transcript);
    INSERT INTO sessions_fts(rowid, transcript) VALUES (new.rowid, new.transcript);
END;
"""

MIGRATIONS = (
    ("sessions", "summary", "TEXT NOT NULL DEFAULT ''"),
    ("sessions", "audio_bytes", "INTEGER"),
    ("sessions", "created_at", "TEXT NOT NULL DEFAULT ''"),
)


@dataclass(frozen=True)
class Conversation:
    """One finished recording: what the worker writes and the API returns."""

    id: str
    device: str
    started_at: str | None
    duration_s: float | None
    language: str | None
    transcript: str
    audio_bytes: int | None = None
    summary: str = ""
    ended_at: str | None = None
    status: str = "transcribed"


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    for table, column, decl in MIGRATIONS:
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    conn.commit()


def upsert(conn: sqlite3.Connection, conv: Conversation) -> None:
    conn.execute(
        """INSERT INTO sessions (id, device, started_at, ended_at, duration_s, language,
                                 transcript, summary, audio_bytes, created_at, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               ended_at   = excluded.ended_at,
               duration_s = excluded.duration_s,
               language   = excluded.language,
               transcript = excluded.transcript,
               summary    = excluded.summary,
               audio_bytes= excluded.audio_bytes,
               status     = excluded.status""",
        (
            conv.id, conv.device, conv.started_at, conv.ended_at, conv.duration_s,
            conv.language, conv.transcript, conv.summary, conv.audio_bytes,
            datetime.now(timezone.utc).isoformat(timespec="seconds"), conv.status,
        ),
    )
    conn.commit()


def get(conn: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def recent(conn: sqlite3.Connection, limit: int = 20) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT id, device, started_at, duration_s, language, summary, audio_bytes,
                  status, COALESCE(substr(started_at, 1, 10), '') AS day
           FROM sessions ORDER BY started_at DESC, id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def search(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT s.id, s.device, s.started_at, s.summary, s.transcript,
                  snippet(sessions_fts, 0, '[', ']', '…', 12) AS snippet
           FROM sessions_fts
           JOIN sessions s ON s.rowid = sessions_fts.rowid
           WHERE sessions_fts MATCH ?
           ORDER BY rank LIMIT ?""",
        (query, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def list_days(conn: sqlite3.Connection, limit: int = 60) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT substr(started_at, 1, 10) AS day, COUNT(*) AS sessions,
                  SUM(duration_s) AS seconds
           FROM sessions
           WHERE started_at IS NOT NULL AND started_at != ''
           GROUP BY day ORDER BY day DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [
        {"day": row["day"], "sessions": int(row["sessions"]),
         "seconds": round(float(row["seconds"] or 0.0), 1)}
        for row in rows
    ]


def sessions_for_day(conn: sqlite3.Connection, day: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT id, device, started_at, duration_s, language, summary, audio_bytes, status
           FROM sessions
           WHERE substr(started_at, 1, 10) = ?
           ORDER BY started_at, id""",
        (day,),
    ).fetchall()
    return [dict(row) for row in rows]


def count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
