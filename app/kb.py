"""Knowledge-base storage: sessions, events, entities.

M1 of docs/knowledge-base.md. No model involvement - this is schema and CRUD only.
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
    status      TEXT NOT NULL DEFAULT 'receiving'
);

CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('decision', 'action', 'question', 'topic')),
    text        TEXT NOT NULL,
    evidence    TEXT NOT NULL,
    owner       TEXT,
    due         TEXT,
    t_start     REAL,
    confidence  REAL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    name        TEXT NOT NULL,
    aliases     TEXT
);

CREATE TABLE IF NOT EXISTS event_entities (
    event_id    TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    entity_id   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    PRIMARY KEY (event_id, entity_id, role)
);

CREATE TABLE IF NOT EXISTS event_links (
    from_event  TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    to_event    TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    relation    TEXT NOT NULL,
    PRIMARY KEY (from_event, to_event, relation)
);

CREATE INDEX IF NOT EXISTS events_session ON events(session_id, kind);
CREATE INDEX IF NOT EXISTS events_kind ON events(kind);
"""


@dataclass(frozen=True)
class Session:
    id: str
    device: str
    started_at: str | None
    ended_at: str | None
    duration_s: float | None
    language: str | None
    transcript: str = ""
    status: str = "receiving"


@dataclass(frozen=True)
class Event:
    id: str
    session_id: str
    kind: str
    text: str
    evidence: str
    owner: str | None = None
    due: str | None = None
    t_start: float | None = None
    confidence: float | None = None


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def upsert_session(conn: sqlite3.Connection, session: Session) -> None:
    conn.execute(
        """INSERT INTO sessions (id, device, started_at, ended_at, duration_s,
                                 language, transcript, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               ended_at   = excluded.ended_at,
               duration_s = excluded.duration_s,
               language   = excluded.language,
               transcript = excluded.transcript,
               status     = excluded.status""",
        (
            session.id,
            session.device,
            session.started_at,
            session.ended_at,
            session.duration_s,
            session.language,
            session.transcript,
            session.status,
        ),
    )
    conn.commit()


def get_session(conn: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def insert_event(conn: sqlite3.Connection, event: Event) -> bool:
    """Insert an event. Returns False when the id already exists."""
    try:
        conn.execute(
            """INSERT INTO events (id, session_id, kind, text, evidence,
                                   owner, due, t_start, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.id,
                event.session_id,
                event.kind,
                event.text,
                event.evidence,
                event.owner,
                event.due,
                event.t_start,
                event.confidence,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        conn.rollback()
        return False


def list_events(
    conn: sqlite3.Connection, session_id: str | None = None, kind: str | None = None, limit: int = 200
) -> list[dict[str, Any]]:
    clauses, params = [], []
    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)
    if kind:
        clauses.append("kind = ?")
        params.append(kind)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""SELECT id, session_id, kind, text, evidence, owner, due, t_start, confidence
            FROM events {where}
            ORDER BY t_start IS NULL, t_start, id
            LIMIT ?""",
        (*params, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def upsert_entity(conn: sqlite3.Connection, entity_id: str, kind: str, name: str, aliases: list[str] | None = None) -> None:
    import json

    conn.execute(
        """INSERT INTO entities (id, kind, name, aliases) VALUES (?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET kind = excluded.kind, name = excluded.name,
                                         aliases = excluded.aliases""",
        (entity_id, kind, name, json.dumps(aliases or [])),
    )
    conn.commit()


def link_event_entity(conn: sqlite3.Connection, event_id: str, entity_id: str, role: str) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO event_entities (event_id, entity_id, role)
           VALUES (?, ?, ?)""",
        (event_id, entity_id, role),
    )
    conn.commit()


def link_events(conn: sqlite3.Connection, from_event: str, to_event: str, relation: str) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO event_links (from_event, to_event, relation)
           VALUES (?, ?, ?)""",
        (from_event, to_event, relation),
    )
    conn.commit()


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in ("sessions", "events", "entities")
    }
