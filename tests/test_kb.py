import sqlite3
import tempfile
from pathlib import Path

import pytest

from app import kb


@pytest.fixture()
def conn():
    tmp = tempfile.TemporaryDirectory()
    connection = kb.connect(Path(tmp.name) / "kb.db")
    kb.init(connection)
    yield connection
    connection.close()
    tmp.cleanup()


def test_store_and_read_back_a_hand_written_event(conn):
    kb.upsert_session(
        conn,
        kb.Session(id="rep01-20261107-143012", device="rep01", started_at="2026-11-07T14:30:12",
                   ended_at="2026-11-07T15:12:00", duration_s=2510.0, language="English",
                   transcript="...", status="ready"),
    )
    assert kb.insert_event(
        conn,
        kb.Event(
            id="rep01-20261107-143012-1",
            session_id="rep01-20261107-143012",
            kind="decision",
            text="delay the launch to Q2",
            evidence="we will delay the launch to Q2",
            t_start=412.5,
            confidence=0.9,
        ),
    )

    session = kb.get_session(conn, "rep01-20261107-143012")
    assert session["status"] == "ready"
    assert session["duration_s"] == 2510.0

    events = kb.list_events(conn, "rep01-20261107-143012")
    assert len(events) == 1
    assert events[0]["kind"] == "decision"
    assert events[0]["evidence"] == "we will delay the launch to Q2"
    assert kb.counts(conn) == {"sessions": 1, "events": 1, "entities": 0}


def test_event_without_a_session_is_rejected(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO events (id, session_id, kind, text, evidence, created_at)
               VALUES ('x', 'missing', 'decision', 't', 'e', 'now')"""
        )


def test_unknown_kind_is_rejected(conn):
    kb.upsert_session(conn, kb.Session("s", "rep01", None, None, None, None))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO events (id, session_id, kind, text, evidence, created_at)
               VALUES ('x', 's', 'nonsense', 't', 'e', 'now')"""
        )


def test_duplicate_event_id_is_ignored(conn):
    kb.upsert_session(conn, kb.Session("s", "rep01", None, None, None, None))
    event = kb.Event("e1", "s", "topic", "pricing", "we talked about pricing")
    assert kb.insert_event(conn, event) is True
    assert kb.insert_event(conn, event) is False
    assert len(kb.list_events(conn, "s")) == 1


def test_events_are_ordered_by_time(conn):
    kb.upsert_session(conn, kb.Session("s", "rep01", None, None, None, None))
    kb.insert_event(conn, kb.Event("b", "s", "topic", "second", "e2", t_start=20.0))
    kb.insert_event(conn, kb.Event("a", "s", "topic", "first", "e1", t_start=10.0))
    kb.insert_event(conn, kb.Event("c", "s", "topic", "no time", "e3"))

    assert [e["id"] for e in kb.list_events(conn, "s")] == ["a", "b", "c"]


def test_entity_link_round_trip(conn):
    kb.upsert_session(conn, kb.Session("s", "rep01", None, None, None, None))
    kb.insert_event(conn, kb.Event("e1", "s", "action", "send the spec", "Raki will send the spec"))
    kb.upsert_entity(conn, "person:raki", "person", "Raki", ["Raki Delmoro"])
    kb.link_event_entity(conn, "e1", "person:raki", "owner")

    rows = conn.execute("SELECT role FROM event_entities WHERE event_id = 'e1'").fetchall()
    assert [r["role"] for r in rows] == ["owner"]
