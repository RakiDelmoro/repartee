"""Knowledge base: one row per conversation, day grouping, FTS search."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import kb


@pytest.fixture()
def conn(tmp_path: Path):
    c = kb.connect(tmp_path / "kb.db")
    kb.init(c)
    yield c
    c.close()


def conversation(session_id: str = "20261113-090000-room1.wav",
                 started_at: str | None = "2026-11-13T09:00:00",
                 summary: str = "Launch moved to Q2.",
                 transcript: str = "We moved the launch to Q2. Tom documents it."):
    return kb.Conversation(
        id=session_id, device="room1", started_at=started_at, duration_s=12.0,
        language=None, transcript=transcript, audio_bytes=2048, summary=summary,
    )


def test_a_conversation_is_one_row_with_its_summary(conn):
    conv = conversation()
    kb.upsert(conn, conv)
    row = kb.get(conn, conv.id)
    assert row["transcript"] == conv.transcript
    assert row["summary"] == conv.summary
    assert row["audio_bytes"] == 2048
    assert row["status"] == "transcribed"
    assert kb.count(conn) == 1


def test_upsert_replaces_rather_than_duplicates(conn):
    kb.upsert(conn, conversation(summary=""))
    kb.upsert(conn, conversation(summary="Launch moved to Q2."))
    assert kb.count(conn) == 1
    assert kb.get(conn, conversation().id)["summary"] == "Launch moved to Q2."


def test_days_group_conversations_and_carry_totals(conn):
    kb.upsert(conn, conversation("20261112-090000-room1.wav", "2026-11-12T09:00:00"))
    kb.upsert(conn, conversation("20261112-150000-room1.wav", "2026-11-12T15:00:00"))
    kb.upsert(conn, conversation("20261113-080000-room2.wav", "2026-11-13T08:00:00",
                                 summary="Hosting decision."))

    days = kb.list_days(conn)
    assert [d["day"] for d in days] == ["2026-11-13", "2026-11-12"]
    assert days[1]["sessions"] == 2
    assert days[1]["seconds"] == 24.0

    today = kb.sessions_for_day(conn, "2026-11-12")
    assert [s["id"] for s in today] == ["20261112-090000-room1.wav",
                                        "20261112-150000-room1.wav"]
    assert today[0]["summary"] == "Launch moved to Q2."


def test_search_finds_the_transcript_and_returns_the_summary(conn):
    kb.upsert(conn, conversation())
    hits = kb.search(conn, "launch")
    assert len(hits) == 1
    assert hits[0]["summary"] == "Launch moved to Q2."
    assert "launch" in hits[0]["snippet"].lower()
    assert kb.search(conn, "nonexistentword") == []


def test_a_conversation_without_a_start_time_is_not_a_day(conn):
    kb.upsert(conn, conversation("mystery.wav", started_at=None))
    assert kb.list_days(conn) == []
    assert kb.recent(conn)[0]["day"] == ""
