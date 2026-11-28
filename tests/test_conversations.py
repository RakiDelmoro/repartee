"""Conversation store: schema, upsert, and FTS search."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from app import conversations


def _conv(session_id: str, transcript: str, summary: str = "") -> conversations.Conversation:
    return conversations.Conversation(
        id=session_id,
        device="rep01",
        started_at="2026-11-28T12:00:00",
        duration_s=12.5,
        language="English",
        transcript=transcript,
        summary=summary,
        audio_path=f"/tmp/{session_id}",
        audio_bytes=1234,
        model="fake",
        status="summarized" if summary else "transcribed",
    )


def _db(tmp_path: Path):
    conn = conversations.connect(tmp_path / "conversations.db")
    conversations.init(conn)
    return conn


def test_upsert_round_trips_one_conversation(tmp_path):
    conn = _db(tmp_path)
    conversations.upsert(conn, _conv("a.wav", "We discussed the visa paperwork.", "Visa talk."))

    row = conversations.get(conn, "a.wav")

    assert row["device"] == "rep01"
    assert row["started_at"] == "2026-11-28T12:00:00"
    assert row["transcript"] == "We discussed the visa paperwork."
    assert row["summary"] == "Visa talk."
    assert row["status"] == "summarized"
    assert conversations.count(conn) == 1


def test_search_matches_transcript_and_summary(tmp_path):
    conn = _db(tmp_path)
    conversations.upsert(conn, _conv("a.wav", "We discussed the visa paperwork.", "Travel plans."))
    conversations.upsert(conn, _conv("b.wav", "The cat slept.", "Holiday memories."))

    by_transcript = conversations.search(conn, "visa")
    by_summary = conversations.search(conn, "holiday")

    assert [row["id"] for row in by_transcript] == ["a.wav"]
    assert "[visa]" in by_transcript[0]["transcript_snippet"]
    assert [row["id"] for row in by_summary] == ["b.wav"]
    assert "[holiday]" in by_summary[0]["summary_snippet"].lower()


def test_upsert_reindexes_changed_text(tmp_path):
    conn = _db(tmp_path)
    conversations.upsert(conn, _conv("a.wav", "The old transcript.", "Old summary."))
    conversations.upsert(conn, _conv("a.wav", "The new transcript.", "New summary."))

    assert conversations.search(conn, "old") == []
    assert [row["id"] for row in conversations.search(conn, "new")] == ["a.wav"]


def test_search_ignores_punctuation_and_empty_queries(tmp_path):
    conn = _db(tmp_path)
    conversations.upsert(conn, _conv("a.wav", "Visa, paperwork, and travel."))

    assert [row["id"] for row in conversations.search(conn, "visa!")] == ["a.wav"]
    assert conversations.search(conn, "!!!") == []


def test_recent_orders_by_start_and_respects_limit(tmp_path):
    conn = _db(tmp_path)
    older = _conv("a.wav", "older")
    newer = replace(_conv("b.wav", "newer"), started_at="2026-11-29T09:00:00")
    conversations.upsert(conn, older)
    conversations.upsert(conn, newer)

    rows = conversations.recent(conn, limit=1)

    assert [row["id"] for row in rows] == ["b.wav"]
