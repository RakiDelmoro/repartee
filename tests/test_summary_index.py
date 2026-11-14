"""Summary-level knowledge base: one row per conversation, transcript + summary."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import kb
from app.pipeline import store_conversation


class StubModel:
    def __init__(self, summary="Launch moved to Q2; Tom documents it.", boom=False):
        self.summary, self.boom = summary, boom
        self.seen = []

    def summarize(self, text):
        self.seen.append(text)
        if self.boom:
            raise RuntimeError("model crashed")
        return self.summary


def conversation(transcript: str = "We moved the launch to Q2. Tom documents it."):
    return kb.Conversation(
        id="20261113-070000-room1.wav", device="room1",
        started_at="2026-11-13T07:00:00", duration_s=12.0, language=None,
        transcript=transcript, audio_bytes=1234,
    )


def test_summary_is_stored_with_the_transcript(tmp_path: Path):
    conn = kb.connect(tmp_path / "kb.db")
    kb.init(conn)
    model = StubModel()

    stored = store_conversation(conn, conversation(), model)
    assert stored.summary == model.summary
    assert stored.status == "summarized"
    row = kb.get(conn, "20261113-070000-room1.wav")
    assert row["summary"] == model.summary
    assert row["transcript"].startswith("We moved")
    assert model.seen == [conversation().transcript]


def test_empty_transcript_is_transcribed_not_summarized(tmp_path: Path):
    conn = kb.connect(tmp_path / "kb.db")
    kb.init(conn)
    model = StubModel(summary="")
    stored = store_conversation(conn, conversation(transcript="  "), model)
    assert stored.summary == ""
    assert stored.status == "transcribed"
    row = kb.get(conn, "20261113-070000-room1.wav")
    assert row["summary"] == ""
    assert model.seen == []


def test_a_missing_summarizer_still_stores_the_transcript(tmp_path: Path):
    conn = kb.connect(tmp_path / "kb.db")
    kb.init(conn)

    class TranscribeOnly:
        pass

    stored = store_conversation(conn, conversation(), TranscribeOnly())
    assert stored.summary == ""
    row = kb.get(conn, "20261113-070000-room1.wav")
    assert row["summary"] == ""
    assert row["transcript"]


def test_a_failing_summarizer_still_stores_the_transcript(tmp_path: Path):
    conn = kb.connect(tmp_path / "kb.db")
    kb.init(conn)
    stored = store_conversation(conn, conversation(), StubModel(boom=True))
    assert stored.summary == ""
    row = kb.get(conn, "20261113-070000-room1.wav")
    assert row["status"] == "transcribed"
    assert row["transcript"].startswith("We moved")


def test_importing_a_transcript_stores_it_as_a_dated_conversation(tmp_path: Path):
    from app.pipeline import import_transcript

    conn = kb.connect(tmp_path / "kb.db")
    kb.init(conn)
    src = tmp_path / "meeting.txt"
    src.write_text("We moved the launch to Q2. Tom documents it.")

    conv = import_transcript(conn, src, "meeting", "2026-11-13T10:00:00", StubModel())
    assert conv.id == "20261113-100000-meeting-meeting.txt"
    assert conv.status == "imported"
    row = kb.get(conn, conv.id)
    assert row["summary"] == StubModel().summary
    assert row["audio_bytes"] is None
    assert kb.list_days(conn) == [{"day": "2026-11-13", "sessions": 1, "seconds": 0.0}]
