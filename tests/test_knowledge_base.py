"""File-based knowledge base: render, read, search, and listing."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import knowledge_base


def _entry(
    session_id: str = "20261128-120000-rep01",
    *,
    started_at: str = "2026-11-28T12:00:00",
    summary: str = "Visa paperwork.",
    transcript: str = "We discussed the visa paperwork.",
) -> knowledge_base.Entry:
    return knowledge_base.Entry(
        id=session_id,
        device="rep01",
        started_at=started_at,
        ended_at="2026-11-28T12:00:12",
        duration_s=12.5,
        language="English",
        model="fake",
        audio="data/recordings/20261128-120000-rep01.wav",
        audio_bytes=1234,
        status="summarized",
        created_at="2026-11-28T12:00:13+00:00",
        summary=summary,
        transcript=transcript,
    )


def test_write_and_read_round_trip(tmp_path):
    root = tmp_path / "knowledge-base"

    path = knowledge_base.write(root, _entry())
    entry = knowledge_base.read(path)

    assert path == root / "2026-11-28" / "20261128-120000-rep01.md"
    assert entry.id == "20261128-120000-rep01"
    assert entry.started_at == "2026-11-28T12:00:00"
    assert entry.summary == "Visa paperwork."
    assert entry.transcript == "We discussed the visa paperwork."
    assert entry.audio == "data/recordings/20261128-120000-rep01.wav"


def test_existing_file_is_kept_unless_forced(tmp_path):
    root = tmp_path / "knowledge-base"
    knowledge_base.write(root, _entry(summary="Old summary."))

    knowledge_base.write(root, _entry(summary="New summary."))
    kept = knowledge_base.read(knowledge_base.find(root, "20261128-120000-rep01"))

    knowledge_base.write(root, _entry(summary="New summary."), force=True)
    replaced = knowledge_base.read(knowledge_base.find(root, "20261128-120000-rep01"))

    assert kept.summary == "Old summary."
    assert replaced.summary == "New summary."


def test_search_matches_summary_and_transcript(tmp_path):
    root = tmp_path / "knowledge-base"
    knowledge_base.write(
        root, _entry(summary="Travel plans.", transcript="We discussed the visa paperwork.")
    )
    knowledge_base.write(
        root,
        _entry(
            "20261129-090000-rep01",
            started_at="2026-11-29T09:00:00",
            summary="Holiday memories.",
            transcript="The cat slept.",
        ),
    )

    by_transcript = knowledge_base.search(root, "visa")
    by_summary = knowledge_base.search(root, "holiday")

    assert [row["id"] for row in by_transcript] == ["20261128-120000-rep01"]
    assert "visa" in by_transcript[0]["snippet"].lower()
    assert [row["id"] for row in by_summary] == ["20261129-090000-rep01"]
    assert "holiday" in by_summary[0]["snippet"].lower()


def test_search_is_case_insensitive_and_limited(tmp_path):
    root = tmp_path / "knowledge-base"
    for day in range(1, 4):
        knowledge_base.write(
            root,
            _entry(
                f"202611{day:02d}-120000-rep01",
                started_at=f"2026-11-{day:02d}T12:00:00",
                transcript="The bank was busy.",
            ),
        )

    rows = knowledge_base.search(root, "BANK", limit=2)

    assert len(rows) == 2
    assert all("bank" in row["snippet"].lower() for row in rows)


def test_recent_orders_by_start_and_respects_limit(tmp_path):
    root = tmp_path / "knowledge-base"
    knowledge_base.write(root, _entry())
    knowledge_base.write(
        root,
        _entry("20261129-090000-rep01", started_at="2026-11-29T09:00:00"),
    )

    rows = knowledge_base.recent(root, limit=1)

    assert [row["id"] for row in rows] == ["20261129-090000-rep01"]


def test_get_returns_full_entry_or_none(tmp_path):
    root = tmp_path / "knowledge-base"
    knowledge_base.write(root, _entry())

    assert knowledge_base.get(root, "20261128-120000-rep01")["transcript"] == (
        "We discussed the visa paperwork."
    )
    assert knowledge_base.get(root, "missing") is None
    assert knowledge_base.count(root) == 1


def test_duplicate_ids_are_rejected(tmp_path):
    root = tmp_path / "knowledge-base"
    knowledge_base.write(root, _entry())
    other = root / "2026-11-29" / "20261128-120000-rep01.md"
    other.parent.mkdir(parents=True)
    other.write_text("not read")

    with pytest.raises(ValueError, match="duplicate conversation id"):
        knowledge_base.find(root, "20261128-120000-rep01")
