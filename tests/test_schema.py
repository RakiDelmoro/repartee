"""Unit tests for the canonical Repartee schema."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.schema import (
    CONTROL_IDS,
    FRAME_RATE,
    Conversation,
    Turn,
    active_frames,
    align_text_to_frames,
    control_histogram,
    derive_control_labels,
    frame_count,
    read_manifest,
    write_manifest,
)


def test_frame_count_and_constants() -> None:
    assert FRAME_RATE == 12.5
    assert frame_count(0.0) == 0
    assert frame_count(0.08) == 1
    assert frame_count(1.0) == 13  # 1.0 s lands mid-frame
    assert frame_count(1.6) == 20
    assert frame_count(1.6000000001) == 21


def test_active_frames_half_open_boundaries() -> None:
    turns = [Turn("user", 0.0, 0.96), Turn("agent", 0.96, 1.6)]
    user = active_frames(turns, 1.6, speaker="user")
    agent = active_frames(turns, 1.6, speaker="agent")
    assert user[:12].all() and not user[12:].any()
    assert agent[12:20].all() and not agent[:12].any()


def test_control_labels_backchannel_and_stop_precedence() -> None:
    turns = [
        Turn("user", 0.0, 1.0),          # LISTEN frames 0..12
        Turn("user", 1.2, 1.8),          # overlaps agent backchannel
        Turn("agent", 1.2, 1.6),         # 0.4 s -> BACKCHANNEL frames 15..19
        Turn("user", 2.2, 3.0),          # onset while agent speaks -> STOP window
        Turn("agent", 2.0, 3.5),         # SPEAK, except STOP frames 27..29
    ]
    labels = derive_control_labels(turns, 4.0)

    assert (labels[0:13] == CONTROL_IDS["LISTEN"]).all()
    assert (labels[13:15] == CONTROL_IDS["PAD"]).all()  # 1.04..1.2 s silence
    assert (labels[15:20] == CONTROL_IDS["BACKCHANNEL"]).all()
    # 1.6..2.0 s: agent silent, user still speaking => LISTEN, then PAD.
    assert (labels[20:23] == CONTROL_IDS["LISTEN"]).all()
    assert (labels[23:25] == CONTROL_IDS["PAD"]).all()
    assert (labels[25:27] == CONTROL_IDS["SPEAK"]).all()
    assert (labels[27:30] == CONTROL_IDS["STOP"]).all()
    assert (labels[30:44] == CONTROL_IDS["SPEAK"]).all()
    assert (labels[44:] == CONTROL_IDS["PAD"]).all()

    histogram = control_histogram(labels)
    assert histogram["STOP"] == 3
    assert histogram["BACKCHANNEL"] == 5
    assert histogram["LISTEN"] == 16
    assert histogram["PAD"] == 10
    assert histogram["SPEAK"] == 16


def test_stop_wins_over_backchannel_at_same_onset() -> None:
    turns = [
        Turn("agent", 0.0, 0.4),   # short and overlaps user => backchannel
        Turn("user", 0.2, 1.0),    # onset inside agent speech => STOP opens
    ]
    labels = derive_control_labels(turns, 1.5)
    # 0.2 s is frame 2; STOP covers frames 2..4 inside the short agent turn.
    assert (labels[0:2] == CONTROL_IDS["BACKCHANNEL"]).all()
    assert (labels[2:5] == CONTROL_IDS["STOP"]).all()
    assert (labels[5:13] == CONTROL_IDS["LISTEN"]).all()


def test_no_stop_on_clean_handoff() -> None:
    turns = [Turn("agent", 0.0, 1.0), Turn("user", 1.0, 2.0)]
    labels = derive_control_labels(turns, 2.0)
    assert not (labels == CONTROL_IDS["STOP"]).any()
    # Frame 12 (0.96..1.04 s) contains both speakers, so SPEAK takes
    # precedence at 80 ms resolution; the rest is LISTEN.
    assert labels[12] == CONTROL_IDS["SPEAK"]
    assert (labels[13:25] == CONTROL_IDS["LISTEN"]).all()


def test_touching_turns_do_not_overlap() -> None:
    turns = [Turn("user", 0.0, 0.8), Turn("agent", 0.8, 1.6)]
    labels = derive_control_labels(turns, 1.6)
    assert (labels[:10] == CONTROL_IDS["LISTEN"]).all()
    assert (labels[10:] == CONTROL_IDS["SPEAK"]).all()


class _WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [100 + i for i, _ in enumerate(text.split())]


def test_align_text_to_frames_spreads_tokens() -> None:
    turns = [Turn("agent", 0.0, 0.8, "one two three four")]
    text = align_text_to_frames(turns, 0.8, _WhitespaceTokenizer())
    assert text.shape == (10,)
    # Four tokens spread over ten frames: 0, 2, 5, 7.
    assert text.tolist() == [100, 0, 101, 0, 0, 102, 0, 103, 0, 0]


def test_conversation_validation_and_manifest_roundtrip(tmp_path) -> None:
    conversation = Conversation(
        id="conv_1",
        source="unit",
        license="CC-BY-4.0",
        duration_s=1.6,
        channels={"user": "a.flac", "agent": "b.flac"},
        turns=[Turn("user", 0.0, 0.8, "hi"), Turn("agent", 0.8, 1.6, "hello")],
        split="valid",
    )
    path = tmp_path / "manifest.jsonl"
    assert write_manifest(path, [conversation]) == 1
    loaded = list(read_manifest(path))
    assert loaded == [conversation]
    assert json.loads(path.read_text())["turns"][0]["speaker"] == "user"

    with pytest.raises(ValueError, match="overlapping user"):
        Conversation(
            id="bad",
            source="unit",
            license="CC-BY-4.0",
            duration_s=2.0,
            channels={"user": "a", "agent": "b"},
            turns=[Turn("user", 0.0, 1.0), Turn("user", 0.5, 1.5)],
        )

    with pytest.raises(ValueError, match="beyond duration"):
        Conversation(
            id="bad2",
            source="unit",
            license="CC-BY-4.0",
            duration_s=1.0,
            channels={"user": "a", "agent": "b"},
            turns=[Turn("agent", 0.0, 1.5)],
        )

    with pytest.raises(ValueError, match="channels"):
        Conversation(
            id="bad3",
            source="unit",
            license="CC-BY-4.0",
            duration_s=1.0,
            channels={"user": "a"},
            turns=[Turn("agent", 0.0, 1.0)],
        )

    with pytest.raises(ValueError, match="sorted"):
        Conversation(
            id="bad4",
            source="unit",
            license="CC-BY-4.0",
            duration_s=2.0,
            channels={"user": "a", "agent": "b"},
            turns=[Turn("user", 1.0, 1.5), Turn("agent", 0.0, 0.5)],
        )
