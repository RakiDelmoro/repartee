"""Pure-function tests for :mod:`src.build_real` (no network, no downloads)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src import build_real as br
from src import schema


def _tone(seconds: float, sample_rate: int = 16000, amplitude: float = 0.2) -> np.ndarray:
    t = np.arange(int(round(seconds * sample_rate)), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * math.pi * 220.0 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Turn normalisation
# ---------------------------------------------------------------------------


def test_normalize_turns_groups_words_by_gap_and_marks_events() -> None:
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.3, "type": "word"},
        {"word": "there", "start": 0.35, "end": 0.6, "type": "word"},
        # gap 0.8 s > 0.6 s starts a new turn
        {"word": "well", "start": 1.4, "end": 1.6, "type": "filler"},
        {"word": "ha", "start": 1.62, "end": 1.8, "type": "laugh"},
        {"word": "okay", "start": 1.85, "end": 2.0, "type": "word"},
    ]
    turns = br.normalize_turns(words, speaker="user", max_gap_s=0.6)
    assert len(turns) == 2
    assert turns[0].text == "Hello there"
    assert turns[0].start == 0.0 and turns[0].end == 0.6
    assert turns[1].text == "well [laugh] okay"
    assert all(turn.speaker == "user" for turn in turns)


def test_normalize_turns_merges_same_speaker_overlap() -> None:
    words = [
        {"word": "one two", "start": 0.0, "end": 1.0},
        {"word": "three", "start": 0.8, "end": 1.4},
    ]
    turns = br.normalize_turns(words, speaker="agent", max_gap_s=0.1)
    assert len(turns) == 1
    assert turns[0].start == 0.0 and turns[0].end == 1.4
    assert "two" in turns[0].text


def test_normalize_turns_accepts_objects_and_empty_timings() -> None:
    class Word:
        def __init__(self, word: str, start: float, end: float) -> None:
            self.word = word
            self.start = start
            self.end = end

    turns = br.normalize_turns([Word("hi", 0.1, 0.3)], speaker="user")
    assert len(turns) == 1
    assert turns[0].text == "hi"
    assert br.normalize_turns([], speaker="user") == []


@pytest.mark.parametrize(
    "word",
    [
        {"word": "bad", "start": 1.0, "end": 1.0},
        {"word": "bad", "start": -1.0, "end": 0.0},
        {"word": "bad", "start": float("nan"), "end": 1.0},
        {"word": "bad", "start": 0.0, "end": "nope"},
    ],
)
def test_normalize_turns_rejects_invalid_timing(word: dict) -> None:
    with pytest.raises(ValueError):
        br.normalize_turns([word], speaker="user")


def test_normalize_turns_rejects_unknown_speaker() -> None:
    with pytest.raises(ValueError):
        br.normalize_turns([], speaker="narrator")


def test_sanitize_word_timings_drops_a_few_bad_words() -> None:
    words = [
        {"word": f"w{i}", "start": i * 0.2, "end": i * 0.2 + 0.15}
        for i in range(40)
    ]
    words.insert(20, {"word": "bad", "start": 5.0, "end": 4.0})  # corrupt timing
    clean, dropped = br._sanitize_word_timings(words)
    assert dropped == 1 and len(clean) == 40
    with pytest.raises(ValueError):
        br._sanitize_word_timings(
            [{"word": "x", "start": 1.0, "end": 0.0}] * 3, max_drop_ratio=0.05
        )


# ---------------------------------------------------------------------------
# Holdout / split
# ---------------------------------------------------------------------------


def test_select_holdout_is_order_independent_and_exact() -> None:
    keys = [f"spk_{i:03d}" for i in range(200)]
    first = br.select_holdout(keys, 50, salt="salt")
    second = br.select_holdout(list(reversed(keys)), 50, salt="salt")
    assert first == second
    assert len(first) == 50
    assert first <= set(keys)
    assert br.select_holdout(keys[:3], 50, salt="salt") == set(keys[:3])
    assert br.select_holdout(keys, 0, salt="salt") == set()
    with pytest.raises(ValueError):
        br.select_holdout(keys, -1, salt="salt")


def test_assign_split_holds_out_conversation_and_speaker() -> None:
    assert br.assign_split("c1", ["s1"]) == "train"
    assert br.assign_split("c1", ["s1"], holdout_conversations={"c1"}) == "test"
    assert br.assign_split("c2", ["s1", "s9"], holdout_speakers={"s9"}) == "test"
    # The speaker holdout wins even when the conversation id is different.
    assert br.assign_split("c3", ["s9"], holdout_speakers={"s9"}) == "test"


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------


def test_passes_quality_accepts_clean_audio() -> None:
    audio = _tone(1.0)
    ok, reason = br.passes_quality(user_audio=audio, agent_audio=audio, sample_rate=16000)
    assert ok, reason


def test_passes_quality_drops_missing_and_short_channels() -> None:
    audio = _tone(1.0)
    assert br.passes_quality(user_audio=None, agent_audio=audio, sample_rate=16000) == (
        False,
        "missing channel",
    )
    ok, reason = br.passes_quality(
        user_audio=_tone(0.2),
        agent_audio=_tone(0.2),
        sample_rate=16000,
        min_duration_s=0.8,
    )
    assert not ok and "too short" in reason


def test_passes_quality_drops_excessive_clipping() -> None:
    clipped = np.ones(16000, dtype=np.float32)
    clean = _tone(1.0)
    ok, reason = br.passes_quality(user_audio=clipped, agent_audio=clean, sample_rate=16000)
    assert not ok and "clipping" in reason
    # Exactly 20 % is not "more than 20 %".
    audio = np.zeros(16000, dtype=np.float32)
    audio[:3200] = 1.0
    ok, _ = br.passes_quality(user_audio=audio, agent_audio=clean, sample_rate=16000)
    assert ok


# ---------------------------------------------------------------------------
# SRT parsing (otoSpeech turn-104h)
# ---------------------------------------------------------------------------


SRT = """1
00:00:01,000 --> 00:00:02,500
[Start] Hello there.

2
00:00:02,700 --> 00:00:04,000
[Acknowledgement Backchannel] mm-hmm.

3
00:00:03,900 --> 00:00:05,000
[Normal Turn] I agree.
"""


def test_parse_srt_reads_labels_and_milliseconds() -> None:
    cues = br.parse_srt(SRT)
    assert len(cues) == 3
    assert cues[0].start == 1.0 and cues[0].end == 2.5
    assert cues[0].label == "Start" and cues[0].text == "Hello there."
    assert cues[1].label == "Acknowledgement Backchannel"
    assert cues[1].text == "mm-hmm."
    assert cues[2].label == "Normal Turn"


def test_parse_srt_rejects_bad_block() -> None:
    with pytest.raises(ValueError):
        br.parse_srt("1\nnot a timestamp\nhello\n")


def test_srt_to_turns_merges_same_speaker_overlap() -> None:
    cues = br.parse_srt(SRT)
    turns = br.srt_to_turns(cues, speaker="user")
    # Cue 3 overlaps cue 2 (3.9 < 4.0) for the same speaker and is merged.
    assert len(turns) == 2
    assert turns[0].start == 1.0 and turns[0].end == 2.5
    assert turns[1].start == 2.7 and turns[1].end == 5.0
    assert "mm-hmm" in turns[1].text and "I agree" in turns[1].text


# ---------------------------------------------------------------------------
# Energy VAD and tar member classification
# ---------------------------------------------------------------------------


def test_energy_turns_finds_speech_burst() -> None:
    sample_rate = 16000
    silence = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
    speech = _tone(0.6, sample_rate=sample_rate, amplitude=0.3)
    audio = np.concatenate([silence, speech, silence])
    intervals = br.energy_turns(audio, sample_rate)
    assert len(intervals) == 1
    start, end = intervals[0]
    assert 0.35 <= start <= 0.6
    assert 0.95 <= end <= 1.2


def test_energy_turns_all_silence() -> None:
    assert br.energy_turns(np.zeros(16000, dtype=np.float32), 16000) == []


def test_energy_turns_rejects_bad_rate() -> None:
    with pytest.raises(ValueError):
        br.energy_turns(np.zeros(10, dtype=np.float32), 0)


def test_classify_open_yap_member_released_and_webdataset_names() -> None:
    assert br.classify_open_yap_member("conv_abc.a.flac") == ("conv_abc", "user")
    assert br.classify_open_yap_member("conv_abc.b.flac") == ("conv_abc", "agent")
    assert br.classify_open_yap_member("conv_abc.json") == ("conv_abc", "meta")
    assert br.classify_open_yap_member("conv_abc.preview.mp3") is None
    assert br.classify_open_yap_member("conv_abc/a.flac") == ("conv_abc", "user")
    assert br.classify_open_yap_member("conv_abc/json") == ("conv_abc", "meta")
    assert br.classify_open_yap_member("a.flac") == ("", "user")


def test_classify_otospeech_member() -> None:
    assert br.classify_otospeech_member("session_1.flac") == ("session_1", "audio")
    assert br.classify_otospeech_member("data/train/shard-1/session_1.json") == (
        "session_1",
        "meta",
    )
    assert br.classify_otospeech_member("flac") == ("", "audio")
    assert br.classify_otospeech_member("json") == ("", "meta")
    assert br.classify_otospeech_member("README.md") is None


# ---------------------------------------------------------------------------
# Frame alignment and schema integration
# ---------------------------------------------------------------------------


def test_grid_frames_and_clamp_turns() -> None:
    assert br._grid_frames(1.0) == 12
    frames = br._grid_frames(2.0)
    duration = frames / schema.FRAME_RATE
    turns = [
        schema.Turn("user", 0.0, 0.5, "hello"),
        schema.Turn("agent", 0.5, 10.0, "too long"),
        schema.Turn("user", 99.0, 100.0, "outside"),
    ]
    clamped = br._clamp_turns(turns, duration)
    assert all(turn.end <= duration for turn in clamped)
    assert all(turn.end > turn.start for turn in clamped)
    with pytest.raises(br.QualityDrop):
        br._grid_frames(0.01)


def test_safe_name_is_filesystem_safe() -> None:
    assert br._safe_name("conv/abc:1") == "conv_abc_1"
    assert br._safe_name("...") == "record"


# ---------------------------------------------------------------------------
# Gated access and tar grouping
# ---------------------------------------------------------------------------


def test_open_yap_requires_access_without_local_copy(tmp_path) -> None:
    with pytest.raises(br.AccessRequired) as error:
        br._open_yap_tars(raw_root=tmp_path, input_tar=None, sample_only=False)
    assert "theagenticdatacompany.com/open-yap-1k" in str(error.value)
    assert error.value.url in str(error.value)


def test_iter_tar_groups_released_and_bare_layouts(tmp_path) -> None:
    import io
    import tarfile

    archive = tmp_path / "sample.tar"

    def add(handle, name: str, payload: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))

    with tarfile.open(archive, "w") as handle:
        add(handle, "conv_a.a.flac", b"audio-a")
        add(handle, "conv_a.b.flac", b"audio-b")
        add(handle, "conv_a.json", b'{"conversation_id": "conv_a"}')
        add(handle, "missing.b.flac", b"audio-b")
        add(handle, "missing.json", b'{"conversation_id": "missing"}')
        add(handle, "bare.a.flac", b"audio-a")
        add(handle, "bare.b.flac", b"audio-b")
        add(handle, "bare.json", b'{"conversation_id": "bare"}')

    groups = list(br._iter_tar_groups([archive], br.classify_open_yap_member))
    by_key = {key: members for _, key, members in groups}
    assert set(by_key["conv_a"]) == {"user", "agent", "meta"}
    assert set(by_key["missing"]) == {"agent", "meta"}
    assert set(by_key["bare"]) == {"user", "agent", "meta"}
    assert by_key["conv_a"]["user"] == b"audio-a"
