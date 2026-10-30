"""Scheduler, jitter-buffer and engine-protocol tests.

These are scheduler/AEC evidence only.  They do not load the checkpoint and
they do not claim Gate 7.
"""

from __future__ import annotations

import json
import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.aec import NLMSAEC  # noqa: E402
from serve.loop import (  # noqa: E402
    EngineOutput,
    FakeDuplexEngine,
    JitterBuffer,
    Lfm3DuplexEngine,
    OUT_FRAME_SAMPLES,
    TickClock,
    WavFileSink,
    WavFileSource,
    _rms_dbfs,
    run_session,
)

MIC = 1280


class ArraySource:
    def __init__(self, frames) -> None:
        self._frames = deque(frames)

    def read(self):
        return self._frames.popleft() if self._frames else None


class ZeroSource:
    def __init__(self, count: int) -> None:
        self._remaining = count

    def read(self):
        if self._remaining <= 0:
            return None
        self._remaining -= 1
        return np.zeros(MIC, dtype=np.float32)


class RecordingSink:
    def __init__(self, keep: bool = True) -> None:
        self.keep = keep
        self.frames: list[np.ndarray] = []
        self.writes = 0
        self.threads: set[int] = set()
        self._entered = False

    def write(self, audio) -> None:
        assert not self._entered, "sink.write is not re-entrant; there must be one writer"
        self._entered = True
        try:
            array = np.asarray(audio, dtype=np.float32).reshape(-1)
            self.writes += 1
            self.threads.add(threading.get_ident())
            if self.keep:
                self.frames.append(array.copy())
        finally:
            self._entered = False


class BurstEngine:
    """Speaks a four-frame burst, then STOPs on the next tick."""

    def reset(self) -> None:
        return None

    def step(self, user_audio_16k, tick_index: int) -> EngineOutput:
        if tick_index == 0:
            frames = [np.full(OUT_FRAME_SAMPLES, value, dtype=np.float32) for value in (1.0, 2.0, 3.0, 4.0)]
            return EngineOutput("SPEAK", np.concatenate(frames))
        if tick_index == 1:
            return EngineOutput("STOP")
        return EngineOutput("PAD")


class BackchannelEngine:
    def reset(self) -> None:
        return None

    def step(self, user_audio_16k, tick_index: int) -> EngineOutput:
        if tick_index == 0:
            return EngineOutput("BACKCHANNEL", np.ones(24_000 * 3 // 2, dtype=np.float32))
        return EngineOutput("PAD")


class ChattyEngine:
    """Keeps saying SPEAK even while the user talks over it."""

    def reset(self) -> None:
        return None

    def step(self, user_audio_16k, tick_index: int) -> EngineOutput:
        return EngineOutput("SPEAK", np.full(OUT_FRAME_SAMPLES, 0.1, dtype=np.float32))


class BargeInEngine:
    """Starts speaking immediately and STOPs one frame after user speech."""

    def __init__(self) -> None:
        self._prev_active = False

    def reset(self) -> None:
        self._prev_active = False

    def step(self, user_audio_16k, tick_index: int) -> EngineOutput:
        active = _rms_dbfs(user_audio_16k) > -40.0
        if self._prev_active:
            return EngineOutput("STOP")
        self._prev_active = active
        return EngineOutput("SPEAK", np.full(OUT_FRAME_SAMPLES, 0.1, dtype=np.float32))


def _speech_source(ticks: int, onset: int) -> ArraySource:
    frames = []
    for tick in range(ticks):
        if tick < onset:
            frames.append(np.zeros(MIC, dtype=np.float32))
        else:
            t = np.arange(MIC, dtype=np.float32) / 16_000.0
            frames.append((0.1 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32))
    return ArraySource(frames)


def test_real_engine_is_lazy_and_needs_no_checkpoint_to_construct():
    engine = Lfm3DuplexEngine("not/a/real/checkpoint")
    assert engine.model is None
    assert engine.processor is None


# ---------------------------------------------------------------------------
# Jitter buffer
# ---------------------------------------------------------------------------


def test_jitter_buffer_is_bounded_and_counts_overruns():
    buffer = JitterBuffer(target_fill_ms=30.0, max_fill_ms=200.0)
    frame = np.ones(OUT_FRAME_SAMPLES, dtype=np.float32)
    for _ in range(5):
        buffer.push(frame)
    assert buffer.depth_ms <= 200.0 + 1e-9
    assert buffer.overruns >= 1
    out = buffer.pop(OUT_FRAME_SAMPLES)
    assert out.shape == (OUT_FRAME_SAMPLES,)
    assert out.dtype == np.float32
    assert buffer.depth_samples <= buffer.max_samples


def test_jitter_buffer_drops_oldest_not_newest():
    buffer = JitterBuffer(target_fill_ms=30.0, max_fill_ms=200.0)  # 4800 samples
    buffer.push(np.full(2400, 1.0, dtype=np.float32))
    buffer.push(np.full(2400, 2.0, dtype=np.float32))
    buffer.push(np.full(2400, 3.0, dtype=np.float32))
    # The oldest 2400-sample block is evicted; the two newest remain.
    assert buffer.depth_samples == 4800
    out = buffer.pop(OUT_FRAME_SAMPLES)
    assert np.all(out == 2.0)


def test_jitter_buffer_underrun_and_lone_clip_playout():
    buffer = JitterBuffer(target_fill_ms=30.0, max_fill_ms=1000.0)
    buffer.push(np.ones(1000, dtype=np.float32))
    assert buffer.pop(OUT_FRAME_SAMPLES).sum() == 0.0  # one-tick start-up reserve
    out = buffer.pop(OUT_FRAME_SAMPLES)  # lone clip is released, padded
    assert out[0] == 1.0
    assert buffer.underruns == 1


def test_stop_drop_keeps_exact_80ms_tail():
    buffer = JitterBuffer(target_fill_ms=30.0, max_fill_ms=1000.0)
    for value in (1.0, 2.0, 3.0, 4.0):
        buffer.push(np.full(OUT_FRAME_SAMPLES, value, dtype=np.float32))
    buffer.drop(keep_samples=OUT_FRAME_SAMPLES)
    assert buffer.depth_samples == OUT_FRAME_SAMPLES
    tail = buffer.pop(OUT_FRAME_SAMPLES)
    assert np.all(tail == 4.0)


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------


def test_tick_clock_uses_absolute_deadlines():
    clock = TickClock(period=0.08, realtime=False, start=100.0)
    for _ in range(5):
        assert clock.wait() == 0.0
    assert clock.tick_index == 5
    assert clock._deadline == pytest.approx(100.0 + 5 * 0.08)
    stats = clock.stats()
    assert stats["count"] == 5
    assert stats["p50"] == 0.0 and stats["p95"] == 0.0


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------


def test_fake_engine_through_10s_is_125_ticks_and_one_writer():
    ticks = 125  # 10 s at 12.5 Hz
    sink = RecordingSink()
    metrics = run_session(ZeroSource(ticks), sink, FakeDuplexEngine(), ticks, realtime=False)

    assert metrics["ticks"] == ticks
    assert metrics["requested_ticks"] == ticks
    assert sink.writes == ticks
    assert all(frame.shape == (OUT_FRAME_SAMPLES,) for frame in sink.frames)
    assert sink.threads == {threading.get_ident()}
    assert sum(metrics["controls"].values()) == ticks
    assert metrics["controls"]["SPEAK"] >= 15  # ticks 10..24 plus the later script
    assert metrics["controls"]["BACKCHANNEL"] == 1
    assert metrics["controls"]["STOP"] == 1  # tick 25
    assert metrics["jitter_buffer"]["max_depth_ms"] <= 1000.0 + 1e-9
    assert metrics["gate7_claimed"] is False


def test_stop_drops_queue_but_writes_80ms_tail():
    sink = RecordingSink()
    metrics = run_session(ZeroSource(4), sink, BurstEngine(), 4, realtime=False)
    assert sink.writes == 4
    assert np.all(sink.frames[0] == 1.0)
    assert np.all(sink.frames[1] == 4.0)  # STOP retained the newest 80 ms
    assert np.all(sink.frames[2] == 0.0)
    assert metrics["barge_in"]["stop_events"] == 1


def test_backchannel_clip_is_truncated_to_short_clip():
    sink = RecordingSink(keep=False)
    metrics = run_session(ZeroSource(2), sink, BackchannelEngine(), 2, realtime=False)
    # 0.8 s maximum clip = 800 ms of queued audio.
    assert metrics["jitter_buffer"]["max_depth_ms"] <= 800.0 + 1e-9
    assert metrics["controls"]["BACKCHANNEL"] == 1


def test_barge_in_stop_within_two_frames():
    ticks = 10
    sink = RecordingSink(keep=False)
    metrics = run_session(_speech_source(ticks, onset=5), sink, BargeInEngine(), ticks, realtime=False)
    latencies = metrics["barge_in"]["latency_frames"]
    assert latencies, metrics["barge_in"]
    assert max(latencies) <= 2.0
    assert metrics["barge_in"]["stats"]["count"] == len(latencies)


def test_barge_in_guard_bounds_latency_when_model_never_stops():
    ticks = 12
    sink = RecordingSink()
    metrics = run_session(
        _speech_source(ticks, onset=5), sink, ChattyEngine(), ticks, realtime=False
    )
    assert metrics["barge_in"]["guard_drops"] >= 1
    assert metrics["barge_in"]["latency_frames"]
    assert max(metrics["barge_in"]["latency_frames"]) <= 2.0
    # After the guard fires, agent audio is dropped to silence.
    assert np.all(sink.frames[-1] == 0.0)


def test_no_buffer_growth_over_long_run():
    ticks = 1250  # 100 s
    sink = RecordingSink(keep=False)
    metrics = run_session(ZeroSource(ticks), sink, FakeDuplexEngine(), ticks, realtime=False)
    assert metrics["ticks"] == ticks
    assert metrics["jitter_buffer"]["max_depth_ms"] <= 1000.0 + 1e-9
    assert metrics["jitter_buffer"]["final_depth_ms"] == 0.0
    assert sum(metrics["controls"].values()) == ticks


def test_session_with_aec_uses_reference_and_writes_every_tick():
    ticks = 20
    sink = RecordingSink(keep=False)
    metrics = run_session(
        ZeroSource(ticks), sink, FakeDuplexEngine(), ticks, aec=NLMSAEC(), realtime=False
    )
    assert metrics["ticks"] == ticks
    assert sink.writes == ticks
    assert metrics["gate7_claimed"] is False


def test_source_eof_stops_session_and_metrics_json(tmp_path):
    sink = RecordingSink(keep=False)
    path = tmp_path / "metrics.json"
    metrics = run_session(ZeroSource(3), sink, FakeDuplexEngine(), 10, path, realtime=False)
    assert metrics["ticks"] == 3
    assert metrics["source_exhausted"] is True
    assert sink.writes == 3
    saved = json.loads(path.read_text())
    for key in ("ticks", "controls", "jitter_buffer", "ttfa_ms", "barge_in", "rss", "gate7_claimed"):
        assert key in saved
    assert saved["gate7_claimed"] is False


# ---------------------------------------------------------------------------
# WAV file source / sink
# ---------------------------------------------------------------------------


def test_wav_file_sink_and_source_roundtrip(tmp_path):
    path = tmp_path / "agent.wav"
    frames = [
        (value * np.ones(OUT_FRAME_SAMPLES, dtype=np.float32) * 0.05) for value in range(1, 13)
    ]
    sink = WavFileSink(path)
    try:
        for frame in frames:
            sink.write(frame)
    finally:
        sink.close()
    assert path.exists()

    source = WavFileSource(path, sample_rate=24_000, frame_samples=OUT_FRAME_SAMPLES)
    try:
        read = []
        while True:
            frame = source.read()
            if frame is None:
                break
            read.append(frame)
    finally:
        source.close()
    assert len(read) == 12
    for expected, actual in zip(frames, read):
        assert np.allclose(expected, actual, atol=2.0 / 32768.0)
