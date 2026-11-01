"""Fixed 80 ms full-duplex serving loop for Repartee.

The loop owns the wall clock, the jitter buffer and the only call to the audio
sink.  Engines are pure functions of one 80 ms user frame and the tick index:

    step(user_audio_16k, tick_index) -> EngineOutput(control, audio_24k, text)

Design notes and deviations
--------------------------
* Tick: absolute ``time.monotonic`` deadlines, so a slow tick never shifts the
  following deadlines.  p50/p95 wake-up jitter is recorded.
* The jitter buffer is bounded (default 1 s).  It primes playback with one
  decoded frame plus a configurable 20--40 ms reserve (default 30 ms); because
  LiquidCodec emits 80 ms frames, the effective steady reserve is one 80 ms
  quantum.  The reserve target is therefore a scheduling target, not a promise
  of a 30 ms end-to-end buffer.
* ``STOP`` drops queued audio but leaves the newest 80 ms so speech does not
  click.  A backchannel clip is clipped to `schema.BACKCHANNEL_MAX_SECONDS`
  before it is queued.
* A single thread calls ``sink.write`` exactly once per tick.  Engine code
  never writes to the sink.
* TTFA is measured in tick quanta (80 ms), not wall time: the offline WAV path
  runs faster than realtime, so wall-clock TTFA would be meaningless.  The
  metric is explicitly labelled as tick-quantised.
* Gate 7 is *not* claimed by these metrics.  They are scheduler/AEC evidence;
  a real 10-minute live run is required for Gate 7.

``Lfm3DuplexEngine`` is the real model engine.  It uses the checkpoint's
streaming APIs directly:

* ``processor.feature_extractor.stream`` for log-mel frames;
* ``model.prefill_audio`` for the encoder and user-token prefill (the helper
  returns the user-token hidden states, both caches and the attention mask);
* ``lm_head`` restricted to the five control ids for the per-frame decision;
* ``model.sample_audio_frame`` from the post-control hidden state and
  ``model.model.get_audio_out_embeds`` for the agent frame;
* ``LiquidCodec`` streaming decode with its decoder cache.

Every heavy dependency is imported lazily, so importing this module works on a
CPU-only host without the checkpoint.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from src.schema import (
    AGENT_SAMPLE_RATE,
    BACKCHANNEL_MAX_SECONDS,
    CONTROL_LABELS,
    CONTROL_LABEL_TO_TOKEN_NAME,
    FRAME_SECONDS,
    FRAME_SAMPLES,
    USER_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)

MIC_FRAME_SAMPLES = FRAME_SAMPLES[USER_SAMPLE_RATE]  # 1280
OUT_FRAME_SAMPLES = FRAME_SAMPLES[AGENT_SAMPLE_RATE]  # 1920
REFERENCE_SAMPLES = int(0.5 * USER_SAMPLE_RATE)  # last 500 ms played


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _rms_dbfs(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if audio.size == 0:
        return -math.inf
    rms = float(np.sqrt(np.mean(audio * audio)))
    return 20.0 * math.log10(max(rms, 1e-12))


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "max": None, "count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "count": int(array.size),
    }


def _rss_mb() -> float | None:
    """Resident set size in MiB, or ``None`` when the platform cannot report it."""
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError, AttributeError):
        try:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except (ImportError, OSError):
            return None


def _rss_slope(samples: list[float], times: list[float]) -> float | None:
    if len(samples) < 2 or len(times) < 2:
        return None
    t = np.asarray(times, dtype=np.float64)
    r = np.asarray(samples, dtype=np.float64)
    t = t - t[0]
    if t[-1] <= 0:
        return None
    slope_per_second = float(np.polyfit(t, r, 1)[0])
    return slope_per_second * 60.0


def _resample_24k_to_16k(audio: np.ndarray) -> np.ndarray:
    """Linear 24 kHz -> 16 kHz conversion for the AEC reference only.

    The reference path does not need studio quality; it needs to be cheap and
    deterministic inside the 80 ms tick.  Linear interpolation is adequate for
    an echo reference and avoids loading a resampler on CPU-only imports.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    out_samples = int(round(audio.size * USER_SAMPLE_RATE / AGENT_SAMPLE_RATE))
    if out_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    positions = np.arange(out_samples, dtype=np.float64) * (AGENT_SAMPLE_RATE / USER_SAMPLE_RATE)
    return np.interp(positions, np.arange(audio.size, dtype=np.float64), audio).astype(np.float32)


def _fit_mic_frame(frame: np.ndarray, samples: int = MIC_FRAME_SAMPLES) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32).reshape(-1)
    if frame.size == samples:
        return frame
    if frame.size > samples:
        return frame[:samples]
    return np.pad(frame, (0, samples - frame.size))


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------


class TickClock:
    """Absolute monotonic 80 ms deadlines with wake-up jitter statistics."""

    def __init__(self, period: float = FRAME_SECONDS, *, realtime: bool = True, start: float | None = None) -> None:
        if period < 0:
            raise ValueError("period must be non-negative")
        self.period = float(period)
        self.realtime = bool(realtime)
        self._start = time.monotonic() if start is None else float(start)
        self._deadline = self._start
        self._ticks = 0
        self.jitters_ms: list[float] = []

    def reset(self) -> None:
        self._start = time.monotonic()
        self._deadline = self._start
        self._ticks = 0
        self.jitters_ms.clear()

    @property
    def tick_index(self) -> int:
        return self._ticks

    def wait(self) -> float:
        """Block until the next absolute deadline and return wake-up jitter in ms.

        In ``realtime=False`` mode no sleeping occurs and zero jitter is
        recorded; the deadline arithmetic is still advanced.
        """
        deadline = self._deadline
        self._deadline += self.period
        self._ticks += 1
        if not self.realtime:
            self.jitters_ms.append(0.0)
            return 0.0
        delay = deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        jitter = (time.monotonic() - deadline) * 1000.0
        self.jitters_ms.append(jitter)
        return jitter

    def stats(self) -> dict[str, float | None]:
        return _percentiles(self.jitters_ms)

    @property
    def p50_jitter_ms(self) -> float | None:
        return self.stats()["p50"]

    @property
    def p95_jitter_ms(self) -> float | None:
        return self.stats()["p95"]


# ---------------------------------------------------------------------------
# Jitter buffer
# ---------------------------------------------------------------------------


class JitterBuffer:
    """Bounded mono sample queue with prime/underrun/overrun accounting.

    Playback primes once the queue holds one output frame plus the configured
    20--40 ms reserve.  This adds at most two 80 ms frames of start-up latency
    while keeping the steady reserve at one decoded quantum.
    """

    def __init__(
        self,
        sample_rate: int = AGENT_SAMPLE_RATE,
        *,
        target_fill_ms: float = 30.0,
        max_fill_ms: float = 1000.0,
        output_frame_samples: int = OUT_FRAME_SAMPLES,
    ) -> None:
        if not 20.0 <= float(target_fill_ms) <= 40.0:
            raise ValueError("target_fill_ms must be within the 20-40 ms design range")
        if max_fill_ms <= 0 or max_fill_ms <= target_fill_ms:
            raise ValueError("max_fill_ms must exceed target_fill_ms")
        if output_frame_samples <= 0:
            raise ValueError("output_frame_samples must be positive")
        self.sample_rate = int(sample_rate)
        self.target_fill_ms = float(target_fill_ms)
        self.max_fill_ms = float(max_fill_ms)
        self.output_frame_samples = int(output_frame_samples)
        self.max_samples = int(round(max_fill_ms * sample_rate / 1000.0))
        self.prime_threshold_samples = output_frame_samples + int(round(target_fill_ms * sample_rate / 1000.0))
        self._chunks: deque[np.ndarray] = deque()
        self._length = 0
        self._primed = False
        self._active = False
        self._hold_ticks = 0
        self.underruns = 0
        self.overruns = 0
        self.max_depth_samples = 0

    # -- introspection ----------------------------------------------------
    @property
    def depth_samples(self) -> int:
        return self._length

    @property
    def depth_ms(self) -> float:
        return self._length * 1000.0 / self.sample_rate

    @property
    def primed(self) -> bool:
        return self._primed

    @property
    def active(self) -> bool:
        return self._active

    # -- queue operations -------------------------------------------------
    def push(self, audio: np.ndarray) -> None:
        """Append mono audio, dropping the oldest samples if the bound is hit."""
        frame = np.asarray(audio, dtype=np.float32).reshape(-1)
        if frame.size == 0:
            return
        if frame.size > self.max_samples:
            frame = frame[-self.max_samples :].copy()
            self.overruns += 1
        overflow = self._length + frame.size - self.max_samples
        if overflow > 0:
            self.overruns += 1
            while overflow > 0 and self._chunks:
                front = self._chunks[0]
                if front.size <= overflow:
                    self._chunks.popleft()
                    self._length -= front.size
                    overflow -= front.size
                else:
                    self._chunks[0] = front[overflow:].copy()
                    self._length -= overflow
                    overflow = 0
        self._chunks.append(np.ascontiguousarray(frame, dtype=np.float32))
        self._length += frame.size
        self._active = True
        self._hold_ticks = 0
        self.max_depth_samples = max(self.max_depth_samples, self._length)

    def pop(self, n: int) -> np.ndarray:
        """Remove exactly ``n`` samples; pad with silence on underrun."""
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)
        if not self._active or not self._chunks:
            self._active = False
            return out
        if not self._primed:
            if self._length < self.prime_threshold_samples:
                # Hold one tick to accumulate the reserve, but never strand a
                # lone short clip below the prime mark forever.
                self._hold_ticks += 1
                if self._hold_ticks <= 1:
                    return out
            self._primed = True
        if self._length < n:
            self.underruns += 1
            offset = 0
            while self._chunks and offset < n:
                frame = self._chunks.popleft()
                take = min(frame.size, n - offset)
                out[offset : offset + take] = frame[:take]
                offset += take
                self._length -= take
            self._active = False
            self._primed = False
            return out
        offset = 0
        while offset < n:
            frame = self._chunks[0]
            take = min(frame.size, n - offset)
            out[offset : offset + take] = frame[:take]
            offset += take
            self._length -= take
            if take == frame.size:
                self._chunks.popleft()
            else:
                self._chunks[0] = frame[take:]
        if self._length == 0:
            self._active = False
            self._primed = False
        return out

    def drop(self, *, keep_samples: int = 0) -> None:
        """Drop queued audio, keeping at most ``keep_samples`` from the tail."""
        keep_samples = max(0, int(keep_samples))
        if keep_samples == 0 or not self._chunks:
            self._chunks.clear()
            self._length = 0
            self._active = False
            self._primed = False
            return
        if self._length <= keep_samples:
            self._active = bool(self._length)
            self._primed = bool(self._length)
            return
        drop_count = self._length - keep_samples
        while drop_count > 0 and self._chunks:
            front = self._chunks[0]
            if front.size <= drop_count:
                self._chunks.popleft()
                self._length -= front.size
                drop_count -= front.size
            else:
                self._chunks[0] = front[drop_count:].copy()
                self._length -= drop_count
                drop_count = 0
        self._active = True
        # The retained tail must play immediately, even below the prime mark.
        self._primed = True

    def reset(self) -> None:
        self._chunks.clear()
        self._length = 0
        self._primed = False
        self._active = False
        self._hold_ticks = 0

    def snapshot(self) -> dict[str, float | int]:
        return {
            "underruns": self.underruns,
            "overruns": self.overruns,
            "max_depth_ms": self.max_depth_samples * 1000.0 / self.sample_rate,
            "final_depth_ms": self.depth_ms,
            "target_fill_ms": self.target_fill_ms,
            "max_fill_ms": self.max_fill_ms,
        }


# ---------------------------------------------------------------------------
# Engine protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineOutput:
    """One 80 ms engine decision."""

    control: str
    audio_24k: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    text: str = ""

    def __post_init__(self) -> None:
        control = str(self.control).upper()
        if control not in CONTROL_LABELS:
            raise ValueError(f"control must be one of {CONTROL_LABELS}, got {control!r}")
        audio = (
            np.zeros(0, dtype=np.float32)
            if self.audio_24k is None
            else np.asarray(self.audio_24k, dtype=np.float32).reshape(-1)
        )
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "audio_24k", audio)
        object.__setattr__(self, "text", str(self.text))


@runtime_checkable
class DuplexEngine(Protocol):
    def step(self, user_audio_16k: np.ndarray, tick_index: int) -> EngineOutput:
        ...


@runtime_checkable
class AudioSource(Protocol):
    def read(self) -> np.ndarray | None:
        """Return 1280 samples at 16 kHz, or ``None`` at end of stream."""
        ...


@runtime_checkable
class AudioSink(Protocol):
    def write(self, audio_24k: np.ndarray) -> None:
        """Consume 1920 samples at 24 kHz."""
        ...


# ---------------------------------------------------------------------------
# File / device sources and sinks
# ---------------------------------------------------------------------------


class WavFileSource:
    """Streaming 16 kHz mono WAV source (exactly one 80 ms frame per read)."""

    def __init__(self, path: str | Path, sample_rate: int = USER_SAMPLE_RATE, frame_samples: int = MIC_FRAME_SAMPLES):
        self.path = Path(path)
        self.sample_rate = int(sample_rate)
        self.frame_samples = int(frame_samples)
        import soundfile as sf  # lazy; audio dependency

        self._sf = sf
        self._handle = sf.SoundFile(str(self.path))
        self._eof = False
        self._buffer: np.ndarray | None = None
        self._cursor = 0
        if self._handle.samplerate != self.sample_rate:
            # Fall back to the shared resampler, which loads and converts the
            # file once.  Keeps the common 16 kHz path streaming.
            self._handle.close()
            from src.audio import read_audio

            audio = read_audio(self.path, self.sample_rate)
            self._buffer = audio
            self._handle = None

    def read(self) -> np.ndarray | None:
        if self._buffer is not None:
            if self._cursor >= self._buffer.size:
                return None
            chunk = self._buffer[self._cursor : self._cursor + self.frame_samples]
            self._cursor += chunk.size
            if chunk.size < self.frame_samples:
                chunk = np.pad(chunk, (0, self.frame_samples - chunk.size))
            return np.asarray(chunk, dtype=np.float32)
        if self._eof:
            return None
        data = self._handle.read(frames=self.frame_samples, dtype="float32", always_2d=True)
        if data.shape[0] == 0:
            self._eof = True
            return None
        if data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data[:, 0]
        chunk = np.asarray(data, dtype=np.float32)
        if chunk.size < self.frame_samples:
            chunk = np.pad(chunk, (0, self.frame_samples - chunk.size))
            self._eof = True
        return chunk

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "WavFileSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class WavFileSink:
    """Streaming 24 kHz mono PCM_16 WAV sink."""

    def __init__(self, path: str | Path, sample_rate: int = AGENT_SAMPLE_RATE):
        self.path = Path(path)
        self.sample_rate = int(sample_rate)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        import soundfile as sf  # lazy; audio dependency

        self._sf = sf
        self._handle = sf.SoundFile(
            str(self.path), mode="w", samplerate=self.sample_rate, channels=1, subtype="PCM_16"
        )
        self.samples_written = 0

    def write(self, audio_24k: np.ndarray) -> None:
        audio = np.asarray(audio_24k, dtype=np.float32).reshape(-1)
        if audio.size:
            self._handle.write(audio)
            self.samples_written += audio.size

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "WavFileSink":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SoundDeviceSource:
    """Live 16 kHz mono microphone source (optional ``sounddevice``)."""

    def __init__(self, sample_rate: int = USER_SAMPLE_RATE, frame_samples: int = MIC_FRAME_SAMPLES, device=None):
        import sounddevice as sd  # lazy, optional

        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_samples,
            device=device,
        )
        self._stream.start()
        self.frame_samples = frame_samples

    def read(self) -> np.ndarray:
        data, overflowed = self._stream.read(self.frame_samples)
        if overflowed:
            logger.warning("sounddevice input overflow")
        return np.asarray(data[:, 0], dtype=np.float32)

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class SoundDeviceSink:
    """Live 24 kHz mono speaker sink (optional ``sounddevice``)."""

    def __init__(self, sample_rate: int = AGENT_SAMPLE_RATE, device=None):
        import sounddevice as sd  # lazy, optional

        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            # The device period is in the 20-40 ms range even though the loop
            # writes 80 ms blocks.
            blocksize=int(0.03 * sample_rate),
            device=device,
        )
        self._stream.start()

    def write(self, audio_24k: np.ndarray) -> None:
        self._stream.write(np.asarray(audio_24k, dtype=np.float32).reshape(-1))

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


# ---------------------------------------------------------------------------
# Fake engine (CI only)
# ---------------------------------------------------------------------------


class FakeDuplexEngine:
    """Deterministic scripted engine used by tests and ``python -m serve --fake``.

    This is not a model and is never used on the production path.  The script
    exercises LISTEN, SPEAK, STOP, BACKCHANNEL and a user barge-in.
    """

    def __init__(self, *, tone_hz: float = 440.0, amplitude: float = 0.1, user_speech_dbfs: float = -40.0):
        self.tone_hz = float(tone_hz)
        self.amplitude = float(amplitude)
        self.user_speech_dbfs = float(user_speech_dbfs)

    def reset(self) -> None:
        return None

    def _tone(self, tick: int, seconds: float = FRAME_SECONDS) -> np.ndarray:
        samples = int(round(seconds * AGENT_SAMPLE_RATE))
        t = np.arange(samples, dtype=np.float64) / AGENT_SAMPLE_RATE + tick * FRAME_SECONDS
        envelope = np.ones(samples, dtype=np.float64)
        fade = min(240, samples // 4)
        if fade > 0:
            envelope[:fade] = np.linspace(0.0, 1.0, fade)
            envelope[-fade:] = np.linspace(1.0, 0.0, fade)
        return (self.amplitude * np.sin(2.0 * np.pi * self.tone_hz * t) * envelope).astype(np.float32)

    def step(self, user_audio_16k: np.ndarray, tick_index: int) -> EngineOutput:
        user_active = _rms_dbfs(user_audio_16k) > self.user_speech_dbfs
        if 10 <= tick_index < 25:
            return EngineOutput("SPEAK", self._tone(tick_index), text="hello")
        if tick_index == 25:
            return EngineOutput("STOP")
        if tick_index == 26:
            # A short backchannel clip, several frames long.
            return EngineOutput("BACKCHANNEL", self._tone(tick_index, seconds=0.4), text="mm-hm")
        if 60 <= tick_index < 80:
            if user_active and tick_index > 60:
                return EngineOutput("STOP")
            return EngineOutput("SPEAK", self._tone(tick_index))
        return EngineOutput("PAD")


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------


class _PlayedReference:
    """Rolling last-500 ms 16 kHz reference buffer."""

    def __init__(self, samples: int = REFERENCE_SAMPLES):
        self._buffer = np.zeros(samples, dtype=np.float32)

    @property
    def value(self) -> np.ndarray:
        return self._buffer

    def update(self, played_24k: np.ndarray) -> None:
        converted = _resample_24k_to_16k(played_24k)
        if converted.size == 0:
            return
        if converted.size >= self._buffer.size:
            self._buffer[:] = converted[-self._buffer.size :]
            return
        shift = converted.size
        self._buffer[:-shift] = self._buffer[shift:]
        self._buffer[-shift:] = converted


def run_session(
    source: AudioSource,
    sink: AudioSink,
    engine: DuplexEngine,
    ticks: int,
    metrics_path: str | Path | None = None,
    *,
    aec=None,
    realtime: bool = False,
    clock: TickClock | None = None,
    user_speech_dbfs: float = -40.0,
    jitter_target_ms: float = 30.0,
    jitter_max_ms: float = 1000.0,
    stop_on_eof: bool = True,
    barge_in_guard: bool = True,
    barge_in_guard_frames: int = 2,
) -> dict:
    """Run ``ticks`` fixed 80 ms steps and return (optionally write) metrics.

    This function is the sole writer: it calls ``sink.write`` once per tick and
    no engine receives the sink.  ``aec`` is any object with
    ``process(mic, reference)``; the reference is the last 500 ms of played
    audio resampled to 16 kHz.
    """
    if ticks < 0:
        raise ValueError("ticks must be non-negative")
    if hasattr(engine, "reset"):
        engine.reset()
    clock = clock or TickClock(realtime=realtime)
    jitter = JitterBuffer(target_fill_ms=jitter_target_ms, max_fill_ms=jitter_max_ms)
    reference = _PlayedReference() if aec is not None else None

    controls = {name: 0 for name in CONTROL_LABELS}
    ttfa_ms: list[float] = []
    barge_latencies: list[float] = []
    guard_drops: list[int] = []
    self_interruptions: list[int] = []
    self_interruption_frames: list[int] = []
    rss_samples: list[float] = []
    rss_times: list[float] = []

    actual_ticks = 0
    engine_seconds = 0.0
    stop_events = 0
    source_exhausted = False
    prev_control: str | None = None
    user_was_active = False
    user_onset_tick: int | None = None
    barge_onset_tick: int | None = None
    guard_latched = False
    silent_frames = 0
    wall_start = time.monotonic()

    try:
        for tick in range(ticks):
            clock.wait()
            item = source.read()
            if item is None:
                source_exhausted = True
                if stop_on_eof:
                    break
                mic = np.zeros(MIC_FRAME_SAMPLES, dtype=np.float32)
            else:
                mic = _fit_mic_frame(item)

            clean = aec.process(mic, reference.value) if aec is not None else mic
            active = _rms_dbfs(clean) > user_speech_dbfs
            if active and not user_was_active:
                user_onset_tick = tick
            if active:
                silent_frames = 0
            else:
                silent_frames += 1

            engine_start = time.perf_counter()
            output = engine.step(clean, tick)
            engine_seconds += time.perf_counter() - engine_start
            if not isinstance(output, EngineOutput):
                try:
                    output = EngineOutput(
                        output.control, getattr(output, "audio_24k", None), getattr(output, "text", "")
                    )
                except AttributeError as exc:
                    raise TypeError(
                        f"engine.step must return EngineOutput, got {type(output)!r}"
                    ) from exc
            control = output.control
            controls[control] += 1

            playing_before = jitter.depth_samples > 0 or prev_control in ("SPEAK", "BACKCHANNEL")
            if active and playing_before and barge_onset_tick is None and not guard_latched:
                barge_onset_tick = tick
            if control == "SPEAK" and active:
                self_interruption_frames.append(tick)
                if prev_control != "SPEAK":
                    self_interruptions.append(tick)

            if output.audio_24k.size and control in ("SPEAK", "BACKCHANNEL"):
                if user_onset_tick is not None:
                    ttfa_ms.append((tick - user_onset_tick) * FRAME_SECONDS * 1000.0)
                    user_onset_tick = None

            skip_push = False
            if active and control == "SPEAK" and barge_in_guard:
                if guard_latched:
                    # Keep the agent silent while the user barges in.
                    jitter.drop(keep_samples=OUT_FRAME_SAMPLES)
                    skip_push = True
                elif (
                    barge_onset_tick is not None
                    and tick - barge_onset_tick >= barge_in_guard_frames
                ):
                    # The local guard bounds barge-in latency even if the model
                    # fails to emit STOP.  The metric records the guard.
                    jitter.drop(keep_samples=OUT_FRAME_SAMPLES)
                    barge_latencies.append(float(tick - barge_onset_tick))
                    guard_drops.append(tick)
                    barge_onset_tick = None
                    guard_latched = True
                    skip_push = True
            elif not active or control != "SPEAK":
                guard_latched = False

            if control == "STOP":
                stop_events += 1
                # Drop everything except the newest 80 ms tail.
                jitter.drop(keep_samples=OUT_FRAME_SAMPLES)
                if barge_onset_tick is not None:
                    barge_latencies.append(float(tick - barge_onset_tick))
                    barge_onset_tick = None
                guard_latched = False
            elif output.audio_24k.size and not skip_push:
                clip = output.audio_24k
                if control == "BACKCHANNEL":
                    limit = int(round(BACKCHANNEL_MAX_SECONDS * AGENT_SAMPLE_RATE))
                    clip = clip[:limit]
                jitter.push(clip)

            # ----- the single writer -----
            written = jitter.pop(OUT_FRAME_SAMPLES)
            sink.write(written)

            if reference is not None:
                reference.update(written)

            actual_ticks += 1
            elapsed = time.monotonic() - wall_start
            rss = _rss_mb()
            if rss is not None:
                rss_samples.append(rss)
                rss_times.append(elapsed)

            prev_control = control
            user_was_active = active
            if not active and silent_frames >= 3:
                user_onset_tick = None
                barge_onset_tick = None
    finally:
        if hasattr(engine, "end_utterance"):
            try:
                engine.end_utterance()
            except Exception:  # noqa: BLE001 - cleanup must not hide the session result
                logger.exception("engine.end_utterance failed")

    wall_seconds = max(time.monotonic() - wall_start, 1e-9)
    audio_seconds = actual_ticks * FRAME_SECONDS
    metrics = {
        "ticks": actual_ticks,
        "requested_ticks": ticks,
        "audio_seconds": audio_seconds,
        "wall_seconds": wall_seconds,
        "rtf": (engine_seconds / audio_seconds) if audio_seconds else None,
        "rtf_engine": (engine_seconds / audio_seconds) if audio_seconds else None,
        "rtf_wall": wall_seconds / audio_seconds if audio_seconds else None,
        "jitter_ms": clock.stats(),
        "ttfa_ms": _percentiles(ttfa_ms),
        "controls": controls,
        "jitter_buffer": jitter.snapshot(),
        "underruns": jitter.underruns,
        "overruns": jitter.overruns,
        "self_interruptions": {
            "count": len(self_interruptions),
            "ticks": self_interruptions,
            "frames": len(self_interruption_frames),
            "frame_ticks": self_interruption_frames,
        },
        "barge_in": {
            "stop_events": stop_events,
            "latency_frames": barge_latencies,
            "stats": _percentiles(barge_latencies),
            "guard_drops": len(guard_drops),
            "guard_ticks": guard_drops,
        },
        "rss": {
            "start_mb": rss_samples[0] if rss_samples else None,
            "end_mb": rss_samples[-1] if rss_samples else None,
            "slope_mb_per_min": _rss_slope(rss_samples, rss_times),
        },
        "source_exhausted": source_exhausted,
        "gate7_claimed": False,
        "notes": [
            "Scheduler/AEC evidence only; Gate 7 requires a real 10-minute live run.",
            "TTFA is quantised to 80 ms ticks and measured from RMS speech onset.",
            "The jitter-buffer reserve is quantised by 80 ms decoded frames.",
        ],
    }
    if metrics_path is not None:
        path = Path(metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# Real LFM3 engine
# ---------------------------------------------------------------------------


class Lfm3DuplexEngine:
    """Duplex engine over LiquidAI/LFM3-Omni-2B (or a fine-tuned checkpoint).

    The constructor loads the checkpoint; importing this module never does.
    The control vocabulary is resolved from the tokenizer: the checkpoint must
    carry ``<|listen|>``, ``<|speak|>``, ``<|backchannel|>`` and ``<|stop|>``
    (plus a pad token).  A base checkpoint that has not gone through the token
    surgery fails with a clear error.

    Text output is sampled from the same hidden state as the control decision
    and returned for display.  The blueprint's serving loop feeds only the
    control token and the agent audio frame back into the LM, which is what
    this implementation does.
    """

    def __init__(
        self,
        model_id: str = "LiquidAI/LFM3-Omni-2B",
        *,
        int4: bool = False,
        device: str | None = None,
        dtype=None,
        temperature: float = 0.0,
        top_k: int | None = None,
        end_silence_frames: int = 6,
        user_speech_dbfs: float = -50.0,
        max_audio_history: int = 25,
    ) -> None:
        self.model_id = model_id
        self.int4 = bool(int4)
        self.requested_device = device
        self.requested_dtype = dtype
        self.temperature = float(temperature)
        self.top_k = top_k
        self.end_silence_frames = int(end_silence_frames)
        self.user_speech_dbfs = float(user_speech_dbfs)
        self.max_audio_history = int(max_audio_history)
        self._loaded = False
        self._state_ready = False
        self._torch = None
        self.model = None
        self.processor = None
        self._device = None
        self._lm_device = None
        self._encoder_device = None
        self._depth_device = None
        self._codebook_device = None
        self._codec_device = None
        self._silence_codes = None
        self._control_ids: list[int] = []
        self._control_id_tensor = None
        self._bos_id = 0
        self._lm_cache = None
        self._attention_mask = None
        self._last_hidden = None
        self._feature_cache = None
        self._encoder_cache = None
        self._codec_cache = None
        self._audio_history = None
        self._silent_frames = 0
        self._finalized = True
        self._had_speech = False

    # -- loading ----------------------------------------------------------
    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self._torch = torch
        load_kwargs: dict = {"trust_remote_code": True}
        if self.int4:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("--int4 requires the optional bitsandbytes package") from exc
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = {"": self.requested_device or ("cuda" if torch.cuda.is_available() else "cpu")}
        else:
            load_kwargs["dtype"] = self.requested_dtype or (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            load_kwargs["device_map"] = {"": self.requested_device} if self.requested_device else "auto"

        try:
            model = AutoModelForMultimodalLM.from_pretrained(self.model_id, **load_kwargs)
        except (ImportError, ModuleNotFoundError) as exc:
            # ``device_map="auto"`` needs accelerate; without it, load on CPU
            # (or the explicitly requested device) and move the model.
            if "accelerate" not in str(exc).lower() or "device_map" not in str(load_kwargs):
                raise
            load_kwargs.pop("device_map")
            model = AutoModelForMultimodalLM.from_pretrained(self.model_id, **load_kwargs)
            device = self.requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
        except TypeError:
            # transformers 5 renamed torch_dtype -> dtype; support both.
            if "dtype" not in load_kwargs:
                raise
            load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
            model = AutoModelForMultimodalLM.from_pretrained(self.model_id, **load_kwargs)

        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        model.eval()
        self.model = model
        self.processor = processor
        self._device = next(model.parameters()).device
        self._lm_device = next(model.model.language_model.parameters()).device
        self._encoder_device = next(model.model.audio_encoder.parameters()).device
        self._depth_device = next(model.model.depth_decoder.parameters()).device
        self._codebook_device = next(model.model.codebook_embedding.parameters()).device
        tokenizer = processor.tokenizer
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = getattr(model.config.text_config, "pad_token_id", None)
        if pad_id is None:
            raise RuntimeError("checkpoint tokenizer has no pad token; cannot resolve the PAD control label")
        unk = tokenizer.unk_token_id
        control_ids: list[int] = []
        for label in CONTROL_LABELS:
            if label == "PAD":
                control_ids.append(int(pad_id))
                continue
            token = CONTROL_LABEL_TO_TOKEN_NAME[label]
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == unk:
                raise RuntimeError(
                    f"checkpoint {self.model_id!r} has no {token} token; "
                    "serving requires a duplex checkpoint with the four control tokens"
                )
            control_ids.append(int(token_id))
        self._control_ids = control_ids
        self._control_id_tensor = torch.tensor(control_ids, dtype=torch.long, device=self._lm_device)
        bos = tokenizer.bos_token_id
        if bos is None:
            bos = getattr(model.config.text_config, "bos_token_id", None) or pad_id
        self._bos_id = int(bos)
        if processor.audio_tokenizer is None:
            raise RuntimeError("checkpoint processor did not load the LiquidCodec audio tokenizer")
        # The codec is small; keep it on the model device when one is used.
        processor.audio_tokenizer.eval()
        try:
            processor.audio_tokenizer.to(self._device)
        except (RuntimeError, ValueError):
            logger.warning("could not move LiquidCodec to %s; decoding on its current device", self._device)
        self._codec_device = next(processor.audio_tokenizer.parameters()).device
        # Every training frame contains an agent-audio slot, including PAD and
        # LISTEN.  Cache the codec representation of 80 ms of silence so the
        # inference sequence keeps the same [user, control, agent-audio] shape.
        with torch.inference_mode():
            silence = torch.zeros(
                (1, int(getattr(processor.audio_tokenizer.config, "frame_samples", 1920))),
                dtype=torch.float32,
                device=self._codec_device,
            )
            self._silence_codes = processor.audio_tokenizer.encode(silence).audio_codes[:, :, 0].contiguous()
        self._loaded = True

    # -- state ------------------------------------------------------------
    def reset(self) -> None:
        """Start a new conversation: fresh KV/encoder/codec caches and bos prefill."""
        self._load()
        torch = self._torch
        from transformers.cache_utils import DynamicCache

        self._lm_cache = DynamicCache(config=self.model.config.text_config)
        self._attention_mask = torch.zeros((1, 0), dtype=torch.long, device=self._lm_device)
        self._last_hidden = None
        self._feature_cache = None
        self._encoder_cache = None
        self._codec_cache = None
        self._audio_history = None
        self._silent_frames = 0
        self._finalized = True
        self._had_speech = False
        bos = torch.tensor([[self._bos_id]], dtype=torch.long, device=self._lm_device)
        with torch.inference_mode():
            embeds = self.model.model.get_input_embeddings()(bos)
            self._last_hidden = self._append_embeds(embeds)
        self._state_ready = True

    def _ensure_state(self) -> None:
        self._load()
        if not self._state_ready:
            self.reset()

    def _append_embeds(self, embeds):
        """Append embeddings to the LM cache, extending the attention mask."""
        torch = self._torch
        n = int(embeds.shape[1])
        past = int(self._lm_cache.get_seq_length())
        new_mask = torch.cat(
            (self._attention_mask, torch.ones((1, n), dtype=torch.long, device=self._lm_device)), dim=1
        )
        position_ids = torch.arange(past, past + n, dtype=torch.long, device=self._lm_device)[None]
        out = self.model.model.language_model(
            inputs_embeds=embeds.to(self._lm_device),
            attention_mask=new_mask,
            position_ids=position_ids,
            past_key_values=self._lm_cache,
            use_cache=True,
        )
        self._lm_cache = out.past_key_values
        self._attention_mask = new_mask
        self._last_hidden = out.last_hidden_state[:, -1:]
        return out.last_hidden_state

    def _new_codec_cache(self):
        codec = self.processor.audio_tokenizer
        try:  # transformers >= 5 may export it from the top level
            from transformers import LiquidCodecStreamingCache

            return LiquidCodecStreamingCache(codec.config)
        except (ImportError, AttributeError):
            pass
        try:
            import importlib

            module = importlib.import_module(type(codec).__module__)
            cls = getattr(module, "LiquidCodecStreamingCache", None)
            if cls is not None:
                return cls(codec.config)
        except (ImportError, AttributeError, TypeError):
            pass
        logger.warning("LiquidCodecStreamingCache unavailable; decoding frames independently")
        return None

    def _finalize_audio_stream(self) -> None:
        codec = self.processor.audio_tokenizer
        if self._codec_cache is None or getattr(self._codec_cache, "finished", True):
            self._codec_cache = None
            self._audio_history = None
            return
        torch = self._torch
        try:
            empty = torch.zeros(
                (1, int(self.model.config.num_codebooks), 0), dtype=torch.long, device=self._codec_device
            )
            codec.decode(empty, self._codec_cache, finalize=True)
        except (RuntimeError, ValueError):
            logger.exception("LiquidCodec finalize failed")
        self._codec_cache = None
        self._audio_history = None

    def _finalize_user_stream(self) -> None:
        if self._feature_cache is None:
            self._finalized = True
            return
        try:
            features, _ = self.processor.feature_extractor.stream(
                np.zeros(0, dtype=np.float32),
                self._feature_cache,
                is_final=True,
                sampling_rate=USER_SAMPLE_RATE,
            )
            out = self.model.prefill_audio(
                features["input_features"].to(self._encoder_device),
                past_key_values=self._lm_cache,
                encoder_cache=self._encoder_cache,
                attention_mask=self._attention_mask,
                is_final=True,
            )
            self._lm_cache = out.past_key_values
            self._encoder_cache = out.encoder_cache
            self._attention_mask = out.attention_mask
            if out.last_hidden_state is not None:
                self._last_hidden = out.last_hidden_state[:, -1:]
        except (RuntimeError, ValueError):
            logger.exception("streaming encoder finalize failed")
        self._feature_cache = None
        self._encoder_cache = None
        self._finalized = True
        self._had_speech = False

    def end_utterance(self) -> None:
        """Flush user and agent streams at session end."""
        if not self._state_ready:
            return
        with self._torch.inference_mode():
            self._finalize_user_stream()
            self._finalize_audio_stream()

    # -- step -------------------------------------------------------------
    def _control_label(self, hidden) -> str:
        torch = self._torch
        head_dtype = self.model.lm_head.weight.dtype
        logits = self.model.lm_head(hidden.to(head_dtype))[0, -1].float()
        index = int(torch.argmax(logits[self._control_id_tensor]).item())
        return CONTROL_LABELS[index]

    def _sample_text_token(self, hidden) -> int | None:
        """Pick one text token after the control token, mirroring training order."""
        torch = self._torch
        head_dtype = self.model.lm_head.weight.dtype
        logits = self.model.lm_head(hidden.to(head_dtype))[0, -1].float().clone()
        logits[self._control_id_tensor] = -float("inf")
        for special in self.processor.tokenizer.all_special_ids:
            if 0 <= int(special) < logits.numel():
                logits[int(special)] = -float("inf")
        token = int(torch.argmax(logits).item())
        return token if token not in self._control_ids else None

    def step(self, user_audio_16k: np.ndarray, tick_index: int) -> EngineOutput:
        self._ensure_state()
        # The depth decoder and the LM are both training-mode-capable; serving
        # must not build an autograd graph across frames.
        with self._torch.inference_mode():
            return self._step_impl(user_audio_16k, tick_index)

    def _step_impl(self, user_audio_16k: np.ndarray, tick_index: int) -> EngineOutput:
        self._ensure_state()
        torch = self._torch
        audio = _fit_mic_frame(user_audio_16k)
        level = _rms_dbfs(audio)
        speech = level > self.user_speech_dbfs

        if speech:
            if self._finalized:
                # A new user utterance starts a new encoder stream.
                self._feature_cache = None
                self._encoder_cache = None
                self._finalized = False
            self._had_speech = True
            self._silent_frames = 0
        else:
            self._silent_frames += 1
            if (
                self._had_speech
                and not self._finalized
                and self._silent_frames >= self.end_silence_frames
            ):
                self._finalize_user_stream()

        if not self._finalized:
            features, self._feature_cache = self.processor.feature_extractor.stream(
                audio, cache=self._feature_cache, sampling_rate=USER_SAMPLE_RATE
            )
            out = self.model.prefill_audio(
                features["input_features"].to(self._encoder_device),
                past_key_values=self._lm_cache,
                encoder_cache=self._encoder_cache,
                attention_mask=self._attention_mask,
                is_final=False,
            )
            self._lm_cache = out.past_key_values
            self._encoder_cache = out.encoder_cache
            self._attention_mask = out.attention_mask
            if out.last_hidden_state is not None and out.last_hidden_state.shape[1]:
                self._last_hidden = out.last_hidden_state[:, -1:]

        if self._last_hidden is None:
            raise RuntimeError("engine state has no hidden state; reset() must prefill bos")

        label = self._control_label(self._last_hidden)
        text = ""
        control_id = torch.tensor(
            [[self._control_ids[CONTROL_LABELS.index(label)]]], dtype=torch.long, device=self._lm_device
        )
        control_embeds = self.model.model.get_input_embeddings()(control_id)
        audio_hidden = self._append_embeds(control_embeds)

        # Training order is [user, control, optional text, agent audio].
        if label in ("SPEAK", "BACKCHANNEL"):
            token_id = self._sample_text_token(audio_hidden)
            if token_id is not None:
                token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=self._lm_device)
                token_embed = self.model.model.get_input_embeddings()(token_tensor)
                audio_hidden = self._append_embeds(token_embed)
                text = self.processor.tokenizer.decode([token_id], skip_special_tokens=True).strip()

        agent_audio = np.zeros(0, dtype=np.float32)
        speaking = label in ("SPEAK", "BACKCHANNEL")
        if speaking:
            codes = self.model.sample_audio_frame(
                audio_hidden[0, -1].to(self._depth_device),
                temperature=self.temperature,
                top_k=self.top_k,
                history=self._audio_history,
            )
            eos_code = int(getattr(self.model.config, "audio_eos_code", 2016))
            if bool((codes == eos_code).all()):
                speaking = False
                self._finalize_audio_stream()
            else:
                codes_tensor = codes[None]
                if self._audio_history is None:
                    self._audio_history = codes_tensor
                else:
                    self._audio_history = torch.cat(
                        (self._audio_history, codes_tensor), dim=0
                    )[-self.max_audio_history :]
                audio_embeds = self.model.model.get_audio_out_embeds(
                    codes_tensor.to(self._codebook_device)
                )
                self._append_embeds(audio_embeds[None])

                codec = self.processor.audio_tokenizer
                if self._codec_cache is None:
                    self._codec_cache = self._new_codec_cache()
                decode_codes = codes_tensor[:, :, None].to(self._codec_device)
                with torch.inference_mode():
                    decoded = codec.decode(decode_codes, self._codec_cache, finalize=False).audio_values
                agent_audio = decoded[0].detach().float().cpu().numpy().astype(np.float32)
        if not speaking:
            silence = self._silence_codes.to(self._codebook_device)
            self._append_embeds(self.model.model.get_audio_out_embeds(silence)[None])

        if label == "STOP":
            self._finalize_audio_stream()

        return EngineOutput(label, agent_audio, text)


__all__ = [
    "AudioSink",
    "AudioSource",
    "DuplexEngine",
    "EngineOutput",
    "FakeDuplexEngine",
    "JitterBuffer",
    "Lfm3DuplexEngine",
    "MIC_FRAME_SAMPLES",
    "OUT_FRAME_SAMPLES",
    "SoundDeviceSink",
    "SoundDeviceSource",
    "TickClock",
    "WavFileSink",
    "WavFileSource",
    "run_session",
]
