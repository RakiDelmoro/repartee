"""Acoustic echo cancellation for the Repartee 80 ms serving loop.

The loop feeds one 16 kHz / 80 ms microphone frame (1280 samples) to
:meth:`AcousticEchoCanceller.process` together with the last 500 ms of audio
that was written to the loudspeaker.  The reference must be 16 kHz mono: the
serving loop resamples its 24 kHz agent output for this purpose.  The return
value is the echo-suppressed microphone frame (same shape, float32).

Backends
--------
``WebRtcAec3``
    Preferred.  Used when the optional ``webrtc_audio_processing`` Python
    binding of the WebRTC audio processing module is importable.  That
    binding only accepts 10 ms int16 blocks, so an 80 ms frame is processed as
    eight blocks, with the matching reference block fed first.  The binding
    exposes WebRTC's echo canceller (``aec_type=2``); the module is built from
    WebRTC's AEC3-capable audio processing tree, but the binding does not let
    us force the AEC3 algorithm explicitly.  This class is therefore named
    after the intended backend and not a claim about the internal algorithm.
    It is imported lazily, so importing :mod:`serve` never requires it.

``NLMSAEC``
    Deterministic fallback.  A single-channel block-normalised LMS filter with
    a coarse-to-fine delay search over 0--500 ms.  It models a linear echo path
    only: non-linear loudspeaker echo, rapidly moving paths, and heavy
    double-talk are handled poorly.  It is intentionally boring and testable,
    not a replacement for AEC3 in production.

The synthetic-echo test (``tests/test_aec.py``) checks >=10 dB ERLE
improvement on a delayed-gain echo and retention of the near-end signal.  The
serving tests and these numbers are scheduler/AEC evidence only; they are not
Gate 7 results.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1_280  # 80 ms
REFERENCE_SECONDS = 0.5


@runtime_checkable
class AcousticEchoCanceller(Protocol):
    """One 80 ms / 16 kHz frame in, one echo-suppressed frame out."""

    def process(self, mic: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Cancel the echo of ``reference`` from ``mic``.

        Args:
            mic: Microphone frame, ``float32`` of shape ``(1280,)`` at 16 kHz.
            reference: Played audio ending at the current frame, ``float32`` at
                16 kHz, up to 500 ms (8000 samples).  Implementations use the
                newest samples first and tolerate a shorter buffer.

        Returns:
            ``float32`` array of shape ``(1280,)``.
        """
        ...


# ---------------------------------------------------------------------------
# WebRTC backend
# ---------------------------------------------------------------------------


class WebRtcAec3:
    """Thin wrapper around the optional ``webrtc_audio_processing`` binding.

    The binding's API is fixed by ``audio_processing_module.h``::

        AP(aec_type, enable_ns, agc_type, enable_vad)
        AP.set_stream_format(rate, channels)
        AP.set_reverse_stream_format(rate, channels)
        AP.set_aec_level(level)          # 0..2
        AP.set_system_delay(ms)
        AP.process_reverse_stream(int16 bytes)
        AP.process_stream(int16 bytes) -> int16 bytes

    Only 10 ms blocks are supported, so this class splits the 80 ms frame.
    """

    name = "webrtc-aec3"

    def __init__(self, sample_rate: int = SAMPLE_RATE, frame_samples: int = FRAME_SAMPLES) -> None:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"WebRTC backend expects 16 kHz, got {sample_rate}")
        if frame_samples % (sample_rate // 100):
            raise ValueError("frame_samples must be a multiple of 10 ms")
        from webrtc_audio_processing import AudioProcessingModule  # lazy, optional

        # aec_type=2 enables the echo canceller in the binding; keyword names
        # differ between SWIG builds, so use the documented positional form.
        self._ap = AudioProcessingModule(2, False, 0, False)
        self._ap.set_stream_format(sample_rate, 1)
        self._ap.set_reverse_stream_format(sample_rate, 1)
        self._ap.set_aec_level(2)
        self._ap.set_system_delay(0)
        self._frame_samples = frame_samples
        self._block = sample_rate // 100  # 10 ms

    def process(self, mic: np.ndarray, reference: np.ndarray) -> np.ndarray:
        mic = np.asarray(mic, dtype=np.float32).reshape(-1)
        if mic.size != self._frame_samples:
            raise ValueError(f"mic must have {self._frame_samples} samples, got {mic.size}")
        reference = np.asarray(reference, dtype=np.float32).reshape(-1)
        if reference.size < mic.size:
            reference = np.pad(reference, (mic.size - reference.size, 0))
        reference = reference[-mic.size :]
        return _webrtc_process(self._ap, mic, reference, self._block)


def _call_binding(function, payload: bytes):
    """Call a SWIG ``std::string`` method with bytes or a latin-1 string.

    Older SWIG builds expose ``std::string`` as ``str`` (the binding's README
    passes a Python string); newer ones accept bytes.  A failed argument
    conversion happens before the C++ call, so retrying is safe.
    """
    try:
        return function(payload)
    except TypeError:
        return function(payload.decode("latin-1"))


def _webrtc_process(ap, mic: np.ndarray, reference: np.ndarray, block: int) -> np.ndarray:
    """Process int16 blocks with the binding; separated for unit testing."""
    out = np.empty_like(mic)
    for start in range(0, mic.size, block):
        stop = start + block
        _call_binding(ap.process_reverse_stream, _to_bytes(reference[start:stop]))
        data = _call_binding(ap.process_stream, _to_bytes(mic[start:stop]))
        out[start:stop] = _from_bytes(data)
    return out


def _to_bytes(audio: np.ndarray) -> bytes:
    block = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (block * 32768.0).astype("<i2").tobytes()


def _from_bytes(data) -> np.ndarray:
    if isinstance(data, str):
        data = data.encode("latin-1")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# NLMS fallback
# ---------------------------------------------------------------------------


class NLMSAEC:
    """Block-normalised LMS echo canceller with a 0--500 ms delay search.

    The far-end reference is kept in a rolling history.  For every frame the
    delay with the highest normalised cross-correlation between the delayed
    reference and the microphone is selected on a coarse grid, then refined to
    one sample.  A 256-tap (16 ms) FIR filter at that delay models the linear
    echo path and is adapted with a block-gradient NLMS update.

    Limitations (documented rather than hidden):

    * linear echo path only; no non-linear residual echo suppression;
    * delay is re-estimated once per 80 ms frame; path changes inside a frame
      are not tracked;
    * the delay search is limited by the available reference history (roughly
      0--400 ms on the first frame, up to 500 ms once the history is warm);
    * double-talk is handled by the normalised gradient only, there is no
      explicit double-talk detector, so strong near-end speech slows
      convergence;
    * the 16 kHz reference is assumed to be aligned with the end of the
      microphone frame.
    """

    name = "nlms"

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_samples: int = FRAME_SAMPLES,
        *,
        max_delay_ms: float = 500.0,
        filter_taps: int = 256,
        step_size: float = 0.5,
        delay_stride_samples: int = 32,
        eps: float = 1e-8,
    ) -> None:
        if sample_rate <= 0 or frame_samples <= 0:
            raise ValueError("sample_rate and frame_samples must be positive")
        if filter_taps <= 0 or filter_taps > frame_samples:
            raise ValueError("filter_taps must be in (0, frame_samples]")
        if not 0.0 < step_size <= 1.0:
            raise ValueError("step_size must be in (0, 1]")
        if delay_stride_samples <= 0:
            raise ValueError("delay_stride_samples must be positive")
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.filter_taps = filter_taps
        self.step_size = step_size
        self.delay_stride_samples = delay_stride_samples
        self.eps = eps
        self.max_delay_samples = int(round(max_delay_ms * sample_rate / 1000.0))
        self._max_history = self.max_delay_samples + frame_samples + filter_taps + delay_stride_samples
        self.reset()

    def reset(self) -> None:
        """Forget the echo path, delay estimate and far-end history."""
        self._weights = np.zeros(self.filter_taps, dtype=np.float32)
        self._far = np.zeros(0, dtype=np.float32)
        self._delay = 0
        self._delay_confidence = 0.0
        self._frames = 0

    @property
    def delay_samples(self) -> int:
        """Current delay estimate in samples (useful for diagnostics/tests)."""
        return self._delay

    def _feasible_max_delay(self) -> int:
        # Enough far-end history for a whole frame plus the FIR tail.
        return max(0, min(self.max_delay_samples, self._far.size - self.frame_samples - self.filter_taps))

    @staticmethod
    def _similarity(mic: np.ndarray, delayed: np.ndarray) -> float:
        denom = float(np.sqrt(np.dot(mic, mic) * np.dot(delayed, delayed))) + 1e-12
        return float(np.dot(mic, delayed) / denom)

    def _estimate_delay(self, mic: np.ndarray) -> int:
        max_delay = self._feasible_max_delay()
        if max_delay <= 0:
            return 0
        end = self._far.size

        def score(delay: int) -> float:
            delayed = self._far[end - self.frame_samples - delay : end - delay]
            return self._similarity(mic, delayed)

        coarse = range(0, max_delay + 1, self.delay_stride_samples)
        best = max(coarse, key=score)
        lo = max(0, best - self.delay_stride_samples)
        hi = min(max_delay, best + self.delay_stride_samples)
        refined = max(range(lo, hi + 1), key=score)
        refined_score = score(refined)

        if refined_score < 0.10:
            # No trustworthy correlation (silence or near-end only): keep the
            # previous estimate rather than chasing noise.
            self._delay_confidence *= 0.9
            return self._delay
        # A one-frame outlier is allowed, but confidence is tracked so callers
        # can log an unstable delay.
        self._delay_confidence = 0.8 * self._delay_confidence + 0.2 * refined_score
        return refined

    def process(self, mic: np.ndarray, reference: np.ndarray) -> np.ndarray:
        mic = np.asarray(mic, dtype=np.float32).reshape(-1)
        if mic.size != self.frame_samples:
            raise ValueError(f"mic must have {self.frame_samples} samples, got {mic.size}")
        reference = np.asarray(reference, dtype=np.float32).reshape(-1)
        if reference.size == 0:
            reference = np.zeros(self.frame_samples, dtype=np.float32)

        if self._far.size == 0:
            # First call: the whole 500 ms reference is history.
            self._far = reference[-self._max_history :].astype(np.float32, copy=True)
        else:
            # Calls advance one 80 ms frame, so only the newest samples are new.
            self._far = np.concatenate((self._far, reference[-self.frame_samples :].astype(np.float32)))
            if self._far.size > self._max_history:
                self._far = self._far[-self._max_history :].copy()

        self._delay = self._estimate_delay(mic)

        # x_seg covers [frame_end - delay - taps + 1, frame_end - delay].
        end = self._far.size - self._delay
        need = self.frame_samples + self.filter_taps - 1
        if end < need:
            x_seg = np.pad(self._far[: max(end, 0)], (need - max(end, 0), 0))
        else:
            x_seg = self._far[end - need : end]

        echo = np.convolve(x_seg, self._weights, mode="valid")
        error = mic - echo

        # Block NLMS: gradient is the correlation of the error with the tap
        # delay line; the normaliser is the far-end frame energy.
        norm = float(np.dot(x_seg, x_seg)) + self.eps
        if norm > 1e-9:
            # np.correlate's first output corresponds to the *last* FIR tap;
            # reverse so index k is the gradient of w[k].
            gradient = np.correlate(x_seg, error, mode="valid")[::-1]
            self._weights += (self.step_size * gradient / norm).astype(np.float32)
            # Small leakage keeps the filter from drifting during silence.
            self._weights *= np.float32(0.9995)

        self._frames += 1
        return np.ascontiguousarray(error, dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_echo_canceller(
    *,
    sample_rate: int = SAMPLE_RATE,
    frame_samples: int = FRAME_SAMPLES,
    backend: str = "auto",
    max_delay_ms: float = 500.0,
) -> AcousticEchoCanceller:
    """Create the preferred available backend.

    ``backend`` is ``"auto"``, ``"webrtc"`` or ``"nlms"``.  ``"auto"`` falls
    back to :class:`NLMSAEC` when the optional binding is absent.  ``"webrtc"``
    raises the import error instead of silently degrading.
    """
    backend = backend.lower()
    if backend not in {"auto", "webrtc", "nlms"}:
        raise ValueError(f"unknown AEC backend {backend!r}")
    if backend in {"auto", "webrtc"}:
        try:
            return WebRtcAec3(sample_rate=sample_rate, frame_samples=frame_samples)
        except (ImportError, AttributeError, OSError, RuntimeError) as exc:
            if backend == "webrtc":
                raise
            logger.info("webrtc_audio_processing unavailable (%s); using NLMS fallback", exc)
    return NLMSAEC(sample_rate=sample_rate, frame_samples=frame_samples, max_delay_ms=max_delay_ms)


__all__ = [
    "AcousticEchoCanceller",
    "FRAME_SAMPLES",
    "NLMSAEC",
    "REFERENCE_SECONDS",
    "SAMPLE_RATE",
    "WebRtcAec3",
    "create_echo_canceller",
]
