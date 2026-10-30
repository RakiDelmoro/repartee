"""Synthetic-echo tests for the AEC backends (scheduler/AEC evidence only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.aec import FRAME_SAMPLES, NLMSAEC, create_echo_canceller  # noqa: E402

FS = 16_000


def _delayed_echo(far: np.ndarray, gain: float, delay: int) -> np.ndarray:
    echo = np.zeros_like(far)
    if delay <= 0:
        echo[:] = gain * far
    elif delay < far.size:
        echo[delay:] = gain * far[:-delay]
    return echo


def _run_frames(aec, mic: np.ndarray, far: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mic)
    for start in range(0, mic.size - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        end = start + FRAME_SAMPLES
        # The loop passes the last 500 ms played, ending at this frame.
        reference = far[max(0, end - int(0.5 * FS)) : end]
        out[start:end] = aec.process(mic[start:end], reference)
    return out


def _erle_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref_power = float(np.mean(np.asarray(reference, dtype=np.float64) ** 2))
    est_power = float(np.mean(np.asarray(estimate, dtype=np.float64) ** 2)) + 1e-15
    return 10.0 * np.log10(max(ref_power, 1e-15) / est_power)


def test_nlms_erle_improves_by_10db_on_synthetic_echo():
    rng = np.random.default_rng(0)
    seconds = 10.0
    n = int(seconds * FS)
    far = (0.2 * rng.standard_normal(n)).astype(np.float32)
    gain = 0.6
    delay = int(0.25 * FS)
    mic = _delayed_echo(far, gain, delay)

    aec = NLMSAEC()
    out = _run_frames(aec, mic, far)

    warmup = int(3.0 * FS)
    measured = slice(warmup, n)
    raw_erle = _erle_db(mic[measured], mic[measured])  # 0 dB by definition
    processed_erle = _erle_db(mic[measured], out[measured])
    improvement = processed_erle - raw_erle
    assert improvement >= 10.0, f"ERLE improvement {improvement:.2f} dB < 10 dB"
    assert abs(aec.delay_samples - delay) <= 64, aec.delay_samples


def test_nlms_retains_near_end_signal():
    rng = np.random.default_rng(1)
    seconds = 10.0
    n = int(seconds * FS)
    t = np.arange(n) / FS
    # A tonal near-end stands in for speech and is uncorrelated with the
    # white-noise far-end.
    near = (
        0.06 * np.sin(2 * np.pi * 220.0 * t)
        + 0.04 * np.sin(2 * np.pi * 440.0 * t)
        + 0.02 * np.sin(2 * np.pi * 660.0 * t)
    ).astype(np.float32)
    far = (0.2 * rng.standard_normal(n)).astype(np.float32)
    mic = near + _delayed_echo(far, 0.6, int(0.25 * FS))

    aec = NLMSAEC()
    out = _run_frames(aec, mic, far)
    warmup = int(3.0 * FS)
    corr = float(np.corrcoef(out[warmup:], near[warmup:])[0, 1])
    assert corr >= 0.8, f"near-end correlation {corr:.3f} < 0.8"


def test_process_shape_dtype_and_reset():
    aec = NLMSAEC()
    mic = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    out = aec.process(mic, np.zeros(8000, dtype=np.float32))
    assert out.shape == (FRAME_SAMPLES,)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    aec.reset()
    assert aec.delay_samples == 0


def test_factory_nlms_backend():
    aec = create_echo_canceller(backend="nlms")
    assert isinstance(aec, NLMSAEC)


def test_factory_auto_falls_back_without_optional_binding():
    # If the optional binding is installed this still returns a valid AEC; if
    # not, it must not raise.
    aec = create_echo_canceller(backend="auto")
    out = aec.process(np.zeros(FRAME_SAMPLES, dtype=np.float32), np.zeros(8000, dtype=np.float32))
    assert out.shape == (FRAME_SAMPLES,)


def test_webrtc_backend_only_when_importable():
    try:
        import webrtc_audio_processing  # noqa: F401
    except ImportError:
        return
    from serve.aec import WebRtcAec3

    aec = WebRtcAec3()
    out = aec.process(np.zeros(FRAME_SAMPLES, dtype=np.float32), np.zeros(8000, dtype=np.float32))
    assert out.shape == (FRAME_SAMPLES,)
