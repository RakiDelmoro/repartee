"""Audio utility tests (no model, no network)."""
from __future__ import annotations

import numpy as np

from src.audio import clipping_ratio, fit_frames, peak_dbfs, read_audio, rms_dbfs, write_audio
from src.schema import FRAME_SAMPLES


def test_write_read_roundtrip(tmp_path) -> None:
    rate = 16000
    t = np.arange(rate, dtype=np.float32) / rate
    audio = 0.25 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "tone.wav"
    write_audio(path, audio, rate)
    loaded = read_audio(path, rate)
    assert loaded.shape == audio.shape
    assert np.max(np.abs(loaded - audio)) < 1e-3


def test_resample_changes_length() -> None:
    from src.audio import resample

    audio = np.zeros(24000, dtype=np.float32)
    assert resample(audio, 24000, 16000).shape == (16000,)


def test_fit_frames_exact_length() -> None:
    audio = np.ones(1000, dtype=np.float32)
    fitted = fit_frames(audio, 2, 16000)
    assert fitted.shape == (2 * FRAME_SAMPLES[16000],)
    assert np.all(fitted[:1000] == 1.0) and np.all(fitted[1000:] == 0.0)
    assert fit_frames(np.ones(5000, dtype=np.float32), 2, 16000).shape == (2560,)


def test_level_helpers() -> None:
    assert clipping_ratio(np.array([1.0, -1.0, 0.0], dtype=np.float32)) == 2 / 3
    assert rms_dbfs(np.zeros(10, dtype=np.float32)) == -240.0
    assert peak_dbfs(np.zeros(10, dtype=np.float32)) == -240.0
    assert -6.1 < rms_dbfs(np.full(10, 0.5, dtype=np.float32)) < -5.9
