"""Pure helpers for the bigger-device STT stage."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from app import stt


def _tone(seconds: float, value: float, sr: int = 16_000) -> np.ndarray:
    return np.full(int(seconds * sr), value, dtype=np.float32)


def test_plan_chunks_cuts_at_pause_before_hard_limit():
    sr = 16_000
    wav = np.concatenate([
        _tone(55, 0.05), _tone(2, 0.0),
        _tone(53, 0.05), _tone(2, 0.0),
        _tone(38, 0.05),
    ])

    chunks = stt.plan_chunks(wav, sr, max_seconds=60)

    assert [round(c.start_s, 1) for c in chunks] == [0.0, 56.0, 111.0]
    assert all(c.end_s - c.start_s <= 60.0 for c in chunks)
    assert chunks[-1].end_s == pytest.approx(len(wav) / sr)


def test_plan_chunks_hard_cuts_when_speech_never_pauses():
    sr = 16_000
    chunks = stt.plan_chunks(_tone(150, 0.05), sr, max_seconds=60)

    assert [(c.start_s, c.end_s) for c in chunks] == [
        (0.0, 60.0),
        (60.0, 120.0),
        (120.0, 150.0),
    ]


def test_plan_chunks_keeps_short_recording_in_one_piece():
    chunks = stt.plan_chunks(_tone(12, 0.05), 16_000, max_seconds=60)

    assert chunks == [stt.Chunk(0.0, 12.0)]


def test_plan_chunks_empty_audio_has_no_chunks():
    assert stt.plan_chunks(np.zeros(0, dtype=np.float32), 16_000, max_seconds=60) == []


def test_load_audio_mixes_to_mono_and_resamples(tmp_path):
    sr = 8_000
    stereo = np.stack([_tone(2, 0.25, sr), _tone(2, -0.25, sr)], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, sr, subtype="PCM_16")

    wav, out_sr = stt.load_audio(path)

    assert out_sr == 16_000
    assert wav.shape == (2 * 16_000,)
    assert np.max(np.abs(wav)) < 0.01


def test_clean_text_drops_empty_think_block():
    assert stt.clean_text("<think></think> Hello there. ") == "Hello there."
    assert stt.clean_text("<think>still open") == ""
    assert stt.clean_text("plain") == "plain"
