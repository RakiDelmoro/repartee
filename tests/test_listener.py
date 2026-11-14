"""Listener VAD: starts on speech, closes after the configured silence, spools atomically."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import soundfile as sf

from app.listener import (FRAME_SAMPLES, SAMPLE_RATE, Segmenter, VadConfig,
                          capture_file)


def speech(seconds: float, amp: float = 0.15) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    # 220 Hz plus harmonics: a voiced-looking frame well above the energy floor
    return (amp * (np.sin(2 * np.pi * 220 * t) + 0.4 * np.sin(2 * np.pi * 440 * t))
            ).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def feed(seg: Segmenter, audio: np.ndarray) -> list[np.ndarray]:
    out = []
    for i in range(0, len(audio) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        done = seg.feed(audio[i:i + FRAME_SAMPLES])
        if done is not None:
            out.append(done)
    return out


def test_starts_on_speech_and_closes_after_silence():
    cfg = VadConfig(stop_silence_s=0.3, min_speech_ms=90, min_record_s=0.3,
                    preroll_s=0.15)
    seg = Segmenter(cfg)
    # ideal speech, unlike the toolkit default: no noise floor to learn from
    seg.floor = 0.0

    assert feed(seg, silence(0.5)) == []          # nothing while quiet
    got = feed(seg, np.concatenate([speech(0.6), silence(0.45)]))
    assert len(got) == 1, "one speech burst must produce exactly one recording"
    audio = got[0]
    assert len(audio) / SAMPLE_RATE >= 0.6
    # pre-roll and the trailing silence are part of the recording
    assert len(audio) / SAMPLE_RATE <= 0.15 + 0.6 + 0.45 + 0.1


def test_two_bursts_produce_two_recordings():
    cfg = VadConfig(stop_silence_s=0.2, min_speech_ms=60, min_record_s=0.2,
                    preroll_s=0.1)
    seg = Segmenter(cfg)
    seg.floor = 0.0
    audio = np.concatenate([silence(0.3), speech(0.4), silence(0.3),
                           speech(0.4), silence(0.3)])
    got = feed(seg, audio)
    assert len(got) == 2
    for rec in got:
        assert 0.2 <= len(rec) / SAMPLE_RATE <= 1.2


def test_short_blip_is_discarded():
    cfg = VadConfig(stop_silence_s=0.2, min_speech_ms=30, min_record_s=0.8)
    seg = Segmenter(cfg)
    seg.floor = 0.0
    for frame in np.split(speech(0.09), 3):       # three loud frames
        seg.feed(frame)
    for frame in np.split(silence(0.09), 3):
        seg.feed(frame)
    assert feed(seg, silence(0.6)) == [], 'a sub-minimum blip must be discarded'


def test_noise_floor_does_not_trigger_on_quiet_room():
    cfg = VadConfig(stop_silence_s=0.2, min_speech_ms=30, min_record_s=0.2)
    seg = Segmenter(cfg)
    rng = np.random.default_rng(0)
    quiet = (rng.normal(0, 0.002, FRAME_SAMPLES) * 3).astype(np.float32)
    got = []
    for _ in range(200):                          # 6 s of a stable quiet floor
        done = seg.feed(quiet)
        if done is not None:
            got.append(done)
    assert got == []


def test_capture_file_writes_pipeline_filenames(tmp_path: Path):
    src = tmp_path / "clip.wav"
    sf.write(src, np.concatenate([silence(0.2), speech(0.5), silence(0.4),
                                  speech(0.5), silence(0.4)]), SAMPLE_RATE)
    spool = tmp_path / "spool"
    n = capture_file(src, spool, "room1", VadConfig(stop_silence_s=0.25,
                                                   min_speech_ms=60,
                                                   min_record_s=0.2))
    files = sorted(spool.glob("*.wav"))
    assert n == len(files) == 2
    for f in files:
        assert re.match(r"^\d{8}-\d{6}-room1\.wav$", f.name)
        assert sf.info(str(f)).samplerate == SAMPLE_RATE
        assert not list(spool.glob("*.part"))
