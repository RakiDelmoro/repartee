#!/usr/bin/env python3
"""Trim silent codec frames from teacher audio before training."""

import numpy as np
import soundfile as sf

FRAME_SAMPLES = 1920  # 80 ms at 24 kHz
THRESHOLD = 0.01


def trim_codes(codes: np.ndarray, wav_path: str, *, lead: bool = True, tail: bool = True,
               keep: int = 2) -> np.ndarray:
    """codes: (num_codebooks, num_frames). Drops silent frames at the ends."""
    wav, _ = sf.read(str(wav_path), dtype="float32")
    wav = wav if wav.ndim == 1 else wav.mean(1)
    n = codes.shape[1]
    rms = []
    for i in range(n):
        chunk = wav[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
        rms.append(float(np.sqrt((chunk ** 2).mean())) if len(chunk) else 0.0)
    rms = np.asarray(rms)
    speech = np.where(rms > THRESHOLD)[0]
    if len(speech) == 0:
        return codes
    start = int(speech[0]) if lead else 0
    end = int(speech[-1]) + 1 if tail else n
    start = max(0, start)
    end = min(n, end)
    if end - start < keep:
        return codes
    return codes[:, start:end]
