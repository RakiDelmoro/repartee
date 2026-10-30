"""Audio loading, resampling and quality checks shared by data and serving.

Uses soundfile for I/O and torch for resampling.  Both are training-stack
dependencies; this module stays import-light so data builds do not need CUDA.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def read_audio(path: str | Path, sample_rate: int, *, mono: bool = True) -> np.ndarray:
    """Load ``path`` at exactly ``sample_rate`` and return float32 ``[samples]``.

    Stereo files are averaged.  Resampling is polyphase and deterministic.
    """
    data, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data.mean(axis=1) if mono else data
    audio = np.asarray(audio, dtype=np.float32)
    if source_rate != sample_rate:
        audio = resample(audio, source_rate, sample_rate)
    return np.ascontiguousarray(audio, dtype=np.float32)


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    import torch

    tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    target = int(round(tensor.numel() * target_rate / source_rate))
    if target <= 0:
        return np.zeros(0, dtype=np.float32)
    import torchaudio.functional as F

    resampled = F.resample(tensor, source_rate, target_rate)
    del target, F
    return resampled.numpy().astype(np.float32, copy=False)


def write_audio(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")


def fit_frames(audio: np.ndarray, frames: int, sample_rate: int) -> np.ndarray:
    """Trim or zero-pad ``audio`` to exactly ``frames`` 80 ms frames."""
    from .schema import FRAME_SAMPLES

    per_frame = FRAME_SAMPLES[sample_rate]
    target = frames * per_frame
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size >= target:
        return np.ascontiguousarray(audio[:target])
    return np.pad(audio, (0, target - audio.size))


def clipping_ratio(audio: np.ndarray, threshold: float = 0.999) -> float:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= threshold))


def rms_dbfs(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return -math.inf
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-12))


def peak_dbfs(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return -math.inf
    return 20.0 * math.log10(max(float(np.max(np.abs(audio))), 1e-12))


def channel_report(audio: np.ndarray) -> dict[str, Any]:
    return {
        "samples": int(np.asarray(audio).size),
        "clipping_ratio": clipping_ratio(audio),
        "rms_dbfs": rms_dbfs(audio),
        "peak_dbfs": peak_dbfs(audio),
    }
