#!/usr/bin/env python3
"""Analyze Open Yap conversations at 80 ms: speech, overlap, backchannels."""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

FRAME = 0.08
SR = 48_000
VAD_WIN = 0.02


def speech_mask(path: Path, n_frames: int, words: list[dict]) -> np.ndarray:
    info = sf.info(str(path))
    total = int(info.frames / SR / FRAME) + 1
    mask = np.zeros(total, dtype=bool)
    for w in words:
        if w.get("type") != "word":
            continue
        a, b = int(w["start"] / FRAME), int(np.ceil(w["end"] / FRAME))
        mask[a:min(b, total)] = True
    return mask[:n_frames]


def energy_mask(path: Path, n_frames: int) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32")
    x = x if x.ndim == 1 else x.mean(1)
    hop = int(VAD_WIN * sr)
    n = len(x) // hop
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1))
    floor = np.percentile(rms, 10)
    active = rms > max(floor * 2.5, 0.004)
    # 80 ms frame is active if at least two 20 ms windows are active
    per = int(FRAME / VAD_WIN)
    k = n // per
    return (active[: k * per].reshape(k, per).sum(1) >= 2)[:n_frames]


def run(conv: Path) -> dict:
    meta = json.load(open(conv / "meta.json"))
    n_frames = int(meta["duration_seconds"] / FRAME)
    masks = {}
    for s in ("a", "b"):
        words = json.load(open(conv / f"speaker_{s}_transcript.json"))["words"]
        masks[s] = energy_mask(conv / f"speaker_{s}.flac", n_frames)
    a, b = masks["a"], masks["b"]
    overlap = a & b
    # agent = a, user = b: backchannel = a run <= 0.96 s that overlaps b
    runs, start = [], None
    for i, v in enumerate(a):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(a)))
    bc_frames = 0
    bc_count = 0
    for s0, s1 in runs:
        if (s1 - s0) * FRAME <= 0.96 and overlap[s0:s1].mean() > 0.5:
            bc_frames += s1 - s0
            bc_count += 1
    total = n_frames
    return {
        "conv": conv.name,
        "duration_min": round(meta["duration_seconds"] / 60, 1),
        "frames": total,
        "a_speech_pct": round(a.mean() * 100, 1),
        "b_speech_pct": round(b.mean() * 100, 1),
        "overlap_pct_of_frames": round(overlap.mean() * 100, 2),
        "overlap_pct_of_speech": round(overlap.sum() / max((a | b).sum(), 1) * 100, 2),
        "backchannel_segments": bc_count,
        "backchannel_frames_pct": round(bc_frames / max(a.sum(), 1) * 100, 2),
        "bc_per_min": round(bc_count / (meta["duration_seconds"] / 60), 2),
        "reported_overlap": json.load(open(conv.parent.parent / "conversations" / "verification.json"))["conversations"][conv.name]["overlap_pct_of_speech_time"] if False else None,
    }


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(run(Path(d)), indent=1))
