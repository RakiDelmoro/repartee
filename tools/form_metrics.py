"""Form metrics from a duplex stereo WAV (L = user, R = model)."""

from __future__ import annotations

import argparse
import json

import numpy as np
import soundfile as sf

FRAME = 1920  # 80 ms at 24 kHz


def flags(x: np.ndarray) -> tuple[np.ndarray, float]:
    n = len(x) // FRAME
    rms = np.array([np.sqrt(np.mean(x[i * FRAME:(i + 1) * FRAME] ** 2)) for i in range(n)])
    floor = float(np.percentile(rms, 10))
    thr = max(3 * floor, 0.005)
    return rms > thr, float(thr)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i)); start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    a = ap.parse_args()
    x, sr = sf.read(a.wav, dtype="float32")
    if x.ndim != 2:
        raise SystemExit("expected stereo")
    L, R = x[:, 0], x[:, 1]
    fu, tu = flags(L)
    fa, ta = flags(R)
    n = min(len(fu), len(fa)); fu, fa = fu[:n], fa[:n]
    dur = n * 0.08
    overlap = int((fu & fa).sum())
    dead = runs(~(fu | fa))
    longest_dead = max((b - a) for a, b in dead) * 0.08 if dead else 0.0
    # backchannels: model speech runs < 0.6 s overlapping user speech
    bc = 0
    for s, e in runs(fa):
        if (e - s) * 0.08 < 0.6 and fu[s:e].any():
            bc += 1
    # response latency: after a user run ends, frames until agent starts (if within 1 s)
    lat = []
    for s, e in runs(fu):
        nxt = np.where(fa[e:e + 13])[0]
        if len(nxt):
            lat.append(int(nxt[0]) * 0.08)
    print(json.dumps({
        "seconds": round(dur, 1),
        "user_speech_pct": round(100 * fu.mean(), 1),
        "agent_speech_pct": round(100 * fa.mean(), 1),
        "overlap_pct": round(100 * overlap / n, 3),
        "agent_only_pct": round(100 * (fa & ~fu).sum() / n, 1),
        "longest_dead_air_s": round(longest_dead, 2),
        "backchannels": bc,
        "backchannels_per_min": round(bc / (dur / 60), 2),
        "response_latency_mean_s": round(float(np.mean(lat)), 3) if lat else None,
        "thresholds": [round(tu, 5), round(ta, 5)],
    }, indent=2))


if __name__ == "__main__":
    main()
