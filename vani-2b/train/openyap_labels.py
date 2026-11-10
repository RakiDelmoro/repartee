#!/usr/bin/env python3
"""Build per-frame duplex labels for the Open Yap pilot.

Labels are for one agent role (speaker a); the loader can swap roles.
Speech regions come from word timestamps; true overlap and backchannels
come from the calibrated energy VAD and backchannel word content.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "vani-2b/train")
from openyap_frames import FRAME, energy_mask

PILOT = Path("data/open-yap/pilot")
OUT = PILOT / "labels"
BC = {"yeah", "right", "okay", "ok", "mm", "hmm", "mhm", "sure", "yep",
      "alright", "oh", "ah", "uh-huh", "i see", "got it", "exactly", "true"}


def word_mask(words, n):
    m = np.zeros(n, dtype=bool)
    for w in words:
        if w.get("type") != "word":
            continue
        a, b = int(w["start"] / FRAME), int(np.ceil(w["end"] / FRAME))
        m[a:min(b, n)] = True
    return m


def word_runs(words, n):
    runs = []
    for w in words:
        if w.get("type") != "word":
            continue
        a, b = int(w["start"] / FRAME), int(np.ceil(w["end"] / FRAME))
        runs.append((a, min(b, n), w["word"]))
    return runs


def run(conv: Path) -> dict:
    meta = json.load(open(conv / "meta.json"))
    n = int(meta["duration_seconds"] / FRAME)
    words = {s: json.load(open(conv / f"speaker_{s}_transcript.json"))["words"] for s in ("a", "b")}
    speak = {s: word_mask(words[s], n) for s in ("a", "b")}
    energy = {s: energy_mask(conv / f"speaker_{s}.flac", n) for s in ("a", "b")}
    overlap = energy["a"] & energy["b"]

    bc = np.zeros(n, dtype=bool)
    bc_count = 0
    for a, b, w in word_runs(words["a"], n):
        mid = (a + b) // 2
        if w.strip(".,?!").lower() in BC and 0 <= mid < n and energy["b"][mid]:
            bc[a:b] = True
            bc_count += 1

    # agent speech starts while the user is already speaking and lasts > 1 s
    interrupt = np.zeros(n, dtype=bool)
    int_count = 0
    prev = False
    for i, v in enumerate(speak["a"]):
        if v and not prev and energy["b"][i]:
            j = i
            while j < n and speak["a"][j]:
                j += 1
            if (j - i) * FRAME > 1.0:
                interrupt[i] = True
                int_count += 1
        prev = v

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / f"{conv.name}.npz",
        a_speak=speak["a"], b_speak=speak["b"], overlap=overlap,
        backchannel=bc, interrupt=interrupt,
    )
    minutes = meta["duration_seconds"] / 60
    return {
        "id": conv.name, "minutes": round(minutes, 1), "frames": n,
        "a_speech_pct": round(speak["a"].mean() * 100, 1),
        "b_speech_pct": round(speak["b"].mean() * 100, 1),
        "overlap_pct_of_speech": round(overlap.sum() / max((speak["a"] | speak["b"]).sum(), 1) * 100, 2),
        "backchannels": bc_count, "bc_per_min": round(bc_count / minutes, 2),
        "interrupts": int_count, "int_per_min": round(int_count / minutes, 2),
    }


def main() -> None:
    rows = []
    for conv in sorted((PILOT / "conversations").glob("conv_*")):
        r = run(conv)
        rows.append(r)
        print(r, flush=True)
    (PILOT / "labels_summary.json").write_text(json.dumps(rows, indent=1))
    print("total_minutes", round(sum(r["minutes"] for r in rows), 1))


if __name__ == "__main__":
    main()
