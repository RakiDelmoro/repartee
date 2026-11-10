#!/usr/bin/env python3
"""Pair prefixes with target clauses and switch points."""

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("data/probe/mouth")
OUT = ROOT / "switch"
PREFIXES = json.loads((OUT / "prefixes.json").read_text())
PAIRS = json.loads((ROOT / "train_pairs.json").read_text())
META = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "meta.jsonl").read_text().splitlines() if '"error"' not in l}
rng = random.Random(0)
FRACTIONS = [0.3, 0.5, 0.7]


def pick_switch(pref, frac):
    wav, _ = sf.read(str(OUT / "prefixes" / f"{pref['id']}.wav"), dtype="float32")
    wav = wav if wav.ndim == 1 else wav.mean(1)
    n = pref["frames"]
    rms = np.array([np.sqrt((wav[i * 1920:(i + 1) * 1920] ** 2).mean()) for i in range(n)])
    target = max(2, round(frac * n))
    lo, hi = max(1, target - 4), min(n, target + 5)
    return lo + int(np.argmin(rms[lo:hi]))

rows = []
for split, per_clause in (("train", 2), ("holdout", 1)):
    clauses = [p for p in PAIRS if p["split"] == split]
    for clause in clauses:
        for _ in range(per_clause):
            pref = rng.choice(PREFIXES)
            frac = rng.choice(FRACTIONS)
            switch = pick_switch(pref, frac)
            rows.append({
                "id": f"s{len(rows):05d}",
                "split": split,
                "b_id": clause["id"],
                "b_text": clause["text"],
                "b_frames": META[clause["id"]]["frames"],
                "prefix_id": pref["id"],
                "prefix_text": pref["text"],
                "prefix_frames": pref["frames"],
                "switch_frames": switch,
            })

(OUT / "pairs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print("pairs", len(rows), "train", sum(r["split"] == "train" for r in rows),
      "holdout", sum(r["split"] == "holdout" for r in rows))
for r in rows[:4]:
    print(r["split"], r["prefix_text"], "|", r["b_text"], "| switch", r["switch_frames"], "/", r["prefix_frames"])
