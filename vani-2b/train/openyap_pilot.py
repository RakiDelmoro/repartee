#!/usr/bin/env python3
"""Extract a high-overlap pilot from the Open Yap tar parts."""

import json
import tarfile
from pathlib import Path

ROOT = Path("data/open-yap/open-yap-1k_2026-11-05_flac_26a4d80b")
PILOT = Path("data/open-yap/pilot")
INDEX = json.load(open("data/open-yap/index.json"))
VER = json.load(open(ROOT / "conversations" / "verification.json"))["conversations"]
N = 20


def main() -> None:
    PILOT.mkdir(parents=True, exist_ok=True)
    cands = []
    for cid, v in VER.items():
        if v["overlap_pct_of_speech_time"] < 10:
            continue
        if max(v["clipping_pct"].values()) >= 0.5 or not v["anchored"]:
            continue
        cands.append((cid, v["overlap_pct_of_speech_time"]))
    cands.sort(key=lambda x: -x[1])
    chosen = cands[:N]
    print("candidates", len(cands), "chosen", len(chosen))

    manifest = []
    by_part = {}
    for cid, ov in chosen:
        by_part.setdefault(INDEX[cid], []).append((cid, ov))

    for part_name, items in by_part.items():
        part = ROOT / "parts" / part_name
        with tarfile.open(part) as tar:
            members = [m for m in tar.getmembers()
                       if any(m.name.startswith(f"conversations/{cid}/") for cid, _ in items)]
            tar.extractall(PILOT, members=members, filter="data")
        for cid, ov in items:
            meta = json.load(open(PILOT / "conversations" / cid / "meta.json"))
            manifest.append({
                "id": cid, "part": part_name, "overlap": ov,
                "duration_min": round(meta["duration_seconds"] / 60, 1),
                "is_segment": meta["is_segment"], "relationship": meta["relationship"],
                "turns_per_min": meta.get("turns_per_min"),
            })
        print(part_name, "extracted", len(items), flush=True)

    (PILOT / "pilot.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
