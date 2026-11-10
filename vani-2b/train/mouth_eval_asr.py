#!/usr/bin/env python3
"""ASR-only report for the generated mouth adapter clips."""

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "vani-2b/train")
from mouth_asr import norm, wer

ROOT = Path("data/probe/mouth")
eval_dir = ROOT / "eval"
pairs = {p["id"]: p for p in json.load(open(ROOT / "train_pairs.json")) if p["split"] == "holdout"}
meta = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "meta.jsonl") if '"error"' not in l}
cache_path = ROOT / "eval_asr_cache.jsonl"
cache = {}
if cache_path.exists():
    for line in cache_path.read_text().splitlines():
        r = json.loads(line)
        cache[r["id"]] = r["hyp"]

import soundfile as sf
todo = [p for p in sorted(eval_dir.glob("*.wav")) if p.stem not in cache]
if todo:
    from app.asr import QwenTranscriber
    t = QwenTranscriber()
    with cache_path.open("a") as f:
        for i, p in enumerate(todo, 1):
            hyp = t.transcribe(p).text
            cache[p.stem] = hyp
            f.write(json.dumps({"id": p.stem, "hyp": hyp}) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"asr {i}/{len(todo)}", flush=True)

rows = []
for pid in cache:
    if pid not in pairs or pid not in meta:
        continue
    p = eval_dir / f"{pid}.wav"
    dur = sf.info(p).duration
    frames = round(dur / 0.08)
    rows.append({
        "id": pid, "text": pairs[pid]["text"], "hyp": cache[pid],
        "wer": round(wer(norm(pairs[pid]["text"]), norm(cache[pid])), 4),
        "audio_s": round(dur, 2), "frames": frames,
        "teacher_frames": meta[pid]["frames"],
        "frame_ratio": round(frames / meta[pid]["frames"], 3),
        "words_per_s": round(len(pairs[pid]["text"].split()) / max(dur, 1e-9), 2),
    })

report = {
    "n": len(rows),
    "exact": round(sum(r["wer"] == 0 for r in rows) / len(rows), 4),
    "mean_wer": round(st.mean(r["wer"] for r in rows), 4),
    "median_words_per_s": round(st.median(r["words_per_s"] for r in rows), 2),
    "median_frame_ratio": round(st.median(r["frame_ratio"] for r in rows), 3),
    "dropped_tail": round(sum(r["frame_ratio"] < 0.7 for r in rows) / len(rows), 4),
    "rows": rows,
}
(ROOT / "eval_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
for r in sorted(rows, key=lambda x: -x["wer"])[:6]:
    print(f"\nWER {r['wer']}  {r['text']}\n        {r['hyp']}")
