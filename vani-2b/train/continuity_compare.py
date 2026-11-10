#!/usr/bin/env python3
"""Controlled v1 (switch) vs v2 (continuity) on the same lead-ins and cuts."""

import json
import random
import shutil
import sys
from pathlib import Path

import soundfile as sf
import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

sys.path.insert(0, ".")
sys.path.insert(0, "vani-2b/train")
from continuity_eval import b_ratio, gap_ms, synth
from inject_probe import DEVICE

MODEL_ID = "models/LFM3-Omni-2B-hf"
ROOT = Path("data/probe/mouth")
CT = ROOT / "continuity"
OUT = CT / "listen"
SELECTED = ["c0080", "c0082", "c0084", "c0089", "c0085"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "blind").mkdir(exist_ok=True)
    meta = {m["id"]: m for m in json.load(open(CT / "meta.json"))}
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    codec = proc.audio_tokenizer.to(device=DEVICE, dtype=torch.float32).eval()
    base = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()
    mouth = PeftModel.from_pretrained(base, ROOT / "lora").merge_and_unload().eval()
    v1 = PeftModel.from_pretrained(mouth, ROOT / "switch" / "lora").merge_and_unload().eval()

    rows = []
    for i, cid in enumerate(SELECTED, 1):
        m = meta[cid]
        label = f"set{i}"
        wav = synth(v1, proc, codec, m["lead_text"], m["b_text"], m["cut_frames"])
        sf.write(OUT / f"{label}_v1.wav", wav, SR if (SR := 24_000) else 24_000)
        shutil.copy(CT / "eval" / f"{cid}.wav", OUT / f"{label}_v2.wav")
        rows.append({"label": label, "id": cid, "lead": m["lead_text"], "b": m["b_text"], "cut_s": m["cut_s"]})
        print(label, cid, m["lead_text"], "|", m["b_text"], flush=True)

    del v1, mouth, base
    torch.cuda.empty_cache()
    from app.asr import QwenTranscriber
    t = QwenTranscriber()
    for r in rows:
        for kind in ("v1", "v2"):
            path = OUT / f"{r['label']}_{kind}.wav"
            hyp = t.transcribe(path).text
            r[kind] = {"hyp": hyp, "s": round(sf.info(path).duration, 2),
                       "gap_ms": gap_ms(path, r["cut_s"]),
                       "b_recall": round(b_ratio(hyp, r["b"]), 3)}

    rng = random.Random(21); key = {}
    for r in rows:
        opts = [("A", "v1"), ("B", "v2")]; rng.shuffle(opts); mapping = {}
        for letter, kind in opts:
            shutil.copy(OUT / f"{r['label']}_{kind}.wav", OUT / "blind" / f"{r['label']}_{letter}.wav")
            mapping[letter] = kind
        key[r["label"]] = mapping
    (OUT / "blind" / "key.json").write_text(json.dumps(key, indent=2))

    lines = ["# Continuity v1 vs v2", "",
             "v1 = switch adapter trained on separate A/B recordings. v2 = continuity adapter, A+B rendered as one utterance.",
             "", "| Set | Lead | Answer | Cut | v1 ASR | v1 gap | v2 ASR | v2 gap |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['lead']} | {r['b']} | {r['cut_s']:.2f}s | {r['v1']['hyp']} | {r['v1']['gap_ms']}ms | {r['v2']['hyp']} | {r['v2']['gap_ms']}ms |")
    lines += ["", "Blind: `blind/setN_A/B.wav`, key in `blind/key.json`.", "",
              "Listen for whether v2 continues the lead's pitch and rhythm, or still starts a new sentence."]
    (OUT / "MANIFEST.md").write_text("\n".join(lines) + "\n")
    (CT / "compare_report.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
