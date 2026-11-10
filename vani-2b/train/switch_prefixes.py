#!/usr/bin/env python3
"""Render the switch prefixes (A) with the TTS teacher."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE, load

ROOT = Path("data/probe/mouth/switch")
SR = 24_000
PREFIXES = [
    "Let me check that for you.",
    "One moment, please.",
    "Good question, let me see.",
    "I'm looking that up now.",
    "Let me pull that up.",
    "The short answer is.",
    "Just a second.",
    "Let me think about that.",
    "Okay, I have that here.",
    "Right, so.",
    "Give me a moment.",
    "Sure, let me check.",
]


@torch.no_grad()
def main() -> None:
    (ROOT / "prefixes").mkdir(parents=True, exist_ok=True)
    model, proc, codec, load_s = load()
    print("loaded", round(load_s, 1), flush=True)
    meta = []
    t0 = time.perf_counter()
    for i, text in enumerate(PREFIXES):
        pid = f"p{i:02d}"
        conv = [
            {"role": "system", "content": "Perform TTS in American English using a female voice."},
            {"role": "user", "content": text},
        ]
        inputs = proc.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(DEVICE)
        out = model.generate(
            **inputs, max_new_tokens=120, generation_mode="audio",
            audio_temperature=0.4, audio_repetition_penalty=1.2,
        )
        n = int(out.audio_lengths[0])
        codes = out.audio_codes[0, :, :n].to(torch.int16).cpu().numpy()
        wav = codec.decode(out.audio_codes[:1, :, :n]).audio_values[0].float().cpu().numpy().squeeze()
        sf.write(ROOT / "prefixes" / f"{pid}.wav", wav, SR)
        np.save(ROOT / "prefixes" / f"{pid}.npy", codes)
        meta.append({"id": pid, "text": text, "frames": n, "audio_s": round(len(wav) / SR, 2)})
        print(pid, round(len(wav) / SR, 2), text, flush=True)
    (ROOT / "prefixes.json").write_text(json.dumps(meta, indent=2))
    print("wall", round(time.perf_counter() - t0, 1), "s")


if __name__ == "__main__":
    main()
