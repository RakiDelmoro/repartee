#!/usr/bin/env python3
"""Generate a small Vani voice validation set."""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "vani-2b/train")
from inject_probe import load
from revocalize import tts

OUT = Path("data/probe/samples/vani_voice")
SR = 24_000
LINES = [
    ("warm", "Hi, I'm Vani. I'll be listening in on your conversation."),
    ("neutral", "The launch is scheduled for the third of March."),
    ("names", "Dr. Okafor will review the vendor contract tomorrow."),
    ("backchannel", "That's a good point. Let me look into it."),
    ("business", "We spent most of the meeting on the billing migration."),
    ("helpful", "I can help with that once you're ready."),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model, proc, codec, load_s = load()
    print("loaded", round(load_s, 1), flush=True)
    manifest = []
    for name, text in LINES:
        wav = tts(model, proc, codec, text)
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        path = OUT / f"{name}.wav"
        sf.write(path, np.clip(wav, -1, 1), SR)
        manifest.append({"file": path.name, "text": text, "duration_s": round(len(wav) / SR, 2)})
        print(path.name, len(wav) / SR, flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
