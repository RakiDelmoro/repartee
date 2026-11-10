#!/usr/bin/env python3
"""Re-voice the agent side of short Open Yap conversations with the product TTS."""

import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE, load

PILOT = Path("data/open-yap/pilot")
OUT = Path("data/open-yap/pilot_tts")
SR = 24_000
CONVS = ["conv_a69239616d4e", "conv_3e2001c6b7e9", "conv_96c9e53e9310",
         "conv_63a80d7053c1", "conv_714ff6b8cd50", "conv_79be53d2a257",
         "conv_cd9656663384", "conv_eec711c3ca25", "conv_6f1d55110124",
         "conv_31a410c2d2ed", "conv_0e7dab27dc32"]
MIN_GAP = 0.4


@torch.no_grad()
def tts(model, proc, codec, text):
    conv = [
        {"role": "system", "content": "Perform TTS in American English using a female voice."},
        {"role": "user", "content": text},
    ]
    inputs = proc.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(DEVICE)
    out = model.generate(
        **inputs, max_new_tokens=240, generation_mode="audio",
        audio_temperature=0.4, audio_repetition_penalty=1.2,
    )
    n = int(out.audio_lengths[0])
    return codec.decode(out.audio_codes[:1, :, :n]).audio_values[0].float().cpu().numpy().squeeze()


def turns(words):
    out = []
    cur = []
    for w in words:
        if w.get("type") != "word":
            continue
        if cur and w["start"] - cur[-1]["end"] > MIN_GAP:
            out.append(cur); cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model, proc, codec, load_s = load()
    print("loaded", round(load_s, 1), flush=True)
    t0 = time.perf_counter()
    for cid in CONVS:
        if (OUT / cid / "agent_tts.wav").exists():
            print("skip", cid, flush=True); continue
        conv = PILOT / "conversations" / cid
        meta = json.load(open(conv / "meta.json"))
        total = int(meta["duration_seconds"] * SR)
        track = np.zeros(total, dtype=np.float32)
        words = json.load(open(conv / "speaker_a_transcript.json"))["words"]
        meta_turns = []
        for ti, turn in enumerate(turns(words)):
            text = " ".join(w["word"] for w in turn)
            s = int(turn[0]["start"] * SR); e = int(turn[-1]["end"] * SR)
            target = max(e - s, int(0.2 * SR))
            wav = tts(model, proc, codec, text)
            if len(wav) < 400:
                continue
            rate = len(wav) / target
            if abs(rate - 1.0) > 0.02:
                wav = librosa.effects.time_stretch(wav, rate=rate)
            wav = wav[:target]
            track[s:s + len(wav)] += wav
            meta_turns.append({"start": round(s / SR, 2), "end": round((s + len(wav)) / SR, 2), "text": text})
        (OUT / cid).mkdir(exist_ok=True)
        sf.write(OUT / cid / "agent_tts.wav", np.clip(track, -1, 1), SR)
        (OUT / cid / "turns.json").write_text(json.dumps(meta_turns, indent=1))
        print(cid, len(meta_turns), "turns", round(time.perf_counter() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
