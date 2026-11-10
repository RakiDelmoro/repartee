#!/usr/bin/env python3
"""Continuity switch data: render lead-in + answer as one utterance, find the join."""

import json
import random
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE, load

ROOT = Path("data/probe/mouth")
OUT = ROOT / "continuity"
SR = 24_000
LEADS = [
    "Okay, so",
    "Right, so",
    "Well,",
    "So,",
    "Alright, so",
    "Okay, so here's the thing",
]
N_TRAIN = 80
N_HOLD = 40
SEED = 0


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
    codes = out.audio_codes[0, :, :n].to(torch.int16).cpu().numpy()
    wav = codec.decode(out.audio_codes[:1, :, :n]).audio_values[0].float().cpu().numpy().squeeze()
    return wav, codes


def find_join(prefix_wav: Path, combined_wav: Path) -> float:
    a, sr = librosa.load(prefix_wav, sr=SR, mono=True)
    c, _ = librosa.load(combined_wav, sr=SR, mono=True)
    ma = librosa.feature.mfcc(y=a, sr=sr, n_mfcc=20, hop_length=512)
    mc = librosa.feature.mfcc(y=c, sr=sr, n_mfcc=20, hop_length=512)
    _, wp = librosa.sequence.dtw(X=mc, Y=ma, subseq=True)
    return float(wp[:, 0].max() * 512 / sr)


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    (OUT / "leads").mkdir(parents=True, exist_ok=True)
    (OUT / "pairs").mkdir(parents=True, exist_ok=True)
    model, proc, codec, load_s = load()

    leads_meta = []
    for i, text in enumerate(LEADS):
        wid = f"l{i:02d}"
        wav, codes = tts(model, proc, codec, text)
        sf.write(OUT / "leads" / f"{wid}.wav", wav, SR)
        np.save(OUT / "leads" / f"{wid}.npy", codes)
        leads_meta.append({"id": wid, "text": text, "frames": codes.shape[1]})
        print("lead", wid, text, flush=True)

    pairs = json.load(open(ROOT / "train_pairs.json"))
    train_b = [p for p in pairs if p["split"] == "train"][:N_TRAIN]
    hold_b = [p for p in pairs if p["split"] == "holdout"][:N_HOLD]
    meta = []
    t0 = time.perf_counter()
    for i, p in enumerate(train_b + hold_b):
        lead = leads_meta[i % len(leads_meta)]
        text = f"{lead['text']} {p['text']}"
        wid = f"c{i:04d}"
        wav, codes = tts(model, proc, codec, text)
        sf.write(OUT / "pairs" / f"{wid}.wav", wav, SR)
        np.save(OUT / "pairs" / f"{wid}.npy", codes)
        cut = find_join(OUT / "leads" / f"{lead['id']}.wav", OUT / "pairs" / f"{wid}.wav")
        meta.append({
            "id": wid, "split": "train" if i < N_TRAIN else "holdout",
            "lead_id": lead["id"], "lead_text": lead["text"], "b_id": p["id"], "b_text": p["text"],
            "combined_text": text, "frames": int(codes.shape[1]),
            "cut_s": round(cut, 2), "cut_frames": max(2, round(cut / 0.08)),
        })
        if (i + 1) % 10 == 0:
            print(i + 1, len(train_b) + len(hold_b), "elapsed", round(time.perf_counter() - t0, 1), flush=True)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print("pairs", len(meta))


if __name__ == "__main__":
    main()
