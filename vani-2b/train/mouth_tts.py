#!/usr/bin/env python3
"""Render the mouth corpus with the model's TTS mode (resumable)."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE, load

ROOT = Path("data/probe/mouth")
WAV = ROOT / "wav"
CODES = ROOT / "codes"
META = ROOT / "meta.jsonl"
SR = 24_000


@torch.no_grad()
def main() -> None:
    WAV.mkdir(parents=True, exist_ok=True)
    CODES.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (ROOT / "corpus.jsonl").read_text().splitlines()]
    done = {p.stem for p in WAV.glob("*.wav")}
    todo = [r for r in rows if r["id"] not in done]
    print(f"corpus={len(rows)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return

    model, proc, codec, load_s = load()
    print(f"loaded in {load_s:.1f}s", flush=True)
    t_start = time.perf_counter()
    with META.open("a") as meta:
        for i, row in enumerate(todo, 1):
            conv = [
                {"role": "system", "content": "Perform TTS in American English using a female voice."},
                {"role": "user", "content": row["text"]},
            ]
            try:
                inputs = proc.apply_chat_template(
                    conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                ).to(DEVICE)
                t0 = time.perf_counter()
                out = model.generate(
                    **inputs, max_new_tokens=200, generation_mode="audio",
                    audio_temperature=0.4, audio_repetition_penalty=1.2,
                )
                torch.cuda.synchronize()
                gen_s = time.perf_counter() - t0
                n = int(out.audio_lengths[0])
                codes = out.audio_codes[0, :, :n].to(torch.int16).cpu().numpy()
                wav = codec.decode(out.audio_codes[:1, :, :n]).audio_values[0].float().cpu().numpy().squeeze()
                sf.write(WAV / f"{row['id']}.wav", wav, SR)
                np.save(CODES / f"{row['id']}.npy", codes)
                rec = {
                    "id": row["id"], "text": row["text"], "family": row["family"],
                    "split": row["split"], "frames": n, "audio_s": round(len(wav) / SR, 2),
                    "gen_s": round(gen_s, 2),
                    "words_per_s": round(len(row["text"].split()) / max(len(wav) / SR, 1e-9), 2),
                }
                meta.write(json.dumps(rec) + "\n")
                meta.flush()
            except Exception as exc:  # keep the batch alive; log and continue
                meta.write(json.dumps({"id": row["id"], "error": str(exc)}) + "\n")
                meta.flush()
                print("ERR", row["id"], exc, flush=True)
            if i % 10 == 0:
                elapsed = time.perf_counter() - t_start
                rate = i / elapsed
                print(f"{i}/{len(todo)}  {rate:.2f} clips/s  eta {(len(todo)-i)/max(rate,1e-9)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
