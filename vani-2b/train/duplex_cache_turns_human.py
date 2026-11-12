#!/usr/bin/env python3
"""Diagnostic cache: turn-boundary windows with the human agent (no re-voicing).

If duplex learns to take the floor here, the failure was scale/turn coverage, not
the re-voicing. Voice consistency is not expected in this diagnostic.
"""

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE

MODEL_ID = "models/LFM3-Audio-2B-hf"
PILOT = Path("data/open-yap/pilot")
OUT = Path("data/probe/duplex/cache_turns_human")
SR_U, SR_A = 16_000, 24_000
FRAMES = 63
PER_CONV = 12


def main() -> None:
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    OUT.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()
    codec = proc.audio_tokenizer.to(device=DEVICE, dtype=torch.float32).eval()
    enc = model.model

    manifest = []
    for conv in sorted((PILOT / "conversations").glob("conv_*")):
        lab = np.load(PILOT / "labels" / f"{conv.name}.npz")
        a_speak, b_speak, overlap = lab["a_speak"], lab["b_speak"], lab["overlap"]
        n = len(a_speak)
        starts = []
        for i in range(1, n - FRAMES):
            if b_speak[i - 1] and not b_speak[i] and a_speak[i:i + 38].sum() > 5:
                starts.append(max(0, i - 25))
        for s in range(0, n - FRAMES, 25):
            if a_speak[s:s + FRAMES].sum() > 20:
                starts.append(s)
        starts = sorted(set(starts))[:PER_CONV]
        if not starts:
            continue
        for s in starts:
            t0 = s * 0.08
            u, _ = librosa.load(conv / "speaker_b.flac", sr=SR_U, mono=True, offset=t0, duration=FRAMES * 0.08)
            a, _ = librosa.load(conv / "speaker_a.flac", sr=SR_A, mono=True, offset=t0, duration=FRAMES * 0.08)
            if len(a) < 400:
                continue
            sf.write("/tmp/turn_h_u.wav", u, SR_U)
            inp = proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "audio", "audio": "/tmp/turn_h_u.wav"}]}],
                add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt",
            )
            with torch.no_grad():
                ue, _ = enc.get_audio_features_stream(inp["input_features"].to(DEVICE), None, True)
                codes = codec.encode(torch.tensor(a)[None].to(DEVICE)).audio_codes
            k = min(ue.shape[1], codes.shape[-1], FRAMES)
            wid = f"{conv.name}_{s:06d}"
            np.savez_compressed(
                OUT / f"{wid}.npz",
                u=ue[0, :k].half().cpu().numpy(),
                codes=codes[0, :, :k].t().short().cpu().numpy(),
                speak=a_speak[s:s + k], b_speak=b_speak[s:s + k], overlap=overlap[s:s + k],
            )
            manifest.append({"id": wid, "conv": conv.name, "start_s": round(t0, 2),
                             "frames": k, "speak": int(a_speak[s:s + k].sum())})
            print(wid, k, int(a_speak[s:s + k].sum()), flush=True)
    (Path("data/probe/duplex/cache_turns_human_manifest.json")).write_text(json.dumps(manifest, indent=1))
    print("windows", len(manifest))


if __name__ == "__main__":
    main()
