#!/usr/bin/env python3
"""Turn-boundary duplex cache: windows around user stops, TTS-energy labels."""

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
TTS = Path("data/open-yap/pilot_tts")
OUT = Path("data/probe/duplex/cache_turns")
SR_U, SR_A = 16_000, 24_000
FRAMES = 63
PER_CONV = 100


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
    for cdir in sorted(TTS.glob("conv_*")):
        cid = cdir.name
        conv = PILOT / "conversations" / cid
        lab = np.load(PILOT / "labels" / f"{cid}.npz")
        b_speak, overlap = lab["b_speak"], lab["overlap"]
        y, _ = librosa.load(cdir / "agent_tts.wav", sr=SR_A, mono=True)
        hop = int(0.08 * SR_A); m = len(y) // hop
        rms = np.sqrt((y[:m * hop].reshape(m, hop) ** 2).mean(1))
        a_speak = rms > 0.01
        n = min(len(a_speak), len(b_speak), m)

        starts = []
        for i in range(1, n - FRAMES):
            if b_speak[i - 1] and not b_speak[i]:
                s = max(0, i - 25)
                if a_speak[i:i + 38].sum() > 5:
                    starts.append(s)
        # add agent-speech-rich windows for balance
        for s in range(0, n - FRAMES, 25):
            if a_speak[s:s + FRAMES].sum() > 20:
                starts.append(s)
        starts = sorted(set(starts))
        if len(starts) > PER_CONV:
            idx = np.linspace(0, len(starts) - 1, PER_CONV).astype(int)
            starts = [starts[i] for i in idx]
        if not starts:
            continue
        for s in starts:
            t0 = s * 0.08
            u, _ = librosa.load(conv / "speaker_b.flac", sr=SR_U, mono=True, offset=t0, duration=FRAMES * 0.08)
            a, _ = librosa.load(cdir / "agent_tts.wav", sr=SR_A, mono=True, offset=t0, duration=FRAMES * 0.08)
            if len(a) < 400:
                continue
            sf.write("/tmp/turn_u.wav", u, SR_U)
            inp = proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "audio", "audio": "/tmp/turn_u.wav"}]}],
                add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt",
            )
            with torch.no_grad():
                ue, _ = enc.get_audio_features_stream(inp["input_features"].to(DEVICE), None, True)
                codes = codec.encode(torch.tensor(a)[None].to(DEVICE)).audio_codes
            k = min(ue.shape[1], codes.shape[-1], FRAMES)
            wid = f"{cid}_{s:06d}"
            np.savez_compressed(
                OUT / f"{wid}.npz",
                u=ue[0, :k].half().cpu().numpy(),
                codes=codes[0, :, :k].t().short().cpu().numpy(),
                speak=a_speak[s:s + k], b_speak=b_speak[s:s + k], overlap=overlap[s:s + k],
            )
            manifest.append({"id": wid, "conv": cid, "start_s": round(t0, 2), "frames": k,
                             "speak": int(a_speak[s:s + k].sum())})
            print(wid, k, int(a_speak[s:s + k].sum()), flush=True)
    (Path("data/probe/duplex/cache_turns_manifest.json")).write_text(json.dumps(manifest, indent=1))
    print("windows", len(manifest))


if __name__ == "__main__":
    main()
