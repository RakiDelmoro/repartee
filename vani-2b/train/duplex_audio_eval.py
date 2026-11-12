#!/usr/bin/env python3
"""Evaluate pure audio-only duplex: audio in, audio out, silence frames for listening."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE
from joint_eval import sample_audio_frame
from app.asr import ASR

MODEL_ID = "models/LFM3-Audio-2B-hf"
ROOT = Path("data/probe/mouth")
CT = ROOT / "continuity"
DUP = Path("data/probe/duplex")
AUDIO = DUP / "audio_lora"
CACHE = DUP / "cache_turns"
OUT = Path("data/probe/samples/duplex_audio")
SR = 16000


def speech_frames(wav: np.ndarray, sr: int = SR) -> int:
    hop = int(0.08 * sr)
    n = len(wav) // hop
    total = 0
    for i in range(n):
        rms = float(np.sqrt(np.mean(wav[i * hop:(i + 1) * hop] ** 2)))
        total += rms > 0.01
    return total


def main() -> None:
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model.config.use_cache = False
    mouth = PeftModel.from_pretrained(model, ROOT / "lora").merge_and_unload()
    sw = PeftModel.from_pretrained(mouth, ROOT / "switch" / "lora").merge_and_unload()
    cont = PeftModel.from_pretrained(sw, CT / "lora_v2").merge_and_unload()
    model = PeftModel.from_pretrained(cont, AUDIO).merge_and_unload().eval()
    embed = model.get_input_embeddings()
    bos = embed(torch.tensor([[model.config.text_config.bos_token_id]], device=DEVICE))[0, 0]
    silence = None
    for path in sorted(CACHE.glob("*.npz")):
        if path.stem.split("_")[0] != "conv_c866":
            continue
        d = np.load(path)
        speak = d["speak"].astype(bool)
        if not speak.any():
            continue
        u = torch.from_numpy(d["u"]).to(DEVICE).half()
        n = len(u); t0 = path.stem.split("_")[1]
        print("window", path.name, "user_s", round(int(t0) * 5, 1), flush=True)
        with torch.no_grad():
            seq = torch.cat([bos.view(1, -1), u[0].view(1, -1)], 0)[None]
            modes = []
            for t in range(n):
                out = model.model.language_model(
                    inputs_embeds=seq,
                    attention_mask=torch.ones((1, seq.shape[1]), dtype=torch.long, device=DEVICE),
                    use_cache=False,
                )
                h = out.last_hidden_state[0, -1]
                codes = sample_audio_frame(model, h, seq.shape[1])
                if codes is None:
                    modes.append(None); break
                modes.append(codes)
                a = model.model.get_audio_out_embeds(torch.tensor([codes], device=DEVICE))[0, 0]
                seq = torch.cat([seq, a.view(1, 1, -1)], 1)
                if t + 1 < n:
                    seq = torch.cat([seq, u[t + 1].view(1, 1, -1)], 1)
        frames = [m for m in modes if m is not None]
        flat = [c for f in frames for c in f[:12]]
        if flat:
            audio = proc.decode_audio(torch.tensor([flat], device=DEVICE))[0]
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        else:
            audio = np.zeros(0, dtype=np.float32)
        path_out = OUT / f"{path.stem}.wav"
        path_out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path_out, audio, SR)
        print("frames", len(frames), "speech_frames", speech_frames(audio), flush=True)
        asr = ASR(os.environ.get("REPARTEE_ASR_MODEL", "models/Qwen3-ASR-0.6B-hf"))
        print("asr", repr(asr.transcribe(str(path_out))), flush=True)


if __name__ == "__main__":
    main()
