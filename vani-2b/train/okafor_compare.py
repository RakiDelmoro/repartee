#!/usr/bin/env python3
"""Same sentence, same protocol: base injection vs mouth adapter vs teacher."""

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
from inject_probe import DEVICE
from mouth_listen import synth

MODEL_ID = "models/LFM3-Omni-2B-hf"
ROOT = Path("data/probe/mouth")
OUT = ROOT / "listen" / "okafor"
SR = 24_000
TEXT = "Doctor Okafor approved three point two million pounds for the Harrow project."


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    codec = proc.audio_tokenizer.to(device=DEVICE, dtype=torch.float32).eval()
    base = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()

    wav = synth(base, proc, codec, TEXT)
    sf.write(OUT / "base.wav", wav, SR)
    print("base done", round(len(wav) / SR, 2), flush=True)

    model = PeftModel.from_pretrained(base, ROOT / "lora").merge_and_unload().eval()
    wav = synth(model, proc, codec, TEXT)
    sf.write(OUT / "adapter.wav", wav, SR)
    print("adapter done", round(len(wav) / SR, 2), flush=True)

    shutil.copy("data/probe/injection/audio/01_tts_baseline_exact.wav", OUT / "teacher.wav")
    shutil.copy("data/probe/injection/audio/03_naive_stop_truncated.wav", OUT / "old_naive_truncated.wav")

    del model, base
    torch.cuda.empty_cache()

    from app.asr import QwenTranscriber
    t = QwenTranscriber()
    asr = {}
    for name in ("teacher", "base", "adapter", "old_naive_truncated"):
        asr[name] = {
            "s": round(sf.info(OUT / f"{name}.wav").duration, 2),
            "hyp": t.transcribe(OUT / f"{name}.wav").text,
        }
    (OUT / "asr.json").write_text(json.dumps(asr, indent=2))

    kinds = ["teacher", "base", "adapter"]
    random.Random(3).shuffle(kinds)
    key = {}
    for j, kind in enumerate(kinds):
        letter = "ABC"[j]
        shutil.copy(OUT / f"{kind}.wav", OUT / f"blind_{letter}.wav")
        key[letter] = kind
    (OUT / "blind_key.json").write_text(json.dumps(key, indent=2))
    print(json.dumps({"text": TEXT, "asr": asr, "blind": key}, indent=2))


if __name__ == "__main__":
    main()
