#!/usr/bin/env python3
"""Build a listening set: teacher TTS vs fine-tuned injection vs base injection."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, DynamicCache

sys.path.insert(0, ".")
sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE

MODEL_ID = "models/LFM3-Omni-2B-hf"
ROOT = Path("data/probe/mouth")
OUT = ROOT / "listen"
SR = 24_000
MAX_FRAMES = 200
SELECTED = [
    ("m0014", "01_question"),
    ("m0033", "02_name"),
    ("m0058", "03_name_date"),
    ("m0103", "04_date"),
]


@torch.no_grad()
def synth(model, proc, codec, text: str):
    conv = [
        {"role": "system", "content": "Respond with interleaved text and audio."},
        {"role": "user", "content": "Read the following sentence exactly."},
    ]
    prompt = proc.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )["input_ids"][0]
    ids = torch.cat([
        prompt,
        torch.tensor([model.config.mixed_start_token_id]),
        torch.tensor(proc.tokenizer.encode(text, add_special_tokens=False)),
        torch.tensor([model.config.text_end_token_id]),
    ]).to(DEVICE)[None]
    out = model.model(
        input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True,
        past_key_values=DynamicCache(config=model.config.text_config),
    )
    hidden, cache = out.last_hidden_state[:, -1], out.past_key_values
    mask = torch.ones_like(ids)

    def feed(e):
        nonlocal hidden, cache, mask
        if e.ndim == 2:
            e = e.unsqueeze(1)
        mask = torch.cat((mask, mask.new_ones((1, 1))), dim=1)
        step = model.model.language_model(
            inputs_embeds=e, attention_mask=mask,
            position_ids=torch.tensor([[cache.get_seq_length()]], device=DEVICE),
            past_key_values=cache, use_cache=True,
        )
        hidden, cache = step.last_hidden_state[:, -1], step.past_key_values

    frames = []
    for _ in range(MAX_FRAMES):
        history = torch.stack(frames[-64:]) if frames else None
        codes = model.sample_audio_frame(hidden[0], temperature=0.4, repetition_penalty=1.2, history=history)
        if bool((codes == model.config.audio_eos_code).any()):
            break
        frames.append(codes)
        feed(model.model.get_audio_out_embeds(codes[None]))
    codes = torch.stack(frames).t()
    wav = codec.decode(codes[None]).audio_values[0].float().cpu().numpy().squeeze()
    return wav


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = {p["id"]: p for p in json.load(open(ROOT / "train_pairs.json"))}
    base = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    codec = proc.audio_tokenizer.to(device=DEVICE, dtype=torch.float32).eval()

    for pid, label in SELECTED:
        wav = synth(base, proc, codec, pairs[pid]["text"])
        sf.write(OUT / f"{label}_base.wav", wav, SR)
        shutil.copy(ROOT / "wav" / f"{pid}.wav", OUT / f"{label}_teacher.wav")
        shutil.copy(ROOT / "eval" / f"{pid}.wav", OUT / f"{label}_adapter.wav")
        print("wrote", label, pairs[pid]["text"], flush=True)

    del base
    torch.cuda.empty_cache()

    from app.asr import QwenTranscriber
    t = QwenTranscriber()
    asr = {}
    for pid, label in SELECTED:
        asr[label] = {
            "text": pairs[pid]["text"],
            "teacher": sf.info(OUT / f"{label}_teacher.wav").duration,
            "adapter": t.transcribe(OUT / f"{label}_adapter.wav").text,
            "base": t.transcribe(OUT / f"{label}_base.wav").text,
        }
    (OUT / "asr.json").write_text(json.dumps(asr, indent=2))
    print(json.dumps(asr, indent=2))


if __name__ == "__main__":
    main()
