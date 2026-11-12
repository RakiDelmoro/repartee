#!/usr/bin/env python3
"""Mid-utterance injection: speak A, then force B, then continue."""

import json
import sys
from pathlib import Path

import soundfile as sf
import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor, DynamicCache

sys.path.insert(0, ".")
sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE

MODEL_ID = "models/LFM3-Audio-2B-hf"
ROOT = Path("data/probe/mouth")
OUT = ROOT / "listen" / "mid"
SR = 24_000
MAX_FRAMES = 250
SWITCH = 15  # 1.2 s into A
A = "Doctor Okafor approved the budget for the new project."
B = "The meeting moved to two in the afternoon."


@torch.no_grad()
def speak_with_switch(model, proc, codec, switch_frames: int):
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
        torch.tensor(proc.tokenizer.encode(A, add_special_tokens=False)),
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

    frames, switch_at = [], None
    for i in range(MAX_FRAMES):
        if i == switch_frames:
            switch_at = i
            for t in proc.tokenizer.encode(B, add_special_tokens=False):
                feed(model.get_input_embeddings()(torch.tensor([[t]], device=DEVICE)))
            feed(model.get_input_embeddings()(torch.tensor([[model.config.text_end_token_id]], device=DEVICE)))
        history = torch.stack(frames[-64:]) if frames else None
        codes = model.sample_audio_frame(hidden[0], temperature=0.4, repetition_penalty=1.2, history=history)
        if bool((codes == model.config.audio_eos_code).any()):
            break
        frames.append(codes)
        feed(model.model.get_audio_out_embeds(codes[None]))
    codes = torch.stack(frames).t()
    wav = codec.decode(codes[None]).audio_values[0].float().cpu().numpy().squeeze()
    return wav, switch_at, len(frames)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    codec = proc.audio_tokenizer.to(device=DEVICE, dtype=torch.float32).eval()
    base = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()

    wav, at, n = speak_with_switch(base, proc, codec, SWITCH)
    sf.write(OUT / "base_mid.wav", wav, SR)
    print("base", round(len(wav) / SR, 2), "switch_at", at, flush=True)

    model = PeftModel.from_pretrained(base, ROOT / "lora").merge_and_unload().eval()
    wav, at, n = speak_with_switch(model, proc, codec, SWITCH)
    sf.write(OUT / "adapter_mid.wav", wav, SR)
    print("adapter", round(len(wav) / SR, 2), "switch_at", at, flush=True)

    del model, base
    torch.cuda.empty_cache()

    from app.asr import QwenTranscriber
    t = QwenTranscriber()
    asr = {name: {"s": round(sf.info(OUT / f"{name}_mid.wav").duration, 2),
                  "hyp": t.transcribe(OUT / f"{name}_mid.wav").text}
           for name in ("base", "adapter")}
    (OUT / "asr.json").write_text(json.dumps({"A": A, "B": B, "switch_s": SWITCH * 0.08, "asr": asr}, indent=2))
    print(json.dumps({"A": A, "B": B, "switch_s": SWITCH * 0.08, "asr": asr}, indent=2))


if __name__ == "__main__":
    main()
