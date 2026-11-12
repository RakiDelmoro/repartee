#!/usr/bin/env python3
"""No hardcoded filler: the model improvises a hold, then the answer is injected."""

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
SW = ROOT / "switch"
OUT = SW / "improvise"
SR = 24_000
TEXT = "The meeting moved to two in the afternoon."
BLOCKS = 2  # ~2 s of self-generated filler


@torch.no_grad()
def run(model, proc, codec):
    conv = [
        {"role": "system", "content": "Respond with interleaved text and audio."},
        {"role": "user", "content": "The user asked a complex question. You need a moment to think. Stall naturally for a few seconds. Do not answer yet."},
    ]
    prompt = proc.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )["input_ids"][0].to(DEVICE)
    out = model.model(
        input_ids=prompt[None], attention_mask=torch.ones_like(prompt)[None], use_cache=True,
        past_key_values=DynamicCache(config=model.config.text_config),
    )
    hidden, cache = out.last_hidden_state[:, -1], out.past_key_values
    mask = torch.ones_like(prompt)[None]

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

    embed = model.get_input_embeddings()
    suppress = [model.config.mixed_start_token_id, model.config.audio_start_token_id,
                model.config.text_end_token_id, model.config.text_config.eos_token_id,
                model.config.audio_out_token_id, model.config.audio_token_id]
    frames, text_tokens = [], []

    feed(embed(torch.tensor([[model.config.mixed_start_token_id]], device=DEVICE)))
    for _ in range(BLOCKS):
        for _ in range(6):
            logits = model.lm_head(hidden).float()
            logits[:, suppress] = -float("inf")
            token = int(logits.argmax(-1))
            text_tokens.append(token)
            feed(embed(torch.tensor([[token]], device=DEVICE)))
        for _ in range(12):
            history = torch.stack(frames[-64:]) if frames else None
            codes = model.sample_audio_frame(hidden[0], temperature=0.4, repetition_penalty=1.2, history=history)
            if bool((codes == model.config.audio_eos_code).any()):
                break
            frames.append(codes)
            feed(model.model.get_audio_out_embeds(codes[None]))
    filler_text = proc.decode(torch.tensor(text_tokens), skip_special_tokens=True)

    for t in proc.tokenizer.encode(TEXT, add_special_tokens=False):
        feed(embed(torch.tensor([[t]], device=DEVICE)))
    feed(embed(torch.tensor([[model.config.text_end_token_id]], device=DEVICE)))
    for _ in range(200):
        history = torch.stack(frames[-64:]) if frames else None
        codes = model.sample_audio_frame(hidden[0], temperature=0.4, repetition_penalty=1.2, history=history)
        if bool((codes == model.config.audio_eos_code).any()):
            break
        frames.append(codes)
        feed(model.model.get_audio_out_embeds(codes[None]))

    codes = torch.stack(frames).t()
    wav = codec.decode(codes[None]).audio_values[0].float().cpu().numpy().squeeze()
    return wav, filler_text


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    codec = proc.audio_tokenizer.to(device=DEVICE, dtype=torch.float32).eval()
    base = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()
    mouth = PeftModel.from_pretrained(base, ROOT / "lora").merge_and_unload().eval()
    model = PeftModel.from_pretrained(mouth, SW / "lora").merge_and_unload().eval()

    wav, filler = run(model, proc, codec)
    sf.write(OUT / "improvise.wav", wav, SR)
    json.dump({"answer": TEXT, "model_filler": filler, "s": round(len(wav) / SR, 2)},
              open(OUT / "improvise.json", "w"), indent=2)
    print("filler:", repr(filler))
    print("s:", round(len(wav) / SR, 2))

    del model, mouth, base
    torch.cuda.empty_cache()
    from app.asr import QwenTranscriber
    t = QwenTranscriber()
    print("asr:", repr(t.transcribe(OUT / "improvise.wav").text))


if __name__ == "__main__":
    main()
