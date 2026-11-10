#!/usr/bin/env python3
"""Block-synchronous text injection probe for LFM3-Omni-2B.

Interleaved generation is 6 text slots followed by 12 speech frames
(80 ms each = 960 ms). This script forces a target clause into the text
slots and lets the model voice it in the following frames, without
retraining and without letting the model reword the clause.

It also renders the same clause as plain TTS, which is the fidelity
ceiling for the same checkpoint.

Outputs data/probe/inject.wav and data/probe/tts.wav plus a JSON summary.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, DynamicCache

MODEL_ID = "models/LFM3-Omni-2B-hf"
OUT = Path("data/probe")
DEVICE = "cuda"
N_TEXT = 6
N_AUDIO = 12
SR = 24_000

TARGET = "Doctor Okafor approved three point two million pounds for the Harrow project."


def load():
    t0 = time.perf_counter()
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).eval().to(DEVICE)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    # The codec ships in bf16; Turing has no bf16 kernels.
    codec = proc.audio_tokenizer.to(device=DEVICE, dtype=torch.float32).eval()
    return model, proc, codec, time.perf_counter() - t0


def prompt_ids(proc, system: str, user: str) -> torch.LongTensor:
    conv = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    inputs = proc.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    return inputs["input_ids"].to(DEVICE)


@torch.no_grad()
def force_clause(model, proc, codec, target: str):
    """Prefill, force <|mixed_start|>, then alternate forced text / sampled speech."""
    ids = prompt_ids(proc, "Respond with interleaved text and audio.", "Give me the status update.")
    cfg = model.config.text_config
    out = model.model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        use_cache=True,
        past_key_values=DynamicCache(config=cfg),
    )
    hidden = out.last_hidden_state[:, -1]
    cache = out.past_key_values
    mask = torch.ones((1, ids.shape[1]), dtype=torch.long, device=DEVICE)
    embed = model.get_input_embeddings()

    def feed_embed(e):
        nonlocal hidden, cache, mask
        if e.ndim == 2:
            e = e.unsqueeze(1)
        mask = torch.cat((mask, mask.new_ones((1, 1))), dim=1)
        pos = torch.tensor([[cache.get_seq_length()]], device=DEVICE)
        step = model.model.language_model(
            inputs_embeds=e,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        )
        hidden, cache = step.last_hidden_state[:, -1], step.past_key_values

    def feed_token(tok: int):
        feed_embed(embed(torch.tensor([[tok]], device=DEVICE)))

    tokens = proc.tokenizer.encode(target, add_special_tokens=False)
    t_start = time.perf_counter()
    feed_token(model.config.mixed_start_token_id)
    t_marker = time.perf_counter()

    frames: list[torch.Tensor] = []
    text_steps = 0
    audio_steps = 0

    def speak(n_frames: int):
        nonlocal audio_steps
        for _ in range(n_frames):
            history = torch.stack(frames[-64:]) if frames else None
            codes = model.sample_audio_frame(
                hidden[0],
                temperature=0.4,
                repetition_penalty=1.2,
                history=history,
            )
            frames.append(codes)
            audio_steps += 1
            feed_embed(model.model.get_audio_out_embeds(codes[None]))

    for start in range(0, len(tokens), N_TEXT):
        group = tokens[start:start + N_TEXT]
        for tok in group:
            feed_token(tok)
            text_steps += 1
        speak(N_AUDIO)
    t_end = time.perf_counter()

    codes = torch.stack(frames, dim=-1)  # (num_codebooks, num_frames)
    wav = codec.decode(codes[None]).audio_values[0].float().cpu().numpy().squeeze()
    sf.write(OUT / "inject.wav", wav, SR)

    return {
        "target": target,
        "tokens": len(tokens),
        "text_steps": text_steps,
        "audio_frames": audio_steps,
        "audio_s": round(audio_steps * 0.080, 3),
        "tts_marker_s": round(t_marker - t_start, 3),
        "gen_s": round(t_end - t_marker, 3),
        "text_tok_per_s": round(text_steps / max(t_end - t_marker, 1e-9), 2),
        "audio_frames_per_s": round(audio_steps / max(t_end - t_marker, 1e-9), 2),
        "realtime_factor": round((audio_steps * 0.080) / max(t_end - t_marker, 1e-9), 3),
        "wav_s": round(len(wav) / SR, 3),
    }


@torch.no_grad()
def tts_baseline(model, proc, codec, target: str):
    ids = prompt_ids(proc, "Perform TTS in American English using a female voice.", target)
    t0 = time.perf_counter()
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=512,
        generation_mode="audio",
        audio_temperature=0.4,
        audio_repetition_penalty=1.2,
    )
    n = int(out.audio_lengths[0])
    wav = codec.decode(out.audio_codes[:1, :, :n]).audio_values[0].float().cpu().numpy().squeeze()
    sf.write(OUT / "tts.wav", wav, SR)
    dt = time.perf_counter() - t0
    return {
        "target": target,
        "audio_frames": n,
        "gen_s": round(dt, 3),
        "wav_s": round(len(wav) / SR, 3),
        "realtime_factor": round((n * 0.080) / max(dt, 1e-9), 3),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    model, proc, codec, load_s = load()

    tts = tts_baseline(model, proc, codec, TARGET)
    inject = force_clause(model, proc, codec, TARGET)

    peak = torch.cuda.max_memory_allocated() / 1e9
    summary = {
        "model": MODEL_ID,
        "dtype": "float16",
        "load_s": round(load_s, 1),
        "peak_vram_gb": round(peak, 2),
        "tts": tts,
        "inject": inject,
    }
    (OUT / "inject.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
