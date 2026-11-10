#!/usr/bin/env python3
"""Feasibility: one [U][A] duplex forward/backward with a manual audio loss."""

import sys
from pathlib import Path

import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForMultimodalLM, AutoProcessor

sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE

MODEL_ID = "models/LFM3-Omni-2B-hf"
CONV = Path("data/open-yap/pilot/conversations/conv_cd9656663384")
SECONDS = 5.0


def main() -> None:
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)",
    ))
    base = model.base_model.model
    print("trainable", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    u, _ = librosa.load(CONV / "speaker_b.flac", sr=16_000, mono=True, duration=SECONDS)
    a, _ = librosa.load(CONV / "speaker_a.flac", sr=24_000, mono=True, duration=SECONDS)
    sf.write("/tmp/duplex_u.wav", u, 16_000)
    inputs = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "audio", "audio": "/tmp/duplex_u.wav"}]}],
        add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt",
    )
    feats = inputs["input_features"].to(DEVICE)
    with torch.no_grad():
        u_emb, _ = base.model.get_audio_features_stream(feats, None, True)
        codec = proc.audio_tokenizer.to(device=DEVICE, dtype=torch.float32).eval()
        codes = codec.encode(torch.tensor(a)[None].to(DEVICE)).audio_codes

    n = min(u_emb.shape[1], codes.shape[-1])
    u_emb = u_emb[0, :n]
    codes_t = codes[0, :, :n].t().long()
    a_emb = base.model.get_audio_out_embeds(codes_t).to(u_emb.dtype)
    seq = torch.stack([u_emb, a_emb], dim=1).reshape(1, 2 * n, u_emb.shape[-1])
    print("U", u_emb.shape, "A", codes_t.shape, "seq", seq.shape, flush=True)

    out = base.model.language_model(
        inputs_embeds=seq,
        attention_mask=torch.ones((1, 2 * n), dtype=torch.long, device=DEVICE),
        use_cache=False,
    )
    hidden = out.last_hidden_state
    latents = hidden[0, 0::2]  # U_t predicts A_t
    masked = torch.ones_like(codes_t, dtype=torch.bool)
    logits = base.model.depth_decoder(latents, codes_t, masked)
    loss = F.cross_entropy(logits.transpose(1, 2).float(), codes_t)
    print("loss", round(float(loss.detach()), 4), "vram", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
    loss.backward()
    print("backward ok; peak", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)


if __name__ == "__main__":
    main()
