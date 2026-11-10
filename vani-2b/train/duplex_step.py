#!/usr/bin/env python3
"""One duplex training step: control-token mode loss + depthformer audio loss."""

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForMultimodalLM, AutoProcessor

sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE

MODEL_ID = "models/LFM3-Omni-2B-hf"
CONV = Path("data/open-yap/pilot/conversations/conv_cd9656663384")
LABELS = Path("data/open-yap/pilot/labels/conv_cd9656663384.npz")
SECONDS = 5.0
START = 25.0


def main() -> None:
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model.config.use_cache = False
    tok = proc.tokenizer
    tok.add_special_tokens({"additional_special_tokens": ["<|listen|>"]})
    model.resize_token_embeddings(len(tok))
    listen_id = tok.convert_tokens_to_ids("<|listen|>")
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)",
    ))
    base = model.base_model.model

    u, _ = librosa.load(CONV / "speaker_b.flac", sr=16_000, mono=True, offset=START, duration=SECONDS)
    a, _ = librosa.load(CONV / "speaker_a.flac", sr=24_000, mono=True, offset=START, duration=SECONDS)
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
    labels = np.load(LABELS)
    start_f = int(START / 0.08)
    speak = torch.tensor(labels["a_speak"][start_f:start_f + n], device=DEVICE)
    print("frames", n, "speak", int(speak.sum()), "listen", int((~speak).sum()), flush=True)

    a_emb = base.model.get_audio_out_embeds(codes_t).to(u_emb.dtype)
    listen_emb = base.get_input_embeddings()(torch.tensor([listen_id], device=DEVICE)).expand(n, -1)
    a_in = torch.where(speak[:, None], a_emb, listen_emb)
    seq = torch.stack([u_emb, a_in], dim=1).reshape(1, 2 * n, u_emb.shape[-1])

    out = base.model.language_model(
        inputs_embeds=seq,
        attention_mask=torch.ones((1, 2 * n), dtype=torch.long, device=DEVICE),
        use_cache=False,
    )
    h_u = out.last_hidden_state[0, 0::2]  # predicts A_t
    mode = torch.where(speak, torch.full_like(speak, base.config.audio_out_token_id, dtype=torch.long),
                       torch.full_like(speak, listen_id, dtype=torch.long))
    mode_logits = base.lm_head(h_u)
    mode_loss = F.cross_entropy(mode_logits.float(), mode)

    if speak.any():
        audio_h = h_u[speak]
        audio_codes = codes_t[speak]
        masked = torch.rand(audio_codes.shape, device=DEVICE) < 0.5
        masked |= ~masked.any(-1, keepdim=True)
        audio_logits = base.model.depth_decoder(audio_h, audio_codes, masked)
        audio_loss = F.cross_entropy(audio_logits.transpose(1, 2).float(), audio_codes)
    else:
        audio_loss = h_u.sum() * 0.0

    loss = mode_loss + audio_loss
    print("mode", round(float(mode_loss.detach()), 4), "audio", round(float(audio_loss.detach()), 4),
          "total", round(float(loss.detach()), 4), flush=True)
    loss.backward()
    print("backward ok; peak", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)


if __name__ == "__main__":
    main()
