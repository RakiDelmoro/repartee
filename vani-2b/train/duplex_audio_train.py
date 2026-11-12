#!/usr/bin/env python3
"""Pure audio-only duplex trainer with curriculum and scheduled sampling.

A_t is always a codec frame. Silence frames are listening; speech frames are
talking. No control tokens, no mode head, no text path.

Env:
  MODE=speech|listen|all   filter windows
  INIT=<adapter dir>       start from an existing adapter
  OUT=<adapter dir>        output
  EPOCHS, MIX, MASK_EOS, LR
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForMultimodalLM, AutoProcessor

sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE

MODEL_ID = "models/LFM3-Omni-2B-hf"
ROOT = Path("data/probe/mouth")
CT = ROOT / "continuity"
DUP = Path("data/probe/duplex")
CACHE = DUP / "cache_turns"
MODE = os.environ.get("MODE", "all")
OUT = DUP / os.environ.get("OUT", "audio_lora")
INIT = os.environ.get("INIT", "")
EPOCHS = int(os.environ.get("EPOCHS", "6"))
MIX = os.environ.get("MIX", "1") == "1"
MASK_EOS = os.environ.get("MASK_EOS", "0") == "1"
REPEAT_SPEECH = int(os.environ.get("REPEAT_SPEECH", "0"))
OVERFIT = int(os.environ.get("OVERFIT", "0"))
TRAIN_DEPTH = os.environ.get("TRAIN_DEPTH", "0") == "1"
DEPTH_LORA = os.environ.get("DEPTH_LORA", "0") == "1"
DEPTH_LR = float(os.environ.get("DEPTH_LR", "1e-5"))
LR = float(os.environ.get("LR", "1e-4"))
SEED = 0


def keep(path: Path) -> bool:
    d = np.load(path)
    s = int(d["speak"].sum()); b = int(d["b_speak"].sum())
    if MODE == "speech":
        return s >= 45
    if MODE == "listen":
        return (s <= 10 and b >= 30) or s >= 45
    return True


def main() -> None:
    random.seed(SEED); torch.manual_seed(SEED)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model.config.use_cache = False
    mouth = PeftModel.from_pretrained(model, ROOT / "lora").merge_and_unload()
    sw = PeftModel.from_pretrained(mouth, ROOT / "switch" / "lora").merge_and_unload()
    cont = PeftModel.from_pretrained(sw, CT / "lora_v2").merge_and_unload()
    if INIT:
        cont = PeftModel.from_pretrained(cont, Path(INIT)).merge_and_unload()
    target = r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)"
    if DEPTH_LORA:
        target += r"|model\.depth_decoder\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3|depth_linear|logits)"
    model = get_peft_model(cont, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=target,
    ))
    base = model.base_model.model
    if TRAIN_DEPTH:
        for p in base.model.depth_decoder.parameters():
            p.requires_grad = True
        n_depth = sum(p.numel() for p in base.model.depth_decoder.parameters())
        print({"train_depth": True, "depth_params": n_depth}, flush=True)
    eos = int(base.config.audio_eos_code)
    embed = base.get_input_embeddings()
    bos = embed(torch.tensor([[base.config.text_config.bos_token_id]], device=DEVICE))[0, 0]
    depth_ids = {id(p) for p in base.model.depth_decoder.parameters()} if TRAIN_DEPTH else set()
    lora_params = [p for p in model.parameters() if p.requires_grad and id(p) not in depth_ids]
    depth_params = [p for p in model.parameters() if p.requires_grad and id(p) in depth_ids]
    groups = [{"params": lora_params, "lr": LR}]
    if depth_params:
        groups.append({"params": depth_params, "lr": DEPTH_LR})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    files = [p for p in sorted(CACHE.glob("*.npz")) if keep(p)]
    if REPEAT_SPEECH:
        heavy = [p for p in files if int(np.load(p)["speak"].sum()) >= 45]
        files += heavy * REPEAT_SPEECH
    if OVERFIT:
        speech = [p for p in files if int(np.load(p)["speak"].sum()) >= 45][:OVERFIT]
        user = [p for p in files if int(np.load(p)["b_speak"].sum()) >= 30 and int(np.load(p)["speak"].sum()) <= 10][:OVERFIT]
        files = speech if MODE == "speech" else speech + user
    print({"mode": MODE, "init": INIT or None, "windows": len(files), "mask_eos": MASK_EOS, "mix": MIX, "repeat_speech": REPEAT_SPEECH}, flush=True)
    for epoch in range(EPOCHS):
        random.shuffle(files)
        mix_p = 0.10 + 0.25 * epoch / max(EPOCHS - 1, 1) if MIX else 0.0
        total_loss = 0.0; n_steps = 0
        for path in files:
            d = np.load(path)
            u = torch.from_numpy(d["u"]).to(DEVICE).half()
            codes = torch.from_numpy(d["codes"]).long().to(DEVICE)
            speak = torch.from_numpy(d["speak"]).bool().to(DEVICE)
            n = min(len(u), len(codes))
            u, codes, speak = u[:n], codes[:n], speak[:n]
            with torch.no_grad():
                a_emb = base.model.get_audio_out_embeds(codes)
                seq = torch.cat([bos.view(1, -1), torch.stack([u, a_emb], dim=1).reshape(2 * n, -1)], 0)[None]
                out = base.model.language_model(
                    inputs_embeds=seq,
                    attention_mask=torch.ones((1, 1 + 2 * n), dtype=torch.long, device=DEVICE),
                    use_cache=False,
                )
                h_u = out.last_hidden_state[0, 1::2][:n]
                mixed = codes
                if MIX:
                    masked_all = torch.ones_like(codes, dtype=torch.bool)
                    logits_all = base.model.depth_decoder(h_u, codes, masked_all)
                    probs = (logits_all.float() / 0.8).softmax(-1)
                    sampled = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape_as(codes)
                    mix = (torch.rand((n, 1), device=DEVICE) < mix_p).expand(-1, codes.shape[-1])
                    mixed = torch.where(mix, sampled, codes)
                a_mixed = base.model.get_audio_out_embeds(mixed)
            seq = torch.cat([bos.view(1, -1), torch.stack([u, a_mixed], dim=1).reshape(2 * n, -1)], 0)[None]
            out = base.model.language_model(
                inputs_embeds=seq,
                attention_mask=torch.ones((1, 1 + 2 * n), dtype=torch.long, device=DEVICE),
                use_cache=False,
            )
            h_u = out.last_hidden_state[0, 1::2][:n]
            masked = torch.rand(codes.shape, device=DEVICE) < 0.5
            masked |= ~masked.any(-1, keepdim=True)
            logits = base.model.depth_decoder(h_u, mixed, masked)
            if MASK_EOS:
                logits = logits.clone()
                logits[..., eos] = -1e4
            ce = F.cross_entropy(logits.transpose(1, 2).float(), codes, reduction="none")
            w = torch.where(speak, 1.0, 0.3)
            loss = (ce.mean(-1) * w).mean()
            if not torch.isfinite(loss):
                opt.zero_grad(); print("skip non-finite", flush=True); continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            total_loss += float(loss.detach()); n_steps += 1
        print({"epoch": epoch, "loss": round(total_loss / max(n_steps, 1), 3), "mix": round(mix_p, 2)}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
