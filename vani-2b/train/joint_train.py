#!/usr/bin/env python3
"""Joint pilot: duplex + injection + mouth in one adapter."""

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForMultimodalLM, AutoProcessor

sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE
from continuity_train import build as ct_build
from mouth_train import build as mouth_build

MODEL_ID = "models/LFM3-Omni-2B-hf"
ROOT = Path("data/probe/mouth")
CT = ROOT / "continuity"
DUP = Path("data/probe/duplex")
JOINT = DUP / "joint"
EPOCHS = 2
CYCLE = 10  # 5 duplex, 3 injection, 2 mouth
LR = 1e-4
SEED = 0


def main() -> None:
    random.seed(SEED); torch.manual_seed(SEED)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model.config.use_cache = False
    mouth = PeftModel.from_pretrained(model, ROOT / "lora").merge_and_unload()
    base = PeftModel.from_pretrained(mouth, CT / "lora_v2").merge_and_unload()
    model = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)",
    ))
    base = model.base_model.model
    mode_head = nn.Linear(base.config.text_config.hidden_size, 2).to(DEVICE)
    model.print_trainable_parameters()
    tok = proc.tokenizer
    listen_id = tok.encode(" listen", add_special_tokens=False)[0]
    bos_id = base.config.text_config.bos_token_id
    print("listen", listen_id, tok.convert_ids_to_tokens(listen_id), "bos", bos_id, flush=True)

    dup_rows = sorted((DUP / "cache_turns").glob("*.npz"))
    ct_rows = [m for m in json.load(open(CT / "meta.json")) if m["split"] == "train"]
    mouth_rows = [p for p in json.load(open(ROOT / "train_pairs.json")) if p["split"] == "train"]
    dup_rows = dup_rows[:250]
    print(f"duplex={len(dup_rows)} ct={len(ct_rows)} mouth={len(mouth_rows)}", flush=True)

    def duplex_loss(path):
        d = np.load(path)
        u = torch.from_numpy(d["u"]).to(DEVICE).half()
        codes = torch.from_numpy(d["codes"]).long().to(DEVICE)
        speak = torch.from_numpy(d["speak"]).bool().to(DEVICE)
        n = min(len(u), len(codes), len(speak))
        u, codes, speak = u[:n], codes[:n], speak[:n]
        a_emb = base.model.get_audio_out_embeds(codes)
        listen = base.get_input_embeddings()(torch.tensor([listen_id], device=DEVICE)).expand(n, -1).half()
        a_in = torch.where(speak[:, None], a_emb, listen)
        bos = base.get_input_embeddings()(torch.tensor([bos_id], device=DEVICE)).half()
        seq = torch.cat([bos.view(1, -1), torch.stack([u, a_in], dim=1).reshape(2 * n, -1)], dim=0)[None]
        out = base.model.language_model(
            inputs_embeds=seq,
            attention_mask=torch.ones((1, 1 + 2 * n), dtype=torch.long, device=DEVICE),
            use_cache=False,
        )
        h_u = out.last_hidden_state[0, 1::2][:n]
        mode_loss = F.cross_entropy(mode_head(h_u.float()), speak.long())
        if speak.any():
            audio_h = h_u[speak]; audio_codes = codes[speak]
            masked = torch.rand(audio_codes.shape, device=DEVICE) < 0.5
            masked |= ~masked.any(-1, keepdim=True)
            logits = base.model.depth_decoder(audio_h, audio_codes, masked)
            audio_loss = F.cross_entropy(logits.transpose(1, 2).float(), audio_codes)
        else:
            audio_loss = h_u.sum() * 0.0
        return mode_loss + audio_loss, float(mode_loss), float(audio_loss)

    def text_loss(kind, row):
        if kind == "ct":
            ids, place, labels = ct_build(proc, base.config, row)
        else:
            ids, place, labels = mouth_build(proc, row, base.config)
        out = model(
            input_ids=ids[None].to(DEVICE),
            attention_mask=torch.ones_like(ids)[None].to(DEVICE),
            audio_out_codes=place.to(DEVICE),
            audio_labels=labels.to(DEVICE),
            use_cache=False, return_dict=True,
        )
        return out.loss

    params = [p for p in model.parameters() if p.requires_grad] + list(mode_head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    total = CYCLE * 20 * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total, pct_start=0.1, anneal_strategy="cos")
    t0 = time.perf_counter(); step = 0
    for epoch in range(EPOCHS):
        for _ in range(20):
            kinds = ["d"] * 5 + ["c"] * 3 + ["m"] * 2
            random.shuffle(kinds)
            for kind in kinds:
                opt.zero_grad()
                if kind == "d":
                    loss, ml, al = duplex_loss(random.choice(dup_rows))
                elif kind == "c":
                    loss = text_loss("ct", random.choice(ct_rows)); ml = al = 0.0
                else:
                    loss = text_loss("mouth", random.choice(mouth_rows)); ml = al = 0.0
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); step += 1
                if step % 20 == 0:
                    print({"step": step, "kind": kind, "loss": round(float(loss.detach()), 3),
                           "mode": round(ml, 2), "audio": round(al, 2), "s": round(time.perf_counter() - t0, 1)}, flush=True)
    JOINT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(JOINT / "lora")
    torch.save(mode_head.state_dict(), JOINT / "mode_head.pt")
    print("saved", JOINT / "lora", "steps", step, "peak", round(torch.cuda.max_memory_allocated() / 1e9, 2))


if __name__ == "__main__":
    main()
