#!/usr/bin/env python3
"""Mouth LoRA: force text, predict the teacher's codec frames plus EOS."""

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForMultimodalLM, AutoProcessor

sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE
from audio_trim import trim_codes

MODEL_ID = "models/LFM3-Omni-2B-hf"
ROOT = Path("data/probe/mouth")
ADAPTER = ROOT / "lora"
EPOCHS = 2
ACCUM = 4
LR = 2e-4
SEED = 0


def build(proc, row, cfg):
    raw = np.load(ROOT / "codes" / f"{row['id']}.npy")
    raw = trim_codes(raw, ROOT / "wav" / f"{row['id']}.wav")
    codes = torch.from_numpy(raw).long().t()  # (N, 12)
    eos = torch.full((1, 12), cfg.audio_eos_code, dtype=torch.long)
    # one placeholder per teacher frame, then one to predict EOS, then an unused context frame
    place = torch.cat([codes, codes[-1:], eos])
    labels = torch.cat([codes, eos, eos])
    conv = [
        {"role": "system", "content": "Respond with interleaved text and audio."},
        {"role": "user", "content": "Read the following sentence exactly."},
    ]
    prompt = proc.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )["input_ids"][0]
    text = torch.tensor(proc.tokenizer.encode(row["text"], add_special_tokens=False))
    ids = torch.cat([
        prompt,
        torch.tensor([cfg.mixed_start_token_id]),
        text,
        torch.tensor([cfg.text_end_token_id]),
        torch.full((place.shape[0],), cfg.audio_out_token_id),
    ])
    return ids, place, labels


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    pairs = json.load(open(ROOT / "train_pairs.json"))
    train = [p for p in pairs if p["split"] == "train"]
    holdout = [p for p in pairs if p["split"] == "holdout"]
    print(f"train={len(train)} holdout={len(holdout)}", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)",
    ))
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / ACCUM)
    total = steps_per_epoch * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=total, pct_start=min(0.1, 20 / total), anneal_strategy="cos"
    )

    def batch_loss(rows):
        model.train()
        ids, places, labels = zip(*(build(proc, r, model.config) for r in rows))
        # batch 1 keeps the variable-length frame counts simple
        loss = 0.0
        for i, p, l in zip(ids, places, labels):
            out = model(
                input_ids=i[None].to(DEVICE),
                attention_mask=torch.ones_like(i)[None].to(DEVICE),
                audio_out_codes=p.to(DEVICE),
                audio_labels=l.to(DEVICE),
                use_cache=False,
                return_dict=True,
            )
            loss = loss + out.loss / len(rows)
        return loss

    t0 = time.perf_counter()
    step = 0
    log = []
    for epoch in range(EPOCHS):
        random.shuffle(train)
        opt.zero_grad()
        for i in range(0, len(train), ACCUM):
            chunk = train[i:i + ACCUM]
            loss = batch_loss(chunk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0:
                rec = {"step": step, "epoch": epoch, "loss": round(float(loss.detach()), 4),
                       "lr": round(sched.get_last_lr()[0], 7),
                       "elapsed": round(time.perf_counter() - t0, 1)}
                log.append(rec)
                print(rec, flush=True)

    model.eval()
    def eval_loss(rows):
        vals = []
        with torch.no_grad():
            for r in rows:
                ids, places, labels = build(proc, r, model.config)
                out = model(
                    input_ids=ids[None].to(DEVICE),
                    attention_mask=torch.ones_like(ids)[None].to(DEVICE),
                    audio_out_codes=places.to(DEVICE),
                    audio_labels=labels.to(DEVICE),
                    use_cache=False,
                    return_dict=True,
                )
                vals.append(float(out.loss))
        return sum(vals) / len(vals)

    result = {
        "train_n": len(train), "holdout_n": len(holdout),
        "train_loss": round(eval_loss(train[:80]), 4),
        "holdout_loss": round(eval_loss(holdout), 4),
        "steps": step, "wall_s": round(time.perf_counter() - t0, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "log": log,
    }
    ADAPTER.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER)
    (ROOT / "train_report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "log"}, indent=2))
    print("adapter:", ADAPTER)


if __name__ == "__main__":
    main()
