#!/usr/bin/env python3
"""Train the continuity switch: A and B targets come from one continuous utterance."""

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForMultimodalLM, AutoProcessor

sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE
from audio_trim import trim_codes
from switch_train import switch_example

MODEL_ID = "models/LFM3-Audio-2B-hf"
ROOT = Path("data/probe/mouth")
CT = ROOT / "continuity"
EPOCHS = 2
ACCUM = 4
LR = 1e-4
SEED = 0


def prompt_ids(proc):
    conv = [
        {"role": "system", "content": "Respond with interleaved text and audio."},
        {"role": "user", "content": "Read the following sentence exactly."},
    ]
    return proc.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )["input_ids"][0]


def build(proc, cfg, row):
    raw = np.load(CT / "pairs" / f"{row['id']}.npy")  # (12, N)
    cut = min(row["cut_frames"], raw.shape[1] - 5)
    a, b = raw[:, :cut], raw[:, cut:]
    eos = np.full((12, 1), cfg.audio_eos_code, dtype=np.int16)
    place = torch.from_numpy(np.concatenate([a, b, b[:, -1:], eos], axis=1)).long().t()
    labels = torch.cat([
        torch.full((a.shape[1], 12), -100, dtype=torch.long),
        torch.from_numpy(b).long().t(),
        torch.full((1, 12), cfg.audio_eos_code, dtype=torch.long),
        torch.full((1, 12), cfg.audio_eos_code, dtype=torch.long),
    ])
    ids = torch.cat([
        prompt_ids(proc),
        torch.tensor([cfg.mixed_start_token_id]),
        torch.tensor(proc.tokenizer.encode(row["lead_text"], add_special_tokens=False)),
        torch.tensor([cfg.text_end_token_id]),
        torch.full((a.shape[1],), cfg.audio_out_token_id),
        torch.tensor(proc.tokenizer.encode(row["b_text"], add_special_tokens=False)),
        torch.tensor([cfg.text_end_token_id]),
        torch.full((b.shape[1] + 2,), cfg.audio_out_token_id),
    ])
    return ids, place, labels


def retain(proc, cfg, row):
    raw = np.load(ROOT / "codes" / f"{row['id']}.npy")
    raw = trim_codes(raw, ROOT / "wav" / f"{row['id']}.wav")
    b = torch.from_numpy(raw).long().t()
    eos = torch.full((1, 12), cfg.audio_eos_code, dtype=torch.long)
    place = torch.cat([b, b[-1:], eos])
    labels = torch.cat([b, eos, eos])
    ids = torch.cat([
        prompt_ids(proc),
        torch.tensor([cfg.mixed_start_token_id]),
        torch.tensor(proc.tokenizer.encode(row["text"], add_special_tokens=False)),
        torch.tensor([cfg.text_end_token_id]),
        torch.full((place.shape[0],), cfg.audio_out_token_id),
    ])
    return ids, place, labels


def main() -> None:
    random.seed(SEED); torch.manual_seed(SEED)
    meta = json.load(open(CT / "meta.json"))
    train = [m for m in meta if m["split"] == "train"]
    hold = [m for m in meta if m["split"] == "holdout"]
    base_rows = [p for p in json.load(open(ROOT / "train_pairs.json")) if p["split"] == "train"]
    retention = random.sample(base_rows, 50)
    print(f"train={len(train)} hold={len(hold)} retain={len(retention)}", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    base = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    base.config.use_cache = False
    mouth = PeftModel.from_pretrained(base, ROOT / "lora").merge_and_unload()
    model = PeftModel.from_pretrained(mouth, ROOT / "switch" / "lora").merge_and_unload()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)",
    ))
    model.print_trainable_parameters()

    switch_rows = [json.loads(l) for l in (ROOT / "switch" / "pairs.jsonl").read_text().splitlines()]
    switch_rows = [r for r in switch_rows if r.get("split") == "train"][:120]
    examples = [("ct", r) for r in train] + [("sw", r) for r in switch_rows] + [("retain", r) for r in retention]

    def loss_of(kind, row):
        fn = {"ct": build, "sw": switch_example, "retain": retain}[kind]
        ids, place, labels = fn(proc, model.config, row)
        out = model(
            input_ids=ids[None].to(DEVICE),
            attention_mask=torch.ones_like(ids)[None].to(DEVICE),
            audio_out_codes=place.to(DEVICE),
            audio_labels=labels.to(DEVICE),
            use_cache=False, return_dict=True,
        )
        return out.loss

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    steps = math.ceil(len(examples) / ACCUM) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1, anneal_strategy="cos")
    t0 = time.perf_counter(); step = 0
    for epoch in range(EPOCHS):
        random.shuffle(examples); opt.zero_grad()
        for i in range(0, len(examples), ACCUM):
            chunk = examples[i:i + ACCUM]
            loss = sum(loss_of(k, r) for k, r in chunk) / len(chunk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 10 == 0:
                print({"step": step, "loss": round(float(loss.detach()), 4), "s": round(time.perf_counter() - t0, 1)}, flush=True)

    model.eval()
    with torch.no_grad():
        hold_loss = sum(float(loss_of("ct", r)) for r in hold) / len(hold)
        train_loss = sum(float(loss_of("ct", r)) for r in train[:40]) / min(40, len(train))
    result = {"train_n": len(train), "hold_n": len(hold), "retain_n": len(retention),
              "train_loss": round(train_loss, 4), "holdout_loss": round(hold_loss, 4),
              "steps": step, "wall_s": round(time.perf_counter() - t0, 1),
              "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    (CT / "lora_v2").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CT / "lora_v2")
    (CT / "train_report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
