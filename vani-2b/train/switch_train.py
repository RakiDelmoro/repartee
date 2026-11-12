#!/usr/bin/env python3
"""Train the actor-switch LoRA on top of the merged mouth adapter."""

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

MODEL_ID = "models/LFM3-Audio-2B-hf"
ROOT = Path("data/probe/mouth")
SW = ROOT / "switch"
EPOCHS = 2
ACCUM = 4
LR = 1e-4
SEED = 0


def codes(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.load(path)).long().t()


def prompt_ids(proc) -> torch.Tensor:
    conv = [
        {"role": "system", "content": "Respond with interleaved text and audio."},
        {"role": "user", "content": "Read the following sentence exactly."},
    ]
    return proc.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )["input_ids"][0]


def switch_example(proc, cfg, row):
    a = codes(SW / "prefixes" / f"{row['prefix_id']}.npy")[: row["switch_frames"]]
    raw_b = np.load(ROOT / "codes" / f"{row['b_id']}.npy")
    b = torch.from_numpy(trim_codes(raw_b, ROOT / "wav" / f"{row['b_id']}.wav")).long().t()
    eos = torch.full((1, 12), cfg.audio_eos_code, dtype=torch.long)
    place = torch.cat([a, b, b[-1:], eos])
    labels = torch.cat([torch.full((a.shape[0], 12), -100, dtype=torch.long), b, eos, eos])
    ids = torch.cat([
        prompt_ids(proc),
        torch.tensor([cfg.mixed_start_token_id]),
        torch.tensor(proc.tokenizer.encode(row["prefix_text"], add_special_tokens=False)),
        torch.tensor([cfg.text_end_token_id]),
        torch.full((a.shape[0],), cfg.audio_out_token_id),
        torch.tensor(proc.tokenizer.encode(row["b_text"], add_special_tokens=False)),
        torch.tensor([cfg.text_end_token_id]),
        torch.full((b.shape[0] + 2,), cfg.audio_out_token_id),
    ])
    return ids, place, labels


def retain_example(proc, cfg, row):
    raw_b = np.load(ROOT / "codes" / f"{row['id']}.npy")
    b = torch.from_numpy(trim_codes(raw_b, ROOT / "wav" / f"{row['id']}.wav")).long().t()
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
    random.seed(SEED)
    torch.manual_seed(SEED)
    pairs = [json.loads(l) for l in (SW / "pairs.jsonl").read_text().splitlines()]
    train = [p for p in pairs if p["split"] == "train"]
    holdout = [p for p in pairs if p["split"] == "holdout"]
    base_rows = [p for p in json.loads((ROOT / "train_pairs.json").read_text()) if p["split"] == "train"]
    retain = random.sample(base_rows, 100)
    print(f"switch={len(train)} retain={len(retain)} holdout={len(holdout)}", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    base = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, ROOT / "lora").merge_and_unload()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)",
    ))
    model.print_trainable_parameters()

    examples = [("switch", r) for r in train] + [("retain", r) for r in retain]

    def loss_of(kind, row):
        ids, place, labels = (switch_example if kind == "switch" else retain_example)(
            proc, model.config, row
        )
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
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=steps, pct_start=min(0.1, 20 / steps), anneal_strategy="cos"
    )

    t0 = time.perf_counter()
    step, log = 0, []
    for epoch in range(EPOCHS):
        random.shuffle(examples)
        opt.zero_grad()
        for i in range(0, len(examples), ACCUM):
            chunk = examples[i:i + ACCUM]
            loss = sum(loss_of(k, r) for k, r in chunk) / len(chunk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 10 == 0:
                rec = {"step": step, "epoch": epoch, "loss": round(float(loss.detach()), 4),
                       "elapsed": round(time.perf_counter() - t0, 1)}
                log.append(rec); print(rec, flush=True)

    model.eval()
    with torch.no_grad():
        hold = [float(loss_of("switch", r)) for r in holdout]
        train_loss = [float(loss_of("switch", r)) for r in train[:60]]
    result = {
        "switch_train_n": len(train), "retain_n": len(retain), "holdout_n": len(holdout),
        "train_loss": round(sum(train_loss) / len(train_loss), 4),
        "holdout_loss": round(sum(hold) / len(hold), 4),
        "steps": step, "wall_s": round(time.perf_counter() - t0, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "log": log,
    }
    (SW / "lora").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(SW / "lora")
    (SW / "switch_train_report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "log"}, indent=2))


if __name__ == "__main__":
    main()
