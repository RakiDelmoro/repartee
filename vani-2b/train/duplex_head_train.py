#!/usr/bin/env python3
"""Train only the duplex mode head on a frozen mouth+continuity model.

Speak-input dropout teaches the head to say 'speak' from a listen context, which
is the state inference actually starts from.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE

MODEL_ID = "models/LFM3-Omni-2B-hf"
ROOT = Path("data/probe/mouth")
CT = ROOT / "continuity"
DUP = Path("data/probe/duplex")
JOINT = DUP / "joint"
CACHE = DUP / "cache_turns"
EPOCHS = 6
LR = 1e-3
DROP = 0.5
SPEAK_WEIGHT = 1.0
SEED = 0


def main() -> None:
    random.seed(SEED); torch.manual_seed(SEED)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()
    mouth = PeftModel.from_pretrained(model, ROOT / "lora").merge_and_unload().eval()
    sw = PeftModel.from_pretrained(mouth, ROOT / "switch" / "lora").merge_and_unload().eval()
    cont = PeftModel.from_pretrained(sw, CT / "lora_v2").merge_and_unload().eval()
    base = PeftModel.from_pretrained(cont, JOINT / "lora").merge_and_unload().eval()
    for p in base.parameters():
        p.requires_grad_(False)
    mode_head = nn.Linear(base.config.text_config.hidden_size + 3, 2).to(DEVICE)
    opt = torch.optim.AdamW(mode_head.parameters(), lr=LR, weight_decay=0.0)
    embed = base.get_input_embeddings()
    listen_id = proc.tokenizer.encode(" listen", add_special_tokens=False)[0]
    bos_id = base.config.text_config.bos_token_id
    listen_emb = embed(torch.tensor([listen_id], device=DEVICE)).half()
    bos = embed(torch.tensor([bos_id], device=DEVICE)).half()

    files = sorted(CACHE.glob("*.npz"))
    print("windows", len(files), flush=True)
    for epoch in range(EPOCHS):
        random.shuffle(files)
        total = correct = pred_speak = label_speak = 0
        for path in files:
            d = np.load(path)
            u = torch.from_numpy(d["u"]).to(DEVICE).half()
            codes = torch.from_numpy(d["codes"]).long().to(DEVICE)
            speak = torch.from_numpy(d["speak"]).bool().to(DEVICE)
            n = min(len(u), len(codes), len(speak))
            if n < 8:
                continue
            u, codes, speak = u[:n], codes[:n], speak[:n]
            with torch.no_grad():
                a_emb = base.model.get_audio_out_embeds(codes)
                listen_n = listen_emb.expand(n, -1)
                a_in = torch.where(speak[:, None], a_emb, listen_n)
                # simulate the model's own wrong 'listen' decisions
                drop = (torch.rand(n, device=DEVICE) < DROP) & speak
                a_in = torch.where(drop[:, None], listen_n, a_in)
                seq = torch.cat([bos.view(1, -1), torch.stack([u, a_in], dim=1).reshape(2 * n, -1)], 0)[None]
                out = base.model.language_model(
                    inputs_embeds=seq,
                    attention_mask=torch.ones((1, 1 + 2 * n), dtype=torch.long, device=DEVICE),
                    use_cache=False,
                )
            h_u = out.last_hidden_state[0, 1::2][:n]
            b = torch.from_numpy(d["b_speak"])[:n].bool().to(DEVICE)
            sil = torch.zeros(n, device=DEVICE)
            run = 0
            for i in range(n):
                run = 0 if b[i] else run + 1
                sil[i] = min(run, 25) / 25.0
            feats = torch.stack([b.float(), sil, 1.0 - b.float()], -1)
            logits = mode_head(torch.cat([h_u.float(), feats], -1))
            ce = F.cross_entropy(logits, speak.long(), reduction="none")
            loss = (ce * torch.where(speak, SPEAK_WEIGHT, 1.0)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pred = logits.argmax(-1)
                total += n; correct += int((pred == speak.long()).sum())
                pred_speak += int(pred.sum()); label_speak += int(speak.sum())
        print({"epoch": epoch, "acc": round(correct / max(total, 1), 4),
               "pred_speak": pred_speak, "label_speak": label_speak,
               "frames": total}, flush=True)
    JOINT.mkdir(parents=True, exist_ok=True)
    torch.save(mode_head.state_dict(), JOINT / "mode_head.pt")
    print("saved", JOINT / "mode_head.pt")


if __name__ == "__main__":
    main()
