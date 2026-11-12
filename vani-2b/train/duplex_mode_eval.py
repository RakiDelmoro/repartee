#!/usr/bin/env python3
"""Teacher-forced mode accuracy on cached duplex windows.

If the model cannot predict 'speak' even with the correct previous audio token,
the mode head is the problem. If it can, the generation loop or turn coverage is.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE

MODEL_ID = "models/LFM3-Audio-2B-hf"
ROOT = Path("data/probe/mouth")
CT = ROOT / "continuity"
DUP = Path("data/probe/duplex")
JOINT = DUP / "joint"


@torch.no_grad()
def main() -> None:
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else DUP / "cache_tts"
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()
    mouth = PeftModel.from_pretrained(model, ROOT / "lora").merge_and_unload().eval()
    cont = PeftModel.from_pretrained(mouth, CT / "lora_v2").merge_and_unload().eval()
    model = PeftModel.from_pretrained(cont, JOINT / "lora").merge_and_unload().eval()
    mode_head = nn.Linear(model.config.text_config.hidden_size, 2).to(DEVICE)
    mode_head.load_state_dict(torch.load(JOINT / "mode_head.pt", map_location=DEVICE))
    mode_head.eval()
    base = model
    listen_id = proc.tokenizer.encode(" listen", add_special_tokens=False)[0]
    bos_id = base.config.text_config.bos_token_id
    embed = base.get_input_embeddings()

    total = correct = pred_speak = label_speak = 0
    rows = []
    for path in sorted(cache.glob("*.npz"))[:40]:
        d = np.load(path)
        u = torch.from_numpy(d["u"]).to(DEVICE).half()
        codes = torch.from_numpy(d["codes"]).long().to(DEVICE)
        speak = torch.from_numpy(d["speak"]).bool().to(DEVICE)
        n = min(len(u), len(codes), len(speak))
        if n < 8:
            continue
        u, codes, speak = u[:n], codes[:n], speak[:n]
        a_emb = base.model.get_audio_out_embeds(codes)
        listen = embed(torch.tensor([listen_id], device=DEVICE)).expand(n, -1).half()
        a_in = torch.where(speak[:, None], a_emb, listen)
        bos = embed(torch.tensor([bos_id], device=DEVICE)).half()
        seq = torch.cat([bos.view(1, -1), torch.stack([u, a_in], dim=1).reshape(2 * n, -1)], 0)[None]
        out = base.model.language_model(
            inputs_embeds=seq, attention_mask=torch.ones((1, 1 + 2 * n), dtype=torch.long, device=DEVICE),
            use_cache=False,
        )
        h_u = out.last_hidden_state[0, 1::2][:n]
        is_speak = mode_head(h_u.float()).argmax(-1) == 1
        total += n
        correct += int((is_speak == speak).sum())
        pred_speak += int(is_speak.sum())
        label_speak += int(speak.sum())
        rows.append({"id": path.stem, "frames": n, "label_speak": int(speak.sum()),
                     "pred_speak": int(is_speak.sum())})
    report = {"windows": len(rows), "frames": total, "label_speak": label_speak,
              "pred_speak": pred_speak, "accuracy": round(correct / max(total, 1), 4),
              "rows": rows}
    (DUP / "mode_eval.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
