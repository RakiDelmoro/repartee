"""Generate a stereo sample from a dual-stream checkpoint.

Left  = human user (reference conversation)
Right = model-generated agent frames

Usage: python -m train.sample --ckpt data/ckpt/dual-0000500.pt --out data/proof/dual
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from train.dual import DualBackbone, SENTINEL, embed

ROOT = Path(__file__).resolve().parents[1]
FRAMES = 300


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="data/proof/dual")
    ap.add_argument("--frames", type=int, default=FRAMES)
    a = ap.parse_args()

    ckpt = a.ckpt
    if ckpt is None:
        latest = ROOT / "data/ckpt/dual_latest.txt"
        ckpt = str(ROOT / "data/ckpt" / latest.read_text().strip())

    idx = [json.loads(l) for l in open(ROOT / "data/codes/index.jsonl")]
    dev = [r for r in idx if r["split"] == "dev" and r["frames"] >= a.frames + 1]
    row = dev[0]
    U = torch.tensor(np.load(ROOT / "data/codes" / row["user"]), dtype=torch.long, device="cuda")[None]
    A = torch.tensor(np.load(ROOT / "data/codes" / row["agent"]), dtype=torch.long, device="cuda")[None]
    T = min(U.shape[1], A.shape[1])
    Uw, Aw = U[:, :a.frames], A[:, :a.frames]

    model = AutoModelForMultimodalLM.from_pretrained(
        str(ROOT / "models/LFM3-Omni-2B"), dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()
    proc = AutoProcessor.from_pretrained(str(ROOT / "models/LFM3-Omni-2B"), trust_remote_code=True)
    codec = proc.audio_tokenizer.to("cuda", dtype=torch.float32).eval()
    depth = model.model.depth_decoder
    table = model.model.codebook_embedding.weight.float()

    net = DualBackbone().cuda().eval()
    st = torch.load(ckpt, map_location="cuda")
    net.load_state_dict(st["model"])
    print(f"ckpt step {st['step']}  dev {row['id']}", flush=True)

    hist = [torch.full((12,), SENTINEL, dtype=torch.long, device="cuda")]
    out = []
    with torch.no_grad():
        for k in range(a.frames):
            ue = embed(table, Uw[:, : k + 1])
            ae = embed(table, torch.stack(hist, 0)[None])
            _, ha = net(ue, ae)
            h = ha[:, -1]
            codes = torch.full((1, 12), SENTINEL, dtype=torch.long, device="cuda")
            m = torch.ones((1, 12), dtype=torch.bool, device="cuda")
            for keep in (4, 4, 4):
                lo = depth(h.to(torch.bfloat16), codes, m)
                conf, pick = lo.softmax(-1).max(-1)
                top = conf.masked_fill(~m, -1).topk(min(keep, int(m.sum())), -1).indices
                codes[0, top] = pick[0, top]
                m[0, top] = False
            out.append(codes[0]); hist.append(codes[0])
    g = torch.stack(out)

    with torch.no_grad():
        gen = codec.decode(g.T.float().unsqueeze(0)).audio_values[0].float().cpu().numpy()
        usr = codec.decode(Uw[0].T.float().unsqueeze(0)).audio_values[0].float().cpu().numpy()
        ref = codec.decode(Aw[0].T.float().unsqueeze(0)).audio_values[0].float().cpu().numpy()
    n = min(len(gen), len(usr), len(ref))
    d = ROOT / a.out / f"step{st['step']:07d}"
    d.mkdir(parents=True, exist_ok=True)
    sf.write(d / "model_agent.wav", gen, 24000)
    sf.write(d / "reference_agent.wav", ref, 24000)
    sf.write(d / "duplex_stereo.wav", np.stack([usr[:n], gen[:n]], 1), 24000)
    print("wrote", d, flush=True)


if __name__ == "__main__":
    main()
