"""Evaluate a dual-stream checkpoint: stereo sample + real vs mismatched user gap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import AutoModelForMultimodalLM, AutoProcessor

from train.dual import DualBackbone, SENTINEL, embed, WINDOW

ROOT = Path(__file__).resolve().parents[1]
FRAMES = 300


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="data/proof/dual")
    a = ap.parse_args()
    ckpt = a.ckpt or str(ROOT / "data/ckpt" / (ROOT / "data/ckpt/dual_latest.txt").read_text().strip())

    idx = [json.loads(l) for l in open(ROOT / "data/codes/index.jsonl")]
    dev = [r for r in idx if r["split"] == "dev" and r["frames"] >= FRAMES + 1]
    row = dev[0]
    U = torch.tensor(np.load(ROOT / "data/codes" / row["user"]), dtype=torch.long, device="cuda")[None]
    A = torch.tensor(np.load(ROOT / "data/codes" / row["agent"]), dtype=torch.long, device="cuda")[None]
    T = min(U.shape[1], A.shape[1])
    Uw, Aw = U[:, :FRAMES], A[:, :FRAMES]

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
    print(f"ckpt step {st['step']}", flush=True)

    # ---- stereo sample ----
    hist = [torch.full((12,), SENTINEL, dtype=torch.long, device="cuda")]
    out = []
    with torch.no_grad():
        for k in range(FRAMES):
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

    # ---- gap: agent CE with real vs mismatched user (history dropped) ----
    rng = np.random.default_rng(31)
    ces = {"real": [], "mismatch": []}
    for i, r in enumerate(dev[:8]):
        u = np.load(ROOT / "data/codes" / r["user"], mmap_mode="r")
        al = np.load(ROOT / "data/codes" / r["agent"], mmap_mode="r")
        nn = min(len(u), len(al)); s = int(rng.integers(0, max(1, nn - WINDOW)))
        U0 = torch.tensor(np.asarray(u[s:s + WINDOW]), dtype=torch.long, device="cuda")[None]
        A0 = torch.tensor(np.asarray(al[s:s + WINDOW]), dtype=torch.long, device="cuda")[None]
        r2 = dev[(i + 1) % 8]
        u2 = np.load(ROOT / "data/codes" / r2["user"], mmap_mode="r")
        U1 = torch.tensor(np.asarray(u2[s:s + WINDOW]), dtype=torch.long, device="cuda")[None]
        prev = torch.full_like(A0, SENTINEL)
        mask = torch.ones(1, WINDOW, 12, dtype=torch.bool, device="cuda")
        for name, uu in (("real", U0), ("mismatch", U1)):
            with torch.no_grad():
                _, ha = net(embed(table, uu), embed(table, prev))
                lo = depth(ha.reshape(-1, 2048).to(torch.bfloat16), A0.reshape(-1, 12), mask.reshape(-1, 12))
            ces[name].append(F.cross_entropy(lo.reshape(-1, 2017).float(), A0.reshape(-1)).item())
    real, mis = float(np.mean(ces["real"])), float(np.mean(ces["mismatch"]))
    print(json.dumps({"step": st["step"], "agent_ce_real": round(real, 3),
                      "agent_ce_mismatch": round(mis, 3), "gap": round(mis - real, 3)}, indent=2))


if __name__ == "__main__":
    main()
