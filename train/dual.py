"""Dual-stream duplex trainer.

Symmetric two-stream model: each stream has a causal encoder and a decoder that
cross-attends to the other stream. Both next frames are predicted (auxiliary
user loss + agent loss); the agent head is what we listen to.

Form is evaluated separately from a checkpoint (speech activity, overlap,
backchannels). This script trains and checkpoints.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMultimodalLM, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
D, HEADS, MLP, MAXPOS = 384, 6, 1024, 4096
ENC_L, DEC_L = 2, 3
SENTINEL = 2016
WINDOW, BATCH, STEPS, LR = 128, 4, 5000, 1e-3
CKPT_EVERY, EVAL_EVERY = 500, 250


class DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(D); self.self_attn = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.n2 = nn.LayerNorm(D); self.cross_attn = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.n3 = nn.LayerNorm(D); self.mlp = nn.Sequential(nn.Linear(D, MLP), nn.GELU(), nn.Linear(MLP, D))

    def forward(self, x, memory, causal):
        h = self.n1(x)
        x = x + self.self_attn(h, h, h, attn_mask=causal, need_weights=False)[0]
        h = self.n2(x)
        x = x + self.cross_attn(h, memory, memory, need_weights=False)[0]
        return x + self.mlp(self.n3(x))


class Stream(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2048, D, bias=False)
        enc = nn.TransformerEncoderLayer(D, HEADS, MLP, dropout=0.0, activation="gelu",
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, ENC_L, enable_nested_tensor=False)
        self.dec = nn.ModuleList(DecoderLayer() for _ in range(DEC_L))

    def forward(self, x, memory, causal):
        x = self.enc(self.proj(x), mask=causal)
        for layer in self.dec:
            x = layer(x, memory, causal)
        return x


class DualBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.randn(MAXPOS, D) * 0.02)
        self.user = Stream()
        self.agent = Stream()
        self.out = nn.Linear(D, 2048, bias=False)

    def forward(self, ue, ae):
        t = ue.shape[1]
        causal = torch.triu(torch.ones(t, t, device=ue.device, dtype=torch.bool), diagonal=1)
        hu_in = ue + self.pos[:t]
        ha_in = ae + self.pos[:t]
        # encoders first (each stream's own history), then cross decode
        cu = self.user.enc(self.user.proj(hu_in), mask=causal)
        ca = self.agent.enc(self.agent.proj(ha_in), mask=causal)
        hu = cu
        for layer in self.user.dec:
            hu = layer(hu, ca, causal)
        ha = ca
        for layer in self.agent.dec:
            ha = layer(ha, cu, causal)
        return self.out(hu), self.out(ha)


@torch.no_grad()
def embed(table, codes):
    offs = torch.arange(12, device=codes.device) * 2017
    idx = codes + offs
    return table[idx.reshape(-1)].reshape(*idx.shape, table.shape[1]).sum(-2)


def window_of(row, rng):
    u = np.load(ROOT / "data/codes" / row["user"], mmap_mode="r")
    a = np.load(ROOT / "data/codes" / row["agent"], mmap_mode="r")
    n = min(len(u), len(a)); s = int(rng.integers(0, max(1, n - WINDOW)))
    return np.asarray(u[s:s + WINDOW]), np.asarray(a[s:s + WINDOW])


def batch_of(rows, rng, n):
    us, as_ = [], []
    for _ in range(n):
        u, a = window_of(rows[int(rng.integers(len(rows)))], rng)
        us.append(u); as_.append(a)
    return (torch.tensor(np.stack(us), dtype=torch.long),
            torch.tensor(np.stack(as_), dtype=torch.long))


def main() -> None:
    idx = [json.loads(l) for l in open(ROOT / "data/codes/index.jsonl")]
    train = [r for r in idx if r["split"] == "train" and r["frames"] >= WINDOW + 1]
    dev = [r for r in idx if r["split"] == "dev" and r["frames"] >= WINDOW + 1]
    print(f"train {len(train)} dev {len(dev)}", flush=True)

    model = AutoModelForMultimodalLM.from_pretrained(
        str(ROOT / "models/LFM3-Omni-2B"), dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    depth = model.model.depth_decoder.eval()
    table = model.model.codebook_embedding.weight.float()

    net = DualBackbone().cuda()
    print(f"params {sum(p.numel() for p in net.parameters())/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    CK = ROOT / "data/ckpt"; CK.mkdir(exist_ok=True)
    logger = open(CK / "dual_metrics.jsonl", "a")
    legacy = ROOT / "checkpoints/dual"; legacy.mkdir(parents=True, exist_ok=True)
    legacy_log = open(legacy / "metrics.jsonl", "a")
    rng = np.random.default_rng(21)
    t0 = time.time()

    dev_batches = [batch_of(dev, rng, 1) for _ in range(8)]

    for step in range(1, STEPS + 1):
        U, A = batch_of(train, rng, BATCH)
        U, A = U.cuda(), A.cuda()
        pu = torch.cat([torch.full_like(U[:, :1], SENTINEL), U[:, :-1]], 1)
        pa = torch.cat([torch.full_like(A[:, :1], SENTINEL), A[:, :-1]], 1)
        with torch.no_grad():
            ue, ae = embed(table, pu), embed(table, pa)
        hu, ha = net(ue, ae)
        mu = torch.rand_like(U, dtype=torch.float) < 0.5; mu[:, :, 0] = True
        ma = torch.rand_like(A, dtype=torch.float) < 0.5; ma[:, :, 0] = True
        lu = depth(hu.reshape(-1, 2048).to(torch.bfloat16), U.reshape(-1, 12), mu.reshape(-1, 12))
        la = depth(ha.reshape(-1, 2048).to(torch.bfloat16), A.reshape(-1, 12), ma.reshape(-1, 12))
        def ce(lg, tgt, mask):
            lg = lg.reshape(-1, 2017); t = tgt.reshape(-1); m = mask.reshape(-1)
            return F.cross_entropy(lg[m].float(), t[m])
        loss = ce(lu, U, mu) + ce(la, A, ma)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()

        if step % EVAL_EVERY == 0:
            net.eval()
            with torch.no_grad():
                ces_u, ces_a = [], []
                for U0, A0 in dev_batches:
                    U0, A0 = U0.cuda(), A0.cuda()
                    pu0 = torch.cat([torch.full_like(U0[:, :1], SENTINEL), U0[:, :-1]], 1)
                    pa0 = torch.cat([torch.full_like(A0[:, :1], SENTINEL), A0[:, :-1]], 1)
                    hu0, ha0 = net(embed(table, pu0), embed(table, pa0))
                    m0u = torch.rand(1, WINDOW, 12, device="cuda") < 0.5; m0u[:, :, 0] = True
                    m0a = torch.rand(1, WINDOW, 12, device="cuda") < 0.5; m0a[:, :, 0] = True
                    ces_u.append(ce(depth(hu0.reshape(-1, 2048).to(torch.bfloat16), U0.reshape(-1, 12), m0u.reshape(-1, 12)), U0, m0u).item())
                    ces_a.append(ce(depth(ha0.reshape(-1, 2048).to(torch.bfloat16), A0.reshape(-1, 12), m0a.reshape(-1, 12)), A0, m0a).item())
            net.train()
            rec = {"step": step, "train_loss": round(loss.item(), 3),
                   "dev_user_ce": round(float(np.mean(ces_u)), 3),
                   "dev_agent_ce": round(float(np.mean(ces_a)), 3),
                   "frames_s": round(step * BATCH * WINDOW / (time.time() - t0), 1),
                   "s_step": round((time.time() - t0) / step, 2)}
            logger.write(json.dumps(rec) + "\n"); logger.flush(); print(json.dumps(rec), flush=True)
            legacy_log.write(json.dumps({
                "step": step, "loss": rec["dev_agent_ce"], "text_loss": rec["dev_user_ce"],
                "lr": LR, "seconds_per_step": rec["s_step"], "frames_per_s": rec["frames_s"],
                "tokens_per_s": round(rec["frames_s"] * 12, 1),
            }) + "\n"); legacy_log.flush()
        if step % CKPT_EVERY == 0:
            torch.save({"step": step, "model": net.state_dict()}, CK / f"dual-{step:07d}.pt")
            fs = sorted(CK.glob("dual-*.pt"))
            for old in fs[:-5]:
                old.unlink()
            (CK / "dual_latest.txt").write_text(sorted(CK.glob("dual-*.pt"))[-1].name)
    print("done", flush=True)


if __name__ == "__main__":
    main()
