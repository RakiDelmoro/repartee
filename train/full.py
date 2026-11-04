"""Full dual-stream training run (long, resumable, scheduled).

- one epoch = 23,449 steps = 12.0M frames at batch 4 x window 128
- linear warmup then cosine decay
- newest-5 circular checkpoints with optimizer state
- metrics in the dashboard schema (checkpoints/dual/metrics.jsonl)

Run:
    python -m train.full                    # one epoch, resume if possible
    python -m train.full --steps 50000      # longer
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForMultimodalLM, AutoProcessor

from train.dual import DualBackbone, SENTINEL, embed

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/codes"
CK = ROOT / "data/ckpt"
LEGACY = ROOT / "checkpoints/dual"


def lr_at(step: int, total: int, base: float, warmup: int, floor: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def load_arrays(rows, window):
    out = []
    for r in rows:
        u = np.load(DATA / r["user"])
        a = np.load(DATA / r["agent"])
        n = min(len(u), len(a))
        out.append((u[:n], a[:n]))
    print(f"preloaded {len(out)} conversations, {sum(len(u) for u, _ in out)/1e6:.1f}M frames", flush=True)
    return out


def batch_of(arrays, rng, n, window):
    us, as_ = [], []
    for _ in range(n):
        u, a = arrays[int(rng.integers(len(arrays)))]
        s = int(rng.integers(0, max(1, len(u) - window)))
        us.append(u[s:s + window]); as_.append(a[s:s + window])
    return (torch.tensor(np.stack(us), dtype=torch.long),
            torch.tensor(np.stack(as_), dtype=torch.long))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=23449)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lr-floor", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--keep", type=int, default=5)
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    idx = [json.loads(l) for l in open(DATA / "index.jsonl")]
    train = [r for r in idx if r["split"] == "train" and r["frames"] >= a.window + 1]
    dev = [r for r in idx if r["split"] == "dev" and r["frames"] >= a.window + 1]
    print(f"train {len(train)} conv  dev {len(dev)} conv  target {a.steps} steps", flush=True)

    model = AutoModelForMultimodalLM.from_pretrained(
        str(ROOT / "models/LFM3-Omni-2B"), dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    depth = model.model.depth_decoder.eval()
    table = model.model.codebook_embedding.weight.float()

    net = DualBackbone().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    start = 0
    latest = CK / "dual_latest.txt"
    if not a.no_resume and latest.exists():
        f = CK / latest.read_text().strip()
        st = torch.load(f, map_location="cuda")
        net.load_state_dict(st["model"])
        if "opt" in st:
            opt.load_state_dict(st["opt"])
        start = st["step"]
        print(f"resumed {f.name} at step {start}", flush=True)

    train_arrays = load_arrays(train, a.window)
    dev_arrays = load_arrays(dev, a.window)
    rng = np.random.default_rng(1234)
    dev_batches = [batch_of(dev_arrays, rng, 1, a.window) for _ in range(8)]

    CK.mkdir(parents=True, exist_ok=True); LEGACY.mkdir(parents=True, exist_ok=True)
    log = open(CK / "dual_metrics.jsonl", "a")
    legacy = open(LEGACY / "metrics.jsonl", "a")
    t0 = time.time()

    for step in range(start + 1, a.steps + 1):
        lr = lr_at(step, a.steps, a.lr, a.warmup, a.lr_floor)
        for g in opt.param_groups:
            g["lr"] = lr

        U, A = batch_of(train_arrays, rng, a.batch, a.window)
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
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if step % a.eval_every == 0:
            net.eval()
            with torch.no_grad():
                ces_u, ces_a = [], []
                for U0, A0 in dev_batches:
                    U0, A0 = U0.cuda(), A0.cuda()
                    pu0 = torch.cat([torch.full_like(U0[:, :1], SENTINEL), U0[:, :-1]], 1)
                    pa0 = torch.cat([torch.full_like(A0[:, :1], SENTINEL), A0[:, :-1]], 1)
                    hu0, ha0 = net(embed(table, pu0), embed(table, pa0))
                    m0u = torch.rand(1, a.window, 12, device="cuda") < 0.5; m0u[:, :, 0] = True
                    m0a = torch.rand(1, a.window, 12, device="cuda") < 0.5; m0a[:, :, 0] = True
                    ces_u.append(ce(depth(hu0.reshape(-1, 2048).to(torch.bfloat16), U0.reshape(-1, 12), m0u.reshape(-1, 12)), U0, m0u).item())
                    ces_a.append(ce(depth(ha0.reshape(-1, 2048).to(torch.bfloat16), A0.reshape(-1, 12), m0a.reshape(-1, 12)), A0, m0a).item())
            net.train()
            done = step - start
            rec = {"step": step, "train_loss": round(loss.item(), 3),
                   "dev_user_ce": round(float(np.mean(ces_u)), 3),
                   "dev_agent_ce": round(float(np.mean(ces_a)), 3),
                   "lr": round(lr, 6),
                   "frames_s": round(done * a.batch * a.window / (time.time() - t0), 1),
                   "s_step": round((time.time() - t0) / done, 3), "ts": round(time.time(), 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            legacy.write(json.dumps({"step": step, "loss": rec["dev_agent_ce"], "text_loss": rec["dev_user_ce"],
                                     "lr": rec["lr"], "seconds_per_step": rec["s_step"],
                                     "frames_per_s": rec["frames_s"], "tokens_per_s": round(rec["frames_s"] * 12, 1)}) + "\n")
            legacy.flush()
            print(json.dumps(rec), flush=True)

        if step % a.ckpt_every == 0:
            torch.save({"step": step, "model": net.state_dict(), "opt": opt.state_dict(), "lr": lr},
                       CK / f"dual-{step:07d}.pt")
            fs = sorted(CK.glob("dual-[0-9]*.pt"))
            for old in fs[:-a.keep]:
                old.unlink()
            (CK / "dual_latest.txt").write_text(sorted(CK.glob("dual-[0-9]*.pt"))[-1].name)
    print("done", flush=True)


if __name__ == "__main__":
    main()
