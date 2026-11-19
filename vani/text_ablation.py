"""Does the current-frame text actually change the audio prediction?

Same temporal hidden, same previous audio, three text conditions on the depth
head: the real word, PAD, and a random word.  Reports how often cb0/cb1 change,
the KL from the real distribution, and accuracy against the true codes.

    python -m vani.text_ablation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import TEXT_PAD, VaniM, load_vani_state

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "checkpoints" / "p1_text29h" / "last.pt")
    ap.add_argument("--grid", type=Path,
                    default=ROOT / "data" / "vani_p0_45" / "conv_d65ed16a6092.grid.npy")
    ap.add_argument("--qwen", default=str(ROOT / "models" / "Qwen3-0.6B-Base"))
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--voice-rms", type=float, default=0.01)
    ap.add_argument("--text-cond", choices=["sparse", "word"], default="sparse")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    grid = np.load(args.grid, mmap_mode="r")
    rms_path = args.grid.with_name(args.grid.name.replace(".grid.npy", ".rms.npy"))
    word_path = args.grid.with_name(args.grid.name.replace(".grid.npy", ".word.npy"))
    rms = np.load(rms_path) if rms_path.exists() else None
    words = np.load(word_path) if word_path.exists() else None
    model = VaniM(args.qwen, dtype=torch.bfloat16).to(args.device)
    load_vani_state(model, args.checkpoint)
    model.eval()

    vocab = model.text_embed.num_embeddings
    g = torch.Generator(device="cpu").manual_seed(0)
    acc = {k: {"n": 0, "changed0": 0, "cb1_changed": 0, "kl0": 0.0, "kl1": 0.0,
               "acc0_real": 0, "acc0_pad": 0, "acc0_rand": 0,
               "cb1_acc_real": 0, "cb1_acc_pad": 0, "cb1_acc_rand": 0}
           for k in ("word", "word_voiced", "voiced")}

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(grid) - args.window, args.stride):
            t = torch.from_numpy(np.asarray(grid[start : start + args.window], dtype=np.int64))[None]
            t = t.to(args.device)
            inputs = torch.zeros_like(t)
            inputs[:, 1:] = t[:, :-1]
            inputs[:, 0, 0] = TEXT_PAD
            if args.text_cond == "word" and words is not None:
                real = torch.from_numpy(
                    np.asarray(words[start : start + args.window], dtype=np.int64)
                )[None].to(args.device)
            else:
                real = t[:, :, 0]
            pad = torch.full_like(real, TEXT_PAD)
            rand = real.clone()
            rand[real != TEXT_PAD] = torch.randint(
                0, vocab, (int((real != TEXT_PAD).sum()),), generator=g
            ).to(args.device)
            hidden = model.temporal_hidden(inputs)
            logits = {
                "real": model.audio_logits(hidden, t[:, :, 1:9], real),
                "pad": model.audio_logits(hidden, t[:, :, 1:9], pad),
                "rand": model.audio_logits(hidden, t[:, :, 1:9], rand),
            }
            word = (real[0] != TEXT_PAD)
            voiced = torch.zeros_like(word)
            if rms is not None:
                voiced = torch.from_numpy(
                    (np.asarray(rms[start : start + args.window]) > args.voice_rms)
                ).to(args.device)
            masks = {
                "word": word,
                "word_voiced": word & voiced,
                "voiced": voiced,
            }
            p = {k: torch.softmax(v.float(), dim=-1) for k, v in logits.items()}
            pred = {k: v.argmax(-1)[0] for k, v in logits.items()}
            for name, mask in masks.items():
                n = int(mask.sum())
                if not n:
                    continue
                a = acc[name]
                a["n"] += n
                for cb in (0, 1):
                    y = t[0, :, 1 + cb]
                    pr = pred["real"][mask, cb]
                    pp = pred["pad"][mask, cb]
                    a[f"changed{cb}" if cb == 0 else "cb1_changed"] += int(
                        (pr != pp).sum()
                    )
                    kl = (p["real"][0][mask, cb] *
                          (torch.log(p["real"][0][mask, cb] + 1e-9) -
                           torch.log(p["pad"][0][mask, cb] + 1e-9))).sum(-1)
                    a["kl0" if cb == 0 else "kl1"] += float(kl.sum())
                    for cond in ("real", "pad", "rand"):
                        key = f"acc{cb}_{cond}" if cb == 0 else f"cb1_acc_{cond}"
                        a[key] += int((pred[cond][mask, cb] == y[mask]).sum())

    def norm(a):
        n = max(1, a["n"])
        return {
            "frames": a["n"],
            "cb0_argmax_changed_vs_pad": round(a["changed0"] / n, 4),
            "cb1_argmax_changed_vs_pad": round(a["cb1_changed"] / n, 4),
            "cb0_kl_vs_pad": round(a["kl0"] / n, 4),
            "cb1_kl_vs_pad": round(a["kl1"] / n, 4),
            "cb0_acc_real": round(a["acc0_real"] / n, 4),
            "cb0_acc_pad": round(a["acc0_pad"] / n, 4),
            "cb0_acc_random": round(a["acc0_rand"] / n, 4),
            "cb1_acc_real": round(a["cb1_acc_real"] / n, 4),
            "cb1_acc_pad": round(a["cb1_acc_pad"] / n, 4),
            "cb1_acc_random": round(a["cb1_acc_rand"] / n, 4),
        }

    result = {
        "checkpoint": str(args.checkpoint),
        "grid": str(args.grid),
        "text_cond": args.text_cond,
        **{k: norm(v) for k, v in acc.items()},
    }
    print(json.dumps(result, indent=2))
    (args.checkpoint.parent / "text_ablation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
