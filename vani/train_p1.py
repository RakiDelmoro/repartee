"""P1 teacher-forced alignment on Open Yap agent channels.

    python -m vani.train_p1 --steps 200 --window 192 --active-codebooks 2
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .model import CODEBOOK_SIZE, TEXT_PAD, VaniM, load_vani_state

ROOT = Path(__file__).resolve().parents[1]
VOICE_RMS = 0.01


def save_checkpoint(path: Path, payload: dict) -> None:
    """Write atomically so a crash cannot leave a half checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def prune_checkpoints(out: Path, keep: int) -> None:
    """Keep only the newest ``keep`` step-*.pt files; last.pt is separate."""
    files = sorted(
        out.glob("step-*.pt"),
        key=lambda p: int(re.search(r"step-(\d+)", p.name).group(1)),
    )
    for old in files[:-keep]:
        old.unlink()


class GridWindows(Dataset):
    def __init__(self, root: Path, window: int = 192, stride: int = 128,
                 holdout: str | None = None, min_voiced: float = 0.0):
        self.files = sorted(root.glob("*.grid.npy"))
        if holdout:
            self.files = [p for p in self.files if holdout not in p.name]
            if not self.files:
                raise FileNotFoundError(f"holdout removed every grid: {holdout}")
        if not self.files:
            raise FileNotFoundError(f"no grids under {root}")
        self.window = window
        self.arrays = [np.load(p, mmap_mode="r") for p in self.files]
        self.lengths = [len(a) for a in self.arrays]
        self.rms = []
        self.words = []
        for path in self.files:
            rms_path = path.with_name(path.name.replace(".grid.npy", ".rms.npy"))
            word_path = path.with_name(path.name.replace(".grid.npy", ".word.npy"))
            self.rms.append(np.load(rms_path, mmap_mode="r") if rms_path.exists() else None)
            self.words.append(np.load(word_path, mmap_mode="r") if word_path.exists() else None)
        self.index = []
        for i, length in enumerate(self.lengths):
            for start in range(0, max(1, length - window), stride):
                if min_voiced > 0 and self.rms[i] is not None:
                    voiced = np.asarray(self.rms[i][start : start + window]) > VOICE_RMS
                    if voiced.size == 0 or voiced.mean() < min_voiced:
                        continue
                self.index.append((i, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        file_index, start = self.index[item]
        grid = np.asarray(self.arrays[file_index][start : start + self.window], dtype=np.int64)
        if len(grid) < self.window:
            pad = np.zeros((self.window - len(grid), grid.shape[1]), dtype=np.int64)
            pad[:, 0] = TEXT_PAD
            grid = np.concatenate([grid, pad], axis=0)
        if self.rms[file_index] is None:
            rms = np.full(len(grid), VOICE_RMS, dtype=np.float32)
        else:
            rms = np.asarray(self.rms[file_index][start : start + self.window], dtype=np.float32)
            if len(rms) < len(grid):
                rms = np.pad(rms, (0, len(grid) - len(rms)))
        if self.words[file_index] is None:
            word = np.asarray(grid[:, 0], dtype=np.int64)
        else:
            word = np.asarray(self.words[file_index][start : start + self.window], dtype=np.int64)
            if len(word) < len(grid):
                word = np.pad(word, (0, len(grid) - len(word)), constant_values=TEXT_PAD)
        return torch.from_numpy(grid), torch.from_numpy(rms), torch.from_numpy(word)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=Path, default=ROOT / "data" / "vani_p0_45")
    ap.add_argument("--qwen", default=str(ROOT / "models" / "Qwen3-0.6B-Base"))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--active-codebooks", type=int, default=2)
    ap.add_argument("--text-weight", type=float, default=0.5)
    ap.add_argument("--audio-weight", type=float, default=1.0)
    ap.add_argument("--lr-qwen", type=float, default=1e-5)
    ap.add_argument("--lr-new", type=float, default=1e-4)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--keep-checkpoints", type=int, default=5,
                    help="newest step-*.pt files to retain; minimum 5")
    ap.add_argument("--step-offset", type=int, default=0,
                    help="display offset for resumed runs (dashboard progression)")
    ap.add_argument("--text-cond", choices=["sparse", "word"], default="sparse",
                    help="sparse = deployable per-frame text stream; word = held-word probe")
    ap.add_argument("--cb-weights", default="1,1,1,1,1,1,1,1",
                    help="per-codebook loss weights, comma separated, 8 values")
    ap.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "p1_text29h")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--loss-scale", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--self-cond", type=float, default=0.0,
                    help="max probability of feeding own previous audio predictions")
    ap.add_argument("--self-cond-codebooks", type=int, default=8,
                    help="number of previous codebooks replaced by predictions; "
                         "8 matches free-running generation")
    ap.add_argument("--self-cond-warmup", type=int, default=1000)
    ap.add_argument("--min-voiced", type=float, default=0.25,
                    help="minimum fraction of voiced frames in a sampled window")
    ap.add_argument("--voice-weight", type=float, default=2.0,
                    help="extra loss weight for frames above VOICE_RMS")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    data = GridWindows(args.grids, args.window, args.stride, args.holdout,
                       min_voiced=args.min_voiced)
    loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0)
    print(f"grids={data.files[0].parent} windows={len(data)} files={len(data.files)} "
          f"window={args.window} holdout={args.holdout} "
          f"min_voiced={args.min_voiced}", flush=True)

    model = VaniM(args.qwen, dtype=torch.bfloat16).to(args.device)
    if args.resume:
        state = load_vani_state(model, args.resume)
        print(f"resumed {args.resume} at step {state.get('step')}", flush=True)
    model.train()
    model.temporal.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.temporal.config.use_cache = False
    model.temporal.enable_input_require_grads()

    qwen_params = [p for n, p in model.named_parameters() if n.startswith("qwen.")]
    new_params = [p for n, p in model.named_parameters() if not n.startswith("qwen.")]
    print(
        f"params qwen={sum(p.numel() for p in qwen_params)/1e6:.1f}M "
        f"new={sum(p.numel() for p in new_params)/1e6:.1f}M",
        flush=True,
    )
    from bitsandbytes.optim import AdamW8bit

    opt = AdamW8bit(
        [
            {"params": qwen_params, "lr": args.lr_qwen},
            {"params": new_params, "lr": args.lr_new},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    keep = max(5, args.keep_checkpoints)
    active = min(args.active_codebooks, 8)
    cb_weights = [float(x) for x in args.cb_weights.split(",")]
    if len(cb_weights) != 8:
        raise ValueError("--cb-weights needs exactly 8 values")
    if sum(cb_weights[:active]) <= 0:
        raise ValueError("--cb-weights has no positive active codebook")
    (args.out / "run.json").write_text(json.dumps({
        "steps": args.steps + args.step_offset,
        "step_offset": args.step_offset,
        "additional_steps": args.steps,
        "accum": args.accum,
        "window": args.window,
        "stride": args.stride,
        "grids": str(args.grids),
        "holdout": args.holdout,
        "min_voiced": args.min_voiced,
        "voice_weight": args.voice_weight,
        "active_codebooks": active,
        "keep_checkpoints": keep,
        "resume": str(args.resume) if args.resume else None,
        "text_cond": args.text_cond,
        "cb_weights": cb_weights,
    }, indent=2) + "\n")
    started = time.time()
    running = []
    skipped = 0
    it = iter(loader)
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        micro_losses = []
        micro_text = []
        micro_cb = []
        skipped_step = False
        for _ in range(args.accum):
            try:
                target, rms, word = next(it)
            except StopIteration:
                it = iter(loader)
                target, rms, word = next(it)
            target = target.to(args.device)
            rms = rms.to(args.device)
            word = word.to(args.device)
            cond = word if args.text_cond == "word" else target[:, :, 0]
            frame_w = 1.0 + args.voice_weight * (rms > VOICE_RMS).float()
            inputs = torch.zeros_like(target)
            inputs[:, 1:, :] = target[:, :-1, :]
            inputs[:, 0, 0] = TEXT_PAD

            p_self = min(args.self_cond, args.self_cond * step / max(1, args.self_cond_warmup))
            if p_self > 0:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    _, tf_audio = model(inputs, target, text_cond=cond)
                pred = tf_audio.argmax(-1)
                mixed = inputs.clone()
                use = torch.rand(
                    (target.shape[0], target.shape[1] - 1, 1),
                    device=target.device,
                ) < p_self
                cond = min(args.self_cond_codebooks, 8)
                mixed[:, 1:, 1:1 + cond] = torch.where(
                    use,
                    pred[:, :-1, :cond],
                    mixed[:, 1:, 1:1 + cond],
                )
                inputs = mixed

            with torch.autocast("cuda", dtype=torch.bfloat16):
                text_logits, audio_logits = model(inputs, target, text_cond=cond)
                text_mask = target[:, :, 0] != TEXT_PAD
                if text_mask.any():
                    token_ce = F.cross_entropy(
                        text_logits.float().reshape(-1, text_logits.shape[-1]),
                        target[:, :, 0].reshape(-1),
                        reduction="none",
                    ).view(target.shape[0], target.shape[1])
                    text_ce = (token_ce * frame_w)[text_mask].sum() / (
                        frame_w * text_mask
                    ).sum().clamp(min=1)
                else:
                    text_ce = torch.zeros((), device=args.device)
                cb_losses = []
                for cb in range(active):
                    per_frame = F.cross_entropy(
                        audio_logits[:, :, cb].float().reshape(-1, CODEBOOK_SIZE),
                        target[:, :, 1 + cb].reshape(-1),
                        reduction="none",
                    ).view(target.shape[0], target.shape[1])
                    cb_losses.append((per_frame * frame_w).sum() / frame_w.sum())
                cb_w = torch.tensor(cb_weights[:active], device=args.device)
                audio_ce = (torch.stack(cb_losses) * cb_w).sum() / cb_w.sum()
                loss = args.text_weight * text_ce + args.audio_weight * audio_ce

            if not torch.isfinite(loss):
                skipped_step = True
                break
            ((loss / args.accum) * args.loss_scale).backward()
            micro_losses.append(float(loss.detach()))
            micro_text.append(float(text_ce.detach()))
            micro_cb.append([float(v.detach()) for v in cb_losses])

        if skipped_step:
            opt.zero_grad(set_to_none=True)
            skipped += 1
            print(f"step {step} skip non-finite loss", flush=True)
            continue

        finite = True
        for param in model.parameters():
            if param.grad is None:
                continue
            param.grad.div_(args.loss_scale)
            if not torch.isfinite(param.grad).all():
                finite = False
                break
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            opt.zero_grad(set_to_none=True)
            skipped += 1

        loss_value = sum(micro_losses) / len(micro_losses)
        text_value = sum(micro_text) / len(micro_text)
        cb_value = [sum(v[i] for v in micro_cb) / len(micro_cb) for i in range(active)]
        running.append(loss_value)
        if step == 1 or step % 10 == 0:
            cbs = " ".join(f"cb{i}={v:.3f}" for i, v in enumerate(cb_value))
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(
                f"step {step} loss={loss_value:.3f} text={text_value:.3f} "
                f"{cbs} maxmem={mem:.2f}GB skip={skipped} accum={args.accum} p={p_self:.2f} "
                f"{time.time()-started:.0f}s",
                flush=True,
            )
        if step % args.save_every == 0:
            ckpt = args.out / f"step-{step}.pt"
            save_checkpoint(
                ckpt,
                {
                    "model": model.state_dict(),
                    "step": step,
                    "args": vars(args),
                },
            )
            prune_checkpoints(args.out, keep)
            print(f"saved {ckpt} (ring={keep})", flush=True)
    ckpt = args.out / "last.pt"
    save_checkpoint(ckpt, {"model": model.state_dict(), "step": args.steps,
                           "args": vars(args)})
    print(f"saved {ckpt}", flush=True)


if __name__ == "__main__":
    main()
