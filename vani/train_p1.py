"""P1 teacher-forced alignment on Open Yap agent channels.

    python -m vani.train_p1 --steps 200 --window 192 --active-codebooks 2
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .model import CODEBOOK_SIZE, TEXT_PAD, VaniM

ROOT = Path(__file__).resolve().parents[1]


class GridWindows(Dataset):
    def __init__(self, root: Path, window: int = 192, stride: int = 128, holdout: str | None = None):
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
        self.index = []
        for i, length in enumerate(self.lengths):
            for start in range(0, max(1, length - window), stride):
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
        return torch.from_numpy(grid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=Path, default=ROOT / "data" / "vani_p0_10h")
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
    ap.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "p1")
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
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    data = GridWindows(args.grids, args.window, args.stride, args.holdout)
    loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0)
    print(f"windows={len(data)} files={len(data.files)} window={args.window} holdout={args.holdout}", flush=True)

    model = VaniM(args.qwen, dtype=torch.bfloat16).to(args.device)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
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
    active = min(args.active_codebooks, 8)
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
                target = next(it).to(args.device)
            except StopIteration:
                it = iter(loader)
                target = next(it).to(args.device)
            inputs = torch.zeros_like(target)
            inputs[:, 1:, :] = target[:, :-1, :]
            inputs[:, 0, 0] = TEXT_PAD

            p_self = min(args.self_cond, args.self_cond * step / max(1, args.self_cond_warmup))
            if p_self > 0:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    _, tf_audio = model(inputs, target)
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
                text_logits, audio_logits = model(inputs, target)
                text_mask = target[:, :, 0] != TEXT_PAD
                if text_mask.any():
                    text_ce = F.cross_entropy(
                        text_logits.float()[text_mask], target[:, :, 0][text_mask]
                    )
                else:
                    text_ce = torch.zeros((), device=args.device)
                cb_losses = []
                for cb in range(active):
                    cb_losses.append(
                        F.cross_entropy(
                            audio_logits[:, :, cb].float().reshape(-1, CODEBOOK_SIZE),
                            target[:, :, 1 + cb].reshape(-1),
                        )
                    )
                audio_ce = torch.stack(cb_losses).mean()
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
            torch.save(
                {
                    "model": model.state_dict(),
                    "step": step,
                    "args": vars(args),
                },
                ckpt,
            )
            print(f"saved {ckpt}", flush=True)
    ckpt = args.out / "last.pt"
    torch.save({"model": model.state_dict(), "step": args.steps, "args": vars(args)}, ckpt)
    print(f"saved {ckpt}", flush=True)


if __name__ == "__main__":
    main()
