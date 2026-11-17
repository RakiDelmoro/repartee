"""Teacher-forced P1 evaluation on one held-out conversation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data.mimi import SAMPLE_RATE, decode_chunks, load_mimi
from .model import CODEBOOK_SIZE, TEXT_PAD, VaniM

ROOT = Path(__file__).resolve().parents[1]


def windows(path: Path, length: int, stride: int):
    grid = np.load(path, mmap_mode="r")
    for start in range(0, max(1, len(grid) - length), stride):
        chunk = np.asarray(grid[start : start + length], dtype=np.int64)
        if len(chunk) == length:
            yield torch.from_numpy(chunk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "p1" / "last.pt")
    ap.add_argument("--grid", type=Path, default=ROOT / "data" / "vani_p0_10h" / "conv_d65ed16a6092.grid.npy")
    ap.add_argument("--qwen", default=str(ROOT / "models" / "Qwen3-0.6B-Base"))
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--audio-out", type=Path, default=None)
    ap.add_argument("--mimi", default=str(ROOT / "models" / "mimi"))
    args = ap.parse_args()

    model = VaniM(args.qwen, dtype=torch.bfloat16).to(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    sums = torch.zeros(8, dtype=torch.float64)
    correct = torch.zeros(8, dtype=torch.float64)
    text_loss = 0.0
    text_correct = 0
    text_total = 0
    count = 0
    predicted_codes = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for target in windows(args.grid, args.window, args.stride):
            target = target.to(args.device)[None]
            inputs = torch.zeros_like(target)
            inputs[:, 1:, :] = target[:, :-1, :]
            inputs[:, 0, 0] = TEXT_PAD
            text_logits, audio_logits = model(inputs, target)
            text_logits = text_logits.float()
            text_target = target[:, :, 0]
            mask = text_target != TEXT_PAD
            if mask.any():
                text_loss += float(F.cross_entropy(text_logits[mask], text_target[mask])) * int(mask.sum())
                text_correct += int((text_logits.argmax(-1)[mask] == text_target[mask]).sum())
                text_total += int(mask.sum())
            predicted_codes.append(audio_logits.argmax(-1)[0].cpu().numpy())
            for cb in range(8):
                logits = audio_logits[:, :, cb].float()
                y = target[:, :, 1 + cb]
                sums[cb] += float(F.cross_entropy(
                    logits.reshape(-1, CODEBOOK_SIZE), y.reshape(-1)
                )) * y.numel()
                correct[cb] += int((logits.argmax(-1) == y).sum())
            count += target.shape[1]

    result = {
        "checkpoint": str(args.checkpoint),
        "frames": count,
        "text_ce": round(text_loss / max(1, text_total), 4),
        "text_acc": round(text_correct / max(1, text_total), 4),
        "audio_ce": [round(float(sums[cb] / max(1, count)), 4) for cb in range(8)],
        "audio_acc": [round(float(correct[cb] / max(1, count)), 4) for cb in range(8)],
    }
    print(json.dumps(result, indent=2))
    (args.checkpoint.parent / "eval.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.audio_out is not None and predicted_codes:
        codes = np.concatenate(predicted_codes, axis=0)
        mimi = load_mimi(args.mimi, args.device)
        audio = decode_chunks(mimi, codes)
        args.audio_out.parent.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        sf.write(args.audio_out, audio, SAMPLE_RATE)
        print(f"wrote {args.audio_out} ({len(audio)/SAMPLE_RATE:.1f}s)")


if __name__ == "__main__":
    main()
