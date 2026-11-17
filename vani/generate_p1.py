"""Free-run P1 sampling: prompt with held-out audio, continue agent mouth."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .data.mimi import CODEBOOKS, SAMPLE_RATE, decode_chunks, load_mimi
from .model import CODEBOOK_SIZE, TEXT_PAD, VaniM

ROOT = Path(__file__).resolve().parents[1]


def sample(logits: torch.Tensor, top_k: int, temperature: float) -> int:
    logits = logits.float() / max(temperature, 1e-5)
    if top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.numel()))
        probs = torch.softmax(values, dim=-1)
        return int(indices[torch.multinomial(probs, 1)])
    return int(logits.argmax())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "p1" / "last.pt")
    ap.add_argument("--grid", type=Path, default=ROOT / "data" / "vani_p0_10h" / "conv_d65ed16a6092.grid.npy")
    ap.add_argument("--qwen", default=str(ROOT / "models" / "Qwen3-0.6B-Base"))
    ap.add_argument("--mimi", default=str(ROOT / "models" / "mimi"))
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "vani_p1_gen")
    ap.add_argument("--prompt-frames", type=int, default=128)
    ap.add_argument("--frames", type=int, default=192)
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = VaniM(args.qwen, dtype=torch.bfloat16).to(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    grid = np.load(args.grid, mmap_mode="r")
    prompt = np.asarray(grid[: args.prompt_frames], dtype=np.int64)
    history = torch.from_numpy(prompt)[None].to(args.device)
    generated = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(args.frames):
            x = history[:, -args.window :]
            hidden = model.temporal_hidden(x)
            text_logits = model.lm_head(hidden[:, -1:])[0, 0]
            text = sample(text_logits, args.top_k, args.temperature)
            h = hidden[0, -1]
            prev = [CODEBOOK_SIZE] * CODEBOOKS
            codes = []
            for cb in range(CODEBOOKS):
                prev_t = torch.tensor(prev, device=args.device, dtype=torch.long)[None]
                depth_in = (
                    model.to_depth(h)[None, None, :]
                    + model.code_embed(prev_t)
                    + model.code_pos.weight[None, :, :]
                )
                depth_out = model.depth(depth_in)
                codes.append(sample(model.audio_head[cb](depth_out[0, cb]), args.top_k, args.temperature))
                prev[cb] = codes[-1]
            frame = torch.zeros((1, 1, 17), dtype=torch.long, device=args.device)
            frame[0, 0, 0] = text
            frame[0, 0, 1 : 1 + CODEBOOKS] = torch.tensor(codes, device=args.device)
            history = torch.cat([history, frame], dim=1)
            generated.append(codes)

    codes = np.asarray(generated, dtype=np.int64)
    mimi = load_mimi(args.mimi, args.device)
    audio = decode_chunks(mimi, codes)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "codes.npy", codes)
    np.save(args.out / "text.npy", history[0, args.prompt_frames :, 0].cpu().numpy())
    sf.write(args.out / "generated.wav", audio, SAMPLE_RATE)
    prompt_audio = decode_chunks(mimi, prompt[:, 1 : 1 + CODEBOOKS])
    sf.write(args.out / "prompt.wav", prompt_audio, SAMPLE_RATE)
    print(
        f"wrote {args.out} codes={codes.shape} "
        f"prompt={len(prompt_audio)/SAMPLE_RATE:.1f}s audio={len(audio)/SAMPLE_RATE:.1f}s"
    )


if __name__ == "__main__":
    main()
