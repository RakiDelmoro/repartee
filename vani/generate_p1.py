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
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--prompt-frames", type=int, default=128)
    ap.add_argument("--frames", type=int, default=192)
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--text-mode", choices=["free", "target"], default="free")
    ap.add_argument("--audio-mode", choices=["free", "gt0_1", "gt2_7"], default="free")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = VaniM(args.qwen, dtype=torch.bfloat16).to(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    grid = np.load(args.grid, mmap_mode="r")
    end = args.start_frame + args.prompt_frames + args.frames
    target_grid = np.asarray(grid[args.start_frame:end], dtype=np.int64)
    prompt = target_grid[: args.prompt_frames]
    history = torch.from_numpy(prompt)[None].to(args.device)
    generated = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(args.frames):
            x = history[:, -args.window :]
            hidden = model.temporal_hidden(x)
            text_logits = model.lm_head(hidden[:, -1:])[0, 0]
            if args.text_mode == "target":
                text = int(target_grid[args.prompt_frames + len(generated), 0])
            else:
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
            if args.audio_mode == "gt0_1":
                truth = target_grid[args.prompt_frames + len(generated), 1:3]
                codes[0:2] = [int(v) for v in truth]
            elif args.audio_mode == "gt2_7":
                truth = target_grid[args.prompt_frames + len(generated), 3:9]
                codes[2:8] = [int(v) for v in truth]
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
    text_ids = history[0, args.prompt_frames :, 0].cpu().numpy()
    np.save(args.out / "text.npy", text_ids)
    target_ids = target_grid[args.prompt_frames :, 0]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.qwen)
    (args.out / "target_text.txt").write_text(
        tokenizer.decode(target_ids.tolist(), skip_special_tokens=True).strip() + "\n"
    )
    (args.out / "generated_text.txt").write_text(
        tokenizer.decode(text_ids.tolist(), skip_special_tokens=True).strip() + "\n"
    )
    sf.write(args.out / "generated.wav", audio, SAMPLE_RATE)
    prompt_audio = decode_chunks(mimi, prompt[:, 1 : 1 + CODEBOOKS])
    sf.write(args.out / "prompt.wav", prompt_audio, SAMPLE_RATE)
    print(
        f"wrote {args.out} codes={codes.shape} "
        f"prompt={len(prompt_audio)/SAMPLE_RATE:.1f}s audio={len(audio)/SAMPLE_RATE:.1f}s "
        f"text={args.text_mode} audio={args.audio_mode} start={args.start_frame}"
    )


if __name__ == "__main__":
    main()
