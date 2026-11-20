"""Tokenize one unseen Open Yap conversation and pick a speakable segment.

The conversation must not be in the 100 h training set.  Writes the grid,
the .rms sidecar, a spaced reference for exactly the target segment, and
selection.json for the gate.

    python -m vani.new_holdout
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
from transformers import AutoTokenizer

from .data.mimi import load_mimi
from .data.openyap import build, index

ROOT = Path(__file__).resolve().parents[1]
FRAME_SAMPLES = 1920


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=ROOT / "open-yap-1k_2026-11-05_flac_26a4d80b")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "vani_p0_newhold")
    ap.add_argument("--existing", type=Path, default=ROOT / "data" / "vani_p0_100h")
    ap.add_argument("--prompt-frames", type=int, default=64)
    ap.add_argument("--frames", type=int, default=192)
    ap.add_argument("--mimi", default=str(ROOT / "models" / "mimi"))
    ap.add_argument("--tokenizer", default=str(ROOT / "models" / "Qwen3-0.6B-Base"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    existing = {
        json.loads(p.read_text())["source"]
        for p in args.existing.glob("conv_*.json")
    }
    conversations = [
        c for c in index(str(args.root)) if c.conversation_id not in existing
    ]
    if not conversations:
        raise RuntimeError("no unseen conversation left")
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_mimi(args.mimi, args.device)
    pad = tokenizer.pad_token_id

    for conversation in conversations:
        meta = build(model, tokenizer, conversation, args.out, pad)
        frames = int(meta["frames"])
        if frames < args.prompt_frames + args.frames:
            continue
        stem = conversation.conversation_id
        grid_path = args.out / f"{stem}.grid.npy"
        grid = np.load(grid_path, mmap_mode="r")
        text = np.asarray(grid[:, 0], dtype=np.int64)
        rms_path = grid_path.with_name(grid_path.name.replace(".grid.npy", ".rms.npy"))
        original = args.out / f"{stem}.original.wav"
        x, _ = sf.read(original, dtype="float32")
        count = min(len(x) // FRAME_SAMPLES, frames)
        rms = np.sqrt((x[: count * FRAME_SAMPLES].reshape(count, FRAME_SAMPLES) ** 2).mean(1))
        rms = np.pad(rms, (0, frames - len(rms))).astype(np.float32)
        np.save(rms_path, rms)

        token_frames = (text != pad).astype(np.float32)
        voiced = (rms > 0.01).astype(np.float32)
        token_sum = np.convolve(token_frames, np.ones(args.frames), "valid")
        voiced_sum = np.convolve(voiced, np.ones(args.frames), "valid")
        lo = args.prompt_frames
        hi = len(text) - args.frames
        if hi <= lo:
            continue
        score = token_sum[lo:hi] * (voiced_sum[lo:hi] / args.frames)
        target_start = lo + int(np.argmax(score))
        start = target_start - args.prompt_frames

        stem_name = "a" if conversation.agent_is_a else "b"
        with tarfile.open(conversation.part) as handle:
            name = (f"conversations/{conversation.conversation_id}/"
                    f"speaker_{stem_name}_transcript.json")
            words = json.loads(handle.extractfile(name).read())["words"]
        lo_s = target_start / 12.5
        hi_s = (target_start + args.frames) / 12.5
        reference = " ".join(
            w["word"] for w in words
            if w.get("type") == "word" and w.get("word")
            and lo_s <= float(w["start"]) < hi_s
        )
        (args.out / "reference.txt").write_text(reference + "\n")
        selection = {
            "source": conversation.conversation_id,
            "grid": str(grid_path),
            "start_frame": start,
            "prompt_frames": args.prompt_frames,
            "frames": args.frames,
            "reference": str(args.out / "reference.txt"),
            "target_start": target_start,
            "reference_words": len(reference.split()),
        }
        (args.out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        print(json.dumps(selection, indent=2), flush=True)
        return
    raise RuntimeError("no unseen conversation long enough")


if __name__ == "__main__":
    main()
