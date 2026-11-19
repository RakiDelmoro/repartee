"""Tokenize additional Open Yap conversations until a target hour count.

Existing grids are symlinked into the output directory unchanged; new
conversations get the same 17-stream grid, 80 ms text alignment, and an
.rms.npy voice-activity sidecar.  The held-word sidecars are not built:
the deployable line uses the sparse per-frame text stream.

    python -m vani.build_100h --target-hours 100
"""

from __future__ import annotations

import argparse
import json
import os
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
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "vani_p0_100h")
    ap.add_argument("--existing", type=Path, default=ROOT / "data" / "vani_p0_45")
    ap.add_argument("--target-hours", type=float, default=100.0)
    ap.add_argument("--max-conversations", type=int, default=200)
    ap.add_argument("--mimi", default=str(ROOT / "models" / "mimi"))
    ap.add_argument("--tokenizer", default=str(ROOT / "models" / "Qwen3-0.6B-Base"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    existing_meta = {}
    for path in sorted(args.existing.glob("conv_*.json")):
        meta = json.loads(path.read_text())
        existing_meta[meta["source"]] = meta
    existing_hours = sum(float(m["seconds"]) for m in existing_meta.values()) / 3600
    print(f"existing: {len(existing_meta)} conversations, {existing_hours:.2f} h",
          flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.existing.glob("conv_*.grid.npy")):
        stem = path.name[: -len(".grid.npy")]
        for suffix in (".grid.npy", ".json", ".rms.npy"):
            src = args.existing / f"{stem}{suffix}"
            dst = args.out / f"{stem}{suffix}"
            if src.exists() and not dst.exists():
                os.symlink(src.resolve(), dst)

    model = load_mimi(args.mimi, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    pad_id = tokenizer.pad_token_id
    conversations = [
        c for c in index(str(args.root)) if c.conversation_id not in existing_meta
    ]
    total_hours = existing_hours
    built = 0
    for conversation in conversations:
        if built >= args.max_conversations or total_hours >= args.target_hours:
            break
        try:
            meta = build(model, tokenizer, conversation, args.out, pad_id)
        except Exception as exc:  # a bad/missing tar member must not stop the run
            print(f"skip {conversation.conversation_id}: {exc}", flush=True)
            continue
        grid = args.out / f"{conversation.conversation_id}.grid.npy"
        rms_path = grid.with_name(grid.name.replace(".grid.npy", ".rms.npy"))
        x, _ = sf.read(
            args.out / f"{conversation.conversation_id}.original.wav", dtype="float32"
        )
        frames = int(meta["frames"])
        count = min(len(x) // FRAME_SAMPLES, frames)
        rms = np.sqrt((x[: count * FRAME_SAMPLES].reshape(count, FRAME_SAMPLES) ** 2).mean(1))
        rms = np.pad(rms, (0, frames - len(rms))).astype(np.float32)
        np.save(rms_path, rms)
        built += 1
        total_hours += float(meta["seconds"]) / 3600
        print(f"{built:3d} {conversation.conversation_id} {meta['seconds']/60:.1f} min "
              f"total {total_hours:.1f} h", flush=True)
    print(f"done: {built} new conversations, {total_hours:.2f} h total", flush=True)


if __name__ == "__main__":
    main()
