"""Build per-frame current-word sidecars from Open Yap transcripts.

For every word, hold its first token from the word's start frame to its end
frame.  Frames between words carry PAD.  The sidecar is only the *conditioning*
stream; the grid's sparse text column remains the text-prediction target.

    python -m vani.build_word_cond
"""

from __future__ import annotations

import argparse
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from .data.openyap import index

ROOT = Path(__file__).resolve().parents[1]
FRAME_RATE = 12.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=ROOT / "open-yap-1k_2026-11-05_flac_26a4d80b")
    ap.add_argument("--grids", type=Path, default=ROOT / "data" / "vani_p0_45")
    ap.add_argument("--tokenizer", default=str(ROOT / "models" / "Qwen3-0.6B-Base"))
    args = ap.parse_args()

    metas = {}
    for path in sorted(args.grids.glob("conv_*.json")):
        meta = json.loads(path.read_text())
        metas[meta["source"]] = meta
    if not metas:
        raise FileNotFoundError(f"no grid metadata under {args.grids}")
    conversations = [c for c in index(str(args.root)) if c.conversation_id in metas]
    if len(conversations) != len(metas):
        raise RuntimeError(f"indexed {len(conversations)} of {len(metas)} grids")
    by_part: dict[Path, list] = defaultdict(list)
    for conversation in conversations:
        by_part[conversation.part].append(conversation)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    pad = int(tokenizer.pad_token_id)
    total_filled = 0
    total_frames = 0
    for part, items in sorted(by_part.items()):
        with tarfile.open(part) as handle:
            for conversation in items:
                meta = metas[conversation.conversation_id]
                stem = "a" if meta["agent_is_a"] else "b"
                name = (f"conversations/{conversation.conversation_id}/"
                        f"speaker_{stem}_transcript.json")
                words = json.loads(handle.extractfile(name).read())["words"]
                frames = int(meta["frames"])
                word_cond = np.full(frames, pad, dtype=np.int32)
                for entry in words:
                    if entry.get("type") != "word" or not entry.get("word"):
                        continue
                    ids = tokenizer.encode(entry["word"], add_special_tokens=False)
                    if not ids:
                        continue
                    start = max(0, min(frames - 1, round(float(entry["start"]) * FRAME_RATE)))
                    end = round(float(entry["end"]) * FRAME_RATE)
                    end = max(start + 1, min(frames, end))
                    word_cond[start:end] = int(ids[0])
                grid_path = args.grids / f"{conversation.conversation_id}.grid.npy"
                sidecar = grid_path.with_name(
                    grid_path.name.replace(".grid.npy", ".word.npy")
                )
                np.save(sidecar, word_cond)
                filled = int((word_cond != pad).sum())
                total_filled += filled
                total_frames += frames
                print(f"{conversation.conversation_id} frames={frames} "
                      f"word_frames={filled} -> {sidecar}", flush=True)
    print(f"wrote {len(conversations)} sidecars; word frames "
          f"{total_filled}/{total_frames} ({total_filled / max(1, total_frames):.2%})")


if __name__ == "__main__":
    main()
