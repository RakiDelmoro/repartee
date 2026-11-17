"""P0: Open Yap -> Mimi 17-stream grid, with decode round-trip.

    python -m vani.p0 --root open-yap-1k_2026-11-05_flac_26a4d80b --limit 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .data.mimi import load_mimi
from .data.openyap import build, index

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "open-yap-1k_2026-11-05_flac_26a4d80b")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "vani_p0")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mimi", default=str(ROOT / "models" / "mimi"))
    parser.add_argument("--tokenizer", default=str(ROOT / "models" / "Qwen3-0.6B-Base"))
    args = parser.parse_args()

    model = load_mimi(args.mimi, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    conversations = index(args.root)
    print(f"{len(conversations)} conversations indexed", flush=True)

    results = []
    for conversation in conversations[: args.limit]:
        meta = build(model, tokenizer, conversation, args.out, tokenizer.pad_token_id)
        results.append(meta)
        print(json.dumps(meta), flush=True)
    (args.out / "p0.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
