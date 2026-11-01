#!/usr/bin/env python3
"""Merge a PEFT/LoRA adapter into its base checkpoint for serving.

    python scripts/merge_adapter.py \
        --base checkpoints/stage1_proto \
        --adapter checkpoints/stage2_proto/adapter \
        --out checkpoints/stage2_400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src import model as model_lib


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="checkpoints/stage1_proto")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"{adapter} has no adapter_config.json")
    model = model_lib.load_model(
        args.base,
        quantize="none",
        device=args.device,
        local_files_only=args.local_files_only,
    )
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    model.eval()
    processor = model_lib.load_processor(args.base, local_files_only=args.local_files_only)
    tokenizer = processor.tokenizer
    control = model_lib.control_ids_from_checkpoint(args.base, tokenizer)
    model_lib.save_prepared(
        model,
        tokenizer,
        control,
        args.out,
        base_model=str(args.base),
        extra={
            "stage": 2,
            "mode": "lora-merged",
            "source_adapter": str(adapter),
            "source_step": 400,
        },
    )
    print(f"merged {adapter} into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
