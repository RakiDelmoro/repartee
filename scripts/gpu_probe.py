#!/usr/bin/env python3
"""Measure whether this GPU can hold a Repartee training step.

This is a hardware probe, not a training run.  It loads the prepared or base
checkpoint in fp16, builds a synthetic duplex batch with the real token layout,
runs one forward/backward, and reports peak VRAM.  It never updates weights and
never claims a gate result.

Example:
    python scripts/gpu_probe.py --model checkpoints/repartee-2b-base \
        --frames 64 256 512 --batch 1
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src import model as model_lib
from src.schema import NUM_CODEBOOKS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="checkpoints/repartee-2b-base")
    parser.add_argument("--frames", type=int, nargs="+", default=[64, 256, 512])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--no-backward", action="store_true")
    parser.add_argument("--int4", action="store_true")
    parser.add_argument("--qlora", action="store_true", help="4-bit base + LoRA adapters (stage-2 style)")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("artifacts/gpu_probe.json"))
    return parser.parse_args()


def build_batch(
    model: Any,
    tokenizer: Any,
    *,
    frames: int,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    config = model.config
    control = model_lib.control_token_ids(tokenizer)
    control_ids = control.by_label()
    # One frame contributes [user audio, control, agent audio_out].
    sequence = []
    labels = [-100] * (3 * frames)
    for frame in range(frames):
        sequence.extend([config.audio_token_id, control_ids["SPEAK"], config.audio_out_token_id])
        labels[3 * frame + 1] = control_ids["SPEAK"]
    input_ids = torch.tensor([sequence] * batch, dtype=torch.long, device=device)
    labels_t = torch.tensor([labels] * batch, dtype=torch.long, device=device)
    # 16 kHz, 10 ms hop, stride 8: exactly one encoder token per 80 ms frame.
    input_features = torch.randn(batch, 128, frames * 8, device=device, dtype=dtype) * 0.1
    feature_mask = torch.ones(batch, frames * 8, dtype=torch.long, device=device)
    # The batch builder flattens speech placeholders across rows.
    codes = torch.randint(0, 2017, (batch * frames, NUM_CODEBOOKS), device=device)
    return {
        "input_ids": input_ids,
        "labels": labels_t,
        "input_features": input_features,
        "feature_attention_mask": feature_mask,
        "audio_out_codes": codes,
        "audio_labels": codes.clone(),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this Python environment")
    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("bfloat16 is not supported on this GPU; use float16")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"GPU: {torch.cuda.get_device_name(0)} | free/peak before load: "
          f"{torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB")
    t0 = time.perf_counter()
    model = model_lib.load_model(
        args.model,
        quantize="int4" if args.int4 else "none",
        device="cuda",
        local_files_only=Path(args.model).is_dir(),
    )
    processor = model_lib.load_processor(
        args.model, local_files_only=Path(args.model).is_dir()
    )
    tokenizer = processor.tokenizer
    control = model_lib.control_token_ids(tokenizer)
    lora_info = None
    if args.qlora:
        from src.train_dpo import apply_lora

        model, lora_info = apply_lora(model, r=args.lora_r, alpha=2 * args.lora_r)
        # Train only the connector, the four control rows, and the LoRA adapters.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        for parameter in model_lib.find_audio_projector(base).parameters():
            parameter.requires_grad_(True)
        embedding = base.get_input_embeddings()
        embedding.weight.requires_grad_(True)
        model_lib.install_embedding_row_mask(embedding, list(control.by_label().values()))
        for name, parameter in model.named_parameters():
            if "lora_" in name and "language_model" in name:
                parameter.requires_grad_(True)
        model_lib.freeze_norms(base)
    else:
        model_lib.configure_trainability(model, 1, control)
    checkpoint_target = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(checkpoint_target, "gradient_checkpointing_enable"):
        try:
            checkpoint_target.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            checkpoint_target.gradient_checkpointing_enable()
    model.train()
    load_s = time.perf_counter() - t0
    load_peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"loaded in {load_s:.1f}s, allocated {load_peak:.2f} GiB, "
          f"torch {torch.__version__}")

    results = []
    for frames in args.frames:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            batch = build_batch(
                model, tokenizer, frames=frames, batch=args.batch, device=device, dtype=dtype
            )
            t1 = time.perf_counter()
            if args.no_backward:
                with torch.no_grad():
                    out = model(**batch)
                loss = out.loss
                step_s = time.perf_counter() - t1
                peak = torch.cuda.max_memory_allocated() / 2**30
                record = {
                    "frames": frames,
                    "batch": args.batch,
                    "tokens": int(batch["input_ids"].shape[1]),
                    "status": "forward_ok",
                    "loss": float(loss.detach()) if loss is not None else None,
                    "seconds": step_s,
                    "peak_gib": peak,
                }
            else:
                out = model(**batch)
                loss = out.loss
                loss.backward()
                step_s = time.perf_counter() - t1
                peak = torch.cuda.max_memory_allocated() / 2**30
                record = {
                    "frames": frames,
                    "batch": args.batch,
                    "tokens": int(batch["input_ids"].shape[1]),
                    "status": "forward_backward_ok",
                    "loss": float(loss.detach()),
                    "seconds": step_s,
                    "peak_gib": peak,
                }
            print(json.dumps(record))
        except torch.cuda.OutOfMemoryError as error:
            record = {
                "frames": frames,
                "batch": args.batch,
                "status": "OOM",
                "error": str(error).splitlines()[0],
            }
            print(json.dumps(record))
            torch.cuda.empty_cache()
        results.append(record)
        del batch
        gc.collect()

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": args.dtype,
        "int4": args.int4,
        "qlora": args.qlora,
        "lora_r": args.lora_r if args.qlora else None,
        "lora": lora_info,
        "load_seconds": load_s,
        "load_peak_gib": load_peak,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
