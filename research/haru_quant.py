#!/usr/bin/env python3
"""Run Haru-4B with a quantized LM backbone, fp16 audio path.

The 10.5 GB checkpoint is split: Qwen3.5-4B backbone (``lm.*``), token embedding,
MeanFlow head, audio projections and the VibeVoice codec.  Only the LM Linears are
quantized (bitsandbytes NF4 by default); everything that turns hidden states into
audio stays fp16, and the tied token embedding stays fp16 because the text logits
are computed from its weight directly.

    .venv/bin/python research/haru_quant.py --model models/Haru-4B --frames 40

Reports load time, VRAM, ms/frame and the model's text, and writes the model audio.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from haru import DuplexConfig, DuplexSession, load_model  # noqa: E402
from haru.audio import load_audio, prepare_reference, save_audio  # noqa: E402

SR = 24_000
FRAME = 3_200
SYSTEM_PROMPT = (
    "You are the person described below, talking with the other person by voice.\n"
    "[Your Persona]\nA calm, concise assistant who answers in short sentences.\n"
    "[Your Name]\nAri\n[Your Gender]\nmale"
)


def replace_lm_linears(model, compute_dtype: torch.dtype, bits: int) -> int:
    """Replace every Linear under ``lm`` with a bitsandbytes quantized module.

    The tied token embedding and any lm_head stay float: Haru computes text logits
    from the embedding weight directly, so quantizing it would put dequantization
    on the per-frame hot path and lose rare-token fidelity.
    """
    if bits == 4:
        from bitsandbytes.nn import Linear4bit as QuantLinear
    elif bits == 8:
        from bitsandbytes.nn import Linear8bitLt as QuantLinear
    else:
        raise ValueError("bits must be 4 or 8")

    replaced = 0
    for name, module in list(model.lm.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, QuantLinear):
            continue
        if name.endswith("embed_tokens") or name.endswith("lm_head"):
            continue
        if bits == 4:
            kwargs = dict(compute_dtype=compute_dtype, quant_type="nf4", compress_statistics=True)
        else:
            kwargs = dict(has_fp16_weights=False, threshold=6.0)
        quantized = QuantLinear(
            module.in_features, module.out_features, bias=module.bias is not None, **kwargs
        )
        quantized.load_state_dict(module.state_dict())
        parent = model.lm.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model.lm
        setattr(parent, name.rsplit(".", 1)[-1], quantized)
        replaced += 1
    model.requires_grad_(False)
    return replaced


def cast_to(model, dtype: torch.dtype) -> None:
    """Cast everything except quantized weights to ``dtype`` (fp16 on Turing)."""
    from bitsandbytes.nn import Linear4bit, Linear8bitLt

    for module in model.modules():
        if isinstance(module, (Linear4bit, Linear8bitLt)):
            if getattr(module, "bias", None) is not None:
                module.bias.data = module.bias.data.to(dtype)
            continue
        for parameter in module.parameters(recurse=False):
            if parameter.is_floating_point():
                parameter.data = parameter.data.to(dtype)
        # Buffers (rotary freqs, latent scale/bias, codec states) stay float32:
        # the model upcasts them explicitly and they set numeric scale.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "Haru-4B"))
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "haru_quant")
    parser.add_argument("--reference", type=Path, default=ROOT / "speaker_a.flac")
    parser.add_argument("--reference-seconds", type=float, default=4.0)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--bits", type=int, default=4, choices=(4, 8))
    parser.add_argument("--codec-backend", default="torch", choices=("torch", "triton", "auto"))
    parser.add_argument("--profile", action="store_true", help="print a CUDA kernel profile of the frame loop")
    parser.add_argument("--compile", default="", choices=("", "default", "reduce-overhead"),
                        help="torch.compile the model forward")
    parser.add_argument("--backend", default="bnb", choices=("bnb", "int8"),
                        help="bnb: bitsandbytes; int8: torch packed-int8 weight-only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--float-dtype", default="bf16", choices=("bf16", "fp16"),
                        help="dtype of the non-quantized parts and bnb compute")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--speak-first", action="store_true", help="force the model to open the conversation")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    # The checkpoint is bf16; quantized modules need a float weight to load into.
    model = load_model(args.model, device="cpu", dtype=torch.bfloat16)
    load_s = time.perf_counter() - started
    float_dtype = torch.bfloat16 if args.float_dtype == "bf16" else torch.float16
    if args.backend == "int8":
        from int8_linear import replace_lm_linears_int8

        replaced = replace_lm_linears_int8(model)
        args.bits = 8
    else:
        replaced = replace_lm_linears(model, float_dtype, args.bits)
    cast_to(model, float_dtype)
    model = model.to(args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()
    ready_s = time.perf_counter() - started

    if args.compile:
        model.forward = torch.compile(model.forward, mode=args.compile, dynamic=True, fullgraph=False)
    voice = prepare_reference(load_audio(args.reference))
    voice = voice[: int(args.reference_seconds * SR)]
    session = DuplexSession(
        model, SYSTEM_PROMPT, voice,
        DuplexConfig(max_frames=args.frames, seed=args.seed, speak_first=args.speak_first,
                     codec_backend=args.codec_backend),
    )
    audio = [session.start()]
    times = []
    profiler = None
    warmup = min(3, max(0, args.frames - 2)) if args.profile else 0
    for _ in range(warmup):
        audio.append(session.step(np.zeros(FRAME, dtype=np.float32)))
    if args.device == "cuda":
        torch.cuda.synchronize()
    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        profiler.start()
    for _ in range(1 + warmup, args.frames):
        tick = time.perf_counter()
        frame = session.step(np.zeros(FRAME, dtype=np.float32))
        if args.device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - tick)
        audio.append(frame)
    if profiler is not None:
        profiler.stop()
        print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    model_audio = np.concatenate([frame.audio for frame in audio])
    text = "".join(frame.text for frame in audio).strip()

    args.out.mkdir(parents=True, exist_ok=True)
    save_audio(args.out / f"model_{args.bits}bit.wav", model_audio, SR)
    summary = {
        "bits": args.bits,
        "replaced_linears": replaced,
        "load_s": round(load_s, 2),
        "ready_s": round(ready_s, 2),
        "frames": len(audio),
        "frame_ms_median": round(1000 * float(np.median(times)), 1) if times else None,
        "frame_ms_p90": round(1000 * float(np.percentile(times, 90)), 1) if times else None,
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if args.device == "cuda" else None,
        "model_dtype": str(model.dtype),
        "text": text,
    }
    (args.out / f"summary_{args.bits}bit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
