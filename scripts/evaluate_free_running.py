#!/usr/bin/env python3
"""Free-running duplex evaluation: the model sees its own generated agent frames.

Teacher-forced evaluation feeds the ground-truth agent audio codes at every
frame.  That can hide a policy that only works when the future is correct.
This script drives the real serving engine over a held-out user WAV and
compares the emitted control tokens with the reference labels from the shard.

Example:
    python scripts/evaluate_free_running.py \
        --model checkpoints/stage2_400 \
        --user artifacts/offline_test/user_90s.wav \
        --tokens data/tokens_proto --conversation conv_0c917fc63b3b \
        --frames 1125 --out artifacts/eval_free_running.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from serve.loop import Lfm3DuplexEngine, WavFileSource
from src.dataset import ShardIndex
from src.schema import CONTROL_LABELS
from src.tokenize import read_record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--user", type=Path, required=True, help="16 kHz mono user WAV")
    parser.add_argument("--tokens", type=Path, default=Path("data/tokens_proto"))
    parser.add_argument("--conversation", required=True)
    parser.add_argument("--frames", type=int, default=1125)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval_free_running.json"))
    parser.add_argument("--audio-out", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def _dtype(name: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _macro_f1(predicted: list[str], reference: list[str]) -> tuple[dict, float]:
    classes = list(CONTROL_LABELS)
    stats = {name: {"frames": 0, "correct": 0, "predicted": 0, "recall": 0.0, "precision": 0.0, "f1": 0.0} for name in classes}
    for pred, ref in zip(predicted, reference):
        if ref not in stats:
            continue
        stats[ref]["frames"] += 1
        if pred == ref:
            stats[ref]["correct"] += 1
        if pred in stats:
            stats[pred]["predicted"] += 1
    f1s = []
    for name, row in stats.items():
        row["recall"] = row["correct"] / row["frames"] if row["frames"] else 0.0
        row["precision"] = row["correct"] / row["predicted"] if row["predicted"] else 0.0
        row["f1"] = (
            2 * row["precision"] * row["recall"] / (row["precision"] + row["recall"])
            if row["precision"] + row["recall"]
            else 0.0
        )
        if row["frames"]:
            f1s.append(row["f1"])
    return stats, sum(f1s) / len(f1s) if f1s else 0.0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    index = ShardIndex.load(args.tokens)
    entry = next((item for item in index.conversations if item.id == args.conversation), None)
    if entry is None:
        raise SystemExit(f"conversation {args.conversation} not found in {args.tokens}")
    record = read_record(
        args.tokens / entry.shard,
        entry.offset,
        n_codebooks=index.n_codebooks,
        frames=entry.T,
    )
    reference = [CONTROL_LABELS[int(value)] for value in record["control"][: args.frames]]

    engine = Lfm3DuplexEngine(
        args.model,
        device=args.device,
        dtype=_dtype(args.dtype),
    )
    engine.reset()
    source = WavFileSource(args.user)
    predicted: list[str] = []
    audio: list[np.ndarray] = []
    started = time.perf_counter()
    for tick in range(args.frames):
        frame = source.read()
        if frame is None:
            break
        output = engine.step(frame, tick)
        predicted.append(output.control)
        if output.audio_24k is not None and len(output.audio_24k):
            audio.append(output.audio_24k)
    engine.end_utterance()
    elapsed = time.perf_counter() - started
    if args.audio_out is not None and audio:
        import soundfile as sf

        args.audio_out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.audio_out, np.concatenate(audio).astype(np.float32), 24000, subtype="PCM_16")

    n = min(len(predicted), len(reference))
    predicted, reference = predicted[:n], reference[:n]
    correct = sum(p == r for p, r in zip(predicted, reference))
    per_class, macro = _macro_f1(predicted, reference)
    histogram = {name: predicted.count(name) for name in CONTROL_LABELS}
    reference_histogram = {name: reference.count(name) for name in CONTROL_LABELS}
    result = {
        "status": "ok",
        "model": str(args.model),
        "conversation": args.conversation,
        "frames": n,
        "seconds": elapsed,
        "rtf": elapsed / (n * 0.08) if n else None,
        "control_accuracy": correct / n if n else 0.0,
        "control_macro_f1": macro,
        "predicted_histogram": histogram,
        "reference_histogram": reference_histogram,
        "per_class": per_class,
        "predicted": predicted,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "predicted"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
