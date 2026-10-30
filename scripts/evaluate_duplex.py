#!/usr/bin/env python3
"""Held-out duplex evaluation for a Repartee checkpoint.

Computes total/text/control/audio loss and control-token accuracy/F1 on the
token-shard ``test`` split.  This is the acceptance check between stages:

    python scripts/evaluate_duplex.py --model checkpoints/stage1_proto \
        --tokens data/tokens_proto --split test --windows 64 \
        --out artifacts/eval_stage1.json

Use ``--baseline artifacts/eval_base.json`` to require a measurable improvement
before moving to the next stage.

This is a behavioural checkpoint check, not FDB and not a text benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from src import model as model_lib
from src.dataset import DuplexShardDataset, ShardIndex
from src.train_common import (
    DuplexBatchBuilder,
    causal_shift,
    control_loss,
    load_codec,
    make_precision,
    text_loss,
)
from src.schema import CONTROL_LABELS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="full HF checkpoint directory")
    parser.add_argument("--tokens", default="data/tokens_proto")
    parser.add_argument("--split", default="test")
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--seq-frames", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=("auto", "fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--codec-device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval_duplex.json"))
    parser.add_argument("--baseline", type=Path, default=None, help="prior evaluation JSON to beat")
    parser.add_argument("--min-loss-improvement", type=float, default=0.05)
    parser.add_argument("--max-text-regression", type=float, default=0.05)
    parser.add_argument("--min-control-accuracy", type=float, default=0.80)
    parser.add_argument("--min-control-macro-f1", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def _model_kwargs(batch) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
    }
    if batch.input_features is not None:
        kwargs["input_features"] = batch.input_features
        kwargs["feature_attention_mask"] = batch.feature_attention_mask
    if batch.audio_out_codes is not None:
        kwargs["audio_out_codes"] = batch.audio_out_codes
    if batch.audio_labels is not None:
        kwargs["audio_labels"] = batch.audio_labels
    return kwargs


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda" and args.codec_device == "cuda":
        torch.cuda.empty_cache()
    model = model_lib.load_model(
        args.model,
        quantize="none",
        device=device,
        local_files_only=args.local_files_only,
    )
    processor = model_lib.load_processor(args.model, local_files_only=args.local_files_only)
    tokenizer = processor.tokenizer
    control = model_lib.control_token_ids(tokenizer)
    control_by_id = {int(value): label for label, value in control.by_label().items()}
    codec = load_codec(processor, source=args.model)
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    if args.codec_device:
        codec.to(torch.device(args.codec_device))
    builder = DuplexBatchBuilder(
        codec=codec,
        feature_extractor=processor.feature_extractor,
        control_ids=control.by_label(),
        audio_token_id=int(model.config.audio_token_id),
        audio_out_token_id=int(model.config.audio_out_token_id),
        pad_token_id=int(control.pad),
        codec_device=args.codec_device,
    )
    index = ShardIndex.load(args.tokens)
    dataset = DuplexShardDataset(
        index,
        seq_frames=args.seq_frames,
        split=args.split,
        samples_per_epoch=args.windows,
        seed=args.seed,
    )
    precision = make_precision(device, args.precision)
    model.eval()
    totals: dict[str, float] = {}
    control_total = 0
    correct = 0
    class_total = {label: 0 for label in CONTROL_LABELS}
    class_correct = {label: 0 for label in CONTROL_LABELS}
    class_predicted = {label: 0 for label in CONTROL_LABELS}
    windows = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for index_item in range(len(dataset)):
            item = dataset[index_item]
            batch = builder.build_items([item]).to(device)
            with precision.autocast():
                outputs = model(**_model_kwargs(batch))
                audio_value = getattr(outputs, "audio_loss", None)
                token_loss, text_count = text_loss(outputs.logits, batch.labels, control.by_label())
                cmd_loss, cmd_count = control_loss(outputs.logits, batch.labels, control.by_label())
                audio_scalar = (
                    float(audio_value.detach())
                    if audio_value is not None and batch.audio_labels is not None
                    else 0.0
                )
                total = 3.0 * token_loss + 2.0 * cmd_loss + audio_scalar
                shifted_logits, shifted_labels = causal_shift(outputs.logits, batch.labels)
                control_ids = set(control.by_label().values())
                mask = torch.zeros_like(shifted_labels, dtype=torch.bool)
                for token_id in control_ids:
                    mask |= shifted_labels == int(token_id)
                if bool(mask.any()):
                    predictions = shifted_logits[mask].argmax(dim=-1)
                    targets = shifted_labels[mask]
                    correct += int((predictions == targets).sum())
                    control_total += int(targets.numel())
                    for predicted, target in zip(predictions.tolist(), targets.tolist()):
                        label = control_by_id.get(int(target))
                        if label is None:
                            continue
                        class_total[label] += 1
                        if int(predicted) == int(target):
                            class_correct[label] += 1
                        predicted_label = control_by_id.get(int(predicted))
                        if predicted_label is not None:
                            class_predicted[predicted_label] += 1
            totals["loss"] = totals.get("loss", 0.0) + float(total.detach())
            totals["text_loss"] = totals.get("text_loss", 0.0) + float(token_loss.detach())
            totals["control_loss"] = totals.get("control_loss", 0.0) + float(cmd_loss.detach())
            totals["audio_loss"] = totals.get("audio_loss", 0.0) + audio_scalar
            totals["text_tokens"] = totals.get("text_tokens", 0.0) + float(text_count)
            totals["control_tokens"] = totals.get("control_tokens", 0.0) + float(cmd_count)
            windows += 1
        if device.type == "cuda":
            torch.cuda.synchronize()

    elapsed = time.perf_counter() - started
    means = {key: value / max(1, windows) for key, value in totals.items()}
    per_class: dict[str, Any] = {}
    for label in CONTROL_LABELS:
        total = class_total[label]
        precision_value = class_correct[label] / class_predicted[label] if class_predicted[label] else 0.0
        recall = class_correct[label] / total if total else 0.0
        f1 = 2 * precision_value * recall / (precision_value + recall) if precision_value + recall else 0.0
        per_class[label] = {
            "frames": total,
            "recall": recall,
            "precision": precision_value,
            "f1": f1,
        }
    present = [name for name in CONTROL_LABELS if per_class[name]["frames"]]
    macro_f1 = sum(per_class[name]["f1"] for name in present) / len(present) if present else 0.0
    result: dict[str, Any] = {
        "status": "ok",
        "model": str(args.model),
        "split": args.split,
        "seq_frames": args.seq_frames,
        "windows": windows,
        "seconds": elapsed,
        "seconds_per_window": elapsed / max(1, windows),
        **{f"mean_{key}": value for key, value in means.items()},
        "control_accuracy": correct / control_total if control_total else 0.0,
        "control_macro_f1": macro_f1,
        "thresholds": {
            "min_loss_improvement": args.min_loss_improvement,
            "max_text_regression": args.max_text_regression,
            "min_control_accuracy": args.min_control_accuracy,
            "min_control_macro_f1": args.min_control_macro_f1,
        },
        "per_class": per_class,
    }
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        result["baseline"] = str(args.baseline)
        result["loss_improvement"] = (
            (baseline.get("mean_loss", 0.0) - means["loss"]) / baseline.get("mean_loss", 1.0)
            if baseline.get("mean_loss")
            else None
        )
        result["text_regression"] = (
            (means["text_loss"] - baseline.get("mean_text_loss", 0.0))
            / baseline.get("mean_text_loss", 1.0)
            if baseline.get("mean_text_loss")
            else None
        )
        result["passes"] = bool(
            result["loss_improvement"] is not None
            and result["loss_improvement"] >= args.min_loss_improvement
            and (result["text_regression"] is None or result["text_regression"] <= args.max_text_regression)
            and result["control_accuracy"] >= args.min_control_accuracy
            and result["control_macro_f1"] >= args.min_control_macro_f1
        )
    else:
        result["passes"] = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.windows <= 0:
        raise SystemExit("--windows must be positive")
    result = evaluate(args)
    print(json.dumps(result, indent=2))
    if result.get("passes") is False:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
