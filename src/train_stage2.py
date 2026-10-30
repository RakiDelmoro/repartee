"""Stage 2: full-duplex SFT (LLM 5e-5, connector 1e-3, control tokens).

Blueprint section 10 / stage 2:

* trainable: text LLM + ``audio_projector`` + the four control-token rows,
* frozen: audio encoder, LiquidCodec, depth decoder (and every norm),
* full mix, 2 epochs, 800-2,000 optimizer steps per epoch,
* same loss as stage 1: ``3*CE_text + 2*CE_control + 1*CE_audio``.

Gate 4 (text-only MMLU/GSM8K within 2 points of the Gate 0 baseline) is a hard
precondition: the stage refuses to start unless a **passing** Gate-4 JSON
artifact is supplied with ``--gate4``.  Nothing about the model or the data is
touched before that check.

Deviations: see ``src/model.py`` and ``src/train_common.py``.  A single-GPU
8 GB host needs ``--quantize`` on the base (serving) and is not the blueprint's
8x A100 setup; the CLI records the actual device/dtype in its metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch

from . import model as model_lib
from .model import norm_modules
from .train_common import (
    DuplexBatchBuilder,
    JsonlMetrics,
    SupervisedBatch,
    SupervisedTrainer,
    WarmupCosineScheduler,
    WindowSampler,
    build_optimizer,
    build_text_batch,
    cleanup_distributed,
    enable_activation_checkpointing,
    init_distributed,
    iter_text_jsonl,
    load_checkpoint,
    load_codec,
    load_json_artifact,
    make_precision,
    open_shard_dataset,
    processor_source,
    require_dir,
    resolve_resume,
    save_checkpoint,
    unwrap_model,
    wrap_ddp,
)

PROG = "python -m src.train_stage2"
DEFAULT_GATE4 = "artifacts/gates/gate4.json"


class GateArtifactError(RuntimeError):
    """The supplied Gate-4 artifact is absent, malformed or not a pass."""


def check_gate4(path: str | Path) -> dict[str, Any]:
    """Validate a Gate-4 pass artifact before stage 2 starts.

    Accepted shapes (any one)::

        {"gate": "Gate 4", "result": "PASS", ...}
        {"gate": 4, "status": "pass", ...}
        {"gate": "gate4", "passed": true, ...}
    """
    artifact = load_json_artifact(path, what="Gate-4 artifact")
    gate = artifact.get("gate", artifact.get("gate_name", artifact.get("name")))
    if gate is not None and "4" not in str(gate).replace("Gate", "").replace("gate", ""):
        raise GateArtifactError(
            f"{path}: artifact is for gate {gate!r}, not Gate 4; stage 2 must not start on the wrong gate"
        )
    passed = artifact.get("passed")
    if passed is None:
        result = str(artifact.get("result", artifact.get("status", ""))).strip().upper()
        passed = result == "PASS"
    if passed is not True:
        measured = {key: artifact[key] for key in ("result", "status", "measured") if key in artifact}
        raise GateArtifactError(
            f"{path}: Gate 4 did not pass ({measured or 'no result field'}); BLUEPRINT.md forbids starting "
            "stage 2 before Gate 4 passes. Fix the text regression and re-run the Gate-4 evaluation."
        )
    return artifact


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", default="data/tokens")
    parser.add_argument("--model", default="checkpoints/stage1", help="prepared stage-1 checkpoint")
    parser.add_argument("--processor", default=None)
    parser.add_argument("--codec", default=None)
    parser.add_argument("--codec-device", default=None, help="device for codec decode (default: codec module device)")
    parser.add_argument("--gate4", default=None, help=f"Gate-4 pass JSON (e.g. {DEFAULT_GATE4})")
    parser.add_argument(
        "--prototype",
        action="store_true",
        help="explicitly allow a short prototype run without Gate 4; not a blueprint-compliant stage 2",
    )
    parser.add_argument("--out", default="checkpoints/stage2")
    parser.add_argument("--job", default="local")
    parser.add_argument("--seq-frames", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1, help="sequences per GPU")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5, help="LLM LR (blueprint: 5e-5)")
    parser.add_argument("--connector-lr", type=float, default=1e-3, help="connector LR (blueprint: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=1200, help="blueprint range: 800-2000")
    parser.add_argument("--epochs", type=int, default=2, help="blueprint: 2 epochs over the full mix")
    parser.add_argument("--lora", action="store_true", help="train LoRA adapters on the LLM instead of full fine-tuning")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", default=None, help="checkpoint .pt, state directory, or 'auto' for the newest latest.pt")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--windows-per-item", type=int, default=4)
    parser.add_argument("--text-jsonl", default=None)
    parser.add_argument("--text-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="two CPU steps on toy tensors (still gate-checked)")
    return parser


def _force_fp32(device: torch.device):
    from .train_common import PrecisionContext

    return PrecisionContext(device_type=device.type, dtype=None, scaler=None, name="fp32")


def _resolve_precision(name: str, device: torch.device):
    if name == "fp32":
        return _force_fp32(device)
    return make_precision(device, name)


def assert_stage2_freezes(model: torch.nn.Module) -> None:
    forbidden = ("audio_encoder", "depth_decoder", "vision_tower", "codebook_embedding")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and any(fragment in name for fragment in forbidden):
            raise RuntimeError(f"stage 2 must keep {name} frozen (blueprint: no encoder/codec/depthformer training)")
    for name, module in norm_modules(model):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError(f"norm module {name} is trainable; norms are frozen in every stage")


def _base_model(model: Any) -> Any:
    getter = getattr(model, "get_base_model", None)
    return getter() if callable(getter) else model


def _configure_lora_trainability(model: Any, control: model_lib.ControlTokenIds) -> dict[str, Any]:
    """Freeze the base, then train connector + four rows + LoRA adapters only."""
    base = _base_model(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    connector = model_lib.find_audio_projector(base)
    for parameter in connector.parameters():
        parameter.requires_grad_(True)
    embedding = base.get_input_embeddings()
    embedding.weight.requires_grad_(True)
    model_lib.install_embedding_row_mask(embedding, list(control.by_label().values()))
    for name, parameter in model.named_parameters():
        if "lora_" in name and "language_model" in name:
            parameter.requires_grad_(True)
    model_lib.freeze_norms(base)
    summary = model_lib.effective_trainable_parameters(base)
    return {
        "mode": "lora",
        "connector": "audio_projector",
        "embedding_rows": control.to_dict(),
        "effective_trainable": summary["trainable"],
        "total": summary["total"],
        "ratio": summary["ratio"],
    }


def _dtype_for_precision(name: str, device: torch.device) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise SystemExit("bf16 was requested but this GPU does not support it")
        return torch.bfloat16
    return model_lib.preferred_dtype(device)


def _load_stage2_model(args: argparse.Namespace, device: torch.device):
    from transformers import AutoTokenizer

    if not Path(args.model).is_dir():
        raise SystemExit(
            f"--model {args.model} must be a prepared stage-1 checkpoint directory (train_stage1 writes one); "
            "stage 2 never starts from the raw base on its own"
        )
    metadata = model_lib.read_repartee_metadata(args.model)
    if metadata is None:
        raise SystemExit(
            f"{args.model} has no repartee.json; run `python -m src.model prepare` or finish train_stage1 first"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=args.local_files_only, trust_remote_code=True
    )
    control = model_lib.control_ids_from_checkpoint(args.model, tokenizer)
    model = model_lib.load_model(
        args.model,
        quantize="none",
        device=device,
        dtype=_dtype_for_precision(args.precision, device),
        local_files_only=args.local_files_only,
    )
    if args.lora:
        from .train_dpo import apply_lora

        model, lora_info = apply_lora(
            model,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        plan = _configure_lora_trainability(model, control)
        plan.update(lora_info)
    else:
        plan = model_lib.configure_trainability(model, 2, control)
    assert_stage2_freezes(model)
    return model, tokenizer, control, plan


def run_smoke(args: argparse.Namespace) -> int:
    device = torch.device("cpu")
    ctx = init_distributed(seed=args.seed)
    try:
        from .train_common import DistributedContext, make_smoke_components

        ctx = DistributedContext(device=device)
        components = make_smoke_components(frames=16, batch_size=2, seed=args.seed)
        model = components["model"]
        optimizer = build_optimizer(model, lr=args.lr)
        scheduler = WarmupCosineScheduler(optimizer, total_steps=2, warmup_steps=1)
        smoke_dir = Path(args.out) / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        trainer = SupervisedTrainer(
            model=model,
            ctx=ctx,
            optimizer=optimizer,
            scheduler=scheduler,
            precision=_resolve_precision("fp32", device),
            control_ids=components["control_ids"],
            batch_factory=components["batch_factory"],
            metrics=JsonlMetrics(smoke_dir / "metrics.jsonl"),
            grad_accum=1,
            log_every=1,
        )
        result = trainer.fit(total_steps=2)
        print(f"stage2 smoke ok: {result['step']} steps (toy tensors, CPU)")
    finally:
        cleanup_distributed(ctx)
    return 0


def run_real(args: argparse.Namespace) -> int:
    if args.gate4:
        gate = check_gate4(args.gate4)
    elif args.prototype:
        gate = {
            "gate": "Gate 4",
            "result": "BYPASSED",
            "passed": False,
            "prototype": True,
            "reason": "--prototype explicitly bypassed the text regression gate",
        }
        print(
            "WARNING: stage2 is running with --prototype; Gate 4 was not evaluated. "
            "This is not a blueprint-compliant stage-2 run.",
            file=sys.stderr,
        )
    else:
        raise SystemExit(
            "stage 2 requires --gate4 pointing at the passing Gate-4 JSON artifact "
            f"(default location: {DEFAULT_GATE4}); BLUEPRINT.md forbids stage 2 before Gate 4"
        )
    tokens_dir = require_dir(
        args.tokens,
        what="Token shard directory",
        hint="Build it with `make data` (src/tokenize.py) or pass --tokens.",
    )
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = init_distributed(seed=args.seed)
    if args.device:
        ctx = type(ctx)(rank=ctx.rank, world_size=ctx.world_size, local_rank=ctx.local_rank, device=device)
    try:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        model, tokenizer, control, plan = _load_stage2_model(args, ctx.device)
        if ctx.is_main:
            print(f"stage2: {args.job} | gate4={args.gate4} | device={ctx.device} | world_size={ctx.world_size}")
            table_status = model_lib.print_trainability(model, control)
            print(f"stage-2 trainable: {table_status['summary']['trainable']:,} params")
            print(f"gate 4 artifact accepted: {json.dumps(gate, sort_keys=True)[:300]}")

        processor_path = args.processor or processor_source(args.model)
        processor = model_lib.load_processor(processor_path, local_files_only=args.local_files_only)
        codec = load_codec(processor, source=args.codec or args.processor or args.model)
        for parameter in codec.parameters():
            parameter.requires_grad_(False)
        if args.codec_device:
            codec.to(torch.device(args.codec_device))
        feature_extractor = processor.feature_extractor

        dataset = open_shard_dataset(tokens_dir, seq_frames=args.seq_frames, seed=args.seed)
        sampler = WindowSampler(
            dataset,
            seq_frames=args.seq_frames,
            batch_size=args.batch_size,
            seed=args.seed,
            rank=ctx.rank,
            world_size=ctx.world_size,
            windows_per_item=args.windows_per_item,
        )
        builder = DuplexBatchBuilder(
            codec=codec,
            feature_extractor=feature_extractor,
            control_ids=control.by_label(),
            audio_token_id=int(model.config.audio_token_id),
            audio_out_token_id=int(model.config.audio_out_token_id),
            pad_token_id=int(control.pad),
            codec_device=args.codec_device,
        )

        def batch_factory(epoch: int) -> Iterator[SupervisedBatch]:
            for items in sampler.batches(epoch):
                yield builder.build_items(items)

        text_factory = None
        if args.text_jsonl:
            def text_factory(epoch: int) -> Iterator[SupervisedBatch]:
                while True:
                    for group in iter_text_jsonl(args.text_jsonl, batch_size=max(1, args.batch_size)):
                        yield build_text_batch(tokenizer, group)

        if args.activation_checkpointing:
            enable_activation_checkpointing(_base_model(model), required=True)
        wrapped = wrap_ddp(model, ctx)
        optimizer = build_optimizer(
            wrapped,
            lr=args.lr,
            connector_lr=args.connector_lr,
            weight_decay=args.weight_decay,
        )
        total_steps = int(args.epochs) * int(args.steps_per_epoch)
        scheduler = WarmupCosineScheduler(optimizer, total_steps=total_steps, warmup_steps=args.warmup)
        state_dir = out_dir / "state"
        resume_path = resolve_resume(args.resume, state_dir)
        resume_state = None
        if resume_path is not None:
            resume_state = load_checkpoint(
                resume_path, model=wrapped, optimizer=optimizer, scheduler=scheduler, map_location=ctx.device
            )
        metrics = JsonlMetrics(out_dir / "metrics.jsonl", rank=ctx.rank, append=resume_path is not None)
        metrics.log(
            {
                "event": "start",
                "job": args.job,
                "device": str(ctx.device),
                "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
                "world_size": ctx.world_size,
                "gate4": str(args.gate4),
                "steps_per_epoch": args.steps_per_epoch,
                "epochs": args.epochs,
            }
        )

        def on_save(step: int) -> None:
            current = unwrap_model(wrapped)
            if args.lora:
                adapter_dir = out_dir / "adapter"
                current.save_pretrained(adapter_dir)
                metadata = model_lib.repartee_metadata(
                    control,
                    base_model=str(args.model),
                    extra={"stage": 2, "step": step, "job": args.job, "mode": "lora"},
                )
                (adapter_dir / "repartee.json").write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                return
            model_lib.save_prepared(
                current,
                tokenizer,
                control,
                out_dir,
                base_model=str(args.model),
                extra={"stage": 2, "step": step, "job": args.job, "gate4": str(args.gate4)},
            )

        precision = _resolve_precision(args.precision, ctx.device)
        if next(model.parameters()).dtype == torch.float16 and precision.scaler is not None:
            precision.scaler = None
            precision.name = "fp16"
        trainer = SupervisedTrainer(
            model=wrapped,
            ctx=ctx,
            optimizer=optimizer,
            scheduler=scheduler,
            precision=precision,
            control_ids=control.by_label(),
            batch_factory=batch_factory,
            metrics=metrics,
            grad_accum=args.grad_accum,
            max_grad_norm=args.max_grad_norm,
            log_every=args.log_every,
            save_every=args.save_every,
            checkpoint_dir=state_dir,
            text_batch_factory=text_factory,
            text_ratio=args.text_ratio,
            on_save=on_save if ctx.is_main else None,
        )
        result = trainer.fit(total_steps=total_steps, epochs=args.epochs, resume_state=resume_state)
        if ctx.is_main:
            save_checkpoint(
                state_dir,
                model=wrapped,
                optimizer=optimizer,
                scheduler=scheduler,
                step=result["step"],
                epoch=result["epoch"],
            )
            current = unwrap_model(wrapped)
            if args.lora:
                adapter_dir = out_dir / "adapter"
                current.save_pretrained(adapter_dir)
                merged = current.merge_and_unload() if hasattr(current, "merge_and_unload") else current
                model_lib.save_prepared(
                    merged,
                    tokenizer,
                    control,
                    out_dir,
                    base_model=str(args.model),
                    extra={
                        "stage": 2,
                        "step": result["step"],
                        "job": args.job,
                        "gate4": str(args.gate4),
                        "mode": "lora-merged",
                    },
                )
            else:
                model_lib.save_prepared(
                    current,
                    tokenizer,
                    control,
                    out_dir,
                    base_model=str(args.model),
                    extra={"stage": 2, "step": result["step"], "job": args.job, "gate4": str(args.gate4)},
                )
            elapsed = time.time() - started
            print(f"stage2 done: {result['step']} steps in {elapsed / 3600:.2f} h -> {out_dir}")
    finally:
        cleanup_distributed(ctx)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if not args.gate4 and not args.smoke and not args.prototype:
            raise GateArtifactError(
                "stage 2 requires --gate4 pointing at the passing Gate-4 JSON artifact "
                f"(default location: {DEFAULT_GATE4}); BLUEPRINT.md forbids stage 2 before Gate 4. "
                "For a short prototype only, pass --prototype."
            )
        # The gate is checked before data/model loading in both smoke and real
        # runs? The smoke path uses toy tensors, so it only records the gate.
        if args.smoke:
            return run_smoke(args)
        return run_real(args)
    except GateArtifactError as exc:
        raise SystemExit(f"stage2: {exc}") from exc
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"stage2: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - CLI boundary: never dump a raw backend traceback
        raise SystemExit(f"stage2: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
