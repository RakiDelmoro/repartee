"""Stage 1: adapter warm-up (connector + four control-token embedding rows).

Blueprint section 10 / stage 1: 200 h mix, 4096-frame windows, 8 sequences per
GPU x 8 GPUs x grad-accum 2, LR 1e-3 with 500 warmup steps, 3,000-6,000 steps,
bf16 + activation checkpointing.  The frozen encoder/codec/depth decoder and
``LiquidCodec`` are never updated.

The connector is the checkpoint's own ``model.model.audio_projector``; no module
is added.  Stage-1 trainable parameters are asserted below 2 % of the model.

``--smoke`` runs two CPU steps on synthetic token-shard items with a toy model,
so it needs neither the checkpoint nor the token shards.

Deviations: see ``src/model.py`` (row-masked embedding; effective parameter
count) and ``src/train_common.py`` (codec decode cost, fallback shard reader).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch

from . import model as model_lib
from .train_common import (
    DuplexBatchBuilder,
    DistributedContext,
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
    make_precision,
    make_smoke_components,
    open_shard_dataset,
    processor_source,
    require_dir,
    resolve_resume,
    save_checkpoint,
    unwrap_model,
    wrap_ddp,
)

PROG = "python -m src.train_stage1"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", default="data/tokens", help="RPT1 token-shard directory")
    parser.add_argument("--model", default=model_lib.BASE_MODEL, help="base checkpoint or prepared stage-0 dir")
    parser.add_argument("--processor", default=None, help="directory with the LFM3 processor (default: model dir or base)")
    parser.add_argument("--codec", default=None, help="checkpoint containing audio_tokenizer/ (default: processor source)")
    parser.add_argument("--codec-device", default=None, help="device for codec decode (default: codec module device)")
    parser.add_argument("--out", default="checkpoints/stage1", help="checkpoint/metrics output directory")
    parser.add_argument("--job", default="local", help="job label recorded in metrics (scheduler-agnostic)")
    parser.add_argument("--seq-frames", type=int, default=4096, help="frames per sequence (4096 ~ 5.5 min)")
    parser.add_argument("--batch-size", type=int, default=8, help="sequences per GPU")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3, help="control-token LR (blueprint: 1e-3)")
    parser.add_argument("--connector-lr", type=float, default=None, help="connector LR (default: --lr)")
    parser.add_argument("--audio-weight", type=float, default=1.0, help="audio CE multiplier for audio-preserving retraining")
    parser.add_argument("--distill-base", default=None, help="frozen base checkpoint for audio-hidden distillation")
    parser.add_argument("--distill-weight", type=float, default=1.0, help="weight for the audio-hidden distillation loss")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="never applied to embeddings")
    parser.add_argument("--warmup", type=int, default=500, help="warmup optimizer steps")
    parser.add_argument("--steps", type=int, default=4000, help="blueprint range: 3000-6000")
    parser.add_argument("--epochs", type=int, default=1, help="unused once --steps is reached; kept for resume symmetry")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="blueprint requires activation checkpointing (default on)",
    )
    parser.add_argument("--resume", default=None, help="checkpoint .pt, state directory, or 'auto' for the newest latest.pt")
    parser.add_argument("--save-every", type=int, default=500, help="optimizer steps between state checkpoints (0 = off)")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--windows-per-item", type=int, default=4, help="windows sampled per conversation per epoch")
    parser.add_argument("--text-jsonl", default=None, help="optional plain-text JSONL for regression batches")
    parser.add_argument("--text-ratio", type=float, default=0.0, help="fraction of optimizer steps that are text-only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="two CPU steps on toy synthetic data; no downloads")
    return parser


def _resolve_precision(name: str, device: torch.device):
    if name == "fp32":
        return make_precision(device, "auto") if device.type != "cuda" else _force_fp32(device)
    return make_precision(device, name)


def _force_fp32(device: torch.device):
    from .train_common import PrecisionContext

    return PrecisionContext(device_type=device.type, dtype=None, scaler=None, name="fp32")


def _audio_latents(hidden, input_ids, audio_labels, audio_out_token_id):
    """Hidden states immediately before each supervised agent-audio slot."""
    placeholders = input_ids == int(audio_out_token_id)
    preceding = placeholders.clone()
    preceding[:, 0] = False
    rows = (audio_labels != -100).all(-1)[preceding[placeholders]]
    if not bool(rows.any()):
        return hidden.new_zeros((0, hidden.shape[-1]))
    return hidden[preceding.roll(-1, dims=1)][rows]


def _load_distill_teacher(path: str, device: torch.device, local_only: bool):
    from transformers import AutoModelForMultimodalLM, BitsAndBytesConfig

    if device.type != "cuda":
        raise SystemExit("--distill-base requires a CUDA device")
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    teacher = AutoModelForMultimodalLM.from_pretrained(
        path,
        quantization_config=config,
        device_map={"": str(device)},
        trust_remote_code=True,
        local_files_only=local_only,
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _make_distill_loss(teacher, control, audio_out_token_id: int, weight: float):
    import torch.nn.functional as F

    control_ids = list(control.by_label().values())

    def extra_loss(outputs, batch):
        ids = batch.input_ids.clone()
        for control_id in control_ids:
            ids[ids == control_id] = int(control.pad)
        kwargs = {
            "input_ids": ids,
            "attention_mask": batch.attention_mask,
        }
        if batch.input_features is not None:
            kwargs["input_features"] = batch.input_features
            kwargs["feature_attention_mask"] = batch.feature_attention_mask
        if batch.audio_out_codes is not None:
            kwargs["audio_out_codes"] = batch.audio_out_codes
        kwargs["output_hidden_states"] = True
        with torch.no_grad():
            teacher_outputs = teacher(**kwargs)
        student_latents = _audio_latents(
            outputs.hidden_states[-1], batch.input_ids, batch.audio_labels, audio_out_token_id
        )
        teacher_latents = _audio_latents(
            teacher_outputs.hidden_states[-1], batch.input_ids, batch.audio_labels, audio_out_token_id
        )
        if student_latents.numel() == 0 or teacher_latents.numel() == 0:
            return None
        similarity = F.cosine_similarity(
            student_latents.float(), teacher_latents.float(), dim=-1
        ).mean()
        return similarity.new_tensor(weight) * (1.0 - similarity)

    return extra_loss


def run_smoke(args: argparse.Namespace) -> int:
    device = torch.device("cpu")
    ctx = DistributedContext(device=device)
    components = make_smoke_components(
        frames=16,
        batch_size=max(1, min(args.batch_size, 2)),
        seed=args.seed,
    )
    control_ids = components["control_ids"]
    model = components["model"]
    optimizer = build_optimizer(model, lr=args.lr)
    scheduler = WarmupCosineScheduler(optimizer, total_steps=2, warmup_steps=1)
    precision = _resolve_precision("fp32", device)
    smoke_dir = Path(args.out) / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    metrics = JsonlMetrics(smoke_dir / "metrics.jsonl")
    trainer = SupervisedTrainer(
        model=model,
        ctx=ctx,
        optimizer=optimizer,
        scheduler=scheduler,
        precision=precision,
        control_ids=control_ids,
        batch_factory=components["batch_factory"],
        metrics=metrics,
        grad_accum=1,
        log_every=1,
    )
    result = trainer.fit(total_steps=2)
    loss_text = (smoke_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    if not loss_text:
        raise SystemExit("stage1 smoke failed: no metrics were written")
    print(f"stage1 smoke ok: {result['step']} steps, {len(loss_text)} metric records, device=cpu")
    return 0


def _dtype_for_precision(name: str, device: torch.device) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise SystemExit("bf16 was requested but this GPU does not support it")
        return torch.bfloat16
    return model_lib.preferred_dtype(device)


def _load_stage1_model(args: argparse.Namespace, device: torch.device):
    """Load a base or prepared checkpoint, add tokens when needed, freeze, budget-check."""
    from transformers import AutoTokenizer

    local_only = bool(args.local_files_only)
    tokenizer: Any | None = None
    control: model_lib.ControlTokenIds | None = None
    is_local = Path(args.model).is_dir()
    if is_local:
        metadata = model_lib.read_repartee_metadata(args.model)
        if metadata is not None:
            tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=local_only, trust_remote_code=True)
            control = model_lib.ControlTokenIds.from_dict(metadata["control_ids"])
            for label, token in zip(model_lib.CONTROL_LABELS, model_lib.CONTROL_TOKENS):
                if tokenizer.convert_tokens_to_ids(token) != control.by_label()[label]:
                    raise SystemExit(f"{args.model}: repartee.json and tokenizer disagree about {token}")
    model = model_lib.load_model(
        args.model,
        quantize="none",
        device=device,
        dtype=_dtype_for_precision(args.precision, device),
        local_files_only=local_only,
    )
    if tokenizer is None:
        processor = model_lib.load_processor(args.model, local_files_only=local_only)
        tokenizer = processor.tokenizer
        control = model_lib.add_control_tokens(tokenizer, model)
    assert control is not None and tokenizer is not None
    plan = model_lib.configure_trainability(model, 1, control)
    summary = model_lib.assert_stage1_budget(model)
    return model, tokenizer, control, plan, summary


def run_real(args: argparse.Namespace) -> int:
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = init_distributed(seed=args.seed)
    if args.device:
        ctx = DistributedContext(rank=ctx.rank, world_size=ctx.world_size, local_rank=ctx.local_rank, device=device)
    try:
        tokens_dir = require_dir(
            args.tokens,
            what="Token shard directory",
            hint="Build it with `make data` (src/tokenize.py) or pass --tokens.",
        )
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        model, tokenizer, control, plan, summary = _load_stage1_model(args, ctx.device)
        if ctx.is_main:
            print(f"stage1: {args.job} | device={ctx.device} | world_size={ctx.world_size}")
            model_lib.print_trainability(model, control)
            print(f"stage-1 budget: {summary['ratio']:.4%} of total parameters (< 2%)")

        processor_path = args.processor or processor_source(args.model)
        processor = model_lib.load_processor(processor_path, local_files_only=args.local_files_only)
        feature_extractor = processor.feature_extractor
        codec = load_codec(
            processor,
            source=args.codec or args.processor or args.model,
        )
        for parameter in codec.parameters():
            parameter.requires_grad_(False)
        if args.codec_device:
            codec.to(torch.device(args.codec_device))

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
            enable_activation_checkpointing(model, required=True)
        wrapped = wrap_ddp(model, ctx)
        distill_teacher = None
        extra_loss = None
        if args.distill_base:
            distill_teacher = _load_distill_teacher(args.distill_base, ctx.device, args.local_files_only)
            extra_loss = _make_distill_loss(
                distill_teacher,
                control,
                int(model.config.audio_out_token_id),
                args.distill_weight,
            )
            if ctx.is_main:
                print(f"audio hidden distillation: {args.distill_base} (weight {args.distill_weight})")
        if args.audio_weight != 1.0:
            model.config.audio_loss_weight = float(args.audio_weight)
            if ctx.is_main:
                print(f"stage1 audio loss weight: {args.audio_weight}")
        optimizer = build_optimizer(
            wrapped,
            lr=args.lr,
            connector_lr=args.connector_lr if args.connector_lr is not None else args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = WarmupCosineScheduler(optimizer, total_steps=args.steps, warmup_steps=args.warmup)
        state_dir = out_dir / "state"
        resume_path = resolve_resume(args.resume, state_dir)
        resume_state = None
        if resume_path is not None:
            resume_state = load_checkpoint(
                resume_path, model=wrapped, optimizer=optimizer, scheduler=scheduler, map_location=ctx.device
            )
        precision = _resolve_precision(args.precision, ctx.device)
        if next(model.parameters()).dtype == torch.float16 and precision.scaler is not None:
            # GradScaler cannot unscale fp16 parameter gradients; the weights
            # are already fp16, so use autocast without a scaler.
            precision.scaler = None
            precision.name = "fp16"
        metrics = JsonlMetrics(out_dir / "metrics.jsonl", rank=ctx.rank, append=resume_path is not None)
        metrics.log(
            {
                "event": "start",
                "job": args.job,
                "device": str(ctx.device),
                "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
                "world_size": ctx.world_size,
                "seq_frames": args.seq_frames,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "steps": args.steps,
            }
        )

        def on_save(step: int) -> None:
            # full HF checkpoint in the background of the lean state dict
            model_lib.save_prepared(
                unwrap_model(wrapped),
                tokenizer,
                control,
                out_dir,
                base_model=str(args.model),
                extra={"stage": 1, "step": step, "job": args.job},
            )

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
            extra_loss=extra_loss,
            audio_weight=args.audio_weight,
        )
        result = trainer.fit(total_steps=args.steps, epochs=args.epochs, resume_state=resume_state)
        if ctx.is_main:
            save_checkpoint(
                state_dir,
                model=wrapped,
                optimizer=optimizer,
                scheduler=scheduler,
                step=result["step"],
                epoch=result["epoch"],
            )
            model_lib.save_prepared(
                unwrap_model(wrapped),
                tokenizer,
                control,
                out_dir,
                base_model=str(args.model),
                extra={"stage": 1, "step": result["step"], "job": args.job},
            )
            elapsed = time.time() - started
            print(f"stage1 done: {result['step']} steps in {elapsed / 3600:.2f} h -> {out_dir}")
    finally:
        cleanup_distributed(ctx)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.smoke:
            return run_smoke(args)
        return run_real(args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"stage1: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - CLI boundary: never dump a raw backend traceback
        raise SystemExit(f"stage1: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
