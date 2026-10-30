"""Stage 3: DPO on response timing (LoRA r=16, beta=0.1, LR 5e-6).

The preference pairs share one context (user codes, agent codes, text) and
differ only in the **control-token timing** — e.g. a 200 ms versus a 1,200 ms
response onset, or barge-in versus talk-over.  Pair construction/labelling is
an evaluation artifact (FDB metrics); this module consumes the labelled pairs
and validates that the context really is identical.

The DPO loss uses ``reference = the same model with the LoRA adapter disabled``
(``peft``'s ``disable_adapter()`` context), which is exact and avoids a second
2 B-parameter copy.

``--smoke`` runs two optimizer steps on tiny tensors and needs neither the
checkpoint nor the pairs.

Limitations: LoRA target modules are discovered from the LM's ``nn.Linear``
leaf names (e.g. ``self_attn.q_proj``); after injection every trainable
parameter outside ``model.language_model`` is frozen again so the frozen
encoder/connector cannot receive adapters.  Without ``--pairs`` the stage fails
with an actionable message rather than inventing preference labels.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import model as model_lib
from .train_common import (
    DuplexBatchBuilder,
    JsonlMetrics,
    SupervisedBatch,
    WarmupCosineScheduler,
    build_optimizer,
    cleanup_distributed,
    enable_activation_checkpointing,
    init_distributed,
    load_checkpoint,
    load_codec,
    make_precision,
    open_shard_dataset,
    processor_source,
    require_dir,
    save_checkpoint,
    unwrap_model,
    window_item,
    wrap_ddp,
)

PROG = "python -m src.train_dpo"
DEFAULT_PAIRS = "data/pairs/dpo_timing.jsonl"


# ---------------------------------------------------------------------------
# DPO maths
# ---------------------------------------------------------------------------


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    *,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Bradley-Terry DPO with a frozen reference: ``-logsigmoid(beta*(dlogp_pi-dlogp_ref))``."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    for name, value in (
        ("policy_chosen", policy_chosen),
        ("policy_rejected", policy_rejected),
        ("reference_chosen", reference_chosen),
        ("reference_rejected", reference_rejected),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen - reference_rejected
    logits = beta * (policy_margin - reference_margin)
    loss = -F.logsigmoid(logits).mean()
    detached = logits.detach()
    metrics = {
        "dpo_loss": float(loss.detach()),
        "reward_accuracy": float((detached > 0).float().mean()),
        "policy_margin": float(policy_margin.detach().mean()),
        "reference_margin": float(reference_margin.detach().mean()),
        "logit_margin": float(detached.mean()),
    }
    return loss, metrics


def sequence_log_prob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of log-probabilities of the labelled (control/text) positions.

    The causal shift matches the model: ``logits[t]`` predicts ``labels[t+1]``.
    ``-100`` labels are ignored; a row with no labelled positions gets 0.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be [batch, sequence, vocab], got {tuple(logits.shape)}")
    if labels.shape != logits.shape[:2]:
        raise ValueError(f"labels shape {tuple(labels.shape)} does not match logits {tuple(logits.shape[:2])}")
    shifted_log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = labels[:, 1:]
    mask = targets != -100
    safe_targets = targets.clamp_min(0)
    gathered = shifted_log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(dim=-1)


@contextlib.contextmanager
def adapter_disabled(model: Any) -> Iterator[Any]:
    """Run the reference forward with the LoRA adapter turned off."""
    disable = getattr(model, "disable_adapter", None)
    if disable is None:
        raise RuntimeError(
            "DPO needs a peft model exposing disable_adapter(); load the stage-2 checkpoint and apply LoRA first"
        )
    with disable():
        yield model


# ---------------------------------------------------------------------------
# Preference pairs
# ---------------------------------------------------------------------------


@dataclass
class DPOPair:
    id: str
    chosen: dict[str, Any]
    rejected: dict[str, Any]


def _as_item(value: Mapping[str, Any], *, key: str) -> dict[str, Any]:
    item = {name: value[name] for name in ("user_codes", "agent_codes", "text", "control") if name in value}
    missing = {"user_codes", "agent_codes", "text", "control"} - set(item)
    if missing:
        raise ValueError(f"pair item {key!r} is missing {sorted(missing)}")
    item["user_codes"] = np.asarray(item["user_codes"])
    item["agent_codes"] = np.asarray(item["agent_codes"])
    item["text"] = np.asarray(item["text"])
    item["control"] = np.asarray(item["control"])
    if "valid" in value:
        valid = np.asarray(value["valid"], dtype=bool)
        if valid.ndim != 1 or len(valid) != item["control"].shape[0]:
            raise ValueError(f"pair item {key!r}: valid mask must be [T] matching control")
        item["valid"] = valid
        frames = int(valid.sum())
    else:
        frames = int(value.get("valid_frames", item["text"].shape[0]))
    item["valid_frames"] = frames
    item["start_frame"] = int(value.get("start_frame", 0))
    item["id"] = str(value.get("id", key))
    return item


def assert_same_context(chosen: Mapping[str, Any], rejected: Mapping[str, Any]) -> None:
    """The pair must differ only in control timing, never in context/audio."""
    for key in ("user_codes", "agent_codes", "text"):
        if not np.array_equal(np.asarray(chosen[key]), np.asarray(rejected[key])):
            raise ValueError(f"DPO pair {chosen.get('id', '?')}: {key} differs; pairs must share one context")
    if int(chosen["valid_frames"]) != int(rejected["valid_frames"]):
        raise ValueError(f"DPO pair {chosen.get('id', '?')}: valid_frames differ")
    if "valid" in chosen or "valid" in rejected:
        if not np.array_equal(np.asarray(chosen.get("valid")), np.asarray(rejected.get("valid"))):
            raise ValueError(f"DPO pair {chosen.get('id', '?')}: valid masks differ")
    if np.array_equal(np.asarray(chosen["control"]), np.asarray(rejected["control"])):
        raise ValueError(
            f"DPO pair {chosen.get('id', '?')}: chosen and rejected control labels are identical; "
            "there is no preference signal"
        )


def timing_variants(
    item: Mapping[str, Any],
    *,
    onset_frame: int,
    early_delay_frames: int = 3,
    late_delay_frames: int = 15,
    response_label: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two control arrays with a 200 ms (3 frame) vs 1,200 ms (15 frame) onset.

    ``response_label`` defaults to ``CONTROL_IDS['SPEAK']``.  This helper only
    creates candidate timings for a pair builder; the FDB-labelled artifact
    decides which side is chosen.
    """
    control = np.asarray(item["control"]).copy()
    frames = int(item.get("valid_frames", control.shape[0]))
    if not 0 <= onset_frame < frames:
        raise ValueError(f"onset_frame {onset_frame} outside 0..{frames - 1}")
    if early_delay_frames < 0 or late_delay_frames < 0 or early_delay_frames == late_delay_frames:
        raise ValueError("delays must differ and be non-negative")

    def variant(delay: int) -> dict[str, Any]:
        new = dict(item)
        labels = control.copy()
        position = onset_frame + delay
        if position >= frames:
            raise ValueError(f"delayed onset {position} exceeds the {frames}-frame item")
        labels[position] = response_label
        new["control"] = labels
        return new

    early, late = variant(early_delay_frames), variant(late_delay_frames)
    assert_same_context(early, late)
    return early, late


def iter_dpo_pairs(
    path: str | Path,
    *,
    dataset: Any | None = None,
) -> Iterator[DPOPair]:
    """Stream labelled pairs.

    Three accepted line shapes::

        {"id": ..., "chosen": <item>, "rejected": <item>}
        {"id": ..., "item": <item>, "chosen_control": [...], "rejected_control": [...]}
        {"id": ..., "dataset_index": 12, "chosen_control": [...], "rejected_control": [...]}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"DPO pair artifact not found at {path}. Build/label the 5,000 timing pairs (FDB metrics, "
            "200 ms vs 1200 ms onset; barge-in vs talk-over) and pass --pairs; this stage does not invent labels."
        )
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                pair_id = str(record.get("id", f"pair_{line_number}"))
                if "chosen" in record and "rejected" in record:
                    chosen = _as_item(record["chosen"], key=f"{pair_id}.chosen")
                    rejected = _as_item(record["rejected"], key=f"{pair_id}.rejected")
                else:
                    controls = record.get("chosen_control"), record.get("rejected_control")
                    if controls[0] is None or controls[1] is None:
                        raise ValueError("expected chosen/rejected items or chosen_control/rejected_control")
                    if "item" in record:
                        base = _as_item(record["item"], key=f"{pair_id}.item")
                    elif "dataset_index" in record:
                        if dataset is None:
                            raise ValueError("dataset_index pairs need --tokens to resolve the shard item")
                        base = _as_item(dataset[int(record["dataset_index"])], key=f"{pair_id}.dataset")
                    else:
                        raise ValueError("pair needs 'item' or 'dataset_index' next to the control arrays")
                    chosen = dict(base)
                    rejected = dict(base)
                    chosen["control"] = np.asarray(controls[0])
                    rejected["control"] = np.asarray(controls[1])
                assert_same_context(chosen, rejected)
            except Exception as exc:  # noqa: BLE001 - include the line number
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            yield DPOPair(id=pair_id, chosen=chosen, rejected=rejected)


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


def lora_target_modules(model: nn.Module, *, root: str = "model.language_model") -> list[str]:
    """Leaf names of the LM's linear layers, for ``peft`` ``target_modules``."""
    suffixes: set[str] = set()
    for name, module in model.named_modules():
        if not name.startswith(root) or not isinstance(module, nn.Linear):
            continue
        parts = name.split(".")
        if len(parts) >= 2:
            suffixes.add(".".join(parts[-2:]))
        else:
            suffixes.add(parts[-1])
    if not suffixes:
        raise RuntimeError(f"No nn.Linear layers under {root}; cannot attach LoRA")
    return sorted(suffixes)


def apply_lora(model: nn.Module, *, r: int = 16, alpha: int = 32, dropout: float = 0.05) -> tuple[nn.Module, dict[str, Any]]:
    """Wrap the frozen stage-2 model with a LoRA adapter on the text LLM only."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("stage 3 needs `peft` (pip install peft)") from exc
    targets = lora_target_modules(model)
    config = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    peft_model = get_peft_model(model, config)
    # Keep adapters strictly on the LLM: anything peft attached elsewhere
    # (e.g. the frozen connector) is frozen again.
    for name, parameter in peft_model.named_parameters():
        is_adapter = any(marker in name for marker in ("lora_A", "lora_B", "lora_embedding"))
        if not is_adapter or "language_model" not in name:
            parameter.requires_grad_(False)
    information = {
        "r": int(r),
        "lora_alpha": int(alpha),
        "lora_dropout": float(dropout),
        "target_modules": targets,
        "beta": None,
    }
    return peft_model, information


# ---------------------------------------------------------------------------
# Pair batches
# ---------------------------------------------------------------------------


def _batch_kwargs(batch: SupervisedBatch) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
    if batch.input_features is not None:
        kwargs["input_features"] = batch.input_features
        kwargs["feature_attention_mask"] = batch.feature_attention_mask
    if batch.audio_out_codes is not None:
        kwargs["audio_out_codes"] = batch.audio_out_codes
    return kwargs


def log_prob_for_batch(model: nn.Module, batch: SupervisedBatch) -> torch.Tensor:
    outputs = model(**_batch_kwargs(batch))
    return sequence_log_prob(outputs.logits, batch.labels)


class PairBatchFactory:
    """Yield ``(chosen, rejected)`` batch pairs from a labelled pair stream."""

    def __init__(
        self,
        pairs: Sequence[DPOPair],
        builder: DuplexBatchBuilder,
        *,
        seq_frames: int,
        seed: int = 0,
    ) -> None:
        self.pairs = list(pairs)
        if not self.pairs:
            raise ValueError("no DPO pairs")
        self.builder = builder
        self.seq_frames = int(seq_frames)
        self.seed = int(seed)

    def __call__(self, epoch: int) -> Iterator[tuple[SupervisedBatch, SupervisedBatch]]:
        rng = random.Random(self.seed + epoch)
        order = list(range(len(self.pairs)))
        rng.shuffle(order)
        for index in order:
            pair = self.pairs[index]
            total = int(pair.chosen["valid_frames"])
            start = 0
            if total > self.seq_frames:
                start = rng.randrange(0, total - self.seq_frames + 1)
            chosen_item = window_item(pair.chosen, start, self.seq_frames)
            rejected_item = window_item(pair.rejected, start, self.seq_frames)
            prepared = self.builder.prepare(chosen_item)
            chosen = self.builder.build([prepared], [chosen_item["control"]])
            rejected = self.builder.build([prepared], [rejected_item["control"]])
            yield chosen, rejected


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def run_smoke(*, steps: int = 2, seed: int = 0) -> int:
    torch.manual_seed(seed)
    policy = torch.tensor([0.4, -0.2], requires_grad=True)
    reference = torch.tensor([0.1, 0.0])
    optimizer = torch.optim.SGD([policy], lr=0.1)
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        loss, metrics = dpo_loss(
            policy[0].reshape(1),
            policy[1].reshape(1),
            reference[0].reshape(1),
            reference[1].reshape(1),
            beta=0.1,
        )
        loss.backward()
        optimizer.step()
        print(f"dpo smoke step {step}: loss={float(loss.detach()):.4f} acc={metrics['reward_accuracy']:.2f}")
    return 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_dpo_training(
    *,
    model: nn.Module,
    ctx: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    precision: Any,
    pair_factory: PairBatchFactory,
    metrics: JsonlMetrics,
    total_steps: int,
    grad_accum: int,
    beta: float,
    max_grad_norm: float = 1.0,
    log_every: int = 10,
    save_every: int = 0,
    state_dir: str | Path | None = None,
    resume_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if grad_accum < 1:
        raise ValueError("grad_accum must be >= 1")
    step = int(resume_state.get("step", 0)) if resume_state else 0
    epoch = int(resume_state.get("epoch", 0)) if resume_state else 0
    iterator = iter(pair_factory(epoch))
    model.train()
    while step < total_steps:
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, float] = {}
        nonfinite = False
        for micro in range(grad_accum):
            item = next(iterator, None)
            while item is None:
                epoch += 1
                iterator = iter(pair_factory(epoch))
                item = next(iterator, None)
                if item is None:
                    raise RuntimeError("DPO pair factory produced no batches")
            chosen, rejected = item
            chosen = chosen.to(ctx.device)
            rejected = rejected.to(ctx.device)
            sync = micro == (grad_accum - 1)
            no_sync = getattr(model, "no_sync", None)
            context = contextlib.nullcontext()
            if not sync and no_sync is not None:
                context = no_sync()
            with context, precision.autocast():
                policy_chosen = log_prob_for_batch(model, chosen)
                policy_rejected = log_prob_for_batch(model, rejected)
                with torch.no_grad(), adapter_disabled(model):
                    reference_chosen = log_prob_for_batch(model, chosen)
                    reference_rejected = log_prob_for_batch(model, rejected)
                loss, parts = dpo_loss(
                    policy_chosen,
                    policy_rejected,
                    reference_chosen,
                    reference_rejected,
                    beta=beta,
                )
            if not torch.isfinite(loss).all():
                nonfinite = True
                break
            precision.backward(loss / grad_accum)
            for key, value in parts.items():
                accumulated[key] = accumulated.get(key, 0.0) + float(value) / grad_accum
        if nonfinite:
            optimizer.zero_grad(set_to_none=True)
            step += 1
            scheduler.step()
            if ctx.is_main:
                metrics.log({"step": step, "skipped": "non_finite_loss"})
            continue
        precision.unscale(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]], max_grad_norm
        )
        precision.step(optimizer)
        scheduler.step()
        step += 1
        if ctx.is_main and (step % max(1, log_every) == 0 or step == 1):
            metrics.log(
                {
                    "step": step,
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "beta": beta,
                    "seconds": time.perf_counter() - started,
                    **accumulated,
                }
            )
        if state_dir is not None and save_every and step % save_every == 0:
            save_checkpoint(
                state_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                epoch=epoch,
            )
    if state_dir is not None:
        save_checkpoint(
            state_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            epoch=epoch,
        )
    return {"step": step, "epoch": epoch}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", default="data/tokens", help="token shards (only needed for dataset_index pairs)")
    parser.add_argument("--model", default="checkpoints/stage2", help="prepared stage-2 checkpoint")
    parser.add_argument("--pairs", default=DEFAULT_PAIRS, help="labelled timing-pair JSONL")
    parser.add_argument("--processor", default=None)
    parser.add_argument("--codec", default=None)
    parser.add_argument("--codec-device", default=None, help="device for codec decode (default: codec module device)")
    parser.add_argument("--out", default="checkpoints/stage3")
    parser.add_argument("--job", default="local")
    parser.add_argument("--seq-frames", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6, help="blueprint: 5e-6")
    parser.add_argument("--beta", type=float, default=0.1, help="blueprint: 0.1")
    parser.add_argument("--lora-r", type=int, default=16, help="blueprint: r=16")
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--steps", type=int, default=1500, help="blueprint range: 1000-2000")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="two steps on tiny tensors; no downloads")
    return parser


def _load_stage2_checkpoint(args: argparse.Namespace, device: torch.device):
    from transformers import AutoTokenizer

    model_dir = Path(args.model)
    if not model_dir.is_dir():
        raise SystemExit(f"--model {args.model} must be a prepared stage-2 checkpoint directory")
    metadata = model_lib.read_repartee_metadata(model_dir)
    if metadata is None:
        raise SystemExit(f"{model_dir} has no repartee.json; finish train_stage2 first")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=args.local_files_only, trust_remote_code=True)
    control = model_lib.control_ids_from_checkpoint(model_dir, tokenizer)
    model = model_lib.load_model(str(model_dir), quantize="none", device=device, local_files_only=args.local_files_only)
    model_lib.configure_trainability(model, "dpo", control)
    peft_model, info = apply_lora(
        model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    info["beta"] = args.beta
    return peft_model, tokenizer, control, info


def run_real(args: argparse.Namespace) -> int:
    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        raise SystemExit(
            f"DPO pairs not found at {pairs_path}. label the timing pairs with FDB metrics and pass --pairs; "
            "stage 3 refuses to fabricate preference labels"
        )
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = init_distributed(seed=args.seed)
    if args.device:
        ctx = type(ctx)(rank=ctx.rank, world_size=ctx.world_size, local_rank=ctx.local_rank, device=device)
    try:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        model, tokenizer, control, lora_info = _load_stage2_checkpoint(args, ctx.device)
        if ctx.is_main:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"dpo: {args.job} | device={ctx.device} | beta={args.beta} | LoRA r={args.lora_r} | "
                f"trainable {trainable:,}/{total:,}"
            )
            print(f"LoRA target modules: {lora_info['target_modules']}")

        processor_path = args.processor or processor_source(args.model)
        processor = model_lib.load_processor(processor_path, local_files_only=args.local_files_only)
        codec = load_codec(processor, source=args.codec or args.processor or args.model)
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

        dataset = None
        if _pairs_need_dataset(pairs_path):
            dataset = open_shard_dataset(
                require_dir(
                    args.tokens,
                    what="Token shard directory",
                    hint="dataset_index pairs resolve their context through --tokens; run `make data`.",
                ),
                seq_frames=args.seq_frames,
                seed=args.seed,
            )
        pairs = list(iter_dpo_pairs(pairs_path, dataset=dataset))
        if ctx.is_main:
            print(f"loaded {len(pairs)} DPO pairs from {pairs_path}")
        pair_factory = PairBatchFactory(pairs, builder, seq_frames=args.seq_frames, seed=args.seed)

        if args.activation_checkpointing:
            enable_activation_checkpointing(model, required=True)
        wrapped = wrap_ddp(model, ctx)
        optimizer = build_optimizer(wrapped, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = WarmupCosineScheduler(optimizer, total_steps=args.steps, warmup_steps=args.warmup)
        resume_state = None
        if args.resume:
            resume_state = load_checkpoint(
                args.resume, model=wrapped, optimizer=optimizer, scheduler=scheduler, map_location=ctx.device
            )
        metrics = JsonlMetrics(out_dir / "metrics.jsonl", rank=ctx.rank, append=bool(args.resume))
        metrics.log(
            {
                "event": "start",
                "job": args.job,
                "device": str(ctx.device),
                "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
                "beta": args.beta,
                "lora_r": args.lora_r,
                "steps": args.steps,
                "pairs": str(args.pairs),
            }
        )
        state_dir = out_dir / "state"
        result = run_dpo_training(
            model=wrapped,
            ctx=ctx,
            optimizer=optimizer,
            scheduler=scheduler,
            precision=make_precision(ctx.device, args.precision),
            pair_factory=pair_factory,
            metrics=metrics,
            total_steps=args.steps,
            grad_accum=args.grad_accum,
            beta=args.beta,
            max_grad_norm=args.max_grad_norm,
            log_every=args.log_every,
            save_every=args.save_every,
            state_dir=state_dir,
            resume_state=resume_state,
        )
        if ctx.is_main:
            model_lib.save_prepared(
                unwrap_model(wrapped),
                tokenizer,
                control,
                out_dir,
                base_model=str(args.model),
                extra={
                    "stage": "dpo",
                    "step": result["step"],
                    "job": args.job,
                    "peft": lora_info,
                    "pairs": str(pairs_path),
                },
            )
            elapsed = time.time() - started
            print(f"dpo done: {result['step']} steps in {elapsed / 3600:.2f} h -> {out_dir}")
    finally:
        cleanup_distributed(ctx)
    return 0


def _pairs_need_dataset(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "dataset_index" in record and "item" not in record and "chosen" not in record:
                return True
    return False


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.smoke:
            return run_smoke(steps=2, seed=args.seed)
        return run_real(args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"dpo: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - CLI boundary: never dump a raw backend traceback
        raise SystemExit(f"dpo: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
