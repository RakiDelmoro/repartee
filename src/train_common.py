"""Shared training machinery for Repartee stages 1-3.

Contents
--------
* torchrun/DDP initialisation, precision (bf16, fp16 fallback + scaler), cosine
  LR with warmup, activation checkpointing, gradient accumulation and clipping,
  JSONL metrics, throughput, atomic checkpoints and resume;
* the duplex batch builder over 12.5 Hz token-shard items;
* the Repartee loss (``3*CE_text + 2*CE_control + 1*CE_audio``) with the
  blueprint's control class weights;
* tiny CPU-only toy components used by the ``--smoke`` paths and by the
  contract tests (no checkpoint download, no GPU).

Item protocol (owned by ``src/dataset.py``)
-------------------------------------------
One dataset item is a mapping with:

=================  ==========================  ==========================================
key                shape                       meaning
=================  ==========================  ==========================================
``user_codes``     ``[T, 12]`` uint16          LiquidCodec codes of the user channel
``agent_codes``    ``[T, 12]`` uint16          LiquidCodec codes of the agent channel
``text``           ``[T]`` int32               coarse agent text token per frame, 0 = none
``control``        ``[T]`` uint8               ``src.schema.CONTROL_IDS`` label per frame
``valid``          ``[T]`` bool                real frames (a true prefix); tail is silence
``valid_frames``   int                         legacy fallback used by the toy smoke path
``id``             str                         conversation id (optional)
``start_frame``    int                         offset of the window (optional)
=================  ==========================  ==========================================

``src/dataset.py`` pads every window to ``seq_frames`` and marks the real frames
with ``valid``; padding rows already carry a codec-encoded silence frame and
``PAD`` control labels.  Only the true prefix is fed to the model.
Codebooks are the **second** axis; the codec's ``decode`` wants
``[batch, 12, frames]``, so the batch builder transposes once.

Training inputs
---------------
For every frame the sequence is ``[<|audio|>(user), control, text?, <|audio_out|>(agent)]``.
``labels`` are the control token id and the optional text token id only; the
audio placeholders carry ``audio_loss`` supervision through ``audio_labels``.
The user channel is decoded with the frozen LiquidCodec to 24 kHz and
resampled to 16 kHz, then passed through the model's log-mel feature extractor.
The model consumes **flattened** placeholder counts, so the batch builder
asserts that the summed user tokens, ``audio_out`` placeholders, code rows and
``audio_labels`` rows all agree (:func:`validate_duplex_batch`).

Deviations / limitations (parent writes REPORT.md)
--------------------------------------------------
* Codec decoding per item is the expensive part of batch construction.  There
  is no silent shortcut: ``--codec-device`` controls where it runs and the
  cost is logged once (``prepare_seconds``).  Token shards deliberately store
  codes, not features (blueprint section 8).
* Control-token embeddings are trained with a gradient row mask (see
  ``src/model.py``); the stage-1 optimizer uses ``weight_decay=0`` for all
  embedding weights so the masked rows cannot drift through weight decay.
* DPO pair construction/labelling is an evaluation artifact (FDB metrics);
  this module only loads the pairs and validates them.
"""
from __future__ import annotations

import contextlib
import inspect
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .schema import (
    AGENT_SAMPLE_RATE,
    AUDIO_EOS_CODE,
    CODEBOOK_SIZE,
    CONTROL_IDS,
    CONTROL_LABELS,
    FRAME_SAMPLES,
    NUM_CODEBOOKS,
    USER_SAMPLE_RATE,
)

#: Blueprint stage-2 control weights.  These multiply the per-token CE inside
#: the control term; the term itself is weighted by 2 in the total loss.
CONTROL_CLASS_WEIGHTS: dict[str, float] = {
    "PAD": 0.3,
    "LISTEN": 1.0,
    "SPEAK": 8.0,
    "BACKCHANNEL": 4.0,
    "STOP": 8.0,
}
TEXT_LOSS_WEIGHT = 3.0
CONTROL_LOSS_WEIGHT = 2.0
AUDIO_LOSS_WEIGHT = 1.0


# ---------------------------------------------------------------------------
# Distributed / precision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed(*, seed: int = 0, backend: str | None = None) -> DistributedContext:
    """Read torchrun's env vars; safe to call when not launched by torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    cuda = torch.cuda.is_available()
    if world_size > 1:
        if not torch.distributed.is_initialized():
            if backend is None:
                backend = "nccl" if cuda else "gloo"
            torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(local_rank)
    # Different ranks need different RNG streams, but the data shuffle is
    # seeded once in the sampler.
    seed_everything(seed + rank)
    return DistributedContext(rank=rank, world_size=world_size, local_rank=local_rank, device=device)


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def wrap_ddp(model: nn.Module, ctx: DistributedContext) -> nn.Module:
    if not ctx.distributed:
        return model
    from torch.nn.parallel import DistributedDataParallel

    if ctx.device.type == "cuda":
        return DistributedDataParallel(model, device_ids=[ctx.local_rank], find_unused_parameters=True)
    return DistributedDataParallel(model, find_unused_parameters=True)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.Module) and hasattr(model, "module") else model


@dataclass
class PrecisionContext:
    """Autocast + optional fp16 gradient scaler."""

    device_type: str
    dtype: torch.dtype | None
    scaler: Any | None = None
    name: str = "fp32"

    @contextlib.contextmanager
    def autocast(self) -> Iterator[None]:
        if self.dtype is None:
            yield
        else:
            with torch.autocast(device_type=self.device_type, dtype=self.dtype):
                yield

    def backward(self, loss: torch.Tensor) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def unscale(self, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()


def make_precision(device: torch.device, prefer: str = "auto") -> PrecisionContext:
    """bfloat16 autocast on CUDA when available, fp16 + scaler otherwise, fp32 on CPU."""
    if device.type != "cuda":
        return PrecisionContext(device_type="cpu", dtype=None, scaler=None, name="fp32")
    use_bf16 = prefer in ("auto", "bf16") and torch.cuda.is_bf16_supported()
    if use_bf16:
        return PrecisionContext(device_type="cuda", dtype=torch.bfloat16, scaler=None, name="bf16")
    if prefer == "bf16":
        raise RuntimeError("bf16 was requested but this GPU does not support it; use --precision auto or fp16")
    scaler = torch.amp.GradScaler("cuda")
    return PrecisionContext(device_type="cuda", dtype=torch.float16, scaler=scaler, name="fp16")


# ---------------------------------------------------------------------------
# Scheduler / optimizer
# ---------------------------------------------------------------------------


def cosine_lr_at(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    *,
    min_ratio: float = 0.0,
) -> float:
    """Linear warmup to ``base_lr`` then cosine decay to ``min_ratio * base_lr``."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError("min_ratio must lie in [0, 1]")
    if step < warmup_steps:
        return base_lr * float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    factor = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * factor


class WarmupCosineScheduler:
    """Per-parameter-group warmup + cosine schedule with a resumable state dict."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
        min_ratio: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_ratio = float(min_ratio)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.step_index = 0
        self.apply()

    def apply(self) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = cosine_lr_at(
                self.step_index, self.total_steps, self.warmup_steps, base_lr, min_ratio=self.min_ratio
            )

    def step(self) -> None:
        self.step_index += 1
        self.apply()

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_ratio": self.min_ratio,
            "base_lrs": list(self.base_lrs),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        # Only the position in the schedule is restored.  The schedule length
        # and warmup come from the current command: resuming a 10-step run with
        # --steps 300 must not keep total_steps=10 (which drives the LR to 0
        # immediately).  base_lrs come from the freshly built optimizer.
        self.step_index = int(state.get("step_index", 0))
        self.apply()


def _is_connector(name: str) -> bool:
    return "audio_projector" in name


def _is_embedding(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1].lower()
    return "embed" in leaf or "lm_head" in name.lower()


def build_optimizer(
    model: nn.Module,
    *,
    lr: float,
    connector_lr: float | None = None,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """AdamW with a connector LR group and no weight decay on embeddings.

    Embedding weights must not decay: stage 1 only trains four rows through a
    gradient mask, and AdamW's decoupled decay would still move the other rows.
    """
    connector_lr = lr if connector_lr is None else connector_lr
    groups: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group_lr = connector_lr if _is_connector(name) else lr
        decay = 0.0 if _is_embedding(name) else float(weight_decay)
        groups.setdefault((group_lr, decay), []).append(parameter)
    if not groups:
        raise RuntimeError("No trainable parameters; configure_trainability() or LoRA first")
    param_groups = [
        {"params": params, "lr": group_lr, "weight_decay": decay}
        for (group_lr, decay), params in sorted(groups.items())
    ]
    return torch.optim.AdamW(param_groups, betas=betas, eps=eps)


# ---------------------------------------------------------------------------
# Activation checkpointing
# ---------------------------------------------------------------------------


def enable_activation_checkpointing(model: nn.Module, *, required: bool = False) -> bool:
    """Turn on HF gradient checkpointing; returns whether it is active."""
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if enable is None:
        if required:
            raise RuntimeError("This model has no gradient_checkpointing_enable(); cannot honour --activation-checkpointing")
        return False
    try:
        enable()
    except TypeError:
        # some models require an explicit reentrant flag
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return True


# ---------------------------------------------------------------------------
# Metrics / throughput / checkpoints
# ---------------------------------------------------------------------------


class JsonlMetrics:
    """Rank-0-only JSONL metric log (one object per line, sorted keys)."""

    def __init__(
        self, path: str | Path, *, rank: int = 0, enabled: bool = True, append: bool = False
    ) -> None:
        self.path = Path(path)
        self.rank = rank
        self.enabled = enabled and rank == 0
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not append:
                self.path.write_text("", encoding="utf-8")

    def log(self, record: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), sort_keys=True, default=str) + "\n")
        if "step" in record:
            if "skipped" in record:
                print(f"step {record['step']:>6} | skipped: {record['skipped']}", flush=True)
            else:
                print(
                    "step {step:>6} | loss {loss:<8.4f} | text {text:<7.4f} | "
                    "control {control:<7.4f} | audio {audio:<7.4f} | "
                    "lr {lr:<9.2e} | {secs:>6.1f}s | {tok:>6.1f} tok/s".format(
                        step=record["step"],
                        loss=float(record.get("loss", float("nan"))),
                        text=float(record.get("text_loss", float("nan"))),
                        control=float(record.get("control_loss", float("nan"))),
                        audio=float(record.get("audio_loss", float("nan"))),
                        lr=float(record.get("lr", 0.0)),
                        secs=float(record.get("seconds_per_step", 0.0)),
                        tok=float(record.get("tokens_per_s", 0.0)),
                    ),
                    flush=True,
                )


@dataclass
class ThroughputMeter:
    window: int = 20
    _frames: deque = field(default_factory=deque)
    _tokens: deque = field(default_factory=deque)
    _times: deque = field(default_factory=deque)

    def update(self, *, frames: int, tokens: int, elapsed: float) -> None:
        self._frames.append(int(frames))
        self._tokens.append(int(tokens))
        self._times.append(float(elapsed))
        while len(self._times) > self.window:
            self._frames.popleft()
            self._tokens.popleft()
            self._times.popleft()

    def rates(self) -> dict[str, float]:
        seconds = sum(self._times)
        if seconds <= 0:
            return {"frames_per_s": 0.0, "tokens_per_s": 0.0, "seconds_per_step": 0.0}
        return {
            "frames_per_s": sum(self._frames) / seconds,
            "tokens_per_s": sum(self._tokens) / seconds,
            "seconds_per_step": seconds / max(1, len(self._times)),
        }


def atomic_save(state: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(state), temporary)
    os.replace(temporary, path)
    return path


def _link_or_copy(source: Path, destination: Path) -> None:
    """Make ``destination`` point at ``source`` without duplicating 5 GB."""
    temporary = destination.with_name(destination.name + ".link")
    try:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    except OSError:
        pass
    for operation in (os.link, os.symlink):
        try:
            operation(source.name if operation is os.symlink else source, temporary)
            os.replace(temporary, destination)
            return
        except OSError:
            try:
                if temporary.exists() or temporary.is_symlink():
                    temporary.unlink()
            except OSError:
                pass
    import shutil

    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _prune_checkpoints(directory: Path, keep_last: int) -> None:
    snapshots = sorted(directory.glob("step_*.pt"))
    for old in snapshots[:-keep_last] if keep_last > 0 else snapshots:
        try:
            old.unlink()
        except OSError:
            pass


def save_checkpoint(
    directory: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    step: int = 0,
    epoch: int = 0,
    name: str = "latest.pt",
    extra: Mapping[str, Any] | None = None,
    keep_last: int = 5,
) -> Path:
    """Save a rolling checkpoint; at most ``keep_last`` snapshots are kept.

    ``latest.pt`` is a link to the newest ``step_XXXXXX.pt``.  The directory
    therefore holds at most five 5 GB state files, not one per save.
    """
    state: dict[str, Any] = {
        "model": unwrap_model(model).state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "extra": dict(extra or {}),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    directory = Path(directory)
    if name == "latest.pt" and keep_last > 0:
        snapshot = directory / f"step_{int(step):06d}.pt"
        atomic_save(state, snapshot)
        _link_or_copy(snapshot, directory / "latest.pt")
        _prune_checkpoints(directory, keep_last)
        return directory / "latest.pt"
    return atomic_save(state, directory / name)


def resolve_resume(value: str | Path | None, state_dir: str | Path) -> Path | None:
    """Resolve ``--resume auto`` to the newest ``<state_dir>/latest.pt``.

    ``None`` or an empty string means a fresh run; a missing auto checkpoint is
    also a fresh run (the caller logs that it started from scratch).
    """
    if value in (None, ""):
        return None
    if str(value) == "auto":
        candidate = Path(state_dir) / "latest.pt"
        return candidate if candidate.exists() else None
    return Path(value)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "latest.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint {path} does not exist; omit --resume to start from scratch or point at latest.pt"
        )
    state = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        unwrap_model(model).load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and hasattr(scheduler, "load_state_dict") and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return state


# ---------------------------------------------------------------------------
# Duplex input construction
# ---------------------------------------------------------------------------


def resample_24k_to_16k(waveform: torch.Tensor) -> torch.Tensor:
    """24 kHz codec output -> 16 kHz encoder input (exact 2/3 resampling)."""
    if waveform.ndim != 1:
        raise ValueError(f"expected one mono waveform, got shape {tuple(waveform.shape)}")
    if waveform.numel() == 0:
        return waveform
    import torchaudio.functional as audio_functional

    return audio_functional.resample(waveform.float(), AGENT_SAMPLE_RATE, USER_SAMPLE_RATE)


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def item_frame_count(item: Mapping[str, Any]) -> int:
    """Number of real frames in an item.

    ``src/dataset.py`` items carry a boolean ``valid`` prefix mask; the toy
    smoke items and legacy readers may carry ``valid_frames``.  Non-prefix
    masks are rejected because the model consumes a contiguous 80 ms grid.
    """
    if "valid" in item:
        valid = np.asarray(item["valid"], dtype=bool)
        if valid.ndim != 1:
            raise ValueError(f"valid must be [T], got {tuple(valid.shape)}")
        count = int(valid.sum())
        if count and not valid[:count].all():
            raise ValueError("valid must be a true prefix (dataset windows are right-padded)")
        return count
    if "valid_frames" in item:
        return int(item["valid_frames"])
    return int(np.asarray(item["text"]).shape[0])


def _as_frames(array: Any, frames: int, name: str, *, width: int | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(array))
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be [T, C], got shape {tuple(tensor.shape)}")
    if width is not None and tensor.shape[1] != width:
        raise ValueError(f"{name} must have {width} codebooks on its second axis, got {tensor.shape[1]}")
    if tensor.shape[0] < frames:
        raise ValueError(f"{name} has {tensor.shape[0]} rows but the item has {frames} real frames")
    return tensor[:frames].long()


@dataclass
class PreparedItem:
    """One window with the expensive user-audio work already done."""

    frames: int
    feature_tokens: int
    input_features: torch.Tensor  # [128, mel_frames]
    feature_attention_mask: torch.Tensor  # [mel_frames]
    agent_codes: torch.Tensor  # [frames, 12]
    text: torch.Tensor  # [frames]
    item_id: str = ""

    def to(self, device: torch.device | str) -> "PreparedItem":
        return PreparedItem(
            frames=self.frames,
            feature_tokens=self.feature_tokens,
            input_features=self.input_features.to(device),
            feature_attention_mask=self.feature_attention_mask.to(device),
            agent_codes=self.agent_codes.to(device),
            text=self.text.to(device),
            item_id=self.item_id,
        )


@dataclass
class SupervisedBatch:
    """Tensors for one model forward, plus frame/token counts for throughput."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    input_features: torch.Tensor | None = None
    feature_attention_mask: torch.Tensor | None = None
    audio_out_codes: torch.Tensor | None = None
    audio_labels: torch.Tensor | None = None
    frame_count: int = 0
    token_count: int = 0

    def to(self, device: torch.device | str) -> "SupervisedBatch":
        def move(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.to(device)

        return SupervisedBatch(
            input_ids=move(self.input_ids),
            attention_mask=move(self.attention_mask),
            labels=move(self.labels),
            input_features=move(self.input_features),
            feature_attention_mask=move(self.feature_attention_mask),
            audio_out_codes=move(self.audio_out_codes),
            audio_labels=move(self.audio_labels),
            frame_count=self.frame_count,
            token_count=self.token_count,
        )


def build_frame_sequence(
    control: Any,
    text: Any,
    *,
    control_ids: Mapping[str, int],
    audio_token_id: int,
    audio_out_token_id: int,
) -> tuple[list[int], list[int]]:
    """Per-frame ``[user, control, text?, audio_out]`` ids and labels.

    Labels are only the control id and the optional text id.  ``text`` uses 0
    for "no token at this frame" (the tokenize.py contract).
    """
    controls = [int(value) for value in control]
    texts = [int(value) for value in text] if text is not None else [0] * len(controls)
    if len(texts) != len(controls):
        raise ValueError(f"control and text lengths differ: {len(controls)} vs {len(texts)}")
    missing = set(CONTROL_LABELS) - set(control_ids)
    if missing:
        raise ValueError(f"control_ids is missing {sorted(missing)}")
    ids: list[int] = []
    labels: list[int] = []
    for frame, (label_index, text_id) in enumerate(zip(controls, texts)):
        if not 0 <= label_index < len(CONTROL_LABELS):
            raise ValueError(f"frame {frame}: invalid control label {label_index}")
        control_id = int(control_ids[CONTROL_LABELS[label_index]])
        ids.append(int(audio_token_id))
        labels.append(-100)
        ids.append(control_id)
        labels.append(control_id)
        if text_id != 0:
            ids.append(text_id)
            labels.append(text_id)
        ids.append(int(audio_out_token_id))
        labels.append(-100)
    return ids, labels


def validate_duplex_batch(
    input_ids: torch.Tensor,
    *,
    audio_feature_tokens: int,
    audio_token_id: int,
    audio_out_token_id: int,
    audio_out_codes: torch.Tensor | None = None,
    audio_labels: torch.Tensor | None = None,
    num_codebooks: int = NUM_CODEBOOKS,
    codebook_size: int = CODEBOOK_SIZE,
) -> dict[str, int]:
    """Mirror ``Lfm3OmniModel.forward``/``audio_loss`` placeholder checks before the forward.

    The model compares **flattened** counts: the summed number of user audio
    tokens across the batch, the number of ``audio_out`` rows, and the row
    count of ``audio_labels``.  A mismatch here is a batch-construction bug, so
    fail with a precise message instead of a CUDA assert deep in the model.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [batch, sequence], got {tuple(input_ids.shape)}")
    user_tokens = int((input_ids == audio_token_id).sum())
    audio_placeholders = int((input_ids == audio_out_token_id).sum())
    if user_tokens != int(audio_feature_tokens):
        raise ValueError(
            f"Batch contract violated: {user_tokens} user audio placeholders but {audio_feature_tokens} "
            "encoder tokens would be produced (check feature extractor lengths/frames)"
        )
    if audio_out_codes is None:
        if audio_placeholders != 0:
            raise ValueError(
                f"Batch contract violated: {audio_placeholders} audio_out placeholders but no audio_out_codes"
            )
        return {"user_tokens": user_tokens, "audio_frames": 0}
    if audio_out_codes.ndim != 2 or audio_out_codes.shape[1] != num_codebooks:
        raise ValueError(
            f"audio_out_codes must be [frames, {num_codebooks}], got {tuple(audio_out_codes.shape)}"
        )
    if audio_out_codes.shape[0] != audio_placeholders:
        raise ValueError(
            f"Batch contract violated: {audio_placeholders} audio_out placeholders but "
            f"{audio_out_codes.shape[0]} code rows"
        )
    if int(audio_out_codes.min()) < 0 or int(audio_out_codes.max()) >= codebook_size:
        raise ValueError(f"audio_out_codes must lie in [0, {codebook_size})")
    if audio_labels is not None:
        if audio_labels.shape != audio_out_codes.shape:
            raise ValueError(
                f"audio_labels shape {tuple(audio_labels.shape)} != audio_out_codes shape "
                f"{tuple(audio_out_codes.shape)}"
            )
        valid = (audio_labels != -100).all(-1)
        values = audio_labels[valid]
        if values.numel() and (int(values.min()) < 0 or int(values.max()) >= codebook_size):
            raise ValueError(f"audio_labels must lie in [0, {codebook_size}) or be -100 rows")
    return {"user_tokens": user_tokens, "audio_frames": audio_placeholders}


class DuplexBatchBuilder:
    """Turn token-shard items into model-ready duplex batches.

    ``codec`` only needs the LiquidCodec ``decode(codes[1,12,F]) -> .audio_values``
    interface; ``feature_extractor`` only needs ``num_encoder_tokens`` and the
    normal ``__call__``.  Both are injected so the smoke path can use toys.
    """

    def __init__(
        self,
        *,
        codec: nn.Module,
        feature_extractor: Any,
        control_ids: Mapping[str, int],
        audio_token_id: int,
        audio_out_token_id: int,
        pad_token_id: int,
        num_codebooks: int = NUM_CODEBOOKS,
        codebook_size: int = CODEBOOK_SIZE,
        codec_device: str | torch.device | None = None,
    ) -> None:
        self.codec = codec
        self.feature_extractor = feature_extractor
        self.control_ids = dict(control_ids)
        self.audio_token_id = int(audio_token_id)
        self.audio_out_token_id = int(audio_out_token_id)
        self.pad_token_id = int(pad_token_id)
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.codec_device = torch.device(codec_device) if codec_device is not None else _module_device(codec)
        self.prepare_seconds = 0.0

    # -- item -----------------------------------------------------------------
    def prepare(self, item: Mapping[str, Any]) -> PreparedItem:
        started = time.perf_counter()
        frames = item_frame_count(item)
        if frames <= 0:
            raise ValueError(f"item {item.get('id', '?')} has no valid frames")
        user_codes = _as_frames(item["user_codes"], frames, "user_codes", width=self.num_codebooks)
        agent_codes = _as_frames(item["agent_codes"], frames, "agent_codes", width=self.num_codebooks)
        text = torch.as_tensor(np.asarray(item["text"]))[:frames].long()
        if text.ndim != 1:
            raise ValueError(f"text must be [T], got {tuple(text.shape)}")
        if int(agent_codes.max()) >= self.codebook_size:
            raise ValueError(f"agent code {int(agent_codes.max())} outside [0, {self.codebook_size})")

        user_wave = self._decode_user(user_codes)
        expected_samples = frames * FRAME_SAMPLES[USER_SAMPLE_RATE]
        if abs(user_wave.numel() - expected_samples) > 2:
            raise ValueError(
                f"Codec/resampler produced {user_wave.numel()} samples for {frames} 80 ms frames "
                f"(expected {expected_samples}); the token shards and the audio rates disagree"
            )
        if user_wave.numel() != expected_samples:
            user_wave = F.pad(user_wave, (0, expected_samples - user_wave.numel()))[:expected_samples]

        features = self.feature_extractor(
            [user_wave.detach().cpu().numpy()],
            sampling_rate=USER_SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = features["input_features"][0]
        feature_mask = features["feature_attention_mask"][0].long()
        tokens = int(self.feature_extractor.num_encoder_tokens(int(feature_mask.sum())))
        if tokens != frames:
            raise ValueError(
                f"Feature extractor produced {tokens} encoder tokens for {frames} frames "
                f"({int(feature_mask.sum())} mel frames); the 80 ms grid disagrees with the encoder stride"
            )
        self.prepare_seconds += time.perf_counter() - started
        return PreparedItem(
            frames=frames,
            feature_tokens=tokens,
            input_features=input_features,
            feature_attention_mask=feature_mask,
            agent_codes=agent_codes,
            text=text,
            item_id=str(item.get("id", "")),
        )

    def _decode_user(self, user_codes: torch.Tensor) -> torch.Tensor:
        codes = user_codes.transpose(0, 1).unsqueeze(0).to(self.codec_device)
        was_training = getattr(self.codec, "training", False)
        if hasattr(self.codec, "eval"):
            self.codec.eval()
        with torch.no_grad():
            output = self.codec.decode(codes)
        if was_training and hasattr(self.codec, "train"):
            self.codec.train()
        audio = getattr(output, "audio_values", output)
        if not isinstance(audio, torch.Tensor):
            raise RuntimeError("codec.decode must return audio_values")
        wave = audio[0].float().cpu()
        return resample_24k_to_16k(wave)

    # -- batch ----------------------------------------------------------------
    def build(
        self,
        prepared: Sequence[PreparedItem],
        controls: Sequence[Any] | None = None,
    ) -> SupervisedBatch:
        if not prepared:
            raise ValueError("cannot build an empty batch")
        if controls is None:
            raise ValueError("controls are required (item['control'] or explicit control arrays)")
        if len(controls) != len(prepared):
            raise ValueError(f"{len(prepared)} items but {len(controls)} control arrays")
        sequences: list[tuple[list[int], list[int]]] = []
        for item, control in zip(prepared, controls):
            control_values = np.asarray(control)[: item.frames]
            if control_values.shape[0] != item.frames:
                raise ValueError(
                    f"control has {control_values.shape[0]} frames but item {item.item_id} has {item.frames}"
                )
            sequences.append(
                build_frame_sequence(
                    control_values,
                    item.text,
                    control_ids=self.control_ids,
                    audio_token_id=self.audio_token_id,
                    audio_out_token_id=self.audio_out_token_id,
                )
            )
        max_length = max(len(ids) for ids, _ in sequences)
        batch_size = len(sequences)
        input_ids = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        for row, (ids, row_labels) in enumerate(sequences):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            labels[row, : len(ids)] = torch.tensor(row_labels, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1

        feature_size = prepared[0].input_features.shape[0]
        max_mel = max(int(item.input_features.shape[-1]) for item in prepared)
        input_features = torch.zeros((batch_size, feature_size, max_mel), dtype=torch.float32)
        feature_attention_mask = torch.zeros((batch_size, max_mel), dtype=torch.long)
        for row, item in enumerate(prepared):
            width = int(item.input_features.shape[-1])
            input_features[row, :, :width] = item.input_features.to(torch.float32)
            feature_attention_mask[row, :width] = item.feature_attention_mask

        audio_out_codes = torch.cat([item.agent_codes for item in prepared], dim=0).long()
        audio_labels = audio_out_codes.clone()

        validate_duplex_batch(
            input_ids,
            audio_feature_tokens=sum(item.feature_tokens for item in prepared),
            audio_token_id=self.audio_token_id,
            audio_out_token_id=self.audio_out_token_id,
            audio_out_codes=audio_out_codes,
            audio_labels=audio_labels,
            num_codebooks=self.num_codebooks,
            codebook_size=self.codebook_size,
        )
        return SupervisedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_out_codes=audio_out_codes,
            audio_labels=audio_labels,
            frame_count=sum(item.frames for item in prepared),
            token_count=sum(len(ids) for ids, _ in sequences),
        )

    def build_items(
        self, items: Sequence[Mapping[str, Any]], controls: Sequence[Any] | None = None
    ) -> SupervisedBatch:
        prepared = [self.prepare(item) for item in items]
        if controls is None:
            controls = [item["control"] for item in items]
        return self.build(prepared, controls)


def build_text_batch(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int = 2048,
) -> SupervisedBatch:
    """A plain text LM batch for ``--text-jsonl`` regression batches."""
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].long()
    attention_mask = encoded["attention_mask"].long()
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return SupervisedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        token_count=int((labels != -100).sum()),
    )


def window_item(item: Mapping[str, Any], start_frame: int, frames: int) -> dict[str, Any]:
    """Slice a conversation item to a contiguous window (keeps the item protocol)."""
    total = item_frame_count(item)
    if start_frame < 0 or frames <= 0:
        raise ValueError(f"invalid window start={start_frame} frames={frames}")
    if start_frame >= total:
        raise ValueError(f"window start {start_frame} beyond the item's {total} real frames")
    end = min(start_frame + frames, total)
    window: dict[str, Any] = dict(item)
    for key in ("user_codes", "agent_codes", "text", "control"):
        window[key] = np.asarray(item[key])[start_frame:end]
    if "valid" in item:
        valid = np.asarray(item["valid"], dtype=bool)[start_frame:end]
        window["valid"] = valid
        window["valid_frames"] = int(valid.sum())
    else:
        window["valid_frames"] = int(end - start_frame)
    window["start_frame"] = int(item.get("start_frame", 0)) + start_frame
    return window


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def causal_shift(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 3:
        raise ValueError(f"logits must be [batch, sequence, vocab], got {tuple(logits.shape)}")
    if labels.shape != logits.shape[:2]:
        raise ValueError(f"labels shape {tuple(labels.shape)} does not match logits {tuple(logits.shape[:2])}")
    return logits[:, :-1, :], labels[:, 1:]


def _unreduced_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(labels.shape)


def control_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    control_ids: Mapping[str, int],
    *,
    class_weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, int]:
    """Weighted mean CE over control-token targets only.

    The class weights are applied per token and the result is normalised by the
    sum of the applied weights (a proper weighted mean), so adding weight to a
    rare class does not explode the loss scale.
    """
    weights = dict(CONTROL_CLASS_WEIGHTS if class_weights is None else class_weights)
    shifted_logits, shifted_labels = causal_shift(logits, labels)
    losses = _unreduced_cross_entropy(shifted_logits, shifted_labels)
    id_to_weight = {int(control_ids[label]): float(weights[label]) for label in CONTROL_LABELS}
    control_mask = torch.zeros_like(shifted_labels, dtype=torch.bool)
    for token_id in id_to_weight:
        control_mask |= shifted_labels == token_id
    control_mask &= shifted_labels != -100
    count = int(control_mask.sum())
    if count == 0:
        return shifted_logits.sum() * 0.0, 0
    token_weights = torch.ones_like(losses)
    for token_id, weight in id_to_weight.items():
        token_weights = torch.where(shifted_labels == token_id, torch.full_like(losses, weight), token_weights)
    applied = token_weights[control_mask]
    return (losses[control_mask] * applied).sum() / applied.sum().clamp_min(1e-6), count


def text_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    control_ids: Mapping[str, int],
) -> tuple[torch.Tensor, int]:
    """Mean CE over every non-control, non-ignored text target."""
    shifted_logits, shifted_labels = causal_shift(logits, labels)
    losses = _unreduced_cross_entropy(shifted_logits, shifted_labels)
    mask = shifted_labels != -100
    for token_id in control_ids.values():
        mask &= shifted_labels != int(token_id)
    count = int(mask.sum())
    if count == 0:
        return shifted_logits.sum() * 0.0, 0
    return losses[mask].mean(), count


def duplex_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    control_ids: Mapping[str, int],
    *,
    audio_loss: torch.Tensor | None = None,
    text_weight: float = TEXT_LOSS_WEIGHT,
    control_weight: float = CONTROL_LOSS_WEIGHT,
    audio_weight: float = AUDIO_LOSS_WEIGHT,
    class_weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """``text_weight*CE_text + control_weight*CE_control + audio_weight*CE_audio``.

    A term whose batch has no supervised positions is omitted (and reported as
    zero) instead of contributing a NaN.  The speech term is the model's own
    depth-decoder loss, so codebook masking stays identical to pretraining.
    """
    token_loss, text_tokens = text_loss(logits, labels, control_ids)
    command_loss, control_tokens = control_loss(logits, labels, control_ids, class_weights=class_weights)
    total = token_loss.new_zeros(())
    if text_tokens:
        total = total + text_weight * token_loss
    if control_tokens:
        total = total + control_weight * command_loss
    audio_value = None
    if audio_loss is not None:
        total = total + audio_weight * audio_loss
        audio_value = float(audio_loss.detach())
    parts = {
        "loss": float(total.detach()),
        "text_loss": float(token_loss.detach()),
        "control_loss": float(command_loss.detach()),
        "audio_loss": audio_value,
        "text_tokens": text_tokens,
        "control_tokens": control_tokens,
    }
    return total, parts


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class SupervisedTrainer:
    """A small, boring training loop shared by stages 1 and 2."""

    def __init__(
        self,
        *,
        model: nn.Module,
        ctx: DistributedContext,
        optimizer: torch.optim.Optimizer,
        scheduler: WarmupCosineScheduler,
        precision: PrecisionContext,
        control_ids: Mapping[str, int],
        batch_factory: Callable[[int], Iterable[SupervisedBatch]],
        metrics: JsonlMetrics,
        grad_accum: int = 1,
        max_grad_norm: float = 1.0,
        log_every: int = 10,
        save_every: int = 0,
        checkpoint_dir: str | Path | None = None,
        text_batch_factory: Callable[[int], Iterable[SupervisedBatch]] | None = None,
        text_ratio: float = 0.0,
        on_save: Callable[[int], None] | None = None,
        extra_loss: Callable[[Any, SupervisedBatch], torch.Tensor] | None = None,
        audio_weight: float = 1.0,
    ) -> None:
        if grad_accum < 1:
            raise ValueError("grad_accum must be >= 1")
        if not 0.0 <= text_ratio <= 1.0:
            raise ValueError("text_ratio must lie in [0, 1]")
        self.model = model
        self.ctx = ctx
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.precision = precision
        self.control_ids = dict(control_ids)
        self.batch_factory = batch_factory
        self.text_batch_factory = text_batch_factory
        self.text_ratio = float(text_ratio)
        self.metrics = metrics
        self.grad_accum = int(grad_accum)
        self.max_grad_norm = float(max_grad_norm)
        self.log_every = int(log_every)
        self.save_every = int(save_every)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.on_save = on_save
        self.extra_loss = extra_loss
        self.audio_weight = float(audio_weight)
        self.throughput = ThroughputMeter()
        self._rng = random.Random(1234)
        self._text_iterator: Iterator[SupervisedBatch] | None = None

    def _next_batch(
        self, iterator: Iterator[SupervisedBatch], epoch: int
    ) -> tuple[SupervisedBatch, Iterator[SupervisedBatch], int]:
        if self.text_batch_factory is not None and self._rng.random() < self.text_ratio:
            if self._text_iterator is None:
                self._text_iterator = iter(self.text_batch_factory(epoch))
            batch = next(self._text_iterator, None)
            if batch is None:
                self._text_iterator = iter(self.text_batch_factory(epoch))
                batch = next(self._text_iterator)
            return batch, iterator, epoch
        batch = next(iterator, None)
        while batch is None:
            epoch += 1
            iterator = iter(self.batch_factory(epoch))
            batch = next(iterator, None)
            if batch is None:
                raise RuntimeError(
                    "batch_factory produced no batches; check that the token shards contain conversations "
                    "and that --batch-size/--seq-frames are usable"
                )
        return batch, iterator, epoch

    def _forward(self, batch: SupervisedBatch) -> tuple[torch.Tensor, dict[str, Any]]:
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
        if self.extra_loss is not None:
            kwargs["output_hidden_states"] = True
        outputs = self.model(**kwargs)
        audio_loss = getattr(outputs, "audio_loss", None)
        loss, parts = duplex_loss(
            outputs.logits,
            batch.labels,
            self.control_ids,
            audio_loss=audio_loss if batch.audio_labels is not None else None,
            audio_weight=self.audio_weight,
        )
        if self.extra_loss is not None:
            extra = self.extra_loss(outputs, batch)
            if extra is not None:
                loss = loss + extra
                parts["distill_loss"] = float(extra.detach())
        return loss, parts

    def fit(
        self,
        *,
        total_steps: int,
        epochs: int = 1,
        resume_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        start_step = int(resume_state.get("step", 0)) if resume_state else 0
        start_epoch = int(resume_state.get("epoch", 0)) if resume_state else 0
        step = start_step
        epoch = start_epoch
        iterator = iter(self.batch_factory(epoch))
        self.model.train()
        while step < total_steps:
            step_started = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            accumulated: dict[str, Any] = {}
            micro_frames = micro_tokens = 0
            nonfinite = False
            for micro in range(self.grad_accum):
                batch, iterator, epoch = self._next_batch(iterator, epoch)
                batch = batch.to(self.ctx.device)
                micro_frames += batch.frame_count
                micro_tokens += batch.token_count
                sync = micro == (self.grad_accum - 1)
                no_sync = getattr(self.model, "no_sync", None)
                context = contextlib.nullcontext()
                if not sync and no_sync is not None:
                    context = no_sync()
                with context, self.precision.autocast():
                    loss, parts = self._forward(batch)
                if not torch.isfinite(loss).all():
                    nonfinite = True
                    break
                self.precision.backward(loss / self.grad_accum)
                for key, value in parts.items():
                    if isinstance(value, (int, float)):
                        accumulated[key] = accumulated.get(key, 0.0) + float(value) / self.grad_accum
            if nonfinite:
                self.optimizer.zero_grad(set_to_none=True)
                step += 1
                self.scheduler.step()
                if self.ctx.is_main:
                    self.metrics.log({"step": step, "skipped": "non_finite_loss"})
                continue
            self.precision.unscale(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizer.param_groups for p in group["params"]],
                self.max_grad_norm,
            )
            self.precision.step(self.optimizer)
            self.scheduler.step()
            step += 1
            elapsed = time.perf_counter() - step_started
            self.throughput.update(frames=micro_frames, tokens=micro_tokens, elapsed=elapsed)
            if self.ctx.is_main and (step % max(1, self.log_every) == 0 or step == 1):
                record = {
                    "step": step,
                    "epoch": epoch,
                    "lr": self.scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "frames": micro_frames,
                    "tokens": micro_tokens,
                    **{k: v for k, v in accumulated.items()},
                    **self.throughput.rates(),
                }
                self.metrics.log(record)
            if self.checkpoint_dir is not None and self.save_every and step % self.save_every == 0:
                save_checkpoint(
                    self.checkpoint_dir,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=step,
                    epoch=epoch,
                )
                if self.on_save is not None:
                    self.on_save(step)
            if epoch >= epochs and step >= total_steps:
                break
        if self.checkpoint_dir is not None:
            save_checkpoint(
                self.checkpoint_dir,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                step=step,
                epoch=epoch,
            )
        return {"step": step, "epoch": epoch}


# ---------------------------------------------------------------------------
# Dataset access
# ---------------------------------------------------------------------------


class NpzIndexDataset:
    """Minimal fallback reader for ``data/tokens/index.json`` + one ``.npz`` per conversation.

    ``src/dataset.py`` (``DuplexShardDataset``) is authoritative; this reader
    exists so the stage scripts can be exercised before/independently of it and
    documents the item keys.  Each ``.npz`` holds ``user_codes [T,12]``,
    ``agent_codes [T,12]``, ``text [T]``, ``control [T]`` and optionally
    ``valid_frames``, ``start_frame`` and ``id``.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        index_path = self.root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"{index_path} is missing")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = index.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"{index_path} needs a non-empty 'shards' list")
        self.shards = [self.root / (entry["path"] if isinstance(entry, Mapping) else entry) for entry in shards]

    def __len__(self) -> int:
        return len(self.shards)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.shards[index]
        with np.load(path) as archive:
            total = int(archive["text"].shape[0])
            frames = int(archive["valid_frames"]) if "valid_frames" in archive else total
            valid = (
                np.asarray(archive["valid"], dtype=bool).copy()
                if "valid" in archive
                else np.array([True] * frames + [False] * (total - frames), dtype=bool)
            )
            item = {
                "user_codes": archive["user_codes"],
                "agent_codes": archive["agent_codes"],
                "text": archive["text"],
                "control": archive["control"],
                "valid": valid,
                "valid_frames": int(valid.sum()),
                "start_frame": int(archive["start_frame"]) if "start_frame" in archive else 0,
                "id": str(archive["id"]) if "id" in archive else path.stem,
            }
        return item


def open_shard_dataset(
    tokens_dir: str | Path,
    *,
    split: str | None = None,
    seq_frames: int | None = None,
    seed: int = 0,
) -> Any:
    """Open ``src.dataset.DuplexShardDataset`` when present, else the fallback reader."""
    root = Path(tokens_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Token shards not found at {root}. Build them with `make data` (tokenize.py writes "
            "data/tokens/tokens_*.bin + index.json) or pass --tokens."
        )
    dataset_error: Exception | None = None
    try:
        from . import dataset as dataset_module
    except Exception as exc:  # noqa: BLE001 - fall back with context
        dataset_error = exc
        dataset_module = None
    if dataset_module is not None:
        cls = getattr(dataset_module, "DuplexShardDataset", None)
        if cls is not None:
            kwargs: dict[str, Any] = {}
            try:
                signature = inspect.signature(cls)
                if "split" in signature.parameters and split is not None:
                    kwargs["split"] = split
                if "seq_frames" in signature.parameters and seq_frames is not None:
                    kwargs["seq_frames"] = int(seq_frames)
                if "seed" in signature.parameters:
                    kwargs["seed"] = int(seed)
                return cls(str(root), **kwargs)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"src.dataset.DuplexShardDataset could not open {root}: {exc}. Check that tokenize.py "
                    "produced the RPT1 shards and index.json, or remove --split."
                ) from exc
    try:
        return NpzIndexDataset(root)
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"No usable token dataset at {root}: src.dataset is unavailable ({dataset_error}) and the fallback "
            f"npz index failed ({exc}). Run `make data` / src/tokenize.py first."
        ) from exc


class WindowSampler:
    """Yield shuffled batches of contiguous windows over a conversation dataset."""

    def __init__(
        self,
        dataset: Any,
        *,
        seq_frames: int,
        batch_size: int,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        windows_per_item: int = 4,
    ) -> None:
        if seq_frames <= 0 or batch_size <= 0:
            raise ValueError("seq_frames and batch_size must be positive")
        self.dataset = dataset
        self.seq_frames = int(seq_frames)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.windows_per_item = max(1, int(windows_per_item))

    def __len__(self) -> int:
        return len(self.dataset)

    def _windows(self, epoch: int) -> Iterator[dict[str, Any]]:
        order = list(range(len(self.dataset)))
        rng = random.Random(self.seed + epoch)
        rng.shuffle(order)
        order = order[self.rank :: self.world_size]
        length = len(self.dataset)
        for index in order:
            # DuplexShardDataset derives its random window from the item index;
            # offsetting by epoch keeps successive epochs on different windows.
            lookup = (index + epoch * max(1, length)) % max(1, length)
            item = self.dataset[lookup]
            total = item_frame_count(item)
            if total <= 0:
                continue
            count = min(self.windows_per_item, max(1, math.ceil(total / self.seq_frames)))
            if total <= self.seq_frames:
                starts = [0]
            else:
                last = total - self.seq_frames
                starts = sorted(
                    {
                        rng.randrange(0, last + 1)
                        for _ in range(count)
                    }
                )
                starts[0] = min(starts[0], last)
            for start in starts:
                window = window_item(item, start, self.seq_frames)
                if int(window["valid_frames"]) > 0:
                    yield window

    def batches(self, epoch: int) -> Iterator[list[dict[str, Any]]]:
        batch: list[dict[str, Any]] = []
        for window in self._windows(epoch):
            batch.append(window)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def iter_text_jsonl(path: str | Path, *, batch_size: int = 2) -> Iterator[list[str]]:
    """Plain text batches for ``--text-jsonl`` (fields: ``text`` or ``messages``)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"--text-jsonl {path} does not exist")
    batch: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text")
            if text is None and "messages" in record:
                text = "\n".join(str(message.get("content", "")) for message in record["messages"])
            if not text:
                continue
            batch.append(str(text))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Tiny CPU components for --smoke and tests
# ---------------------------------------------------------------------------


class ToyTokenizer:
    """The minimum tokenizer surface the smoke path and layout tests need."""

    def __init__(self) -> None:
        self.pad_token_id = 3
        self.unk_token_id = 0
        self._tokens = {"<|listen|>": 4, "<|speak|>": 5, "<|backchannel|>": 6, "<|stop|>": 7}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._tokens.get(token, self.unk_token_id)

    def __len__(self) -> int:
        return 64


TOY_CONTROL_IDS: dict[str, int] = {"PAD": 3, "LISTEN": 4, "SPEAK": 5, "BACKCHANNEL": 6, "STOP": 7}
TOY_AUDIO_TOKEN_ID = 1
TOY_AUDIO_OUT_TOKEN_ID = 2


class ToyFeatureExtractor:
    """Implements the LFM3 80 ms grid arithmetic without downloading a checkpoint.

    ``n_fft=400``, ``hop_length=160``, ``encoder_stride=8``; 1280 samples per
    80 ms frame yields seven mel frames and exactly one encoder token.
    """

    sampling_rate = USER_SAMPLE_RATE
    feature_size = 128
    n_fft = 400
    hop_length = 160
    encoder_stride = 8

    def pad_length(self, samples: int) -> int:
        if samples < self.n_fft:
            return self.n_fft
        return samples + (-(samples - self.n_fft) % self.hop_length)

    def num_frames(self, samples: int) -> int:
        return 1 + (self.pad_length(samples) - self.n_fft) // self.hop_length

    def num_encoder_tokens(self, mel_frames: int) -> int:
        return -(mel_frames // -self.encoder_stride)

    def __call__(self, clips: Sequence[Any], *, sampling_rate: int | None = None, return_tensors: str | None = None):
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(f"toy extractor expects {self.sampling_rate} Hz, got {sampling_rate}")
        if isinstance(clips, np.ndarray) and clips.ndim == 1:
            clips = [clips]
        widths = [self.num_frames(int(np.asarray(clip).size)) for clip in clips]
        width = max(widths)
        features = torch.zeros(len(clips), self.feature_size, width)
        mask = torch.zeros(len(clips), width, dtype=torch.long)
        for row, row_width in enumerate(widths):
            mask[row, :row_width] = 1
        if return_tensors == "pt" or return_tensors is None:
            return {"input_features": features, "feature_attention_mask": mask}
        return {"input_features": features.numpy(), "feature_attention_mask": mask.numpy()}


class ToyCodec(nn.Module):
    """Decode [B, 12, F] codes to deterministic 24 kHz audio (zero signal)."""

    def __init__(self) -> None:
        super().__init__()
        self.frame_samples = FRAME_SAMPLES[AGENT_SAMPLE_RATE]

    def decode(self, codes: torch.Tensor):
        if codes.ndim != 3 or codes.shape[1] != NUM_CODEBOOKS:
            raise ValueError(f"toy codec expects [B, {NUM_CODEBOOKS}, F], got {tuple(codes.shape)}")
        samples = int(codes.shape[-1]) * self.frame_samples
        audio = torch.zeros(codes.shape[0], samples)
        return type("ToyCodecOutput", (), {"audio_values": audio})()


class ToyDuplexModel(nn.Module):
    """A tiny stand-in exposing the LFM3 forward contract used by the trainer."""

    def __init__(
        self,
        *,
        vocab_size: int = 64,
        hidden_size: int = 16,
        num_codebooks: int = NUM_CODEBOOKS,
        codebook_size: int = CODEBOOK_SIZE,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.encoder_stride = 8
        self.audio_token_id = TOY_AUDIO_TOKEN_ID
        self.audio_out_token_id = TOY_AUDIO_OUT_TOKEN_ID
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.audio_head = nn.Linear(self.hidden_size, self.codebook_size)
        self.config = type("ToyConfig", (), {"tie_word_embeddings": False, "use_cache": False})()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_out_codes: torch.Tensor | None = None,
        audio_labels: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_,
    ):
        user_tokens = int((input_ids == self.audio_token_id).sum())
        if input_features is not None:
            if feature_attention_mask is None:
                raise ValueError("input_features require feature_attention_mask")
            expected = sum(
                -(int(row) // -self.encoder_stride) for row in feature_attention_mask.sum(-1).tolist()
            )
            if user_tokens != expected:
                raise ValueError(f"placeholder/feature mismatch: {user_tokens} user tokens vs {expected}")
        elif user_tokens:
            raise ValueError("user audio placeholders require input_features")
        placeholders = int((input_ids == self.audio_out_token_id).sum())
        if audio_out_codes is not None and placeholders != audio_out_codes.shape[0]:
            raise ValueError(f"placeholder/code mismatch: {placeholders} placeholders vs {audio_out_codes.shape[0]}")
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        audio_loss = None
        if audio_labels is not None:
            if audio_out_codes is None:
                raise ValueError("audio_labels require audio_out_codes")
            selected = hidden[input_ids == self.audio_out_token_id]
            predictions = self.audio_head(selected)
            valid = (audio_labels != -100).all(-1)
            if valid.any():
                audio_loss = F.cross_entropy(predictions[valid].float(), audio_labels[valid][:, 0])
            else:
                audio_loss = hidden.sum() * 0.0
        return type("ToyOutput", (), {"logits": logits, "audio_loss": audio_loss})()

    def gradient_checkpointing_enable(self) -> None:
        self._gradient_checkpointing = True


def make_synthetic_items(
    *,
    count: int = 2,
    frames: int = 16,
    seed: int = 0,
    text_frames: int | None = None,
) -> list[dict[str, Any]]:
    """Synthetic token-shard items with the real array layout (uint16/int32/uint8)."""
    rng = np.random.default_rng(seed)
    items: list[dict[str, Any]] = []
    for index in range(count):
        user = rng.integers(0, CODEBOOK_SIZE, size=(frames, NUM_CODEBOOKS), dtype=np.uint16)
        agent = rng.integers(0, CODEBOOK_SIZE, size=(frames, NUM_CODEBOOKS), dtype=np.uint16)
        control = rng.integers(0, len(CONTROL_LABELS), size=frames).astype(np.uint8)
        text = np.zeros(frames, dtype=np.int32)
        if text_frames:
            positions = rng.choice(frames, size=min(text_frames, frames), replace=False)
            text[positions] = rng.integers(8, 40, size=positions.size, dtype=np.int32)
        items.append(
            {
                "id": f"smoke_{index}",
                "user_codes": user,
                "agent_codes": agent,
                "text": text,
                "control": control,
                "valid": np.ones(frames, dtype=bool),
                "valid_frames": frames,
                "start_frame": 0,
                "duration_s": frames / 12.5,
            }
        )
    return items


def make_smoke_components(
    *,
    frames: int = 16,
    batch_size: int = 2,
    seed: int = 0,
    seq_frames: int | None = None,
) -> dict[str, Any]:
    """Toy model/builder/batches for ``--smoke`` (2 steps, CPU, no downloads)."""
    tokenizer = ToyTokenizer()
    control_ids = dict(TOY_CONTROL_IDS)
    builder = DuplexBatchBuilder(
        codec=ToyCodec(),
        feature_extractor=ToyFeatureExtractor(),
        control_ids=control_ids,
        audio_token_id=TOY_AUDIO_TOKEN_ID,
        audio_out_token_id=TOY_AUDIO_OUT_TOKEN_ID,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = ToyDuplexModel()
    items = make_synthetic_items(count=batch_size, frames=frames, seed=seed, text_frames=max(1, frames // 4))

    def batch_factory(epoch: int) -> Iterator[SupervisedBatch]:
        while True:
            yield builder.build_items(items)

    return {
        "tokenizer": tokenizer,
        "control_ids": control_ids,
        "builder": builder,
        "model": model,
        "batch_factory": batch_factory,
    }


# ---------------------------------------------------------------------------
# Actionable preflight helpers used by the stage CLIs
# ---------------------------------------------------------------------------


def processor_source(model_path: str | Path) -> str:
    """Prefer the model directory's own processor config, else the base checkpoint."""
    from .model import BASE_MODEL

    if (Path(model_path) / "processor_config.json").exists():
        return str(model_path)
    return BASE_MODEL


def load_codec(processor: Any, *, source: str | Path | None = None) -> nn.Module:
    """Return the frozen LiquidCodec attached to the processor.

    The released checkpoint bundles it under ``audio_tokenizer/`` and
    ``AutoProcessor`` loads it; if a stripped processor is passed, try the
    source checkpoint explicitly.  Never fabricate a codec.
    """
    codec = getattr(processor, "audio_tokenizer", None)
    if isinstance(codec, nn.Module):
        return codec
    if source is not None:
        try:
            from transformers import AutoModel

            codec = AutoModel.from_pretrained(
                str(source), subfolder="audio_tokenizer", trust_remote_code=True
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"No LiquidCodec on the processor and loading audio_tokenizer/ from {source} failed: {exc}. "
                "Pass --processor /path/to/LiquidAI/LFM3-Omni-2B (with its audio_tokenizer folder) or --codec."
            ) from exc
        if isinstance(codec, nn.Module):
            return codec
    raise RuntimeError(
        "No LiquidCodec available: the processor was loaded without audio_tokenizer. "
        "Point --processor/--codec at the base checkpoint that bundles audio_tokenizer/."
    )


def require_path(path: str | Path | None, *, what: str, hint: str) -> Path:
    if path is None:
        raise SystemExit(f"{what} is required. {hint}")
    resolved = Path(path)
    if not resolved.exists():
        raise SystemExit(f"{what} not found at {resolved}. {hint}")
    return resolved


def require_dir(path: str | Path | None, *, what: str, hint: str) -> Path:
    resolved = require_path(path, what=what, hint=hint)
    if not resolved.is_dir():
        raise SystemExit(f"{what} at {resolved} is not a directory. {hint}")
    return resolved


def load_json_artifact(path: str | Path, *, what: str) -> dict[str, Any]:
    resolved = require_path(path, what=what, hint="Produce it first; never start a stage on a missing gate artifact.")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{what} at {resolved} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{what} at {resolved} must be a JSON object")
    return value


__all__ = [
    "CONTROL_CLASS_WEIGHTS",
    "TEXT_LOSS_WEIGHT",
    "CONTROL_LOSS_WEIGHT",
    "AUDIO_LOSS_WEIGHT",
    "DistributedContext",
    "PrecisionContext",
    "PreparedItem",
    "SupervisedBatch",
    "SupervisedTrainer",
    "JsonlMetrics",
    "ThroughputMeter",
    "WarmupCosineScheduler",
    "DuplexBatchBuilder",
    "WindowSampler",
    "NpzIndexDataset",
    "ToyTokenizer",
    "ToyFeatureExtractor",
    "ToyCodec",
    "ToyDuplexModel",
    "TOY_CONTROL_IDS",
    "TOY_AUDIO_TOKEN_ID",
    "TOY_AUDIO_OUT_TOKEN_ID",
    "atomic_save",
    "build_frame_sequence",
    "build_optimizer",
    "build_text_batch",
    "causal_shift",
    "cleanup_distributed",
    "control_loss",
    "cosine_lr_at",
    "duplex_loss",
    "enable_activation_checkpointing",
    "init_distributed",
    "iter_text_jsonl",
    "load_checkpoint",
    "load_codec",
    "make_precision",
    "make_smoke_components",
    "make_synthetic_items",
    "open_shard_dataset",
    "processor_source",
    "require_dir",
    "require_path",
    "resample_24k_to_16k",
    "save_checkpoint",
    "seed_everything",
    "text_loss",
    "unwrap_model",
    "validate_duplex_batch",
    "window_item",
    "wrap_ddp",
]
