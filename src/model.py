"""Model surgery for Repartee: the four control tokens, the connector check and the freeze sets.

The base checkpoint is ``LiquidAI/LFM3-Omni-2B`` (text LM 2048-wide, LiquidCodec
12 codebooks, 16 kHz causal audio encoder, pretrained ``audio_projector``
connector).  This module deliberately adds **no new module**: the connector is
the checkpoint's existing ``model.model.audio_projector`` and the only new
parameters are the four control-token embedding rows.

What this module does
---------------------
* loads the base model with ``AutoModelForMultimodalLM`` in bfloat16 on CUDA
  and float32 elsewhere (``--quantize int4`` is a serving-only option),
* adds ``<|listen|> <|speak|> <|backchannel|> <|stop|>`` as additional special
  tokens and initialises each new embedding row with the **mean** of the
  pre-existing embedding rows,
* verifies the connector shape/parameter count,
* freezes the audio encoder, codec, depth decoder, vision tower and every norm
  module in every stage,
* exposes per-stage freeze plans (stage 1: connector + four embedding rows;
  stage 2: the LLM + connector + tokens; DPO: everything, LoRA comes later),
* saves ``save_pretrained`` weights plus the tokenizer and ``repartee.json``.

Deviations / limitations (also sent to REPORT.md by the parent)
---------------------------------------------------------------
* PyTorch cannot mark individual rows of one ``nn.Embedding`` parameter as
  ``requires_grad``.  Stage 1 therefore turns the whole embedding weight on
  and installs a **gradient row mask** (a tensor hook).  Rows outside the four
  control ids receive a zero gradient, and the stage-1 optimizer is built with
  ``weight_decay=0`` for embedding weights (see ``train_common``).  The printed
  trainable count is the *effective* count (four rows), which is what the 2 %
  budget is asserted against.
* The LiquidCodec is not a submodule of the language model; when it is attached
  to the process the codec is frozen through the same helper, but it is not
  saved with the LM.
* ``<|listen|>`` etc. are added to ``additional_special_tokens``; the mapping
  is written to ``repartee.json`` so training never has to guess ids.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

BASE_MODEL = "LiquidAI/LFM3-Omni-2B"
REPARTEE_VERSION = 1
REPARTEE_FILENAME = "repartee.json"

# Order matches src.schema.CONTROL_TOKEN_NAMES; PAD maps to tokenizer.pad_token_id.
CONTROL_TOKENS = ("<|listen|>", "<|speak|>", "<|backchannel|>", "<|stop|>")
CONTROL_LABELS = ("LISTEN", "SPEAK", "BACKCHANNEL", "STOP")
ALL_CONTROL_LABELS = ("PAD",) + CONTROL_LABELS

# The connector target from BLUEPRINT.md section 9.
CONNECTOR_TARGET_RANGE = (5_000_000, 30_000_000)
STAGE1_MAX_TRAINABLE_RATIO = 0.02


# ---------------------------------------------------------------------------
# Control-token ids
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlTokenIds:
    """Vocabulary ids of the five frame-level control labels."""

    pad: int
    listen: int
    speak: int
    backchannel: int
    stop: int

    def __post_init__(self) -> None:
        values = [self.pad, self.listen, self.speak, self.backchannel, self.stop]
        if any(not isinstance(value, int) or value < 0 for value in values):
            raise ValueError(f"control ids must be non-negative ints, got {values}")
        if len(set(values)) != len(values):
            raise ValueError(f"control ids must be distinct, got {values}")

    def by_label(self) -> dict[str, int]:
        return {
            "PAD": self.pad,
            "LISTEN": self.listen,
            "SPEAK": self.speak,
            "BACKCHANNEL": self.backchannel,
            "STOP": self.stop,
        }

    def to_dict(self) -> dict[str, int]:
        return dict(self.by_label())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ControlTokenIds":
        missing = set(ALL_CONTROL_LABELS) - set(value)
        if missing:
            raise ValueError(f"control id mapping is missing {sorted(missing)}")
        return cls(**{label.lower(): int(value[label]) for label in ALL_CONTROL_LABELS})


def control_token_ids(tokenizer: Any) -> ControlTokenIds:
    """Map PAD to ``tokenizer.pad_token_id`` and the four new tokens to their ids.

    The control tokens must already exist in the tokenizer; call
    :func:`add_control_tokens` first.  The pad token must exist because PAD is
    the "nobody is speaking" label.
    """
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        raise ValueError("The tokenizer has no pad_token_id; Repartee needs it for the PAD control label")
    unk = getattr(tokenizer, "unk_token_id", None)
    ids: dict[str, int] = {}
    for label, token in zip(CONTROL_LABELS, CONTROL_TOKENS):
        value = tokenizer.convert_tokens_to_ids(token)
        if value is None or value == unk:
            raise ValueError(
                f"{token} is not a known token; call add_control_tokens(tokenizer, model) before reading ids"
            )
        ids[label] = int(value)
    return ControlTokenIds(pad=int(pad), **{label.lower(): value for label, value in ids.items()})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preferred_dtype(device: torch.device) -> torch.dtype:
    """bfloat16 on CUDA when supported, otherwise float16 on CUDA; float32 on CPU.

    The training GPU may be Turing (no bf16); float32 weights (2.17 B ~ 8.7 GB)
    do not fit there, so CUDA always gets a half-precision dtype.
    """
    if device.type == "cuda":
        # Turing exposes an emulated bf16 path that is much slower than fp16 and
        # is not compatible with GradScaler.  Use native fp16 on pre-Ampere.
        try:
            major = torch.cuda.get_device_capability(0)[0]
        except Exception:  # noqa: BLE001 - no device index in exotic setups
            major = 0
        if major >= 8 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def quantization_config(name: str | None) -> Any:
    """Return a ``BitsAndBytesConfig`` for ``int4`` serving, ``None`` otherwise."""
    if name in (None, "none"):
        return None
    if name != "int4":
        raise ValueError(f"Unknown quantization {name!r}; choose 'none' or 'int4'")
    if not torch.cuda.is_available():
        raise RuntimeError("--quantize int4 needs CUDA; bitsandbytes cannot quantize on CPU")
    from transformers import BitsAndBytesConfig

    # Turing has no bf16 compute; use fp16 there.
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        # Only the text LLM is quantized.  The connector, encoder, depth
        # decoder, vision tower and tied text head stay fp16: the connector
        # must remain trainable, and quantizing these small modules only adds
        # dequantization overhead.
        llm_int8_skip_modules=[
            "model.audio_encoder",
            "model.audio_projector",
            "model.depth_decoder",
            "model.codebook_embedding",
            "model.vision_tower",
            "model.multi_modal_projector",
            "model.language_model.embed_tokens",
            "lm_head",
        ],
    )


def load_model(
    checkpoint: str | Path,
    *,
    quantize: str = "none",
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
) -> Any:
    """Load the base (or a prepared Repartee) checkpoint."""
    from transformers import AutoModelForMultimodalLM

    target = resolve_device(device)
    quant = quantization_config(quantize)
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if quant is not None:
        kwargs["quantization_config"] = quant
        kwargs["device_map"] = {"": target.index or 0} if target.type == "cuda" else "auto"
    else:
        kwargs["dtype"] = dtype if dtype is not None else preferred_dtype(target)
    model = AutoModelForMultimodalLM.from_pretrained(str(checkpoint), **kwargs)
    if quant is None:
        model.to(target)
    model.eval()
    return model


def load_processor(
    checkpoint: str | Path,
    *,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
) -> Any:
    """Load ``AutoProcessor``; it carries the tokenizer and the feature extractor."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        str(checkpoint), trust_remote_code=trust_remote_code, local_files_only=local_files_only
    )


def load_tokenizer(checkpoint: str | Path, *, local_files_only: bool = False) -> Any:
    """Tokenizer from a saved/prepared directory, with AutoProcessor as the first try."""
    try:
        processor = load_processor(checkpoint, local_files_only=local_files_only)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
    except Exception:  # noqa: BLE001 - prepared dirs may only carry tokenizer files
        pass
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=local_files_only, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Adding the four tokens
# ---------------------------------------------------------------------------


def find_audio_projector(model: Any) -> nn.Module:
    """Return the checkpoint's existing encoder -> LLM connector.

    The blueprint forbids adding a new connector module, so a missing
    ``model.model.audio_projector`` is a hard error, not something to patch.
    """
    inner = getattr(model, "model", None)
    projector = getattr(inner, "audio_projector", None) if inner is not None else None
    if projector is None:
        raise RuntimeError(
            "Model has no `model.audio_projector`; the Repartee surgery expects the pretrained connector "
            "of LiquidAI/LFM3-Omni-2B (do not add a new module)"
        )
    return projector


def module_shape(module: nn.Module) -> tuple[int, ...] | None:
    """Input/output width of a projector-like module, or the first 2-D weight."""
    in_features = getattr(module, "in_features", None)
    out_features = getattr(module, "out_features", None)
    if isinstance(in_features, int) and isinstance(out_features, int):
        return (in_features, out_features)
    for parameter in module.parameters():
        if parameter.ndim == 2:
            return (int(parameter.shape[1]), int(parameter.shape[0]))
    return None


def count_parameters(module: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())


def mean_initialize_rows(
    embedding: nn.Embedding,
    rows: Sequence[int],
    *,
    reference_rows: int | None = None,
) -> torch.Tensor:
    """Initialise ``rows`` of ``embedding`` from the mean of the existing rows.

    ``reference_rows`` is the vocabulary size *before* the tokenizer grew.  Using
    the mean (not the module's normal init) keeps new tokens close to the
    pretrained embedding distribution.
    """
    rows = [int(row) for row in rows]
    if not rows:
        raise ValueError("mean_initialize_rows needs at least one row")
    weight = embedding.weight
    count = weight.shape[0] if reference_rows is None else int(reference_rows)
    if count <= 0 or count > weight.shape[0]:
        raise ValueError(f"reference_rows must be in 1..{weight.shape[0]}, got {reference_rows}")
    if any(row < 0 or row >= weight.shape[0] for row in rows):
        raise ValueError(f"rows {rows} out of range for an embedding with {weight.shape[0]} rows")
    with torch.no_grad():
        mean = weight[:count].detach().float().mean(dim=0)
        for row in rows:
            weight[row].copy_(mean.to(dtype=weight.dtype, device=weight.device))
    return mean


def assert_tied_embeddings(model: Any) -> bool:
    """Make sure resizing did not break ``lm_head`` <-> ``embed_tokens`` tying."""
    if not bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False)):
        return False
    get_input = getattr(model, "get_input_embeddings", None)
    get_output = getattr(model, "get_output_embeddings", None)
    if get_input is None or get_output is None:
        return False
    input_embedding = get_input()
    output_embedding = get_output()
    if input_embedding is None or output_embedding is None:
        return False
    input_weight = input_embedding.weight
    output_weight = output_embedding.weight
    if input_weight.shape[0] != output_weight.shape[0]:
        raise RuntimeError(
            "Embedding rows diverged after resizing: "
            f"input {tuple(input_weight.shape)} vs output {tuple(output_weight.shape)}"
        )
    if output_weight.data_ptr() != input_weight.data_ptr():
        tie = getattr(model, "tie_weights", None)
        if tie is not None:
            tie()
        output_weight = get_output().weight
    if output_weight.data_ptr() != input_weight.data_ptr():
        raise RuntimeError(
            "config.tie_word_embeddings is true but lm_head.weight and embed_tokens.weight are not the same "
            "parameter after resize; refusing to train (the four rows would update two divergent matrices)"
        )
    return True


def add_control_tokens(tokenizer: Any, model: Any | None = None) -> ControlTokenIds:
    """Add the four control tokens and mean-initialise their embedding rows.

    Idempotent: if the tokenizer already knows the tokens they are not added
    again and no rows are re-initialised.  When ``model`` is given its input
    embeddings are resized to ``len(tokenizer)``.
    """
    # transformers 5 renamed the keyword; support both so a 5.x checkpoint and
    # a pinned 4.x environment behave identically.
    import inspect

    parameters = inspect.signature(tokenizer.add_special_tokens).parameters
    kwargs: dict[str, Any] = {}
    if "replace_extra_special_tokens" in parameters:
        kwargs["replace_extra_special_tokens"] = False
    elif "replace_additional_special_tokens" in parameters:
        kwargs["replace_additional_special_tokens"] = False
    unk = getattr(tokenizer, "unk_token_id", None)
    known_before = all(
        tokenizer.convert_tokens_to_ids(token) not in (None, unk) for token in CONTROL_TOKENS
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(CONTROL_TOKENS)},
        **kwargs,
    )
    control = control_token_ids(tokenizer)
    if model is None:
        return control

    embedding = model.get_input_embeddings()
    if embedding is None:
        raise RuntimeError("Model has no input embeddings")
    old_rows = int(embedding.weight.shape[0])
    target_rows = max(old_rows, len(tokenizer))
    if target_rows != old_rows:
        model.resize_token_embeddings(target_rows)
        embedding = model.get_input_embeddings()
        # LFM3's vocabulary lives in the nested text config; keep both the
        # checkpoint config and the tokenizer consistent with the new rows.
        config = getattr(model, "config", None)
        if config is not None:
            if hasattr(config, "vocab_size"):
                config.vocab_size = int(target_rows)
            text_config = getattr(config, "text_config", None)
            if text_config is not None and hasattr(text_config, "vocab_size"):
                text_config.vocab_size = int(target_rows)
    if int(embedding.weight.shape[0]) < len(tokenizer):
        raise RuntimeError(
            f"Tokenizer grew to {len(tokenizer)} tokens but the embedding has {embedding.weight.shape[0]} rows"
        )
    # Tokenizers such as LFM3's assign added special tokens to unused slots
    # *inside* the existing vocabulary (125033..125036 for a 128000-row table).
    # Those rows are not freshly allocated, but they were never trained as
    # control tokens; initialise them whenever the tokenizer did not know them
    # before this call.
    if not known_before:
        mean_initialize_rows(embedding, list(control.by_label().values()), reference_rows=old_rows)
    assert_tied_embeddings(model)
    return control


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------


def freeze_module(module: nn.Module) -> int:
    """Set every parameter of ``module`` to ``requires_grad=False``; return the count."""
    frozen = 0
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        frozen += int(parameter.numel())
    return frozen


def _get_child(model: Any, dotted: str) -> nn.Module | None:
    """Resolve ``dotted`` from the model, tolerating an extra ``model.`` prefix."""
    candidates = [dotted]
    if dotted.startswith("model."):
        candidates.append(dotted[len("model.") :])
    else:
        candidates.append("model." + dotted)
    for candidate in candidates:
        current: Any = model
        for part in candidate.split("."):
            if current is None:
                break
            current = getattr(current, part, None)
        if isinstance(current, nn.Module):
            return current
    return None


#: Modules that are frozen in every stage.  ``model.codec`` / ``audio_tokenizer``
#: are optional because the LiquidCodec normally lives on the processor.
ALWAYS_FROZEN_PATHS = (
    "model.audio_encoder",
    "model.depth_decoder",
    "model.vision_tower",
    "model.multi_modal_projector",
    "model.codebook_embedding",
    "model.codec",
    "model.audio_tokenizer",
    "audio_tokenizer",
    "codec",
)


def norm_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """All modules that are norms, by class name or leaf attribute name."""
    found: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1].lower() if name else ""
        if "norm" in type(module).__name__.lower() or leaf.endswith("norm") or leaf.startswith("norm"):
            found.append((name, module))
    return found


def freeze_norms(model: nn.Module) -> list[str]:
    """Freeze every norm module; returns the frozen module names."""
    names: list[str] = []
    for name, module in norm_modules(model):
        freeze_module(module)
        names.append(name)
    return names


def freeze_always(model: nn.Module) -> tuple[str, ...]:
    """Apply the always-frozen set: encoder, codec, depth decoder, vision tower, norms."""
    frozen: list[str] = []
    for path in ALWAYS_FROZEN_PATHS:
        module = _get_child(model, path)
        if module is not None:
            freeze_module(module)
            frozen.append(path)
    freeze_norms(model)
    return tuple(frozen)


def remove_embedding_row_mask(embedding: nn.Embedding) -> None:
    state = getattr(embedding, "_repartee_row_mask", None)
    if not state:
        return
    handle = state.get("handle")
    if handle is not None:
        handle.remove()
    delattr(embedding, "_repartee_row_mask")


def install_embedding_row_mask(embedding: nn.Embedding, rows: Sequence[int]) -> torch.Tensor:
    """Zero the embedding gradient outside ``rows`` (stage 1 row-level freezing)."""
    remove_embedding_row_mask(embedding)
    rows = tuple(int(row) for row in rows)
    weight = embedding.weight
    mask = torch.zeros(weight.shape[0], dtype=torch.bool, device=weight.device)
    mask[list(rows)] = True

    def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
        return grad * mask.unsqueeze(-1).to(dtype=grad.dtype, device=grad.device)

    handle = weight.register_hook(_mask_grad)
    embedding._repartee_row_mask = {"handle": handle, "rows": rows, "mask": mask}
    return mask


def embedding_row_mask(embedding: nn.Embedding) -> torch.Tensor | None:
    state = getattr(embedding, "_repartee_row_mask", None)
    return None if not state else state["mask"]


@dataclass(frozen=True)
class FreezePlan:
    stage: str
    frozen_modules: tuple[str, ...]
    trainable_modules: tuple[str, ...]
    embedding_rows: tuple[int, ...] | None = None


def configure_trainability(model: nn.Module, stage: int | str, control: ControlTokenIds) -> FreezePlan:
    """Freeze everything, then unfreeze exactly the parameters of ``stage``.

    * stage 1: ``audio_projector`` and the four control embedding rows,
    * stage 2: the text LLM (all non-norm parameters), ``audio_projector`` and
      the control-token rows (the embedding is fully trainable then),
    * ``dpo``: nothing (LoRA adapters are attached afterwards).

    Norms are re-frozen last in every stage.
    """
    stage = str(stage)
    if stage not in {"1", "2", "dpo"}:
        raise ValueError(f"stage must be '1', '2' or 'dpo', got {stage!r}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    frozen = freeze_always(model)
    trainable: list[str] = []
    embedding_rows: tuple[int, ...] | None = None

    if stage == "1":
        projector = find_audio_projector(model)
        for parameter in projector.parameters():
            parameter.requires_grad_(True)
        trainable.append("audio_projector")
        embedding = model.get_input_embeddings()
        embedding.weight.requires_grad_(True)
        embedding_rows = tuple(control.by_label()[label] for label in CONTROL_LABELS)
        install_embedding_row_mask(embedding, embedding_rows)
        trainable.append("control_token_embeddings")
    elif stage == "2":
        language_model = _get_child(model, "model.language_model")
        if language_model is None:
            raise RuntimeError("Model has no model.language_model to unfreeze for stage 2")
        for parameter in language_model.parameters():
            parameter.requires_grad_(True)
        trainable.append("language_model")
        projector = find_audio_projector(model)
        for parameter in projector.parameters():
            parameter.requires_grad_(True)
        trainable.append("audio_projector")
        embedding = model.get_input_embeddings()
        embedding.weight.requires_grad_(True)
        remove_embedding_row_mask(embedding)
        trainable.append("control_token_embeddings")
        # An untied output head is not part of language_model; it must learn too.
        get_output = getattr(model, "get_output_embeddings", None)
        output_embedding = get_output() if get_output is not None else None
        if output_embedding is not None and output_embedding.weight.data_ptr() != embedding.weight.data_ptr():
            for parameter in output_embedding.parameters():
                parameter.requires_grad_(True)
            trainable.append("lm_head")

    # norms never train, even when their parent module was unfrozen above
    freeze_norms(model)
    assert_tied_embeddings(model)
    return FreezePlan(
        stage=stage,
        frozen_modules=frozen,
        trainable_modules=tuple(trainable),
        embedding_rows=embedding_rows,
    )


def effective_trainable_parameters(model: nn.Module) -> dict[str, Any]:
    """Count trainable parameters, honouring the stage-1 embedding row mask."""
    embedding = model.get_input_embeddings()
    mask = embedding_row_mask(embedding) if embedding is not None else None
    total = 0
    effective = 0
    raw = 0
    for _, parameter in model.named_parameters():
        total += int(parameter.numel())
        if not parameter.requires_grad:
            continue
        raw += int(parameter.numel())
        if mask is not None and parameter is embedding.weight:
            effective += int(mask.sum()) * int(parameter.shape[1])
        else:
            effective += int(parameter.numel())
    ratio = effective / total if total else 0.0
    return {"total": total, "trainable": effective, "raw_trainable": raw, "ratio": ratio}


def assert_stage1_budget(model: nn.Module, max_ratio: float = STAGE1_MAX_TRAINABLE_RATIO) -> dict[str, Any]:
    summary = effective_trainable_parameters(model)
    if summary["ratio"] >= max_ratio:
        raise RuntimeError(
            f"Stage-1 trainable parameters are {summary['ratio']:.4%} of the {summary['total']:,} total, "
            f"at or above the blueprint's {max_ratio:.0%} limit; refusing to train"
        )
    return summary


def _family_for(name: str) -> str:
    families = (
        "model.language_model",
        "model.audio_encoder",
        "model.audio_projector",
        "model.codebook_embedding",
        "model.depth_decoder",
        "model.vision_tower",
        "model.multi_modal_projector",
        "lm_head",
        "audio_tokenizer",
        "codec",
    )
    for family in families:
        if name == family or name.startswith(family + "."):
            return family
    return "other"


def module_freeze_table(model: nn.Module) -> list[dict[str, Any]]:
    """Per-module params/trainable/dtype rows for the REPORT.md freeze table."""
    embedding = model.get_input_embeddings()
    mask = embedding_row_mask(embedding) if embedding is not None else None
    rows: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        family = _family_for(name)
        row = rows.setdefault(
            family,
            {"module": family, "params": 0, "trainable": 0, "dtypes": set(), "frozen_parts": 0, "total_parts": 0},
        )
        count = int(parameter.numel())
        trainable = count if parameter.requires_grad else 0
        if mask is not None and parameter is embedding.weight:
            trainable = int(mask.sum()) * int(parameter.shape[1])
        row["params"] += count
        row["trainable"] += trainable
        row["dtypes"].add(str(parameter.dtype).replace("torch.", ""))
        row["total_parts"] += 1
        row["frozen_parts"] += 0 if parameter.requires_grad else 1
    table: list[dict[str, Any]] = []
    for family in (
        "model.language_model",
        "model.audio_encoder",
        "model.audio_projector",
        "model.codebook_embedding",
        "model.depth_decoder",
        "model.vision_tower",
        "model.multi_modal_projector",
        "lm_head",
        "audio_tokenizer",
        "codec",
        "other",
    ):
        row = rows.get(family)
        if row is None:
            continue
        row = dict(row)
        row["dtype"] = ",".join(sorted(row.pop("dtypes"))) or "-"
        row["frozen"] = row["trainable"] == 0
        table.append(row)
    return table


def format_freeze_table(table: Iterable[Mapping[str, Any]]) -> str:
    header = f"{'module':32} {'params':>14} {'trainable':>14} {'dtype':>10}  frozen"
    lines = [header, "-" * len(header)]
    for row in table:
        lines.append(
            f"{str(row['module']):32} {int(row['params']):14,} {int(row['trainable']):14,} "
            f"{str(row['dtype']):>10}  {bool(row['frozen'])}"
        )
    return "\n".join(lines)


def print_trainability(model: nn.Module, control: ControlTokenIds | None = None) -> dict[str, Any]:
    table = module_freeze_table(model)
    summary = effective_trainable_parameters(model)
    print(format_freeze_table(table))
    norm_names = [name for name, _ in norm_modules(model)]
    print(
        f"\nnorms frozen: {len(norm_names)} | total params: {summary['total']:,} | "
        f"trainable (effective): {summary['trainable']:,} ({summary['ratio']:.4%})"
    )
    if control is not None:
        print(f"control ids: {control.to_dict()}")
    return {"table": table, "summary": summary, "norms": norm_names}


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def repartee_metadata(
    control: ControlTokenIds,
    *,
    base_model: str = BASE_MODEL,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "version": REPARTEE_VERSION,
        "name": "repartee-2b",
        "base_model": str(base_model),
        "frame_rate": 12.5,
        "num_codebooks": 12,
        "codebook_size": 2017,
        "audio_eos_code": 2016,
        "control_tokens": {token: control.by_label()[label] for label, token in zip(CONTROL_LABELS, CONTROL_TOKENS)},
        "control_ids": control.to_dict(),
        "pad_token_id": control.pad,
    }
    if extra:
        metadata.update(dict(extra))
    return metadata


_PROCESSOR_CACHE: dict[str, Any] = {}


def _processor_for(base_model: str | Path) -> Any | None:
    """Load (once) the processor that carries the feature extractor and codec."""
    key = str(base_model)
    if key in _PROCESSOR_CACHE:
        return _PROCESSOR_CACHE[key]
    try:
        local = Path(key).is_dir()
        processor = load_processor(key, local_files_only=local)
    except Exception as error:  # noqa: BLE001 - a weights-only checkpoint is still usable with --processor
        logger.warning("could not load processor from %s for saving (%s)", base_model, error)
        _PROCESSOR_CACHE[key] = None
        return None
    _PROCESSOR_CACHE[key] = processor
    return processor


def save_prepared(
    model: Any,
    tokenizer: Any,
    control: ControlTokenIds,
    output: str | Path,
    *,
    base_model: str = BASE_MODEL,
    extra: Mapping[str, Any] | None = None,
    processor: Any | None = None,
) -> Path:
    """``save_pretrained`` + tokenizer + ``repartee.json`` (ids and version)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # The upstream checkpoint ships a greedy generation_config with
    # temperature=0/top_k=1.  transformers 5 refuses to save that combination
    # because the sampling fields are unused when do_sample=False.
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and not bool(getattr(generation_config, "do_sample", False)):
        for field in ("temperature", "top_k", "top_p", "min_p"):
            if hasattr(generation_config, field):
                setattr(generation_config, field, None)
    model.save_pretrained(str(output))
    processor = processor if processor is not None else _processor_for(base_model)
    if processor is not None:
        # Writes processor_config.json + feature extractor + audio_tokenizer/.
        processor.save_pretrained(str(output))
    # Save the grown tokenizer last so base processor files cannot overwrite
    # the four added control tokens.
    tokenizer.save_pretrained(str(output))
    metadata = repartee_metadata(control, base_model=base_model, extra=extra)
    (output / REPARTEE_FILENAME).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def read_repartee_metadata(checkpoint: str | Path) -> dict[str, Any] | None:
    path = Path(checkpoint) / REPARTEE_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def control_ids_from_checkpoint(checkpoint: str | Path, tokenizer: Any) -> ControlTokenIds:
    """Read ids from ``repartee.json`` when present; otherwise from the tokenizer."""
    metadata = read_repartee_metadata(checkpoint)
    if metadata and "control_ids" in metadata:
        control = ControlTokenIds.from_dict(metadata["control_ids"])
        for label, token in zip(CONTROL_LABELS, CONTROL_TOKENS):
            if tokenizer.convert_tokens_to_ids(token) != control.by_label()[label]:
                raise RuntimeError(f"repartee.json and tokenizer disagree about {token}; refusing to train")
        return control
    return control_token_ids(tokenizer)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_model_facts(model: Any, tokenizer: Any) -> None:
    config = getattr(model, "config", None)
    print(f"model class: {type(model).__name__}")
    try:
        parameter = next(model.parameters())
        print(f"device: {parameter.device} | dtype: {parameter.dtype}")
    except StopIteration:
        print("device: - | dtype: -")
    for name in ("audio_token_id", "audio_out_token_id", "audio_start_token_id", "text_end_token_id"):
        print(f"{name}: {getattr(config, name, None)}")
    if config is not None and getattr(config, "text_config", None) is not None:
        print(f"text vocab: {config.text_config.vocab_size} | hidden: {config.text_config.hidden_size}")
    print(f"tokenizer length: {len(tokenizer)} | pad id: {tokenizer.pad_token_id}")
    for token in CONTROL_TOKENS:
        value = tokenizer.convert_tokens_to_ids(token)
        print(f"{token}: {value}")
    projector = find_audio_projector(model)
    shape = module_shape(projector)
    params = count_parameters(projector)
    low, high = CONNECTOR_TARGET_RANGE
    status = "in target range" if low <= params <= high else "OUTSIDE the 5-30M target"
    print(f"audio_projector: {type(projector).__name__} shape={shape} params={params:,} ({status})")


def _cmd_inspect(args: argparse.Namespace) -> int:
    model = load_model(args.checkpoint, quantize=args.quantize, device=args.device)
    processor = load_processor(args.checkpoint, local_files_only=args.local_files_only)
    tokenizer = processor.tokenizer
    _print_model_facts(model, tokenizer)
    print("\n" + format_freeze_table(module_freeze_table(model)))
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    if args.quantize != "none":
        raise SystemExit("prepare must save full-precision weights; --quantize int4 is only for serving")
    tokenizer = load_tokenizer(args.checkpoint, local_files_only=args.local_files_only)
    model = load_model(
        args.checkpoint,
        quantize="none",
        device=args.device,
        local_files_only=args.local_files_only,
    )
    control = add_control_tokens(tokenizer, model)
    plan = configure_trainability(model, args.stage, control)
    summary = assert_stage1_budget(model) if plan.stage == "1" else effective_trainable_parameters(model)
    print_trainability(model, control)
    print(f"\nfreeze plan: {plan}")
    if plan.stage == "1":
        print(f"stage-1 budget: {summary['ratio']:.4%} < 2% OK")
    save_prepared(model, tokenizer, control, args.output, base_model=args.checkpoint, extra={"stage": plan.stage})
    print(f"\nsaved prepared checkpoint to {args.output}")
    return 0


def _cmd_freeze_table(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer(args.checkpoint, local_files_only=args.local_files_only)
    control = control_ids_from_checkpoint(args.checkpoint, tokenizer)
    model = load_model(args.checkpoint, quantize=args.quantize, device=args.device)
    configure_trainability(model, args.stage, control)
    print_trainability(model, control)
    if args.stage == "1":
        assert_stage1_budget(model)
    return 0


def _cmd_save(args: argparse.Namespace) -> int:
    if args.quantize != "none":
        raise SystemExit("save must write full-precision weights; --quantize int4 is only for serving")
    tokenizer = load_tokenizer(args.checkpoint, local_files_only=args.local_files_only)
    control = control_ids_from_checkpoint(args.checkpoint, tokenizer)
    model = load_model(args.checkpoint, quantize="none", device=args.device)
    embedding_rows = model.get_input_embeddings().weight.shape[0]
    if embedding_rows != len(tokenizer):
        raise SystemExit(
            f"{args.checkpoint}: embedding has {embedding_rows} rows but the tokenizer has {len(tokenizer)} "
            "tokens; run `python -m src.model prepare` on a base checkpoint instead of save"
        )
    save_prepared(
        model,
        tokenizer,
        control,
        args.output,
        base_model=getattr(model.config, "_name_or_path", BASE_MODEL) or BASE_MODEL,
        extra={"stage": args.stage},
    )
    print(f"saved {args.output}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.model",
        description="Repartee model surgery: inspect, prepare, freeze-table, save.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--checkpoint", default=BASE_MODEL, help="HF id or local checkpoint directory")
        sub.add_argument("--quantize", choices=("none", "int4"), default="none", help="int4 is serving-only")
        sub.add_argument("--device", default=None, help="cuda, cuda:0, cpu, ... (default: auto)")
        sub.add_argument("--local-files-only", action="store_true", help="never touch the network")

    inspect = subparsers.add_parser("inspect", help="load the base model and print connector/token facts")
    add_common(inspect)
    inspect.set_defaults(func=_cmd_inspect)

    prepare = subparsers.add_parser("prepare", help="add the four tokens, freeze stage 1, save")
    add_common(prepare)
    prepare.add_argument("--output", required=True, help="directory for the prepared checkpoint")
    prepare.add_argument("--stage", choices=("1", "2"), default="1")
    prepare.set_defaults(func=_cmd_prepare)

    table = subparsers.add_parser("freeze-table", help="print the per-module freeze table")
    add_common(table)
    table.add_argument("--stage", choices=("1", "2", "dpo"), default="1")
    table.set_defaults(func=_cmd_freeze_table)

    save = subparsers.add_parser("save", help="re-save weights with tokenizer + repartee.json")
    add_common(save)
    save.add_argument("--output", required=True)
    save.add_argument("--stage", choices=("1", "2", "dpo"), default="1")
    save.set_defaults(func=_cmd_save)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (RuntimeError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report backend failures concisely
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
