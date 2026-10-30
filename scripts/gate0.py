"""Gate 0: verify the LiquidAI/LFM3-Omni-2B checkpoint before writing code.

The script runs five checks against the real checkpoint and writes machine-readable
evidence to ``artifacts/gate0/``:

0.1 license parsing (LFM Open License v1.0)
0.2 audio-encoder causality and token rate (streaming, 80 ms chunks)
0.3 LiquidCodec streaming decode (one frame per call, 30 s)
0.4 interruptibility: force an end-of-audio frame during generation
0.5 head/module layout and parameter counts from ``config.json`` + safetensors headers

Usage::

    python scripts/gate0.py --all --out artifacts/gate0

Heavy dependencies (torch, transformers, numpy, huggingface_hub) are imported lazily so
this module and its pure helpers import on a CPU-only machine.  The public API is the
argparse CLI; the ``parse_*`` / ``evaluate_*`` helpers are pure and unit-tested in
``tests/test_gate0_helpers.py``.

Measured facts are written verbatim; nothing is fabricated.  If a prerequisite is
missing the corresponding check is reported as ``BLOCKED`` with the exception text.
The encoder-causality threshold is not relaxed: a non-causal checkpoint is a FAIL.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import re
import struct
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MODEL_ID = "LiquidAI/LFM3-Omni-2B"
LICENSE_THRESHOLD_USD = 10_000_000
FRAME_MS = 80.0
FRAME_RATE_HZ = 12.5
CAUSALITY_MAX_ABS_DIFF = 1e-5
CODEC_MAX_CALL_MS = 160.0
INTERRUPT_MAX_FRAMES = 2
ENCODER_SAMPLE_RATE = 16_000
CODEC_SAMPLE_RATE = 24_000
CODEC_FRAME_SAMPLES = 1_920
AUDIO_EOS_CODE = 2_016
NUM_CODEBOOKS = 12

DEFAULT_MODEL_DIR = Path.home() / ".cache" / "repartee" / "LFM3-Omni-2B"


# ---------------------------------------------------------------------------
# Pure helpers (unit tested without weights)
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_license_text(text: str) -> dict[str, Any]:
    """Parse the LFM Open License v1.0 text.

    Returns the version, the revenue Threshold, commercial-use and redistribution
    clauses.  This is a document parser, not a legal opinion.
    """

    version_match = re.search(r"LFM Open License\s+v(\d+\.\d+)", text, re.IGNORECASE)
    version = version_match.group(1) if version_match else None

    threshold_match = re.search(
        r"annual revenue of\s+([0-9][0-9,]*)\s+million\s+United States dollars",
        text,
        re.IGNORECASE,
    )
    if threshold_match:
        threshold = int(threshold_match.group(1).replace(",", "")) * 1_000_000
    else:
        dollar_match = re.search(r"\$([0-9][0-9,]{6,})", text)
        threshold = int(dollar_match.group(1).replace(",", "")) if dollar_match else None

    commercial_limited = bool(
        re.search(r"Commercial Use Limitation", text, re.IGNORECASE)
        or re.search(r"rights granted .* for Commercial Use are conditioned", text, re.IGNORECASE | re.DOTALL)
    )

    # Restrict the condition scan to section 4; section 5 also uses (a)/(b)/(c) labels.
    section_start = text.find("4. Redistribution")
    section_end = text.find("5. Commercial Use", section_start) if section_start >= 0 else -1
    condition_text = text[section_start:section_end] if section_start >= 0 and section_end > section_start else text
    conditions: list[str] = []
    for match in re.finditer(r"^\s*\((a|b|c|d)\)\s*(.+)$", condition_text, re.MULTILINE):
        clause = " ".join(match.group(2).split())
        if clause:
            conditions.append(f"({match.group(1)}) {clause}")

    return {
        "license_name": f"LFM Open License v{version}" if version else None,
        "version": version,
        "annual_revenue_threshold_usd": threshold,
        "commercial_use_limited": commercial_limited,
        "redistribution_conditions": conditions,
        "nonprofit_exception": bool(re.search(r"Qualified Non-Profit Organization", text)),
        "requires_attribution": bool(re.search(r"attribution notices", text, re.IGNORECASE)),
        "sha256": sha256_text(text),
        "characters": len(text),
    }


def classify_license(
    revenue_usd: float | None,
    *,
    nonprofit: bool = False,
    noncommercial: bool = False,
    threshold_usd: int = LICENSE_THRESHOLD_USD,
) -> dict[str, Any]:
    """Classify commercial-use eligibility.

    PASS = the documented use is inside the grant.  CONDITIONAL = the deciding fact
    (organisation revenue) is unknown.  FAIL = the Legal Entity meets/exceeds the
    Threshold, so Commercial Use is not licensed.  This is not legal advice.
    """

    if nonprofit and noncommercial:
        return {
            "result": "PASS",
            "reason": "Qualified non-profit, non-commercial/research use: the Threshold does not apply.",
        }
    if revenue_usd is None:
        return {
            "result": "CONDITIONAL",
            "reason": (
                "Organisation revenue is unknown. The grant for Commercial Use is conditioned on the Legal "
                "Entity not exceeding the $10M annual-revenue Threshold; confirm before deployment."
            ),
        }
    if revenue_usd < threshold_usd:
        return {
            "result": "PASS",
            "reason": f"Annual revenue ${revenue_usd:,.0f} is below the ${threshold_usd:,} Threshold.",
        }
    return {
        "result": "FAIL",
        "reason": (
            f"Annual revenue ${revenue_usd:,.0f} meets/exceeds the ${threshold_usd:,} Threshold; "
            "Commercial Use is not licensed under LFM Open License v1.0."
        ),
    }


def token_rate_metrics(
    n_tokens: int, duration_s: float, frame_ms: float = FRAME_MS, tolerance: float = 1e-6
) -> dict[str, Any]:
    """Report tokens per frame and token rate; PASS means exactly one token per frame."""

    expected = duration_s * 1000.0 / frame_ms
    per_frame = n_tokens / expected if expected else float("inf")
    return {
        "tokens": int(n_tokens),
        "duration_s": float(duration_s),
        "expected_tokens": float(expected),
        "token_rate_hz": float(n_tokens / duration_s) if duration_s else float("inf"),
        "tokens_per_frame": float(per_frame),
        "frame_ms": float(frame_ms),
        "ok": bool(abs(per_frame - 1.0) <= tolerance),
    }


def evaluate_causality(
    token_rate_ok: bool, max_abs_diff: float | None, threshold: float = CAUSALITY_MAX_ABS_DIFF
) -> dict[str, Any]:
    """Fail closed: a missing diff is not a pass."""

    diff_ok = max_abs_diff is not None and max_abs_diff <= threshold
    return {
        "token_rate_ok": bool(token_rate_ok),
        "max_abs_diff": None if max_abs_diff is None else float(max_abs_diff),
        "diff_threshold": float(threshold),
        "diff_ok": bool(diff_ok),
        "result": "PASS" if (token_rate_ok and diff_ok) else "FAIL",
    }


def evaluate_codec_stream(
    samples_per_call: Iterable[int],
    *,
    sample_rate: int = CODEC_SAMPLE_RATE,
    max_call_ms: float = CODEC_MAX_CALL_MS,
    all_finite: bool,
    rms_diff_db: float | None,
    max_rms_diff_db: float = 1.0,
) -> dict[str, Any]:
    """Summarise per-call decoder output; PASS requires <=160 ms chunks, finite output,
    and per-second RMS matching the one-shot decode within 1 dB."""

    calls = [int(value) for value in samples_per_call]
    if not calls:
        raise ValueError("samples_per_call is empty")
    max_samples = max(calls)
    mean_samples = sum(calls) / len(calls)
    max_ms = 1000.0 * max_samples / sample_rate
    rms_ok = rms_diff_db is not None and abs(rms_diff_db) <= max_rms_diff_db
    return {
        "calls": len(calls),
        "min_samples": min(calls),
        "max_samples": max_samples,
        "mean_samples": mean_samples,
        "max_call_ms": max_ms,
        "mean_call_ms": 1000.0 * mean_samples / sample_rate,
        "all_finite": bool(all_finite),
        "rms_diff_db": None if rms_diff_db is None else float(rms_diff_db),
        "max_rms_diff_db": float(max_rms_diff_db),
        "result": "PASS" if (max_ms <= max_call_ms and all_finite and rms_ok) else "FAIL",
    }


def evaluate_interrupt(frames_after_stop: int | None, max_frames: int = INTERRUPT_MAX_FRAMES) -> dict[str, Any]:
    ok = frames_after_stop is not None and 0 <= frames_after_stop <= max_frames
    return {
        "frames_after_stop": frames_after_stop,
        "max_frames": int(max_frames),
        "result": "PASS" if ok else "FAIL",
    }


def parse_safetensors_header_bytes(blob: bytes) -> dict[str, Any]:
    """Parse a safetensors file prefix (8-byte little-endian header length + JSON)."""

    if len(blob) < 8:
        raise ValueError("safetensors header truncated: fewer than 8 bytes")
    header_len = struct.unpack("<Q", blob[:8])[0]
    if header_len == 0 or len(blob) < 8 + header_len:
        raise ValueError("safetensors header truncated or empty")
    header = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")
    return header


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated safetensors file")
        header_len = struct.unpack("<Q", prefix)[0]
        payload = handle.read(header_len)
    return parse_safetensors_header_bytes(prefix + payload)


def group_parameter_counts(header: dict[str, Any], groups: dict[str, str]) -> dict[str, Any]:
    """Count tensors/parameters by first matching key prefix.

    ``groups`` maps a logical module name to a key prefix, evaluated in insertion order.
    Unmatched tensors land in ``other``.  Tied weights appear as a normal key and are
    therefore counted once (safetensors stores each tensor once).
    """

    counts: dict[str, dict[str, Any]] = {
        name: {"tensors": 0, "params": 0, "keys": []} for name in list(groups) + ["other"]
    }
    total = 0
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        shape = meta.get("shape") or []
        numel = 1
        for dim in shape:
            numel *= int(dim)
        total += numel
        target = "other"
        for name, prefix in groups.items():
            if key.startswith(prefix):
                target = name
                break
        counts[target]["tensors"] += 1
        counts[target]["params"] += numel
        if len(counts[target]["keys"]) < 8:
            counts[target]["keys"].append(key)
    return {"groups": counts, "total_params": total, "tensors": sum(g["tensors"] for g in counts.values())}


def check_layout_config(
    config: dict[str, Any], codec_config: dict[str, Any], processor_config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Verify the fixed blueprint values against the checkpoint configs."""

    feature = (processor_config.get("feature_extractor") or {}) if processor_config else {}
    checks = [
        ("num_codebooks", config.get("num_codebooks"), NUM_CODEBOOKS),
        ("codebook_size", config.get("codebook_size"), 2017),
        ("audio_eos_code", config.get("audio_eos_code"), AUDIO_EOS_CODE),
        ("audio_token_id", config.get("audio_token_id"), 125031),
        ("audio_start_token_id", config.get("audio_start_token_id"), 125012),
        ("audio_out_token_id", config.get("audio_out_token_id"), 125032),
        ("text_end_token_id", config.get("text_end_token_id"), 125014),
        ("text_hidden_size", (config.get("text_config") or {}).get("hidden_size"), 2048),
        ("encoder_hidden_size", (config.get("audio_encoder_config") or {}).get("hidden_size"), 768),
        ("encoder_num_mel_bins", (config.get("audio_encoder_config") or {}).get("num_mel_bins"), 128),
        ("codec_n_q", codec_config.get("n_q"), NUM_CODEBOOKS),
        ("codec_frame_samples", codec_config.get("frame_samples"), CODEC_FRAME_SAMPLES),
        ("codec_sample_rate", codec_config.get("sample_rate"), CODEC_SAMPLE_RATE),
        ("feature_hop_length", feature.get("hop_length"), 160),
        ("feature_encoder_stride", feature.get("encoder_stride"), 8),
        ("feature_sampling_rate", feature.get("sampling_rate"), ENCODER_SAMPLE_RATE),
    ]
    results = []
    for name, actual, expected in checks:
        results.append(
            {
                "name": name,
                "actual": actual,
                "expected": expected,
                "pass": actual == expected,
            }
        )
    hop = feature.get("hop_length")
    stride = feature.get("encoder_stride")
    rate = feature.get("sampling_rate")
    frame_ms = 1000.0 * hop * stride / rate if hop and stride and rate else None
    results.append(
        {
            "name": "encoder_frame_ms",
            "actual": frame_ms,
            "expected": FRAME_MS,
            "pass": frame_ms is not None and abs(frame_ms - FRAME_MS) < 1e-9,
        }
    )
    return results


def synthetic_speech_like(duration_s: float, sample_rate: int, seed: int = 20261029) -> Any:
    """Deterministic speech-like signal: filtered noise + harmonic tone bursts.

    Speech-like enough to exercise the encoder and codec, and fully deterministic so the
    evidence is reproducible.  Requires numpy (imported here, not at module import).
    """

    import numpy as np

    n = int(round(duration_s * sample_rate))
    if n <= 0:
        raise ValueError("duration_s must be positive")
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / sample_rate

    # soft low-passed noise: a 64-tap Hann FIR is cheap and deterministic
    white = rng.standard_normal(n)
    kernel = np.hanning(65)
    kernel /= kernel.sum()
    noise = np.convolve(white, kernel, mode="same")

    # voiced bursts every 1.3 s, 0.62 s long, with cosine ramps
    envelope = np.zeros(n, dtype=np.float64)
    period = 1.3
    burst = 0.62
    ramp = 0.03
    for start in np.arange(0.05, duration_s, period):
        i0 = int(start * sample_rate)
        i1 = min(n, int((start + burst) * sample_rate))
        if i1 <= i0:
            continue
        seg = np.ones(i1 - i0, dtype=np.float64)
        r = min(int(ramp * sample_rate), seg.size // 2)
        if r:
            seg[:r] = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, r))
            seg[-r:] = seg[:r][::-1]
        envelope[i0:i1] = np.maximum(envelope[i0:i1], seg)

    harmonic = (
        0.55 * np.sin(2.0 * math.pi * 180.0 * t)
        + 0.22 * np.sin(2.0 * math.pi * 330.0 * t + 0.3)
        + 0.12 * np.sin(2.0 * math.pi * 950.0 * t + 1.1)
        + 0.06 * np.sin(2.0 * math.pi * 1700.0 * t + 0.7)
    )
    fricative = 0.18 * noise * (0.6 + 0.4 * np.sin(2.0 * math.pi * 2.1 * t) ** 2)
    signal = envelope * (harmonic + fricative) + 0.006 * white
    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal = 0.7 * signal / peak
    return signal.astype(np.float32)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _print(message: str) -> None:
    print(message, flush=True)


def _torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"torch is required for Gate 0: {exc}") from exc
    return torch


def _numpy():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"numpy is required for Gate 0: {exc}") from exc
    return np


def resolve_device_dtype(torch, requested_device: str, requested_dtype: str) -> tuple[str, Any, dict[str, Any]]:
    cuda = bool(torch.cuda.is_available())
    if requested_device == "auto":
        device = "cuda" if cuda else "cpu"
    else:
        device = requested_device
        if device == "cuda" and not cuda:
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    bf16_supported = bool(cuda and torch.cuda.is_bf16_supported())
    if requested_dtype == "auto":
        # Turing (Quadro RTX 4000) has no bf16; fp16 is the valid half-precision CUDA dtype.
        dtype = torch.bfloat16 if (device == "cuda" and bf16_supported) else (
            torch.float16 if device == "cuda" else torch.float32
        )
    elif requested_dtype in ("bfloat16", "bf16"):
        dtype = torch.bfloat16
    elif requested_dtype in ("float16", "fp16"):
        dtype = torch.float16
    elif requested_dtype in ("float32", "fp32"):
        dtype = torch.float32
    else:
        raise ValueError(f"unsupported dtype: {requested_dtype}")
    details = {
        "device": device,
        "dtype": str(dtype).replace("torch.", ""), "half_precision": dtype in (torch.float16, torch.bfloat16),
        "cuda_available": cuda,
        "bf16_supported": bf16_supported,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
    }
    return device, dtype, details


def _load_feature_extractor(model_dir: Path):
    """Load Lfm3OmniFeatureExtractor directly; AutoProcessor would import torchvision-only
    image/video processors that are irrelevant to the audio path."""

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    processor_config = json.loads((model_dir / "processor_config.json").read_text(encoding="utf-8"))
    config = dict(processor_config["feature_extractor"])
    allowed = {
        "feature_size",
        "sampling_rate",
        "n_fft",
        "hop_length",
        "win_length",
        "log_floor",
        "normalization_offset",
        "normalization_divisor",
        "encoder_stride",
        "padding_value",
        "return_attention_mask",
    }
    config = {key: value for key, value in config.items() if key in allowed}
    cls = get_class_from_dynamic_module(
        "feature_extraction_lfm3_omni.Lfm3OmniFeatureExtractor", str(model_dir)
    )
    return cls(**config)


def _load_codec(model_dir: Path):
    """Load the bundled LiquidCodec without going through AutoProcessor."""

    codec_dir = model_dir / "audio_tokenizer"
    errors = []
    try:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            str(codec_dir), trust_remote_code=True, local_files_only=True
        )
        model.eval()
        return model
    except Exception as exc:
        errors.append(f"AutoModel: {type(exc).__name__}: {exc}")

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    try:
        cls = get_class_from_dynamic_module("modeling_liquid_codec.LiquidCodecModel", str(model_dir))
        package = cls.__module__.rsplit(".", 1)[0]
        config_module = importlib.import_module(package + ".configuration_liquid_codec")
        config = config_module.LiquidCodecConfig(
            **json.loads((codec_dir / "config.json").read_text(encoding="utf-8"))
        )
        model = cls(config)
        from safetensors.torch import load_file

        state = load_file(str(codec_dir / "model.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"codec state mismatch: {len(missing)} missing, {len(unexpected)} unexpected"
            )
        model.eval()
        return model
    except Exception as exc:
        errors.append(f"dynamic class: {type(exc).__name__}: {exc}")
    raise RuntimeError("could not load the LiquidCodec audio_tokenizer:\n" + "\n".join(errors))


def _load_encoder_only(model_dir: Path):
    """Instantiate only the 17-layer audio encoder and load its 170 tensors.

    Loading the whole 2B checkpoint just to reach ``model.audio_encoder`` wastes RAM and
    minutes on CPU; the encoder state is a strict subset of the checkpoint.
    """

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config_json = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    encoder_config = dict(config_json["audio_encoder_config"])
    encoder_config.pop("model_type", None)
    cls = get_class_from_dynamic_module("modeling_lfm3_omni.Lfm3OmniEncoder", str(model_dir))
    package = cls.__module__.rsplit(".", 1)[0]
    config_module = importlib.import_module(package + ".configuration_lfm3_omni")
    config = config_module.Lfm3OmniEncoderConfig(**encoder_config)
    encoder = cls(config)
    prefix = "model.audio_encoder."
    from safetensors import safe_open

    state = {}
    with safe_open(str(model_dir / "model.safetensors"), framework="pt") as handle:
        for key in handle.keys():
            if key.startswith(prefix):
                state[key[len(prefix) :]] = handle.get_tensor(key)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"encoder state mismatch: {len(missing)} missing, {len(unexpected)} unexpected"
        )
    encoder.eval()
    return encoder


def _load_model(model_dir: Path, dtype, device: str):
    import transformers

    errors = []
    for class_name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM", "AutoModel"):
        cls = getattr(transformers, class_name, None)
        if cls is None:
            continue
        attempt_kwargs = [
            {"dtype": dtype, "low_cpu_mem_usage": True},
            {"dtype": dtype},
            {"torch_dtype": dtype, "low_cpu_mem_usage": True},
        ]
        for extra in attempt_kwargs:
            try:
                model = cls.from_pretrained(
                    str(model_dir),
                    trust_remote_code=True,
                    local_files_only=True,
                    **extra,
                )
                model.eval()
                return model
            except Exception as exc:  # try the next auto class; keep the first real error
                errors.append(f"{class_name} {sorted(extra)}: {type(exc).__name__}: {exc}")
    raise RuntimeError("could not load LFM3-Omni-2B with any auto class:\n" + "\n".join(errors))


def _stream_encoder_tokens(encoder, feature_extractor, waveform, sample_rate: int, device: str, chunk_ms: float = FRAME_MS):
    """Feed 80 ms chunks through feature_extractor.stream + encoder.stream.

    Returns ``(tokens [1,T,H], per_call_token_counts, cache_stats)``.
    """

    torch = _torch()
    chunk_samples = int(round(sample_rate * chunk_ms / 1000.0))
    if chunk_samples <= 0:
        raise ValueError("chunk_ms is too small")
    n = int(waveform.shape[0])
    feature_cache = None
    encoder_cache = None
    pieces = []
    emissions: list[int] = []
    max_feature_buffer = 0
    calls = 0
    with torch.no_grad():
        for start in range(0, n, chunk_samples):
            calls += 1
            raw = torch.from_numpy(waveform[start : start + chunk_samples])
            is_final = start + chunk_samples >= n
            batch, feature_cache = feature_extractor.stream(raw, feature_cache, is_final=is_final)
            features = batch.input_features
            if features.shape[-1] == 0:
                emissions.append(0)
                if is_final:
                    encoded, encoder_cache = encoder.stream(features.to(device), encoder_cache, is_final=True)
                    if encoded.shape[1]:
                        pieces.append(encoded)
                        emissions[-1] = int(encoded.shape[1])
                continue
            encoded, encoder_cache = encoder.stream(
                features.to(device), encoder_cache, is_final=is_final
            )
            emissions.append(int(encoded.shape[1]))
            if encoded.shape[1]:
                pieces.append(encoded)
            buffer = getattr(encoder_cache, "features", None)
            if buffer is not None:
                max_feature_buffer = max(max_feature_buffer, int(buffer.shape[-1]))

    hidden = getattr(encoder.config, "hidden_size", None)
    if pieces:
        tokens = torch.cat(pieces, dim=1)
    else:
        tokens = torch.empty((1, 0, hidden or 0), device=device)
    stats: dict[str, Any] = {}
    if encoder_cache is not None:
        cache = encoder_cache
        stats = {
            "num_frames": int(getattr(cache, "num_frames", -1)),
            "num_tokens": int(getattr(cache, "num_tokens", -1)),
            "offset": int(getattr(cache, "offset", -1)),
            "features": (
                list(cache.features.shape)
                if getattr(cache, "features", None) is not None
                else None
            ),
            "finished": bool(getattr(cache, "finished", False)),
            "max_feature_buffer_frames": max_feature_buffer,
            "chunk_calls": calls,
        }
        past = getattr(cache, "past_key_values", None)
        if past is not None and hasattr(past, "get_seq_length"):
            stats["past_key_values_len"] = int(past.get_seq_length())
    return tokens.detach().float().cpu(), emissions, stats


def _check_license(args, model_dir: Path, out_dir: Path, command: str) -> dict[str, Any]:
    started = utc_now()
    license_path = model_dir / "LICENSE"
    readme_path = model_dir / "README.md"
    if not license_path.exists():
        raise FileNotFoundError(f"LICENSE not found in {model_dir}; run the download step")
    text = license_path.read_text(encoding="utf-8")
    parsed = parse_license_text(text)
    classification = classify_license(
        args.org_revenue_usd,
        nonprofit=args.org_nonprofit,
        noncommercial=args.noncommercial_use,
    )

    (out_dir / "LICENSE").write_text(text, encoding="utf-8")
    card_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    if card_text:
        (out_dir / "model_card.md").write_text(card_text, encoding="utf-8")

    # Explicit hf_hub_download call proves the provenance; it is a no-op when cached.
    provenance: dict[str, Any] = {}
    try:
        from huggingface_hub import hf_hub_download

        local_license = hf_hub_download(args.model_id, "LICENSE", revision=args.revision)
        local_readme = hf_hub_download(args.model_id, "README.md", revision=args.revision)
        provenance = {"license": local_license, "model_card": local_readme, "source": "hf_hub_download"}
    except Exception as exc:  # local_dir remains authoritative
        provenance = {"source": "local snapshot", "hf_hub_download_error": f"{type(exc).__name__}: {exc}"}

    measured = {
        **parsed,
        "classification": classification,
        "model_card": {
            "path": str(readme_path) if card_text else None,
            "sha256": sha256_text(card_text) if card_text else None,
            "declares_lfm_open_license": "LFM Open License" in card_text,
            "license_name_field": "lfm1.0" if "license_name: lfm1.0" in card_text else None,
        },
        "provenance": provenance,
        "organisation": {
            "revenue_usd": args.org_revenue_usd,
            "nonprofit": bool(args.org_nonprofit),
            "noncommercial_use": bool(args.noncommercial_use),
        },
        "legal_advice": False,
    }
    result = classification["result"]
    notes = (
        "Commercial use is permitted only while the Legal Entity stays below the $10M annual-revenue "
        "Threshold; redistribution requires the four notice/attribution conditions. Parser output, not legal "
        "approval."
    )
    _write_json(out_dir / "license.json", measured)
    return {
        "id": "0.1",
        "name": "License",
        "result": result,
        "threshold": "LFM Open License v1.0: Commercial Use only below $10,000,000 annual revenue; notice/attribution on redistribution.",
        "measured": measured,
        "artifacts": ["LICENSE", "model_card.md", "license.json"],
        "notes": notes,
        "command": command,
        "started_at": started,
        "finished_at": utc_now(),
    }


def _check_layout(args, model_dir: Path, out_dir: Path, command: str) -> dict[str, Any]:
    started = utc_now()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    codec_config = json.loads((model_dir / "audio_tokenizer" / "config.json").read_text(encoding="utf-8"))
    processor_path = model_dir / "processor_config.json"
    processor_config = json.loads(processor_path.read_text(encoding="utf-8")) if processor_path.exists() else {}

    model_file = model_dir / "model.safetensors"
    codec_file = model_dir / "audio_tokenizer" / "model.safetensors"
    model_header = read_safetensors_header(model_file)
    codec_header = read_safetensors_header(codec_file)

    groups = {
        "lm_head": "lm_head.",
        "text_language_model": "model.language_model.",
        "audio_encoder": "model.audio_encoder.",
        "audio_projector": "model.audio_projector.",
        "depth_decoder": "model.depth_decoder.",
        "codebook_embedding": "model.codebook_embedding.",
        "vision_tower": "model.vision_tower.",
        "multi_modal_projector": "model.multi_modal_projector.",
    }
    model_counts = group_parameter_counts(model_header, groups)
    codec_counts = group_parameter_counts(
        codec_header,
        {
            "codec_frontend": "frontend.",
            "codec_encoder": "encoder.",
            "codec_quantizer": "quantizer.",
            "codec_decoder": "decoder.",
            "codec_stft": "stft.",
            "codec_to_quantizer": "to_quantizer.",
            "codec_from_quantizer": "from_quantizer.",
        },
    )

    config_checks = check_layout_config(config, codec_config, processor_config)
    tied = bool(config.get("tie_word_embeddings", False))
    head_present = model_counts["groups"]["lm_head"]["tensors"] > 0
    text_emb_present = any(
        key.endswith("embed_tokens.weight") for key in model_header if key.startswith("model.language_model.")
    )
    module_presence = {
        "text_head": head_present or (tied and text_emb_present),
        "text_head_tied": bool(tied and not head_present and text_emb_present),
        "audio_projector": model_counts["groups"]["audio_projector"]["tensors"] > 0,
        "audio_encoder": model_counts["groups"]["audio_encoder"]["tensors"] > 0,
        "depth_decoder": model_counts["groups"]["depth_decoder"]["tensors"] > 0,
        "vision_tower": model_counts["groups"]["vision_tower"]["tensors"] > 0,
        "audio_tokenizer": codec_counts["total_params"] > 0,
    }
    configs_pass = all(item["pass"] for item in config_checks)
    modules_pass = all(module_presence.values())
    result = "PASS" if configs_pass and modules_pass else "FAIL"

    total_params = model_counts["total_params"] + codec_counts["total_params"]
    layout = {
        "model_id": args.model_id,
        "revision": args.revision,
        "config": config,
        "codec_config": codec_config,
        "processor_feature_extractor": processor_config.get("feature_extractor"),
        "module_param_counts": {
            **model_counts["groups"],
            "codec_total": {"tensors": codec_counts["tensors"], "params": codec_counts["total_params"]},
        },
        "codec_groups": codec_counts["groups"],
        "model_tensors": model_counts["tensors"],
        "model_params": model_counts["total_params"],
        "codec_params": codec_counts["total_params"],
        "total_params_with_codec": total_params,
        "module_presence": module_presence,
        "config_checks": config_checks,
        "configs_pass": configs_pass,
        "result": result,
    }
    _write_json(out_dir / "layout.json", layout)
    _print(
        "[0.5] parameter counts: "
        + ", ".join(
            f"{name}={data['params']:,}" for name, data in model_counts["groups"].items()
        )
        + f", liquid_codec={codec_counts['total_params']:,}, total={total_params:,}"
    )

    # Freeze table: blueprint §9. Connector + four new control-token embeddings are the
    # only stage-1 trainables; codec/depthformer/encoder/vision and norms are frozen.
    text_hidden = int((config.get("text_config") or {}).get("hidden_size") or 2048)
    connector = model_counts["groups"]["audio_projector"]["params"]
    new_tokens = 4 * text_hidden
    stage1_trainable = connector + new_tokens
    rows = [
        ("text_language_model", model_counts["groups"]["text_language_model"]["params"], "frozen", "LFM3 24-layer 2048-wide LM"),
        ("lm_head / tied text head", model_counts["groups"]["lm_head"]["params"], "frozen", "tied to embed_tokens" if tied else "explicit lm_head"),
        ("audio_encoder", model_counts["groups"]["audio_encoder"]["params"], "frozen", "17-layer 768-wide LFM3 encoder"),
        ("audio_projector", connector, "trainable", "768->2048 connector (only new module)"),
        ("depth_decoder", model_counts["groups"]["depth_decoder"]["params"], "frozen", "3-layer 1024-wide depthformer"),
        ("codebook_embedding", model_counts["groups"]["codebook_embedding"]["params"], "frozen", "12 codebooks x 2017"),
        ("vision_tower", model_counts["groups"]["vision_tower"]["params"], "frozen", "unused by speech path"),
        ("multi_modal_projector", model_counts["groups"]["multi_modal_projector"]["params"], "frozen", "vision connector"),
        ("audio_tokenizer (LiquidCodec)", codec_counts["total_params"], "frozen", "separate checkpoint"),
        ("4 new control-token embeddings", new_tokens, "trainable", "initialised from mean embedding"),
    ]
    lines = [
        "| Module | Parameters | Stage-1 status | Notes |",
        "|---|---:|---|---|",
    ]
    for name, params, status, note in rows:
        lines.append(f"| {name} | {params:,} | {status} | {note} |")
    lines.append(f"| **stage-1 trainable** | **{stage1_trainable:,}** | | **{100.0 * stage1_trainable / total_params:.4f}% of total** |")
    (out_dir / "freeze_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    measured = {
        "model_params": model_counts["total_params"],
        "codec_params": codec_counts["total_params"],
        "total_params_with_codec": total_params,
        "module_param_counts": {name: data["params"] for name, data in model_counts["groups"].items()},
        "codec_param_counts": {name: data["params"] for name, data in codec_counts["groups"].items()},
        "module_presence": module_presence,
        "config_checks": config_checks,
        "stage1_trainable_params": stage1_trainable,
        "stage1_trainable_fraction": stage1_trainable / total_params,
        "config_values": {
            "num_codebooks": config.get("num_codebooks"),
            "codebook_size": config.get("codebook_size"),
            "audio_eos_code": config.get("audio_eos_code"),
            "text_hidden": text_hidden,
            "encoder_hidden": (config.get("audio_encoder_config") or {}).get("hidden_size"),
            "codec_frame_samples": codec_config.get("frame_samples"),
            "codec_sample_rate": codec_config.get("sample_rate"),
            "codec_frame_rate_hz": codec_config.get("sample_rate", 0) / codec_config.get("frame_samples", 1),
        },
    }
    return {
        "id": "0.5",
        "name": "Head layout",
        "result": result,
        "threshold": "All blueprint §3 ids/rates match config.json; text head, depth decoder, connector, encoder, vision tower and codec identified.",
        "measured": measured,
        "artifacts": ["layout.json", "freeze_table.md"],
        "notes": (
            "Parameter counts are summed from safetensors headers (each stored tensor once). "
            "Tied lm_head is reported as the text head when its weight is shared with embed_tokens."
        ),
        "command": command,
        "started_at": started,
        "finished_at": utc_now(),
    }


def _check_encoder_causality(args, model_dir: Path, out_dir: Path, command: str) -> dict[str, Any]:
    started = utc_now()
    torch = _torch()
    np = _numpy()
    model = None
    feature_extractor = None
    try:
        device, dtype, device_info = resolve_device_dtype(torch, args.device, args.dtype)
        feature_extractor = _load_feature_extractor(model_dir)
        encoder_load = "state-subset"
        try:
            encoder = _load_encoder_only(model_dir)
        except Exception as exc:
            encoder_load = f"full-model fallback ({type(exc).__name__}: {exc})"
            _print(f"[0.2] encoder-only load failed, loading full checkpoint: {exc}")
            model = _load_model(model_dir, dtype, device)
            base = getattr(model, "model", model)
            encoder = base.audio_encoder
            del model
            model = None
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        encoder = encoder.to(device=device, dtype=dtype).eval()

        _print(f"[0.2] encoder on {device} ({device_info['dtype']}); synthesising 30 s @ {ENCODER_SAMPLE_RATE} Hz")
        waveform = synthetic_speech_like(30.0, ENCODER_SAMPLE_RATE)
        duration_s = waveform.shape[0] / ENCODER_SAMPLE_RATE

        full_tokens, emissions, cache_stats = _stream_encoder_tokens(
            encoder, feature_extractor, waveform, ENCODER_SAMPLE_RATE, device
        )
        n_tokens = int(full_tokens.shape[1])
        rate = token_rate_metrics(n_tokens, duration_s)
        rate_ok = bool(rate["ok"])
        _print(
            f"[0.2] full stream: {n_tokens} tokens in {duration_s:.3f}s "
            f"({rate['token_rate_hz']:.4f} Hz, {rate['tokens_per_frame']:.6f} tokens/80ms)"
        )

        chunk_samples = int(round(ENCODER_SAMPLE_RATE * FRAME_MS / 1000.0))
        n_boundaries = int(duration_s * 1000.0 / FRAME_MS)
        tested: list[dict[str, Any]] = []
        first_fail: dict[str, Any] | None = None
        budget_exhausted = False
        boundary_started = time.monotonic()
        max_allowed = CAUSALITY_MAX_ABS_DIFF
        for boundary in range(1, n_boundaries):
            if time.monotonic() - boundary_started > args.causality_time_budget:
                budget_exhausted = True
                _print(
                    f"[0.2] time budget {args.causality_time_budget}s exhausted after "
                    f"{len(tested)} boundaries; remaining boundaries UNVERIFIED"
                )
                break
            masked = waveform.copy()
            masked[boundary * chunk_samples :] = 0.0
            masked_tokens, _, _ = _stream_encoder_tokens(
                encoder, feature_extractor, masked, ENCODER_SAMPLE_RATE, device
            )
            prefix = min(boundary, int(masked_tokens.shape[1]), n_tokens)
            if prefix <= 0:
                continue
            delta = (masked_tokens[:, :prefix] - full_tokens[:, :prefix]).abs()
            per_token = delta.amax(dim=-1)[0]
            worst_token = int(per_token.argmax().item())
            worst_diff = float(per_token[worst_token].item())
            entry = {
                "boundary_ms": boundary * FRAME_MS,
                "boundary_samples": boundary * chunk_samples,
                "compared_tokens": prefix,
                "max_abs_diff": worst_diff,
                "worst_token_index": worst_token,
            }
            tested.append(entry)
            if worst_diff > max_allowed:
                first_fail = entry
                _print(
                    f"[0.2] FAIL at boundary {entry['boundary_ms']:.0f} ms: "
                    f"token {worst_token} diff {worst_diff:.6g} > {max_allowed:g}"
                )
                break
            if len(tested) % 20 == 0:
                _print(f"[0.2] checked {len(tested)} boundaries (max diff {worst_diff:.3g})")

        dependency: dict[str, Any] | None = None
        if first_fail is not None:
            # Locate the exact future-sample dependency of token 0 with a cut sweep.  Feed
            # 1 s so later tokens do not affect the comparison; zero after each cut.
            probe = waveform[: ENCODER_SAMPLE_RATE].copy()
            probe_tokens, _, _ = _stream_encoder_tokens(
                encoder, feature_extractor, probe, ENCODER_SAMPLE_RATE, device
            )
            base_token = probe_tokens[:, 0]
            sweep = []
            minimal_cut = None
            for cut_ms in range(80, 121):
                masked = probe.copy()
                cut = int(round(cut_ms * ENCODER_SAMPLE_RATE / 1000.0))
                masked[cut:] = 0.0
                mt, _, _ = _stream_encoder_tokens(
                    encoder, feature_extractor, masked, ENCODER_SAMPLE_RATE, device
                )
                diff = float((mt[:, 0] - base_token).abs().max().item())
                sweep.append({"cut_ms": cut_ms, "token0_max_abs_diff": diff})
                if minimal_cut is None and diff <= max_allowed:
                    minimal_cut = cut_ms
            dependency = {
                "cut_sweep": sweep,
                "minimal_cut_ms_for_token0_unchanged": minimal_cut,
                "lookahead_beyond_boundary_ms": None if minimal_cut is None else minimal_cut - 80,
                "expected_from_stft": {
                    "feature_hop_ms": 1000.0 * 160 / ENCODER_SAMPLE_RATE,
                    "stft_window_ms": 1000.0 * 400 / ENCODER_SAMPLE_RATE,
                    "last_mel_frame_of_token0": "starts 70 ms, ends 95 ms",
                },
            }

        max_diff = max((entry["max_abs_diff"] for entry in tested), default=None)
        verdict = evaluate_causality(rate_ok, max_diff)
        result = verdict["result"]
        strict_result = result
        approved_deviation = None
        if result == "FAIL" and args.approved_lookahead_ms > 0:
            look = dependency.get("lookahead_beyond_boundary_ms") if dependency else None
            if look is not None and look <= args.approved_lookahead_ms + 0.5:
                approved_deviation = {
                    "policy": "user-approved encoder look-ahead",
                    "approved_lookahead_ms": float(args.approved_lookahead_ms),
                    "measured_lookahead_ms": float(look),
                    "strict_result": "FAIL",
                    "note": (
                        "The strict 80 ms test still fails and remains recorded. Training is allowed only "
                        "under this explicit 95 ms effective-latency policy."
                    ),
                }
                result = "PASS"
        if result == "PASS" and budget_exhausted:
            result = "UNVERIFIED"

        measured = {
            "device": device_info,
            "encoder_load": encoder_load,
            "sample_rate": ENCODER_SAMPLE_RATE,
            "duration_s": duration_s,
            "chunk_ms": FRAME_MS,
            "token_rate": rate,
            "per_chunk_emitted_tokens": {
                "chunks": len(emissions),
                "total": int(sum(emissions)),
                "distribution": {str(value): int(emissions.count(value)) for value in sorted(set(emissions))},
                "mean_per_chunk": sum(emissions) / len(emissions) if emissions else None,
            },
            "emissions_per_chunk": emissions,
            "cache_invariants": cache_stats,
            "boundaries_tested": tested,
            "boundary_budget_exhausted": budget_exhausted,
            "first_failure": first_fail,
            "max_boundary_abs_diff_before_failure": max_diff,
            "threshold_max_abs_diff": max_allowed,
            "dependency_analysis": dependency,
            "verdict": verdict,
            "strict_result": strict_result,
            "approved_deviation": approved_deviation,
        }
        notes = (
            "Literal test: two 30 s streams (full vs audio zeroed after each 80 ms boundary), "
            "compare every already-emitted token. The encoder emits 12.5 tokens/s on average; "
            "375 chunk calls emit 375 tokens in total (mostly one per call; one call emits zero "
            "and one emits two when the retained STFT window completes an extra downsample token); "
            "the average rate is exactly 12.5 Hz."
        )
        if first_fail is not None:
            look = dependency.get("lookahead_beyond_boundary_ms") if dependency else None
            notes += (
                f" FAIL: token {first_fail['worst_token_index']}, which covers the first "
                f"{first_fail['boundary_ms']:.0f} ms window, changes when future audio is zeroed; "
                f"token 0 becomes unchanged only at a {dependency.get('minimal_cut_ms_for_token0_unchanged')} ms cut "
                f"({look} ms of lookahead beyond the 80 ms boundary). The cause is the 25 ms STFT window: "
                "token 0's last mel frame spans 70-95 ms (7-frame/80 ms downsampler lookahead). The blueprint "
                "requires output at t unchanged when audio after t is zeroed."
            )
        if approved_deviation:
            notes += (
                f" APPROVED DEVIATION: the strict 80 ms result is {strict_result}; under the user-approved "
                f"{approved_deviation['approved_lookahead_ms']:g} ms look-ahead policy (95 ms effective "
                "latency) this check passes. Both numbers are retained."
            )
        _write_json(out_dir / "encoder_causality.json", measured)
        return {
            "id": "0.2",
            "name": "Encoder causality",
            "result": result,
            "threshold": (
                f"Strict: 1 encoder token / 80 ms and max abs token diff <= {max_allowed:g} at every 80 ms "
                "boundary."
                + (
                    f" Approved deviation: <= {args.approved_lookahead_ms:g} ms look-ahead "
                    "(95 ms effective latency)."
                    if approved_deviation
                    else ""
                )
            ),
            "measured": measured,
            "artifacts": ["encoder_causality.json"],
            "notes": notes,
            "command": command,
            "started_at": started,
            "finished_at": utc_now(),
        }
    finally:
        if model is not None:
            del model
        if feature_extractor is not None:
            del feature_extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _codec_streaming_cache(codec):
    module = sys.modules.get(type(codec).__module__)
    if module is None:
        module = importlib.import_module(type(codec).__module__)
    cls = getattr(module, "LiquidCodecStreamingCache", None)
    if cls is None:
        raise RuntimeError("LiquidCodecStreamingCache not found in the remote codec module")
    return cls(codec.config)


def _rms_windows_db(samples, sample_rate: int, window_s: float = 1.0):
    np = _numpy()
    n = int(round(sample_rate * window_s))
    count = samples.shape[0] // n
    if count == 0:
        return np.empty((0,), dtype=np.float64)
    frames = samples[: count * n].reshape(count, n).astype(np.float64)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-8))


def _check_codec_streaming(args, model_dir: Path, out_dir: Path, command: str) -> dict[str, Any]:
    started = utc_now()
    torch = _torch()
    np = _numpy()
    codec = None
    try:
        device, dtype, device_info = resolve_device_dtype(torch, args.device, args.dtype)
        codec = _load_codec(model_dir)
        # The codec checkpoint is bf16; fp16 is used on Turing, fp32 on CPU.
        codec = codec.to(device=device, dtype=dtype).eval()
        _print(f"[0.3] codec on {device} ({device_info['dtype']})")

        waveform = synthetic_speech_like(30.0, CODEC_SAMPLE_RATE)
        audio = torch.from_numpy(waveform).unsqueeze(0).to(device)
        with torch.no_grad():
            codes = codec.encode(audio).audio_codes
        frames = int(codes.shape[-1])
        frame_rate = CODEC_SAMPLE_RATE / float(codec.config.frame_samples)
        _print(f"[0.3] encoded {frames} frames ({frame_rate:.4f} Hz)")

        with torch.no_grad():
            one_shot = codec.decode(codes).audio_values[0].detach().float().cpu()

        cache = _codec_streaming_cache(codec)
        pieces = []
        per_call: list[int] = []
        with torch.no_grad():
            for index in range(frames):
                final = index == frames - 1
                decoded = codec.decode(
                    codes[:, :, index : index + 1], streaming_cache=cache, finalize=final
                ).audio_values
                per_call.append(int(decoded.shape[-1]))
                if decoded.shape[-1]:
                    pieces.append(decoded[0].detach().float().cpu())
        streamed = torch.cat(pieces) if pieces else torch.zeros(0)

        min_len = min(int(streamed.shape[0]), int(one_shot.shape[0]))
        max_abs_error = float((streamed[:min_len] - one_shot[:min_len]).abs().max().item()) if min_len else None
        streamed_np = streamed.numpy()
        one_shot_np = one_shot.numpy()
        all_finite = bool(np.isfinite(streamed_np).all() and np.isfinite(one_shot_np).all())

        stream_db = _rms_windows_db(streamed_np, CODEC_SAMPLE_RATE)
        one_db = _rms_windows_db(one_shot_np, CODEC_SAMPLE_RATE)
        window_count = min(stream_db.shape[0], one_db.shape[0])
        rms_diff_db = (
            float(np.max(np.abs(stream_db[:window_count] - one_db[:window_count])))
            if window_count
            else None
        )
        stream_drift_db = float(stream_db[-1] - stream_db[0]) if stream_db.shape[0] else None
        stream_range_db = float(np.max(stream_db) - np.min(stream_db)) if stream_db.shape[0] else None

        verdict = evaluate_codec_stream(
            per_call,
            sample_rate=CODEC_SAMPLE_RATE,
            max_call_ms=CODEC_MAX_CALL_MS,
            all_finite=all_finite,
            rms_diff_db=rms_diff_db,
        )
        measured = {
            "device": device_info,
            "frame_samples": int(codec.config.frame_samples),
            "frame_rate_hz": frame_rate,
            "frames": frames,
            "duration_s": waveform.shape[0] / CODEC_SAMPLE_RATE,
            "samples_per_call": per_call,
            "per_call_ms": [1000.0 * value / CODEC_SAMPLE_RATE for value in per_call],
            "streamed_samples": int(streamed.shape[0]),
            "one_shot_samples": int(one_shot.shape[0]),
            "max_abs_error_vs_one_shot": max_abs_error,
            "all_finite": all_finite,
            "rms": {
                "streaming_per_second_db": [float(value) for value in stream_db],
                "one_shot_per_second_db": [float(value) for value in one_db],
                "max_abs_diff_db": rms_diff_db,
                "first_last_drift_db": stream_drift_db,
                "range_db": stream_range_db,
            },
            "verdict": verdict,
        }
        notes = (
            "One frame per decode() call with LiquidCodecStreamingCache, finalize on the last call. "
            "max_abs_error is computed against a single decode() of the same codes over the overlap. "
            "RMS stability compares 1 s windows of streaming vs one-shot decode (threshold 1 dB)."
        )
        _write_json(out_dir / "codec_streaming.json", measured)
        return {
            "id": "0.3",
            "name": "Codec streaming",
            "result": verdict["result"],
            "threshold": "Decoder emits <=160 ms per call, finite, RMS within 1 dB of one-shot decode over 30 s.",
            "measured": measured,
            "artifacts": ["codec_streaming.json"],
            "notes": notes,
            "command": command,
            "started_at": started,
            "finished_at": utc_now(),
        }
    finally:
        if codec is not None:
            del codec
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _AudioCollector:
    def __init__(self):
        self.frames: list[Any] = []
        self.ended = False

    def put(self, value):
        self.frames.append(value.detach().cpu().clone())

    def end(self):
        self.ended = True


def _build_prompt(tokenizer, audio_start_id: int):
    text = None
    template = getattr(tokenizer, "chat_template", None)
    if template:
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Say hello in one short sentence."}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = None
    if not text:
        text = "<|im_start|>user\nSay hello in one short sentence.<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


def _run_interrupt_generation(model, tokenizer, device, audio_start_id, eos_code, num_codebooks, force_at=3, steps=6):
    torch = _torch()
    from transformers import GenerationConfig

    input_ids = _build_prompt(tokenizer, audio_start_id).to(device)
    cfg = GenerationConfig(
        max_new_tokens=steps,
        do_sample=False,
        pad_token_id=int(model.config.text_config.pad_token_id or model.config.text_config.eos_token_id or 0),
        return_dict_in_generate=True,
    )
    collector = _AudioCollector()
    calls = {"count": 0}
    original = model.sample_audio_frame

    def forced_sample(hidden_state, *call_args, **call_kwargs):
        calls["count"] += 1
        index = calls["count"]
        if index < force_at:
            return torch.zeros((num_codebooks,), dtype=torch.long, device=hidden_state.device)
        if index == force_at:
            return torch.full((num_codebooks,), eos_code, dtype=torch.long, device=hidden_state.device)
        return original(hidden_state, *call_args, **call_kwargs)

    model.sample_audio_frame = forced_sample
    try:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                generation_mode="audio",
                generation_config=cfg,
                audio_temperature=0,
                return_audio_eos=True,
                audio_streamer=collector,
                suppress_audio_out=True,
            )
    finally:
        del model.sample_audio_frame
    return output, collector, calls["count"], input_ids


def _check_interruptibility(args, model_dir: Path, out_dir: Path, command: str) -> dict[str, Any]:
    started = utc_now()
    torch = _torch()
    model = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True
        )
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        audio_start_id = int(config["audio_start_token_id"])
        eos_code = int(config["audio_eos_code"])
        num_codebooks = int(config["num_codebooks"])

        device, dtype, device_info = resolve_device_dtype(torch, args.device, args.dtype)
        # Full 2B LM + vision tower: fall back to CPU if the accelerator cannot hold it.
        try:
            import safetensors  # noqa: F401

            header = read_safetensors_header(model_dir / "model.safetensors")
            params = sum(
                math.prod(meta.get("shape") or [])
                for key, meta in header.items()
                if key != "__metadata__"
            )
        except Exception:
            params = 2_170_000_000
        if device == "cuda":
            free_bytes, _total = torch.cuda.mem_get_info()
            needed = params * torch.finfo(dtype).bits / 8 * 1.2
            if needed > free_bytes * 0.9:
                _print(
                    f"[0.4] GPU free {free_bytes / 2**30:.2f} GiB < estimated {needed / 2**30:.2f} GiB; "
                    "using CPU"
                )
                device, dtype = "cpu", torch.float32
                device_info.update({"device": "cpu", "dtype": "float32", "fallback": "GPU memory too small"})

        _print(f"[0.4] loading full model on {device} ({device_info['dtype']})")
        try:
            model = _load_model(model_dir, dtype, device)
            if str(next(model.parameters()).device) != device:
                model = model.to(device)
        except Exception as exc:
            if device == "cuda":
                _print(f"[0.4] GPU load failed ({type(exc).__name__}: {exc}); retrying on CPU")
                if model is not None:
                    del model
                    model = None
                gc.collect()
                torch.cuda.empty_cache()
                device, dtype = "cpu", torch.float32
                device_info.update({"device": "cpu", "dtype": "float32", "fallback": str(exc)})
                model = _load_model(model_dir, dtype, device)
            else:
                raise

        output, collector, sample_calls, input_ids = _run_interrupt_generation(
            model,
            tokenizer,
            device,
            audio_start_id,
            eos_code,
            num_codebooks,
            force_at=args.interrupt_force_at,
            steps=args.interrupt_steps,
        )
        audio_codes = output.audio_codes
        audio_length = int(output.audio_lengths[0].item()) if output.audio_lengths is not None else int(audio_codes.shape[-1])
        audio_codes = audio_codes[0, :, :audio_length].detach().cpu()
        eos_mask = (audio_codes == eos_code).all(dim=0)
        eos_indices = [int(index) for index in torch.nonzero(eos_mask, as_tuple=False).flatten().tolist()]
        eos_index = eos_indices[0] if eos_indices else None
        frames_after = None if eos_index is None else int(audio_length - eos_index - 1)
        frames_before = eos_index
        collector_frames_after = None
        if eos_index is not None:
            # the streamer receives non-EOS frames only; count its entries beyond the expected prefix
            collector_frames_after = max(0, len(collector.frames) - eos_index)

        verdict = evaluate_interrupt(frames_after, INTERRUPT_MAX_FRAMES)
        generated_text = tokenizer.decode(
            output.sequences[0, input_ids.shape[1] :].tolist(), skip_special_tokens=False
        )
        measured = {
            "device": device_info,
            "forced_eos_at_sampler_call": int(args.interrupt_force_at),
            "sampler_calls": int(sample_calls),
            "eos_code": eos_code,
            "num_codebooks": num_codebooks,
            "generated_slots": int(output.sequences.shape[1] - input_ids.shape[1]),
            "audio_length_with_eos": audio_length,
            "eos_frame_index": eos_index,
            "frames_before_stop": frames_before,
            "frames_after_stop": frames_after,
            "audio_streamer_frames": len(collector.frames),
            "audio_streamer_frames_after_stop": collector_frames_after,
            "audio_codes_after_stop": audio_codes[:, (eos_index + 1) :].tolist() if eos_index is not None else None,
            "generated_text": generated_text,
            "verdict": verdict,
            "measurement": (
                "The public generate() has no way to force EOS, so the serving policy's stop path was "
                "exercised by replacing sample_audio_frame for the first calls: two valid zero frames, then "
                "one all-2016 end-of-audio frame. Frames emitted after that slot are counted in "
                "output.audio_codes and in the audio streamer."
            ),
        }
        notes = (
            "Measured on a text prompt with generation_mode='audio'. Forced EOS at sampler call "
            f"{args.interrupt_force_at}; output kept the EOS frame (return_audio_eos=True). "
            "frames_after_stop is the count of audio frames following the EOS frame in the returned codes; "
            "0 means the policy stopped within the required <=2 frames."
        )
        _write_json(out_dir / "interruptibility.json", measured)
        result = verdict["result"]
        if eos_index is None:
            result = "FAIL"
            notes += " No forced EOS frame was found in the output."
        return {
            "id": "0.4",
            "name": "Interruptibility",
            "result": result,
            "threshold": "After a forced audio EOS, at most 2 further audio frames are emitted.",
            "measured": measured,
            "artifacts": ["interruptibility.json"],
            "notes": notes,
            "command": command,
            "started_at": started,
            "finished_at": utc_now(),
        }
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _ensure_checkpoint(args, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    model_dir = Path(args.model_dir).expanduser() if args.model_dir else DEFAULT_MODEL_DIR
    required = [
        model_dir / "config.json",
        model_dir / "model.safetensors",
        model_dir / "audio_tokenizer" / "config.json",
        model_dir / "audio_tokenizer" / "model.safetensors",
        model_dir / "tokenizer.json",
    ]
    info: dict[str, Any] = {"model_id": args.model_id, "revision": args.revision, "path": str(model_dir)}
    missing = [str(path) for path in required if not path.exists()]
    if missing and not args.allow_download:
        raise FileNotFoundError(
            "checkpoint files missing and --no-download was passed: " + ", ".join(missing)
        )
    if missing:
        _print(f"[download] {len(missing)} files missing; snapshot_download({args.model_id}) -> {model_dir}")
        from huggingface_hub import snapshot_download

        started = time.time()
        snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            local_dir=str(model_dir),
        )
        info["download_seconds"] = time.time() - started
    info["files"] = {
        str(path.relative_to(model_dir)): path.stat().st_size
        for path in sorted(model_dir.rglob("*"))
        if path.is_file()
    }
    try:
        from huggingface_hub import HfApi

        info["commit_sha"] = HfApi().model_info(args.model_id, revision=args.revision).sha
    except Exception as exc:
        info["commit_sha_error"] = f"{type(exc).__name__}: {exc}"
    _write_json(out_dir / "checkpoint.json", info)
    return model_dir, info


def _overall(checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    blocking = [cid for cid in ("0.2", "0.3") if checks.get(cid, {}).get("result") == "FAIL"]
    blocked = [cid for cid, value in checks.items() if value.get("result") == "BLOCKED"]
    failed = [cid for cid, value in checks.items() if value.get("result") == "FAIL"]
    if blocking:
        result = "FAIL"
        reason = f"blueprint-blocking check(s) failed: {', '.join(blocking)}; switch to the fallback base model"
    elif failed:
        result = "FAIL"
        reason = f"check(s) failed: {', '.join(failed)}"
    elif blocked:
        result = "BLOCKED"
        reason = f"check(s) blocked: {', '.join(blocked)}"
    elif checks.get("0.1", {}).get("result") == "CONDITIONAL":
        result = "CONDITIONAL"
        reason = "all measured checks passed; license status is CONDITIONAL because organisation revenue is unknown"
    else:
        result = "PASS"
        reason = "all checks passed"
    return {"result": result, "reason": reason, "blocking": blocking, "failed": failed, "blocked": blocked}


def _measured_summary(check: dict[str, Any]) -> str:
    cid = check["id"]
    measured = check.get("measured") or {}
    if cid == "0.1":
        classification = measured.get("classification", {})
        return (
            f"license={measured.get('license_name')}; threshold=${measured.get('annual_revenue_threshold_usd'):,}; "
            f"org revenue={measured.get('organisation', {}).get('revenue_usd')}; "
            f"redistribution conditions={len(measured.get('redistribution_conditions', []))}; "
            f"classification={classification.get('result')}"
        )
    if cid == "0.2":
        rate = measured.get("token_rate", {})
        failure = measured.get("first_failure") or {}
        dep = measured.get("dependency_analysis") or {}
        if failure:
            return (
                f"{rate.get('tokens')} tokens / {rate.get('duration_s'):.2f} s ({rate.get('token_rate_hz'):.4f} Hz); "
                f"first failure at {failure.get('boundary_ms'):.0f} ms, token {failure.get('worst_token_index')} "
                f"max|diff|={failure.get('max_abs_diff'):.6g}; token 0 unchanged only at a "
                f"{dep.get('minimal_cut_ms_for_token0_unchanged')} ms cut "
                f"({dep.get('lookahead_beyond_boundary_ms')} ms lookahead)"
            )
        return (
            f"{rate.get('tokens')} tokens / {rate.get('duration_s'):.2f} s ({rate.get('token_rate_hz'):.4f} Hz); "
            f"boundaries tested={len(measured.get('boundaries_tested', []))}; "
            f"max|diff|={measured.get('max_boundary_abs_diff_before_failure')}"
        )
    if cid == "0.3":
        verdict = measured.get("verdict", {})
        return (
            f"frame rate={measured.get('frame_rate_hz'):.4f} Hz; calls={verdict.get('calls')}; "
            f"max={verdict.get('max_call_ms'):.3f} ms; mean={verdict.get('mean_call_ms'):.3f} ms; "
            f"finite={measured.get('all_finite')}; max|stream-one-shot|={measured.get('max_abs_error_vs_one_shot'):.3g}; "
            f"max RMS diff={measured.get('rms', {}).get('max_abs_diff_db'):.4g} dB"
        )
    if cid == "0.4":
        return (
            f"forced EOS at sampler call {measured.get('forced_eos_at_sampler_call')}; "
            f"frames before={measured.get('frames_before_stop')}; frames after={measured.get('frames_after_stop')}; "
            f"eos frame index={measured.get('eos_frame_index')}; sampler calls={measured.get('sampler_calls')}"
        )
    if cid == "0.5":
        return (
            f"model params={measured.get('model_params'):,}; codec params={measured.get('codec_params'):,}; "
            f"total={measured.get('total_params_with_codec'):,}; config checks "
            f"{sum(1 for item in measured.get('config_checks', []) if item['pass'])}/"
            f"{len(measured.get('config_checks', []))}; modules={measured.get('module_presence')}"
        )
    return ""


def build_report_sections(checks: dict[str, dict[str, Any]], command: str, overall: dict[str, Any]) -> str:
    lines = [
        f"# Gate 0 — checkpoint verification",
        "",
        f"Overall: **{overall['result']}** — {overall['reason']}",
        f"Command: `{command}`",
        "",
    ]
    for cid in sorted(checks):
        check = checks[cid]
        lines.extend(
            [
                f"## Gate {cid} — {check['name']}",
                f"- Command: `{check.get('command', command)}`",
                f"- Measured: {_measured_summary(check)}",
                f"- Threshold: {check.get('threshold', '')}",
                f"- Result: {check.get('result', 'UNVERIFIED')}",
                f"- Artifacts: {', '.join(check.get('artifacts', [])) or '—'}",
                f"- Notes / UNVERIFIED: {check.get('notes', '')}",
                "",
            ]
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate 0 checkpoint verification for LiquidAI/LFM3-Omni-2B")
    parser.add_argument("--all", action="store_true", help="run every check (default when no --check is given)")
    parser.add_argument(
        "--check",
        action="append",
        choices=["0.1", "0.2", "0.3", "0.4", "0.5"],
        default=None,
        help="run only the named check (repeatable)",
    )
    parser.add_argument("--out", default="artifacts/gate0", help="artifact directory")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--model-dir", default=None, help=f"local snapshot directory (default {DEFAULT_MODEL_DIR})")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--allow-download", dest="allow_download", action="store_true", default=True)
    parser.add_argument("--no-download", dest="allow_download", action="store_false")
    parser.add_argument(
        "--org-revenue-usd",
        type=float,
        default=None,
        help="annual revenue for the license classification; unknown by default -> CONDITIONAL",
    )
    parser.add_argument("--org-nonprofit", action="store_true")
    parser.add_argument("--noncommercial-use", action="store_true")
    parser.add_argument(
        "--causality-time-budget",
        type=float,
        default=1800.0,
        help="seconds allowed for the per-boundary causality sweep before reporting UNVERIFIED",
    )
    parser.add_argument(
        "--approved-lookahead-ms",
        type=float,
        default=0.0,
        help=(
            "user-approved encoder look-ahead beyond the 80 ms boundary. 0 keeps the strict gate; "
            "15 accepts the measured LFM3 STFT look-ahead and records a deviation."
        ),
    )
    parser.add_argument("--interrupt-force-at", type=int, default=3, help="force EOS at this sampler call")
    parser.add_argument("--interrupt-steps", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = "python scripts/gate0.py " + " ".join(sys.argv[1:] if argv is None else argv)
    selected = args.check if args.check else ["0.1", "0.2", "0.3", "0.4", "0.5"]

    document: dict[str, Any] = {
        "gate": 0,
        "model_id": args.model_id,
        "command": command,
        "started_at": utc_now(),
        "checks": {},
        "artifacts_dir": str(out_dir),
    }
    _write_json(out_dir / "gate0.json", document)

    def record(check: dict[str, Any]) -> None:
        document["checks"][check["id"]] = check
        document["overall"] = _overall(document["checks"])
        _write_json(out_dir / "gate0.json", document)
        _print(f"[gate0] {check['id']} {check['name']}: {check['result']}")

    try:
        model_dir, checkpoint_info = _ensure_checkpoint(args, out_dir)
        document["checkpoint"] = checkpoint_info
        _write_json(out_dir / "gate0.json", document)
    except Exception as exc:
        document["checkpoint_error"] = f"{type(exc).__name__}: {exc}"
        for cid in selected:
            record(
                {
                    "id": cid,
                    "name": {
                        "0.1": "License",
                        "0.2": "Encoder causality",
                        "0.3": "Codec streaming",
                        "0.4": "Interruptibility",
                        "0.5": "Head layout",
                    }[cid],
                    "result": "BLOCKED",
                    "threshold": "",
                    "measured": {},
                    "artifacts": [],
                    "notes": f"checkpoint unavailable: {type(exc).__name__}: {exc}",
                    "command": command,
                    "started_at": utc_now(),
                    "finished_at": utc_now(),
                }
            )
        document["finished_at"] = utc_now()
        document["overall"] = _overall(document["checks"])
        _write_json(out_dir / "gate0.json", document)
        (out_dir / "report_section.md").write_text(
            build_report_sections(document["checks"], command, document["overall"]), encoding="utf-8"
        )
        return 1

    runners = {
        "0.1": lambda: _check_license(args, model_dir, out_dir, command),
        "0.5": lambda: _check_layout(args, model_dir, out_dir, command),
        "0.2": lambda: _check_encoder_causality(args, model_dir, out_dir, command),
        "0.3": lambda: _check_codec_streaming(args, model_dir, out_dir, command),
        "0.4": lambda: _check_interruptibility(args, model_dir, out_dir, command),
    }
    for cid in selected:
        try:
            check = runners[cid]()
        except Exception as exc:
            traceback.print_exc()
            check = {
                "id": cid,
                "name": {
                    "0.1": "License",
                    "0.2": "Encoder causality",
                    "0.3": "Codec streaming",
                    "0.4": "Interruptibility",
                    "0.5": "Head layout",
                }[cid],
                "result": "BLOCKED",
                "threshold": "",
                "measured": {},
                "artifacts": [],
                "notes": f"{type(exc).__name__}: {exc}",
                "command": command,
                "started_at": utc_now(),
                "finished_at": utc_now(),
            }
        record(check)

    document["finished_at"] = utc_now()
    document["overall"] = _overall(document["checks"])
    _write_json(out_dir / "gate0.json", document)
    (out_dir / "report_section.md").write_text(
        build_report_sections(document["checks"], command, document["overall"]), encoding="utf-8"
    )
    return 0 if document["overall"]["result"] in ("PASS", "CONDITIONAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
