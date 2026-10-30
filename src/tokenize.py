"""Manifest -> 12.5 Hz binary token shards (blueprint phase 2).

Each manifest record names a dual-channel conversation.  Both channels are
read at 24 kHz, trimmed or zero-padded to exactly ``T = frame_count(duration)``
80 ms frames, encoded with the checkpoint's LiquidCodec, and packed into one
binary record::

    header   magic b"RPT1" | version u16 | n_codebooks u16 | T u32   (12 bytes)
    user     uint16 little-endian [T, n_codebooks]
    agent    uint16 little-endian [T, n_codebooks]
    text     int32 little-endian  [T]
    control  uint8                [T]

The solid record size is ``12 + T * (2*2*n_codebooks + 4 + 1)`` bytes.  With
12 codebooks that is 53 bytes per frame, about 2.38 MB per audio hour; 9,300 h
therefore lands near 22 GB, inside the blueprint's 15-30 GB target.  That
target is a projection, not a measured claim.

Shards are ``tokens_XXXXX.bin`` files capped by *conversation audio hours*
(``--shard-hours``); a single longer conversation still gets its own shard.
``index.json`` is the single source of truth and is rewritten atomically.  The
writer is resumable: conversations already present in the index are skipped,
and an unindexed tail left by a crash is truncated before appending.

Deviations and limitations
--------------------------
* The blueprint says ``user[T,8] | agent[T,8]``.  The published checkpoint's
  LiquidCodec has 12 codebooks (``config.n_q == 12``), matching
  ``src.schema.NUM_CODEBOOKS``; the on-disk layout stores 12.
* The binary header and ``index.json`` are this module's invention; the
  blueprint only sketched the array order.
* Text targets are the schema's deliberately coarse, turn-level alignment
  (tokens spread uniformly over an agent turn), not forced alignment.
* Codec encoding is whole-file (the checkpoint exposes no streaming encoder).
  Peak memory therefore scales with conversation length.
* A deterministic silence frame (the codec's encoding of 1920 zero samples) is
  stored in ``index.json`` so the dataset can pad without loading a codec.
* Streaming verification decodes chunks of at most 2 frames (160 ms).  The
  codec's streaming cache, not this module, defines boundary behaviour.
* ASR WER and UTMOS are optional and report ``UNVERIFIED`` when the model or
  package is unavailable.  Numbers are never synthesized.
* The index writer assumes a single writer per output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from .audio import fit_frames, read_audio, resample, write_audio
from .schema import (
    CODEBOOK_SIZE,
    CODEC_SAMPLE_RATE,
    CONTROL_LABELS,
    FRAME_RATE,
    FRAME_SAMPLES,
    NUM_CODEBOOKS,
    Conversation,
    align_text_to_frames,
    control_histogram,
    derive_control_labels,
    frame_count,
    read_manifest,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# On-disk format
# ---------------------------------------------------------------------------

MAGIC = b"RPT1"
FORMAT_VERSION = 1
INDEX_FORMAT = "repartee-token-shards"
INDEX_VERSION = 1
HEADER_STRUCT = struct.Struct("<4sHHI")
HEADER_SIZE = HEADER_STRUCT.size  # 12

DEFAULT_CODEC_MODEL = "LiquidAI/LFM3-Omni-2B"
DEFAULT_SHARD_HOURS = 50.0
DEFAULT_FLUSH_EVERY = 100
MAX_CHUNK_SECONDS = 0.160
WER_THRESHOLD = 0.05
UTMOS_DELTA_THRESHOLD = 0.1
STREAMING_TOLERANCE = 1e-3

_INDEX_NAME = "index.json"
_SHARD_GLOB = "tokens_*.bin"

_UINT16 = np.dtype("<u2")
_INT32 = np.dtype("<i4")
_UINT8 = np.dtype("|u1")


# ---------------------------------------------------------------------------
# Record layout helpers (shared with src.dataset)
# ---------------------------------------------------------------------------


def record_offsets(frames: int, n_codebooks: int = NUM_CODEBOOKS) -> dict[str, int]:
    """Byte offsets inside one record.  ``total`` is also the record size."""
    if frames < 0:
        raise ValueError(f"frames must be >= 0, got {frames}")
    if not 1 <= n_codebooks <= 0xFFFF:
        raise ValueError(f"n_codebooks must be in [1, 65535], got {n_codebooks}")
    channel_bytes = frames * n_codebooks * _UINT16.itemsize
    user = HEADER_SIZE
    agent = user + channel_bytes
    text = agent + channel_bytes
    control = text + frames * _INT32.itemsize
    total = control + frames * _UINT8.itemsize
    return {
        "user": user,
        "agent": agent,
        "text": text,
        "control": control,
        "total": total,
    }


def record_bytes(frames: int, n_codebooks: int = NUM_CODEBOOKS) -> int:
    """Size of one binary record in bytes."""
    return record_offsets(frames, n_codebooks)["total"]


def encode_record(
    user_codes: np.ndarray,
    agent_codes: np.ndarray,
    text: np.ndarray,
    control: np.ndarray,
) -> bytes:
    """Validate and serialize one conversation record."""
    user = np.asarray(user_codes)
    agent = np.asarray(agent_codes)
    text_array = np.asarray(text).reshape(-1)
    control_array = np.asarray(control).reshape(-1)

    if user.ndim != 2 or agent.ndim != 2:
        raise ValueError("user_codes and agent_codes must be 2-D [T, n_codebooks]")
    if user.shape != agent.shape:
        raise ValueError(f"channel shapes differ: {user.shape} vs {agent.shape}")
    frames, n_codebooks = user.shape
    if frames > 0xFFFFFFFF:
        raise ValueError(f"too many frames for u32: {frames}")
    if n_codebooks < 1 or n_codebooks > 0xFFFF:
        raise ValueError(f"invalid codebook count {n_codebooks}")
    if text_array.shape != (frames,):
        raise ValueError(f"text must have shape ({frames},), got {text_array.shape}")
    if control_array.shape != (frames,):
        raise ValueError(f"control must have shape ({frames},), got {control_array.shape}")

    for name, codes in (("user", user), ("agent", agent)):
        if codes.size:
            if not np.issubdtype(codes.dtype, np.integer):
                raise TypeError(f"{name}_codes must be integer, got {codes.dtype}")
            low, high = int(codes.min()), int(codes.max())
            if low < 0 or high >= CODEBOOK_SIZE:
                raise ValueError(f"{name}_codes out of range [0, {CODEBOOK_SIZE}): {low}..{high}")
    if not np.issubdtype(text_array.dtype, np.integer):
        raise TypeError(f"text must be integer, got {text_array.dtype}")
    if text_array.size:
        low, high = int(text_array.min()), int(text_array.max())
        if low < 0 or high > np.iinfo(np.int32).max:
            raise ValueError(f"text ids out of int32 range: {low}..{high}")
    if not np.issubdtype(control_array.dtype, np.integer):
        raise TypeError(f"control must be integer, got {control_array.dtype}")
    if control_array.size and int(control_array.max()) >= len(CONTROL_LABELS):
        raise ValueError(f"control ids must be < {len(CONTROL_LABELS)}")

    header = HEADER_STRUCT.pack(MAGIC, FORMAT_VERSION, n_codebooks, frames)
    return b"".join(
        (
            header,
            np.ascontiguousarray(user, dtype=_UINT16).tobytes(),
            np.ascontiguousarray(agent, dtype=_UINT16).tobytes(),
            np.ascontiguousarray(text_array, dtype=_INT32).tobytes(),
            np.ascontiguousarray(control_array, dtype=_UINT8).tobytes(),
        )
    )


def decode_record(
    data: bytes | bytearray | memoryview,
    *,
    n_codebooks: int | None = None,
    frames: int | None = None,
) -> dict[str, Any]:
    """Parse a record from the front of ``data``.

    ``n_codebooks``/``frames``, when given, must match the header.  Trailing
    bytes beyond the record are ignored; :func:`read_record` passes exactly one
    record.
    """
    if len(data) < HEADER_SIZE:
        raise ValueError(f"record shorter than {HEADER_SIZE}-byte header: {len(data)}")
    magic, version, stored_codebooks, stored_frames = HEADER_STRUCT.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"bad record magic {magic!r}, expected {MAGIC!r}")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported record version {version}, expected {FORMAT_VERSION}")
    if n_codebooks is not None and stored_codebooks != n_codebooks:
        raise ValueError(f"record has {stored_codebooks} codebooks, index says {n_codebooks}")
    if frames is not None and stored_frames != frames:
        raise ValueError(f"record has {stored_frames} frames, index says {frames}")
    offsets = record_offsets(stored_frames, stored_codebooks)
    if len(data) < offsets["total"]:
        raise ValueError(f"truncated record: {len(data)} < {offsets['total']} bytes")

    user = np.frombuffer(data, dtype=_UINT16, count=stored_frames * stored_codebooks, offset=offsets["user"])
    user = user.reshape(stored_frames, stored_codebooks)
    agent = np.frombuffer(data, dtype=_UINT16, count=stored_frames * stored_codebooks, offset=offsets["agent"])
    agent = agent.reshape(stored_frames, stored_codebooks)
    text = np.frombuffer(data, dtype=_INT32, count=stored_frames, offset=offsets["text"])
    control = np.frombuffer(data, dtype=_UINT8, count=stored_frames, offset=offsets["control"])
    return {
        "magic": magic,
        "version": version,
        "n_codebooks": stored_codebooks,
        "frames": stored_frames,
        "user_codes": np.ascontiguousarray(user),
        "agent_codes": np.ascontiguousarray(agent),
        "text": np.ascontiguousarray(text),
        "control": np.ascontiguousarray(control),
    }


def read_record(
    path: str | Path,
    offset: int,
    *,
    n_codebooks: int | None = None,
    frames: int | None = None,
) -> dict[str, Any]:
    """Read and parse the record at ``offset`` in a shard file."""
    path = Path(path)
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    with path.open("rb") as handle:
        handle.seek(offset)
        header = handle.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            raise ValueError(f"{path}: truncated header at offset {offset}")
        magic, version, stored_codebooks, stored_frames = HEADER_STRUCT.unpack(header)
        if magic != MAGIC or version != FORMAT_VERSION:
            raise ValueError(f"{path}:{offset}: bad record header")
        if n_codebooks is not None and stored_codebooks != n_codebooks:
            raise ValueError(f"{path}:{offset}: {stored_codebooks} codebooks != {n_codebooks}")
        if frames is not None and stored_frames != frames:
            raise ValueError(f"{path}:{offset}: {stored_frames} frames != {frames}")
        size = record_bytes(stored_frames, stored_codebooks)
        body = handle.read(size - HEADER_SIZE)
        if len(body) != size - HEADER_SIZE:
            raise ValueError(f"{path}:{offset}: truncated record")
    return decode_record(header + body, n_codebooks=stored_codebooks, frames=stored_frames)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class CodecBackend(Protocol):
    """Only the encoder is needed to build shards."""

    n_codebooks: int
    sample_rate: int

    def encode(self, audio: np.ndarray) -> np.ndarray:
        """Return integer codes shaped ``[T, n_codebooks]`` for 24 kHz audio."""


class DecodingCodecBackend(CodecBackend, Protocol):
    """Encoder plus one-shot and streaming decoders (used by ``verify``)."""

    def decode(
        self,
        codes: np.ndarray,
        streaming_cache: Any | None = None,
        finalize: bool = True,
    ) -> np.ndarray:
        """Return 24 kHz float32 mono audio for ``[T, n_codebooks]`` codes."""

    def new_streaming_cache(self) -> Any:
        """Fresh decoder state for consecutive :meth:`decode` chunks."""


class TextTokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Token ids for ``text``."""


class HFTokenizer:
    """``transformers`` tokenizer adapter used for the coarsely aligned text."""

    def __init__(self, model_name: str = DEFAULT_CODEC_MODEL, **kwargs: Any) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **kwargs)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [int(token) for token in self._tokenizer.encode(text, add_special_tokens=add_special_tokens)]


class LiquidCodecBackend:
    """LiquidCodec loaded from the checkpoint's bundled ``audio_tokenizer``.

    Weights are bf16 on CUDA and fp32 otherwise, as in the model card.  The
    codec is loaded through ``AutoProcessor`` so the remote-code class and the
    ``audio_tokenizer`` subfolder resolve exactly as they do for inference.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CODEC_MODEL,
        *,
        device: str | None = None,
        dtype: Any | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        import torch
        from transformers import AutoProcessor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        codec = getattr(processor, "audio_tokenizer", None)
        if codec is None:
            raise RuntimeError(
                "processor has no `audio_tokenizer`; expected the bundled LiquidCodec in the checkpoint"
            )
        self._processor = processor
        self._codec = codec.to(device=device, dtype=dtype).eval()
        self.model_name = model_name
        self.device = str(device)
        self.dtype = dtype
        self.sample_rate = CODEC_SAMPLE_RATE
        self.frame_samples = int(getattr(codec.config, "frame_size", FRAME_SAMPLES[CODEC_SAMPLE_RATE]))
        self.n_codebooks = int(getattr(codec.config, "n_q", NUM_CODEBOOKS))
        if self.n_codebooks <= 0:
            raise RuntimeError(f"invalid codec codebook count {self.n_codebooks}")

    def _to_tensor(self, array: np.ndarray, dtype: Any) -> Any:
        import torch

        contiguous = np.ascontiguousarray(array)
        if not contiguous.flags.writeable:
            contiguous = contiguous.copy()
        return torch.from_numpy(contiguous).to(device=self.device, dtype=dtype)

    def encode(self, audio: np.ndarray) -> np.ndarray:
        import torch

        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            raise ValueError("audio must be non-empty")
        if not np.isfinite(waveform).all():
            raise ValueError("audio contains non-finite samples")
        tensor = self._to_tensor(waveform, torch.float32)
        with torch.inference_mode():
            output = self._codec.encode(tensor[None], return_dict=True)
        codes = output.audio_codes[0].T.cpu().numpy().astype(np.int64)
        if codes.size and (codes.min() < 0 or codes.max() >= CODEBOOK_SIZE):
            raise ValueError("codec produced codes outside the known vocabulary")
        return codes

    def decode(
        self,
        codes: np.ndarray,
        streaming_cache: Any | None = None,
        finalize: bool = True,
    ) -> np.ndarray:
        import torch

        array = np.asarray(codes)
        if array.ndim != 2 or array.shape[1] != self.n_codebooks:
            raise ValueError(f"expected codes [T, {self.n_codebooks}], got {array.shape}")
        tensor = self._to_tensor(array.astype(np.int64, copy=False), torch.long)
        # Stored/served layout is [T, codebooks]; LiquidCodec wants [B, codebooks, T].
        codec_codes = tensor.transpose(0, 1).contiguous()[None]
        with torch.inference_mode():
            output = self._codec.decode(codec_codes, streaming_cache, finalize=finalize, return_dict=True)
        return output.audio_values[0].float().cpu().numpy()

    def new_streaming_cache(self) -> Any:
        import importlib
        import sys

        module_name = type(self._codec).__module__
        module = sys.modules.get(module_name)
        if module is None:
            module = importlib.import_module(module_name)
        cache_class = getattr(module, "LiquidCodecStreamingCache", None)
        if cache_class is None:
            raise RuntimeError("codec module does not expose LiquidCodecStreamingCache")
        return cache_class(self._codec.config)


# ---------------------------------------------------------------------------
# Index I/O
# ---------------------------------------------------------------------------


def _empty_index(
    n_codebooks: int = NUM_CODEBOOKS,
    codec_model: str | None = None,
    silence_frame: Sequence[int] | None = None,
) -> dict[str, Any]:
    return {
        "format": INDEX_FORMAT,
        "version": INDEX_VERSION,
        "codec_model": codec_model,
        "n_codebooks": int(n_codebooks),
        "frame_rate": FRAME_RATE,
        "frame_samples": FRAME_SAMPLES[CODEC_SAMPLE_RATE],
        "codebook_size": CODEBOOK_SIZE,
        "silence_frame": None if silence_frame is None else [int(value) for value in silence_frame],
        "shards": [],
        "conversations": [],
    }


def index_path(directory: str | Path) -> Path:
    return Path(directory) / _INDEX_NAME


def load_index(directory: str | Path) -> dict[str, Any]:
    """Load ``index.json`` and validate its envelope and entry types."""
    path = index_path(directory)
    if not path.is_file():
        raise FileNotFoundError(f"no token index at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: index must be an object")
    if data.get("format") != INDEX_FORMAT:
        raise ValueError(f"{path}: format must be {INDEX_FORMAT!r}, got {data.get('format')!r}")
    version = data.get("version")
    if not isinstance(version, int) or version != INDEX_VERSION:
        raise ValueError(f"{path}: unsupported index version {version!r}, expected {INDEX_VERSION}")
    n_codebooks = data.get("n_codebooks")
    if not isinstance(n_codebooks, int) or not 1 <= n_codebooks <= 0xFFFF:
        raise ValueError(f"{path}: invalid n_codebooks {n_codebooks!r}")
    if n_codebooks != NUM_CODEBOOKS:
        raise ValueError(f"{path}: {n_codebooks} codebooks, schema fixes {NUM_CODEBOOKS}")
    if float(data.get("frame_rate", 0.0)) != FRAME_RATE:
        raise ValueError(f"{path}: frame_rate must be {FRAME_RATE}, got {data.get('frame_rate')!r}")
    if int(data.get("frame_samples", 0)) != FRAME_SAMPLES[CODEC_SAMPLE_RATE]:
        raise ValueError(
            f"{path}: frame_samples must be {FRAME_SAMPLES[CODEC_SAMPLE_RATE]}, "
            f"got {data.get('frame_samples')!r}"
        )
    if int(data.get("codebook_size", 0)) != CODEBOOK_SIZE:
        raise ValueError(
            f"{path}: codebook_size must be {CODEBOOK_SIZE}, got {data.get('codebook_size')!r}"
        )
    shards = data.get("shards")
    conversations = data.get("conversations")
    if not isinstance(shards, list) or not isinstance(conversations, list):
        raise ValueError(f"{path}: shards and conversations must be lists")
    silence = data.get("silence_frame")
    if silence is not None:
        if not isinstance(silence, list) or len(silence) != n_codebooks:
            raise ValueError(f"{path}: silence_frame must be a {n_codebooks}-element list")
        if any(not isinstance(value, int) or not 0 <= value < CODEBOOK_SIZE for value in silence):
            raise ValueError(f"{path}: silence_frame codes out of range")
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError(f"{path}: shard entry must be an object")
        missing = {"file", "bytes", "records", "hours"} - set(shard)
        if missing:
            raise ValueError(f"{path}: shard entry missing {sorted(missing)}")
        if not isinstance(shard["file"], str) or Path(shard["file"]).name != shard["file"]:
            raise ValueError(f"{path}: shard file must be a bare name, got {shard['file']!r}")
        if shard["bytes"] < 0 or shard["records"] < 0 or shard["hours"] < 0:
            raise ValueError(f"{path}: negative shard metadata for {shard['file']!r}")
    seen: set[str] = set()
    for entry in conversations:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: conversation entry must be an object")
        missing = {"id", "shard", "offset", "bytes", "T", "duration_s", "source", "license", "split"} - set(entry)
        if missing:
            raise ValueError(f"{path}: conversation entry missing {sorted(missing)}")
        if entry["id"] in seen:
            raise ValueError(f"{path}: duplicate conversation id {entry['id']!r}")
        seen.add(entry["id"])
        if not isinstance(entry["offset"], int) or entry["offset"] < 0:
            raise ValueError(f"{path}: invalid offset for {entry['id']!r}")
        if not isinstance(entry["T"], int) or entry["T"] <= 0:
            raise ValueError(f"{path}: invalid frame count for {entry['id']!r}")
        if entry["bytes"] != record_bytes(entry["T"], n_codebooks):
            raise ValueError(f"{path}: byte size for {entry['id']!r} does not match its frame count")
        if entry["split"] not in ("train", "valid", "test"):
            raise ValueError(f"{path}: invalid split {entry['split']!r} for {entry['id']!r}")
    return data


def _write_index_atomic(directory: str | Path, data: dict[str, Any]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _INDEX_NAME
    fd, temporary = tempfile.mkstemp(dir=directory, prefix=".index-", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


# ---------------------------------------------------------------------------
# Audio -> arrays
# ---------------------------------------------------------------------------


def _encode_channel(
    backend: CodecBackend,
    audio: np.ndarray,
    frames: int,
    *,
    n_codebooks: int,
    channel: str,
) -> np.ndarray:
    codes = np.asarray(backend.encode(np.asarray(audio, dtype=np.float32)))
    if codes.shape != (frames, n_codebooks):
        raise ValueError(
            f"{channel} codec returned shape {codes.shape}, expected ({frames}, {n_codebooks})"
        )
    if not np.issubdtype(codes.dtype, np.integer):
        raise TypeError(f"{channel} codec must return integer codes, got {codes.dtype}")
    if codes.size:
        low, high = int(codes.min()), int(codes.max())
        if low < 0 or high >= CODEBOOK_SIZE:
            raise ValueError(f"{channel} codes out of range [0, {CODEBOOK_SIZE}): {low}..{high}")
    return np.ascontiguousarray(codes, dtype=_UINT16)


def _has_agent_text(conversation: Conversation) -> bool:
    return any(turn.speaker == "agent" and turn.text.strip() for turn in conversation.turns)


def conversation_arrays(
    conversation: Conversation,
    backend: CodecBackend,
    text_tokenizer: TextTokenizer | None = None,
) -> dict[str, Any]:
    """Load, resample, fit and encode both channels of one conversation.

    Returns ``user_codes``/``agent_codes`` ``uint16 [T, 12]``, ``text``
    ``int32 [T]``, ``control`` ``uint8 [T]`` and the frame count ``T``.
    """
    frames = frame_count(conversation.duration_s)
    if frames <= 0:
        raise ValueError(f"{conversation.id}: duration yields no frames")
    n_codebooks = int(getattr(backend, "n_codebooks", NUM_CODEBOOKS))
    if n_codebooks != NUM_CODEBOOKS:
        raise ValueError(
            f"{conversation.id}: backend has {n_codebooks} codebooks, schema fixes {NUM_CODEBOOKS}"
        )

    user_audio = fit_frames(
        read_audio(conversation.channels["user"], CODEC_SAMPLE_RATE), frames, CODEC_SAMPLE_RATE
    )
    agent_audio = fit_frames(
        read_audio(conversation.channels["agent"], CODEC_SAMPLE_RATE), frames, CODEC_SAMPLE_RATE
    )
    user_codes = _encode_channel(
        backend, user_audio, frames, n_codebooks=n_codebooks, channel="user"
    )
    agent_codes = _encode_channel(
        backend, agent_audio, frames, n_codebooks=n_codebooks, channel="agent"
    )

    if _has_agent_text(conversation):
        if text_tokenizer is None:
            raise ValueError(f"{conversation.id}: agent text requires a text tokenizer")
        text = align_text_to_frames(conversation.turns, conversation.duration_s, text_tokenizer)
    else:
        text = np.zeros(frames, dtype=np.int32)
    text = np.ascontiguousarray(text, dtype=_INT32)
    control = np.ascontiguousarray(derive_control_labels(conversation), dtype=_UINT8)
    if text.shape != (frames,) or control.shape != (frames,):
        raise RuntimeError("schema produced misaligned text/control arrays")

    return {
        "frames": frames,
        "user_codes": user_codes,
        "agent_codes": agent_codes,
        "text": text,
        "control": control,
        "control_histogram": control_histogram(control),
        "user_audio_seconds": float(user_audio.size) / CODEC_SAMPLE_RATE,
        "agent_audio_seconds": float(agent_audio.size) / CODEC_SAMPLE_RATE,
    }


def _silence_frame(backend: CodecBackend, n_codebooks: int) -> list[int]:
    """Deterministic padding frame: the codec encoding of 1920 zero samples."""
    silence = np.zeros(FRAME_SAMPLES[CODEC_SAMPLE_RATE], dtype=np.float32)
    codes = _encode_channel(backend, silence, 1, n_codebooks=n_codebooks, channel="silence")
    return [int(value) for value in codes[0]]


# ---------------------------------------------------------------------------
# Resumable writer
# ---------------------------------------------------------------------------


class TokenWriter:
    """Append conversations to shard files and maintain ``index.json``.

    The constructor either resumes the existing index (skipping its records) or
    starts a pristine one when ``overwrite=True``.  An unindexed tail in the
    last shard is truncated so a crashed run can always append safely.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        shard_hours: float = DEFAULT_SHARD_HOURS,
        overwrite: bool = False,
        n_codebooks: int = NUM_CODEBOOKS,
        codec_model: str | None = None,
        silence_frame: Sequence[int] | None = None,
        flush_every: int = DEFAULT_FLUSH_EVERY,
    ) -> None:
        if shard_hours <= 0 or not math.isfinite(shard_hours):
            raise ValueError(f"shard_hours must be finite and positive, got {shard_hours}")
        if n_codebooks != NUM_CODEBOOKS:
            raise ValueError(f"n_codebooks must be {NUM_CODEBOOKS}, got {n_codebooks}")
        if flush_every <= 0:
            raise ValueError(f"flush_every must be positive, got {flush_every}")

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_hours = float(shard_hours)
        self.n_codebooks = int(n_codebooks)
        self.codec_model = codec_model
        self.flush_every = int(flush_every)
        self._since_flush = 0
        self._file = None
        self._shard: dict[str, Any] | None = None
        self._closed = False
        self._path = index_path(self.out_dir)

        if overwrite:
            self._remove_existing()
            self._index = _empty_index(n_codebooks, codec_model, silence_frame)
        elif self._path.is_file():
            self._index = load_index(self.out_dir)
            existing_model = self._index.get("codec_model")
            if codec_model is not None and existing_model is not None and existing_model != codec_model:
                raise ValueError(
                    f"index was built with {existing_model!r}, refusing to mix with {codec_model!r}; "
                    "pass --overwrite to rebuild"
                )
            if silence_frame is not None:
                self.set_silence_frame(silence_frame)
        else:
            self._index = _empty_index(n_codebooks, codec_model, silence_frame)

        self._ids = {entry["id"] for entry in self._index["conversations"]}
        self._open_append_target()

    # -- public state ------------------------------------------------------

    @property
    def index(self) -> dict[str, Any]:
        return self._index

    @property
    def silence_frame(self) -> list[int] | None:
        return self._index["silence_frame"]

    def set_silence_frame(self, frame: Sequence[int]) -> None:
        values = [int(value) for value in frame]
        if len(values) != self.n_codebooks:
            raise ValueError(f"silence frame must have {self.n_codebooks} codes, got {len(values)}")
        if any(value < 0 or value >= CODEBOOK_SIZE for value in values):
            raise ValueError("silence frame codes out of range")
        existing = self._index["silence_frame"]
        if existing is not None and existing != values:
            raise ValueError("backend produced a different silence frame than the index; use --overwrite")
        self._index["silence_frame"] = values

    def __contains__(self, conversation_id: str) -> bool:
        return conversation_id in self._ids

    # -- disk management ---------------------------------------------------

    def _remove_existing(self) -> None:
        if self._path.is_file():
            try:
                data = load_index(self.out_dir)
            except Exception:  # noqa: BLE001 - overwrite still cleans unreadable indexes
                data = None
            if data is not None:
                for shard in data.get("shards", []):
                    try:
                        (self.out_dir / shard["file"]).unlink()
                    except FileNotFoundError:
                        pass
            else:
                for path in self.out_dir.glob(_SHARD_GLOB):
                    path.unlink()
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass
            return
        for path in self.out_dir.glob(_SHARD_GLOB):
            path.unlink()

    def _next_shard_name(self) -> str:
        used = {Path(shard["file"]).name for shard in self._index["shards"]}
        used.update(path.name for path in self.out_dir.glob(_SHARD_GLOB))
        suffix = 0
        while f"tokens_{suffix:05d}.bin" in used:
            suffix += 1
        return f"tokens_{suffix:05d}.bin"

    def _open_append_target(self) -> None:
        shards = self._index["shards"]
        if not shards:
            self._file = None
            self._shard = None
            return
        shard = shards[-1]
        path = self.out_dir / shard["file"]
        if not path.is_file():
            raise FileNotFoundError(f"index references missing shard {path}")
        actual = path.stat().st_size
        if actual < shard["bytes"]:
            raise ValueError(
                f"{path}: file is {actual} bytes but index records {shard['bytes']}; shard is truncated"
            )
        if actual > shard["bytes"]:
            with path.open("r+b") as handle:
                handle.truncate(shard["bytes"])
        self._shard = shard
        self._file = path.open("ab")

    def _start_new_shard(self) -> None:
        self._close_file()
        name = self._next_shard_name()
        path = self.out_dir / name
        self._file = path.open("xb")
        self._shard = {"file": name, "bytes": 0, "records": 0, "hours": 0.0}
        self._index["shards"].append(self._shard)

    def _close_file(self) -> None:
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None

    def flush(self) -> None:
        if self._closed:
            return
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())
        _write_index_atomic(self.out_dir, self._index)
        self._since_flush = 0

    def close(self) -> None:
        if self._closed:
            return
        self._close_file()
        _write_index_atomic(self.out_dir, self._index)
        self._closed = True

    def __enter__(self) -> "TokenWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    # -- records -----------------------------------------------------------

    def add(self, conversation: Conversation, arrays: dict[str, Any]) -> dict[str, Any]:
        """Append one already-encoded conversation and return its index entry."""
        if self._closed:
            raise RuntimeError("writer is closed")
        if conversation.id in self._ids:
            raise ValueError(f"duplicate conversation id {conversation.id!r}")
        frames = int(arrays["frames"])
        if frames != frame_count(conversation.duration_s):
            raise ValueError(
                f"{conversation.id}: {frames} frames != frame_count({conversation.duration_s})"
            )
        data = encode_record(
            arrays["user_codes"], arrays["agent_codes"], arrays["text"], arrays["control"]
        )
        if len(data) != record_bytes(frames, self.n_codebooks):
            raise RuntimeError("record size mismatch")

        if self._file is None:
            self._start_new_shard()
        assert self._shard is not None
        if (
            self._shard["records"] > 0
            and self._shard["hours"] + conversation.duration_s / 3600.0
            > self.shard_hours + 1e-12
        ):
            self._start_new_shard()
        assert self._shard is not None and self._file is not None

        offset = int(self._shard["bytes"])
        self._file.write(data)
        self._shard["bytes"] += len(data)
        self._shard["records"] += 1
        self._shard["hours"] += conversation.duration_s / 3600.0
        entry = {
            "id": conversation.id,
            "shard": self._shard["file"],
            "offset": offset,
            "bytes": len(data),
            "T": frames,
            "duration_s": float(conversation.duration_s),
            "source": conversation.source,
            "license": conversation.license,
            "split": conversation.split,
        }
        self._index["conversations"].append(entry)
        self._ids.add(conversation.id)
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self.flush()
        return entry

    def result(
        self,
        *,
        new_conversations: int = 0,
        skipped: int = 0,
        limit_reached: bool = False,
        new_hours: float = 0.0,
        new_bytes: int = 0,
    ) -> dict[str, Any]:
        conversations = self._index["conversations"]
        hours = sum(float(entry["duration_s"]) for entry in conversations) / 3600.0
        total_bytes = sum(int(entry["bytes"]) for entry in conversations)
        shard_bytes = sum(int(shard["bytes"]) for shard in self._index["shards"])
        index_bytes = self._path.stat().st_size if self._path.is_file() else 0
        return {
            "out": str(self.out_dir),
            "index": str(self._path),
            "conversations": len(conversations),
            "new_conversations": int(new_conversations),
            "skipped": int(skipped),
            "limit_reached": bool(limit_reached),
            "hours": hours,
            "new_hours": float(new_hours),
            "shards": len(self._index["shards"]),
            "record_bytes": total_bytes,
            "new_record_bytes": int(new_bytes),
            "shard_bytes": shard_bytes,
            "index_bytes": index_bytes,
            "bytes_per_hour": total_bytes / hours if hours > 0 else 0.0,
            "bytes_per_second": total_bytes / (hours * 3600.0) if hours > 0 else 0.0,
            "new_bytes_per_hour": new_bytes / new_hours if new_hours > 0 else 0.0,
            "n_codebooks": self.n_codebooks,
            "codec_model": self._index.get("codec_model"),
        }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_index(
    manifests: Sequence[str | Path],
    out_dir: str | Path,
    backend: CodecBackend,
    *,
    text_tokenizer: TextTokenizer | None = None,
    limit: int | None = None,
    shard_hours: float = DEFAULT_SHARD_HOURS,
    overwrite: bool = False,
    codec_model: str | None = None,
    flush_every: int = DEFAULT_FLUSH_EVERY,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Encode every new manifest conversation into ``out_dir``.

    Existing index ids are skipped unless ``overwrite`` is set.  ``limit``
    caps the number of *new* conversations, not the manifest length.
    """
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    n_codebooks = int(getattr(backend, "n_codebooks", NUM_CODEBOOKS))
    if n_codebooks != NUM_CODEBOOKS:
        raise ValueError(f"backend has {n_codebooks} codebooks, schema fixes {NUM_CODEBOOKS}")

    log = progress or (lambda message: logger.info(message))
    writer = TokenWriter(
        out_dir,
        shard_hours=shard_hours,
        overwrite=overwrite,
        n_codebooks=n_codebooks,
        codec_model=codec_model,
        flush_every=flush_every,
    )
    new_conversations = 0
    skipped = 0
    limit_reached = False
    new_hours = 0.0
    new_bytes = 0
    seen_manifest_ids: set[str] = set()
    started = False

    def ensure_tokenizer(conversation: Conversation) -> TextTokenizer | None:
        nonlocal text_tokenizer
        if _has_agent_text(conversation) and text_tokenizer is None:
            model = codec_model or DEFAULT_CODEC_MODEL
            log(f"loading text tokenizer {model}")
            text_tokenizer = HFTokenizer(model)
        return text_tokenizer

    try:
        if not started:
            computed_silence = _silence_frame(backend, n_codebooks)
            writer.set_silence_frame(computed_silence)
            started = True

        for manifest in manifests:
            for conversation in read_manifest(manifest):
                if conversation.id in seen_manifest_ids:
                    raise ValueError(f"duplicate conversation id {conversation.id!r} across manifests")
                seen_manifest_ids.add(conversation.id)
                if conversation.id in writer:
                    skipped += 1
                    continue
                if limit is not None and new_conversations >= limit:
                    limit_reached = True
                    break

                tokenizer = ensure_tokenizer(conversation)
                arrays = conversation_arrays(conversation, backend, tokenizer)
                entry = writer.add(conversation, arrays)
                new_conversations += 1
                new_hours += conversation.duration_s / 3600.0
                new_bytes += int(entry["bytes"])
                if new_conversations == 1 or new_conversations % 10 == 0:
                    log(
                        f"{new_conversations} conversations, {new_hours:.2f} h, "
                        f"{new_bytes / 2**30:.3f} GiB records"
                    )
            if limit_reached:
                break
    finally:
        writer.close()

    return writer.result(
        new_conversations=new_conversations,
        skipped=skipped,
        limit_reached=limit_reached,
        new_hours=new_hours,
        new_bytes=new_bytes,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


_WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)


def normalize_words(text: str) -> list[str]:
    """Lowercase word tokens used by :func:`word_error_rate`."""
    return _WORD_RE.findall(text.lower().replace("’", "'"))


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_word in enumerate(reference, 1):
        current = [row]
        for column, hyp_word in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate after simple case-fold/punctuation-free tokenization.

    Returns ``0.0`` for two empty strings and ``1.0`` when only the hypothesis
    has words.
    """
    ref_words = normalize_words(reference)
    hyp_words = normalize_words(hypothesis)
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def check_streaming_decode(
    backend: DecodingCodecBackend,
    codes: np.ndarray,
    *,
    chunk_frames: int = 2,
    tolerance: float = STREAMING_TOLERANCE,
    max_frames: int = 512,
) -> dict[str, Any]:
    """Compare chunked streaming decode against one-shot decode.

    The chunk duration must stay at or below 160 ms (2 frames at 12.5 Hz).
    Only ``max_frames`` (default 512 = 40.96 s) are compared one-shot; decoding
    a multi-hour shard in one call exhausts host memory.  The full record is
    still decoded chunk-by-chunk for the round-trip check.
    """
    chunk_seconds = chunk_frames / FRAME_RATE
    if chunk_frames <= 0:
        raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")
    if chunk_seconds > MAX_CHUNK_SECONDS + 1e-12:
        raise ValueError(
            f"chunk of {chunk_frames} frames is {chunk_seconds:.3f}s, above {MAX_CHUNK_SECONDS}s"
        )
    array = np.ascontiguousarray(np.asarray(codes, dtype=np.int64))
    total_frames = int(array.shape[0])
    if array.shape[0] > max_frames:
        array = array[:max_frames]
    frames = int(array.shape[0])
    if frames == 0:
        return {
            "status": "SKIPPED",
            "reason": "no frames",
            "frames": 0,
            "chunk_frames": chunk_frames,
            "max_chunk_seconds": chunk_seconds,
        }
    one_shot = np.asarray(backend.decode(array), dtype=np.float32).reshape(-1)
    cache = backend.new_streaming_cache()
    parts: list[np.ndarray] = []
    for start in range(0, frames, chunk_frames):
        end = min(frames, start + chunk_frames)
        chunk = array[start:end]
        parts.append(
            np.asarray(
                backend.decode(chunk, streaming_cache=cache, finalize=end == frames),
                dtype=np.float32,
            ).reshape(-1)
        )
    streamed = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    same_shape = streamed.shape == one_shot.shape
    if same_shape and one_shot.size:
        delta = np.abs(streamed - one_shot)
        max_abs = float(delta.max())
        scale = float(np.abs(one_shot).max())
        allclose = bool(np.allclose(streamed, one_shot, atol=tolerance, rtol=tolerance))
    else:
        max_abs = math.inf if not same_shape else 0.0
        scale = float(np.abs(one_shot).max()) if one_shot.size else 0.0
        allclose = bool(same_shape)
    return {
        "status": "PASS" if allclose else "FAIL",
        "frames": frames,
        "total_frames": total_frames,
        "compared_frames": frames,
        "chunks": len(parts),
        "chunk_frames": chunk_frames,
        "max_chunk_seconds": chunk_seconds,
        "one_shot_samples": int(one_shot.size),
        "streamed_samples": int(streamed.size),
        "max_abs_diff": max_abs,
        "signal_peak": scale,
        "tolerance": tolerance,
    }


class LiquidASR:
    """LFM3-Omni ASR wrapper using the model card's transcription prompt."""

    def __init__(
        self,
        model_name: str = DEFAULT_CODEC_MODEL,
        *,
        device: str | None = None,
        language: str = "English",
        trust_remote_code: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        self.model_name = model_name
        self.device = str(device)
        self.language = language
        self._model = AutoModelForMultimodalLM.from_pretrained(
            model_name, dtype=dtype, device_map=self.device, trust_remote_code=trust_remote_code
        ).eval()
        self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self._torch = torch

    def transcribe(self, audio: np.ndarray, sample_rate: int = CODEC_SAMPLE_RATE) -> str:
        torch = self._torch
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = handle.name
        try:
            write_audio(path, audio, sample_rate)
            conversation = [
                {"role": "system", "content": f"Perform ASR in {self.language}."},
                {"role": "user", "content": [{"type": "audio", "audio": path}]},
            ]
            inputs = self._processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs, max_new_tokens=1024, generation_mode="text"
                )
            prompt_length = int(inputs["input_ids"].shape[1])
            end = int(outputs.sequences.shape[1])
            lengths = getattr(outputs, "sequence_lengths", None)
            if lengths is not None:
                end = int(lengths[0])
            return self._processor.decode(
                outputs.sequences[0, prompt_length:end], skip_special_tokens=True
            )
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


class UTMOSScorer:
    """Optional UTMOS wrapper; prefers the UTMOS22 API from the ``utmos`` package."""

    def __init__(self, device: str | None = None) -> None:
        try:
            from utmos import UTMOS22ScoreModel  # type: ignore[import-not-found]

            try:
                self._model = UTMOS22ScoreModel(device=device) if device else UTMOS22ScoreModel()
            except TypeError:
                self._model = UTMOS22ScoreModel()
        except Exception as first_error:  # noqa: BLE001 - fall back to the older API
            try:
                from utmos import UTMOSScore  # type: ignore[import-not-found]

                try:
                    self._model = UTMOSScore(device=device) if device else UTMOSScore()
                except TypeError:
                    self._model = UTMOSScore()
            except Exception as second_error:  # noqa: BLE001
                raise RuntimeError(
                    "UTMOS unavailable; install the optional `utmos` package "
                    f"(first error: {first_error}; second error: {second_error})"
                ) from second_error
        self.device = device

    def score(self, audio: np.ndarray, sample_rate: int) -> float:
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        try:
            value = self._model(waveform, sample_rate)
        except TypeError:
            value = self._model(waveform)
        if isinstance(value, (tuple, list, np.ndarray)):
            value = np.asarray(value).reshape(-1)[0]
        return float(value)


def _decode_channel(
    backend: DecodingCodecBackend,
    codes: np.ndarray,
    channel: str,
    *,
    chunk_frames: int = 2,
) -> dict[str, Any]:
    # A shard record can hold hours of frames.  One-shot decode then allocates
    # the whole latent sequence; streaming decode keeps memory bounded and is
    # the path serving uses.
    frames = int(np.asarray(codes).shape[0])
    if frames > chunk_frames and hasattr(backend, "new_streaming_cache"):
        cache = backend.new_streaming_cache()
        parts: list[np.ndarray] = []
        for start in range(0, frames, chunk_frames):
            end = min(frames, start + chunk_frames)
            parts.append(
                np.asarray(
                    backend.decode(codes[start:end], streaming_cache=cache, finalize=end == frames),
                    dtype=np.float32,
                ).reshape(-1)
            )
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    else:
        audio = np.asarray(backend.decode(codes), dtype=np.float32).reshape(-1)
    finite = bool(np.isfinite(audio).all()) if audio.size else True
    return {
        "audio": audio,
        "samples": int(audio.size),
        "seconds": audio.size / CODEC_SAMPLE_RATE,
        "finite": finite,
        "peak": float(np.abs(audio).max()) if audio.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0,
        "channel": channel,
    }


def _reference_text(conversation: Conversation) -> str:
    return " ".join(
        turn.text.strip()
        for turn in conversation.turns
        if turn.speaker == "agent" and turn.text.strip()
    )


def _load_manifest_map(paths: Sequence[str | Path] | None) -> dict[str, Conversation]:
    if not paths:
        return {}
    mapping: dict[str, Conversation] = {}
    for path in paths:
        for conversation in read_manifest(path):
            mapping[conversation.id] = conversation
    return mapping


def verify_index(
    out_dir: str | Path,
    *,
    samples: int = 10,
    seed: int = 0,
    backend: DecodingCodecBackend | None = None,
    codec_model: str = DEFAULT_CODEC_MODEL,
    device: str | None = None,
    manifests: Sequence[str | Path] | None = None,
    audio_dir: str | Path | None = None,
    run_asr: bool = False,
    run_utmos: bool = False,
    asr: Any | None = None,
    utmos: Any | None = None,
    language: str = "English",
    chunk_frames: int = 2,
    tolerance: float = STREAMING_TOLERANCE,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Decode sampled records, check streaming equivalence and optional scores.

    The codec backend is loaded lazily.  If it cannot be loaded the result is
    ``UNVERIFIED`` with the exception text; no audio numbers are invented.
    Likewise ASR WER and UTMOS are only reported when their model/package
    actually ran.
    """
    if samples <= 0:
        raise ValueError(f"samples must be positive, got {samples}")
    chunk_seconds = chunk_frames / FRAME_RATE
    if chunk_seconds > MAX_CHUNK_SECONDS + 1e-12:
        raise ValueError(f"chunk_frames={chunk_frames} exceeds {MAX_CHUNK_SECONDS}s")
    root = Path(out_dir)
    try:
        index = load_index(root)
    except Exception as exc:  # noqa: BLE001 - report, do not fabricate
        return {
            "status": "UNVERIFIED",
            "reason": f"cannot load index: {exc}",
            "index": str(root),
        }
    entries = list(index["conversations"])
    if not entries:
        return {"status": "UNVERIFIED", "reason": "index has no conversations", "index": str(root)}

    if backend is None:
        try:
            backend = LiquidCodecBackend(codec_model, device=device)
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "UNVERIFIED",
                "reason": f"codec backend unavailable: {exc}",
                "index": str(root),
                "samples_requested": samples,
            }
    if not hasattr(backend, "decode"):
        return {
            "status": "UNVERIFIED",
            "reason": "backend has no decode(); cannot resynthesize",
            "index": str(root),
        }

    n_codebooks = int(index["n_codebooks"])
    frame_samples = int(index.get("frame_samples", FRAME_SAMPLES[CODEC_SAMPLE_RATE]))
    rng = np.random.default_rng(seed)
    count = min(samples, len(entries))
    positions = rng.choice(len(entries), size=count, replace=False)
    outputs: list[dict[str, Any]] = []
    core_failures: list[str] = []
    streaming_failures: list[str] = []
    worst_streaming: dict[str, Any] | None = None
    references = _load_manifest_map(manifests)
    audio_root = Path(audio_dir) if audio_dir is not None else None
    if audio_root is not None:
        audio_root.mkdir(parents=True, exist_ok=True)

    for position in positions:
        entry = entries[int(position)]
        record = read_record(
            root / entry["shard"],
            int(entry["offset"]),
            n_codebooks=n_codebooks,
            frames=int(entry["T"]),
        )
        verified_frames = int(entry["T"])
        if max_frames is not None:
            if max_frames <= 0:
                raise ValueError("max_frames must be positive")
            verified_frames = min(verified_frames, int(max_frames))
            for channel in ("user", "agent"):
                record[f"{channel}_codes"] = record[f"{channel}_codes"][:verified_frames]
        expected_samples = verified_frames * frame_samples
        decoded: dict[str, dict[str, Any]] = {}
        sample: dict[str, Any] = {
            "id": entry["id"],
            "frames": verified_frames,
            "record_frames": int(entry["T"]),
            "duration_s": float(entry["duration_s"]),
            "shard": entry["shard"],
            "offset": int(entry["offset"]),
        }
        try:
            for channel in ("user", "agent"):
                codes = record[f"{channel}_codes"]
                info = _decode_channel(backend, codes, channel)
                if info["samples"] != expected_samples:
                    core_failures.append(
                        f"{entry['id']}:{channel} decoded {info['samples']} != {expected_samples} samples"
                    )
                if not info["finite"]:
                    core_failures.append(f"{entry['id']}:{channel} non-finite audio")
                decoded[channel] = info
                sample[channel] = {
                    key: value for key, value in info.items() if key != "audio"
                }
                if audio_root is not None:
                    write_audio(
                        audio_root / f"{entry['id']}_{channel}.wav",
                        info["audio"],
                        CODEC_SAMPLE_RATE,
                    )
        except Exception as exc:  # noqa: BLE001
            core_failures.append(f"{entry['id']}: decode failed: {exc}")
            sample["error"] = str(exc)
            outputs.append(sample)
            continue

        for channel in ("user", "agent"):
            try:
                streaming = check_streaming_decode(
                    backend,
                    record[f"{channel}_codes"],
                    chunk_frames=chunk_frames,
                    tolerance=tolerance,
                )
            except Exception as exc:  # noqa: BLE001
                streaming = {"status": "UNVERIFIED", "reason": f"streaming decode failed: {exc}"}
            sample[f"{channel}_streaming"] = streaming
            if streaming.get("status") == "FAIL":
                streaming_failures.append(f"{entry['id']}:{channel}")
            if streaming.get("status") in ("PASS", "FAIL"):
                candidate = streaming.get("max_abs_diff")
                if worst_streaming is None or (
                    isinstance(candidate, (int, float))
                    and candidate > worst_streaming.get("max_abs_diff", -1)
                ):
                    worst_streaming = dict(streaming, id=entry["id"], channel=channel)

        sample["reference_available"] = entry["id"] in references
        sample["agent_audio"] = decoded["agent"]["audio"]
        outputs.append(sample)

    # External optional checks -------------------------------------------------
    asr_status: dict[str, Any]
    if run_asr:
        if not references:
            asr_status = {
                "status": "UNVERIFIED",
                "reason": "no --manifests reference transcripts provided; WER not computed",
            }
        else:
            try:
                scorer = asr if asr is not None else LiquidASR(
                    codec_model, device=device, language=language
                )
            except Exception as exc:  # noqa: BLE001
                asr_status = {"status": "UNVERIFIED", "reason": f"ASR model unavailable: {exc}"}
            else:
                scores = []
                for sample in outputs:
                    conversation = references.get(sample["id"])
                    if conversation is None or "agent_audio" not in sample:
                        continue
                    reference = _reference_text(conversation)
                    if not reference:
                        continue
                    try:
                        hypothesis = scorer.transcribe(sample["agent_audio"])
                    except Exception as exc:  # noqa: BLE001 - report instead of fabricating WER
                        sample["asr_error"] = str(exc)
                        continue
                    sample["asr_hypothesis"] = hypothesis
                    sample["wer"] = word_error_rate(reference, hypothesis)
                    scores.append(sample["wer"])
                if scores:
                    mean_wer = float(np.mean(scores))
                    asr_status = {
                        "status": "PASS" if mean_wer < WER_THRESHOLD else "FAIL",
                        "wer": mean_wer,
                        "samples": len(scores),
                        "threshold": WER_THRESHOLD,
                    }
                else:
                    errors = [sample["asr_error"] for sample in outputs if "asr_error" in sample]
                    asr_status = {
                        "status": "UNVERIFIED",
                        "reason": (
                            f"ASR transcription failed for {len(errors)} sample(s): {errors[0]}"
                            if errors
                            else "no sampled conversation had a reference transcript and audio"
                        ),
                    }
    else:
        asr_status = {"status": "SKIPPED", "reason": "pass --asr to measure WER"}

    utmos_status: dict[str, Any]
    if run_utmos:
        if not references:
            utmos_status = {
                "status": "UNVERIFIED",
                "reason": "no --manifests source channels provided; UTMOS delta not computed",
            }
        else:
            try:
                scorer = utmos if utmos is not None else UTMOSScorer(device=device)
            except Exception as exc:  # noqa: BLE001
                utmos_status = {"status": "UNVERIFIED", "reason": f"UTMOS unavailable: {exc}"}
            else:
                deltas = []
                for sample in outputs:
                    conversation = references.get(sample["id"])
                    if conversation is None or "agent_audio" not in sample:
                        continue
                    try:
                        synthesized = resample(sample["agent_audio"], CODEC_SAMPLE_RATE, 16000)
                        source = read_audio(conversation.channels["agent"], 16000)
                        synthesized_score = scorer.score(synthesized, 16000)
                        source_score = scorer.score(source, 16000)
                    except Exception as exc:  # noqa: BLE001 - keep per-sample diagnostics
                        sample["utmos_error"] = str(exc)
                        continue
                    sample["utmos_synthesized"] = synthesized_score
                    sample["utmos_source"] = source_score
                    sample["utmos_delta"] = synthesized_score - source_score
                    deltas.append(abs(sample["utmos_delta"]))
                if deltas:
                    mean_delta = float(np.mean(deltas))
                    utmos_status = {
                        "status": "PASS" if mean_delta <= UTMOS_DELTA_THRESHOLD else "FAIL",
                        "mean_abs_delta": mean_delta,
                        "samples": len(deltas),
                        "threshold": UTMOS_DELTA_THRESHOLD,
                    }
                else:
                    utmos_status = {
                        "status": "UNVERIFIED",
                        "reason": "no sampled conversation had source agent audio",
                    }
    else:
        utmos_status = {"status": "SKIPPED", "reason": "pass --utmos to measure UTMOS"}

    for sample in outputs:
        sample.pop("agent_audio", None)

    checks = [asr_status, utmos_status]
    reasons: list[str] = []
    status = "PASS"
    if core_failures:
        status = "FAIL"
        reasons.append(f"{len(core_failures)} core decode failure(s)")
    if streaming_failures:
        status = "FAIL"
        reasons.append(f"streaming mismatch on {len(streaming_failures)} channel(s)")
    for check in checks:
        if check["status"] == "FAIL":
            status = "FAIL"
            reasons.append(check.get("reason", f"{check.get('wer', check.get('mean_abs_delta'))} above threshold"))
    if status != "FAIL":
        unverified = [check for check in checks if check["status"] == "UNVERIFIED"]
        if any(
            not output.get(channel, {}).get("finite", True)
            for output in outputs
            for channel in ("user", "agent")
        ):
            status = "FAIL"
            reasons.append("non-finite decoded samples")
        elif unverified:
            status = "UNVERIFIED"
            reasons.extend(check.get("reason", "unverified") for check in unverified)
        elif streaming_failures == [] and any(
            sample.get(f"{channel}_streaming", {}).get("status") == "UNVERIFIED"
            for sample in outputs
            for channel in ("user", "agent")
        ):
            status = "UNVERIFIED"
            reasons.append("streaming decode unavailable")

    return {
        "status": status,
        "reason": "; ".join(reasons) if reasons else None,
        "index": str(root),
        "samples_requested": samples,
        "samples_verified": len(outputs),
        "n_codebooks": n_codebooks,
        "frame_samples": frame_samples,
        "gate": {
            "wer_threshold": WER_THRESHOLD,
            "utmos_delta_threshold": UTMOS_DELTA_THRESHOLD,
            "max_chunk_seconds": MAX_CHUNK_SECONDS,
            "streaming_tolerance": tolerance,
        },
        "core": {
            "status": "FAIL" if core_failures else "PASS",
            "failures": core_failures,
        },
        "streaming": {
            "status": "FAIL" if streaming_failures else "PASS",
            "failures": streaming_failures,
            "worst": worst_streaming,
        },
        "asr": asr_status,
        "utmos": utmos_status,
        "samples": outputs,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.tokenize",
        description="Encode conversation manifests into 12.5 Hz RPT1 token shards.",
    )
    parser.add_argument("--manifests", nargs="+", required=True, help="manifest JSONL file(s)")
    parser.add_argument("--out", required=True, help="output directory (shards + index.json)")
    parser.add_argument(
        "--codec-model",
        "--model",
        dest="codec_model",
        default=DEFAULT_CODEC_MODEL,
        help=f"checkpoint holding the LiquidCodec (default: {DEFAULT_CODEC_MODEL})",
    )
    parser.add_argument("--limit", type=int, default=None, help="maximum new conversations this run")
    parser.add_argument(
        "--shard-hours",
        type=float,
        default=DEFAULT_SHARD_HOURS,
        help=f"conversation audio hours per shard file (default: {DEFAULT_SHARD_HOURS})",
    )
    parser.add_argument("--overwrite", action="store_true", help="discard existing shards and index")
    parser.add_argument("--device", default=None, help="codec device (default: cuda when available)")
    return parser


def _verify_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.tokenize verify",
        description="Decode sampled shard records and check round-trip/streaming quality.",
    )
    parser.add_argument("--out", required=True, help="token directory containing index.json")
    parser.add_argument("--samples", type=int, default=10, help="number of conversations to sample")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument(
        "--codec-model",
        "--model",
        dest="codec_model",
        default=DEFAULT_CODEC_MODEL,
        help=f"checkpoint holding the LiquidCodec (default: {DEFAULT_CODEC_MODEL})",
    )
    parser.add_argument("--device", default=None, help="codec/ASR/UTMOS device")
    parser.add_argument(
        "--manifests",
        nargs="+",
        default=None,
        help="manifest JSONL file(s) with reference transcripts and source channels",
    )
    parser.add_argument("--audio-dir", default=None, help="optional directory for resynthesized WAVs")
    parser.add_argument("--asr", action="store_true", help="measure ASR WER with LFM3-Omni")
    parser.add_argument("--utmos", action="store_true", help="measure UTMOS delta with the utmos package")
    parser.add_argument("--language", default="English", help="ASR language name")
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=2,
        help="streaming decode chunk size in frames (max 2 = 160 ms)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=STREAMING_TOLERANCE,
        help="streaming one-shot comparison tolerance",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="verify only the first N frames of each sampled record (quick smoke)",
    )
    parser.add_argument("--report", default=None, help="optional path to write the JSON result")
    return parser


def _run_build(args: argparse.Namespace) -> int:
    backend = LiquidCodecBackend(args.codec_model, device=args.device)
    result = build_index(
        args.manifests,
        args.out,
        backend,
        limit=args.limit,
        shard_hours=args.shard_hours,
        overwrite=args.overwrite,
        codec_model=args.codec_model,
    )
    print(json.dumps(result, indent=2))
    return 0


def _run_verify(args: argparse.Namespace) -> int:
    result = verify_index(
        args.out,
        samples=args.samples,
        seed=args.seed,
        codec_model=args.codec_model,
        device=args.device,
        manifests=args.manifests,
        audio_dir=args.audio_dir,
        run_asr=args.asr,
        run_utmos=args.utmos,
        language=args.language,
        chunk_frames=args.chunk_frames,
        tolerance=args.tolerance,
        max_frames=args.max_frames,
    )
    text = json.dumps(result, indent=2)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    if result["status"] == "PASS":
        return 0
    if result["status"] == "FAIL":
        return 1
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point supporting both the bare build form and ``verify``."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        if arguments and arguments[0] == "verify":
            return _run_verify(_verify_parser().parse_args(arguments[1:]))
        if arguments and arguments[0] == "build":
            arguments = arguments[1:]
        return _run_build(_build_parser().parse_args(arguments))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - a CLI should print, not dump a traceback
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
