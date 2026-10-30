"""Memory-mapped PyTorch dataset over Repartee RPT1 token shards.

``ShardIndex`` validates ``index.json`` (format version, codebook count, record
offsets, shard sizes) and exposes conversations with split filtering.
``DuplexShardDataset`` never opens an audio file: each item is a fixed
``seq_frames`` window read from the shard's byte offsets, with a random start
per conversation so long conversations are sampled at many alignments.

When a window runs past a conversation's end it is padded from the
deterministic silence frame stored in ``index.json`` (the codec's encoding of
1920 zero samples).  The returned ``valid`` mask marks real frames, and
``collate`` turns it into per-sequence lengths.

Deviations and limitations
--------------------------
* The blueprint says ``[T, 8]`` codes per channel; the published LiquidCodec
  has 12 codebooks, so each channel is ``[F, 12]`` (``src.schema`` fixes 12).
* Items are right-padded to a fixed ``seq_frames`` and carry a mask.  The
  collator does not pack multiple short conversations into one sequence.
* Conversations are sampled in proportion to their frame count, then a window
  start is drawn uniformly in ``[0, T - seq_frames]`` when ``T > seq_frames``;
  shorter conversations start at frame 0 and are padded.
* ``ShardIndex`` tolerates shard files larger than the indexed byte count (a
  crashed writer's unindexed tail) but rejects all internal offset gaps,
  overlaps and short files.
* Randomness is derived per item index and seed, so worker processes and
  repeated accesses are deterministic without sharing an RNG.
* ``measure_random_batch_load`` excludes one warm-up batch (page cache/initial
  memmap fault) from its statistics and compares the mean against the 5 ms
  gate; median and p95 are reported alongside.
* The 15-30 GB corpus target is a projection from bytes/hour, never asserted
  here.  Use fixture numbers only as a dataloader-overhead measurement.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from .schema import (
    CODEBOOK_SIZE,
    CONTROL_IDS,
    CONTROL_LABEL_TO_TOKEN_NAME,
    CONTROL_LABELS,
    CONTROL_TOKEN_NAMES,
    FRAME_RATE,
    NUM_CODEBOOKS,
)
from .tokenize import (
    FORMAT_VERSION,
    HEADER_SIZE,
    HEADER_STRUCT,
    MAGIC,
    load_index,
    record_bytes,
    record_offsets,
)


@dataclass(frozen=True)
class ShardInfo:
    """One ``tokens_XXXXX.bin`` file from the index."""

    file: str
    bytes: int
    records: int
    hours: float


@dataclass(frozen=True)
class ConversationRecord:
    """Location and metadata of one packed conversation."""

    id: str
    shard: str
    offset: int
    bytes: int
    T: int
    duration_s: float
    source: str
    license: str
    split: str


class ShardIndex:
    """Validated view over one token directory's ``index.json``."""

    def __init__(
        self,
        directory: str | Path,
        *,
        n_codebooks: int,
        frame_rate: float,
        frame_samples: int,
        codebook_size: int,
        shards: Sequence[ShardInfo],
        conversations: Sequence[ConversationRecord],
        silence_frame: Sequence[int] | None = None,
    ) -> None:
        if n_codebooks != NUM_CODEBOOKS:
            raise ValueError(f"index has {n_codebooks} codebooks, schema fixes {NUM_CODEBOOKS}")
        if float(frame_rate) != FRAME_RATE:
            raise ValueError(f"index frame_rate is {frame_rate}, expected {FRAME_RATE}")
        self.directory = Path(directory)
        self.n_codebooks = int(n_codebooks)
        self.frame_rate = float(frame_rate)
        self.frame_samples = int(frame_samples)
        self.codebook_size = int(codebook_size)
        self.shards = tuple(shards)
        self.conversations = tuple(conversations)
        self.silence_frame = (
            None if silence_frame is None else tuple(int(value) for value in silence_frame)
        )
        self._shards_by_file = {shard.file: shard for shard in self.shards}
        self._by_id = {conversation.id: conversation for conversation in self.conversations}
        self._validate()

    # -- construction ------------------------------------------------------

    @classmethod
    def load(cls, directory: str | Path) -> "ShardIndex":
        """Load and validate ``index.json`` and every indexed offset."""
        directory = Path(directory)
        data = load_index(directory)
        shards = [
            ShardInfo(
                file=str(item["file"]),
                bytes=int(item["bytes"]),
                records=int(item["records"]),
                hours=float(item["hours"]),
            )
            for item in data.get("shards", [])
        ]
        conversations = [
            ConversationRecord(
                id=str(item["id"]),
                shard=str(item["shard"]),
                offset=int(item["offset"]),
                bytes=int(item["bytes"]),
                T=int(item["T"]),
                duration_s=float(item["duration_s"]),
                source=str(item["source"]),
                license=str(item["license"]),
                split=str(item["split"]),
            )
            for item in data.get("conversations", [])
        ]
        index = cls(
            directory,
            n_codebooks=int(data["n_codebooks"]),
            frame_rate=float(data["frame_rate"]),
            frame_samples=int(data.get("frame_samples", 1920)),
            codebook_size=int(data.get("codebook_size", CODEBOOK_SIZE)),
            shards=shards,
            conversations=conversations,
            silence_frame=data.get("silence_frame"),
        )
        return index

    # -- validation --------------------------------------------------------

    def _validate(self) -> None:
        if self.codebook_size != CODEBOOK_SIZE:
            raise ValueError(f"index codebook_size is {self.codebook_size}, expected {CODEBOOK_SIZE}")
        if self.silence_frame is not None:
            if len(self.silence_frame) != self.n_codebooks:
                raise ValueError("silence_frame has the wrong length")
            if any(value < 0 or value >= CODEBOOK_SIZE for value in self.silence_frame):
                raise ValueError("silence_frame codes out of range")

        for shard in self.shards:
            path = self.directory / shard.file
            if not path.is_file():
                raise ValueError(f"shard file missing: {path}")
            actual = path.stat().st_size
            if actual < shard.bytes:
                raise ValueError(
                    f"{path}: file is {actual} bytes but index records {shard.bytes}"
                )
        if len({shard.file for shard in self.shards}) != len(self.shards):
            raise ValueError("duplicate shard file in index")

        grouped: dict[str, list[ConversationRecord]] = {shard.file: [] for shard in self.shards}
        seen_ids: set[str] = set()
        for conversation in self.conversations:
            if conversation.id in seen_ids:
                raise ValueError(f"duplicate conversation id {conversation.id!r}")
            seen_ids.add(conversation.id)
            if conversation.shard not in grouped:
                raise ValueError(f"{conversation.id}: unknown shard {conversation.shard!r}")
            if conversation.split not in ("train", "valid", "test"):
                raise ValueError(f"{conversation.id}: invalid split {conversation.split!r}")
            if conversation.T <= 0 or conversation.duration_s <= 0:
                raise ValueError(f"{conversation.id}: non-positive frames/duration")
            if conversation.bytes != record_bytes(conversation.T, self.n_codebooks):
                raise ValueError(f"{conversation.id}: byte size disagrees with frame count")
            grouped[conversation.shard].append(conversation)

        for shard in self.shards:
            records = sorted(grouped[shard.file], key=lambda item: item.offset)
            expected = 0
            for conversation in records:
                if conversation.offset != expected:
                    raise ValueError(
                        f"{shard.file}: offset {conversation.offset} for {conversation.id!r} "
                        f"does not follow {expected} (gap or overlap)"
                    )
                expected += conversation.bytes
            if expected != shard.bytes:
                raise ValueError(
                    f"{shard.file}: records end at {expected} bytes, shard metadata says {shard.bytes}"
                )
            if len(records) != shard.records:
                raise ValueError(
                    f"{shard.file}: {len(records)} records, metadata says {shard.records}"
                )
            expected_hours = sum(item.duration_s for item in records) / 3600.0
            if abs(expected_hours - shard.hours) > 1e-6 * max(1.0, expected_hours):
                raise ValueError(
                    f"{shard.file}: {expected_hours:.6f} hours, metadata says {shard.hours:.6f}"
                )
            if records:
                path = self.directory / shard.file
                with path.open("rb") as handle:
                    header = handle.read(HEADER_SIZE)
                if len(header) != HEADER_SIZE:
                    raise ValueError(f"{path}: missing record header")
                magic, version, n_codebooks, frames = HEADER_STRUCT.unpack(header)
                first = records[0]
                if magic != MAGIC or version != FORMAT_VERSION:
                    raise ValueError(f"{path}: bad record magic/version")
                if n_codebooks != self.n_codebooks or frames != first.T:
                    raise ValueError(
                        f"{path}: first record header disagrees with index "
                        f"({n_codebooks} codebooks, {frames} frames)"
                    )

    # -- queries -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.conversations)

    def __iter__(self) -> Iterator[ConversationRecord]:
        return iter(self.conversations)

    def by_split(self, split: str = "train") -> tuple[ConversationRecord, ...]:
        """Conversations belonging to ``split``; ``all`` returns everything."""
        if split in ("all", "*"):
            return self.conversations
        if split not in ("train", "valid", "test"):
            raise ValueError(f"split must be train/valid/test/all, got {split!r}")
        return tuple(conversation for conversation in self.conversations if conversation.split == split)

    def conversation(self, conversation_id: str) -> ConversationRecord:
        try:
            return self._by_id[conversation_id]
        except KeyError as exc:
            raise KeyError(f"unknown conversation id {conversation_id!r}") from exc

    @property
    def hours(self) -> float:
        return sum(conversation.duration_s for conversation in self.conversations) / 3600.0

    def split_hours(self, split: str = "train") -> float:
        return sum(conversation.duration_s for conversation in self.by_split(split)) / 3600.0

    @property
    def bytes(self) -> int:
        return sum(shard.bytes for shard in self.shards)


def control_id_for_name(name: str) -> int:
    """Control id for a schema label (``SPEAK``) or token name (``<|speak|>``)."""
    if not isinstance(name, str):
        raise TypeError(f"control name must be a string, got {type(name).__name__}")
    key = name.strip()
    if key in CONTROL_IDS:
        return CONTROL_IDS[key]
    upper = key.upper()
    if upper in CONTROL_IDS:
        return CONTROL_IDS[upper]
    inverse = {token: label for label, token in CONTROL_LABEL_TO_TOKEN_NAME.items()}
    if key in inverse:
        return CONTROL_IDS[inverse[key]]
    raise KeyError(
        f"unknown control name {name!r}; expected one of {CONTROL_LABELS + CONTROL_TOKEN_NAMES}"
    )


class DuplexShardDataset(torch.utils.data.Dataset):
    """Fixed-size random windows over dual-channel token shards."""

    def __init__(
        self,
        index: ShardIndex | str | Path,
        seq_frames: int = 4096,
        split: str = "train",
        samples_per_epoch: int | None = None,
        seed: int = 0,
        silence_frame: Sequence[int] | None = None,
    ) -> None:
        if seq_frames <= 0:
            raise ValueError(f"seq_frames must be positive, got {seq_frames}")
        if not isinstance(index, ShardIndex):
            index = ShardIndex.load(index)
        self.index = index
        self.seq_frames = int(seq_frames)
        self.split = split
        self.seed = int(seed)
        self.conversations = index.by_split(split)
        if not self.conversations:
            raise ValueError(f"no conversations in split {split!r}")

        silence_source = silence_frame if silence_frame is not None else index.silence_frame
        if silence_source is None:
            raise ValueError(
                "index has no silence_frame; rebuild tokens with src.tokenize or pass silence_frame"
            )
        silence = np.asarray(silence_source, dtype=np.uint16).reshape(-1)
        if silence.shape != (index.n_codebooks,):
            raise ValueError(
                f"silence frame must have {index.n_codebooks} codes, got {silence.shape}"
            )
        if silence.size and int(silence.max()) >= CODEBOOK_SIZE:
            raise ValueError("silence frame codes out of range")
        self.silence_frame = np.ascontiguousarray(silence)

        self._frames = np.asarray([record.T for record in self.conversations], dtype=np.int64)
        self._cumulative = np.cumsum(self._frames)
        self._total_frames = int(self._cumulative[-1])
        self._memmaps: dict[str, np.memmap] = {}
        for record in self.conversations:
            if record.shard not in self._memmaps:
                self._memmaps[record.shard] = np.memmap(
                    index.directory / record.shard, dtype=np.uint8, mode="r"
                )
        if samples_per_epoch is not None:
            if samples_per_epoch <= 0:
                raise ValueError(f"samples_per_epoch must be positive, got {samples_per_epoch}")
            self.samples_per_epoch = int(samples_per_epoch)
        else:
            self.samples_per_epoch = max(1, self._total_frames // self.seq_frames)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _rng(self, item: int) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence([self.seed & 0xFFFFFFFF, int(item) & 0xFFFFFFFF])
        )

    def _view(
        self,
        shard: str,
        dtype: np.dtype | str,
        shape: tuple[int, ...],
        offset: int,
    ) -> np.ndarray:
        return np.ndarray(
            shape=shape,
            dtype=np.dtype(dtype),
            buffer=self._memmaps[shard],
            offset=offset,
        )

    def __getitem__(self, item: int) -> dict[str, Any]:
        if not 0 <= item < len(self):
            raise IndexError(item)
        rng = self._rng(item)
        target = int(rng.integers(0, self._total_frames))
        position = int(np.searchsorted(self._cumulative, target, side="right"))
        position = min(position, len(self.conversations) - 1)
        conversation = self.conversations[position]

        frames = conversation.T
        max_start = max(0, frames - self.seq_frames)
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        take = min(self.seq_frames, frames - start)

        offsets = record_offsets(frames, self.index.n_codebooks)
        base = conversation.offset
        user_view = self._view(
            conversation.shard,
            "<u2",
            (frames, self.index.n_codebooks),
            base + offsets["user"],
        )
        agent_view = self._view(
            conversation.shard,
            "<u2",
            (frames, self.index.n_codebooks),
            base + offsets["agent"],
        )
        text_view = self._view(conversation.shard, "<i4", (frames,), base + offsets["text"])
        control_view = self._view(conversation.shard, "|u1", (frames,), base + offsets["control"])

        user_codes = np.empty((self.seq_frames, self.index.n_codebooks), dtype=np.uint16)
        agent_codes = np.empty_like(user_codes)
        text = np.empty(self.seq_frames, dtype=np.int32)
        control = np.empty(self.seq_frames, dtype=np.uint8)
        valid = np.empty(self.seq_frames, dtype=bool)

        stop = start + take
        user_codes[:take] = user_view[start:stop]
        agent_codes[:take] = agent_view[start:stop]
        text[:take] = text_view[start:stop]
        control[:take] = control_view[start:stop]
        valid[:take] = True
        if take < self.seq_frames:
            user_codes[take:] = self.silence_frame
            agent_codes[take:] = self.silence_frame
            text[take:] = 0
            control[take:] = CONTROL_IDS["PAD"]
            valid[take:] = False

        return {
            "user_codes": user_codes,
            "agent_codes": agent_codes,
            "text": text,
            "control": control,
            "valid": valid,
            "id": conversation.id,
            "start_frame": int(start),
            "duration_s": conversation.duration_s,
        }

    def frame_batches(
        self,
        batch_size: int = 8,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
    ):
        """Iterate one epoch of collated batches."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        from torch.utils.data import DataLoader

        generator = torch.Generator()
        generator.manual_seed(self.seed & 0x7FFFFFFFFFFFFFFF)
        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "collate_fn": collate,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "generator": generator if shuffle else None,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = prefetch_factor
            kwargs["persistent_workers"] = True
        loader = DataLoader(self, **kwargs)
        return iter(loader)


def collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack dataset items and derive per-sequence lengths from ``valid``."""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    lengths = [item["user_codes"].shape[0] for item in batch]
    if len(set(lengths)) != 1:
        raise ValueError(f"all items must share seq_frames, got {sorted(set(lengths))}")

    # Keep the shard dtypes.  Casting the 6 MB code tensor to int64 here cost
    # ~1 ms and bought nothing: DuplexBatchBuilder casts the handful of tensors
    # that need long/float before use.  This keeps a batch at the 5 ms gate.
    channel_stack = np.stack(
        [
            np.stack([item["user_codes"] for item in batch]),
            np.stack([item["agent_codes"] for item in batch]),
        ]
    )
    codes = torch.from_numpy(channel_stack)
    user = codes[0]
    agent = codes[1]
    text = torch.from_numpy(np.stack([item["text"] for item in batch]))
    control = torch.from_numpy(np.stack([item["control"] for item in batch]))
    valid = torch.from_numpy(np.stack([item["valid"] for item in batch]).astype(bool))
    return {
        "user_codes": user,
        "agent_codes": agent,
        "text": text,
        "control": control,
        "valid": valid,
        "lengths": valid.sum(dim=1).to(torch.long),
        "ids": [str(item["id"]) for item in batch],
        "start_frame": torch.tensor(
            [int(item["start_frame"]) for item in batch], dtype=torch.long
        ),
        "duration_s": torch.tensor(
            [float(item["duration_s"]) for item in batch], dtype=torch.float32
        ),
    }


def measure_random_batch_load(
    index: ShardIndex | str | Path,
    n: int = 100,
    *,
    batch_size: int = 8,
    seq_frames: int = 4096,
    split: str = "train",
    seed: int = 0,
    samples_per_epoch: int | None = None,
    warmup: int = 1,
) -> dict[str, Any]:
    """Time random batch construction (dataset reads + collate), in ms.

    One warm-up batch is excluded.  ``passed`` compares the mean against the
    blueprint's 5 ms/batch gate; median, p95 and max are reported too.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not isinstance(index, ShardIndex):
        index = ShardIndex.load(index)
    dataset = DuplexShardDataset(
        index,
        seq_frames=seq_frames,
        split=split,
        samples_per_epoch=samples_per_epoch,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    length = len(dataset)

    def one_batch() -> None:
        positions = rng.integers(0, length, size=batch_size)
        collate([dataset[int(position)] for position in positions])

    for _ in range(max(0, warmup)):
        one_batch()
    times = np.empty(n, dtype=np.float64)
    for step in range(n):
        start = time.perf_counter()
        one_batch()
        times[step] = (time.perf_counter() - start) * 1000.0

    return {
        "batches": int(n),
        "batch_size": int(batch_size),
        "seq_frames": int(seq_frames),
        "split": split,
        "samples": int(len(dataset)),
        "warmup": int(max(0, warmup)),
        "mean_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "max_ms": float(times.max()),
        "min_ms": float(times.min()),
        "gate_ms": 5.0,
        "passed": bool(float(times.mean()) < 5.0),
        "batches_per_second": float(1000.0 / times.mean()) if times.mean() > 0 else math.inf,
    }


__all__ = [
    "ShardInfo",
    "ConversationRecord",
    "ShardIndex",
    "DuplexShardDataset",
    "collate",
    "control_id_for_name",
    "measure_random_batch_load",
]
