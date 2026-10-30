"""Canonical Repartee data schema and 12.5 Hz control-label derivation.

This module is the single source of truth for:

* the conversation manifest JSONL (blueprint section 6),
* the 80 ms frame grid used by both channels,
* the control-token inventory.

Only ``numpy`` is required, so the schema can be validated on a CPU-only host.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Fixed decisions (verified against LiquidAI/LFM3-Omni-2B config.json)
# ---------------------------------------------------------------------------

FRAME_RATE = 12.5
FRAME_SECONDS = 1.0 / FRAME_RATE  # 0.08 s
FRAME_SAMPLES = {
    16000: 1280,
    24000: 1920,
}

# The checkpoint's LiquidCodec has n_q=12 codebooks.  The blueprint's "T,8"
# is an error; see REPORT.md "Deviations".
NUM_CODEBOOKS = 12
CODEBOOK_SIZE = 2017
AUDIO_EOS_CODE = 2016

USER_SAMPLE_RATE = 16000   # LFM3 audio encoder input
AGENT_SAMPLE_RATE = 24000  # LiquidCodec output
CODEC_SAMPLE_RATE = 24000

CONTROL_LABELS = ("PAD", "LISTEN", "SPEAK", "BACKCHANNEL", "STOP")
CONTROL_IDS = {name: index for index, name in enumerate(CONTROL_LABELS)}
CONTROL_TOKEN_NAMES = ("<|listen|>", "<|speak|>", "<|backchannel|>", "<|stop|>")
CONTROL_LABEL_TO_TOKEN_NAME = {
    "LISTEN": "<|listen|>",
    "SPEAK": "<|speak|>",
    "BACKCHANNEL": "<|backchannel|>",
    "STOP": "<|stop|>",
    # PAD intentionally maps to the tokenizer's existing pad token.
}

BACKCHANNEL_MAX_SECONDS = 0.8
STOP_WINDOW_SECONDS = 0.2

_SPEAKERS = ("user", "agent")
_SPLITS = ("train", "valid", "test")


def frame_count(duration_s: float, frame_rate: float = FRAME_RATE) -> int:
    """Number of 80 ms frames needed to cover ``duration_s``."""
    if not math.isfinite(duration_s) or duration_s < 0:
        raise ValueError(f"duration_s must be finite and non-negative, got {duration_s!r}")
    return int(math.ceil(duration_s * frame_rate - 1e-9))


def frame_time(index: int, frame_rate: float = FRAME_RATE) -> float:
    return index / frame_rate


def seconds_to_frames(seconds: float, frame_rate: float = FRAME_RATE) -> int:
    return int(math.floor(seconds * frame_rate + 1e-9))


@dataclass(frozen=True)
class Turn:
    """One contiguous speech interval for one speaker."""

    speaker: Literal["user", "agent"]
    start: float
    end: float
    text: str = ""

    def __post_init__(self) -> None:
        if self.speaker not in _SPEAKERS:
            raise ValueError(f"speaker must be one of {_SPEAKERS}, got {self.speaker!r}")
        for name, value in (("start", self.start), ("end", self.end)):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"turn {name} must be finite, got {value!r}")
        if self.start < 0:
            raise ValueError(f"turn start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(f"turn end must exceed start, got {self.start}..{self.end}")
        if not isinstance(self.text, str):
            raise ValueError("turn text must be a string")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {"speaker": self.speaker, "start": self.start, "end": self.end, "text": self.text}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Turn":
        return cls(
            speaker=value["speaker"],
            start=float(value["start"]),
            end=float(value["end"]),
            text=str(value.get("text", "")),
        )


@dataclass
class Conversation:
    """One dual-channel conversation, matching the manifest JSONL record."""

    id: str
    source: str
    license: str
    duration_s: float
    channels: dict[str, str]
    turns: list[Turn]
    sample_rates: dict[str, int] = field(
        default_factory=lambda: {"user": USER_SAMPLE_RATE, "agent": AGENT_SAMPLE_RATE}
    )
    split: Literal["train", "valid", "test"] = "train"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.turns = [turn if isinstance(turn, Turn) else Turn.from_dict(turn) for turn in self.turns]
        self.validate()

    def validate(self, *, tolerance: float = 1e-3) -> None:
        if not self.id or any(ch.isspace() for ch in self.id):
            raise ValueError(f"id must be non-empty without whitespace, got {self.id!r}")
        if not self.source:
            raise ValueError("source must be non-empty")
        if not self.license:
            raise ValueError("license must be non-empty")
        if not isinstance(self.duration_s, (int, float)) or not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError(f"duration_s must be finite and positive, got {self.duration_s!r}")
        if set(self.channels) != set(_SPEAKERS):
            raise ValueError(f"channels must contain exactly {_SPEAKERS}, got {sorted(self.channels)}")
        for speaker, path in self.channels.items():
            if not isinstance(path, str) or not path:
                raise ValueError(f"channel {speaker!r} path must be non-empty")
        for speaker in _SPEAKERS:
            if self.sample_rates.get(speaker) != {"user": USER_SAMPLE_RATE, "agent": AGENT_SAMPLE_RATE}[speaker]:
                raise ValueError(
                    f"{speaker} channel must be {USER_SAMPLE_RATE if speaker == 'user' else AGENT_SAMPLE_RATE} Hz, "
                    f"got {self.sample_rates.get(speaker)!r}"
                )
        if self.split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}, got {self.split!r}")
        if not self.turns:
            raise ValueError("at least one turn is required")
        previous_start = -math.inf
        for turn in self.turns:
            if turn.start < previous_start - tolerance:
                raise ValueError(f"turns must be sorted by start time; {turn!r} follows {previous_start}")
            if turn.end > self.duration_s + tolerance:
                raise ValueError(f"turn ends at {turn.end}s beyond duration {self.duration_s}s")
            previous_start = turn.start
        # A speaker cannot be in two places at once.  Cross-speaker overlap is the point.
        for speaker in _SPEAKERS:
            spans = sorted((turn.start, turn.end) for turn in self.turns if turn.speaker == speaker)
            for (_, previous_end), (next_start, _) in zip(spans, spans[1:]):
                if next_start < previous_end - tolerance:
                    raise ValueError(f"overlapping {speaker} turns: {previous_end}s vs {next_start}s")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "license": self.license,
            "duration_s": self.duration_s,
            "channels": dict(self.channels),
            "sample_rates": dict(self.sample_rates),
            "split": self.split,
            "turns": [turn.to_dict() for turn in self.turns],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Conversation":
        return cls(
            id=value["id"],
            source=value["source"],
            license=value["license"],
            duration_s=float(value["duration_s"]),
            channels=dict(value["channels"]),
            turns=[Turn.from_dict(turn) for turn in value.get("turns", [])],
            sample_rates=dict(value.get("sample_rates", {"user": USER_SAMPLE_RATE, "agent": AGENT_SAMPLE_RATE})),
            split=value.get("split", "train"),
            meta=dict(value.get("meta", {})),
        )

    def turns_for(self, speaker: str) -> list[Turn]:
        return [turn for turn in self.turns if turn.speaker == speaker]


def active_frames(
    turns: Sequence[Turn],
    duration_s: float,
    *,
    speaker: str,
    frame_rate: float = FRAME_RATE,
) -> np.ndarray:
    """Boolean mask, one entry per 80 ms frame, true when ``speaker`` is active.

    A frame is active when the turn intersects its half-open interval
    ``[i / frame_rate, (i + 1) / frame_rate)``.  Touching boundaries are not
    overlap: a turn ending exactly at 1.0 s does not activate frame 12.5.
    """
    if speaker not in _SPEAKERS:
        raise ValueError(f"unknown speaker {speaker!r}")
    count = frame_count(duration_s, frame_rate)
    mask = np.zeros(count, dtype=bool)
    for turn in turns:
        if turn.speaker != speaker:
            continue
        start = max(0, int(math.floor(turn.start * frame_rate + 1e-9)))
        end = min(count, int(math.ceil(turn.end * frame_rate - 1e-9)))
        if end > start:
            mask[start:end] = True
    return mask


def derive_control_labels(
    turns: Sequence[Turn] | Conversation,
    duration_s: float | None = None,
    *,
    frame_rate: float = FRAME_RATE,
    backchannel_max_s: float = BACKCHANNEL_MAX_SECONDS,
    stop_window_s: float = STOP_WINDOW_SECONDS,
) -> np.ndarray:
    """Derive the per-frame control label (uint8) from turn timings.

    Precedence, highest first: ``STOP`` > ``BACKCHANNEL`` > ``SPEAK`` >
    ``LISTEN`` > ``PAD``.

    * ``LISTEN``  - agent silent, user active.
    * ``SPEAK``   - agent active (onset and continuation).
    * ``BACKCHANNEL`` - an agent turn shorter than 0.8 s that overlaps user
      speech is a backchannel, not a turn claim.
    * ``STOP`` - each user speech onset that happens while the agent is
      active opens a 200 ms stop window.  The label tells the policy to drop
      queued audio at barge-in.
    * ``PAD`` - silence on both channels.
    """
    if isinstance(turns, Conversation):
        duration_s = turns.duration_s
        turns = turns.turns
    if duration_s is None:
        raise ValueError("duration_s is required when passing turns")
    if backchannel_max_s <= 0 or stop_window_s <= 0:
        raise ValueError("backchannel_max_s and stop_window_s must be positive")

    count = frame_count(duration_s, frame_rate)
    user = active_frames(turns, duration_s, speaker="user", frame_rate=frame_rate)
    agent = active_frames(turns, duration_s, speaker="agent", frame_rate=frame_rate)
    labels = np.full(count, CONTROL_IDS["PAD"], dtype=np.uint8)

    labels[user & ~agent] = CONTROL_IDS["LISTEN"]
    labels[agent] = CONTROL_IDS["SPEAK"]

    def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
        """True only for positive-duration overlap; touching edges do not count."""
        return min(a_end, b_end) - max(a_start, b_start) > 1e-9

    # Backchannels: short agent utterances that genuinely overlap user speech.
    for turn in turns:
        if turn.speaker != "agent" or turn.duration >= backchannel_max_s:
            continue
        if not any(
            overlaps(turn.start, turn.end, other.start, other.end)
            for other in turns
            if other.speaker == "user"
        ):
            continue
        start = max(0, int(math.floor(turn.start * frame_rate + 1e-9)))
        end = min(count, int(math.ceil(turn.end * frame_rate - 1e-9)))
        if end > start:
            labels[start:end] = CONTROL_IDS["BACKCHANNEL"]

    # Barge-in: a user onset strictly inside an agent utterance opens a stop
    # window.  Simultaneous onset is a backchannel/talk-over decision, and a
    # user turn that merely touches the agent's end is a clean handoff.
    stop_frames = max(1, int(math.ceil(stop_window_s * frame_rate - 1e-9)))
    for turn in turns:
        if turn.speaker != "user":
            continue
        onset = max(0, int(math.floor(turn.start * frame_rate + 1e-9)))
        if onset >= count:
            continue
        interrupting = any(
            other.start + 1e-9 < turn.start < other.end - 1e-9
            for other in turns
            if other.speaker == "agent"
        )
        if interrupting:
            labels[onset : min(count, onset + stop_frames)] = CONTROL_IDS["STOP"]

    return labels


def control_histogram(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(np.asarray(labels, dtype=np.uint8), return_counts=True)
    histogram = {name: 0 for name in CONTROL_LABELS}
    for value, count in zip(values.tolist(), counts.tolist()):
        if value >= len(CONTROL_LABELS):
            raise ValueError(f"invalid control label {value}")
        histogram[CONTROL_LABELS[value]] = int(count)
    return histogram


def align_text_to_frames(
    turns: Sequence[Turn],
    duration_s: float,
    tokenizer: Any,
    *,
    frame_rate: float = FRAME_RATE,
) -> np.ndarray:
    """Coarse per-frame agent text target.

    Source manifests give turn-level transcripts, not word timings.  Tokens
    for an agent turn are spread uniformly over the turn's frame span; the
    first token that lands on a frame wins.  This is an explicitly coarse
    alignment, not forced alignment.  ``0`` means "no text token here".
    """
    count = frame_count(duration_s, frame_rate)
    out = np.zeros(count, dtype=np.int32)
    for turn in turns:
        if turn.speaker != "agent" or not turn.text.strip():
            continue
        ids = tokenizer.encode(turn.text, add_special_tokens=False)
        if not ids:
            continue
        start = max(0, int(math.floor(turn.start * frame_rate + 1e-9)))
        end = min(count, int(math.ceil(turn.end * frame_rate - 1e-9)))
        span = max(1, end - start)
        if len(ids) > span:
            ids = ids[:span]
        for index, token_id in enumerate(ids):
            frame = start + int(index * span / len(ids))
            frame = min(frame, end - 1)
            if frame >= 0 and out[frame] == 0:
                out[frame] = int(token_id)
    return out


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def read_manifest(path: str | Path) -> Iterator[Conversation]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Conversation.from_dict(json.loads(line))
            except Exception as exc:  # noqa: BLE001 - add location to the parse error
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


def write_manifest(path: str | Path, conversations: Iterable[Conversation]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for conversation in conversations:
            conversation.validate()
            handle.write(json.dumps(conversation.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def summarize_manifest(path: str | Path) -> dict[str, Any]:
    """Hours, sources, licenses and split counts for the license audit."""
    hours: dict[str, float] = {}
    licenses: dict[str, float] = {}
    splits: dict[str, int] = {}
    total = 0.0
    conversations = 0
    for conversation in read_manifest(path):
        hours[conversation.source] = hours.get(conversation.source, 0.0) + conversation.duration_s / 3600
        licenses[conversation.license] = licenses.get(conversation.license, 0.0) + conversation.duration_s / 3600
        splits[conversation.split] = splits.get(conversation.split, 0) + 1
        total += conversation.duration_s / 3600
        conversations += 1
    return {
        "conversations": conversations,
        "hours": total,
        "hours_by_source": hours,
        "hours_by_license": licenses,
        "splits": splits,
    }


__all__ = [
    "FRAME_RATE",
    "FRAME_SECONDS",
    "FRAME_SAMPLES",
    "NUM_CODEBOOKS",
    "CODEBOOK_SIZE",
    "AUDIO_EOS_CODE",
    "USER_SAMPLE_RATE",
    "AGENT_SAMPLE_RATE",
    "CODEC_SAMPLE_RATE",
    "CONTROL_LABELS",
    "CONTROL_IDS",
    "CONTROL_TOKEN_NAMES",
    "CONTROL_LABEL_TO_TOKEN_NAME",
    "BACKCHANNEL_MAX_SECONDS",
    "STOP_WINDOW_SECONDS",
    "Turn",
    "Conversation",
    "frame_count",
    "frame_time",
    "seconds_to_frames",
    "active_frames",
    "derive_control_labels",
    "control_histogram",
    "align_text_to_frames",
    "read_manifest",
    "write_manifest",
    "summarize_manifest",
]
