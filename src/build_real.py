"""Build Repartee conversation manifests from real dual-channel corpora.

Sources and verified layouts (2026-10-29)
-----------------------------------------
Open Yap 1K (``TheAgenticDataCompany/open-yap-1k``)
    The public artefact is a WebDataset tar.  Members are
    ``<conv_id>.preview.mp3``, ``<conv_id>.a.flac``, ``<conv_id>.b.flac`` and
    ``<conv_id>.json`` (verified by walking the tar header of
    ``open-yap-1k-sample.tar``).  Sample audio is mono 48 kHz FLAC; ``a`` is
    mapped to ``user`` and ``b`` to ``agent``.  ``json["transcripts"][ch]``
    holds ``{"text", "words": [{"word", "start", "end", "type"}]}`` with
    word-level timings and event types (``word``, ``filler``, ...).  There are
    no utterance boundaries, so turns are reconstructed from word gaps
    (:func:`normalize_turns`).  The sample tar is public; the 1,000 h corpus is
    delivered on request under the Open Yap 1K Data Use Agreement
    (https://theagenticdatacompany.com/open-yap-1k).  The public
    ``shard-*.tar`` files on the Hub are also WebDataset shards.

otoSpeech full-duplex 280 h (``otoearth/otoSpeech-full-duplex-280h``)
    Gated (``auto``), CC-BY-4.0, WebDataset tar shards under
    ``data/train/shard-*.tar``.  Each sample has ``flac`` (48 kHz **stereo**,
    channel 0 = speaker A, channel 1 = B) and ``json`` session metadata
    (``session_id``, ``duration``, ``speakers[].channel/speaker_id``,
    ``redacted_segments``, ...).  This layout is taken verbatim from the
    dataset card; the shard payload itself is **UNVERIFIED here** because the
    repository is gated.  The card documents **no transcript**, so turns are
    segmented with a deterministic energy VAD and ``text`` is empty.  This is
    a documented limitation, not fabricated ASR.

otoSpeech full-duplex turn-104 h (``otoearth/otoSpeech-full-duplex-turn-104h``)
    Manually gated, **non-commercial** oto Speech License v1.0 (the dataset
    card tags it ``other``; the card explicitly says production/commercial use
    requires a separate written license).  One directory per conversation:
    ``speaker_1_audio.wav`` / ``speaker_2_audio.wav`` (mono 48 kHz float),
    ``combined_audio.wav``, ``speaker_{1,2}_annotation_a.srt`` and
    ``metadata.json``.  SRT cues are ``HH:MM:SS,mmm --> HH:MM:SS,mmm`` followed
    by ``[Event Label] transcript``.  This layout is from the dataset card and
    is **UNVERIFIED on the gated payload**.  ``speaker_1`` maps to ``user`` and
    ``speaker_2`` to ``agent``.  The actual license string is recorded as
    ``oto-speech-license-v1.0``; ``meta["commercial_use"]`` is ``False`` and
    the license audit will flag it.

Normalisation
-------------
Every kept conversation is validated with :class:`src.schema.Conversation`,
trimmed to an integer number of 80 ms frames, and written as two FLACs:
user mono 16 kHz, agent mono 24 kHz.  Conversations with a missing channel,
more than 20 % clipped samples on either channel, or no turns are *dropped*
and counted (these are expected quality filters).  Malformed JSON, SRT or
audio raises :class:`ManifestError` naming the tar/file and record id; records
are never skipped silently.

Held-out speakers/conversations
-------------------------------
Fifty distinct speaker ids and 100 conversation ids are selected by a salted
BLAKE2b hash over the whole build (``select_holdout``).  Any record touching a
held-out speaker or conversation is ``split="test"`` and must never be
trained on (``assign_split``).  The remainder is ``train``.

CLI
---
``python -m src.build_real --source open_yap|otospeech|all --out data/manifests
--raw data/raw [--limit N] [--sample-only] [--overwrite] [--input-tar TAR]``

Local copies (no Hub token needed) are discovered under
``<raw>/open_yap/input/*.tar`` (or ``--input-tar``),
``<raw>/otospeech_280h/input/*.tar`` and
``<raw>/otospeech_turn104h/input/<task_id>/``.

Limitations
-----------
* Full Open Yap access is a manual data-use agreement; the CLI raises
  :class:`AccessRequired` with the URL when no licensed local tar is present.
* otoSpeech downloads are intentionally shard-at-a-time; ``--limit`` cannot
  avoid downloading the first tar shard.
* The actual repositories use WebDataset tar (280 h) and audiofolder
  directories (turn-104 h); neither publishes a parquet layout, so no
  speculative parquet column mapping is implemented.
* Energy VAD for the 280 h corpus is a heuristic (no reference transcript);
  thresholds are deterministic but not validated against human annotations.
* ``--sample-only`` only affects Open Yap; the otoSpeech datasets have no
  public sample.  The public Open Yap sample has only 8 speakers and 16
  conversations, so selecting 50 speakers / 100 conversations holds out the
  entire sample (``split="test"``).  That is intentional: the sample is for
  inspection, never training.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from . import schema

# ---------------------------------------------------------------------------
# Constants (verified from the repositories; see module docstring)
# ---------------------------------------------------------------------------

OPEN_YAP_REPO = "TheAgenticDataCompany/open-yap-1k"
OPEN_YAP_ACCESS_URL = "https://theagenticdatacompany.com/open-yap-1k"
OPEN_YAP_SAMPLE_URL = (
    "https://huggingface.co/datasets/TheAgenticDataCompany/open-yap-1k/"
    "resolve/main/open-yap-1k-sample.tar"
)
OTO_280_REPO = "otoearth/otoSpeech-full-duplex-280h"
OTO_104_REPO = "otoearth/otoSpeech-full-duplex-turn-104h"

SOURCE_OPEN_YAP = "open_yap_1k"
SOURCE_OTO_280 = "otospeech_280h"
SOURCE_OTO_104 = "otospeech_turn104h"

LICENSE_OPEN_YAP = "CC-BY-4.0"
LICENSE_OTO_280 = "CC-BY-4.0"
# The card tags the repository "other"; the actual licence is the oto Speech
# License v1.0, a non-commercial research licence.
LICENSE_OTO_104 = "oto-speech-license-v1.0"

LICENSE_URLS = {
    SOURCE_OPEN_YAP: "https://huggingface.co/datasets/TheAgenticDataCompany/open-yap-1k/blob/main/LICENSE.txt",
    SOURCE_OTO_280: "https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-280h",
    SOURCE_OTO_104: "https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-turn-104h/blob/main/LICENSE.md",
}

SOURCE_FILES = {
    SOURCE_OPEN_YAP: "open_yap_1k.jsonl",
    SOURCE_OTO_280: "otospeech_280h.jsonl",
    SOURCE_OTO_104: "otospeech_turn104h.jsonl",
}
COMBINED_FILE = "repartee-mix.jsonl"

MAX_CLIPPING_RATIO = 0.20
HOLDOUT_SPEAKERS = 50
HOLDOUT_CONVERSATIONS = 100
HOLDOUT_SPEAKER_SALT = "repartee/speaker-holdout/v1"
HOLDOUT_CONVERSATION_SALT = "repartee/conversation-holdout/v1"

_NON_LEXICAL_KINDS = {
    "laugh",
    "laughter",
    "cough",
    "noise",
    "non-speech",
    "non_speech",
    "non_speech_noise",
    "breath",
    "sigh",
}


class BuildError(RuntimeError):
    """Base error for data builds."""


class InstallError(BuildError):
    """A required optional dependency is missing."""


class ManifestError(BuildError):
    """A source record is malformed and cannot be converted."""


class AccessRequired(BuildError):
    """A gated/request-only dataset is not available locally."""

    def __init__(self, *, dataset: str, url: str, instructions: str) -> None:
        self.dataset = dataset
        self.url = url
        self.instructions = instructions
        super().__init__(f"{dataset} requires access at {url}. {instructions}")


class QualityDrop(Exception):
    """An expected quality filter rejected a record (not a build failure)."""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def stable_hash(value: str, *, salt: str = "") -> str:
    """Deterministic hex digest used for holdouts and sampling."""
    digest = hashlib.blake2b(f"{salt}\x1f{value}".encode("utf-8"), digest_size=16)
    return digest.hexdigest()


def select_holdout(
    keys: Iterable[str],
    count: int,
    *,
    salt: str,
) -> set[str]:
    """Select exactly ``min(count, len(unique keys))`` keys by salted hash.

    Sorting on the digest (with the key as tie-breaker) makes the selection
    independent of input order while still being a uniform sample.
    """
    if count < 0:
        raise ValueError("count must be >= 0")
    unique = {str(key) for key in keys}
    ordered = sorted(unique, key=lambda key: (stable_hash(key, salt=salt), key))
    return set(ordered[:count])


def assign_split(
    conversation_id: str,
    speaker_ids: Iterable[str] = (),
    *,
    holdout_speakers: Iterable[str] = (),
    holdout_conversations: Iterable[str] = (),
) -> str:
    """Return ``"test"`` when any speaker or the conversation is held out.

    The test split is a *speaker* holdout: every recording that contains a
    held-out speaker is excluded from training, even if that speaker appears
    under a different conversation id.  Otherwise the record is ``"train"``.
    """
    conversation_id = str(conversation_id)
    speakers = {str(speaker) for speaker in speaker_ids}
    held_speakers = {str(speaker) for speaker in holdout_speakers}
    held_conversations = {str(cid) for cid in holdout_conversations}
    if conversation_id in held_conversations or speakers & held_speakers:
        return "test"
    return "train"


def passes_quality(
    user_audio: np.ndarray | None = None,
    agent_audio: np.ndarray | None = None,
    sample_rate: int = 16000,
    *,
    max_clipping_ratio: float = MAX_CLIPPING_RATIO,
    min_duration_s: float = 0.8,
) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for the mandatory real-data quality filters.

    Missing channels, empty/non-finite audio, more than ``max_clipping_ratio``
    clipped samples on either channel, and recordings shorter than
    ``min_duration_s`` are rejected.
    """
    if user_audio is None or agent_audio is None:
        return False, "missing channel"
    user = np.asarray(user_audio, dtype=np.float32).reshape(-1)
    agent = np.asarray(agent_audio, dtype=np.float32).reshape(-1)
    if user.size == 0 or agent.size == 0:
        return False, "empty channel"
    if not np.isfinite(user).all() or not np.isfinite(agent).all():
        return False, "non-finite audio"
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    duration_s = min(user.size, agent.size) / float(sample_rate)
    if duration_s < min_duration_s:
        return False, f"too short ({duration_s:.3f}s < {min_duration_s:.3f}s)"
    for name, channel in (("user", user), ("agent", agent)):
        clipped = float(np.mean(np.abs(channel) >= 0.999))
        if clipped > max_clipping_ratio:
            return False, f"{name} clipping {clipped:.1%} > {max_clipping_ratio:.0%}"
    return True, "ok"


def _word_get(word: Any, key: str, default: Any = None) -> Any:
    if isinstance(word, Mapping):
        return word.get(key, default)
    return getattr(word, key, default)


def _sanitize_word_timings(
    words: Sequence[Any], *, max_drop_ratio: float = 0.05
) -> tuple[list[Any], int]:
    """Drop individual words with corrupt timings from a real ASR transcript.

    ASR word streams can contain a locally corrupt interval (non-finite, a
    negative start, or ``end <= start``).  That is a word-level defect, not a
    malformed conversation, so a small number are dropped; losing more than
    ``max_drop_ratio`` of a channel's words means the transcript is unusable
    and the record fails.
    """
    kept: list[Any] = []
    dropped = 0
    for word in words:
        start = _word_get(word, "start")
        end = _word_get(word, "end")
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if (
            not math.isfinite(start_f)
            or not math.isfinite(end_f)
            or start_f < 0
            or end_f <= start_f
        ):
            dropped += 1
            continue
        kept.append(word)
    total = len(words)
    if total and dropped / total > max_drop_ratio:
        raise ValueError(
            f"{dropped}/{total} words have unusable timings "
            f"(> {max_drop_ratio:.0%}); transcript is malformed"
        )
    return kept, dropped


def normalize_turns(
    words: Sequence[Any],
    speaker: str,
    *,
    max_gap_s: float = 0.6,
) -> list[schema.Turn]:
    """Group word-level timings into contiguous speaker turns.

    ``words`` accepts mappings or objects with ``word``/``text``, ``start``,
    ``end`` and optional ``type``.  A new turn starts whenever the silence
    between consecutive words exceeds ``max_gap_s``.  Non-lexical event types
    (laughter, cough, noise, ...) are rendered as ``[laugh]``-style text.
    Same-speaker overlaps are merged.
    """
    if speaker not in ("user", "agent"):
        raise ValueError(f"unknown speaker {speaker!r}")
    if max_gap_s < 0:
        raise ValueError("max_gap_s must be >= 0")
    prepared: list[tuple[float, float, str]] = []
    for index, word in enumerate(words):
        text = _word_get(word, "word", _word_get(word, "text", ""))
        start = _word_get(word, "start")
        end = _word_get(word, "end")
        kind = str(_word_get(word, "type", "word") or "word")
        if text is None:
            text = ""
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"word {index} has non-numeric timing: {start!r}..{end!r}") from exc
        if not math.isfinite(start_f) or not math.isfinite(end_f):
            raise ValueError(f"word {index} has non-finite timing: {start_f}..{end_f}")
        if start_f < 0 or end_f <= start_f:
            raise ValueError(f"word {index} has invalid interval: {start_f}..{end_f}")
        token = str(text).strip()
        if kind.lower() in _NON_LEXICAL_KINDS:
            token = f"[{kind.lower()}]"
        elif not token:
            continue
        prepared.append((start_f, end_f, token))

    if not prepared:
        return []
    prepared.sort(key=lambda item: (item[0], item[1]))

    turns: list[schema.Turn] = []
    current_start, current_end = prepared[0][0], prepared[0][1]
    current_tokens: list[str] = [prepared[0][2]]
    for start, end, token in prepared[1:]:
        gap = start - current_end
        if gap <= max_gap_s:
            current_end = max(current_end, end)
            current_tokens.append(token)
            continue
        turns.append(
            schema.Turn(
                speaker=speaker,
                start=round(current_start, 3),
                end=round(current_end, 3),
                text=" ".join(current_tokens),
            )
        )
        current_start, current_end = start, end
        current_tokens = [token]
    turns.append(
        schema.Turn(
            speaker=speaker,
            start=round(current_start, 3),
            end=round(current_end, 3),
            text=" ".join(current_tokens),
        )
    )
    return merge_same_speaker_turns(turns)


def merge_same_speaker_turns(turns: Sequence[schema.Turn]) -> list[schema.Turn]:
    """Sort turns and merge overlapping intervals from the same speaker.

    The schema forbids a speaker occupying two places at once.  Source
    transcripts occasionally contain overlapping ASR segments; merging them is
    the only lossless-enough repair (silently dropping one would lose text).
    """
    ordered = sorted(turns, key=lambda turn: (turn.start, turn.end, turn.speaker))
    merged: list[schema.Turn] = []
    for turn in ordered:
        if merged and turn.speaker == merged[-1].speaker and turn.start < merged[-1].end - 1e-9:
            previous = merged[-1]
            text = " ".join(part for part in (previous.text, turn.text) if part)
            merged[-1] = schema.Turn(
                speaker=previous.speaker,
                start=previous.start,
                end=max(previous.end, turn.end),
                text=text,
            )
        else:
            merged.append(turn)
    return merged


@dataclass(frozen=True)
class SrtCue:
    index: int
    start: float
    end: float
    label: str
    text: str


_SRT_TIME = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{3})\s*-->\s*"
    r"(?P<h2>\d{1,2}):(?P<m2>\d{2}):(?P<s2>\d{2})[,.](?P<ms2>\d{3})"
)
_SRT_LABEL = re.compile(r"^\[(?P<label>[^\]]+)\]\s*(?P<text>.*)$")


def _srt_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(text: str) -> list[SrtCue]:
    """Parse an otoSpeech-style SRT file with ``[Event Label] text`` cues."""
    normalised = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    cues: list[SrtCue] = []
    for block_index, block in enumerate(re.split(r"\n\s*\n", normalised.strip())):
        lines = [line.rstrip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        time_index = next((i for i, line in enumerate(lines) if _SRT_TIME.search(line)), None)
        if time_index is None:
            raise ValueError(f"SRT block {block_index + 1} has no timestamp line: {lines!r}")
        match = _SRT_TIME.search(lines[time_index])
        assert match is not None
        start = _srt_seconds(match["h"], match["m"], match["s"], match["ms"])
        end = _srt_seconds(match["h2"], match["m2"], match["s2"], match["ms2"])
        if end <= start:
            raise ValueError(f"SRT block {block_index + 1} has non-positive duration {start}..{end}")
        body = " ".join(lines[time_index + 1 :]).strip()
        label = ""
        label_match = _SRT_LABEL.match(body)
        if label_match:
            label = label_match["label"].strip()
            body = label_match["text"].strip()
        cues.append(
            SrtCue(
                index=int(lines[0]) if lines[0].isdigit() else block_index + 1,
                start=start,
                end=end,
                label=label,
                text=body,
            )
        )
    return cues


def srt_to_turns(
    cues: Sequence[SrtCue],
    *,
    speaker: str,
    offset_s: float = 0.0,
) -> list[schema.Turn]:
    """Convert SRT cues to schema turns, preserving event labels in ``meta``? No.

    Labels are timing annotations, not transcripts, and are therefore not
    concatenated into the text.  Callers can count them separately.
    """
    turns = [
        schema.Turn(
            speaker=speaker,
            start=round(cue.start + offset_s, 3),
            end=round(cue.end + offset_s, 3),
            text=cue.text,
        )
        for cue in cues
        if cue.end > cue.start
    ]
    return merge_same_speaker_turns(turns)


def energy_turns(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_s: float = 0.02,
    threshold_db: float | None = None,
    min_speech_s: float = 0.20,
    min_silence_s: float = 0.30,
    pad_s: float = 0.05,
) -> list[tuple[float, float]]:
    """Deterministic energy-VAD speech intervals for one channel.

    Used only for the otoSpeech 280 h corpus, whose card documents no
    transcript.  The threshold defaults to the 10th-percentile frame RMS plus
    10 dB, clipped to ``[-55, -20]`` dBFS.  Gaps shorter than
    ``min_silence_s`` are bridged and runs shorter than ``min_speech_s`` are
    discarded.  This is a heuristic; see the module limitations.
    """
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if mono.size == 0:
        return []
    frame_len = max(1, int(round(frame_s * sample_rate)))
    usable = (mono.size // frame_len) * frame_len
    if usable == 0:
        return []
    frames = mono[:usable].reshape(-1, frame_len).astype(np.float64)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    if threshold_db is None:
        noise_floor = float(np.percentile(db, 10))
        speech_level = float(np.percentile(db, 90))
        threshold_db = noise_floor + 10.0
        if speech_level - threshold_db < 10.0:
            # Low-dynamic-range channel (e.g. continuous speech): keep the
            # detector sensitive instead of refusing all activity.
            threshold_db = speech_level - 10.0
        threshold_db = min(-20.0, max(-55.0, threshold_db))
    active = db > float(threshold_db)
    if not active.any():
        return []

    intervals: list[list[float]] = []
    start_index: int | None = None
    for index, is_active in enumerate(active):
        if is_active and start_index is None:
            start_index = index
        elif not is_active and start_index is not None:
            intervals.append([start_index * frame_s, index * frame_s])
            start_index = None
    if start_index is not None:
        intervals.append([start_index * frame_s, len(active) * frame_s])

    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start - merged[-1][1] < min_silence_s:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    total = mono.size / float(sample_rate)
    out: list[tuple[float, float]] = []
    for start, end in merged:
        if end - start < min_speech_s:
            continue
        padded_start = max(0.0, start - pad_s)
        padded_end = min(total, end + pad_s)
        out.append((round(padded_start, 3), round(padded_end, 3)))
    return out


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "record"


def _grid_frames(duration_s: float) -> int:
    """Largest number of complete 80 ms frames that fits in ``duration_s``."""
    frames = int(math.floor(duration_s * schema.FRAME_RATE + 1e-9))
    if frames < 1:
        raise QualityDrop(f"shorter than one 80 ms frame ({duration_s:.4f}s)")
    return frames


def _clamp_turns(turns: Sequence[schema.Turn], duration_s: float) -> list[schema.Turn]:
    clamped: list[schema.Turn] = []
    for turn in turns:
        start = max(0.0, min(float(turn.start), duration_s))
        end = max(start, min(float(turn.end), duration_s))
        if end - start < 1e-6:
            continue
        clamped.append(schema.Turn(turn.speaker, round(start, 3), round(end, 3), turn.text))
    return merge_same_speaker_turns(clamped)


# ---------------------------------------------------------------------------
# Tar/source record iteration
# ---------------------------------------------------------------------------


def classify_open_yap_member(name: str) -> tuple[str, str] | None:
    """Map an Open Yap tar member to ``(conversation_key, role)``.

    Supports both the released naming (``<key>.a.flac``) and WebDataset
    keyed layout (``<key>/a.flac``).  Preview MP3s and unknown members return
    ``None``.
    """
    path = PurePosixPath(name)
    filename = path.name
    for suffix, role in ((".a.flac", "user"), (".b.flac", "agent"), (".json", "meta")):
        if filename.endswith(suffix) and len(filename) > len(suffix):
            return filename[: -len(suffix)], role
    parent = path.parent.as_posix()
    if filename in ("a.flac", "b.flac", "json"):
        role = {"a.flac": "user", "b.flac": "agent", "json": "meta"}[filename]
        key = "" if parent in ("", ".") else parent
        return key, role
    return None


def classify_otospeech_member(name: str) -> tuple[str, str] | None:
    """Map an otoSpeech WebDataset member to ``(session_key, role)``."""
    path = PurePosixPath(name)
    filename = path.name
    parent = path.parent.as_posix()
    if filename in ("flac", "json"):
        role = "meta" if filename == "json" else "audio"
        key = "" if parent in ("", ".") else parent
        return key, role
    if filename.endswith(".flac") or filename.endswith(".json"):
        suffix = ".flac" if filename.endswith(".flac") else ".json"
        role = "meta" if suffix == ".json" else "audio"
        return filename[: -len(suffix)], role
    return None


def _iter_tar_members(handle: tarfile.TarFile, tar_path: Path) -> Iterator[tarfile.TarInfo]:
    """Iterate tar members, translating stream corruption into ManifestError."""
    try:
        yield from handle
    except (tarfile.TarError, OSError) as exc:
        raise ManifestError(f"{tar_path}: tar read failed: {exc}") from exc


def _iter_tar_groups(
    tar_paths: Sequence[Path],
    classify: Callable[[str], tuple[str, str] | None],
) -> Iterator[tuple[Path, str, dict[str, bytes]]]:
    """Yield ``(tar_path, key, role_bytes)`` groups from one or more tars.

    A group is emitted as soon as it contains metadata plus both audio roles,
    or when the tar ends / a new bare-name sample starts.  Incomplete groups
    are emitted too, so the caller can count and report missing channels.
    """
    for tar_path in tar_paths:
        try:
            handle = tarfile.open(tar_path, mode="r|*")
        except (tarfile.TarError, OSError) as exc:
            raise ManifestError(f"{tar_path}: cannot open tar: {exc}") from exc
        with handle:
            pending: dict[str, dict[str, bytes]] = {}
            order: list[str] = []

            def flush(key: str) -> tuple[Path, str, dict[str, bytes]] | None:
                members = pending.pop(key, None)
                if members is None:
                    return None
                if key in order:
                    order.remove(key)
                return tar_path, key, members

            for member in _iter_tar_members(handle, tar_path):
                if not member.isfile():
                    continue
                classified = classify(member.name)
                if classified is None:
                    continue
                key, role = classified
                try:
                    extracted = handle.extractfile(member)
                    if extracted is None:
                        raise ManifestError(f"{tar_path}:{member.name}: not a regular file")
                    payload = extracted.read()
                except (tarfile.TarError, OSError) as exc:
                    raise ManifestError(f"{tar_path}:{member.name}: read failed: {exc}") from exc
                if key not in pending:
                    pending[key] = {}
                    order.append(key)
                pending[key][role] = payload
                if "meta" in pending[key] and "user" in pending[key] and "agent" in pending[key]:
                    item = flush(key)
                    if item:
                        yield item
                elif key == "" and "meta" in pending[key]:
                    # Bare WebDataset names are reused per sample; emit as
                    # soon as the metadata member closes the sample.
                    item = flush(key)
                    if item:
                        yield item
            for key in list(order):
                item = flush(key)
                if item:
                    yield item


# ---------------------------------------------------------------------------
# Audio helpers (heavy dependencies imported lazily)
# ---------------------------------------------------------------------------


def _read_audio_bytes(payload: bytes, *, source: str) -> tuple[np.ndarray, int, int]:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - dependency is in the venv
        raise InstallError("soundfile is required to decode source audio: pip install soundfile") from exc
    try:
        with sf.SoundFile(io.BytesIO(payload)) as handle:
            sample_rate = int(handle.samplerate)
            channels = int(handle.channels)
            frames = int(handle.frames)
            data = handle.read(dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises various LibsndfileError subclasses
        raise ManifestError(f"{source}: cannot decode audio: {exc}") from exc
    if frames == 0:
        raise ManifestError(f"{source}: audio has no samples")
    return np.asarray(data, dtype=np.float32), sample_rate, channels


def _write_channels(
    *,
    raw_root: Path,
    source_dir: str,
    conversation_id: str,
    user_audio: np.ndarray,
    agent_audio: np.ndarray,
    source_rate: int,
    frames: int,
    overwrite: bool,
) -> dict[str, str]:
    from . import audio as audio_utils

    out_dir = raw_root / source_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_name(conversation_id)
    user_path = out_dir / f"{safe}_user.flac"
    agent_path = out_dir / f"{safe}_agent.flac"
    if overwrite or not user_path.exists():
        user = audio_utils.resample(user_audio, source_rate, schema.USER_SAMPLE_RATE)
        user = audio_utils.fit_frames(user, frames, schema.USER_SAMPLE_RATE)
        audio_utils.write_audio(user_path, user, schema.USER_SAMPLE_RATE)
    if overwrite or not agent_path.exists():
        agent = audio_utils.resample(agent_audio, source_rate, schema.AGENT_SAMPLE_RATE)
        agent = audio_utils.fit_frames(agent, frames, schema.AGENT_SAMPLE_RATE)
        audio_utils.write_audio(agent_path, agent, schema.AGENT_SAMPLE_RATE)
    return {"user": str(user_path), "agent": str(agent_path)}


def _make_conversation(
    *,
    conversation_id: str,
    source: str,
    license_name: str,
    user_audio: np.ndarray,
    agent_audio: np.ndarray,
    source_rate: int,
    turns: Sequence[schema.Turn],
    meta: dict[str, Any],
    raw_root: Path,
    source_dir: str,
    overwrite: bool,
    context: str = "",
) -> schema.Conversation:
    """Validate, frame-align, write audio and return a schema conversation."""
    ok, reason = passes_quality(
        user_audio=user_audio,
        agent_audio=agent_audio,
        sample_rate=source_rate,
    )
    if not ok:
        raise QualityDrop(reason)
    min_duration = min(user_audio.size, agent_audio.size) / float(source_rate)
    frames = _grid_frames(min_duration)
    duration_s = frames / schema.FRAME_RATE
    clipped_turns = _clamp_turns(turns, duration_s)
    if not clipped_turns:
        raise QualityDrop("no turns after frame alignment")
    channels = _write_channels(
        raw_root=raw_root,
        source_dir=source_dir,
        conversation_id=conversation_id,
        user_audio=user_audio,
        agent_audio=agent_audio,
        source_rate=source_rate,
        frames=frames,
        overwrite=overwrite,
    )
    label = context or conversation_id
    meta_out = dict(meta)
    meta_out.setdefault("license_url", LICENSE_URLS.get(source, ""))
    meta_out.setdefault("frames", frames)
    meta_out.setdefault("source_sample_rate", source_rate)
    meta_out.setdefault("commercial_use", source != SOURCE_OTO_104)
    try:
        conversation = schema.Conversation(
            id=conversation_id,
            source=source,
            license=license_name,
            duration_s=round(duration_s, 6),
            channels=channels,
            turns=list(clipped_turns),
            split="train",
            meta=meta_out,
        )
        conversation.validate()
    except Exception as exc:
        raise ManifestError(f"{label}: schema validation failed: {exc}") from exc
    return conversation


# ---------------------------------------------------------------------------
# Open Yap
# ---------------------------------------------------------------------------


def _download_public(url: str, destination: Path) -> Path:
    """Resume-capable download for the public Open Yap sample tar."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    part = destination.with_suffix(destination.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "repartee-build-real/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if offset and getattr(response, "status", 200) != 206:
                # Server ignored Range: restart rather than corrupt the part.
                offset = 0
            with part.open("ab" if offset else "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
    except urllib.error.HTTPError as exc:
        raise BuildError(f"failed to download {url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise BuildError(f"failed to download {url}: {exc.reason}") from exc
    part.replace(destination)
    return destination


def _open_yap_tars(
    *,
    raw_root: Path,
    input_tar: Path | None,
    sample_only: bool,
) -> list[Path]:
    if input_tar is not None:
        if not input_tar.exists():
            raise BuildError(f"--input-tar does not exist: {input_tar}")
        return [input_tar]
    local_inputs = sorted((raw_root / "open_yap" / "input").glob("*.tar"))
    local_inputs += sorted((raw_root / "open_yap").glob("*.tar"))
    if sample_only:
        if local_inputs:
            return local_inputs
        destination = raw_root / "_downloads" / "open-yap-1k-sample.tar"
        return [_download_public(OPEN_YAP_SAMPLE_URL, destination)]
    if local_inputs:
        return local_inputs
    raise AccessRequired(
        dataset="Open Yap 1K",
        url=OPEN_YAP_ACCESS_URL,
        instructions=(
            "The full 1,000 h corpus is delivered under a data-use agreement. "
            "Request access, then pass the delivered tar via --input-tar PATH "
            f"(or place it in {raw_root / 'open_yap' / 'input'}). "
            "For the public 8.92 h sample use --sample-only."
        ),
    )


def _open_yap_transcript_turns(meta: dict[str, Any], *, duration_s: float) -> tuple[dict[str, list[schema.Turn]], int]:
    transcripts = meta.get("transcripts")
    if not isinstance(transcripts, dict):
        raise ManifestError("json has no 'transcripts' object")
    out: dict[str, list[schema.Turn]] = {}
    dropped_words = 0
    for channel, speaker in (("a", "user"), ("b", "agent")):
        transcript = transcripts.get(channel)
        if not isinstance(transcript, dict):
            raise ManifestError(f"transcripts.{channel} is missing or not an object")
        words = transcript.get("words")
        if words is None:
            words = transcript.get("segments")
        if words is None:
            raise ManifestError(f"transcripts.{channel} has no words/segments timings")
        try:
            clean_words, dropped = _sanitize_word_timings(words)
            turns = normalize_turns(clean_words, speaker=speaker)
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"transcripts.{channel}: {exc}") from exc
        dropped_words += dropped
        out[speaker] = _clamp_turns(turns, duration_s)
    return out, dropped_words


def _build_open_yap_conversation(
    *,
    tar_path: Path,
    key: str,
    members: dict[str, bytes],
    raw_root: Path,
    overwrite: bool,
) -> schema.Conversation:
    record_id = key or "<bare-key>"
    source = f"{tar_path}:{record_id}"
    if "meta" not in members:
        raise QualityDrop("missing metadata member")
    try:
        meta = json.loads(members["meta"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{source}: invalid json: {exc}") from exc
    if not isinstance(meta, dict):
        raise ManifestError(f"{source}: json root is not an object")
    conversation_id = str(meta.get("conversation_id") or key or "")
    if not conversation_id:
        raise ManifestError(f"{source}: json has no conversation_id and tar member has no key")
    if "user" not in members or "agent" not in members:
        raise QualityDrop("missing channel member")
    user_audio, user_rate, user_channels = _read_audio_bytes(members["user"], source=f"{source}:a")
    agent_audio, agent_rate, agent_channels = _read_audio_bytes(members["agent"], source=f"{source}:b")
    if user_rate != agent_rate:
        raise ManifestError(f"{source}: channel sample rates differ ({user_rate} vs {agent_rate})")
    user_mono = user_audio.mean(axis=1) if user_channels > 1 else user_audio[:, 0]
    agent_mono = agent_audio.mean(axis=1) if agent_channels > 1 else agent_audio[:, 0]
    min_samples = min(user_mono.size, agent_mono.size)
    user_mono = user_mono[:min_samples]
    agent_mono = agent_mono[:min_samples]
    source_duration = min_samples / float(user_rate)
    declared_duration = meta.get("duration_seconds")
    if isinstance(declared_duration, (int, float)) and math.isfinite(float(declared_duration)):
        source_duration = min(source_duration, float(declared_duration))
    # Frame-align before deriving turns so every turn fits inside duration.
    frames = _grid_frames(source_duration)
    duration_s = frames / schema.FRAME_RATE
    frame_samples = int(round(frames / schema.FRAME_RATE * user_rate))
    user_mono = user_mono[:frame_samples]
    agent_mono = agent_mono[:frame_samples]
    try:
        turn_map, dropped_words = _open_yap_transcript_turns(meta, duration_s=duration_s)
    except ManifestError as exc:
        raise ManifestError(f"{source}: {exc}") from exc
    turns = merge_same_speaker_turns(turn_map["user"] + turn_map["agent"])
    if not turns:
        raise QualityDrop("no transcript turns")

    speakers = meta.get("speakers") if isinstance(meta.get("speakers"), dict) else {}
    speaker_ids = []
    for channel in ("a", "b"):
        info = speakers.get(channel)
        speaker_id = info.get("speaker_id") if isinstance(info, dict) else None
        if speaker_id:
            speaker_ids.append(str(speaker_id))
    meta_out = {
        "relationship": meta.get("relationship"),
        "topics": meta.get("topics", []),
        "language": meta.get("language", "en"),
        "turn_taking_gap_ms": meta.get("turn_taking_gap_ms"),
        "turns_per_minute": meta.get("turns_per_minute"),
        "speech_dominance": meta.get("speech_dominance"),
        "speaker_ids": speaker_ids,
        "speaker_channels": {"user": "a", "agent": "b"},
        "transcript_source": "deepgram-word-timings",
        "dropped_word_timings": dropped_words,
        "measurement_schema_version": meta.get("measurement_schema_version"),
        "tar": str(tar_path),
    }
    return _make_conversation(
        conversation_id=conversation_id,
        source=SOURCE_OPEN_YAP,
        license_name=LICENSE_OPEN_YAP,
        user_audio=user_mono,
        agent_audio=agent_mono,
        source_rate=user_rate,
        turns=turns,
        meta=meta_out,
        raw_root=raw_root,
        source_dir="open_yap",
        overwrite=overwrite,
        context=source,
    )


def build_open_yap(
    *,
    raw_root: Path,
    input_tar: Path | None,
    sample_only: bool,
    limit: int | None,
    overwrite: bool,
) -> Iterator[tuple[schema.Conversation | None, str, str]]:
    """Yield ``(conversation, reason, record_id)`` for Open Yap records."""
    tar_paths = _open_yap_tars(raw_root=raw_root, input_tar=input_tar, sample_only=sample_only)
    kept = 0
    for tar_path, key, members in _iter_tar_groups(tar_paths, classify_open_yap_member):
        if limit is not None and kept >= limit:
            break
        record_id = key or "<bare-key>"
        try:
            conversation = _build_open_yap_conversation(
                tar_path=tar_path,
                key=key,
                members=members,
                raw_root=raw_root,
                overwrite=overwrite,
            )
        except QualityDrop as drop:
            yield None, str(drop), record_id
            continue
        kept += 1
        yield conversation, "ok", record_id


# ---------------------------------------------------------------------------
# otoSpeech
# ---------------------------------------------------------------------------


def _require_huggingface_hub():
    try:
        import huggingface_hub
    except ImportError as exc:
        raise InstallError(
            "huggingface_hub is required for gated otoSpeech downloads: "
            "pip install huggingface_hub"
        ) from exc
    return huggingface_hub


def _is_access_error(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status in (401, 403):
        return True
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "gated" in name or "401" in message or "403" in message or "authenticate" in message


def _access_required(repo_id: str, exc: BaseException, *, instructions: str) -> AccessRequired:
    return AccessRequired(
        dataset=repo_id,
        url=f"https://huggingface.co/datasets/{repo_id}",
        instructions=instructions,
    )


def _hf_download(repo_id: str, filename: str, *, cache_dir: Path) -> Path:
    hub = _require_huggingface_hub()
    try:
        return Path(
            hub.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                cache_dir=str(cache_dir),
            )
        )
    except Exception as exc:  # noqa: BLE001 - translate gated access errors
        if _is_access_error(exc):
            raise _access_required(
                repo_id,
                exc,
                instructions=(
                    "Accept the dataset terms on the Hugging Face page (and run "
                    "`hf auth login` if the repository is gated), then rerun. "
                    "Alternatively place a licensed local copy under "
                    f"{cache_dir.parent.parent}/otospeech_280h/input or "
                    f"{cache_dir.parent.parent}/otospeech_turn104h/input/<task_id>."
                ),
            ) from exc
        raise BuildError(f"{repo_id}/{filename}: download failed: {exc}") from exc


def _hf_list_files(repo_id: str) -> list[str]:
    hub = _require_huggingface_hub()
    try:
        return list(hub.HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset"))
    except Exception as exc:  # noqa: BLE001
        if _is_access_error(exc):
            raise _access_required(
                repo_id,
                exc,
                instructions=(
                    "This dataset is gated. Accept the terms on the Hugging Face "
                    "page and authenticate (`hf auth login`), then rerun."
                ),
            ) from exc
        raise BuildError(f"{repo_id}: cannot list files: {exc}") from exc


def _local_otospeech_tars(raw_root: Path) -> list[Path]:
    root = raw_root / "otospeech_280h"
    return sorted(root.glob("input/*.tar")) + sorted(root.glob("*.tar"))


def _otospeech_280_tar_groups(
    *,
    raw_root: Path,
    limit: int | None,
    cache_dir: Path,
) -> Iterator[tuple[Path, str, dict[str, bytes]]]:
    tars = _local_otospeech_tars(raw_root)
    if not tars:
        files = sorted(
            name
            for name in _hf_list_files(OTO_280_REPO)
            if name.startswith("data/train/") and name.endswith(".tar")
        )
        if not files:
            raise BuildError(f"{OTO_280_REPO}: no data/train/shard-*.tar files")
        # One shard at a time keeps peak disk usage bounded.
        for filename in files:
            tars = [_hf_download(OTO_280_REPO, filename, cache_dir=cache_dir)]
            yield from _iter_tar_groups(tars, classify_otospeech_member)
            if limit is not None:
                # The caller enforces the successful-record limit; the caller
                # stops consuming, which closes this generator.
                pass
        return
    yield from _iter_tar_groups(tars, classify_otospeech_member)


def _build_otospeech_280_conversation(
    *,
    tar_path: Path,
    key: str,
    members: dict[str, bytes],
    raw_root: Path,
    overwrite: bool,
) -> schema.Conversation:
    record_id = key or "<bare-key>"
    source = f"{tar_path}:{record_id}"
    if "meta" not in members:
        raise QualityDrop("missing metadata member")
    try:
        meta = json.loads(members["meta"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{source}: invalid json: {exc}") from exc
    if not isinstance(meta, dict):
        raise ManifestError(f"{source}: json root is not an object")
    conversation_id = str(meta.get("session_id") or key or "")
    if not conversation_id:
        raise ManifestError(f"{source}: json has no session_id and tar member has no key")
    if "audio" not in members:
        raise QualityDrop("missing audio member")
    audio, sample_rate, channels = _read_audio_bytes(members["audio"], source=f"{source}:flac")
    if channels < 2:
        raise QualityDrop("missing channel: archive has mono audio")
    user_raw = audio[:, 0]
    agent_raw = audio[:, 1]
    min_samples = min(user_raw.size, agent_raw.size)
    user_raw = user_raw[:min_samples]
    agent_raw = agent_raw[:min_samples]
    source_duration = min_samples / float(sample_rate)
    declared = meta.get("duration")
    if isinstance(declared, (int, float)) and math.isfinite(float(declared)):
        source_duration = min(source_duration, float(declared))
    frames = _grid_frames(source_duration)
    duration_s = frames / schema.FRAME_RATE
    frame_samples = int(round(frames / schema.FRAME_RATE * sample_rate))
    user_raw = user_raw[:frame_samples]
    agent_raw = agent_raw[:frame_samples]

    user_turns = [
        schema.Turn("user", start, end, "")
        for start, end in energy_turns(user_raw, sample_rate)
    ]
    agent_turns = [
        schema.Turn("agent", start, end, "")
        for start, end in energy_turns(agent_raw, sample_rate)
    ]
    turns = merge_same_speaker_turns(_clamp_turns(user_turns + agent_turns, duration_s))
    if not turns:
        raise QualityDrop("no speech detected by energy VAD")

    speakers = meta.get("speakers") if isinstance(meta.get("speakers"), list) else []
    channel_speakers: dict[str, str] = {}
    for speaker in speakers:
        if not isinstance(speaker, dict):
            continue
        channel = speaker.get("channel")
        speaker_id = speaker.get("speaker_id")
        try:
            channel_int = int(channel)
        except (TypeError, ValueError):
            continue
        if channel_int in (0, 1) and speaker_id:
            channel_speakers[str(channel_int)] = str(speaker_id)
    instruction = meta.get("session_instruction_metadata")
    topics = instruction.get("topics", []) if isinstance(instruction, dict) else []
    meta_out = {
        "session_type": meta.get("session_type"),
        "language": meta.get("language", "en"),
        "topics": topics,
        "speaker_ids": [channel_speakers.get("0", ""), channel_speakers.get("1", "")],
        "speaker_channels": {"user": 0, "agent": 1},
        "redacted_segments": meta.get("redacted_segments", []),
        "transcript_source": "energy-vad (no transcript in dataset card)",
        "tar": str(tar_path),
    }
    return _make_conversation(
        conversation_id=conversation_id,
        source=SOURCE_OTO_280,
        license_name=LICENSE_OTO_280,
        user_audio=user_raw,
        agent_audio=agent_raw,
        source_rate=sample_rate,
        turns=turns,
        meta=meta_out,
        raw_root=raw_root,
        source_dir="otospeech_280h",
        overwrite=overwrite,
        context=source,
    )


def build_otospeech_280(
    *,
    raw_root: Path,
    limit: int | None,
    overwrite: bool,
    cache_dir: Path,
) -> Iterator[tuple[schema.Conversation | None, str, str]]:
    kept = 0
    for tar_path, key, members in _otospeech_280_tar_groups(
        raw_root=raw_root, limit=limit, cache_dir=cache_dir
    ):
        if limit is not None and kept >= limit:
            break
        record_id = key or "<bare-key>"
        try:
            conversation = _build_otospeech_280_conversation(
                tar_path=tar_path,
                key=key,
                members=members,
                raw_root=raw_root,
                overwrite=overwrite,
            )
        except QualityDrop as drop:
            yield None, str(drop), record_id
            continue
        kept += 1
        yield conversation, "ok", record_id


def _local_turn104_dirs(raw_root: Path) -> list[Path]:
    root = raw_root / "otospeech_turn104h"
    candidates: list[Path] = []
    for base in (root / "input", root):
        if base.exists():
            candidates.extend(
                path
                for path in base.iterdir()
                if path.is_dir() and (path / "metadata.json").exists()
            )
    return sorted({path.resolve() for path in candidates})


def _download_turn104_dirs(
    *,
    raw_root: Path,
    limit: int | None,
    cache_dir: Path,
) -> list[Path]:
    files = _hf_list_files(OTO_104_REPO)
    keys = sorted({name.split("/", 1)[0] for name in files if "/" in name})
    if not keys:
        raise BuildError(f"{OTO_104_REPO}: no conversation directories found")
    if limit is not None:
        keys = keys[:limit]
    required = (
        "metadata.json",
        "speaker_1_audio.wav",
        "speaker_2_audio.wav",
        "speaker_1_annotation_a.srt",
        "speaker_2_annotation_a.srt",
    )
    local_root = raw_root / "otospeech_turn104h" / "input"
    local_root.mkdir(parents=True, exist_ok=True)
    dirs: list[Path] = []
    for key in keys:
        target = local_root / key
        target.mkdir(parents=True, exist_ok=True)
        available = {name for name in files if name.startswith(f"{key}/")}
        for filename in required:
            if f"{key}/{filename}" not in available:
                raise ManifestError(f"{OTO_104_REPO}:{key}: missing required file {filename}")
            path = _hf_download(OTO_104_REPO, f"{key}/{filename}", cache_dir=cache_dir)
            destination = target / filename
            if destination.exists() and destination.stat().st_size == path.stat().st_size:
                continue
            destination.unlink(missing_ok=True)
            try:
                os.link(path, destination)
            except OSError:
                try:
                    destination.symlink_to(path)
                except OSError:
                    shutil.copy2(path, destination)
        dirs.append(target)
    return dirs


def _build_turn104_conversation(
    *,
    directory: Path,
    raw_root: Path,
    overwrite: bool,
) -> schema.Conversation:
    source = str(directory)
    metadata_path = directory / "metadata.json"
    try:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{metadata_path}: invalid metadata: {exc}") from exc
    if not isinstance(meta, dict):
        raise ManifestError(f"{metadata_path}: metadata root is not an object")
    task_id = str(meta.get("task_id") or directory.name)
    if not task_id:
        raise ManifestError(f"{source}: no task_id")

    channels: dict[str, np.ndarray] = {}
    sample_rate: int | None = None
    for speaker, filename in (("user", "speaker_1_audio.wav"), ("agent", "speaker_2_audio.wav")):
        path = directory / filename
        if not path.exists():
            raise QualityDrop(f"missing channel file {filename}")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ManifestError(f"{path}: cannot read: {exc}") from exc
        audio, rate, count = _read_audio_bytes(payload, source=str(path))
        if count > 1:
            audio = audio.mean(axis=1, keepdims=True)
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            raise ManifestError(f"{source}: channel sample rates differ")
        channels[speaker] = audio[:, 0]
    assert sample_rate is not None
    min_samples = min(channels["user"].size, channels["agent"].size)
    user_raw = channels["user"][:min_samples]
    agent_raw = channels["agent"][:min_samples]
    frames = _grid_frames(min_samples / float(sample_rate))
    duration_s = frames / schema.FRAME_RATE

    turns: list[schema.Turn] = []
    label_counts: dict[str, int] = {}
    for speaker, filename in (("user", "speaker_1_annotation_a.srt"), ("agent", "speaker_2_annotation_a.srt")):
        path = directory / filename
        if not path.exists():
            raise ManifestError(f"{path}: missing SRT annotation")
        try:
            cues = parse_srt(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ManifestError(f"{path}: cannot parse SRT: {exc}") from exc
        for cue in cues:
            label_counts[cue.label] = label_counts.get(cue.label, 0) + 1
        turns.extend(srt_to_turns(cues, speaker=speaker))
    turns = merge_same_speaker_turns(_clamp_turns(turns, duration_s))
    if not turns:
        raise QualityDrop("no SRT turns")

    meta_out = {
        "conversation_type": meta.get("conversation_type"),
        "speaker_ids": [
            str(meta.get("speaker_1_actor_id", "")),
            str(meta.get("speaker_2_actor_id", "")),
        ],
        "speaker_channels": {"user": "speaker_1", "agent": "speaker_2"},
        "genders": [
            meta.get("speaker_1_actor_gender", ""),
            meta.get("speaker_2_actor_gender", ""),
        ],
        "turn_event_counts": label_counts,
        "transcript_source": "srt-turn-annotations",
        "directory": str(directory),
    }
    return _make_conversation(
        conversation_id=task_id,
        source=SOURCE_OTO_104,
        license_name=LICENSE_OTO_104,
        user_audio=user_raw,
        agent_audio=agent_raw,
        source_rate=sample_rate,
        turns=turns,
        meta=meta_out,
        raw_root=raw_root,
        source_dir="otospeech_turn104h",
        overwrite=overwrite,
        context=source,
    )


def build_otospeech_104(
    *,
    raw_root: Path,
    limit: int | None,
    overwrite: bool,
    cache_dir: Path,
) -> Iterator[tuple[schema.Conversation | None, str, str]]:
    directories = _local_turn104_dirs(raw_root)
    if not directories:
        directories = _download_turn104_dirs(
            raw_root=raw_root, limit=limit, cache_dir=cache_dir
        )
    kept = 0
    for directory in directories:
        if limit is not None and kept >= limit:
            break
        record_id = directory.name
        try:
            conversation = _build_turn104_conversation(
                directory=directory,
                raw_root=raw_root,
                overwrite=overwrite,
            )
        except QualityDrop as drop:
            yield None, str(drop), record_id
            continue
        kept += 1
        yield conversation, "ok", record_id


# ---------------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------------


def _write_temp_source(
    *,
    out_dir: Path,
    source: str,
    records: Iterable[tuple[schema.Conversation | None, str, str]],
    printer: Callable[[str], None],
) -> tuple[Path, int, int]:
    temp_path = out_dir / f".{source}.tmp.jsonl"
    kept = 0
    dropped = 0
    drops: dict[str, int] = {}
    with temp_path.open("w", encoding="utf-8") as handle:
        for conversation, reason, record_id in records:
            if conversation is None:
                dropped += 1
                drops[reason] = drops.get(reason, 0) + 1
                printer(f"  drop {source}/{record_id}: {reason}")
                continue
            conversation.validate()
            handle.write(json.dumps(conversation.to_dict(), ensure_ascii=False) + "\n")
            kept += 1
    if drops:
        printer(f"  {source}: dropped {dropped} record(s): {json.dumps(drops, sort_keys=True)}")
    return temp_path, kept, dropped


def _finalise_manifests(
    *,
    out_dir: Path,
    temp_paths: dict[str, Path],
    overwrite: bool,
    printer: Callable[[str], None],
) -> dict[str, dict[str, Any]]:
    final_paths = {source: out_dir / SOURCE_FILES[source] for source in temp_paths}
    if not overwrite:
        existing = [str(path) for path in final_paths.values() if path.exists()]
        if existing:
            raise BuildError(
                f"output manifest(s) already exist: {', '.join(existing)}; pass --overwrite"
            )
    records: dict[str, list[dict[str, Any]]] = {}
    speaker_ids: set[str] = set()
    conversation_ids: set[str] = set()
    for source, temp_path in temp_paths.items():
        source_records: list[dict[str, Any]] = []
        with temp_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                source_records.append(value)
                conversation_ids.add(str(value["id"]))
                for speaker in value.get("meta", {}).get("speaker_ids", []) or []:
                    if speaker:
                        speaker_ids.add(str(speaker))
        records[source] = source_records

    holdout_speakers = select_holdout(speaker_ids, HOLDOUT_SPEAKERS, salt=HOLDOUT_SPEAKER_SALT)
    holdout_conversations = select_holdout(
        conversation_ids, HOLDOUT_CONVERSATIONS, salt=HOLDOUT_CONVERSATION_SALT
    )
    printer(
        f"holdout: {len(holdout_speakers)} speaker(s), "
        f"{len(holdout_conversations)} conversation(s) -> split=test"
    )

    stats: dict[str, dict[str, Any]] = {}
    combined_path = out_dir / COMBINED_FILE
    with combined_path.open("w", encoding="utf-8") as combined:
        for source in (SOURCE_OPEN_YAP, SOURCE_OTO_280, SOURCE_OTO_104):
            if source not in records:
                continue
            path = final_paths[source]
            count = 0
            hours = 0.0
            by_split: dict[str, dict[str, float]] = {}
            with path.open("w", encoding="utf-8") as handle:
                for value in records[source]:
                    conversation = schema.Conversation.from_dict(value)
                    conversation.split = assign_split(
                        conversation.id,
                        value.get("meta", {}).get("speaker_ids", []) or [],
                        holdout_speakers=holdout_speakers,
                        holdout_conversations=holdout_conversations,
                    )
                    conversation.validate()
                    line = json.dumps(conversation.to_dict(), ensure_ascii=False)
                    handle.write(line + "\n")
                    combined.write(line + "\n")
                    count += 1
                    duration_h = conversation.duration_s / 3600.0
                    hours += duration_h
                    bucket = by_split.setdefault(conversation.split, {"hours": 0.0, "conversations": 0})
                    bucket["hours"] += duration_h
                    bucket["conversations"] += 1
            stats[source] = {
                "conversations": count,
                "hours": hours,
                "by_split": {
                    split: {"hours": round(value["hours"], 4), "conversations": int(value["conversations"])}
                    for split, value in sorted(by_split.items())
                },
                "license": records[source][0]["license"] if records[source] else None,
            }
            temp_path.unlink(missing_ok=True)
    return stats


def _print_report(out_dir: Path, stats: dict[str, dict[str, Any]], printer: Callable[[str], None]) -> None:
    printer("hours by source/split:")
    for source, info in stats.items():
        printer(
            f"  {source}: {info['hours']:.2f} h, {info['conversations']} conversation(s), "
            f"license={info['license']}"
        )
        for split, bucket in info["by_split"].items():
            printer(f"    {split}: {bucket['hours']:.2f} h ({bucket['conversations']})")
    summary = schema.summarize_manifest(out_dir / COMBINED_FILE)
    printer("combined manifest:")
    printer(f"  conversations: {summary['conversations']}")
    printer(f"  hours: {summary['hours']:.2f}")
    for license_name, hours in sorted(summary["hours_by_license"].items()):
        printer(f"  license {license_name}: {hours:.2f} h")
    for source, hours in sorted(summary["hours_by_source"].items()):
        printer(f"  source {source}: {hours:.2f} h")
    for split, count in sorted(summary["splits"].items()):
        printer(f"  split {split}: {count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(
    *,
    source: str,
    out_dir: Path,
    raw_root: Path,
    limit: int | None = None,
    sample_only: bool = False,
    overwrite: bool = False,
    input_tar: Path | None = None,
    printer: Callable[[str], None] = print,
) -> dict[str, dict[str, Any]]:
    """Build the requested sources and return per-source summary statistics."""
    if source not in ("open_yap", "otospeech", "all"):
        raise ValueError("source must be one of open_yap, otospeech, all")
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")
    if sample_only and source == "otospeech":
        raise BuildError("--sample-only only applies to Open Yap; otoSpeech has no public sample")
    if source == "otospeech" and input_tar is not None:
        raise BuildError("--input-tar only applies to Open Yap")

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    cache_dir = raw_root / "_downloads" / "hf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if limit == 0:
        printer("limit=0: nothing to build")
        return {}

    selected = {
        "open_yap": (SOURCE_OPEN_YAP,),
        "otospeech": (SOURCE_OTO_280, SOURCE_OTO_104),
        "all": (SOURCE_OPEN_YAP, SOURCE_OTO_280, SOURCE_OTO_104),
    }[source]
    if not overwrite:
        existing = [str(out_dir / SOURCE_FILES[item]) for item in selected if (out_dir / SOURCE_FILES[item]).exists()]
        if existing:
            raise BuildError(f"output manifest(s) already exist: {', '.join(existing)}; pass --overwrite")

    temp_paths: dict[str, Path] = {}

    for item in selected:
        if item == SOURCE_OPEN_YAP:
            printer(f"building {SOURCE_OPEN_YAP} ...")
            records = build_open_yap(
                raw_root=raw_root,
                input_tar=input_tar,
                sample_only=sample_only,
                limit=limit,
                overwrite=overwrite,
            )
        elif item == SOURCE_OTO_280:
            printer(f"building {SOURCE_OTO_280} ...")
            records = build_otospeech_280(
                raw_root=raw_root,
                limit=limit,
                overwrite=overwrite,
                cache_dir=cache_dir,
            )
        else:
            printer(f"building {SOURCE_OTO_104} ...")
            records = build_otospeech_104(
                raw_root=raw_root,
                limit=limit,
                overwrite=overwrite,
                cache_dir=cache_dir,
            )
        temp_path, kept, _ = _write_temp_source(
            out_dir=out_dir, source=item, records=records, printer=printer
        )
        if kept:
            temp_paths[item] = temp_path
        else:
            temp_path.unlink(missing_ok=True)
            printer(f"  {item}: no records kept")

    if not temp_paths:
        printer("no records built")
        return {}

    stats = _finalise_manifests(
        out_dir=out_dir, temp_paths=temp_paths, overwrite=overwrite, printer=printer
    )
    _print_report(out_dir, stats, printer)
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("open_yap", "otospeech", "all"), default="all")
    parser.add_argument("--out", type=Path, default=Path("data/manifests"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--limit", type=int, default=None, help="max conversations per source")
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Open Yap: use the public 8.92 h sample instead of a licensed full copy",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--input-tar",
        type=Path,
        default=None,
        help="licensed Open Yap tar (full corpus or public shards)",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")
    try:
        run(
            source=args.source,
            out_dir=args.out,
            raw_root=args.raw,
            limit=args.limit,
            sample_only=args.sample_only,
            overwrite=args.overwrite,
            input_tar=args.input_tar,
        )
    except AccessRequired as exc:
        print(f"access required: {exc}", file=sys.stderr)
        print(f"  dataset: {exc.dataset}", file=sys.stderr)
        print(f"  url: {exc.url}", file=sys.stderr)
        print(f"  next: {exc.instructions}", file=sys.stderr)
        return 3
    except BuildError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"build failed (I/O): {exc}", file=sys.stderr)
        return 2
    return 0


__all__ = [
    "MAX_CLIPPING_RATIO",
    "HOLDOUT_SPEAKERS",
    "HOLDOUT_CONVERSATIONS",
    "SOURCE_OPEN_YAP",
    "SOURCE_OTO_280",
    "SOURCE_OTO_104",
    "LICENSE_OPEN_YAP",
    "LICENSE_OTO_280",
    "LICENSE_OTO_104",
    "BuildError",
    "InstallError",
    "ManifestError",
    "AccessRequired",
    "QualityDrop",
    "SrtCue",
    "stable_hash",
    "select_holdout",
    "assign_split",
    "passes_quality",
    "normalize_turns",
    "merge_same_speaker_turns",
    "parse_srt",
    "srt_to_turns",
    "energy_turns",
    "classify_open_yap_member",
    "classify_otospeech_member",
    "run",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
