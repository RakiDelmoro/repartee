"""Synthetic full-duplex dialogue generation for Repartee (blueprint 7.2).

Everything in this module is a *real* generation path.  Nothing produces
placeholder audio: the TTS backends either import their upstream package or
raise :class:`InstallError` with install instructions.  ``--dry-run`` is the
only mode that emits no audio, and its output is explicitly labelled
``"dry_run": true, "audio": false, "manifest": false``.

Pipeline
--------
1. An LLM writes 2-8 turn dialogues (70 % assistant/task, 20 % casual, 10 %
   backchannel-heavy) with a fixed agent persona and a fixed agent voice id
   (:class:`LocalHFDialogueWriter`; local Hugging Face weights only, no
   OpenAI/network API keys).  ``ScriptedDialogueWriter`` exists solely for
   ``--dry-run`` planning and refuses to run in a real build.
2. TTS renders each utterance with a commercially licensed backend: Kokoro
   (Apache-2.0), Dia (Apache-2.0), Chatterbox (MIT) or CosyVoice2
   (Apache-2.0).  XTTS-v2, F5-TTS and Fish weights are never used (see
   ``src/license_audit.py``).
3. :func:`compose_timeline` places turns at 12.5 Hz-compatible times:
   * 0.64 s inter-turn gap for 80 % of turns (otherwise 0.15-0.55 s),
   * barge-in for 20 % of agent turns: cut the agent audio and leave silence,
   * a short agent backchannel inside 10 % of user turns,
   * 100-400 ms cross-speaker overlap at 15 % of turn boundaries.
4. :func:`apply_augmentation` augments the user channel: gain -24..+15 dB
   (50 %), additive noise at -30..+6 dB SNR (30 %), synthetic reverb (30 %)
   and delayed agent echo at 0..20 % gain / 100-500 ms (30 %).
5. ``ECAPAVoiceChecker`` rejects a dialogue if any agent utterance has
   cosine similarity < 0.8 to the fixed-voice reference (optional
   ``speechbrain``; enabled by default, ``--no-voice-check`` turns it off and
   records that limitation in the manifest).

Reproducibility
---------------
Each dialogue id is ``synth-<seed>-<index:07d>`` and owns an independent RNG
seed.  Dialogue text comes from a deterministic prompt+sampling seed; TTS is
deterministic for a fixed backend.  Completed dialogues are written atomically
to ``<raw>/synth/<id>/`` with a ``record.json`` sidecar, so rerunning the CLI
skips them (resumable by dialogue id).

CLI
---
``python -m src.build_synth --out data/manifests --raw data/raw --hours 8000
--backend kokoro --writers 8 [--dry-run] [--writer-model MODEL]
[--seed 0] [--voice-check|--no-voice-check]``

Limitations
-----------
* None of the TTS backends were executed in this environment (their weights
  are not installed); the Dia/Chatterbox/CosyVoice2 wrappers follow their
  current public APIs and raise ``InstallError``/``BackendUnavailable`` rather
  than falling back to placeholder audio.  Kokoro is the reference backend.
* ``--writers`` creates one TTS backend instance per worker thread; large
  backends (Dia/CosyVoice2) multiply GPU memory by that count.
* Reverb is a cheap synthetic RIR, not a measured room impulse response.
* The LLM prompt is English-only; no non-English synthetic data is produced.
* Voice similarity uses an ECAPA speaker embedding; it detects drift, not
  identity leakage or TTS artefacts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from . import schema

# ---------------------------------------------------------------------------
# Fixed design constants
# ---------------------------------------------------------------------------

CATEGORIES = ("assistant", "casual", "backchannel")
CATEGORY_WEIGHTS = {"assistant": 70, "casual": 20, "backchannel": 10}

DEFAULT_GAP_S = 0.64
DEFAULT_GAP_PROBABILITY = 0.80
DEFAULT_BARGE_IN_PROBABILITY = 0.20
DEFAULT_BACKCHANNEL_PROBABILITY = 0.10
DEFAULT_OVERLAP_PROBABILITY = 0.15
MIN_OVERLAP_S = 0.10
MAX_OVERLAP_S = 0.40

AGENT_PERSONA = (
    "Repartee: a warm, concise, curious conversational partner. It listens, "
    "answers directly, uses natural spoken English, and never mentions being "
    "an AI unless asked."
)
USER_PERSONAS = (
    "A busy parent planning a weekend trip.",
    "A software engineer venting about a difficult bug.",
    "A student preparing for a presentation.",
    "A neighbour catching up after work.",
    "A hobbyist excited about a new recipe.",
    "A traveller asking for local recommendations.",
)
DEFAULT_BACKCHANNELS: tuple[tuple[str, float], ...] = (
    ("mm-hmm", 0.45),
    ("yeah", 0.40),
    ("right", 0.42),
    ("okay", 0.48),
)

SYNTH_SOURCE = "synthetic"
SYNTH_LICENSE = "synthetic-commercial"
SYNTH_MANIFEST = "repartee-synth.jsonl"

USER_VOICES_BY_BACKEND: dict[str, tuple[str, ...]] = {
    "kokoro": ("am_michael", "af_bella", "am_adam", "af_alloy", "af_nicole"),
    "dia": ("S1",),
    "chatterbox": ("user",),
    "cosyvoice2": ("user",),
}
AGENT_VOICE_BY_BACKEND: dict[str, str] = {
    "kokoro": "af_heart",
    "dia": "S2",
    "chatterbox": "agent",
    "cosyvoice2": "agent",
}


class SynthError(RuntimeError):
    """Base error for synthetic generation."""


class InstallError(SynthError):
    """A TTS/LLM dependency is not installed."""


class BackendUnavailable(SynthError):
    """A backend is installed but its weights/configuration are missing."""


class DialogueError(SynthError):
    """The dialogue writer returned malformed or invalid text."""


class VoiceCheckError(SynthError):
    """Voice-similarity rejection."""


# ---------------------------------------------------------------------------
# Dialogue model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Utterance:
    speaker: str
    text: str

    def __post_init__(self) -> None:
        if self.speaker not in ("user", "agent"):
            raise ValueError(f"speaker must be user or agent, got {self.speaker!r}")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("utterance text must be a non-empty string")


@dataclass(frozen=True)
class Dialogue:
    id: str
    category: str
    utterances: tuple[Utterance, ...]
    persona: str
    seed: int
    user_voice: str = ""
    agent_voice: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("dialogue id must be non-empty")
        if self.category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}, got {self.category!r}")
        if not 2 <= len(self.utterances) <= 8:
            raise ValueError(f"a dialogue needs 2-8 utterances, got {len(self.utterances)}")
        if not self.persona.strip():
            raise ValueError("persona must be non-empty")

    @property
    def text(self) -> str:
        return " ".join(utterance.text for utterance in self.utterances)


def _stable_digest(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def dialogue_mix(count: int, seed: int = 0) -> list[str]:
    """Return ``count`` categories with exact 70/20/10 largest-remainder quotas.

    The list is shuffled with ``seed`` so categories do not appear in blocks.
    """
    if count < 0:
        raise ValueError("count must be >= 0")
    quotas = {name: count * weight / 100.0 for name, weight in CATEGORY_WEIGHTS.items()}
    counts = {name: int(math.floor(quota)) for name, quota in quotas.items()}
    remaining = count - sum(counts.values())
    order = sorted(
        CATEGORY_WEIGHTS,
        key=lambda name: (-(quotas[name] - counts[name]), CATEGORIES.index(name)),
    )
    for name in order[:remaining]:
        counts[name] += 1
    values = [name for name in CATEGORIES for _ in range(counts[name])]
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    return values


def category_for_index(index: int, *, seed: int = 0) -> str:
    """Category of the ``index``-th dialogue in the deterministic 10-block mix."""
    if index < 0:
        raise ValueError("index must be >= 0")
    block, position = divmod(index, 10)
    block_mix = dialogue_mix(10, seed=seed * 1_000_003 + block)
    return block_mix[position]


def persona_for(dialogue_id: str) -> tuple[str, str]:
    """Return fixed agent persona and a deterministic user persona."""
    user_persona = USER_PERSONAS[_stable_digest("user:" + dialogue_id) % len(USER_PERSONAS)]
    return AGENT_PERSONA, user_persona


# ---------------------------------------------------------------------------
# Dialogue writers
# ---------------------------------------------------------------------------


@runtime_checkable
class DialogueWriter(Protocol):
    """Writes one dialogue.  Implementations must be deterministic per seed."""

    def write(self, dialogue_id: str, *, category: str, seed: int) -> Dialogue:
        ...


_SCRIPTED_TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    "assistant": (
        ("user", "I need to plan a two-day trip to Lisbon. Where should I start?"),
        ("agent", "Start with Alfama in the morning and leave the tram for the afternoon."),
        ("user", "Is the tram usually crowded on weekends?"),
        ("agent", "Very crowded after ten, so go early or walk instead."),
        ("user", "That helps. Any place for dinner near the river?"),
        ("agent", "Try the small places in Cais do Sodre, away from the main square."),
    ),
    "casual": (
        ("user", "You will not believe what happened at the shop today."),
        ("agent", "Tell me. I could use a good story."),
        ("user", "The cashier recognised me from the bakery and gave me a free loaf."),
        ("agent", "That is a much nicer ending than I expected."),
    ),
    "backchannel": (
        ("user", "So I was walking home, and the whole street was blocked by a film crew."),
        ("agent", "mm-hmm, go on."),
        ("user", "They had lights everywhere and a fake cafe sign on the corner."),
        ("agent", "right."),
        ("user", "I waited twenty minutes, and they never even started filming."),
        ("agent", "that sounds about right."),
    ),
}


class ScriptedDialogueWriter:
    """Deterministic template writer for ``--dry-run`` only.

    This is *not* an LLM and its text must never be rendered into training
    data; :meth:`write` is used by the dry-run planner and the pipeline
    rejects it for real runs.
    """

    is_scripted = True

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed

    def write(self, dialogue_id: str, *, category: str, seed: int) -> Dialogue:
        template = _SCRIPTED_TEMPLATES.get(category)
        if template is None:
            raise ValueError(f"unknown category {category!r}")
        agent_persona, user_persona = persona_for(dialogue_id)
        persona = f"{user_persona} Speaking with {agent_persona}"
        return Dialogue(
            id=dialogue_id,
            category=category,
            utterances=tuple(Utterance(speaker, text) for speaker, text in template),
            persona=persona,
            seed=seed,
        )


_SYSTEM_PROMPT = (
    "You write realistic two-party spoken English conversations. Return only "
    "valid JSON, with no markdown fences or commentary. The schema is "
    '{"turns": [{"speaker": "user" or "agent", "text": "..."}, ...]}. '
    "Write 2 to 8 turns. Keep each turn speakable in under 20 seconds. "
    "Do not use stage directions, emoji, bullet points, or speaker labels "
    "inside the text."
)

_CATEGORY_PROMPTS = {
    "assistant": (
        "Write an assistant/task dialogue: the user asks for help and the "
        "agent gives concrete, natural spoken answers."
    ),
    "casual": (
        "Write a casual chat between two people who know each other. Keep it "
        "warm and meandering, with no task to complete."
    ),
    "backchannel": (
        "Write a backchannel-heavy dialogue: the user does most of the talking "
        "and the agent uses short acknowledgements such as 'mm-hmm', 'right', "
        "'yeah', and 'okay' while the user continues."
    ),
}


class LocalHFDialogueWriter:
    """Local Hugging Face causal-LM dialogue writer (no API keys, no network APIs).

    Weights are loaded lazily with ``transformers``/``torch``.  Generation is
    deterministic for a fixed seed.  Malformed model output raises
    :class:`DialogueError`; it is never silently replaced by template text.
    """

    is_scripted = False

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.9,
        top_p: float = 0.95,
        local_files_only: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.local_files_only = local_files_only
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise InstallError(
                "LocalHFDialogueWriter needs torch and transformers: "
                "pip install torch transformers"
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, local_files_only=self.local_files_only
            )
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype="auto",
                    local_files_only=self.local_files_only,
                )
            except TypeError:
                # transformers>=5 renamed torch_dtype to dtype.
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    dtype="auto",
                    local_files_only=self.local_files_only,
                )
            if self.device != "auto":
                self._model.to(self.device)
            self._model.eval()
        except Exception as exc:  # noqa: BLE001 - surface a clear local-weight error
            raise BackendUnavailable(
                f"cannot load dialogue writer {self.model_id!r}: {exc}. "
                "Pass a local path or a Hugging Face model id you have access to."
            ) from exc
        self._torch = torch

    def _prompt(self, category: str, *, agent_persona: str, user_persona: str) -> str:
        return (
            f"Agent persona: {agent_persona}\n"
            f"User persona: {user_persona}\n"
            f"{_CATEGORY_PROMPTS[category]}\n"
            "Return the JSON object now."
        )

    def write(self, dialogue_id: str, *, category: str, seed: int) -> Dialogue:
        self._load()
        assert self._model is not None and self._tokenizer is not None
        torch = self._torch
        torch.manual_seed(seed)
        agent_persona, user_persona = persona_for(dialogue_id)
        user_prompt = self._prompt(
            category, agent_persona=agent_persona, user_persona=user_persona
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # tokenizer without a chat template
            prompt = f"{_SYSTEM_PROMPT}\n\n{user_prompt}\n\nJSON:"
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        utterances = parse_dialogue_text(text, dialogue_id=dialogue_id)
        return Dialogue(
            id=dialogue_id,
            category=category,
            utterances=tuple(utterances),
            persona=f"{user_persona} Speaking with {agent_persona}",
            seed=seed,
        )


def parse_dialogue_text(text: str, *, dialogue_id: str) -> list[Utterance]:
    """Parse the writer's JSON response into validated utterances."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise DialogueError(f"{dialogue_id}: model did not return a JSON object: {text[:200]!r}")
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise DialogueError(f"{dialogue_id}: invalid JSON from writer: {exc}: {text[:300]!r}") from exc
    if isinstance(value, dict):
        raw_turns = value.get("turns")
    elif isinstance(value, list):
        raw_turns = value
    else:
        raw_turns = None
    if not isinstance(raw_turns, list):
        raise DialogueError(f"{dialogue_id}: JSON has no 'turns' list")
    utterances: list[Utterance] = []
    for index, raw in enumerate(raw_turns):
        if not isinstance(raw, dict):
            raise DialogueError(f"{dialogue_id}: turn {index} is not an object")
        speaker = str(raw.get("speaker", "")).strip().lower()
        line = raw.get("text", raw.get("content", ""))
        if speaker not in ("user", "agent"):
            raise DialogueError(f"{dialogue_id}: turn {index} has speaker {speaker!r}")
        if not isinstance(line, str) or not line.strip():
            raise DialogueError(f"{dialogue_id}: turn {index} has empty text")
        utterances.append(Utterance(speaker, line.strip()))
    if not 2 <= len(utterances) <= 8:
        raise DialogueError(f"{dialogue_id}: writer returned {len(utterances)} turns, expected 2-8")
    return utterances


# ---------------------------------------------------------------------------
# Timeline composition (pure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingTurn:
    speaker: str
    duration: float
    text: str = ""

    def __post_init__(self) -> None:
        if self.speaker not in ("user", "agent"):
            raise ValueError(f"speaker must be user or agent, got {self.speaker!r}")
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError(f"duration must be finite and positive, got {self.duration!r}")


@dataclass(frozen=True)
class Placement:
    """One placed speech interval on the shared timeline.

    ``end`` is the audible speech end; ``nominal_end`` is where the turn would
    have ended without a barge-in.  ``pad_after`` is the silence that replaces
    the cut agent tail (barge-in only).  ``source_index`` indexes the original
    dialogue utterance, or ``-1`` for an inserted backchannel.
    """

    source_index: int
    speaker: str
    text: str
    kind: str
    start: float
    end: float
    nominal_end: float
    pad_after: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def _intervals_overlap(start: float, end: float, other_start: float, other_end: float) -> bool:
    return start < other_end - 1e-9 and end > other_start + 1e-9


def compose_timeline(
    turns: Sequence[TimingTurn],
    *,
    seed: int = 0,
    gap_s: float = DEFAULT_GAP_S,
    gap_probability: float = DEFAULT_GAP_PROBABILITY,
    barge_in_probability: float = DEFAULT_BARGE_IN_PROBABILITY,
    backchannel_probability: float = DEFAULT_BACKCHANNEL_PROBABILITY,
    backchannel_options: Sequence[tuple[str, float]] = DEFAULT_BACKCHANNELS,
    overlap_probability: float = DEFAULT_OVERLAP_PROBABILITY,
    min_overlap_s: float = MIN_OVERLAP_S,
    max_overlap_s: float = MAX_OVERLAP_S,
) -> list[Placement]:
    """Place dialogue turns on one dual-channel timeline.

    The returned placements are sorted by start time and never contain two
    simultaneous intervals from the same speaker.  Cross-speaker overlap is
    deliberate.  Barge-ins shorten the preceding agent placement and record
    ``pad_after`` silence; the user turn starts exactly at the cut.
    """
    if not turns:
        return []
    if gap_s < 0:
        raise ValueError("gap_s must be >= 0")
    for name, probability in (
        ("gap_probability", gap_probability),
        ("barge_in_probability", barge_in_probability),
        ("backchannel_probability", backchannel_probability),
        ("overlap_probability", overlap_probability),
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if min_overlap_s < 0 or max_overlap_s < min_overlap_s:
        raise ValueError("overlap bounds must satisfy 0 <= min <= max")
    for text, duration in backchannel_options:
        if not text or duration <= 0:
            raise ValueError("backchannel options need non-empty text and positive duration")

    rng = np.random.default_rng(seed)
    mutable: list[dict[str, Any]] = []
    cursor = 0.0

    for index, turn in enumerate(turns):
        start: float | None = None
        if mutable:
            previous = mutable[-1]
            cross_speaker = previous["speaker"] != turn.speaker
            if (
                cross_speaker
                and previous["speaker"] == "agent"
                and turn.speaker == "user"
                and rng.random() < barge_in_probability
            ):
                cut = previous["start"] + max(0.30, previous["duration"] * rng.uniform(0.35, 0.70))
                cut = min(cut, previous["end"])
                previous["end"] = cut
                previous["pad_after"] = max(0.0, previous["nominal_end"] - cut)
                start = cut
            elif cross_speaker and rng.random() < overlap_probability:
                overlap = rng.uniform(min_overlap_s, max_overlap_s)
                candidate = previous["end"] - overlap
                if candidate >= previous["start"] + 0.05:
                    start = candidate
            if start is None:
                gap = gap_s if rng.random() < gap_probability else rng.uniform(0.15, 0.55)
                start = cursor + gap
        else:
            start = 0.0

        same_speaker_end = max(
            (
                placement["nominal_end"] if placement["pad_after"] > 0 else placement["end"]
                for placement in mutable
                if placement["speaker"] == turn.speaker
            ),
            default=-math.inf,
        )
        start = max(start, same_speaker_end)
        nominal_end = start + turn.duration
        mutable.append(
            {
                "source_index": index,
                "speaker": turn.speaker,
                "text": turn.text,
                "kind": "normal",
                "start": start,
                "end": nominal_end,
                "nominal_end": nominal_end,
                "duration": turn.duration,
                "pad_after": 0.0,
            }
        )
        cursor = max(cursor, nominal_end)

    # Backchannels are inserted after the main layout so their overlap with
    # user speech can be checked against every agent interval.
    user_indices = [i for i, placement in enumerate(mutable) if placement["speaker"] == "user"]
    for placement_index in user_indices:
        user = mutable[placement_index]
        if user["duration"] < 1.0 or rng.random() >= backchannel_probability:
            continue
        text, duration = backchannel_options[int(rng.integers(0, len(backchannel_options)))]
        max_duration = user["duration"] - 0.40
        duration = min(duration, max_duration)
        if duration < 0.20:
            continue
        low = user["start"] + 0.15
        high = user["end"] - duration - 0.15
        if high <= low:
            continue
        start = float(rng.uniform(low, high))
        end = start + duration
        if any(
            placement["speaker"] == "agent"
            and _intervals_overlap(
                start,
                end,
                placement["start"],
                placement["nominal_end"] if placement["pad_after"] > 0 else placement["end"],
            )
            for placement in mutable
        ):
            continue
        mutable.append(
            {
                "source_index": -1,
                "speaker": "agent",
                "text": text,
                "kind": "backchannel",
                "start": start,
                "end": end,
                "nominal_end": end,
                "duration": duration,
                "pad_after": 0.0,
            }
        )

    mutable.sort(key=lambda item: (item["start"], item["speaker"], item["end"]))
    return [
        Placement(
            source_index=int(item["source_index"]),
            speaker=str(item["speaker"]),
            text=str(item["text"]),
            kind=str(item["kind"]),
            start=round(float(item["start"]), 4),
            end=round(float(item["end"]), 4),
            nominal_end=round(float(item["nominal_end"]), 4),
            pad_after=round(float(item["pad_after"]), 4),
        )
        for item in mutable
    ]


def timeline_end(placements: Sequence[Placement]) -> float:
    """End of the timeline including barge-in silence padding."""
    return max((placement.nominal_end for placement in placements), default=0.0)


# ---------------------------------------------------------------------------
# Augmentation (pure numpy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentationSpec:
    gain_db: float = 0.0
    snr_db: float | None = None
    reverb: bool = False
    echo_gain: float = 0.0
    echo_delay_s: float = 0.0

    @property
    def echo(self) -> bool:
        return self.echo_gain > 0.0 and self.echo_delay_s > 0.0


def sample_augmentation(
    seed: int,
    *,
    gain_probability: float = 0.50,
    noise_probability: float = 0.30,
    reverb_probability: float = 0.30,
    echo_probability: float = 0.30,
) -> AugmentationSpec:
    """Sample the blueprint augmentation recipe deterministically from ``seed``."""
    rng = np.random.default_rng(seed)
    gain_db = float(rng.uniform(-24.0, 15.0)) if rng.random() < gain_probability else 0.0
    snr_db = float(rng.uniform(-30.0, 6.0)) if rng.random() < noise_probability else None
    reverb = bool(rng.random() < reverb_probability)
    if rng.random() < echo_probability:
        echo_gain = float(rng.uniform(0.0, 0.20))
        echo_delay_s = float(rng.uniform(0.10, 0.50))
    else:
        echo_gain, echo_delay_s = 0.0, 0.0
    return AugmentationSpec(
        gain_db=gain_db,
        snr_db=snr_db,
        reverb=reverb,
        echo_gain=echo_gain,
        echo_delay_s=echo_delay_s,
    )


def _synthetic_rir(sample_rate: int, *, seed: int, length_s: float = 0.25) -> np.ndarray:
    length = max(1, int(round(length_s * sample_rate)))
    rng = np.random.default_rng(seed)
    times = np.arange(length, dtype=np.float64) / sample_rate
    decay = np.exp(-times / 0.06)
    tail = rng.normal(0.0, 1.0, size=length) * decay * 0.12
    rir = tail.astype(np.float32)
    rir[0] += 1.0
    for delay_s, gain in ((0.011, 0.35), (0.023, 0.22), (0.041, 0.14)):
        index = int(round(delay_s * sample_rate))
        if index < length:
            rir[index] += gain
    energy = float(np.sqrt(np.sum(np.square(rir, dtype=np.float64))))
    if energy > 0:
        rir /= energy
    return rir


def apply_augmentation(
    user: np.ndarray,
    sample_rate: int,
    spec: AugmentationSpec,
    *,
    agent: np.ndarray | None = None,
    seed: int = 0,
    peak_limit: bool = True,
) -> np.ndarray:
    """Apply gain/noise/reverb/agent-echo to the user channel.

    ``agent`` (same sample rate, same or longer length) is required only when
    ``spec.echo`` is true; it models the agent leaking into the user's
    microphone.  The input is never modified.  The result is peak-limited to
    0.99 when ``peak_limit`` is true.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    out = np.array(user, dtype=np.float64, copy=True).reshape(-1)
    if out.size == 0:
        return out.astype(np.float32)
    if not math.isfinite(spec.gain_db):
        raise ValueError("gain_db must be finite")

    if spec.gain_db:
        out *= 10.0 ** (spec.gain_db / 20.0)

    if spec.snr_db is not None:
        if not math.isfinite(spec.snr_db):
            raise ValueError("snr_db must be finite")
        signal_rms = float(np.sqrt(np.mean(np.square(out))))
        if signal_rms > 0:
            rng = np.random.default_rng(seed + 1)
            noise = rng.normal(0.0, 1.0, size=out.size)
            noise_rms = float(np.sqrt(np.mean(np.square(noise))))
            target_noise_rms = signal_rms * 10.0 ** (-spec.snr_db / 20.0)
            if noise_rms > 0:
                out += noise * (target_noise_rms / noise_rms)

    if spec.reverb:
        rir = _synthetic_rir(sample_rate, seed=seed + 2)
        out = np.convolve(out, rir, mode="full")[: out.size]

    if spec.echo:
        if agent is None:
            raise ValueError("agent audio is required for echo augmentation")
        reference = np.asarray(agent, dtype=np.float64).reshape(-1)
        if reference.size < out.size:
            reference = np.pad(reference, (0, out.size - reference.size))
        else:
            reference = reference[: out.size]
        delay = int(round(spec.echo_delay_s * sample_rate))
        if delay <= 0:
            raise ValueError("echo_delay_s must be positive when echo is enabled")
        if delay >= out.size:
            raise ValueError("echo delay exceeds the user channel length")
        echo = np.zeros_like(out)
        echo[delay:] = reference[:-delay] * spec.echo_gain
        out += echo

    if peak_limit:
        peak = float(np.max(np.abs(out)))
        if peak > 0.99:
            out *= 0.99 / peak
    return np.ascontiguousarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# TTS backends (all heavy imports lazy; real packages only)
# ---------------------------------------------------------------------------


@runtime_checkable
class TTSBackend(Protocol):
    name: str
    sample_rate: int
    agent_voice: str
    user_voices: tuple[str, ...]

    def synthesize(self, text: str, *, voice: str, seed: int = 0) -> np.ndarray:
        ...


def _as_numpy_audio(value: Any) -> np.ndarray:
    if isinstance(value, tuple) and value:
        value = value[-1]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    audio = np.asarray(value, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise BackendUnavailable("TTS returned empty audio")
    return audio


def _kokoro_result_audio(result: Any) -> np.ndarray:
    if hasattr(result, "audio"):
        return _as_numpy_audio(result.audio)
    if isinstance(result, (tuple, list)) and len(result) >= 3:
        return _as_numpy_audio(result[2])
    return _as_numpy_audio(result)


class KokoroBackend:
    """Kokoro-82M (Apache-2.0), 24 kHz English TTS."""

    name = "kokoro"
    sample_rate = 24000

    def __init__(self, *, lang_code: str = "a", speed: float = 1.0) -> None:
        self.lang_code = lang_code
        self.speed = speed
        self.agent_voice = AGENT_VOICE_BY_BACKEND["kokoro"]
        self.user_voices = USER_VOICES_BY_BACKEND["kokoro"]
        self._pipeline = None

    def check_available(self) -> None:
        try:
            from kokoro import KPipeline  # noqa: F401
        except ImportError as exc:
            raise InstallError(
                "Kokoro is not installed. Install it with: "
                'pip install "kokoro>=0.9.4" "misaki[en]" soundfile'
            ) from exc

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        self.check_available()
        from kokoro import KPipeline

        self._pipeline = KPipeline(lang_code=self.lang_code)

    def synthesize(self, text: str, *, voice: str, seed: int = 0) -> np.ndarray:
        del seed
        self._load()
        assert self._pipeline is not None
        chunks: list[np.ndarray] = []
        for result in self._pipeline(text, voice=voice, speed=self.speed, split_pattern=r"\n+"):
            chunks.append(_kokoro_result_audio(result))
        if not chunks:
            raise BackendUnavailable(f"Kokoro produced no audio for {text[:80]!r}")
        return np.concatenate(chunks).astype(np.float32, copy=False)


class DiaBackend:
    """Dia-1.6B (Apache-2.0), 44.1 kHz dialogue TTS.

    Dia renders speaker tags (``[S1]``/``[S2]``) itself; ``voice`` is expected
    to be one of those tags.
    """

    name = "dia"
    sample_rate = 44100

    def __init__(self, model_id: str = "nari-labs/Dia-1.6B-0626", *, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self.agent_voice = AGENT_VOICE_BY_BACKEND["dia"]
        self.user_voices = USER_VOICES_BY_BACKEND["dia"]
        self._model = None

    def check_available(self) -> None:
        try:
            from dia.model import Dia  # noqa: F401
        except ImportError as exc:
            raise InstallError(
                "Dia is not installed. Install it with: "
                "pip install git+https://github.com/nari-labs/dia.git"
            ) from exc

    def _load(self) -> None:
        if self._model is not None:
            return
        self.check_available()
        from dia.model import Dia

        try:
            self._model = Dia.from_pretrained(self.model_id)
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"cannot load Dia weights {self.model_id!r}: {exc}") from exc

    def synthesize(self, text: str, *, voice: str, seed: int = 0) -> np.ndarray:
        del seed
        self._load()
        assert self._model is not None
        tag = voice if voice in ("S1", "S2") else self.user_voices[0]
        try:
            output = self._model.generate(
                f"[{tag}] {text}", use_torch_compile=False, verbose=False
            )
        except TypeError:
            output = self._model.generate(f"[{tag}] {text}")
        return _as_numpy_audio(output)


class ChatterboxBackend:
    """Chatterbox (MIT), 24 kHz TTS with reference-audio voice prompting.

    A fixed agent voice requires a reference wav.  Supply
    ``voice_prompts={"agent": path, "user": path}``; missing prompts raise
    :class:`BackendUnavailable` instead of silently changing voices.
    """

    name = "chatterbox"
    sample_rate = 24000

    def __init__(
        self,
        *,
        voice_prompts: Mapping[str, str] | None = None,
        device: str | None = None,
    ) -> None:
        self.voice_prompts = dict(voice_prompts or {})
        self.device = device
        self.agent_voice = AGENT_VOICE_BY_BACKEND["chatterbox"]
        self.user_voices = USER_VOICES_BY_BACKEND["chatterbox"]
        self._model = None

    def check_available(self) -> None:
        try:
            from chatterbox.tts import ChatterboxTTS  # noqa: F401
        except ImportError as exc:
            raise InstallError(
                "Chatterbox is not installed. Install it with: pip install chatterbox-tts"
            ) from exc
        missing = [role for role in ("agent", "user") if not self.voice_prompts.get(role)]
        if missing:
            raise BackendUnavailable(
                "Chatterbox needs reference wavs for fixed voices; missing "
                f"{missing}. Pass voice_prompts={{'agent': '...', 'user': '...'}} "
                "via --backend-options."
            )
        for role, prompt in self.voice_prompts.items():
            if not Path(prompt).exists():
                raise BackendUnavailable(f"Chatterbox {role} reference wav not found: {prompt}")

    def _load(self) -> None:
        if self._model is not None:
            return
        self.check_available()
        from chatterbox.tts import ChatterboxTTS

        try:
            device = self.device or "cpu"
            self._model = ChatterboxTTS.from_pretrained(device=device)
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"cannot load Chatterbox weights: {exc}") from exc
        self.sample_rate = int(getattr(self._model, "sr", 24000))

    def synthesize(self, text: str, *, voice: str, seed: int = 0) -> np.ndarray:
        del seed
        self._load()
        assert self._model is not None
        prompt = self.voice_prompts.get(voice)
        if not prompt:
            raise BackendUnavailable(
                f"Chatterbox voice {voice!r} needs a reference wav; "
                "pass voice_prompts={'agent': '...', 'user': '...'}"
            )
        if not Path(prompt).exists():
            raise BackendUnavailable(f"Chatterbox reference wav not found: {prompt}")
        return _as_numpy_audio(self._model.generate(text, audio_prompt_path=prompt))


class CosyVoice2Backend:
    """CosyVoice2 (Apache-2.0), 24 kHz TTS from a local model directory.

    The upstream API expects a speaker id from the model's ``spk2info``.
    Supply ``speaker_ids={"agent": "...", "user": "..."}`` from that model.
    """

    name = "cosyvoice2"

    def __init__(
        self,
        model_dir: str,
        *,
        speaker_ids: Mapping[str, str] | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_dir = model_dir
        self.speaker_ids = dict(speaker_ids or {})
        self.device = device
        self.agent_voice = AGENT_VOICE_BY_BACKEND["cosyvoice2"]
        self.user_voices = USER_VOICES_BY_BACKEND["cosyvoice2"]
        self.sample_rate = 24000
        self._model = None

    def check_available(self) -> None:
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: F401
        except ImportError as exc:
            raise InstallError(
                "CosyVoice2 is not installed. Install it from "
                "https://github.com/FunAudioLLM/CosyVoice and set PYTHONPATH."
            ) from exc
        missing = [role for role in ("agent", "user") if not self.speaker_ids.get(role)]
        if missing:
            raise BackendUnavailable(
                f"CosyVoice2 speaker ids missing for {missing}; pass "
                "speaker_ids={'agent': '...', 'user': '...'} via --backend-options."
            )

    def _load(self) -> None:
        if self._model is not None:
            return
        self.check_available()
        from cosyvoice.cli.cosyvoice import CosyVoice2

        try:
            self._model = CosyVoice2(
                self.model_dir, load_jit=False, load_trt=False, fp16=False
            )
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(
                f"cannot load CosyVoice2 from {self.model_dir!r}: {exc}"
            ) from exc
        self.sample_rate = int(getattr(self._model, "sample_rate", 24000))

    def synthesize(self, text: str, *, voice: str, seed: int = 0) -> np.ndarray:
        del seed
        self._load()
        assert self._model is not None
        speaker = self.speaker_ids.get(voice)
        if not speaker:
            raise BackendUnavailable(
                f"CosyVoice2 speaker for {voice!r} is not configured; "
                "pass speaker_ids={'agent': '...', 'user': '...'}"
            )
        chunks = []
        for output in self._model.inference_sft(text, speaker, stream=False):
            chunks.append(_as_numpy_audio(output["tts_speech"]))
        if not chunks:
            raise BackendUnavailable(f"CosyVoice2 produced no audio for {text[:80]!r}")
        return np.concatenate(chunks).astype(np.float32, copy=False)


def create_backend(name: str, *, options: Mapping[str, Any] | None = None) -> TTSBackend:
    """Instantiate a real TTS backend by name."""
    options = dict(options or {})
    if name == "kokoro":
        return KokoroBackend(**options)
    if name == "dia":
        return DiaBackend(**options)
    if name == "chatterbox":
        return ChatterboxBackend(**options)
    if name == "cosyvoice2":
        if "model_dir" not in options:
            raise BackendUnavailable("CosyVoice2 requires --backend-dir / model_dir")
        return CosyVoice2Backend(**options)
    raise ValueError(f"unknown TTS backend {name!r}; choose kokoro, dia, chatterbox, cosyvoice2")


# ---------------------------------------------------------------------------
# Voice similarity
# ---------------------------------------------------------------------------


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity of two embedding vectors."""
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        raise ValueError("embeddings must be non-empty and the same size")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


class ECAPAVoiceChecker:
    """ECAPA-TDNN speaker encoder from ``speechbrain`` (optional dependency)."""

    def __init__(
        self,
        *,
        source: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.source = source
        self.savedir = str(savedir or Path.home() / ".cache" / "repartee" / "ecapa")
        self.device = device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from speechbrain.inference.speaker import SpeakerRecognition
        except ImportError:
            try:
                from speechbrain.inference import SpeakerRecognition  # type: ignore[no-redef]
            except ImportError as exc:
                raise InstallError(
                    "Voice similarity checking needs speechbrain: "
                    "pip install speechbrain. Use --no-voice-check to build "
                    "without it (this limitation is recorded in the manifest)."
                ) from exc
        try:
            self._model = SpeakerRecognition.from_hparams(
                source=self.source, savedir=self.savedir, run_opts={"device": self.device}
            )
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"cannot load ECAPA model {self.source!r}: {exc}") from exc

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self._load()
        assert self._model is not None
        import torch

        from . import audio as audio_utils

        float_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != 16000:
            float_audio = audio_utils.resample(float_audio, sample_rate, 16000)
        tensor = torch.from_numpy(float_audio).unsqueeze(0)
        device = next(self._model.parameters()).device
        with torch.no_grad():
            embedding = self._model.encode_batch(tensor.to(device))
        return embedding.detach().cpu().numpy().reshape(-1)

    def similarity(self, audio: np.ndarray, reference: np.ndarray, sample_rate: int) -> float:
        return cosine_similarity(self.embed(audio, sample_rate), self.embed(reference, sample_rate))


# ---------------------------------------------------------------------------
# Channel rendering
# ---------------------------------------------------------------------------


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.ascontiguousarray(audio, dtype=np.float32)
    from . import audio as audio_utils

    return audio_utils.resample(audio, source_rate, target_rate)


def _fit(audio: np.ndarray, samples: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples <= 0:
        return np.zeros(0, dtype=np.float32)
    if audio.size >= samples:
        return np.ascontiguousarray(audio[:samples])
    return np.pad(audio, (0, samples - audio.size))


def render_channels(
    placements: Sequence[Placement],
    *,
    utterance_audio: Sequence[np.ndarray],
    backchannel_audio: Mapping[str, np.ndarray],
    backend_rate: int,
    tail_s: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Render placements into 16 kHz user and 24 kHz agent channels.

    Returns ``(user, agent, duration_s)`` where both channels have exactly the
    same number of 80 ms frames.
    """
    end_s = timeline_end(placements) + tail_s
    frames = max(1, int(math.ceil(end_s * schema.FRAME_RATE - 1e-9)))
    duration_s = frames / schema.FRAME_RATE
    user = np.zeros(frames * schema.FRAME_SAMPLES[schema.USER_SAMPLE_RATE], dtype=np.float32)
    agent = np.zeros(frames * schema.FRAME_SAMPLES[schema.AGENT_SAMPLE_RATE], dtype=np.float32)
    for placement in placements:
        if placement.kind == "backchannel":
            source = backchannel_audio.get(placement.text)
        else:
            source = utterance_audio[placement.source_index]
        if source is None:
            raise SynthError(f"no rendered audio for placement {placement!r}")
        target_rate = schema.USER_SAMPLE_RATE if placement.speaker == "user" else schema.AGENT_SAMPLE_RATE
        channel = user if placement.speaker == "user" else agent
        rendered = _resample(source, backend_rate, target_rate)
        speech_samples = int(round(placement.duration * target_rate))
        rendered = _fit(rendered, speech_samples)
        offset = int(round(placement.start * target_rate))
        if offset < 0:
            raise SynthError("negative placement start")
        if offset + speech_samples > channel.size:
            raise SynthError("placement exceeds rendered channel length")
        channel[offset : offset + speech_samples] += rendered
    return user, agent, duration_s


# ---------------------------------------------------------------------------
# Dialogue rendering
# ---------------------------------------------------------------------------


def _estimate_seconds(text: str) -> float:
    words = max(1, len(text.split()))
    return max(0.35, words / 2.6 + 0.30)


_THREAD_CONTEXT = threading.local()


def _thread_context(backend_name: str, backend_options: Mapping[str, Any], voice_check: bool, device: str):
    local = _THREAD_CONTEXT
    if (
        not hasattr(local, "backend")
        or local.backend_name != backend_name
        or local.voice_check != voice_check
    ):
        local.backend = create_backend(backend_name, options=backend_options)
        local.backend_name = backend_name
        local.voice_check = voice_check
        local.checker = ECAPAVoiceChecker(device=device) if voice_check else None
        local.reference = None
        local.reference_embedding = None
    return local


def _render_dialogue(
    dialogue: Dialogue,
    *,
    backend_name: str,
    backend_options: Mapping[str, Any],
    voice_check: bool,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[schema.Turn], float, AugmentationSpec, dict[str, Any]]:
    """Synthesize, compose, augment and return one dialogue's audio + turns."""
    context = _thread_context(backend_name, backend_options, voice_check, device)
    backend = context.backend
    user_voice = dialogue.user_voice or backend.user_voices[
        _stable_digest(dialogue.id) % len(backend.user_voices)
    ]
    agent_voice = dialogue.agent_voice or backend.agent_voice

    utterance_audio: list[np.ndarray] = []
    for index, utterance in enumerate(dialogue.utterances):
        voice = user_voice if utterance.speaker == "user" else agent_voice
        utterance_audio.append(
            backend.synthesize(utterance.text, voice=voice, seed=dialogue.seed + index)
        )
    backchannel_audio = {
        text: backend.synthesize(text, voice=agent_voice, seed=dialogue.seed + 100 + index)
        for index, (text, _) in enumerate(DEFAULT_BACKCHANNELS)
    }
    options = tuple(
        (text, len(backchannel_audio[text]) / backend.sample_rate)
        for text, _ in DEFAULT_BACKCHANNELS
    )

    if voice_check:
        checker = context.checker
        assert checker is not None
        if context.reference is None:
            reference_text = "Hello, I am your conversational assistant. How can I help today?"
            context.reference = backend.synthesize(
                reference_text, voice=agent_voice, seed=dialogue.seed + 999
            )
            context.reference_embedding = checker.embed(
                context.reference, backend.sample_rate
            )
        for index, utterance in enumerate(dialogue.utterances):
            if utterance.speaker != "agent":
                continue
            similarity = cosine_similarity(
                checker.embed(utterance_audio[index], backend.sample_rate),
                context.reference_embedding,
            )
            if similarity < 0.80:
                raise VoiceCheckError(
                    f"{dialogue.id}: agent voice similarity {similarity:.3f} < 0.80 "
                    f"on utterance {index}"
                )

    durations = [audio.size / backend.sample_rate for audio in utterance_audio]
    timing = [
        TimingTurn(utterance.speaker, duration, utterance.text)
        for utterance, duration in zip(dialogue.utterances, durations)
    ]
    placements = compose_timeline(
        timing,
        seed=dialogue.seed,
        backchannel_options=options,
    )
    user, agent, duration_s = render_channels(
        placements,
        utterance_audio=utterance_audio,
        backchannel_audio=backchannel_audio,
        backend_rate=backend.sample_rate,
    )

    spec = sample_augmentation(dialogue.seed + 7)
    agent_for_echo = _resample(agent, schema.AGENT_SAMPLE_RATE, schema.USER_SAMPLE_RATE)
    augmented_user = apply_augmentation(
        user,
        schema.USER_SAMPLE_RATE,
        spec,
        agent=agent_for_echo if spec.echo else None,
        seed=dialogue.seed + 11,
    )
    augmented_user = _fit(augmented_user, user.size)

    turns = [
        schema.Turn(
            speaker=placement.speaker,
            start=round(placement.start, 4),
            end=round(placement.end, 4),
            text=placement.text,
        )
        for placement in placements
    ]
    turns = sorted(turns, key=lambda turn: (turn.start, turn.end, turn.speaker))
    meta = {
        "category": dialogue.category,
        "persona": dialogue.persona,
        "writer": "local-hf",
        "tts_backend": backend_name,
        "agent_voice": agent_voice,
        "user_voice": user_voice,
        "voice_check": voice_check,
        "augmentation": {
            "gain_db": spec.gain_db,
            "snr_db": spec.snr_db,
            "reverb": spec.reverb,
            "echo_gain": spec.echo_gain,
            "echo_delay_s": spec.echo_delay_s,
        },
        "timeline": [
            {
                "speaker": placement.speaker,
                "text": placement.text,
                "kind": placement.kind,
                "start": placement.start,
                "end": placement.end,
                "nominal_end": placement.nominal_end,
                "pad_after": placement.pad_after,
            }
            for placement in placements
        ],
    }
    return augmented_user, agent, turns, duration_s, spec, meta


def _write_dialogue(
    *,
    dialogue: Dialogue,
    user: np.ndarray,
    agent: np.ndarray,
    turns: Sequence[schema.Turn],
    duration_s: float,
    meta: dict[str, Any],
    raw_root: Path,
) -> schema.Conversation:
    from . import audio as audio_utils

    directory = raw_root / "synth" / dialogue.id
    directory.mkdir(parents=True, exist_ok=True)
    user_path = directory / "user.flac"
    agent_path = directory / "agent.flac"
    audio_utils.write_audio(user_path, user, schema.USER_SAMPLE_RATE)
    audio_utils.write_audio(agent_path, agent, schema.AGENT_SAMPLE_RATE)
    conversation = schema.Conversation(
        id=dialogue.id,
        source=SYNTH_SOURCE,
        license=SYNTH_LICENSE,
        duration_s=round(duration_s, 6),
        channels={"user": str(user_path), "agent": str(agent_path)},
        turns=list(turns),
        split="train",
        meta=dict(meta),
    )
    conversation.validate()
    record_path = directory / "record.json"
    temp_path = directory / ".record.json.tmp"
    temp_path.write_text(json.dumps(conversation.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(record_path)
    return conversation


def _load_completed(raw_root: Path, seed: int) -> tuple[int, set[int], float]:
    """Scan completed dialogue sidecars without holding them all in memory."""
    completed = 0
    completed_indices: set[int] = set()
    hours = 0.0
    root = raw_root / "synth"
    if not root.exists():
        return completed, completed_indices, hours
    pattern = re.compile(rf"^synth-{re.escape(str(seed))}-(\d{{7}})$")
    for record_path in root.glob("*/record.json"):
        match = pattern.match(record_path.parent.name)
        if not match:
            continue
        try:
            conversation = schema.Conversation.from_dict(
                json.loads(record_path.read_text(encoding="utf-8"))
            )
        except Exception as exc:  # noqa: BLE001 - a corrupt sidecar is a real error
            raise SynthError(f"{record_path}: invalid record sidecar: {exc}") from exc
        if not all(Path(path).exists() for path in conversation.channels.values()):
            continue
        completed += 1
        completed_indices.add(int(match.group(1)))
        hours += conversation.duration_s / 3600.0
    return completed, completed_indices, hours


def _write_manifest(out_dir: Path, raw_root: Path, seed: int) -> tuple[Path, int, float]:
    """Stream completed dialogue sidecars into the combined manifest."""
    path = out_dir / SYNTH_MANIFEST
    temp = out_dir / f".{SYNTH_MANIFEST}.tmp"
    pattern = re.compile(rf"^synth-{re.escape(str(seed))}-(\d{{7}})$")
    record_paths = sorted(
        (candidate for candidate in (raw_root / "synth").glob("*/record.json")
         if pattern.match(candidate.parent.name)),
        key=lambda candidate: candidate.parent.name,
    )
    count = 0
    hours = 0.0
    with temp.open("w", encoding="utf-8") as handle:
        for record_path in record_paths:
            try:
                conversation = schema.Conversation.from_dict(
                    json.loads(record_path.read_text(encoding="utf-8"))
                )
            except Exception as exc:  # noqa: BLE001
                raise SynthError(f"{record_path}: invalid record sidecar: {exc}") from exc
            if not all(Path(channel).exists() for channel in conversation.channels.values()):
                continue
            conversation.validate()
            handle.write(json.dumps(conversation.to_dict(), ensure_ascii=False) + "\n")
            count += 1
            hours += conversation.duration_s / 3600.0
    temp.replace(path)
    return path, count, hours


def dry_run_plan(
    *,
    hours: float,
    seed: int,
    dialogues: int,
    writer: DialogueWriter | None = None,
) -> dict[str, Any]:
    """Emit text + timing plans without audio or manifests."""
    writer = writer or ScriptedDialogueWriter(seed=seed)
    estimated_per_dialogue = 60.0
    planned = max(1, int(math.ceil(hours * 3600.0 / estimated_per_dialogue)))
    examples = []
    total_s = 0.0
    for index in range(min(dialogues, planned)):
        category = category_for_index(index, seed=seed)
        dialogue_id = f"synth-{seed}-{index:07d}"
        dialogue = writer.write(dialogue_id, category=category, seed=seed * 1_000_003 + index)
        timings = [TimingTurn(u.speaker, _estimate_seconds(u.text), u.text) for u in dialogue.utterances]
        placements = compose_timeline(timings, seed=dialogue.seed)
        duration = timeline_end(placements)
        total_s += duration
        examples.append(
            {
                "id": dialogue.id,
                "category": dialogue.category,
                "persona": dialogue.persona,
                "turns": [
                    {"speaker": u.speaker, "text": u.text} for u in dialogue.utterances
                ],
                "timeline": [
                    {
                        "speaker": p.speaker,
                        "kind": p.kind,
                        "text": p.text,
                        "start": p.start,
                        "end": p.end,
                        "nominal_end": p.nominal_end,
                        "pad_after": p.pad_after,
                    }
                    for p in placements
                ],
                "estimated_duration_s": round(duration, 3),
            }
        )
    return {
        "dry_run": True,
        "audio": False,
        "manifest": False,
        "warning": "dry run: no audio was rendered and no manifest was written",
        "target_hours": hours,
        "planned_dialogues_estimate": planned,
        "category_mix": CATEGORY_WEIGHTS,
        "examples": examples,
        "example_hours": round(total_s / 3600.0, 4),
    }


# ---------------------------------------------------------------------------
# Real build
# ---------------------------------------------------------------------------


def build(
    *,
    out_dir: Path,
    raw_root: Path,
    hours: float,
    backend_name: str,
    writers: int = 1,
    seed: int = 0,
    writer: DialogueWriter | None = None,
    backend_options: Mapping[str, Any] | None = None,
    voice_check: bool = True,
    device: str = "cpu",
    max_failures: int = 20,
    printer: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Generate synthetic dialogues until ``hours`` of completed records exist."""
    if hours <= 0:
        raise ValueError("hours must be positive")
    if writers < 1:
        raise ValueError("writers must be >= 1")
    backend_options = dict(backend_options or {})
    # Validate backend name/configuration before generating anything.
    probe = create_backend(backend_name, options=backend_options)
    if writer is None:
        raise SynthError(
            "a DialogueWriter is required for a real build; pass --writer-model "
            "or use --dry-run"
        )
    if getattr(writer, "is_scripted", False):
        raise SynthError("ScriptedDialogueWriter is dry-run only and cannot produce training data")
    check = getattr(probe, "check_available", None)
    if callable(check):
        check()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    if voice_check:
        checker = ECAPAVoiceChecker(device=device)
        checker._load()
    printer(f"backend={probe.name} rate={probe.sample_rate} voice={probe.agent_voice}")

    completed_count, completed_indices, completed_hours = _load_completed(raw_root, seed)
    printer(f"resume: {completed_count} dialogue(s), {completed_hours:.3f} h already complete")
    failures = 0
    index = 0
    chunk_size = max(1, writers)

    def render(dialogue: Dialogue):
        return (
            dialogue,
            _render_dialogue(
                dialogue,
                backend_name=backend_name,
                backend_options=backend_options,
                voice_check=voice_check,
                device=device,
                seed=seed,
            ),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=writers) as pool:
        while completed_hours < hours:
            dialogues: list[Dialogue] = []
            while len(dialogues) < chunk_size and completed_hours < hours:
                while index in completed_indices:
                    index += 1
                category = category_for_index(index, seed=seed)
                dialogue_id = f"synth-{seed}-{index:07d}"
                dialogue_seed = seed * 1_000_003 + index
                try:
                    dialogue = writer.write(dialogue_id, category=category, seed=dialogue_seed)
                    if dialogue.id != dialogue_id:
                        raise DialogueError(
                            f"writer returned id {dialogue.id!r}, expected {dialogue_id!r}"
                        )
                    dialogues.append(dialogue)
                except DialogueError as exc:
                    failures += 1
                    printer(f"  writer failure index={index}: {exc}")
                    if failures > max_failures:
                        raise SynthError(
                            f"aborting after {failures} generation failures; last: {exc}"
                        ) from exc
                index += 1
            if not dialogues:
                continue

            futures = [pool.submit(render, dialogue) for dialogue in dialogues]
            for future in concurrent.futures.as_completed(futures):
                try:
                    dialogue, (user, agent, turns, duration_s, spec, meta) = future.result()
                    meta = dict(meta)
                    meta.setdefault("writer_model", getattr(writer, "model_id", "unknown"))
                    conversation = _write_dialogue(
                        dialogue=dialogue,
                        user=user,
                        agent=agent,
                        turns=turns,
                        duration_s=duration_s,
                        meta=meta,
                        raw_root=raw_root,
                    )
                except VoiceCheckError as exc:
                    failures += 1
                    printer(f"  voice reject: {exc}")
                    if failures > max_failures:
                        raise SynthError(f"aborting after {failures} failures") from exc
                    continue
                except (InstallError, BackendUnavailable):
                    raise
                except Exception as exc:  # noqa: BLE001 - keep the batch alive, count it
                    failures += 1
                    printer(f"  render failure: {exc}")
                    if failures > max_failures:
                        raise SynthError(f"aborting after {failures} failures") from exc
                    continue
                else:
                    failures = 0
                completed_count += 1
                completed_hours += conversation.duration_s / 3600.0
                printer(
                    f"  {dialogue.id}: {conversation.duration_s:.2f}s "
                    f"({completed_hours:.3f}/{hours:.3f} h)"
                )

    path, total_count, total_hours = _write_manifest(out_dir, raw_root, seed)
    printer(f"wrote {total_count} dialogue(s), {total_hours:.3f} h -> {path}")
    return {
        "dialogues": total_count,
        "hours": total_hours,
        "manifest": str(path),
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _optional_json(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    stripped = value.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    path = Path(value)
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/manifests"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--hours", type=float, default=8000.0)
    parser.add_argument(
        "--backend",
        choices=("kokoro", "dia", "chatterbox", "cosyvoice2"),
        default="kokoro",
    )
    parser.add_argument("--writers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-dialogues", type=int, default=3)
    parser.add_argument(
        "--writer-model",
        default=os.environ.get("REPARTEE_WRITER_MODEL"),
        help="local path or Hugging Face id for the dialogue LLM (required for real builds)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--voice-check", dest="voice_check", action="store_true", default=True)
    parser.add_argument("--no-voice-check", dest="voice_check", action="store_false")
    parser.add_argument(
        "--backend-options",
        default=None,
        help="JSON file or inline JSON object passed to the backend constructor",
    )
    parser.add_argument("--max-failures", type=int, default=20)
    args = parser.parse_args(argv)
    if args.hours <= 0:
        parser.error("--hours must be positive")
    if args.writers < 1:
        parser.error("--writers must be >= 1")
    if args.dry_run_dialogues < 1:
        parser.error("--dry-run-dialogues must be >= 1")

    if args.dry_run:
        plan = dry_run_plan(
            hours=args.hours,
            seed=args.seed,
            dialogues=args.dry_run_dialogues,
        )
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    if not args.writer_model:
        # Still surface a missing TTS dependency first: the requested backend
        # is what users usually need to install.
        try:
            options = _optional_json(args.backend_options)
            probe = create_backend(args.backend, options=options or {})
            check = getattr(probe, "check_available", None)
            if callable(check):
                check()
        except InstallError as exc:
            print(f"missing dependency: {exc}", file=sys.stderr)
            return 3
        except BackendUnavailable as exc:
            print(f"backend unavailable: {exc}", file=sys.stderr)
            return 4
        except (OSError, json.JSONDecodeError) as exc:
            print(f"config error: --backend-options is invalid: {exc}", file=sys.stderr)
            return 2
        print(
            "config error: real synthesis needs --writer-model MODEL_OR_LOCAL_PATH "
            "(or REPARTEE_WRITER_MODEL). Use --dry-run to inspect planning only.",
            file=sys.stderr,
        )
        return 2
    try:
        options = _optional_json(args.backend_options)
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"--backend-options is not valid JSON/path: {exc}")
    try:
        build(
            out_dir=args.out,
            raw_root=args.raw,
            hours=args.hours,
            backend_name=args.backend,
            writers=args.writers,
            seed=args.seed,
            writer=LocalHFDialogueWriter(
                args.writer_model,
                device=args.device,
            ),
            backend_options=options or {},
            voice_check=args.voice_check,
            device=args.device,
            max_failures=args.max_failures,
        )
    except InstallError as exc:
        print(f"missing dependency: {exc}", file=sys.stderr)
        return 3
    except BackendUnavailable as exc:
        print(f"backend unavailable: {exc}", file=sys.stderr)
        return 4
    except SynthError as exc:
        print(f"synthesis failed: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"synthesis failed (I/O): {exc}", file=sys.stderr)
        return 2
    return 0


__all__ = [
    "CATEGORIES",
    "CATEGORY_WEIGHTS",
    "DEFAULT_GAP_S",
    "DEFAULT_GAP_PROBABILITY",
    "DEFAULT_BARGE_IN_PROBABILITY",
    "DEFAULT_BACKCHANNEL_PROBABILITY",
    "DEFAULT_OVERLAP_PROBABILITY",
    "DEFAULT_BACKCHANNELS",
    "AGENT_PERSONA",
    "SYNTH_SOURCE",
    "SYNTH_LICENSE",
    "SYNTH_MANIFEST",
    "SynthError",
    "InstallError",
    "BackendUnavailable",
    "DialogueError",
    "VoiceCheckError",
    "Utterance",
    "Dialogue",
    "DialogueWriter",
    "ScriptedDialogueWriter",
    "LocalHFDialogueWriter",
    "parse_dialogue_text",
    "dialogue_mix",
    "category_for_index",
    "persona_for",
    "TimingTurn",
    "Placement",
    "compose_timeline",
    "timeline_end",
    "AugmentationSpec",
    "sample_augmentation",
    "apply_augmentation",
    "TTSBackend",
    "KokoroBackend",
    "DiaBackend",
    "ChatterboxBackend",
    "CosyVoice2Backend",
    "create_backend",
    "cosine_similarity",
    "ECAPAVoiceChecker",
    "render_channels",
    "dry_run_plan",
    "build",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
