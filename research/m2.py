#!/usr/bin/env python3
"""M2: one generation, several text blocks — no end-of-speech, no re-prefill.

Phrase 1 is spoken with the audio end-of-speech code masked; the generator stops at an
audio-block boundary whose last frames are quiet, the next phrase's text is fed into
the *same* cache, and the audio continues. The seam is inside the model.

Measured: lead silence before phrase 2, per-phrase ASR recall, total length, and
whether masking EOS makes it run past the sentence.

  .venv/bin/python research/m2.py --out data/m2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import config  # noqa: E402
from seam_test import Actor  # noqa: E402

MODEL = config.ASR_LFM3_MODEL
SR_A = 24_000
FRAME_SAMPLES = int(0.08 * SR_A)
BLOCK = 12
PHRASE1 = "One more thing, the Harrow project was cancelled last Friday."
PHRASE2 = "Its budget moved to the satellite team."


def disable_eos(model) -> None:
    """Suppress the audio end-of-speech code in the logits, so the stream never closes."""
    cls = type(model)
    original = cls._sample.__func__
    eos = model.config.audio_eos_code

    def patched(cls_, logits, temperature, top_k, repetition_penalty=None,
                history=None, suppress=(), **kwargs):
        return original(cls_, logits, temperature, top_k, repetition_penalty,
                        history, (*suppress, eos), **kwargs)

    cls._sample = classmethod(patched)


def masked_frame(actor: Actor) -> bool:
    """One audio frame; end-of-speech cannot be sampled while it is suppressed."""
    hist = torch.stack(actor.frames[-64:]) if actor.frames else None
    codes = actor.model.sample_audio_frame(
        actor.hidden[0], temperature=0.4, top_k=128,
        repetition_penalty=1.2, history=hist,
    )
    actor.frames.append(codes)
    actor._feed(actor.model.model.get_audio_out_embeds(codes[None]))
    return False


def loud_tail(actor: Actor, frames: int = 2, thr: float = 0.01) -> bool:
    """True when the last `frames` frames hold speech (i.e. not a quiet boundary)."""
    if len(actor.frames) < frames:
        return True
    wav = actor.samples(actor.frames[-frames:])
    rms = np.sqrt((wav ** 2).mean()) if len(wav) else 0.0
    return float(rms) > thr


def budget_frames(text: str, wps: float = 2.6) -> int:
    """Frames a phrase needs at the actor's measured rate, plus a margin."""
    words = len(re.findall(r"[a-z0-9']+", text.lower()))
    return int(round((words / wps) / 0.08 * 1.15))


def speak_until_pause(actor: Actor, min_frames: int, max_blocks: int = 10) -> dict:
    produced = 0
    for block in range(max_blocks):
        for _ in range(BLOCK):
            if masked_frame(actor):
                return {"frames": produced, "end": "eos_stuck"}
            produced += 1
        if produced >= min_frames and not loud_tail(actor):
            return {"frames": produced, "end": "quiet_boundary"}
    return {"frames": produced, "end": "cap"}


def lead_ms(wav: np.ndarray, thr: float = 0.01) -> int:
    hop = int(0.01 * SR_A)
    n = len(wav) // hop
    if n == 0:
        return 0
    rms = np.sqrt((wav[: n * hop].reshape(n, hop) ** 2).mean(1))
    loud = np.flatnonzero(rms > thr)
    return int(loud[0]) * 10 if len(loud) else n * 10


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def recall(ref: str, hyp: str) -> float:
    import difflib

    a, b = words(ref), words(hyp)
    if not a:
        return 0.0
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return round(sum(x.size for x in sm.get_matching_blocks()) / len(a), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "m2")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    actor = Actor()
    disable_eos(actor.model)

    def start(text: str, budget: int, tries: int = 4) -> dict:
        """The renderer occasionally stalls at frame 0; a fresh seedling fixes it."""
        for attempt in range(tries):
            actor.prefill_messages([
                {"role": "system", "content": "Respond with interleaved text and audio."},
                {"role": "user", "content": "Continue the conversation."},
            ])
            actor.force_text(text)
            got = speak_until_pause(actor, budget)
            if got["frames"] > 0:
                got["attempt"] = attempt
                return got
            print({"retry": text[:20], "attempt": attempt, "end": got["end"]}, flush=True)
        raise SystemExit("model stalled at frame 0 for every attempt")

    first = start(PHRASE1, budget_frames(PHRASE1))
    junction = len(actor.frames)
    actor.force_text(PHRASE2)
    second = speak_until_pause(actor, budget_frames(PHRASE2))

    wav = actor.samples()
    p1 = actor.samples(actor.frames[:junction])
    p2 = actor.samples(actor.frames[junction:])
    if len(p1) == 0 or len(p2) == 0:
        raise SystemExit(f"empty segment: p1={len(p1)} p2={len(p2)}")
    sf.write(args.out / "m2_full.wav", wav, SR_A)
    sf.write(args.out / "m2_phrase1.wav", p1, SR_A)
    sf.write(args.out / "m2_phrase2.wav", p2, SR_A)

    from app.asr import QwenTranscriber
    judge = QwenTranscriber()
    hyp1 = judge.transcribe(args.out / "m2_phrase1.wav").text
    hyp2 = judge.transcribe(args.out / "m2_phrase2.wav").text

    report = {
        "phrase1": {**first, "s": round(len(p1) / SR_A, 2), "lead_ms": lead_ms(p1),
                    "recall": recall(PHRASE1, hyp1), "asr": hyp1},
        "seam": {"lead_ms": lead_ms(p2),
                 "gap_at_junction": round(junction * 0.08, 2)},
        "phrase2": {**second, "s": round(len(p2) / SR_A, 2),
                    "recall": recall(PHRASE2, hyp2), "asr": hyp2},
        "total_s": round(len(wav) / SR_A, 2),
    }
    (args.out / "m2_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
