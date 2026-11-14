#!/usr/bin/env python3
"""Full loop, third iteration: cue that fits the conversation, capped turns.

  user audio -> actor samples a short reply (text) and speaks it in phrases
  -> topic-shift cue -> injected fact in phrases -> user audio -> actor reacts

Every spoken agent turn goes through the same phrase renderer: clean turn, <=7
words, punctuation-terminated, so the voice, rate and endings are consistent
between the actor's own words and the injected text.

  .venv/bin/python research/full_flow.py --out data/flow4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import config  # noqa: E402
from seam_test import Actor  # noqa: E402

MODEL = config.ASR_LFM3_MODEL
SR_A = 24_000
SR_U = 16_000
PAUSE_HANDOVER_S = 0.12
PHRASE_GAP_S = 0.06
TRIM_GUARD_MS = 60
TRIM_THR = 0.003
FACT = ("The Harrow project was cancelled last Friday. "
        "Its budget moved to the satellite team.")
CUE = "One more thing,"
MAX_WORDS = 10  # comma/sentence clauses first; only split mid-clause above this


def split_phrases(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Clauses first (commas and sentence ends), so phrases end where prosody does.

    A clause longer than max_words is split at the last comma inside the window, or
    at the word boundary; that is the only place the renderer can be asked to stop
    mid-thought, and it is what caused repeated words at phrase edges.
    """
    clauses: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        for clause in re.split(r"(?<=,)\s+", sentence.strip()):
            if clause:
                clauses.append(clause.strip())
    phrases: list[str] = []
    for clause in clauses:
        words = clause.split()
        while words:
            take = min(max_words, len(words))
            for k in range(take, 3, -1):
                if words[k - 1].endswith(","):
                    take = k
                    break
            phrases.append(" ".join(words[:take]))
            words = words[take:]
    return [p for p in phrases if p]


def choose_cue(fact: str, context: str) -> str:
    """A topic-shift cue; a bridge when the fact shares vocabulary with the talk."""
    stop = {"the", "and", "its", "was", "were", "to", "a", "of", "in", "on", "for"}
    def keys(t: str) -> set[str]:
        return {w for w in re.findall(r"[a-z]+", t.lower()) if w not in stop and len(w) > 3}
    if keys(fact) & keys(context):
        return "Speaking of that,"
    return CUE


class Flow:
    def __init__(self) -> None:
        self.proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            MODEL, dtype=torch.float16, trust_remote_code=True
        ).to("cuda").eval()

    def reply_text(self, wav16: np.ndarray, max_new: int = 200) -> str:
        """The actor's own words, taken from its interleaved pass (text-only echoes).

        The audio it generates here is discarded; the text goes through the phrase
        renderer so every agent turn shares one voice, rate and ending style.
        """
        messages = [
            {"role": "system", "content": "Respond with interleaved text and audio."},
            {"role": "user", "content": [{"type": "audio"}]},
        ]
        prompt = self.proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.proc(text=prompt, audio=[wav16], sampling_rate=SR_U,
                           return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new, generation_mode="interleaved"
            )
        seq = out.sequences if hasattr(out, "sequences") else out
        text = self.proc.batch_decode(
            seq[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
        complete = [x for x in sentences if re.search(r"[.!?]$", x)]
        keep = complete[:2] if complete else sentences[:1]
        return " ".join(keep).strip()

    def speak(self, text: str, cue: str | None = None, single: bool = False) -> np.ndarray:
        """Render text. single=True keeps one turn (no internal pause, tail risk);
        single=False uses one clean turn per phrase (full words, intra pause)."""
        actor = Actor(self.model, self.proc)
        parts = []
        phrases = split_phrases(text)
        if single:
            phrases = [text]
        if cue and phrases:
            phrases = [f"{cue} {phrases[0]}".strip()] + phrases[1:]
        for phrase in phrases:
            actor.prefill_messages([
                {"role": "system", "content": "Respond with interleaved text and audio."},
                {"role": "user", "content": "Continue the conversation."},
            ])
            actor.force_text(phrase)
            actor.speak(80 if single else 50)
            parts.append(trim_silence(actor.samples()))
        gap = np.zeros(int(PHRASE_GAP_S * SR_A), dtype=np.float32)
        seq = []
        for part in parts:
            seq += [part, gap]
        return np.concatenate(seq[:-1])


def trim_silence(audio: np.ndarray, thr: float = TRIM_THR,
                 guard_ms: int = TRIM_GUARD_MS) -> np.ndarray:
    """Drop the model's onset/offset padding; keep a guard so onsets are not clipped."""
    hop = int(0.01 * SR_A)
    n = len(audio) // hop
    if n == 0:
        return audio
    rms = np.sqrt((audio[: n * hop].reshape(n, hop) ** 2).mean(1))
    loud = np.flatnonzero(rms > thr)
    if len(loud) == 0:
        return audio
    guard = max(1, guard_ms // 10)
    start = max(0, loud[0] - guard) * hop
    end = min(len(audio), (loud[-1] + 1 + guard) * hop)
    return audio[start:end]


def load_user(path: Path, start_s: float, seconds: float) -> np.ndarray:
    wav, _ = librosa.load(path, sr=SR_U, mono=True, offset=start_s, duration=seconds)
    return wav.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "flow4")
    ap.add_argument("--user", type=Path,
                    default=ROOT / "samples" / "two-speakers-interview-clean.wav")
    ap.add_argument("--start", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--after", type=float, default=70.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    flow = Flow()
    user1 = load_user(args.user, args.start, args.seconds)
    reply = flow.reply_text(user1)
    print({"step": "reply text", "text": reply}, flush=True)
    reply_audio = flow.speak(reply)

    cue = choose_cue(FACT, reply)
    print({"step": "cue", "cue": cue}, flush=True)
    fact_audio = flow.speak(FACT, cue=cue)

    user2 = load_user(args.user, args.after, args.seconds * 0.6)
    reaction = flow.reply_text(user2, max_new=60)
    print({"step": "reaction text", "text": reaction}, flush=True)
    reaction_audio = flow.speak(reaction)

    pause = np.zeros(int(PAUSE_HANDOVER_S * SR_A), dtype=np.float32)
    user1_24 = librosa.resample(user1, orig_sr=SR_U, target_sr=SR_A)
    user2_24 = librosa.resample(user2, orig_sr=SR_U, target_sr=SR_A)

    timeline, mix = [], np.zeros(0, dtype=np.float32)

    def add(audio: np.ndarray, label: str, kind: str) -> None:
        nonlocal mix
        timeline.append({"kind": kind, "label": label,
                         "onset_s": round(len(mix) / SR_A, 2),
                         "duration_s": round(len(audio) / SR_A, 2)})
        mix = np.concatenate([mix, audio])

    add(user1_24, "user: first utterance", "user")
    add(reply_audio, "actor: own reply", "agent")
    add(pause, "handover pause", "silence")
    add(fact_audio, "actor: injected fact (cue + phrases)", "agent")
    add(user2_24, "user: second utterance", "user")
    add(reaction_audio, "actor: reaction after re-listening", "agent")

    out = args.out / "full_flow.wav"
    sf.write(out, np.clip(mix, -1, 1), SR_A)
    (args.out / "full_flow.json").write_text(json.dumps({
        "user": str(args.user), "out": str(out), "reply_text": reply,
        "reaction_text": reaction, "cue": cue, "fact": FACT,
        "phrases": split_phrases(FACT), "timeline": timeline,
    }, indent=1))
    print(json.dumps({"out": str(out), "total_s": round(len(mix) / SR_A, 2),
                      "timeline": timeline}, indent=1))


if __name__ == "__main__":
    main()
