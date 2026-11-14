#!/usr/bin/env python3
"""Full loop, final build: verified phrase rendering.

  user audio -> reply (sampled text, rendered in verified phrases)
  -> cue merged into the fact's first phrase -> fact phrases
  -> user audio again -> reaction

Each phrase: clean turn, <=10 words, punctuation-terminated, trimmed of turn padding,
and checked against the duration its word count predicts. A stalled or short render is
retried instead of shipped.

  .venv/bin/python research/full_flow2.py --out data/final_flow
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from full_flow import (CUE, FACT, Flow, SR_A, SR_U, load_user, split_phrases,
                       trim_silence)  # noqa: E402
from cues import compose, cue_for  # noqa: E402
from seam_test import Actor  # noqa: E402

WPS = 2.8          # measured actor speaking rate
PHRASE_GAP_S = 0.06
HANDOVER_S = 0.12


def expected_s(phrase: str) -> float:
    words = len(re.findall(r"[a-z0-9']+", phrase.lower()))
    return max(0.6, words / WPS)


class VerifiedFlow(Flow):
    def render_phrase(self, phrase: str, attempts: int = 3) -> tuple[np.ndarray, int, float]:
        want = expected_s(phrase)
        best = None
        for attempt in range(attempts):
            actor = Actor(self.model, self.proc)
            actor.prefill_messages([
                {"role": "system", "content": "Respond with interleaved text and audio."},
                {"role": "user", "content": "Continue the conversation."},
            ])
            actor.force_text(phrase)
            actor.speak(60)
            wav = trim_silence(actor.samples())
            got = len(wav) / SR_A
            score = abs(got - want) / want
            if best is None or score < best[0]:
                best = (score, wav, attempt, got)
            if 0.6 * want <= got <= 1.6 * want:
                break
        _, wav, attempt, got = best
        return wav, attempt, got

    def speak_verified(self, text: str, cue: str | None = None) -> tuple[np.ndarray, list[dict]]:
        if cue:
            # the cue is words too: split tighter so cue + first phrase stays <= 10
            cue_words = len(re.findall(r"[a-z0-9']+", cue.lower()))
            phrases = split_phrases(text, max_words=max(4, 10 - cue_words))
            phrases = [compose(phrases[0], cue)] + phrases[1:]
        else:
            phrases = split_phrases(text)
        parts, meta = [], []
        for phrase in phrases:
            wav, attempt, got = self.render_phrase(phrase)
            parts.append(wav)
            meta.append({"phrase": phrase, "attempt": attempt,
                         "s": round(got, 2), "want_s": round(expected_s(phrase), 2)})
            print({"phrase": phrase[:60], "attempt": attempt, "s": round(got, 2),
                   "want_s": round(expected_s(phrase), 2)}, flush=True)
        gap = np.zeros(int(PHRASE_GAP_S * SR_A), dtype=np.float32)
        seq = []
        for part in parts:
            seq += [part, gap]
        return np.concatenate(seq[:-1]), meta

    def release(self) -> None:
        import torch

        del self.model
        torch.cuda.empty_cache()


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
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "final_flow")
    ap.add_argument("--user", type=Path,
                    default=ROOT / "samples" / "two-speakers-interview-clean.wav")
    ap.add_argument("--start", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--after", type=float, default=70.0)
    ap.add_argument("--fact", default=FACT)
    ap.add_argument("--relation", default="shift",
                    choices=["shift", "correct", "remind", "answer", "continue"])
    ap.add_argument("--topic", default=None,
                    help="content words the cue may name; omit for no cue")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    flow = VerifiedFlow()
    user1 = load_user(args.user, args.start, args.seconds)
    reply = flow.reply_text(user1)
    reply_audio, reply_meta = flow.speak_verified(reply)

    fact = args.fact
    cue = cue_for(args.relation, args.topic, fact)
    fact_audio, fact_meta = flow.speak_verified(fact, cue=cue)

    user2 = load_user(args.user, args.after, args.seconds * 0.6)
    reaction = flow.reply_text(user2, max_new=120)
    reaction_audio, reaction_meta = flow.speak_verified(reaction)
    flow.release()

    pause = np.zeros(int(HANDOVER_S * SR_A), dtype=np.float32)
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
    add(reply_audio, "actor: reply (verified phrases)", "agent")
    add(pause, "handover pause", "silence")
    add(fact_audio, f"actor: {cue} + injected fact", "agent")
    add(user2_24, "user: second utterance", "user")
    add(reaction_audio, "actor: reaction", "agent")

    out = args.out / "full_flow.wav"
    sf.write(out, np.clip(mix, -1, 1), SR_A)

    from app.asr import QwenTranscriber

    judge = QwenTranscriber()
    def seg_asr(onset: float, dur: float) -> str:
        z = mix[int(onset * SR_A):int((onset + dur) * SR_A)]
        p = args.out / "segment.wav"
        sf.write(p, z, SR_A)
        return judge.transcribe(p).text

    rows = {}
    for item in timeline:
        if item["kind"] != "agent":
            continue
        hyp = seg_asr(item["onset_s"], item["duration_s"])
        rows[item["label"]] = {"asr": hyp, "duration_s": item["duration_s"]}

    report = {
        "user": str(args.user), "out": str(out),
        "reply_text": reply, "reaction_text": reaction, "cue": cue, "fact": fact,
        "timeline": timeline,
        "phrases": {"reply": reply_meta, "fact": fact_meta, "reaction": reaction_meta},
        "asr": {
            "reply": {**rows.get("actor: reply (verified phrases)", {}),
                      "recall": recall(reply, rows.get("actor: reply (verified phrases)", {}).get("asr", ""))},
            "fact": {**rows.get(f"actor: {cue} + injected fact", {}),
                     "recall": recall(fact, rows.get(f"actor: {cue} + injected fact", {}).get("asr", ""))},
            "reaction": {**rows.get("actor: reaction", {}),
                         "recall": recall(reaction, rows.get("actor: reaction", {}).get("asr", ""))},
        },
        "total_s": round(len(mix) / SR_A, 2),
    }
    (args.out / "full_flow.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("asr", "total_s")}, indent=1))


if __name__ == "__main__":
    main()
