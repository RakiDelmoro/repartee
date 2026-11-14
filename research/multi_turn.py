#!/usr/bin/env python3
"""Multi-turn flow on a longer user recording.

  U1 -> A1 -> U2 -> A2 -> cue + injected fact -> U3 -> A3 -> U4 -> A4

User turns are 7 s speech-rich windows picked from the recording. Every agent turn
uses the verified phrase renderer; the fact uses the relation-based cue.

  .venv/bin/python research/multi_turn.py --user speaker_a.flac \
      --fact "Light from the Sun takes eight minutes to reach Earth." --out data/multi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from full_flow import SR_A, SR_U  # noqa: E402
from cues import cue_for  # noqa: E402
from full_flow2 import VerifiedFlow  # noqa: E402

HANDOVER_S = 0.12
TURN_S = 7.0
N_TURNS = 4


def speech_windows(wav: np.ndarray, dur: float = TURN_S, n: int = N_TURNS,
                   gap_s: float = 12.0, min_speech: float = 0.45,
                   exclude_s: tuple[float, ...] = ()) -> list[tuple[int, int]]:
    """Speech-rich windows, snapped to a speech onset, skipping previously used ones."""
    hop = int(0.08 * SR_U)
    frames = len(wav) // hop
    rms = np.sqrt((wav[: frames * hop].reshape(frames, hop) ** 2).mean(1))
    voiced = rms > 0.012
    win = int(dur / 0.08)
    step = int(0.5 / 0.08)
    exclude = [e / 0.08 for e in exclude_s]
    scored = []
    for start in range(0, frames - win, step):
        scored.append((float(voiced[start:start + win].mean()), start))
    scored.sort(reverse=True)
    picked: list[int] = []
    for frac, start in scored:
        if frac < min_speech:
            break
        if any(abs(start - e) * 0.08 < 6.0 for e in exclude):
            continue
        if all(abs(start - other) * 0.08 >= gap_s for other in picked):
            picked.append(start)
        if len(picked) == n:
            break
    picked.sort()
    spans = []
    for start in picked:
        onsets = np.flatnonzero(voiced[start:start + int(1.0 / 0.08)])
        if len(onsets):
            start += int(onsets[0])
        start = min(start, max(0, frames - win))
        spans.append((start * hop, int(dur * SR_U),
                      round(float(voiced[start:start + win].mean()), 2)))
    return spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=Path, default=ROOT / "speaker_a.flac")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "multi")
    ap.add_argument("--fact", default="Light from the Sun takes eight minutes to reach Earth.")
    ap.add_argument("--relation", default="shift")
    ap.add_argument("--topic", default=None)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--gap", type=float, default=12.0)
    ap.add_argument("--min-speech", type=float, default=0.45)
    ap.add_argument("--exclude", default="")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    wav, _ = librosa.load(args.user, sr=SR_U, mono=True)
    spans = speech_windows(
        wav, n=args.turns, gap_s=args.gap, min_speech=args.min_speech,
        exclude_s=tuple(float(x) for x in args.exclude.split(",") if x.strip()))
    print({"user": str(args.user),
           "spans": [{"start_s": round(a / SR_U, 1), "speech": f} for a, _, f in spans]},
          flush=True)
    if len(spans) < 3:
        raise SystemExit("not enough speech windows in the recording")

    flow = VerifiedFlow()
    segments: list[tuple[str, np.ndarray, dict]] = []
    meta: list[dict] = []
    for i, (start, length, _frac) in enumerate(spans):
        user = wav[start:start + length]
        reply = flow.reply_text(user, max_new=160)
        audio, phrases = flow.speak_verified(reply)
        segments.append(("user", user, {"index": i, "start_s": round(start / SR_U, 1)}))
        segments.append(("agent", audio, {"text": reply, "phrases": phrases,
                                          "recall_ref": reply}))
        meta.append({"turn": i, "user_start_s": round(start / SR_U, 1),
                     "reply": reply, "phrases": phrases})
        print({"turn": i, "user_at": round(start / SR_U, 1), "reply": reply[:80]}, flush=True)
        if i == 1:
            cue = cue_for(args.relation, args.topic, args.fact)
            fact_audio, fact_phrases = flow.speak_verified(args.fact, cue=cue)
            segments.append(("silence", np.zeros(int(HANDOVER_S * SR_A), np.float32), {}))
            segments.append(("agent", fact_audio, {"text": args.fact, "phrases": fact_phrases,
                                                   "recall_ref": args.fact, "cue": cue}))
            meta.append({"turn": "fact", "cue": cue, "fact": args.fact,
                         "phrases": fact_phrases})
    flow.release()

    timeline, mix = [], np.zeros(0, dtype=np.float32)
    for kind, audio, info in segments:
        if kind == "user":
            audio = librosa.resample(audio, orig_sr=SR_U, target_sr=SR_A)
        timeline.append({"kind": kind, "onset_s": round(len(mix) / SR_A, 2),
                         "duration_s": round(len(audio) / SR_A, 2), **info})
        mix = np.concatenate([mix, audio])

    out = args.out / "multi_turn.wav"
    sf.write(out, np.clip(mix, -1, 1), SR_A)

    from app.asr import QwenTranscriber
    judge = QwenTranscriber()

    def asr(audio: np.ndarray) -> str:
        p = args.out / "segment.wav"
        sf.write(p, audio, SR_A)
        return judge.transcribe(p).text

    for item in timeline:
        if item["kind"] == "agent":
            hyp = asr(mix[int(item["onset_s"] * SR_A):
                          int((item["onset_s"] + item["duration_s"]) * SR_A)])
            item["asr"] = hyp
    (args.out / "multi_turn.json").write_text(json.dumps(
        {"user": str(args.user), "out": str(out), "turns": meta, "timeline": timeline},
        indent=1))
    print(json.dumps({"out": str(out), "total_s": round(len(mix) / SR_A, 2),
                      "segments": len(timeline)}, indent=1))
    for item in timeline:
        if item["kind"] == "agent":
            print({"agent": item["duration_s"], "asr": item["asr"][:110]})


if __name__ == "__main__":
    main()
