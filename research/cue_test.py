#!/usr/bin/env python3
"""C1/C2: does a topic-bearing cue beat a generic shift marker, or no cue at all?

Same fact, three openings:
  A  no cue at all
  B  generic shift   "One more thing,"
  C  topic-bearing   "About the Sun,"

Each is rendered through the verified phrase path and transcribed, so the ear can judge
and the numbers can confirm nothing was lost.

  .venv/bin/python research/cue_test.py --out data/cue
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from full_flow import SR_A  # noqa: E402
from full_flow2 import VerifiedFlow  # noqa: E402

FACT = "Light from the Sun takes eight minutes to reach Earth."
VARIANTS = {
    "A_no_cue": None,
    "B_generic": "One more thing,",
    "C_topic": "About the Sun,",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "cue")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    flow = VerifiedFlow()
    rendered = {}
    for name, cue in VARIANTS.items():
        audio, phrases = flow.speak_verified(FACT, cue=cue)
        sf.write(args.out / f"{name}.wav", audio, SR_A)
        rendered[name] = {"cue": cue, "s": round(len(audio) / SR_A, 2), "phrases": phrases}
        print({name: rendered[name]["s"], "phrases": phrases}, flush=True)
    flow.release()

    from app.asr import QwenTranscriber
    judge = QwenTranscriber()
    for name in rendered:
        rendered[name]["asr"] = judge.transcribe(args.out / f"{name}.wav").text
        print({name: rendered[name]["asr"]}, flush=True)
    (args.out / "cue_report.json").write_text(json.dumps(rendered, indent=1))


if __name__ == "__main__":
    main()
