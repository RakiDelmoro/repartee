"""ASR a generated P1 WAV against its target text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.asr import QwenTranscriber  # noqa: E402


def wer(ref: str, hyp: str) -> float | None:
    r = ref.lower().split()
    h = hyp.lower().split()
    if not r:
        return None
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (rw != hw),
            )
        prev = cur
    return prev[-1] / len(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--target", type=Path, default=None)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3-ASR-0.6B-hf"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--language", default=None)
    args = ap.parse_args()

    transcriber = QwenTranscriber(model_id=args.model, language=args.language)
    result = transcriber.transcribe(args.audio)
    target = args.target.read_text().strip() if args.target and args.target.exists() else ""
    record = {
        "audio": str(args.audio),
        "transcript": result.text,
        "language": result.language,
        "duration_s": result.duration_s,
        "target": target,
        "wer": wer(target, result.text),
    }
    print(json.dumps(record, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
