#!/usr/bin/env python3
"""One duplex sample: a user recording, with a sentence injected over it.

    .venv/bin/python research/haru_sample.py --lm-bits 4 \
        --user speaker_a.flac \
        --text "Light from the Sun takes about eight minutes to reach Earth." \
        --at 2.0 --seconds 10 --out data/samples/space_fact.wav

Writes a stereo WAV (user left, Haru right), the two mono channels, and metadata.
The injected sentence is spoken while the user audio keeps feeding the model, which is
the full-duplex case: it speaks over the user without losing the input stream.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from haru import DuplexConfig, load_model  # noqa: E402
from haru.audio import load_audio, prepare_reference, save_audio  # noqa: E402
from haru_inject import InjectedSession, speech_window  # noqa: E402

SR = 24_000
FRAME = 3_200
FPS = 7.5
CARD = (
    "You are the person described below, talking with the other person by voice.\n"
    "[Your Persona]\nA calm, concise assistant who answers in short sentences.\n"
    "[Your Name]\nAri\n[Your Gender]\nmale"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "Haru-4B"))
    parser.add_argument("--lm-bits", type=int, default=4, choices=(0, 4, 8))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lm-dtype", default="bf16", choices=("bf16", "fp16"))
    parser.add_argument("--backend", default="bnb", choices=("bnb", "int8"),
                        help="bnb: bitsandbytes NF4; int8: torch packed-int8 weight-only")
    parser.add_argument("--user", type=Path, default=ROOT / "speaker_a.flac")
    parser.add_argument("--reference", type=Path, default=None, help="voice clip (default: --user)")
    parser.add_argument("--reference-seconds", type=float, default=4.0)
    parser.add_argument("--seconds", type=float, default=10.0, help="user audio window")
    parser.add_argument("--at", type=float, default=2.0, help="injection time in seconds")
    parser.add_argument("--text", required=True)
    parser.add_argument("--scripted", action="store_true", help="say only the injected text")
    parser.add_argument("--mute-audio", action="store_true",
                        help="feed silence instead of the model's previous audio during injection")
    parser.add_argument("--gap", type=int, default=0, help="PAD frames between injected words (0 works)")
    parser.add_argument("--no-eop", action="store_true", help="do not insert an EOP before the text")
    parser.add_argument("--no-inject", action="store_true", help="control run: do not inject anything")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "samples" / "sample.wav")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--asr", action="store_true", help="transcribe the model channel")
    parser.add_argument("--asr-language", default="English")
    parser.add_argument("--asr-model", default=None, help="ASR checkpoint (default: the app's)")
    args = parser.parse_args()

    started = time.perf_counter()
    if args.lm_bits:
        from haru_quant import cast_to, replace_lm_linears

        model = load_model(args.model, device="cpu", dtype=torch.bfloat16)
        float_dtype = torch.bfloat16 if args.lm_dtype == "bf16" else torch.float16
        if args.backend == "int8":
            from int8_linear import replace_lm_linears_int8

            replace_lm_linears_int8(model)
            args.lm_bits = 8
        else:
            replace_lm_linears(model, float_dtype, args.lm_bits)
        cast_to(model, float_dtype)
        model = model.to("cuda")
        torch.cuda.synchronize()
    else:
        model = load_model(args.model, device=args.device, dtype=torch.bfloat16)
    print(f"model ready in {time.perf_counter() - started:.1f} s", flush=True)

    user_source = load_audio(args.user)
    voice_source = load_audio(args.reference) if args.reference else user_source
    reference = prepare_reference(voice_source)[: int(args.reference_seconds * SR)]
    user = speech_window(user_source, seconds=args.seconds)
    frames = max(1, int(args.seconds * FPS) + 1)
    config = DuplexConfig(max_frames=frames, seed=args.seed, temperature=0.7)
    session = InjectedSession(model, CARD, reference, config, scripted=args.scripted,
                              mute_audio=args.mute_audio)
    collected = [session.start()]
    inject_frame = max(1, min(frames - 1, int(args.at * FPS)))
    schedule: list[int] = []
    step_times: list[float] = []
    for index in range(1, frames):
        if index == inject_frame and not args.no_inject:
            schedule = session.inject(
                args.text, gap=args.gap, tail_pad=4, lead_eop=False if args.no_eop else None
            )
        chunk = np.zeros(FRAME, dtype=np.float32)
        piece = user[index * FRAME : (index + 1) * FRAME]
        if len(piece):
            chunk[: len(piece)] = piece
        tick = time.perf_counter()
        collected.append(session.step(chunk))
        step_times.append(time.perf_counter() - tick)
    elapsed = time.perf_counter() - started

    model_audio = np.concatenate([frame.audio for frame in collected])
    user_audio = np.zeros_like(model_audio)
    usable = min(len(user), len(model_audio))
    user_audio[:usable] = user[:usable]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_audio(args.out, np.stack([user_audio, model_audio], axis=1), SR)
    save_audio(args.out.with_name(args.out.stem + ".model.wav"), model_audio, SR)
    save_audio(args.out.with_name(args.out.stem + ".user.wav"), user_audio, SR)
    meta = {
        "text": args.text,
        "inject_frame": inject_frame,
        "schedule_frames": len(schedule),
        "scripted": args.scripted,
        "frames": len(collected),
        "duration_s": round(len(model_audio) / SR, 2),
        "frame_ms_median": round(1000 * float(np.median(step_times)), 1) if step_times else None,
        "frame_ms_p90": round(1000 * float(np.percentile(step_times, 90)), 1) if step_times else None,
        "peak": round(float(np.abs(model_audio).max()), 3),
        "rms": round(float(np.sqrt(np.mean(model_audio ** 2))), 4),
        "model_text": "".join(frame.text for frame in collected).strip(),
        "out": str(args.out),
    }
    if args.asr:
        del session, model
        torch.cuda.empty_cache()
        from app.asr import QwenTranscriber

        transcriber = QwenTranscriber(model_id=args.asr_model, language=args.asr_language)             if args.asr_model else QwenTranscriber(language=args.asr_language)
        transcript = transcriber.transcribe(args.out.with_name(args.out.stem + ".model.wav"))
        meta["asr"] = transcript.text
    args.out.with_name(args.out.stem + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
