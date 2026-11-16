#!/usr/bin/env python3
"""Offline continuous-duplex demo: no turn-taking gate, overlap allowed.

This is the honest capture of the live behaviour: the model can speak while the
user speaks, and the fact is injected mid-turn from live audio context.

  0-6    user 1
  14-19  user 2
  21     inject "Light from the Sun takes about eight minutes to reach Earth."
  28-33  user 3
  0-44   model runs freely throughout (overlap expected)

8-bit int8 LM + Triton GEMV, bf16 audio path.
"""

from __future__ import annotations

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
from haru_quant import cast_to  # noqa: E402
from int8_linear import replace_lm_linears_int8  # noqa: E402

SR = 24_000
FRAME = 3_200
FPS = 7.5
CARD = (
    "You are the person described below, talking with the other person by voice.\n"
    "[Your Persona]\nA calm, concise assistant who answers in short sentences.\n"
    "[Your Name]\nAri\n[Your Gender]\nmale"
)
FACT = "Light from the Sun takes about eight minutes to reach Earth."
SEGMENTS = [(25.0, 0.0), (85.0, 14.0), (105.0, 28.0)]
INJECT_AT = 21.0
TOTAL = 44.0


def user_track(audio: np.ndarray) -> np.ndarray:
    track = np.zeros(int(TOTAL * SR), dtype=np.float32)
    for source, place in SEGMENTS:
        piece = speech_window(audio[int(source * SR) : int((source + 18) * SR)], seconds=5.0)
        dst = int(place * SR)
        track[dst : dst + len(piece)] = piece
    return track


def main() -> None:
    started = time.perf_counter()
    model = load_model(str(ROOT / "models" / "Haru-4B"), device="cpu", dtype=torch.bfloat16)
    replaced = replace_lm_linears_int8(model)
    cast_to(model, torch.bfloat16)
    model = model.to("cuda")
    torch.cuda.synchronize()
    print(f"model ready in {time.perf_counter() - started:.1f} s, {replaced} int8 linears, "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    source = load_audio(ROOT / "speaker_a.flac")
    reference = prepare_reference(load_audio(ROOT / "speaker_a.flac"))[: 4 * SR]
    user = user_track(source)
    frames = int(TOTAL * FPS) + 1
    session = InjectedSession(
        model, CARD, reference,
        DuplexConfig(max_frames=frames, seed=7, temperature=0.7),
    )
    collected = [session.start()]
    inject_frame = int(INJECT_AT * FPS)
    schedule: list[int] = []
    step_times: list[float] = []
    for index in range(1, frames):
        if index == inject_frame:
            schedule = session.inject(FACT, gap=0, tail_pad=4)
            print(f"inject at {INJECT_AT:.1f}s, {len(schedule)} frames", flush=True)
        piece = user[index * FRAME : (index + 1) * FRAME]
        chunk = np.zeros(FRAME, dtype=np.float32)
        if len(piece):
            chunk[: len(piece)] = piece
        tick = time.perf_counter()
        frame = session.step(chunk)
        step_times.append(time.perf_counter() - tick)
        collected.append(frame)

    model_audio = np.concatenate([f.audio for f in collected])
    user_audio = np.zeros_like(model_audio)
    usable = min(len(user), len(model_audio))
    user_audio[:usable] = user[:usable]
    out = ROOT / "data" / "samples" / "duplex_demo_8bit.wav"
    save_audio(out, np.stack([user_audio, model_audio], axis=1), SR)
    save_audio(out.with_name(out.stem + ".model.wav"), model_audio, SR)
    save_audio(out.with_name(out.stem + ".user.wav"), user_audio, SR)
    energies = [float(np.sqrt(np.mean(model_audio[i * SR : (i + 1) * SR] ** 2)))
                for i in range(int(len(model_audio) / SR))]
    meta = {
        "model": "8-bit int8 + Triton GEMV",
        "mode": "continuous duplex, no turn-taking gate",
        "fact": FACT,
        "inject_at_s": INJECT_AT,
        "segments": SEGMENTS,
        "frames": len(collected),
        "duration_s": round(len(model_audio) / SR, 2),
        "frame_ms_median": round(1000 * float(np.median(step_times)), 1),
        "frame_ms_p90": round(1000 * float(np.percentile(step_times, 90)), 1),
        "rms_per_second": [round(x, 3) for x in energies],
        "model_text": "".join(f.text for f in collected).strip(),
    }
    out.with_name(out.stem + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)

    del session, model
    torch.cuda.empty_cache()
    from app.asr import QwenTranscriber

    transcript = QwenTranscriber(model_id=str(ROOT / "models" / "Qwen3-ASR-1.7B-hf"),
                                 language="English").transcribe(out.with_name(out.stem + ".model.wav"))
    meta["asr"] = transcript.text
    out.with_name(out.stem + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print("ASR:", transcript.text, flush=True)


if __name__ == "__main__":
    main()
