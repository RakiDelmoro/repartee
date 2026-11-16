#!/usr/bin/env python3
"""One multi-turn duplex sample with an injected science fact in the middle.

Timeline (seconds):
  0-6    user speaks (turn 1)
  6-17   silence: model answers in duplex
  17-22  user speaks (turn 2)
  22-32  silence: model answers
  32     inject the science fact; model switches from its own words to the fact
  ~38-44 silence after the fact
  44-49  user speaks (turn 3)
  49-60  silence: model answers

Model: 8-bit LM (torch int8 weight-only + Triton GEMV), bf16 audio path.
Writes a stereo WAV (user left, model right), mono channels, and metadata.
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

# (start, duration) of user speech; everything else is silence
# (source second in speaker_a, timeline second, duration)
SEGMENTS = [
    (25.0, 0.0),         # "Okay, I'm glad your internet connection is stable..."
    (85.0, 13.0),        # "...little jobs, and when they call us..."
    (105.0, 26.0),       # "Money, so they only call when the customers are there..."
]
TOTAL = 42.0


def user_track(audio: np.ndarray) -> np.ndarray:
    """Concatenate speaker_a windows and silence into a 60 s user track."""
    track = np.zeros(int(TOTAL * SR), dtype=np.float32)
    for source, place in SEGMENTS:
        start = int(source * SR)
        piece = speech_window(audio[start : start + 18 * SR], seconds=5.0)
        dst = int(place * SR)
        track[dst : dst + len(piece)] = piece
    return track


class TurnTakingSession(InjectedSession):
    """Duplex model, but the harness owns the floor: no model speech over the user."""

    END_SILENCE = 5       # frames (~0.7 s) after the user stops
    MODEL_END = 6         # frames (~0.8 s) of model silence ends its turn
    MAX_RESPONSE = 75     # 10 s
    REPROMPT = 8          # if the model has not started, prompt again

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = "LISTEN"
        self.force_pad = True
        self._had_user = False
        self._user_silence = 0
        self._model_rms = 0.0
        self._model_spoke = False
        self._model_silence = 0
        self._response_frames = 0

    def inject_now(self, text: str, **kwargs):
        """Scripted line that takes the floor even if the controller is listening."""
        self.state = "RESPOND"
        self.force_pad = False
        self._model_spoke = True       # suppress the reprompt path
        self._model_silence = 0
        self._response_frames = 0
        schedule = super().inject(text, **kwargs)
        self._hold_off_until = self.index + len(schedule) + 2
        return schedule

    def start(self):
        frame = super().start()
        self._model_rms = float(np.sqrt(np.mean(frame.audio ** 2)))
        return frame

    def step(self, user_audio):
        speech = float(np.sqrt(np.mean(np.asarray(user_audio) ** 2))) > 0.012
        if self.state == "LISTEN":
            if speech:
                self._had_user, self._user_silence = True, 0
            elif self._had_user:
                self._user_silence += 1
                if self._user_silence >= self.END_SILENCE:
                    self.state = "RESPOND"
                    self._had_user = False
                    self._model_spoke = False
                    self._model_silence = 0
                    self._response_frames = 0
                    self.release(prompt=True)
        else:
            if speech:
                self.state, self._had_user, self._user_silence = "LISTEN", True, 0
                self.hold()
            elif self.index < getattr(self, "_hold_off_until", -1):
                pass  # a scripted line is playing; do not cut it for model silence
            else:
                self._response_frames += 1
                if self._model_rms > 0.01:
                    self._model_spoke, self._model_silence = True, 0
                elif self._model_spoke:
                    self._model_silence += 1
                    if self._model_silence >= self.MODEL_END:
                        self.state = "LISTEN"
                        self.hold()
                if not self._model_spoke and self._response_frames == self.REPROMPT:
                    self.release(prompt=True)
                if self._response_frames >= self.MAX_RESPONSE:
                    self.state = "LISTEN"
                    self.hold()
        frame = super().step(user_audio)
        self._model_rms = float(np.sqrt(np.mean(frame.audio ** 2)))
        return frame


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
    reference = prepare_reference(source)[: 4 * SR]
    user = user_track(source)
    frames = int(TOTAL * FPS) + 1
    session = InjectedSession(
        model, CARD, reference,
        DuplexConfig(max_frames=frames, seed=7, temperature=0.7),
    )
    session.force_pad = True   # the user has the floor until release()
    collected = [session.start()]
    schedule: list[int] = []
    step_times: list[float] = []
    injected = False
    last_model_rms = 0.0
    release_at = {int(5 * FPS), int(18 * FPS), int(31 * FPS)}
    hold_at = {int(12 * FPS), int(25 * FPS)}
    for index in range(1, frames):
        if index in release_at:
            session.release(prompt=True)
            print(f"release at {index / FPS:.1f}s", flush=True)
        if index in hold_at:
            session.hold()
            print(f"hold at {index / FPS:.1f}s", flush=True)
        # one short fact only: Haru sustains about one sentence per forced burst
        if not injected and index >= int(18.5 * FPS):
            schedule = session.inject(FACT, gap=0, tail_pad=4)
            injected = True
            print(f"inject at {index / FPS:.1f}s, {len(schedule)} frames", flush=True)
        piece = user[index * FRAME : (index + 1) * FRAME]
        chunk = np.zeros(FRAME, dtype=np.float32)
        if len(piece):
            chunk[: len(piece)] = piece
        tick = time.perf_counter()
        frame = session.step(chunk)
        step_times.append(time.perf_counter() - tick)
        collected.append(frame)
        last_model_rms = float(np.sqrt(np.mean(frame.audio ** 2)))
    elapsed = time.perf_counter() - started

    model_audio = np.concatenate([frame.audio for frame in collected])
    user_audio = np.zeros_like(model_audio)
    usable = min(len(user), len(model_audio))
    user_audio[:usable] = user[:usable]
    out = ROOT / "data" / "samples" / "multiturn_8bit.wav"
    save_audio(out, np.stack([user_audio, model_audio], axis=1), SR)
    save_audio(out.with_name(out.stem + ".model.wav"), model_audio, SR)
    save_audio(out.with_name(out.stem + ".user.wav"), user_audio, SR)
    energies = [float(np.sqrt(np.mean(model_audio[i * SR : (i + 1) * SR] ** 2)))
                for i in range(int(len(model_audio) / SR))]
    text = "".join(frame.text for frame in collected).strip()
    meta = {
        "model": "8-bit int8 + Triton GEMV",
        "fact": FACT,
        "segments": SEGMENTS,
        "schedule_frames": len(schedule),
        "frames": len(collected),
        "duration_s": round(len(model_audio) / SR, 2),
        "frame_ms_median": round(1000 * float(np.median(step_times)), 1),
        "frame_ms_p90": round(1000 * float(np.percentile(step_times, 90)), 1),
        "rms_per_second": [round(x, 3) for x in energies],
        "model_text": text,
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
