#!/usr/bin/env python3
"""Haru-4B text injection: force the model's text stream and measure what it says.

Haru's text stream is an *input* to the language model at every frame: the
previous frame's text token, model audio latent and user audio latent are summed
into one position.  The served API only samples that token, so injection is a
sampler override: at the frame where the model would emit a token, emit the next
token of the injected sentence instead.  The model consumes it at the next
position, so the audio follows one frame later (133 ms); when the model is silent
the EOP marker takes that frame and the first word's audio starts two frames after
the injection (266 ms).

  .venv/bin/python research/haru_inject.py --model models/Haru-4B --lm-bits 4 \
      --out data/haru_inject_4          # 4-bit LM, fp16 audio path
  .venv/bin/python research/haru_inject.py --model models/Haru-4B --device cpu \
      --dtype bfloat16 --only listen_inject   # slow bf16 reference

Scenarios answer the three cases the product needs:
  listen  - model is silent (listening), text arrives, model speaks it
  speak   - model is mid-utterance, text arrives, model pivots
  user    - the user is talking while the text is injected (true overlap)
plus a no-injection baseline for the model's own token cadence and a scripted
run (PAD whenever the queue is empty) where the model never free-samples a word.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from haru import DuplexConfig, DuplexSession, Frame, load_model  # noqa: E402
from haru.audio import load_audio, prepare_reference, save_audio  # noqa: E402
from app.inject import TextInjector  # noqa: E402

SR = 24_000
FRAME = 3_200


# ------------------------------------------------------------------ injection


class InjectedSession(DuplexSession):
    """A DuplexSession whose text token can be forced for chosen frames.

    The parent's normal path is untouched; only frames with a queued token take
    the forced branch, which repeats the parent's frame assembly (token, audio
    latent, decode, detokenizer) with the token supplied instead of sampled.
    """

    def __init__(self, *args, scripted: bool = False, idle_frames: int = 4,
                 mute_audio: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.scripted = scripted
        self.mute_audio = mute_audio
        self.force_pad = False
        self.injector = TextInjector(
            self.model.tokenizer, self.pad_id, self.eop_id, idle_frames=idle_frames
        )
        self.forced: list[tuple[int, int]] = []

    @property
    def model_idle(self) -> bool:
        """A pause, not a word gap: enough consecutive PAD frames to be silence."""
        return self.injector.idle

    def inject(
        self,
        text: str,
        gap: int = 0,
        lead_eop: bool | None = None,
        tail_pad: int = 0,
        interrupt: bool = True,
        wait: bool = False,
    ) -> list[int]:
        """Queue ``text``; see :class:`app.inject.TextInjector` for the semantics.

        ``lead_eop=None`` inserts EOP when the model is idle.  ``wait=True``
        holds the sentence until the next pause instead of cutting in.
        """
        return self.injector.speak(
            text, wait=wait, interrupt=interrupt, gap=gap, lead_eop=lead_eop, tail_pad=tail_pad
        )

    def hold(self) -> None:
        """Stop the model now and keep it silent until ``release``."""
        self.force_pad = True
        self.injector.speak("", lead_eop=False, tail_pad=0)  # drop pending words

    def release(self, prompt: bool = True) -> None:
        """Let the model speak again; an EOP prompt starts its turn."""
        self.force_pad = False
        if prompt:
            self.inject("", lead_eop=True)

    @torch.no_grad()
    def _emit(self, hidden: torch.Tensor, started: float) -> Frame:
        forced = self.injector.poll()
        if forced is None and (self.scripted or self.force_pad):
            forced = self.pad_id  # hold the model silent while the user talks
        if forced is None:
            frame = super()._emit(hidden, started)
            self.injector.observe(frame.token)
            return frame

        model, cfg = self.model, self.config
        token = torch.tensor([forced], device=hidden.device, dtype=torch.long)

        # The audio of this frame comes from the same hidden state as the
        # parent's; the forced token is consumed by the next position, so the
        # audio follows one frame later (two from silence, where EOP takes one).
        noise = torch.randn(
            1, model.config.latent_dim, device=hidden.device, dtype=model.dtype, generator=self._generator
        )
        latent = model.unscale_latents(model.head.sample(hidden, noise, cfg.guidance_scale))
        audio = self._codec.decode(latent[None])[0].float().clamp(-1.0, 1.0)

        self._token, self._latent = token, latent
        if self.force_pad:
            # Text PAD alone does not silence the audio head when the user is
            # talking; cut the output and feed silence back so the model cannot
            # build a response it would later unmute mid-word.
            audio = torch.zeros_like(audio)  # mute output, keep the model's audio context
        if self.mute_audio:
            from haru.prompt import silence_latent

            # Do not feed the model its old audio plan while its words are forced;
            # otherwise the hidden state continues the previous speech.
            self._latent = silence_latent(model.codec)[None]
        if self._dry is not None:
            self._dry.update(forced)
        frame = Frame(
            audio=audio.cpu().numpy(),
            token=forced,
            text=self._detokenizer.push(forced),
            index=self.index,
            seconds=time.perf_counter() - started,
        )
        self.index += 1
        frame.done = self.index >= cfg.max_frames
        self.injector.observe(forced)
        self.forced.append((frame.index, forced))
        return frame


# ------------------------------------------------------------------ media


def speech_window(audio: np.ndarray, seconds: float = 6.0, hop: float = 0.08) -> np.ndarray:
    """Speech-rich ``seconds`` of ``audio``, snapped forward to a speech onset."""
    step = int(hop * SR)
    count = len(audio) // step
    rms = np.sqrt((audio[: count * step].reshape(count, step) ** 2).mean(axis=1) + 1e-12)
    voiced = rms > 0.012
    win = int(seconds / hop)
    best, best_score = 0, -1.0
    for start in range(0, count - win, int(0.5 / hop)):
        score = float(voiced[start : start + win].mean())
        if score > best_score:
            best, best_score = start, score
    head = np.flatnonzero(voiced[best : best + int(1.0 / hop)])
    if len(head):
        best += int(head[0])
    begin = best * step
    return audio[begin : begin + int(seconds * SR)]


def rms_per_frame(audio: np.ndarray) -> list[float]:
    count = len(audio) // FRAME
    return [float(np.sqrt(np.mean(audio[i * FRAME : (i + 1) * FRAME] ** 2))) for i in range(count)]


# ------------------------------------------------------------------ metrics


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def recall(injected: list[str], heard: list[str]) -> float:
    return lcs_length(injected, heard) / len(injected) if injected else 0.0


# ------------------------------------------------------------------ scenario


def run_scenario(
    model,
    out: Path,
    name: str,
    text: str,
    *,
    inject_frame: int,
    frames: int,
    speak_first: bool,
    user: np.ndarray | None,
    gap: int,
    tail_pad: int,
    seed: int,
    scripted: bool = False,
) -> dict:
    config = DuplexConfig(
        seed=seed,
        speak_first=speak_first,
        max_frames=frames,
        temperature=0.7,
        top_k=25,
    )
    session = InjectedSession(model, SYSTEM_PROMPT, REFERENCE, config, scripted=scripted)
    started = time.perf_counter()
    frame0 = session.start()
    prompt_s = time.perf_counter() - started
    collected = [frame0]
    schedule: list[int] = []
    injected_at = None
    for index in range(1, frames):
        if index == inject_frame:
            schedule = session.inject(text, gap=gap, tail_pad=tail_pad)
            injected_at = index
        chunk = np.zeros(FRAME, dtype=np.float32) if user is None else user[index * FRAME : (index + 1) * FRAME]
        if len(chunk) < FRAME:
            chunk = np.pad(chunk, (0, FRAME - len(chunk)))
        collected.append(session.step(chunk))
    elapsed = time.perf_counter() - started

    model_audio = np.concatenate([f.audio for f in collected])
    user_audio = np.zeros_like(model_audio)
    if user is not None:
        usable = min(len(user), len(model_audio))
        user_audio[:usable] = user[:usable]
    directory = out / name
    directory.mkdir(parents=True, exist_ok=True)
    save_audio(directory / "model.wav", model_audio, SR)
    save_audio(directory / "user.wav", user_audio, SR)
    save_audio(directory / "stereo.wav", np.stack([user_audio, model_audio], axis=1), SR)
    with (directory / "frames.jsonl").open("w", encoding="utf-8") as file:
        for f in collected:
            file.write(json.dumps({"frame": f.index, "token": f.token, "text": f.text}) + "\n")

    def f0_stats(signal: np.ndarray) -> dict:
        if signal.size < 2048:
            return {}
        try:
            import librosa
        except Exception:
            return {}
        values, voiced, _ = librosa.pyin(signal, fmin=60, fmax=400, sr=SR)
        chosen = values[voiced] if voiced is not None else np.array([])
        chosen = chosen[np.isfinite(chosen)]
        if not len(chosen):
            return {"f0_median_hz": None, "voiced_fraction": 0.0}
        return {"f0_median_hz": round(float(np.median(chosen)), 1),
                "voiced_fraction": round(float(np.mean(voiced)), 3)}

    energies = rms_per_frame(model_audio)
    onset = None
    for i in range(inject_frame, len(energies)):
        if energies[i] > 0.01:
            onset = i
            break
    spoken = "".join(f.text for f in collected).strip()
    injected_windows = [f.text for f in collected if f.index in set(range(inject_frame, inject_frame + len(schedule)))]
    summary = {
        "name": name,
        "text": text,
        "gap": gap,
        "scripted": scripted,
        "inject_frame": inject_frame,
        "schedule_frames": len(schedule),
        "frames": len(collected),
        "prompt_s": round(prompt_s, 2),
        "run_s": round(elapsed, 2),
        "frame_ms": round(1000 * elapsed / max(1, len(collected)), 1),
        "model_speech_onset_frame": onset,
        "onset_latency_ms": None if onset is None else round((onset - inject_frame) * 133.3),
        "forced_tokens": len(session.forced),
        "text_stream_injected_span": "".join(injected_windows),
        "text_stream_full": spoken,
        "model_rms_max": round(max(energies), 4),
        "duration_s": round(len(model_audio) / SR, 2),
        "f0": f0_stats(model_audio),
        "injected_f0": f0_stats(model_audio[inject_frame * FRAME : (inject_frame + len(schedule)) * FRAME]),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


# ------------------------------------------------------------------ main


SYSTEM_PROMPT = ""
REFERENCE = None


def main() -> None:
    global SYSTEM_PROMPT, REFERENCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "Haru-4B"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    parser.add_argument("--lm-bits", type=int, default=0, choices=(0, 4, 8),
                        help="quantize the LM backbone with bitsandbytes (needs CUDA)")
    parser.add_argument("--lm-dtype", default="bf16", choices=("bf16", "fp16"),
                        help="dtype of the non-quantized parts with --lm-bits")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "haru_inject")
    parser.add_argument("--user", type=Path, default=ROOT / "speaker_a.flac")
    parser.add_argument("--reference-seconds", type=float, default=4.0)
    parser.add_argument("--gap", type=int, default=0, help="PAD frames between injected words (0 works)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--only", nargs="*", help="scenario names to run")
    args = parser.parse_args()

    SYSTEM_PROMPT = (
        "You are the person described below, talking with the other person by voice.\n"
        "[Your Persona]\nA calm, concise assistant who answers in short sentences.\n"
        "[Your Name]\nAri\n[Your Gender]\nmale"
    )

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    started = time.perf_counter()
    if args.lm_bits:
        from haru_quant import cast_to, replace_lm_linears

        print(f"loading {args.model} on cpu, {args.lm_bits}-bit LM", flush=True)
        model = load_model(args.model, device="cpu", dtype=torch.bfloat16)
        float_dtype = torch.bfloat16 if args.lm_dtype == "bf16" else torch.float16
        replaced = replace_lm_linears(model, float_dtype, args.lm_bits)
        cast_to(model, float_dtype)
        model = model.to("cuda")
        torch.cuda.synchronize()
        print(f"loaded in {time.perf_counter() - started:.1f} s, {replaced} LM linears quantized, "
              f"{torch.cuda.memory_allocated() / 1e9:.2f} GB VRAM", flush=True)
    else:
        print(f"loading {args.model} on {args.device} ({args.dtype})", flush=True)
        model = load_model(args.model, device=args.device, dtype=dtype)
        print(f"loaded in {time.perf_counter() - started:.1f} s", flush=True)

    voice = prepare_reference(load_audio(args.user))
    REFERENCE = voice[: int(args.reference_seconds * SR)]
    user_talk = speech_window(load_audio(args.user), seconds=6.0)

    args.out.mkdir(parents=True, exist_ok=True)
    scenarios = [
        dict(
            name="baseline_own_speech",
            text="",
            inject_frame=10**9,
            frames=40,
            speak_first=True,
            user=None,
            gap=args.gap,
            tail_pad=0,
        ),
        dict(
            name="listen_inject",
            text="The Harrow project was cancelled last Friday.",
            inject_frame=8,
            frames=60,
            speak_first=False,
            user=None,
            gap=args.gap,
            tail_pad=4,
        ),
        dict(
            name="speak_inject",
            text="Actually, the budget moved to the satellite team.",
            inject_frame=22,
            frames=70,
            speak_first=True,
            user=None,
            gap=args.gap,
            tail_pad=4,
        ),
        dict(
            name="user_talking_inject",
            text="One more thing: the review is on Tuesday.",
            inject_frame=24,
            frames=70,
            speak_first=False,
            user=user_talk,
            gap=args.gap,
            tail_pad=4,
        ),
        dict(
            name="scripted_control",
            text="Good morning. The review is on Tuesday at nine.",
            inject_frame=8,
            frames=50,
            speak_first=False,
            user=None,
            gap=args.gap,
            tail_pad=0,
            scripted=True,
        ),
    ]
    if args.only:
        scenarios = [scenario for scenario in scenarios if scenario["name"] in args.only]
    results = []
    for scenario in scenarios:
        print(f"running {scenario['name']}", flush=True)
        summary = run_scenario(model, args.out, seed=args.seed, **scenario)
        print(
            f"  {summary['frame_ms']} ms/frame, onset {summary['onset_latency_ms']} ms, "
            f"text: {summary['text_stream_full'][:160]!r}",
            flush=True,
        )
        results.append(summary)

    if not args.skip_asr:
        del model
        torch.cuda.empty_cache()
        from app.asr import QwenTranscriber

        transcriber = QwenTranscriber()
        for summary in results:
            wav = args.out / summary["name"] / "model.wav"
            try:
                transcript = transcriber.transcribe(wav)
            except Exception as error:  # ASR is a judge, not the product path
                print(f"  ASR failed for {summary['name']}: {error}", flush=True)
                continue
            heard = words(transcript.text)
            injected = words(summary["text"]) if summary["text"] else []
            summary["asr"] = transcript.text
            summary["injected_recall"] = round(recall(injected, heard), 3)
            print(f"  ASR: {transcript.text!r} recall={summary['injected_recall']}", flush=True)
        (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    table = ["| scenario | frame ms | onset ms | forced | recall | text stream |", "|---|---|---|---|---|---|"]
    for s in results:
        table.append(
            f"| {s['name']} | {s['frame_ms']} | {s['onset_latency_ms']} | {s['forced_tokens']} | "
            f"{s.get('injected_recall', '-')} | {s['text_stream_full'][:60]!r} |"
        )
    (args.out / "results.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print("\n".join(table), flush=True)


if __name__ == "__main__":
    main()
