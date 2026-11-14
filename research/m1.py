#!/usr/bin/env python3
"""M1: carry the previous phrase's audio into the next generation.

Baseline (today): phrase 2 is a cold turn — history is system + user text only.
M1: the same turn, but preceded by an assistant turn whose content is the previous
phrase as `audio_out` (codec frames), so the model continues its own voice.

Measured on phrase 2: leading silence, pitch step against phrase 1's tail, words/s,
ASR recall.

  .venv/bin/python research/m1.py --out data/m1
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import config  # noqa: E402
from seam_test import Actor  # noqa: E402

MODEL = config.ASR_LFM3_MODEL
SR_A = 24_000
PHRASE1 = "One more thing, the Harrow project was cancelled last Friday."
PHRASE2 = "Its budget moved to the satellite team."


def prefill_audio_out(actor: Actor, prev_wav: np.ndarray, user_text: str) -> None:
    """History: system, user text, assistant <audio_out> = the previous phrase."""
    messages = [
        {"role": "system", "content": "Respond with interleaved text and audio."},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": [{"type": "audio_out"}]},
    ]
    prompt = actor.proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = actor.proc(text=prompt, audio_out=[prev_wav], sampling_rate=SR_A,
                        return_tensors="pt")
    ids = inputs["input_ids"].to("cuda")
    codes = inputs["audio_out_codes"]
    if codes.ndim == 3:
        codes = codes[0]
    codes = codes.to("cuda")

    embeds = actor.model.get_input_embeddings()(ids)
    placeholder = ids[0] == actor.proc.audio_out_token_id
    n = int(placeholder.sum())
    if n != codes.shape[0]:
        raise RuntimeError(f"{n} audio_out slots for {codes.shape[0]} frames")
    out = actor.model.model.get_audio_out_embeds(codes)
    embeds = embeds.clone()
    embeds[0, placeholder] = out.to(embeds.dtype)

    step = actor.model.model.language_model(
        inputs_embeds=embeds,
        attention_mask=torch.ones(ids.shape, dtype=torch.long, device="cuda"),
        use_cache=True,
    )
    actor.hidden = step.last_hidden_state[:, -1]
    actor.cache = step.past_key_values
    actor.mask = torch.ones(ids.shape, dtype=torch.long, device="cuda")
    actor.frames = []
    actor.eos = False


def render(actor: Actor, phrase: str, prev_wav: np.ndarray | None) -> np.ndarray:
    if prev_wav is None:
        actor.prefill_messages([
            {"role": "system", "content": "Respond with interleaved text and audio."},
            {"role": "user", "content": "Continue the conversation."},
        ])
    else:
        prefill_audio_out(actor, prev_wav, "Continue the conversation.")
    actor.force_text(phrase)
    actor.speak(60)
    return actor.samples()


def leading_ms(wav: np.ndarray, thr: float = 0.01) -> int:
    hop = int(0.01 * SR_A)
    n = len(wav) // hop
    rms = np.sqrt((wav[: n * hop].reshape(n, hop) ** 2).mean(1))
    loud = np.flatnonzero(rms > thr)
    return int(loud[0]) * 10 if len(loud) else n * 10


def median_f0(wav: np.ndarray) -> float | None:
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, wav, SR_A)
    y, _ = librosa.load(path, sr=SR_A, mono=True)
    os.unlink(path)
    f0, _, _ = librosa.pyin(y, fmin=70, fmax=400, sr=SR_A,
                            frame_length=2048, hop_length=480)
    f0 = f0[~np.isnan(f0)]
    return float(np.median(f0)) if len(f0) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "m1")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    actor = Actor()
    sys.path.insert(0, str(ROOT))
    from app.asr import QwenTranscriber  # noqa: PLC0415

    judge = QwenTranscriber()

    p1 = render(actor, PHRASE1, None)
    p2_base = render(actor, PHRASE2, None)
    p2_m1 = render(actor, PHRASE2, p1)

    rows = {"phrase1": p1, "baseline_phrase2": p2_base, "m1_phrase2": p2_m1}
    report = {}
    f0_p1_tail = median_f0(p1[-int(0.30 * SR_A):])
    for name, wav in rows.items():
        path = args.out / f"{name}.wav"
        sf.write(path, wav, SR_A)
        hyp = judge.transcribe(path).text
        ref = PHRASE2 if "phrase2" in name else PHRASE1
        ref_w = re.findall(r"[a-z0-9']+", ref.lower())
        got = re.findall(r"[a-z0-9']+", hyp.lower())
        sm = __import__("difflib").SequenceMatcher(a=ref_w, b=got, autojunk=False)
        hit = sum(b.size for b in sm.get_matching_blocks())
        f0 = median_f0(wav[: int(0.30 * SR_A)])
        report[name] = {
            "s": round(len(wav) / SR_A, 2),
            "lead_ms": leading_ms(wav),
            "words_per_s": round(len(got) / max(len(wav) / SR_A, 1e-9), 2),
            "recall": round(hit / max(len(ref_w), 1), 3),
            "f0_start": None if f0 is None else round(f0, 1),
            "f0_jump_vs_p1_tail": (None if (f0 is None or f0_p1_tail is None)
                                   else round(abs(f0 - f0_p1_tail), 1)),
            "asr": hyp,
        }
        print({name: report[name]}, flush=True)

    (args.out / "m1_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
