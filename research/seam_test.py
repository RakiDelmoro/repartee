#!/usr/bin/env python3
"""Seam test: is a text handover spliced because of the moment, the cue, or the actor?

Four conditions, same answer text, same actor:

  A  inject at a natural boundary   (after the filler's own end-of-speech)
  B  inject mid-speech              (three frames into the first voiced run)
  C  boundary + cue phrase          (filler ends "...right, so —")
  D  filler only                    (reference for what the filler alone sounds like)

The answer is fed as 6-token blocks with audio between them, which is the
interleaving the model was trained on; one giant text block is out of distribution
and ends the utterance early.

Metrics: silence gap at the seam, answer word recall (ASR on the answer slice),
duplicate word at the seam, pitch step across the seam vs within the answer.

  .venv/bin/python research/seam_test.py --out data/seam
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

from app import config  # noqa: E402

MODEL = config.ASR_LFM3_MODEL
SR_A = 24_000
FRAME = 0.08
FRAME_SAMPLES = int(SR_A * FRAME)
N_TEXT = 6
N_AUDIO = 12
TEMPERATURE = 0.7
TOP_K = 50
TEXT_REP = 1.1
AUDIO_TEMP = 0.4
AUDIO_TOPK = 128
AUDIO_REP = 1.2
RMS_THR = 0.01

FILLER = "Let me think about that for a second."
CUE = "Let me think about that for a second. Right, so,"
ANSWER = ("The launch is moving to Q2, in the second week of April. "
          "Tom will document it so nobody keeps planning against March.")
QUESTION = "The team asked: when is the launch now, and who owns it?"
CLAUSES = ["The launch is moving to Q2,", "in the second week of April.",
           "Tom will document it", "so nobody keeps planning against March."]
SENTENCES = ["The launch is moving to Q2, in the second week of April.",
             "Tom will document it so nobody keeps planning against March."]


class Actor:
    def __init__(self, model=None, proc=None) -> None:
        self.proc = proc or AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        self.model = model or AutoModelForMultimodalLM.from_pretrained(
            MODEL, dtype=torch.float16, trust_remote_code=True
        ).to("cuda").eval()
        self.cfg = self.model.config
        self.codec = self.proc.audio_tokenizer.to(device="cuda", dtype=torch.float32).eval()
        self.stop = {self.cfg.text_end_token_id}
        self.frames: list[torch.Tensor] = []
        self.eos = False

    def prefill(self, user_text: str) -> None:
        self.prefill_messages([
            {"role": "system", "content": "Respond with interleaved text and audio."},
            {"role": "user", "content": user_text},
        ])

    def prefill_messages(self, messages: list[dict]) -> None:
        ids = self.proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        )["input_ids"]
        ids = torch.tensor(ids, dtype=torch.long, device="cuda").reshape(1, -1)
        mix = torch.tensor([[self.cfg.mixed_start_token_id]], device="cuda")
        ids = torch.cat([ids, mix], 1)
        embeds = self.model.get_input_embeddings()(ids)
        out = self.model.model.language_model(
            inputs_embeds=embeds,
            attention_mask=torch.ones(ids.shape, dtype=torch.long, device="cuda"),
            use_cache=True,
        )
        self.hidden = out.last_hidden_state[:, -1]
        self.cache = out.past_key_values
        self.mask = torch.ones(ids.shape, dtype=torch.long, device="cuda")
        self.frames = []
        self.eos = False

    def _feed(self, e: torch.Tensor) -> None:
        if e.ndim == 2:
            e = e.unsqueeze(1)
        self.mask = torch.cat([self.mask, self.mask.new_ones((1, 1))], dim=1)
        out = self.model.model.language_model(
            inputs_embeds=e, attention_mask=self.mask,
            position_ids=torch.tensor([[self.cache.get_seq_length()]], device="cuda"),
            past_key_values=self.cache, use_cache=True,
        )
        self.hidden = out.last_hidden_state[:, -1]
        self.cache = out.past_key_values

    def feed_token(self, tok: int) -> None:
        self._feed(self.model.get_input_embeddings()(
            torch.tensor([[tok]], device="cuda")))

    def force_text(self, text: str) -> None:
        for t in self.proc.tokenizer.encode(text, add_special_tokens=False):
            self.feed_token(t)
        self.feed_token(self.cfg.text_end_token_id)

    def frame(self) -> bool:
        """One audio frame; returns True on end-of-speech (never decoded)."""
        hist = torch.stack(self.frames[-64:]) if self.frames else None
        codes = self.model.sample_audio_frame(
            self.hidden[0], temperature=AUDIO_TEMP, top_k=AUDIO_TOPK,
            repetition_penalty=AUDIO_REP, history=hist,
        )
        if bool((codes == self.cfg.audio_eos_code).any()):
            return True
        self.frames.append(codes)
        self._feed(self.model.model.get_audio_out_embeds(codes[None]))
        return False

    def speak(self, max_frames: int) -> int:
        n = 0
        while n < max_frames:
            if self.frame():
                break
            n += 1
        return n

    def inject(self, text: str, max_frames: int = 140) -> None:
        """Handed text as one text block, then speech — the pattern `speak_text` used
        in the product demo, which is the only one measured to voice a full answer."""
        self.force_text(text)
        self.speak(max_frames)

    def samples(self, frames=None) -> np.ndarray:
        frames = self.frames if frames is None else frames
        if not frames:
            return np.zeros(0, dtype=np.float32)
        codes = torch.stack(frames).t()
        with torch.no_grad():
            wav = self.codec.decode(codes[None]).audio_values[0].float().cpu().numpy()
        return np.asarray(wav).squeeze()


def frame_rms(wav: np.ndarray) -> np.ndarray:
    n = len(wav) // FRAME_SAMPLES
    if n == 0:
        return np.zeros(0)
    return np.sqrt((wav[: n * FRAME_SAMPLES].reshape(n, FRAME_SAMPLES) ** 2).mean(1))


def first_voiced(rms: np.ndarray) -> int:
    idx = np.flatnonzero(rms > RMS_THR)
    return int(idx[0]) if len(idx) else 0


PAUSE_S = 0.24  # a turn boundary, not a splice


def run_condition(actor: Actor, filler: str, mode: str) -> dict:
    """mode: boundary | mid | cue | filler_only | cue_pause | cue_pause_turn"""
    torch.manual_seed(0)
    np.random.seed(0)
    actor.prefill(QUESTION)
    actor.force_text(filler)
    if mode == "mid":
        cap = 80
        actor.speak(cap)
        rms = frame_rms(actor.samples())
        inject_at = first_voiced(rms) + 3
        # rewind: we generated too far; rebuild up to the injection point
        target = max(3, inject_at)
        actor.prefill(QUESTION)
        actor.force_text(filler)
        actor.speak(target)
    else:
        # block boundary: two full audio blocks, before the filler's end-of-speech.
        # Injecting after end-of-speech was measured dead: the model treats the
        # utterance as finished and speaks nothing.
        actor.speak(2 * N_AUDIO)

    junction = len(actor.frames)
    filler_wav = actor.samples()

    if mode == "phrases":
        parts = []
        for phrase in CLAUSES:
            actor.prefill_messages([
                {"role": "system", "content": "Respond with interleaved text and audio."},
                {"role": "user", "content": QUESTION},
            ])
            actor.force_text(phrase)
            actor.speak(30)
            parts.append(actor.samples())
        gap = np.zeros(int(0.12 * SR_A), dtype=np.float32)
        seq = []
        for part in parts:
            seq += [part, gap]
        wav = np.concatenate([filler_wav, np.zeros(int(PAUSE_S * SR_A), np.float32)] + seq[:-1])
        offset = len(filler_wav) + int(PAUSE_S * SR_A)
        return {"wav": wav, "junction_frame": junction,
                "junction_samples": offset, "natural_end": actor.eos,
                "filler_frames": junction,
                "injected_frames": int((len(wav) - len(filler_wav)) / FRAME_SAMPLES),
                "answer_offset_samples": offset,
                "sentence_samples": [len(x) for x in parts]}

    if mode == "chunked_clean":
        parts = []
        for sentence in SENTENCES:
            actor.prefill_messages([
                {"role": "system", "content": "Respond with interleaved text and audio."},
                {"role": "user", "content": QUESTION},
            ])
            toks = actor.proc.tokenizer.encode(sentence, add_special_tokens=False)
            for i in range(0, len(toks), N_TEXT):
                for t in toks[i:i + N_TEXT]:
                    actor.feed_token(t)
                if i + N_TEXT >= len(toks):
                    actor.feed_token(actor.cfg.text_end_token_id)
                actor.speak(N_AUDIO)
            parts.append(actor.samples())
        p1 = np.zeros(int(PAUSE_S * SR_A), dtype=np.float32)
        p2 = np.zeros(int(0.12 * SR_A), dtype=np.float32)
        wav = np.concatenate([filler_wav, p1, parts[0], p2, parts[1]])
        offset = len(filler_wav) + len(p1)
        return {"wav": wav, "junction_frame": junction,
                "junction_samples": offset, "natural_end": actor.eos,
                "filler_frames": junction,
                "injected_frames": int((len(wav) - len(filler_wav)) / FRAME_SAMPLES),
                "answer_offset_samples": offset,
                "sentence_samples": [len(x) for x in parts]}

    if mode == "clean_turn":
        # the actor as its own TTS: a fresh turn per sentence, no filler text in the
        # history (that is what leaked before), cue kept in audio only
        parts = []
        for sentence in SENTENCES:
            actor.prefill_messages([
                {"role": "system", "content": "Respond with interleaved text and audio."},
                {"role": "user", "content": QUESTION},
            ])
            actor.force_text(sentence)
            actor.speak(40)
            parts.append(actor.samples())
        p1 = np.zeros(int(PAUSE_S * SR_A), dtype=np.float32)
        p2 = np.zeros(int(0.12 * SR_A), dtype=np.float32)
        wav = np.concatenate([filler_wav, p1, parts[0], p2, parts[1]])
        offset = len(filler_wav) + len(p1)
        return {"wav": wav, "junction_frame": junction,
                "junction_samples": offset, "natural_end": actor.eos,
                "filler_frames": junction,
                "injected_frames": int((len(wav) - len(filler_wav)) / FRAME_SAMPLES),
                "answer_offset_samples": offset,
                "sentence_samples": [len(x) for x in parts]}

    if mode == "final":
        # one generation, same voice: cue only in audio, one short sentence at a time,
        # pauses spliced at the boundaries
        marks = []
        for sentence in SENTENCES:
            actor.force_text(sentence)
            actor.speak(30)
            marks.append(len(actor.frames))
        j0, j1, j2 = junction, marks[0], marks[-1]
        filler_wav = actor.samples(actor.frames[:j0])
        s1 = actor.samples(actor.frames[j0:j1])
        s2 = actor.samples(actor.frames[j1:j2])
        p1 = np.zeros(int(PAUSE_S * SR_A), dtype=np.float32)
        p2 = np.zeros(int(0.12 * SR_A), dtype=np.float32)
        wav = np.concatenate([filler_wav, p1, s1, p2, s2])
        offset = len(filler_wav) + len(p1)
        return {"wav": wav, "junction_frame": junction,
                "junction_samples": offset, "natural_end": actor.eos,
                "filler_frames": junction,
                "injected_frames": int((len(wav) - len(filler_wav)) / FRAME_SAMPLES),
                "answer_offset_samples": offset,
                "sentence_samples": [len(s1), len(s2)]}

    if mode == "sentences":
        parts, spoken = [], []
        for sentence in SENTENCES:
            actor.prefill_messages([
                {"role": "system", "content": "Respond with interleaved text and audio."},
                {"role": "user", "content": QUESTION},
                {"role": "assistant", "content": " ".join([filler] + spoken)},
            ])
            actor.force_text(sentence)
            actor.speak(40)
            parts.append(actor.samples())
            spoken.append(sentence)
        pause = np.zeros(int(PAUSE_S * SR_A), dtype=np.float32)
        wav = np.concatenate([filler_wav, pause, parts[0], pause, parts[1]])
        return {"wav": wav, "junction_frame": junction,
                "junction_samples": len(filler_wav) + len(pause),
                "natural_end": actor.eos, "filler_frames": junction,
                "injected_frames": int((len(wav) - len(filler_wav)) / FRAME_SAMPLES),
                "answer_offset_samples": len(filler_wav) + len(pause)}

    if mode == "clauses":
        actor.prefill_messages([
            {"role": "system", "content": "Respond with interleaved text and audio."},
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": filler},
        ])
        for clause in CLAUSES:
            actor.force_text(clause)
            actor.speak(20)
        answer_wav = actor.samples()
        pause = np.zeros(int(PAUSE_S * SR_A), dtype=np.float32)
        wav = np.concatenate([filler_wav, pause, answer_wav])
        return {"wav": wav, "junction_frame": junction,
                "junction_samples": len(filler_wav) + len(pause),
                "natural_end": actor.eos, "filler_frames": junction,
                "injected_frames": int(len(answer_wav) / FRAME_SAMPLES),
                "answer_offset_samples": len(filler_wav) + len(pause)}

    if mode in ("cue_pause", "cue_pause_turn"):
        if mode == "cue_pause_turn":
            # the answer is the next assistant turn, with the filler in history
            actor.prefill_messages([
                {"role": "system", "content": "Respond with interleaved text and audio."},
                {"role": "user", "content": QUESTION},
                {"role": "assistant", "content": filler},
            ])
        actor.inject(ANSWER)
        answer_wav = actor.samples()
        pause = np.zeros(int(PAUSE_S * SR_A), dtype=np.float32)
        wav = np.concatenate([filler_wav, pause, answer_wav])
        return {"wav": wav, "junction_frame": junction,
                "junction_samples": len(filler_wav) + len(pause),
                "natural_end": actor.eos, "filler_frames": junction,
                "injected_frames": int(len(answer_wav) / FRAME_SAMPLES),
                "answer_offset_samples": len(filler_wav) + len(pause)}

    if mode != "filler_only":
        actor.inject(ANSWER)
    wav = actor.samples()
    return {"wav": wav, "junction_frame": junction,
            "junction_samples": int(junction * FRAME * SR_A),
            "natural_end": actor.eos, "filler_frames": junction,
            "injected_frames": len(actor.frames) - junction,
            "answer_offset_samples": int(junction * FRAME * SR_A)}


def normalize(text: str) -> list[str]:
    text = text.lower().replace("q2", "q two").replace("q 2", "q two")
    return re.findall(r"[a-z']+", text)


def word_recall(ref: list[str], got: list[str]) -> float:
    if not ref:
        return 0.0
    sm = __import__("difflib").SequenceMatcher(a=ref, b=got, autojunk=False)
    matched = sum(block.size for block in sm.get_matching_blocks())
    return matched / len(ref)


def seam_metrics(wav: np.ndarray, junction: int, junction_samples: int | None = None) -> dict:
    rms = frame_rms(wav)
    j = (int(junction_samples / FRAME_SAMPLES) if junction_samples is not None
         else min(junction, len(rms)))
    before = 0
    i = j - 1
    while i >= 0 and rms[i] < RMS_THR:
        before += 1
        i -= 1
    after = 0
    i = j
    while i < len(rms) and rms[i] < RMS_THR:
        after += 1
        i += 1
    return {"gap_ms": round((before + after) * FRAME * 1000, 1),
            "frames": len(rms), "junction_frame": j}


def pitch_step(wav: np.ndarray, junction: int,
               answer_offset: int | None = None) -> dict:
    fd, path = __import__("tempfile").mkstemp(suffix=".wav")
    import os
    os.close(fd)
    sf.write(path, wav, SR_A)
    y, _ = librosa.load(path, sr=SR_A, mono=True)
    os.unlink(path)
    f0, _, _ = librosa.pyin(y, fmin=70, fmax=400, sr=SR_A,
                            frame_length=2048, hop_length=480)
    t = np.arange(len(f0)) * 480 / SR_A
    if answer_offset is None:
        jt = junction * FRAME
        before = f0[(t > jt - 0.30) & (t < jt) & ~np.isnan(f0)]
        after = f0[(t >= jt) & (t < jt + 0.30) & ~np.isnan(f0)]
    else:
        jt = answer_offset / SR_A
        before = f0[(t > jt - 0.55) & (t < jt - PAUSE_S - 0.02) & ~np.isnan(f0)]
        after = f0[(t >= jt) & (t < jt + 0.30) & ~np.isnan(f0)]
    within = np.abs(np.diff(f0[~np.isnan(f0)]))
    step = (float(abs(np.median(after) - np.median(before)))
            if len(before) and len(after) else None)
    return {"pitch_step_hz": None if step is None else round(step, 1),
            "within_step_hz": round(float(np.median(within)), 1) if len(within) else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "seam")
    ap.add_argument("--conditions", default="A,B,C,E,F")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    actor = Actor()
    want = set((args.conditions or "A,B,C,E,F").split(","))
    catalogue = {
        "A": ("A_boundary", FILLER, "boundary"),
        "B": ("B_mid_speech", FILLER, "mid"),
        "C": ("C_cued_boundary", CUE, "cue"),
        "D": ("D_filler_only", FILLER, "filler_only"),
        "E": ("E_cue_pause", CUE, "cue_pause"),
        "F": ("F_cue_pause_turn", CUE, "cue_pause_turn"),
        "G": ("G_clauses", CUE, "clauses"),
        "H": ("H_sentence_turns", CUE, "sentences"),
        "I": ("I_final", CUE, "final"),
        "J": ("J_clean_turn", CUE, "clean_turn"),
        "K": ("K_chunked_clean", CUE, "chunked_clean"),
        "L": ("L_phrases", CUE, "phrases"),
    }
    conditions = [v for k, v in catalogue.items() if k in want]
    rendered = {}
    for name, filler, mode in conditions:
        r = run_condition(actor, filler, mode)
        path = args.out / f"{name}.wav"
        sf.write(path, r["wav"], SR_A)
        r["path"] = path
        rendered[name] = r
        print({"condition": name, "s": round(len(r["wav"]) / SR_A, 2),
               "junction_s": round(r["junction_frame"] * FRAME, 2),
               "natural_end": r["natural_end"],
               "injected_frames": r["injected_frames"]}, flush=True)

    del actor
    torch.cuda.empty_cache()

    from app.asr import QwenTranscriber
    judge = QwenTranscriber()
    ref = normalize(ANSWER)
    report = {"answer": ANSWER, "conditions": {}}
    for name, r in rendered.items():
        wav = r["wav"]
        j = r["junction_frame"]
        m = seam_metrics(wav, j, r.get("junction_samples"))
        p = pitch_step(wav, j, r.get("answer_offset_samples"))
        if name == "D_filler_only":
            answer_wav = np.zeros(0, dtype=np.float32)
        else:
            answer_wav = wav[r["answer_offset_samples"]:]
        apath = args.out / f"{name}.answer.wav"
        if len(answer_wav):
            sf.write(apath, answer_wav, SR_A)
            hyp = judge.transcribe(apath).text
        else:
            hyp = ""
        got = normalize(hyp)
        report["conditions"][name] = {
            **m, **p, "answer_recall": round(word_recall(ref, got), 3),
            "duplicate_at_seam": any(a == b for a, b in zip(got, got[1:])),
            "asr_answer": hyp,
        }
        print({"condition": name, **{k: v for k, v in report["conditions"][name].items()
                                     if k != "asr_answer"}}, flush=True)

    (args.out / "seam_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
