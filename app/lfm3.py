"""LFM3-Audio-2B on the bigger device: speech to text, then the summary.

One model, loaded once, serving both stage-1 jobs. The summary is prose because
that is what a 2B can do reliably; structured, quote-grounded events measured worse
than a coin-flip-free 9B (see docs/goal.md) and are not part of this contract.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import soundfile as sf

from . import config
from .asr import Transcript

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
MAX_SUMMARY_CHARS = 6000
SUMMARY_PROMPT = ("Summarize the conversation in three sentences. Do not explain your "
                  "reasoning. Output only the summary.")



def clean_output(raw: str) -> str:
    """Drop a thinking block; an unclosed one means the visible answer never arrived."""
    if "<think>" in raw or "\n\n" in raw and raw.lstrip().startswith("<think>"):
        head, sep, tail = raw.partition("</think>")
        return tail.strip() if sep else ""
    return raw.strip()


class Lfm3:
    """Base LFM3-Audio-2B: audio in for STT, text in for the summary."""

    def __init__(self, model_id: str = config.ASR_LFM3_MODEL,
                 max_new_tokens: int = config.ASR_MAX_NEW_TOKENS) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._processor = None
        self._model = None

    # ---------- model ----------
    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32
        kwargs: dict = {"dtype": dtype}
        if use_cuda:
            kwargs["device_map"] = "auto"

        log.info("loading %s (cuda=%s, dtype=%s)", self.model_id, use_cuda, dtype)
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        try:
            self._model = AutoModelForMultimodalLM.from_pretrained(
                self.model_id, trust_remote_code=True, **kwargs
            )
        except TypeError:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            self._model = AutoModelForMultimodalLM.from_pretrained(
                self.model_id, trust_remote_code=True, **kwargs
            )
        log.info("model ready on %s", self._model.device)

    def _generate(self, messages: list[dict], max_new_tokens: int) -> str:
        import torch

        self._load()
        prompt = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self._processor(text=prompt, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, generation_mode="text"
            )
        seq = output.sequences if hasattr(output, "sequences") else output
        generated = seq[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    # ---------- jobs ----------
    def transcribe(self, path: Path) -> Transcript:
        import librosa

        info = sf.info(str(path))
        wav, sr = sf.read(str(path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)

        self._load()
        import torch

        messages = [
            {"role": "system",
             "content": ("Transcribe the speech verbatim. Output only the words that "
                         "were spoken, with normal punctuation and no commentary.")},
            {"role": "user", "content": [{"type": "audio"}]},
        ]
        prompt = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self._processor(
            text=prompt, audio=[wav], sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).to(self._model.device)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, generation_mode="text"
            )
        seq = output.sequences if hasattr(output, "sequences") else output
        generated = seq[:, inputs["input_ids"].shape[1]:]
        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]

        return Transcript(text=text.strip(), language=None,
                          duration_s=float(info.duration))

    def summarize(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if len(text) <= MAX_SUMMARY_CHARS:
            return self._summary_once(text)
        # long transcript: summarize chunks, then summarize the summaries
        parts, buf = [], ""
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if len(buf) + len(sentence) + 1 > MAX_SUMMARY_CHARS and buf:
                parts.append(buf)
                buf = ""
            buf = f"{buf} {sentence}".strip()
        if buf:
            parts.append(buf)
        partials = [self._summary_once(p) for p in parts]
        return self._summary_once(" ".join(partials))

    def _summary_once(self, text: str) -> str:
        raw = self._generate(
            [{"role": "system", "content": SUMMARY_PROMPT},
             {"role": "user", "content": text}],
            max_new_tokens=400,
        )
        return clean_output(raw)
