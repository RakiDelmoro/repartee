"""Speech recognition: LFM3-Audio-2B (runtime) or Qwen3-ASR (reference)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from . import config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None
    duration_s: float


class QwenTranscriber:
    """Lazy wrapper around Qwen3-ASR. Loads the model on first use."""

    def __init__(
        self,
        model_id: str = config.ASR_MODEL,
        language: str | None = config.ASR_LANGUAGE,
        max_new_tokens: int = config.ASR_MAX_NEW_TOKENS,
    ) -> None:
        self.model_id = model_id
        self.language = language
        self.max_new_tokens = max_new_tokens
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        use_cuda = torch.cuda.is_available()
        # Turing (sm_75) has no native bfloat16, so prefer fp16 there.
        dtype = torch.float16 if use_cuda else torch.float32
        kwargs: dict = {"dtype": dtype}
        if use_cuda:
            kwargs["device_map"] = "auto"

        log.info("loading %s (cuda=%s, dtype=%s)", self.model_id, use_cuda, dtype)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        try:
            self._model = AutoModelForMultimodalLM.from_pretrained(self.model_id, **kwargs)
        except TypeError:
            # transformers < 5 spells it torch_dtype
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            self._model = AutoModelForMultimodalLM.from_pretrained(self.model_id, **kwargs)
        log.info("model ready on %s", self._model.device)

    def transcribe(self, path: Path) -> Transcript:
        info = sf.info(str(path))
        self._load()

        inputs = self._processor.apply_transcription_request(
            audio=str(path), language=self.language
        ).to(self._model.device, self._model.dtype)

        output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        parsed = self._processor.decode(generated, return_format="parsed")[0]

        return Transcript(
            text=(parsed.get("transcription") or "").strip(),
            language=parsed.get("language"),
            duration_s=float(info.duration),
        )


def make_transcriber():
    """Runtime transcriber: LFM3-Audio-2B by default, Qwen3-ASR on request.

    The LFM3 object also carries `summarize`; Qwen only transcribes, so a Qwen run
    stores transcripts without a summary.
    """
    if config.ASR_ENGINE == "qwen":
        return QwenTranscriber()
    if config.ASR_ENGINE == "lfm3":
        from .lfm3 import Lfm3

        return Lfm3()
    raise ValueError(f"unknown ASR engine: {config.ASR_ENGINE!r}")
