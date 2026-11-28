"""Bigger device: LFM3-Omni-2B speech to text for receiver WAVs.

The model is 2.17 B parameters, so it runs on the bigger device, not on the
ESP32. The device side stays a VAD + recorder; this module turns its finished
WAVs into text.

Recordings are split into <= 60 s chunks. A chunk boundary is moved backwards
to the last pause in the final 20 % of the window, so a word is not cut in
half; a hard cut at 60 s is the fallback when the speech never pauses. The
encoder is quadratic in audio length, which is why 60 s is the ceiling on an
8 GB GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from . import config

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.03
PAUSE_SECONDS = 0.25
MIN_CHUNK_SECONDS = 0.5


@dataclass(frozen=True)
class Chunk:
    start_s: float
    end_s: float


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str
    duration_s: float
    chunks: tuple[Chunk, ...]
    model: str
    elapsed_s: float


def clean_text(raw: str) -> str:
    """Drop the empty ``<think></think>`` prefix the Omni template emits."""
    raw = raw.strip()
    if raw.startswith("<think>"):
        _, sep, tail = raw.partition("</think>")
        raw = tail if sep else ""
    return raw.strip()


def load_audio(path: Path) -> tuple["np.ndarray", int]:
    """Read any WAV/FLAC as contiguous 16 kHz mono float32."""
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = wav.mean(axis=1, dtype=np.float32) if wav.shape[1] > 1 else wav[:, 0]
    if sr != SAMPLE_RATE:
        import librosa

        mono = librosa.resample(mono, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.ascontiguousarray(mono, dtype=np.float32), SAMPLE_RATE


def _frame_rms(wav: "np.ndarray", frame: int) -> "np.ndarray":
    import numpy as np

    count = len(wav) // frame
    if count == 0:
        return np.empty(0, dtype=np.float32)
    frames = wav[: count * frame].reshape(count, frame)
    return np.sqrt(np.mean(frames * frames, axis=1, dtype=np.float32) + 1e-12)


def _pause_cut(rms: "np.ndarray", sr: int, start_s: float, limit_s: float) -> float | None:
    """Last quiet run in ``[start_s, limit_s]``, cut in its middle."""
    frame_s = FRAME_SECONDS
    quiet = rms < _quiet_threshold(rms)
    pause_frames = max(1, round(PAUSE_SECONDS / frame_s))
    lo = max(0, int(start_s / frame_s))
    i = min(len(quiet) - 1, int(limit_s / frame_s))
    while i >= lo:
        if not quiet[i]:
            i -= 1
            continue
        j = i
        while j > lo and quiet[j - 1]:
            j -= 1
        if i - j + 1 >= pause_frames:
            return (j + (i - j + 1) / 2) * frame_s
        i = j - 1
    return None


def _quiet_threshold(rms: "np.ndarray") -> float:
    import numpy as np

    # Relative to the clip's own noise floor, but capped below the loud level:
    # a file with no pauses has noise == speech, and a pure noise-floor
    # threshold would call every frame a pause.
    noise = float(np.percentile(rms, 10)) if rms.size else 0.0
    loud = float(np.percentile(rms, 90)) if rms.size else 0.0
    return max(0.003, min(noise * 2.5, loud * 0.5))


def plan_chunks(wav: "np.ndarray", sr: int, max_seconds: float = 60.0) -> list[Chunk]:
    """Split audio into contiguous chunks of at most ``max_seconds``.

    The cut is placed at the last >= 250 ms pause in the final 20 % of the
    window. If there is no pause, the hard limit wins.
    """
    if max_seconds <= MIN_CHUNK_SECONDS:
        raise ValueError(f"max_seconds must exceed {MIN_CHUNK_SECONDS}: {max_seconds}")
    total_s = len(wav) / sr
    if total_s <= 0:
        return []
    if total_s <= max_seconds:
        return [Chunk(0.0, total_s)]

    frame = int(sr * FRAME_SECONDS)
    rms = _frame_rms(wav, frame)
    search_from = max(MIN_CHUNK_SECONDS, max_seconds * 0.8)
    chunks: list[Chunk] = []
    start_s = 0.0
    while start_s < total_s:
        remaining = total_s - start_s
        if remaining <= max_seconds:
            chunks.append(Chunk(start_s, total_s))
            break
        limit_s = start_s + max_seconds
        cut = _pause_cut(rms, sr, start_s + search_from, limit_s)
        if cut is None or cut <= start_s + MIN_CHUNK_SECONDS:
            cut = limit_s
        chunks.append(Chunk(start_s, cut))
        start_s = cut
    return chunks


class OmniTranscriber:
    """Lazy LFM3-Omni-2B transcriber. Loads the model on first use."""

    def __init__(
        self,
        model_id: str = config.STT_MODEL,
        language: str = config.STT_LANGUAGE,
        max_chunk_seconds: float = config.STT_CHUNK_SECONDS,
        max_new_tokens: int = config.STT_MAX_NEW_TOKENS,
    ) -> None:
        self.model_id = model_id
        self.language = language
        self.max_chunk_seconds = max_chunk_seconds
        self.max_new_tokens = max_new_tokens
        self._processor = None
        self._model = None

    # ---------- model ----------
    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        use_cuda = torch.cuda.is_available() and config.STT_DEVICE != "cpu"
        if use_cuda:
            # Turing (sm_75) reports bf16 support but has no native kernel; fp16
            # is ~4x faster there. Ampere and newer keep bf16.
            major, _ = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if major >= 8 else torch.float16
            device = config.STT_DEVICE if config.STT_DEVICE != "auto" else "cuda"
        else:
            dtype = torch.float32
            device = "cpu"

        log.info("loading %s (device=%s, dtype=%s)", self.model_id, device, dtype)
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        kwargs: dict = {"dtype": dtype, "device_map": device}
        try:
            self._model = AutoModelForMultimodalLM.from_pretrained(
                self.model_id, trust_remote_code=True, **kwargs
            )
        except TypeError:
            # transformers < 5 spells the dtype argument torch_dtype
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            self._model = AutoModelForMultimodalLM.from_pretrained(
                self.model_id, trust_remote_code=True, **kwargs
            )
        self._model.eval()
        log.info("model ready on %s", self._model.device)

    def _transcribe_chunk(self, wav: "np.ndarray", depth: int = 0) -> str:
        import soundfile as sf
        import torch

        self._load()
        fd, name = tempfile.mkstemp(suffix=".wav", prefix="repartee-")
        os.close(fd)
        tmp = Path(name)
        try:
            sf.write(tmp, wav, SAMPLE_RATE, subtype="PCM_16")
            conversation = [
                {"role": "system", "content": f"Perform ASR in {self.language}."},
                {"role": "user", "content": [{"type": "audio", "audio": str(tmp)}]},
            ]
            inputs = self._processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model.device)
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    generation_mode="text",
                )
            seq = output.sequences if hasattr(output, "sequences") else output
            end = (
                int(output.sequence_lengths[0])
                if hasattr(output, "sequence_lengths")
                else seq.shape[1]
            )
            text = self._processor.decode(
                seq[0, inputs["input_ids"].shape[1] : end],
                skip_special_tokens=True,
            )
            text = clean_text(text)
        finally:
            tmp.unlink(missing_ok=True)

        # An empty result on a long chunk is the model's early-EOS failure;
        # split once and retry, rather than silently losing the speech.
        if not text and len(wav) > SAMPLE_RATE * 4 and depth < 1:
            log.warning("empty transcript for %.1fs chunk; retrying split", len(wav) / SAMPLE_RATE)
            half = len(wav) // 2
            left = self._transcribe_chunk(wav[:half], depth + 1)
            right = self._transcribe_chunk(wav[half:], depth + 1)
            return f"{left} {right}".strip()
        return text

    # ---------- job ----------
    def transcribe(self, path: Path) -> Transcript:
        started = time.monotonic()
        wav, sr = load_audio(path)
        duration_s = len(wav) / sr
        chunks = plan_chunks(wav, sr, self.max_chunk_seconds)
        log.info("%s: %.2f s, %d chunk(s)", path.name, duration_s, len(chunks))

        pieces: list[str] = []
        for index, chunk in enumerate(chunks, 1):
            log.info("chunk %d/%d: %.2f-%.2f s", index, len(chunks), chunk.start_s, chunk.end_s)
            part = self._transcribe_chunk(
                wav[int(chunk.start_s * sr) : int(chunk.end_s * sr)]
            )
            if part:
                pieces.append(part)

        return Transcript(
            text=" ".join(pieces).strip(),
            language=self.language,
            duration_s=duration_s,
            chunks=tuple(chunks),
            model=self.model_id,
            elapsed_s=time.monotonic() - started,
        )


def _as_dict(transcript: Transcript, path: Path) -> dict:
    return {
        "audio": str(path),
        "model": transcript.model,
        "language": transcript.language,
        "duration_s": round(transcript.duration_s, 3),
        "elapsed_s": round(transcript.elapsed_s, 3),
        "chunks": [
            {"start_s": round(c.start_s, 3), "end_s": round(c.end_s, 3)}
            for c in transcript.chunks
        ],
        "text": transcript.text,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="repartee transcribe",
        description="Transcribe a receiver WAV with LFM3-Omni-2B.",
    )
    parser.add_argument("audio", type=Path, help="WAV or FLAC recording")
    parser.add_argument(
        "--model",
        default=config.STT_MODEL,
        help="HF id or local checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--language",
        default=config.STT_LANGUAGE,
        help="ASR language name (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=config.STT_CHUNK_SECONDS,
        help="maximum audio per forward pass (default: %(default)s)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=config.STT_MAX_NEW_TOKENS,
        help="generation cap per chunk (default: %(default)s)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit transcript plus chunks and timings as JSON",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="write to this file instead of stdout",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.audio.is_file():
        parser.error(f"not a file: {args.audio}")

    transcriber = OmniTranscriber(
        model_id=args.model,
        language=args.language,
        max_chunk_seconds=args.chunk_seconds,
        max_new_tokens=args.max_new_tokens,
    )
    transcript = transcriber.transcribe(args.audio)
    payload = (
        json.dumps(_as_dict(transcript, args.audio), indent=2) + "\n"
        if args.json
        else transcript.text + "\n"
    )
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
    else:
        import sys

        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
