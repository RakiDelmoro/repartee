"""Deeper audio/code diagnostics for one P1 generated segment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.asr import QwenTranscriber  # noqa: E402
from vani.asr_p1 import wer  # noqa: E402
from vani.data.mimi import SAMPLE_RATE, decode_chunks, load_mimi  # noqa: E402

START = 21753
PROMPT = 64
FRAMES = 192
GRID = ROOT / "data" / "vani_p0_45" / "conv_d65ed16a6092.grid.npy"
GEN_CODES = ROOT / "data" / "vani_p1_dense" / "textforced" / "codes.npy"
TARGET_TEXT = ROOT / "data" / "vani_p1_dense" / "target_dense.txt"
OUT = ROOT / "data" / "vani_p1_deep"


def boundary_ratio(x: np.ndarray, frame: int = 1920) -> float:
    d = np.abs(np.diff(x))
    if len(x) <= frame:
        return 0.0
    idx = np.arange(frame, len(x), frame)
    b = np.abs(x[idx] - x[idx - 1])
    return float(np.median(b) / (np.median(d) + 1e-9))


def spectral_flux(x: np.ndarray) -> float:
    s = np.abs(librosa.stft(x.astype(np.float32), n_fft=512, hop_length=240))
    return float(np.mean(np.abs(np.diff(s, axis=1))))


def code_metrics(gen: np.ndarray, truth: np.ndarray) -> dict:
    out = {}
    trans, runs, acc = [], [], []
    for cb in range(8):
        g, t = gen[:, cb], truth[:, cb]
        trans.append(float(np.mean(g[1:] != g[:-1])))
        runs.append(float(1.0 / (np.mean(g[1:] != g[:-1]) + 1e-9)))
        acc.append(float(np.mean(g == t)))
    out["transition_rate"] = [round(v, 4) for v in trans]
    out["mean_run_frames"] = [round(v, 2) for v in runs]
    out["accuracy"] = [round(v, 4) for v in acc]
    offsets = {}
    for off in (-2, -1, 0, 1, 2):
        vals = []
        for cb in range(8):
            if off < 0:
                g, t = gen[-off:, cb], truth[:off, cb]
            elif off > 0:
                g, t = gen[:-off, cb], truth[off:, cb]
            else:
                g, t = gen[:, cb], truth[:, cb]
            vals.append(float(np.mean(g == t)))
        offsets[str(off)] = [round(v, 4) for v in vals]
    out["offset_accuracy"] = offsets
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--language", default="English")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    grid = np.load(GRID, mmap_mode="r")
    target = np.asarray(grid[START + PROMPT : START + PROMPT + FRAMES, 1:9], dtype=np.int64)
    gen = np.load(GEN_CODES).astype(np.int64)
    text = TARGET_TEXT.read_text().strip()

    masks = [
        ("gen_none", []),
        ("gt_cb0", [0]),
        ("gt_cb0_1", [0, 1]),
        ("gt_cb0_2", [0, 1, 2]),
        ("gt_cb0_3", [0, 1, 2, 3]),
        ("gt_cb0_4", [0, 1, 2, 3, 4]),
        ("gt_cb0_5", [0, 1, 2, 3, 4, 5]),
        ("gt_cb0_6", [0, 1, 2, 3, 4, 5, 6]),
        ("gt_all", list(range(8))),
    ]

    mimi = load_mimi(str(ROOT / "models" / "mimi"), "cuda")
    variants = {}
    for name, cbs in masks:
        codes = gen.copy()
        for cb in cbs:
            codes[:, cb] = target[:, cb]
        audio = decode_chunks(mimi, codes)
        raw = args.out / f"{name}.wav"
        sf.write(raw, audio, SAMPLE_RATE)
        x = audio / (np.abs(audio).max() + 1e-9) * 0.95
        norm = args.out / f"{name}_normalized.wav"
        sf.write(norm, x, SAMPLE_RATE)
        variants[name] = {
            "wav": str(raw),
            "normalized": str(norm),
            "rms": round(float(np.sqrt(np.mean(audio**2))), 6),
            "peak": round(float(np.abs(audio).max()), 4),
            "boundary_ratio": round(boundary_ratio(audio), 3),
            "spectral_flux": round(spectral_flux(audio), 4),
            "gt_codebooks": cbs,
        }
    del mimi
    torch.cuda.empty_cache()

    # ASR all variants with one loaded model.
    asr = QwenTranscriber(model_id=str(ROOT / "models" / "Qwen3-ASR-0.6B-hf"), language=args.language)
    for name, info in variants.items():
        result = asr.transcribe(Path(info["normalized"]))
        info["transcript"] = result.text
        info["wer"] = round(wer(text, result.text), 4) if text else None

    metrics = code_metrics(gen, target)
    summary = {
        "segment_start_frame": START,
        "prompt_frames": PROMPT,
        "generated_frames": FRAMES,
        "target_words": len(text.split()),
        "target_text": text,
        "code_metrics_generated_vs_target": metrics,
        "variants": variants,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
