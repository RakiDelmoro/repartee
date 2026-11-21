"""LibriTTS-R -> Vani-M 17-stream P1 grids (chapter-level, dense read speech).

Each chapter is concatenated with 100 ms gaps.  Text tokens are distributed
across each utterance's frames.  User streams are zero (P1).  A held-out
speaker and a 192-frame target segment are selected for the gate.

    python -m vani.build_libritts --root data/libritts_r --out data/vani_p0_libritts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from transformers import AutoTokenizer

from .data.mimi import CODEBOOKS, SAMPLE_RATE, encode_chunks, load_mimi

TEXT_PAD = 151_643
FRAME_SAMPLES = 1_920
GAP = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)


def align_text(tokenizer, text: str, start: int, frames: int, out: np.ndarray) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return 0
    n = len(ids)
    for i, token in enumerate(ids):
        frame = start if n == 1 else start + round(i * (frames - 1) / (n - 1))
        while frame < start + frames and out[frame] != TEXT_PAD:
            frame += 1
        if frame < start + frames:
            out[frame] = int(token)
    return len(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/libritts_r"))
    ap.add_argument("--out", type=Path, default=Path("data/vani_p0_libritts"))
    ap.add_argument("--tokenizer", default="models/Qwen3-0.6B-Base")
    ap.add_argument("--mimi", default="models/mimi")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--holdout-speaker", default=None)
    args = ap.parse_args()

    wavs = sorted(args.root.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"no wav files under {args.root}")
    chapters: dict[Path, list[Path]] = {}
    for wav in wavs:
        chapters.setdefault(wav.parent, []).append(wav)
    speakers = sorted({p.parent.name for p in chapters})
    holdout = args.holdout_speaker or speakers[-1]
    print(f"chapters={len(chapters)} speakers={len(speakers)} holdout={holdout}",
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_mimi(args.mimi, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    total_seconds = 0.0
    for chapter, files in sorted(chapters.items()):
        speaker = chapter.parent.name
        audio_parts = []
        text = np.full(0, TEXT_PAD, dtype=np.int32)
        cursor = 0
        for wav in sorted(files):
            txt_path = wav.with_suffix(".normalized.txt")
            if not txt_path.exists():
                txt_path = wav.with_suffix(".original.txt")
            if not txt_path.exists():
                continue
            x, sr = sf.read(wav, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            assert sr == SAMPLE_RATE, f"{wav}: {sr}"
            frames = len(x) // FRAME_SAMPLES
            text = np.pad(text, (0, frames), constant_values=TEXT_PAD)
            align_text(tokenizer, txt_path.read_text().strip(), cursor, frames, text)
            audio_parts.append(x)
            audio_parts.append(GAP)
            cursor += frames + len(GAP) // FRAME_SAMPLES
        if not audio_parts:
            continue
        audio = np.concatenate(audio_parts)
        codes = encode_chunks(model, audio)
        frames = len(codes)
        text = text[:frames]
        if len(text) < frames:
            text = np.pad(text, (0, frames - len(text)), constant_values=TEXT_PAD)
        if len(audio) < frames * FRAME_SAMPLES:
            audio = np.pad(audio, (0, frames * FRAME_SAMPLES - len(audio)))
        grid = np.zeros((frames, 1 + 2 * CODEBOOKS), dtype=np.int32)
        grid[:, 0] = text
        grid[:, 1 : 1 + CODEBOOKS] = codes
        stem = f"conv_{speaker}_{chapter.name}"
        np.save(args.out / f"{stem}.grid.npy", grid)
        sf.write(args.out / f"{stem}.original.wav", audio, SAMPLE_RATE)
        rms = np.sqrt(
            (audio[: frames * FRAME_SAMPLES].reshape(frames, FRAME_SAMPLES) ** 2).mean(1)
        ).astype(np.float32)
        np.save(args.out / f"{stem}.rms.npy", rms)
        meta = {
            "source": f"{speaker}_{chapter.name}",
            "speaker": speaker,
            "frames": frames,
            "seconds": round(frames / 12.5, 2),
            "text_tokens": int((text != TEXT_PAD).sum()),
            "agent_is_a": True,
        }
        (args.out / f"{stem}.json").write_text(json.dumps(meta) + "\n")
        total_frames += frames
        total_seconds += frames / 12.5
    print(f"wrote {len(chapters)} chapters, {total_frames} frames, "
          f"{total_seconds/3600:.2f} h", flush=True)

    # Held-out gate: longest chapter of the holdout speaker, a 192-frame target.
    holdout_chapters = sorted(p for p in chapters if p.parent.name == holdout)
    selection = None
    for chapter in holdout_chapters:
        stem = f"conv_{holdout}_{chapter.name}"
        grid_path = args.out / f"{stem}.grid.npy"
        if not grid_path.exists():
            continue
        grid = np.load(grid_path, mmap_mode="r")
        text = np.asarray(grid[:, 0], dtype=np.int64)
        if len(text) < 320:
            continue
        # target 192 frames starting after a 64-frame prompt
        best = None
        for target in range(64, len(text) - 192, 16):
            n = int((text[target : target + 192] != TEXT_PAD).sum())
            if best is None or n > best[1]:
                best = (target, n)
        if best is None or best[1] < 8:
            continue
        target = best[0]
        ids = [int(t) for t in text[target : target + 192] if t != TEXT_PAD]
        reference = tokenizer.decode(ids, skip_special_tokens=True).strip()
        if len(reference.split()) < 5:
            continue
        selection = {
            "speaker": holdout,
            "grid": str(grid_path),
            "start_frame": target - 64,
            "prompt_frames": 64,
            "frames": 192,
            "reference": str(args.out / "holdout_reference.txt"),
            "target_start": target,
        }
        (args.out / "holdout_reference.txt").write_text(reference + "\n")
        break
    (args.out / "holdout.json").write_text(
        json.dumps({"speaker": holdout, "selection": selection}, indent=2) + "\n"
    )
    print(f"holdout {holdout} selection {selection}", flush=True)


if __name__ == "__main__":
    main()
