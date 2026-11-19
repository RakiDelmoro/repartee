#!/bin/bash
set -u
cd "$(dirname "$0")/.."
START=${START:-21753}
PROMPT=${PROMPT:-64}
FRAMES=${FRAMES:-192}
CKPT=${CKPT:-checkpoints/p1_text29h/last.pt}
GRID=${GRID:-data/vani_p0_45/conv_d65ed16a6092.grid.npy}
REF=${REF:-data/vani_p1_gate/reference.txt}
OUT_ROOT=${OUT_ROOT:-data/vani_p1_gate}
mkdir -p "$OUT_ROOT"
export START PROMPT FRAMES REF GRID OUT_ROOT
for MODE in textforced gt0_1 gt2_7 gt_all; do
  case "$MODE" in
    textforced) AUDIO=free ;;
    gt0_1) AUDIO=gt0_1 ;;
    gt2_7) AUDIO=gt2_7 ;;
    gt_all) AUDIO=gt_all ;;
  esac
  OUT=$OUT_ROOT/$MODE
  .venv/bin/python -u -m vani.generate_p1 \
    --checkpoint "$CKPT" --grid "$GRID" \
    --start-frame "$START" --prompt-frames "$PROMPT" --frames "$FRAMES" --window 192 \
    --text-mode target --audio-mode "$AUDIO" --reference "$REF" --out "$OUT" \
    2>&1 | tee "$OUT.log"
  .venv/bin/python -u -m vani.asr_p1 "$OUT/generated.wav" \
    --target "$OUT/target_text.txt" --language English \
    --out "$OUT/asr_en.json" 2>&1 | tee "$OUT/asr_en.log"
done
.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
import numpy as np
import soundfile as sf
from vani.asr_p1 import wer

start = int(os.environ["START"]) + int(os.environ["PROMPT"])
frames = int(os.environ["FRAMES"])
grid = np.load(os.environ["GRID"], mmap_mode="r")
truth = np.asarray(grid[start : start + frames, 1:9], dtype=np.int64)
target = open(os.environ["REF"]).read().strip()
root = Path(os.environ["OUT_ROOT"])
grid_path = Path(os.environ["GRID"])
original = grid_path.with_name(grid_path.name.replace(".grid.npy", ".original.wav"))
x, sr = sf.read(original, dtype="float32")
seg = x[int(start / 12.5 * sr) : int((start + frames) / 12.5 * sr)]
sf.write(root / "original.wav", seg, sr)
seg_n = seg / (float(np.abs(seg).max()) + 1e-9) * 0.95
sf.write(root / "original_normalized.wav", seg_n, sr)
for mode in ("textforced", "gt0_1", "gt2_7", "gt_all"):
    out = os.environ["OUT_ROOT"] + "/" + mode
    codes = np.load(f"{out}/codes.npy").astype(np.int64)
    asr = json.load(open(f"{out}/asr_en.json"))
    record = {
        "mode": mode,
        "checked_frames": int(min(len(codes), len(truth))),
        "code_accuracy": [
            round(float(np.mean(codes[: len(truth), cb] == truth[:, cb])), 4)
            for cb in range(8)
        ],
        "wer": round(wer(target, asr["transcript"]), 4),
        "transcript": asr["transcript"],
    }
    open(f"{out}/metrics.json", "w").write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
PY
