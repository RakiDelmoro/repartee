#!/bin/bash
set -u
cd "$(dirname "$0")/.."
START=21753
PROMPT=64
FRAMES=192
CKPT=checkpoints/p1_28h/last.pt
GRID=data/vani_p0_45/conv_d65ed16a6092.grid.npy
for MODE in textforced gt0_1 gt2_7; do
  case "$MODE" in
    textforced) AUDIO=free ;;
    gt0_1) AUDIO=gt0_1 ;;
    gt2_7) AUDIO=gt2_7 ;;
  esac
  OUT=data/vani_p1_dense/$MODE
  .venv/bin/python -u -m vani.generate_p1 \
    --checkpoint "$CKPT" --grid "$GRID" \
    --start-frame $START --prompt-frames $PROMPT --frames $FRAMES --window 192 \
    --text-mode target --audio-mode "$AUDIO" --out "$OUT" \
    2>&1 | tee "$OUT.log"
  .venv/bin/python -u -m vani.asr_p1 "$OUT/generated.wav" \
    --target "$OUT/target_text.txt" --language English \
    --out "$OUT/asr_en.json" 2>&1 | tee "$OUT/asr_en.log"
done
