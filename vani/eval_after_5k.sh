#!/bin/bash
set -u
cd "$(dirname "$0")/.."
# Wait for the free-run 5k WAV (also implies last.pt exists).
while [ ! -f data/vani_p1_28h_gen/generated.wav ]; do sleep 20; done

.venv/bin/python -u -m vani.eval_p1 \
  --checkpoint checkpoints/p1_28h/last.pt \
  --window 192 --stride 192 \
  --audio-out data/vani_p1_28h_eval/teacher_forced.wav \
  2>&1 | tee data/vani_p1_28h_eval.log

.venv/bin/python -u -m vani.generate_p1 \
  --checkpoint checkpoints/p1_28h/last.pt \
  --grid data/vani_p0_45/conv_d65ed16a6092.grid.npy \
  --prompt-frames 128 --frames 192 --window 192 \
  --text-mode target --out data/vani_p1_28h_textforced \
  2>&1 | tee data/vani_p1_28h_textforced.log

.venv/bin/python -u -m vani.asr_p1 \
  data/vani_p1_28h_gen/generated.wav \
  --target data/vani_p1_28h_gen/target_text.txt \
  --out data/vani_p1_28h_gen/asr.json \
  2>&1 | tee data/vani_p1_28h_gen/asr.log

.venv/bin/python -u -m vani.asr_p1 \
  data/vani_p1_28h_textforced/generated.wav \
  --target data/vani_p1_28h_textforced/target_text.txt \
  --out data/vani_p1_28h_textforced/asr.json \
  2>&1 | tee data/vani_p1_28h_textforced/asr.log
