#!/bin/bash
set -u
cd "$(dirname "$0")/.."
mkdir -p data/libritts_r data/vani_p0_libritts checkpoints/p1_libritts
if [ ! -d data/libritts_r/LibriTTS_R/train-clean-100 ]; then
  curl -L -C - --retry 5 --retry-delay 5 \
    -o data/libritts_r/train_clean_100.tar.gz \
    https://openslr.trmal.net/resources/141/train_clean_100.tar.gz
  curl -L -o data/libritts_r/md5sum.txt \
    https://openslr.trmal.net/resources/141/md5sum.txt
  (cd data/libritts_r && grep train_clean_100 md5sum.txt | md5sum -c -)
  tar -xzf data/libritts_r/train_clean_100.tar.gz -C data/libritts_r
fi
.venv/bin/python -u -m vani.build_libritts \
  --root data/libritts_r --out data/vani_p0_libritts \
  2>&1 | tee data/vani_p0_libritts/build.log
HOLDOUT=$(.venv/bin/python -c "import json;print(json.load(open('data/vani_p0_libritts/holdout.json'))['speaker'])")
.venv/bin/python -u -m vani.train_p1 \
  --steps 5000 --save-every 200 --keep-checkpoints 5 --accum 2 \
  --active-codebooks 8 --cb-weights 1,1,1,1,1,1,1,1 \
  --text-cond sparse --grids data/vani_p0_libritts \
  --min-voiced 0.7 --voice-weight 2.0 \
  --out checkpoints/p1_libritts --holdout "$HOLDOUT" \
  --resume checkpoints/p1_100h/last.pt \
  2>&1 | tee checkpoints/p1_libritts/train.log
mapfile -t SEL < <(.venv/bin/python -c "import json; s=json.load(open('data/vani_p0_libritts/holdout.json'))['selection']; print(s['start_frame']); print(s['prompt_frames']); print(s['frames']); print(s['grid']); print(s['reference'])")
START=${SEL[0]} PROMPT=${SEL[1]} FRAMES=${SEL[2]} GRID=${SEL[3]} REF=${SEL[4]} \
  CKPT=checkpoints/p1_libritts/last.pt OUT_ROOT=data/vani_p1_libritts \
  bash vani/dense_eval.sh
.venv/bin/python -u -m vani.eval_p1 \
  --checkpoint checkpoints/p1_libritts/last.pt --grid "${SEL[3]}" \
  --window 192 --stride 192 --text-cond sparse \
  2>&1 | tee checkpoints/p1_libritts/eval.log
