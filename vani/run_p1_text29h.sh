#!/bin/bash
set -u
cd "$(dirname "$0")/.."
mkdir -p checkpoints/p1_text29h
.venv/bin/python -u -m vani.train_p1 \
  --steps 5000 --window 192 --stride 128 --accum 2 --active-codebooks 8 \
  --grids data/vani_p0_45 \
  --min-voiced 0.25 --voice-weight 2.0 \
  --out checkpoints/p1_text29h --holdout conv_d65ed16a6092 \
  --resume checkpoints/p1_28h/last.pt 2>&1 | tee checkpoints/p1_text29h/train.log
