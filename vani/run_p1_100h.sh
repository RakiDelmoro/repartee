#!/bin/bash
set -u
cd "$(dirname "$0")/.."
mkdir -p data/vani_p0_100h checkpoints/p1_100h
.venv/bin/python -u -m vani.build_100h \
  --target-hours 100 \
  --out data/vani_p0_100h \
  --existing data/vani_p0_45 \
  2>&1 | tee data/vani_p0_100h/build.log
.venv/bin/python -u -m vani.train_p1 \
  --steps 5000 --save-every 200 --keep-checkpoints 5 --accum 2 \
  --active-codebooks 2 --text-cond sparse \
  --grids data/vani_p0_100h --min-voiced 0.7 --voice-weight 2.0 \
  --out checkpoints/p1_100h --holdout conv_d65ed16a6092 \
  --resume checkpoints/p1_textslot2k/last.pt \
  2>&1 | tee checkpoints/p1_100h/train.log
