#!/bin/bash
set -u
cd "$(dirname "$0")/.."
# Current run must finish at 5,000 first.
while [ ! -f checkpoints/p1_28h/last.pt ]; do sleep 20; done
mkdir -p checkpoints/p1_5ep
# 25,910 total - 5,000 current = 20,910 extra optimizer steps.
# 25,910 * 2 windows / 10,364 windows-per-epoch = 5.0 epochs.
.venv/bin/python -u -m vani.train_p1 \
  --steps 20910 --window 192 --accum 2 --active-codebooks 8 \
  --save-every 1000 --out checkpoints/p1_5ep \
  --holdout conv_d65ed16a6092 --resume checkpoints/p1_28h/last.pt \
  2>&1 | tee checkpoints/p1_5ep/train.log
# Inference only; no training/architecture/loss changes.
.venv/bin/python -u -m vani.generate_p1 \
  --checkpoint checkpoints/p1_5ep/last.pt \
  --grid data/vani_p0_45/conv_d65ed16a6092.grid.npy \
  --prompt-frames 128 --frames 192 --window 192 \
  --out data/vani_p1_5ep_gen 2>&1 | tee data/vani_p1_5ep_gen.log
