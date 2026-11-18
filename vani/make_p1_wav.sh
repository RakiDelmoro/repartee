#!/bin/bash
set -u
cd "$(dirname "$0")/.."
while [ ! -f checkpoints/p1_28h/last.pt ]; do sleep 20; done
.venv/bin/python -u -m vani.generate_p1 \
  --checkpoint checkpoints/p1_28h/last.pt \
  --grid data/vani_p0_45/conv_d65ed16a6092.grid.npy \
  --prompt-frames 128 --frames 192 --window 192 \
  --out data/vani_p1_28h_gen 2>&1 | tee data/vani_p1_28h_gen.log
