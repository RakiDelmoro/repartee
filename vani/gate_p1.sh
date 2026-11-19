#!/bin/bash
set -u
cd "$(dirname "$0")/.."
CKPT=${CKPT:-checkpoints/p1_text29h/last.pt}
GRID=${GRID:-data/vani_p0_45/conv_d65ed16a6092.grid.npy}
export CKPT GRID
.venv/bin/python -u -m vani.eval_p1 \
  --checkpoint "$CKPT" --grid "$GRID" --window 192 --stride 192
bash vani/dense_eval.sh
