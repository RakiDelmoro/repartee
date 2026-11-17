#!/bin/bash
set -u
cd "$(dirname "$0")/.."
while true; do
  n=$(ls data/vani_p0_45/conv_*.json 2>/dev/null | wc -l)
  if [ "$n" -ge 45 ]; then break; fi
  sleep 20
done
.venv/bin/python -u -m vani.train_p1 \
  --steps 5000 --window 192 --accum 2 --active-codebooks 8 \
  --out checkpoints/p1_28h --holdout conv_d65ed16a6092 \
  --resume checkpoints/p1_all/last.pt 2>&1 | tee checkpoints/p1_28h/train.log
