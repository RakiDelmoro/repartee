#!/bin/bash
set -u
cd "$(dirname "$0")/.."
CKPT=${CKPT:-checkpoints/p1_coarse_sc/last.pt}
OUT=${OUT:-data/vani_p1_coarse_eval}
export OUT
while [ ! -f "$CKPT" ]; do sleep 20; done
mkdir -p "$OUT"
.venv/bin/python -u -m vani.generate_p1 \
  --checkpoint "$CKPT" \
  --grid data/vani_p0_45/conv_d65ed16a6092.grid.npy \
  --start-frame 21753 --prompt-frames 64 --frames 192 --window 192 \
  --text-mode target --audio-mode free --out "$OUT" \
  2>&1 | tee "$OUT/generate.log"
.venv/bin/python -u -m vani.asr_p1 "$OUT/generated.wav" \
  --target data/vani_p1_dense/target_dense.txt --language English \
  --out "$OUT/asr_en.json" 2>&1 | tee "$OUT/asr_en.log"
.venv/bin/python - <<'PY'
import json
import os
import numpy as np
from vani.asr_p1 import wer
out=os.environ['OUT']
start=21753+64
t=np.asarray(np.load('data/vani_p0_45/conv_d65ed16a6092.grid.npy',mmap_mode='r')[start:start+192,1:9],dtype=np.int64)
g=np.load(f'{out}/codes.npy').astype(np.int64)
target_text=open('data/vani_p1_dense/target_dense.txt').read().strip()
asr=json.load(open(f'{out}/asr_en.json'))
record={'coarse_accuracy':[round(float(np.mean(g[:,cb]==t[:,cb])),4) for cb in range(8)],
        'wer':round(wer(target_text,asr['transcript']),4), 'transcript':asr['transcript']}
open(f'{out}/metrics.json','w').write(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY
