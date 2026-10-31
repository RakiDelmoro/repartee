# Repartee

A 2B full-duplex speech-to-speech conversation model. Listens and speaks on
the same 80 ms tick; no VAD, no turn detector.

This repository is the implementation of `BLUEPRINT.md`. Measured gate numbers
and deviations live in `REPORT.md`; unrun gates are marked there as
`UNVERIFIED`, never estimated.

**Status: prototype Stage 1 passed under an approved 15 ms look-ahead.**
Gate 0 records the strict 80 ms failure plus the approved 95 ms policy. A
600-step, one-epoch Stage 1 run on the 3.4 h Open Yap prototype split reached
held-out control accuracy 0.8735 and macro-F1 0.5938 on the 8 GB RTX 4000.
Stage 2 LoRA is paused at step 100; `make stage2-proto` resumes automatically from `checkpoints/stage2_proto/state/latest.pt`.

This is **not** a fully trained model: it is one epoch over 3.4 h. Repeating
that epoch would mostly overfit; the next meaningful improvement is more data.
Full 9,300 h training, FDB, pilot, and consumer gates were not run.

## Layout

```
models/LFM3-Omni-2B/  original base weights + tokenizer/processor/codec
checkpoints/          prepared/trained Repartee checkpoints
data/manifests/     conversation JSONL (one per source + repartee-mix.jsonl)
data/raw/           downloaded/normalized audio (gitignored)
data/tokens/        RPT1 shards + index.json (gitignored)
src/schema.py       12.5 Hz frame grid, control labels, manifest I/O
src/build_real.py   Open Yap / otoSpeech -> manifest
src/build_synth.py  LLM dialogue -> TTS -> timeline -> manifest
src/tokenize.py     manifest -> LiquidCodec shards
src/dataset.py      mmap dataset over shards
src/model.py        base load, 4 control tokens, connector, freeze table
src/train_*.py      stage 1 (adapter), stage 2 (duplex SFT), stage 3 (DPO)
serve/loop.py       80 ms scheduler, KV-cache duplex engine, jitter buffer
serve/aec.py        AEC3 backend + NLMS fallback
eval/               FDB, text regression, 20-scenario pilot
scripts/gate0.py    blocking checkpoint verification
```

## Model weights

The base checkpoint is stored in the workspace, not only in the Hugging Face cache:

```
models/LFM3-Omni-2B/                 4.50 GB, 27 files
  model.safetensors                  4.1 GB   sha256 76c0ec2c…19d2763
  audio_tokenizer/model.safetensors  141 MB  sha256 45984a87…655137f
  tokenizer.json, processor_config.json, config.json, chat_template.jinja

checkpoints/repartee-2b-base/        8.3 GB fp32, prepared but UNTRAINED
  model.safetensors                  four control tokens mean-initialised
  repartee.json                      control ids + metadata
```

`make` uses `models/LFM3-Omni-2B` automatically when it exists; otherwise it
falls back to the Hugging Face id.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
# CUDA build of torch for the training host:
.venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## One command each

```bash
make gate0     # blocking checkpoint checks -> artifacts/gate0
make data      # real + synthetic manifests -> 12.5 Hz token shards
make train     # gate 0 check -> stage 1 -> gate 4 -> stage 2 -> FDB -> DPO
make serve     # 80 ms live loop (requires a microphone/speaker)
```

`make data` needs the licensed corpora placed in `data/raw/` (or the public
Open Yap sample, the default) and a real TTS + dialogue model, e.g.
`make data SOURCES=all HOURS=8000 WRITER_MODEL=/models/Qwen2.5-7B-Instruct`.
`make train` refuses to start unless `artifacts/gate0/gate0.json` passed; the
current checkpoint fails Gate 0.2, so it stops there by design.

Useful smaller targets:

```bash
make dashboard   # http://127.0.0.1:8080 — live loss/LR/step-time charts
make test
make licenses
make report
```

## Monitoring and resuming

```bash
make dashboard DASHBOARD_PORT=8080
```

The dashboard reads every `checkpoints/*/metrics.jsonl`, shows loss (total,
text, control, audio), learning rate, step time, throughput, ETA, checkpoint
age, and live GPU memory/utilization. No external JavaScript or CDN is needed.

Prototype training checkpoints every 25 optimizer steps to
`<run>/state/latest.pt` and resumes automatically on the next run:

```bash
make dashboard               # watch metrics at http://127.0.0.1:8080
make stage1-proto            # 300 steps + automatic held-out evaluation
make stage2-proto            # requires stage-1 acceptance; 300 LoRA steps + evaluation
```

`stage1-proto` writes `artifacts/eval_stage1_final.json` and fails if the
checkpoint does not improve loss, reach control accuracy 0.80, or reach
macro-F1 0.40. `stage2-proto` refuses to start without that passing artifact,
then writes `artifacts/eval_stage2_final.json` with a macro-F1 target of 0.50.
You do not need to tweak anything between runs.

Prototype defaults: bf16, 512 frames, batch 1, accumulation 1, LR 1e-3
(stage 1) / LoRA 1e-4 + connector 1e-3 (stage 2), warmup 20, checkpoint and
log every 25/5 steps. fp16 is not used because it diverged on Turing.
`--resume auto` continues from `<run>/state/latest.pt` after Ctrl-C.

## Data and licensing

`make data` needs the Open Yap sample (public) and the gated otoSpeech terms
accepted. The full Open Yap 1K corpus is granted per recipient under a data
use agreement. `src/license_audit.py` lists every source and TTS with terms;
`make licenses` exits non-zero on a non-commercial or unknown source.

Base model: `LiquidAI/LFM3-Omni-2B` under the LFM Open License v1.0
(commercial use permitted below $10M annual revenue; see `REPORT.md` Gate 0).
