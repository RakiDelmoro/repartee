# Repartee — one command per stage.
#
# Override on the command line, e.g.:
#   make data HOURS=1000 SOURCES=all
#   make train GPUS=8
#   make serve INPUT=mic.wav OUTPUT=reply.wav
#   make gate4 GATE4_LIMIT=200

PY ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest

MANIFESTS ?= data/manifests/repartee-mix.jsonl
RAW ?= data/raw
TOKENS ?= data/tokens
CHECKPOINTS ?= checkpoints
ARTIFACTS ?= artifacts
MODEL_DIR ?= models/LFM3-Omni-2B
MODEL ?= $(if $(wildcard $(MODEL_DIR)/config.json),$(MODEL_DIR),LiquidAI/LFM3-Omni-2B)
SOURCES ?= open_yap
SAMPLE_ONLY ?= --sample-only
HOURS ?= 8000
JOB ?= local
NPROC ?= 1
TORCHRUN ?= .venv/bin/torchrun
GATE4_LIMIT ?= 200
APPROVED_LOOKAHEAD_MS ?= 15
GRAD_ACCUM ?= 1
SEQ_FRAMES ?= 256
STEPS ?= 600
WRITER_MODEL ?=
FDB_DATA ?= data/full-duplex-bench
FDB_RESULTS ?= artifacts/fdb/timing_preferences.json

.PHONY: all gate0 gate0-check data real synth tokenize train stage1 stage1-proto stage1-accept eval-base eval-stage1 eval-stage2 gate4 stage2 stage2-proto fdb dpo pairs dashboard serve test licenses report clean-tokens

all: gate0 data train

gate0:
	$(PY) scripts/gate0.py --all --model-id LiquidAI/LFM3-Omni-2B --model-dir $(MODEL_DIR) --approved-lookahead-ms $(APPROVED_LOOKAHEAD_MS) --out $(ARTIFACTS)/gate0

data: real synth tokenize

real:
	$(PY) -m src.build_real --source $(SOURCES) $(SAMPLE_ONLY) --out data/manifests --raw $(RAW)

synth:
	$(PY) -m src.build_synth --out data/manifests --raw $(RAW) --hours $(HOURS) $(if $(WRITER_MODEL),--writer-model $(WRITER_MODEL),)

tokenize:
	$(PY) -m src.tokenize --manifests $(MANIFESTS) --out $(TOKENS) --model $(MODEL)

gate0-check:
	$(PY) scripts/require_gate.py --artifact $(ARTIFACTS)/gate0/gate0.json --name "Gate 0"

train: gate0-check stage1 gate4 stage2 fdb pairs dpo

stage1:
	$(TORCHRUN) --nproc_per_node=$(NPROC) -m src.train_stage1 --tokens $(TOKENS) --model $(MODEL) --out $(CHECKPOINTS)/stage1 --job $(JOB)

# Both suites; writes artifacts/gates/gate4.json for stage 2.
gate4:
	$(PY) scripts/gate4.py --model $(MODEL) --candidate $(CHECKPOINTS)/stage1 --limit $(GATE4_LIMIT) --out $(ARTIFACTS)/gates/gate4.json

stage2:
	$(TORCHRUN) --nproc_per_node=$(NPROC) -m src.train_stage2 --tokens $(TOKENS) --model $(CHECKPOINTS)/stage1 --out $(CHECKPOINTS)/stage2 --gate4 $(ARTIFACTS)/gates/gate4.json --job $(JOB)

# Single-8GB-GPU prototype targets (512 frames, LoRA; Gate 4 explicitly bypassed
# for stage2).  These are not the blueprint recipe.
stage1-accept:
	$(PY) scripts/require_gate.py --artifact $(ARTIFACTS)/eval_stage1_final.json --name "Stage 1 acceptance"

eval-base:
	test -f $(ARTIFACTS)/eval_base.json || $(PY) scripts/evaluate_duplex.py --model $(CHECKPOINTS)/repartee-2b-base --tokens data/tokens_proto --split test --windows 16 --seq-frames $(SEQ_FRAMES) --out $(ARTIFACTS)/eval_base.json --device cuda --precision bf16 --codec-device cuda --local-files-only

eval-stage1: eval-base
	$(PY) scripts/evaluate_duplex.py --model $(CHECKPOINTS)/stage1_proto --tokens data/tokens_proto --split test --windows 16 --seq-frames $(SEQ_FRAMES) --baseline $(ARTIFACTS)/eval_base.json --out $(ARTIFACTS)/eval_stage1_final.json --device cuda --precision bf16 --codec-device cuda --local-files-only

eval-stage2: stage1-accept
	$(PY) scripts/evaluate_duplex.py --model $(CHECKPOINTS)/stage2_proto --tokens data/tokens_proto --split test --windows 16 --seq-frames $(SEQ_FRAMES) --baseline $(ARTIFACTS)/eval_stage1_final.json --min-control-macro-f1 0.50 --out $(ARTIFACTS)/eval_stage2_final.json --device cuda --precision bf16 --codec-device cuda --local-files-only

stage1-proto:
	$(TORCHRUN) --nproc_per_node=1 -m src.train_stage1 --tokens data/tokens_proto --model $(CHECKPOINTS)/repartee-2b-base --out $(CHECKPOINTS)/stage1_proto --seq-frames $(SEQ_FRAMES) --batch-size 1 --grad-accum $(GRAD_ACCUM) --steps $(STEPS) --warmup 20 --save-every 25 --log-every 1 --resume auto --device cuda --precision bf16 --codec-device cuda --local-files-only
	$(MAKE) eval-stage1

stage2-proto: stage1-accept
	$(TORCHRUN) --nproc_per_node=1 -m src.train_stage2 --tokens data/tokens_proto --model $(CHECKPOINTS)/stage1_proto --out $(CHECKPOINTS)/stage2_proto --prototype --lora --lora-r 32 --lora-alpha 64 --lr 1e-4 --connector-lr 1e-3 --seq-frames $(SEQ_FRAMES) --batch-size 1 --grad-accum $(GRAD_ACCUM) --steps-per-epoch $(STEPS) --epochs 1 --warmup 20 --save-every 25 --log-every 1 --resume auto --device cuda --precision bf16 --codec-device cuda --local-files-only
	$(MAKE) eval-stage2

# Gate 5 evidence; FDB v1/v1.5 data must be downloaded first (see eval/fdb.py).
fdb:
	$(PY) -m eval.fdb --data $(FDB_DATA) --engine repartee --model $(CHECKPOINTS)/stage2 --out $(ARTIFACTS)/fdb.json

# Requires the FDB timing-preference artifact for the early/late variants.
pairs:
	$(PY) scripts/build_dpo_pairs.py --tokens $(TOKENS) --fdb-results $(FDB_RESULTS) --out data/pairs/dpo_timing.jsonl

dpo:
	$(TORCHRUN) --nproc_per_node=$(NPROC) -m src.train_dpo --tokens $(TOKENS) --model $(CHECKPOINTS)/stage2 --out $(CHECKPOINTS)/stage3 --pairs data/pairs/dpo_timing.jsonl --job $(JOB)

dashboard:
	$(PY) -m tools.dashboard --runs $(CHECKPOINTS) --host $${DASHBOARD_HOST:-127.0.0.1} --port $${DASHBOARD_PORT:-8080}

serve:
	$(PY) -m serve --model $(MODEL) $(if $(INPUT),--input $(INPUT),) $(if $(OUTPUT),--output $(OUTPUT),) --metrics $(ARTIFACTS)/serve.json

licenses:
	$(PY) -m src.license_audit --manifests $(MANIFESTS) --strict

report:
	@cat $(ARTIFACTS)/gate0/report_section.md

test:
	$(PYTEST) -q

clean-tokens:
	rm -rf $(TOKENS)
