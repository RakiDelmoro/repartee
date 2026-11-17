# Repartee

Local speech software for room recording and full-duplex speech research.

## Tracks

| track | code | state |
|---|---|---|
| Room-recorder ingest | `app/` | ingest, ASR, summary, knowledge-base UI built and tested; device firmware not started |
| Vani-M speech model | `vani/` | 730M local Qwen3 + depth model; P0 codec check passed; P1 alignment in progress |

## Vani-M

Vani-M is the local, trainable, full-duplex mouth:

- Qwen3-0.6B temporal backbone
- 12-layer depth transformer
- Mimi codec, frozen, 12.5 Hz, 8 codebooks
- 17 streams per 80 ms frame: agent text, 8 agent audio codes, 8 user audio codes
- P1 trains the agent mouth only; P2 adds duplex; P3 adds text injection

Current status: **P1 alignment**. The first 8.6 h run learned all 8 codebooks in
teacher-forced evaluation but did not produce intelligible audio. The dataset is
now scaled to 45 Open Yap conversations / ~29.6 h, and a 5,000-step all-codebook
run is training.

Full plan and gates: [`docs/vani-m-plan.md`](docs/vani-m-plan.md).

## Training dashboard

```bash
.venv/bin/uvicorn vani.dashboard:app --host 0.0.0.0 --port 8765
```

Open <http://127.0.0.1:8765>. It shows loss curves, per-codebook CE, progress,
GPU/CPU/RAM, dataset tokenization, checkpoints, and the live log.

Training log:

```text
checkpoints/p1_28h/train.log
```

## Room recorder

The original ingest path is in `app/`; requirements and architecture are in
[`docs/goal.md`](docs/goal.md) and [`docs/architecture.md`](docs/architecture.md).
It records speech on a low-power host, spools WAVs, transcribes, summarizes, and
stores conversations.

## Docs

Start at [`docs/README.md`](docs/README.md).

## Development

```bash
.venv/bin/pip install -e .
.venv/bin/pytest
```

Dataset, model weights, checkpoints, and generated audio are not tracked; see
`.gitignore`.
