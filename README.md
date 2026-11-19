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

Current status: **P1 alignment, gate failed.** The gate isolates the failure: with
true coarse codes the model reaches WER 0.353 (codec ceiling 0.279); with generated
coarse codes it is 0.971. The codec and fine codebooks are good; coarse content
(cb0–1) is the blocker. Token + cb0–1-first reached voiced cb0 **10.59%**; the
held-word run reached **11.82%**, with real-vs-PAD accuracy still within ~1 point.
Text timing/alignment is not the main limiter. Held-word is a **probe**, not the
product path: it needs word-duration pacing at runtime. The deployable line is the
per-frame sparse text token (cb0 10.59%). Next: dense P1 data (speech-only turns
or read/aligned speech), then 100 h, then a backbone change. Listenable WAVs are
in `data/vani_p1_gate/`.

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
