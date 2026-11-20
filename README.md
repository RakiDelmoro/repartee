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
(cb0–1) is the blocker. The deployable sparse text path reached cb0 **10.59%**;
held-word was a probe (11.82%) and is not used. The 100 h Open Yap run (149
conversations, dense windows) moved cb0 to **14.32%** but cb1 fell to **2.11%**.
A fresh unseen conversation confirms cb0 **12.24%**, cb1 **2.37%**, and
`gt0_1` WER **0.890** (old holdout 0.353) while `gt_all` is 0.110. The codec is
fine; the frozen fine codebooks do not generalize, and cb1 is broken on both
domains. A 5,000-step all-8 fine-tune **regressed both halves** (old cb0/cb1
13.08/1.82, new 10.82/2.08; new `gt0_1` 0.986), and the timeboxed cb1-weighted
and fine-only phases also failed (cb1 stayed ~2%; fine-only `gt0_1` 0.849 new,
coarse collapsed). The **0.73 B / 100 h Open Yap P1 recipe is exhausted**;
`p1_100h` remains the best base but fails the gate. Next: dense read/aligned P1
speech, a larger/audio-pretrained backbone, or the product cascade. New-domain
WAVs are in `data/vani_p1_newhold/`; old gate WAVs in `data/vani_p1_gate/`.

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
