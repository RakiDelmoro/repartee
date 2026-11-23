# Repartee

Local speech software for room recording and full-duplex speech research.

## Tracks

| track | code | state |
|---|---|---|
| Room-recorder ingest | `app/` | ingest, ASR, summary, knowledge-base UI built and tested; device firmware not started |
| Full-duplex voice | [`RakiDelmoro/vani`](https://github.com/RakiDelmoro/vani) | separate repository: offline cascade replay, Laya timing, workspace-aware LLM; live loop not built |

## Full-duplex voice

The active duplex design is a VAD-free ASR–LLM–TTS cascade, not a new speech model.
A streaming recognizer emits words every 300–600 ms; a small timing policy decides
`listen`, `speak`, backchannel, or interrupt; an asynchronous text LLM supplies the
words; and a streaming TTS speaks them. Typed control tokens carry the conversation
state, and real Open Yap conversations supply the timing data.

Current stack: Kyutai STT 1B + Laya timing policy on the 8 GB GPU, with
Qwen3-4B-Instruct-2507 on CPU for content and ZONOS3-Mini or Piper on CPU for TTS.
The Qwen3-0.6B timing LoRA remains an experiment. All duplex research work lives in
the separate repository [`RakiDelmoro/vani`](https://github.com/RakiDelmoro/vani):
code, docs, tests, data, checkpoints, samples, models, and vendor tools.

Status: **offline replay built, live loop not built.** The Vani repository's
`replay.py` runs the cascade over Open Yap audio with a rule policy; `replay_laya.py`
runs the trained user-state Laya policy, schedules backchannels at real word gaps, and
gives the content LLM a live duplex-workspace snapshot. The first real-time deliverable
is still the standalone loop (`mic -> ASR -> policy -> content -> TTS -> speaker`), not
production integration. The retired native speech-model code, checkpoints, and
archives have been removed from this repository.

Design, data, gates, build order, and code:
[`RakiDelmoro/vani`](https://github.com/RakiDelmoro/vani).

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
.venv/bin/python -m pytest
```

Dataset, model weights, and generated audio are not tracked; see `.gitignore`.
