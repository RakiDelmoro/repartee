# Docs

Design notes for the room-recorder project and the Vani-M speech-model track. The
device hardware has not arrived, but the listener software and the server pipeline
are built and tested. Vani-M is the newer local full-duplex model: its plan, phases,
gates, and measured results are in [vani-m-plan.md](vani-m-plan.md).

| Document | Covers |
|---|---|
| [goal.md](goal.md) | **the centre**: stage-1 requirements, what exists, vNext and vNextNext constraints |
| [architecture.md](architecture.md) | system overview, data contracts, server pipeline, knowledge-base stage |
| [vani-m-plan.md](vani-m-plan.md) | **current plan**: 600M–1B Moshi-style Vani-M, data, phases, gates |
| [archive/](archive/) | pre-Vani-M Haru/LFM3 research notes, kept for history |
| [knowledge-base.md](knowledge-base.md) | event schema, extraction contract, model sizing, milestones |
| [device.md](device.md) | room-mic hardware, firmware plan, segmentation rule, buffering |
| [asr-notes.md](asr-notes.md) | ASR model choice, measured speed/VRAM/length limits, noise results |

## Vani-M local model

- Training code: `vani/`
- Plan and gates: [vani-m-plan.md](vani-m-plan.md)
- Dashboard: `uvicorn vani.dashboard:app --host 0.0.0.0 --port 8765`, then
  <http://127.0.0.1:8765>
- Training log: `checkpoints/p1_28h/train.log`
- Dataset, checkpoints, generated audio, and model weights are ignored by git.

## The project in one paragraph

A microphone array in a room records conversations when people are speaking and
streams them to a server. The server transcribes the audio, stores the transcript,
and the same LFM3-Audio-2B that transcribes also writes a short summary and topic
list that go into the knowledge base. The room device does no transcription; the
server does no recording.

In parallel, Vani-M trains a local Moshi-style speech model on Open Yap so the
same system can eventually listen and speak simultaneously, with a text-injection
control channel for a larger reasoner. The model is not part of the room-recorder
runtime yet.

## Status

| Piece | State |
|---|---|
| Server: ingest API, spool worker, ASR, summary, one-row store | **built and tested** |
| Dashboard: conversations grouped by day, summary + transcript | **built**; API verified, UI not yet exercised in a browser here |
| Device-side listener (energy VAD, 60 s silence close, spool handoff) | **built and tested** (`app/listener.py`) |
| Runtime STT | **LFM3-Audio-2B** (one family with the device model); Qwen3-ASR stays selectable with `REPARTEE_ASR_ENGINE=qwen` |
| Session grouping for 60 s chunks | not built |
| Summary + topics per conversation (LFM3) | **built and tested**; stored on the session row |
| Structured events (decisions/actions/owners) | future: needs a bigger model, measured in `knowledge-base.md` |
| Device firmware | not started (hardware not delivered); the host listener is the reference behaviour |
| Vani-M P0: Mimi grid + decode round-trip | **passed perceptually** on 45 conversations / 29.6 h |
| Vani-M P1: agent-only alignment, all codebooks | **gate open**: 29.6 h baseline WER 0.971; coarse-only and all-8 self-conditioning also failed |
| Vani-M dashboard | **built** at `vani/dashboard.py`; shows loss, codebooks, GPU/CPU, data, checkpoints, live log |
| Vani-M P2 duplex / P3 injection | blocked until P1 audio gate passes |

## Conventions

- **Measured** means a number came from running code on this machine.
- **Planned** means it is a design decision, not yet verified.
- Timings are from a Quadro RTX 4000 (8 GB) in a WSL2 container; the workspace
  filesystem is a Windows drive mount and is slower than a normal disk.
