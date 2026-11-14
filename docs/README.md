# Docs

Design notes for the room-recorder project. The device hardware has not arrived,
but the listener software and the whole server pipeline are built and tested; see
[goal.md](goal.md) for what is real and what is planned.

| Document | Covers |
|---|---|
| [goal.md](goal.md) | **the centre**: stage-1 requirements, what exists, vNext and vNextNext constraints |
| [model-research.md](model-research.md) | duplex models, text-injection capability, feasibility on 8 GB, research programme R1–R5 |
| [seam-results.md](seam-results.md) | first seam measurement: gap, recall, pitch step, and what it implies |
| [actor-plan.md](actor-plan.md) | stage-3 design: duplex actor in front, big model reasoning behind, state machine, experiments, gates |
| [architecture.md](architecture.md) | system overview, data contracts, server pipeline, knowledge-base stage |
| [knowledge-base.md](knowledge-base.md) | event schema, extraction contract, model sizing, milestones |
| [device.md](device.md) | room-mic hardware, firmware plan, segmentation rule, buffering |
| [asr-notes.md](asr-notes.md) | ASR model choice, measured speed/VRAM/length limits, noise results |

## The project in one paragraph

A microphone array in a room records conversations when people are speaking and
streams them to a server. The server transcribes the audio, stores the transcript,
and the same LFM3-Audio-2B that transcribes also writes a short summary and topic
list that go into the knowledge base. The room device does no transcription; the server does no
recording.

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

## Conventions

- **Measured** means a number came from running code on this machine.
- **Planned** means it is a design decision, not yet verified.
- Timings are from a Quadro RTX 4000 (8 GB) in a WSL2 container; the workspace
  filesystem is a Windows drive mount and is slower than a normal disk.
