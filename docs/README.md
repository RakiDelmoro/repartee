# Docs

Design notes for the room-recorder project. Written while the hardware is still
in transit, so the device sections are **planned**, not built.

| Document | Covers |
|---|---|
| [architecture.md](architecture.md) | system overview, data contracts, server pipeline, knowledge-base stage |
| [knowledge-base.md](knowledge-base.md) | event schema, extraction contract, model sizing, milestones |
| [device.md](device.md) | room-mic hardware, firmware plan, segmentation rule, buffering |
| [asr-notes.md](asr-notes.md) | ASR model choice, measured speed/VRAM/length limits, noise results |

## The project in one paragraph

A microphone array in a room records conversations when people are speaking and
streams them to a server. The server transcribes the audio, stores the transcript,
and a ~9B language model turns it into an event knowledge base: decisions, action
items, owners, topics. The room device does no transcription; the server does no
recording.

## Status

| Piece | State |
|---|---|
| Server: ingest API, spool worker, ASR, transcript store | **built and tested** |
| ASR model selection (1.7B vs 0.6B) | **measured**; 0.6B is the default |
| Session grouping for 60 s chunks | not built |
| Knowledge-base extraction (9B) | not built |
| Device firmware | not started (hardware not delivered) |

## Conventions

- **Measured** means a number came from running code on this machine.
- **Planned** means it is a design decision, not yet verified.
- Timings are from a Quadro RTX 4000 (8 GB) in a WSL2 container; the workspace
  filesystem is a Windows drive mount and is slower than a normal disk.
