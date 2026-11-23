# Docs

Design notes for the room-recorder project and the full-duplex voice path. The device
hardware has not arrived, but the listener software and the server pipeline are built
and tested. The active duplex design is a VAD-free ASR–LLM–TTS micro-turn cascade;
the offline replay, micro-turn data, Laya timing checkpoint, and duplex-workspace LLM
path are built, but the live microphone loop is not.

| Document | Covers |
|---|---|
| [goal.md](goal.md) | **the centre**: stage-1 requirements, what exists, vNext and vNextNext constraints |
| [architecture.md](architecture.md) | system overview, data contracts, server pipeline, knowledge-base stage |
| [`RakiDelmoro/vani`](https://github.com/RakiDelmoro/vani) | **separate repository**: duplex plan and research code — micro-turns, typed control tokens, data, training, runtime, gates |
| [knowledge-base.md](knowledge-base.md) | event schema, extraction contract, model sizing, milestones |
| [device.md](device.md) | room-mic hardware, firmware plan, segmentation rule, buffering |
| [asr-notes.md](asr-notes.md) | ASR model choice, measured speed/VRAM/length limits, noise results |

## Full-duplex voice

- Design and code: [`RakiDelmoro/vani`](https://github.com/RakiDelmoro/vani)
- Status: offline replay built; the live loop
  (`mic -> ASR -> policy -> content -> TTS -> speaker`) is not.
- Built: micro-turn dataset, Laya user-state timing checkpoint, offline replays,
  gap-scheduled backchannels, and the workspace-aware content LLM.
- Data: Open Yap two-channel conversations with word-level timing, plus the deployed
  streaming ASR's real partials.
- Current stack: Kyutai STT 1B + Laya timing on the 8 GB GPU; Qwen3-4B-Instruct-2507
  on CPU for content; ZONOS3-Mini or Piper on CPU for TTS. The Qwen3-0.6B LoRA timing
  path remains as an experiment, not the active policy.
- Runtime target: streaming ASR + timing policy + asynchronous content LLM + streaming
  TTS, with AEC and hard barge-in.
- The retired native speech-model code, checkpoints, and archives have been removed.

## The project in one paragraph

A microphone array in a room records conversations when people are speaking and
streams them to a server. The server transcribes the audio, stores the transcript,
and the same LFM3-Audio-2B that transcribes also writes a short summary and topic
list that go into the knowledge base. The room device does no transcription; the
server does no recording.

In parallel, the duplex path replaces turn-taking VAD with a VAD-free micro-turn
cascade: streaming ASR emits words every 300–600 ms, a small timing policy decides
listen/speak/backchannel/interrupt, a text LLM supplies the words, and a streaming TTS
speaks them. It keeps the text LLM's intelligence and fits small hardware; it is not a
native audio model.

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
| Full-duplex cascade design | **documented** in `RakiDelmoro/vani` |
| Micro-turn dataset | **built** in `RakiDelmoro/vani` |
| Timing policy | **Laya user-state checkpoint trained** in `RakiDelmoro/vani`; Qwen3-0.6B LoRA path remains |
| Offline duplex replay | **built** in `RakiDelmoro/vani` |
| Live loop (streaming ASR/TTS, AEC, barge-in) | not built |

## Conventions

- **Measured** means a number came from running code on this machine.
- **Planned** means it is a design decision, not yet verified.
- Timings are from a Quadro RTX 4000 (8 GB) in a WSL2 container; the workspace
  filesystem is a Windows drive mount and is slower than a normal disk.
