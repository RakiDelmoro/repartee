# Goal and roadmap

This file is the centre of the project: the requirements we build against, what
exists today, and the two future stages we must not foreclose. Everything else in
`docs/` explains a decision inside one stage.

## Stage 1 — initial project (current)

Requirements, as stated:

1. Low power listen for human speech.
2. When human speech detected, record it.
3. When human speech stops for some time (e.g., 1 minute) stop the recording.
4. Send recording off to bigger device for processing.
5. Bigger device runs STT, and then summarization, and adds the conversation and
   data to a knowledgebase.

### Where each requirement lives

| # | Requirement | Implementation | State |
|---|---|---|---|
| 1 | low-power listen | `app/listener.py`: pure-energy VAD, 30 ms frames, no model, noise floor tracked adaptively | built, tested |
| 2 | record on speech | speech for 90 ms opens a recording with 1 s pre-roll; sub-1 s segments discarded | built, tested |
| 3 | stop after silence | `--stop-silence` default 60 s; 1800 s hard cap | built, tested |
| 4 | send off to bigger device | atomic write into the spool as `YYYYMMDD-HHMMSS-device.wav`; `POST /ingest` or rsync into `spool/` | built, tested |
| 5 | STT → summarization → knowledge base | one `Lfm3` instance (`app/lfm3.py`) transcribes, summarizes, and names topics; `app/pipeline.py: summarize_conversation` stores them on the KB session row | built, tested end-to-end |

### Acceptance criteria for stage 1

- Two recordings produced in the same second never collide, and the filename always
  parses to a device and start time.
- The first word of a recording is not clipped (pre-roll) and the last word is not
  clipped (the recording only closes after the configured silence).
- A conversation row exists with the STT text after the worker drains the spool, and
  the audio file ends in `done/`, never silently lost (`failed/` on error).
- The session row carries the summary and topics; a summarization failure leaves the
  transcript intact and only loses the summary.
- Measured on the extraction fixtures: LFM3 writes faithful prose summaries, but
  structured events (decisions/actions/owners, empty when nothing happened) fail —
  fabricated actions on a no-outcome meeting, 1 owner attributed where 3 were named.
  Those events are therefore out of stage 1, and the KB stores the summary instead.
- Measured STT cost on this box: 90 s clip in ~20 s after the model is loaded,
  4.3 GB VRAM, first call pays ~140 s of load (see `asr-notes.md`).

### Out of scope for stage 1 — do not build

- realtime audio, streaming STT, duplex behaviour, barge-in
- audio output / TTS on the device
- any model finetuning; runtime uses the base LFM3-Audio-2B
- on-device summarization or inference beyond VAD
- structured decisions/actions/owners with quote grounding (needs a larger model)

## Stage 2 — vNext (after stage 1 works; possibly new hardware)

- Detect commands in realtime (requires realtime streaming STT).
- When a command is detected, forward the command and its parameters (natural
  language following the command) to the big device for processing.
- TTS turns the big-device response into speech over a speaker.

Design constraints this puts on stage 1, cheap to respect now:

- the VAD is already a frame-at-a-time state machine (`Segmenter.feed`), so a
  streaming transcriber can consume the same frames without rewriting the device;
- the spool contract (`filename` + one file per utterance) stays as the fallback
  when streaming is unavailable;
- the response path is text today (`kb`, `extract`), which is exactly the input a
  TTS stage needs; do not bury text generation inside the audio path;
- the command grammar must not live in the listener: the listener produces audio,
  the server decides meaning.

## Stage 3 — vNextNext (future direction, keep in mind, do not build around)

- realtime responses from a medium-power device (Jetson class);
- space-filling speech so the conversation feels fluid while the bigger model thinks;
- smooth handover between the small speech-to-speech model and the big model answer;
- most conversational work on the medium device; the big model is invoked only when
  the small model judges the task hard;
- good emotional awareness and empathy.

Cheap-to-respect implications:

- keep the session contract stable (a session id + transcript + summary), so a
  smaller model can later produce the same artefacts;
- keep the summary call behind one interface (`Lfm3.summarize`), so a different or
  larger model can replace it without touching the pipeline;
- keep audio in a codec both ends can read (48 kHz capable, mono, PCM or Opus), so a
  device can both listen and speak later;
- latency budget should be measured per stage now (STT, summary) so the handover
  target is grounded in numbers, not guesses.

## Running stage 1

```bash
# device side (host reference implementation)
.venv/bin/python -m app listen --device room1 --stop-silence 60

# bigger device
.venv/bin/python -m app worker          # spool -> STT -> DB -> KB
.venv/bin/python -m app serve           # ingest API + worker together

# existing transcripts (no audio): summarize and file them on a day
.venv/bin/python -m app import tests/fixtures/meeting_01.txt \
    --device meeting --started-at 2026-11-13T10:00:00
```

Status values: `summarized` (audio path, summary stored), `transcribed` (audio path,
summary unavailable), `imported` (transcript path, no audio file).

Environment: `REPARTEE_DATA` (spool, working, done, failed, dbs),
`REPARTEE_ASR_ENGINE=lfm3|qwen`, `REPARTEE_SUMMARY=0` to store transcripts without a
summary, `REPARTEE_DB` for the single SQLite file.
