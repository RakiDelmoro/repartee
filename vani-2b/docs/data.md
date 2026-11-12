# Data

## Open Yap 1K

- 1,602 conversations, 1,000 hours, English.
- 48 kHz FLAC, one channel per speaker, sample-aligned.
- Word timestamps per speaker; per-track VAD/SNR/DNSMOS metadata.
- Overlap: median 8.25% of speech time; 598 conversations ≥10%.
- Licence: Data Use Agreement. Commercial training allowed; no redistribution,
  no voice clones identifiable as corpus speakers.
- Path: `data/open-yap/open-yap-1k_2026-11-05_flac_26a4d80b/`.
- Index: `data/open-yap/index.json`.

## Pilot

- 20 high-overlap conversations, 7.2 hours.
- 80 ms labels: speak, listen, overlap, backchannel, interrupt.
- Path: `data/open-yap/pilot/`.

## Re-voiced agent

- Agent turns replaced with the product TTS voice, time-aligned to original turns.
- User channel left human.
- 11 conversations targeted, ~43 minutes of conversation.
- Path: `data/open-yap/pilot_tts/`.
- Status: first 5 done; 6 in progress.

## Mouth corpus

- 722 clauses, spoken forms (names, numbers, dates).
- 521 train / 201 held out.
- Teacher: LFM3 TTS, female American English.
- Paths: `data/probe/mouth/corpus.jsonl`, `wav/`, `codes/`.

## Continuity corpus

- 120 lead+answer pairs rendered as one utterance; 6 discourse leads.
- 80 train / 40 held out.
- Paths: `data/probe/mouth/continuity/`.

## Licence summary

| Asset | Use |
|---|---|
| LFM3-Audio-2B | LFM Open License v1.0; commercial under $10M revenue |
| Open Yap | DUA; commercial training, no redistribution/clones |
| Qwen3-ASR | judge only; not shipped in runtime |
