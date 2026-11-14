# ASR notes

All numbers **measured** on a Quadro RTX 4000 (8 GB), fp16, unless marked planned.

## Model choice

| Model | Params | Weights | License | Note |
|---|---|---|---|---|
| Qwen3-ASR-1.7B-hf | 1.72 B | 3.44 GB | Apache-2.0 | most accurate on rare terms |
| Qwen3-ASR-0.6B-hf | 782 M | 1.56 GB | Apache-2.0 | reference accuracy; `REPARTEE_ASR_ENGINE=qwen` |
| LFM3-Audio-2B | 2.17 B | 4.3 GB | — | **runtime default**, one family with the device model |
| Granite Speech 5.0 470M | 473 M | 0.95 GB | Apache-2.0 | best meeting WER per MB, different API, 2-min cap |
| Parakeet TDT 0.6B v3 | 600 M | ~1.2 GB | CC-BY-4.0 | attribution required |
| Whisper large-v3-turbo | 809 M | ~1.6 GB | Apache-2.0 | worst meeting accuracy of the group |

Both Qwen models use `AutoProcessor` + `AutoModelForMultimodalLM` and the same
`apply_transcription_request` call, so switching is `REPARTEE_ASR_MODEL=<path>`.

## Measured: LFM3-Audio-2B as the transcriber

`app/asr.py: Lfm3Transcriber` sends the clip as one audio turn with a
"transcribe verbatim" system prompt and generates with `generation_mode="text"`.

| Audio | Wall time (incl. load) | Output |
|---|---|---|
| 10 s interview clip | 155 s, of which ~140 s is model load | 216 characters |
| 90 s interview clip | generation finished inside 190 s total | run failed later at decode, since fixed |

Quality, same clip: LFM3 writes *"playing around with the fortram … brilliant
trees … organ sound"* where the Qwen reference hears the music terms correctly
("four track"). Qwen measured ~3 word errors per 90 s; expect visibly more from
LFM3 on rare terms, at ~3x the VRAM (4.3 GB vs 1.58 GB for 0.6B).

Consequences:

1. Keep clips at 60 s: the 2B forward pass is the cost driver, and the listener's
   60 s silence rule already bounds them.
2. Treat LFM3 output as a draft: the knowledge-base extractor requires every event
   to quote the transcript, so a garbled term becomes a missing event rather than a
   wrong one.
3. Switch with `REPARTEE_ASR_ENGINE=qwen` when accuracy on names/terms matters more
   than staying in one model family.

## Measured: 1.7B vs 0.6B

Same two 90 s samples (clean two-speaker interview, and the same plus vacuum
noise at 10 dB SNR):

| | 1.7B | 0.6B |
|---|---|---|
| Peak VRAM | 4.08 GB | **1.58 GB** |
| Load time | 61.1 s | **21.9 s** |
| Clean, 90 s | 55.5 s | 28.9 s |
| Vacuum, 90 s | 33.3 s | 29.4 s |
| Characters (clean / vacuum) | 1430 / 1414 | 1442 / 1437 |

First inference includes CUDA warmup; steady state is ~33 s (1.7B) and ~29 s
(0.6B) per 90 s, roughly 2.7x faster than realtime.

Quality: word agreement 89 % clean, 87 % vacuum, but almost all of the difference
is contraction expansion (`it'd` -> `it would`) and punctuation. Real errors are
about **3 words per 90 s**, concentrated on unusual terms ("four track" ->
"trunck"/"drum"). Neither model truncates or drops spans.

## Measured: how long can it transcribe

| Audio | Prompt tokens | Compute | Peak VRAM |
|---|---|---|---|
| 1.5 min | 1,185 | 38.9 s | 2.27 GB |
| 3.0 min | 2,355 | 58.2 s | 2.97 GB |
| 6.0 min | 4,695 | 294.3 s | 5.64 GB |
| 12.6 min | — | timed out at 900 s | — |

The audio token rate is constant at **13 tokens per second of audio**. With a
65,536-token context the model could accept ~5,040 s (**84 minutes**) of audio.

But the audio encoder is quadratic: 3 -> 6 minutes quintuples the compute and
almost doubles VRAM. On 8 GB the practical wall is **~6 minutes**, and 60 s is the
sweet spot (~20 s of compute, ~1.7 GB).

**Consequence:** always transcribe in 60 s segments, regardless of meeting length.
This is a memory and compute constraint, not an accuracy one.

## Measured: noise behaviour

Two-speaker interview, 90 s, with added noise:

| Condition | Result |
|---|---|
| clean | 1430 chars, full transcript |
| + vacuum cleaner, 10 dB SNR | 1414 chars, 92.6 % word agreement |
| + steady rain, 10 dB SNR | 1404 chars, coherent |
| + restaurant ambience (competing speech), 20 dB SNR | **80 chars, early EOS** |
| + restaurant ambience, 5 dB SNR | **0 chars** |

The failure mode is **early end-of-sequence**, not gradual degradation: the model
returns `language None<asr_text>` or a short transcript and stops.

`language None` / empty text is the signature of this failure. Treat it as an
error, not an empty conversation.

**Consequence:** steady machine noise is tolerated; competing speech is not. That
is what the XVF3800 beamforming and noise suppression are for.

## Rules derived from these measurements

1. Transcribe in 60 s segments.
2. Cut segments at a pause, not on a fixed grid.
3. Treat empty or suspiciously short transcripts as failures; retry with the audio
   split further.
4. Scale `max_new_tokens` with duration: 1,024 per 60 s is ample.
5. Pick 0.6B if VRAM must be shared with the LLM; pick 1.7B if accuracy on rare
   terms matters more.
