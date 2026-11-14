# Model research: duplex speech + injected text

Question: is there a model that is **full duplex** *and* accepts **text mid-conversation**
that it speaks with a smooth handover? If not, which architecture gets closest, and what
would it take?

Sources are linked inline. Numbers are from the model/serving documentation as of
2026-11-14; measured-on-this-box numbers are marked.

## The four families

| family | example | duplex | unit | text into the live stream | cancel / truncate | size |
|---|---|---|---|---|---|---|
| **A. native duplex, per-frame text stream** | PersonaPlex 7B (Moshi-based) | yes, decides itself when to speak | 80 ms | text is a per-frame stream (inner monologue), generated not injected | **no**: `response.cancel` / `truncate` → `not_supported` | 7B, BF16 backbone 13.15 GB |
| **B. native duplex, thinker/talker** | MiniCPM-o 4.5 | yes | 1000 ms | `/v1/audio/speech` prefills text in the Thinker and conditions the Talker on each token; live session accepts **no instruction updates** | yes (`supports_cancel`, `supports_truncate`) | ~8B |
| **C. turn-based omni** | Qwen3-Omni-30B-A3B | no: VAD ends the turn, speech during a reply interrupts it | turn | text is the answer channel (turn based) | yes | 30B MoE, 2×H100 / 1×H200 |
| **D. streaming cascade** | ASR + LLM + TTS (SGLang cascade, Unmute-style) | listening + filler by the server; answer synthesised | streamed chunks | text is TTS input, i.e. **always injectable** | yes, via forced alignment to the last heard word | 1.7B ASR + 4–27B LLM + 0.6–1.7B TTS |
| **(E) our LFM3-Audio-2B** | interleaved ChatML speech blocks | rules + heads (measured) | 80 ms | forced text after `text_end` is in-distribution and measured verbatim | yes (160 ms cut, measured) | 2.17B, 4.3 GB fp16 |

Two findings that matter most:

1. **The primitive exists, the API doesn't.** MiniCPM-o already has the *inside* of the
   architecture we want: a Thinker that takes text and a Talker that speaks it, in one
   duplex model. What is missing is a way to hand the Talker new text *during* a live
   session — its capabilities advertise `supports_instruction_updates: false`.
2. **PersonaPlex's text stream is the closest thing to an injection channel**: it reads
   one text token and 16 Mimi codes per 80 ms frame and writes the next frame. If that
   text token can be forced, the audio conditions on it at *frame* granularity — finer
   than our 80 ms text-block boundary. But its server refuses cancel/truncate, so it
   cannot be interrupted by a client, which makes it a poor fit for barge-in as served.

There is also a research line for exactly this problem: **TurnGuide** (Interspeech 2026)
interleaves text and speech *at turn level* in full-duplex SLMs, explicitly to keep
acoustic flow while gaining LLM-level semantics. That is the training recipe for making
mid-conversation text injection in-distribution, as opposed to an inference-time hack.

## What "smooth handover" costs per family

- **A/B (native duplex)**: the seam is inside one model, between its own generated audio
  and the injected text. Voice identity is free. The risk is prosody: the model planned a
  continuation, we change the words. Fixes: force at a frame/unit boundary; or train
  turn-level interleaving (TurnGuide); or prefill and let the model re-plan (MiniCPM-o's
  Talker does this per token).
- **C (turn-based)**: the seam is a turn boundary enforced by VAD — natural, but the model
  is deaf while speaking, so it cannot be "duplex" in the user's sense.
- **D (cascade)**: the seam is an utterance boundary between two services. Voice identity
  must be engineered (clone the same voice in the TTS) and the interrupt must be cut at
  the last heard word (forced alignment). This is the industry-proven path and the only
  one that fits small hardware today.
- **E (LFM3)**: the seam is a text-block boundary; measured verbatim, but the model's
  content is generic so it can only be the *voice*, not the reasoner.

## Feasibility on this box (8 GB)

Measured: LFM3-Audio-2B fits (4.3 GB) and is the only one of the above that runs here.
PersonaPlex (13.15 GB backbone), MiniCPM-o (~8B), Nemotron VoiceChat (11B) and
Qwen3-Omni (30B) do not. A cascade of 1.7B ASR + 4B LLM + 0.6B TTS is borderline
(~6.5–7 GB with overhead) and would be the only way to demonstrate family D locally.

## Proposed research programme

| # | experiment | hardware | question it settles |
|---|---|---|---|
| R1 | LFM3 seam mechanics: forced text at a text-block boundary vs mid-block, with/without a cue phrase; measure gap, ASR recall, truncation, duplicate words, F0 step | this box | is a boundary handover acceptable at all? (cheap proxy for every family) |
| R2 | PersonaPlex text forcing at frame granularity: patch its LM to feed a fixed text token stream and measure whether the audio follows the forced words and how natural the seam is | rented ≥24 GB (L4/A10) | can family A do mid-stream injection without training? |
| R3 | MiniCPM-o: does a live session accept a text turn or a Talker prefill mid-call (even undocumented)? If not, can the TTS endpoint be streamed into the ongoing session? | rented ≥24 GB | is family B's thinker/talker usable as an injection channel? |
| R4 | cascade baseline: streaming ASR + small LLM + streaming TTS with cloned voice, cut answer at last heard word (forced alignment) | this box or one rented GPU | is family D good enough that the exotic path is unnecessary? |
| R5 | (only if R1–R3 fail the seam gates) train turn-level text-speech interleaving on LFM3 following TurnGuide's recipe, using the 200 Open Yap conversations | rented GPU, days | can the seam be learned rather than engineered? |

Gates: R1 passes if gap ≤ 200 ms, ASR recall ≥ 0.95, no duplicate word at the seam,
F0 step across the seam ≤ the within-sentence median. R2/R3 pass if the forced text is
spoken with measurable word recall ≥ 0.95 and the same seam metrics. R4 passes if the
cascade matches R1's numbers with a cloned voice.

## Decision this research is meant to support

1. **If R1 passes**: the product design is boundary handover on a model we already run;
   no training, no bigger hardware.
2. **If R2 or R3 passes**: rent the hardware for the duplex model and use it as the actor;
   the big model still supplies text, and the seam is inside the actor.
3. **If R4 passes and R2/R3 fail**: the cascade is the architecture; the duplex model is a
   research bet, not a product path.
4. **If all fail**: the seam is a training problem (R5), which is a weeks-long project with
   data we already have (200 conversations) but no guarantee.

Nothing here commits to LFM3; it is simply the only member of the family that runs on
this machine, which makes it the cheapest place to learn what the seam numbers look like.
