# Duplex log

Chronological record of every duplex decision, experiment, probe, and plan.
Append-only. If a decision changes, add a new row; do not edit history.

## Purpose

The goal is a 2B pure-audio duplex model: audio in, audio out, no text path,
no control tokens, no mode head. It handles ordinary conversation itself and
voices harness-supplied text only when reasoning is needed.

This file records what was tried, what was measured, what failed, and what the
next decision was. It exists so the reasoning can be read later without
reconstructing it from chat.

## Timeline

| Date | Step | Evidence | Outcome / next decision |
|---|---|---|---|
| 2026-11-11 | Remove control tokens, mode head, and text path | User requirement: duplex is audio in / audio out | Pure audio: `A_t` is always a codec frame; silence frames mean listening |
| 2026-11-11 | Re-voice all 20 pilot conversations | 20/20 agent tracks, original timing preserved | One Vani voice; user track untouched |
| 2026-11-11 | Build pure-audio cache | 1,605 windows, 63 frames each, 12.5 Hz | `u`, `codes`, `speak`, `b_speak`, `overlap`; no labels fed to model |
| 2026-11-11 | Teacher-forced pure audio, 8 epochs | Loss 2.866 → 2.067 | Inference mute: EOS at frame 0 |
| 2026-11-11 | EOS penalty on speech frames | Loss identical to the previous run | No effect under teacher forcing; stopped at epoch 2 |
| 2026-11-11 | EOS masked from training logits | Loss identical again | No effect under teacher forcing; stopped at epoch 2 |
| 2026-11-11 | EOS masked at inference only | 0 speech frames without mask; 23 speech frames with mask on user window | Mask forces speech everywhere; EOS is being used as silence |
| 2026-11-11 | Scheduled sampling, 10–35% own frames | Loss 3.177 → 3.186 | Agent window still 0 speech; independent replacement did not break the EOS attractor |
| 2026-11-11 | Curriculum A: speech-only, 365 windows, EOS masked, 6 epochs | Loss 4.118 → 3.320 | Agent window 85 speech frames; content repetitive but speaking behaviour exists |
| 2026-11-11 | Curriculum B: speech + 43 user-silent windows, EOS allowed, 3 epochs | Loss 2.895 → 2.704 | Both probe windows 0 speech; mute returned |
| 2026-11-11 | Curriculum C: all 1,605 windows, EOS masked, 4 epochs | Loss 2.566 → 2.295 | User window 5 speech frames, agent window 0; speech lost again |
| 2026-11-12 | Speech oversampling: all windows + speech-heavy ×3, 2,700 samples, EOS masked, 4 epochs | Loss 2.724 → 2.086 | Saved `audio_lora_balanced` |
| 2026-11-12 | Fair multi-window probe | User median 0.0; agent 0.024; boundary 0.0 speech ratio | Quiet on user turns, still mute on agent turns; oversampling insufficient |
| 2026-11-12 | Add fair multi-window probe | User/agent/boundary categories; distributions, not one window | Probe code: `vani-2b/train/duplex_audio_multi_eval.py` |
| 2026-11-12 | Voice validation samples | 6 generated lines + real re-voiced clip | Current Vani voice is the LFM3 TTS voice; filenames are not style conditions |
| 2026-11-12 | Commit and push duplex code | `32d94e7` | Reproducible branch state |

## Current state

- Data: 20 conversations, ~7.2 h, 1,605 windows.
- Adapters:
  - `audio_lora`: teacher-forced / scheduled; mute at inference.
  - `audio_lora_speech`: speech-only; speaks, repetitive, would over-talk.
  - `audio_lora_listen`: speech + user-silent; mute at inference.
  - `audio_lora_all`: all windows, EOS masked; user near-silent, agent mute.
  - `audio_lora_balanced`: oversampling run; quiet on user, essentially mute on agent.
- No adapter yet shows both listening and speaking in the same checkpoint.
- Injection/continuity path is separate and unaffected.

## Probe philosophy

A single window is not a verdict. The model is 2B with 7.2 h; one window is
noise, and an agent-rich window may start mid-turn without a boundary cue.

The fair probe must have:

1. **Multiple windows per category**: user-speaking, agent-speaking, boundary.
2. **Held-out conversations**, split by conversation, not by window.
3. **Distributions**, not single values: median speech ratio, any-speech rate,
   false-speech rate while the user talks, boundary onset.
4. **Tolerant bands**, not exact frame matching.
5. **Human listening** for naturalness and seam.

Smoke-test expectation (coarse, not acceptance):

- user-speaking window → low speech ratio (backchannels allowed),
- agent window → some speech frames,
- boundary window → speech starts after the user stops.

## What would prove "only scale is missing"

Run these in order:

1. **Overfit probe**: 50 speech + 50 user-silent windows, train to near-zero
   training loss. Can the model represent both behaviours at all?
2. **Held-out suite**: 20+ windows per category from unseen conversations.
3. **Data-scaling curve**: same objective at 10/25/50/100% of conversations;
   evaluate the same held-out suite. Monotonic improvement toward target means
   scale is the lever.
4. **Oracle turn-state diagnostic**: give the model the true turn state at
   inference. If it can then speak correctly, turn-taking is the bottleneck.
5. **Free-running vs teacher-forced gap**: should shrink with more data.

If overfit fails, metrics are flat, or oracle does not help, scale alone is not
the answer.

## Voice note

The Vani voice is the LFM3 TTS voice, not a stored speaker identity. It is
conditioned by the system prompt in `revocalize.tts`:

> "Perform TTS in American English using a female voice."

That prompt describes who, not how. The sample script's filenames (`warm`,
`helpful`) are labels; no style was passed. Robotic delivery is expected from
bare, isolated TTS. A product voice needs a canonical prosody prompt or a
dedicated TTS/voice reference, outside duplex training.

## Open decisions

- The oversampling adapter failed the fair probe (agent median speech ratio 0.024).
- The tiny overfit test (E019) also failed to fit both behaviours cleanly on its own
  80 windows: loss 0.981, agent median 0.024, user median 0.032. This weakens the
  “only scale is missing” hypothesis.
- E020 full depth-decoder fine-tuning diverged (NaN) in fp16; E021 depth LoRA was
  stable (loss 2.095 → 0.812) but still failed the same overfit probe (all categories
  median 0.048). The model cannot cleanly memorize both behaviours even on 80 windows.
- Conclusion: scale is not yet justified. The next step is a capability test on a much
  smaller set (e.g., 5 speech + 5 user windows) with higher LoRA rank, plus the oracle
  turn-state probe. If neither fits, the pure-audio objective/architecture needs
  revision before scale.
- E022: 10-window overfit still failed (loss 0.610; user median 0.0, agent 0.032).
  The model cannot memorize ten alternated windows. Next isolation: speech-only
  overfit; if that also fails, the audio objective/head is the blocker, not data volume.
- Do not add control tokens or a mode head to hide the failure.
- Keep every result, including failures, in this file and in
  [experiments.md](experiments.md).
