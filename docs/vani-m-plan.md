# Vani-M plan: a local, trainable, full-duplex mouth

Status: **P0 passed. The first 8.6 h P1 run failed the listening gate. Dataset
scaled to 45 conversations / 29.6 h; 5,000-step all-codebook retrain is running.
P2 remains blocked.**
This is a new small model, not a patch to Vani-4B/Haru.

## What we want

1. listens and speaks at the same time;
2. natural timing, overlap, backchannels;
3. accepts text injected at **any frame** and speaks it in its own voice;
4. a bigger reasoner supplies facts/answers;
5. fits and **trains** on the 8 GB card;
6. trained on Open Yap, no rented GPU;
7. **speaks immediately from its own knowledge**; injected text is an override, not a
   prerequisite. There is no “waiting for text” state.

## Model

Moshi-style, 600M–1B. Not Haru’s three-stream continuous architecture.

| part | choice |
|---|---|
| temporal backbone | Qwen3-0.6B-Base (text-pretrained) |
| audio codec | Mimi, frozen, 12.5 Hz, 8 RVQ codebooks, 24 kHz |
| streams per 80 ms frame | agent text, 8 agent audio codebooks, 8 user audio codebooks |
| depth transformer | 12 layers (up from 6), per-codebook loss |
| size | ~0.7–1B; fits 8 GB with batch 1 + checkpointing |
| injection channel | agent text stream, forced per frame |

Why Qwen3-0.6B: text-pretrained, the blog’s proven backbone, and text/timing converge.
LFM3-2B is audio-native but turn-based; adapting it to duplex is a new interface too. It
is a fallback, not the first attempt.

## Data

- `open-yap-1k_2026-11-05_flac_26a4d80b/` — tar-only, 149 GB.
- 1,602 conversations, 1,000.2 h, 37.5 min average.
- 48 kHz mono per speaker, aligned, word-level transcripts, real overlap.
- 9.9 M words; 52 k filler tags.

## Training plan

### P0 — tooling and data check (no training)

- Mimi tokenizer for both channels at 24 kHz / 12.5 Hz.
- Text alignment to 80 ms frames; role swap A/B.
- User streams **input-only, zero loss**.
- Tokenize 10–20 h; decode a stored token grid back to clean audio.
- Gate: round-trip clean; frame count and text alignment correct; no missing
  conversations.

### P1 — alignment stage

- Single-speaker clean audio from Open Yap model-speaker turns (or LibriTTS-R/Emilia).
- Agent-only prediction; no duplex.
- Learn codebooks 0–1 first, then unlock 2–7.
- Per-codebook CE and acoustic weights, not one aggregate loss.
- 20–50 h.
- Gate: teacher-forced and free-run audio intelligible; ASR matches text.

### P2 — duplex stage

- Open Yap two-channel conversations; real overlap/backchannels.
- Agent text + audio targets; user audio input-only.
- Per-codebook loss; role swap; 50–100 h.
- Gate: audio still intelligible; turn-taking/overlap natural; no regression vs P1.

### P3 — text injection

- Force the agent text stream at a chosen frame: `cue + answer`.
- ASR the output; does it say the injected sentence?
- Add `<handover>` / `<cue_end>` only if the forced boundary needs learning.
- Gate: injected recall ≥ 0.95; seam acceptable.

### P4 — scale and runtime

- If P1–P3 pass, scale to 1,000 h.
- Streaming runtime at 12.5 Hz, KV cache, injection queue.
- Reasoner integration: fast path, optional stall, cue, answer, barge-in.

## How and why each phase

The order is deliberate: data correctness before compute, clean audio before duplex,
duplex before injection, scale last. Each phase has a stop rule so a failure does not
silently contaminate the next one.

### P0 — tooling and data check

**Why first.** Every later number is meaningless if the token grid or text alignment is
wrong. The 600M experiment spent GPU time before proving the decode round-trip; we do the
cheap checks first. Open Yap is already two-channel and aligned, so the risk is our
conversion, not the corpus.

**How.**

- Resample both 48 kHz channels to 24 kHz for Mimi.
- Mimi-encode each channel to 8 RVQ codebooks at 12.5 Hz.
- Build the 17-stream grid: agent text, 8 agent audio codebooks, 8 user audio codebooks.
- Align the agent transcript to 80 ms frames; role-swap A/B deterministically.
- Mark user streams **input-only, zero loss**.
- Save 10–20 h; decode a stored grid back through Mimi.
- Check: decoded audio is clean; frame counts match; text lands on the right frames; no
  missing conversations.

**Stop rule.** If the round-trip is dirty, fix codec/resampling/alignment. Do not train.

**P0 result (2026-11-17).** One 61-minute conversation produced a `(45,750, 17)` grid
with 6,666 text tokens; decoded audio matched duration and level, first-pass SNR 4.67 dB.
The owner listened to `decoded.wav` against `original.wav` and judged the reconstruction
good enough to retain: **8 Mimi codebooks at 12.5 Hz is an acceptable training target.**
The 16-conversation / ~10 h subset is being tokenized to confirm the distribution; the
gate is not structural anymore, only consistency across conversations.

### P1 — alignment stage

**Why.** The 600M run skipped read-aloud alignment and produced text that made sense over
audio that did not. Fine RVQ codebooks are residual detail; they need clean,
single-speaker supervision before they can be learned inside overlapping dialogue.
This is the stage that decides whether the model can speak at all.

**How.**

- Use model-speaker turns from Open Yap (or LibriTTS-R/Emilia if Open Yap turns are too
  noisy).
- Agent-only prediction: no user stream, no duplex.
- Learn codebooks 0–1 first (semantic/coarse), then unlock 2–7.
- Loss per codebook, with explicit acoustic weights; log each codebook separately.
- 20–50 h; batch 1; gradient checkpointing.

**Stop rule.** Teacher-forced and free-run audio must be intelligible and ASR must match
the text. If fine codebooks stay flat, do not start P2; increase clean hours, depth
capacity, or change per-codebook weighting.

**P1 scale-up (2026-11-17).** P0 now covers 45 conversations / 29.6 h
(1,332,750 frames, 45/45 tokenized). The all-codebook model is retraining for 5,000
optimizer steps with gradient accumulation 2, resuming from the 8.6 h checkpoint.
Dashboard: `http://127.0.0.1:8765`; log: `checkpoints/p1_28h/train.log`. Do not
start P2 until the held-out audio gate passes.

**P1 result (2026-11-17, 8.6 h subset).** Two 2,000-step runs (cb0–1, then all 8) on 14
conversations, holdout `conv_d65ed16a6092`. Held-out teacher-forced accuracy:

| target | cb0 | cb1 | cb2 | cb3 | cb4 | cb5 | cb6 | cb7 | text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| accuracy | 34.7% | 42.7% | 36.4% | 28.4% | 28.1% | 32.4% | 27.3% | 33.3% | 33.6% |
| CE | 4.16 | 3.30 | 3.85 | 4.09 | 4.07 | 4.01 | 4.16 | 4.00 | 3.62 |

Chance accuracy is 0.049% per codebook, so this is real learning, not collapse. But the
teacher-forced predicted-code WAV (`data/vani_p1_eval/teacher_forced.wav`) is not
intelligible even after normalization; free-run cannot be better. **Gate failed.** The
likely causes are training hours, not pipeline: 8.6 h is below the planned 20–50 h and
the stage-2 loss plateaued. Scale to ~28 h before changing architecture or loss weights.
Do not start P2.

### P2 — duplex stage

**Why.** Duplexity is a timing behaviour, not a second model. Once the model can speak
cleanly in one stream, Open Yap teaches when to speak: overlap, quick turns,
backchannels, interruptions. This is where the product behaviour appears.

**How.**

- Two-channel Open Yap conversations; real overlap, not stitched turns.
- Agent text + agent audio are targets; user audio is input-only.
- Per-codebook loss; role swap; 50–100 h first.
- Keep the P1 checkpoint; compare against it on the same audio.

**Stop rule.** Audio must not regress from P1, and turn-taking/overlap must be natural. If
audio regresses, roll back and fix the loss/weights before P3.

### P3 — text injection

**Why.** This is the product control channel: the reasoner writes, the mouth speaks. The
agent text stream is per-frame, so forcing it is the native way to control words. The
model must still listen while the text is injected.

**How.**

- Force the agent text stream at a chosen frame: `cue + answer`.
- Start without special tokens; use a short cue.
- ASR the model audio; compare with the injected sentence.
- Add `<handover>` / `<cue_end>` only if the forced boundary is unstable.
- One short sentence per burst; re-inject failed tails.

**Stop rule.** Injected recall ≥ 0.95 and the seam is acceptable. If not, do not scale;
fix pacing, cue, or add the learned handover.

### P4 — scale and runtime

**Why.** 1,000 h and streaming are expensive. They are only worth it after P1–P3 prove
the recipe. The runtime is also a separate risk: 12.5 Hz, KV cache, injection queue,
reasoner handover.

**How.**

- Scale the passing recipe to 1,000 h.
- Streaming runtime at 80 ms frames; injection queue checked each frame.
- Fast path always available; optional stall; cue + answer; barge-in.
- Evaluate ASR, turn-taking, injection, and barge-in.

**Stop rule.** Any regression at scale sends us back to the last passing subset.

## Decision log

| decision | reason |
|---|---|
| Moshi-style 600M–1B, not Vani-4B | 4B full training does not fit 8 GB; 4-bit 4B audio failed in every test |
| Qwen3-0.6B backbone | text-pretrained; the 600M blog showed text/timing converge; 2B LFM3 is turn-based |
| Mimi frozen | codec training is not the bottleneck; 80M params, CC-BY-4.0 |
| 12.5 Hz, 17 streams | Moshi-proven grid; text leads audio; one temporal step per 80 ms |
| user streams zero loss | the 600M run wasted capacity predicting the user; user audio is input only |
| per-codebook loss | aggregate audio loss hid the fine-codebook failure |
| alignment before duplex | the 600M run skipped it and never reached clean audio |
| Open Yap, real two-channel | real overlap/backchannels; better than the blog’s 100 h synthetic |
| fast path always on | product requirement: speak immediately; injection is an override |
| injection via agent text stream | native per-frame channel; reasoner supplies content |

## Compute

- Model 0.6–1B bf16: 1.2–2 GB.
- Mimi frozen: ~0.16 GB.
- Adam8bit + activations: ~2–3 GB.
- Total training ~5–7 GB; batch 1, short windows, gradient checkpointing.
- 1,000 h = 45 M frames; token grids ~1–2 GB.
- One epoch on 8 GB is slow — start with 10–50 h and gate.

## Fixes / does not fix

Fixes: duplex training fits the hardware; injection is native; Open Yap is real
two-channel data; the blog’s diagnosis is usable (alignment, per-codebook loss, user
streams zero, stronger depth transformer).

Does not fix: clean audio is still the risk; the model’s knowledge is small; new code
path (Mimi, 17 streams); 1,000 h is narrow and English-only; no guarantee.

## Start here

1. Download Qwen3-0.6B-Base and Mimi.
2. Build `Open Yap → Mimi 17-stream` tokenizer.
- Run P0 on 10 h, then P1 alignment on 20 h.
- Report the audio gate before any duplex run.

## Repo state

- `app/`, `docs/`, `tests/`, dataset retained.
- `models/`, `data/`, `vani/`, `research/`, `vendor/` removed.
- Fresh code will live in `vani/` (Moshi-style), separate from `app/`.
