# Vani-M plan: a local, trainable, full-duplex mouth

Status: **P0 passed. P1 remains open. The 0.73 B / 100 h Open Yap recipe is
exhausted: cb1 stayed ~2%, the all-8 and cb1-weighted runs did not fix it, and
the fine-only phase did not adapt cb2–7 (`gt0_1` 0.849 new). Next: dense
read/aligned P1 speech, a larger/audio-pretrained backbone, or the product
cascade. P2 and P3 remain blocked.** This is a new small model, not a patch to
Vani-4B/Haru.

## What we want

1. listens and speaks at the same time;
2. natural timing, overlap, backchannels;
3. accepts text injected at **any frame** and speaks it in its own voice;
4. a bigger reasoner supplies facts/answers;
5. fits and **trains** on the 8 GB card;
6. trained on Open Yap, no rented GPU;
7. **speaks immediately from its own knowledge**; injected text is an override, not a
   prerequisite. There is no “waiting for text” state.

## Reading the stages

The phases are not four variants of the same model. Each adds one capability:

| stage | input | output | listening? |
|---|---|---|---|
| **P0** | Open Yap audio | 17-stream Mimi grid + text alignment | — |
| **P1** | target text + previous **agent** audio | next agent text + 8 agent audio codes | no |
| **P2** | user audio + agent audio | agent text + agent audio while listening | yes |
| **P3** | P2 streams + forced agent text | agent speech containing the injected text | yes |
| **P4** | passing P1–P3 recipe | 1,000 h scale + streaming runtime | yes |

**P1 is not speech-to-speech.** It is agent-only speech continuation: the model is
given target text and its own previous audio frame, and predicts the next agent frame.
The user audio channels exist in the 17-stream grid but are zeroed and receive no loss.
Listening starts in P2.

The shorthand “text → agent speech” means:

```
target text + previous agent audio
    -> next agent text token
    -> next 8 agent audio codes
```

It is not ordinary text-to-speech from a blank start. A prompt of held-out agent audio
establishes voice, timing, and acoustic context; the text controls what is said.

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
| text conditioning | current-frame agent text is an input to the depth transformer; previous-frame text stays in the temporal backbone |

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
- Give the depth transformer the **current** frame's text token; the temporal backbone
  keeps the previous frame's text.
- Learn codebooks 0–1 first (semantic/coarse), then unlock 2–7.
- Loss per codebook, with explicit acoustic weights; log each codebook separately.
- Weight voiced frames; do not let silence dominate the loss or the accuracy metric.
- 20–50 h; batch 1; gradient checkpointing.

**Stop rule.** Teacher-forced and free-run audio must be intelligible and ASR must match
the text. If fine codebooks stay flat, do not start P2; increase clean hours, depth
capacity, or change per-codebook weighting.

**P1 scale-up (2026-11-17).** P0 now covers 45 conversations / 29.6 h
(1,332,750 frames, 45/45 tokenized). The all-codebook model resumed from the 8.6 h
checkpoint and trained for 5,000 optimizer steps with gradient accumulation 2.
Dashboard: `http://127.0.0.1:8765`; log: `checkpoints/p1_28h/train.log`.

**P1 result (2026-11-17, 8.6 h subset).** Two 2,000-step runs (cb0–1, then all 8) on 14
conversations, holdout `conv_d65ed16a6092`. Held-out teacher-forced accuracy:

| target | cb0 | cb1 | cb2 | cb3 | cb4 | cb5 | cb6 | cb7 | text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| accuracy | 34.7% | 42.7% | 36.4% | 28.4% | 28.1% | 32.4% | 27.3% | 33.3% | 33.6% |
| CE | 4.16 | 3.30 | 3.85 | 4.09 | 4.07 | 4.01 | 4.16 | 4.00 | 3.62 |

Chance accuracy is 0.049% per codebook. This table is **all frames**; most of the
accuracy is silence (see the 2026-11-19 correction: voiced-frame cb0 is 3.75%). The
teacher-forced predicted-code WAV (`data/vani_p1_eval/teacher_forced.wav`) is not
intelligible even after normalization; free-run cannot be better. **Gate failed.** The
likely causes are training hours, not pipeline: 8.6 h is below the planned 20–50 h and
the stage-2 loss plateaued. Scale to ~28 h before changing architecture or loss weights.
Do not start P2.

**P1 result (2026-11-18, mislabeled 29.6 h — actually 8.6 h).** The 5,000-step
all-codebook baseline still failed the dense held-out gate: free-run cb0/cb1 accuracy
was roughly 1–3%, and text-forced audio gave WER **0.971**. Two self-conditioning
diagnostics also failed:

| run | cb0 free-run | cb1 free-run | dense WER |
|---|---:|---:|---:|
| 8.6 h baseline (mislabeled 29.6 h) | ~1% | ~2% | 0.971 |
| coarse-only scheduled sampling (29.6 h) | 3.65% | 0.52% | 0.926 |
| all-8 self-conditioning (29.6 h) | 2.60% | 1.04% | 0.985 |

Both 29.6 h self-conditioning runs saved `active_codebooks: 2`; they trained only
cb0/cb1, so neither is an all-8-codebook 29.6 h baseline.

**P1 correction (2026-11-19).** The 2026-11-18 checkpoint's saved args say
`grids=data/vani_p0_10h`; `checkpoints/p1_28h/train.log` starts with
`windows=2824 files=14`. That model saw 8.6 h of audio, not 29.6 h:
`vani/run_p1_28h.sh` waited for the 45 conversations but omitted
`--grids data/vani_p0_45`, so `train_p1.py` used its default 10 h directory. The
only runs that did use the 29.6 h set are the two self-conditioning diagnostics
(`files=44`), and both saved `active_codebooks: 2`; neither trained all 8 audio
codebooks on 29.6 h.

The 37–44% all-frame accuracy is mostly silence. Of 24,192 held-out frames, 7,591
are above 0.01 RMS. Voiced-frame accuracy is **3.75%** (cb0), **4.78%** (cb1) and
0.3–0.6% (cb2–7); voiced CE is 6.5, 5.6 and 7.0–7.4. Chance is 0.049%. The 8.6 h
model is a silence predictor, not evidence of "real learning" on speech.

The dense WER table below was scored against the spaced reference, now kept at
`data/vani_p1_gate/reference.txt`: natural speech 0.294, all-ground-truth Mimi
codes 0.397, the 8.6 h model 0.971. The codec adds ~0.10 WER, not the bulk of the
failure. The old 8.6 h eval artifacts have been removed; the table's numbers are
retained as history.

The old model also never gave the depth transformer the **current** frame's text.
Text entered only through the previous frame's temporal input, so a word's first
token arrived after the word's first audio frame. The corrected model adds a
zero-initialised `text_to_depth` path; the 29.6 h run trains it.

**P1 result (2026-11-19, 29.6 h).** `vani/run_p1_text29h.sh` trained 5,000 steps
on 29.6 h (5,529 voice-filtered windows, batch 1 × accum 2, 4.13 h). The checkpoint
is `checkpoints/p1_text29h/last.pt`. Held-out voiced-frame accuracy: cb0 **7.27%**,
cb1 **5.45%**, cb2–7 0.4–1.1%; voiced CE 5.86, 5.05, 6.69–7.12. This is better than
the 8.6 h model (3.75% / 4.78%) but still near chance.

The gate (one command: `bash vani/gate_p1.sh`) gives:

| variant | codes | WER |
|---|---|---:|
| `gt_all` | all 8 ground truth | **0.279** |
| `gt0_1` | true coarse, generated fine | **0.353** |
| `gt2_7` | generated coarse, true fine | 0.971 |
| `textforced` | all model | 0.927 |

So the codec and fine codebooks pass: with true coarse codes the output is within
0.07 WER of the codec ceiling. The failure is entirely cb0–1: in free-run they are
2.6% / 1.0% accurate on the gate segment and the model says unrelated words.

The text channel is learned (`text_to_depth` norm 11.4 vs 16.3 for the main depth
projection) but it does not control the coarse prediction. The likely causes are
that a word token is present only on its onset frame and that an additive bias is
too weak. Next changes, in order: (1) make the text a dedicated depth token
(Moshi-style) rather than an additive bias; (2) hold the current word across its
duration; (3) a cb0–1-first curriculum. Do not start P2.

Listenable WAVs are in `data/vani_p1_gate/`: `original_normalized.wav`,
`gt_all/`, `gt0_1/`, `gt2_7/`, `textforced/`, each with `generated_normalized.wav`
and `prompt_normalized.wav`.

**P1 token + coarse-first probe (2026-11-19).** A 500-step probe plus a 1,500-step
continuation trained a dedicated text token in the depth transformer, with cb0–1
only (`active_codebooks: 2`). Held-out voiced accuracy rose cb0 **7.27% → 10.59%**
and cb1 **5.45% → 7.88%**; voiced CE fell 5.86→5.51 and 5.05→4.74. On word+voiced
frames the text changes the cb0 argmax on **38.9%** of frames (KL 0.159), and
real-word accuracy is 9.0% vs PAD 7.7%. The token is being used, but it is still
weak. Gate not run; P1 remains open. Next: hold the current word across its
duration, then a 2k run and gate.

**P1 held-word probe (2026-11-19).** A 2,000-step held-word run (resumed from the
2k token checkpoint, cb0–1 only) reached voiced cb0 **11.82%** and cb1 **7.73%**
(from 10.59% / 7.88%). The word now covers 13,178 held frames on the holdout and
changes the cb0 argmax on 32.7% of them, but real-word accuracy is only 11.4% vs
PAD 10.4% on word+voiced frames (random 11.1%). This is the same slow slope as
before, not a step change. **Text timing/alignment is not the main limiter.**
Held-word is a **probe only**: it requires the runtime to hold each word for its
duration, which the reasoner does not supply. The deployable line stays on the
per-frame sparse text token (`p1_textslot2k`: cb0 10.59%), where the injector
queues one token per frame with PADs and needs no word timing. Next: dense P1
data (speech-only turns or read/aligned speech), then 100 h, then a backbone
change. Do not run another text-timing variant first.

**P1 100 h run (2026-11-19).** `vani/run_p1_100h.sh` built 104 new conversations
(101.23 h total, 149 conversations) and trained 5,000 steps on it: sparse text,
cb0–1 only, dense windows (`min_voiced 0.7`), resumed from `p1_textslot2k`. Voiced
cb0 rose **10.59% → 14.32%**, but cb1 fell **7.88% → 2.11%**. The sparse-text
ablation is stronger: the word changes cb0 on 43.0% of word frames (KL 0.191) and
real-word accuracy is 12.7% vs PAD 10.5% on word+voiced. Data scale helps cb0; cb1
regressed on the old-holdout speaker. Likely causes: the old 45 are only ~30% of
the 100 h (old-speaker forgetting) and the fine codebooks were frozen from the
the 29.6 h model. A fresh unseen conversation (`conv_70cb754c9722`, not in the
149) confirms both: voiced cb0 **12.24%**, cb1 **2.37%**; `textforced` WER 0.932;
`gt0_1` WER **0.890** (the old holdout was 0.353); `gt_all` 0.110. The codec
ceiling is excellent; the **frozen fine codebooks do not generalize** to the new
speakers, and cb1 is broken on both domains. Next: unlock cb2–7 on the 100 h data
(all-8 fine-tune) and address cb1; gate both holdouts. Do not add more data blindly.
checkpoint: `checkpoints/p1_100h/last.pt`.

**P1 all-8 100 h result (2026-11-20).** A 5,000-step all-8 fine-tune from
`p1_100h` **regressed both halves**: old-holdout cb0/cb1 13.08% / 1.82% (base
14.32% / 2.11%), new-domain 10.82% / 2.08% (base 12.24% / 2.37%); gate WER
old `gt0_1` 0.912 (29.6 h all-8 was 0.353), new `gt0_1` 0.986 (100 h coarse base
0.890); old `gt2_7` 0.941, new 0.932; textforced old 0.971, new 0.945. `gt_all`
remains 0.279 / 0.110. The fine codebooks did not adapt in 5k; the all-8 loss
degraded the coarse pair. **Do not use `p1_100h_all8` as the base.** Next:
cb1-weighted coarse-only from `p1_100h`, and a separate fine-code phase; the
plain all-8 recipe is falsified.

**P1 timeboxed phases (2026-11-20).** Two single-variable phases from the 100 h
coarse base (`p1_100h`), 2k + 5k steps:

- **A, cb1-weighted cb0–1** (`cb_weights 1,3`): old cb0/cb1 14.08/2.23, new
  11.66/2.14. cb1 did not move; cb0 did not improve.
- **B, fine-only** (`cb_weights 0,0,1,1,1,1,1,1`): the shared depth moved to the
  fine codebooks and collapsed the coarse heads (old cb0/cb1 3.46/2.46, new
  3.38/1.26). It did not adapt the fine codes usefully: `gt0_1` old 0.912, new
  0.849; `textforced` old 0.956, new 0.986. `gt_all` remains 0.279 / 0.110.

The pre-registered contract is exhausted: cb1 stayed ~2%, and the fine-only
phase failed `gt0_1` (new > 0.70). **The 0.73 B / 100 h Open Yap P1 recipe cannot
pass.** `p1_100h` remains the best base (14.32/2.11 old, 12.24/2.37 new) but still
fails the gate. Next: dense read/aligned P1 speech (the plan's fallback), a
larger/audio-pretrained backbone, or the product cascade. No more 0.73 B / 100 h
Open Yap tuning.

The codebook-swap ablation localises the failure to coarse content:

| variant | cb0–1 | cb2–7 | WER |
|---|---|---|---:|
| all ground truth | ground truth | ground truth | 0.397 |
| `gt0_1` | ground truth | generated | 0.412 |
| `gt2_7` | generated | ground truth | 0.985 |

When coarse codes are correct, generated fine codes are sufficient for intelligible
words. When coarse codes are generated freely, content is lost. The P1 gate remains
open; P2 and P3 stay blocked.

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

- `app/`, `docs/`, `tests/` retained.
- `vani/` holds the Moshi-style model, training, generation, and evaluation code.
- Deployable checkpoint: `checkpoints/p1_textslot2k/last.pt` (per-frame sparse
  text token, cb0 10.59%) plus its `eval.json` and `text_ablation.json`.
- Baseline evidence: `checkpoints/p1_text29h/eval.json`/`train.log`,
  `checkpoints/p1_28h/{eval,eval_old}.json`/`run.json`/`train.log`, and the
  `p1_coarse_sc*/train.log` diagnostics.
- `data/vani_p0_45`: grids and `.rms.npy` voice-activity sidecars. The held-word
  `.word.npy` sidecars were deleted (probe only); `vani/build_word_cond.py` can
  rebuild them.
- `data/vani_p1_gate`: validation WAVs, ASR, metrics, `reference.txt`.
- Failed-run and probe checkpoints were deleted on 2026-11-19.
- `models/`, `data/`, `checkpoints/` are local artifacts and are gitignored.
