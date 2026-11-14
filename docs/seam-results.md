# Seam test results (first run)

Actor: LFM3-Audio-2B, base weights. Harness: `research/seam_test.py`, output
`data/seam/`. Question: when handed text is spoken mid-conversation, is the defect
caused by the moment, the cue, or the actor itself?

Setup: one text-only user question, one filler ("Let me think about that for a
second."), one 16-word answer, injected at three points. The answer is fed as one
text block followed by speech, the pattern that spoke a full sentence in the earlier
product demo. Metrics: energy-silence gap at the seam, ASR word recall of the answer
slice (Qwen3-ASR judge, numbers normalised), duplicate word, pitch step versus
within-sentence pitch movement.

| condition | seam gap | pitch step | within-sentence step | answer recall |
|---|---|---|---|---|
| A block boundary (24 frames) | 80 ms | **83.0 Hz** | 1.9 Hz | **0.652** |
| B mid-speech (frame 8) | 0 ms | 86.9 Hz | 1.2 Hz | 0.522 |
| C block boundary + cue phrase | 80 ms | **17.5 Hz** | 1.8 Hz | 0.478 |
| D filler only | 0 ms | – | 1.8 Hz | – |

## What the numbers say

1. **The moment is not the main problem.** A block boundary and a mid-speech
   injection both produce speech, with gaps of 80 and 0 ms; neither is catastrophic.
2. **Pitch continuity is the audible defect.** The step across the seam is 40–70×
   the within-sentence movement, i.e. the answer starts in a different register
   from the filler. The cue phrase cuts it to 17.5 Hz — the one clear win.
3. **Handed-text fidelity is the blocker.** Recall is 0.48–0.65: roughly a third to
   a half of the answer's words are wrong or missing even at the best boundary. That
   is not a boundary-policy problem; the actor does not render the injected text
   reliably.
4. **End-of-speech is a dead end.** Injecting after the filler's own EOS produced
   zero audio (measured twice before the harness was changed to a block boundary).
   The turn is closed at that point, so a handover there must start a new turn.

## Interpretation

The seam test was meant to choose between "train the boundary policy" and "change
the model". It points at the second: no boundary rule recovers a third of the
words. The likely cause is that a text block followed by speech is in-distribution,
but a *long* forced block mid-turn is not — the model was trained on short
interleaved blocks and on turns that end, not on one turn containing two
utterances.

## Full-file ASR (what is actually in each file)

| file | transcript |
|---|---|
| A block boundary | "Let me think about that for a second. **We do too in the second week of April.** Tom will document it so nobody keeps planning against them. Tom will document." |
| B mid-speech | "**Lenny,** the launch is moving to Q two, and nobody keeps planning against March." |
| C cued boundary | "Let me think about that. The launch is moving to Q two in the second week of**anja**." |
| D filler only | "Let me think about that for a second." |

Two things this exposes beyond the metrics:

- **A repeats itself** ("Tom will document" twice) and mangles the answer's opening.
- **B cut the filler mid-word** ("Lenny" is the residue of "Let me…"), so its "0 ms gap"
  is not a clean handover; it is an interruption of the actor's own speech.

So the ranking by metrics (A best on recall, C best on pitch) does not survive the
transcripts: none of the three delivers the answer intact.

## Caveats

One sample per condition, a text-only user turn, an ASR judge with its own errors,
no human listener yet. F0 via `librosa.pyin`; gap via an energy threshold. The files
are in `data/seam/`; listen before trusting the table.

## Second round: fixing the jump

| condition | handover | gap | pitch step | recall | what the transcript shows |
|---|---|---|---|---|---|
| A | block boundary, no pause | 0 ms | 11.7 Hz | 0.522 | answer's second half garbled ("…of ten zero") |
| E | cue + 240 ms pause, same state | 1120 ms | – | 0.522 | extra ~0.9 s of silence after the pause |
| F | cue + pause + new turn | **240 ms** | – | 0.391 | "…of Anubis keeps planning…" |
| G | cue + pause + clause blocks | 1120 ms | – | 0.565 | **first sentence verbatim, second missing entirely** |
| H | cue + pause + one sentence per turn | **240 ms** | 37.2 Hz | 0.565 | both sentences present, first mangled ("Years so, right is moving to Q") |

Findings:

1. **The jump is fixed by a marked boundary**: cue + 240 ms pause (H) gives exactly the
   intended pause and no extra silence. F and H are the only conditions with a clean gap.
2. **The model renders one short sentence well and collapses on more.** G shows it
   plainly: first sentence word-perfect, second sentence produces no audio at all.
3. **Assistant text in the history leaks into the speech.** H's "Years so, right" is the
   cue phrasing bleeding back in because the filler was re-fed as assistant text.
4. Recall never exceeded 0.57 in any condition: the handed-text fidelity, not the seam,
   is the limiting factor.

Rule that follows: hand over **one short sentence per turn**, keep the cue **in audio
only** (never re-feed it as assistant text), and put a 240 ms pause at the boundary.
If recall stays below ~0.9 under that rule, the actor cannot be used for answer-length
speech and the model-level branch is next.

## Final confirmation (condition I)

Same generation, same voice, cue only in audio, one short sentence at a time, 240 ms
pause at the boundary:

| condition | gap | pitch step | recall | transcript |
|---|---|---|---|---|
| A (no cue, no pause) | 0 ms | 11.7 Hz | 0.522 | "…in the second week of ten zero." |
| H (new turn per sentence) | 240 ms | 37.2 Hz | 0.565 | "Years so, right is moving to Q. Tom will document it…" |
| **I (same voice, sentence per handover)** | **240 ms** | **7.6 Hz** | 0.435 | "Right, so the launch is moving to Q two, in the second." |

Two facts settle the architecture question:

1. **The seam itself can be made clean in this voice**: 240 ms gap, pitch step 7.6 Hz
   against 1.6 Hz within-sentence. That is the best boundary of any condition.
2. **A turn holds exactly one utterance.** After the first sentence the model emits
   end-of-speech; any further handed text produces no audio. Multi-sentence answers
   require a new turn (H), which changes the voice and leaks the cue text back in
   ("Years so, right…", and "Right, so" in I's answer slice — the cue echoing from the
   audio context).

So with this actor you can have same-voice **or** multi-sentence, not both, and neither
mode exceeds 0.57 answer recall.

## Third round: the actor as its own TTS (no second model needed)

The earlier failures were self-inflicted: the filler text was re-fed as assistant
history (leaking "Years so, right…") and each handover carried a whole sentence
(16 words), while the actor renders about the first eight words of any handed block.

| condition | handover | gap | recall | transcript |
|---|---|---|---|---|
| J | new turn per sentence, clean history | 720 ms | 0.696 | "…Q two. Tom will document it so nobody keeps planning against." (tails dropped) |
| K | new turn per sentence, 6-token blocks | 640 ms | 0.739 | "…Q two. In the second week, Tom will document it so nobody." |
| **L** | **new turn per phrase, ≤8 words, cue in audio** | **640 ms** | **1.000** | **"The launch is moving to Q two, in the second week of April. Tom will document it so nobody keeps planning against March."** |

So the rule is:

1. hand over **≤8 words at a time**,
2. each phrase as a **clean turn** with only system+user in history — never the filler,
3. cue phrase **in audio only**,
4. 240 ms pause at the handover, 120 ms between phrases.

Under that rule the same model, same voice, renders the full answer word-for-word.
No TTS, no bridge, no training. What remains to judge by ear is the 640 ms seam
(240 ms pause + the model's ~400 ms onset) and whether the pauses between phrases
sound like thinking or like choppiness.

## Full-loop sample

`research/full_flow.py` renders the whole interaction into one file
(`data/flow/full_flow.wav`, 50.1 s):

| t | segment |
|---|---|
| 0.0 | user speaks (8 s of the interview sample) |
| 8.0 | actor answers in its own words (13.6 s, generic content — expected) |
| 21.6 | 240 ms handover pause |
| 21.84 | injected made-up fact, phrase by phrase (7.6 s) |
| 29.4 | user speaks again (4.8 s) |
| 34.2 | actor reacts after re-listening (15.9 s) |

ASR of the injected fact: "Right, so so, the Harrow project was cancelled last
Friday. And its budget moved to the." — cue + both phrases, but the last two words
("satellite team") are dropped, and the cue doubled ("so so").

Adjustments this sample asks for: phrase cap 6 words rather than 8, cue shortened to
"Right," to avoid the double "so", and a length cap on the actor's own responses.

## Full loop, corrected (data/flow3/full_flow.wav)

| segment | duration | words/s | median F0 | last 40 ms |
|---|---|---|---|---|
| user 1 | 8.0 s | 2.0 | 86 Hz | 0.014 |
| actor: own response | 26.0 s | **3.15** | 176 Hz | **0.000** (natural end) |
| handover pause | 0.24 s | – | – | – |
| injected fact | 7.0 s | **2.14** | 186 Hz | **0.000** (natural end) |
| user 2 | 4.8 s | 3.33 | 105 Hz | 0.016 |
| actor: reaction | 18.7 s | 2.51 | 174 Hz | 0.000 (natural end) |

Fact ASR: "Right, the Harrow project was cancelled last Friday. Its budget moved to
the satellite team." — word-for-word, cue not doubled.

Fixed: natural endings for every agent turn (the source utterance is no longer cut
mid-clause), cue reduced to "Right," with no doubling, phrase-final words restored
(the earlier loss was my 30-frame cap, not the word count).

Still open: the injected fact runs at 2.14 words/s against the response's 3.15
(32% slower), so the tempo still changes at the seam; and the actor's own turns
ramble (26 s, 18.7 s) because the 600-token budget lets them finish naturally but
does not cap length.

## Full loop, cue and length fixed (data/flow5/full_flow.wav)

| segment | dur | words/s | F0 | last 40 ms | ASR |
|---|---|---|---|---|---|
| user 1 | 8.0 s | 2.00 | 86 Hz | 0.014 | — |
| actor reply (2 sentences, phrase-rendered) | 14.6 s | 2.05 | 187 Hz | 0.000 | "Yeah, totally… tiny slice of people," |
| handover pause | 0.12 s | – | – | – | – |
| cue + injected fact | 7.5 s | 2.14 | 198 Hz | 0.000 | "Oh one, the Harrow project was cancelled last Friday. Its budget moved to the satellite team." |
| user 2 | 4.8 s | 3.33 | 105 Hz | 0.016 | — |
| actor reaction | 6.5 s | 2.30 | 195 Hz | 0.000 | complete sentence |

Progress: the tempo difference across the seam fell from 32% to **4%** (both the
actor's own words and the injected text now go through the same phrase renderer), the
cue is a topic-shift marker chosen for the fact ("Oh, one more thing,"), every agent
turn ends at 0.000 energy, and the actor's turns are 7–15 s instead of 26 s.

Two defects left, both small:

1. the cue is cut ("Oh one" — the cue turn is capped at 20 frames, needs 30);
2. repeated words at some phrase boundaries ("something something", "valuable, or
   valuable") from splitting a phrase at a non-punctuation edge.

## Why the intra-sentence pause exists, and what removes it

Cause: every phrase is its own generation. Each generation ends with trailing
silence and the next begins with ~300–400 ms of onset silence, plus our splice gap.
Nothing in the splice code creates it; the per-turn design does.

Three attempts, measured:

| attempt | pauses inside the fact | fidelity |
|---|---|---|
| phrase-per-turn, no trim (flow7) | 400–600 ms | full words |
| phrase-per-turn, trim at 0.010 RMS, 20 ms guard (flow8) | 500–540 ms | **clipped** ("last Friday" → "Lab") |
| single turn per answer (flow9) | still 600 ms internal | **truncated** ("…canceled last day") |
| phrase-per-turn, trim at 0.003 RMS, 60 ms guard (flow10) | **~200–240 ms** | one phrase lost ("cancelled last Friday") |

Verdict: the pause can be brought into the natural clause range (~200 ms) by trimming
with a low threshold and a generous guard, but phrase rendering is not deterministic —
sometimes a phrase comes back short. The missing piece is verification: ASR each
phrase, compare with the intended text, re-render when it is far off. That turns a
variance into a retry instead of an audible hole.

## M1: carrying the previous phrase as audio (negative)

History for phrase 2 = system, user text, then the assistant's previous phrase as
`audio_out` codec frames — so the model continues its own voice instead of a cold turn.

| build | lead silence | words/s | recall |
|---|---|---|---|
| phrase 1 (cold turn) | **1220 ms** | 2.08 | 1.00 |
| phrase 2, baseline (cold) | **460 ms** | 3.12 | 1.00 |
| phrase 2, M1 (audio carried) | **1340 ms** | 2.19 | 1.00 |

Fidelity is fine (1.0 everywhere), but the mechanism **worsens** the onset: carrying
the previous phrase makes the model wait *longer* before speaking, consistent with the
history ending in an end-of-speech frame — it reads as a completed turn and pauses
before starting again.

Conclusion: continuity cannot be obtained by re-conditioning a new generation on the
old audio. The model has to keep generating — no end-of-speech, no re-prefill — which is
M2 (mask end-of-speech, cut at audio-block boundaries, feed the next phrase's text into
the same cache).

## M2: one generation, EOS suppressed (decisive)

Logit-level suppression of the audio end-of-speech code (via the sampler's `suppress`
list), cut at a quiet block boundary, next phrase fed into the same cache:

| phrase | frames | lead silence | recall | what the transcript shows |
|---|---|---|---|---|
| 1 | 60 (4.8 s) | 900 ms | 1.00 | correct sentence |
| seam | – | **380 ms** | – | no re-prefill happened |
| 2 | 72 (5.76 s) | – | 1.00 | **the sentence is spoken twice** |

Two findings, and they close the question:

1. **Continuity still does not remove the onset silence.** Even without a re-prefill,
   phrase 2 leads with 380 ms of silence. The pause is the model's speaking style, not
   the splice.
2. **Without end-of-speech there is no stop.** Suppressing EOS lets the stream run, and
   the model repeats the sentence, because its only "this utterance is finished" signal
   is the EOS code itself.

So the trade is structural: EOS on → turn closes → re-prefill → onset + prosody reset;
EOS off → continuous → repetition. There is no third option available at inference
time with this checkpoint. A true stream needs a model trained to interleave text and
speech at turn level (TurnGuide-style), i.e. the training branch.
