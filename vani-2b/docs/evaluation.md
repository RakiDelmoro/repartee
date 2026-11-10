# Evaluation

Five checks. Human ears decide; numbers are evidence, not the verdict.

| # | Check | Pass | How it is measured |
|---|---|---|---|
| 1 | Same voice | filler, backchannel, and injected answer sound like one speaker | listening; speaker-similarity hint |
| 2 | Waits while the user speaks | no speech during user speech | 5 s user-speaking window: speak frames = 0 |
| 3 | Takes the floor | speaks after the user stops | agent-turn window: speak frames > 0, intelligible audio |
| 4 | Interrupt | stops when user starts | interrupt clip: speech ends within 200 ms |
| 5 | Handoff | injected answer continues, not restarts | lead/answer/combined clips; gap ≤ 40 ms; no pitch reset |
| 6 | Words | injected sentence is exact | independent ASR of the combined clip |

Pass = all checks. A single miss is a fix item, not a pass.

## Files

`vani-2b/samples/` holds the current check set:

- `joint_pilot/duplex_user.wav` — user-speaking window.
- `joint_pilot/duplex_agentturn.wav` — agent-turn window.
- `joint_pilot/injection_1.wav`, `injection_2.wav` — handoff and words.
- `joint_pilot/CHECKLIST.md` — filled result.
- `joint_pilot/report.json` — mode counts and ASR.

## Current result — 2026-11-10

| Check | Result |
|---|---|
| Same voice | cannot test; no duplex speech |
| Waits | **pass**: 0/63 speak |
| Takes floor | **pass**: 8–37/63 speak |
| Interrupt | not tested |
| Handoff | 2/2 exact; joins 0–190 ms |
| Words | 2/2 exact |

Verdict: **not ready**. See `experiments.md` E005/E006.

## Rules

- Test on held-out speakers and sentences, never training clips.
- Keep the automated evidence beside the check; do not replace it with a score.
- A gain in one check that costs another is a regression, not progress.
