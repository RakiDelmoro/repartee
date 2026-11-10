# Experiment log

Append-only. Every result, including failures, with the data and the verdict.

| ID | Date | Experiment | Data | Result | Verdict |
|---|---|---|---|---|---|
| E001 | 2026-11-10 | Naive forced injection | 8 runs, one clause | 7/8 content-complete; 150–170 ms join silence | fail seam |
| E002 | 2026-11-10 | Mouth adapter | 722 clauses; 40 held out | 92.5% exact; 2.77 words/s; 0 dropped tails | pass |
| E003 | 2026-11-10 | Hard switch v1 | 12 held-out pairs | 11/12 answer present; median 0 ms join; sounded like two takes | mechanism pass, seam fail |
| E004 | 2026-11-10 | Continuity v2 | 20 held-out pairs | 90% answer present; 98.3% recall; median 0 ms; one garbled name | pass continuity |
| E005 | 2026-11-10 | Joint pilot: duplex + injection + mouth | 50 windows from 5 short conversations; 200 steps | 0/63 speak frames on both user-speaking and agent-turn windows; injection 1/2 | fail duplex |
| E006 | 2026-11-10 | Joint pilot with TTS-energy labels | same windows, labels from re-voiced audio | still 0/63 speak; injection 2/2 exact, gaps 450/0 ms | fail duplex, injection intact |
| E007 | 2026-11-10 | Extended agent re-voicing | 11 conversations, ~43 min | in progress | pending |

## Interpretation

- The mouth and continuity are real, measured, and reproducible.
- The hard switch proves the injection mechanism; continuity proves the natural join.
- The joint duplex failure is not an injection failure. The model never chooses to speak.
- E006 isolates the cause: fixing the silence-labelled-as-speech bug did not make the
  model speak, so the remaining gap is scale and turn coverage.
