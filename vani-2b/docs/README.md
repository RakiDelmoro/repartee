# Vani-2B documentation

One 2B speech model for live conversation and harness-supplied answers.

| Document | Covers |
|---|---|
| [end-goal.md](end-goal.md) | what Vani is for, what it is not, success criteria |
| [architecture.md](architecture.md) | model, stream layout, injection, voice, runtime |
| [decisions.md](decisions.md) | every decision, with rationale and status |
| [experiments.md](experiments.md) | every experiment and result, including failures |
| [plan.md](plan.md) | phases, gates, current todo |
| [evaluation.md](evaluation.md) | the five checks and how to run them |
| [data.md](data.md) | corpora, licenses, paths, layouts |

## Status snapshot — 2026-11-10

| Piece | State |
|---|---|
| Mouth adapter | **works**: 92.5% exact, 2.77 words/s |
| Continuity injection | **works**: 90% answer present, 98.3% recall, median 0 ms join |
| Hard switch | **works**: 11/12 answer present, median 0 ms join |
| Joint duplex pilot | **fails**: 0/63 speak frames on an agent-turn window |
| Cause | 50 windows from 5 short conversations; labels partially silent; no turn-boundary sampling |
| Current work | re-voice 30–60 min of Open Yap, turn-boundary cache, retrain, re-check |

Rules for this folder:

- Append to `decisions.md` and `experiments.md`; do not rewrite history.
- A negative result stays in the log.
- No claim of “works” without a file and a measured result.
