# Vani-2B documentation

One 2B speech model for live conversation and harness-supplied answers.

| Document | Covers |
|---|---|
| [end-goal.md](end-goal.md) | what Vani is for, what it is not, success criteria |
| [architecture.md](architecture.md) | model, stream layout, injection, voice, runtime |
| [decisions.md](decisions.md) | every decision, with rationale and status |
| [experiments.md](experiments.md) | every experiment and result, including failures |
| [duplex-log.md](duplex-log.md) | chronological duplex decisions, failures, probe design, and scaling plan |
| [plan.md](plan.md) | phases, gates, current todo |
| [evaluation.md](evaluation.md) | the five checks and how to run them |
| [data.md](data.md) | corpora, licenses, paths, layouts |

## Status snapshot — 2026-11-12

| Piece | State |
|---|---|
| Mouth adapter | **works**: 92.5% exact, 2.77 words/s |
| Continuity injection | **works**: 90% answer present, 98.3% recall, median 0 ms join |
| Vani voice | **works**: one TTS voice; samples under `data/probe/samples/vani_voice/` |
| Pure-audio duplex | **not yet**: speech-only and listen-only adapters work separately; no single adapter has both |
| EOS finding | the codec EOS is the model's silence shortcut; teacher forcing hides it |
| Current work | speech-oversampled run, then the fair multi-window probe |
| Next decision | scale only after overfit + held-out + scaling-curve + oracle diagnostics |

Rules for this folder:

- Append to `decisions.md` and `experiments.md`; do not rewrite history.
- A negative result stays in the log.
- No claim of “works” without a file and a measured result.
