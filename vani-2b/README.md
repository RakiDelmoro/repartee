# Vani-2B

A 2B speech model for natural live conversation with a harness that owns reasoning.

Vani does two things in one voice:

1. **Duplex conversation** — speech in, speech out. It listens while speaking,
   waits, backchannels, and takes turns.
2. **Injected answers** — when the harness needs to reason, it supplies text
   mid-stream; Vani speaks it as a continuation of what it was already saying.

The harness may call a paid model for reasoning and the knowledge base. Vani is
the mouth, not the brain.

Base: LFM3-Audio-2B, frozen encoders/codec, LoRA adapters.

Documentation: [`docs/`](docs/README.md) — end goal, architecture, decisions, experiments, plan, evaluation, data.

## Layout

```
train/        active training and evaluation scripts
data/         manifests, labels, cached tensors (gitignored)
adapters/     mouth, switch, continuity, joint LoRAs (gitignored)
samples/      model outputs and check sheets
eval/         reports
```

Large corpora stay outside:

- `data/open-yap/` — full corpus, index, pilot, re-voiced pilot
- `models/` — base weights

## Current state

| Piece | State |
|---|---|
| Mouth | 92.5% exact, 2.77 w/s |
| Continuity injection | 90% B present, 98.3% recall, median 0 ms gap |
| Hard switch | 11/12 B, median 0 ms gap |
| Duplex policy | **works**: 0/63 while user speaks, 8–37/63 on agent turn; injection exact |
| Remaining gap | audio-only agent speech loops/fragments; needs a text plan in the agent slot |

## Next

1. Rebuild the duplex labels from the re-voiced audio (not the original transcript).
2. Scale to 30–60 min of re-voiced conversation.
3. Sample windows around turn boundaries.
4. Keep the mode loss unweighted; balance by sampling.
5. Re-run the five checks before any full-corpus run.

## Checks

- same voice for filler, backchannel, and injected answer
- waits while the user speaks
- takes the floor when the user stops
- stops on interruption
- injected answer is a continuation, not a restart
- injected words are exact
