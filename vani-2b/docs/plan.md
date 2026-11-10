# Plan

## Phase 0 — mouth and injection (done)

- Mouth: 92.5% exact, natural rate.
- Continuity: 90% answer present, median 0 ms join.
- Gate: passed. These are the voicing layer.

## Gate 1 — joint duplex pilot (in progress)

1. Re-voice 30–60 min of Open Yap agent speech (in progress).
2. Build 8 s windows around user-stop boundaries: 2 s before, 6 s after.
3. Use speech labels derived from the re-voiced audio.
4. Train one joint model from mouth + continuity, mixed duplex/injection/mouth.
4. Run the five checks in [evaluation.md](evaluation.md).

Pass requires: answer present, join, interruption, and turn-taking; no mute.

If it fails again: add a fixed role context, then more data. Do not scale.

## Gate 2 — full corpus (only after Gate 1)

- Re-voice the agent side of 1,000 h (dominant cost, ~500 GPU-hours TTS).
- Train on a rented 24–48 GB GPU.
- Full duplex eval: pause takeover, interruption takeover, backchannel resume,
  verbatim injection, voice similarity.

## Gate 3 — runtime

- Implement the controller: VAD, injection queue, block-boundary cut, cancel.
- Wire the paid model + knowledge base.
- Measure end-to-end latency on the target box.

## Todo

- [x] mouth adapter
- [x] continuity injection
- [x] hard-switch baseline
- [x] joint training loop and 8 GB feasibility
- [x] re-voice first 5 conversations
- [ ] re-voice 30–60 min total (running)
- [ ] turn-boundary cache
- [ ] joint retrain
- [x] five-check sheet (current result: fail)
- [ ] fix duplex and re-run checks
- [ ] scale re-voicing and full-corpus run
- [ ] runtime controller and paid-harness integration
