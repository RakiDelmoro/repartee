# Plan

## Phase 0 — mouth and injection (done)

- Mouth: 92.5% exact, natural rate.
- Continuity: 90% answer present, median 0 ms join.
- Gate: passed. This is the voicing layer.

## Phase 1 — pure-audio duplex pilot (current)

No control tokens, no mode head, no text path. `A_t` is always a codec frame;
silence frames mean listening.

1. Data: 20 Open Yap conversations, ~7.2 h.
   - Agent side re-voiced to Vani with original timing.
   - User side untouched.
   - 80 ms frames; 12 codebooks per agent frame.
   - 1,605 five-second windows: boundary and agent-rich.
2. Training objective: predict agent frames from user frames and previous agent
   frames; loss on `A` only; speech weight 1.0, silence 0.3.
3. Run the curriculum/oversampling experiments and record every result in
   [experiments.md](experiments.md) and [duplex-log.md](duplex-log.md).
4. Evaluate with the fair probe, not one window:
   - 20+ user-speaking windows,
   - 20+ agent-speaking windows,
   - 20+ boundary windows,
   - held-out conversations; distributions and tolerant bands.
5. Decide scale only after:
   - a tiny overfit test represents both behaviours,
   - held-out metrics improve with more data (10/25/50/100% curve),
   - the oracle turn-state diagnostic shows the model can voice correctly when
     the turn is given,
   - the free-running/teacher-forced gap shrinks.

Gate: the model stays quiet while the user speaks, speaks on its turn, and the
speech is varied enough to listen to. Injection and seam are tested after the
switch works.

If the switch fails:
- overfit test first; if it fails, the objective/architecture is wrong,
- if overfit passes but held-out is flat, data composition/volume is the blocker,
- do not add control tokens or a mode head to hide it.

## Phase 2 — scale (only after Gate 1 evidence)

- Re-voice more conversations in chunks.
- Keep the speech/silence balance that worked in the pilot.
- Train on the 8 GB box in chunks, or rent 24–48 GB for the full corpus.
- Full duplex eval: pause takeover, interruption takeover, backchannel resume,
  verbatim injection, voice similarity.

## Phase 3 — runtime

- VAD/controller, injection queue, block-boundary cut, cancel.
- Wire the paid model + knowledge base.
- Measure end-to-end latency on the target box.

## Todo

- [x] mouth adapter
- [x] continuity injection
- [x] pure-audio data format and cache
- [x] re-voice all 20 pilot conversations
- [x] teacher-forced baseline
- [x] EOS/exposure diagnostics
- [x] speech-only and listen-only curricula
- [x] speech oversampling run
- [ ] fair multi-window probe
- [ ] overfit / oracle / scaling-curve diagnostics
- [ ] injection and seam on the working adapter
- [ ] scale decision
- [ ] runtime controller and paid-harness integration
