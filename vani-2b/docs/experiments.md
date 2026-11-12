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
| E007 | 2026-11-10 | Extended agent re-voicing | 11 conversations, ~43 min | done, one voice | pass |
| E008 | 2026-11-10 | Policy head + VAD features, joint duplex audio | 129 turn-boundary windows | user 0/63 speak; agent 8–37/63 speak; injection 2/2 exact | **timing pass, content fail** |
| E009 | 2026-11-11 | Pure audio teacher-forced | 1,605 windows, 8 epochs | loss 2.866 → 2.067; inference mute, EOS at frame 0 | fail duplex |
| E010 | 2026-11-11 | EOS penalty on speech frames | same data, 2 epochs | loss identical to E009 | no effect |
| E011 | 2026-11-11 | EOS masked from training logits | same data, 2 epochs | loss identical to E009 | no effect |
| E012 | 2026-11-11 | EOS masked only at inference | held-out user/agent windows | 0 frames unmasked; 23 speech frames on user window masked | fail, over-talk |
| E013 | 2026-11-11 | Scheduled sampling 10–35% | 1,605 windows, 6 epochs | loss 3.177 → 3.186; agent 0 speech | fail, mute |
| E014 | 2026-11-11 | Speech-only curriculum | 365 speech windows, EOS masked, 6 epochs | loss 4.118 → 3.320; agent 85 speech frames; repetitive | speaking pass, content fail |
| E015 | 2026-11-11 | Listen curriculum | 408 windows, EOS allowed, 3 epochs | loss 2.895 → 2.704; both probe windows 0 speech | fail, mute |
| E016 | 2026-11-11 | All-window EOS-masked | 1,605 windows, 4 epochs | loss 2.566 → 2.295; user 5, agent 0 speech | fail, speech lost |
| E017 | 2026-11-12 | Speech oversampling | 2,700 samples, EOS masked, 4 epochs | epoch 2 loss 2.266; probe pending | running |

## Interpretation

- The mouth and continuity are real, measured, and reproducible.
- The hard switch proves the injection mechanism; continuity proves the natural join.
- The joint duplex failure is not an injection failure. The model never chooses to speak.
- E006 isolates the cause: fixing the silence-labelled-as-speech bug did not make the
  model speak, so the remaining gap is scale and turn coverage.
- E008 isolates the next gap: the duplex **policy** now listens and takes the floor on
  held-out windows, but audio-only generation loops or produces a fragment. The model
  needs a text plan in the agent slot, not just audio codes.
- E009–E013 show the EOS/exposure problem: teacher-forced loss is not a behaviour metric.
  The model can be mute despite a falling loss.
- E014 proves the model can speak; E015–E016 prove silence training can erase speech.
- E017 tests whether speech oversampling keeps both behaviours. No verdict yet.
- Fair evaluation is now multi-window and distribution-based, not one window
  (`duplex_audio_multi_eval.py`).
