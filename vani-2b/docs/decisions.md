# Decision log

Append-only. Status values: accepted, superseded, rejected.

| ID | Date | Decision | Why | Status |
|---|---|---|---|---|
| D001 | 2026-11-10 | Base model LFM3-Audio-2B | 2.17B fits 8 GB; speech in/out; interleaved text+speech; 32K context | accepted |
| D002 | 2026-11-10 | LoRA only; freeze encoder, projector, codec, depthformer | Fits memory; preserves voice; the mouth is a small adaptation | accepted |
| D003 | 2026-11-10 | Mouth adapter trained on the model’s own TTS | One voice; no external voice licence; 722 clauses produced 92.5% exact | accepted |
| D004 | 2026-11-10 | Hard switch first: force text, close, run speech tail | Fastest path to the injected-answer behaviour | accepted, then improved by D005 |
| D005 | 2026-11-10 | Continuity v2: render lead + answer as one utterance; cut at the join | Hard switch sounded like two takes; continuity inherits the lead’s prosody | accepted |
| D006 | 2026-11-10 | Re-voice the Open Yap agent with the product TTS | One voice across filler, backchannel, injected answer; human user audio stays | accepted |
| D007 | 2026-11-10 | One joint run, initialized from mouth + continuity, 50/30/20 mix, pilot first | Retention is built in; no separate forgetting window; pilot controls cost | accepted |
| D008 | 2026-11-10 | Defer role-context token | User prefers data scale first; role context remains an option if data alone fails | accepted |
| D009 | 2026-11-10 | Derive duplex “speak” labels from re-voiced audio energy | Original transcript labels marked silent TTS frames as speech (30–40%) | accepted |
| D010 | 2026-11-10 | Qwen3-ASR is judge only; runtime STT uses LFM3 | Self-ASR bias; one runtime model; independent evaluation | accepted |
| D011 | 2026-11-10 | Name: Vani; folder `vani-2b/` | Human name meaning voice/speech; fits the mouth role | accepted |
| D012 | 2026-11-10 | Training code under `vani-2b/train` | One home per model; lean repo | accepted |
| D013 | 2026-11-10 | Do not scale to 1,000 h until the five checks pass | Duplex pilot failed; scaling now would multiply a broken behaviour | accepted |
| D014 | 2026-11-10 | Open Yap DUA: commercial training allowed; no redistribution, no voice clones | Licence terms; product voice comes from the TTS teacher | accepted |
| D015 | 2026-11-10 | Local 8 GB for pilots; rented 24–48 GB for full training | 5.43 GB for 5 s duplex windows; full corpus is compute-bound | accepted |
| D016 | 2026-11-11 | Duplex is pure audio in / audio out; no control tokens, no mode head | The model must learn turn-taking and silence from conversation audio itself; tokens are a text shortcut | accepted, supersedes the token/head pilot |
| D017 | 2026-11-12 | EOS is the codec's end-of-audio code, not a listen token | Teacher forcing hides the model's own EOS; masking it forces speech everywhere | accepted |
| D018 | 2026-11-12 | A single window is a smoke test, never a verdict | 2B model, 7.2 h data, one window is sampling noise; agent-rich windows may start mid-turn | accepted |
| D019 | 2026-11-12 | Scale only after overfit + held-out + data-scaling curve + oracle diagnostics | Prevents multiplying a collapsed objective; separates capability from data volume | accepted |
| D020 | 2026-11-12 | Voice system prompt is separate from duplex training | Duplex never sees text; the TTS prompt conditions the voice path only | accepted |
| D021 | 2026-11-12 | Record every duplex decision, experiment, and plan in `duplex-log.md` | The user needs a readable trail during and after the project | accepted |
| D022 | 2026-11-12 | Switch the base model from LFM3-Omni-2B to LFM3-Audio-2B | Audio-native encoder/codec; no vision path; streaming voice chat; closer to Vani than Omni | accepted |
| D023 | 2026-11-12 | Rebuild the duplex cache with Audio-2B’s encoder/codec | Omni cache embeddings/codes are base-specific; old cache kept as `cache_turns_omni` | accepted |
| D024 | 2026-11-12 | Reuse the mouth/switch/continuity LoRAs if they load on Audio-2B | Adapter transfer passed; avoids retraining the voice layer | accepted, voice sample pending |
