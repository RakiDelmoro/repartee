# End goal

Vani is a 2B speech model that holds a natural live conversation and speaks
answers supplied by a harness, in one voice.

## Two behaviours, one model

1. **Duplex conversation.** Speech in, speech out. Vani listens while speaking,
   waits, backchannels, and takes turns. No user text input.
2. **Injected answers.** When the harness needs reasoning, it sends text
   mid-stream. Vani speaks it as a continuation of what it was already saying.

The injected answer must not sound like a different person or a new recording.

## Split of responsibility

| Layer | Owns |
|---|---|
| **Vani** | listening, turn-taking, backchannels, voice, prosody, speaking supplied text |
| **Harness** | facts, knowledge graph, reasoning, when to inject |
| **Paid model** | reasoning that does not fit locally |
| **Judge ASR** | independent verification of what Vani said |

Vani does not decide facts. If an answer needs reasoning, the harness gets it and
hands Vani the words.

## Success criteria

- Same voice for filler, backchannel, and injected answer.
- Waits while the user speaks; takes the floor when the user stops.
- Stops on interruption.
- Injected answer is a continuation, not a restart.
- Injected words are exact.
- Runs faster than real time on the target hardware.

## Not in scope

- Facts invented by Vani.
- Voice cloning from Open Yap speakers.
- Multi-party audio.
- Model-owned chain-of-thought.
