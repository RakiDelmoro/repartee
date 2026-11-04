# Repartee — textless duplex speech model

A full-duplex speech-to-speech model trained only on two-person conversation
audio. The acoustic tokenizer (LiquidCodec) is frozen; the duplex network is
trained from random initialisation.

```
user 16 kHz -> LiquidCodec encoder -> 12 codes @ 80 ms
                                            |
                                 dual-stream backbone
                       (two causal encoders + cross-attention decoders;
                        predicts BOTH next user frame and next agent frame)
                                            |
                        frozen depth decoder (MaskGIT, 12 codebooks)
                                            |
                     LiquidCodec decoder -> 24 kHz agent speech
```

## Layout

```
train/dual.py       dual-stream trainer (both-stream loss, checkpoints, dev eval)
train/dashboard.py  live dashboard (loss curves, checkpoints) on port 8080
train/sample.py     stereo sample from a checkpoint (L = user, R = model)
data/codes/         tokenized conversations (gitignored)
data/ckpt/          rotating checkpoints + metrics (gitignored)
models/LFM3-Omni-2B frozen base weights + codec (gitignored)
```

## Data

`data/codes/index.jsonl` lists tokenized conversations (`uint16`, shape
`frames x 12`, 12.5 Hz). Sources: otoSpeech (stereo L/R) and Open Yap (a/b
tracks), split 90/5/5 by conversation.

## Run

```bash
# train (dual-stream, cross-attention)
python -m train.dual

# dashboard
python -m train.dashboard dual 5000      # mode, total steps -> http://localhost:8080

# sample from the latest checkpoint
python -m train.sample
```

## What is measured

- form: overlap fraction, backchannel rate, dead air, barge-in latency
  (against the human tracks)
- content: ASR of the generated agent channel (reported, not gated)
- sanity: auxiliary user-stream loss (a learnable cross-stream mapping, unlike
  the agent content at this scale)
