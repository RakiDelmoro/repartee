# Room device (software listener built, firmware planned)

Hardware has not arrived yet, but the segmentation decision now has a working
reference implementation on the host: `app/listener.py` (see below). The firmware
section is still a plan.

## Hardware

| Part | Role |
|---|---|
| reSpeaker XVF3800 4-mic array | 4 PDM mics, XMOS DSP: beamforming, AEC, noise suppression, DoG/DoA, VAD |
| XIAO ESP32-S3 (on the array board) | capture, encode, upload |
| I2S firmware on the XVF3800 | 16 kHz or 48 kHz, 2 channels, 32-bit; the array is **not** a USB device in this mode |
| I2C | control: speech energy per beam, LEDs, gain, mute state |

No microSD. Memory is 8 MB PSRAM.

## Memory budget

| Format | 60 s chunk | Slots in ~7 MB usable |
|---|---|---|
| WAV 16 kHz mono 16-bit | 1.92 MB | 3 (~3 min) |
| Opus 24 kbps | 180 KB | ~38 (~38 min) |
| Opus 16 kbps | 120 KB | ~58 (~58 min) |

Opus encoding is therefore **required**, not an optimisation: raw WAV only buys a
3-minute safety net.

## Segmentation: why not cut at exactly 60 s

The recording is one continuous stream; nothing in the audio marks word
boundaries. The cut point is a timestamp the firmware chooses.

Each chunk is transcribed independently, so the ASR never sees the chunk before
it. Cutting mid-word therefore produces:

```
chunk 1: "...the proposal is that we sh"   -> "the proposal is that we"
chunk 2: "ould probably delay the launch"  -> "old probably delay the launch"
```

The fragment at the start of chunk 2 has to be guessed. Cutting during a pause
means neither side of the seam is a fragment.

**Rule:** close the chunk at the first speech-energy dip after ~50 s; hard cap at
60 s.

Speech energy comes from the XVF3800 over I2C, so no extra VAD model is needed on
the device.

## Implemented reference listener (host)

`app/listener.py` is the software form of the device rule: pure-energy VAD on
30 ms frames, no model, no audio-device library. It reads 16 kHz mono s16le PCM
from any command (`arecord`, `parecord`, `ffmpeg`; `REPARTEE_LISTENER_CMD`) and
writes finished recordings atomically to the spool with the device filename
`YYYYMMDD-HHMMSS-device.wav`. Two recordings in the same second cannot collide:
the stamp is forced monotonic, because the ingest regex accepts exactly six time
digits.

```
.venv/bin/python -m app listen --device room1 --stop-silence 60 --preroll 1.0
.venv/bin/python -m app listen --input samples/two-speakers-interview-clean.wav \
    --spool /tmp/spool --stop-silence 0.8 --device room1      # replay a file
```

| knob | default | effect |
|---|---|---|
| `--threshold` | 0.012 RMS | frame counts as speech above max(threshold, 3x noise floor) |
| `--min-speech-ms` | 90 | speech this long starts a recording |
| `--stop-silence` | 60 s | silence this long closes it (the requirement's "1 minute") |
| `--preroll` | 1.0 s | audio kept before the first speech frame, so the first word is not clipped |
| `--max-record` | 1800 s | hard cap so a stuck stream cannot fill the disk |

Segments shorter than 1 s are discarded as noise. Tests cover the state machine,
the noise-floor behaviour, and the spool filename contract (`tests/test_listener.py`).

## Transport: stream frames, segment on the server

Streaming does not reduce latency below the segment length - the ASR still needs
60 s of audio before it can transcribe. What streaming buys:

- no 180 KB burst every 60 s, just ~3 kB/s
- a dropped connection costs a few frames, not a whole chunk
- the same socket can carry playback later (duplex)

```
device:  [20ms][20ms][20ms] ...  over one WebSocket, seq per frame
server:  buffers frames, acks the last contiguous seq
server:  closes a block at a pause or 60 s, transcribes, appends to session
device:  keeps the last 60 s of frames; on reconnect, resends from last ack
```

Keeping the last 60 s in RAM costs 180 KB (Opus), versus 1.9 MB if the device had
to hold a whole WAV chunk for retry.

## Firmware loop

```
record 60 s (cut at a pause after 50 s)
  -> Opus encode (~1 s of CPU, 5-10 % of one core while running)
  -> push to ring buffer (state: queued)
  -> uploader thread sends oldest queued slot
  -> on 200 OK: free the slot
  -> on failure: backoff, keep queued, resend from last acked frame
```

Two buffers minimum, four comfortable, eight if the budget allows.

## Failure modes

| Failure | Consequence | Mitigation |
|---|---|---|
| Wi-Fi down | chunks accumulate, ring fills | resend from ring; if full, drop oldest and record a sequence gap |
| Power cut | un-uploaded buffer lost | ack fast; keep the ring small |
| Server down | same as Wi-Fi down | retry with backoff, never delete on failure |
| Router reboot | a few seconds of frames lost | resend from last acked sequence |
| Mic muted | nothing captured | mute button is hardware-side; surface it in the stream |

## Open items

- Confirm which I2S port carries playback into the array (needed only for the
  future duplex mode).
- Decide ring size after measuring real Wi-Fi reliability at the install point.
- Validate far-field transcription quality on the real device before tuning VAD
  thresholds.
