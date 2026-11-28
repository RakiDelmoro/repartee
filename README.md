# Repartee

USB CDC receiver for the ESP32 conversation recorder. It reads the firmware's
START/AUDIO/END records from the board's USB serial console and writes one 16-bit
PCM WAV per utterance into `data/recordings/`. The bigger-device stage then
transcribes a finished WAV with [LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B).
The firmware is unchanged (it lives in
[`RakiDelmoro/conversation-recorder`](https://github.com/RakiDelmoro/conversation-recorder)).

The 2.17 B model does **not** run on the ESP32: 4.3 GB of fp16 weights versus
8 MB of PSRAM. The device keeps the low-power job — VAD and recording — and the
bigger device runs the model.

## Storage

| file | meaning |
|---|---|
| `data/recordings/YYYYMMDD-HHMMSS-<device>.wav` | one utterance |

The WAV appears at START and is playable while it grows: its RIFF/data sizes are
refreshed every 5 s and fixed on END, so you can listen to a recording before the
device closes the session.

`<device>` is the board id from START, sanitised to `[A-Za-z0-9]`; the timestamp
is the receiver clock at START. Filenames are monotonic within a second, so two
utterances never collide, and an existing file is never overwritten.

## Wire protocol (v1, little-endian)

| record | struct | size | fields |
|---|---|---|---|
| START `1` | `<IIIHHQQ16s` | 48 B | session_id, first_sample, rate, channels, bits, start_unix_ms, uptime_ms, device_id[16] |
| AUDIO `2` | raw PCM | ≤ 6400 B | 16-bit samples as declared in START |
| END `3` | `<IIB3xI` | 16 B | session_id, total_samples, reason, crc |
| ACK `0x81` | `<II` | 8 B | session_id, acked_samples (+ final flag) |

Every AUDIO record is ACKed with the running sample count so the firmware can free
its queue; after a link drop the device sends START with `first_sample` and the
receiver resumes at that offset. The ESP-IDF console shares the port, so the
framer scans for `CREC` and skips log text.

## Run

1. **Attach the board to the Docker host.** Windows, PowerShell as Administrator:

   ```powershell
   winget install usbipd
   usbipd list                     # note the board's BUSID, e.g. 2-1
   usbipd bind --busid 2-1         # once per device
   usbipd attach --wsl --busid 2-1 # after every replug; --auto-attach keeps it bound
   ```

2. **Check the container sees the node.** It must exist before the container is
   created, because `runArgs` maps it in:

   ```bash
   ls -l /dev/ttyACM0
   ```

   Docker Desktop's engine runs in its own VM and cannot see a device attached to
   the WSL distro. Then either run Docker Engine inside the WSL distro (so
   `--device` works), or keep Desktop and run the receiver on the host:
   `repartee receiver --serial /dev/ttyACM0` in WSL, or `--serial COM3` on Windows
   (needs `pip install pyserial`).

3. **Install once inside the container.** No rebuild needed; the image ships the
   Python toolchain, and an older container gets it from the first line:

   ```bash
   command -v python3 >/dev/null || { apt-get update && apt-get install -y --no-install-recommends python3 python3-venv; }
   test -x .venv/bin/python || python3 -m venv .venv
   .venv/bin/pip install -e '.[receiver]'
   ```

4. **Free the port.** Close `espflash monitor`; only one process may own the tty:

   ```bash
   fuser -v /dev/ttyACM0
   ```

5. **Start the receiver:**

   ```bash
   .venv/bin/repartee receiver            # or: .venv/bin/python -m app receiver
   # logs: "serial: /dev/ttyACM0 open" then one line per finalized recording
   ```

   A foreground run dies with its terminal. To keep it alive:
   `mkdir -p data && nohup .venv/bin/repartee receiver > data/receiver.log 2>&1 &`.

6. **Verify:** speak, then `ls -l data/recordings` — each utterance shows up as
   one `*.wav`, playable immediately and finalized on END. Unplug/replug needs no restart: the
   receiver reopens the port and the firmware resumes from the last ACK.

## Bigger device: speech to text

```bash
.venv/bin/pip install -e '.[bigger]'

# A local checkpoint avoids the 4.3 GB Hugging Face download.
REPARTEE_STT_MODEL=/root/.cache/repartee/LFM3-Omni-2B \
  .venv/bin/repartee transcribe data/recordings/20261128-101500-rep01.wav

# JSON with chunk boundaries and timings; --out writes a file.
REPARTEE_STT_MODEL=/root/.cache/repartee/LFM3-Omni-2B \
  .venv/bin/repartee transcribe recording.wav --json --out recording.transcript.json
```

The WAV is downmixed and resampled to 16 kHz mono. It is split into chunks of at
most 60 s; each boundary is moved back to the last pause of at least 250 ms in
the final 20 % of the window, with a hard cut at 60 s when the speech never
pauses. The model's encoder is quadratic in audio length, so 60 s is the ceiling
that fits an 8 GB GPU. An empty result is retried once as two half chunks.

Measured on a Quadro RTX 4000 (8 GB), fp16: 6.6 s to load the model and ~30 s to
generate the transcript of a 60 s clip when the host is idle. The same run took
~100 s while a large filesystem delete was competing for CPU and disk, so treat
the idle number as the machine's ceiling. ASR is `Perform ASR in English.` with
`generation_mode="text"`; the model reads about 13 audio tokens per second.

The test sample is `samples/open-yap-1k/conv_d4005da6db98-60s.wav` — 60 s of a
two-speaker Open Yap 1K conversation (`conv_d4005da6db98`, start 75 s), 16 kHz
mono, CC-BY-4.0. Its `.json` sidecar records the source and excerpt offset.
`samples/` is gitignored; the sidecar is the provenance record.

## Settings

| variable | default | meaning |
|---|---|---|
| `REPARTEE_SERIAL` | `/dev/ttyACM0` | USB CDC device (`COM3` on Windows) |
| `REPARTEE_DATA` | `<cwd>/data` | parent of the recordings folder |
| `REPARTEE_RECORDINGS` | `<data>/recordings` | where WAVs are written |
| `REPARTEE_STT_MODEL` | `LiquidAI/LFM3-Omni-2B` | HF id or local checkpoint |
| `REPARTEE_STT_LANGUAGE` | `English` | ASR language name |
| `REPARTEE_STT_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `REPARTEE_STT_CHUNK_SECONDS` | `60` | maximum audio per forward pass |
| `REPARTEE_STT_MAX_NEW_TOKENS` | `1024` | generation cap per chunk |

`--serial` and `--out` override the receiver settings; `--model`,
`--language`, `--chunk-seconds` and `--max-new-tokens` override the STT ones.

## Tests

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[receiver]'
.venv/bin/python -m pytest
```

The suite drives the protocol over a POSIX pty: ACKs, console-text resync,
duplicate START, link-drop resume, and retry when the device is missing. The STT
tests cover pause-aware chunking, the 60 s hard cut, downmix/resample, and
`<think>` removal; they do not load the 4.3 GB model.

With the model installed, run the end-to-end smoke:

```bash
REPARTEE_STT_MODEL=/root/.cache/repartee/LFM3-Omni-2B \
  .venv/bin/repartee transcribe samples/open-yap-1k/conv_d4005da6db98-60s.wav
```
