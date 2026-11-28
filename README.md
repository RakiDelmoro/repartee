# Repartee

ESP32 recorder → USB → receiver → `data/recordings/` → LFM3-Omni worker →
`knowledge-base/`.

The receiver and worker are **separate processes**. The receiver writes a
`.done` marker when a WAV is complete; the worker polls that marker, runs STT and
summary, then writes one Markdown file.

The model is 2.17 B parameters. It runs on the bigger device, not on the ESP32.

## 0. Attach the board

Windows PowerShell as Administrator:

```powershell
winget install usbipd
usbipd list
usbipd bind --busid <BUSID>       # once per device
usbipd attach --wsl --busid <BUSID>
```

The board is the Espressif `303a:1001` device, not the FTDI `0403:6014`
converter.

Check it in WSL:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc 'ls -l /dev/ttyACM0'
```

Now check inside the container:

```bash
ls -l /dev/ttyACM0
```

### If the container was already running

`usbipd attach` changes WSL only. The container's `/dev` was fixed when the
container was created, so a new host device does not appear by itself. In VS
Code:

```text
Ctrl+Shift+P → Dev Containers: Rebuild Container
```

`Rebuild Container` recreates it with the `--device=/dev/ttyACM0` mapping.
A simple container **restart is not enough**; `--device` is applied only at
creation. You only need this when `/dev/ttyACM0` is missing inside the
container, not after every replug.

If it is still missing after the rebuild, the engine is Docker Desktop, whose
VM cannot see a device attached to WSL. Use Docker Engine inside WSL, or run
the receiver on the WSL host.

Re-attach after every board replug.

## Quick start — this dev container

```bash
cd /workspaces/repartee

# One time: venv, Python packages, model in models/LFM3-Omni-2B.
./scripts/setup.sh

# Every time you want the recorder + worker running.
./scripts/start.sh
```

Check:

```bash
./scripts/status.sh
```

Stop:

```bash
./scripts/stop.sh
```

`setup.sh` copies the checkpoint from `/root/.cache/repartee/LFM3-Omni-2B`
when present, otherwise downloads it. It ends up in the workspace at
`models/LFM3-Omni-2B`, which is gitignored but visible in the file explorer.

The active model is this workspace copy. If `setup.sh` copied from
`/root/.cache`, that root copy is only a source and can be removed.

`start.sh` runs the supervisor in the background. It survives closing the
terminal, but not the container stopping. The supervisor restarts either child
after 5 s.

Host systemd or Docker needs its own host-visible model copy; see below.
If you are not in this container, use the systemd or Docker steps.

## Run on Ubuntu/WSL with systemd

Use the WSL/Ubuntu host, not this container. This container has no systemd.

### Install once

```bash
# 1. Service account and directories.
sudo useradd --system --user-group --home /var/lib/repartee --shell /usr/sbin/nologin repartee
sudo usermod -aG dialout,video,render repartee
sudo mkdir -p /var/lib/repartee /etc/repartee

# 2. Get the code into /opt/repartee.
sudo git clone https://github.com/RakiDelmoro/repartee.git /opt/repartee
cd /opt/repartee

# 3. Install the app.
python3 -m venv .venv
.venv/bin/pip install -e '.[bigger]'

# 4. Put a host-visible copy of the model in place.
sudo mkdir -p /opt/repartee/models
sudo cp -a /path/to/LFM3-Omni-2B /opt/repartee/models/
sudo chown -R repartee:repartee /opt/repartee /var/lib/repartee

# 5. Configuration and units.
sudo cp deploy/systemd/repartee.env.example /etc/repartee/repartee.env
sudoedit /etc/repartee/repartee.env
sudo cp deploy/systemd/*.service deploy/systemd/*.target /etc/systemd/system/
sudo systemctl daemon-reload
```

`/etc/repartee/repartee.env` must contain:

```bash
REPARTEE_DATA=/var/lib/repartee/data
REPARTEE_KNOWLEDGE_BASE=/var/lib/repartee/knowledge-base
REPARTEE_STT_MODEL=/opt/repartee/models/LFM3-Omni-2B
REPARTEE_STT_DEVICE=cuda
```

### Start

```bash
sudo systemctl enable --now repartee.target
systemctl status repartee.target
sudo journalctl -u repartee-receiver -u repartee-worker -f
```

### Stop

```bash
sudo systemctl stop repartee.target
```

`repartee.target` starts both units. Each has `Restart=always`. This survives a
reboot.

Requirements: systemd enabled in WSL, the board attached to the WSL host, and
the `dialout`/GPU groups available. The model path must be readable by
`repartee`.

## Run with Docker on the host

Use the host, not this container. This container has no `docker` command.

### Install once

```bash
cd /path/to/repartee
mkdir -p data knowledge-base models

# Host-visible copy of the checkpoint.
cp -a /path/to/LFM3-Omni-2B models/

# $PWD is the repo directory from the cd above; Compose needs an absolute path.
export REPARTEE_MODEL_DIR="$PWD/models/LFM3-Omni-2B"
```

### Start

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f worker
```

### Stop

```bash
docker compose down
```

Two services: `receiver` owns `/dev/ttyACM0`; `worker` owns the GPU and writes
`knowledge-base/`. Both use `restart: unless-stopped`.

Requirements: Docker Compose v2, NVIDIA Container Toolkit, and the USB device
visible to the host. If your Compose rejects the GPU reservation, replace the
`deploy:` block in `compose.yaml` with `gpus: all`.

## Use it without a supervisor

Process one file:

```bash
REPARTEE_STT_MODEL="$PWD/models/LFM3-Omni-2B" \
  .venv/bin/repartee process data/recordings/20261128-101500-rep01.wav
```

Transcribe only:

```bash
REPARTEE_STT_MODEL="$PWD/models/LFM3-Omni-2B" \
  .venv/bin/repartee transcribe data/recordings/20261128-101500-rep01.wav --json
```

Query the folder:

```bash
.venv/bin/repartee conversations
.venv/bin/repartee search "bank"
.venv/bin/repartee show 20261128-101500-rep01
```

`show` takes the id without `.wav`. Use `--knowledge-base PATH` for another
folder. The model path above is the workspace copy made by `setup.sh`.

## Paths

| path | meaning |
|---|---|
| `data/recordings/<id>.wav` | one finished utterance |
| `data/recordings/<id>.wav.done` | complete; the worker claims only these |
| `data/recordings/<id>.wav.error` | failed job; retry with `process --force` |
| `knowledge-base/YYYY-MM-DD/<id>.md` | summary + transcript + metadata |
| `models/LFM3-Omni-2B` | local checkpoint used by the dev scripts |
| `scripts/setup.sh` | install deps + copy/download the model |
| `scripts/start.sh` / `stop.sh` / `status.sh` | dev quick start |
| `scripts/supervise.sh` | dev supervisor for receiver + worker |
| `deploy/systemd/` | systemd units and env example |
| `compose.yaml` / `Dockerfile` | two-service Docker stack |

Audio stays in `data/recordings`; the Markdown entry references it.
`knowledge-base/` is gitignored.

## Troubleshooting

| symptom | check |
|---|---|
| no KB entries | `pgrep -af repartee`; both must run |
| worker down | `.done` files wait; restarting the worker drains them |
| `.error` file | `repartee process <wav> --force` |
| no `.done` file | receiver not finalizing; check its log |
| `systemctl` says not booted | you are in this container; use the quick start |
| `docker: command not found` | run Docker on the host |
| worker cannot find model | set `REPARTEE_STT_MODEL` to a readable path |
| serial permission denied | add the user to `dialout` |
| GPU not found | pass `--gpus all` / check NVIDIA toolkit |

Only one process may own `/dev/ttyACM0`. Do not run two receivers. Do not run
two workers against the same GPU.

## Tests

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[receiver,test]'
.venv/bin/python -m pytest
```

The `test` extra includes pytest, numpy, soundfile, and librosa. The model is
not loaded by the tests.

## Settings

| variable | default | meaning |
|---|---|---|
| `REPARTEE_SERIAL` | `/dev/ttyACM0` | USB CDC device |
| `REPARTEE_DATA` | `<cwd>/data` | parent of recordings |
| `REPARTEE_RECORDINGS` | `<data>/recordings` | recordings folder |
| `REPARTEE_KNOWLEDGE_BASE` | `<cwd>/knowledge-base` | Markdown folder |
| `REPARTEE_STT_MODEL` | `LiquidAI/LFM3-Omni-2B` | HF id or local checkpoint |
| `REPARTEE_STT_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `REPARTEE_STT_LANGUAGE` | `English` | ASR language |
| `REPARTEE_SUMMARY` | `1` | `0` stores transcripts without summaries |
| `REPARTEE_WORKER_POLL` | `2` | seconds between polls |

Commands: `receiver`, `transcribe`, `process`, `worker`, `conversations`,
`search`, `show`.

## Protocol (receiver)

| record | struct | size |
|---|---|---|
| START `1` | `<IIIHHQQ16s` | 48 B |
| AUDIO `2` | raw PCM | ≤ 6400 B |
| END `3` | `<IIB3xI` | 16 B |
| ACK `0x81` | `<II` | 8 B |

Every AUDIO record is ACKed. The framer scans for `CREC` and skips ESP-IDF log
text. See `app/receiver.py` for the fields.
