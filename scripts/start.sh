#!/usr/bin/env bash
# Start the receiver + worker supervisor in the background.
set -euo pipefail

cd "$(dirname "$0")/.."

model="$PWD/models/LFM3-Omni-2B"
if [ ! -f "$model/model.safetensors" ] || [ ! -f "$model/audio_tokenizer/model.safetensors" ]; then
    echo "model missing: $model" >&2
    echo "run ./scripts/setup.sh first" >&2
    exit 1
fi

mkdir -p data
pidfile="$PWD/data/supervise.pid"

if pgrep -f 'repartee (receiver|worker)|scripts/supervise.sh' >/dev/null; then
    echo "already running; run ./scripts/stop.sh first"
    echo "check: ./scripts/status.sh"
    exit 0
fi

if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "already running (pid $(cat "$pidfile"))"
    echo "check: ./scripts/status.sh"
    exit 0
fi

nohup env REPARTEE_STT_MODEL="$model" bash scripts/supervise.sh \
    > data/supervise.log 2>&1 &
echo "$!" > "$pidfile"

echo "started (pid $(cat "$pidfile"))"
echo "logs: data/receiver.log, data/worker.log"
echo "check: ./scripts/status.sh"
echo "stop:  ./scripts/stop.sh"
