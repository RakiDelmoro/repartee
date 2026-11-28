#!/usr/bin/env bash
# Stop the supervisor and its receiver/worker children.
set -euo pipefail

cd "$(dirname "$0")/.."

pidfile="data/supervise.pid"
if [ -f "$pidfile" ]; then
    kill -TERM "$(cat "$pidfile")" 2>/dev/null || true
    rm -f "$pidfile"
fi

pkill -TERM -f 'scripts/supervise.sh' 2>/dev/null || true
sleep 1
pkill -TERM -f 'repartee (receiver|worker)' 2>/dev/null || true

if pgrep -af 'repartee (receiver|worker)|scripts/supervise.sh'; then
    echo "some processes still running; check manually" >&2
    exit 1
fi

echo "stopped"
