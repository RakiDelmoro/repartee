#!/usr/bin/env bash
# Show whether the receiver and worker are running.
set -euo pipefail

cd "$(dirname "$0")/.."

if pgrep -af 'repartee (receiver|worker)|scripts/supervise.sh'; then
    :
else
    echo "not running"
fi

echo "--- worker log"
tail -n 10 data/worker.log 2>/dev/null || true
