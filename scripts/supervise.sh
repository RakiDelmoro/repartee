#!/usr/bin/env bash
# Development supervisor: keep the receiver and worker alive inside the container.
# It restarts either child after 5 s. It does NOT survive a container stop or a
# machine reboot; use a Docker restart policy or a systemd unit for that.
set -u
set -m

cd "$(dirname "$0")/.."
mkdir -p data

export REPARTEE_STT_MODEL="${REPARTEE_STT_MODEL:-LiquidAI/LFM3-Omni-2B}"

pids=()

cleanup() {
    trap - INT TERM EXIT
    for pid in "${pids[@]}"; do
        kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null
    done
    wait
}
trap cleanup INT TERM EXIT

supervise() {
    local name="$1"
    shift
    while true; do
        echo "$(date -Is) starting ${name}" >>"data/${name}.log"
        "$@" >>"data/${name}.log" 2>&1
        status=$?
        echo "$(date -Is) ${name} exited (${status}); restarting in 5 s" >>"data/${name}.log"
        sleep 5
    done
}

supervise receiver .venv/bin/repartee receiver &
pids+=("$!")

supervise worker .venv/bin/repartee worker &
pids+=("$!")

wait
