#!/usr/bin/env bash
# One-time dev-container setup: venv, Python dependencies, and the model folder.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data models

if [ ! -x .venv/bin/python ]; then
    echo "creating .venv"
    python3 -m venv .venv
fi

echo "installing Python dependencies"
.venv/bin/pip install -e '.[bigger]'

target="$PWD/models/LFM3-Omni-2B"

is_model() {
    local dir="$1" file
    for file in model.safetensors config.json tokenizer.json \
                audio_tokenizer/model.safetensors modeling_lfm3_omni.py; do
        [ -f "$dir/$file" ] || return 1
    done
}

if is_model "$target"; then
    echo "model already present: $target"
    echo "next: ./scripts/start.sh"
    exit 0
fi

find_source() {
    local src
    for src in "${REPARTEE_MODEL_SOURCE:-}" \
               "/root/.cache/repartee/LFM3-Omni-2B" \
               "$HOME"/.cache/huggingface/hub/models--LiquidAI--LFM3-Omni-2B/snapshots/*; do
        [ -n "$src" ] || continue
        is_model "$src" || continue
        printf '%s\n' "$src"
        return 0
    done
    return 1
}

tmp="$PWD/models/.LFM3-Omni-2B.tmp.$$"
trap 'rm -rf "$tmp"' EXIT
rm -rf "$tmp"

if src="$(find_source)"; then
    echo "copying model from $src"
    cp -aL "$src" "$tmp"
else
    echo "no local checkpoint found; downloading from Hugging Face"
    .venv/bin/hf download LiquidAI/LFM3-Omni-2B --local-dir "$tmp"
fi

if ! is_model "$tmp"; then
    echo "incomplete model checkpoint" >&2
    exit 1
fi

if [ -e "$target" ]; then
    backup="$target.incomplete.$(date +%s)"
    echo "moving incomplete folder to $backup"
    mv "$target" "$backup"
fi
mv "$tmp" "$target"
trap - EXIT

echo "model ready: $target"
echo "next: ./scripts/start.sh"
