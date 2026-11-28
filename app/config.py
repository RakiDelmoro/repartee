"""Runtime configuration. Everything comes from the environment."""

from __future__ import annotations

import os
from pathlib import Path

HOME = Path(os.environ.get("REPARTEE_HOME", Path.cwd())).resolve()
DATA_DIR = Path(os.environ.get("REPARTEE_DATA", HOME / "data"))

# Where finished WAVs are written; created on start.
RECORDINGS_DIR = Path(os.environ.get("REPARTEE_RECORDINGS", DATA_DIR / "recordings"))

# USB CDC device of the ESP32 board.
SERIAL_DEVICE = os.environ.get("REPARTEE_SERIAL", "/dev/ttyACM0")

# Bigger device: LFM3-Omni-2B speech to text. Point STT_MODEL at a local
# checkpoint to avoid a 4.3 GB download.
STT_MODEL = os.environ.get("REPARTEE_STT_MODEL", "LiquidAI/LFM3-Omni-2B")
STT_LANGUAGE = os.environ.get("REPARTEE_STT_LANGUAGE", "English")
STT_DEVICE = os.environ.get("REPARTEE_STT_DEVICE", "auto")
STT_CHUNK_SECONDS = float(os.environ.get("REPARTEE_STT_CHUNK_SECONDS", "60"))
STT_MAX_NEW_TOKENS = int(os.environ.get("REPARTEE_STT_MAX_NEW_TOKENS", "1024"))

# Bigger device: transcript -> summary -> one Markdown file per conversation.
KNOWLEDGE_BASE_DIR = Path(
    os.environ.get("REPARTEE_KNOWLEDGE_BASE", HOME / "knowledge-base")
)
SUMMARY_ENABLED = os.environ.get("REPARTEE_SUMMARY", "1").lower() not in {
    "0", "false", "no", "off",
}
WORKER_POLL_SECONDS = float(os.environ.get("REPARTEE_WORKER_POLL", "2"))


def ensure_dirs() -> None:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
