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


def ensure_dirs() -> None:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
