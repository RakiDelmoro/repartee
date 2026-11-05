"""Runtime configuration. Everything comes from the environment."""

from __future__ import annotations

import os
from pathlib import Path

HOME = Path(os.environ.get("REPARTEE_HOME", Path.cwd())).resolve()
DATA_DIR = Path(os.environ.get("REPARTEE_DATA", HOME / "data"))

SPOOL_DIR = DATA_DIR / "spool"
WORKING_DIR = DATA_DIR / "working"
DONE_DIR = DATA_DIR / "done"
FAILED_DIR = DATA_DIR / "failed"
DB_PATH = Path(os.environ.get("REPARTEE_DB", DATA_DIR / "repartee.db"))

ASR_MODEL_DIR = Path(os.environ.get("REPARTEE_MODEL_DIR", HOME / "model"))
ASR_MODEL_LOCAL = ASR_MODEL_DIR / "Qwen3-ASR-1.7B-hf"
ASR_MODEL = os.environ.get("REPARTEE_ASR_MODEL") or (
    str(ASR_MODEL_LOCAL) if ASR_MODEL_LOCAL.exists() else "Qwen/Qwen3-ASR-1.7B-hf"
)
ASR_LANGUAGE = os.environ.get("REPARTEE_ASR_LANGUAGE") or None
ASR_MAX_NEW_TOKENS = int(os.environ.get("REPARTEE_ASR_MAX_NEW_TOKENS", "4096"))

# When set, /ingest requires `Authorization: Bearer <token>`.
INGEST_TOKEN = os.environ.get("REPARTEE_INGEST_TOKEN") or None
MAX_UPLOAD_BYTES = int(float(os.environ.get("REPARTEE_MAX_UPLOAD_MB", "512")) * 1024 * 1024)

ALL_DIRS = (SPOOL_DIR, WORKING_DIR, DONE_DIR, FAILED_DIR)


def ensure_dirs() -> None:
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
