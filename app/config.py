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
# Summary after transcription (same LFM3 model as STT).
SUMMARY_ENABLED = os.environ.get("REPARTEE_SUMMARY", "1") == "1"

ASR_MODEL_DIR = Path(os.environ.get("REPARTEE_MODEL_DIR", HOME / "models"))
ASR_MODEL_NAME = os.environ.get("REPARTEE_ASR_MODEL_NAME", "Qwen3-ASR-0.6B-hf")
ASR_MODEL_LOCAL = ASR_MODEL_DIR / ASR_MODEL_NAME
ASR_MODEL = os.environ.get("REPARTEE_ASR_MODEL") or (
    str(ASR_MODEL_LOCAL) if ASR_MODEL_LOCAL.exists() else f"Qwen/{ASR_MODEL_NAME}"
)
ASR_LANGUAGE = os.environ.get("REPARTEE_ASR_LANGUAGE") or None
# Runtime STT model (one family with the device listener): LFM3-Audio-2B.
ASR_ENGINE = os.environ.get("REPARTEE_ASR_ENGINE", "lfm3")
ASR_LFM3_MODEL = os.environ.get(
    "REPARTEE_LFM3_MODEL", str(ASR_MODEL_DIR / "LFM3-Audio-2B-hf")
)
ASR_MAX_NEW_TOKENS = int(os.environ.get("REPARTEE_ASR_MAX_NEW_TOKENS", "4096"))

# When set, /ingest requires `Authorization: Bearer <token>`.
INGEST_TOKEN = os.environ.get("REPARTEE_INGEST_TOKEN") or None
MAX_UPLOAD_BYTES = int(float(os.environ.get("REPARTEE_MAX_UPLOAD_MB", "512")) * 1024 * 1024)

ALL_DIRS = (SPOOL_DIR, WORKING_DIR, DONE_DIR, FAILED_DIR)


def ensure_dirs() -> None:
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
