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

ASR_MODEL_DIR = Path(os.environ.get("REPARTEE_MODEL_DIR", HOME / "models"))
ASR_MODEL_NAME = os.environ.get("REPARTEE_ASR_MODEL_NAME", "Qwen3-ASR-0.6B-hf")
ASR_MODEL_LOCAL = ASR_MODEL_DIR / ASR_MODEL_NAME
ASR_MODEL = os.environ.get("REPARTEE_ASR_MODEL") or (
    str(ASR_MODEL_LOCAL) if ASR_MODEL_LOCAL.exists() else f"Qwen/{ASR_MODEL_NAME}"
)
ASR_LANGUAGE = os.environ.get("REPARTEE_ASR_LANGUAGE") or None
ASR_MAX_NEW_TOKENS = int(os.environ.get("REPARTEE_ASR_MAX_NEW_TOKENS", "4096"))

# When set, /ingest requires `Authorization: Bearer <token>`.
INGEST_TOKEN = os.environ.get("REPARTEE_INGEST_TOKEN") or None
MAX_UPLOAD_BYTES = int(float(os.environ.get("REPARTEE_MAX_UPLOAD_MB", "512")) * 1024 * 1024)

# OpenAI-compatible endpoint used by the knowledge-base extractor (llama.cpp, ollama, vLLM).
LLM_BASE_URL = os.environ.get("REPARTEE_LLM_BASE_URL", "http://127.0.0.1:8081/v1")
LLM_MODEL = os.environ.get("REPARTEE_LLM_MODEL", "Qwen3.5-9B")
LLM_API_KEY = os.environ.get("REPARTEE_LLM_API_KEY", "none")
LLM_TIMEOUT = float(os.environ.get("REPARTEE_LLM_TIMEOUT", "600"))

ALL_DIRS = (SPOOL_DIR, WORKING_DIR, DONE_DIR, FAILED_DIR)


def ensure_dirs() -> None:
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
