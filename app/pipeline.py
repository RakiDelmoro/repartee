"""Spool worker: audio file -> transcript -> summary -> one knowledge-base row."""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from . import config, kb

log = logging.getLogger(__name__)

FILENAME_RE = re.compile(
    r"^(?P<date>\d{8})-(?P<time>\d{6})-(?P<device>[A-Za-z0-9]{1,32})"
    r"\.(?P<ext>wav|flac|ogg|opus|mp3|m4a)$"
)


def parse_filename(name: str) -> tuple[str, str | None]:
    """Return (device, ISO start time) parsed from the device's filename."""
    match = FILENAME_RE.match(name)
    if not match:
        stem = Path(name).stem
        return (stem[:32] or "unknown"), None
    date, clock = match["date"], match["time"]
    stamp = (
        f"{date[:4]}-{date[4:6]}-{date[6:]}"
        f"T{clock[:2]}:{clock[2:4]}:{clock[4:6]}"
    )
    return match["device"], stamp


def process(path: Path, transcriber) -> kb.Conversation:
    device, started_at = parse_filename(path.name)
    transcript = transcriber.transcribe(path)
    return kb.Conversation(
        id=path.name,
        device=device,
        started_at=started_at,
        duration_s=transcript.duration_s,
        language=transcript.language,
        transcript=transcript.text,
        audio_bytes=path.stat().st_size,
    )


def store_conversation(conn, conv: kb.Conversation, model) -> kb.Conversation:
    """Summarize, then write the single row. A summary failure keeps the transcript."""
    summary = ""
    if config.SUMMARY_ENABLED and hasattr(model, "summarize") and conv.transcript.strip():
        try:
            summary = (model.summarize(conv.transcript) or "").strip()
        except Exception:  # noqa: BLE001 - the transcript is what must not be lost
            log.exception("%s: summarization failed", conv.id)
    stored = replace(conv, summary=summary,
                     status="summarized" if summary else "transcribed")
    kb.upsert(conn, stored)
    return stored


class Worker(threading.Thread):
    """Claims files from spool/, transcribes them, and files them away."""

    def __init__(self, transcriber, poll_seconds: float = 0.5, conn=None) -> None:
        super().__init__(name="repartee-worker", daemon=True)
        self.transcriber = transcriber
        self.conn = conn
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._owns_conn = conn is None

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        if self.conn is None:
            self.conn = kb.connect(config.DB_PATH)
        kb.init(self.conn)
        backoff = self.poll_seconds
        try:
            while not self._stop.is_set():
                try:
                    self.drain(self.conn)
                    backoff = self.poll_seconds
                except Exception:  # noqa: BLE001 - keep the worker alive
                    # A mounted filesystem can return transient errors; back off
                    # instead of spinning on it once per poll.
                    log.exception("drain failed; backing off %.0fs", backoff)
                    backoff = min(backoff * 2, 30.0)
                self._stop.wait(backoff)
        finally:
            if self._owns_conn and self.conn is not None:
                self.conn.close()

    def drain(self, conn) -> int:
        processed = 0
        for name in sorted(os.listdir(config.SPOOL_DIR)):
            if name.endswith(".part") or self._stop.is_set():
                continue
            self.handle(name, conn)
            processed += 1
        return processed

    def handle(self, name: str, conn) -> None:
        claimed = config.WORKING_DIR / name
        try:
            os.replace(config.SPOOL_DIR / name, claimed)
        except FileNotFoundError:
            return  # another worker took it

        try:
            if kb.get(conn, name) is not None:
                log.info("%s already stored; skipping", name)
            else:
                conv = process(claimed, self.transcriber)
                stored = store_conversation(conn, conv, self.transcriber)
                log.info("%s: %.1fs audio, %d chars, summary %s", name,
                         conv.duration_s or 0.0, len(conv.transcript),
                         "stored" if stored.summary else "unavailable")
            os.replace(claimed, config.DONE_DIR / name)
        except Exception as exc:  # noqa: BLE001 - quarantine and keep going
            log.exception("failed to process %s", name)
            (config.FAILED_DIR / f"{name}.error").write_text(f"{type(exc).__name__}: {exc}\n")
            os.replace(claimed, config.FAILED_DIR / name)


def import_transcript(conn, path: Path, device: str, started_at: str | None,
                      model) -> kb.Conversation:
    """Store an existing transcript: summary now, no audio, status 'imported'."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty transcript: {path}")
    summary = ""
    if config.SUMMARY_ENABLED and hasattr(model, "summarize"):
        summary = (model.summarize(text) or "").strip()
    try:
        stamp = (datetime.fromisoformat(started_at).strftime("%Y%m%d-%H%M%S")
                 if started_at else "imported")
    except ValueError:
        stamp = "imported"
    conv = kb.Conversation(
        id=f"{stamp}-{device}-{path.stem}.txt",
        device=device,
        started_at=started_at,
        duration_s=None,
        language=None,
        transcript=text,
        audio_bytes=None,
        summary=summary,
        status="imported",
    )
    kb.upsert(conn, conv)
    return conv
