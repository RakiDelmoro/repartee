"""Spool worker: turn spooled audio into transcripts."""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path

from . import config, db
from .asr import QwenTranscriber

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


def process(path: Path, transcriber: QwenTranscriber) -> db.Conversation:
    device, started_at = parse_filename(path.name)
    transcript = transcriber.transcribe(path)

    return db.Conversation(
        id=path.name,
        device=device,
        started_at=started_at,
        duration_s=transcript.duration_s,
        language=transcript.language,
        transcript=transcript.text,
        audio_bytes=path.stat().st_size,
    )


class Worker(threading.Thread):
    """Claims files from spool/, transcribes them, and files them away."""

    def __init__(self, transcriber: QwenTranscriber, poll_seconds: float = 0.5) -> None:
        super().__init__(name="repartee-worker", daemon=True)
        self.transcriber = transcriber
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        conn = db.connect(config.DB_PATH)
        db.init(conn)
        backoff = self.poll_seconds
        try:
            while not self._stop.is_set():
                try:
                    self.drain(conn)
                    backoff = self.poll_seconds
                except Exception:  # noqa: BLE001 - keep the worker alive
                    # A mounted filesystem can return transient errors; back off
                    # instead of spinning on it once per poll.
                    log.exception("drain failed; backing off %.0fs", backoff)
                    backoff = min(backoff * 2, 30.0)
                self._stop.wait(backoff)
        finally:
            conn.close()

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
            if db.get(conn, name) is not None:
                log.info("%s already transcribed; skipping", name)
            else:
                conv = process(claimed, self.transcriber)
                db.insert(conn, conv)
                log.info(
                    "%s: %.1fs audio, %s, %d chars",
                    name,
                    conv.duration_s or 0.0,
                    conv.language,
                    len(conv.transcript),
                )
            os.replace(claimed, config.DONE_DIR / name)
        except Exception as exc:  # noqa: BLE001 - quarantine and keep going
            log.exception("failed to transcribe %s", name)
            (config.FAILED_DIR / f"{name}.error").write_text(f"{type(exc).__name__}: {exc}\n")
            os.replace(claimed, config.FAILED_DIR / name)
