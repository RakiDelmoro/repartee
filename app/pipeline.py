"""Bigger device: finished WAV -> transcript -> summary -> knowledge-base row.

The receiver drops a ``<name>.wav.done`` sidecar after it has fixed the RIFF
sizes and closed the file. The worker only claims those files, so it never
transcribes a recording that is still growing. An explicit ``process`` command
can process any WAV, marker or not.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

from . import config, conversations

log = logging.getLogger(__name__)

FILENAME_RE = re.compile(
    r"^(?P<date>\d{8})-(?P<time>\d{6})-(?P<device>[A-Za-z0-9]{1,32})"
    r"\.(?P<ext>wav|flac|ogg|opus|mp3|m4a)$"
)
DONE_SUFFIX = ".done"
ERROR_SUFFIX = ".error"
SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_-]")
MIN_TRANSCRIPT_WORDS = 3


def parse_filename(name: str) -> tuple[str, str | None]:
    """Return (device, ISO start time) parsed from the receiver filename."""
    match = FILENAME_RE.match(name)
    if not match:
        stem = SAFE_STEM_RE.sub("", Path(name).stem)[:32]
        return (stem or "unknown"), None
    date, clock = match["date"], match["time"]
    stamp = (
        f"{date[:4]}-{date[4:6]}-{date[6:]}"
        f"T{clock[:2]}:{clock[2:4]}:{clock[4:6]}"
    )
    return match["device"], stamp


def marker_path(path: Path, suffix: str) -> Path:
    return path.with_name(path.name + suffix)


def done_path(path: Path) -> Path:
    return marker_path(path, DONE_SUFFIX)


def error_path(path: Path) -> Path:
    return marker_path(path, ERROR_SUFFIX)


def _started_at(path: Path, parsed: str | None) -> str:
    if parsed:
        return parsed
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(
        timespec="seconds"
    )


def _ended_at(started_at: str, duration_s: float | None) -> str | None:
    if duration_s is None:
        return None
    try:
        start = datetime.fromisoformat(started_at)
    except ValueError:
        return None
    return (start + timedelta(seconds=duration_s)).isoformat(timespec="seconds")


def process(path: Path, transcriber, conn, force: bool = False) -> dict:
    """Transcribe, summarize, and upsert one recording. Returns the stored row."""
    path = Path(path)
    existing = conversations.get(conn, path.name)
    if existing is not None and not force:
        log.info("%s already stored; skipping", path.name)
        return existing

    transcript = transcriber.transcribe(path)
    text = transcript.text.strip()
    if not text:
        raise RuntimeError(f"empty transcript for {path.name}")

    word_count = len(text.split())
    summary = ""
    if config.SUMMARY_ENABLED and word_count >= MIN_TRANSCRIPT_WORDS:
        try:
            summary = (transcriber.summarize(text) or "").strip()
        except Exception:  # noqa: BLE001 - the transcript is what must not be lost
            log.exception("%s: summarization failed", path.name)

    device, parsed_start = parse_filename(path.name)
    started_at = _started_at(path, parsed_start)
    conversation = conversations.Conversation(
        id=path.name,
        device=device,
        started_at=started_at,
        ended_at=_ended_at(started_at, transcript.duration_s),
        duration_s=transcript.duration_s,
        language=transcript.language,
        transcript=transcript.text,
        summary=summary,
        audio_path=str(path.resolve()),
        audio_bytes=path.stat().st_size,
        model=transcript.model,
        status=(
            "short"
            if word_count < MIN_TRANSCRIPT_WORDS
            else "summarized" if summary else "transcribed"
        ),
    )
    conversations.upsert(conn, conversation)
    error_path(path).unlink(missing_ok=True)
    log.info(
        "%s: %.1f s audio, %d transcript chars, summary %s",
        path.name,
        transcript.duration_s,
        len(transcript.text),
        "stored" if summary else "unavailable",
    )
    return conversations.get(conn, path.name) or {}


def process_ready(
    conn, transcriber, recordings: Path, require_done: bool = True
) -> int:
    """Process every unclaimed finished recording. Returns the number handled."""
    processed = 0
    for path in sorted(Path(recordings).glob("*.wav")):
        if error_path(path).exists():
            log.debug("%s has an error marker; skipping", path.name)
            continue
        if require_done and not done_path(path).exists():
            continue
        if conversations.get(conn, path.name) is not None:
            continue
        try:
            process(path, transcriber, conn)
            processed += 1
        except Exception:  # noqa: BLE001 - quarantine and keep the worker alive
            log.exception("failed to process %s", path.name)
            error_path(path).write_text(traceback.format_exc(), encoding="utf-8")
    return processed


def run_worker(
    db_path: Path,
    recordings: Path,
    poll_seconds: float,
    transcriber,
    once: bool = False,
) -> None:
    conn = conversations.connect(db_path)
    conversations.init(conn)
    log.info("worker: watching %s, db %s", recordings, db_path)
    try:
        while True:
            process_ready(conn, transcriber, recordings)
            if once:
                return
            time.sleep(poll_seconds)
    finally:
        conn.close()


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    parser.add_argument("--model", default=config.STT_MODEL)
    parser.add_argument("--language", default=config.STT_LANGUAGE)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repartee",
        description="Bigger-device pipeline: WAV -> transcript -> summary -> KB.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("process", help="process one recording")
    _add_common(one)
    one.add_argument("audio", type=Path)
    one.add_argument("--force", action="store_true", help="replace an existing row")

    worker = sub.add_parser("worker", help="watch recordings for .done markers")
    _add_common(worker)
    worker.add_argument(
        "--recordings", type=Path, default=config.RECORDINGS_DIR
    )
    worker.add_argument(
        "--poll", type=float, default=config.WORKER_POLL_SECONDS
    )
    worker.add_argument("--once", action="store_true", help="drain once and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from .stt import OmniTranscriber

    transcriber = OmniTranscriber(model_id=args.model, language=args.language)

    if args.command == "process":
        if not args.audio.is_file():
            raise SystemExit(f"not a file: {args.audio}")
        conn = conversations.connect(args.db)
        try:
            conversations.init(conn)
            row = process(args.audio, transcriber, conn, force=args.force)
        finally:
            conn.close()
        print(json.dumps(row, indent=2, ensure_ascii=False))
        return 0

    try:
        run_worker(
            args.db,
            args.recordings,
            args.poll,
            transcriber,
            once=args.once,
        )
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
