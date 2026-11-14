"""Command line entry points."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import config


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("app.server:app", host=args.host, port=args.port, log_level="info")


def _work(args: argparse.Namespace) -> None:
    from .asr import make_transcriber
    from .pipeline import Worker

    config.ensure_dirs()
    worker = Worker(make_transcriber())
    logging.getLogger(__name__).info("worker watching %s", config.SPOOL_DIR)
    try:
        worker.run()
    except KeyboardInterrupt:
        pass


def _import(argv: list[str]) -> int:
    """Store existing transcripts (no audio) with a summary, so they appear in the KB."""
    from .asr import make_transcriber
    from .pipeline import import_transcript
    from . import kb

    parser = argparse.ArgumentParser(prog="repartee import")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--device", default="imported")
    parser.add_argument("--started-at", default=None,
                        help="ISO timestamp, or one per file, comma-separated")
    parser.add_argument("--skip-summary", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config.ensure_dirs()
    conn = kb.connect(config.DB_PATH)
    kb.init(conn)
    model = object() if args.skip_summary else make_transcriber()
    stamps = (args.started_at or "").split(",")
    for i, path in enumerate(args.files):
        started_at = stamps[i] if i < len(stamps) and stamps[i] else stamps[0] or None
        conv = import_transcript(conn, path, args.device, started_at, model)
        logging.info("%s -> %s (%d chars, summary %d chars)", path.name, conv.id,
                     len(conv.transcript), len(conv.summary))
    conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "listen":
        from . import listener

        return listener.main(argv[1:])
    if argv and argv[0] == "import":
        return _import(argv[1:])
    parser = argparse.ArgumentParser(prog="repartee")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the ingest API and the worker together")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)
    serve.set_defaults(func=_serve)

    work = sub.add_parser("worker", help="run only the spool worker (no HTTP)")
    work.set_defaults(func=_work)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
