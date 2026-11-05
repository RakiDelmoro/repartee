"""Command line entry points."""

from __future__ import annotations

import argparse
import logging

from . import config


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("app.server:app", host=args.host, port=args.port, log_level="info")


def _work(args: argparse.Namespace) -> None:
    from .asr import QwenTranscriber
    from .pipeline import Worker

    config.ensure_dirs()
    worker = Worker(QwenTranscriber())
    logging.getLogger(__name__).info("worker watching %s", config.SPOOL_DIR)
    try:
        worker.run()
    except KeyboardInterrupt:
        pass


def main(argv: list[str] | None = None) -> int:
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
