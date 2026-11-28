"""Command line entry point: receiver, transcription, pipeline, and history."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] == "transcribe":
        from . import stt

        return stt.main(argv[1:])

    if argv and argv[0] in {"process", "worker"}:
        from . import pipeline

        return pipeline.main(argv)

    if argv and argv[0] in {"search", "conversations", "show"}:
        from . import conversations

        return conversations.main(argv)

    from . import receiver

    if argv and argv[0] == "receiver":
        argv = argv[1:]
    return receiver.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
