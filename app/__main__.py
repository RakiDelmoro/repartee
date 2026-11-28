"""Command line entry point: the USB receiver and bigger-device transcription."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] == "transcribe":
        from . import stt

        return stt.main(argv[1:])

    from . import receiver

    if argv and argv[0] == "receiver":
        argv = argv[1:]
    return receiver.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
