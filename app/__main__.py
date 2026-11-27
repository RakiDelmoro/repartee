"""Command line entry point: the USB receiver is the only command."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from . import receiver

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "receiver":
        argv = argv[1:]
    return receiver.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
