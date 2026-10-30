#!/usr/bin/env python3
"""Refuse to start a pipeline stage unless its prerequisite gate artifact passed.

Used by ``make train`` so the blueprint's stop conditions are enforced by the
build system, not by memory.  Accepts artifacts with any of the shapes produced
by this repo: ``{"result": "PASS"}``, ``{"passed": true}``, or a Gate-0
``{"overall": {"result": "PASS"}}`` wrapper.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def gate_status(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    overall = value.get("overall")
    if isinstance(overall, dict):
        return gate_status(overall)
    if value.get("passed") is True or value.get("passes") is True:
        return "PASS"
    return str(value.get("result", value.get("status", ""))).strip().upper()


def gate_passed(value: object) -> bool:
    # CONDITIONAL is the license-only state: the technical gates passed, but
    # organisation revenue is unknown.  Prototype training may proceed; this
    # is not permission for commercial deployment.
    return gate_status(value) in ("PASS", "CONDITIONAL")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--name", default="prerequisite gate")
    args = parser.parse_args(argv)
    if not args.artifact.exists():
        print(f"{args.name}: artifact not found: {args.artifact}", file=sys.stderr)
        return 2
    try:
        value = json.loads(args.artifact.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        print(f"{args.name}: cannot parse {args.artifact}: {error}", file=sys.stderr)
        return 2
    if not gate_passed(value):
        print(f"{args.name}: {args.artifact} did not pass; refusing to continue", file=sys.stderr)
        return 1
    status = gate_status(value)
    if status == "CONDITIONAL":
        print(f"{args.name}: CONDITIONAL — technical gates passed, license revenue still unverified ({args.artifact})")
    else:
        print(f"{args.name}: {status} ({args.artifact})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
