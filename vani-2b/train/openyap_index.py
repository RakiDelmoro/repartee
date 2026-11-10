#!/usr/bin/env python3
"""Index Open Yap tar parts: conversation id -> part file."""

import json
import tarfile
from pathlib import Path

ROOT = Path("data/open-yap/open-yap-1k_2026-11-05_flac_26a4d80b")
OUT = Path("data/open-yap/index.json")


def main() -> None:
    index = {}
    for part in sorted((ROOT / "parts").glob("*.tar")):
        with tarfile.open(part) as tar:
            for member in tar:
                name = member.name
                if name.endswith("/meta.json"):
                    conv = name.split("/")[1]
                    index[conv] = part.name
        print(part.name, len(index), flush=True)
    OUT.write_text(json.dumps(index, indent=1))
    print("conversations", len(index))


if __name__ == "__main__":
    main()
