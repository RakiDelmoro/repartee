"""Load fixture transcripts into the knowledge base so the UI has something to show.

    python -m app.seed --llm http://127.0.0.1:8081/v1
"""

from __future__ import annotations

import argparse
import logging
import pathlib

from . import config, kb
from .extract import LLMClient, extract

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="repartee-seed")
    parser.add_argument("--fixtures", default="tests/fixtures", type=pathlib.Path)
    parser.add_argument("--llm", default=config.LLM_BASE_URL)
    parser.add_argument("--db", default=config.DB_PATH, type=pathlib.Path)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = LLMClient(base_url=args.llm)
    conn = kb.connect(args.db)
    kb.init(conn)

    transcripts = sorted(args.fixtures.glob("*.txt"))
    if not transcripts:
        log.error("no .txt fixtures in %s", args.fixtures)
        return 1

    for index, path in enumerate(transcripts):
        session_id = f"fixture-{path.stem}"
        text = path.read_text().strip()

        conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
        kb.upsert_session(
            conn,
            kb.Session(
                id=session_id,
                device="fixture",
                started_at=f"2026-11-07T{9 + index:02d}:00:00",
                ended_at=None,
                duration_s=None,
                language="English",
                transcript=text,
                status="ready",
            ),
        )

        events = extract(text, client)
        for number, event in enumerate(events, start=1):
            kb.insert_event(
                conn,
                kb.Event(
                    id=f"{session_id}-{number:03d}",
                    session_id=session_id,
                    kind=event.kind,
                    text=event.text,
                    evidence=event.evidence,
                    owner=event.owner,
                    due=event.due,
                ),
            )

        print(f"{session_id}: {len(events)} events")

    print("counts:", kb.counts(conn))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
