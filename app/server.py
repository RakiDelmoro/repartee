"""Ingest API and knowledge-base queries."""

from __future__ import annotations

import logging
import os
import secrets
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, db
from . import kb as knowledge
from .asr import QwenTranscriber
from .pipeline import Worker

WEB_DIR = Path(__file__).parent / "web"

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

SAFE_NAME = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
ALLOWED_SUFFIXES = {".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a"}


def safe_filename(raw: str | None) -> str:
    if not raw:
        raise HTTPException(400, "missing X-Filename header")
    name = os.path.basename(raw)
    if name != raw or not name or not set(name) <= SAFE_NAME:
        raise HTTPException(400, "filename may only contain letters, digits, '.', '_' and '-'")
    if name.startswith(".") or name.endswith(".part"):
        raise HTTPException(400, "invalid filename")
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"unsupported extension; expected one of {sorted(ALLOWED_SUFFIXES)}")
    return name


def check_auth(request: Request) -> None:
    if not config.INGEST_TOKEN:
        return
    header = request.headers.get("authorization", "")
    expected = f"Bearer {config.INGEST_TOKEN}"
    if not secrets.compare_digest(header, expected):
        raise HTTPException(401, "bad or missing token")


def already_known(conn, name: str) -> bool:
    """True when this recording is queued, in flight, finished, or indexed."""
    return (
        (config.DONE_DIR / name).exists()
        or (config.WORKING_DIR / name).exists()
        or (config.SPOOL_DIR / name).exists()
        or db.get(conn, name) is not None
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    # Anything left mid-flight by a crash goes back to the queue.
    for stale in config.WORKING_DIR.glob("*"):
        if stale.is_file():
            os.replace(stale, config.SPOOL_DIR / stale.name)
    for partial in config.SPOOL_DIR.glob("*.part"):
        partial.unlink(missing_ok=True)

    conn = db.connect(config.DB_PATH)
    db.init(conn)
    worker = Worker(QwenTranscriber())
    worker.start()
    app.state.conn = conn
    app.state.worker = worker
    log.info("listening; spool=%s db=%s", config.SPOOL_DIR, config.DB_PATH)
    try:
        yield
    finally:
        worker.stop()
        worker.join(timeout=10)
        conn.close()


app = FastAPI(title="repartee ingest", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "conversations": db.count(app.state.conn),
        "spooled": len([p for p in config.SPOOL_DIR.glob("*") if not p.name.endswith(".part")]),
    }


@app.post("/ingest", status_code=200)
async def ingest(
    request: Request,
    x_filename: str | None = Header(None, alias="X-Filename"),
    x_device: str | None = Header(None, alias="X-Device"),
) -> dict:
    check_auth(request)
    name = safe_filename(x_filename)

    if already_known(app.state.conn, name):
        return {"status": "duplicate", "id": name}

    destination = config.SPOOL_DIR / name
    partial = config.SPOOL_DIR / f"{name}.part"

    written = 0
    try:
        with partial.open("wb") as handle:
            async for chunk in request.stream():
                written += len(chunk)
                if written > config.MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "upload too large")
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if written == 0:
            raise HTTPException(400, "empty body")
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    log.info("queued %s (%d bytes, device=%s)", name, written, x_device or "unknown")
    return {"status": "queued", "id": name, "bytes": written}


@app.get("/conversations")
def conversations(limit: int = Query(20, ge=1, le=200)) -> dict:
    return {"items": db.recent(app.state.conn, limit)}


@app.get("/conversations/{conversation_id}")
def conversation(conversation_id: str) -> dict:
    row = db.get(app.state.conn, conversation_id)
    if row is None:
        raise HTTPException(404, "not found")
    return row


@app.get("/search")
def search(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=200)) -> dict:
    try:
        return {"items": db.search(app.state.conn, q, limit)}
    except sqlite3.OperationalError as exc:
        raise HTTPException(400, f"bad query: {exc}") from exc


# --- knowledge base -------------------------------------------------------


@app.get("/api/graph")
def graph(limit: int = Query(800, ge=1, le=5000)) -> dict:
    """Nodes and edges for the graph view: sessions, events, entities."""
    conn = app.state.conn
    nodes: list[dict] = []
    links: list[dict] = []

    sessions = conn.execute(
        "SELECT id, device, started_at, status FROM sessions ORDER BY started_at DESC"
    ).fetchall()

    for session in sessions:
        nodes.append(
            {
                "id": session["id"],
                "label": f"{session['device']} · {(session['started_at'] or '')[:16]}",
                "type": "session",
                "session": session["id"],
            }
        )

    rows = conn.execute(
        """SELECT id, session_id, kind, text, owner FROM events
           ORDER BY session_id, t_start IS NULL, t_start
           LIMIT ?""",
        (limit,),
    ).fetchall()

    for event in rows:
        nodes.append(
            {
                "id": event["id"],
                "label": event["text"][:90],
                "type": event["kind"],
                "session": event["session_id"],
            }
        )
        links.append(
            {"source": event["session_id"], "target": event["id"], "type": "contains"}
        )

    for entity in conn.execute("SELECT id, kind, name FROM entities").fetchall():
        nodes.append(
            {"id": entity["id"], "label": entity["name"], "type": entity["kind"], "session": None}
        )

    for row in conn.execute("SELECT event_id, entity_id, role FROM event_entities").fetchall():
        links.append({"source": row["event_id"], "target": row["entity_id"], "type": row["role"]})

    for row in conn.execute("SELECT from_event, to_event, relation FROM event_links").fetchall():
        links.append(
            {"source": row["from_event"], "target": row["to_event"], "type": row["relation"]}
        )

    return {"nodes": nodes, "links": links}


@app.get("/api/events/{event_id}")
def event_detail(event_id: str) -> dict:
    row = app.state.conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "not found")
    return dict(row)


@app.get("/api/sessions")
def api_sessions() -> dict:
    rows = app.state.conn.execute(
        """SELECT s.id, s.device, s.started_at, s.status,
                  COUNT(e.id) AS events
           FROM sessions s LEFT JOIN events e ON e.session_id = s.id
           GROUP BY s.id ORDER BY s.started_at DESC"""
    ).fetchall()
    return {"items": [dict(row) for row in rows]}


@app.get("/api/stats")
def api_stats() -> dict:
    return knowledge.counts(app.state.conn)
