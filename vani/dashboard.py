"""Local Vani-M training dashboard: log, progress, hardware, data."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import psutil
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent / "web"
CHECKPOINTS = ROOT / "checkpoints"
P0 = ROOT / "data" / "vani_p0_libritts"
P0_TARGET = 558
DEFAULT_STEPS = 5000
DEFAULT_ACCUM = 2
DEFAULT_WINDOW = 192
DEFAULT_STRIDE = 128

LINE = re.compile(
    r"step (\d+) loss=([0-9.]+) text=([0-9.]+) "
    r"(.*?) maxmem=([0-9.]+)GB skip=(\d+) accum=(\d+)"
    r"(?: p=([0-9.]+))? (\d+)s"
)

app = FastAPI(title="Vani-M dashboard")
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.middleware("http")
async def no_store_ui(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store"
    return response


def active_run() -> Path:
    env = os.environ.get("VANI_RUN")
    if env:
        return ROOT / env
    proc = training_process()
    if proc and proc.get("out"):
        return ROOT / proc["out"]
    logs = list(CHECKPOINTS.glob("*/train.log"))
    if logs:
        return max(logs, key=lambda p: p.stat().st_mtime).parent
    return CHECKPOINTS / "p1_28h"


def run_config(run: Path) -> dict:
    path = run / "run.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {
        "steps": int(os.environ.get("VANI_STEPS", DEFAULT_STEPS)),
        "accum": int(os.environ.get("VANI_ACCUM", DEFAULT_ACCUM)),
        "window": DEFAULT_WINDOW,
        "stride": DEFAULT_STRIDE,
    }


def p0_stats() -> dict:
    files = sorted(P0.glob("conv_*.json")) if P0.exists() else []
    total_frames = 0
    total_seconds = 0.0
    latest = None
    for path in files:
        try:
            meta = json.loads(path.read_text())
        except Exception:
            continue
        total_frames += int(meta.get("frames", 0))
        total_seconds += float(meta.get("seconds", 0.0))
        latest = meta
    return {
        "done": len(files),
        "target": P0_TARGET,
        "hours": round(total_seconds / 3600, 2),
        "frames": total_frames,
        "latest": latest.get("source") if latest else None,
        "latest_snr": latest.get("snr_db") if latest else None,
    }


def training_process() -> dict | None:
    for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent", "memory_info", "create_time"]):
        try:
            cmd = " ".join(proc.info["cmdline"] or [])
            if "vani.train_p1" in cmd:
                mem = proc.info["memory_info"]
                parts = cmd.split()
                out = parts[parts.index("--out") + 1] if "--out" in parts else None
                return {
                    "pid": proc.info["pid"],
                    "cpu": round(proc.info["cpu_percent"] or 0.0, 1),
                    "rss_gb": round((mem.rss if mem else 0) / 1e9, 2),
                    "elapsed": round(datetime.now(timezone.utc).timestamp() - proc.info["create_time"], 0),
                    "out": out,
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def read_log() -> dict:
    run = active_run()
    log = run / "train.log"
    cfg = run_config(run)
    steps_total = int(cfg.get("steps", DEFAULT_STEPS))
    step_offset = int(cfg.get("step_offset", 0))
    accum = int(cfg.get("accum", DEFAULT_ACCUM))
    records = []
    lines = []
    if log.exists():
        try:
            lines = log.read_text(errors="replace").splitlines()[-2000:]
        except Exception:
            lines = []
    for line in lines:
        m = LINE.search(line)
        if not m:
            continue
        vals = m.groups()
        cbs = re.findall(r"cb(\d+)=([0-9.]+)", vals[3])
        cb_values = [float(v) for _, v in cbs]
        records.append(
            {
                "step": int(vals[0]) + step_offset,
                "chart_step": int(vals[0]),
                "loss": float(vals[1]),
                "text": float(vals[2]),
                "cb": cb_values,
                "maxmem": float(vals[4]),
                "skip": int(vals[5]),
                "accum": int(vals[6]),
                "p": float(vals[7]) if vals[7] else None,
                "elapsed": int(vals[8]),
            }
        )
    latest = records[-1] if records else None
    proc = training_process()
    p0 = p0_stats()
    windows = max(1, int((p0["frames"] - cfg.get("window", 192)) // max(1, cfg.get("stride", 128)))) if p0["frames"] else 0
    epochs = (latest["step"] * accum / windows) if latest and windows else 0.0
    eta = None
    additional_done = (latest["step"] - step_offset) if latest else 0
    if latest and additional_done > 0 and latest["elapsed"] > 0 and latest["step"] < steps_total:
        eta = round((steps_total - latest["step"]) * latest["elapsed"] / additional_done)
    state = "waiting"
    if proc:
        state = "training" if latest else "starting"
    elif latest and latest["step"] >= steps_total:
        state = "finished"
    elif p0["done"] < P0_TARGET:
        state = "tokenizing"
    return {
        "state": state,
        "run": str(run.relative_to(ROOT)),
        "steps_total": steps_total,
        "chart_max": max(1, steps_total - step_offset),
        "step": latest["step"] if latest else 0,
        "progress": round((latest["step"] / steps_total) if latest and steps_total else 0.0, 4),
        "epochs": round(epochs, 3),
        "loss": latest["loss"] if latest else None,
        "text": latest["text"] if latest else None,
        "cb": latest["cb"] if latest else None,
        "skip": latest["skip"] if latest else 0,
        "accum": latest["accum"] if latest else accum,
        "elapsed": latest["elapsed"] if latest else 0,
        "eta": eta,
        "maxmem": latest["maxmem"] if latest else 0,
        "process": proc,
        "p0": p0,
        "series": records,
        "tail": lines[-40:],
        "config": cfg,
        "updated": datetime.now(timezone.utc).isoformat(),
    }


def gpu_stats() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        parts = [p.strip() for p in out.split(",")]
        return {
            "name": parts[0],
            "memory_used": float(parts[1]),
            "memory_total": float(parts[2]),
            "util": float(parts[3]),
            "temp": float(parts[4]),
            "power": float(parts[5]),
            "power_limit": float(parts[6]),
        }
    except Exception as exc:
        return {"error": str(exc)}


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def system_stats() -> dict:
    vm = psutil.virtual_memory()
    du = shutil.disk_usage(ROOT)
    freq = psutil.cpu_freq()
    return {
        "gpu": gpu_stats(),
        "cpu": {
            "name": cpu_name(),
            "percent": round(psutil.cpu_percent(interval=None), 1),
            "per_core": [round(v, 1) for v in psutil.cpu_percent(interval=None, percpu=True)],
            "cores": psutil.cpu_count(logical=True),
            "freq_mhz": round(freq.current) if freq else None,
            "load": [round(v, 2) for v in psutil.getloadavg()],
        },
        "ram": {
            "used_gb": round(vm.used / 1e9, 2),
            "total_gb": round(vm.total / 1e9, 2),
            "percent": round(vm.percent, 1),
        },
        "disk": {
            "used_gb": round(du.used / 1e9, 1),
            "total_gb": round(du.total / 1e9, 1),
            "percent": round(du.used / du.total * 100, 1),
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "time": datetime.now().isoformat(timespec="seconds"),
        },
        "process": training_process(),
    }


def data_stats() -> dict:
    run = active_run()
    checkpoints = []
    if run.exists():
        for path in sorted(run.glob("*.pt")):
            try:
                checkpoints.append(
                    {
                        "name": path.name,
                        "gb": round(path.stat().st_size / 1e9, 2),
                        "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                    }
                )
            except OSError:
                pass
    grids = list(P0.glob("*.grid.npy")) if P0.exists() else []
    return {
        "p0": p0_stats(),
        "checkpoints": checkpoints,
        "grids": len(grids),
        "grid_bytes": sum(p.stat().st_size for p in grids if p.exists()),
        "log_bytes": (run / "train.log").stat().st_size if (run / "train.log").exists() else 0,
    }


@app.get("/")
def index():
    return FileResponse(WEB / "dashboard.html")


@app.get("/api/status")
def status():
    return read_log()


@app.get("/api/system")
def system():
    return system_stats()


@app.get("/api/data")
def data():
    return data_stats()


@app.get("/api/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}
