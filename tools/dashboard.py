"""Live training dashboard for Repartee (standard library only).

Run:
    python -m tools.dashboard --runs checkpoints --port 8080

Open http://127.0.0.1:8080.  The page polls the metrics JSONL files written by
``src.train_stage1`` / ``src.train_stage2`` and draws loss/learning-rate/step
time charts without any external JavaScript or CDN.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

MAX_POINTS = 1500
RUNNING_WINDOW_S = 180.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    records.append(value)
    except OSError:
        return []
    return records


def _downsample(records: list[dict[str, Any]], limit: int = MAX_POINTS) -> list[dict[str, Any]]:
    if len(records) <= limit:
        return records
    stride = max(1, len(records) // limit)
    sampled = records[::stride]
    if records[-1] is not sampled[-1]:
        sampled.append(records[-1])
    return sampled


def _series(records: list[dict[str, Any]], key: str) -> list[float | None]:
    values: list[float | None] = []
    for record in records:
        value = record.get(key)
        values.append(float(value) if isinstance(value, (int, float)) else None)
    return values


def _eta_seconds(records: list[dict[str, Any]], total_steps: int | None) -> float | None:
    if not total_steps or not records:
        return None
    latest = records[-1]
    step = latest.get("step")
    if not isinstance(step, (int, float)):
        return None
    remaining = max(0.0, float(total_steps) - float(step))
    times = [
        float(r["seconds_per_step"])
        for r in records[-20:]
        if isinstance(r.get("seconds_per_step"), (int, float)) and r["seconds_per_step"] > 0
    ]
    if not times:
        return None
    return remaining * statistics.median(times)


def _checkpoint_info(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "state" / "latest.pt"
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "modified": stat.st_mtime,
        "age_s": max(0.0, time.time() - stat.st_mtime),
        "size_bytes": stat.st_size,
    }


def summarize_run(run_dir: Path, root: Path | None = None) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.jsonl"
    records = [r for r in _read_jsonl(metrics_path) if isinstance(r.get("step"), (int, float))]
    start = next((r for r in _read_jsonl(metrics_path) if r.get("event") == "start"), {})
    latest = records[-1] if records else {}
    total_steps = start.get("steps")
    if total_steps is None and start.get("steps_per_epoch") is not None:
        total_steps = int(start["steps_per_epoch"]) * int(start.get("epochs", 1))
    checkpoint = _checkpoint_info(run_dir)
    age = latest.get("seconds_per_step")
    name = run_dir.name
    if root is not None:
        try:
            name = str(run_dir.resolve().relative_to(root.resolve()))
        except ValueError:
            pass
    return {
        "name": name,
        "path": str(run_dir),
        "metrics": str(metrics_path),
        "start": start,
        "latest": latest,
        "step": latest.get("step", 0),
        "total_steps": total_steps,
        "eta_s": _eta_seconds(records, total_steps),
        "checkpoint": checkpoint,
        "running": metrics_path.exists()
        and (time.time() - metrics_path.stat().st_mtime) < RUNNING_WINDOW_S,
        "updated": metrics_path.stat().st_mtime if metrics_path.exists() else None,
        "step_seconds": age if isinstance(age, (int, float)) else None,
    }


def load_run(run_dir: Path, max_points: int = MAX_POINTS, root: Path | None = None) -> dict[str, Any]:
    summary = summarize_run(run_dir, root)
    raw = _read_jsonl(run_dir / "metrics.jsonl")
    records = _downsample([r for r in raw if isinstance(r.get("step"), (int, float))], max_points)
    series = {
        "step": _series(records, "step"),
        "loss": _series(records, "loss"),
        "text_loss": _series(records, "text_loss"),
        "control_loss": _series(records, "control_loss"),
        "audio_loss": _series(records, "audio_loss"),
        "lr": _series(records, "lr"),
        "grad_norm": _series(records, "grad_norm"),
        "seconds_per_step": _series(records, "seconds_per_step"),
        "frames_per_s": _series(records, "frames_per_s"),
        "tokens_per_s": _series(records, "tokens_per_s"),
    }
    summary["series"] = series
    summary["events"] = [r for r in raw if "event" in r][-20:]
    return summary


def discover_runs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        {path.parent.resolve() for path in root.rglob("metrics.jsonl")},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def gpu_status() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in output.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            rows.append(
                {
                    "name": parts[0],
                    "memory_total_mib": float(parts[1]),
                    "memory_used_mib": float(parts[2]),
                    "memory_free_mib": float(parts[3]),
                    "utilization_pct": float(parts[4]),
                    "temperature_c": float(parts[5]),
                }
            )
        except ValueError:
            continue
    return rows


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Repartee training</title>
<style>
  :root { color-scheme: light dark; --bg:#0f1117; --fg:#e6e8ee; --muted:#9aa3b2;
          --border:#2a2f3a; --surface:#171a22; --accent:#5aa9ff; --success:#43d17a;
          --warning:#e6b450; --error:#ff6b6b; }
  @media (prefers-color-scheme: light) {
    :root { --bg:#f7f8fa; --fg:#16181d; --muted:#5d6572; --border:#dfe3ea;
            --surface:#ffffff; --accent:#1f6feb; --success:#1a7f37;
            --warning:#9a6700; --error:#cf222e; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.5 ui-sans-serif,system-ui,sans-serif; }
  header { padding:18px 24px; border-bottom:1px solid var(--border); display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:18px; margin:0; font-weight:600; }
  .spacer { flex:1; }
  main { padding:20px 24px 40px; max-width:1200px; margin:0 auto; }
  select, button { background:var(--surface); color:var(--fg); border:1px solid var(--border);
                   border-radius:8px; padding:7px 10px; font:inherit; }
  button { cursor:pointer; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:18px; }
  .card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:12px 14px; }
  .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
  .value { font-size:20px; font-weight:650; margin-top:3px; font-variant-numeric:tabular-nums; }
  .small { font-size:12px; color:var(--muted); }
  .progress { height:9px; background:var(--border); border-radius:999px; overflow:hidden; margin-top:8px; }
  .progress > div { height:100%; width:0; background:var(--accent); transition:width .4s ease; }
  .chart { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:12px 14px; margin-bottom:14px; }
  .chart h2 { font-size:13px; margin:0 0 8px; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
  svg { width:100%; height:auto; display:block; overflow:visible; }
  .legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--muted); margin-top:6px; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; vertical-align:1px; }
  .err { color:var(--error); }
  .badge { font-size:11px; border:1px solid var(--border); border-radius:999px; padding:2px 8px; color:var(--muted); }
  .live { color:var(--success); }
  .stale { color:var(--warning); }
</style>
</head>
<body>
<header>
  <h1>Repartee training</h1>
  <select id="runs"></select>
  <button id="pause">Pause</button>
  <span id="state" class="badge">connecting</span>
  <span class="spacer"></span>
  <span id="gpu" class="small"></span>
</header>
<main>
  <div class="grid">
    <div class="card"><div class="label">Step</div><div class="value" id="step">–</div><div class="small" id="total"></div><div class="progress"><div id="bar"></div></div></div>
    <div class="card"><div class="label">Loss</div><div class="value" id="loss">–</div><div class="small" id="loss_parts"></div></div>
    <div class="card"><div class="label">ETA</div><div class="value" id="eta">–</div><div class="small" id="speed"></div></div>
    <div class="card"><div class="label">Checkpoint</div><div class="value" id="ckpt_age">–</div><div class="small" id="ckpt_path"></div></div>
    <div class="card"><div class="label">Run</div><div class="value" id="run_name" style="font-size:15px">–</div><div class="small" id="run_info"></div></div>
  </div>
  <div class="chart"><h2>Loss</h2><svg id="loss_chart" viewBox="0 0 900 260"></svg><div class="legend" id="loss_legend"></div></div>
  <div class="chart"><h2>Learning rate</h2><svg id="lr_chart" viewBox="0 0 900 180"></svg></div>
  <div class="chart"><h2>Step time &amp; throughput</h2><svg id="speed_chart" viewBox="0 0 900 180"></svg></div>
</main>
<script>
const $ = id => document.getElementById(id);
let current = null, paused = false, timer = null;

async function getJSON(url) {
  const response = await fetch(url, {cache: "no-store"});
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  return Number(value).toFixed(digits);
}
function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined || !isFinite(seconds)) return "–";
  if (seconds < 90) return Math.round(seconds) + "s";
  if (seconds < 5400) return (seconds / 60).toFixed(1) + "m";
  if (seconds < 172800) return (seconds / 3600).toFixed(1) + "h";
  return (seconds / 86400).toFixed(1) + "d";
}
function color(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function drawSeries(svg, x, series, opts = {}) {
  const W = 900, H = Number(svg.getAttribute("viewBox").split(" ")[3]);
  const pad = {l: 54, r: 12, t: 12, b: 26};
  svg.innerHTML = "";
  const valid = series.filter(s => s.values.some(v => typeof v === "number" && isFinite(v)));
  if (!valid.length || !x.length) {
    svg.innerHTML = `<text x="${W/2}" y="${H/2}" fill="${color('--muted')}" text-anchor="middle">no data yet</text>`;
    return;
  }
  const xmin = Math.min(...x), xmax = Math.max(...x, xmin + 1);
  let ymin = Infinity, ymax = -Infinity;
  for (const s of valid) for (const v of s.values) if (typeof v === "number" && isFinite(v)) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
  if (opts.log) { ymin = Math.max(1e-6, ymin); }
  if (ymin === ymax) { ymax = ymin + 1; }
  const sx = v => pad.l + (v - xmin) / (xmax - xmin) * (W - pad.l - pad.r);
  const sy = v => H - pad.b - (v - ymin) / (ymax - ymin) * (H - pad.t - pad.b);
  const ns = "http://www.w3.org/2000/svg";
  const add = (name, attrs, text) => {
    const el = document.createElementNS(ns, name);
    for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, v);
    if (text !== undefined) el.textContent = text;
    svg.appendChild(el); return el;
  };
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + i * (H - pad.t - pad.b) / 4;
    const value = ymax - i * (ymax - ymin) / 4;
    add("line", {x1: pad.l, x2: W - pad.r, y1: y, y2: y, stroke: color('--border'), "stroke-width": 1});
    add("text", {x: pad.l - 8, y: y + 4, fill: color('--muted'), "font-size": 11, "text-anchor": "end"}, fmt(value, opts.digits ?? 2));
  }
  add("text", {x: pad.l, y: H - 6, fill: color('--muted'), "font-size": 11}, `step ${Math.round(xmin)}`);
  add("text", {x: W - pad.r, y: H - 6, fill: color('--muted'), "font-size": 11, "text-anchor": "end"}, `step ${Math.round(xmax)}`);
  for (const s of valid) {
    let path = "", pen = false;
    s.values.forEach((v, i) => {
      if (typeof v !== "number" || !isFinite(v)) { pen = false; return; }
      const xx = sx(x[i]), yy = sy(v);
      path += (pen ? " L" : " M") + xx.toFixed(1) + " " + yy.toFixed(1);
      pen = true;
    });
    add("path", {d: path, fill: "none", stroke: color(s.color), "stroke-width": 2, "stroke-linejoin": "round"});
  }
}
function legend(el, series) {
  el.innerHTML = series.map(s => `<span><span class="dot" style="background:${color(s.color)}"></span>${s.label}</span>`).join("");
}

async function refresh() {
  if (paused) return;
  try {
    const gpu = await getJSON("/api/gpu");
    $("gpu").textContent = gpu.length
      ? `${gpu[0].name} · ${Math.round(gpu[0].memory_used_mib)}/${Math.round(gpu[0].memory_total_mib)} MiB · ${Math.round(gpu[0].utilization_pct)}% · ${Math.round(gpu[0].temperature_c)}°C`
      : "GPU unavailable";
    const runs = await getJSON("/api/runs");
    const select = $("runs");
    const chosen = current || (runs[0] && runs[0].name);
    select.innerHTML = runs.map(r => `<option value="${r.name}">${r.name}</option>`).join("");
    if (runs.length === 0) { $("state").textContent = "no runs found"; return; }
    select.value = chosen;
    current = chosen;
    const run = await getJSON(`/api/run?name=${encodeURIComponent(chosen)}`);
    render(run);
    $("state").textContent = run.running ? "live" : "idle";
    $("state").className = "badge " + (run.running ? "live" : "stale");
  } catch (error) {
    $("state").textContent = "error";
    $("state").className = "badge err";
    console.error(error);
  }
}

function render(run) {
  const latest = run.latest || {};
  $("run_name").textContent = run.name;
  const updatedAgo = run.updated ? Math.max(0, Math.round(Date.now() / 1000 - run.updated)) : null;
  $("run_info").textContent = `${run.start.device || ""} ${run.start.dtype || ""} · ${run.start.seq_frames || "?"} frames · ${run.start.grad_accum || "?"} accum` +
    (updatedAgo !== null ? ` · updated ${updatedAgo}s ago` : "");
  $("step").textContent = Math.round(run.step || 0);
  $("total").textContent = run.total_steps ? `of ${run.total_steps}` : "";
  $("bar").style.width = run.total_steps ? `${Math.min(100, 100 * (run.step || 0) / run.total_steps)}%` : "0%";
  $("loss").textContent = fmt(latest.loss);
  $("loss_parts").textContent = `text ${fmt(latest.text_loss)} · control ${fmt(latest.control_loss)} · audio ${fmt(latest.audio_loss)}`;
  $("eta").textContent = fmtDuration(run.eta_s);
  $("speed").textContent = `${fmt(latest.seconds_per_step, 1)} s/step · ${fmt(latest.frames_per_s, 2)} frames/s`;
  const cp = run.checkpoint || {};
  $("ckpt_age").textContent = cp.exists ? fmtDuration(cp.age_s) + " ago" : "none";
  $("ckpt_path").textContent = cp.exists ? `${(cp.size_bytes / 2**30).toFixed(2)} GiB · ${cp.path}` : "no state/latest.pt";
  const s = run.series;
  const lossSeries = [
    {label:"total", values:s.loss, color:"--accent"},
    {label:"text", values:s.text_loss, color:"--success"},
    {label:"control", values:s.control_loss, color:"--warning"},
    {label:"audio", values:s.audio_loss, color:"--error"},
  ];
  drawSeries($("loss_chart"), s.step, lossSeries);
  legend($("loss_legend"), lossSeries);
  drawSeries($("lr_chart"), s.step, [{label:"lr", values:s.lr, color:"--accent"}], {digits:6});
  drawSeries($("speed_chart"), s.step, [
    {label:"s/step", values:s.seconds_per_step, color:"--warning"},
    {label:"frames/s", values:s.frames_per_s, color:"--success"},
  ]);
}

$("runs").addEventListener("change", event => { current = event.target.value; refresh(); });
$("pause").addEventListener("click", () => {
  paused = !paused;
  $("pause").textContent = paused ? "Resume" : "Pause";
});
refresh();
timer = setInterval(refresh, 2000);
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "ReparteeDashboard/1.0"

    def _json(self, value: Any, status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(value, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _html(self) -> None:
        payload = PAGE.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        runs_root: Path = self.server.runs_root  # type: ignore[attr-defined]
        if parsed.path == "/":
            self._html()
        elif parsed.path == "/api/gpu":
            self._json(gpu_status())
        elif parsed.path == "/api/runs":
            self._json([summarize_run(path, runs_root) for path in discover_runs(runs_root)])
        elif parsed.path == "/api/run":
            name = parse_qs(parsed.query).get("name", [""])[0]
            target = (runs_root / name).resolve()
            if not str(target).startswith(str(runs_root.resolve())) or not (target / "metrics.jsonl").exists():
                self._json({"error": "run not found"}, HTTPStatus.NOT_FOUND)
                return
            self._json(load_run(target, root=runs_root))
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("checkpoints"), help="root containing */metrics.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    server.runs_root = args.runs  # type: ignore[attr-defined]
    print(f"dashboard: http://{args.host}:{args.port}  runs={args.runs}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
