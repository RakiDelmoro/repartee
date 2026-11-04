"""Live dashboard: serves either the main training run or the learning probe."""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CKPT = Path(__file__).resolve().parents[1] / "data/ckpt"

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>duplex training</title>
<style>
body{font:13px/1.4 system-ui,sans-serif;background:#0e1116;color:#e6edf3;margin:20px}
h1{font-size:16px;font-weight:600;margin:0 0 4px}
.sub{color:#8b949e;font-size:12px;margin-bottom:14px}
.grid{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:14px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px}
.k{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.v{font-size:20px;font-variant-numeric:tabular-nums;margin-top:2px}
canvas{background:#161b22;border:1px solid #30363d;border-radius:8px;width:100%;height:300px}
canvas+canvas{margin-top:10px}
table{border-collapse:collapse;width:100%;margin-top:12px;font-variant-numeric:tabular-nums}
td,th{border-bottom:1px solid #21262d;padding:4px 6px;text-align:left;color:#c9d1d9}
th{color:#8b949e;font-weight:500}
</style></head><body>
<h1 id="title">duplex backbone</h1><div class="sub" id="sub"></div>
<div class="grid">
 <div class="card"><div class="k">step</div><div class="v" id="step">–</div></div>
 <div class="card"><div class="k" id="k1">dev CE</div><div class="v" id="v1">–</div></div>
 <div class="card"><div class="k" id="k2">floor / loss</div><div class="v" id="v2">–</div></div>
 <div class="card"><div class="k" id="k3">dev top-1</div><div class="v" id="v3">–</div></div>
 <div class="card"><div class="k">frames/s</div><div class="v" id="fps">–</div></div>
 <div class="card"><div class="k">ETA</div><div class="v" id="eta">–</div></div>
</div>
<canvas id="chart" width="1200" height="300"></canvas>
<canvas id="chart2" width="1200" height="150"></canvas>
<table><thead><tr><th>checkpoint</th><th>size</th><th>saved</th></tr></thead><tbody id="ck"></tbody></table>
<script>
const TOTAL = %TOTAL%;
const MODE = "%MODE%";
let main = [], probe = [];
function axes(c, pad){ return {W:c.width,H:c.height,pad}; }
function frame(g,c,pad,minX,maxX,minY,maxY){
  g.strokeStyle='#30363d'; g.lineWidth=1;
  g.beginPath(); g.moveTo(pad,c.height-pad); g.lineTo(c.width-pad,c.height-pad);
  g.moveTo(pad,pad); g.lineTo(pad,c.height-pad); g.stroke();
  g.fillStyle='#8b949e'; g.font='11px system-ui';
  g.fillText(maxY.toFixed(2), 4, pad+4); g.fillText(minY.toFixed(2), 4, c.height-pad);
  g.fillText('step '+maxX, c.width-90, c.height-12);
  return {X:s=>pad+(s-minX)/(maxX-minX||1)*(c.width-2*pad),
          Y:v=>c.height-pad-(v-minY)/(maxY-minY||1)*(c.height-2*pad)};
}
function line(g,pts,color,X,Y){
  g.strokeStyle=color; g.lineWidth=1.6; g.beginPath();
  pts.forEach((p,i)=> i? g.lineTo(X(p[0]),Y(p[1])) : g.moveTo(X(p[0]),Y(p[1])));
  g.stroke();
}
function dashed(g,y,color,X,Y,maxX){
  g.strokeStyle=color; g.setLineDash([4,4]); g.beginPath();
  g.moveTo(X(0),Y(y)); g.lineTo(X(maxX),Y(y)); g.stroke(); g.setLineDash([]);
}
function drawMain(){
  const c=document.getElementById('chart'), g=c.getContext('2d'); g.clearRect(0,0,c.width,c.height);
  const rows=main.filter(d=>d.loss!==undefined); if(!rows.length) return;
  const evs=main.filter(d=>d.eval_loss!==undefined);
  const xs=main.map(d=>d.step), minX=Math.min(...xs), maxX=Math.max(...xs);
  const ys=rows.map(d=>d.loss).concat(evs.map(d=>d.eval_loss));
  const {X,Y}=frame(g,c,40,minX,maxX,Math.min(...ys),Math.max(...ys));
  line(g,rows.map(d=>[d.step,d.loss]),'#58a6ff',X,Y);
  evs.forEach(d=>{ g.fillStyle='#f0883e'; g.beginPath(); g.arc(X(d.step),Y(d.eval_loss),3,0,7); g.fill(); });
}
function drawDual(){
  const c=document.getElementById('chart'), g=c.getContext('2d'); g.clearRect(0,0,c.width,c.height);
  const rows=dual; if(!rows.length) return;
  const xs=rows.map(d=>d.step), minX=Math.min(...xs), maxX=Math.max(...xs);
  const ys=rows.map(d=>d.dev_user_ce).concat(rows.map(d=>d.dev_agent_ce));
  const {X,Y}=frame(g,c,40,minX,maxX,Math.min(...ys),Math.max(...ys));
  line(g,rows.map(d=>[d.step,d.dev_user_ce]),'#58a6ff',X,Y);
  line(g,rows.map(d=>[d.step,d.dev_agent_ce]),'#3fb950',X,Y);
}
function drawProbe(){
  const c=document.getElementById('chart'), g=c.getContext('2d'); g.clearRect(0,0,c.width,c.height);
  const rows=probe.filter(d=>d.dev_ce!==undefined); if(!rows.length) return;
  const gapMode = rows[0].real_ce!==undefined;
  const ce = d=> gapMode? d.real_ce : d.dev_ce;
  const sh = d=> gapMode? d.shuffled_ce : d.unigram_ce;
  const xs=rows.map(d=>d.step), minX=Math.min(...xs), maxX=Math.max(...xs);
  const ys=rows.map(ce).concat(rows.map(sh));
  const {X,Y}=frame(g,c,40,minX,maxX,Math.min(...ys),Math.max(...ys));
  line(g,rows.map(d=>[d.step,sh(d)]),'#f0883e',X,Y);
  line(g,rows.map(d=>[d.step,ce(d)]),'#3fb950',X,Y);
  const c2=document.getElementById('chart2'), g2=c2.getContext('2d'); g2.clearRect(0,0,c2.width,c2.height);
  const tops=rows.map(d=>(d.real_top1!==undefined? d.real_top1 : d.dev_top1)*100);
  const {X:X2,Y:Y2}=frame(g2,c2,40,minX,maxX,0,Math.max(...tops,1));
  line(g2,rows.map(d=>[d.step,d.dev_top1*100]),'#f0883e',X2,Y2);
}
async function tick(){
  const r=await fetch('/metrics'); const d=await r.json();
  main=d.main||[]; probe=d.probe||[]; dual=d.dual||[];
  const rows = MODE==='probe'? probe : MODE==='dual'? dual : main;
  if(rows.length){
    const last=rows[rows.length-1];
    document.getElementById('step').textContent=last.step;
    document.getElementById('fps').textContent=last.frames_s??'–';
    if(MODE==='dual'){
      document.getElementById('k1').textContent='user CE (aux)';
      document.getElementById('k2').textContent='agent CE (voice)';
      document.getElementById('k3').textContent='agent/user ratio';
      document.getElementById('v1').textContent=last.dev_user_ce;
      document.getElementById('v2').textContent=last.dev_agent_ce;
      document.getElementById('v3').textContent=(last.dev_agent_ce/last.dev_user_ce).toFixed(3);
      drawDual();
    } else if(MODE==='probe'){
      const gapMode = last.real_ce!==undefined;
      document.getElementById('k1').textContent = gapMode? 'dev CE (real user)' : 'dev CE';
      document.getElementById('k2').textContent = gapMode? 'dev CE (shuffled user)' : 'unigram floor';
      document.getElementById('k3').textContent = gapMode? 'listening gap' : 'dev top-1';
      document.getElementById('v1').textContent = gapMode? last.real_ce : last.dev_ce;
      document.getElementById('v2').textContent = gapMode? last.shuffled_ce : last.unigram_ce;
      document.getElementById('v3').textContent = gapMode? last.gap : (last.dev_top1*100).toFixed(1)+'%';
      drawProbe();
    } else if(MODE==='dual'){
      drawDual();
    } else {
      const evs=main.filter(x=>x.eval_loss!==undefined); const ev=evs.length?evs[evs.length-1]:null;
      document.getElementById('k1').textContent='loss';
      document.getElementById('k2').textContent='dev loss';
      document.getElementById('k3').textContent='dev top-1';
      document.getElementById('v1').textContent=last.loss??'–';
      document.getElementById('v2').textContent=ev?ev.eval_loss:'–';
      document.getElementById('v3').textContent=ev?(ev.eval_top1*100).toFixed(1)+'%':'–';
      drawMain();
    }
    if(last.s_step && TOTAL){ const rem=(TOTAL-last.step)*last.s_step;
      document.getElementById('eta').textContent=Math.floor(rem/3600)+'h '+Math.floor(rem%3600/60)+'m'; }
  }
  const ck=await (await fetch('/ckpts')).json();
  document.getElementById('ck').innerHTML=ck.map(c=>`<tr><td>${c.name}</td><td>${c.mb} MB</td><td>${c.time}</td></tr>`).join('');
}
tick(); setInterval(tick,5000);
</script></body></html>"""


def read_jsonl(p: Path, limit: int = 2000):
    out = []
    if p.exists():
        for line in p.read_text().splitlines()[-limit:]:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/metrics"):
            body = json.dumps({"main": read_jsonl(CKPT / "metrics.jsonl"),
                               "probe": read_jsonl(CKPT / "probe_metrics.jsonl"),
                               "dual": read_jsonl(CKPT / "dual_metrics.jsonl")}).encode()
            ctype = "application/json"
        elif self.path.startswith("/ckpts"):
            out = []
            for f in sorted(CKPT.glob("duplex-*.pt")):
                st = f.stat()
                out.append({"name": f.name, "mb": round(st.st_size / 1e6, 1),
                            "time": time.strftime("%H:%M:%S", time.localtime(st.st_mtime))})
            body = json.dumps(out).encode()
            ctype = "application/json"
        else:
            body = PAGE.replace("%TOTAL%", str(TOTAL)).replace("%MODE%", MODE).encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


MODE = "probe"
TOTAL = 800
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        MODE = sys.argv[1]
    if len(sys.argv) > 2:
        TOTAL = int(sys.argv[2])
    ThreadingHTTPServer(("0.0.0.0", 8080), H).serve_forever()
