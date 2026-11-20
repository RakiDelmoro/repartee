(() => {
  const $ = (id) => document.getElementById(id);
  const COLORS = ['#39d0ff','#46e6a0','#f4c66a','#a78bfa','#ff9f6e','#7dd3fc','#f472b6','#c4b5fd'];
  const GRID = '#1d2b38';
  const MUTED = '#6f8496';
  let status = null;
  let data = null;

  const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(Number(v))) ? '—' : Number(v).toFixed(d);
  const int = (v) => (v === null || v === undefined) ? '—' : Math.round(v).toLocaleString();
  const dur = (s) => {
    if (s === null || s === undefined) return '—';
    s = Math.max(0, Math.round(s));
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60), h = Math.floor(m / 60);
    if (h) return `${h}h ${m % 60}m`;
    return `${m}m ${s % 60}s`;
  };
  function ema(points, alpha = 0.08) {
    let prev = null;
    return points.map((p) => {
      prev = prev === null ? p.y : prev + alpha * (p.y - prev);
      return { x: p.x, y: prev };
    });
  }
  function niceMax(v) {
    if (!isFinite(v) || v <= 0) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    const n = v / p;
    const m = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
    return m * p;
  }
  function setBar(id, pct) { const el = $(id); if (el) el.style.width = `${Math.max(0, Math.min(100, pct || 0))}%`; }
  function setText(id, text) { const el = $(id); if (el) el.textContent = text; }

  function chart(canvas, series, opts = {}) {
    if (!canvas) return;
    const box = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(200, box.width), h = Math.max(120, box.height);
    canvas.width = Math.floor(w * dpr); canvas.height = Math.floor(h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const pad = { l: 42, r: 12, t: 10, b: 24 };
    const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    const clean = series.map((s) => ({ ...s, data: s.data.filter((p) => isFinite(p.x) && isFinite(p.y)) })).filter((s) => s.data.length);
    const all = clean.flatMap((s) => s.data);
    if (!all.length) {
      ctx.fillStyle = MUTED; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center';
      ctx.fillText('waiting for training log…', w / 2, h / 2);
      return;
    }
    const xmax = opts.xmax || Math.max(...all.map((p) => p.x), 1);
    const ymax = opts.ymax || niceMax(Math.max(...all.map((p) => p.y)) * 1.08);
    const X = (x) => pad.l + (x / xmax) * iw;
    const Y = (y) => pad.t + ih - (y / ymax) * ih;

    ctx.lineWidth = 1; ctx.strokeStyle = GRID;
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = MUTED;
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (ih * i) / 4;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(fmt(ymax * (1 - i / 4), ymax < 10 ? 2 : 1), pad.l - 8, y);
    }
    for (let i = 0; i <= 3; i++) {
      const x = pad.l + (iw * i) / 3;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ih); ctx.stroke();
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(int(xmax * i / 3), x, h - 18);
    }
    for (const s of clean) {
      ctx.beginPath();
      s.data.forEach((p, i) => { if (i) ctx.lineTo(X(p.x), Y(p.y)); else ctx.moveTo(X(p.x), Y(p.y)); });
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.7;
      ctx.globalAlpha = s.alpha === undefined ? 1 : s.alpha;
      ctx.setLineDash(s.dash || []);
      ctx.stroke();
      ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
  }

  function badge(state) {
    const el = $('state');
    el.className = 'pill ' + (['training','tokenizing','starting','finished'].includes(state) ? state : state === 'waiting' ? '' : '');
    const label = { training: 'training', tokenizing: 'tokenizing', starting: 'starting', waiting: 'waiting', finished: 'finished' }[state] || state;
    el.textContent = label;
  }

  function updateStatus(s) {
    status = s;
    const total = s.steps_total || 5000, step = s.step || 0;
    setText('step', int(step)); setText('total', int(total));
    const p0 = s.p0 || {};
    if (isFinite(p0.hours)) {
      setText('run-sub', `Qwen3-0.6B + depth · ${Number(p0.hours).toFixed(1)} h Open Yap`);
    }
    setText('phase', (!s.process ? (s.p0 && s.p0.done < s.p0.target ? 'tokenizing' : 'waiting') : (step ? 'training' : 'starting')));
    setText('epoch', fmt(s.epochs, 2));
    setText('accum', s.accum || 2);
    setText('eta', s.eta === null ? '—' : dur(s.eta));
    setText('loss', s.loss === null ? '—' : fmt(s.loss, 3));
    setText('text', s.text === null ? '—' : fmt(s.text, 3));
    setText('audio', s.cb ? fmt(s.cb.reduce((a, b) => a + b, 0) / s.cb.length, 3) : '—');
    setText('skip', int(s.skip));
    setText('progress', '');
    $('progress').style.width = `${Math.min(100, (s.progress || 0) * 100)}%`;
    badge(s.state || 'waiting');

    const chartX = (r) => (isFinite(r.chart_step) ? r.chart_step : r.step);
    const recs = (s.series || []).filter((r) => isFinite(r.loss) && isFinite(chartX(r)));
    const firstX = recs.length ? chartX(recs[0]) : 0;
    const chartSpan = s.chart_max || total;
    const x = (r) => chartX(r) - firstX;
    const totalPts = recs.map((r) => ({ x: x(r), y: r.loss }));
    const textPts = recs.map((r) => ({ x: x(r), y: r.text }));
    chart($('loss-chart'), [
      { color: '#39d0ff', alpha: 0.25, width: 1.1, data: totalPts },
      { color: '#39d0ff', width: 2.2, data: ema(totalPts) },
      { color: '#f4c66a', width: 1.6, dash: [5, 4], data: ema(textPts) },
    ], { xmax: chartSpan, ymax: 16 });
    const cbSeries = [];
    for (let i = 0; i < 8; i++) {
      const pts = recs.filter((r) => r.cb && isFinite(r.cb[i])).map((r) => ({ x: x(r), y: r.cb[i] }));
      cbSeries.push({ color: COLORS[i], width: 1.8, data: ema(pts) });
    }
    chart($('cb-chart'), cbSeries, { xmax: chartSpan, ymax: 9 });

    const tail = s.tail || [];
    const log = $('log');
    const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    log.textContent = tail.length ? tail.join('\n') : 'waiting for training log…';
    if (atBottom) log.scrollTop = log.scrollHeight;
    setText('updated', s.updated ? `updated ${new Date(s.updated).toLocaleTimeString()}` : '—');
  }

  function updateSystem(sys) {
    const g = sys.gpu || {};
    if (!g.error) {
      setText('gpu-name', g.name);
      setText('gpu-util', `${Math.round(g.util)}%`);
      setText('gpu-power', `${fmt(g.power, 0)} W / ${fmt(g.power_limit, 0)} W`);
      setText('gpu-temp', `${fmt(g.temp, 0)} °C`);
      $('gpu-ring').style.setProperty('--p', Math.round(g.util));
      setText('kpi-gpu', `${Math.round(g.util)}%`);
      setText('kpi-gpu-sub', g.name);
      const vp = g.memory_total ? (g.memory_used / g.memory_total) * 100 : 0;
      setBar('vram-bar', vp);
      setText('vram-label', `${fmt(g.memory_used / 1024, 1)} / ${fmt(g.memory_total / 1024, 1)} GB`);
      setText('kpi-vram', `${fmt(g.memory_used / 1024, 1)} GB`);
      setText('kpi-vram-sub', `${Math.round(vp)}% of ${fmt(g.memory_total / 1024, 0)} GB`);
    } else {
      setText('gpu-name', 'GPU unavailable'); setText('kpi-gpu', '—');
    }
    const c = sys.cpu || {}, r = sys.ram || {}, d = sys.disk || {};
    setText('kpi-cpu', `${fmt(c.percent, 0)}%`);
    setText('kpi-cpu-sub', `${c.cores || '—'} threads · ${fmt(c.freq_mhz, 0)} MHz`);
    setBar('cpu-bar', c.percent);
    setText('cpu-label', `${fmt(c.percent, 0)}%`);
    setText('kpi-ram', `${fmt(r.used_gb, 1)} / ${fmt(r.total_gb, 0)} GB`);
    setText('kpi-ram-sub', `${fmt(r.percent, 0)}% used`);
    setBar('ram-bar', r.percent);
    setText('ram-label', `${fmt(r.used_gb, 1)} GB`);
    setBar('disk-bar', d.percent);
    setText('disk-label', `${fmt(d.used_gb, 0)} / ${fmt(d.total_gb, 0)} GB`);
    setText('host', (sys.host && sys.host.platform) || '—');
    const cores = $('cores');
    const per = c.per_core || [];
    if (cores.childElementCount !== per.length) {
      cores.innerHTML = '';
      for (let i = 0; i < per.length; i++) { const el = document.createElement('div'); el.className = 'core'; cores.appendChild(el); }
    }
    [...cores.children].forEach((el, i) => { el.style.height = `${Math.max(2, per[i] || 0)}%`; });
    const p = sys.process;
    setText('proc-line', p ? `training pid ${p.pid} · ${fmt(p.cpu, 0)}% CPU · ${fmt(p.rss_gb, 2)} GB RSS · ${dur(p.elapsed)}` : 'training process — not running');
  }

  function updateData(d) {
    data = d;
    const p0 = d.p0 || {};
    setText('p0-done', int(p0.done)); setText('p0-target', int(p0.target));
    setText('p0-hours', `${fmt(p0.hours, 1)} h`);
    setText('p0-frames', int(p0.frames));
    setText('p0-grids', int(d.grids));
    setText('p0-latest', p0.latest || '—');
    setText('p0-snr', p0.latest_snr === null || p0.latest_snr === undefined ? '—' : `${fmt(p0.latest_snr, 2)} dB`);
    setText('kpi-p0', `${int(p0.done)} / ${int(p0.target)}`);
    setText('kpi-p0-sub', `${fmt(p0.hours, 1)} h tokenized`);
    setText('kpi-data', `${fmt(p0.hours, 1)} h`);
    setText('kpi-data-sub', `${int(d.grids)} grids`);
    setText('log-size', `${fmt((d.log_bytes || 0) / 1024, 1)} KB`);
    setText('data-updated', `updated ${new Date().toLocaleTimeString()}`);
    const list = $('checkpoints');
    const cps = d.checkpoints || [];
    list.innerHTML = cps.length ? cps.map((c) => `<div class="checkpoint"><span class="name">${c.name}</span><span class="meta">${c.gb} GB · ${c.mtime.slice(11)}</span></div>`).join('') : '<div class="empty">No checkpoints yet</div>';
  }

  async function refresh() {
    try {
      const [s, sys, d] = await Promise.all([
        fetch('/api/status').then((r) => r.json()),
        fetch('/api/system').then((r) => r.json()),
        fetch('/api/data').then((r) => r.json()),
      ]);
      updateStatus(s); updateSystem(sys); updateData(d);
    } catch (e) {
      badge('error'); setText('state', 'offline');
    }
  }

  function legend() {
    $('cb-legend').innerHTML = Array.from({ length: 8 }, (_, i) => `<span class="cb-key"><i style="background:${COLORS[i]}"></i>cb${i}</span>`).join('');
  }

  function clock() { setText('clock', new Date().toLocaleTimeString()); }
  window.addEventListener('resize', () => { if (status) updateStatus(status); });
  legend(); clock(); refresh();
  setInterval(clock, 1000);
  setInterval(refresh, 2000);
})();
