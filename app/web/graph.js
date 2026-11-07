/* Obsidian-style force graph for the knowledge base. No dependencies. */

const COLORS = {
  session: "#a78bfa",
  decision: "#22c55e",
  action: "#3b82f6",
  question: "#f59e0b",
  topic: "#94a3b8",
  person: "#ec4899",
  project: "#14b8a6",
};

const canvas = document.getElementById("graph");
const ctx = canvas.getContext("2d");
const emptyEl = document.getElementById("empty");
const statsEl = document.getElementById("stats");
const filtersEl = document.getElementById("filters");
const sessionsEl = document.getElementById("sessions");
const detailEl = document.getElementById("detail");
const detailBody = document.getElementById("detail-body");

const view = { x: 0, y: 0, k: 1 };
const state = {
  nodes: [],
  links: [],
  byId: new Map(),
  alpha: 1,
  hover: null,
  selected: null,
  hidden: new Set(),
  session: null,
  width: 0,
  height: 0,
  dpr: 1,
  pointer: { x: 0, y: 0, down: false, node: null, panning: false },
};

/* ---------------------------------------------------------------- data */

function degrees(links) {
  const degree = new Map();
  for (const link of links) {
    degree.set(link.source, (degree.get(link.source) || 0) + 1);
    degree.set(link.target, (degree.get(link.target) || 0) + 1);
  }
  return degree;
}

async function load() {
  const data = await fetch("/api/graph").then((r) => r.json());
  const degree = degrees(data.links);
  const golden = Math.PI * (3 - Math.sqrt(5));

  state.nodes = data.nodes.map((node, index) => {
    const angle = index * golden;
    const radius = 40 + 22 * Math.sqrt(index);
    const deg = degree.get(node.id) || 1;
    return {
      ...node,
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
      vx: 0,
      vy: 0,
      r: node.type === "session" ? 13 : 5 + Math.min(9, deg * 1.6),
      degree: deg,
      neighbours: new Set(),
    };
  });

  state.byId = new Map(state.nodes.map((n) => [n.id, n]));

  state.links = data.links
    .map((link) => ({ ...link, a: state.byId.get(link.source), b: state.byId.get(link.target) }))
    .filter((link) => link.a && link.b);

  for (const link of state.links) {
    link.a.neighbours.add(link.b.id);
    link.b.neighbours.add(link.a.id);
  }

  buildFilters();
  buildSessions();

  emptyEl.classList.toggle("hidden", state.nodes.length > 0);
  const counts = state.nodes.reduce((acc, n) => ((acc[n.type] = (acc[n.type] || 0) + 1), acc), {});
  statsEl.textContent = Object.entries(counts)
    .map(([type, n]) => `${n} ${type}${n > 1 ? "s" : ""}`)
    .join(" · ");

  state.alpha = 1;
  fitView();
  requestAnimationFrame(frame);
}

/* ------------------------------------------------------------- filters */

function buildFilters() {
  const types = [...new Set(state.nodes.map((n) => n.type))].sort();
  filtersEl.innerHTML = "";
  for (const type of types) {
    const label = document.createElement("label");
    label.className = "filter";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = true;
    box.addEventListener("change", () => {
      if (box.checked) state.hidden.delete(type);
      else state.hidden.add(type);
      restart();
    });
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = COLORS[type] || "#888";
    label.append(box, dot, document.createTextNode(type));
    filtersEl.append(label);
  }
}

function buildSessions() {
  const sessions = state.nodes.filter((n) => n.type === "session");
  const counts = new Map();
  for (const node of state.nodes) {
    if (node.session && node.type !== "session") {
      counts.set(node.session, (counts.get(node.session) || 0) + 1);
    }
  }

  sessionsEl.innerHTML = "";
  sessions
    .sort((a, b) => b.label.localeCompare(a.label))
    .forEach((session) => {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = session.label;
      const count = document.createElement("span");
      count.className = "count";
      count.textContent = counts.get(session.id) || 0;
      li.append(name, count);
      li.addEventListener("click", () => {
        state.session = state.session === session.id ? null : session.id;
        [...sessionsEl.children].forEach((el) => el.classList.remove("active"));
        if (state.session) li.classList.add("active");
        restart();
      });
      sessionsEl.append(li);
    });
}

/* ---------------------------------------------------------- simulation */

function visible(node) {
  if (state.hidden.has(node.type)) return false;
  if (state.session && node.session && node.session !== state.session) return false;
  return true;
}

function simulate() {
  const nodes = state.nodes.filter(visible);
  const index = new Set(nodes.map((n) => n.id));
  const alpha = state.alpha;

  for (let i = 0; i < nodes.length; i++) {
    const a = nodes[i];
    for (let j = i + 1; j < nodes.length; j++) {
      const b = nodes[j];
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      let d2 = dx * dx + dy * dy;
      if (d2 < 1) {
        d2 = 1;
        dx = Math.random() - 0.5;
        dy = Math.random() - 0.5;
      }
      const force = 2600 / d2;
      const d = Math.sqrt(d2);
      const fx = (dx / d) * force * alpha;
      const fy = (dy / d) * force * alpha;
      a.vx -= fx;
      a.vy -= fy;
      b.vx += fx;
      b.vy += fy;
    }
  }

  for (const link of state.links) {
    if (!index.has(link.a.id) || !index.has(link.b.id)) continue;
    const dx = link.b.x - link.a.x;
    const dy = link.b.y - link.a.y;
    const d = Math.sqrt(dx * dx + dy * dy) || 1;
    const target = link.type === "contains" ? 105 : 70;
    const force = ((d - target) / d) * 0.035 * alpha;
    link.a.vx += dx * force;
    link.a.vy += dy * force;
    link.b.vx -= dx * force;
    link.b.vy -= dy * force;
  }

  for (const node of nodes) {
    if (node.fixed) continue;
    node.vx -= node.x * 0.0035 * alpha;
    node.vy -= node.y * 0.0035 * alpha;
    node.vx *= 0.82;
    node.vy *= 0.82;
    node.x += node.vx;
    node.y += node.vy;
  }

  state.alpha = Math.max(0, state.alpha * 0.985);
}

function restart() {
  state.alpha = 0.9;
  requestAnimationFrame(frame);
}

/* ------------------------------------------------------------ rendering */

function frame() {
  if (state.alpha > 0.004) simulate();
  draw();
  if (state.alpha > 0.004) requestAnimationFrame(frame);
}

function draw() {
  const { width, height, dpr } = state;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  ctx.save();
  ctx.translate(width / 2 + view.x, height / 2 + view.y);
  ctx.scale(view.k, view.k);

  const focus = state.hover || state.selected;
  const related = focus ? new Set([focus.id, ...focus.neighbours]) : null;
  const shown = new Set(state.nodes.filter(visible).map((n) => n.id));

  ctx.lineWidth = 1 / view.k;
  for (const link of state.links) {
    if (!shown.has(link.a.id) || !shown.has(link.b.id)) continue;
    const active = !related || (related.has(link.a.id) && related.has(link.b.id));
    ctx.strokeStyle = active ? "rgba(160,180,210,0.35)" : "rgba(120,135,160,0.08)";
    ctx.beginPath();
    ctx.moveTo(link.a.x, link.a.y);
    ctx.lineTo(link.b.x, link.b.y);
    ctx.stroke();
  }

  for (const node of state.nodes) {
    if (!shown.has(node.id)) continue;
    const active = !related || related.has(node.id);
    const color = COLORS[node.type] || "#8899aa";

    if (node === state.selected) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.r + 6, 0, Math.PI * 2);
      ctx.fillStyle = color + "33";
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
    ctx.fillStyle = active ? color : color + "33";
    ctx.fill();

    const showLabel =
      active &&
      (node.type === "session" || node === focus || node.degree > 3 || view.k > 1.4);
    if (showLabel) {
      ctx.font = `${12 / view.k}px ui-sans-serif, system-ui, sans-serif`;
      ctx.fillStyle = active ? "#dfe6f0" : "#5a6573";
      ctx.textAlign = "center";
      const text = node.label.length > 46 ? node.label.slice(0, 46) + "…" : node.label;
      ctx.fillText(text, node.x, node.y + node.r + 13 / view.k);
    }
  }

  ctx.restore();
}

/* --------------------------------------------------------- interaction */

function toWorld(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left - state.width / 2 - view.x;
  const y = clientY - rect.top - state.height / 2 - view.y;
  return { x: x / view.k, y: y / view.k };
}

function nodeAt(point) {
  let best = null;
  let bestDist = Infinity;
  for (const node of state.nodes) {
    if (!visible(node)) continue;
    const dx = node.x - point.x;
    const dy = node.y - point.y;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d < node.r + 6 && d < bestDist) {
      best = node;
      bestDist = d;
    }
  }
  return best;
}

canvas.addEventListener("mousedown", (event) => {
  const world = toWorld(event.clientX, event.clientY);
  const node = nodeAt(world);
  state.pointer.down = true;
  if (node) {
    state.pointer.node = node;
    node.fixed = true;
    canvas.classList.add("dragging");
  } else {
    state.pointer.panning = true;
    state.pointer.startX = event.clientX - view.x;
    state.pointer.startY = event.clientY - view.y;
    canvas.classList.add("dragging");
  }
});

window.addEventListener("mousemove", (event) => {
  if (state.pointer.down && state.pointer.node) {
    const world = toWorld(event.clientX, event.clientY);
    state.pointer.node.x = world.x;
    state.pointer.node.y = world.y;
    state.pointer.node.vx = 0;
    state.pointer.node.vy = 0;
    draw();
    return;
  }
  if (state.pointer.panning) {
    view.x = event.clientX - state.pointer.startX;
    view.y = event.clientY - state.pointer.startY;
    draw();
    return;
  }
  const hovered = nodeAt(toWorld(event.clientX, event.clientY));
  if (hovered !== state.hover) {
    state.hover = hovered;
    draw();
  }
  canvas.style.cursor = hovered ? "pointer" : "grab";
});

window.addEventListener("mouseup", () => {
  if (state.pointer.node) {
    state.pointer.node.fixed = false;
    state.pointer.node = null;
    restart();
  }
  state.pointer.down = false;
  state.pointer.panning = false;
  canvas.classList.remove("dragging");
});

canvas.addEventListener("click", (event) => {
  const node = nodeAt(toWorld(event.clientX, event.clientY));
  if (!node) {
    state.selected = null;
    detailEl.classList.add("hidden");
    draw();
    return;
  }
  state.selected = node;
  showDetail(node);
  draw();
});

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const cx = event.clientX - rect.left - state.width / 2;
  const cy = event.clientY - rect.top - state.height / 2;
  const factor = Math.exp(-event.deltaY * 0.0015);
  const k = Math.min(4, Math.max(0.15, view.k * factor));
  const ratio = k / view.k;
  view.x = cx - (cx - view.x) * ratio;
  view.y = cy - (cy - view.y) * ratio;
  view.k = k;
  draw();
});

document.getElementById("close").addEventListener("click", () => {
  state.selected = null;
  detailEl.classList.add("hidden");
  draw();
});

/* -------------------------------------------------------------- detail */

async function showDetail(node) {
  detailBody.innerHTML = "";
  const badge = document.createElement("span");
  badge.className = "badge";
  badge.style.background = COLORS[node.type] || "#889";
  badge.textContent = node.type;
  detailBody.append(badge);

  const title = document.createElement("h3");
  title.textContent = node.label;
  detailBody.append(title);

  if (node.type === "session") {
    const meta = document.createElement("div");
    meta.className = "muted";
    meta.textContent = node.label;
    detailBody.append(meta);
    detailEl.classList.remove("hidden");
    return;
  }

  try {
    const event = await fetch(`/api/events/${encodeURIComponent(node.id)}`).then((r) => r.json());
    if (event.evidence) {
      const quote = document.createElement("blockquote");
      quote.textContent = `“${event.evidence}”`;
      detailBody.append(quote);
    }
    const list = document.createElement("dl");
    const rows = [
      ["session", event.session_id || "—"],
      ["owner", event.owner || "—"],
      ["due", event.due || "—"],
      ["at", event.t_start != null ? `${Math.round(event.t_start)}s` : "—"],
    ];
    for (const [key, value] of rows) {
      const dt = document.createElement("dt");
      dt.textContent = key;
      const dd = document.createElement("dd");
      dd.textContent = value;
      list.append(dt, dd);
    }
    detailBody.append(list);
  } catch {
    /* keep the basic panel */
  }

  detailEl.classList.remove("hidden");
}

/* --------------------------------------------------------------- setup */

function resize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  state.dpr = window.devicePixelRatio || 1;
  state.width = rect.width;
  state.height = rect.height;
  canvas.width = Math.round(rect.width * state.dpr);
  canvas.height = Math.round(rect.height * state.dpr);
  draw();
}

function fitView() {
  if (!state.nodes.length) return;
  const xs = state.nodes.map((n) => n.x);
  const ys = state.nodes.map((n) => n.y);
  const spanX = Math.max(...xs) - Math.min(...xs) || 1;
  const spanY = Math.max(...ys) - Math.min(...ys) || 1;
  view.k = Math.min(state.width / (spanX + 200), state.height / (spanY + 200), 1.6);
  view.k = Math.max(0.25, view.k);
  view.x = 0;
  view.y = 0;
}

new ResizeObserver(resize).observe(canvas.parentElement);
resize();
load().catch((error) => {
  statsEl.textContent = "failed to load graph";
  console.error(error);
});
