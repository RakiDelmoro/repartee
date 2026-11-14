"use strict";

const $ = (id) => document.getElementById(id);
const state = { day: null, query: "" };

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

function clock(iso) {
  return iso && iso.length >= 16 ? iso.slice(11, 16) : "--:--";
}

function duration(seconds) {
  const s = Math.round(Number(seconds) || 0);
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

function el(tag, attrs = {}, text = "") {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text) node.textContent = text;
  return node;
}

function showDetail(row) {
  const body = $("detail-body");
  body.replaceChildren();
  body.append(el("h2", {}, clock(row.started_at) + " · " + row.device));
  body.append(el("div", { class: "muted" },
    `${duration(row.duration_s)} · ${row.id} · ${row.status}`));

  body.append(el("h3", {}, "Summary"));
  body.append(el("p", { class: row.summary ? "summary" : "muted" },
    row.summary || "no summary (transcript only)"));

  body.append(el("h3", {}, "Transcript"));
  body.append(el("p", { class: "transcript" }, row.transcript || ""));
}

async function openSession(id) {
  try {
    showDetail(await getJSON(`/api/sessions/${encodeURIComponent(id)}`));
    history.replaceState(null, "", `#s=${encodeURIComponent(id)}`);
    $("detail").classList.remove("hidden");
  } catch (err) {
    console.error(err);
  }
}

function sessionFromHash() {
  const raw = location.hash.replace(/^#s=/, "");
  return location.hash.startsWith("#s=") && raw ? decodeURIComponent(raw) : "";
}

function card(row) {
  const node = el("article", { class: "card", tabindex: "0", role: "button" });
  const head = el("header");
  head.append(el("strong", {}, clock(row.started_at)));
  head.append(el("span", { class: "muted" },
    `${row.device} · ${duration(row.duration_s)}`));
  node.append(head);
  node.append(el("p", { class: row.summary ? "" : "muted" },
    row.summary || "no summary (transcript only)"));
  node.addEventListener("click", () => openSession(row.id));
  node.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" || ev.key === " ") openSession(row.id);
  });
  return node;
}

function renderCards(rows, title, meta) {
  $("day-title").textContent = title;
  $("day-meta").textContent = meta;
  const cards = $("cards");
  cards.replaceChildren(...rows.map(card));
  $("empty").classList.toggle("hidden", rows.length > 0);
}

async function loadDay(day, label) {
  state.day = day;
  state.query = "";
  $("search").value = "";
  for (const li of document.querySelectorAll("#days li")) {
    li.classList.toggle("active", li.dataset.day === day);
  }
  const data = await getJSON(`/api/sessions?day=${encodeURIComponent(day)}`);
  const total = data.items.reduce((sum, r) => sum + (Number(r.duration_s) || 0), 0);
  renderCards(data.items, label || day,
    `${data.items.length} conversation(s) · ${duration(total)}`);
}

async function loadDays() {
  const data = await getJSON("/api/days");
  const list = $("days");
  list.replaceChildren();
  for (const d of data.items) {
    const li = el("li", { "data-day": d.day });
    li.append(el("span", { class: "day" }, d.day));
    li.append(el("span", { class: "muted" },
      `${d.sessions} · ${duration(d.seconds)}`));
    li.addEventListener("click", () => loadDay(d.day));
    list.append(li);
  }
  return data.items;
}

async function search(query) {
  state.query = query;
  if (!query) {
    const days = await loadDays();
    if (days.length) await loadDay(days[0].day);
    return;
  }
  try {
    const data = await getJSON(`/search?q=${encodeURIComponent(query)}&limit=50`);
    for (const li of document.querySelectorAll("#days li")) li.classList.remove("active");
    renderCards(data.items, `Search: ${query}`,
      `${data.items.length} hit(s) · click to open`);
  } catch (err) {
    renderCards([], `Search: ${query}`, String(err));
  }
}

function debounce(fn, ms) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

async function boot() {
  try {
    const stats = await getJSON("/api/stats");
    $("stats").textContent =
      `${stats.sessions} conversation(s) · ${stats.days} day(s)`;
  } catch (err) {
    $("stats").textContent = "stats unavailable";
  }
  const days = await loadDays();
  if (days.length) await loadDay(days[0].day);
  else {
    $("cards").replaceChildren();
    $("empty").classList.remove("hidden");
  }
  $("search").addEventListener("input", debounce((ev) => search(ev.target.value.trim()), 250));
  $("close").addEventListener("click", () => {
    $("detail").classList.add("hidden");
    history.replaceState(null, "", location.pathname);
  });
  window.addEventListener("hashchange", () => {
    const id = sessionFromHash();
    if (id) openSession(id);
  });
  const deep = sessionFromHash();
  if (deep) openSession(deep);
}

boot();
