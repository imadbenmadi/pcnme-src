// TrainingView: the learning methods' training (PLAN V4), finished or running, from what train.py writes per
// episode (sources/training.py). On the left every scenario x method (a running one first, marked live); on the
// right its summary (budget and time left, best and recent on-time, pretraining), curves with a rolling mean
// (hover for the value, click an episode for all its numbers), every episode in a table, and train.log. A
// running training is read again every few seconds, so each new episode appears as train.py writes it.

import { $, esc, methodName } from "../core/format.js";
import { C, alpha } from "../core/theme.js";

const POLL = 5000;
const ROLL = 10; // episodes in the rolling mean
const pct = (v) => (v == null ? "–" : `${(100 * v).toFixed(1)} %`);
const fix = (d) => (v) => (v == null ? "–" : (+v).toFixed(d));
const int = (v) => (v == null ? "–" : Math.round(v).toLocaleString("en"));
const dur = (s) => (s == null ? "–" : s < 90 ? `${s.toFixed(0)} s` : s < 5400 ? `${(s / 60).toFixed(0)} min` : `${(s / 3600).toFixed(1)} h`);
const CHARTS = [
  ["on_time", "On time per episode", pct, "share of the episode's tasks delivered by their deadline (16.3)"],
  ["mean_loss", "Mean loss", fix(3), "mean of the episode's update losses (DQfD: TD + n-step + margin)"],
  ["mean_e_hat", "Mean ê", fix(3), "energy over its reference, min(E / E_ref, ē) (13.2.4); lower is better"],
  ["updates", "Network updates (total)", int, "gradient steps so far"],
  ["tasks", "Tasks per episode", int, "tasks created in the episode"],
  ["episode_s", "Wall time per episode", dur, "how long the episode took (train.py's clock)"],
];
const FIELDS = [["episode", int], ["seed", int], ["on_time", pct], ["mean_e_hat", fix(3)], ["mean_loss", fix(4)], ["tasks", int],
  ["updates", int], ["decisions_used", int], ["episode_s", dur], ["wall_s", dur]];
const STATE = { running: "training now", done: "finished", stopped: "stopped before the end" };

function rolling(ys, n) {
  const out = []; let sum = 0, cnt = 0; const q = [];
  for (const y of ys) { q.push(y); if (y != null) { sum += y; cnt++; } if (q.length > n) { const o = q.shift(); if (o != null) { sum -= o; cnt--; } } out.push(cnt ? sum / cnt : null); }
  return out;
}

class Chart {
  constructor(canvas, ys, fmt, onPick) {
    Object.assign(this, { canvas, ys, fmt, onPick, hover: null });
    this.roll = rolling(ys, ROLL);
    canvas.addEventListener("mousemove", (e) => { this.hover = this.at(e); this.draw(); });
    canvas.addEventListener("mouseleave", () => { this.hover = null; this.draw(); });
    canvas.addEventListener("click", (e) => { const i = this.at(e); if (i != null) onPick(i, e); });
    this.draw();
  }
  at(e) {
    const b = this.canvas.getBoundingClientRect(), x = e.clientX - b.left, n = this.ys.length;
    if (!n || !this.geo) return null;
    const { left, right } = this.geo;
    return Math.max(0, Math.min(n - 1, Math.round(((x - left) / (right - left)) * (n - 1))));
  }
  draw() {
    const cv = this.canvas, box = cv.parentElement.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const w = Math.max(200, box.width - 24), h = 150;
    cv.width = w * dpr; cv.height = h * dpr; cv.style.height = `${h}px`;
    const g = cv.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h); g.font = `11px ${C.font}`;
    const ys = this.ys, pts = ys.map((y, i) => [i, y]).filter(([, y]) => y != null);
    if (!pts.length) { g.fillStyle = C.faint; g.fillText("no episodes yet", 12, h / 2); return; }
    const left = 8, right = w - 66, top = 10, bottom = h - 22;
    this.geo = { left, right };
    let lo = Math.min(...pts.map((p) => p[1])), hi = Math.max(...pts.map((p) => p[1]));
    if (hi - lo < 1e-12) { const d = Math.abs(hi) * 0.01 || 1e-6; lo -= d; hi += d; }
    const n = Math.max(2, ys.length);
    const X = (i) => left + (i / (n - 1)) * (right - left), Y = (v) => bottom - ((v - lo) / (hi - lo)) * (bottom - top);
    g.strokeStyle = C.border; g.lineWidth = 1;
    for (const v of [lo, (lo + hi) / 2, hi]) { g.beginPath(); g.moveTo(left, Y(v)); g.lineTo(right, Y(v)); g.stroke(); }
    g.fillStyle = C.faint;
    for (const v of [lo, hi]) g.fillText(this.fmt(v), right + 6, Y(v) + 4);
    g.fillText("episode 1", left, h - 6);
    const lastLabel = `${ys.length}`; g.fillText(lastLabel, right - g.measureText(lastLabel).width, h - 6);
    g.strokeStyle = alpha(C.accent, 0.35); g.lineWidth = 1; g.beginPath(); // the raw values
    pts.forEach(([i, v], k) => (k ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v)))); g.stroke();
    g.strokeStyle = C.accent; g.lineWidth = 2; g.beginPath(); let started = false; // the rolling mean
    this.roll.forEach((v, i) => { if (v == null) return; if (started) g.lineTo(X(i), Y(v)); else { g.moveTo(X(i), Y(v)); started = true; } }); g.stroke();
    const [li, lv] = pts[pts.length - 1];
    g.fillStyle = C.accent; g.beginPath(); g.arc(X(li), Y(lv), 3, 0, 2 * Math.PI); g.fill();
    if (this.hover != null && ys[this.hover] != null) {
      const i = this.hover, x = X(i), y = Y(ys[i]);
      g.strokeStyle = alpha(C.text, 0.3); g.beginPath(); g.moveTo(x, top); g.lineTo(x, bottom); g.stroke();
      g.fillStyle = C.text; g.beginPath(); g.arc(x, y, 3.5, 0, 2 * Math.PI); g.fill();
      const label = `ep ${i + 1}: ${this.fmt(ys[i])}`, tw = g.measureText(label).width + 12, tx = Math.min(Math.max(left, x - tw / 2), right - tw);
      g.fillStyle = C.surface; g.strokeStyle = C["border-strong"]; g.fillRect(tx, top, tw, 18); g.strokeRect(tx, top, tw, 18);
      g.fillStyle = C.text; g.fillText(label, tx + 6, top + 13);
    }
  }
}

export class TrainingView {
  constructor() { this.timer = null; this.listTimer = null; this.charts = []; }

  async open(adapter, want) {
    this.close();
    this.adapter = adapter;
    const root = $("training");
    root.hidden = false;
    root.innerHTML = `<nav class="train-list" id="train-list"></nav><section class="train-main" id="train-main"></section><div class="ep-pop" id="ep-pop" hidden></div>`;
    root.addEventListener("click", (e) => { if (e.target.closest("[data-close-pop]")) $("ep-pop").hidden = true; });
    this.onResize = () => this.charts.forEach((c) => c.draw());
    addEventListener("resize", this.onResize);
    await this.list(want);
    this.listTimer = setInterval(() => this.list(this.run), 15000);
  }

  async list(want) {
    let runs;
    try { runs = await this.adapter.listRuns(); } catch (e) { $("train-main").innerHTML = `<p class="muted">Could not read the training folders: ${esc(e.message)}</p>`; return; }
    const running = runs.filter((r) => r.state === "running");
    const byMap = {};
    for (const r of runs) (byMap[r.map] = byMap[r.map] || []).push(r);
    $("train-list").innerHTML = (running.length ? `<p class="live-note"><span class="live-state on">${running.length} training now</span></p>` : `<p class="muted small">Nothing is training now. A training started with train.py (RUN_TRAINING.md) appears here and updates by itself.</p>`) +
      Object.entries(byMap).map(([m, rs]) => `<h3>${esc(m === "vfc" ? "VFC grid" : m === "lust" ? "LuST" : m)}</h3>` + rs
        .sort((a, b) => (a.state === "running" ? -1 : b.state === "running" ? 1 : a.run.localeCompare(b.run)))
        .map((r) => `<button data-run="${esc(r.run)}" class="${r.state}"><span>${esc(r.scenario)} · ${esc(methodName(r.method))}</span><small>${r.state === "running" ? "● live" : `${r.episodes} ep.`}</small></button>`).join("")).join("");
    $("train-list").querySelectorAll("button").forEach((b) => (b.onclick = () => this.show(b.dataset.run)));
    if (!runs.length) { $("train-main").innerHTML = `<p class="muted">No training yet.</p>`; return; }
    const pick = runs.some((r) => r.run === want) ? want : (running[0] || runs[0]).run;
    if (pick !== this.run) this.show(pick);
    else this.mark();
  }

  mark() { $("train-list").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.run === this.run))); }

  async show(run) {
    clearTimeout(this.timer);
    const first = run !== this.run;
    this.run = run;
    this.mark();
    history.replaceState(null, "", `#training/${encodeURIComponent(run)}`);
    let c;
    try { c = await this.adapter.curve(run); } catch (e) { $("train-main").innerHTML = `<p class="muted">${esc(e.message)}</p>`; return; }
    if (run !== this.run) return;
    const scroll = first ? 0 : $("train-main").scrollTop;
    this.render(c);
    $("train-main").scrollTop = scroll;
    if (c.state === "running") this.timer = setTimeout(() => this.show(run), POLL);
  }

  summary(c) {
    const e = c.episodes, n = e.episode.length, b = c.budget || {}, d = c.details || {};
    const on = e.on_time.filter((v) => v != null);
    const best = on.length ? Math.max(...on) : null, bestEp = best == null ? null : e.on_time.indexOf(best) + 1;
    const last10 = on.slice(-10), mean10 = last10.length ? last10.reduce((a, x) => a + x, 0) / last10.length : null;
    const wall = n ? e.wall_s[n - 1] : null; // train.py's wall_s counts from the session's start
    const used = b.used ?? e.decisions_used[n - 1], left = b.total && used != null ? b.total - used : null;
    const rate = n >= 2 && e.episode_s[n - 1] ? (e.decisions_used[n - 1] - e.decisions_used[n - 2]) / e.episode_s[n - 1] : null;
    const eta = c.state === "running" && left != null && rate ? left / rate : null;
    const tile = (label, value, hint = "") => `<div class="tile" title="${esc(hint)}"><b>${value}</b><span>${esc(label)}</span></div>`;
    return `<div class="tiles wide">
      ${tile("episodes", int(n))}
      ${tile("on time, last 10 ep.", pct(mean10))}
      ${tile(`best on time (ep. ${bestEp ?? "–"})`, pct(best))}
      ${tile("last loss", fix(3)(e.mean_loss[n - 1]))}
      ${tile("updates", int(e.updates[n - 1]))}
      ${tile("training time", dur(wall))}
      ${c.state === "running" ? tile("time left (estimate)", dur(eta), "decisions left over the last episode's speed") : ""}
      ${d.w_star != null ? tile("w* (time weight)", fix(3)(d.w_star), "the median knee weight of NSGA-II (13.3)") : ""}
      ${d.expert_transitions != null ? tile("expert transitions", int(d.expert_transitions), "NSGA-II demonstrations for DQfD (13.4)") : ""}
      ${d.agreement != null ? tile("expert agreement", pct(d.agreement), "held-out agreement with the expert after pretraining (13.5)") : ""}
    </div>`;
  }

  render(c) {
    const e = c.episodes, n = e.episode.length, b = c.budget;
    e.episode_s = e.wall_s.map((w, i) => (w == null ? null : i && e.wall_s[i - 1] != null && w >= e.wall_s[i - 1] ? w - e.wall_s[i - 1] : w));
    const used = b && b.total ? Math.min(1, b.used / b.total) : null;
    $("train-main").innerHTML = `
      <header class="train-head">
        <div><div class="eyebrow">Training · ${esc(c.run.split("/")[0] === "vfc" ? "VFC grid" : c.run.split("/")[0])}</div>
          <h2>${esc(methodName(c.method))} <small>${esc(c.scenario)}</small></h2>
          <span class="live-state ${c.state === "running" ? "on" : ""}">${esc(STATE[c.state] || c.state)}${c.state === "running" ? " · refreshes every 5 s" : ""}</span></div>
        ${used == null ? "" : `<div class="budget"><span>Decision budget ${(100 * used).toFixed(1)} %</span><div class="bar"><i style="width:${(100 * used).toFixed(1)}%"></i></div>
          <small>${int(b.used)} / ${int(b.total)} decisions (16.1: the same for every method)</small></div>`}
      </header>
      ${this.summary(c)}
      <p class="muted small">Faint line: each episode; bold: the mean of the last ${ROLL}. Hover a curve for its value, click it for that episode.</p>
      <div class="charts">${CHARTS.map(([k, label, f, hint]) => `<figure title="${esc(hint)}"><figcaption>${label}<b>${n ? f(e[k][n - 1]) : "–"}</b></figcaption><canvas data-k="${k}"></canvas></figure>`).join("")}</div>
      <h3>Episodes</h3>
      <div class="ep-table"><table><thead><tr>${FIELDS.map(([k]) => `<th>${k.replace(/_/g, " ")}</th>`).join("")}</tr></thead>
        <tbody>${[...Array(n).keys()].reverse().map((i) => `<tr data-ep="${i}">${FIELDS.map(([k, f]) => `<td>${f(e[k][i])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
      <h3>train.log</h3><pre class="log">${esc(c.log.join("\n"))}</pre>`;
    this.curve = c;
    const pick = (i, ev) => this.episode(i, ev);
    this.charts = [...$("train-main").querySelectorAll("canvas")].map((cv) => new Chart(cv, e[cv.dataset.k], CHARTS.find((x) => x[0] === cv.dataset.k)[2], pick));
    $("train-main").querySelectorAll("tr[data-ep]").forEach((tr) => (tr.onclick = (ev) => this.episode(+tr.dataset.ep, ev)));
    const log = $("train-main").querySelector(".log");
    log.scrollTop = log.scrollHeight;
  }

  // a popup with every number of episode i, and the change from the episode before
  episode(i, ev) {
    const e = this.curve.episodes, pop = $("ep-pop"), n = e.episode.length;
    const rank = e.on_time[i] == null ? null : 1 + e.on_time.filter((v) => v != null && v > e.on_time[i]).length;
    const delta = (k, f) => (i > 0 && e[k][i] != null && e[k][i - 1] != null ? ` <small>${e[k][i] >= e[k][i - 1] ? "+" : "−"}${f(Math.abs(e[k][i] - e[k][i - 1]))}</small>` : "");
    pop.innerHTML = `<div class="panel-head"><div><div class="eyebrow">Episode</div><h2>${i + 1} of ${n}</h2>
      <p>${esc(methodName(this.curve.method))} · ${esc(this.curve.scenario)}${rank ? ` · on-time rank ${rank} of ${n}` : ""}</p></div>
      <button class="icon-btn" data-close-pop aria-label="Close"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 4l8 8M12 4l-8 8"/></svg></button></div>
      <div class="panel-body"><dl class="kv">${FIELDS.map(([k, f]) => `<dt>${k.replace(/_/g, " ")}</dt><dd>${f(e[k][i])}${k === "on_time" ? delta(k, (x) => `${(100 * x).toFixed(1)} pp`) : k === "mean_loss" || k === "mean_e_hat" ? delta(k, fix(3)) : ""}</dd>`).join("")}</dl>
      <p class="note">Each episode is one simulated run on its own training seed; the network keeps learning across them.</p></div>`;
    pop.hidden = false;
    const host = $("training").getBoundingClientRect();
    const x = Math.min(ev.clientX - host.left + 12, host.width - pop.offsetWidth - 12), y = Math.min(ev.clientY - host.top - 10, host.height - pop.offsetHeight - 12);
    pop.style.left = `${Math.max(12, x)}px`; pop.style.top = `${Math.max(12, y)}px`;
  }

  close() {
    clearTimeout(this.timer); clearInterval(this.listTimer);
    if (this.onResize) removeEventListener("resize", this.onResize);
    this.run = null; this.charts = [];
    $("training").hidden = true;
    $("training").innerHTML = "";
  }
}
