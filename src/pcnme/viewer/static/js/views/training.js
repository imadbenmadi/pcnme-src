// TrainingView: the learning methods' training (PLAN V4), finished or running, from what train.py writes per
// episode (sources/training.py). A list of every scenario x method on the left (a running one first, marked
// live); on the right its curves (on-time share, loss, mean ê, updates), the decision budget and the log's last
// lines. A running training is read again every few seconds, so each new episode appears as train.py writes it.

import { $, esc, methodName } from "../core/format.js";
import { C } from "../core/theme.js";

const POLL = 5000;
const CHARTS = [
  ["on_time", "On time (episode)", (v) => `${(100 * v).toFixed(1)} %`],
  ["mean_loss", "Mean loss", (v) => v.toFixed(3)],
  ["mean_e_hat", "Mean ê (energy / reference)", (v) => v.toFixed(3)],
  ["updates", "Network updates (total)", (v) => Math.round(v).toLocaleString("en")],
];
const STATE = { running: "training now", done: "finished", stopped: "stopped before the end" };

function chart(canvas, ys, fmt) {
  const box = canvas.parentElement.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const w = Math.max(200, box.width), h = 150;
  canvas.width = w * dpr; canvas.height = h * dpr; canvas.style.height = `${h}px`;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  const pts = ys.map((y, i) => [i + 1, y]).filter(([, y]) => y != null);
  g.font = `11px ${C.font}`;
  if (!pts.length) { g.fillStyle = C.faint; g.fillText("no episodes yet", 12, h / 2); return; }
  const left = 8, right = w - 64, top = 10, bottom = h - 22;
  let lo = Math.min(...pts.map((p) => p[1])), hi = Math.max(...pts.map((p) => p[1]));
  if (hi - lo < 1e-9) { lo -= 0.5 * Math.abs(lo || 1) * 0.01 + 1e-6; hi += 0.5 * Math.abs(hi || 1) * 0.01 + 1e-6; }
  const n = Math.max(2, pts[pts.length - 1][0]);
  const X = (i) => left + ((i - 1) / (n - 1)) * (right - left), Y = (v) => bottom - ((v - lo) / (hi - lo)) * (bottom - top);
  g.strokeStyle = C.border; g.lineWidth = 1;
  for (const v of [lo, hi]) { g.beginPath(); g.moveTo(left, Y(v)); g.lineTo(right, Y(v)); g.stroke(); }
  g.fillStyle = C.faint;
  g.fillText(fmt(hi), right + 6, Y(hi) + 4); g.fillText(fmt(lo), right + 6, Y(lo) + 4);
  g.fillText("episode 1", left, h - 6);
  const lastLabel = `${pts[pts.length - 1][0]}`;
  g.fillText(lastLabel, right - g.measureText(lastLabel).width, h - 6);
  g.strokeStyle = C.accent; g.lineWidth = 1.5; g.beginPath();
  pts.forEach(([i, v], k) => (k ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))));
  g.stroke();
  const [li, lv] = pts[pts.length - 1];
  g.fillStyle = C.accent; g.beginPath(); g.arc(X(li), Y(lv), 3, 0, 2 * Math.PI); g.fill();
}

export class TrainingView {
  constructor() { this.timer = null; this.listTimer = null; }

  async open(adapter, want) {
    this.close();
    this.adapter = adapter;
    const root = $("training");
    root.hidden = false;
    root.innerHTML = `<nav class="train-list" id="train-list"></nav><section class="train-main" id="train-main"></section>`;
    await this.list(want);
    this.listTimer = setInterval(() => this.list(this.run), 15000);
  }

  async list(want) {
    let runs;
    try { runs = await this.adapter.listRuns(); } catch (e) { $("train-main").innerHTML = `<p class="muted">Could not read the training folders: ${esc(e.message)}</p>`; return; }
    const running = runs.filter((r) => r.state === "running");
    const byMap = {};
    for (const r of runs) (byMap[r.map] = byMap[r.map] || []).push(r);
    $("train-list").innerHTML = (running.length ? "" : `<p class="muted small">Nothing is training now. Start one with train.py (RUN_TRAINING.md); it appears here live.</p>`) +
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
    this.run = run;
    this.mark();
    history.replaceState(null, "", `#training/${encodeURIComponent(run)}`);
    let c;
    try { c = await this.adapter.curve(run); } catch (e) { $("train-main").innerHTML = `<p class="muted">${esc(e.message)}</p>`; return; }
    if (run !== this.run) return;
    this.render(c);
    if (c.state === "running") this.timer = setTimeout(() => this.show(run), POLL);
  }

  render(c) {
    const e = c.episodes, n = e.episode.length, b = c.budget;
    const used = b && b.total ? Math.min(1, b.used / b.total) : null;
    const last = (k) => (n ? e[k][n - 1] : null);
    $("train-main").innerHTML = `
      <header class="train-head">
        <div><h2>${esc(methodName(c.method))} <small>${esc(c.scenario)} · ${esc(c.run.split("/")[0])}</small></h2>
          <span class="live-state ${c.state === "running" ? "on" : ""}">${esc(STATE[c.state] || c.state)} · ${n} episodes${last("wall_s") ? ` · last ${last("wall_s").toFixed(0)} s` : ""}</span></div>
        ${used == null ? "" : `<div class="budget"><span>Decision budget ${(100 * used).toFixed(1)} %</span><div class="bar"><i style="width:${(100 * used).toFixed(1)}%"></i></div>
          <small>${Math.round(b.used).toLocaleString("en")} / ${Math.round(b.total).toLocaleString("en")} decisions</small></div>`}
      </header>
      <div class="charts">${CHARTS.map(([k, label]) => `<figure><figcaption>${label}<b>${last(k) == null ? "–" : CHARTS.find((x) => x[0] === k)[2](last(k))}</b></figcaption><canvas data-k="${k}"></canvas></figure>`).join("")}</div>
      <h3>train.log</h3><pre class="log">${esc(c.log.join("\n"))}</pre>`;
    for (const cv of $("train-main").querySelectorAll("canvas")) chart(cv, e[cv.dataset.k], CHARTS.find((x) => x[0] === cv.dataset.k)[2]);
    const log = $("train-main").querySelector(".log");
    log.scrollTop = log.scrollHeight;
  }

  close() {
    clearTimeout(this.timer); clearInterval(this.listTimer);
    this.run = null;
    $("training").hidden = true;
    $("training").innerHTML = "";
  }
}
