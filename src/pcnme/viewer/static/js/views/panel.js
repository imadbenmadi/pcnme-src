// DetailsPanel: what was clicked (car, task, zone, WAN, cloud), at time t, from the model only.

import { OPTION_NAMES, esc, fmtMs, fmtPct, methodName } from "../core/format.js";
import { C } from "../core/theme.js";

export class DetailsPanel {
  constructor(el, { onClose, onSelectTask }) {
    this.el = el; this.onClose = onClose; this.onSelectTask = onSelectTask;
  }
  render(sel, t, showMethod) {
    const el = this.el;
    if (!sel) { el.hidden = true; el.innerHTML = ""; return; }
    const m = sel.pane.m, who = showMethod ? ` · ${methodName(m.meta.method)}` : "";
    el.innerHTML = this[sel.kind](m, sel, t, who);
    el.hidden = false;
    el.querySelector("[data-close]").onclick = () => this.onClose();
    el.querySelectorAll("[data-task]").forEach((li) => (li.onclick = () => this.onSelectTask(+li.dataset.task)));
    el.querySelectorAll("canvas.spark").forEach((cv) => spark(cv, cv.dataset.series === "wan" ? m.wanSeries() : m.slotSeries(sel.id, cv.dataset.series, "rho"), t / m.end));
  }
  head(title, sub) {
    return `<div class="panel-head"><div><h2>${title}</h2><p>${sub}</p></div>
      <button class="icon-btn" data-close aria-label="Close" title="Close (Esc)"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 4l8 8M12 4l-8 8"/></svg></button></div>`;
  }
  car(m, sel, t, who) {
    const c = sel.id, p = m.carPos(c, t), T = m.tasks;
    const mine = m.byCar[c].filter((i) => T.t[i] <= t);
    const recent = mine.slice(-14).reverse();
    return this.head(`Car ${esc(m.carId(c))}`, `${p ? "driving" : "not in the city now"}${who}`) +
      `<div class="panel-body"><div class="section"><dl class="kv">
        <dt>Serving zone</dt><dd>${esc(m.carZone(c, t) || "none (coverage gap)")}</dd>
        <dt>Speed</dt><dd>${(3.6 * m.carSpeed(c, t)).toFixed(0)} km/h</dd>
        <dt>Position</dt><dd class="mono">${p ? `${p[0].toFixed(0)}, ${p[1].toFixed(0)} m` : "–"}</dd>
        <dt>Tasks so far</dt><dd>${mine.length}</dd>
      </dl></div>
      <div class="section"><h3>Recent tasks</h3><ul class="list">${recent.map((i) => `
        <li data-task="${i}"><span class="mono">${esc(T.id[i])}</span><span>${esc(T.type[i])}</span>
        <span><i class="dot b-${m.cls[i]}"></i>${T.close[i] <= t ? fmtMs(T.latency[i]) : "open"}</span></li>`).join("") || "<li><span>No tasks yet</span></li>"}</ul></div></div>`;
  }
  task(m, sel, t, who) {
    const i = sel.id, T = m.tasks, A = m.attempts, cls = m.cls[i], done = T.close[i] <= t;
    const parts = [["upload", T.upload[i], C.accent], ["forward", T.forward[i], C.muted], ["wait", T.wait[i], C.late],
      ["run", T.run[i], C["on-time"]], ["return", T.back[i], C.faint]].filter(([, v]) => v != null && v > 0);
    const sum = parts.reduce((s, [, v]) => s + v, 0) || 1;
    const attempts = m.byTask[i].map((k) => `<div class="attempt"><b>Attempt ${A.attempt[k]}</b>: ${esc(OPTION_NAMES[A.option[k]] || A.option[k] || "–")} → ${esc(A.executor[k])}${A.slot[k] ? ` (${A.slot[k].toUpperCase()} slot)` : ""}
      <p>${esc((A.result[k] || "").replace(/_/g, " "))}${A.energy[k] != null ? ` · ${A.energy[k].toFixed(2)} J` : ""}${A.start[k] != null && A.finish[k] != null ? ` · ran ${fmtMs(A.finish[k] - A.start[k])}` : ""}</p></div>`).join("");
    return this.head(`Task ${esc(T.id[i])}`, `${esc(T.type[i])} · ${esc(T.cls[i])} deadline${who}`) +
      `<div class="panel-body"><div class="section"><dl class="kv">
        <dt>Outcome</dt><dd class="o-${cls}">${done ? esc(T.outcome[i].replace(/_/g, " ")) : "open"}</dd>
        <dt>Created</dt><dd>${T.t[i].toFixed(2)} s</dd>
        <dt>Deadline</dt><dd>${fmtMs(T.deadline[i])}</dd>
        <dt>Latency</dt><dd>${done ? fmtMs(T.latency[i]) : "–"}</dd>
        <dt>Energy</dt><dd>${T.energy[i] != null ? T.energy[i].toFixed(2) + " J" : "–"}</dd>
        <dt>Anchor zone</dt><dd>${esc(T.anchor[i])}</dd>
      </dl></div>
      <div class="section"><h3>Decision</h3>${attempts || "<p class='note'>No attempt: dropped before a decision.</p>"}</div>
      ${parts.length ? `<div class="section"><h3>Where the time went</h3>
        <div class="timeline">${parts.map(([, v, c]) => `<i style="width:${(100 * v) / sum}%;background:${c}"></i>`).join("")}</div>
        <dl class="tl-legend">${parts.map(([k, v, c]) => `<dt><i class="dot" style="background:${c}"></i>${k}</dt><dd>${fmtMs(v)}</dd>`).join("")}</dl></div>` : ""}
      </div>`;
  }
  zone(m, sel, t, who) {
    const z = m.zone[sel.id], T = m.tasks;
    let n = 0, on = 0;
    for (let i = 0; i < m.n; i++) if (T.anchor[i] === z.id && T.close[i] <= t) { n++; if (m.cls[i] === "on_time") on++; }
    const rows = (type, slots, cap) => (slots ? `<dt>${type.toUpperCase()} slots busy</dt><dd>${m.slot(z.id, type, "busy", t)} of ${slots}</dd>
        <dt>${type.toUpperCase()} queue</dt><dd>${m.slot(z.id, type, "queue", t)} of ${cap}</dd>
        <dt>${type.toUpperCase()} load ρ</dt><dd>${m.slot(z.id, type, "rho", t).toFixed(2)}</dd>` : "");
    const main = z.gpu ? "gpu" : "cpu";
    return this.head(`Zone ${esc(z.id)}`, `${esc(z.hw)} · ${z.gpu} GPU and ${z.cpu} CPU slots${who}`) +
      `<div class="panel-body"><div class="section"><h3>Now</h3><dl class="kv">${rows("gpu", z.gpu, z.qgpu)}${rows("cpu", z.cpu, z.qcpu)}</dl></div>
      <div class="section"><h3>${main.toUpperCase()} load over the run</h3><canvas class="spark" data-series="${main}"></canvas></div>
      <div class="section"><h3>Anchored here so far</h3><dl class="kv"><dt>Tasks closed</dt><dd>${n.toLocaleString("en")}</dd>
        <dt>On time</dt><dd>${fmtPct(n ? on / n : null)}</dd></dl></div></div>`;
  }
  wan(m, sel, t, who) {
    return this.head("WAN link", `metro-edge router to the cloud${who}`) +
      `<div class="panel-body"><div class="section"><dl class="kv"><dt>Queue wait now</dt><dd>${fmtMs(m.wan(t))}</dd>
      <dt>Largest in the run</dt><dd>${fmtMs(Math.max(0, ...m.wanSeries()))}</dd><dt>Scenario</dt><dd>${esc(m.meta.scenario)}, fog load ρ ${m.layout.rho_fog}</dd></dl></div>
      <div class="section"><h3>Queue wait over the run</h3><canvas class="spark" data-series="wan"></canvas></div></div>`;
  }
  cloud(m, sel, t, who) {
    const A = m.attempts;
    const sent = A.executor.filter((e, k) => e === "cloud" && A.send[k] != null && A.send[k] <= t).length;
    return this.head("Cloud", `${m.layout.cloud.gpu} GPU and ${m.layout.cloud.cpu} CPU slots${who}`) +
      `<div class="panel-body"><div class="section"><dl class="kv"><dt>Running now</dt><dd>${m.cloudRunning(t)}</dd>
      <dt>Attempts sent so far</dt><dd>${sent.toLocaleString("en")}</dd></dl></div></div>`;
  }
}

function spark(cv, v, at) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr);
  if (!v.length) return;
  const max = Math.max(1e-9, ...v), n = v.length;
  g.beginPath();
  v.forEach((y, i) => { const x = (i / (n - 1)) * w, yy = h - 2 - (y / max) * (h - 6); i ? g.lineTo(x, yy) : g.moveTo(x, yy); });
  g.strokeStyle = C.accent; g.lineWidth = 1.3; g.stroke();
  g.lineTo(w, h); g.lineTo(0, h); g.closePath(); g.fillStyle = C["accent-soft"] || "rgba(58,92,204,0.1)"; g.globalAlpha = 0.6; g.fill(); g.globalAlpha = 1;
  const xn = at * w;
  g.strokeStyle = C.text; g.globalAlpha = 0.35; g.beginPath(); g.moveTo(xn, 0); g.lineTo(xn, h); g.stroke(); g.globalAlpha = 1;
}
