// DetailsPanel: a popup with the details of what was clicked (zone, cloud, broker, WAN, car, task) at time t,
// from the model only. A zone's agent comes first: its online learning (live runs: updates, loss, Q, ε), what it
// decided (the action mix of its first decisions) and the rewards it got; then its performance and its node.

import { OPTION_NAMES, esc, fmtMs, fmtPct, methodName } from "../core/format.js";
import { C, alpha } from "../core/theme.js";

const LEARNING = new Set(["pcnme", "d3qn-scratch", "d3qn-greedy", "vfc-ppo", "tang-wong", "qeco"]);
const OPTIONS = ["here", "n1", "n2", "n3", "n4", "cloud"];
const WINDOW = 10; // s: "recent" statistics

const num = (x, d = 2) => (x == null || Number.isNaN(x) ? "–" : (+x).toFixed(d));
const int = (x) => (x == null ? "–" : Math.round(x).toLocaleString("en"));
const optColor = (o) => (o === "cloud" ? C.muted : alpha(C.accent, { here: 1, n1: 0.75, n2: 0.58, n3: 0.43, n4: 0.3 }[o] ?? 0.2));

// what the tasks a zone anchored did by time t (closed ones), its first decisions and their rewards
function anchored(m, zone, t) {
  const T = m.tasks, A = m.attempts;
  const s = { n: 0, on: 0, drop: 0, late: 0, lat: 0, latN: 0, e: 0, rn: 0, ron: 0, mix: {}, dec: 0, rew: [], open: 0 };
  for (let i = 0; i < m.n; i++) {
    if (zone && T.anchor[i] !== zone) continue;
    if (T.t[i] > t) continue;
    const ks = m.byTask[i];
    if (ks.length) {
      const k = ks[0];
      if (A.send[k] != null && A.send[k] <= t) { const o = A.option[k] || "?"; s.mix[o] = (s.mix[o] || 0) + 1; s.dec++; }
    }
    if (T.close[i] > t) { s.open++; continue; }
    s.n++;
    const c = m.cls[i];
    if (c === "on_time") s.on++; else if (c === "late") s.late++; else s.drop++;
    if (T.latency[i] != null) { s.lat += T.latency[i]; s.latN++; }
    if (T.energy[i] != null) s.e += T.energy[i];
    if (T.close[i] > t - WINDOW) { s.rn++; if (c === "on_time") s.ron++; }
    for (const k of ks) if (A.reward && A.reward[k] != null) s.rew.push([T.close[i], A.reward[k]]);
  }
  s.rew.sort((a, b) => a[0] - b[0]);
  return s;
}

function mixBar(mix, total) {
  if (!total) return "<p class='note'>No decisions yet.</p>";
  const parts = OPTIONS.filter((o) => mix[o]).map((o) => [o, mix[o] / total]);
  return `<div class="timeline tall">${parts.map(([o, f]) => `<i title="${esc(OPTION_NAMES[o] || o)} ${(100 * f).toFixed(1)} %" style="width:${100 * f}%;background:${optColor(o)}"></i>`).join("")}</div>
    <dl class="tl-legend">${parts.map(([o, f]) => `<dt><i class="dot" style="background:${optColor(o)}"></i>${esc(OPTION_NAMES[o] || o)}</dt><dd>${(100 * f).toFixed(1)} % · ${int(mix[o])}</dd>`).join("")}</dl>`;
}

function mean(xs) { return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null; }

export class DetailsPanel {
  constructor(el, { onClose, onSelectTask }) {
    this.el = el; this.onClose = onClose; this.onSelectTask = onSelectTask;
    this.pos = null;
  }
  // the popup opens next to the click, kept inside the stage
  place(clientX, clientY) { this.pos = [clientX, clientY]; }
  position() {
    const host = this.el.parentElement.getBoundingClientRect(), w = this.el.offsetWidth, h = this.el.offsetHeight;
    let [x, y] = this.pos || [host.right - w - 16, host.top + 16];
    x = x - host.left + 14; y = y - host.top - 20;
    if (x + w > host.width - 8) x = Math.max(8, x - w - 28);
    y = Math.max(8, Math.min(y, host.height - h - 8));
    this.el.style.left = `${x}px`; this.el.style.top = `${y}px`;
  }
  render(sel, t, showMethod) {
    const el = this.el;
    if (!sel) { el.hidden = true; el.innerHTML = ""; this.pos = null; return; }
    const m = sel.pane.m, who = showMethod ? ` · ${methodName(m.meta.method)}` : "";
    const scroll = el.querySelector(".panel-body")?.scrollTop || 0;
    el.innerHTML = this[sel.kind](m, sel, t, who);
    el.hidden = false;
    const body = el.querySelector(".panel-body"); if (body) body.scrollTop = scroll;
    el.querySelector("[data-close]").onclick = () => this.onClose();
    el.querySelectorAll("[data-task]").forEach((li) => (li.onclick = () => this.onSelectTask(+li.dataset.task)));
    el.querySelectorAll("canvas.spark").forEach((cv) => {
      const k = cv.dataset.series;
      const v = k === "wan" ? m.wanSeries() : k === "reward" ? this.rewardSeries : m.slotSeries(sel.id, k, "rho");
      spark(cv, v || [], k === "reward" ? 1 : t / m.end);
    });
    this.position();
  }
  head(eyebrow, title, sub) {
    return `<div class="panel-head"><div><div class="eyebrow">${esc(eyebrow)}</div><h2>${title}</h2><p>${sub}</p></div>
      <button class="icon-btn" data-close aria-label="Close" title="Close (Esc)"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 4l8 8M12 4l-8 8"/></svg></button></div>`;
  }
  tiles(items) {
    return `<div class="tiles">${items.map(([label, value, cls = ""]) => `<div class="tile"><b class="${cls}">${value}</b><span>${esc(label)}</span></div>`).join("")}</div>`;
  }

  // ---- zone -----------------------------------------------------------------------------------------------
  zone(m, sel, t, who) {
    const z = m.zone[sel.id], s = anchored(m, z.id, t), method = m.meta.method;
    const learns = LEARNING.has(method), agent = (m.agents || {})[z.id];
    const recent = s.rew.slice(-200).map((r) => r[1]);
    this.rewardSeries = s.rew.length > 1 ? bucket(s.rew.map((r) => r[1]), 60) : [];
    let rl;
    if (agent && agent.updates != null) {
      rl = this.tiles([["online updates", int(agent.updates)], ["samples learned", int(agent.samples)],
        ["loss (last)", num(agent.loss, 3)], ["mean Q", num(agent.mean_q, 2)], ["ε (exploration)", num(agent.epsilon, 3)],
        ["mean reward, last 200", num(mean(recent), 3)]]);
    } else if (learns) {
      rl = this.tiles([["decisions", int(s.dec)], ["mean reward", num(mean(s.rew.map((r) => r[1])), 3)], ["mean reward, last 200", num(mean(recent), 3)]]) +
        `<p class="note">${m.meta.mode === "live" ? "Waiting for the zone's first learning sample." : "This replay holds the decisions and rewards; the online updates, loss and Q are recorded in live runs."}</p>`;
    } else {
      rl = `<p class="note">${esc(methodName(method))} is a fixed rule: it does not learn. Its decisions:</p>` +
        this.tiles([["decisions", int(s.dec)], ["mean reward", num(mean(s.rew.map((r) => r[1])), 3)]]);
    }
    const node = (type, slots, cap) => (slots ? `<dt>${type.toUpperCase()} slots busy</dt><dd>${m.slot(z.id, type, "busy", t)} of ${slots}</dd>
        <dt>${type.toUpperCase()} queue</dt><dd>${m.slot(z.id, type, "queue", t)} of ${cap}</dd>
        <dt>${type.toUpperCase()} load ρ</dt><dd>${m.slot(z.id, type, "rho", t).toFixed(2)}</dd>` : "");
    let cars = 0;
    for (let c = 0; c < m.carCount(); c++) if (m.carPos(c, t) && m.carZone(c, t) === z.id) cars++;
    const main = z.gpu ? "gpu" : "cpu";
    return this.head("Fog zone", `Zone ${esc(z.id)}`, `${esc(z.hw)} · ${z.gpu} GPU + ${z.cpu} CPU slots · ${esc(methodName(method))}${who}`) +
      `<div class="panel-body">
      <div class="section"><h3>Agent ${learns ? "(reinforcement learning)" : "(rule)"}</h3>${rl}
        ${this.rewardSeries.length ? `<div class="sub">Reward of its decisions over the run</div><canvas class="spark" data-series="reward"></canvas>` : ""}</div>
      <div class="section"><h3>What it decided (first decision per task)</h3>${mixBar(s.mix, s.dec)}</div>
      <div class="section"><h3>Its tasks so far</h3>${this.tiles([
        ["on time", fmtPct(s.n ? s.on / s.n : null), "o-on_time"], ["dropped", fmtPct(s.n ? s.drop / s.n : null), "o-dropped"],
        ["late", fmtPct(s.n ? s.late / s.n : null), "o-late"], ["mean latency", fmtMs(s.latN ? s.lat / s.latN : null)],
        [`on time, last ${WINDOW} s`, fmtPct(s.rn ? s.ron / s.rn : null)], ["J / task", num(s.n ? s.e / s.n : null)]])}
        <dl class="kv small"><dt>Closed</dt><dd>${int(s.n)}</dd><dt>Open now</dt><dd>${int(s.open)}</dd><dt>Cars it serves now</dt><dd>${cars}</dd></dl></div>
      <div class="section"><h3>Its node now</h3><dl class="kv">${node("gpu", z.gpu, z.qgpu)}${node("cpu", z.cpu, z.qcpu)}</dl>
        <div class="sub">${main.toUpperCase()} load over the run</div><canvas class="spark" data-series="${main}"></canvas></div>
      </div>`;
  }

  // ---- cloud, broker, WAN -----------------------------------------------------------------------------------
  cloud(m, sel, t, who) {
    const A = m.attempts, T = m.tasks;
    let sent = 0, total = 0, ran = 0, runSum = 0, delivered = 0, on = 0;
    for (let k = 0; k < A.executor.length; k++) {
      if (A.send[k] == null || A.send[k] > t) continue;
      total++;
      if (A.executor[k] !== "cloud") continue;
      sent++;
      if (A.start[k] != null && A.finish[k] != null && A.finish[k] <= t) { ran++; runSum += A.finish[k] - A.start[k]; }
      if (A.result[k] === "delivered") { const i = A.task[k]; if (T.close[i] <= t) { delivered++; if (m.cls[i] === "on_time") on++; } }
    }
    return this.head("Cloud", "Cloud data centre", `${m.layout.cloud.gpu} GPU + ${m.layout.cloud.cpu} CPU slots, behind the WAN${who}`) +
      `<div class="panel-body"><div class="section"><h3>Now</h3>${this.tiles([["running now", int(m.cloudRunning(t))], ["WAN queue", fmtMs(m.wan(t))]])}</div>
      <div class="section"><h3>So far</h3>${this.tiles([["attempts sent here", int(sent)], ["share of all attempts", fmtPct(total ? sent / total : null)],
        ["mean run here", fmtMs(ran ? runSum / ran : null)], ["its results on time", fmtPct(delivered ? on / delivered : null), "o-on_time"]])}
      <p class="note">The TOF-Broker sends tasks too heavy for any fog node here; an agent may also choose the cloud.</p></div></div>`;
  }
  broker(m, sel, t, who) {
    const A = m.attempts, T = m.tasks;
    let results = 0, recent = 0, acks = 0;
    for (let k = 0; k < A.executor.length; k++) if (A.finish[k] != null && A.finish[k] <= t) { results++; if (A.finish[k] > t - WINDOW) recent++; }
    for (let i = 0; i < m.n; i++) if (T.close[i] <= t && T.latency[i] != null) acks++;
    const how = m.meta.mode === "fast" ? "In fast mode the broker is a modelled delay (5.1)." : "On the testbed: EMQX, every message one shaped backhaul crossing.";
    return this.head("Message bus", "MQTT broker", `results, acks, state snapshots${who}`) +
      `<div class="panel-body"><div class="section"><h3>Counted from the task log</h3>${this.tiles([["results published", int(results)],
        [`results / s, last ${WINDOW} s`, num(recent / WINDOW, 1)], ["acks (tasks confirmed)", int(acks)]])}
      <p class="note">${how} State snapshots are published by every zone on its own period and are not in the task log.</p></div></div>`;
  }
  wan(m, sel, t, who) {
    return this.head("Network", "WAN link", `metro-edge router to the cloud${who}`) +
      `<div class="panel-body"><div class="section">${this.tiles([["queue wait now", fmtMs(m.wan(t))], ["largest in the run", fmtMs(Math.max(0, ...m.wanSeries()))]])}
      <dl class="kv small"><dt>Scenario</dt><dd>${esc(m.meta.scenario)}, fog load ρ ${m.layout.rho_fog}</dd></dl></div>
      <div class="section"><h3>Queue wait over the run</h3><canvas class="spark" data-series="wan"></canvas></div></div>`;
  }

  // ---- car and task ------------------------------------------------------------------------------------------
  car(m, sel, t, who) {
    const c = sel.id, p = m.carPos(c, t), T = m.tasks;
    const mine = m.byCar[c].filter((i) => T.t[i] <= t);
    const closed = mine.filter((i) => T.close[i] <= t), on = closed.filter((i) => m.cls[i] === "on_time").length;
    const types = {};
    for (const i of mine) types[T.type[i]] = (types[T.type[i]] || 0) + 1;
    const recent = mine.slice(-14).reverse();
    return this.head("Vehicle", `Car ${esc(m.carId(c))}`, `${p ? "driving" : "not in the city now"}${who}`) +
      `<div class="panel-body"><div class="section">${this.tiles([["serving zone", esc(m.carZone(c, t) || "none")], ["speed", `${(3.6 * m.carSpeed(c, t)).toFixed(0)} km/h`],
        ["its tasks on time", fmtPct(closed.length ? on / closed.length : null), "o-on_time"], ["tasks so far", int(mine.length)]])}
        <dl class="kv small"><dt>Position</dt><dd class="mono">${p ? `${p[0].toFixed(0)}, ${p[1].toFixed(0)} m` : "–"}</dd>
        <dt>By type</dt><dd>${Object.entries(types).sort().map(([k, v]) => `${esc(k)} ${v}`).join(" · ") || "–"}</dd></dl></div>
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
      ${A.decided && A.decided[k] === false ? '<span class="tag">TOF-Broker / fallback</span>' : A.decided && A.decided[k] ? '<span class="tag">agent decision</span>' : ""}
      <p>${esc((A.result[k] || "").replace(/_/g, " "))}${A.reward && A.reward[k] != null ? ` · reward ${num(A.reward[k], 3)}` : ""}${A.energy[k] != null ? ` · ${A.energy[k].toFixed(2)} J` : ""}${A.start[k] != null && A.finish[k] != null ? ` · ran ${fmtMs(A.finish[k] - A.start[k])}` : ""}</p></div>`).join("");
    return this.head("Task", `${esc(T.id[i])}`, `${esc(T.type[i])} · ${esc(T.cls[i])} deadline · anchored at ${esc(T.anchor[i])}${who}`) +
      `<div class="panel-body"><div class="section">${this.tiles([["outcome", done ? esc(T.outcome[i].replace(/_/g, " ")) : "open", `o-${cls}`],
        ["latency", done ? fmtMs(T.latency[i]) : "–"], ["deadline", fmtMs(T.deadline[i])], ["energy", T.energy[i] != null ? `${T.energy[i].toFixed(2)} J` : "–"]])}
        <dl class="kv small"><dt>Created</dt><dd>${T.t[i].toFixed(2)} s</dd><dt>Attempts</dt><dd>${int(T.attempts[i])}</dd></dl></div>
      <div class="section"><h3>Decisions and attempts</h3>${attempts || "<p class='note'>No attempt: dropped before a decision.</p>"}</div>
      ${parts.length ? `<div class="section"><h3>Where the time went</h3>
        <div class="timeline">${parts.map(([, v, c]) => `<i style="width:${(100 * v) / sum}%;background:${c}"></i>`).join("")}</div>
        <dl class="tl-legend">${parts.map(([k, v, c]) => `<dt><i class="dot" style="background:${c}"></i>${k}</dt><dd>${fmtMs(v)}</dd>`).join("")}</dl></div>` : ""}
      </div>`;
  }
}

function bucket(v, n) { // a long series as n means
  if (v.length <= n) return v;
  const out = [], step = v.length / n;
  for (let b = 0; b < n; b++) { const s = v.slice(Math.floor(b * step), Math.floor((b + 1) * step)); out.push(mean(s)); }
  return out;
}

function spark(cv, v, at) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr);
  if (!v.length) return;
  const lo = Math.min(0, ...v), hi = Math.max(1e-9, ...v), n = v.length;
  const Y = (y) => h - 2 - ((y - lo) / (hi - lo || 1)) * (h - 6);
  g.beginPath();
  v.forEach((y, i) => { const x = (i / Math.max(1, n - 1)) * w; i ? g.lineTo(x, Y(y)) : g.moveTo(x, Y(y)); });
  g.strokeStyle = C.accent; g.lineWidth = 1.3; g.stroke();
  g.lineTo(w, Y(lo)); g.lineTo(0, Y(lo)); g.closePath(); g.fillStyle = C["accent-soft"] || "rgba(58,92,204,0.1)"; g.globalAlpha = 0.6; g.fill(); g.globalAlpha = 1;
  if (lo < 0) { g.strokeStyle = C.faint; g.globalAlpha = 0.5; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(0, Y(0)); g.lineTo(w, Y(0)); g.stroke(); g.setLineDash([]); g.globalAlpha = 1; }
  if (at < 1) { const xn = at * w; g.strokeStyle = C.text; g.globalAlpha = 0.35; g.beginPath(); g.moveTo(xn, 0); g.lineTo(xn, h); g.stroke(); g.globalAlpha = 1; }
}
