// RunPane: one model on screen: a header (method, run, the live statistics strip) above a CityView.

import { esc, fmtMs, fmtPct, methodName } from "../core/format.js";
import { CityView } from "./city.js";

const STATS = [
  ["onTime", "on time", fmtPct],
  ["dropped", "dropped", fmtPct],
  ["latency", "mean latency", fmtMs],
  ["energy", "J / task", (x) => (x == null ? "–" : x.toFixed(2))],
  ["closed", "tasks closed", (x) => x.toLocaleString("en")],
];

function subtitle(meta) {
  const times = { off_peak: "off-peak (11:00)", mid: "midday (15:00)", peak: "peak (08:00)" };
  if (meta.mode === "traces") return `${meta.map === "lust" ? "LuST" : meta.map} · ${times[meta.scenario] || meta.scenario} · ${meta.fleet} vehicles in ${meta.end} s`;
  return `${meta.scenario} · seed ${meta.seed} · ${meta.mode === "testbed" ? "testbed" : meta.mode === "live" ? "live testbed" : "fast mode"}${meta.fleet ? ` · ${meta.fleet} cars` : ""}`;
}

// MissingPane: a selected run that does not exist (yet), and why.
export class MissingPane {
  constructor(root, { what, why, note }) {
    this.el = document.createElement("div");
    this.el.className = "pane";
    this.el.innerHTML = `<div class="missing"><div class="dev-card">
      <div class="eyebrow">No results</div><h1>${esc(what)}</h1><p>${esc(why)}</p>${note ? `<p>${esc(note)}</p>` : ""}</div></div>`;
    root.appendChild(this.el);
  }
  draw() {}
  updateStats() {}
  destroy() { this.el.remove(); }
}

export class RunPane {
  constructor(root, model, { onSelect, speed }) {
    this.m = model;
    const meta = model.meta;
    this.el = document.createElement("div");
    this.el.className = "pane";
    this.el.innerHTML = `
      <div class="pane-head">
        <div class="pane-title">${esc(meta.mode === "traces" ? "Traffic" : methodName(meta.method))}<small>${esc(subtitle(meta))}</small></div>
        ${meta.note ? `<div class="pane-note">${esc(meta.note)}</div>` : `<div class="stats">${STATS.map(([k, label]) => `<div class="stat"><b data-k="${k}">–</b><span>${label}</span></div>`).join("")}</div>`}
      </div>
      <div class="canvas-wrap"></div>`;
    root.appendChild(this.el);
    this.statEls = Object.fromEntries([...this.el.querySelectorAll("[data-k]")].map((e) => [e.dataset.k, e]));
    this.city = new CityView(this.el.querySelector(".canvas-wrap"), model, { onSelect: (hit, ev) => onSelect(hit, this, ev), speed });
  }
  draw(t, sel) { this.city.draw(t, sel && sel.pane === this ? sel : null); }
  updateStats(t) {
    if (this.m.meta.note) return;
    const s = this.m.stats(t);
    for (const [k, , f] of STATS) this.statEls[k].textContent = f(s[k]);
  }
  destroy() { this.city.destroy(); this.el.remove(); }
}
