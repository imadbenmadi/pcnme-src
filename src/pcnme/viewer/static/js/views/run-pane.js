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

export class RunPane {
  constructor(root, model, { onSelect, speed }) {
    this.m = model;
    const meta = model.meta;
    this.el = document.createElement("div");
    this.el.className = "pane";
    this.el.innerHTML = `
      <div class="pane-head">
        <div class="pane-title">${esc(methodName(meta.method))}<small>${esc(meta.scenario)} · seed ${meta.seed} · ${meta.mode === "testbed" ? "testbed" : "fast mode"}${meta.fleet ? ` · ${meta.fleet} cars` : ""}</small></div>
        <div class="stats">${STATS.map(([k, label]) => `<div class="stat"><b data-k="${k}">–</b><span>${label}</span></div>`).join("")}</div>
      </div>
      <div class="canvas-wrap"></div>`;
    root.appendChild(this.el);
    this.statEls = Object.fromEntries([...this.el.querySelectorAll("[data-k]")].map((e) => [e.dataset.k, e]));
    this.city = new CityView(this.el.querySelector(".canvas-wrap"), model, { onSelect: (hit) => onSelect(hit, this), speed });
  }
  draw(t, sel) { this.city.draw(t, sel && sel.pane === this ? sel : null); }
  updateStats(t) {
    const s = this.m.stats(t);
    for (const [k, , f] of STATS) this.statEls[k].textContent = f(s[k]);
  }
  destroy() { this.city.destroy(); this.el.remove(); }
}
