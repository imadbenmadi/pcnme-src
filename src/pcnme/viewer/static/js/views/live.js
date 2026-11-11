// LiveView: a testbed run while it goes (PLAN V3). The same RunPane and CityView as a replay, on a model that
// grows each second (core/model.js liveModel); the clock plays in real time a few seconds behind the newest
// data, because a task is drawn once its anchor zone has reported it closed. Under the header, each zone's agent
// and its online learning (updates, loss, epsilon), from the zones' live samples.

import { $, esc, methodName } from "../core/format.js";
import { liveModel } from "../core/model.js";
import { RunPane } from "./run-pane.js";

const LAG = 3; // seconds behind the newest data
const RETRY = 5000; // ms between looks for InfluxDB or a new run

function card(eyebrow, title, body, pre) {
  return `<div class="pane"><div class="missing"><div class="dev-card"><div class="eyebrow">${esc(eyebrow)}</div><h1>${esc(title)}</h1>
    ${body.map((p) => `<p>${esc(p)}</p>`).join("")}${pre ? `<pre class="cmd">${esc(pre)}</pre>` : ""}</div></div></div>`;
}

const runLabel = (r) => `${r.scenario} · ${methodName(r.method)} · seed ${r.seed} · ${r.run.split("-").slice(-2).join(" ")}`;

export class LiveView {
  constructor({ clock, controls, setPanes, onSelect }) {
    Object.assign(this, { clock, controls, setPanes, onSelect });
    this.stopPoll = null; this.retry = null; this.model = null; this.pane = null; this.last = 0;
    $("live-run").onchange = () => this.follow($("live-run").value);
    $("live-go").onclick = () => this.clock.goLive();
    this.ticker = setInterval(() => this.status(), 1000);
  }

  async open(adapter, want) {
    this.close();
    this.adapter = adapter;
    $("live-bar").hidden = false;
    let list;
    try { list = await adapter.listRuns(); } catch (e) { list = { online: false, error: e.message, runs: [] }; }
    if (!list.online || !list.runs.length) {
      const [help, cmd] = (list.help || "").split("\n");
      $("stage").innerHTML = list.online
        ? card("Live testbed", "No live run yet", [help], cmd)
        : card("Live testbed", "InfluxDB is not running", ["The live view reads the points the zones push to InfluxDB during a live run.", help], cmd);
      $("live-run").innerHTML = ""; $("live-state").textContent = list.online ? "waiting for a live run" : "offline";
      this.retry = setTimeout(() => this.open(adapter, want), RETRY);
      return;
    }
    this.runs = list.runs;
    $("live-run").innerHTML = list.runs.map((r) => `<option value="${esc(r.run)}">${esc(runLabel(r))}</option>`).join("");
    const pick = list.runs.some((r) => r.run === want) ? want : list.runs[0].run;
    $("live-run").value = pick;
    this.follow(pick);
  }

  follow(run) {
    this.drop();
    this.run = run;
    history.replaceState(null, "", `#live/${encodeURIComponent(run)}`);
    $("stage").innerHTML = card("Live testbed", "Waiting for data", [`${run}: the first points arrive within a few seconds of the cars starting.`]);
    this.stopPoll = this.adapter.follow(run, (buf) => this.data(buf), (e) => { this.error = e.message; });
  }

  data(buf) {
    this.error = null; this.last = performance.now();
    if (!buf.ready) return;
    if (!this.model) {
      this.model = liveModel(buf);
      $("stage").innerHTML = "";
      this.pane = new RunPane($("stage"), this.model, { onSelect: this.onSelect, speed: () => this.clock.speed });
      this.strip = document.createElement("div");
      this.strip.className = "agent-strip";
      this.pane.el.querySelector(".pane-head").after(this.strip);
      this.setPanes([this.pane]);
      this.clock.lag = LAG;
      this.clock.setRange(this.model.end);
      this.clock.goLive();
    } else {
      this.model.refresh();
      this.clock.setRange(this.model.end);
    }
    this.controls.setArrivals(this.model.end);
    this.agents();
  }

  agents() {
    const a = this.model.agents, zones = this.model.layout.zones.filter((z) => z.active).map((z) => z.id);
    if (!Object.keys(a).length) {
      this.strip.innerHTML = `<span class="muted">${esc(methodName(this.model.meta.method))} decides by a fixed rule: nothing learns in this run.</span>`;
      return;
    }
    const f = (x, d) => (x == null ? "–" : (+x).toFixed(d));
    this.strip.innerHTML = `<span class="muted">Online learning</span>` + zones.map((z) => {
      const s = a[z] || {};
      return `<span class="chip"><b>${esc(z)}</b> ${s.updates == null ? "–" : (+s.updates).toLocaleString("en")} updates · loss ${f(s.loss, 3)} · Q ${f(s.mean_q, 2)} · ε ${f(s.epsilon, 3)}</span>`;
    }).join("");
  }

  status() {
    if ($("live-bar").hidden || !this.run) return;
    const el = $("live-state");
    if (this.error) { el.textContent = `connection lost: ${this.error}`; el.className = "live-state off"; return; }
    if (!this.model) { el.textContent = "waiting for data"; el.className = "live-state"; return; }
    const age = (performance.now() - this.last) / 1000;
    const fresh = age < 5;
    el.className = `live-state ${fresh ? "on" : "off"}`;
    el.textContent = fresh ? `live · ${this.model.n.toLocaleString("en")} tasks · ${(this.model.end / 60).toFixed(1)} min` : `no new data for ${age.toFixed(0)} s (run over?)`;
  }

  drop() {
    if (this.stopPoll) this.stopPoll();
    this.stopPoll = null; this.model = null;
    if (this.pane) { this.pane.destroy(); this.pane = null; this.setPanes([]); }
    this.clock.lag = null;
  }

  close() {
    clearTimeout(this.retry);
    this.drop();
    this.run = null;
    $("live-bar").hidden = true;
  }
}
