// LiveView: the testbed's live runs (PLAN V3). A run that is going now plays in real time a few seconds behind
// the newest data (a task is drawn once its anchor zone has reported it closed); a finished run is a recording
// and plays like a replay. The same RunPane and CityView as a replay, on a model that grows each second
// (core/model.js liveModel). Under the header, each zone's agent and its online learning.

import { $, esc, methodName } from "../core/format.js";
import { liveModel } from "../core/model.js";
import { RunPane } from "./run-pane.js";

const LAG = 3; // seconds behind the newest data
const RETRY = 5000; // ms between looks for InfluxDB or a run
const LIST_EVERY = 10000; // ms between looks for a new live run
const ENDED = 20; // s without new data: the run is over
const COMMAND = "uv run python run_live.py --scenario S2 --method pcnme --minutes 10";

function card(eyebrow, title, body, pre) {
  return `<div class="pane"><div class="missing"><div class="dev-card"><div class="eyebrow">${esc(eyebrow)}</div><h1>${esc(title)}</h1>
    ${body.map((p) => `<p>${esc(p)}</p>`).join("")}${pre ? `<pre class="cmd">${esc(pre)}</pre>` : ""}</div></div></div>`;
}

const clockTime = (s) => new Date(s * 1000).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
const runLabel = (r) => `${r.live ? "● " : ""}${r.scenario} · ${methodName(r.method)} · seed ${r.seed} · ${clockTime(r.start)} · ${r.live ? "live now" : `finished, ${r.minutes} min`}`;

export class LiveView {
  constructor({ clock, controls, setPanes, onSelect }) {
    Object.assign(this, { clock, controls, setPanes, onSelect });
    this.stopPoll = null; this.retry = null; this.model = null; this.pane = null; this.last = 0; this.runs = [];
    $("live-run").onchange = () => this.follow($("live-run").value);
    $("live-go").onclick = () => this.goLive();
    $("toast").addEventListener("click", (e) => { if (e.target.closest("[data-close]")) this.toast(null); });
    this.ticker = setInterval(() => this.status(), 1000);
  }

  async list() {
    try { return await this.adapter.listRuns(); } catch (e) { return { online: false, error: e.message, runs: [] }; }
  }

  fill(list) {
    this.runs = list.runs;
    const cur = $("live-run").value;
    $("live-run").innerHTML = list.runs.map((r) => `<option value="${esc(r.run)}">${esc(runLabel(r))}</option>`).join("");
    if (cur && list.runs.some((r) => r.run === cur)) $("live-run").value = cur;
    $("live-go").classList.toggle("has-live", !!list.any_live);
  }

  async open(adapter, want) {
    this.close();
    this.adapter = adapter;
    $("live-bar").hidden = false;
    const list = await this.list();
    if (!list.online || !list.runs.length) {
      $("stage").innerHTML = list.online
        ? card("Live testbed", "No live run yet", ["A live run shows the testbed as it goes: the cars, the tasks, the zones' queues and each agent learning online. Start one (Docker Desktop running, nothing else on the testbed):"], COMMAND)
        : card("Live testbed", "InfluxDB is not running", ["The live view reads the points the zones push to InfluxDB during a live run. Starting a live run starts InfluxDB too:"], COMMAND);
      $("live-run").innerHTML = ""; $("live-state").textContent = list.online ? "no live run" : "offline"; $("live-state").className = "live-state";
      this.retry = setTimeout(() => this.open(adapter, want), RETRY);
      return;
    }
    this.fill(list);
    const pick = list.runs.find((r) => r.run === want) || list.runs.find((r) => r.live) || list.runs[0];
    $("live-run").value = pick.run;
    this.follow(pick.run);
    this.lister = setInterval(async () => { const l = await this.list(); if (l.online) this.fill(l); }, LIST_EVERY);
  }

  follow(run) {
    this.drop();
    this.run = run;
    const info = this.runs.find((r) => r.run === run);
    this.isLive = !!(info && info.live);
    history.replaceState(null, "", `#live/${encodeURIComponent(run)}`);
    $("stage").innerHTML = card("Live testbed", this.isLive ? "Waiting for data" : "Loading the recording", [`${run}: ${this.isLive ? "the first points arrive within a few seconds of the cars starting." : "a finished live run, played like a replay."}`]);
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
      this.clock.setRange(this.model.end);
      if (this.isLive) { this.clock.lag = LAG; this.clock.goLive(); }
      else { this.clock.lag = null; this.clock.seek(0); this.clock.play(); }
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
      return `<button class="chip" data-zone="${esc(z)}" title="Details of ${esc(z)}"><b>${esc(z)}</b> ${s.updates == null ? "–" : (+s.updates).toLocaleString("en")} updates · loss ${f(s.loss, 3)} · Q ${f(s.mean_q, 2)} · ε ${f(s.epsilon, 3)}</button>`;
    }).join("");
    this.strip.querySelectorAll("[data-zone]").forEach((b) => (b.onclick = (e) => this.onSelect({ kind: "zone", id: b.dataset.zone }, this.pane, e)));
  }

  // "Go live": jump to the newest data of the run that is going now, or say that none is
  async goLive() {
    if (this.isLive && this.model) { this.clock.goLive(); return; }
    const list = this.adapter ? await this.list() : { runs: [] };
    if (list.online) this.fill(list);
    const live = (list.runs || []).find((r) => r.live);
    if (live) { $("live-run").value = live.run; this.follow(live.run); return; }
    this.toast(`<b>No live run is going now.</b><p>The runs listed are recordings of earlier live runs. Start one, then press Go live:</p><pre class="cmd">${esc(COMMAND)}</pre>`);
  }

  toast(html) {
    const t = $("toast");
    if (!html) { t.hidden = true; return; }
    t.innerHTML = `<button class="icon-btn" data-close aria-label="Close"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 4l8 8M12 4l-8 8"/></svg></button>${html}`;
    t.hidden = false;
    clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => (t.hidden = true), 15000);
  }

  status() {
    if ($("live-bar").hidden || !this.run) return;
    const el = $("live-state");
    if (this.error) { el.textContent = `connection lost: ${this.error}`; el.className = "live-state off"; return; }
    if (!this.model) { el.textContent = this.isLive ? "waiting for data" : "loading"; el.className = "live-state"; return; }
    const age = (performance.now() - this.last) / 1000;
    if (this.isLive && age > ENDED) { // the run ended while we watched: it becomes a recording
      this.isLive = false; this.clock.lag = null;
      const info = this.runs.find((r) => r.run === this.run); if (info) info.live = false;
    }
    const n = this.model.n.toLocaleString("en"), mins = (this.model.end / 60).toFixed(1);
    el.className = `live-state ${this.isLive ? "on" : ""}`;
    el.textContent = this.isLive ? `live · ${n} tasks · ${mins} min · ${LAG} s behind` : `recording (finished) · ${n} tasks · ${mins} min`;
  }

  drop() {
    if (this.stopPoll) this.stopPoll();
    this.stopPoll = null; this.model = null;
    if (this.pane) { this.pane.destroy(); this.pane = null; this.setPanes([]); }
    this.clock.lag = null;
  }

  close() {
    clearTimeout(this.retry); clearInterval(this.lister);
    this.drop();
    this.run = null;
    this.toast(null);
    $("live-bar").hidden = true;
  }
}
