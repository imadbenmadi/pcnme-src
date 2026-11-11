// The city model: what every view reads. The contract (a live model implements the same methods, fed by a
// stream, with `end` growing as data arrives):
//
//   meta, layout                       the run (scenario, method, seed, mode, fleet) and the city (zones, links)
//   end, arrivals                      the last time with data; the time new tasks stop
//   carCount(), carId(c), carPos(c,t)  positions; carZone(c,t), carSpeed(c,t)
//   slot(zone, "gpu"|"cpu", "busy"|"queue"|"rho", t), load(zone, t)
//   wan(t), wanMax, cloudRunning(t)
//   tasksAround(t) -> [i0, i1)         the tasks that may be on screen at t (indices into tasks/attempts)
//   tasks, attempts, cls, byTask, byCar  columnar task and attempt data (see sources/replay.py)
//   stats(t) -> {closed, onTime, dropped, latency, energy}  over the tasks closed by t
//
// ReplayModel builds it from one replay (/api/replay).

import { lowerBound, outcomeClass, upperBound } from "./format.js";

export class ReplayModel {
  constructor(d) {
    this.meta = d.meta; this.layout = d.layout; this.cars = d.cars; this.zones = d.zones; this.pipes = d.pipes;
    this.tasks = d.tasks; this.attempts = d.attempts;
    this.end = d.meta.end; this.arrivals = d.meta.seconds;
    const T = d.tasks, A = d.attempts, n = T.id.length;
    this.n = n;
    this.cls = T.outcome.map(outcomeClass);
    this.byTask = Array.from({ length: n }, () => []);
    A.task.forEach((ti, k) => this.byTask[ti].push(k));
    this.endT = new Float64Array(n);
    let maxLife = 1;
    for (let i = 0; i < n; i++) {
      let e = T.close[i] ?? T.t[i];
      for (const k of this.byTask[i]) e = Math.max(e, A.car[k] ?? 0, A.finish[k] ?? 0, A.arrive[k] ?? 0);
      this.endT[i] = e;
      maxLife = Math.max(maxLife, e - T.t[i]);
    }
    this.maxLife = maxLife + 1;
    const order = [...Array(n).keys()].sort((a, b) => T.close[a] - T.close[b]);
    this.closeT = order.map((i) => T.close[i]);
    const cum = (f) => { const out = new Float64Array(n + 1); order.forEach((i, k) => (out[k + 1] = out[k] + f(i))); return out; };
    this.cumOn = cum((i) => (this.cls[i] === "on_time" ? 1 : 0));
    this.cumDrop = cum((i) => (this.cls[i] === "dropped" ? 1 : 0));
    this.cumLat = cum((i) => T.latency[i] ?? 0);
    this.cumLatN = cum((i) => (T.latency[i] == null ? 0 : 1));
    this.cumE = cum((i) => T.energy[i] ?? 0);
    this.byCar = Array.from({ length: d.cars.ids.length }, () => []);
    T.car.forEach((c, i) => c >= 0 && this.byCar[c].push(i));
    this.cloud = [];
    A.executor.forEach((e, k) => e === "cloud" && A.start[k] != null && this.cloud.push(k));
    this.wanMax = Math.max(0.25, ...(d.pipes.wan || [0]));
    this.zone = Object.fromEntries(d.layout.zones.map((z) => [z.id, z]));
  }
  frame(t) { return Math.max(0, Math.min(this.cars.x[0].length - 1, Math.floor(t / this.cars.step))); }
  carCount() { return this.cars.ids.length; }
  carId(c) { return this.cars.ids[c]; }
  carZone(c, t) { return this.cars.zone[c][this.frame(t)]; }
  carSpeed(c, t) { return this.cars.speed[c][this.frame(t)] || 0; }
  carPos(c, t) {
    const X = this.cars.x[c], Y = this.cars.y[c], f = t / this.cars.step, i = Math.floor(f);
    if (i < 0 || i >= X.length || X[i] == null) return null;
    if (i + 1 >= X.length || X[i + 1] == null) return [X[i], Y[i]];
    const u = f - i;
    return [X[i] + (X[i + 1] - X[i]) * u, Y[i] + (Y[i + 1] - Y[i]) * u];
  }
  slot(z, type, key, t) {
    const s = this.zones[z] && this.zones[z][type];
    return s ? s[key][Math.min(this.frame(t), s[key].length - 1)] : 0;
  }
  slotSeries(z, type, key) { const s = this.zones[z] && this.zones[z][type]; return s ? s[key] : []; }
  load(z, t) { return Math.max(this.slot(z, "gpu", "rho", t), this.slot(z, "cpu", "rho", t)); }
  wan(t) { const w = this.pipes.wan; return w ? w[Math.min(this.frame(t), w.length - 1)] : 0; }
  wanSeries() { return this.pipes.wan || []; }
  cloudRunning(t) {
    const A = this.attempts;
    let n = 0;
    for (const k of this.cloud) if (A.start[k] <= t && t < (A.finish[k] ?? Infinity)) n++;
    return n;
  }
  tasksAround(t) { return [lowerBound(this.tasks.t, t - this.maxLife), upperBound(this.tasks.t, t)]; }
  stats(t) {
    const k = upperBound(this.closeT, t);
    return {
      closed: k,
      onTime: k ? this.cumOn[k] / k : null,
      dropped: k ? this.cumDrop[k] / k : null,
      latency: this.cumLatN[k] ? this.cumLat[k] / this.cumLatN[k] : null,
      energy: k ? this.cumE[k] / k : null,
    };
  }
}

// ---- live ---------------------------------------------------------------------------------------------------
// LiveBuffer keeps what /api/live has sent (tasks by id, zone samples and car fixes by time) and turns it into a
// replay-shaped object; liveModel() wraps a ReplayModel rebuilt from it after each poll, so every view reads a
// live run through the same contract. Rebuilding is O(data) and runs at most once a second.

const STEP = 0.2; // the replay's step (sources/replay.py STEP_S)
const TASK_COLS = ["anchor", "type", "cls", "outcome", "attempts", "deadline", "latency", "energy", "upload", "forward", "wait", "run", "back"];
const ATT_COLS = ["attempt", "executor", "option", "slot", "result", "reward", "decided", "send", "arrive", "start", "finish", "car", "energy"];

function addSorted(list, t, item) { // samples arrive nearly in order; equal times replace
  let i = list.length;
  while (i > 0 && list[i - 1][0] > t) i--;
  if (i > 0 && list[i - 1][0] === t) list[i - 1] = item; else list.splice(i, 0, item);
}

export class LiveBuffer {
  constructor() {
    this.meta = null; this.layout = null; this.cursor = 0; this.now = 0;
    this.tasks = new Map(); this.cars = new Map(); this.zones = new Map(); this.agents = {};
  }
  get ready() { return !!this.layout && this.cars.size > 0 && this.zones.size > 0; }
  add(d) {
    if (d.meta) { this.meta = d.meta; this.layout = d.layout; }
    this.cursor = Math.max(this.cursor, d.cursor || 0);
    let changed = false;
    for (const t of d.tasks) {
      if (t.t == null) continue;
      if (!this.tasks.has(t.id)) changed = true;
      this.tasks.set(t.id, t);
      this.now = Math.max(this.now, t.close ?? t.t);
    }
    for (const [zone, t, f] of d.zones) {
      if (!this.zones.has(zone)) this.zones.set(zone, []);
      addSorted(this.zones.get(zone), t, [t, f]);
      if (f.updates != null || f.loss != null) this.agents[zone] = f;
      this.now = Math.max(this.now, t); changed = true;
    }
    for (const [car, zone, t, x, y, v] of d.cars) {
      if (!this.cars.has(car)) this.cars.set(car, []);
      addSorted(this.cars.get(car), t, [t, x, y, v, zone]);
      this.now = Math.max(this.now, t); changed = true;
    }
    return changed;
  }
  toReplay() {
    const end = Math.max(STEP, this.now), n = Math.floor(end / STEP + 1e-9) + 1;
    const ids = [...this.cars.keys()].sort(), index = new Map(ids.map((c, i) => [c, i]));
    const cars = { ids, step: STEP, x: [], y: [], speed: [], zone: [] };
    for (const c of ids) {
      const X = new Array(n).fill(null), Y = new Array(n).fill(null), S = new Array(n).fill(0), Z = new Array(n).fill("");
      const fixes = this.cars.get(c);
      for (let k = 0; k < fixes.length; k++) { // each fix on its frame; short gaps (< 1 s) filled between fixes
        const [t, x, y, v, z] = fixes[k], f = Math.round(t / STEP);
        if (f < 0 || f >= n) continue;
        X[f] = x; Y[f] = y; S[f] = v; Z[f] = z;
        const next = fixes[k + 1];
        if (!next) continue;
        const g = Math.round(next[0] / STEP);
        if (g - f > 1 && g - f <= 5) for (let j = f + 1; j < g && j < n; j++) {
          const u = (j - f) / (g - f);
          X[j] = x + (next[1] - x) * u; Y[j] = y + (next[2] - y) * u; S[j] = v; Z[j] = z;
        }
      }
      cars.x.push(X); cars.y.push(Y); cars.speed.push(S); cars.zone.push(Z);
    }
    const zones = {};
    for (const [z, samples] of this.zones) {
      const info = (this.layout.zones.find((q) => q.id === z)) || {};
      const out = (zones[z] = {});
      for (const slot of ["gpu", "cpu"]) {
        const cap = info[slot] || 0;
        const s = (out[slot] = { busy: new Array(n).fill(0), queue: new Array(n).fill(0), rho: new Array(n).fill(0) });
        let k = 0, cur = null;
        for (let f = 0; f < n; f++) { // the newest sample at or before each frame
          while (k < samples.length && samples[k][0] <= f * STEP + 1e-9) cur = samples[k++][1];
          if (!cur) continue;
          const busy = cur[`busy_${slot}`] || 0;
          s.busy[f] = busy; s.queue[f] = cur[`queued_${slot}`] || 0; s.rho[f] = cap ? Math.min(1, busy / cap) : 0;
        }
      }
    }
    const list = [...this.tasks.values()].sort((a, b) => a.t - b.t);
    const tasks = Object.fromEntries(["id", "car", "t", "close", ...TASK_COLS].map((k) => [k, []]));
    const attempts = Object.fromEntries(["task", ...ATT_COLS].map((k) => [k, []]));
    list.forEach((o, i) => {
      tasks.id.push(o.id); tasks.car.push(index.has(o.car) ? index.get(o.car) : -1); tasks.t.push(o.t); tasks.close.push(o.close);
      for (const k of TASK_COLS) tasks[k].push(o[k] ?? null);
      for (const a of o.att || []) { attempts.task.push(i); for (const k of ATT_COLS) attempts[k].push(a[k] ?? null); }
    });
    const meta = { ...this.meta, end: end, seconds: end, fleet: ids.length };
    return { meta, layout: this.layout, cars, zones, pipes: {}, tasks, attempts };
  }
}

export function liveModel(buffer) {
  const box = { inner: new ReplayModel(buffer.toReplay()), buffer };
  return new Proxy(box, {
    get(b, k) {
      if (k === "refresh") return () => { b.inner = new ReplayModel(b.buffer.toReplay()); };
      if (k === "agents") return b.buffer.agents;
      if (k === "live") return true;
      const v = b.inner[k];
      return typeof v === "function" ? v.bind(b.inner) : v;
    },
  });
}
