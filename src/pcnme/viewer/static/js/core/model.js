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
