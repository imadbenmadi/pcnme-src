// Source adapters on the page: one per kind (server side: sources/base.py). An adapter turns a source into
// models the views can draw. Add a kind by adding an adapter here; the views stay as they are.
//
//   replay         listRuns() -> runs;  open(runId) -> ReplayModel
//   live-testbed   listRuns() -> {online, runs, help};  follow(run, onData) -> stop(): polls /api/live into a
//                  LiveBuffer and calls onData(buffer) when something new came
//   live-training  listRuns() -> runs with their state;  curve(run) -> episodes and the log's last lines

import { api } from "./api.js";
import { LiveBuffer, ReplayModel } from "./model.js";

class ReplayAdapter {
  constructor(info) { this.info = info; this.cache = new Map(); }
  async listRuns() { return this.runs ?? (this.runs = await api.runs(this.info.id)); }
  async status() { return this.state ?? (this.state = await api.status(this.info.id)); }
  async open(runId) {
    if (!this.cache.has(runId)) this.cache.set(runId, api.replay(this.info.id, runId).then((d) => new ReplayModel(d)));
    return this.cache.get(runId);
  }
}

class LiveAdapter {
  constructor(info) { this.info = info; }
  listRuns() { return api.liveRuns(this.info.id); }
  follow(run, onData, onError, every = 1000) {
    const buffer = new LiveBuffer();
    let stopped = false, timer = null;
    const poll = async () => {
      try {
        const d = await api.live(this.info.id, run, buffer.cursor);
        if (stopped) return;
        if (buffer.add(d)) onData(buffer);
      } catch (e) { if (!stopped) onError(e); }
      if (!stopped) timer = setTimeout(poll, every);
    };
    poll();
    return () => { stopped = true; clearTimeout(timer); };
  }
}

class TrainingAdapter {
  constructor(info) { this.info = info; }
  listRuns() { return api.trainingRuns(this.info.id); }
  curve(run) { return api.training(this.info.id, run); }
}

const ADAPTERS = { replay: ReplayAdapter, "live-testbed": LiveAdapter, "live-training": TrainingAdapter };

export function adapterFor(info) {
  const A = ADAPTERS[info.kind];
  return info.status === "ready" && A ? new A(info) : null;
}
