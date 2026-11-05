// Source adapters on the page: one per kind (server side: sources/base.py). An adapter turns a source into
// models the views can draw. Add a kind by adding an adapter here; the views stay as they are.
//
//   replay         listRuns() -> runs;  open(runId) -> ReplayModel           (now)
//   live-testbed   connect() -> a model fed by /api/stream, the clock follows (later)
//   live-training  connect() -> a training model (episodes, loss, on-time, decisions) for a training view (later)

import { api } from "./api.js";
import { ReplayModel } from "./model.js";

class ReplayAdapter {
  constructor(info) { this.info = info; this.cache = new Map(); }
  async listRuns() { return this.runs ?? (this.runs = await api.runs(this.info.id)); }
  async open(runId) {
    if (!this.cache.has(runId)) this.cache.set(runId, api.replay(this.info.id, runId).then((d) => new ReplayModel(d)));
    return this.cache.get(runId);
  }
}

const ADAPTERS = { replay: ReplayAdapter };

export function adapterFor(info) {
  const A = ADAPTERS[info.kind];
  return info.status === "ready" && A ? new A(info) : null;
}
