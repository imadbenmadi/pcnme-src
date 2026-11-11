// The server's JSON API (server.py). The live kinds are polled.

async function get(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

export const api = {
  sources: () => get("/api/sources"),
  runs: (source) => get(`/api/runs?source=${encodeURIComponent(source)}`),
  status: (source) => get(`/api/status?source=${encodeURIComponent(source)}`),
  replay: (source, run) => get(`/api/replay?source=${encodeURIComponent(source)}&run=${encodeURIComponent(run)}`),
  liveRuns: (source) => get(`/api/live/runs?source=${encodeURIComponent(source)}`),
  live: (source, run, since) => get(`/api/live?source=${encodeURIComponent(source)}&run=${encodeURIComponent(run)}&since=${since}`),
  trainingRuns: (source) => get(`/api/training/runs?source=${encodeURIComponent(source)}`),
  training: (source, run) => get(`/api/training?source=${encodeURIComponent(source)}&run=${encodeURIComponent(run)}`),
};
