// The server's JSON API (server.py). Live sources will add a stream here (EventSource or WebSocket).

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
};
