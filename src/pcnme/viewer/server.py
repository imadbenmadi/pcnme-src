"""`uv run python view.py`: the viewer on http://127.0.0.1:8765 (viewer/README.md).

A standard-library HTTP server: the page (static/) and a small JSON API over the source registry:

    GET /api/sources                         every source: id, kind, label, status, map, notes
    GET /api/runs?source=vfc                 a replay source's runs (scenario, method, seed, mode)
    GET /api/replay?source=vfc&run=<run_id>  one run as a replay (gzip)
    GET /api/status?source=vfc               the design's scenarios x methods and what exists of each
    GET /api/live/runs?source=live           the live testbed runs in InfluxDB (or why there are none)
    GET /api/live?source=live&run=<run>&since=<ns>   what is new in a live run since the cursor
    GET /api/training/runs?source=training   every trained or training scenario x method and its state
    GET /api/training?source=training&run=<map>/<S>-<method>   its episodes and the log's last lines

The live kinds are polled (about once a second; a training run every few seconds): simple, and it survives a
restart of either side. The page picks the view by the source's kind.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import mimetypes
import threading
import webbrowser
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from pcnme.viewer.sources.base import Registry

STATIC = Path(__file__).parent / "static"
TYPES = {".js": "text/javascript", ".css": "text/css", ".html": "text/html", ".svg": "image/svg+xml"}


def api(registry: Registry, path: str, q: dict[str, str]) -> Any:
    """The JSON answer for an /api/ path; KeyError for anything unknown."""
    if path == "/api/sources":
        return registry.describe()
    if path == "/api/runs":
        return [asdict(r) for r in registry.replay_source(q["source"]).runs()]
    if path == "/api/replay":
        return registry.replay_source(q["source"]).replay(q["run"])
    if path == "/api/status":
        return registry.replay_source(q["source"]).status()
    if path == "/api/live/runs":
        return registry.of_kind(q["source"], "live-testbed").runs()  # type: ignore[attr-defined]
    if path == "/api/live":
        src = registry.of_kind(q["source"], "live-testbed")
        return src.delta(q["run"], int(q.get("since", "0") or 0))  # type: ignore[attr-defined]
    if path == "/api/training/runs":
        return registry.of_kind(q["source"], "live-training").runs()  # type: ignore[attr-defined]
    if path == "/api/training":
        return registry.of_kind(q["source"], "live-training").curve(q["run"])  # type: ignore[attr-defined]
    raise KeyError(path)


def handler(registry: Registry) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:  # quiet
            pass

        def _send(self, body: bytes, ctype: str, status: int = 200, zipped: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if zipped:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, status: int = 200) -> None:
            body = json.dumps(obj, separators=(",", ":"), allow_nan=False).encode()
            if "gzip" in self.headers.get("Accept-Encoding", "") and len(body) > 4096:
                self._send(gzip.compress(body, 5), "application/json", status, zipped=True)
            else:
                self._send(body, "application/json", status)

        def do_GET(self) -> None:  # the http.server name
            url = urlsplit(self.path)
            if url.path.startswith("/api/"):
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                try:
                    return self._json(api(registry, url.path, q))
                except KeyError as e:
                    return self._json({"error": f"not found: {e}"}, HTTPStatus.NOT_FOUND)
                except OSError as e:  # a live source's InfluxDB went away: the page says so and retries
                    return self._json({"error": f"unavailable: {e}"}, HTTPStatus.SERVICE_UNAVAILABLE)
            name = "index.html" if url.path in ("", "/") else url.path.lstrip("/")
            f = (STATIC / name).resolve()
            if STATIC.resolve() not in f.parents or not f.is_file():
                return self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
            ctype = TYPES.get(f.suffix) or mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            self._send(f.read_bytes(), f"{ctype}; charset=utf-8" if ctype.startswith("text") else ctype)

    return Handler


def serve(registry: Registry, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), handler(registry))


def main(argv: list[str] | None = None) -> int:
    from pcnme.viewer.sources import default_registry

    ap = argparse.ArgumentParser(description="PCNME viewer")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    httpd = serve(default_registry(), port=a.port)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"PCNME viewer: {url}   (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    with contextlib.suppress(KeyboardInterrupt):
        httpd.serve_forever()
    return 0
