"""The exported viewer's server: python serve.py (copied next to an export by tools/export_viewer.py).

Standard library only, no pcnme import: it runs on any machine with Python 3.9+. It answers the same API as
server.py from files written by the export:

    data/sources.json                  GET /api/sources
    data/runs/<source>.json            GET /api/runs?source=<source>
    data/replay/<source>/<run>.json.gz GET /api/replay?source=<source>&run=<run>
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

HERE = Path(__file__).resolve().parent
STATIC, DATA = HERE / "static", HERE / "data"
TYPES = {".js": "text/javascript", ".css": "text/css", ".html": "text/html", ".svg": "image/svg+xml"}
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")  # source and run ids: never a path


def api_file(path: str, q: dict[str, str]) -> Path | None:
    """The data file for an /api/ request, or None."""
    source, run = q.get("source", ""), q.get("run", "")
    if path == "/api/sources":
        return DATA / "sources.json"
    if path == "/api/runs" and SAFE.match(source):
        return DATA / "runs" / f"{source}.json"
    if path == "/api/replay" and SAFE.match(source) and SAFE.match(run):
        return DATA / "replay" / source / f"{run}.json.gz"
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_: Any) -> None:
        pass

    def send(self, body: bytes, ctype: str, status: int = 200, gz: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if url.path.startswith("/api/"):
            f = api_file(url.path, {k: v[0] for k, v in parse_qs(url.query).items()})
            if f is None or not f.is_file():
                body = json.dumps({"error": "not in this export"}).encode()
                return self.send(body, "application/json", 404)
            return self.send(f.read_bytes(), "application/json", gz=f.suffix == ".gz")
        name = "index.html" if url.path in ("", "/") else url.path.lstrip("/")
        f = (STATIC / name).resolve()
        if STATIC.resolve() not in f.parents or not f.is_file():
            return self.send(b"not found", "text/plain", 404)
        ctype = TYPES.get(f.suffix, "application/octet-stream")
        self.send(f.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text") else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description="PCNME viewer (exported runs)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"PCNME viewer: {url}   (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    with contextlib.suppress(KeyboardInterrupt):
        httpd.serve_forever()


if __name__ == "__main__":
    main()
