"""The exported viewer (src/pcnme/viewer/export.py, standalone.py): a folder that runs with Python alone."""

from __future__ import annotations

import gzip
import importlib.util
import json
import threading
import urllib.request
from pathlib import Path

from pcnme.viewer.export import export
from tests.viewer.test_server import get, registry


def test_an_export_holds_the_page_the_replays_and_a_server_that_needs_no_repo(tmp_path: Path) -> None:
    out = tmp_path / "viewer"
    assert export(out, {3000}, registry(), log=False) == 1
    assert (out / "static" / "index.html").exists() and (out / "serve.py").exists()
    replay = out / "data" / "replay" / "demo" / "S1-greedy-3000-fast.json.gz"
    assert json.loads(gzip.decompress(replay.read_bytes()))["meta"]["run_id"] == "S1-greedy-3000-fast"
    assert "import pcnme" not in (out / "serve.py").read_text(encoding="utf-8")  # standard library only

    spec = importlib.util.spec_from_file_location("exported_serve", out / "serve.py")
    assert spec and spec.loader
    serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve)
    httpd = serve.ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert get(base + "/")[0] == 200 and get(base + "/js/main.js")[1].startswith("text/javascript")
        assert json.loads(get(base + "/api/sources")[2])[0]["id"] == "demo"
        assert json.loads(get(base + "/api/runs?source=demo")[2])[0]["seed"] == 3000
        req = urllib.request.Request(base + "/api/replay?source=demo&run=S1-greedy-3000-fast")
        with urllib.request.urlopen(req) as r:
            assert r.headers["Content-Encoding"] == "gzip"
        assert get(base + "/api/replay?source=demo&run=..%2Fx")[0] == 404
        assert get(base + "/api/runs?source=later")[0] == 404  # a source in development has no runs
    finally:
        httpd.shutdown()
