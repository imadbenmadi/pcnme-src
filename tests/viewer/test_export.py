"""The exported viewer (src/pcnme/viewer/export.py, standalone.py): a folder that runs with Python alone."""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
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


def test_a_run_rerun_under_its_id_replaces_its_exported_replay(tmp_path: Path) -> None:
    """15 Nov: the export kept every replay already in the folder, so the testbed runs rerun after the fixes
    and S6's rerun (Q60) were exported with their old content."""
    import time

    out = tmp_path / "viewer"
    reg = registry()
    src = reg.get("demo")
    export(out, {3000}, reg, log=False)
    replay = out / "data" / "replay" / "demo" / "S1-greedy-3000-fast.json.gz"
    assert json.loads(gzip.decompress(replay.read_bytes()))["meta"]["version"] == 1
    src.version, src.written = 2, time.time() + 60  # type: ignore[attr-defined]  # the run is rerun
    export(out, {3000}, reg, log=False)
    assert json.loads(gzip.decompress(replay.read_bytes()))["meta"]["version"] == 2


def test_the_export_starts_and_stops_with_its_launchers(tmp_path: Path) -> None:
    out = tmp_path / "viewer"
    export(out, {3000}, registry(), log=False)
    assert {"start.bat", "stop.bat", "start.sh", "stop.sh"} <= {f.name for f in out.iterdir()}
    port = 8000 + os.getpid() % 900
    proc = subprocess.Popen([sys.executable, "serve.py", "--port", str(port), "--no-browser"], cwd=out)
    try:
        deadline = time.monotonic() + 15
        while not (out / "serve.pid").exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert (out / "serve.pid").read_text(encoding="utf-8").strip().isdigit()  # the server, not a launcher
        subprocess.run([sys.executable, "serve.py", "--stop"], cwd=out, check=True, capture_output=True)
        assert proc.wait(timeout=15) is not None
    finally:
        if proc.poll() is None:
            proc.kill()
