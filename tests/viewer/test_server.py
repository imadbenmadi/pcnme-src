"""The viewer's server (src/pcnme/viewer/server.py): the API over the source registry and the static page."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import pytest

from pcnme.viewer.server import api, serve
from pcnme.viewer.sources.base import Placeholder, Registry, RunInfo, SourceInfo


class FakeReplay:
    info = SourceInfo("demo", "replay", "Demo", map="vfc")

    def runs(self) -> list[RunInfo]:
        return [RunInfo("S1-greedy-3000-fast", "S1", "greedy", 3000, "fast")]

    def replay(self, run_id: str) -> dict[str, Any]:
        if run_id != "S1-greedy-3000-fast":
            raise KeyError(run_id)
        return {"meta": {"run_id": run_id}}

    def status(self) -> dict[str, Any]:
        return {"scenarios": ["S1"], "methods": ["greedy"], "seeds": [3000], "cells": {}, "notes": {}}


def registry() -> Registry:
    return Registry([FakeReplay(), Placeholder(SourceInfo("later", "replay", "Later", status="development"))])


def test_the_api_lists_sources_runs_and_replays() -> None:
    reg = registry()
    assert [(s["id"], s["status"]) for s in api(reg, "/api/sources", {})] == [
        ("demo", "ready"),
        ("later", "development"),
    ]
    assert api(reg, "/api/runs", {"source": "demo"})[0]["seed"] == 3000
    assert api(reg, "/api/replay", {"source": "demo", "run": "S1-greedy-3000-fast"})["meta"]["run_id"]
    for path, q in (("/api/runs", {"source": "later"}), ("/api/replay", {"source": "demo", "run": "x"}),
                    ("/api/nothing", {})):  # fmt: skip
        with pytest.raises(KeyError):
            api(reg, path, q)


@pytest.fixture()
def base() -> Iterator[str]:
    httpd = serve(registry(), port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def get(url: str) -> tuple[int, str, bytes]:
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.headers["Content-Type"], r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers["Content-Type"], e.read()


def test_the_server_answers_the_page_its_modules_and_the_api(base: str) -> None:
    status, ctype, body = get(base + "/")
    assert status == 200 and ctype.startswith("text/html") and b"js/main.js" in body
    status, ctype, _ = get(base + "/js/main.js")
    assert status == 200 and ctype.startswith("text/javascript")  # ES modules need a JavaScript type
    status, _, body = get(base + "/api/sources")
    assert status == 200 and json.loads(body)[0]["id"] == "demo"
    assert get(base + "/api/runs?source=later")[0] == 404


def test_nothing_outside_the_static_folder_is_served(base: str) -> None:
    assert get(base + "/../server.py")[0] == 404
    assert get(base + "/%2e%2e/server.py")[0] == 404
