"""The running testbed for the @pytest.mark.testbed tests (PLAN M9.1 step 6, M9.2): one compose project
for the session, removed afterwards. Needs Docker (Docker Desktop on the home desktop)."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = [
    "docker",
    "compose",
    "-p",
    "pcnme-test",
    "-f",
    str(ROOT / "docker" / "compose.yaml"),
    "--profile",
    "testbed",
]
SERVICES = ["broker", "broker-net", "wan-gw", "cloud", "zone-1", "zone-2", "zone-3", "zone-4", "car-1"]
CLOSED = "import httpx, json; print(httpx.get('http://127.0.0.1:8000/closed').text)"


def _run(*args: str, timeout: float = 900) -> str:
    return subprocess.run(
        [*COMPOSE, *args], capture_output=True, text=True, check=True, timeout=timeout
    ).stdout


@pytest.fixture(scope="session")
def testbed() -> Iterator[None]:
    if shutil.which("docker") is None:
        pytest.skip("no Docker")
    _run("up", "-d", "--build", *SERVICES)
    try:
        yield
    finally:
        subprocess.run([*COMPOSE, "down", "-v", "--remove-orphans"], capture_output=True, timeout=300)


def closed(zone: str) -> list[list[str]]:
    try:
        out = _run("exec", "-T", zone, "python", "-c", CLOSED, timeout=60).strip()
    except subprocess.CalledProcessError:
        return []  # not serving yet
    rows = json.loads(out) if out else []
    return [r["$t"] if isinstance(r, dict) else r for r in rows]


def exec_py(service: str, code: str, timeout: float = 300) -> str:
    """Run Python in a running container of the testbed."""
    return _run("exec", "-T", service, "python", "-c", code, timeout=timeout)


def exec_sh(service: str, cmd: str, timeout: float = 120) -> str:
    return _run("exec", "-T", service, "sh", "-c", cmd, timeout=timeout)
