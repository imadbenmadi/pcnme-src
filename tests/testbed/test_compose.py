"""The testbed end to end (PLAN M9.1 step 6; final-design 15.4): broker, wan-gw, cloud, 4 zones and one car
container on the shaped networks; tasks go car -> zone -> executor -> car and close on time.

Needs Docker (Docker Desktop on the home desktop): pytest -m testbed tests/testbed/test_compose.py. Runs in
its own compose project, removed afterwards.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
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

pytestmark = pytest.mark.testbed


def _run(*args: str, timeout: float = 900) -> str:
    return subprocess.run(
        [*COMPOSE, *args], capture_output=True, text=True, check=True, timeout=timeout
    ).stdout


@pytest.fixture(scope="module")
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


def test_tasks_go_car_zone_executor_car_and_close_on_time(testbed: None) -> None:
    deadline = time.monotonic() + 180
    outcomes: list[str] = []
    while time.monotonic() < deadline:
        outcomes = [o for z in ("zone-1", "zone-2", "zone-3", "zone-4") for _, o in closed(z)]
        if outcomes.count("on_time") >= 5:
            break
        time.sleep(5)
    logs = "" if outcomes.count("on_time") >= 5 else _run("logs", "--tail", "40", timeout=60)
    assert outcomes.count("on_time") >= 5, f"closed: {outcomes}\n{logs}"
