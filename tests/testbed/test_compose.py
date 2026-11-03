"""The testbed end to end (PLAN M9.1 step 6; final-design 15.4): broker, wan-gw, cloud, 4 zones and one car
container on the shaped networks; tasks go car -> zone -> executor -> car and close on time.

Needs Docker (Docker Desktop on the home desktop): pytest -m testbed tests/testbed/test_compose.py. Runs in
its own compose project, removed afterwards.
"""

from __future__ import annotations

import time

import pytest

from tests.testbed.conftest import _run, closed

pytestmark = pytest.mark.testbed


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
