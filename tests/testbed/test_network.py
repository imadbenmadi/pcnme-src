"""The testbed's emulated network (PLAN M9.2; final-design 15.4 "to verify", 4.2, 18): routing through wan-gw,
netem's delays against the config's intervals, transfers not distorted by jitter reordering, warm TCP, and one
shaped crossing per hop. Runs on the compose stack of conftest.py: pytest -m testbed."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Iterator

import pytest

from pcnme.config import load_config
from tests.testbed.conftest import _run, exec_py, exec_sh

pytestmark = pytest.mark.testbed


@pytest.fixture(scope="module")
def quiet(testbed: None) -> Iterator[None]:
    """The network alone: no car traffic, so the pipes hold no other uploads (W = 0 in 4.2)."""
    _run("stop", "car-1", timeout=120)
    time.sleep(5)
    yield
    _run("start", "car-1", timeout=120)


CFG = load_config()
BH = CFG.links.backhaul_delay  # one backhaul crossing, s (uniform)
WAN = CFG.links.wan_delay  # one WAN crossing, s (uniform)
SLACK = 0.003  # s: container and Python overhead on a round trip

RTT = """
import json, socket, time
xs = []
for _ in range({n}):
    t = time.perf_counter()
    s = socket.create_connection(("{host}", 8000), timeout=5)
    xs.append(time.perf_counter() - t)
    s.close()
    time.sleep(0.02)
print(json.dumps(xs))
"""


def rtts(src: str, host: str, n: int = 150) -> list[float]:
    """TCP connect times: SYN out, SYN-ACK back = one round trip. The first is dropped (ARP, warm-up)."""
    return list(json.loads(exec_py(src, RTT.format(host=host, n=n + 1))))[1:]


def test_cloud_traffic_routes_through_wan_gw(testbed: None) -> None:
    assert "via 10.20.0.3" in exec_sh("zone-1", "ip route get 10.30.0.10")
    assert "via 10.30.0.3" in exec_sh("cloud", "ip route get 10.20.0.11")


@pytest.mark.parametrize("service", ["zone-1", "zone-2", "cloud", "wan-gw", "car-1"])
def test_connections_stay_warm(testbed: None, service: str) -> None:
    assert exec_sh(service, "cat /proc/sys/net/ipv4/tcp_slow_start_after_idle").strip() == "0"


@pytest.mark.parametrize(
    ("service", "shaped", "unshaped"),
    [
        ("zone-1", ["10.20.0.", "10.10.0."], ["10.40.0."]),
        ("wan-gw", ["10.30.0.", "10.20.0."], ["10.40.0."]),
        ("cloud", ["10.30.0."], ["10.40.0."]),
        ("car-1", ["10.10.0."], ["10.40.0."]),
        ("broker-net", ["10.20.0."], ["10.40.0."]),
    ],
)
def test_one_shaped_crossing_per_hop(
    testbed: None, service: str, shaped: list[str], unshaped: list[str]
) -> None:
    """Each egress interface on a data path has exactly one netem; test equipment (mgmt_net) has none."""

    def qdisc(prefix: str) -> str:
        cmd = f"tc qdisc show dev $(ip -o -4 addr show | awk '$4 ~ /^{prefix}/ {{print $2; exit}}')"
        return exec_sh(service, cmd)

    for p in shaped:
        assert qdisc(p).count("netem") == 1, (service, p)
    for p in unshaped:
        assert "netem" not in qdisc(p), (service, p)


@pytest.mark.usefixtures("quiet")
def test_zone_to_zone_round_trip_is_two_backhaul_crossings(testbed: None) -> None:
    xs = rtts("zone-1", "10.20.0.12")
    lo, hi = 2 * BH[0], 2 * BH[1]
    assert min(xs) >= lo - 0.0005 and max(xs) <= hi + SLACK
    assert statistics.mean(xs) == pytest.approx((lo + hi) / 2, abs=0.0015)


@pytest.mark.usefixtures("quiet")
def test_zone_to_cloud_round_trip_is_backhaul_plus_wan_each_way(testbed: None) -> None:
    xs = rtts("zone-1", "10.30.0.10")
    lo, hi = 2 * (BH[0] + WAN[0]), 2 * (BH[1] + WAN[1])
    assert min(xs) >= lo - 0.0005 and max(xs) <= hi + SLACK
    # Q56: with packets kept in order, a packet waits for the one ahead of it, so under the testbed's own
    # traffic (state, results, uploads) each crossing's delay drifts up: +7.6 % idle, +24 % loaded (3 Nov).
    mean = (lo + hi) / 2
    assert 0.95 * mean <= statistics.mean(xs) <= 1.30 * mean


SINK = """
import socket, struct, threading
def serve(c):
    while True:
        h = c.recv(4, socket.MSG_WAITALL)
        if len(h) < 4:
            return
        n = struct.unpack("!I", h)[0]
        while n:
            d = c.recv(min(n, 65536))
            if not d:
                return
            n -= len(d)
        c.sendall(b"k")
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", 9101))
s.listen(4)
while True:
    threading.Thread(target=serve, args=(s.accept()[0],), daemon=True).start()
"""

BULK = """
import json, socket, struct, time
s = socket.create_connection(("10.30.0.10", 9101))
def send(n):
    t = time.perf_counter()
    s.sendall(struct.pack("!I", n) + b"x" * n)
    s.recv(1)
    return time.perf_counter() - t
for _ in range(20):  # warm the connection: its window grows over about 15 transfers (measured 3 Nov)
    send({size})
print(json.dumps([send({size}) for _ in range({n})]))
"""


@pytest.mark.usefixtures("quiet")
def test_a_bulk_upload_takes_the_pipe_model_time(testbed: None) -> None:
    """4.2: a WAN transfer takes S / B_WAN plus its delays; jitter must not reorder packets into TCP recovery
    (Q56: netem with an in-order pfifo child). Raw TCP on one warm connection (15.4), so neither HTTP nor the
    zone's own work is timed: the upload and a one-byte answer, one crossing each way on both links."""
    _run("exec", "-d", "-T", "cloud", "python", "-c", SINK, timeout=60)
    time.sleep(1)
    size = 1_000_000
    xs = json.loads(exec_py("zone-1", BULK.format(n=10, size=size)))
    push = size * 8 / CFG.derived.b_wan
    lo, hi = push + 2 * (BH[0] + WAN[0]), push + 2 * (BH[1] + WAN[1])
    assert lo - 0.005 <= statistics.median(xs) <= hi + 0.02, (xs, lo, hi)
