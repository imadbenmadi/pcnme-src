"""The testbed runner (PLAN M9.3; final-design 15.4, 15.7, 16.3), run on the host:

    python -m pcnme.testbed.runner --scenario S2 --seed 3000 --method greedy --seconds 60

It sets every container's shaping from the config (netem: backhaul, WAN with B_WAN, radio base delays), starts
the testbed with the run's scenario, seed and method, applies the scheduled events (config `testbed.events`,
15.7: a zone joins, zones are killed, the broker restarts), waits for the zones to finish (the run, then the
drain), and merges the rows each zone wrote (its anchored tasks, METRICS.md) into
data/vfc/results/<scenario>-<method>-<seed>-testbed/, the same tables as fast mode. A killed zone
writes nothing: the tasks it anchored are lost (15.7).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from pcnme.config import Config, load_config
from pcnme.metrics.runlog import TABLES, RunLog
from pcnme.paths import MapPaths

ROOT = Path(__file__).resolve().parents[3]
COMPOSE = ["docker", "compose", "-f", str(ROOT / "docker" / "compose.yaml"), "--profile", "testbed"]
ZONES = ["zone-1", "zone-2", "zone-3", "zone-4"]
CARS = [f"car-{i}" for i in range(1, 11)]
CARS_PER_CONTAINER = 20  # compose: PCNME_CARS "0-199/10:k" with PCNME_FLEET = 20 x containers
CORE = ["broker", "broker-net", "wan-gw", "cloud"]


def _ms(x: float) -> str:
    return f"{x * 1e3:g}"


def _interval(prefix: str, lo_hi: tuple[float, float]) -> dict[str, str]:
    """netem's uniform jitter: U(lo, hi) = mean +- half-width (docker/net-setup.sh)."""
    lo, hi = lo_hi
    return {f"{prefix}_DELAY_MS": _ms((lo + hi) / 2), f"{prefix}_JITTER_MS": _ms((hi - lo) / 2)}


def shaping_env(cfg: Config) -> dict[str, str]:
    """15.4: the shaped crossings from the config (18). Rates in whole kbit/s."""
    links, radio = cfg.links, cfg.radio
    return {
        **_interval("BACKHAUL", links.backhaul_delay),
        **_interval("WAN", links.wan_delay),
        **_interval("RADIO_UP", radio.uplink_delay),
        **_interval("RADIO_DOWN", radio.downlink_delay),
        "BACKHAUL_RATE": f"{round(links.backhaul_bw / 1e3)}kbit",
        "WAN_RATE": f"{round(cfg.derived.b_wan / 1e3)}kbit",
    }


def run_env(cfg: Config, scenario: str, seed: int, method: str, seconds: float) -> dict[str, str]:
    """The run's tags for every container. 16.3: testbed runs are evaluations, on evaluation seeds."""
    if seed not in cfg.fast_mode.seed_split["evaluation"]:
        raise ValueError(f"seed {seed} is not an evaluation seed (16.3)")
    if scenario not in cfg.scenarios:
        raise ValueError(f"unknown scenario {scenario!r}")
    return {
        "PCNME_SCENARIO": scenario,
        "PCNME_SEED": str(seed),
        "PCNME_METHOD": method,
        "PCNME_SECONDS": f"{seconds:g}",
        "PCNME_DRAIN": f"{cfg.testbed.drain_s:g}",
    }


def merge(zones_dir: Path, scenario: str, method: str, seed: int) -> RunLog:
    """One run's rows from the zones' files (each zone logs the tasks it anchored)."""
    log = RunLog(scenario, method, seed, "testbed")
    for f in sorted(zones_dir.glob("*.json")):
        rows = json.loads(f.read_text(encoding="utf-8"))
        for table in TABLES:
            log.rows[table].extend(rows.get(table, []))
    return log


def _compose(env: dict[str, str], *args: str, timeout: float = 900) -> None:
    subprocess.run([*COMPOSE, *args], env={**os.environ, **env}, check=True, timeout=timeout)


def run(
    cfg: Config, scenario: str, seed: int, method: str, seconds: float, data_dir: Path, containers: int = 10
) -> Path:
    """One testbed run, end to end; returns the run's results directory. `containers` car containers of 20
    cars each; the per-zone load is the scenario's for any fleet (lambda_car scales with it, 18)."""
    log = RunLog(scenario, method, seed, "testbed")
    zones_dir = data_dir / "results" / log.run_id / "zones"
    zones_dir.mkdir(parents=True, exist_ok=True)
    for f in zones_dir.glob("*.json"):
        f.unlink()  # a rerun of the same run id starts clean
    env = {**shaping_env(cfg), **run_env(cfg, scenario, seed, method, seconds)}
    env["PCNME_RESULTS_HOST"] = str(data_dir / "results")
    env["PCNME_FLEET"] = str(CARS_PER_CONTAINER * containers)
    _compose(env, "down", "-v", "--remove-orphans")
    _compose(env, "up", "-d", "--build", *CORE, *ZONES, *CARS[:containers])
    t0 = time.monotonic()
    for ev in cfg.testbed.events:  # 15.7, in time order
        time.sleep(max(0.0, t0 + ev.at_s - time.monotonic()))
        _compose(env, ev.action, ev.service)
    killed = {ev.service for ev in cfg.testbed.events if ev.action == "kill"}
    expected = len([z for z in ZONES if z not in killed])
    end = t0 + seconds + cfg.testbed.drain_s + 300
    while len(list(zones_dir.glob("*.json"))) < expected and time.monotonic() < end:
        time.sleep(2)  # each zone writes its rows when its run and drain are over
    logs = subprocess.run(
        [*COMPOSE, "logs", "--no-color"], env={**os.environ, **env}, capture_output=True, text=True
    )
    (zones_dir.parent / "containers.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
    _compose(env, "down", "-v", "--remove-orphans")
    out = merge(zones_dir, scenario, method, seed)
    return out.write(data_dir / "results", {"scenario": scenario, "seed": seed, "seconds": seconds})


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - drives Docker
    p = argparse.ArgumentParser(description="One PCNME testbed run (PLAN M9.3)")
    p.add_argument("--scenario", default="S2")
    p.add_argument("--seed", type=int, default=3000)
    p.add_argument("--method", default="greedy", choices=("greedy", "random"))
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--containers", type=int, default=10, help="car containers (20 cars each)")
    p.add_argument("--config", type=Path, default=None)
    a = p.parse_args(argv)
    cfg = load_config(a.config) if a.config else load_config()
    print(run(cfg, a.scenario, a.seed, a.method, a.seconds, MapPaths("vfc").root, a.containers))


if __name__ == "__main__":  # pragma: no cover
    main()
