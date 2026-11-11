"""The testbed runner (PLAN M9.3; final-design 15.4, 15.7, 16.3): shaping from the config, the run's
environment, scheduled events, and the zones' rows merged into one run's Parquet tables."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcnme.metrics.runlog import RunLog
from pcnme.testbed.runner import merge, run_env, shaping_env
from tests.zone.harness import fixed_delay_config


def test_the_shaping_comes_from_the_config() -> None:
    cfg = fixed_delay_config(links__wan_delay=[0.025, 0.100], links__backhaul_delay=[0.001, 0.005])
    env = shaping_env(cfg)
    assert (env["WAN_DELAY_MS"], env["WAN_JITTER_MS"]) == ("62.5", "37.5")  # U(25, 100) ms = 62.5 +- 37.5
    assert (env["BACKHAUL_DELAY_MS"], env["BACKHAUL_JITTER_MS"]) == ("3", "2")
    assert env["WAN_RATE"] == f"{round(cfg.derived.b_wan / 1e3)}kbit"  # B_WAN (18)
    assert env["BACKHAUL_RATE"] == f"{round(cfg.links.backhaul_bw / 1e3)}kbit"
    up, down = cfg.radio.uplink_delay, cfg.radio.downlink_delay
    assert env["RADIO_UP_DELAY_MS"] == f"{(up[0] + up[1]) / 2 * 1e3:g}"
    assert env["RADIO_DOWN_DELAY_MS"] == f"{(down[0] + down[1]) / 2 * 1e3:g}"


def test_the_run_environment_names_scenario_seed_method_and_length() -> None:
    env = run_env(fixed_delay_config(), "S2", 3001, "greedy", 60.0)
    assert (env["PCNME_SCENARIO"], env["PCNME_SEED"], env["PCNME_METHOD"]) == ("S2", "3001", "greedy")
    assert env["PCNME_SECONDS"] == "60"


def test_an_evaluation_run_needs_an_evaluation_seed() -> None:
    with pytest.raises(ValueError, match="evaluation"):
        run_env(fixed_delay_config(), "S2", 2000, "greedy", 60.0)  # 16.3: testbed runs are evaluations


def test_the_zones_rows_merge_into_one_run(tmp_path: Path) -> None:
    zones = tmp_path / "zones"
    zones.mkdir()
    for z, n in (("z1", 2), ("z2", 1)):
        log = RunLog("S2", "greedy", 3000, "testbed")
        for i in range(n):
            log.add("tasks", float(i), anchor=z, task_id=f"{z}-{i}", outcome="on_time")
        (zones / f"{z}.json").write_text(json.dumps(log.rows), encoding="utf-8")
    log = merge(zones, "S2", "greedy", 3000)
    assert log.run_id == "S2-greedy-3000-testbed"
    assert sorted(r["task_id"] for r in log.rows["tasks"]) == ["z1-0", "z1-1", "z2-0"]


def test_live_points_are_switched_on_only_for_a_live_run() -> None:
    """PLAN R2b: measured runs push nothing (PCNME_LIVE empty); a live run names itself for Grafana."""
    from pcnme.testbed.runner import MONITORING, live_env

    assert live_env(None) == {"PCNME_LIVE": "", "PCNME_LIVE_RUN": ""}
    assert live_env("S2-pcnme-3000-1111-120000") == {
        "PCNME_LIVE": "1",
        "PCNME_LIVE_RUN": "S2-pcnme-3000-1111-120000",
    }
    assert MONITORING == ["influxdb", "grafana"]


def test_a_live_run_name_is_readable_and_unique_per_start() -> None:
    from pcnme.pipeline.live import live_name

    a = live_name("S2", "d3qn-greedy", 3000, now=1_800_000_000.0)
    assert a.startswith("S2-d3qn-greedy-3000-") and a != live_name(
        "S2", "d3qn-greedy", 3000, now=1_800_000_001.0
    )
