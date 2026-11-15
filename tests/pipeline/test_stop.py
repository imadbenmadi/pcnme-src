"""stop_all.py (src/pcnme/pipeline/stop.py): what it matches and parses; it stops a real entry point."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from pcnme.pipeline import stop
from pcnme.pipeline.stop import Proc


def test_entry_points_are_matched_and_nothing_else() -> None:
    yes = [
        r'"C:\venv\Scripts\python.exe" train.py --maps vfc',
        "uv run python evaluate.py --maps lust",
        r"C:\Users\x\.local\bin\uv.exe run python F:\projects\pcnme-src\view.py",
        "/home/x/.venv/bin/python3 /home/x/pcnme-src/run_testbed.py --force",
        "python run_live.py --scenario S2",
    ]
    no = [
        "python tools/plot_curves.py",
        "python preview.py",
        "python stop_all.py",
        "code train.py",  # an editor, not a run
        "python -m pytest tests/pipeline/test_train.py",
        "notepad view.py.bak",
    ]
    assert all(stop.is_entry_point(c) for c in yes)
    assert not any(stop.is_entry_point(c) for c in no)


def test_the_compose_down_covers_every_profile_and_never_deletes_volumes_or_images() -> None:
    compose = yaml.safe_load(stop.COMPOSE_FILE.read_text(encoding="utf-8"))
    profiles = stop.compose_profiles(compose)
    assert {"testbed", "monitoring", "zone5"} <= set(profiles)
    assert stop.compose_project(compose) == "pcnme"
    cmd = stop.compose_down(str(stop.COMPOSE_FILE), profiles)
    assert cmd[-2:] == ["down", "--remove-orphans"] and "-v" not in cmd and "--rmi" not in cmd
    assert all(cmd[cmd.index(p) - 1] == "--profile" for p in profiles)
    assert set(stop.VOLUMES) == set(compose["volumes"])  # --volumes deletes exactly InfluxDB's and Grafana's


def test_ports_are_parsed_on_both_systems() -> None:
    netstat = """
  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:3000           0.0.0.0:0              LISTENING       4120
  TCP    127.0.0.1:8765         0.0.0.0:0              LISTENING       900
  TCP    127.0.0.1:8765         127.0.0.1:50000        ESTABLISHED     900
  TCP    0.0.0.0:445            0.0.0.0:0              LISTENING       4
  TCP    [::]:8086              [::]:0                 LISTENING       4120
"""
    assert stop.parse_netstat(netstat, stop.PORTS) == {3000: {4120}, 8765: {900}, 8086: {4120}}
    ss = """LISTEN 0 5 127.0.0.1:8765 0.0.0.0:* users:(("python",pid=321,fd=3))
LISTEN 0 4096 [::]:1883 [::]:*
LISTEN 0 128 0.0.0.0:22 0.0.0.0:* """
    assert stop.parse_ss(ss, stop.PORTS) == {8765: {321}, 1883: set()}


def test_windows_process_lines_and_the_process_tree() -> None:
    lines = "10\t1\tpython.exe\tpython train.py\n11\t10\tpython.exe\tpython -c spawn\n"
    table = stop.parse_windows_procs(lines + "12\t11\tpython.exe\t\nbad line\n")
    assert table[10] == Proc(10, 1, "python.exe", "python train.py") and table[12].cmd == ""
    assert stop.descendants({10}, table) == {10, 11, 12}
    assert stop.ancestors(12, table) == {12, 11, 10}


def test_docker_and_system_processes_are_protected() -> None:
    assert all(
        stop.is_protected(n)
        for n in ("com.docker.backend.exe", "wslrelay.exe", "docker-proxy", "svchost.exe")
    )
    assert not any(stop.is_protected(n) for n in ("python.exe", "python3", "uv.exe", "grafana"))


def test_a_dry_run_prints_the_plan_and_stops_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = Proc(424242, 1, "python", "python train.py --maps vfc")
    monkeypatch.setattr(stop, "docker_lines", lambda args: None)
    monkeypatch.setattr(stop, "list_procs", lambda: {fake.pid: fake})
    monkeypatch.setattr(stop, "port_holders", lambda ports: {8765: {fake.pid}})
    monkeypatch.setattr(stop, "kill", lambda pids: pytest.fail("a dry run kills nothing"))
    monkeypatch.setattr(stop, "run", lambda *a, **k: pytest.fail("a dry run runs nothing"))
    assert stop.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "424242" in out and "docker: not reachable" in out and "port 8765: 424242 python" in out


def test_a_running_entry_point_is_found_and_stopped(tmp_path: Path) -> None:
    script = tmp_path / "view.py"
    script.write_text("import time\ntime.sleep(120)\n", encoding="utf-8")
    child = subprocess.Popen([sys.executable, str(script)])
    try:
        deadline = time.monotonic() + 10
        found: list[Proc] = []
        while not found and time.monotonic() < deadline:
            found = [p for p in stop.list_procs().values() if p.pid == child.pid]
            time.sleep(0.1)
        assert found and stop.is_entry_point(found[0].cmd)
        stop.kill({child.pid})
        assert child.wait(timeout=15) is not None
    finally:
        if child.poll() is None:
            child.kill()
