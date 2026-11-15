"""stop_all.py: stop everything PCNME runs on this machine, on Windows and in WSL / Linux.

    uv run python stop_all.py [--images] [--volumes] [--dry-run]

It stops the project's containers (`docker compose down` over every profile in docker/compose.yaml),
removes the project's networks, stops the entry points (view.py, run_live.py, run_testbed.py, train.py,
evaluate.py) and frees the ports PCNME uses. `--images` also deletes the pcnme images, `--volumes` the
InfluxDB and Grafana volumes. It prints the plan first. It deletes no file: data/ is never touched (the
testbed's data/ is a bind mount, not a volume). Docker's own processes are never killed: a port they hold
is freed by stopping its container.

A Windows shell sees only Windows processes and a WSL shell only Linux ones: run it where the runs were
started.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = ROOT / "docker" / "compose.yaml"
SCRIPTS = ("view.py", "run_live.py", "run_testbed.py", "train.py", "evaluate.py")
PORTS = (8765, 3000, 8086, 1883, 18083, 8000)
VOLUMES = ("influx-data", "grafana-data")  # compose volume keys; docker names them <project>_<key>
IMAGE = "pcnme"  # pcnme:dev and compose-built pcnme-<service> images
# Port holders that belong to Docker or the OS: never killed (stopping the container frees the port).
PROTECTED = (
    "docker",
    "com.docker",
    "containerd",
    "vpnkit",
    "wslrelay",
    "wsl",
    "rootlesskit",
    "system",
    "svchost",
)
SCRIPT_RE = re.compile(r"(?:^|[\s\"'\\/])(" + "|".join(re.escape(s) for s in SCRIPTS) + r")(?=$|[\s\"'])")
PYTHON_RE = re.compile(r"python|(?:^|[\s\"'\\/])uv(?:\.exe)?(?=$|[\s\"'])", re.IGNORECASE)


@dataclass(frozen=True)
class Proc:
    pid: int
    ppid: int
    name: str
    cmd: str


@dataclass(frozen=True)
class Plan:
    containers: list[str]
    networks: list[str]
    volumes: list[str]
    images: list[str]
    procs: list[Proc]
    ports: dict[int, list[Proc]]


# ---------- pure helpers (tested) ----------


def compose_project(compose: dict[str, object]) -> str:
    return str(compose.get("name") or "docker")


def compose_profiles(compose: dict[str, object]) -> list[str]:
    services = compose.get("services") or {}
    assert isinstance(services, dict)
    found = {p for s in services.values() for p in (s or {}).get("profiles", [])}
    return sorted(found)


def compose_down(project_file: str, profiles: list[str], docker: str = "docker") -> list[str]:
    cmd = [docker, "compose", "-f", project_file]
    for p in profiles:
        cmd += ["--profile", p]
    return [*cmd, "down", "--remove-orphans"]  # never -v or --rmi: those are the explicit options


def is_entry_point(cmd: str) -> bool:
    """A python (or `uv run`) process running one of the five entry points."""
    return bool(SCRIPT_RE.search(cmd)) and bool(PYTHON_RE.search(cmd))


def is_protected(name: str) -> bool:
    return name.lower().startswith(PROTECTED)


def ancestors(pid: int, table: dict[int, Proc]) -> set[int]:
    out: set[int] = set()
    while pid in table and pid not in out:
        out.add(pid)
        pid = table[pid].ppid
    return out


def descendants(pids: set[int], table: dict[int, Proc]) -> set[int]:
    out = set(pids)
    grew = True
    while grew:
        more = {p.pid for p in table.values() if p.ppid in out} - out
        out |= more
        grew = bool(more)
    return out


def parse_windows_procs(text: str) -> dict[int, Proc]:
    """Lines `pid<TAB>ppid<TAB>name<TAB>commandline` (see list_procs)."""
    out = {}
    for line in text.splitlines():
        parts = line.split("\t", 3)
        if len(parts) == 4 and parts[0].isdigit() and parts[1].isdigit():
            out[int(parts[0])] = Proc(int(parts[0]), int(parts[1]), parts[2], parts[3])
    return out


def parse_netstat(text: str, ports: tuple[int, ...]) -> dict[int, set[int]]:
    """`netstat -ano` on Windows: port -> listening pids."""
    out: dict[int, set[int]] = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) == 5 and f[0].upper() == "TCP" and f[3].upper() == "LISTENING" and f[4].isdigit():
            port = int(f[1].rsplit(":", 1)[1])
            if port in ports:
                out.setdefault(port, set()).add(int(f[4]))
    return out


def parse_ss(text: str, ports: tuple[int, ...]) -> dict[int, set[int]]:
    """`ss -ltnpH` on Linux: port -> listening pids (only the user's own processes show a pid)."""
    out: dict[int, set[int]] = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) < 4:
            continue
        try:
            port = int(f[3].rsplit(":", 1)[1])
        except (IndexError, ValueError):
            continue
        if port in ports:
            out.setdefault(port, set()).update(int(p) for p in re.findall(r"pid=(\d+)", line))
    return out


# ---------- the machine ----------


def run(cmd: list[str], timeout: float = 120) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def list_procs() -> dict[int, Proc]:
    if sys.platform == "win32":
        ps = (
            "Get-CimInstance Win32_Process | ForEach-Object { "
            '"$($_.ProcessId)`t$($_.ParentProcessId)`t$($_.Name)`t$($_.CommandLine)" }'
        )
        r = run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps])
        return parse_windows_procs(r.stdout) if r and r.returncode == 0 else {}
    out = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            stat = (d / "stat").read_text()
        except OSError:
            continue
        name = stat[stat.index("(") + 1 : stat.rindex(")")]
        ppid = int(stat[stat.rindex(")") + 2 :].split()[1])
        out[int(d.name)] = Proc(int(d.name), ppid, name, cmd)
    return out


def port_holders(ports: tuple[int, ...]) -> dict[int, set[int]]:
    if sys.platform == "win32":
        r = run(["netstat", "-ano", "-p", "TCP"])
        r6 = run(["netstat", "-ano", "-p", "TCPv6"])
        text = (r.stdout if r else "") + (r6.stdout if r6 else "")
        return parse_netstat(text, ports)
    r = run(["ss", "-ltnpH"])
    return parse_ss(r.stdout, ports) if r and r.returncode == 0 else {}


_DOCKER: list[str] = []


def docker() -> str:
    """`docker`; in WSL without Docker Desktop's WSL integration, the Windows `docker.exe` (interop)."""
    if not _DOCKER:
        for c in ("docker", "docker.exe"):
            r = run([c, "ps", "-q"], 30)  # not `info`: the WSL stub without integration answers it
            if r and r.returncode == 0:
                _DOCKER.append(c)
                break
        else:
            _DOCKER.append("docker")
    return _DOCKER[0]


def compose_path() -> str:
    """The compose file as the docker client sees it (a Windows path for docker.exe called from WSL)."""
    if docker() == "docker.exe" and sys.platform != "win32":
        r = run(["wslpath", "-w", str(COMPOSE_FILE)])
        if r and r.returncode == 0:
            return r.stdout.strip()
    return str(COMPOSE_FILE)


def docker_lines(args: list[str]) -> list[str] | None:
    r = run([docker(), *args], timeout=30)
    if r is None or r.returncode != 0:
        return None
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def project_networks(project: str) -> list[str]:
    by_label = docker_lines(["network", "ls", "--filter", f"label=com.docker.compose.project={project}",
                             "--format", "{{.Name}}"]) or []  # fmt: skip
    by_name = docker_lines(["network", "ls", "--format", "{{.Name}}"]) or []
    return sorted(set(by_label) | {n for n in by_name if n.startswith(project + "_")})


def make_plan(images: bool, volumes: bool) -> tuple[Plan, bool]:
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    project = compose_project(compose)
    containers = docker_lines(["ps", "-a", "--filter", f"label=com.docker.compose.project={project}",
                               "--format", "{{.Names}} ({{.Status}})"])  # fmt: skip
    docker_ok = containers is not None
    vols: list[str] = []
    imgs: list[str] = []
    if docker_ok and volumes:
        have = set(docker_lines(["volume", "ls", "--format", "{{.Name}}"]) or [])
        vols = [f"{project}_{v}" for v in VOLUMES if f"{project}_{v}" in have]
    if docker_ok and images:
        refs = docker_lines(["images", "--format", "{{.Repository}}:{{.Tag}}"]) or []
        imgs = sorted(
            {r for r in refs if r.split(":")[0] == IMAGE or r.startswith((IMAGE + "-", IMAGE + "_"))}
        )
    table = list_procs()
    me = ancestors(os.getpid(), table)
    procs = [p for p in table.values() if p.pid not in me and is_entry_point(p.cmd)]
    ports = {port: [table.get(pid, Proc(pid, 0, "?", "")) for pid in sorted(pids) if pid not in me]
             for port, pids in port_holders(PORTS).items()}  # fmt: skip
    plan = Plan(containers or [], project_networks(project) if docker_ok else [], vols, imgs, procs, ports)
    return plan, docker_ok


def show(plan: Plan, docker_ok: bool, images: bool, volumes: bool) -> None:
    print("stop_all: the plan (data/ is never touched)")
    if not docker_ok:
        print("  docker: not reachable, skipping containers, networks, volumes and images")
    else:
        print(f"  containers ({len(plan.containers)}): " + (", ".join(plan.containers) or "none"))
        print(f"  networks ({len(plan.networks)}): " + (", ".join(plan.networks) or "none"))
        if volumes:
            print(f"  volumes ({len(plan.volumes)}): " + (", ".join(plan.volumes) or "none"))
        if images:
            print(f"  images ({len(plan.images)}): " + (", ".join(plan.images) or "none"))
    print(f"  processes ({len(plan.procs)}):" + ("" if plan.procs else " none"))
    for p in plan.procs:
        print(f"    {p.pid}  {p.cmd[:150]}")
    for port in PORTS:
        held = plan.ports.get(port, [])
        what = ", ".join(
            f"{h.pid} {h.name}" + (" (Docker: freed with its container)" if is_protected(h.name) else "")
            for h in held
        )
        print(f"  port {port}: " + (what or ("in use" if port in plan.ports else "free")))


def kill(pids: set[int]) -> None:
    if not pids:
        return
    if sys.platform == "win32":
        for pid in sorted(pids):
            run(["taskkill", "/T", "/F", "/PID", str(pid)], timeout=30)
        return
    table = list_procs()
    everyone = descendants(pids, table) - ancestors(os.getpid(), table)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in everyone:
            with contextlib.suppress(OSError):
                os.kill(pid, sig)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(Path(f"/proc/{p}").exists() for p in everyone):
            time.sleep(0.2)
        everyone = {p for p in everyone if Path(f"/proc/{p}").exists()}
        if not everyone:
            return


def execute(plan: Plan, docker_ok: bool) -> None:
    if docker_ok:
        compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
        project = compose_project(compose)
        print("docker compose down (every profile) ...")
        r = run(compose_down(compose_path(), compose_profiles(compose), docker()), timeout=300)
        if r is None or r.returncode != 0:
            print(
                "  compose down failed: "
                + ((r.stderr.strip().splitlines() or [""])[-1] if r else "no docker")
            )
        left = docker_lines(["ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"]) or []
        if left:
            run([docker(), "rm", "-f", *left])
        for n in project_networks(project):
            run([docker(), "network", "rm", n])
        if plan.volumes:
            run([docker(), "volume", "rm", *plan.volumes])
        if plan.images:
            run([docker(), "rmi", "-f", *plan.images])
    print(f"stopping {len(plan.procs)} process(es) ...")
    kill({p.pid for p in plan.procs})
    table = list_procs()
    me = ancestors(os.getpid(), table)
    holders = port_holders(PORTS)
    rest = {pid for pids in holders.values() for pid in pids if pid not in me}
    kill({pid for pid in rest if not is_protected(table.get(pid, Proc(pid, 0, "?", "")).name)})


def report() -> int:
    held = port_holders(PORTS)
    table = list_procs()
    busy = {port: sorted(pids) for port, pids in held.items()}
    for port, pids in sorted(busy.items()):
        names = ", ".join(f"{p} {table[p].name}" if p in table else str(p) for p in pids) or "unknown owner"
        print(f"  port {port} still in use: {names}")
    print("done" if not busy else "done, but some ports are still in use (see above)")
    return 0 if not busy else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", action="store_true", help="also delete the pcnme images")
    ap.add_argument("--volumes", action="store_true", help="also delete the InfluxDB and Grafana volumes")
    ap.add_argument("--dry-run", action="store_true", help="print the plan only")
    a = ap.parse_args(argv)
    plan, docker_ok = make_plan(a.images, a.volumes)
    show(plan, docker_ok, a.images, a.volumes)
    if a.dry_run:
        return 0
    execute(plan, docker_ok)
    return report()
