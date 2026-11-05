"""S6 check (Imed, 5 Nov): which share of S6 tasks cannot meet their deadline on any option, even with every
queue empty? Read-only: the logged S6 tasks (greedy's 20 evaluation seeds; the tasks are the same for every
method) and the config's time table and delays. Nothing is changed.

For each task and each option (here, every active neighbour within d_nb, the cloud), the queue-free latency is
    logged upload + forward + run + result back
with the run time from the time table (the faster slot type of the executor's class) and the network as in
final-design 4: backhaul delay + 76 KB at 1 Gbit/s, WAN delay + 76 KB at B_WAN, radio downlink delay. Two
bounds: "typical" (mean delays, mean run times) and "best case" (smallest delays, p5 run times). Then the
capacity: the GPU work the hard tasks ask for, and an ideal shared queue's estimate (QUESTIONS Q60).

    uv run python tools/s6_feasibility.py
"""

from __future__ import annotations

import math
from collections import Counter

import pyarrow.parquet as pq

from pcnme.maps import map_config
from pcnme.paths import MapPaths


def main() -> None:
    cfg = map_config("vfc")
    sc, lay, tt = cfg.scenarios["S6"], cfg.layout, cfg.time_table.rows
    active = [z for z in lay.zones if z not in lay.joining_zones]
    nbrs = {
        a: [
            b
            for b in active
            if b != a and math.dist(lay.zone_positions_m[a], lay.zone_positions_m[b]) <= lay.d_nb_m
        ]
        for a in active
    }
    bits = {t: cfg.task_types[t].input_bits for t in ("T1", "T2", "T3", "T4")}
    res_bits = {t: cfg.task_types[t].result_bits for t in bits}

    def run(t: str, hw: str, best: bool) -> float:
        rows = [tt[(t, hw, s)] for s in ("gpu", "cpu") if (t, hw, s) in tt]
        return min((r.p5 if best else r.mean for r in rows), default=math.inf)  # no row: cannot run there

    def delay(rng: tuple[float, float], best: bool) -> float:
        return rng[0] if best else (rng[0] + rng[1]) / 2

    def options(t: str, anchor: str, best: bool) -> dict[str, float]:
        down = delay(cfg.radio.downlink_delay, best)
        bh = delay(cfg.links.backhaul_delay, best)
        wan = delay(cfg.links.wan_delay, best)
        out = {"here": run(t, sc.fog[anchor], best) + down}
        for n in nbrs[anchor]:
            fwd = bh + bits[t] / cfg.links.backhaul_bw
            back = bh + res_bits[t] / cfg.links.backhaul_bw
            out[n] = fwd + run(t, sc.fog[n], best) + back + down
        out["cloud"] = (
            (wan + bits[t] / cfg.derived.b_wan)
            + run(t, "cloud", best)
            + (wan + res_bits[t] / cfg.derived.b_wan)
            + down
        )
        return out

    runs = sorted(MapPaths("vfc").runs.glob("S6-greedy-*-fast"))
    total, infeasible = Counter(), {False: Counter(), True: Counter()}
    for d in runs:
        t = pq.read_table(
            d / "tasks.parquet", columns=["type", "anchor", "upload_s", "deadline_s"]
        ).to_pydict()
        for typ, anchor, up, dl in zip(t["type"], t["anchor"], t["upload_s"], t["deadline_s"], strict=True):
            if anchor not in nbrs or up is None:
                continue  # no anchor (uncovered) or no upload: not a placement question
            key = (typ, sc.fog[anchor])
            total[key] += 1
            for best in (False, True):
                if up + min(options(typ, anchor, best).values()) > dl:
                    infeasible[best][key] += 1
    n = sum(total.values())
    print(f"S6, {len(runs)} seeds (greedy's logged tasks), {n:,} tasks with an anchor and an upload\n")
    print("| task type | anchor class | tasks | infeasible (typical) | infeasible (best case) |")
    print("| --- | --- | --- | --- | --- |")
    for key in sorted(total):
        k = total[key]
        typical, best = infeasible[False][key] / k, infeasible[True][key] / k
        print(f"| {key[0]} | {key[1]} | {k:,} | {typical:.1%} | {best:.1%} |")
    a, b = sum(infeasible[False].values()), sum(infeasible[True].values())
    print(f"| all | | {n:,} | {a / n:.1%} | {b / n:.1%} |\n")
    for anchor in ("z2", "z1"):
        for typ in ("T1", "T3"):
            o = options(typ, anchor, False)
            print(f"{typ} anchored at {anchor} ({sc.fog[anchor]}), queue-free, typical, without upload: "
                  + ", ".join(f"{k} {1e3 * v:.1f} ms" for k, v in o.items()))  # fmt: skip
    # capacity: the GPU work S6 asks for against the GPU slots it has
    work = Counter()
    for d in runs:
        t = pq.read_table(d / "tasks.parquet", columns=["type", "anchor"]).to_pydict()
        for typ, anchor in zip(t["type"], t["anchor"], strict=True):
            if anchor in nbrs:
                work[typ] += tt[(typ, "F-GPU", "gpu")].mean if (typ, "F-GPU", "gpu") in tt else 0.0
    secs = len(runs) * cfg.fast_mode.episode_length
    gpu_slots = sum(cfg.hardware[sc.fog[z]].c_gpu for z in active)
    print("\nGPU slot-seconds asked for per second, every task on an F-GPU GPU slot: "
          + ", ".join(f"{t} {work[t] / secs:.2f}" for t in sorted(work)))  # fmt: skip
    hard = (work["T1"] + work["T3"]) / secs  # hard tasks meet a tight deadline only on a GPU slot
    print(f"Hard tasks (T1, T3): {hard:.2f} for {gpu_slots} GPU slots, utilisation {hard / gpu_slots:.2f}")
    run_s = tt[("T1", "F-GPU", "gpu")].mean
    for slack in (0.014, 0.020):  # a tight T1's slack beyond its queue-free path: to a neighbour, at home
        late = erlang_c(gpu_slots, hard) * math.exp(-(gpu_slots - hard) / run_s * slack)
        print(f"  ideal shared queue (M/M/c, Erlang C), slack {1e3 * slack:.0f} ms: {late:.0%} wait too long")


def erlang_c(c: int, a: float) -> float:
    """The probability that a task waits in an M/M/c queue with c servers and offered load a (a < c)."""
    head = sum(a**k / math.factorial(k) for k in range(c))
    tail = a**c / math.factorial(c) * c / (c - a)
    return tail / (head + tail)


if __name__ == "__main__":
    main()
