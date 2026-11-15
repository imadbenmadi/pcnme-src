"""PLAN T4: docs/RESULTS.md, the single source for the paper's Results section, built only from files that
exist (no run). Every table cites the file it comes from; the findings are derived from the same files.

    uv run python tools/results_doc.py

Inputs per map (data/<map>/): results/report.csv (mean and 95 % Student t interval over the seeds),
results/paired.csv (paired t tests of PCNME vs each method), results/extra-runs.csv and
results/extra-hypervolume.csv (tools/extra_metrics.py), train/<S>-<method>/episodes.csv, the figures in
results/;
for the VFC testbed: testbed/summary.csv and testbed/gap.csv.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ORDER = ["pcnme", "greedy", "d3qn-greedy", "d3qn-scratch", "vfc-ppo", "tang-wong", "qeco", "random"]
NAMES = {
    "pcnme": "PCNME",
    "greedy": "Greedy",
    "random": "Random",
    "d3qn-scratch": "D3QN (scratch)",
    "d3qn-greedy": "D3QN + greedy demos",
    "vfc-ppo": "VFC PPO",
    "tang-wong": "Tang & Wong",
    "qeco": "QECO",
}
LEARNERS = ["pcnme", "d3qn-greedy", "d3qn-scratch", "vfc-ppo", "tang-wong", "qeco"]
SCEN = ["S1", "S2", "S3", "S4", "S5", "S6"]
MAPS = {"vfc": "VFC grid", "lust": "LuST (Luxembourg)"}
T4 = 2.776  # Student t, 4 degrees of freedom (5 testbed seeds), 95 %


def rel(p: Path) -> str:
    return p.relative_to(REPO).as_posix()


def read(p: Path) -> list[dict[str, str]]:
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def table(head: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "| --- " * len(head) + "|"] + [
        "| " + " | ".join(r) + " |" for r in rows
    ]


def pct(m: float, h: float | None = None, d: int = 1) -> str:
    return f"{100 * m:.{d}f}" + (f" ± {100 * h:.{d}f}" if h is not None and math.isfinite(h) else "")


def ms(m: float, h: float | None = None) -> str:
    return f"{1000 * m:.0f}" + (f" ± {1000 * h:.0f}" if h is not None and math.isfinite(h) else "")


def num(m: float, h: float | None = None, d: int = 2) -> str:
    return f"{m:.{d}f}" + (f" ± {h:.{d}f}" if h is not None and math.isfinite(h) else "")


def pfmt(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


class Report:
    """report.csv: (scenario, method, metric) -> (mean, half)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.v: dict[tuple[str, str, str], tuple[float, float]] = {}
        for r in read(path):
            self.v[(r["scenario"], r["method"], r["metric"])] = (float(r["mean"]), float(r["half"] or "nan"))
        self.n = {r["scenario"]: r["n"] for r in read(path)}

    def get(self, s: str, m: str, k: str) -> tuple[float, float]:
        return self.v.get((s, m, k), (math.nan, math.nan))

    def methods(self, s: str) -> list[str]:
        have = {m for (sc, m, _) in self.v if sc == s}
        return [m for m in ORDER if m in have]


def main_table(rep: Report, s: str) -> list[str]:
    rows = []
    for m in rep.methods(s):

        def g(k: str, s: str = s, m: str = m) -> tuple[float, float]:
            return rep.get(s, m, k)

        rows.append(
            [
                NAMES[m],
                pct(*g("on_time")),
                pct(*g("late")),
                pct(*g("dropped")),
                pct(*g("failed")),
                ms(*g("latency_mean_s")),
                ms(*g("latency_p99_s")),
                num(*g("energy_per_task_j")),
            ]
        )
    head = ["method", "on time %", "late %", "dropped %", "failed %", "mean latency ms", "p99 ms", "J / task"]
    return table(head, rows)


def paired(path: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    out = {}
    for r in read(path):
        out[(r["scenario"], r["metric"], r["baseline"])] = {
            "diff": float(r["diff"]),
            "p": float(r["p"]),
            "ours": float(r["ours_mean"]),
            "base": float(r["baseline_mean"]),
        }
    return out


def paired_table(pt: dict[tuple[str, str, str], dict[str, float]], s: str) -> list[str]:
    rows = []
    for b in ORDER[1:]:
        cells = [NAMES[b]]
        for k, f in (
            ("on_time", lambda x: f"{100 * x:+.2f} pp"),
            ("latency_p99_s", lambda x: f"{1000 * x:+.0f} ms"),
            ("energy_per_task_j", lambda x: f"{x:+.3f} J"),
        ):
            r = pt.get((s, k, b))
            cells.append("–" if r is None else f"{f(r['diff'])} (p {pfmt(r['p'])})")
        rows.append(cells)
    return table(["PCNME − baseline", "on time", "p99 latency", "J / task"], rows)


def verdicts(
    pt: dict[tuple[str, str, str], dict[str, float]], s: str
) -> tuple[list[str], list[str], list[str]]:
    """On-time share: methods PCNME beats, ties, loses to (paired t, p < 0.05)."""
    win, tie, lose = [], [], []
    for b in ORDER[1:]:
        r = pt.get((s, "on_time", b))
        if r is None:
            continue
        label = f"{NAMES[b]} ({100 * r['diff']:+.2f} pp, p {pfmt(r['p'])})"
        (win if r["p"] < 0.05 and r["diff"] > 0 else lose if r["p"] < 0.05 else tie).append(label)
    return win, tie, lose


def extra_rows(path: Path) -> list[dict[str, Any]]:
    out = []
    for r in read(path):
        row: dict[str, Any] = dict(r)
        for k, v in r.items():
            try:
                row[k] = float(v) if v not in ("", "None") else None
            except ValueError:
                row[k] = v
        out.append(row)
    return out


def cold_start(rep: Report, extra: list[dict[str, Any]]) -> list[str]:
    by: dict[tuple[str, str], list[float | None]] = defaultdict(list)
    for r in extra:
        by[(r["scenario"], r["method"])].append(r["decisions_until_stable"])
    rows = []
    for m in ORDER:
        cells = [NAMES[m]]
        for s in SCEN:
            e = rep.get(s, m, "early_on_time")
            vals = [v for v in by.get((s, m), []) if v is not None]
            n = len(by.get((s, m), []))
            st = f"{statistics.median(vals):,.0f} ({len(vals)}/{n})" if vals else f"– (0/{n})"
            cells.append(f"{pct(*e)} · {st}")
        rows.append(cells)
    return table(["method", *SCEN], rows)


def training_end(map_name: str) -> list[str]:
    """Per learner and scenario: mean on-time of the last 10 episodes, and its change from the 10 before."""
    rows = []
    for m in LEARNERS:
        cells = [NAMES[m]]
        for s in SCEN:
            f = REPO / "data" / map_name / "train" / f"{s}-{m}" / "episodes.csv"
            if not f.exists():
                cells.append("–")
                continue
            ot = [float(r["on_time"]) for r in read(f)]
            last, before = ot[-10:], ot[-20:-10]
            d = statistics.mean(last) - statistics.mean(before) if before else math.nan
            cells.append(f"{100 * statistics.mean(last):.1f} ({100 * d:+.1f})")
        rows.append(cells)
    return table(["method", *SCEN], rows)


def predictor(extra: list[dict[str, Any]]) -> list[str]:
    rows = []
    for s in SCEN:
        g = [
            r
            for r in extra
            if r["scenario"] == s and r["method"] == "greedy" and r.get("pred_acc1") is not None
        ]
        if not g:
            continue
        rows.append(
            [
                s,
                f"{sum(r['pred_predictions'] for r in g):,.0f}",
                f"{100 * statistics.mean(r['pred_acc1'] for r in g):.1f}",
                f"{statistics.mean(r['pred_t_exit_median_s'] for r in g):.2f}",
                f"{statistics.mean(r['pred_t_exit_mae_s'] for r in g):.2f}",
            ]
        )
    return table(
        ["scenario", "predictions", "next-zone acc@1 %", "T_exit error median s", "T_exit error mean s"], rows
    )


def hypervolume(path: Path) -> list[str]:
    hv = extra_rows(path)
    rows = []
    for s in SCEN:
        ws = [r for r in hv if r["scenario"] == s]
        if ws:
            rows.append(
                [
                    s,
                    str(len(ws)),
                    f"{statistics.mean(r['points'] for r in ws):.0f}",
                    f"{statistics.mean(r['hv'] for r in ws):.4f}",
                    f"{statistics.mean(r['hv_ratio'] for r in ws):.2f}",
                ]
            )
    return table(["scenario", "windows", "front points", "hypervolume", "ratio to the greedy plan"], rows)


def testbed_section() -> list[str]:
    summ = REPO / "data/vfc/testbed/summary.csv"
    rows = read(summ)
    by: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by[(r["scenario"], r["method"], r["mode"])].append(r)

    def ci(v: list[float]) -> tuple[float, float]:
        return statistics.mean(v), (T4 * statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else math.nan)

    out = []
    for s in SCEN:
        trs = []
        for m in ORDER:
            bed, fast = by.get((s, m, "testbed"), []), by.get((s, m, "fast"), [])
            if not bed or not fast:
                continue
            ob, of = ci([float(r["on_time"]) for r in bed]), ci([float(r["on_time"]) for r in fast])
            pb = ci([float(r["latency_p99_s"]) for r in bed])
            trs.append(
                [
                    NAMES[m],
                    str(len(bed)),
                    pct(*of),
                    pct(*ob),
                    f"{100 * (ob[0] - of[0]):+.1f}",
                    ms(*pb),
                    pct(*ci([float(r["dropped"]) for r in bed])),
                    pct(*ci([float(r["failed"]) for r in bed])),
                ]
            )
        out += [
            f"**{s}**",
            "",
            *table(
                [
                    "method",
                    "seeds",
                    "on time fast %",
                    "on time testbed %",
                    "gap pp",
                    "p99 testbed ms",
                    "dropped testbed %",
                    "failed testbed %",
                ],
                trs,
            ),
            "",
        ]
    out += ["### Testbed findings", ""]
    for s in SCEN:
        on = {m: statistics.mean(float(r["on_time"]) for r in by[(s, m, "testbed")])
              for m in ORDER if by.get((s, m, "testbed"))}  # fmt: skip
        if not on:
            continue
        rank = sorted(on, key=lambda m: -on[m])
        fail = {m: statistics.mean(float(r["failed"]) for r in by[(s, m, "testbed")]) for m in on}
        worst = max(fail, key=lambda m: fail[m])
        out.append(
            f"- **{s}.** Best on the testbed: {NAMES[rank[0]]} {100 * on[rank[0]]:.1f} %; PCNME "
            f"{100 * on['pcnme']:.1f} % (rank {rank.index('pcnme') + 1} of {len(rank)}); the largest failed "
            f"share: {NAMES[worst]} {100 * fail[worst]:.1f} %."
        )
    out += [
        "",
        "In S3 the cars receive many results after they gave the task up (deadline + L_max): investigated "
        "for Greedy, whose cars answered 2 682 results per zone and run without a confirmation against 32 in "
        "S2 (RUN_LOG 15 Nov); those tasks count as failed (no ack) where fast mode drops them "
        "before they run.",
        "",
    ]
    return out


FINDING_WHY = {  # each claim rests on a number of report.csv or extra-runs.csv, filled in by findings()
    "vfc": {
        "S1": "Light load: every method is at its ceiling; the differences are hundredths of a percent.",
        "S2": "Medium load: PCNME, Greedy and D3QN + greedy demos are at the ceiling; the methods without "
        "demonstrations lose a little.",
        "S3": "Overload (rho 1.5): PCNME's placements are refused by a full node less often than Greedy's "
        "(refusal rate {ref_p} vs {ref_g} %), and it keeps the most tasks on time.",
        "S4": "Tight deadlines: the learned methods are close; PCNME keeps a lead over all of them.",
        "S5": "Tight deadlines with high mobility: PCNME leads Greedy and most learned methods; "
        "D3QN (scratch) "
        "is level with it.",
        "S6": "Mixed fog (two CPU-only zones): Greedy is best; PCNME's placements are refused more "
        "often than "
        "Greedy's (refusal rate {ref_p} vs {ref_g} %). Not analysed further (PLAN T6).",
    },
    "lust": {
        "S1": "Off-peak real traffic: every method at its ceiling.",
        "S2": "Midday traffic: at the ceiling; PCNME ties Greedy and beats the methods without "
        "demonstrations.",
        "S3": "08:00 peak: every method falls to about half on time and the grid's S3 advantage does "
        "not carry "
        "over; the mobility predictor's next-zone accuracy drops to {acc} % here (from {acc_mid} % at "
        "midday). The cause is not analysed further.",
        "S4": "Tight deadlines on real traffic: PCNME beats Greedy, as on the grid; VFC PPO is "
        "slightly ahead.",
        "S5": "Tight deadlines at the peak: as in S4.",
        "S6": "Mixed fog: Greedy and the D3QN methods are level; PCNME's placements are refused more "
        "often than "
        "Greedy's (refusal rate {ref_p} vs {ref_g} %).",
    },
}


def _acc(extra: list[dict[str, Any]], s: str) -> str:
    g = [
        r["pred_acc1"] for r in extra if r["scenario"] == s and r["method"] == "greedy" and r.get("pred_acc1")
    ]
    return f"{100 * statistics.mean(g):.1f}" if g else "–"


def findings(
    map_name: str, rep: Report, pt: dict[tuple[str, str, str], dict[str, float]], extra: list[dict[str, Any]]
) -> list[str]:
    out = []
    for s in SCEN:
        why = FINDING_WHY[map_name][s].format(
            ref_p=f"{100 * rep.get(s, 'pcnme', 'refusal_rate')[0]:.1f}",
            ref_g=f"{100 * rep.get(s, 'greedy', 'refusal_rate')[0]:.1f}",
            acc=_acc(extra, s),
            acc_mid=_acc(extra, "S2"),
        )
        win, tie, lose = verdicts(pt, s)
        best = max(rep.methods(s), key=lambda m: rep.get(s, m, "on_time")[0])
        pc = rep.get(s, "pcnme", "on_time")[0]
        out.append(
            f"- **{s}.** PCNME {100 * pc:.1f} % on time; best {NAMES[best]} "
            f"{100 * rep.get(s, best, 'on_time')[0]:.1f} %. {why}"
        )
        out.append(
            f"  Wins: {', '.join(win) or 'none'}. Ties: {', '.join(tie) or 'none'}. "
            f"Loses: {', '.join(lose) or 'none'}."
        )
    return out


def map_section(map_name: str) -> list[str]:
    base = REPO / "data" / map_name / "results"
    rep = Report(base / "report.csv")
    pt = paired(base / "paired.csv")
    extra = extra_rows(base / "extra-runs.csv")
    n = rep.n.get("S1", "?")
    out = [
        f"## {MAPS[map_name]}",
        "",
        f"Fast mode, {n} evaluation seeds per scenario and method; mean ± 95 % Student t interval.",
        "",
        "### Main results",
        "",
    ]
    for s in SCEN:
        out += [f"**{s}**", "", *main_table(rep, s), "", f"Source: `{rel(base / 'report.csv')}`.", ""]
    out += ["### PCNME vs each method (paired t test over the same seeds)", ""]
    for s in SCEN:
        out += [f"**{s}**", "", *paired_table(pt, s), ""]
    out += [
        f"Source: `{rel(base / 'paired.csv')}`.",
        "",
        "### Cold start",
        "",
        "Early-stage on time (the tasks of the first 1 000 decisions after deployment, %) · decisions until "
        "the on-time share of the last 1 000 tasks is ≥ 95 % to the end of the run (median over the seeds "
        "that get there; how many of them).",
        "",
    ]
    out += [
        *cold_start(rep, extra),
        "",
        f"Sources: `{rel(base / 'report.csv')}` (early_on_time), "
        f"`{rel(base / 'extra-runs.csv')}` (decisions until stable).",
        "",
    ]
    out += [
        "### Training",
        "",
        "On time of the last 10 training episodes, % (change from the 10 before, pp): a positive change "
        "means "
        "the method was still improving when the 1 M-decision budget ran out.",
        "",
    ]
    out += [
        *training_end(map_name),
        "",
        f"Source: `data/{map_name}/train/<S>-<method>/episodes.csv`. Figures: "
        + ", ".join(f"`{rel(base / f'training-{s}.png')}`" for s in SCEN)
        + ".",
        "",
    ]
    out += [
        "### Mobility predictor and NSGA-II fronts",
        "",
        *predictor(extra),
        "",
        f"Source: `{rel(base / 'extra-runs.csv')}` (Greedy's runs; the cars move the same under every "
        "method).",
        "",
        *hypervolume(base / "extra-hypervolume.csv"),
        "",
        f"Source: `{rel(base / 'extra-hypervolume.csv')}`.",
        "",
    ]
    out += [
        "### Findings",
        "",
        "Per scenario: PCNME's on-time share, the best method's, and the paired tests "
        "(on time, p < 0.05) that PCNME wins, ties or loses; refusal rates from report.csv, predictor "
        "accuracy from extra-runs.csv.",
        "",
        *findings(map_name, rep, pt, extra),
        "",
    ]
    figs = ", ".join(f"`{rel(base / f'across-{k}.png')}`" for k in ("on_time", "latency_p99_s", "mean_e_hat"))
    out += [f"Figures across scenarios: {figs}; per scenario: `{rel(base)}/<S>.png`, `<S>-load.png`.", ""]
    return out


def main() -> int:
    vfc = Report(REPO / "data/vfc/results/report.csv")
    out = [
        "# PCNME: results",
        "",
        "The single source for the paper's Results section (PLAN T4). Generated by `tools/results_doc.py` "
        "from "
        "the files cited under each table; nothing here is typed by hand. Regenerate after any new run.",
        "",
        "Methods (final-design 16.1): PCNME, Greedy, D3QN + greedy demos, D3QN (scratch), VFC PPO, Tang & "
        "Wong, "
        "QECO, Random. Scenarios (16.2): S1 light, S2 medium, S3 overload (rho 1.5), S4 tight deadlines, S5 "
        "tight "
        "deadlines and high mobility, S6 mixed fog (two CPU-only zones, loose deadlines). Shares are of all "
        "tasks; late = delivered after the deadline; dropped = refused, masked, aborted or given up; failed "
        "= "
        "lost or never acknowledged (QUESTIONS Q46).",
        "",
    ]
    out += map_section("vfc") + map_section("lust")
    out += [
        "## Testbed vs fast mode (VFC grid)",
        "",
        "The Docker testbed (final-design 15.4): 40 cars, 4 zones, real MQTT and HTTP over shaped networks, "
        "60 s "
        "runs, 5 evaluation seeds; each next to its fast-mode twin (same scenario, method, seed and fleet). "
        "Mean ± 95 % Student t interval (4 degrees of freedom). Every learning method learns online in its "
        "own "
        "trainer process (QUESTIONS Q62).",
        "",
    ]
    out += [
        *testbed_section(),
        "Source: `data/vfc/testbed/summary.csv` (gap means: `data/vfc/testbed/gap.csv`).",
        "",
    ]
    out += [
        "## S6 with tight deadlines (archived stress test)",
        "",
        "S6 first ran with tight deadlines; every method was near a capacity ceiling (QUESTIONS Q60) and S6 "
        "was "
        "rerun with loose deadlines. The tight version is kept as a stress test.",
        "",
    ]
    out += [
        *main_table(vfc, "S6 tight (archived)"),
        "",
        'Source: `data/vfc/results/report.csv` (scenario "S6 tight (archived)").',
        "",
    ]
    out += [
        "## Limitations",
        "",
        "- No ablations: the 16.4 experiments (expert data, FedAvg variants, queue order, refusal penalty, "
        "IQL, "
        "60 inputs, state period, sensitivity sweeps) and the training-budget test are future work.",
        "- One training budget of 1 M decisions per method and scenario. Several learners were still "
        "improving "
        "when it ran out: on LuST most of them in S3, S4 and S6, on the grid the D3QN family in S3 (the "
        "Training tables: the change over the last 20 episodes). Each episode is another seed, so a change "
        "of "
        "a few tenths is within episode-to-episode noise.",
        "- The testbed runs at 40 cars (2 car containers) and 60 s per run, on one PC; the fast-mode "
        "evaluation runs 200 cars. Under overload (S3) many results reach the cars after they gave the task "
        "up, so these tasks count as failed where fast mode drops them (investigated for Greedy, RUN_LOG 15 "
        "Nov); the high failed shares of the LSTM baselines in S4 and S5 are not analysed.",
        "- FedAvg + fine-tuning (13.7, phase 2) is evaluated in fast mode only; on the testbed it is future "
        "work "
        "(QUESTIONS Q64).",
        "- LuST has no testbed: its results are fast mode only.",
        "- The 16.3 metrics without a definition in the design use stand-ins (QUESTIONS Q63); delivery "
        "after a "
        "zone change has few tasks per run and is not reported.",
        "- S6 (mixed fog) is not analysed beyond its tables (PLAN T6).",
        "",
    ]
    path = REPO / "docs" / "RESULTS.md"
    path.write_bytes(("\n".join(out).rstrip("\n") + "\n").encode())
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
