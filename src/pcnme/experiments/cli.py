"""Command-line entry point `pcnme` (final-design 15.6 pipeline, 16). Subcommands are filled in milestone by
milestone.

    pcnme run --scenario S2 --method greedy --seeds 3000-3004 [--seconds 60] [--cars 200] [--out results]
    pcnme report results [more run directories] [--out results/report]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Each subcommand not built yet names the PLAN.md milestone that implements it.
_PENDING: dict[str, str] = {
    "plan": "M5: NSGA-II plans and demonstrations (tools/nsga_runs.py, tools/make_demos.py for now)",
    "pretrain": "M6: DQfD pretraining (tools/pretrain.py for now)",
    "calibrate": "C: measure the time table on the GPU",
    "testbed": "M9: start a scenario on the Docker testbed",
}


def _results() -> Path:
    """$PCNME_DATA_DIR/results, default ./results (CLAUDE.md)."""
    return Path(os.environ.get("PCNME_DATA_DIR", ".")) / "results"


def _run_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    from pcnme.methods.learners import METHODS

    p = sub.add_parser("run", help="M7: train a method, then run it on evaluation seeds (fast mode)")
    p.add_argument("--scenario", required=True, help="S1-S6")
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--seeds", required=True, help="evaluation seeds, for example 3000-3019 or 3000,3002")
    p.add_argument("--config", type=Path, default=None, help="config file (default config/default.yaml)")
    p.add_argument(
        "--out", type=Path, default=None, help="results directory (default $PCNME_DATA_DIR/results)"
    )
    p.add_argument("--seconds", type=float, default=None, help="evaluation episode, simulated s")
    p.add_argument("--cars", type=int, default=None, help="fleet size")
    p.add_argument("--budget", type=int, default=None, help="training decisions (learning methods)")
    p.add_argument("--train-seconds", type=float, default=None, help="training episode, simulated s")
    p.add_argument("--train-seed", type=int, default=None, help="first training seed (2000-2999)")
    p.add_argument("--demos", type=Path, nargs="*", default=(), help="pcnme: knee demonstration files")
    p.add_argument("--windows", type=int, default=None, help="d3qn-greedy: windows for its demonstrations")
    p.add_argument("--pretrain-updates", type=int, default=None)
    p.add_argument("--population", type=int, default=None, help="knee-reference: NSGA-II population")
    p.add_argument("--generations", type=int, default=None, help="knee-reference: NSGA-II generations")
    p.add_argument("--workers", type=int, default=1, help="knee-reference: NSGA-II evaluations in parallel")


def _run(args: argparse.Namespace) -> int:
    from pcnme.config import DEFAULT_CONFIG_PATH, load_yaml
    from pcnme.experiments.runner import RunSpec, parse_seeds, run

    try:
        spec = RunSpec(
            args.scenario,
            args.method,
            parse_seeds(args.seeds),
            seconds=args.seconds,
            n_cars=args.cars,
            budget=args.budget,
            train_seconds=args.train_seconds,
            train_seed=args.train_seed,
            demos=tuple(args.demos),
            windows=args.windows,
            pretrain_updates=args.pretrain_updates,
            nsga_population=args.population,
            nsga_generations=args.generations,
            workers=args.workers,
        )
        result = run(load_yaml(args.config or DEFAULT_CONFIG_PATH), spec, args.out or _results())
    except ValueError as e:  # seeds out of range (SeedError), unknown scenario, missing demonstrations
        print(f"pcnme run: {e}", file=sys.stderr)
        return 2
    for d in result.runs:
        print(d)
    return 0


def _report_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("report", help="M7: results tables and plots from run directories (Parquet only)")
    p.add_argument("runs", type=Path, nargs="+", help="run directories, or directories of runs")
    p.add_argument("--config", type=Path, default=None, help="config file (default config/default.yaml)")
    p.add_argument("--out", type=Path, default=None, help="report directory (default <results>/report)")
    p.add_argument("--early", type=int, default=None, help="N of early-stage performance (fast_mode)")
    p.add_argument("--no-plots", action="store_true")


def _report(args: argparse.Namespace) -> int:
    from pcnme.config import DEFAULT_CONFIG_PATH, load_config
    from pcnme.metrics.report import report

    early = args.early or load_config(args.config or DEFAULT_CONFIG_PATH).fast_mode.early_decisions
    try:
        files = report(args.runs, args.out or _results() / "report", early, plots=not args.no_plots)
    except (ValueError, FileNotFoundError) as e:
        print(f"pcnme report: {e}", file=sys.stderr)
        return 2
    for f in files:
        print(f)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pcnme", description="PCNME experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    _run_parser(sub)
    _report_parser(sub)
    for name, help_text in _PENDING.items():
        sub.add_parser(name, help=help_text)
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "report":
        return _report(args)
    print(f"'{args.command}' is not implemented yet ({_PENDING[args.command]}).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
