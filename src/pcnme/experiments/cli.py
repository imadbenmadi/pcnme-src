"""Command-line entry point `pcnme` (final-design 15.6 pipeline). Subcommands are filled in milestone by milestone."""

from __future__ import annotations

import argparse
import sys

# Each subcommand names the PLAN.md milestone that implements it.
_SUBCOMMANDS: dict[str, str] = {
    "run": "M3: run one scenario in fast mode (scenario, method, seed)",
    "plan": "M5: NSGA-II plans and demonstrations",
    "pretrain": "M6: DQfD pretraining",
    "experiment": "M7: scenario x method x seed grid with statistics",
    "calibrate": "C: measure the time table on the GPU",
    "testbed": "M9: start a scenario on the Docker testbed",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pcnme", description="PCNME experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in _SUBCOMMANDS.items():
        sub.add_parser(name, help=help_text)
    args = parser.parse_args(argv)
    print(f"'{args.command}' is not implemented yet ({_SUBCOMMANDS[args.command]}).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
