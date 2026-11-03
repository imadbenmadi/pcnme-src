"""PCNME evaluation: every scenario x method x seed on the trained models, summary.csv, tables and
figures. See RUN_TRAINING.md.

uv run python evaluate.py --maps vfc
"""

from pcnme.pipeline.evaluate import main

if __name__ == "__main__":  # the guard keeps spawned worker processes from rerunning main
    raise SystemExit(main())
