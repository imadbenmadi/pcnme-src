"""PCNME Docker testbed (PLAN M9). See RUN_TRAINING.md.

uv run python run_testbed.py --maps vfc
"""

from pcnme.pipeline.testbed import main

if __name__ == "__main__":  # the guard keeps spawned worker processes from rerunning main
    raise SystemExit(main())
