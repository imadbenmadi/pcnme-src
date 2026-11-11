"""A live testbed run to watch in Grafana and the viewer (PLAN R2b, V3). See RUN_TRAINING.md.

uv run python run_live.py --scenario S2 --method pcnme --minutes 10
"""

from pcnme.pipeline.live import main

if __name__ == "__main__":
    raise SystemExit(main())
