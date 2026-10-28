---
name: check
description: Run the PCNME verification suite (tests, lint, determinism, config arithmetic) and report the result. Use before reporting any item as done.
---

# Check

Run, in this order, from the repo root:

1. `pytest -q` — all tests must pass. Report failures with the test name and the first assertion message.
2. `ruff check src tests` — no errors.
3. Determinism: if the item touches the simulator, run the same short scenario twice with the same seed and compare the event-log hashes (there is a test for this from M1 on; make sure it ran).
4. `python tools/calc_defaults.py` if the config or the time table changed; compare with the values the config loader derives.

Report one line per step: pass or fail, and what failed. Never mark an item done with a failing step, and never weaken or skip a test to make it pass: if a test looks wrong, say why and ask Imed.
