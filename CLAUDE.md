# PCNME — instructions for Claude Code

PCNME places the computing tasks of moving cars across fog zones and the cloud. A fixed rule (TOF-Broker) sends tasks too heavy for any fog node to the cloud. A D3QN agent in every zone places the rest. The agent is pretrained by DQfD on NSGA-II knee-point plans, then learns online. This repo is the MSc thesis implementation (Imed-eddin Benmadi, ELTE).

## Sources of truth (read in this order, nothing else)

1. `docs/final-design.md` — the complete design. It is the specification. Every behaviour, formula, message and rule in the code must come from it, and the code cites the section it implements (for example `# final-design 8.4 step 2`).
2. `docs/PLAN.md` — the build order. Work on exactly one milestone item at a time.
3. `config/default.yaml` — every number. Never hard-code a number that is in the config.
4. `docs/CHANGES.md` and `docs/QUESTIONS.md` — open design issues found while coding.

Not specifications: `visual/`, `excalidraw/` (old illustrations with example numbers), `reference/vfc/` (the VFC paper's code, read-only reference). If any of these disagree with `docs/final-design.md`, the final design wins.

## Hard rules

- **Never change the design in code.** If the spec is unclear, contradictory, or impossible to implement as written: stop, write the issue to `docs/QUESTIONS.md` (or use the design-change skill), and ask Imed. Never invent behaviour to fill a gap, and never "simplify" a rule silently.
- **Tests first.** For every rule you implement, write the test that checks it, then the code. A milestone item is done only when its acceptance tests in `docs/PLAN.md` pass.
- **Say when something touches the agent.** Any change that alters the agent's 48 inputs, its 6 actions, the masks or the reward must be called out explicitly to Imed.
- **Determinism.** Every random draw comes from the seeded stream for its purpose (arrivals, run times, link delays, mobility, …). The same config and seed must give a bit-identical event log. No `random` module globals, no wall-clock time in fast mode, no unseeded UUIDs (ids are built as in final-design 5.5).
- **No clock comparisons across machines** (final-design 5.4): messages carry budgets, durations and versions, never another machine's timestamp in a decision.
- **Do not copy code from `reference/vfc/`.** It has no license. Reimplement from the paper and the final design; reading it to check a formula or constant is fine.
- **Small steps.** One milestone item per branch / PR. Keep functions small and typed. No new dependencies without saying why.
- **Units.** Internally: seconds, bits, bits per second, watts, joules, metres. Convert at the edges (config loader, logs).

## Workflow for each item

Use the `implement-step` skill. In short: read the PLAN item → read every final-design section it cites, in full → list the rules you will implement → write the tests → implement → run `check` → report what was done, what was not, and any question.

## Environment

- Python 3.12 (not 3.14). Dev machine: Windows 10, repo on drive F:. Also used from Claude Code cloud sessions (Linux, no GPU, no Docker).
- Install: `pip install -e ".[dev]"`. Later milestones add `.[ml]` (PyTorch, pymoo).
- Run tests: `pytest -q`. Lint: `ruff check src tests`.
- Check the numbers fit together: `python tools/calc_defaults.py`.
- Fast mode must run on Windows and Linux alike: use `pathlib`, no shell-specific code; multiprocessing must work with the `spawn` start method.

## Layout

```
src/pcnme/
  config.py        config loader + derived values (B_WAN, λ_car, E_ref, ē)       final-design 18
  ids.py           task ids, attempt keys, snapshot versions                     5.5
  sim/             discrete-event clock, seeded random streams                   15.2
  net/             FIFO pipe, radio (Shannon), backhaul, WAN in series            4
  node/            time table, slots, queues, scheduler, wait estimates          8, 14.4
  zone/            intake, TOF-Broker, masks, anchor table, state store          6, 7, 9
  car/             car app: tasks, uploads, GPS, duplicate filter                6, 9.5, 10
  mobility/        grid / LuST traces, attach + hysteresis, predictor            10
  bus/             fast-mode in-process bus and calls (testbed later: MQTT/HTTP) 5, 15.2
  agent/           48 inputs, reward, D3QN, DQfD, replay                        11, 13.5
  expert/          NSGA-II plans, knee, demonstrations                           13.2–13.4
  methods/         greedy, random, baselines                                     16.1
  energy.py        energy accounting                                             12
  metrics/         logs, metrics, statistics                                     16.3
tests/             mirrors src/ ; one test file per module
tools/             calc_defaults.py and small scripts
```
