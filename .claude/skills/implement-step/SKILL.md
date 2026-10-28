---
name: implement-step
description: Implement exactly one PLAN.md item of PCNME from docs/final-design.md, tests first. Use whenever asked to build, code, or continue a milestone item.
---

# Implement one PLAN item

Follow these steps in order. Do not skip any.

1. **Pick the item.** Open `docs/PLAN.md`. Work on the item Imed named, or else the first `[ ]` item of the current milestone. Mark it `[~]`. One item only.
2. **Read the spec in full.** Read every section of `docs/final-design.md` that the item cites, completely, plus any section those sections point to. Read the matching rows of section 18 and `config/default.yaml`.
3. **Write the rules list.** Before any code, write a numbered list of every rule, formula and number the item must implement, each with its section (for example "R3 — skip a hard head when t + Ê_low > d_k (8.4 step 1)"). Show it to Imed in your reply.
4. **Stop on any gap.** If a rule is unclear, contradicts another section, or cannot be built as written: do not guess. Add a row to `docs/QUESTIONS.md` (or run the `design-change` skill if the spec is wrong), tell Imed, and stop work on the parts that depend on it.
5. **Tests first.** For each rule, write a test in `tests/` that would fail without it. Include the acceptance tests listed for the milestone when the item completes it. Use fixed seeds and hand-computed expected values.
6. **Implement.** Smallest code that passes the tests. Cite the section in a short comment where a rule is implemented. Read every number from the config. Typed functions, no globals, no unseeded randomness, no wall-clock time.
7. **Run the `check` skill.** All tests, lint, determinism.
8. **Report.** In your reply: the rules list with ✓ per rule, the tests added, what is not done, any question raised, and whether anything touches the agent's inputs, actions, masks or reward. Mark the item `[x]` only after Imed approves; until then leave it `[~]`.
