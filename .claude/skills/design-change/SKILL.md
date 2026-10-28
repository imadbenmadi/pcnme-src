---
name: design-change
description: Record a problem in docs/final-design.md found while coding PCNME and propose a fix for Imed's approval. Use when the spec is wrong, contradictory, incomplete or impossible to implement. Never change the design without it.
---

# Propose a design change

1. **Do not patch around it.** Leave the code for the affected rule unwritten (or behind a clearly failing test) until the change is approved.
2. **Describe it exactly.** Add a row to `docs/CHANGES.md`: next id (C2, C3, …), date, section, the problem with a concrete example (numbers, a failing scenario), the smallest change that fixes it, whether it touches the agent's inputs, actions, masks or reward, and status `Proposed`.
3. **Check side effects.** Search `docs/final-design.md` for every section that uses the same rule, symbol or number, and list them in the row.
4. **Ask Imed.** Explain it in 3–5 sentences: what breaks, the fix, what else it touches. Offer at most two options and recommend one.
5. **After approval only:** set the status to `Approved`, edit `docs/final-design.md` (every affected section), then implement. Remind Imed to apply the same change to the master doc on claude.ai and to its "Changes after the printed copy" table, so his printed copy stays valid.
