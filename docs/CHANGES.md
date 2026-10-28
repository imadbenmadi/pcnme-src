# Design changes found while coding

Every place where coding showed that `docs/final-design.md` must change. Nothing here is applied to the code until Imed approves it. After approval: update `docs/final-design.md`, then tell Imed so the master doc on claude.ai (and its "Changes after the printed copy" table) gets the same change.

| Id | Date | Section | Problem | Proposed change | Touches the agent? | Status |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | 2026-10-28 | 18 (Radio) | The VFC code confirms the carrier is 5.9 GHz. It also computes the noise as 40 MHz × 10^(−17.4) with the signal in watts: that is −174 dBm/Hz read as W/Hz, so its noise is 1 000× (30 dB) too high and its rates are lower (about 150–350 Mbps) than the physics gives. | PCNME uses correct units: noise = 10^(−20.4) W/Hz × B, giving 540–930 Mbps at 10–300 m (as section 18 says). Mention the VFC slip in the paper's environment section. | No | Approved (decided 28 Oct) |
