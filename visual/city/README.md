# PCNME — interactive system visual (v1)

Predictive Cloud-Native Mobile Edge: task offloading for connected cars in an IoT–Fog–Cloud network.

## Run it
Double-click `index.html` (Chrome / Edge / Firefox). Needs internet on load (three.js from unpkg, IBM Plex fonts from Google Fonts).

## Files
- `index.html` — UI shell: flows, pause/step, MQTT topics, decision panel, trace, fog-node inspector, testbed view, legend, open design items.
- `pcnme-v1.js` — simulation engine + 10 flows + 3D city (three.js) + SVG system poster.
- `support.js` — runtime for the page component.
- `v0/` — first version, kept for comparison.

## Using it in the meeting
- Views: 3D city · System poster (Print poster button) · Testbed.
- Flows 1–10 (C1–C14, S3). Turn on **Step mode** and press **Next step** to explain each step.
- Hover a car → serving zone, T_exit, in-flight tasks. Click a server cabinet → fog-node internals.
- Q-values, signal strengths and latencies are **example values**. Items tagged **proposed** or shown as dashed **?** (open) are not final.

## Where to edit for v2
- Flows: `FL` object inside `runFlow()` in `pcnme-v1.js`.
- Zones / neighbours / roads: `ZDEF`, `NEIGH`, `RING` at the top of `pcnme-v1.js`.
- Colours: `COL` at the top of `pcnme-v1.js`.
