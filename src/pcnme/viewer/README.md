# PCNME viewer

```bash
uv run python view.py            # http://127.0.0.1:8765, opens the browser
uv run python view.py --port 9000 --no-browser
```

Version 1 replays finished runs: every scenario, method and seed under `data/<map>/results/runs/`, and the
testbed runs under `data/<map>/testbed/results/`. Two runs on the same seed can be compared side by side. Everything
on screen comes from a run's logs (METRICS.md); nothing is simulated in the page.

It shows what exists and says what does not. In fast mode the picker lists the whole design (every scenario,
method and seed, from `/api/status`); a choice without a run is marked "not evaluated" or "not trained" and its
pane explains why (for example a scenario being rerun, QUESTIONS Q60). A trace map with traces but no results
(LuST before its training) shows a traffic preview per time of day: the crop's real roads, the zones and SUMO's
vehicles, with no tasks. A map with neither is listed "in development".

No build step and no new dependency: a standard-library HTTP server and plain ES modules with Canvas 2D.

## Another machine (a laptop): export

```bash
uv run python -m pcnme.viewer.export                  # seed 3000 of every run -> ../pcnme-history-viewer (~25 MB)
uv run python -m pcnme.viewer.export --seeds all      # every seed (~0.5 GB)
```

The folder holds the page, the replays pre-built and gzipped, and `serve.py` (`standalone.py`: standard library
only). Copy it anywhere and run `python serve.py` (Windows: `start.bat`); no repo, uv or packages needed.
Export again after new runs; replays already in the folder are kept.

## Layout

```
viewer/
  server.py              HTTP: the page (static/) and the JSON API over the source registry
  export.py              writes a portable folder (page + pre-built replays + standalone.py as serve.py)
  standalone.py          the exported folder's server: the same API from files, standard library only
  sources/
    base.py              the interface: SourceInfo, Source, ReplaySource, Placeholder, Registry
    replay.py            ParquetSource: a map's run directories -> runs() and replay(run_id)
    traffic.py           MapSource (runs + a trace map's traffic previews), status() of the design
    __init__.py          default_registry(): VFC and LuST, each with whatever runs and traces exist
  static/
    index.html, css/style.css
    js/main.js           the shell: source switcher, view by kind, clock, panel, keyboard
    js/core/             no DOM views here
      api.js             fetch helpers for the API
      sources.js         page-side adapters, one per source kind (replay now)
      model.js           the city-model contract every view reads, and ReplayModel
      clock.js           play / pause / speed / seek; views draw "the state at clock.t"
      theme.js           colour tokens from the CSS for the canvases
      format.js          names, number formats, bisection
    js/views/            DOM and canvas; they read a model, never a source
      city.js            CityView: the map, zones, links, WAN, cloud, cars, task packets
      run-pane.js        a header with the statistics strip above a CityView
      panel.js           the details panel (car, task, zone, WAN, cloud)
      controls.js        transport controls
      picker.js          scenario / method / seed / source and the comparison method
      placeholder.js     the "in development" state of a source that is not ready
```

## The data flow

```
source (Python)  ->  /api/...  ->  adapter (core/sources.js)  ->  model (core/model.js)  ->  views
```

A view only calls the model's methods (the contract at the top of `core/model.js`: positions, slots, queues,
the WAN, tasks around a time, running statistics). It does not know whether the model was built from a finished
run or is being fed live.

## Add a source

1. Python: a class with `info: SourceInfo` (a new `kind`, or `replay`) and the methods of its kind in
   `sources/<name>.py`; register it in `sources/__init__.py:default_registry()`.
   A source that is not ready yet is a `Placeholder(SourceInfo(..., status="development", notes={...}))`; the
   page shows its title, summary and steps.
2. API: a replay source needs nothing new. A live kind adds its endpoint in `server.py:api()` (or a streaming
   handler next to it).
3. Page: an adapter class in `core/sources.js`, added to `ADAPTERS` under its kind.

A replay source also answers `status()`: the design's scenarios, methods and seeds, and per scenario x method
"evaluated" (with its seeds), "trained" or "not trained", plus notes per scenario.

Planned kinds:

- `live-testbed` (PLAN V3): the server subscribes to the testbed's MQTT broker and streams the same records the
  logs hold (positions, zone rows, task and attempt rows) as server-sent events or over a WebSocket at
  `/api/stream?source=...`. The adapter appends them to a `LiveModel` that implements the model contract with
  growing arrays; the clock follows the newest time (seeking back replays the buffer). CityView, the panel and the
  statistics need no change.
- `live-training`: the training loop (pipeline/train.py) already reports per episode; the server streams those
  reports (episode, on-time share, mean ê, loss, updates) and sampled decisions. It gets its own model and views
  (training curves, a decision table), chosen by kind in `main.js`.

## Add a view

A view is a class in `js/views/` that takes a container and a model and draws `(t, selection)`. Read the model's
methods only; use `core/theme.js` colours and `core/format.js` formats. Wire it in `main.js` (or inside a pane like
`run-pane.js`). If it needs data the model does not offer, add a method to the contract in `core/model.js` and to
every model that implements it, and the field to `sources/replay.py`.

## Design rules

One accent colour (`--accent`); green, amber and red only for task outcomes. Every motion shows data: cars move
with their logged positions, packets travel during their logged upload, forward and result times, coverage fill
follows the zone's load, queue bars and the WAN pipe follow their logged values. Light and dark themes follow the
system.

Keys: Space play/pause, arrow keys step 1 s (Shift: 5 s), `[` `]` speed, H hide the bar, F full screen, Esc
close the panel. The URL keeps the selection (`#vfc/S3/pcnme/3000/fast/greedy`), so a view can be shared as a link.
