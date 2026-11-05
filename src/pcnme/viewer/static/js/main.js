// The viewer's shell: sources in the switcher, the view for the chosen source's kind, the clock, the panel,
// the keyboard. Adding a source kind: an adapter in core/sources.js and a view; this file picks by kind.

import { $, esc } from "./core/format.js";
import { Clock, SPEEDS } from "./core/clock.js";
import { api } from "./core/api.js";
import { adapterFor } from "./core/sources.js";
import { onTheme, readTheme } from "./core/theme.js";
import { TransportControls } from "./views/controls.js";
import { DetailsPanel } from "./views/panel.js";
import { RunPicker } from "./views/picker.js";
import { renderPlaceholder } from "./views/placeholder.js";
import { RunPane } from "./views/run-pane.js";

const clock = new Clock();
const controls = new TransportControls(clock);
const state = { sources: [], source: null, adapter: null, panes: [], sel: null, lastUi: 0, lastPanel: 0, loadToken: 0 };

const panel = new DetailsPanel($("panel"), {
  onClose: () => select(null),
  onSelectTask: (i) => select({ kind: "task", id: i }, state.sel && state.sel.pane),
});
const picker = new RunPicker((sel) => showRuns(sel));

function select(hit, pane) {
  state.sel = hit ? { ...hit, pane: pane || state.panes[0] } : null;
  panel.render(state.sel, clock.t, state.panes.length > 1);
  draw(true);
}

function draw(force = false) {
  for (const p of state.panes) p.draw(clock.t, state.sel);
  const now = performance.now();
  if (force || now - state.lastUi > 100) {
    state.lastUi = now;
    controls.frame(clock.t);
    state.panes.forEach((p) => p.updateStats(clock.t));
  }
  if (state.sel && (force || now - state.lastPanel > 300)) {
    state.lastPanel = now;
    panel.render(state.sel, clock.t, state.panes.length > 1);
  }
}
clock.onFrame((_, force) => draw(force));

// ---- sources ------------------------------------------------------------------------------------------------
async function setSource(id, hash = {}) {
  const info = state.sources.find((s) => s.id === id) || state.sources[0];
  state.source = info;
  $("sources").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === info.id)));
  state.adapter = adapterFor(info);
  const ready = !!state.adapter;
  document.body.classList.toggle("dev-mode", !ready);
  $("dev").hidden = ready;
  $("pickers").style.visibility = ready ? "visible" : "hidden";
  if (!ready) {
    clock.pause(); select(null); clearPanes();
    renderPlaceholder($("dev"), info);
    history.replaceState(null, "", `#${info.id}`);
    return;
  }
  if (info.kind === "replay") {
    picker.setRuns(await state.adapter.listRuns(), hash);
    await showRuns(picker.selection());
  }
}

function clearPanes() { state.panes.forEach((p) => p.destroy()); state.panes = []; }

async function showRuns(sel) {
  const token = ++state.loadToken;
  const ids = picker.runIds(sel);
  history.replaceState(null, "", `#${[state.source.id, sel.scenario, sel.method, sel.seed, sel.mode, sel.method2 || ""].join("/").replace(/\/$/, "")}`);
  $("progress").classList.add("on");
  try {
    const models = await Promise.all(ids.map((id) => state.adapter.open(id)));
    if (token !== state.loadToken) return; // a newer choice won
    state.sel = null; panel.render(null);
    clearPanes();
    state.panes = models.map((m) => new RunPane($("stage"), m, { onSelect: select, speed: () => clock.speed }));
    controls.setArrivals(models[0].arrivals);
    clock.setRange(Math.max(...models.map((m) => m.end)));
  } catch (e) {
    $("stage").innerHTML = `<div class="empty">Could not load ${esc(ids.join(", "))}: ${esc(e.message)}</div>`;
  } finally {
    if (token === state.loadToken) $("progress").classList.remove("on");
  }
}

// ---- shell: bar, full screen, keyboard ------------------------------------------------------------------
function toggleBar() { document.body.classList.toggle("bar-hidden"); }
function fullscreen() { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(); }
$("hide-bar").onclick = $("show-bar").onclick = toggleBar;
$("fullscreen").onclick = fullscreen;
let idle;
addEventListener("mousemove", () => {
  document.body.classList.remove("idle"); clearTimeout(idle);
  idle = setTimeout(() => document.body.classList.add("idle"), 1800);
});
addEventListener("keydown", (e) => {
  const tag = e.target.tagName;
  if (tag === "SELECT" || (tag === "INPUT" && e.target.type !== "range" && e.target.type !== "checkbox")) return;
  const k = e.key;
  if (k === " ") { e.preventDefault(); clock.toggle(); }
  else if (k === "ArrowRight" || k === "ArrowLeft") { e.preventDefault(); clock.seek(clock.t + (k === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 5 : 1)); }
  else if (k === "f" || k === "F") fullscreen();
  else if (k === "h" || k === "H") toggleBar();
  else if (k === "Escape") select(null);
  else if (k === "]" || k === "[") { const i = SPEEDS.indexOf(clock.speed) + (k === "]" ? 1 : -1); if (SPEEDS[i]) clock.setSpeed(SPEEDS[i]); }
});

function parseHash() {
  const [id, scenario = "S3", method = "pcnme", seed = "3000", mode = "fast", method2] = location.hash.slice(1).split("/");
  return { id: id || "vfc", want: { scenario, method, seed, mode, method2 } };
}
addEventListener("hashchange", () => { const h = parseHash(); setSource(h.id, h.want); }); // a pasted or typed link

async function boot() {
  readTheme();
  onTheme(() => { state.panes.forEach((p) => p.city.resize()); });
  state.sources = await api.sources();
  $("sources").innerHTML = state.sources.map((s) => `<button data-id="${esc(s.id)}">${esc(s.label)}${s.status === "development" ? '<span class="tag">in development</span>' : ""}</button>`).join("");
  $("sources").querySelectorAll("button").forEach((b) => (b.onclick = () => setSource(b.dataset.id)));
  const h = parseHash();
  await setSource(h.id, h.want);
}
boot();
