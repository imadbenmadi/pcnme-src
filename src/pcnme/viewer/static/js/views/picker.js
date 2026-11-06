// RunPicker: scenario, method, seed and source (fast mode, testbed, traffic) of a replay source. In fast mode it
// lists the whole design (every scenario and method, from /api/status), so a missing run is visible as missing:
// "not evaluated" or "not trained", never silently absent. Calls onChange(selection).

import { $, METHOD_ORDER, esc, methodName } from "../core/format.js";

const MODES = { fast: "Fast mode", testbed: "Testbed", traces: "Traffic (SUMO)" };
const TIMES = { off_peak: "Off-peak", mid: "Midday", peak: "Peak" };
const SUFFIX = { trained: " · not evaluated", "not trained": " · not trained" };

export class RunPicker {
  constructor(onChange) {
    this.onChange = onChange;
    for (const id of ["scenario", "method", "seed", "mode"]) $(id).onchange = () => { this.fill({}); this.emit(); };
    $("method2").onchange = () => this.emit();
    $("compare").onchange = () => { $("method2").disabled = !$("compare").checked; this.emit(); };
  }
  setRuns(runs, status, want = {}) {
    this.runs = runs; this.status = status;
    this.ids = new Set(runs.map((r) => r.run_id));
    this.fill(want);
  }
  cell(scenario, method) { return (this.status && this.status.cells[`${scenario}/${method}`]) || null; }
  fill(want) {
    const set = (id, values, label, pick) => {
      const el = $(id), cur = pick ?? el.value;
      el.innerHTML = values.map((v) => `<option value="${esc(v)}">${esc(label(v))}</option>`).join("");
      el.value = values.map(String).includes(String(cur)) ? cur : values[0];
    };
    const runs = this.runs, st = this.status;
    const have = new Set(runs.map((r) => r.mode));
    const modes = [...new Set(["fast", ...have])].filter((m) => m === "fast" ? st || have.has("fast") : true);
    const firstWithRuns = modes.find((m) => have.has(m)) || "fast";
    // a link asks for a mode; one without runs falls back to one with runs (LuST opens on its traffic)
    const pick = want.mode === undefined ? undefined : have.has(want.mode) ? want.mode : firstWithRuns;
    set("mode", modes, (v) => (MODES[v] || v) + (have.has(v) ? "" : " · no runs yet"), pick);
    $("mode-field").style.display = modes.length > 1 ? "" : "none";
    const mode = $("mode").value, inMode = runs.filter((r) => r.mode === mode);
    const design = mode === "fast" && st;
    const scenarios = [...new Set([...(design ? st.scenarios : []), ...inMode.map((r) => r.scenario)])].sort();
    set("scenario", scenarios, (v) => (TIMES[v] ? `${TIMES[v]}` : v), want.scenario);
    const sc = $("scenario").value, inSc = inMode.filter((r) => r.scenario === sc);
    const present = new Set([...(design ? st.methods : []), ...inSc.map((r) => r.method)]);
    const methods = METHOD_ORDER.filter((m) => present.has(m)).concat([...present].filter((m) => !METHOD_ORDER.includes(m)));
    const mlabel = (m) => methodName(m) + (design ? (SUFFIX[(this.cell(sc, m) || {}).state] || "") : "");
    set("method", methods, mlabel, want.method);
    const me = $("method").value;
    const seeds = [...new Set([...inSc.filter((r) => r.method === me).map((r) => r.seed), ...(design ? st.seeds : [])])].sort((a, b) => a - b);
    set("seed", seeds, (v) => (mode === "traces" ? "recording" : String(v)), want.seed);
    const seed = +$("seed").value;
    const others = methods.filter((m) => m !== me && (design || inSc.some((r) => r.method === m && r.seed === seed)));
    set("method2", others, mlabel, want.method2 ?? (others.includes("greedy") ? "greedy" : others[0]));
    if (want.method2) $("compare").checked = true;
    if (!others.length) $("compare").checked = false;
    $("compare").disabled = !others.length;
    $("method2").disabled = !$("compare").checked;
  }
  selection() {
    const s = { scenario: $("scenario").value, method: $("method").value, seed: $("seed").value, mode: $("mode").value };
    if ($("compare").checked && $("method2").value) s.method2 = $("method2").value;
    return s;
  }
  runIds(s = this.selection()) {
    const id = (m) => (s.mode === "traces" ? `traffic-${s.scenario}` : `${s.scenario}-${m}-${s.seed}-${s.mode}`);
    return s.method2 ? [id(s.method), id(s.method2)] : [id(s.method)];
  }
  exists(runId) { return this.ids.has(runId); }
  // why a selected run is missing, in words
  missing(s, method) {
    const c = this.cell(s.scenario, method), note = this.status && this.status.notes[s.scenario];
    const what = `${s.scenario} · ${methodName(method)} · seed ${s.seed}`;
    let why = "Not in this run's results.";
    if (c && c.state === "not trained") why = "Not trained yet on this map, so there is nothing to replay.";
    else if (c && c.state === "trained") why = "Trained, but not evaluated yet.";
    else if (c && c.seeds.length) why = `Evaluated on seeds ${c.seeds[0]}–${c.seeds[c.seeds.length - 1]}, not this one.`;
    return { what, why, note };
  }
  emit() { this.onChange(this.selection()); }
}
