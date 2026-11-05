// RunPicker: scenario, method, seed and source (fast mode or testbed) from a replay source's runs, and the
// comparison method (the same scenario and seed). Calls onChange(selection) with the run ids to show.

import { $, METHOD_ORDER, esc, methodName } from "../core/format.js";

export class RunPicker {
  constructor(onChange) {
    this.onChange = onChange;
    for (const id of ["scenario", "method", "seed", "mode"]) $(id).onchange = () => { this.fill({}); this.emit(); };
    $("method2").onchange = () => this.emit();
    $("compare").onchange = () => { $("method2").disabled = !$("compare").checked; this.emit(); };
  }
  setRuns(runs, want = {}) { this.runs = runs; this.fill(want); }
  fill(want) {
    const set = (id, values, label, pick) => {
      const el = $(id), cur = pick ?? el.value;
      el.innerHTML = values.map((v) => `<option value="${esc(v)}">${esc(label(v))}</option>`).join("");
      el.value = values.map(String).includes(String(cur)) ? cur : values[0];
    };
    const runs = this.runs;
    const modes = [...new Set(runs.map((r) => r.mode))].sort();
    set("mode", modes, (v) => (v === "fast" ? "Fast mode" : "Testbed"), want.mode);
    $("mode-field").style.display = modes.length > 1 ? "" : "none";
    const inMode = runs.filter((r) => r.mode === $("mode").value);
    set("scenario", [...new Set(inMode.map((r) => r.scenario))].sort(), String, want.scenario);
    const inSc = inMode.filter((r) => r.scenario === $("scenario").value);
    const present = new Set(inSc.map((r) => r.method));
    const methods = METHOD_ORDER.filter((m) => present.has(m)).concat([...present].filter((m) => !METHOD_ORDER.includes(m)));
    set("method", methods, methodName, want.method);
    const me = $("method").value;
    set("seed", [...new Set(inSc.filter((r) => r.method === me).map((r) => r.seed))].sort((a, b) => a - b), String, want.seed);
    const seed = +$("seed").value;
    const others = methods.filter((m) => m !== me && inSc.some((r) => r.method === m && r.seed === seed));
    set("method2", others, methodName, want.method2 ?? (others.includes("greedy") ? "greedy" : others[0]));
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
    const id = (m) => `${s.scenario}-${m}-${s.seed}-${s.mode}`;
    return s.method2 ? [id(s.method), id(s.method2)] : [id(s.method)];
  }
  emit() { this.onChange(this.selection()); }
}
