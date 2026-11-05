// TransportControls: play / pause, the clock, the time slider (with the drain after the last arrival), speed.

import { $ } from "../core/format.js";
import { SPEEDS } from "../core/clock.js";

export class TransportControls {
  constructor(clock) {
    this.clock = clock;
    $("play").onclick = () => clock.toggle();
    $("time").oninput = (e) => clock.seek(+e.target.value);
    $("speeds").innerHTML = SPEEDS.map((s) => `<button data-s="${s}">${s}×</button>`).join("");
    $("speeds").querySelectorAll("button").forEach((b) => (b.onclick = () => clock.setSpeed(+b.dataset.s)));
    clock.onState(() => this.sync());
    this.sync();
  }
  setArrivals(arrivals) { this.arrivals = arrivals; this.sync(); }
  sync() {
    const c = this.clock;
    $("icon-play").style.display = c.playing ? "none" : "";
    $("icon-pause").style.display = c.playing ? "" : "none";
    $("play").setAttribute("aria-label", c.playing ? "Pause" : "Play");
    $("speeds").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.s === c.speed)));
    $("time").max = c.end || 1;
    if (c.end && this.arrivals != null) {
      $("drain").style.width = `${(100 * Math.max(0, c.end - this.arrivals)) / c.end}%`;
      $("mark-end").style.left = `${(100 * this.arrivals) / c.end}%`;
    }
  }
  frame(t) {
    $("time").value = t;
    $("clock").innerHTML = `<b>${t.toFixed(1)} s</b> / ${this.clock.end.toFixed(0)} s`;
  }
}
