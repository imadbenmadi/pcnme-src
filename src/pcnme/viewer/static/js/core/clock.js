// The viewer's time: play, pause, speed, seek. Views draw "the state at clock.t" and never keep time themselves.
// Live mode (`lag` set): the clock runs in real time but stays `lag` seconds behind the newest data (a task shows
// once its zone has reported it closed); it never stops at the end, and seeking back replays the buffer.

export const SPEEDS = [0.25, 0.5, 1, 2, 5, 10, 20];

export class Clock {
  constructor() {
    this.t = 0; this.end = 0; this.playing = false; this.speed = 1;
    this.frameListeners = []; this.stateListeners = [];
    this.last = 0;
    this.lag = null; // live mode: seconds behind the newest data
    requestAnimationFrame((ts) => this.tick(ts));
  }
  onFrame(fn) { this.frameListeners.push(fn); }   // every drawn frame (canvas)
  onState(fn) { this.stateListeners.push(fn); }   // play / pause / speed / range changed (controls)
  emitState() { this.stateListeners.forEach((f) => f(this)); }
  emitFrame(force = false) { this.frameListeners.forEach((f) => f(this, force)); }

  setRange(end) { this.end = end; this.t = Math.min(this.t, end); this.emitState(); this.emitFrame(true); }
  play() { if (this.lag == null && this.t >= this.end) this.t = 0; this.playing = true; this.emitState(); }
  goLive() { this.t = Math.max(0, this.end - (this.lag || 0)); this.speed = 1; this.play(); }
  pause() { this.playing = false; this.emitState(); }
  toggle() { this.playing ? this.pause() : this.play(); }
  setSpeed(s) { this.speed = s; this.emitState(); }
  seek(t) { this.t = Math.max(0, Math.min(this.end, t)); this.emitFrame(true); }

  tick(ts) {
    const dt = Math.min(0.1, (ts - (this.last || ts)) / 1000);
    this.last = ts;
    if (this.playing) {
      this.t += dt * this.speed;
      if (this.lag != null) this.t = Math.max(0, Math.min(this.t, this.end - this.lag));
      else if (this.t >= this.end) { this.t = this.end; this.pause(); }
      this.emitFrame();
    }
    requestAnimationFrame((x) => this.tick(x));
  }
}
