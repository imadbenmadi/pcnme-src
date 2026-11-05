// CityView: the city on a canvas at time t, drawn from a model (core/model.js). It knows nothing of where the
// model's data comes from. Cars on the roads; zones with their coverage (fill = load), fog nodes (slots busy,
// queues); backhaul links; the metro-edge router, the WAN pipe (fill = queue wait) and the cloud; tasks as
// packets car -> anchor -> executor -> car, coloured by their outcome; drops as a fading ring at the anchor.

import { fmtMs } from "../core/format.js";
import { C, alpha } from "../core/theme.js";

export class CityView {
  constructor(container, model, { onSelect, speed }) {
    this.m = model;
    this.speed = speed; // () -> playback speed: a drop marker stays visible for a few hundred ms of wall time
    this.canvas = document.createElement("canvas");
    container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    this.hits = [];
    const [W, H] = model.layout.area;
    this.router = [W + 110, H / 2];
    this.cloudPos = [W + 560, H / 2];
    this.bounds = [-40, -40, W + 640, H + 40];
    this.t = 0; this.sel = null;
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(container);
    this.canvas.addEventListener("click", (e) => onSelect(this.hit(e)));
    this.canvas.addEventListener("mousemove", (e) => this.canvas.classList.toggle("hover", !!this.hit(e)));
  }
  destroy() { this.observer.disconnect(); this.canvas.remove(); }

  resize() {
    const box = this.canvas.parentElement.getBoundingClientRect();
    if (!box.width || !box.height) return;
    const dpr = window.devicePixelRatio || 1;
    this.w = box.width; this.h = box.height; this.dpr = dpr;
    this.canvas.width = Math.round(box.width * dpr);
    this.canvas.height = Math.round(box.height * dpr);
    const [x0, y0, x1, y1] = this.bounds, pad = 28, card = 64; // the cloud card is drawn in pixels: keep room for it
    this.s = Math.min((this.w - 2 * pad - card) / (x1 - x0), (this.h - 2 * pad) / (y1 - y0));
    this.ox = (this.w - card - (x1 - x0) * this.s) / 2 - x0 * this.s;
    this.oy = (this.h - (y1 - y0) * this.s) / 2 - y0 * this.s;
    this.buildStatic();
    this.draw(this.t, this.sel);
  }
  X(x) { return this.ox + x * this.s; }
  Y(y) { return this.oy + y * this.s; }
  node(id) {
    if (id === "cloud") return this.cloudPos;
    const z = this.m.zone[id];
    return z ? [z.x, z.y] : this.router;
  }
  route(from, to) { // the cloud is behind the metro-edge router and the WAN
    const a = Array.isArray(from) ? from : this.node(from), b = Array.isArray(to) ? to : this.node(to);
    return from === "cloud" || to === "cloud" ? [a, this.router, b] : [a, b];
  }

  buildStatic() { // the roads and blocks: drawn once per size and theme
    const c = document.createElement("canvas");
    c.width = this.canvas.width; c.height = this.canvas.height;
    const g = c.getContext("2d");
    g.scale(this.dpr, this.dpr);
    const L = this.m.layout, [W, H] = L.area, B = L.block, road = Math.max(4, 16 * this.s) / this.s;
    g.fillStyle = C["map-bg"]; g.fillRect(0, 0, this.w, this.h);
    g.fillStyle = C.road;
    roundRect(g, this.X(-road), this.Y(-road), (W + 2 * road) * this.s, (H + 2 * road) * this.s, 4); g.fill();
    g.fillStyle = C.block;
    for (let x = 0; x < W; x += B)
      for (let y = 0; y < H; y += B) {
        roundRect(g, this.X(x + road / 2), this.Y(y + road / 2), (B - road) * this.s, (B - road) * this.s, 3);
        g.fill();
      }
    this.static = c;
  }

  draw(t, sel) {
    this.t = t; this.sel = sel;
    if (!this.static) return;
    const g = this.ctx, m = this.m, L = m.layout;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.drawImage(this.static, 0, 0, this.w, this.h);
    this.hits = [];
    this.drawCoverage(g, t, sel);
    this.drawLinks(g);
    this.drawWan(g, t, sel);
    this.drawCars(g, t);
    this.drawPackets(g, t);
    for (const z of L.zones) this.drawNode(g, z, t, sel);
    this.drawSelection(g, t, sel);
  }

  drawCoverage(g, t, sel) {
    for (const z of this.m.layout.zones) {
      const cx = this.X(z.x), cy = this.Y(z.y);
      g.beginPath(); g.arc(cx, cy, z.r * this.s, 0, Math.PI * 2);
      if (!z.active) {
        g.setLineDash([4, 5]); g.strokeStyle = alpha(C.faint, 0.45); g.lineWidth = 1; g.stroke(); g.setLineDash([]);
        continue;
      }
      const load = Math.min(this.m.load(z.id, t), 1.5) / 1.5;
      g.fillStyle = alpha(C.accent, 0.025 + 0.10 * load); g.fill();
      const picked = sel && sel.kind === "zone" && sel.id === z.id;
      g.strokeStyle = alpha(C.accent, picked ? 0.9 : 0.28); g.lineWidth = picked ? 2 : 1; g.stroke();
    }
  }

  drawLinks(g) {
    const m = this.m, L = m.layout;
    g.lineWidth = 1.2; g.strokeStyle = C.link;
    for (const [a, b] of L.edges) {
      if (!m.zone[a].active || !m.zone[b].active) continue;
      line(g, this.X(m.zone[a].x), this.Y(m.zone[a].y), this.X(m.zone[b].x), this.Y(m.zone[b].y));
    }
    g.setLineDash([3, 4]);
    for (const z of L.zones) if (z.active) line(g, this.X(z.x), this.Y(z.y), this.X(this.router[0]), this.Y(this.router[1]));
    g.setLineDash([]);
  }

  drawCars(g, t) {
    const m = this.m;
    g.fillStyle = C.car;
    for (let c = 0; c < m.carCount(); c++) {
      const p = m.carPos(c, t);
      if (!p) continue;
      g.beginPath(); g.arc(this.X(p[0]), this.Y(p[1]), 2.3, 0, Math.PI * 2); g.fill();
    }
  }

  drawPackets(g, t) {
    const m = this.m, T = m.tasks, A = m.attempts;
    const [i0, i1] = m.tasksAround(t);
    const dropDur = Math.max(0.3, 0.35 * this.speed());
    for (let i = i0; i < i1; i++) {
      if (m.endT[i] + dropDur < t) continue;
      const col = C.outcome[m.cls[i]], car = T.car[i], t0 = T.t[i], up = T.upload[i];
      if (up != null && t < t0 + up) { // upload: car -> anchor
        const p = car >= 0 ? m.carPos(car, t0) : null;
        if (p) this.packet(g, [p, this.node(T.anchor[i])], (t - t0) / up, col, i);
        continue;
      }
      for (const k of m.byTask[i]) {
        const ex = A.executor[k];
        if (ex !== T.anchor[i] && A.send[k] != null && A.arrive[k] != null && t >= A.send[k] && t < A.arrive[k])
          this.packet(g, this.route(T.anchor[i], ex), (t - A.send[k]) / (A.arrive[k] - A.send[k]), col, i);
        if (A.finish[k] != null && A.car[k] != null && t >= A.finish[k] && t < A.car[k]) { // result: executor -> car
          const p = car >= 0 ? m.carPos(car, A.car[k]) : null;
          if (p) this.packet(g, this.route(ex, p), (t - A.finish[k]) / (A.car[k] - A.finish[k]), col, i);
        }
      }
      if (m.cls[i] === "dropped") {
        const tc = T.close[i];
        if (t >= tc && t < tc + dropDur) {
          const [x, y] = this.node(T.anchor[i]), u = (t - tc) / dropDur;
          g.beginPath(); g.arc(this.X(x), this.Y(y), 8 + 18 * u, 0, Math.PI * 2);
          g.strokeStyle = alpha(C.dropped, 0.65 * (1 - u)); g.lineWidth = 1.5; g.stroke();
        }
      }
    }
  }

  packet(g, path, u, col, task) {
    u = Math.max(0, Math.min(1, u));
    const p = along(path, u), q = along(path, Math.max(0, u - 0.12));
    const x = this.X(p[0]), y = this.Y(p[1]);
    g.strokeStyle = alpha(col, 0.35); g.lineWidth = 2;
    line(g, this.X(q[0]), this.Y(q[1]), x, y);
    g.fillStyle = col;
    g.beginPath(); g.arc(x, y, 3, 0, Math.PI * 2); g.fill();
    this.hits.push({ x, y, task });
  }

  drawWan(g, t, sel) {
    const m = this.m;
    const rx = this.X(this.router[0]), ry = this.Y(this.router[1]);
    const cx = this.X(this.cloudPos[0]), cy = this.Y(this.cloudPos[1]);
    const cw = 104, ch = 50, x0 = rx + 16, x1 = cx - cw / 2 - 14, h = 12;
    const wait = m.wan(t), fill = Math.min(1, wait / m.wanMax);
    const pickedWan = sel && sel.kind === "wan";
    g.fillStyle = C.surface; g.strokeStyle = pickedWan ? C.accent : C["border-strong"]; g.lineWidth = pickedWan ? 2 : 1;
    roundRect(g, x0, ry - h / 2, x1 - x0, h, h / 2); g.fill(); g.stroke();
    if (fill > 0.002) {
      g.fillStyle = alpha(wait > 0.5 ? C.late : C.accent, 0.6);
      roundRect(g, x0 + 2, ry - h / 2 + 2, Math.max(h - 4, (x1 - x0 - 4) * fill), h - 4, (h - 4) / 2); g.fill();
    }
    g.font = `500 11px ${C.font}`; g.textAlign = "center"; g.fillStyle = C.muted;
    g.fillText("WAN", (x0 + x1) / 2, ry - h / 2 - 8);
    g.fillStyle = wait > 0.5 ? C.late : C.faint; g.font = `11px ${C.font}`;
    g.fillText(wait > 0.0005 ? `queue ${fmtMs(wait)}` : "no queue", (x0 + x1) / 2, ry + h / 2 + 15);
    this.hits.push({ x: (x0 + x1) / 2, y: ry, kind: "wan", w: (x1 - x0) / 2, h: 14 });
    g.beginPath(); g.arc(rx, ry, 7, 0, Math.PI * 2); // the metro-edge router
    g.fillStyle = C.surface; g.fill(); g.strokeStyle = C.muted; g.lineWidth = 1.5; g.stroke();
    g.fillStyle = C.faint; g.fillText("metro edge", rx, ry - 14);
    const picked = sel && sel.kind === "cloud"; // the cloud
    g.fillStyle = C.surface; g.strokeStyle = picked ? C.accent : C["border-strong"]; g.lineWidth = picked ? 2 : 1;
    roundRect(g, cx - cw / 2, cy - ch / 2, cw, ch, 7); g.fill(); g.stroke();
    g.textAlign = "left"; g.fillStyle = C.text; g.font = `600 12px ${C.font}`;
    g.fillText("Cloud", cx - cw / 2 + 11, cy - 4);
    g.fillStyle = C.muted; g.font = `11px ${C.font}`;
    g.fillText(`${m.cloudRunning(t)} running`, cx - cw / 2 + 11, cy + 13);
    this.hits.push({ x: cx, y: cy, kind: "cloud", w: cw / 2, h: ch / 2 });
  }

  drawNode(g, z, t, sel) {
    if (this.s * this.m.layout.area[0] < 400) return this.drawNodeCompact(g, z, t, sel);
    if (!z.active) {
      g.fillStyle = C.faint; g.font = `11px ${C.font}`; g.textAlign = "center";
      g.fillText(`${z.id} joins later`, this.X(z.x), this.Y(z.y) + 4);
      return;
    }
    const m = this.m, cx = this.X(z.x), cy = this.Y(z.y), w = 108, h = 60, x = cx - w / 2, y = cy - h / 2;
    const picked = sel && sel.kind === "zone" && sel.id === z.id;
    g.fillStyle = C.surface; g.strokeStyle = picked ? C.accent : C["border-strong"]; g.lineWidth = picked ? 2 : 1;
    g.shadowColor = "rgba(0,0,0,0.07)"; g.shadowBlur = 8; g.shadowOffsetY = 2;
    roundRect(g, x, y, w, h, 7); g.fill(); g.shadowColor = "transparent"; g.stroke();
    g.textAlign = "left"; g.fillStyle = C.text; g.font = `600 12px ${C.font}`;
    g.fillText(z.id, x + 10, y + 17);
    g.textAlign = "right"; g.fillStyle = C.faint; g.font = `11px ${C.font}`;
    g.fillText(z.hw, x + w - 10, y + 17);
    [["GPU", z.gpu, z.qgpu, "gpu"], ["CPU", z.cpu, z.qcpu, "cpu"]].forEach(([label, slots, cap, type], j) => {
      const yy = y + 32 + j * 15;
      g.textAlign = "left"; g.fillStyle = C.faint; g.font = `10px ${C.font}`;
      g.fillText(label, x + 10, yy + 4);
      if (!slots) { g.fillText("none", x + 38, yy + 4); return; }
      const busy = m.slot(z.id, type, "busy", t), q = m.slot(z.id, type, "queue", t);
      for (let s = 0; s < slots; s++) {
        g.fillStyle = s < busy ? C.accent : alpha(C.faint, 0.25);
        roundRect(g, x + 38 + s * 11, yy - 3, 8, 8, 2); g.fill();
      }
      const bx = x + 64, bw = w - 74, f = cap ? Math.min(1, q / cap) : 0; // the queue against its capacity
      g.fillStyle = alpha(C.faint, 0.18); roundRect(g, bx, yy - 1, bw, 4, 2); g.fill();
      if (f > 0) { g.fillStyle = f > 0.75 ? C.late : C.accent; roundRect(g, bx, yy - 1, Math.max(3, bw * f), 4, 2); g.fill(); }
    });
    this.hits.push({ x: cx, y: cy, kind: "zone", id: z.id, w: w / 2, h: h / 2 });
  }

  drawNodeCompact(g, z, t, sel) { // a small canvas: the node as a dot, its fill = GPU (or CPU) slots busy
    const cx = this.X(z.x), cy = this.Y(z.y);
    if (!z.active) return;
    const type = z.gpu ? "gpu" : "cpu", slots = z.gpu || z.cpu, busy = this.m.slot(z.id, type, "busy", t);
    const picked = sel && sel.kind === "zone" && sel.id === z.id;
    g.beginPath(); g.arc(cx, cy, 9, 0, Math.PI * 2);
    g.fillStyle = C.surface; g.fill(); g.strokeStyle = picked ? C.accent : C["border-strong"]; g.lineWidth = picked ? 2 : 1; g.stroke();
    if (busy) { g.beginPath(); g.moveTo(cx, cy); g.arc(cx, cy, 7, -Math.PI / 2, -Math.PI / 2 + (2 * Math.PI * busy) / slots); g.fillStyle = C.accent; g.fill(); }
    g.fillStyle = C.muted; g.font = `600 11px ${C.font}`; g.textAlign = "center"; g.fillText(z.id, cx, cy - 14);
    this.hits.push({ x: cx, y: cy, kind: "zone", id: z.id, w: 12, h: 12 });
  }

  drawSelection(g, t, sel) {
    if (!sel) return;
    let p = null;
    if (sel.kind === "car") { const q = this.m.carPos(sel.id, t); if (q) p = [this.X(q[0]), this.Y(q[1])]; }
    if (sel.kind === "task") { const h = this.hits.find((k) => k.task === sel.id); if (h) p = [h.x, h.y]; }
    if (!p) return;
    g.beginPath(); g.arc(p[0], p[1], 7.5, 0, Math.PI * 2); g.strokeStyle = C.accent; g.lineWidth = 2; g.stroke();
  }

  hit(e) {
    const b = this.canvas.getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top, m = this.m;
    let best = null, bd = 64;
    for (const h of this.hits) if (h.task != null) { const d = (h.x - x) ** 2 + (h.y - y) ** 2; if (d < bd) { bd = d; best = { kind: "task", id: h.task }; } }
    if (best) return best;
    for (const h of this.hits) if (h.kind && Math.abs(h.x - x) <= h.w && Math.abs(h.y - y) <= h.h) return { kind: h.kind, id: h.id };
    bd = 49;
    for (let c = 0; c < m.carCount(); c++) {
      const p = m.carPos(c, this.t);
      if (!p) continue;
      const d = (this.X(p[0]) - x) ** 2 + (this.Y(p[1]) - y) ** 2;
      if (d < bd) { bd = d; best = { kind: "car", id: c }; }
    }
    if (best) return best;
    for (const z of m.layout.zones) if (z.active && Math.hypot(this.X(z.x) - x, this.Y(z.y) - y) < z.r * this.s) return { kind: "zone", id: z.id };
    return null;
  }
}

function line(g, x0, y0, x1, y1) { g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); }
function roundRect(g, x, y, w, h, r) {
  g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
}
function along(path, u) {
  let total = 0; const seg = [];
  for (let i = 1; i < path.length; i++) { const d = Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]); seg.push(d); total += d; }
  let goal = u * total;
  for (let i = 0; i < seg.length; i++) {
    if (goal <= seg[i] || i === seg.length - 1) {
      const v = seg[i] ? Math.min(1, goal / seg[i]) : 1, a = path[i], b = path[i + 1];
      return [a[0] + (b[0] - a[0]) * v, a[1] + (b[1] - a[1]) * v];
    }
    goal -= seg[i];
  }
  return path[path.length - 1];
}
