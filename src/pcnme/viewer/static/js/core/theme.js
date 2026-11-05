// The colour tokens of css/style.css, read once for the canvases (and again when the OS theme changes).

export const C = {};
const listeners = [];
const TOKENS = ["bg", "surface", "accent-soft", "border", "border-strong", "text", "muted", "faint", "accent", "on-time", "late",
  "dropped", "map-bg", "block", "road", "car", "link"];

export function readTheme() {
  const s = getComputedStyle(document.documentElement);
  for (const k of TOKENS) C[k] = s.getPropertyValue(`--${k}`).trim();
  C.outcome = { on_time: C["on-time"], late: C.late, dropped: C.dropped };
  C.font = getComputedStyle(document.body).fontFamily;
}

export function onTheme(fn) { listeners.push(fn); }

matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readTheme(); listeners.forEach((f) => f()); });

export function alpha(hex, a) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.replace(/./g, "$&$&") : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
