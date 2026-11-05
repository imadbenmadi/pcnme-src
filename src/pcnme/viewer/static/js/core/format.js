// Names, number formats and small helpers shared by every view.

export const METHOD_ORDER = ["pcnme", "greedy", "d3qn-greedy", "d3qn-scratch", "vfc-ppo", "tang-wong", "qeco", "random"];
export const METHOD_NAMES = {
  pcnme: "PCNME", greedy: "Greedy", random: "Random", "d3qn-scratch": "D3QN (scratch)",
  "d3qn-greedy": "D3QN + greedy demos", "vfc-ppo": "VFC PPO", "tang-wong": "Tang & Wong", qeco: "QECO",
};
export const OPTION_NAMES = { here: "here", n1: "neighbour 1", n2: "neighbour 2", n3: "neighbour 3", n4: "neighbour 4", cloud: "cloud" };

export const methodName = (m) => METHOD_NAMES[m] || m;
export const outcomeClass = (o) => (o === "on_time" ? "on_time" : o === "late" ? "late" : "dropped");
export const fmtMs = (s) => (s == null ? "–" : `${(s * 1000).toFixed(s < 0.1 ? 1 : 0)} ms`);
export const fmtPct = (x) => (x == null || isNaN(x) ? "–" : `${(100 * x).toFixed(1)} %`);
export const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
export const $ = (id) => document.getElementById(id);

export function lowerBound(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; }
  return lo;
}
export function upperBound(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= x) lo = m + 1; else hi = m; }
  return lo;
}
