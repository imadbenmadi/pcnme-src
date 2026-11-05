// PlaceholderView: a source that is not ready yet (status "development"): its title, summary and steps.

import { esc } from "../core/format.js";

export function renderPlaceholder(el, info) {
  const n = info.notes || {};
  el.innerHTML = `<div class="dev-card">
    <div class="eyebrow">In development</div>
    <h1>${esc(n.title || info.label)}</h1>
    <p>${esc(n.summary || "Not available yet.")}</p>
    ${n.steps ? `<ul class="dev-steps">${n.steps.map(([what, state, done]) => `<li class="${done ? "done" : ""}"><span>${esc(what)}</span><span>${esc(state)}</span></li>`).join("")}</ul>` : ""}
  </div>`;
}
