"use strict";
/* Įrankinė — naršyklės dalis. Be priklausomybių. */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null) el.append(c.nodeType ? c : document.createTextNode(c));
  return el;
};
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.detail || msg; } catch {}
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r.text();
};
const fmtDate = (t) => new Date(t * 1000).toLocaleString("lt-LT", { dateStyle: "short", timeStyle: "short" });

const S = { info: null, tool: null, files: {}, values: {}, job: null, poll: null, logNext: 0, preview: null };

/* ------------------------------------------------------------ meniu */
async function boot() {
  S.info = await api("api/info");
  document.title = S.info.title;
  $("#brand-title").textContent = S.info.title;
  const groups = {};
  for (const t of S.info.tools) (groups[t.group] ||= []).push({ ...t, kind: "tool" });
  for (const l of S.info.links || []) (groups[l.group || "Kita"] ||= []).push({ ...l, kind: "link" });
  const nav = $("#nav");
  for (const [g, items] of Object.entries(groups)) {
    nav.append(h("div", { class: "nav-group" }, g));
    for (const it of items) {
      nav.append(it.kind === "tool"
        ? h("a", { class: "nav-item", href: `#/t/${it.id}`, "data-tool": it.id }, h("span", { class: "ic", "aria-hidden": "true" }, it.icon || "•"), it.title)
        : h("a", { class: "nav-item", href: it.url }, h("span", { class: "ic", "aria-hidden": "true" }, it.icon || "•"), it.title, h("span", { class: "ext", title: "Atskiras puslapis" }, "↗")));
    }
  }
  const list = $("#home-list");
  for (const t of S.info.tools) list.append(h("a", { class: "home-item", href: `#/t/${t.id}` }, h("h2", {}, t.title), h("p", {}, t.description)));
  for (const l of S.info.links || []) list.append(h("a", { class: "home-item", href: l.url }, h("h2", {}, l.title + " ↗"), h("p", {}, l.description || "Atskiras puslapis")));
  $("#btn-history").onclick = () => (location.hash = "#/history");
  window.addEventListener("hashchange", route);
  route();
}

function show(view) {
  for (const v of ["home", "tool", "history"]) $(`#view-${v}`).hidden = v !== view;
}

async function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  $$(".nav-item").forEach((a) => a.removeAttribute("aria-current"));
  if (parts[0] === "t" && parts[1]) {
    const tool = S.info.tools.find((t) => t.id === parts[1]);
    if (!tool) return (location.hash = "#/");
    $(`.nav-item[data-tool="${tool.id}"]`)?.setAttribute("aria-current", "page");
    show("tool");
    if (S.tool?.id !== tool.id) openTool(tool);
    if (parts[2] && S.job?.id !== parts[2]) openJob(parts[2]);
    if (!parts[2] && S.job) resetResults();
  } else if (parts[0] === "history") {
    show("history"); loadHistory();
  } else show("home");
}

/* ------------------------------------------------------------ įrankio forma */
function openTool(tool) {
  S.tool = tool; S.files = {}; resetResults();
  $("#tool-title").textContent = tool.title;
  $("#tool-desc").textContent = tool.description;
  $("#tool-help").innerHTML = tool.help || "";
  $("#tool-help-wrap").hidden = !tool.help;
  const defaults = Object.fromEntries(tool.params.map((p) => [p.key, p.default]));
  S.values = { ...defaults, ...store.get(`tb:params:${tool.id}`, {}) };
  renderInputs(); renderParams();
}

function renderInputs() {
  const box = $("#inputs"); box.replaceChildren();
  for (const spec of S.tool.inputs) {
    S.files[spec.key] ||= [];
    const input = h("input", { type: "file", hidden: true, accept: spec.accept, multiple: spec.multiple });
    const list = h("ul", { class: "files" });
    const drop = h("div", { class: "drop", tabindex: "0", role: "button" },
      h("strong", {}, spec.label), h("br"), `Nutempkite failus arba spauskite čia${spec.accept ? ` (${spec.accept})` : ""}`);
    const add = (fl) => {
      const arr = [...fl];
      S.files[spec.key] = spec.multiple ? [...S.files[spec.key], ...arr] : arr.slice(0, 1);
      draw();
    };
    const draw = () => {
      list.replaceChildren(...S.files[spec.key].map((f, i) => h("li", {}, h("span", {}, f.name, " ", h("small", {}, `${(f.size / 1024).toFixed(0)} kB`)),
        h("button", { type: "button", "aria-label": `Pašalinti ${f.name}`, onclick: () => { S.files[spec.key].splice(i, 1); draw(); } }, "✕"))));
    };
    drop.onclick = () => input.click();
    drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } };
    input.onchange = () => { add(input.files); input.value = ""; };
    drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
    drop.ondragleave = () => drop.classList.remove("over");
    drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); add(e.dataTransfer.files); };
    box.append(drop, input, list); draw();
  }
}

function renderParams() {
  const box = $("#params"); box.replaceChildren();
  const groups = {};
  for (const p of S.tool.params) (groups[p.group] ||= []).push(p);
  for (const [g, ps] of Object.entries(groups)) {
    const fs = h("fieldset", {}, h("legend", {}, g));
    for (const p of ps) {
      const id = `p-${p.key}`;
      let row;
      const save = (v) => { S.values[p.key] = v; store.set(`tb:params:${S.tool.id}`, S.values); syncEnabled(); };
      if (p.type === "bool") {
        const cb = h("input", { type: "checkbox", id, checked: !!S.values[p.key], onchange: (e) => save(e.target.checked) });
        row = h("div", { class: "field check" }, cb, h("label", { for: id }, p.label));
      } else if (p.type === "choice") {
        const sel = h("select", { id, onchange: (e) => save(e.target.value) }, ...p.choices.map(([v, l]) => h("option", { value: v, selected: v === S.values[p.key] }, l)));
        row = h("div", { class: "field" }, h("label", { for: id }, p.label), sel);
      } else if (p.type === "str") {
        const inp = h("input", { type: "text", id, value: S.values[p.key] ?? "", oninput: (e) => save(e.target.value) });
        row = h("div", { class: "field wide" }, h("label", { for: id }, p.label), inp);
      } else {
        const inp = h("input", { type: "text", inputmode: "decimal", id, value: S.values[p.key] ?? "",
          oninput: (e) => save(e.target.value.replace(",", ".")) });
        row = h("div", { class: "field" }, h("label", { for: id }, p.label), h("div", { class: "num" }, inp, p.unit ? h("span", { class: "u" }, p.unit) : null));
      }
      row.dataset.key = p.key;
      if (p.enabled_by) row.dataset.enabledBy = p.enabled_by;
      if (p.help) row.append(h("div", { class: "hint" }, p.help));
      fs.append(row);
    }
    box.append(fs);
  }
  syncEnabled();
}

function syncEnabled() {
  for (const row of $$("#params [data-enabled-by]")) {
    const on = !!S.values[row.dataset.enabledBy];
    row.classList.toggle("off", !on);
    $$("input,select", row).forEach((i) => (i.disabled = !on));
  }
}

$("#btn-reset").onclick = () => {
  S.values = Object.fromEntries(S.tool.params.map((p) => [p.key, p.default]));
  store.set(`tb:params:${S.tool.id}`, S.values); renderParams();
};
$("#btn-export").onclick = () => {
  const blob = new Blob([JSON.stringify({ tool: S.tool.id, params: S.values }, null, 2)], { type: "application/json" });
  const a = h("a", { href: URL.createObjectURL(blob), download: `${S.tool.id}_nustatymai.json` });
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};
$("#preset-file").onchange = async (e) => {
  try {
    const j = JSON.parse(await e.target.files[0].text());
    if (j.tool && j.tool !== S.tool.id) throw new Error(`Nustatymai skirti įrankiui „${j.tool}“`);
    S.values = { ...S.values, ...(j.params || j) };
    store.set(`tb:params:${S.tool.id}`, S.values); renderParams(); formError("");
  } catch (err) { formError(`Nepavyko įkelti nustatymų: ${err.message}`); }
  e.target.value = "";
};

function formError(msg) { const el = $("#form-error"); el.textContent = msg; el.hidden = !msg; }

$("#tool-form").onsubmit = async (e) => {
  e.preventDefault(); formError("");
  for (const spec of S.tool.inputs) if (spec.required && !S.files[spec.key]?.length) return formError(`Įkelkite failą: ${spec.label}`);
  const fd = new FormData();
  for (const [k, arr] of Object.entries(S.files)) for (const f of arr) fd.append(k, f, f.name);
  fd.append("params", JSON.stringify(S.values));
  $("#btn-run").disabled = true;
  try {
    const { job_id } = await api(`api/tools/${S.tool.id}/run`, { method: "POST", body: fd });
    location.hash = `#/t/${S.tool.id}/${job_id}`;
  } catch (err) { formError(err.message); $("#btn-run").disabled = false; }
};

/* ------------------------------------------------------------ užduotis */
function resetResults() {
  clearInterval(S.poll); S.job = null; S.preview = null; S.logNext = 0;
  $("#log").textContent = ""; $("#report").textContent = ""; $("#charts").replaceChildren();
  $("#results").hidden = true; setStatus("", "");
  $("#btn-run").disabled = false; plan.set(null);
}

function setStatus(cls, text) { const el = $("#job-status"); el.className = `job-status ${cls}`; el.textContent = text; }

async function openJob(id) {
  resetResults();
  S.job = { id };
  $("#btn-run").disabled = true;
  setStatus("running", "Skaičiuojama…");
  selectTab("log");
  const tick = async () => {
    let j;
    try { j = await api(`api/jobs/${id}?log_from=${S.logNext}`); } catch (err) {
      clearInterval(S.poll); setStatus("error", err.message); $("#btn-run").disabled = false; return;
    }
    if (S.job?.id !== id) return;
    if (j.log.length) { $("#log").textContent += j.log.join("\n") + "\n"; S.logNext = j.log_next; }
    if (j.status === "done" || j.status === "error") {
      clearInterval(S.poll); S.job = j; $("#btn-run").disabled = false;
      if (S.tool && j.params) { S.values = { ...S.values, ...j.params }; renderParams(); }
      if (j.status === "error") { setStatus("error", `Klaida: ${j.error}`); return; }
      setStatus("done", `Baigta ${fmtDate(j.finished)}`);
      await showResults(j);
    }
  };
  await tick();
  if (S.job && !S.job.status) S.poll = setInterval(tick, 800);
}

async function showResults(j) {
  const base = `api/jobs/${j.id}/files/`;
  $("#downloads").replaceChildren(...j.outputs.map((o) =>
    h("a", { class: "dl", href: base + encodeURIComponent(o.name), download: o.name }, o.label, h("small", {}, o.name))));
  const rep = j.outputs.find((o) => o.kind === "text");
  if (rep) $("#report").textContent = await api(base + encodeURIComponent(rep.name) + "?inline=true");
  $("#summary").replaceChildren();
  if (j.has_preview) {
    S.preview = await api(`api/jobs/${j.id}/preview`);
    if (S.preview.table?.length) $("#summary").append(table(S.preview.table));
    plan.set(S.preview.plan);
    setLegend(S.preview.legend);
    drawCharts(S.preview.charts || []);
  }
  $("#results").hidden = false;
  selectTab(S.preview?.plan ? "plan" : "report");
}

const LEGEND_DEFAULT = [["line", "tiesė"], ["arc", "lankas"], ["pt", "įvesties taškai"], ["bad", "už tolerancijos"]];
function setLegend(items) {
  const box = $("#plan-legend");
  box.replaceChildren(...(items || LEGEND_DEFAULT).map(([k, label]) =>
    h("span", {}, h("i", { class: `sw sw-${k}` }), label)));
}

function table(rows) {
  const cols = Object.keys(rows[0]);
  return h("table", { class: "tbl" },
    h("thead", {}, h("tr", {}, cols.map((c) => h("th", {}, c)))),
    h("tbody", {}, rows.map((r) => h("tr", {}, cols.map((c) => {
      const v = r[c]; const n = typeof v === "number";
      return h("td", { class: (n ? "n" : "") + (c === "Už tol." && v > 0 ? " warn" : "") }, n ? v.toLocaleString("lt-LT") : v);
    })))));
}

/* ------------------------------------------------------------ skirtukai */
function selectTab(name) {
  $$(".tabs [role=tab]").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === name));
  $$(".tabpane").forEach((p) => (p.hidden = p.dataset.pane !== name));
  if (name === "plan") plan.resize();
  if (name === "chart") drawCharts(S.preview?.charts || []);
}
$$(".tabs [role=tab]").forEach((b) => (b.onclick = () => selectTab(b.dataset.tab)));

/* ------------------------------------------------------------ plano peržiūra */
const plan = (() => {
  const cv = $("#plan"), ctx = cv.getContext("2d"), tip = $("#plan-tip");
  const css = getComputedStyle(document.documentElement);
  const col = (n) => css.getPropertyValue(n).trim();
  let data = null, view = null, W = 0, H = 0, dpr = 1, hover = null;

  const toS = (x, y) => [(x - view.cx) * view.s + W / 2, H / 2 - (y - view.cy) * view.s];
  const toW = (sx, sy) => [(sx - W / 2) / view.s + view.cx, (H / 2 - sy) / view.s + view.cy];

  function bounds() {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    const add = ([x, y]) => { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; };
    for (const a of data) { a.points.forEach(add); a.elements.forEach((e) => e.xy.forEach(add)); }
    return { x0, y0, x1, y1 };
  }
  function fit() {
    if (!data?.length) return;
    const b = bounds(), pad = 40;
    const s = Math.min((W - 2 * pad) / Math.max(b.x1 - b.x0, 1e-3), (H - 2 * pad) / Math.max(b.y1 - b.y0, 1e-3));
    view = { s, cx: (b.x0 + b.x1) / 2, cy: (b.y0 + b.y1) / 2 };
    draw();
  }
  function resize() {
    const r = cv.getBoundingClientRect();
    if (!r.width) return;
    dpr = window.devicePixelRatio || 1; W = r.width; H = r.height;
    cv.width = W * dpr; cv.height = H * dpr;
    if (!view) fit(); else draw();
  }
  function path(pts) {
    ctx.beginPath();
    pts.forEach((p, i) => { const [x, y] = toS(p[0], p[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  }
  function draw() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (!data || !view) return;
    drawScale();
    // taškai retinami ekrane: piešiamas tik jei nuo ankstesnio nupiešto > minPx (kitaip uždengia linijas)
    const dots = (arr, minPx, fn) => {
      let lx = -1e9, ly = -1e9;
      for (const p of arr) {
        const [x, y] = toS(p[0], p[1]);
        if (x < -10 || y < -10 || x > W + 10 || y > H + 10) continue;
        if (Math.abs(x - lx) < minPx && Math.abs(y - ly) < minPx) continue;
        lx = x; ly = y; fn(x, y);
      }
    };
    ctx.fillStyle = col("--pt-c");
    for (const a of data) dots(a.points, 4, (x, y) => ctx.fillRect(x - 1.5, y - 1.5, 3, 3));
    for (const a of data) for (const e of a.elements) {
      path(e.xy);
      ctx.strokeStyle = e.type === "arc" ? col("--arc-c") : e.type === "removed" ? col("--bad-c") : col("--line-c");
      ctx.setLineDash(e.type === "removed" ? [6, 4] : []);
      ctx.lineWidth = hover?.e === e ? 4 : 2; ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.fillStyle = "#fff";
    for (const a of data) dots(a.joints, 8, (x, y) => { ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 7); ctx.fill(); });
    ctx.strokeStyle = col("--bad-c"); ctx.lineWidth = 2;
    for (const a of data) for (const p of a.bad) { const [x, y] = toS(p[0], p[1]); ctx.beginPath(); ctx.arc(x, y, 7, 0, 7); ctx.stroke(); }
  }
  function drawScale() {
    const target = 120 / view.s, pow = 10 ** Math.floor(Math.log10(target));
    const len = [1, 2, 5, 10].map((m) => m * pow).find((v) => v >= target * 0.6) || pow;
    const px = len * view.s, x = W - px - 20, y = H - 18;
    ctx.strokeStyle = "#b9c8d6"; ctx.lineWidth = 1.5; ctx.beginPath();
    ctx.moveTo(x, y - 5); ctx.lineTo(x, y); ctx.lineTo(x + px, y); ctx.lineTo(x + px, y - 5); ctx.stroke();
    ctx.fillStyle = "#b9c8d6"; ctx.font = "12px Barlow, sans-serif"; ctx.textAlign = "center";
    ctx.fillText(len >= 1 ? `${len} m` : `${len * 1000} mm`, x + px / 2, y - 6);
  }
  function pick(sx, sy) {
    let best = null, bd = 10;
    for (const a of data || []) for (const e of a.elements) {
      for (let i = 1; i < e.xy.length; i++) {
        const [x1, y1] = toS(...e.xy[i - 1]), [x2, y2] = toS(...e.xy[i]);
        const dx = x2 - x1, dy = y2 - y1, l2 = dx * dx + dy * dy || 1;
        const t = Math.max(0, Math.min(1, ((sx - x1) * dx + (sy - y1) * dy) / l2));
        const d = Math.hypot(sx - (x1 + t * dx), sy - (y1 + t * dy));
        if (d < bd) { bd = d; best = { a, e }; }
      }
    }
    return best;
  }
  let drag = null;
  cv.addEventListener("pointerdown", (ev) => { drag = { x: ev.clientX, y: ev.clientY, cx: view?.cx, cy: view?.cy }; cv.setPointerCapture(ev.pointerId); cv.classList.add("drag"); });
  cv.addEventListener("pointerup", () => { drag = null; cv.classList.remove("drag"); });
  cv.addEventListener("pointermove", (ev) => {
    if (!view) return;
    const r = cv.getBoundingClientRect(), sx = ev.clientX - r.left, sy = ev.clientY - r.top;
    if (drag) { view.cx = drag.cx - (ev.clientX - drag.x) / view.s; view.cy = drag.cy + (ev.clientY - drag.y) / view.s; tip.hidden = true; draw(); return; }
    const hit = pick(sx, sy);
    if (hit?.e !== hover?.e) { hover = hit; draw(); }
    if (hit) {
      const e = hit.e, n = (v) => v.toLocaleString("lt-LT", { minimumFractionDigits: 3, maximumFractionDigits: 3 });
      if (e.label) tip.replaceChildren(h("strong", {}, e.label), e.L != null ? h("br") : null, e.L != null ? `L ${n(e.L)} m` : null,
        h("br"), h("small", {}, hit.a.name));
      else tip.replaceChildren(h("strong", {}, `${e.type === "arc" ? "Lankas" : "Tiesė"}${e.R ? `, R ${n(e.R)} m` : ""}`), h("br"),
        `L ${n(e.L)} m, PK ${n(e.s0)} – ${n(e.s0 + e.L)}`, h("br"), h("small", {}, hit.a.name));
      tip.style.left = `${Math.min(sx + 14, W - 230)}px`; tip.style.top = `${sy + 14}px`; tip.hidden = false;
    } else tip.hidden = true;
  });
  cv.addEventListener("pointerleave", () => { tip.hidden = true; });
  cv.addEventListener("wheel", (ev) => {
    if (!view) return; ev.preventDefault();
    const r = cv.getBoundingClientRect(), sx = ev.clientX - r.left, sy = ev.clientY - r.top;
    const [wx, wy] = toW(sx, sy), f = Math.exp(-ev.deltaY * 0.0015);
    view.s *= f; view.cx = wx - (sx - W / 2) / view.s; view.cy = wy - (H / 2 - sy) / view.s; draw();
  }, { passive: false });
  cv.addEventListener("dblclick", fit);
  $("#btn-fit").onclick = fit;
  new ResizeObserver(resize).observe(cv);
  return {
    set(d) { data = d; view = null; hover = null; tip.hidden = true; $("#plan-empty").hidden = !!d; $("#plan-legend").hidden = !d; resize(); if (!d) draw(); },
    resize,
  };
})();

/* ------------------------------------------------------------ grafikai */
function drawCharts(charts) {
  const box = $("#charts");
  if ($("[data-pane=chart]").hidden) return;
  box.replaceChildren();
  const colors = ["#5fb3f5", "#7bd88f"];
  for (const c of charts) {
    const cv = h("canvas"); box.append(h("div", { class: "chart" }, h("h3", {}, c.title), cv));
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1, W = r.width, H = r.height;
    cv.width = W * dpr; cv.height = H * dpr;
    const g = cv.getContext("2d"); g.scale(dpr, dpr);
    const L = 44, R = 10, T = 22, B = 24;
    const x0 = Math.min(...c.x), x1 = Math.max(...c.x);
    let ym = 1;
    for (const s of c.series) for (const v of s.y) ym = Math.max(ym, Math.abs(v));
    for (const s of c.series) if (s.tol) ym = Math.max(ym, s.tol * 1.15);
    const X = (v) => L + ((v - x0) / Math.max(x1 - x0, 1e-9)) * (W - L - R);
    const Y = (v) => T + (1 - (v + ym) / (2 * ym)) * (H - T - B);
    g.font = "12px Barlow, sans-serif"; g.fillStyle = "#8ea2b4"; g.strokeStyle = "rgba(255,255,255,.08)";
    for (const v of c.vlines || []) { g.beginPath(); g.moveTo(X(v), T); g.lineTo(X(v), H - B); g.stroke(); }
    g.strokeStyle = "rgba(255,255,255,.35)"; g.beginPath(); g.moveTo(L, Y(0)); g.lineTo(W - R, Y(0)); g.stroke();
    g.textAlign = "right";
    for (const v of [ym, 0, -ym]) g.fillText(v.toFixed(1), L - 6, Y(v) + 4);
    g.textAlign = "center";
    for (let i = 0; i <= 5; i++) { const v = x0 + (i / 5) * (x1 - x0); g.fillText(v.toFixed(0), X(v), H - 6); }
    c.series.forEach((s, si) => {
      const color = colors[si % colors.length];
      if (s.tol) {
        g.setLineDash([5, 4]); g.strokeStyle = color; g.globalAlpha = 0.6;
        for (const t of [s.tol, -s.tol]) { g.beginPath(); g.moveTo(L, Y(t)); g.lineTo(W - R, Y(t)); g.stroke(); }
        g.setLineDash([]); g.globalAlpha = 1;
      }
      g.strokeStyle = color; g.lineWidth = 1.5; g.beginPath();
      s.y.forEach((v, i) => (i ? g.lineTo(X(c.x[i]), Y(v)) : g.moveTo(X(c.x[i]), Y(v)))); g.stroke();
      g.fillStyle = color; g.textAlign = "left";
      g.fillText(s.name + (s.tol ? `  (tol. ±${s.tol})` : ""), L + 4 + si * 190, 14);
    });
  }
}

/* ------------------------------------------------------------ istorija */
async function loadHistory() {
  const tb = $("#history-table tbody");
  tb.replaceChildren(h("tr", {}, h("td", { colspan: 5 }, "Įkeliama…")));
  try {
    const rows = await api("api/jobs?limit=100");
    const st = { done: "Baigta", error: "Klaida", running: "Vykdoma", queued: "Eilėje" };
    tb.replaceChildren(...rows.map((r) => h("tr", {},
      h("td", {}, h("a", { href: `#/t/${r.tool}/${r.id}` }, fmtDate(r.created))),
      h("td", {}, r.title), h("td", {}, Object.values(r.inputs || {}).flat().join(", ")),
      h("td", { class: `st-${r.status}` }, st[r.status] || r.status),
      h("td", {}, h("button", { class: "linkbtn", type: "button", onclick: async () => { await api(`api/jobs/${r.id}`, { method: "DELETE" }); loadHistory(); } }, "Ištrinti")))));
    if (!rows.length) tb.replaceChildren(h("tr", {}, h("td", { colspan: 5 }, "Dar nieko neskaičiuota.")));
  } catch (err) { tb.replaceChildren(h("tr", {}, h("td", { colspan: 5 }, `Nepavyko įkelti: ${err.message}`))); }
}

boot().catch((err) => { document.body.prepend(h("p", { class: "form-error" }, `Nepavyko paleisti: ${err.message}`)); });
