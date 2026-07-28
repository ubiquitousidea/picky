const state = {
  effects: [],
  effect: null,    // name of the effect the Apply panel is set to
  images: [],
  imageId: null,
  nodes: [],       // flat node list for the selected image
  nodeId: null,    // selected node
  presets: [],     // saved effect chains, reusable across images
};

const $ = (id) => document.getElementById(id);

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

// ---------- Gallery ----------

async function refreshGallery() {
  state.images = await api("/api/images");
  const ul = $("gallery");
  ul.innerHTML = "";
  for (const img of state.images) {
    const li = document.createElement("li");
    li.classList.toggle("selected", img.id === state.imageId);
    const thumb = document.createElement("img");
    thumb.src = `/api/nodes/${img.root_node_id}/render?thumb=1`;
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = img.name;
    li.append(thumb, name);
    li.onclick = () => selectImage(img.id);
    ul.appendChild(li);
  }
}

async function selectImage(imageId, nodeId = null) {
  if (imageId !== state.imageId) resetZoom();
  state.imageId = imageId;
  localStorage.setItem("picky:lastImage", imageId ?? "");
  if (imageId === null) {
    state.nodes = [];
    state.nodeId = null;
  } else {
    state.nodes = await api(`/api/images/${imageId}/tree`);
    const valid = state.nodes.some((n) => n.id === nodeId);
    state.nodeId = valid ? nodeId : state.nodes[state.nodes.length - 1].id;
  }
  renderSelection();
}

function renderSelection() {
  exitPreview(false);
  document.querySelectorAll("#gallery li").forEach((li, i) => {
    li.classList.toggle("selected", state.images[i]?.id === state.imageId);
  });
  const hasImage = state.imageId !== null;
  $("preview").hidden = !hasImage;
  $("drop-hint").hidden = hasImage;
  $("zoom-hud").hidden = !hasImage;
  $("delete-btn").hidden = !hasImage;
  $("export-btn").hidden = !hasImage;
  $("apply-btn").disabled = !hasImage;
  $("preview-btn").disabled = !hasImage;
  if (hasImage) {
    $("preview").src = `/api/nodes/${state.nodeId}/render?t=${Date.now()}`;
    $("export-btn").href = `/api/nodes/${state.nodeId}/render?download=1`;
  }
  if (state.effect === "blend") renderEffectControls();
  renderTree();
  updatePresetControls();
  updateClusterPlot();
}

// ---------- Zoom / pan ----------

const view = { zoom: 1, panX: 0, panY: 0 };

function applyViewTransform() {
  $("preview").style.transform =
    `translate(${view.panX}px, ${view.panY}px) scale(${view.zoom})`;
}

function resetZoom() {
  view.zoom = 1;
  view.panX = 0;
  view.panY = 0;
  applyViewTransform();
}

function initZoom() {
  const panel = $("preview-panel");
  const preview = $("preview");
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  panel.addEventListener(
    "wheel",
    (e) => {
      if (state.imageId === null) return;
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const next = Math.min(16, Math.max(1, view.zoom * factor));
      // keep the point under the cursor fixed while zooming
      const rect = panel.getBoundingClientRect();
      const cx = e.clientX - rect.left - rect.width / 2;
      const cy = e.clientY - rect.top - rect.height / 2;
      const scale = next / view.zoom;
      view.panX = cx - (cx - view.panX) * scale;
      view.panY = cy - (cy - view.panY) * scale;
      view.zoom = next;
      if (view.zoom === 1) {
        view.panX = 0;
        view.panY = 0;
      }
      applyViewTransform();
    },
    { passive: false }
  );

  preview.addEventListener("mousedown", (e) => {
    if (view.zoom === 1) return;
    e.preventDefault();
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    view.panX += e.clientX - lastX;
    view.panY += e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    applyViewTransform();
  });
  window.addEventListener("mouseup", () => (dragging = false));
  preview.addEventListener("dblclick", resetZoom);
}

// ---------- Work tree ----------

function nodeLabel(node) {
  if (!node.effect) return "Original";
  if (node.effect === "blend") return "Blend";
  const spec = state.effects.find((e) => e.name === node.effect);
  return spec ? spec.label : node.effect;
}

function nodeParamsText(node) {
  if (node.effect === "blend") {
    return `${node.params.mode} · with #${node.parent2_id}`;
  }
  return Object.entries(node.params)
    .map(([k, v]) => `${k}=${v}`)
    .join(", ");
}

// The work tree is a DAG (blend nodes have two parents), drawn like a git
// commit graph: one row per node in id order (ids are topological — parents
// always precede children), with colored lanes in a left gutter that fork at
// branches and merge into blend rows.

const RAIL = { rowH: 28, dotR: 4 };
const SVG_NS = "http://www.w3.org/2000/svg";

let lanePaletteCache = null;
function laneColor(k) {
  if (!lanePaletteCache) {
    const cs = getComputedStyle(document.documentElement);
    lanePaletteCache = ["red", "orange", "yellow", "green", "teal", "blue", "violet"]
      .map((c) => cs.getPropertyValue(`--rb-${c}`).trim());
  }
  return lanePaletteCache[k % lanePaletteCache.length];
}

function layoutGraph(nodes) {
  const childCount = new Map();
  for (const n of nodes) {
    if (n.parent_id !== null) childCount.set(n.parent_id, (childCount.get(n.parent_id) || 0) + 1);
    if (n.parent2_id !== null) childCount.set(n.parent2_id, (childCount.get(n.parent2_id) || 0) + 1);
  }
  // lanes[k] = {nodeId, remaining}: edges from nodeId down to its `remaining`
  // not-yet-rendered children pass through lane k. null = free.
  const lanes = [];
  const rows = [];
  let laneCount = 0;
  for (const n of nodes) {
    const activeAbove = [];
    lanes.forEach((s, k) => { if (s) activeAbove.push(k); });

    const parentIds = [];
    if (n.parent_id !== null) parentIds.push(n.parent_id);
    if (n.parent2_id !== null) parentIds.push(n.parent2_id);
    // a self-pair blend (parent_id === parent2_id) hits the same slot twice
    const parentLanes = parentIds.map((pid) => {
      const k = lanes.findIndex((s) => s && s.nodeId === pid);
      if (k !== -1) lanes[k].remaining--;
      return k;
    });

    // take over an exhausted parent lane (primary first), else leftmost free
    let lane = parentLanes.find((k) => k !== -1 && lanes[k].remaining === 0) ?? -1;
    if (lane === -1) {
      lane = lanes.findIndex((s) => s === null);
      if (lane === -1) lane = lanes.push(null) - 1;
    }
    for (const k of parentLanes) {
      if (k !== -1 && k !== lane && lanes[k] && lanes[k].remaining === 0) lanes[k] = null;
    }

    const kids = childCount.get(n.id) || 0;
    lanes[lane] = kids > 0 ? { nodeId: n.id, remaining: kids } : null;
    while (lanes.length && lanes[lanes.length - 1] === null) lanes.pop();
    laneCount = Math.max(laneCount, lanes.length, lane + 1);

    const passThrough = activeAbove.filter((k) => k !== lane && lanes[k]);
    const parentLinks = [];
    for (const k of parentLanes) {
      if (k === -1 || parentLinks.some((l) => l.fromLane === k)) continue;
      // fork (parent lane survives below): the peeling branch gets the child's
      // lane color; merge (lane ended here): the dying lane flows into the dot
      const colorLane = k !== lane && lanes[k] ? lane : k;
      parentLinks.push({ fromLane: k, colorLane });
    }
    rows.push({ node: n, lane, continues: kids > 0, passThrough, parentLinks });
  }
  return { rows, laneCount };
}

function svgEl(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function buildRailCell(row, laneCount, laneWidth) {
  const H = RAIL.rowH;
  const mid = H / 2;
  const laneX = (k) => laneWidth / 2 + k * laneWidth;
  const svg = svgEl("svg", {
    width: laneCount * laneWidth,
    height: H,
    class: `tree-rail fx-${row.node.effect || "original"}`,
  });
  for (const k of row.passThrough) {
    svg.appendChild(svgEl("line", {
      x1: laneX(k), y1: 0, x2: laneX(k), y2: H,
      stroke: laneColor(k), "stroke-width": 2,
    }));
  }
  for (const { fromLane, colorLane } of row.parentLinks) {
    if (fromLane === row.lane) {
      svg.appendChild(svgEl("line", {
        x1: laneX(fromLane), y1: 0, x2: laneX(fromLane), y2: mid,
        stroke: laneColor(colorLane), "stroke-width": 2,
      }));
    } else {
      const x1 = laneX(fromLane);
      const x2 = laneX(row.lane);
      svg.appendChild(svgEl("path", {
        d: `M ${x1} 0 C ${x1} ${mid}, ${x2} 0, ${x2} ${mid}`,
        stroke: laneColor(colorLane), "stroke-width": 2,
        fill: "none", "stroke-linecap": "round",
      }));
    }
  }
  if (row.continues) {
    svg.appendChild(svgEl("line", {
      x1: laneX(row.lane), y1: mid, x2: laneX(row.lane), y2: H,
      stroke: laneColor(row.lane), "stroke-width": 2,
    }));
  }
  const dot = svgEl("circle", {
    cx: laneX(row.lane), cy: mid, r: RAIL.dotR, fill: "currentColor",
  });
  if (row.node.id === state.nodeId) {
    dot.setAttribute("stroke", "#e6e9ef");
    dot.setAttribute("stroke-width", 1.5);
  }
  svg.appendChild(dot);
  return svg;
}

function buildTreeRow(row, laneCount, laneWidth) {
  const node = row.node;
  const wrap = document.createElement("div");
  wrap.className = "tree-row";
  wrap.style.height = `${RAIL.rowH}px`;
  wrap.appendChild(buildRailCell(row, laneCount, laneWidth));

  const div = document.createElement("div");
  div.className = `tree-node fx-${node.effect || "original"}`;
  div.classList.toggle("selected", node.id === state.nodeId);
  const idSpan = document.createElement("span");
  idSpan.className = "node-id";
  idSpan.textContent = `#${node.id}`;
  const label = document.createElement("span");
  label.className = "fx-label";
  label.textContent = nodeLabel(node);
  div.append(idSpan, label);
  div.title = `#${node.id} ${nodeLabel(node)}${node.params ? " · " + nodeParamsText(node) : ""}`;
  if (node.params) {
    const span = document.createElement("span");
    span.className = "params";
    span.textContent = nodeParamsText(node);
    div.appendChild(span);
  }
  if (node.parent_id !== null) {
    const edit = document.createElement("button");
    edit.className = "node-edit";
    edit.textContent = "✎";
    edit.title = "Change this effect's settings (re-renders everything below it)";
    edit.onclick = (e) => {
      e.stopPropagation();
      openEdit(node);
    };
    div.appendChild(edit);

    const del = document.createElement("button");
    del.className = "node-del";
    del.textContent = "×";
    del.title = "Delete this effect (and everything below it)";
    del.onclick = (e) => {
      e.stopPropagation();
      deleteNode(node);
    };
    div.appendChild(del);
  }
  div.onclick = () => {
    state.nodeId = node.id;
    renderSelection();
  };
  wrap.appendChild(div);
  return wrap;
}

function renderTree() {
  const container = $("tree");
  container.innerHTML = "";
  if (state.imageId === null) {
    container.textContent = "No image selected.";
    return;
  }
  const { rows, laneCount } = layoutGraph(state.nodes);
  const laneWidth = laneCount <= 8 ? 12 : Math.max(8, Math.floor(96 / laneCount));
  for (const row of rows) container.appendChild(buildTreeRow(row, laneCount, laneWidth));
}

// ---------- Effect picker ----------

// Each button gets a small graphic of what the effect does. They paint with
// currentColor, so an icon takes its hue from the .fx-* class already on the
// button — the same idiom as the work tree's rail dot.
const iconSvg = () => svgEl("svg", { viewBox: "0 0 24 24" });

// a 2D Gaussian as a monochromatic heatmap; kept smooth on purpose, a cell grid
// here would read as pixelate
function iconBlur() {
  const svg = iconSvg();
  const grad = svgEl("radialGradient", { id: "fx-gauss" });
  for (const [offset, opacity] of [[0, 1], [25, 0.78], [50, 0.37], [75, 0.11], [100, 0]]) {
    grad.appendChild(svgEl("stop", {
      offset: `${offset}%`, "stop-color": "currentColor", "stop-opacity": opacity,
    }));
  }
  const defs = svgEl("defs", {});
  defs.appendChild(grad);
  svg.append(defs, svgEl("rect", { x: 0, y: 0, width: 24, height: 24, fill: "url(#fx-gauss)" }));
  return svg;
}

// roughly parallel squiggles — the offsets keep them from looking ruled
function iconEdges() {
  const svg = iconSvg();
  [0, -1.5, 0.8].forEach((dx, i) => {
    const y = 5.5 + i * 6.5;
    svg.appendChild(svgEl("path", {
      d: `M ${1.5 + dx} ${y} q 3.2 -4 6.4 0 t 6.4 0 t 6.4 0`,
      fill: "none", stroke: "currentColor", "stroke-width": 1.6, "stroke-linecap": "round",
    }));
  });
  return svg;
}

// bars stepping in tone: the flat color bands posterizing produces
function iconPosterize() {
  const svg = iconSvg();
  [1, 0.75, 0.5, 0.3, 0.15].forEach((opacity, i) => {
    svg.appendChild(svgEl("rect", {
      x: 2 + i * 4.2, y: 3, width: 3.4, height: 18, rx: 0.6,
      fill: "currentColor", "fill-opacity": opacity,
    }));
  });
  return svg;
}

// a 4x4 grid of blocks, toned like a heavily downsampled image
function iconPixelate() {
  const svg = iconSvg();
  const tones = [
    0.95, 0.7, 0.45, 0.2,
    0.75, 0.9, 0.3, 0.5,
    0.4, 0.55, 0.85, 0.65,
    0.15, 0.35, 0.6, 1,
  ];
  tones.forEach((opacity, i) => {
    svg.appendChild(svgEl("rect", {
      x: 1.5 + (i % 4) * 5.4, y: 1.5 + Math.floor(i / 4) * 5.4,
      width: 5, height: 5, fill: "currentColor", "fill-opacity": opacity,
    }));
  });
  return svg;
}

// two nodes merging into one, drawn with the same curve the DAG rail uses
function iconBlend() {
  const svg = iconSvg();
  for (const x of [5, 19]) {
    svg.appendChild(svgEl("path", {
      d: `M ${x} 5 C ${x} 19, 12 5, 12 19`,
      fill: "none", stroke: "currentColor", "stroke-width": 1.8, "stroke-linecap": "round",
    }));
  }
  for (const [cx, cy] of [[5, 5], [19, 5], [12, 19]]) {
    svg.appendChild(svgEl("circle", { cx, cy, r: 2.6, fill: "currentColor" }));
  }
  return svg;
}

// The picker groups posterize + dither behind one button — same idea (reduce to
// N colors), different palette algorithm — so choosing between them is a Method
// dropdown rather than two icons. Presentation only: the registry, the stored
// effect names, and the tree still treat them as two separate effects.
const EFFECT_BUTTONS = [
  { key: "blur", label: "Gaussian blur", icon: iconBlur, effects: ["blur"] },
  { key: "edges", label: "Sobel edges", icon: iconEdges, effects: ["edges"] },
  {
    key: "posterize",
    label: "Posterize",
    icon: iconPosterize,
    effects: ["posterize", "dither"],
    methods: { posterize: "k-means clustering", dither: "Floyd–Steinberg dither" },
  },
  { key: "pixelate", label: "Pixelate", icon: iconPixelate, effects: ["pixelate"] },
  { key: "blend", label: "Blend with…", icon: iconBlend, effects: ["blend"] },
];

const groupFor = (name) => EFFECT_BUTTONS.find((g) => g.effects.includes(name));

// which effect a grouped button was last left on, so leaving posterize on
// Floyd–Steinberg and coming back does not silently reset to k-means
const lastMethod = {};

function buildEffectButtons() {
  const box = $("effect-buttons");
  for (const group of EFFECT_BUTTONS) {
    const btn = document.createElement("button");
    btn.className = `fx-btn fx-${group.key}`;
    btn.dataset.group = group.key;
    // an inline <svg> has no alt, so the name of the effect goes on the button
    btn.title = group.label;
    btn.setAttribute("aria-label", group.label);
    const svg = group.icon();
    svg.setAttribute("aria-hidden", "true");
    btn.appendChild(svg);
    btn.onclick = () => setEffect(lastMethod[group.key] || group.effects[0]);
    box.appendChild(btn);
  }
}

// The selected effect lives in state, not in the DOM. Single write path: every
// change rebuilds the params and re-renders any running preview.
function setEffect(name) {
  const group = groupFor(name);
  state.effect = name;
  lastMethod[group.key] = name;
  document.querySelectorAll("#effect-buttons .fx-btn").forEach((btn) => {
    const on = btn.dataset.group === group.key;
    btn.classList.toggle("selected", on);
    btn.setAttribute("aria-pressed", on);
  });
  renderEffectControls();
  if (preview.active) schedulePreview();
}

// A grouped button picks its effect here rather than in buildParamControls(),
// which the edit modal shares: PATCH edits params, not the effect, so the modal
// must never grow this control. No data-param, so readParams() walks past it.
function buildMethodRow(group) {
  const row = document.createElement("div");
  row.className = "param-row";
  const label = document.createElement("label");
  const nameSpan = document.createElement("span");
  nameSpan.textContent = "Method";
  label.appendChild(nameSpan);
  const sel = document.createElement("select");
  sel.className = "fx-method";
  for (const [name, text] of Object.entries(group.methods)) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = text;
    sel.appendChild(opt);
  }
  sel.value = state.effect;
  sel.onchange = () => setEffect(sel.value);
  row.append(label, sel);
  return row;
}

// ---------- Effects ----------

// The Apply panel and the edit modal both build their controls here and read
// them back through readParams(), so the two can never disagree about ranges,
// defaults, or how a value is coerced. `values` seeds the controls — empty for
// Apply (fresh defaults), an existing node's params when editing.
function buildParamControls(container, effectName, values = {}) {
  const effect = state.effects.find((e) => e.name === effectName);
  container.innerHTML = "";
  for (const p of effect.params) {
    const current = values[p.name] ?? p.default;
    const row = document.createElement("div");
    row.className = "param-row";
    const label = document.createElement("label");
    const nameSpan = document.createElement("span");
    nameSpan.textContent = p.label;
    label.appendChild(nameSpan);
    if (p.type === "choice") {
      const sel = document.createElement("select");
      sel.dataset.param = p.name;
      for (const optName of p.options) {
        const opt = document.createElement("option");
        opt.value = optName;
        opt.textContent = optName;
        sel.appendChild(opt);
      }
      sel.value = p.options.includes(current) ? current : p.default;
      row.append(label, sel);
    } else {
      const valSpan = document.createElement("span");
      valSpan.textContent = current;
      label.appendChild(valSpan);
      const input = document.createElement("input");
      input.type = "range";
      input.min = p.min;
      input.max = p.max;
      if (p.step !== undefined) input.step = p.step;
      input.value = current;
      input.dataset.param = p.name;
      input.oninput = () => (valSpan.textContent = input.value);
      row.append(label, input);
    }
    container.appendChild(row);
  }
  return effect;
}

// Blend's second input is not a registry param (it names a node), so it gets
// appended separately. `maxId` mirrors the server's cycle guard when editing:
// ids are topological, so only a smaller id is guaranteed not to be downstream.
// Returns false when there is nothing to blend with.
function appendBlendTarget(container, { text, excludeId, selected = null, maxId = null }) {
  const others = state.nodes.filter(
    (n) => n.id !== excludeId && (maxId === null || n.id < maxId)
  );
  const row = document.createElement("div");
  row.className = "param-row";
  const label = document.createElement("label");
  const nameSpan = document.createElement("span");
  nameSpan.textContent = text;
  label.appendChild(nameSpan);
  const sel = document.createElement("select");
  sel.className = "blend-with";
  for (const n of others) {
    const opt = document.createElement("option");
    opt.value = n.id;
    opt.textContent = `#${n.id} ${nodeLabel(n)}${n.params ? " · " + nodeParamsText(n) : ""}`;
    sel.appendChild(opt);
  }
  if (selected !== null && others.some((n) => n.id === selected)) sel.value = selected;
  row.append(label, sel);
  container.appendChild(row);

  // the weight param only affects the "average" mode
  const modeSel = container.querySelector('[data-param="mode"]');
  const weightRow = container.querySelector('[data-param="weight"]')?.closest(".param-row");
  if (modeSel && weightRow) {
    const syncWeight = () => (weightRow.hidden = modeSel.value !== "average");
    modeSel.addEventListener("change", syncWeight);
    syncWeight();
  }
  return others.length > 0;
}

function renderEffectControls() {
  const box = $("effect-params");
  const effect = buildParamControls(box, state.effect);
  let canApply = state.imageId !== null;
  if (effect.name === "blend") {
    const hasTarget = appendBlendTarget(box, {
      text: "Blend selected node with",
      excludeId: state.nodeId,
    });
    canApply = canApply && hasTarget;
  }
  const group = groupFor(state.effect);
  if (group.methods) box.prepend(buildMethodRow(group));
  $("apply-btn").disabled = !canApply;
  $("preview-btn").disabled = !canApply;
}

function readParams(container, effectName) {
  const params = {};
  container.querySelectorAll("[data-param]").forEach((el) => {
    params[el.dataset.param] = el.type === "range" ? Number(el.value) : el.value;
  });
  let parent2_id = null;
  if (effectName === "blend") {
    const withSel = container.querySelector(".blend-with");
    parent2_id = withSel && withSel.value ? Number(withSel.value) : null;
  }
  return { effect: effectName, params, parent2_id };
}

function readEffectForm() {
  return readParams($("effect-params"), state.effect);
}

async function applyEffect() {
  const btn = $("apply-btn");
  const { effect, params, parent2_id } = readEffectForm();
  if (effect === "blend" && parent2_id === null) return;
  exitPreview(false);
  const body = { parent_id: state.nodeId, effect, params };
  if (effect === "blend") body.parent2_id = parent2_id;
  btn.disabled = true;
  btn.classList.add("busy");
  btn.textContent = "Applying…";
  try {
    const node = await api(`/api/images/${state.imageId}/nodes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    await selectImage(state.imageId, node.id);
  } catch (err) {
    alert(`Failed to apply effect: ${err.message}`);
  } finally {
    btn.disabled = state.imageId === null;
    btn.classList.remove("busy");
    btn.textContent = "Apply";
  }
}

// ---------- Effect preview (uncommitted) ----------

// `source` is whatever is currently driving the preview: the Apply panel, or the
// edit modal. It returns the request to render, or null when the form can't
// produce one yet. Keeping it a single slot means one debounce, one stale-response
// counter, and one owner of the blob URL no matter who is previewing.
const preview = { active: false, url: null, seq: 0, timer: null, source: null };

function togglePreview() {
  if (preview.active) {
    exitPreview(true);
  } else {
    preview.active = true;
    preview.source = applyPreviewRequest;
    $("preview-btn").classList.add("active");
    refreshPreview();
  }
}

// Apply composes the effect on top of the selected node, which is exactly what
// POST /api/nodes/{id}/preview does.
function applyPreviewRequest() {
  if (state.nodeId === null) return null;
  const { effect, params, parent2_id } = readEffectForm();
  if (effect === "blend" && parent2_id === null) return null;
  const body = { effect, params };
  if (parent2_id !== null) body.parent2_id = parent2_id;
  return { nodeId: state.nodeId, body, busyEl: $("preview-btn") };
}

function exitPreview(restoreSrc) {
  clearTimeout(preview.timer);
  preview.timer = null;
  preview.seq++; // invalidate in-flight fetches
  preview.active = false;
  preview.source = null;
  $("preview-btn").classList.remove("active");
  if (restoreSrc && state.nodeId !== null) {
    $("preview").src = `/api/nodes/${state.nodeId}/render?t=${Date.now()}`;
  }
  if (preview.url) {
    URL.revokeObjectURL(preview.url);
    preview.url = null;
  }
}

function schedulePreview() {
  clearTimeout(preview.timer);
  preview.timer = setTimeout(refreshPreview, 250);
}

async function refreshPreview() {
  if (!preview.active || !preview.source) return;
  const req = preview.source();
  if (!req) return;
  const seq = ++preview.seq;
  req.busyEl?.classList.add("busy");
  try {
    const res = await fetch(`/api/nodes/${req.nodeId}/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    if (seq !== preview.seq || !preview.active) return; // stale response or exited
    const url = URL.createObjectURL(blob);
    $("preview").src = url;
    if (preview.url) URL.revokeObjectURL(preview.url);
    preview.url = url;
  } catch (err) {
    console.warn("preview failed:", err); // keep the last frame; no alert mid-drag
  } finally {
    req.busyEl?.classList.remove("busy");
  }
}

// ---------- Presets (saved effect chains, reusable across images) ----------

let presetBusy = false;

async function refreshPresets() {
  state.presets = await api("/api/presets");
  renderPresets();
}

function renderPresets() {
  const list = $("preset-list");
  list.innerHTML = "";
  if (!state.presets.length) {
    const empty = document.createElement("li");
    empty.className = "hint";
    empty.textContent = "Pick a node, then save its chain to reuse it here.";
    list.appendChild(empty);
    return;
  }
  for (const preset of state.presets) {
    const li = document.createElement("li");
    li.className = "preset-row";
    li.title = `Apply to the selected node: ${preset.summary}`;

    const text = document.createElement("div");
    text.className = "preset-text";
    const name = document.createElement("div");
    name.className = "preset-name";
    name.textContent = preset.name;
    const summary = document.createElement("div");
    summary.className = "preset-summary";
    summary.textContent = preset.summary;
    text.append(name, summary);

    const del = document.createElement("button");
    del.className = "preset-del";
    del.textContent = "×";
    del.title = "Delete preset";
    del.onclick = (e) => {
      e.stopPropagation();
      deletePreset(preset);
    };

    li.append(text, del);
    li.onclick = () => applyPreset(preset, li);
    list.appendChild(li);
  }
  updatePresetControls();
}

function updatePresetControls() {
  const node = state.nodes.find((n) => n.id === state.nodeId);
  const canSave = state.imageId !== null && !!node && node.parent_id !== null;
  const saveBtn = $("save-preset-btn");
  saveBtn.disabled = !canSave || presetBusy;
  saveBtn.title = canSave
    ? "Save the edits leading to the selected node"
    : "Select an edited node — the original has no chain to save";
  document.querySelectorAll("#preset-list .preset-row").forEach((li) => {
    li.classList.toggle("disabled", state.imageId === null || presetBusy);
  });
}

async function savePreset() {
  const node = state.nodes.find((n) => n.id === state.nodeId);
  if (!node || node.parent_id === null) return;
  const name = prompt("Name this preset:", "");
  if (name === null || !name.trim()) return;
  try {
    await api("/api/presets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name.trim(), node_id: state.nodeId }),
    });
    await refreshPresets();
  } catch (err) {
    alert(`Could not save preset: ${err.message}`);
  }
}

async function applyPreset(preset, row) {
  if (presetBusy || state.imageId === null) return;
  exitPreview(false);
  presetBusy = true;
  row.classList.add("busy");
  updatePresetControls();
  try {
    const res = await api(`/api/nodes/${state.nodeId}/apply-preset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preset_id: preset.id }),
    });
    // selectImage refetches the tree, so all of the preset's new nodes show up
    await selectImage(state.imageId, res.terminal_node_id);
  } catch (err) {
    alert(`Could not apply preset: ${err.message}`);
  } finally {
    presetBusy = false;
    row.classList.remove("busy");
    updatePresetControls();
  }
}

async function deletePreset(preset) {
  if (!confirm(`Delete the preset “${preset.name}”?`)) return;
  try {
    await api(`/api/presets/${preset.id}`, { method: "DELETE" });
    await refreshPresets();
  } catch (err) {
    alert(`Could not delete preset: ${err.message}`);
  }
}

// ---------- Library stats ----------

function formatBytes(n) {
  if (n >= 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${n} B`;
}

function statRow(container, label, value, muted = false) {
  const row = document.createElement("div");
  row.className = muted ? "stat-row muted" : "stat-row";
  const l = document.createElement("span");
  l.textContent = label;
  const v = document.createElement("span");
  v.textContent = value;
  row.append(l, v);
  container.appendChild(row);
}

function statHeading(container, text) {
  const h = document.createElement("div");
  h.className = "stat-heading";
  h.textContent = text;
  container.appendChild(h);
}

async function openStats() {
  const body = $("stats-body");
  body.textContent = "Loading…";
  $("stats-modal").showModal();
  let s;
  try {
    s = await api("/api/stats"); // a snapshot — refetched on every open
  } catch (err) {
    body.textContent = `Could not load stats: ${err.message}`;
    return;
  }
  body.innerHTML = "";
  statHeading(body, "Contents");
  statRow(body, "Images", s.images);
  statRow(body, "Nodes", `${s.nodes} (${s.edits} effects + ${s.images} originals)`);
  statRow(body, "Presets", s.presets);
  for (const e of s.by_effect) {
    // nodeLabel() wants a node; here we only have the effect name
    const spec = state.effects.find((x) => x.name === e.effect);
    const label = e.effect === "blend" ? "Blend" : spec ? spec.label : e.effect;
    statRow(body, label, e.count, true);
  }

  const st = s.storage;
  statHeading(body, "On disk");
  statRow(body, "Database", formatBytes(st.database.bytes));
  statRow(body, "Originals", `${formatBytes(st.originals.bytes)} · ${st.originals.files} files`);
  const cacheBytes = st.renders.bytes + st.thumbs.bytes + st.clusters.bytes;
  statRow(body, "Render cache", formatBytes(cacheBytes));
  statRow(body, "renders", `${formatBytes(st.renders.bytes)} · ${st.renders.files} files`, true);
  statRow(body, "thumbnails", `${formatBytes(st.thumbs.bytes)} · ${st.thumbs.files} files`, true);
  statRow(body, "cluster data", `${formatBytes(st.clusters.bytes)} · ${st.clusters.files} files`, true);
  statRow(body, "Total", formatBytes(st.database.bytes + st.originals.bytes + cacheBytes));

  const note = document.createElement("div");
  note.className = "hint";
  note.textContent =
    "The render cache rebuilds itself from the originals and the work tree — only the database and originals are irreplaceable.";
  body.appendChild(note);
}

// ---------- Editing an existing node's settings ----------

// Unlike Apply, this changes a node in place: it keeps its id and its children,
// and every render derived from it is thrown away and rebuilt.
const edit = { node: null };

function descendantsOf(nodeId) {
  const found = new Set([nodeId]);
  // state.nodes is in id order and parents always precede children, so one
  // forward pass reaches the whole closure
  for (const n of state.nodes) {
    if (found.has(n.parent_id) || found.has(n.parent2_id)) found.add(n.id);
  }
  found.delete(nodeId);
  return [...found];
}

function editPreviewRequest() {
  const node = edit.node;
  if (!node) return null;
  const { effect, params, parent2_id } = readParams($("edit-params"), node.effect);
  if (effect === "blend" && parent2_id === null) return null;
  const body = { effect, params };
  if (parent2_id !== null) body.parent2_id = parent2_id;
  // previewing an edit to node N means re-applying N's effect to N's *input*
  return { nodeId: node.parent_id, body, busyEl: $("edit-status") };
}

function openEdit(node) {
  // select the node first: renderSelection() is the choke point that exits any
  // running preview, so it has to happen before the modal starts its own
  if (state.nodeId !== node.id) {
    state.nodeId = node.id;
    renderSelection();
  } else {
    exitPreview(false);
  }
  edit.node = node;
  $("edit-title").textContent = `#${node.id} ${nodeLabel(node)}`;
  buildParamControls($("edit-params"), node.effect, node.params || {});
  if (node.effect === "blend") {
    appendBlendTarget($("edit-params"), {
      text: "Blend with",
      excludeId: node.id,
      selected: node.parent2_id,
      maxId: node.id, // matches the server's cycle guard
    });
  }
  const downstream = descendantsOf(node.id).length;
  $("edit-warning").textContent = downstream
    ? `${downstream} downstream effect${downstream === 1 ? "" : "s"} will be re-rendered.`
    : "";
  $("edit-modal").showModal();
  preview.active = true;
  preview.source = editPreviewRequest;
  refreshPreview();
}

function closeEdit(restoreSrc) {
  edit.node = null;
  exitPreview(restoreSrc);
  $("edit-modal").close();
}

async function saveEdit() {
  const node = edit.node;
  if (!node) return;
  const { params, parent2_id } = readParams($("edit-params"), node.effect);
  if (node.effect === "blend" && parent2_id === null) return;
  const btn = $("edit-save-btn");
  btn.disabled = true;
  btn.classList.add("busy");
  btn.textContent = "Saving…";
  try {
    await api(`/api/nodes/${node.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params, parent2_id }),
    });
    // the node id did not change, so the cluster plot would otherwise keep
    // showing the clusters it cached for the old params
    cluster.nodeId = null;
    closeEdit(false);
    await selectImage(state.imageId, node.id);
  } catch (err) {
    alert(`Could not save: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.classList.remove("busy");
    btn.textContent = "Save";
  }
}

// ---------- Cluster plot (3D RGB scatter for posterize nodes) ----------

const cluster = {
  nodeId: null,
  points: [],
  centroids: [],
  yaw: 0.8,
  pitch: -0.45,
  spin: true,
  raf: null,
};

async function updateClusterPlot() {
  const node = state.nodes.find((n) => n.id === state.nodeId);
  const show = !!node && node.effect === "posterize";
  $("cluster-section").hidden = !show;
  if (!show) {
    stopClusterLoop();
    cluster.nodeId = null;
    return;
  }
  if (cluster.nodeId !== node.id) {
    cluster.nodeId = node.id;
    try {
      const data = await api(`/api/nodes/${node.id}/clusters`);
      if (cluster.nodeId !== node.id) return; // stale response
      cluster.points = data.points;
      cluster.centroids = data.centroids;
    } catch (err) {
      cluster.points = [];
      cluster.centroids = [];
    }
  }
  startClusterLoop();
}

function startClusterLoop() {
  if (cluster.raf) return;
  const tick = () => {
    if (cluster.spin) cluster.yaw += 0.004;
    drawClusterPlot();
    cluster.raf = requestAnimationFrame(tick);
  };
  cluster.raf = requestAnimationFrame(tick);
}

function stopClusterLoop() {
  if (cluster.raf) cancelAnimationFrame(cluster.raf);
  cluster.raf = null;
}

function drawClusterPlot() {
  const canvas = $("cluster-canvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  // half-diagonal of the RGB cube is 127.5 * sqrt(3) ~ 221
  const scale = (Math.min(w, h) / 2 - 10) / 221;
  const cy = Math.cos(cluster.yaw);
  const sy = Math.sin(cluster.yaw);
  const cp = Math.cos(cluster.pitch);
  const sp = Math.sin(cluster.pitch);
  const project = (r, g, b) => {
    const x = r - 127.5;
    const y = g - 127.5;
    const z = b - 127.5;
    const x1 = x * cy + z * sy;
    const z1 = z * cy - x * sy;
    const y1 = y * cp - z1 * sp;
    const z2 = y * sp + z1 * cp;
    return [w / 2 + x1 * scale, h / 2 - y1 * scale, z2];
  };

  ctx.strokeStyle = "rgba(139, 147, 163, 0.35)";
  ctx.lineWidth = 1;
  for (const a of [0, 255]) {
    for (const b of [0, 255]) {
      for (const [p, q] of [
        [[0, a, b], [255, a, b]],
        [[a, 0, b], [a, 255, b]],
        [[a, b, 0], [a, b, 255]],
      ]) {
        const [x1, y1] = project(...p);
        const [x2, y2] = project(...q);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }
  }

  const pts = cluster.points
    .map(([r, g, b, label]) => {
      const [x, y, z] = project(r, g, b);
      return { x, y, z, label };
    })
    .sort((a, b) => a.z - b.z);
  for (const p of pts) {
    const c = cluster.centroids[p.label];
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    ctx.fillRect(p.x - 1.25, p.y - 1.25, 2.5, 2.5);
  }

  for (const c of cluster.centroids) {
    const [x, y] = project(c[0], c[1], c[2]);
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    ctx.strokeStyle = "#e6e9ef";
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

function initClusterPlot() {
  const canvas = $("cluster-canvas");
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  canvas.addEventListener("mousedown", (e) => {
    e.preventDefault();
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    cluster.yaw += (e.clientX - lastX) * 0.01;
    cluster.pitch += (e.clientY - lastY) * 0.01;
    cluster.pitch = Math.max(-1.5, Math.min(1.5, cluster.pitch));
    lastX = e.clientX;
    lastY = e.clientY;
  });
  window.addEventListener("mouseup", () => (dragging = false));
  canvas.addEventListener("dblclick", () => (cluster.spin = !cluster.spin));
}

// ---------- Upload / delete ----------

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  try {
    const image = await api("/api/images", { method: "POST", body: form });
    await refreshGallery();
    await selectImage(image.id, image.root_node_id);
  } catch (err) {
    alert(`Upload failed: ${err.message}`);
  }
}

async function deleteNode(node) {
  const hasChildren = state.nodes.some(
    (n) => n.parent_id === node.id || n.parent2_id === node.id
  );
  const msg = hasChildren
    ? "Delete this effect and all effects branching from it?"
    : "Delete this effect?";
  if (!confirm(msg)) return;
  try {
    const res = await api(`/api/nodes/${node.id}`, { method: "DELETE" });
    const keep = res.deleted.includes(state.nodeId) ? res.parent_id : state.nodeId;
    await selectImage(state.imageId, keep);
  } catch (err) {
    alert(`Delete failed: ${err.message}`);
  }
}

async function deleteImage() {
  if (!confirm("Delete this image and its entire work tree?")) return;
  await api(`/api/images/${state.imageId}`, { method: "DELETE" });
  await refreshGallery();
  await selectImage(state.images.length ? state.images[0].id : null);
}

// ---------- Init ----------

async function init() {
  state.effects = await api("/api/effects");
  buildEffectButtons();
  setEffect(EFFECT_BUTTONS[0].effects[0]);

  initZoom();
  initClusterPlot();
  $("apply-btn").onclick = applyEffect;
  $("preview-btn").onclick = togglePreview;
  $("effect-params").addEventListener("input", () => {
    if (preview.active) schedulePreview();
  });
  $("delete-btn").onclick = deleteImage;
  $("save-preset-btn").onclick = savePreset;

  $("stats-btn").onclick = openStats;
  $("stats-close-btn").onclick = () => $("stats-modal").close();
  $("edit-cancel-btn").onclick = () => closeEdit(true);
  $("edit-save-btn").onclick = saveEdit;
  $("edit-params").addEventListener("input", schedulePreview);
  // Esc closes a <dialog> without going through our buttons, so clean up here too
  $("edit-modal").addEventListener("close", () => {
    if (edit.node) closeEdit(true);
  });
  $("file-input").onchange = (e) => {
    if (e.target.files[0]) uploadFile(e.target.files[0]);
    e.target.value = "";
  };

  document.body.addEventListener("dragover", (e) => {
    e.preventDefault();
    document.body.classList.add("dragging");
  });
  document.body.addEventListener("dragleave", () => {
    document.body.classList.remove("dragging");
  });
  document.body.addEventListener("drop", (e) => {
    e.preventDefault();
    document.body.classList.remove("dragging");
    const file = [...e.dataTransfer.files].find(
      (f) => f.type === "image/jpeg" || /\.jpe?g$/i.test(f.name)
    );
    if (file) uploadFile(file);
  });

  await refreshGallery();
  await refreshPresets();
  // deep links (?image=2&node=28) win over the last-viewed image
  const q = new URLSearchParams(location.search);
  const qImage = Number(q.get("image"));
  const last = Number(localStorage.getItem("picky:lastImage"));
  const target =
    state.images.find((i) => i.id === qImage) ||
    state.images.find((i) => i.id === last) ||
    state.images[0];
  if (target) await selectImage(target.id, Number(q.get("node")) || null);
}

init();
