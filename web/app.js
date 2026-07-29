const state = {
  effects: [],
  effect: null,    // name of the effect the Apply panel is set to
  images: [],
  imageId: null,
  nodes: [],       // flat node list for the selected image
  nodeId: null,    // selected node
  presets: [],     // saved effect chains, reusable across images
  masks: [],       // saved masks — image-scoped, so refetched per image
  // The Apply panel's current selection. It lives here rather than in the DOM
  // because the effect controls are torn down and rebuilt on every effect
  // switch, which used to take the pick with them. nodeId/imageId record what
  // it was picked against, so pruneSelection() can tell when it goes stale.
  selection: { value: null, nodeId: null, imageId: null },
  // Preview runs by default; the checkbox is how you get back to the original.
  livePreview: true,
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
  if (imageId !== state.imageId) {
    resetZoom();
    clearOverlayCache(); // overlays are image-scoped, and so are their mask ids
  }
  state.imageId = imageId;
  localStorage.setItem("picky:lastImage", imageId ?? "");
  if (imageId === null) {
    state.nodes = [];
    state.nodeId = null;
    state.masks = [];
  } else {
    // masks are image-scoped, so unlike presets they are fetched on every image
    // change; after that only a save/rename/delete refreshes them
    [state.nodes, state.masks] = await Promise.all([
      api(`/api/images/${imageId}/tree`),
      api(`/api/images/${imageId}/masks`),
    ]);
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
  $("preview-wrap").hidden = !hasImage;
  $("drop-hint").hidden = hasImage;
  $("zoom-hud").hidden = !hasImage;
  $("delete-btn").hidden = !hasImage;
  $("export-btn").hidden = !hasImage;
  $("apply-btn").disabled = !hasImage;
  if (hasImage) {
    $("preview").src = `/api/nodes/${state.nodeId}/render?t=${Date.now()}`;
    $("export-btn").href = `/api/nodes/${state.nodeId}/render?download=1`;
  }
  // the Apply panel describes an operation on the *selected* node, so it is
  // rebuilt whenever the selection moves: blend's target list, and the curve
  // editor's histogram, both read the node we are now pointing at. Selection
  // first — the effect controls no longer own it, and it reads as step 1.
  renderSelectControls();
  renderEffectControls();
  renderTree();
  updatePresetControls();
  updateClusterPlot();
  // and last, re-enter the preview this function's exitPreview() just left. The
  // node's own render is already painted above, so the unedited pixels show
  // immediately and the effect lands on top when the debounce fires.
  if (state.livePreview) enterApplyPreview();
}

// ---------- Zoom / pan ----------

const view = { zoom: 1, panX: 0, panY: 0 };

function applyViewTransform() {
  // the wrapper, not the img, so the mask overlay zooms and pans in lockstep
  $("preview-wrap").style.transform =
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
  const wrap = $("preview-wrap");
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  let downX = 0;
  let downY = 0;

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

  wrap.addEventListener("mousedown", (e) => {
    downX = e.clientX; // remembered even unzoomed, to tell a pick from a pan
    downY = e.clientY;
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
  wrap.addEventListener("dblclick", resetZoom);

  // Click-to-segment picking. The img's bounding rect is post-transform, so
  // mapping client -> full-res pixel needs no pan/zoom inversion (the same
  // reasoning as the curve editor's atEvent); a drag beyond a few px is a
  // pan that happened to end on the image, not a pick.
  wrap.addEventListener("click", (e) => {
    if (!selPicker.armed) return;
    if (Math.hypot(e.clientX - downX, e.clientY - downY) > 3) return;
    const img = $("preview");
    const rect = img.getBoundingClientRect();
    if (
      e.clientX < rect.left || e.clientX >= rect.right ||
      e.clientY < rect.top || e.clientY >= rect.bottom
    ) {
      return; // letterbox area around the image
    }
    const x = Math.round(((e.clientX - rect.left) / rect.width) * img.naturalWidth);
    const y = Math.round(((e.clientY - rect.top) / rect.height) * img.naturalHeight);
    selPicker.armed.pick(
      Math.min(x, img.naturalWidth - 1),
      Math.min(y, img.naturalHeight - 1)
    );
  });
}

// ---------- Work tree ----------

function nodeLabel(node) {
  if (!node.effect) return "Original";
  if (node.effect === "blend") return "Blend";
  const spec = state.effects.find((e) => e.name === node.effect);
  return spec ? spec.label : node.effect;
}

function nodeParamsText(node) {
  let text;
  if (node.effect === "blend") {
    text = `${node.params.mode} · with #${node.parent2_id}`;
  } else if (node.effect === "curves") {
    // spelling out every control point would swamp the row and its tooltip
    text = `${node.params.points.length} points`;
  } else {
    text = Object.entries(node.params)
      .map(([k, v]) => `${k}=${v}`)
      .join(", ");
  }
  if (node.selection) {
    text += node.selection.invert ? " · masked (inverted)" : " · masked";
  }
  return text;
}

// The work tree is a DAG (blend nodes have two parents), drawn like a git
// commit graph laid on its side: one *column* per node in id order (ids are
// topological — parents always precede children), flowing left to right, with
// colored lanes in a gutter above the chips that fork at branches and merge
// into blend columns.
//
// layoutGraph() below is unaffected by that orientation — it is a lane packer
// whose output ("this step sits in lane k; these lanes pass it by; these
// parents link in") names no axis. Only buildRailCell() and the DOM know which
// way the graph runs.

const RAIL = { colW: 108, dotR: 4 };
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

// The flow axis is x (0 → mid → colW) and the cross axis is y (one lane per
// track). This is the vertical rail's geometry with the two swapped.
function buildRailCell(row, laneCount, laneWidth) {
  const W = RAIL.colW;
  const mid = W / 2;
  const laneY = (k) => laneWidth / 2 + k * laneWidth;
  // a linear chain occupies one lane; floor the height so it is a rail and not
  // a sliver, and so every column in the strip lines up
  const H = Math.max(laneCount, 2) * laneWidth;
  const svg = svgEl("svg", {
    width: W,
    height: H,
    class: `tree-rail fx-${row.node.effect || "original"}`,
  });
  for (const k of row.passThrough) {
    svg.appendChild(svgEl("line", {
      x1: 0, y1: laneY(k), x2: W, y2: laneY(k),
      stroke: laneColor(k), "stroke-width": 2,
    }));
  }
  for (const { fromLane, colorLane } of row.parentLinks) {
    if (fromLane === row.lane) {
      svg.appendChild(svgEl("line", {
        x1: 0, y1: laneY(fromLane), x2: mid, y2: laneY(fromLane),
        stroke: laneColor(colorLane), "stroke-width": 2,
      }));
    } else {
      const y1 = laneY(fromLane);
      const y2 = laneY(row.lane);
      svg.appendChild(svgEl("path", {
        d: `M 0 ${y1} C ${mid} ${y1}, 0 ${y2}, ${mid} ${y2}`,
        stroke: laneColor(colorLane), "stroke-width": 2,
        fill: "none", "stroke-linecap": "round",
      }));
    }
  }
  if (row.continues) {
    svg.appendChild(svgEl("line", {
      x1: mid, y1: laneY(row.lane), x2: W, y2: laneY(row.lane),
      stroke: laneColor(row.lane), "stroke-width": 2,
    }));
  }
  const dot = svgEl("circle", {
    cx: mid, cy: laneY(row.lane), r: RAIL.dotR, fill: "currentColor",
  });
  if (row.node.id === state.nodeId) {
    dot.setAttribute("stroke", "#e6e9ef");
    dot.setAttribute("stroke-width", 1.5);
  }
  svg.appendChild(dot);
  return svg;
}

function buildTreeCol(row, laneCount, laneWidth) {
  const node = row.node;
  const wrap = document.createElement("div");
  wrap.className = "tree-col";
  wrap.style.width = `${RAIL.colW}px`;
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
  // the badge rides on the id line, not the label: the label ellipsises at this
  // width, and an ellipsised badge is an invisible one
  if (node.selection) {
    const badge = document.createElement("span");
    badge.className = "sel-badge";
    badge.textContent = "◎";
    idSpan.append(" ", badge);
  }
  div.append(idSpan, label);
  // a column is too narrow for the params; the tooltip carries them instead
  div.title = `#${node.id} ${nodeLabel(node)}${node.params ? " · " + nodeParamsText(node) : ""}`;
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
  const laneWidth = laneCount <= 4 ? 14 : Math.max(8, Math.floor(56 / laneCount));
  let selected = null;
  for (const row of rows) {
    const col = buildTreeCol(row, laneCount, laneWidth);
    if (row.node.id === state.nodeId) selected = col;
    container.appendChild(col);
  }
  // the strip grows to the right as work accumulates, so the newest node — and
  // the one Apply just created — would otherwise land off-screen
  selected?.scrollIntoView({ inline: "nearest", block: "nearest" });
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

// an S-curve in a box: the editor's own grid, in miniature. No gradient — the
// one in iconBlur uses a hardcoded id, so a second would have to invent another
function iconCurves() {
  const svg = iconSvg();
  svg.appendChild(svgEl("rect", {
    x: 2.5, y: 2.5, width: 19, height: 19, rx: 2,
    fill: "none", stroke: "currentColor", "stroke-width": 1.3, "stroke-opacity": 0.5,
  }));
  svg.appendChild(svgEl("path", {
    d: "M 4 20 C 9 20, 8 8, 12 8 S 15 4, 20 4",
    fill: "none", stroke: "currentColor", "stroke-width": 1.8, "stroke-linecap": "round",
  }));
  svg.appendChild(svgEl("circle", { cx: 12, cy: 8, r: 1.9, fill: "currentColor" }));
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

// Some buttons stand for two effects that are one idea with two algorithms —
// posterize/dither (reduce to N colors), curves/gamma (reshape tone) — so
// choosing between them is a Method dropdown rather than two icons.
// Presentation only: the registry, the stored effect names, and the tree still
// treat each pair as two separate effects.
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
  {
    key: "curves",
    label: "Tone curve",
    icon: iconCurves,
    effects: ["curves", "gamma"],
    methods: { curves: "Multi-point curve", gamma: "Gamma correction" },
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

// ---------- Tone curve editor ----------

// Line-for-line mirror of `_curve_lut` in server/effects.py: Fritsch–Carlson
// monotone cubic Hermite interpolation, sampled at all 256 input levels. The
// duplication is deliberate — the editor has to draw the exact transfer
// function the server will apply, and asking the server per drag frame would
// cost a round trip per pointermove. Change one, change the other.
function curveLut(points) {
  const pts = [...points].sort((a, b) => a[0] - b[0]);
  const x = pts.map((p) => p[0]);
  const y = pts.map((p) => p[1]);
  const n = x.length;

  const h = [], delta = [];
  for (let i = 0; i < n - 1; i++) {
    h.push(x[i + 1] - x[i]);
    delta.push((y[i + 1] - y[i]) / h[i]);
  }
  // tangents: the average of the two adjacent secants, one-sided at the ends
  const m = new Array(n);
  m[0] = delta[0];
  m[n - 1] = delta[n - 2];
  for (let i = 1; i < n - 1; i++) m[i] = (delta[i - 1] + delta[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (delta[i] === 0) {
      // a flat segment must stay flat, or the cubic bulges out of it
      m[i] = m[i + 1] = 0;
    } else {
      const a = m[i] / delta[i], b = m[i + 1] / delta[i];
      const s = a * a + b * b;
      if (s > 9) {
        const t = 3 / Math.sqrt(s);
        m[i] = t * a * delta[i];
        m[i + 1] = t * b * delta[i];
      }
    }
  }

  const lut = new Uint8Array(256);
  let seg = 0;
  for (let level = 0; level < 256; level++) {
    while (seg < n - 2 && level >= x[seg + 1]) seg++;
    const t = (level - x[seg]) / h[seg];
    const t2 = t * t, t3 = t2 * t;
    const v =
      (2 * t3 - 3 * t2 + 1) * y[seg] +
      (t3 - 2 * t2 + t) * h[seg] * m[seg] +
      (-2 * t3 + 3 * t2) * y[seg + 1] +
      (t3 - t2) * h[seg] * m[seg + 1];
    lut[level] = Math.min(255, Math.max(0, Math.round(v)));
  }
  return lut;
}

const CURVE = { size: 256, hit: 10, maxPoints: 16 };

// Unlike the cluster plot, this is an *input*: buildParamControls() throws it
// away and rebuilds it on every effect change, so it can't be an init()-time
// singleton on a fixed id. All state lives in this closure and every listener
// is per-instance — dragging uses pointer capture on the canvas rather than the
// cluster plot's window-level mousemove/mouseup, which would leak one pair of
// listeners per rebuild.
function buildCurveEditor(p, current, sourceNodeId) {
  let points = (Array.isArray(current) ? current : p.default).map((pt) => [pt[0], pt[1]]);
  let hist = null;

  const row = document.createElement("div");
  row.className = "param-row";

  const label = document.createElement("label");
  const nameSpan = document.createElement("span");
  nameSpan.textContent = p.label;
  const reset = document.createElement("button");
  reset.type = "button";
  reset.className = "curve-reset";
  reset.textContent = "reset";
  label.append(nameSpan, reset);

  const canvas = document.createElement("canvas");
  canvas.className = "curve-canvas";
  canvas.width = canvas.height = CURVE.size;
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", "Tone curve editor");

  // the points travel as JSON on a hidden input so readParams() finds them the
  // same way it finds every other control: by [data-param]
  const input = document.createElement("input");
  input.type = "hidden";
  input.dataset.param = p.name;

  const hint = document.createElement("div");
  hint.className = "hint";
  hint.textContent = "drag to move · click to add · double-click to remove";

  row.append(label, canvas, input, hint);

  // ---- drawing

  const toCanvas = (v) => (v / 255) * CURVE.size;
  const flip = (v) => CURVE.size - toCanvas(v); // y grows downward on a canvas

  function draw() {
    const ctx = canvas.getContext("2d");
    const s = CURVE.size;
    ctx.clearRect(0, 0, s, s);

    if (hist) {
      // sqrt so one spike (a clipped sky, a black border) doesn't flatten the rest
      const peak = Math.sqrt(Math.max(...hist)) || 1;
      ctx.fillStyle = "rgba(214, 218, 226, 0.16)";
      for (let i = 0; i < 256; i++) {
        const bh = (Math.sqrt(hist[i]) / peak) * s;
        ctx.fillRect((i / 256) * s, s - bh, s / 256 + 0.5, bh);
      }
    }

    ctx.strokeStyle = "rgba(139, 147, 163, 0.25)";
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      const at = (i / 4) * s;
      ctx.beginPath();
      ctx.moveTo(at, 0); ctx.lineTo(at, s);
      ctx.moveTo(0, at); ctx.lineTo(s, at);
      ctx.stroke();
    }

    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = "rgba(139, 147, 163, 0.45)";
    ctx.beginPath();
    ctx.moveTo(0, s); ctx.lineTo(s, 0);
    ctx.stroke();
    ctx.setLineDash([]);

    const lut = curveLut(points);
    // canvas has no currentColor, so read the .fx-* hue off the element itself
    ctx.strokeStyle = getComputedStyle(canvas).color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < 256; i++) {
      const cx = toCanvas(i), cy = flip(lut[i]);
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    ctx.fillStyle = getComputedStyle(canvas).color;
    for (const [px, py] of points) {
      ctx.beginPath();
      ctx.arc(toCanvas(px), flip(py), 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function commit() {
    input.value = JSON.stringify(points);
    draw();
    // what drives live preview: the delegated `input` listeners on
    // #effect-params and #edit-params, through the usual 250 ms debounce
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }

  // ---- interaction

  // the canvas is CSS-scaled to the panel width, so pointer coordinates have to
  // come off the rendered box, not the backing store
  function atEvent(e) {
    const r = canvas.getBoundingClientRect();
    const scale = 255 / r.width;
    return {
      x: Math.round((e.clientX - r.left) * scale),
      y: Math.round((r.bottom - e.clientY) * (255 / r.height)),
      slop: CURVE.hit * scale,
    };
  }

  const nearest = (pos) => {
    let best = -1, bestD = Infinity;
    points.forEach(([px, py], i) => {
      const d = Math.hypot(px - pos.x, py - pos.y);
      if (d < bestD) { bestD = d; best = i; }
    });
    return bestD <= pos.slop ? best : -1;
  };

  const clamp = (v) => Math.max(0, Math.min(255, v));
  let dragging = -1;

  canvas.addEventListener("pointerdown", (e) => {
    const pos = atEvent(e);
    let i = nearest(pos);
    if (i === -1) {
      if (points.length >= CURVE.maxPoints) return;
      // a new interior point; the endpoints keep the domain at 0..255
      const x = Math.max(1, Math.min(254, pos.x));
      if (points.some((pt) => pt[0] === x)) return;
      i = points.findIndex((pt) => pt[0] > x);
      points.splice(i, 0, [x, clamp(pos.y)]);
    }
    dragging = i;
    canvas.setPointerCapture(e.pointerId);
    commit();
  });

  canvas.addEventListener("pointermove", (e) => {
    if (dragging === -1) return;
    const pos = atEvent(e);
    const pt = points[dragging];
    // endpoints are x-locked at 0 and 255 (that is what guarantees a
    // full-domain curve); interior points stay strictly between their neighbours
    if (dragging > 0 && dragging < points.length - 1) {
      pt[0] = Math.max(points[dragging - 1][0] + 1,
                       Math.min(points[dragging + 1][0] - 1, pos.x));
    }
    pt[1] = clamp(pos.y);
    commit();
  });

  const endDrag = (e) => {
    if (dragging === -1) return;
    dragging = -1;
    canvas.releasePointerCapture(e.pointerId);
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  canvas.addEventListener("dblclick", (e) => {
    const i = nearest(atEvent(e));
    if (i > 0 && i < points.length - 1) {
      points.splice(i, 1);
      commit();
    }
  });

  reset.onclick = () => {
    points = p.default.map((pt) => [pt[0], pt[1]]);
    commit();
  };

  // ---- histogram backdrop

  input.value = JSON.stringify(points);
  draw();
  if (sourceNodeId !== null) {
    api(`/api/nodes/${sourceNodeId}/histogram`)
      .then((data) => {
        // a rebuild can land first, leaving this canvas detached
        if (!canvas.isConnected) return;
        hist = data.luma;
        draw();
      })
      .catch((err) => console.warn("histogram failed:", err)); // the grid draws fine without it
  }
  return row;
}

// ---------- Effects ----------

// The Apply panel and the edit modal both build their controls here and read
// them back through readParams(), so the two can never disagree about ranges,
// defaults, or how a value is coerced. `values` seeds the controls — empty for
// Apply (fresh defaults), an existing node's params when editing.
// `sourceNodeId` is the node whose pixels the effect will be applied to — the
// same node the preview composes on — and is what the curve editor draws its
// histogram from.
function buildParamControls(container, effectName, values = {}, sourceNodeId = null) {
  const effect = state.effects.find((e) => e.name === effectName);
  container.innerHTML = "";
  for (const p of effect.params) {
    const current = values[p.name] ?? p.default;
    if (p.type === "points") {
      container.appendChild(buildCurveEditor(p, current, sourceNodeId));
      continue;
    }
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

// A selection outlives an effect switch — that is the whole point of keeping it
// in state — but not a move to pixels it was never picked against. Click points
// are tied to the node they were clicked on (their coords are in that node's
// pixel space); saved masks' pixels are frozen and fit any node of their image.
function pruneSelection() {
  const s = state.selection;
  if (!s.value) return;
  const stale = s.value.masks ? s.imageId !== state.imageId : s.nodeId !== state.nodeId;
  if (stale) Object.assign(s, { value: null, nodeId: null, imageId: null });
  // a surviving mask ref now belongs to the node we moved to: nodeId records
  // what the selection is aimed at, and everything downstream (the overlay's
  // source node, saveMask's node_id) reads it expecting the current one
  else s.nodeId = state.nodeId;
}

// Step 1 of the panel. Its own section rather than a row inside #effect-params,
// because selecting comes *before* choosing an effect and outlives every switch
// between them — living in the effect's box meant buildParamControls()'s
// `innerHTML = ""` tore the control down underneath a live selection.
// pruneSelection() runs here, and only here, so a selection has exactly one
// place it can expire; renderSelection() calls this before renderEffectControls().
function renderSelectControls() {
  pruneSelection();
  const box = $("select-controls");
  box.innerHTML = "";
  if (state.nodeId === null) {
    clearMaskOverlay();
    box.textContent = "No image selected.";
    return;
  }
  appendSelectionControls(box, {
    store: state.selection,
    sourceNodeId: state.nodeId,
    allowManage: true,
  });
}

function renderEffectControls() {
  const box = $("effect-params");
  const effect = buildParamControls(box, state.effect, {}, state.nodeId);
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
}

// `store` is the selection store the container's controls were bound to — the
// Apply panel's `state.selection` or the edit modal's `edit.selection`. It is a
// parameter rather than a global for the same reason the blend target is looked
// up scoped to its container: both forms can be on screen at once.
function readParams(container, effectName, store = null) {
  const params = {};
  container.querySelectorAll("[data-param]").forEach((el) => {
    if (el.type === "range") params[el.dataset.param] = Number(el.value);
    // the curve editor's control points, which only fit in a control as JSON
    else if (el.type === "hidden") params[el.dataset.param] = JSON.parse(el.value);
    else params[el.dataset.param] = el.value;
  });
  let parent2_id = null;
  if (effectName === "blend") {
    const withSel = container.querySelector(".blend-with");
    parent2_id = withSel && withSel.value ? Number(withSel.value) : null;
  }
  return { effect: effectName, params, parent2_id, selection: store?.value ?? null };
}

function readEffectForm() {
  return readParams($("effect-params"), state.effect, state.selection);
}

async function applyEffect() {
  const btn = $("apply-btn");
  const { effect, params, parent2_id, selection } = readEffectForm();
  if (effect === "blend" && parent2_id === null) return;
  exitPreview(false);
  const body = { parent_id: state.nodeId, effect, params };
  if (effect === "blend") body.parent2_id = parent2_id;
  if (selection) body.selection = selection;
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
const preview = {
  active: false, url: null, seq: 0, timer: null, source: null, abort: null,
};

// Hand the preview to the Apply panel. Debounced rather than immediate, so
// clicking quickly through tree nodes coalesces into one render instead of
// firing a full-resolution one per node.
function enterApplyPreview() {
  if (state.nodeId === null) return;
  preview.active = true;
  preview.source = applyPreviewRequest;
  schedulePreview();
}

function toggleLivePreview() {
  state.livePreview = $("live-preview").checked;
  localStorage.setItem("picky:livePreview", state.livePreview ? "1" : "");
  // unchecking restores the selected node's own render — that *is* "show me the
  // original", so there is no separate control for it
  if (state.livePreview) enterApplyPreview();
  else exitPreview(true);
}

// Apply composes the effect on top of the selected node, which is exactly what
// POST /api/nodes/{id}/preview does.
function applyPreviewRequest() {
  if (state.nodeId === null) return null;
  const { effect, params, parent2_id, selection } = readEffectForm();
  if (effect === "blend" && parent2_id === null) return null;
  const body = { effect, params };
  if (parent2_id !== null) body.parent2_id = parent2_id;
  if (selection) body.selection = selection;
  return { nodeId: state.nodeId, body, busyEl: $("preview-status") };
}

function exitPreview(restoreSrc) {
  clearTimeout(preview.timer);
  preview.timer = null;
  preview.seq++; // invalidate in-flight fetches
  preview.abort?.abort();
  preview.abort = null;
  preview.active = false;
  preview.source = null;
  // The overlay is deliberately *not* torn down here: its lifetime tracks the
  // selection store, not preview mode, so unchecking live preview leaves a live
  // pick lit. A mask still cannot outlive the node it was picked on, because
  // every path that changes the selection runs renderSelectControls(), which
  // prunes the store and repaints the overlay from whatever is left.
  $("preview-status").classList.remove("busy");
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
  // A preview is a full-resolution render — over a second on a 24-megapixel
  // original. The debounce coalesces a slider drag, but discrete clicks through
  // the work tree are seconds apart, so each one starts a fresh render; abort
  // the one it supersedes rather than leaving it to finish into a dropped
  // response. (The server cannot interrupt a render already running in the
  // threadpool, but this stops the queue behind it from filling up.)
  preview.abort?.abort();
  const ctl = new AbortController();
  preview.abort = ctl;
  req.busyEl?.classList.add("busy");
  try {
    const res = await fetch(`/api/nodes/${req.nodeId}/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
      signal: ctl.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    if (seq !== preview.seq || !preview.active) return; // stale response or exited
    const url = URL.createObjectURL(blob);
    $("preview").src = url;
    if (preview.url) URL.revokeObjectURL(preview.url);
    preview.url = url;
  } catch (err) {
    if (err.name === "AbortError") return; // superseded on purpose, not a failure
    console.warn("preview failed:", err); // keep the last frame; no alert mid-drag
  } finally {
    req.busyEl?.classList.remove("busy");
  }
}

// ---------- Click-to-segment selection ----------

// One overlay, one armed picker — the same single-slot model as `preview`:
// whichever controls instance (Apply panel or edit modal) last drove it owns it.
// `cache` maps an overlay's identity to its blob URL and owns every URL's
// lifetime, so nothing here revokes on replacement — only on a cache clear.
const selPicker = { armed: null, seq: 0, cache: new Map() };

function disarmPick() {
  selPicker.armed = null;
  $("preview-wrap").classList.remove("picking");
  document.querySelectorAll(".sel-pick").forEach((b) => b.classList.remove("armed"));
}

// What an overlay depends on. Saved masks are frozen pixels, so their PNG is a
// pure function of the ids and invert — `compute_mask` ignores the node
// entirely for that shape. Click points are in one node's pixel space, so they
// key on it too. Without this, walking the tree with a mask selected refetched
// a byte-identical half-megapixel PNG per node, at ~0.5 s each.
function overlayKey(sourceNodeId, selection) {
  return selection.masks
    ? `m:${JSON.stringify(selection)}`
    : `p:${sourceNodeId}:${JSON.stringify(selection)}`;
}

// Masks can be deleted or renamed and nodes re-rendered underneath a cached
// point overlay, so the cache does not outlive the image it was filled for.
function clearOverlayCache() {
  for (const url of selPicker.cache.values()) URL.revokeObjectURL(url);
  selPicker.cache.clear();
}

function paintOverlay(url) {
  const overlay = $("mask-overlay");
  overlay.src = url;
  overlay.hidden = false;
}

async function updateMaskOverlay(sourceNodeId, selection, busyEl) {
  const key = overlayKey(sourceNodeId, selection);
  const hit = selPicker.cache.get(key);
  if (hit) {
    selPicker.seq++; // a cached paint still supersedes anything in flight
    paintOverlay(hit);
    return;
  }
  const seq = ++selPicker.seq;
  busyEl?.classList.add("busy");
  try {
    const res = await fetch(`/api/nodes/${sourceNodeId}/mask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(selection),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    // cache even a stale response: it cost a render, and the key says what it
    // is, so a later selection that lands back here gets it for free
    selPicker.cache.set(key, url);
    if (seq !== selPicker.seq) return; // superseded; keep the pixels, drop the paint
    paintOverlay(url);
  } catch (err) {
    console.warn("mask failed:", err); // the live preview still shows the result
  } finally {
    busyEl?.classList.remove("busy");
  }
}

function clearMaskOverlay() {
  selPicker.seq++; // invalidate in-flight fetches
  const overlay = $("mask-overlay");
  overlay.hidden = true;
  // the src stays pointing at a cached blob URL; the cache owns it, and
  // re-selecting the same thing then repaints without a round trip
  disarmPick();
}

// The overlay always shows exactly what the store holds — including nothing.
function syncMaskOverlay(store, sourceNodeId, busyEl) {
  if (store.value) updateMaskOverlay(sourceNodeId, store.value, busyEl);
  else clearMaskOverlay();
}

// A selection is a union in either of its two shapes: `{points: [...], invert}`
// re-segmented by SAM, or `{masks: [...], invert}` loaded from frozen PNGs.
// These helpers are the only things that know the shapes, so the control below
// can treat "what is selected" as one list either way.
const selPoints = (sel) => (sel && !sel.masks ? sel.points : []);
const selMasks = (sel) => (sel && sel.masks ? sel.masks : []);

function selSummary(sel) {
  if (!sel) return "";
  const names = selMasks(sel).map(
    (id) => state.masks.find((m) => m.id === id)?.name ?? `mask #${id}`
  );
  if (names.length) return names.join(" + ");
  const pts = selPoints(sel);
  return pts.length === 1 ? `@ ${pts[0].x}, ${pts[0].y}` : `${pts.length} points`;
}

// The appendBlendTarget analogue for selections: any effect can be masked, so
// like blend's target this is not a registry param. The state lives in `store`
// — `state.selection` for the Apply panel, `edit.selection` for the modal — so
// that the two forms can be on screen at once. Every user change dispatches a
// bubbling `input` event, so the existing delegated listeners on the containers
// drive the shared preview debounce with no new wiring.
//
// The saved-mask list is part of this control rather than a section of its own:
// choosing masks *is* the selection, so keeping the picker, the level/invert
// options, Save-as-mask, and the list of masks in one component is what stops
// the panel's rendered state and the store from drifting apart. `allowManage`
// is off in the edit modal — its page is inert, so a `prompt()` could not be
// answered, and deleting a mask the node uses would 409 anyway.
function appendSelectionControls(
  container, { store, sourceNodeId, allowPick = true, allowManage = false }
) {
  // rebuilding the control is exactly when a previously armed picker becomes
  // invalid: its commit closure now writes into a detached row
  disarmPick();

  const row = document.createElement("div");
  row.className = "param-row";

  const label = document.createElement("label");
  const nameSpan = document.createElement("span");
  nameSpan.textContent = "Limit to object";
  const coords = document.createElement("span");
  coords.className = "sel-coords";
  label.append(nameSpan, coords);

  const pick = document.createElement("button");
  pick.type = "button";
  pick.className = "sel-pick";
  const clear = document.createElement("button");
  pick.title = allowPick
    ? "Then click an object in the image to select it"
    : "The click point was made on this node's input — re-picking needs a new node from the Apply panel";
  clear.type = "button";
  clear.className = "sel-clear";
  clear.textContent = "clear";
  const pickRow = document.createElement("div");
  pickRow.className = "sel-row";
  pickRow.append(pick, clear);

  const level = document.createElement("select");
  level.className = "sel-level";
  for (const name of ["auto", "whole", "part", "subpart"]) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name === "auto" ? "auto (best match)" : name;
    level.appendChild(opt);
  }
  const invertLabel = document.createElement("label");
  invertLabel.className = "sel-invert";
  const invert = document.createElement("input");
  invert.type = "checkbox";
  invertLabel.append(invert, document.createTextNode("invert"));
  const optRow = document.createElement("div");
  optRow.className = "sel-row";
  optRow.append(level, invertLabel);

  row.append(label, pickRow, optRow);

  // Save-as-mask, then the masks themselves — the order the workflow runs in:
  // pick an object, freeze it, repeat, then tick the ones this effect applies to.
  const save = document.createElement("button");
  save.type = "button";
  save.className = "sel-save-mask";
  save.textContent = "Save selection as mask";
  if (allowManage) row.appendChild(save);

  const list = document.createElement("ul");
  list.className = "sel-mask-list";
  row.appendChild(list);
  container.appendChild(row);

  const current = () => store.value;
  const commit = (sel) => {
    store.value = sel;
    store.nodeId = sel ? sourceNodeId : null;
    store.imageId = sel ? state.imageId : null;
    render(sel);
    syncMaskOverlay(store, sourceNodeId, pick);
    row.dispatchEvent(new Event("input", { bubbles: true }));
  };

  // Ticking a mask unions it in; unticking takes it out. Crossing over from a
  // click selection replaces it — the two shapes cannot mix, since one is a
  // recipe and the other is pixels.
  const toggleMask = (id) => {
    const ids = selMasks(current());
    const next = ids.includes(id) ? ids.filter((m) => m !== id) : [...ids, id];
    commit(next.length ? { masks: next, invert: !!current()?.invert } : null);
  };

  // The rows are built once, not on every toggle: `state.masks` is fixed for
  // this control's lifetime (anything that changes it goes through
  // refreshMasks(), which rebuilds the whole control), so ticking one only ever
  // needs to restyle the existing rows. Rebuilding them would detach the very
  // element the click came from mid-handler.
  const rows = [];
  if (!state.masks.length) {
    const empty = document.createElement("li");
    empty.className = "hint";
    empty.textContent = state.imageId === null
      ? "Pick an image to see its masks."
      : "Select an object above, then save it to reuse it on any node.";
    list.appendChild(empty);
  }
  for (const mask of state.masks) {
    const li = document.createElement("li");
    li.className = "mask-row";
    li.title = mask.used_by
      ? `Used by ${mask.used_by} node${mask.used_by === 1 ? "" : "s"} — delete those first to remove it`
      : "Include this mask in the selection";

    const box = document.createElement("input");
    box.type = "checkbox";
    box.tabIndex = -1; // the row owns the click; the box is the indicator

    const text = document.createElement("div");
    text.className = "preset-text";
    const name = document.createElement("div");
    name.className = "preset-name";
    name.textContent = mask.name;
    const summary = document.createElement("div");
    summary.className = "preset-summary";
    summary.textContent =
      `${mask.width}×${mask.height}${mask.used_by ? ` · used ${mask.used_by}×` : ""}`;
    text.append(name, summary);

    li.append(box, text);
    if (allowManage) {
      const del = document.createElement("button");
      del.className = "mask-del";
      del.textContent = "×";
      del.title = "Delete mask";
      del.onclick = (e) => {
        e.stopPropagation();
        deleteMask(mask);
      };
      li.appendChild(del);
    }
    li.onclick = () => toggleMask(mask.id);
    list.appendChild(li);
    rows.push({ mask, li, box });
  }

  function render(sel) {
    const points = selPoints(sel);
    coords.textContent = selSummary(sel);
    pick.textContent = points.length ? "Re-pick object" : "Select object";
    pick.disabled = !allowPick;
    // a frozen mask has no candidate masks left to re-rank
    level.value = points.length === 1 ? points[0].level : "auto";
    level.disabled = points.length !== 1;
    invert.checked = sel ? !!sel.invert : false;
    invert.disabled = clear.disabled = !sel;
    // a saved mask is already saved; only a fresh pick has anything to freeze
    save.disabled = maskBusy || state.imageId === null || !points.length;
    save.title = save.disabled
      ? "Select an object first — a saved mask is already saved"
      : "Freeze this selection's pixels so you can reuse it without re-picking";
    const chosen = selMasks(sel);
    for (const { mask, li, box } of rows) {
      box.checked = chosen.includes(mask.id);
      li.classList.toggle("selected", box.checked);
      li.classList.toggle("disabled", maskBusy);
    }
  }

  if (allowPick) {
    pick.onclick = () => {
      if (selPicker.armed) {
        disarmPick();
        return;
      }
      // single-shot: the click handler in initZoom does not disarm, so the
      // closure does it before committing
      selPicker.armed = {
        pick: (x, y) => {
          disarmPick();
          commit({ points: [{ x, y, level: level.value }], invert: invert.checked });
        },
      };
      pick.classList.add("armed");
      $("preview-wrap").classList.add("picking");
    };
  }
  level.onchange = () => {
    const pts = selPoints(current());
    if (pts.length === 1) {
      commit({ points: [{ ...pts[0], level: level.value }], invert: current().invert });
    }
  };
  invert.onchange = () => {
    const sel = current();
    if (sel) commit({ ...sel, invert: invert.checked });
  };
  clear.onclick = () => commit(null);
  save.onclick = () => saveMask();

  // seed without dispatching `input` — nothing user-driven happened yet — but do
  // repaint the overlay unconditionally, so it always agrees with the store:
  // that is what carries a pick across an effect switch, and what stops a dead
  // overlay lingering over an empty one
  render(store.value);
  syncMaskOverlay(store, sourceNodeId, pick);
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

// ---------- Saved masks (frozen selections, reusable across an image's tree) ----------
//
// Presets are global and fetched once; masks are image-scoped, so the analogous
// rule is: fetched on image change (in selectImage), refreshed only after a
// save/rename/delete. The list itself is drawn by appendSelectionControls(),
// because ticking a mask *is* editing the selection — see the comment there.

let maskBusy = false;

async function refreshMasks() {
  state.masks = await api(`/api/images/${state.imageId}/masks`);
  // a saved or deleted mask changes what the ids in a cached overlay key mean
  clearOverlayCache();
  // the mask list and the selection are one control, so redrawing it is how
  // the list refreshes — and it reseeds the picker from the new list too
  renderSelectControls();
}

async function saveMask() {
  const sel = state.selection.value;
  if (!sel || sel.masks || state.imageId === null) return;
  const name = prompt("Name this mask:", "");
  if (name === null || !name.trim()) return;
  maskBusy = true;
  renderSelectControls();
  try {
    const created = await api(`/api/images/${state.imageId}/masks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: name.trim(),
        node_id: state.selection.nodeId,
        selection: sel,
      }),
    });
    // Switch the live selection over to the frozen pixels, so the next Apply
    // uses them rather than re-running SAM. `invert: false` because the PNG was
    // frozen post-invert — what was on screen is what got saved.
    Object.assign(state.selection, {
      value: { masks: [created.id], invert: false },
      nodeId: state.nodeId,
      imageId: state.imageId,
    });
    maskBusy = false;
    await refreshMasks();
  } catch (err) {
    alert(`Could not save mask: ${err.message}`);
  } finally {
    maskBusy = false;
    renderSelectControls();
  }
}

async function deleteMask(mask) {
  if (maskBusy) return;
  if (!confirm(`Delete the mask “${mask.name}”?`)) return;
  try {
    await api(`/api/masks/${mask.id}`, { method: "DELETE" });
    const sel = state.selection.value;
    if (sel?.masks?.includes(mask.id)) {
      const rest = sel.masks.filter((id) => id !== mask.id);
      // the rest of the union survives; only an emptied one clears the store
      if (rest.length) sel.masks = rest;
      else Object.assign(state.selection, { value: null, nodeId: null, imageId: null });
    }
    await refreshMasks();
  } catch (err) {
    alert(`Could not delete mask: ${err.message}`);
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
  statRow(body, "Masks", s.masks);
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
  statRow(body, "Masks", `${formatBytes(st.masks.bytes)} · ${st.masks.files} files`);
  const cacheBytes =
    st.renders.bytes + st.thumbs.bytes + st.clusters.bytes + st.embeddings.bytes;
  statRow(body, "Render cache", formatBytes(cacheBytes));
  statRow(body, "renders", `${formatBytes(st.renders.bytes)} · ${st.renders.files} files`, true);
  statRow(body, "thumbnails", `${formatBytes(st.thumbs.bytes)} · ${st.thumbs.files} files`, true);
  statRow(body, "cluster data", `${formatBytes(st.clusters.bytes)} · ${st.clusters.files} files`, true);
  statRow(body, "embeddings", `${formatBytes(st.embeddings.bytes)} · ${st.embeddings.files} files`, true);
  statRow(
    body,
    "Total",
    formatBytes(st.database.bytes + st.originals.bytes + st.masks.bytes + cacheBytes)
  );

  const note = document.createElement("div");
  note.className = "hint";
  note.textContent =
    "The render cache rebuilds itself from the originals and the work tree — only the database, originals, and saved masks are irreplaceable.";
  body.appendChild(note);
}

// ---------- Editing an existing node's settings ----------

// Unlike Apply, this changes a node in place: it keeps its id and its children,
// and every render derived from it is thrown away and rebuilt.
// `selection` is the modal's own selection store, the counterpart of
// state.selection for the Apply panel — the two never share one.
const edit = { node: null, selection: null };

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
  const { effect, params, parent2_id, selection } = readParams(
    $("edit-params"), node.effect, edit.selection
  );
  if (effect === "blend" && parent2_id === null) return null;
  const body = { effect, params };
  if (parent2_id !== null) body.parent2_id = parent2_id;
  if (selection) body.selection = selection;
  // previewing an edit to node N means re-applying N's effect to N's *input*
  return { nodeId: node.parent_id, body, busyEl: $("edit-status") };
}

function openEdit(node) {
  // Select the node first: renderSelection() is the choke point that exits any
  // running preview, so it has to happen before the modal starts its own. Then
  // exit unconditionally — with live preview, renderSelection() *re-enters* and
  // leaves a debounce timer armed, and that timer would otherwise fire against
  // whatever source it finds later. exitPreview() is what clears it.
  if (state.nodeId !== node.id) {
    state.nodeId = node.id;
    renderSelection();
  }
  exitPreview(false);
  edit.node = node;
  $("edit-title").textContent = `#${node.id} ${nodeLabel(node)}`;
  // the node's *parent* — editing re-applies the effect to the node's input,
  // which is both what the preview shows and what the histogram describes
  buildParamControls($("edit-params"), node.effect, node.params || {}, node.parent_id);
  if (node.effect === "blend") {
    appendBlendTarget($("edit-params"), {
      text: "Blend with",
      excludeId: node.id,
      selected: node.parent2_id,
      maxId: node.id, // matches the server's cycle guard
    });
  }
  // No re-picking: showModal() makes the page inert, so clicking a new point is
  // impossible from here — that means a new node from the Apply panel. Ticking
  // *saved* masks still works, since frozen pixels need no click. Coords are in
  // the parent's space, which is what this previews. allowManage is off for the
  // same inertness reason: saveMask()'s prompt() could not be answered.
  edit.selection = { value: node.selection, nodeId: node.parent_id, imageId: state.imageId };
  appendSelectionControls($("edit-params"), {
    store: edit.selection,
    sourceNodeId: node.parent_id,
    allowPick: false,
    allowManage: false,
  });
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
  edit.selection = null;
  exitPreview(restoreSrc);
  // the modal borrowed the one overlay; hand it back to the Apply panel's own
  // selection, which openEdit() already left pointing at this same node
  syncMaskOverlay(state.selection, state.nodeId, null);
  $("edit-modal").close();
  // and hand the preview back too. The Save path re-enters via selectImage(),
  // but Cancel and Esc end here, and live preview should survive both.
  if (state.livePreview && restoreSrc) enterApplyPreview();
}

async function saveEdit() {
  const node = edit.node;
  if (!node) return;
  const { params, parent2_id, selection } = readParams(
    $("edit-params"), node.effect, edit.selection
  );
  if (node.effect === "blend" && parent2_id === null) return;
  const btn = $("edit-save-btn");
  btn.disabled = true;
  btn.classList.add("busy");
  btn.textContent = "Saving…";
  try {
    await api(`/api/nodes/${node.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params, parent2_id, selection }), // null clears
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
  // the preference outlives the session; only an explicit "" means opted out
  state.livePreview = localStorage.getItem("picky:livePreview") !== "";
  $("live-preview").checked = state.livePreview;
  $("live-preview").onchange = toggleLivePreview;
  // both control containers drive the one debounce; the selection lives in its
  // own section now, so it needs its own listener
  for (const id of ["effect-params", "select-controls"]) {
    $(id).addEventListener("input", () => {
      if (preview.active) schedulePreview();
    });
  }
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
  renderSelectControls(); // the empty state, for a library with no images yet
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
