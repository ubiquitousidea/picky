const state = {
  effects: [],
  // Name of the effect the Apply panel is set to, or null when no feature button
  // is lit — which is how you look at the selected node's own pixels. There is
  // no separate live-preview control: an armed effect *is* the preview running.
  effect: null,
  images: [],
  imageId: null,
  nodes: [],       // flat node list for the selected image
  nodeId: null,    // selected node
  presets: [],     // saved effect chains, reusable across images
  masks: [],       // saved masks — image-scoped, so refetched per image
  // The image's one framing, and the geometry it implies. The crop is an output
  // stage outside the work tree, so this describes *every* node of the image at
  // once — including the `inverse` affine that turns a click on the framed
  // preview back into a coordinate in the node's own (uncropped) pixel space.
  crop: null,
  geometry: null,
  // The Apply panel's current selection. It lives here rather than in the DOM
  // because the effect controls are torn down and rebuilt on every effect
  // switch, which used to take the pick with them. nodeId/imageId record what
  // it was picked against, so pruneSelection() can tell when it goes stale.
  selection: { value: null, nodeId: null, imageId: null },
  // The filmstrip is off by default — the Image map is the picker it defers to.
  filmstrip: false,
  // The command row's conversation, held here because the Messages API is
  // stateless and this app has no sessions — it is posted back on every turn and
  // replaced by whatever comes back. `busy` is what stops a second Enter while
  // the first turn is still applying effects. `wire` is the same conversation
  // as the API saw it, one entry per turn, kept for the Wire dialog to read —
  // it is a record, never posted back, and dies with the history it describes.
  agent: { history: [], wire: [], busy: false },
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

// A gallery thumb is cached by URL, and re-framing an image changes its pixels
// without changing its node id — so the crop rides along as a version tag, the
// same trick the mask grid's `?v=<created_at>` uses. Keyed on the crop itself
// because an image row carries no updated_at, and absent entirely when there is
// no crop, so an unframed library's URLs are exactly what they always were.
const cropTag = (crop) =>
  crop ? `&v=${crop.angle},${crop.rect.map((n) => n.toFixed(4)).join(",")}` : "";

async function refreshGallery() {
  state.images = await api("/api/images");
  const ul = $("gallery");
  ul.innerHTML = "";
  for (const img of state.images) {
    const li = document.createElement("li");
    li.classList.toggle("selected", img.id === state.imageId);
    const thumb = document.createElement("img");
    thumb.src = `/api/nodes/${img.root_node_id}/render?thumb=1${cropTag(img.crop)}`;
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = img.name;
    li.append(thumb, name);
    li.onclick = () => selectImage(img.id);
    ul.appendChild(li);
  }
}

// The filmstrip is the last row of the bar, so showing it never reorders
// anything above — the preview simply gives up the height.
function setFilmstrip(on) {
  state.filmstrip = on;
  localStorage.setItem("picky:filmstrip", on ? "1" : "");
  $("film-section").hidden = !on;
  $("film-btn").classList.toggle("on", on);
  // the strip scrolls, and the selected image is anywhere in the library
  if (on) document.querySelector("#gallery li.selected")
    ?.scrollIntoView({ inline: "nearest", block: "nearest" });
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
    state.crop = null;
    state.geometry = null;
  } else {
    // masks are image-scoped, so unlike presets they are fetched on every image
    // change; after that only a save/rename/delete refreshes them. The image
    // itself comes along for its crop and geometry, which the click picker needs
    // before the user's first click.
    let image;
    [state.nodes, state.masks, image] = await Promise.all([
      api(`/api/images/${imageId}/tree`),
      api(`/api/images/${imageId}/masks`),
      api(`/api/images/${imageId}`),
    ]);
    state.crop = image.crop;
    state.geometry = image.geometry;
    const valid = state.nodes.some((n) => n.id === nodeId);
    state.nodeId = valid ? nodeId : state.nodes[state.nodes.length - 1].id;
  }
  renderSelection();
}

function renderSelection() {
  exitPreview(false);
  // crop mode borrows the preview image for its proxy, so it leaves through the
  // same choke point the effect preview does. `false`: we are already repainting.
  exitCropMode(false);
  document.querySelectorAll("#gallery li").forEach((li, i) => {
    const on = state.images[i]?.id === state.imageId;
    li.classList.toggle("selected", on);
    // the strip scrolls sideways, so the current image can be off its end —
    // most visibly right after an upload, which lands at one
    if (on && state.filmstrip) li.scrollIntoView({ inline: "nearest", block: "nearest" });
  });
  const hasImage = state.imageId !== null;
  $("preview-wrap").hidden = !hasImage;
  $("drop-hint").hidden = hasImage;
  $("zoom-hud").hidden = !hasImage;
  $("delete-btn").hidden = !hasImage;
  $("export-btn").hidden = !hasImage;
  $("apply-btn").disabled = !hasImage;
  $("frame-btn").disabled = !hasImage;
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
  // and last, re-enter the preview this function's exitPreview() just left — if
  // a feature button is lit, which is the whole of "is preview on". The node's
  // own render is already painted above, so the unedited pixels show immediately
  // and the effect lands on top when the debounce fires.
  if (state.effect) enterApplyPreview();
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

// Wheel deltas are not comparable across devices: a mouse notch arrives as one
// event carrying ~100px, where a trackpad's two-finger scroll arrives as a
// stream of events a few pixels each — plus a tail of momentum after the fingers
// lift. Stepping by a fixed factor per *event* is what made the trackpad
// unusable, since a flick is sixty of them; zoom is therefore exponential in the
// distance scrolled, with ZOOM_WHEEL_PX chosen so one mouse notch is still
// exactly the 1.15x it has always been. Shared by the preview panel and the
// Image map, which zoom the same way and must keep feeling the same.
const ZOOM_WHEEL_STEP = 1.15;
const ZOOM_WHEEL_PX = 100; // Chrome/macOS pixel-mode delta for one notch
const WHEEL_LINE_PX = 33; // DOM_DELTA_LINE: Firefox sends 3 lines per notch
const WHEEL_PAGE_PX = 400; // DOM_DELTA_PAGE, rare
// A trackpad pinch reaches the page as a ctrlKey wheel carrying much smaller
// deltas, so it needs its own gain to read as a pinch rather than a nudge. Both
// callers preventDefault(), which is also what stops it zooming the browser.
const ZOOM_PINCH_GAIN = 6;

function wheelZoomFactor(e) {
  const unit =
    e.deltaMode === 1 ? WHEEL_LINE_PX : e.deltaMode === 2 ? WHEEL_PAGE_PX : 1;
  let px = e.deltaY * unit * (e.ctrlKey ? ZOOM_PINCH_GAIN : 1);
  // one event must never do more than about a notch and a half, however hard
  // the gesture was flung
  px = Math.max(-150, Math.min(150, px));
  return ZOOM_WHEEL_STEP ** (-px / ZOOM_WHEEL_PX);
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
      const next = Math.min(16, Math.max(1, view.zoom * wheelZoomFactor(e)));
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
    const ox = ((e.clientX - rect.left) / rect.width) * img.naturalWidth;
    const oy = ((e.clientY - rect.top) / rect.height) * img.naturalHeight;
    const [x, y] = toNodeSpace(ox, oy);
    selPicker.armed.pick(x, y);
  });
}

// A click lands on the *framed* preview, but a selection spec is stored in the
// node's own pixel space — the crop is an output stage, so nodes and saved masks
// never see it. The server hands over the inverse affine with the image
// (`geometry.inverse`), so this is six multiplies and no trigonometry here: the
// browser never needs to know how PIL rounds an expanded rotation.
function toNodeSpace(ox, oy) {
  const g = state.geometry;
  const [a, b, c, d, e, f] = g?.inverse || [1, 0, 0, 0, 1, 0];
  const [w, h] = g?.source || [Infinity, Infinity];
  return [
    Math.min(Math.max(Math.round(a * ox + b * oy + c), 0), w - 1),
    Math.min(Math.max(Math.round(d * ox + e * oy + f), 0), h - 1),
  ];
}

// ---------- Work tree ----------

// How an effect is named wherever the name stands on its own — a tree row, a
// stats line, a filter chip. Blend is special-cased away from its registry
// label, which is "Blend with…": that is the wording of the *control* that asks
// for a second parent, and it reads as an unfinished sentence anywhere else.
function effectLabel(name) {
  if (name === "blend") return "Blend";
  const spec = state.effects.find((e) => e.name === name);
  return spec ? spec.label : name;
}

function nodeLabel(node) {
  return node.effect ? effectLabel(node.effect) : "Original";
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

// out-of-focus highlights: the shapes a fast lens turns points into, round on
// the axis and clipped to cat's eyes by the barrel as they leave it — the swirl
// the effect's slider is a dial for. One point stays crisp: that is the subject.
function iconBokeh() {
  const svg = iconSvg();
  // The aperture is the intersection of two equal circles whose centres lie on
  // the radial line — `_lens_runs`' construction, which SVG can draw exactly
  // rather than approximate, since it is only ever two arcs. `R` is the long
  // half-axis and it points *across* the radius: highlights lie down tangentially
  // as they go out, which is what makes a ring of them read as a swirl.
  //
  // Aspect follows the server's `1 - swirl*rho^2`, but off the icon's own scale:
  // measured against the frame diagonal the way `_swirl_aspect` measures, no
  // highlight that fits in 24 units gets below t ≈ 0.8, which at 28 px on screen
  // is just a circle. So rho is distance over 11 and swirl is 0.9 — chosen to
  // read at button size. This is decoration, not a transfer function; unlike
  // `curveLut()` it has no pact with the server, and drifting from it costs
  // nothing.
  const catsEye = (cx, cy, R, opacity) => {
    const dx = cx - 12, dy = cy - 12;
    const theta = Math.atan2(dy, dx);
    const t = 1 - 0.9 * Math.min(Math.hypot(dx, dy) / 11, 1) ** 2;
    const rc = (R * (1 / t + t)) / 2; // radius of each of the two circles
    // the two tips, at ±R across theta. At t = 1 the arcs become semicircles and
    // this is a plain circle — the same degeneracy `_lens_runs` has at d = 0.
    const [ux, uy] = [-Math.sin(theta) * R, Math.cos(theta) * R];
    const [x1, y1, x2, y2] = [cx + ux, cy + uy, cx - ux, cy - uy];
    return svgEl("path", {
      d: `M ${x1} ${y1} A ${rc} ${rc} 0 0 1 ${x2} ${y2} A ${rc} ${rc} 0 0 1 ${x1} ${y1}`,
      fill: "currentColor", "fill-opacity": opacity,
      stroke: "currentColor", "stroke-width": 0.9, "stroke-opacity": opacity + 0.3,
    });
  };
  // distances vary on purpose: the near one is nearly round and the far one is
  // pointed, so the icon shows the ramp and not just four identical lenses
  const highlights = [
    [17.5, 7.5, 5, 0.22], [8, 8.5, 3.6, 0.34], [16, 17.5, 4.4, 0.27], [6, 17.5, 3, 0.42],
  ];
  for (const [cx, cy, R, opacity] of highlights) svg.appendChild(catsEye(cx, cy, R, opacity));
  // on the axis, and so the one thing measured from rather than deformed
  svg.appendChild(svgEl("circle", { cx: 12, cy: 12, r: 2, fill: "currentColor" }));
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
  { key: "blur", label: "Blur", icon: iconBlur, effects: ["blur"] },
  { key: "bokeh", label: "Bokeh", icon: iconBokeh, effects: ["bokeh"] },
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
    btn.onclick = () => toggleGroup(group);
    box.appendChild(btn);
  }
}

// The buttons are toggles, not a radio group: clicking the lit one turns the
// effect off. That is the only way back to the unedited node, so there is no
// separate live-preview control for it to disagree with.
function toggleGroup(group) {
  const lit = groupFor(state.effect) === group;
  setEffect(lit ? null : lastMethod[group.key] || group.effects[0]);
}

// The selected effect lives in state, not in the DOM. Single write path: every
// change rebuilds the params and starts, restarts or stops the preview — `null`
// included, which is what makes "no button lit" a state and not just an absence.
function setEffect(name) {
  const group = name === null ? null : groupFor(name);
  state.effect = name;
  if (group) lastMethod[group.key] = name;
  document.querySelectorAll("#effect-buttons .fx-btn").forEach((btn) => {
    const on = !!group && btn.dataset.group === group.key;
    btn.classList.toggle("selected", on);
    btn.setAttribute("aria-pressed", on);
  });
  renderEffectControls();
  // turning the effect off restores the node's own render — that *is* "show me
  // the original", so there is no separate control for it
  if (!state.effect) return exitPreview(true);
  // Bokeh is the one effect with a model behind it, and the server 409s rather
  // than fetching 99 MB inside a render. So the button is what triggers the
  // download, the way the search box triggers the text tower, and the preview
  // waits for it — a live preview that 409'd on every slider drag would be a
  // worse way to learn the same thing.
  if (state.effect === "bokeh" && !depthReady) prepareDepthModel();
  else enterApplyPreview();
}

// Latched, so the round trip happens once a session rather than on every visit
// to the button. The server answers `done` synchronously when the file is
// already there, so even the first visit is one request and no poll.
let depthReady = false;
let depthSeq = 0;
// The markup's own "rendering…", captured the first time anything is about to
// overwrite it rather than spelled out a second time here.
let previewStatusText = null;

async function prepareDepthModel() {
  const seq = ++depthSeq;
  const status = $("preview-status");
  if (previewStatusText === null) previewStatusText = status.textContent;
  const say = (text) => {
    status.textContent = text;
    status.classList.add("busy"); // it is `visibility: hidden` otherwise
  };
  // Switching effects mid-download abandons the poll — including the enter into
  // preview at the end, which would otherwise fire against whatever is lit now.
  // Abandoning puts the line back as it goes, so a half-written progress
  // message cannot survive to be shown as the next effect's "rendering…".
  const stale = () => {
    if (seq === depthSeq && state.effect === "bokeh") return false;
    status.textContent = previewStatusText;
    status.classList.remove("busy");
    return true;
  };
  try {
    let job = await api("/api/depth-model/prepare", { method: "POST" });
    while (job.state === "running") {
      if (stale()) return;
      say(depthJobText(job));
      await new Promise((done) => setTimeout(done, EMBED_POLL_MS));
      if (stale()) return;
      job = await api("/api/depth-model/progress");
    }
    if (stale()) return;
    if (!job.ready) throw new Error(job.error || "not available");
  } catch (err) {
    if (stale()) return;
    // Reported and not retried, for runEmbedSearch()'s reason: falling through
    // would repeat the whole download to reach the same error.
    say(`Could not load the depth model: ${err.message}`);
    return;
  }
  depthReady = true;
  status.textContent = previewStatusText; // back to what refreshPreview() shows
  status.classList.remove("busy");
  enterApplyPreview();
}

// One phase and one unit, like textJobText() — and separate from it for the
// same reason, that naming the model is most of what the sentence is for.
function depthJobText(job) {
  if (job.phase === "download") {
    const mb = (bytes) => Math.round(bytes / 1e6);
    const of = job.total ? ` of ${mb(job.total)}` : ""; // no Content-Length
    return `Downloading the depth model — ${mb(job.done)}${of} MB (first use only)`;
  }
  return "Loading the depth model…";
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
function buildParamControls(
  container, effectName, values = {}, sourceNodeId = null, { allowPick = true } = {}
) {
  const effect = state.effects.find((e) => e.name === effectName);
  container.innerHTML = "";
  // by param name, for syncParamVisibility() below: a row is found here, while
  // it is being built, rather than dug back out of the DOM afterwards — the
  // curve editor's is not a `.param-row` around a `[data-param]` and would need
  // a second rule to reach.
  const rows = new Map();
  for (const p of effect.params) {
    const current = values[p.name] ?? p.default;
    if (p.type === "points") {
      const editor = buildCurveEditor(p, current, sourceNodeId);
      rows.set(p.name, editor);
      container.appendChild(editor);
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
      if (p.pick === "depth") {
        row.appendChild(buildDepthPick(container, p, sourceNodeId, allowPick));
      }
    }
    rows.set(p.name, row);
    container.appendChild(row);
  }
  syncParamVisibility(container, effect, rows);
  return effect;
}

// A param spec's `when` says the control only applies in some mode of another
// control — bokeh's two aperture sliders, blend's weight — and this is the one
// place that reads it. Presentation, like `pick`: the row is hidden, never
// removed, so readParams() still finds it (it walks every `[data-param]`,
// visible or not). That is deliberate on both counts — the params dict Apply
// sends stays complete, and a slider dialled in one mode still holds its value
// when you come back to it. Which of them the effect then ignores is the
// server's business, and `bokeh()` already resolves exactly that.
//
// A pass of its own, after every row exists, because a gate may point forward:
// bokeh's `density` names `show`, which is declared after it.
function syncParamVisibility(container, effect, rows) {
  const syncs = new Map(); // controlling <select> → the syncs it drives
  for (const p of effect.params) {
    const row = rows.get(p.name);
    if (!p.when || !row) continue;
    // Every named control must hold a listed value. One entry is all any spec
    // uses today; `every` is what makes a second one mean the obvious thing
    // rather than silently the first one.
    const gates = Object.entries(p.when).map(([name, allowed]) => [
      container.querySelector(`[data-param="${name}"]`),
      allowed,
    ]);
    const sync = () =>
      (row.hidden = !gates.every(([el, allowed]) => el && allowed.includes(el.value)));
    for (const [el] of gates) {
      if (el) syncs.set(el, [...(syncs.get(el) || []), sync]);
    }
    sync(); // the panel is built in whatever mode it opens in
  }
  for (const [el, fns] of syncs) {
    el.addEventListener("change", () => fns.forEach((fn) => fn()));
  }
}

// A slider whose number nobody can read off a photo — bokeh's focal plane is a
// position in one frame's normalized depth — gets a second way in: click the
// thing you want sharp. The button carries `.sel-pick` deliberately, not just
// for the styling: that class is what disarmPick() sweeps, so arming this and
// arming click-to-segment cannot both look lit, and either one disarms the
// other for free. `.depth-pick` beside it is what tells the two apart — this one
// sits under a slider and fills its width, the selection's own sits beside
// `clear` in a shrink-to-fit row — and is also how the commit closure below
// finds this button again. The slider stays the only thing Apply reads.
function buildDepthPick(container, p, sourceNodeId, allowPick) {
  const btn = document.createElement("button");
  btn.type = "button"; // inside no form, but readParams() walks past it either way
  btn.className = "sel-pick depth-pick";
  const idle = "Focus on a click";
  btn.textContent = idle;
  // Disabled rather than absent in the edit modal, exactly as the selection's
  // own pick is and for the same reason: showModal() makes the image inert, so
  // there is nothing to click. The slider still works there.
  btn.disabled = !allowPick || sourceNodeId === null;
  btn.title = btn.disabled
    ? "Focusing by click needs the image — use the Bokeh button to make a new node"
    : "Click the subject in the image to focus on it";

  btn.onclick = () => {
    if (selPicker.armed?.owner === btn) return disarmPick(); // a second press cancels
    disarmPick(); // and a first press takes the slot off whoever else held it
    selPicker.armed = {
      owner: btn,
      pick: async (x, y) => {
        disarmPick();
        // The panel is torn down and rebuilt whenever the selection changes,
        // which is exactly the window this pick was waiting through — so the
        // slider is looked up again rather than captured, and a rebuild that
        // dropped it (a switch away from bokeh, say) lands on nothing.
        const live = document.getElementById(container.id);
        const input = live?.querySelector(`[data-param="${p.name}"]`);
        const button = live?.querySelector(".sel-pick.depth-pick");
        if (!input) return;
        if (button) button.textContent = "Reading depth…";
        try {
          const { depth } = await api(
            `/api/nodes/${sourceNodeId}/depth-at?x=${x}&y=${y}`
          );
          // The slider quantizes this to its own step, which is the point:
          // what gets rendered and stored is a plain focus value someone can
          // then nudge, not a hidden reading the number merely reports.
          input.value = depth;
          // one dispatch does both jobs: the input's own handler repaints the
          // number beside the label, and the bubble reaches the delegated
          // listener that schedules the preview
          input.dispatchEvent(new Event("input", { bubbles: true }));
          if (button) button.textContent = idle;
        } catch (err) {
          if (button) {
            button.textContent = "Could not read depth";
            button.title = err.message;
          }
        }
      },
    };
    btn.classList.add("armed");
    btn.textContent = "Click the subject";
    $("preview-wrap").classList.add("picking");
  };
  // disarmPick() only removes `.armed`, so the label is put back here — the
  // one place that knows what it said before.
  btn.addEventListener("pickdisarmed", () => (btn.textContent = idle));
  btn.classList.add("depth-pick");
  return btn;
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
  // The weight row's "average only" rule is not here: `mode` and `weight` are
  // both registry params, so BLEND_SPEC's `when` states it and
  // buildParamControls() has already wired it by the time this runs.
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
  // An empty params box is the off state made visible — it is the other half of
  // the unlit button, so this early-out and setEffect()'s `.selected` class have
  // to agree. Apply goes with them: with nothing to apply it is hidden outright
  // rather than greyed, so the row is just the buttons and the image below it.
  // So does the render indicator — it reserves its line so the row does not
  // reflow when a render starts, and there are no renders to start.
  if (state.effect === null) {
    box.innerHTML = "";
    $("apply-btn").hidden = true;
    $("preview-status").hidden = true;
    return;
  }
  $("apply-btn").hidden = false;
  $("preview-status").hidden = false;
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
  if (state.effect === null) return; // no button lit, nothing to apply
  const { effect, params, parent2_id, selection } = readEffectForm();
  if (effect === "blend" && parent2_id === null) return;
  exitPreview(false);
  btn.disabled = true;
  btn.classList.add("busy");
  btn.textContent = "Applying…";
  // Using a picked object banks it first, so it outlives the click that made
  // it — see bankSelection. Failing to bank is not a reason to refuse the
  // effect, so fall back to the click and say so once the node is in.
  let banked = selection;
  let bankErr = null;
  try {
    banked = await bankSelection(state.selection);
  } catch (err) {
    bankErr = err;
  }
  const body = { parent_id: state.nodeId, effect, params };
  if (effect === "blend") body.parent2_id = parent2_id;
  if (banked) body.selection = banked;
  try {
    const node = await api(`/api/images/${state.imageId}/nodes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    await selectImage(state.imageId, node.id);
    // The effect is committed, so the panel goes back to nothing selected: what
    // you see is the node you just made, not the same effect armed again on top
    // of it. setEffect stays the single write path — it unlights the button,
    // empties the params and stops the preview renderSelection() just re-armed.
    // After selectImage, not before: its exitPreview(true) repaints from
    // state.nodeId, which has to be the new node and not the old one.
    setEffect(null);
    if (bankErr) {
      alert(
        "The effect was applied, but the selection could not be saved for reuse: " +
          bankErr.message
      );
    }
  } catch (err) {
    // a bank that succeeded is never unwound — the mask is durable user data and
    // the retry reuses it — but the grid has not drawn its tile yet
    renderSelectControls();
    alert(`Failed to apply effect: ${err.message}`);
  } finally {
    btn.disabled = state.imageId === null || state.effect === null;
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

// Apply composes the effect on top of the selected node, which is exactly what
// POST /api/nodes/{id}/preview does.
function applyPreviewRequest() {
  if (state.nodeId === null || state.effect === null) return null;
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

// ---------- Crop & rotate: the image's one framing ----------
//
// Not an effect and not a node — a crop applies after the whole work tree, so it
// belongs to the image rather than to any node, and the panel sits outside the
// Apply flow. Crop mode swaps the preview for a *rotated but unframed* proxy, so
// the user can see what falls outside the frame and where the black corners will
// be; the SVG overlay draws the frame on top of it.

const CROP = { handlePx: 9, grabPx: 12, minFrac: 0.01, minLinePx: 8 };
const cropTool = { armed: null };
// `active` is crop mode; `angle`/`rect` are the unsaved edit. `seq` drops stale
// proxy responses the same way preview.seq drops stale renders. `canvas` is the
// *full-size* rotated canvas the current proxy stands for, straight off the
// proxy response — the readout's pixel count comes from it rather than from the
// downscaled proxy on screen. `level` is the straighten tool, `line` the line it
// is drawing (fractions of the displayed proxy, for painting only).
const crop = {
  active: false, angle: 0, rect: [0, 0, 1, 1],
  seq: 0, url: null, canvas: null, level: false, line: null,
};

// `toggleAttribute`, not `.hidden`: the overlay is an <svg>, and `hidden` is an
// HTMLElement property that SVGElement does not implement — assigning it sets a
// plain JS expando and leaves the attribute (and so `display: none`) in place.
const showCropOverlay = (on) => $("crop-overlay").toggleAttribute("hidden", !on);

function disarmCrop() {
  cropTool.armed = null;
  $("preview-wrap").classList.remove("cropping");
  showCropOverlay(false);
}

// Client coords -> a fraction of the displayed image. The img's bounding rect is
// post-transform, so this needs no pan/zoom inversion — the same reasoning as
// the click picker's mapping and the curve editor's atEvent.
function cropAtEvent(e) {
  const r = $("preview").getBoundingClientRect();
  return {
    x: (e.clientX - r.left) / r.width,
    y: (e.clientY - r.top) / r.height,
    // a grab radius in fractions, so hit-testing is in the same units as the rect
    slopX: CROP.grabPx / r.width,
    slopY: CROP.grabPx / r.height,
  };
}

// Which part of the frame is under the pointer: a corner, an edge, the inside,
// or nothing. Corners win over edges, so the diagonal grab is never stolen.
function cropHit(rect, pos) {
  const [x, y, w, h] = rect;
  const nearL = Math.abs(pos.x - x) <= pos.slopX;
  const nearR = Math.abs(pos.x - (x + w)) <= pos.slopX;
  const nearT = Math.abs(pos.y - y) <= pos.slopY;
  const nearB = Math.abs(pos.y - (y + h)) <= pos.slopY;
  const inX = pos.x >= x - pos.slopX && pos.x <= x + w + pos.slopX;
  const inY = pos.y >= y - pos.slopY && pos.y <= y + h + pos.slopY;
  if (inX && inY) {
    const edges = (nearL ? "w" : nearR ? "e" : "") + (nearT ? "n" : nearB ? "s" : "");
    if (edges) return edges;
    if (pos.x > x && pos.x < x + w && pos.y > y && pos.y < y + h) return "move";
  }
  return null;
}

const CROP_CURSORS = {
  nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize",
  n: "ns-resize", s: "ns-resize", w: "ew-resize", e: "ew-resize", move: "move",
};

const clamp01 = (v, size) => Math.min(Math.max(v, 0), 1 - size);

// What the frame will actually export, in whole pixels. This is the box block of
// `crop_geometry` in server/effects.py — including its "at least one pixel each
// way" clamp — deliberately spelled twice, for the same reason `_curve_lut` is:
// the panel has to name the number before anything is saved, and asking the
// server per pointermove is a round trip per drag. Change one, change the other.
// The *canvas* is not recomputed here; it comes from the proxy response, so
// PIL's rotate-expand rounding stays where it is known.
function frameSizePx(rect, canvas) {
  const [cw, ch] = canvas;
  const [x, y, w, h] = rect;
  const left = pyRound(x * cw), top = pyRound(y * ch);
  const right = Math.max(left + 1, pyRound((x + w) * cw));
  const bottom = Math.max(top + 1, pyRound((y + h) * ch));
  return [right - left, bottom - top];
}

// Python's round(): half to *even*, where Math.round is half up. An edge landing
// on an exact half pixel is rare (0.625 × 3604) but not never — about one frame
// in three hundred — and a readout that exists to be believed should not be off
// by a pixel from the file it describes.
function pyRound(v) {
  const f = Math.floor(v);
  const diff = v - f;
  if (diff !== 0.5) return diff > 0.5 ? f + 1 : f;
  return f % 2 === 0 ? f : f + 1;
}

// Repaint the frame, and the straighten line when one is being drawn. Everything
// is drawn in the *displayed image's* pixel space (the viewBox), so the fractions
// in the store survive the natural-size change that straightening causes.
function drawCropOverlay(rect, line) {
  const svg = $("crop-overlay");
  const img = $("preview");
  const iw = img.naturalWidth;
  const ih = img.naturalHeight;
  if (!iw || !ih) return;
  // intrinsic dimensions: this is what makes the SVG resolve to the same box as
  // the fitted img, with no measurement (see #crop-overlay in style.css)
  svg.setAttribute("width", iw);
  svg.setAttribute("height", ih);
  svg.setAttribute("viewBox", `0 0 ${iw} ${ih}`);
  svg.replaceChildren();

  const [fx, fy, fw, fh] = rect;
  const x = fx * iw, y = fy * ih, w = fw * iw, h = fh * ih;
  // handles keep a roughly constant size on screen, so they stay grabbable on a
  // 24-megapixel image and do not swallow the frame on a small one
  const box = svg.getBoundingClientRect();
  const unit = box.width ? iw / box.width : 1;

  // An invisible but *painted* full-size rect, purely as a hit target. SVG hit
  // testing only finds painted geometry, and at a full-frame rect the dim path
  // below encloses zero area — so without this the first "drag across the image"
  // gesture, the one the whole tool opens with, would fall straight through to
  // the page. `fill-opacity: 0` still counts as painted; `fill: none` would not.
  svg.appendChild(svgEl("rect", {
    x: 0, y: 0, width: iw, height: ih, fill: "#000", "fill-opacity": 0,
  }));
  // everything outside the frame, dimmed — one evenodd path rather than four
  // rects, so the frame's interior is a hole and nothing overlaps at the seams
  svg.appendChild(svgEl("path", {
    d: `M 0 0 H ${iw} V ${ih} H 0 Z M ${x} ${y} H ${x + w} V ${y + h} H ${x} Z`,
    "fill-rule": "evenodd", fill: "#000", "fill-opacity": 0.55,
  }));
  for (let i = 1; i < 3; i++) {
    svg.appendChild(svgEl("path", {
      d: `M ${x + (w * i) / 3} ${y} V ${y + h} M ${x} ${y + (h * i) / 3} H ${x + w}`,
      stroke: "#fff", "stroke-opacity": 0.35, "stroke-width": unit,
    }));
  }
  svg.appendChild(svgEl("rect", {
    x, y, width: w, height: h,
    fill: "none", stroke: "#fff", "stroke-width": 1.5 * unit,
  }));
  const s = CROP.handlePx * unit;
  for (const [hx, hy] of [
    [x, y], [x + w / 2, y], [x + w, y],
    [x, y + h / 2], [x + w, y + h / 2],
    [x, y + h], [x + w / 2, y + h], [x + w, y + h],
  ]) {
    svg.appendChild(svgEl("rect", {
      x: hx - s / 2, y: hy - s / 2, width: s, height: s,
      fill: "#fff", stroke: "#000", "stroke-opacity": 0.6, "stroke-width": unit,
    }));
  }
  if (!line) return;
  // The straighten line, drawn last so it sits over the dimming. A white core in
  // a dark casing, like the mask outline: it has to read on any photograph, and
  // it is dragged across the whole frame. End caps mark the two points, since a
  // line laid along a horizon is otherwise hard to see against it.
  const [x0, y0, x1, y1] = [line.x0 * iw, line.y0 * ih, line.x1 * iw, line.y1 * ih];
  const d = `M ${x0} ${y0} L ${x1} ${y1}`;
  for (const [stroke, width] of [["#000", 4 * unit], ["#fff", 1.5 * unit]]) {
    svg.appendChild(svgEl("path", { d, stroke, "stroke-width": width, fill: "none" }));
  }
  for (const [cx, cy] of [[x0, y0], [x1, y1]]) {
    svg.appendChild(svgEl("circle", {
      cx, cy, r: 3 * unit,
      fill: "#fff", stroke: "#000", "stroke-opacity": 0.6, "stroke-width": unit,
    }));
  }
}

// How far a drawn line is off horizontal, in degrees, positive when it falls to
// the right — which is also how far counter-clockwise the image must turn to
// level it, PIL's direction. Measured in *displayed* pixels rather than in the
// stored fractions: the two axes have different denominators, so an angle taken
// straight from fractions would be skewed by the image's aspect ratio.
function lineAngle(line) {
  const r = $("preview").getBoundingClientRect();
  let dx = (line.x1 - line.x0) * r.width;
  let dy = (line.y1 - line.y0) * r.height;
  if (dx < 0) { dx = -dx; dy = -dy; } // a line has no direction
  return { deg: (Math.atan2(dy, dx) * 180) / Math.PI, len: Math.hypot(dx, dy) };
}

// One pointerdown -> one gesture. Pointer capture on the overlay, like the curve
// editor's canvas, so nothing leaks a window-level listener per rebuild.
function initCropOverlay() {
  const svg = $("crop-overlay");
  let drag = null;

  svg.addEventListener("pointermove", (e) => {
    if (!cropTool.armed || drag) return;
    // the straighten tool owns the whole surface, so the frame's edges and
    // corners stop advertising themselves while it is armed
    const mode = crop.level ? null : cropHit(cropTool.armed.rect, cropAtEvent(e));
    svg.style.cursor = CROP_CURSORS[mode] || "crosshair";
  });

  svg.addEventListener("pointerdown", (e) => {
    const tool = cropTool.armed;
    if (!tool || e.button !== 0) return;
    e.preventDefault(); // or the wrapper's pan handler takes the drag
    e.stopPropagation();
    const pos = cropAtEvent(e);
    if (crop.level) {
      // straightening: this gesture draws a line and leaves the frame alone
      drag = { mode: "level" };
      crop.line = { x0: pos.x, y0: pos.y, x1: pos.x, y1: pos.y };
      tool.paint();
      svg.setPointerCapture(e.pointerId);
      return;
    }
    let mode = cropHit(tool.rect, pos);
    // A frame filling the canvas has no slack to slide into, so "move" would be
    // a no-op — and that is the *opening* state, which would leave "drag on the
    // image to frame" doing nothing at all. Fall through to a fresh frame
    // instead. Edges and corners still resize, and a frame with slack in either
    // axis still moves.
    if (mode === "move" && tool.rect[2] >= 1 && tool.rect[3] >= 1) mode = null;
    // on the dimmed area: start a fresh frame from this corner, which is the
    // "drag across the image" gesture
    drag = mode
      ? { mode, start: pos, rect: [...tool.rect] }
      : { mode: "se", start: pos, rect: [pos.x, pos.y, CROP.minFrac, CROP.minFrac] };
    if (!mode) tool.commit(drag.rect);
    svg.setPointerCapture(e.pointerId);
  });

  svg.addEventListener("pointermove", (e) => {
    const tool = cropTool.armed;
    if (!drag || !tool) return;
    const pos = cropAtEvent(e);
    if (drag.mode === "level") {
      // deliberately unclamped: pinning the far end to the edge would pin one
      // axis and not the other, which would bend the very angle being measured
      crop.line = { ...crop.line, x1: pos.x, y1: pos.y };
      tool.paint(); // the panel shows the pending angle — no render, so no lag
      return;
    }
    const dx = pos.x - drag.start.x;
    const dy = pos.y - drag.start.y;
    let [x, y, w, h] = drag.rect;
    if (drag.mode === "move") {
      // moving keeps the size and slides inside the image
      x = clamp01(x + dx, w);
      y = clamp01(y + dy, h);
    } else {
      // resizing works on edges, then normalizes — dragging one edge past its
      // opposite flips the frame rather than pinning it at zero
      let l = x, t = y, r = x + w, b = y + h;
      if (drag.mode.includes("w")) l = Math.min(Math.max(pos.x, 0), 1);
      if (drag.mode.includes("e")) r = Math.min(Math.max(pos.x, 0), 1);
      if (drag.mode.includes("n")) t = Math.min(Math.max(pos.y, 0), 1);
      if (drag.mode.includes("s")) b = Math.min(Math.max(pos.y, 0), 1);
      [x, w] = l <= r ? [l, r - l] : [r, l - r];
      [y, h] = t <= b ? [t, b - t] : [b, t - b];
      w = Math.max(w, CROP.minFrac);
      h = Math.max(h, CROP.minFrac);
      x = clamp01(x, w);
      y = clamp01(y, h);
    }
    tool.commit([x, y, w, h]);
  });

  const endDrag = (e, apply) => {
    if (!drag) return;
    const wasLevel = drag.mode === "level";
    drag = null;
    svg.releasePointerCapture(e.pointerId);
    if (!wasLevel) return;
    const line = crop.line; // null if crop mode exited under the pointer
    crop.line = null;
    // a tap is not a line: drop it and stay armed, so a stray click on the image
    // does not silently throw the framing away
    if (!apply || !line || lineAngle(line).len < CROP.minLinePx) {
      cropTool.armed?.paint();
      return;
    }
    // a *delta*: the proxy on screen is already turned by crop.angle, so a line
    // drawn on it is measured in that rotated canvas — which is what makes
    // straightening iterative, a second line refining the first
    const next = Math.min(90, Math.max(-90, crop.angle + lineAngle(line).deg));
    crop.angle = Math.round(next * 10) / 10; // CROP_SPEC's step
    crop.level = false; // one line, then back to framing
    renderCropControls();
    refreshCropProxy(); // one gesture, one render — no debounce to lag behind
  };
  svg.addEventListener("pointerup", (e) => endDrag(e, true));
  svg.addEventListener("pointercancel", (e) => endDrag(e, false));

  // The proxy changes size whenever the angle does, so both the overlay's
  // intrinsic dimensions and the panel's pixel readout have to be re-derived
  // from whatever just loaded — hence paint(), not drawCropOverlay() alone.
  $("preview").addEventListener("load", () => cropTool.armed?.paint());

  // show(), not showModal(): the frame is dragged on the preview underneath,
  // which showModal() would make inert — the same constraint openEdit() lives
  // under, taken the other way. Opening the dialog and entering crop mode are
  // one thing, so `close` is the single teardown path however it was reached;
  // exitCropMode() closes the dialog in turn, and its own `crop.active` guard is
  // what stops the two re-entering each other.
  $("frame-btn").onclick = () => {
    $("crop-modal").show();
    enterCropMode();
  };
  $("crop-modal").addEventListener("close", () => exitCropMode(true));
  $("crop-close-btn").onclick = () => $("crop-modal").close();
  // Only *modal* dialogs get Esc from the platform, so the one dialog that
  // deliberately is not one has to listen for it — otherwise Frame would be the
  // single panel in the app that Esc could not dismiss. The `:modal` guard is
  // for the case where something else is up: that dialog owns the key.
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || !$("crop-modal").open) return;
    if (!document.querySelector("dialog:modal")) $("crop-modal").close();
  });
  $("crop-save-btn").onclick = saveCrop;
  $("crop-reset-btn").onclick = () => {
    crop.angle = 0;
    crop.rect = [0, 0, 1, 1];
    crop.level = false;
    crop.line = null;
    renderCropControls();
    refreshCropProxy();
  };
}

// The panel: the angle with its straighten tool, a frame readout, and the hint.
// Rebuilt whenever the numbers change, and it is what arms the overlay — so,
// like appendSelectionControls, a rebuild is the moment a stale commit closure
// would start writing into a detached row, and it disarms first.
//
// There is no angle slider. A slider's every step scheduled a fresh server
// rotation of the proxy, so the image lurched along behind the thumb; and an
// angle is not what a user knows anyway — they know where the horizon is. The
// Straighten tool takes that directly, and costs exactly one render per line.
function renderCropControls() {
  const box = $("crop-controls");
  box.replaceChildren();
  disarmCrop();
  if (!crop.active) return;

  const row = document.createElement("div");
  row.className = "param-row";
  const label = document.createElement("label");
  const name = document.createElement("span");
  name.textContent = "Rotate (° ccw)";
  const val = document.createElement("span");
  label.append(name, val);
  const levelBtn = document.createElement("button");
  levelBtn.type = "button";
  levelBtn.className = "crop-level" + (crop.level ? " armed" : "");
  levelBtn.textContent = crop.level ? "Cancel" : "Straighten";
  levelBtn.onclick = () => {
    crop.level = !crop.level;
    crop.line = null;
    renderCropControls();
  };
  row.append(label, levelBtn);

  const readout = document.createElement("div");
  readout.className = "hint";
  const hint = document.createElement("div");
  hint.className = "hint";
  hint.textContent = crop.level
    ? "drag a line along what should be horizontal"
    : "drag on the image to frame · drag the handles to adjust";

  box.append(row, readout, hint);

  function paint() {
    // the pending line moves the angle on screen before anything is rendered,
    // which is the whole of the tool's feedback while the pointer is down
    const pending = crop.line ? lineAngle(crop.line).deg : 0;
    val.textContent = (crop.angle + pending).toFixed(1);
    // the true output size, never the proxy's: `canvas` is the full-size rotated
    // canvas the server reported for these pixels, so this is what will export
    const size = crop.canvas ? frameSizePx(crop.rect, crop.canvas) : null;
    const px = size ? ` · ${size[0]}×${size[1]} px` : "";
    readout.textContent =
      `${Math.round(crop.rect[2] * 100)}% × ${Math.round(crop.rect[3] * 100)}%${px}`;
    if (cropTool.armed) drawCropOverlay(crop.rect, crop.line);
  }

  cropTool.armed = {
    rect: crop.rect,
    paint,
    commit(next) {
      crop.rect = next;
      cropTool.armed.rect = next;
      paint();
    },
  };
  showCropOverlay(true);
  $("preview-wrap").classList.add("cropping");
  paint();
}

function enterCropMode() {
  if (state.imageId === null || crop.active) return;
  // crop mode owns the preview image, so the effect preview has to let go of it
  exitPreview(false);
  clearMaskOverlay(); // the framed outline means nothing over an unframed proxy
  crop.active = true;
  const saved = state.crop || {};
  crop.angle = saved.angle || 0;
  crop.rect = Array.isArray(saved.rect) ? [...saved.rect] : [0, 0, 1, 1];
  crop.level = false;
  crop.line = null;
  crop.canvas = null; // the first proxy response says how big these pixels are
  renderCropControls();
  refreshCropProxy();
}

// `repaint` is false when renderSelection() is the caller: it is about to redraw
// everything anyway, and re-entering it here would recurse.
function exitCropMode(repaint) {
  if (!crop.active) return;
  crop.active = false;
  crop.seq++; // drop any proxy still in flight
  crop.level = false;
  crop.line = null;
  crop.canvas = null;
  disarmCrop();
  $("crop-controls").replaceChildren();
  $("crop-status").classList.remove("busy");
  if (crop.url) {
    URL.revokeObjectURL(crop.url);
    crop.url = null;
  }
  $("crop-modal").close(); // a no-op when the close event is what got us here
  if (repaint) renderSelection();
}

// Undebounced on purpose: with the slider gone, the only thing that changes the
// angle is a completed straighten line, so one gesture is one render.
async function refreshCropProxy() {
  if (!crop.active || state.nodeId === null) return;
  const seq = ++crop.seq;
  $("crop-status").classList.add("busy");
  try {
    const res = await fetch(`/api/images/${state.imageId}/crop-preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_id: state.nodeId, angle: crop.angle }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    if (seq !== crop.seq || !crop.active) return; // stale response or exited
    // The full-size canvas these downscaled pixels stand for. Set *before* the
    // src, so the load-driven repaint already reports the new output size.
    const cw = Number(res.headers.get("X-Canvas-Width"));
    const ch = Number(res.headers.get("X-Canvas-Height"));
    crop.canvas = cw > 0 && ch > 0 ? [cw, ch] : null;
    const url = URL.createObjectURL(blob);
    $("preview").src = url; // the load handler redraws the frame at the new size
    if (crop.url) URL.revokeObjectURL(crop.url);
    crop.url = url;
  } catch (err) {
    console.warn("crop preview failed:", err); // keep the last frame, as preview does
  } finally {
    $("crop-status").classList.remove("busy");
  }
}

async function saveCrop() {
  if (!crop.active) return;
  const body = { crop: { angle: crop.angle, rect: crop.rect } };
  try {
    const image = await api(`/api/images/${state.imageId}/crop`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.crop = image.crop;
    state.geometry = image.geometry;
  } catch (err) {
    alert(`Could not save the frame: ${err.message}`);
    return;
  }
  // every node's framed output just changed, so the overlay cache — which is
  // keyed on the selection, not the crop — would otherwise serve outlines
  // framed the old way
  clearOverlayCache();
  // and the gallery's thumb URLs carry the crop, so they need rebuilding from
  // the refreshed image rows or the tile keeps the old framing
  await refreshGallery();
  exitCropMode(true);
}

// ---------- Click-to-segment selection ----------

// One overlay, one armed picker — the same single-slot model as `preview`:
// whichever controls instance (Apply panel or edit modal) last drove it owns it.
// `armed.owner` is the button that lit itself, so "press the lit button to
// cancel" means the lit one: two different controls arm this slot (selecting an
// object, and bokeh's focus pick) and each must be able to take it in one press.
// `cache` maps an overlay's identity to its blob URL and owns every URL's
// lifetime, so nothing here revokes on replacement — only on a cache clear.
const selPicker = { armed: null, seq: 0, cache: new Map() };

function disarmPick() {
  selPicker.armed = null;
  $("preview-wrap").classList.remove("picking");
  document.querySelectorAll(".sel-pick").forEach((b) => {
    b.classList.remove("armed");
    // A button that rewrote its own label to say it was armed puts it back;
    // the selection's own does not need this, since render(sel) sets its text
    // from the store every time. Dispatched to every pick button rather than
    // handled here, so this stays ignorant of what any of them say.
    b.dispatchEvent(new Event("pickdisarmed"));
  });
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

// Latched for prepareDepthModel()'s reason: the round trip happens once a
// session rather than on every Find.
let selectReady = false;

// Fetch the model behind "name an object", reporting progress through `say`.
//
// Unlike prepareDepthModel() this reports into the caller's own line rather
// than #preview-status: the control that waits for this is built into the edit
// modal too, where that status line is not on screen. It returns whether the
// model is usable, so the caller decides what not to do — falling through to
// the request would 409 and say something less useful than the reason.
//
// That is also why this needs none of prepareDepthModel()'s staleness guard:
// its line is shared, so a poll that outlives its reason overwrites something
// else's message, where `say` here writes into one control's own span and a
// rebuild simply detaches it.
async function prepareSelectModel(say) {
  if (selectReady) return true;
  try {
    let job = await api("/api/select-model/prepare", { method: "POST" });
    while (job.state === "running") {
      say(selectJobText(job));
      await new Promise((done) => setTimeout(done, EMBED_POLL_MS));
      job = await api("/api/select-model/progress");
    }
    if (!job.ready) throw new Error(job.error || "not available");
  } catch (err) {
    // Reported and not retried, for runEmbedSearch()'s reason: falling through
    // would repeat the whole download to reach the same error.
    say(`Could not load the object model: ${err.message}`);
    return false;
  }
  selectReady = true;
  return true;
}

// One phase and one unit, like textJobText() and depthJobText() — and separate
// from both for their reason, that naming the model is most of what the
// sentence is for.
function selectJobText(job) {
  if (job.phase === "download") {
    const mb = (bytes) => Math.round(bytes / 1e6);
    const of = job.total ? ` of ${mb(job.total)}` : ""; // no Content-Length
    return `Downloading the object model — ${mb(job.done)}${of} MB (first use only)`;
  }
  return "Loading the object model…";
}

// The appendBlendTarget analogue for selections: any effect can be masked, so
// like blend's target this is not a registry param. The state lives in `store`
// — `state.selection` for the Apply panel, `edit.selection` for the modal — so
// that the two forms can be on screen at once. Every user change dispatches a
// bubbling `input` event, so the existing delegated listeners on the containers
// drive the shared preview debounce with no new wiring.
//
// The saved-mask grid is part of this control rather than a section of its own:
// choosing masks *is* the selection, so keeping the picker, the level/invert
// options, Save, and the saved objects in one component is what stops the
// panel's rendered state and the store from drifting apart. `allowManage` is off
// in the edit modal — deleting a mask the node uses would 409 anyway.
function appendSelectionControls(
  container, { store, sourceNodeId, allowPick = true, allowManage = false }
) {
  // rebuilding the control is exactly when a previously armed picker becomes
  // invalid: its commit closure now writes into a detached row
  disarmPick();

  const row = document.createElement("div");
  row.className = "param-row";

  // No heading: the pick button says "Select object" itself, so a title above it
  // only repeated the control. What is left is the summary of what is picked —
  // a bare span rather than a <label>, since it now labels nothing.
  const coords = document.createElement("span");
  coords.className = "sel-coords";

  const pick = document.createElement("button");
  pick.type = "button";
  pick.className = "sel-pick";
  pick.title = allowPick
    ? "Then click an object in the image to select it"
    : "The click point was made on this node's input — re-picking needs a new node from the Apply panel";
  const clear = document.createElement("button");
  clear.type = "button";
  clear.className = "sel-clear";
  clear.textContent = "clear";
  const pickRow = document.createElement("div");
  pickRow.className = "sel-row";
  pickRow.append(pick, clear);

  // The other way to say which object: name it. A typed query resolves on the
  // server to the point a click would have made, so everything below this line
  // — the level, invert, Save, the overlay — cannot tell the two apart, and
  // this needs no state of its own beyond the words in the box.
  const query = document.createElement("input");
  query.type = "text";
  query.className = "sel-text";
  query.placeholder = "or name an object…";
  const find = document.createElement("button");
  find.type = "button";
  find.className = "sel-find";
  find.textContent = "Find";
  const findRow = document.createElement("div");
  findRow.className = "sel-row";
  findRow.append(query, find);
  // What the search found, in words: the outline says *where*, this says how
  // sure. Inline rather than an alert, which would block a control you press
  // repeatedly, and empty most of the time — `:empty` hides it.
  const note = document.createElement("span");
  note.className = "sel-note";

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

  row.append(coords, pickRow, findRow, optRow);

  // Save, then the saved objects themselves — the order the workflow runs in:
  // pick an object, use it (which banks it), repeat, then tick the ones this
  // effect applies to.
  const save = document.createElement("button");
  save.type = "button";
  save.className = "sel-save-mask";
  save.textContent = "Save selection";
  if (allowManage) row.appendChild(save);
  // after the controls and before the strip: in the bar the note takes a line
  // of its own, and putting it here keeps Save on the line with the pickers
  // rather than pushed below a sentence.
  row.appendChild(note);

  const list = document.createElement("ul");
  list.className = "sel-mask-list";
  row.appendChild(list);
  container.appendChild(row);

  const current = () => store.value;
  const commit = (sel) => {
    // every path through here supersedes whatever the last search reported
    note.textContent = "";
    store.value = sel;
    store.nodeId = sel ? sourceNodeId : null;
    store.imageId = sel ? state.imageId : null;
    render(sel);
    syncMaskOverlay(store, sourceNodeId, pick);
    // a preset run is limited to this selection too, and its rows say which
    // object — this is the write path renderSelection() does not cover, since
    // the store changes without the node or image changing
    updatePresetControls();
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
      : "Select an object and apply an effect — it gets saved here for reuse.";
    list.appendChild(empty);
  }
  for (const mask of state.masks) {
    const li = document.createElement("li");
    li.className = "mask-chip";
    // the silhouette carries the identity, so the words live in the tooltip
    li.title = [
      mask.name,
      `${mask.width}×${mask.height}`,
      mask.used_by
        ? `used by ${mask.used_by} node${mask.used_by === 1 ? "" : "s"} — delete those first to remove it`
        : "click to include it in the selection",
    ].join(" · ");

    const img = document.createElement("img");
    // created_at busts the cache: the icon is served with a year-long max-age,
    // but mask ids are rowids and SQLite reuses them
    img.src = `/api/masks/${mask.id}/thumb?v=${encodeURIComponent(mask.created_at)}`;
    img.alt = mask.name;
    img.loading = "lazy";
    img.draggable = false;

    const box = document.createElement("input");
    box.type = "checkbox";
    box.tabIndex = -1; // the chip owns the click; the box is the indicator

    li.append(img, box);
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

  // The typed counterpart of the pick closure below, and it commits the same
  // shape: a point and a level, which is all a click ever produced either.
  //
  // `prepareSelectModel` runs first rather than after a 409, the way
  // runEmbedSearch() calls the text model's prepare before searching — the
  // server answers `done` synchronously once the files are there, so this is one
  // extra request on the first Find of a session and none after it.
  let finding = false;
  async function findObject() {
    const text = query.value.trim();
    if (!text || finding || sourceNodeId === null) return;
    finding = true;
    render(current()); // the row goes busy the same way it comes back
    const say = (message) => (note.textContent = message);
    try {
      say(`Looking for “${text}”…`);
      if (!(await prepareSelectModel(say))) return;
      const found = await api(`/api/nodes/${sourceNodeId}/select-text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text }),
      });
      // The server's level rather than the dropdown's, which `render()` then
      // pushes into the control: it was chosen to match how much of the frame
      // the phrase covers, and that is the whole of how "the sky" and "a cloud"
      // end up as different selections from the same kind of click.
      commit({
        points: [{ x: found.x, y: found.y, level: found.level }],
        invert: invert.checked,
      });
      const covers = `${(found.coverage * 100).toFixed(1)}% of the frame`;
      say(
        found.confident
          ? `Found “${text}” — ${covers}.`
          : `Weak match for “${text}” — ${covers}. Check the outline.`
      );
    } catch (err) {
      say(`Could not find that: ${err.message}`);
    } finally {
      // A repaint, not two assignments: `commit()` above re-rendered the whole
      // control while this was still running, so the row is currently painted
      // busy and only running it again with the flag down puts it back.
      finding = false;
      render(current());
    }
  }

  function render(sel) {
    const points = selPoints(sel);
    coords.textContent = selSummary(sel);
    pick.textContent = points.length ? "Re-pick object" : "Select object";
    pick.disabled = !allowPick;
    // Naming an object needs no click on the picture, so unlike the picker this
    // stays live in the edit modal — showModal() making the image inert is the
    // whole of why that one is disabled there.
    query.disabled = find.disabled = finding || sourceNodeId === null;
    // a frozen mask has no candidate masks left to re-rank
    level.value = points.length === 1 ? points[0].level : "auto";
    level.disabled = points.length !== 1;
    invert.checked = sel ? !!sel.invert : false;
    invert.disabled = clear.disabled = !sel;
    // a saved mask is already saved; only a fresh pick has anything to freeze
    save.disabled = maskBusy || state.imageId === null || !points.length;
    save.title = save.disabled
      ? "Select an object first — a saved mask is already saved"
      : "Save this object now — applying an effect to it saves it too";
    const chosen = selMasks(sel);
    for (const { mask, li, box } of rows) {
      box.checked = chosen.includes(mask.id);
      li.classList.toggle("selected", box.checked);
      li.classList.toggle("disabled", maskBusy);
    }
  }

  if (allowPick) {
    pick.onclick = () => {
      // pressing the *lit* button cancels; pressing a different one takes the
      // slot over, so arming this never costs two presses because something
      // else (the Focus pick) was waiting on a click
      if (selPicker.armed?.owner === pick) {
        disarmPick();
        return;
      }
      disarmPick();
      // single-shot: the click handler in initZoom does not disarm, so the
      // closure does it before committing
      selPicker.armed = {
        owner: pick,
        pick: (x, y) => {
          disarmPick();
          commit({ points: [{ x, y, level: level.value }], invert: invert.checked });
        },
      };
      pick.classList.add("armed");
      $("preview-wrap").classList.add("picking");
    };
  }
  find.onclick = findObject;
  query.onkeydown = (event) => {
    // Enter is the gesture the box invites; there is no form to submit, so
    // nothing else would happen to it.
    if (event.key === "Enter") {
      event.preventDefault();
      findObject();
    }
  };
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
    // the title names the ticked object, which changes without a rebuild —
    // updatePresetControls owns it, and finds the preset back through this id
    li.dataset.presetId = preset.id;

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
  // A ticked object limits the whole chain, so the row has to say so — and this
  // runs on every selection change (renderSelection calls it), unlike the row
  // build, which only runs when the preset list itself changes.
  const limit = selSummary(state.selection.value);
  document.querySelectorAll("#preset-list .preset-row").forEach((li) => {
    li.classList.toggle("disabled", state.imageId === null || presetBusy);
    const preset = state.presets.find((p) => p.id === +li.dataset.presetId);
    if (!preset) return;
    li.title = limit
      ? `Apply to the selected node, limited to ${limit}: ${preset.summary}`
      : `Apply to the selected node: ${preset.summary}`;
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
  // A ticked object limits every step of the chain, exactly as it limits a
  // single effect — so this banks the pick first, for the same reasons
  // applyEffect does (and here it also means one mask for the whole replay
  // rather than one per masked step). A failed bank still runs the preset,
  // with the click selection, and says so once afterwards.
  let banked = state.selection.value;
  let bankErr = null;
  try {
    banked = await bankSelection(state.selection);
  } catch (err) {
    bankErr = err;
  }
  const body = { preset_id: preset.id };
  if (banked) body.selection = banked;
  try {
    const res = await api(`/api/nodes/${state.nodeId}/apply-preset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    // selectImage refetches the tree, so all of the preset's new nodes show up
    await selectImage(state.imageId, res.terminal_node_id);
    if (bankErr) {
      alert(
        "The preset was applied, but the selection could not be saved for reuse: " +
          bankErr.message
      );
    }
  } catch (err) {
    // as in applyEffect: a bank that succeeded stays, but the grid has not
    // drawn its tile yet
    renderSelectControls();
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

// Freeze a store's click selection and swap the store over to the frozen
// pixels. This is what makes a picked object durable: a {points} selection dies
// with its node in pruneSelection(), because its coordinates only mean anything
// in that node's pixel space, while a {masks} one lives as long as the image.
//
// So every path that *uses* a pick banks it first — which is also what stops the
// next Apply saving a second copy of the same object, since it now finds the
// mask shape and no-ops. Doing this here rather than in the create_node endpoint
// is deliberate: from the server the store would still be holding the points, so
// every Apply would freeze another duplicate, and preset replay (which builds
// nodes through db.create_node) would mint a mask per masked step.
//
// The server names it. `invert: false` because the PNG is frozen post-invert —
// what was on screen is what got saved, and a reference's own invert toggles on
// top of that.
async function bankSelection(store) {
  const sel = store.value;
  if (!sel || sel.masks || state.imageId === null) return sel;
  const created = await api(`/api/images/${state.imageId}/masks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ node_id: store.nodeId, selection: sel }),
  });
  const value = { masks: [created.id], invert: false };
  Object.assign(store, { value, nodeId: state.nodeId, imageId: state.imageId });
  // Adopt the row locally rather than waiting for the refetch: a caller whose
  // *next* step fails (Apply erroring after a successful bank) would otherwise
  // leave the store naming a mask the grid has no tile for, and selSummary()
  // falling back to `mask #12`. A new id also changes what a cached overlay key
  // means, since rowids come back around.
  state.masks = [...state.masks, created];
  clearOverlayCache();
  return value;
}

// The explicit button: bank an object you are not ready to use yet. Applying an
// effect banks the selection too, so this is a shortcut, not the only way in.
async function saveMask() {
  const sel = state.selection.value;
  if (!sel || sel.masks || state.imageId === null) return;
  maskBusy = true;
  renderSelectControls();
  try {
    await bankSelection(state.selection);
    maskBusy = false;
    await refreshMasks();
  } catch (err) {
    alert(`Could not save the selection: ${err.message}`);
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
    statRow(body, effectLabel(e.effect), e.count, true);
  }

  const st = s.storage;
  statHeading(body, "On disk");
  statRow(body, "Database", formatBytes(st.database.bytes));
  statRow(body, "Originals", `${formatBytes(st.originals.bytes)} · ${st.originals.files} files`);
  statRow(body, "Masks", `${formatBytes(st.masks.bytes)} · ${st.masks.files} files`);
  statRow(body, "Models", `${formatBytes(st.models.bytes)} · ${st.models.files} files`);
  const cacheBytes =
    st.renders.bytes + st.outputs.bytes + st.thumbs.bytes +
    st.clusters.bytes + st.embeddings.bytes;
  statRow(body, "Render cache", formatBytes(cacheBytes));
  statRow(body, "renders", `${formatBytes(st.renders.bytes)} · ${st.renders.files} files`, true);
  statRow(body, "framed output", `${formatBytes(st.outputs.bytes)} · ${st.outputs.files} files`, true);
  statRow(body, "thumbnails", `${formatBytes(st.thumbs.bytes)} · ${st.thumbs.files} files`, true);
  statRow(body, "cluster data", `${formatBytes(st.clusters.bytes)} · ${st.clusters.files} files`, true);
  statRow(body, "embeddings", `${formatBytes(st.embeddings.bytes)} · ${st.embeddings.files} files`, true);
  statRow(
    body,
    "Total",
    formatBytes(
      st.database.bytes + st.originals.bytes + st.masks.bytes +
      st.models.bytes + cacheBytes
    )
  );

  const note = document.createElement("div");
  note.className = "hint";
  note.textContent =
    "The render cache rebuilds itself from the originals and the work tree, and the models re-download — only the database, originals, and saved masks are irreplaceable.";
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
  // allowPick is off for the same inertness reason the selection's is, below
  buildParamControls($("edit-params"), node.effect, node.params || {}, node.parent_id, {
    allowPick: false,
  });
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
  // and hand the preview back too, to whatever feature button was left lit. The
  // Save path re-enters via selectImage(), but Cancel and Esc end here, and the
  // panel's own preview should survive both.
  if (state.effect && restoreSrc) enterApplyPreview();
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
  // Off, like the map's: a plot that is moving while you are reading it is
  // harder to read, and the depth the spin conveys is one drag away. Double-click
  // turns it on, and it stays on for the session.
  spin: false,
  raf: null,
};

async function updateClusterPlot() {
  const node = state.nodes.find((n) => n.id === state.nodeId);
  const show = !!node && node.effect === "posterize";
  // the plot lives in a dialog now, so selecting a posterize node only offers
  // it — it does not open it
  $("cluster-btn").hidden = !show;
  if (!show) {
    $("cluster-modal").close();
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
  // Only spin while the dialog is up: the loop is animation, and animating
  // behind a closed dialog is a frame budget spent on nothing. Moving between
  // posterize nodes with the plot open re-points it at the new data in place.
  if ($("cluster-modal").open) startClusterLoop();
}

function startClusterLoop() {
  if (cluster.raf) return;
  const tick = () => {
    // The loop ends itself when the dialog goes away, rather than trusting every
    // close path to remember to call stopClusterLoop(). One condition, checked
    // where the cost is actually paid, is what keeps "it only spins while you
    // are looking at it" true no matter how the dialog got dismissed.
    if (!$("cluster-modal").open) {
      cluster.raf = null;
      return;
    }
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

  $("cluster-btn").onclick = () => {
    $("cluster-modal").showModal();
    startClusterLoop();
  };
  $("cluster-close-btn").onclick = () => $("cluster-modal").close();
  // Esc closes without going through the button, and the loop would keep
  // painting a canvas nobody can see
  $("cluster-modal").addEventListener("close", stopClusterLoop);
}

// ---------- Image map (3D embedding scatter over the whole library) ----------

// The library as a point cloud, one point per image, positioned by a CLIP
// embedding so photos of similar things land near each other. Deliberately a
// sibling of the cluster plot rather than a generalization of it: they share
// the projection *math* (copied below) but nothing else — that one plots pixels
// of one node in RGB space, this one plots every image of the library in a
// fitted space, and it owns the whole viewport rather than a small dialog.
const embedMap = {
  points: [],
  // t-SNE rather than PCA: the clumps are what the map is opened to read, and
  // the fit is seeded (see main._project), so reopening shows the same layout.
  method: "tsne",
  yaw: 0.8,
  pitch: -0.45,
  // zoom multiplies the derived world→screen scale, pan offsets the projected
  // result — so pan lives in backing-store pixels, like everything else here.
  zoom: 1,
  panX: 0,
  panY: 0,
  // The world point rotation turns about, and the point pan positions: pan is
  // its screen offset from the canvas centre. The cloud's own centre until you
  // grab somewhere else.
  pivot: { x: 0, y: 0, z: 0 },
  // Whether the pivot follows the pick instead of the grab (see
  // pivotOnSelection). Outlives an open, like spriteScale and matchZ — it is a
  // taste for how turning the cloud should work, not part of where you are
  // looking. On by default because the map now opens with the image you were
  // working on picked and centred (see loadEmbedMap), so there is something to
  // turn about from the first frame; before that there never was until you
  // clicked, and the tick did nothing.
  orbit: true,
  // Off by default, and the reason is the reverse of `orbit`'s: turning is worth
  // defaulting on because it costs nothing to look at, where *moving* is a thing
  // you have to wait out before you can read a label or aim at a point. It shows
  // depth the flat projection cannot, so it stays one click (or double-click)
  // away — and outlives an open once asked for, like the two fields above.
  spin: false,
  // Whether the next frame would differ from the one on screen. With spin on by
  // default the loop could repaint unconditionally and nobody noticed; still by
  // default, that is a full-viewport redraw of the whole library, sixty times a
  // second, to produce the same picture. See startEmbedLoop.
  dirty: true,
  painted: 0,
  raf: null,
  selected: null,
  // Multiplies the sprite size, 0.1 to 1. Not folded into `zoom`: zoom moves
  // the camera and this resizes the things in front of it, so shrinking the
  // sprites opens gaps between them where zooming out only makes the same
  // collage smaller. Deliberately outlives an open, like yaw/pitch and unlike
  // the framing — it is a reading preference for a library of a given density,
  // not part of where you happen to be looking.
  spriteScale: 1,
  // Groups of similar images, each named by the nearest CLIP text label the
  // server could find (see server/labels.py). Positions are in the same
  // normalized cube as the points, so they project through the same closure.
  clusters: [],
  // How many groups to ask for. 0 means "you choose", which is what the server
  // does from the library size — so the slider is initialized from the first
  // response rather than duplicating that formula here, and only pins a value
  // once you actually drag it.
  clusterCount: 0,
  clusterTimer: null,
  // Thumbnails to draw at each point, keyed by URL (see embedSprite). Kept
  // across opens: a baked sprite is immutable for its key, so a reopened map
  // should be instant rather than re-fetching a library it already has.
  sprites: new Map(),
  // The semantic search. `scores` is image_id -> z, or null for "no search
  // running", which is the whole off state: null means every point draws, so
  // there is no second flag anywhere saying whether filtering is on.
  //
  // These reset on every open, unlike yaw/pitch/spriteScale and like the
  // framing — a query is a thing you were looking for once, not a reading
  // preference for a library of a given density.
  query: "",
  scores: null,
  // How strong a match to keep, in standard deviations above the query's own
  // mean over the library. Not a raw cosine: CLIP's image-text cosines sit in a
  // narrow band whose position moves with the phrasing, so an absolute
  // threshold means something different for every query (see main.py's
  // search_embedding_map). Outlives an open, like spriteScale — it is a taste
  // for how strict searching should feel.
  matchZ: 1.2,
  searchTimer: null,
  searchSeq: 0,
  // What the assistant found, when a turn opened the map: `{query, ids: Set}`,
  // or null for off, the same off state `edits` has. Ids and not a query
  // because the map has to show the photos the log directly above it just
  // named — re-running the words would answer a slightly different question,
  // and Match could then hide photos the assistant said it had found.
  //
  // It is also the one filter in here nobody set by hand, which is why it is
  // the only one that ships with its own visible button (`#embed-pinned`): the
  // others are the control you already pressed, this one has to say what it is
  // and how to be rid of it.
  pinned: null,
  // The Near filter's slider position, 0..1 — an exponent, not a radius (see
  // embedNearRadius), and 1 is the whole travel's worth of "off". Resets on
  // every open, like the query and unlike matchZ: hiding all but a
  // neighbourhood is something you did once, and a map that reopens showing
  // three images reads as broken. It is anchored on `selected`, so it also does
  // nothing at all until something is picked.
  nearT: 1,
  // The edit filter: a Set of tokens, or null for off — the same single-sentinel
  // shape `scores` uses, so there is still no second boolean anywhere saying
  // whether filtering is on. Tokens are effect names plus "any" (the image
  // carries at least one effect node) and "none" (it carries none), and they are
  // OR'd: this is a filter you widen by lighting more chips. Resets on every
  // open, like the query — it is a thing you were looking for once.
  edits: null,
  // `open` is the sentinel that keeps closeEmbedMap() from recursing when Esc
  // fires the dialog's `close` event, the same idiom #edit-modal uses.
  open: false,
  seq: 0,
};

const EMBED_HIT_PX = 14; // click tolerance for a point still drawn as a dot
const EMBED_IDLE_MS = 200; // longest a still map may go without a repaint
const EMBED_DRAG_PX = 4; // beyond this a pointerdown/up pair was a drag, not a click
const EMBED_POLL_MS = 400;
// A sprite's long edge, as a fraction of the unit cube the cloud is normalized
// into — so it is a world size, not a screen size, and zoom grows it exactly as
// it grows the wireframe cube.
const EMBED_SPRITE_WORLD = 0.26;
// Resolution the sprite is baked at, once, in device pixels. Blitting a
// pre-scaled canvas is much cheaper per frame than resampling a 320px JPEG,
// which is the whole cost of drawing a library sixty times a second. Sprites
// only reach this size past ~8x zoom, and are merely soft beyond it.
const EMBED_SPRITE_PX = 256;
// Long enough that typing a phrase is one request rather than one per letter,
// short enough that pausing mid-thought still shows you something. The server
// caches by phrase, so a backspace-and-retype is free anyway.
const EMBED_SEARCH_MS = 250;
// The Near slider's travel, as a radius in the same unit cube the points live
// in. Its position is an exponent rather than the radius itself: what you are
// looking for when you drag it is a neighbourhood, and at cube scale those are
// all in the bottom fifth of the range, which a linear slider crosses in one
// step. The two ends are clipped rather than extrapolated, so the two answers
// people actually want — "only this one" and "all of them" — are the ends of
// the travel instead of somewhere near them.
const EMBED_NEAR_MIN = 0.02; // radius one step up from the bottom
const EMBED_NEAR_MAX = 2 * Math.sqrt(3); // the cube's diagonal: everything

function embedNearRadius(t) {
  if (t <= 0) return 0; // only the pick itself, which is at distance 0
  if (t >= 1) return Infinity; // off
  return EMBED_NEAR_MIN * Math.pow(EMBED_NEAR_MAX / EMBED_NEAR_MIN, t);
}

// One hue per cluster, spun by the golden angle so that however many groups
// there are, adjacent ids land far apart on the wheel and no two of a dozen
// collide. Derived rather than sent: the server's job is to say which group a
// point is in, not what colour to paint it.
function clusterColor(index, alpha = 1) {
  return `hsla(${(index * 137.508) % 360}, 62%, 62%, ${alpha})`;
}

function setClusterSlider(count) {
  $("embed-clusters").value = count;
  $("embed-cluster-count").textContent = count;
}

// The readout says the radius, not the slider position — the position is an
// exponent and means nothing on its own. The ends say what they do instead of
// the numbers they would round to, since "0.02" and "3.46" are not the claims
// those ends make.
function setNearSlider(t) {
  embedMap.nearT = t;
  $("embed-near").value = t;
  const radius = embedNearRadius(t);
  $("embed-near-num").textContent =
    radius === 0 ? "0" : radius === Infinity ? "∞" : radius.toFixed(2);
}

// Zoomed all the way out there is only one sensible framing — the whole cloud,
// centred on its own centre — so 1x, "no pan" and "no pivot" are one state, and
// reaching any of them restores all three. Resetting the pivot is not optional:
// leave it on some grab point and pan 0 would centre *that*, not the cloud.
// This is also what saves the map a reset button it has no room for.
//
// Orbit is deliberately not honoured here: at 1x the whole cloud is framed and
// its own centre is the only sensible pivot, so the tick takes effect again at
// the next pick or grab rather than fighting the reframing.
function resetEmbedView() {
  embedMap.zoom = 1;
  embedMap.panX = 0;
  embedMap.panY = 0;
  embedMap.pivot = { x: 0, y: 0, z: 0 };
}

// Rotation turns the cloud about the point you grabbed rather than about its
// centre. A grab names a 2D point, so it names a whole ray through the cloud;
// this takes the point on that ray lying in the plane through the centre normal
// to the view. Depth along the ray does not change what you see now, but it does
// once you turn — and a pivot at the cloud's own depth is the one that doesn't
// make the cloud swing.
//
// Screen coordinates are backing-store pixels, as everywhere else here.
function setEmbedPivot(screenX, screenY) {
  const { w, h, scale, cy, sy, cp, sp } = embedCamera();
  const pivot = embedMap.pivot;
  // view-space depth of the cloud's centre, measured from the current pivot;
  // the new pivot has to share it, and that is the whole choice of plane
  const oz1 = -pivot.z * cy + pivot.x * sy;
  const z2 = -pivot.y * sp + oz1 * cp;
  // undo project(): screen back to view space, then the inverse rotation
  const x1 = (screenX - (w / 2 + embedMap.panX)) / scale;
  const y1 = -(screenY - (h / 2 + embedMap.panY)) / scale;
  const y = y1 * cp + z2 * sp;
  const z1 = z2 * cp - y1 * sp;
  embedMap.pivot = {
    x: pivot.x + x1 * cy - z1 * sy,
    y: pivot.y + y,
    z: pivot.z + x1 * sy + z1 * cy,
  };
  // Re-anchoring pan on the grab is what makes the switch invisible: the new
  // pivot projects exactly where it was clicked, so nothing on screen moves.
  embedMap.panX = screenX - w / 2;
  embedMap.panY = screenY - h / 2;
}

// With Orbit ticked, rotation turns about the picked image instead of about the
// grab — which is the thing you want once you have found something and only the
// view around it is still wrong. It buys the same invisibility setEmbedPivot()
// does, and gets it without any of the trig: a pivot placed *on* a point
// projects to the pan anchor by construction, and the draw loop has already
// cached where that point currently is (p.sx/p.sy, refreshed every frame).
//
// Returns whether it could — the caller falls back to the grab when it could
// not, so "orbit on, nothing picked" is not a special case anywhere else.
function pivotOnSelection() {
  const p = embedMap.selected;
  if (!p || !Number.isFinite(p.sx)) return false; // never drawn: nothing to hold still
  const { w, h } = embedCamera();
  embedMap.pivot = { x: p.x, y: p.y, z: p.z };
  embedMap.panX = p.sx - w / 2;
  embedMap.panY = p.sy - h / 2;
  return true;
}

// Put the pick in the middle of the canvas: pivot on it and drop the pan, which
// centres it by construction — the projection places the pivot at
// `w/2 + panX, h/2 + panY`.
//
// The counterpart to pivotOnSelection(), and the opposite of it: that one buys
// invisibility by re-anchoring the pan so the pick stays exactly where it
// already was, which is what you want when the view is already framed on
// something. This is for the moments where the view is not framed on anything
// yet — opening the map, or re-projecting, where the coordinates just changed
// under it. It needs no cached p.sx for the same reason, so it can run before
// the first frame has been drawn.
//
// Zoom is left alone: at 1x the whole cloud is framed, so this only slides that
// framing over to the pick.
function centerOnSelection() {
  const p = embedMap.selected;
  if (!p) return;
  embedMap.pivot = { x: p.x, y: p.y, z: p.z };
  embedMap.panX = 0;
  embedMap.panY = 0;
}

// Picking, as a gesture: the pick is also what the Near filter measures from,
// so choosing one re-filters the cloud. applyEmbedFilter()'s own
// `selectEmbedPoint(null)` deliberately does *not* go through here — that is
// what keeps the two from recursing.
function pickEmbedPoint(point) {
  selectEmbedPoint(point);
  applyEmbedFilter();
}

// `pin` is `{query, image_ids}` from an agent turn's `map`, or nothing at all
// when a person pressed the header button — see runAgent().
async function openEmbedMap(pin = null) {
  embedMap.open = true;
  embedMap.selected = null;
  // yaw/pitch persist across opens on purpose, but the framing does not: a map
  // reopened still parked at 16x on some corner reads as a broken map.
  resetEmbedView();
  // A query is part of the framing, not part of the viewpoint: reopening the
  // map still filtered to a search you finished last time hides most of the
  // library for no visible reason.
  $("embed-search").value = "";
  clearEmbedSearch();
  // Part of the framing too, and for a stronger version of the same reason: a
  // map reopened still hiding everything but one neighbourhood is a map with
  // most of the library missing and nothing on screen saying why.
  setNearSlider(1);
  // Same argument again for the edit chips, which hide even more than Near can.
  embedMap.edits = null;
  // And the strongest version of it for the assistant's pinned set: a map
  // reopened still showing the five photos a conversation two minutes ago
  // turned up is hiding the library for a reason nothing on screen gives, and
  // this is the one filter the person never chose. Cleared unconditionally
  // here and re-applied below, so one place still decides what an opened map
  // looks like and `pin` is an explicit override rather than leftover state.
  embedMap.pinned = null;
  // The two checkboxes are the other direction: their flags outlive the open, so
  // the markup has to be told what they are. Without this the box and the flag
  // are two sources of truth that agree only until you toggle one.
  $("embed-orbit").checked = embedMap.orbit;
  $("embed-spin").checked = embedMap.spin;
  $("embed-card").hidden = true;
  $("embed-status").textContent = "Preparing…";
  $("embed-modal").showModal();
  // sized only now: a closed <dialog> has a zero-size bounding rect
  sizeEmbedCanvas();
  // centred on the image you are working on — see loadEmbedMap
  if (!(await prepareEmbedMap())) return;
  await loadEmbedMap(true);
  // Only now, and only on a map that loaded: prepareEmbedMap() leaves its
  // explanation in #embed-status when the vision model could not be had, and
  // applyEmbedFilter() is the sole writer of that line — filtering over a
  // failed prepare would paint "0 of 0" over the sentence saying what went
  // wrong. Ids and nothing else, so this never touches the search box, which
  // is what keeps "only the search box fetches the 254 MB text tower" true of
  // a map the assistant opened.
  if (pin) {
    embedMap.pinned = { query: pin.query, ids: new Set(pin.image_ids) };
    applyEmbedFilter();
  }
}

// The embedding pass runs on a thread of the server's, and this polls it so the
// status line can say *what* is slow. On a fresh install that is a 335 MB model
// download, which as one blocking request was indistinguishable from a hang.
// Returns false when the map should not be drawn: either the dialog closed
// meanwhile, or the pass failed — and in that case falling through to
// /api/embedding-map would just repeat the whole download to reach the same
// error, so the failure is reported rather than retried.
async function prepareEmbedMap() {
  const seq = ++embedMap.seq;
  const stale = () => seq !== embedMap.seq || !embedMap.open;
  try {
    let job = await api("/api/embedding-map/prepare", { method: "POST" });
    while (job.state === "running") {
      if (stale()) return false;
      $("embed-status").textContent = embedJobText(job);
      await new Promise((done) => setTimeout(done, EMBED_POLL_MS));
      if (stale()) return false;
      job = await api("/api/embedding-map/progress");
    }
    if (stale()) return false;
    if (job.state === "error") {
      $("embed-status").textContent = `Could not embed the library: ${job.error}`;
      return false;
    }
    $("embed-status").textContent = "Projecting…";
    return true;
  } catch (err) {
    if (stale()) return false;
    $("embed-status").textContent = `Could not embed the library: ${err.message}`;
    return false;
  }
}

// `done`/`total` change units with the phase — bytes while downloading, images
// while embedding — so each phase gets its own sentence rather than one bar
// whose numbers silently mean something else halfway through.
function embedJobText(job) {
  const mb = (bytes) => Math.round(bytes / 1e6);
  if (job.phase === "download") {
    const of = job.total ? ` of ${mb(job.total)}` : ""; // no Content-Length
    return `Downloading the CLIP model — ${mb(job.done)}${of} MB (first run only)`;
  }
  if (job.phase === "embed" && job.total) {
    return `Embedding the library — ${job.done} of ${job.total} images`;
  }
  return "Preparing…";
}

function sizeEmbedCanvas() {
  const canvas = $("embed-canvas");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  // A backing store at device resolution, unlike the cluster canvas's fixed
  // 268x268 — at this size a 1x buffer is visibly soft. Everything downstream
  // (drawing and hit testing alike) therefore works in backing-store pixels,
  // and pointer coordinates are scaled into them on the way in.
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  // Assigning width/height also clears the backing store, so a still map would
  // otherwise sit on a blank canvas until something else happened to it.
  markEmbedDirty();
}

// `center` says the coordinates the view is framed on have just changed — the
// map is opening, or the projection was swapped — so the framing has to be
// rebuilt around the pick. A re-fetch that leaves the layout alone (the cluster
// slider) passes false, or a drag would yank the view out from under itself.
async function loadEmbedMap(center = false) {
  const seq = ++embedMap.seq;
  try {
    const data = await api(
      `/api/embedding-map?method=${embedMap.method}&clusters=${embedMap.clusterCount}`,
    );
    if (seq !== embedMap.seq || !embedMap.open) return; // stale or closed meanwhile
    embedMap.points = data.points;
    embedMap.clusters = data.clusters;
    // Adopt the count the server picked, so the slider shows the truth from the
    // first frame. Only while `clusterCount` is still 0 — past that the slider
    // is what asked for this number, and writing it back would fight a drag.
    if (!embedMap.clusterCount && data.clusters.length) {
      setClusterSlider(data.clusters.length);
    }
    // Built from the points, so only kinds of edit the library actually holds
    // get a chip. Before the pick below, which re-runs the filter the chips are
    // read by.
    renderEditChips();
    // The pick survives a re-fetch by image id, never by identity: these are
    // fresh objects, so the old one is no longer in `points` and the draw
    // loop's `p === embedMap.selected` would never match again. On open there
    // is no pick yet, and the image you are working on is where to start.
    const keepId = embedMap.selected
      ? embedMap.selected.image_id
      : center
        ? state.imageId
        : null;
    // Also the point applyEmbedFilter() measures Near from, so it is picked
    // rather than merely selected — and it writes the status line, which is why
    // nothing sets it here. These are fresh objects, so their `hidden` flags
    // start clear; a search survives the re-fetch anyway, since `scores` is
    // keyed by image_id, which re-projecting the same library cannot change.
    pickEmbedPoint(embedMap.points.find((p) => p.image_id === keepId) || null);
    if (center) centerOnSelection();
    markEmbedDirty();
    startEmbedLoop();
  } catch (err) {
    if (seq !== embedMap.seq || !embedMap.open) return;
    embedMap.points = [];
    markEmbedDirty();
    $("embed-status").textContent = `Could not build the map: ${err.message}`;
  }
}

// The status line the map wears when nothing is filtered. Shared by the two
// places that restore it, so the legend cannot drift between them.
function embedIdleStatus() {
  return embedMap.points.length
    ? `${embedMap.points.length} images · drag to rotate · scroll to zoom · right-drag to pan · click a point · double-click to spin`
    : "No images yet.";
}

// Score the library against the box's current contents.
//
// Two requests deep on a cold machine: the text tower is 254 MB and is fetched
// only now — not when the map opens — because most sessions never search, and
// nobody should pay for a model they will not use. That is also why the server
// 409s instead of downloading inside the GET, and why this polls a job for it
// exactly as prepareEmbedMap() does for the vision model.
async function runEmbedSearch() {
  const query = $("embed-search").value.trim();
  embedMap.query = query;
  if (!query) return clearEmbedSearch();

  const seq = ++embedMap.searchSeq;
  const stale = () => seq !== embedMap.searchSeq || !embedMap.open;
  try {
    let job = await api("/api/text-model/prepare", { method: "POST" });
    while (job.state === "running") {
      if (stale()) return;
      $("embed-status").textContent = textJobText(job);
      await new Promise((done) => setTimeout(done, EMBED_POLL_MS));
      if (stale()) return;
      job = await api("/api/text-model/progress");
    }
    if (stale()) return;
    if (!job.ready) {
      // Reported rather than retried, for prepareEmbedMap()'s reason: falling
      // through to the search would repeat the whole download to reach the
      // same error.
      $("embed-status").textContent =
        `Could not load the text model: ${job.error || "not available"}`;
      return;
    }
    const data = await api(`/api/embedding-map/search?q=${encodeURIComponent(query)}`);
    if (stale()) return;
    embedMap.scores = data.scores;
    applyEmbedFilter();
  } catch (err) {
    if (stale()) return;
    $("embed-status").textContent = `Could not search: ${err.message}`;
  }
}

// text_job has one phase and one unit, so this is embedJobText() minus the
// half that changes units — kept separate rather than sharing, because the two
// jobs name different models and the sentence is the whole value.
function textJobText(job) {
  if (job.phase === "download") {
    const mb = (bytes) => Math.round(bytes / 1e6);
    const of = job.total ? ` of ${mb(job.total)}` : ""; // no Content-Length
    return `Downloading the CLIP text model — ${mb(job.done)}${of} MB (first search only)`;
  }
  return "Loading the text model…";
}

// ---------- The edit filter (which kinds of edit an image carries) ----------

// The two non-effect tokens. "any" and "none" are not effect names and never
// collide with one, since every effect name is a key of the registry.
const EDIT_CHIP_LABELS = { any: "Edited", none: "Untouched" };
const EDIT_PHRASES = { any: "an edit", none: "no edits" };

const editChipLabel = (token) => EDIT_CHIP_LABELS[token] || effectLabel(token);
const editPhrase = (token) => EDIT_PHRASES[token] || effectLabel(token);

// Build the chip row from the *library*, not from the effect registry: a chip
// for an effect nothing has ever been run through is a filter whose only
// possible answer is "nothing". Their order is the registry's, so the chips sit
// in the order the app's own effect buttons do, and each carries its count —
// which is most of what the row is for, since it says what there is to find
// before you press anything.
function renderEditChips() {
  const row = $("embed-edit-row");
  row.textContent = "";
  const counts = new Map();
  let edited = 0;
  for (const p of embedMap.points) {
    if (p.effects && p.effects.length) edited++;
    for (const name of p.effects || []) counts.set(name, (counts.get(name) || 0) + 1);
  }
  const chips = [];
  // Both or neither: with nothing edited these say the same thing as each other,
  // and with nothing untouched "Edited" is the whole library.
  if (edited && edited < embedMap.points.length) {
    chips.push(["any", edited], ["none", embedMap.points.length - edited]);
  }
  for (const spec of state.effects) {
    if (counts.has(spec.name)) chips.push([spec.name, counts.get(spec.name)]);
  }
  row.hidden = !chips.length;
  for (const [token, count] of chips) {
    const btn = document.createElement("button");
    // Explicitly not a submit button: this row lives inside a <dialog>, where
    // the default type would close the map on the first chip pressed.
    btn.type = "button";
    btn.className = "embed-chip";
    btn.classList.toggle("on", !!embedMap.edits && embedMap.edits.has(token));
    btn.textContent = `${editChipLabel(token)} ${count}`;
    btn.onclick = () => toggleEditToken(token);
    row.append(btn);
  }
}

function toggleEditToken(token) {
  const set = new Set(embedMap.edits || []);
  if (!set.delete(token)) set.add(token);
  // Empty is off, not "an empty union matches nothing" — the alternative is a
  // map that goes blank when you press the last lit chip a second time.
  embedMap.edits = set.size ? set : null;
  renderEditChips();
  applyEmbedFilter();
}

// Whether anything is hiding points at all — the question the cluster pills and
// the status line both used to ask as `embedMap.scores`, which was the same
// question only while the search was the only filter that could empty a group.
// Near's radius is not consulted here: it does nothing without a pick, which is
// exactly the condition below.
function embedFiltering() {
  return !!(
    embedMap.scores ||
    embedMap.edits ||
    embedMap.pinned ||
    (embedMap.selected && embedNearRadius(embedMap.nearT) !== Infinity)
  );
}

// Whether one image answers the lit chips. OR, not AND: the chips are how you
// widen a filter, and asking for the images carrying *both* bokeh and a dither
// is a question about one recipe rather than about a library.
function matchesEdits(p) {
  const edited = !!(p.effects && p.effects.length);
  for (const token of embedMap.edits) {
    const hit =
      token === "any" ? edited : token === "none" ? !edited : edited && p.effects.includes(token);
    if (hit) return true;
  }
  return false;
}

// Every filter the map has, resolved into per-point visibility — the one writer
// of `p.hidden` and of the status line, so the two can neither drift nor
// disagree about what is on screen. Purely local: the sliders and the chips
// re-run this and nothing else, so they are instant the way Size is, where a new
// query costs a request the way Groups does.
//
// The four filters compose, and are not symmetric. The search, the edit chips
// and the assistant's pinned set each ask a question of an image on its own, so
// they resolve together in one pass; Near asks one *about the pick*, so it is
// applied second, over the survivors, and turns itself off when there is no
// pick to measure from.
function applyEmbedFilter() {
  $("embed-match-row").hidden = !embedMap.scores;
  // The pin's own control, written here with the Match row rather than
  // wherever the pin is set: this function owns what the header says about
  // what is hidden, exactly as it owns the status line below.
  $("embed-pinned").hidden = !embedMap.pinned;
  if (embedMap.pinned) {
    $("embed-pinned").textContent =
      `Assistant: “${embedMap.pinned.query}” ${embedMap.pinned.ids.size}`;
  }
  for (const p of embedMap.points) {
    const entry = embedMap.scores && embedMap.scores[p.image_id];
    p.hidden =
      (!!embedMap.scores && (!entry || entry.z < embedMap.matchZ)) ||
      (!!embedMap.edits && !matchesEdits(p)) ||
      (!!embedMap.pinned && !embedMap.pinned.ids.has(p.image_id));
  }
  // A pick that just went invisible would otherwise leave the card floating
  // over a point nobody can see, with Open still wired to it. It has to happen
  // between the passes: it is also what the distance below is measured from.
  // Cleared directly rather than through pickEmbedPoint(), which would call
  // this again from inside itself.
  if (embedMap.selected && embedMap.selected.hidden) selectEmbedPoint(null);

  const anchor = embedMap.selected;
  const radius = embedNearRadius(embedMap.nearT);
  // The pick is at distance 0 from itself, so it survives even a radius of 0 —
  // which is what makes the bottom of the travel "only this one" rather than
  // "nothing".
  const near = anchor && radius !== Infinity;
  if (near) {
    for (const p of embedMap.points) {
      if (p.hidden) continue;
      p.hidden =
        Math.hypot(p.x - anchor.x, p.y - anchor.y, p.z - anchor.z) > radius;
    }
  }
  markEmbedDirty();

  if (!embedMap.scores && !embedMap.edits && !embedMap.pinned && !near) {
    $("embed-status").textContent = embedIdleStatus();
    return;
  }
  let kept = 0;
  for (const p of embedMap.points) if (!p.hidden) kept++;
  // One clause per active filter rather than a ternary over the combinations:
  // with two filters that was two branches, with three it is seven, and the
  // branch for "edits, no query, no pick" is one that reads `anchor.name` off a
  // null pick. The trailing hint is by precedence instead — whichever filter is
  // hardest to notice you left on is the one worth telling you how to clear.
  const clauses = [];
  if (embedMap.pinned) clauses.push(`are what the assistant found for “${embedMap.pinned.query}”`);
  if (embedMap.scores) clauses.push(`match “${embedMap.query}”`);
  if (embedMap.edits) clauses.push(`have ${[...embedMap.edits].map(editPhrase).join(" or ")}`);
  if (near) clauses.push(`are near ${anchor.name}`);
  // The pin goes first in the precedence for the same reason it is the only
  // filter with a button of its own: every other one names a control the person
  // reached for, so it is the one they are least likely to work out they can
  // switch off.
  const hint = embedMap.pinned
    ? "press the assistant's chip to show the library again"
    : embedMap.scores
      ? "drag Match to widen · clear the box to show all"
      : embedMap.edits
        ? "press a lit chip to clear it"
        : "drag Near to widen · pick another image to move the circle";
  // Two clauses read "a and b", more than two "a, b and c" — the second
  // spelling became reachable with the third filter, and "a and b and c" was
  // the sentence that made it obvious a list had grown out of a ternary.
  const list =
    clauses.length > 2
      ? `${clauses.slice(0, -1).join(", ")} and ${clauses[clauses.length - 1]}`
      : clauses.join(" and ");
  $("embed-status").textContent =
    `${kept} of ${embedMap.points.length} ${list} · ${hint}`;
}

function clearEmbedSearch() {
  embedMap.query = "";
  embedMap.scores = null;
  embedMap.searchSeq++; // abandon an in-flight search, download poll included
  clearTimeout(embedMap.searchTimer);
  // Not a loop clearing `p.hidden` here: Near may still be filtering, and
  // revealing the whole library because a *query* ended would show images the
  // other filter says are out. applyEmbedFilter() owns that flag.
  applyEmbedFilter();
}

// The assistant's filter, dismissed. Shaped like clearEmbedSearch() and for its
// reason: `p.hidden` is not cleared here, because the chips or Near may still
// be filtering and revealing the whole library would show images they say are
// out. applyEmbedFilter() owns that flag.
function clearEmbedPin() {
  embedMap.pinned = null;
  applyEmbedFilter();
}

function closeEmbedMap() {
  embedMap.open = false;
  // Without this the rAF loop keeps running behind a hidden dialog forever —
  // and Esc never touches the Close button, which is why the teardown hangs off
  // the dialog's `close` event too.
  stopEmbedLoop();
  embedMap.seq++; // abandon any in-flight fetch
  embedMap.searchSeq++; // including a search, and any model download it was polling
  clearTimeout(embedMap.clusterTimer); // and any re-fetch a drag left pending
  clearTimeout(embedMap.searchTimer);
  $("embed-modal").close();
}

// Anything that changes what the next frame should look like says so here. Its
// callers are every gesture, every filter, and the sprite loads that arrive
// after the frame that asked for them.
function markEmbedDirty() {
  embedMap.dirty = true;
}

// The single write path for the spin, because there are two ways to ask for it —
// the checkbox and the canvas's double-click — and a flag written by one of them
// leaves the other showing the opposite of what is happening.
function setEmbedSpin(on) {
  embedMap.spin = on;
  $("embed-spin").checked = on;
  markEmbedDirty();
}

// The loop still runs whenever the map is open, but only *paints* when the
// picture would differ. With the spin on by default that distinction cost
// nothing to skip; still by default, painting anyway means compositing the whole
// library sixty times a second to produce the frame already on screen.
//
// EMBED_IDLE_MS is the safety net, not the mechanism: a dirty flag's failure
// mode is a frozen canvas the moment one mutation forgets to mark itself, and a
// frame that can be at most a fifth of a second stale is a far better worst case
// than a map that stops responding. It is also what covers the one input nothing
// can mark — an image the browser decoded between frames.
function startEmbedLoop() {
  if (embedMap.raf) return;
  const tick = () => {
    if (embedMap.spin) embedMap.yaw += 0.003;
    const now = performance.now();
    if (embedMap.spin || embedMap.dirty || now - embedMap.painted > EMBED_IDLE_MS) {
      embedMap.dirty = false;
      embedMap.painted = now;
      drawEmbedMap();
    }
    embedMap.raf = requestAnimationFrame(tick);
  };
  embedMap.raf = requestAnimationFrame(tick);
}

function stopEmbedLoop() {
  if (embedMap.raf) cancelAnimationFrame(embedMap.raf);
  embedMap.raf = null;
}

// The camera, derived from the view state. Shared by drawEmbedMap() and the
// pivot arithmetic, which has to invert exactly the projection that was drawn —
// two copies of this trig would be two chances for a grab to land elsewhere.
function embedCamera() {
  const canvas = $("embed-canvas");
  const w = canvas.width;
  const h = canvas.height;
  return {
    w,
    h,
    // coordinates arrive normalized into the unit cube, half-diagonal √3
    scale: ((Math.min(w, h) / 2 - 14) / Math.sqrt(3)) * embedMap.zoom,
    cy: Math.cos(embedMap.yaw),
    sy: Math.sin(embedMap.yaw),
    cp: Math.cos(embedMap.pitch),
    sp: Math.sin(embedMap.pitch),
  };
}

// The thumbnail to draw at a point, or null while it loads — and null for good
// if it fails, so one broken render falls back to its dot instead of refetching
// sixty times a second.
//
// Keyed by URL with the crop tag in it, because re-framing an image changes the
// bytes served under a node id that has not changed: that is the hazard
// cropTag() exists for in the gallery, and here a stale entry would not be an
// old thumbnail but the wrong picture entirely. Superseded keys are left in the
// map; there are at most a few per image per session, and the alternative is a
// second index from image to key that can only get out of step.
function embedSprite(point) {
  if (point.url === undefined) {
    // Resolved once per point rather than once per point per frame — a library
    // scan sixty times a second is quadratic in its size. Nothing invalidates
    // it: opening the map always fetches a fresh array of points, and the map
    // is closed for every path that could re-frame an image.
    const image = state.images.find((i) => i.id === point.image_id);
    point.url =
      `/api/nodes/${point.node_id}/render?thumb=1${cropTag(image && image.crop)}`;
  }
  const url = point.url;
  let entry = embedMap.sprites.get(url);
  if (!entry) {
    entry = { canvas: null };
    embedMap.sprites.set(url, entry);
    const img = new Image();
    // The one input no gesture can mark: the frame that asked for this sprite
    // has long since been painted, so the loop has to be told there is more to
    // draw. (Covered by EMBED_IDLE_MS either way — this only saves the wait.)
    img.onload = () => {
      entry.canvas = bakeEmbedSprite(img);
      markEmbedDirty();
    };
    img.src = url;
  }
  return entry.canvas;
}

// Scale the thumbnail down once, into a canvas of its own, rather than letting
// drawImage resample the full 320px JPEG on every frame. Never upscales: a
// source smaller than the bake size is already as sharp as it will ever be.
function bakeEmbedSprite(img) {
  const k = Math.min(1, EMBED_SPRITE_PX / Math.max(img.width, img.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(img.width * k));
  canvas.height = Math.max(1, Math.round(img.height * k));
  canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function drawEmbedMap() {
  const canvas = $("embed-canvas");
  const ctx = canvas.getContext("2d");
  const { w, h, scale, cy, sy, cp, sp } = embedCamera();
  ctx.clearRect(0, 0, w, h);
  const pivot = embedMap.pivot;
  // the same orthographic yaw-then-pitch projection drawClusterPlot() uses;
  // z comes back only to sort by depth (a shared offset cannot reorder it).
  // Pivot, zoom and pan ride along here and nowhere else, so the wireframe cube
  // below stays welded to the cloud.
  const project = (x, y, z) => {
    const dx = x - pivot.x;
    const dy = y - pivot.y;
    const dz = z - pivot.z;
    const x1 = dx * cy + dz * sy;
    const z1 = dz * cy - dx * sy;
    const y1 = dy * cp - z1 * sp;
    const z2 = dy * sp + z1 * cp;
    return [
      w / 2 + embedMap.panX + x1 * scale,
      h / 2 + embedMap.panY - y1 * scale,
      z2,
    ];
  };

  ctx.strokeStyle = "rgba(139, 147, 163, 0.25)";
  ctx.lineWidth = 1;
  for (const a of [-1, 1]) {
    for (const b of [-1, 1]) {
      for (const [p, q] of [
        [[-1, a, b], [1, a, b]],
        [[a, -1, b], [a, 1, b]],
        [[a, b, -1], [a, b, 1]],
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

  const dpr = window.devicePixelRatio || 1;
  const r = 4 * dpr;
  // Project once and cache the screen position back onto the point, so
  // hit testing reads exactly the pixels that were drawn rather than
  // re-deriving them against a cloud that has since rotated.
  let zMin = Infinity;
  let zMax = -Infinity;
  for (const p of embedMap.points) {
    const [sx, sy2, z] = project(p.x, p.y, p.z);
    p.sx = sx;
    p.sy = sy2;
    p.sz = z;
    zMin = Math.min(zMin, z);
    zMax = Math.max(zMax, z);
  }

  // A thumbnail's long edge, in screen pixels. Derived from `scale` — the
  // world→screen factor — so a sprite is a fixed size *in the cloud*, welded to
  // the wireframe cube the way the points are: zoom in and the pictures grow.
  // A billboard costs nothing to orient here, because the projection is
  // orthographic: there is no perspective divide to face away from, so "always
  // face the viewer" is just an axis-aligned rect at the projected point.
  const size = EMBED_SPRITE_WORLD * scale * embedMap.spriteScale;
  // Sprites are sorted near-last and overlap, which is the only depth cue an
  // orthographic view gets for free — nothing shrinks with distance. So the
  // back of the cloud is dimmed too, over the depth range the cloud actually
  // spans, or a dense library reads as one flat collage.
  //
  // Dimming is a wash of the background colour laid *over* an opaque sprite,
  // not transparency: fading the sprite itself would let the pictures behind it
  // show through, and once they do, overlap stops reading as one thing in front
  // of another — which is the depth cue this is meant to reinforce.
  const fade = (z) =>
    zMax > zMin ? 0.5 * ((zMax - z) / (zMax - zMin)) : 0;

  // Filtered points are dropped here and nowhere else — this is the single
  // place a point becomes visible. Note they were still projected above, and
  // still count towards zMin/zMax: the depth wash is a property of how deep the
  // *cloud* is, so a search must not make the survivors re-shade themselves.
  for (const p of embedMap.points.filter((p) => !p.hidden).sort((a, b) => a.sz - b.sz)) {
    const selected = p === embedMap.selected;
    const sprite = embedSprite(p);
    const dim = selected ? 0 : fade(p.sz); // the pick never dims
    // A library too small to group (or one served by an older backend) has no
    // cluster on its points, and keeps the hairline it always had.
    const edge =
      p.cluster == null
        ? "rgba(20, 22, 27, 0.65)"
        : clusterColor(p.cluster, 1 - dim * 1.4);
    if (sprite) {
      // half-extents, cached for the hit test: what you click is the box you saw
      const k = size / Math.max(sprite.width, sprite.height);
      p.hw = (sprite.width * k) / 2;
      p.hh = (sprite.height * k) / 2;
      ctx.drawImage(sprite, p.sx - p.hw, p.sy - p.hh, p.hw * 2, p.hh * 2);
      if (dim) {
        ctx.fillStyle = `rgba(20, 22, 27, ${dim})`;
        ctx.fillRect(p.sx - p.hw, p.sy - p.hh, p.hw * 2, p.hh * 2);
      }
      // The frame carries the group: a thumbnail says what it is a picture of,
      // but nothing about it says which label out on the cloud is claiming it.
      // Dimmed along with the sprite, or the frames of the back of the cloud
      // would sit in front of the depth cue the wash exists to create.
      ctx.strokeStyle = selected ? "#e6e9ef" : edge;
      ctx.lineWidth = (selected ? 3 : 2) * dpr;
      ctx.strokeRect(p.sx - p.hw, p.sy - p.hh, p.hw * 2, p.hh * 2);
    } else {
      // Until its thumbnail arrives a point is the dot it always was, in the
      // image's mean colour — so the map is populated the moment it opens and
      // fills in, rather than starting empty. `hw` of 0 is what tells the hit
      // test to fall back to a radius here.
      p.hw = 0;
      p.hh = 0;
      // a dot has nothing behind it to show through, so here the wash is just
      // the same dimming applied to the colour
      const [cr, cg, cb] = p.color.map((c) => Math.round(c * (1 - dim)));
      ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, selected ? r * 1.6 : r, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = selected ? "#e6e9ef" : edge;
      ctx.lineWidth = (selected ? 2 : 1) * dpr;
      ctx.stroke();
    }
  }

  drawEmbedLabels(ctx, project, dpr, fade);
}

// One text label per cluster, floated over the cloud where its images are.
//
// A separate pass, after every sprite, and deliberately not depth-sorted in
// with them: a label drawn in its own depth order disappears behind whatever
// happens to be nearer, and in a dense collage that is most of them — which
// would hide exactly the labels sitting on the busiest, most worth naming part
// of the map. So a label is never *occluded*, but it still recedes: it shares
// the sprites' `fade` curve, so a group at the back of the cloud is exactly as
// far back as its own pictures.
//
// Note this fades by transparency where the sprites use a wash of the
// background colour, which above is called out as the wrong thing to do. The
// reason it inverts here is the same reason the pass exists: a wash preserves
// overlap as a depth cue, and these never overlap anything. With occlusion
// given up, alpha is the only depth cue a label has left.
function drawEmbedLabels(ctx, project, dpr, fade) {
  const font = 13 * dpr;
  ctx.font = `600 ${font}px system-ui, sans-serif`;
  ctx.textBaseline = "middle";
  const padX = 8 * dpr;
  const padY = 5 * dpr;
  const lineHeight = font + padY * 2;

  // While anything is filtering, count what each group still has showing. The
  // groups themselves are deliberately not refetched — they describe the
  // library, and recomputing k-means per keystroke would make the labels dance —
  // but a bare count beside a filtered cloud claims pictures that are not there.
  const shown = embedFiltering()
    ? embedMap.points.reduce((counts, p) => {
        if (!p.hidden && p.cluster != null) counts[p.cluster] = (counts[p.cluster] || 0) + 1;
        return counts;
      }, {})
    : null;

  // Nearest last, so that where two labels genuinely cannot both be placed the
  // one in front is the one on top — the same rule the sprites follow.
  //
  // A group with nothing left showing drops out entirely rather than reading
  // "0/44": the filter exists to reduce the cloud to what matched, and a dozen
  // pills labelling empty space is most of what there would be left to look at.
  const ordered = embedMap.clusters
    .filter((c) => c.label && (!shown || shown[c.id]))
    .map((c) => ({ ...c, p: project(c.x, c.y, c.z) }))
    .sort((a, b) => a.p[2] - b.p[2]);

  const placed = [];
  for (const cluster of ordered) {
    const suffix = shown
      ? ` · ${shown[cluster.id] || 0}/${cluster.count}`
      : ` · ${cluster.count}`;
    const textW = ctx.measureText(cluster.label).width;
    const w = textW + ctx.measureText(suffix).width + padX * 2;
    const h = lineHeight;
    const [x, home] = cluster.p;
    // Nudge off anything already placed rather than solving a layout: a label
    // that moved a line still points at its own group, and the alternative is a
    // constraint solver for a dozen boxes that move every frame anyway.
    //
    // Offsets alternate below/above the true centre and grow — 0, +1, -1, +2 —
    // so a crowd opens outwards from where the group actually is instead of
    // sliding one way and pulling the last label a long way south. Bounded, so
    // a genuine pile-up (twenty groups over a tight t-SNE) degrades to overlap
    // rather than marching labels off the canvas.
    const step = lineHeight + 2 * dpr;
    let y = home;
    for (let tries = 0; tries < 12; tries++) {
      y = home + Math.ceil(tries / 2) * step * (tries % 2 ? 1 : -1);
      const clash = placed.some(
        (b) =>
          Math.abs(b.x - x) < (b.w + w) / 2 && Math.abs(b.y - y) < (b.h + h) / 2,
      );
      if (!clash) break;
    }
    placed.push({ x, y, w, h });

    const left = x - w / 2;
    const top = y - h / 2;
    // `fade` tops out at a 0.5 wash, so the furthest label is drawn at half
    // opacity — receding without becoming a thing you have to squint at.
    ctx.globalAlpha = 1 - fade(cluster.p[2]);
    // A pill, not bare text: `fillText` over a collage of photographs is
    // unreadable about half the time, and which half changes as the cloud spins.
    ctx.beginPath();
    ctx.roundRect(left, top, w, h, h / 2);
    ctx.fillStyle = "rgba(20, 22, 27, 0.82)";
    ctx.fill();
    ctx.strokeStyle = clusterColor(cluster.id);
    ctx.lineWidth = 1.5 * dpr;
    ctx.stroke();

    ctx.fillStyle = "#e6e9ef";
    ctx.fillText(cluster.label, left + padX, y + 0.5 * dpr);
    ctx.fillStyle = "rgba(230, 233, 239, 0.45)";
    ctx.fillText(suffix, left + padX + textW, y + 0.5 * dpr);
  }
  // The context is shared with the next frame's sprites, which assume 1.
  ctx.globalAlpha = 1;
}

// Hit testing runs against the extents the last frame *drew* (`hw`/`hh`), not
// against a tolerance, so a click lands on the picture under the cursor however
// far the view is zoomed.
function embedPointAt(x, y) {
  let best = null;
  for (const p of embedMap.points) {
    // Filtered out, so not drawn — and `hw` is only ever written *inside* the
    // draw loop, which means a point that stops being drawn keeps last frame's
    // extents and would otherwise stay clickable. This guard is what keeps this
    // function's contract with the pixels, not defensiveness.
    if (p.hidden) continue;
    if (p.sx === undefined || !p.hw) continue;
    if (Math.abs(p.sx - x) > p.hw || Math.abs(p.sy - y) > p.hh) continue;
    // overlapping sprites go to the one nearest the camera — the one on top,
    // and so the only one the click could have looked like it hit
    if (!best || p.sz > best.sz) best = p;
  }
  if (best) return best;
  // Points still waiting on a thumbnail are 4px dots, too small to hit exactly,
  // and keep the radial tolerance they have always had.
  let bestD = EMBED_HIT_PX * (window.devicePixelRatio || 1);
  for (const p of embedMap.points) {
    if (p.hidden) continue;
    if (p.sx === undefined || p.hw) continue;
    const d = Math.hypot(p.sx - x, p.sy - y);
    if (d < bestD || (d === bestD && best && p.sz > best.sz)) {
      best = p;
      bestD = d;
    }
  }
  return best;
}

function selectEmbedPoint(point) {
  embedMap.selected = point;
  markEmbedDirty(); // the pick is drawn: its ring, and the card's leader line
  const card = $("embed-card");
  if (!point) {
    card.hidden = true;
    return;
  }
  const image = state.images.find((i) => i.id === point.image_id);
  $("embed-card-img").src =
    `/api/nodes/${point.node_id}/render?thumb=1${cropTag(image && image.crop)}`;
  $("embed-card-name").textContent = point.name;
  $("embed-card-name").title = point.name;
  card.hidden = false;
  // Picking moves the rotation centre at once rather than waiting for a drag,
  // so a spin already running starts orbiting the new pick too. Nothing on
  // screen moves.
  if (embedMap.orbit) pivotOnSelection();
}

function initEmbedMap() {
  const canvas = $("embed-canvas");
  let dragging = false;
  let panning = false; // a secondary-button drag pans instead of rotating
  let lastX = 0;
  let lastY = 0;
  let downX = 0;
  let downY = 0;

  canvas.addEventListener("mousedown", (e) => {
    // also what suppresses Windows Chrome's middle-click autoscroll
    e.preventDefault();
    dragging = true;
    panning = e.button !== 0;
    // A rotate drag turns the cloud about wherever it was grabbed, so the pivot
    // is chosen here, before the first move — unless Orbit has nailed it to the
    // pick, which is why pivotOnSelection() is asked first and the grab is the
    // fallback. A pan drag keeps the pivot it has either way: panning slides the
    // whole cloud, pivot included.
    if (!panning && !(embedMap.orbit && pivotOnSelection())) {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      setEmbedPivot((e.clientX - rect.left) * dpr, (e.clientY - rect.top) * dpr);
    }
    lastX = downX = e.clientX;
    lastY = downY = e.clientY;
    markEmbedDirty();
  });
  // or the menu would land in the middle of a right-drag
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  // on window, not the canvas, so a drag survives leaving it
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    if (panning) {
      // pan is applied after projection, so it takes screen deltas straight —
      // scaled into backing-store pixels, the unit drawEmbedMap() works in
      const dpr = window.devicePixelRatio || 1;
      embedMap.panX += (e.clientX - lastX) * dpr;
      embedMap.panY += (e.clientY - lastY) * dpr;
    } else {
      embedMap.yaw += (e.clientX - lastX) * 0.01;
      embedMap.pitch += (e.clientY - lastY) * 0.01;
      embedMap.pitch = Math.max(-1.5, Math.min(1.5, embedMap.pitch));
    }
    lastX = e.clientX;
    lastY = e.clientY;
    markEmbedDirty();
  });
  window.addEventListener("mouseup", (e) => {
    if (!dragging) return;
    dragging = false;
    // A rotation or pan gesture must not also pick a point, so a left-button
    // mouseup that barely moved is the only thing that counts as a click.
    if (panning || e.button !== 0) return;
    if (Math.hypot(e.clientX - downX, e.clientY - downY) > EMBED_DRAG_PX) return;
    const rect = canvas.getBoundingClientRect();
    if (
      e.clientX < rect.left || e.clientX > rect.right ||
      e.clientY < rect.top || e.clientY > rect.bottom
    ) return;
    const dpr = window.devicePixelRatio || 1;
    pickEmbedPoint(
      embedPointAt((e.clientX - rect.left) * dpr, (e.clientY - rect.top) * dpr)
    );
  });
  canvas.addEventListener("dblclick", () => setEmbedSpin(!embedMap.spin));
  // The preview panel's initZoom() arithmetic, in backing-store pixels against
  // the canvas centre: keep whatever is under the cursor under the cursor.
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const next = Math.min(16, Math.max(1, embedMap.zoom * wheelZoomFactor(e)));
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const cx = (e.clientX - rect.left) * dpr - canvas.width / 2;
      const cy = (e.clientY - rect.top) * dpr - canvas.height / 2;
      const k = next / embedMap.zoom;
      embedMap.panX = cx - (cx - embedMap.panX) * k;
      embedMap.panY = cy - (cy - embedMap.panY) * k;
      embedMap.zoom = next;
      if (next === 1) resetEmbedView(); // scrolling back out reframes the cloud
      markEmbedDirty();
    },
    { passive: false }
  );

  // No redraw to schedule beyond saying one is due: the rAF loop is already
  // running, so the slider only has to move the number it reads.
  $("embed-size").oninput = (e) => {
    embedMap.spriteScale = Number(e.target.value);
    markEmbedDirty();
  };
  // Unlike Size, this one costs a request — a different k is a different
  // k-means fit — so the readout tracks the drag while the fetch waits for it
  // to settle. `loadEmbedMap`'s own seq guard makes a late response harmless;
  // the debounce is only there to stop a slow drag firing twenty of them.
  $("embed-clusters").oninput = (e) => {
    embedMap.clusterCount = Number(e.target.value);
    setClusterSlider(embedMap.clusterCount);
    clearTimeout(embedMap.clusterTimer);
    embedMap.clusterTimer = setTimeout(loadEmbedMap, 150);
  };
  // Debounced, like the cluster slider and for the same reason — but note what
  // is *not* debounced: emptying the box restores the whole library at once,
  // since waiting a beat to undo a filter reads as a stuck UI.
  $("embed-search").oninput = (e) => {
    clearTimeout(embedMap.searchTimer);
    if (!e.target.value.trim()) return clearEmbedSearch();
    embedMap.searchTimer = setTimeout(runEmbedSearch, EMBED_SEARCH_MS);
  };
  $("embed-search").addEventListener("keydown", (e) => {
    // Enter means "I have finished typing", so it pre-empts the debounce. It
    // must also not reach the dialog, where it would submit and close it.
    if (e.key !== "Enter") return;
    e.preventDefault();
    clearTimeout(embedMap.searchTimer);
    runEmbedSearch();
  });
  // Costs nothing but a redraw: the scores are already here, and the threshold
  // is applied against them locally. Like Size, unlike Groups.
  $("embed-match").oninput = (e) => {
    embedMap.matchZ = Number(e.target.value);
    $("embed-match-num").textContent = embedMap.matchZ.toFixed(1);
    applyEmbedFilter();
  };
  // Costs nothing but a redraw too — the coordinates are already here and the
  // radius is measured against them locally, so this is Match's twin and not
  // Groups'.
  $("embed-near").oninput = (e) => {
    setNearSlider(Number(e.target.value));
    applyEmbedFilter();
  };
  // Costs nothing but the pivot, like Size and unlike Groups — and re-pivoting
  // on the tick is what makes the box read as an answer rather than a promise:
  // a spin already running swings onto the pick immediately, without moving it.
  $("embed-orbit").onchange = (e) => {
    embedMap.orbit = e.target.checked;
    if (embedMap.orbit) pivotOnSelection();
    markEmbedDirty();
  };
  $("embed-spin").onchange = (e) => setEmbedSpin(e.target.checked);
  $("embed-method").onchange = (e) => {
    embedMap.method = e.target.value;
    // Keep yaw/pitch, so re-projecting doesn't also throw away the viewpoint —
    // and keep the pick, which is the same image wherever the new fit puts it.
    // The framing cannot be kept: pivot and pan are in the old space, so
    // loadEmbedMap re-centres on the pick.
    $("embed-status").textContent = "Projecting…";
    loadEmbedMap(true);
  };
  $("embed-open-btn").onclick = async () => {
    const point = embedMap.selected;
    if (!point) return;
    closeEmbedMap();
    await selectImage(point.image_id);
    await refreshGallery();
  };
  // Wrapped, not passed: openEmbedMap takes a pin now, and a bare handler would
  // hand it the click event — a map opened filtered to whatever `image_ids` a
  // PointerEvent has, which is nothing, so an empty cloud.
  $("embed-btn").onclick = () => openEmbedMap();
  $("embed-pinned").onclick = clearEmbedPin;
  $("embed-close-btn").onclick = closeEmbedMap;
  $("embed-modal").addEventListener("close", () => {
    if (embedMap.open) closeEmbedMap();
  });
  // the cloud is sized from its rendered box, so a resized window needs a resize
  window.addEventListener("resize", () => {
    if (embedMap.open) sizeEmbedCanvas();
  });
}

// ---------- Upload / delete ----------

// Uploads are serialized, not raced: each POST reads a whole JPG into server
// memory, and going in order is what makes the counter meaningful and lands the
// images in the order they were picked. One bad file does not stop the batch —
// the good ones are already saved by then, so failures are collected and
// reported once at the end, naming the files that did not make it.
// A file whose name is already in the library comes back `duplicate: true` with
// the *existing* image, not an error — re-dropping a folder you already imported
// is the case this exists for. Skips are reported in the label rather than the
// failure alert below, since nothing went wrong; `last` is assigned either way,
// so an all-duplicates batch still lands on an image instead of nowhere.
let uploadLabelTimer = null;

async function uploadFiles(files) {
  const label = $("upload-label");
  const input = $("file-input");
  const failed = [];
  const skipped = [];
  let last = null;
  // a summary still counting down from an earlier batch would otherwise wipe
  // this one's progress partway through
  clearTimeout(uploadLabelTimer);
  input.disabled = true;
  try {
    for (const [i, file] of files.entries()) {
      label.textContent =
        files.length > 1 ? `Adding ${i + 1} / ${files.length}…` : "Adding…";
      const form = new FormData();
      form.append("file", file);
      try {
        last = await api("/api/images", { method: "POST", body: form });
        if (last.duplicate) skipped.push(file.name);
      } catch (err) {
        failed.push(`${file.name}: ${err.message}`);
      }
    }
  } finally {
    if (skipped.length) {
      const added = files.length - skipped.length - failed.length;
      const dupes =
        skipped.length === 1
          ? `${skipped[0]} already in library`
          : `${skipped.length} already in library`;
      label.textContent = added ? `${added} added, ${dupes}` : dupes;
      uploadLabelTimer = setTimeout(() => {
        label.textContent = "+ Add JPG";
      }, 4000);
    } else {
      label.textContent = "+ Add JPG";
    }
    input.disabled = false;
  }
  // one refresh for the whole batch, then land on the last image that made it
  if (last) {
    await refreshGallery();
    await selectImage(last.id, last.root_node_id);
  }
  if (failed.length) alert(`Upload failed:\n${failed.join("\n")}`);
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

// ---------- The command row ----------
//
// A sentence goes to /api/agent, which drives the same endpoints these buttons
// do and reports what it touched. Two things come back besides words: `focus`,
// the node to navigate to, and `pending`, a destructive action the server
// refused to perform until someone presses a button here.
//
// The conversation is held here rather than on the server: the Messages API is
// stateless and this app has no sessions, so the browser is the natural owner.
// It is posted back verbatim and replaced with whatever comes back.

function agentLine(text, kind) {
  const div = document.createElement("div");
  div.className = `agent-line${kind ? ` agent-${kind}` : ""}`;
  div.textContent = text;
  $("agent-log").append(div);
  // The log is a strip inside a scrolling row, so the newest line has to be
  // walked to rather than waited for.
  div.scrollIntoView({ block: "nearest", inline: "nearest" });
  return div;
}

// The queued delete, rendered as the button the server declined to be. The
// agent never holds this trigger — what it sent was a description, and this is
// the ordinary DELETE the tree's own buttons call, reached by a human click.
function renderAgentPending(pending) {
  const row = document.createElement("div");
  row.className = "agent-line agent-pending";
  row.append(`${pending.description}?`);

  const go = document.createElement("button");
  go.textContent = "Delete";
  go.className = "agent-danger";
  const cancel = document.createElement("button");
  cancel.textContent = "Cancel";

  const settle = (note) => {
    row.replaceChildren();
    row.className = "agent-line agent-note";
    row.textContent = note;
  };
  cancel.onclick = () => settle("Cancelled — nothing was deleted.");
  go.onclick = async () => {
    go.disabled = cancel.disabled = true;
    try {
      if (pending.action === "delete_node") {
        const res = await api(`/api/nodes/${pending.id}`, { method: "DELETE" });
        const keep = res.deleted.includes(state.nodeId) ? res.parent_id : state.nodeId;
        await selectImage(state.imageId, keep);
      } else {
        await api(`/api/images/${pending.id}`, { method: "DELETE" });
        await refreshGallery();
        await selectImage(state.images.length ? state.images[0].id : null);
      }
      settle("Deleted.");
    } catch (err) {
      settle(`Delete failed: ${err.message}`);
    }
  };
  row.append(go, cancel);
  $("agent-log").append(row);
  row.scrollIntoView({ block: "nearest", inline: "nearest" });
}

async function runAgent() {
  const input = $("agent-input");
  const prompt = input.value.trim();
  if (!prompt || state.agent.busy) return;

  state.agent.busy = true;
  input.value = "";
  input.disabled = $("agent-go").disabled = true;
  $("agent-status").hidden = false;
  agentLine(prompt, "you");

  try {
    const res = await api("/api/agent", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        prompt,
        history: state.agent.history,
        image_id: state.imageId,
        node_id: state.nodeId,
      }),
    });
    state.agent.history = res.history;
    // The prompt rides along because it is what makes a turn recognisable in
    // the Wire dialog — the messages list carries it too, but buried at the
    // bottom of whatever the trim left of the previous turns.
    if (res.wire) state.agent.wire.push({ prompt, ...res.wire });
    // Steps first, then the sentence: what it did, then what it says about it.
    for (const step of res.steps) {
      agentLine(`${step.ok ? "·" : "×"} ${step.summary}`, step.ok ? "step" : "fail");
    }
    if (res.reply) agentLine(res.reply, "reply");
    // selectImage() is the single choke point every path that changes what you
    // are looking at goes through — it repaints the preview, tree, masks and
    // presets together, so the row does not need to know about any of them.
    if (res.focus) await selectImage(res.focus.image_id, res.focus.node_id);
    if (res.pending) renderAgentPending(res.pending);
    // Last of the three, and after both on purpose. It is a modal, so the steps
    // and the reply above have to be in the log *before* it opens to be there
    // when it closes; renderAgentPending is synchronous DOM while this awaits
    // seconds of I/O, and a confirm button appended under a backdrop that has
    // already taken the screen is a button nobody sees. It centres on
    // state.imageId, which res.focus just set — so the cloud frames the photo
    // the turn landed on.
    if (res.map) await openEmbedMap(res.map);
  } catch (err) {
    // Into the log, never an alert(): alert() blocks, and this is a control you
    // press over and over.
    agentLine(err.message, "fail");
  } finally {
    state.agent.busy = false;
    input.disabled = $("agent-go").disabled = false;
    $("agent-status").hidden = true;
    $("agent-clear").hidden = !state.agent.history.length;
    // Its own count, not the history's: a turn the API refused hands back no
    // wire, and a Wire button that opens an empty dialog is a broken button.
    $("agent-wire").hidden = !state.agent.wire.length;
    input.focus();
  }
}

function clearAgent() {
  state.agent.history = [];
  // The record goes with the conversation it records. Keeping it would leave
  // the Wire dialog describing messages the model has already been made to
  // forget, which is the one thing it must not do.
  state.agent.wire = [];
  $("agent-log").replaceChildren();
  $("agent-clear").hidden = $("agent-wire").hidden = true;
}

// ---------- The wire ----------
//
// Every request the last few turns made to the Messages API, and every response
// back. Nothing here changes what the agent does; it is the loop in `run_turn`
// made readable — a growing `messages` list re-sent each round, a response whose
// `stop_reason` decides whether there is another one, and the system prompt and
// tool schemas that ride along with all of them.
//
// Built with textContent throughout, never innerHTML: this view renders model
// output and tool results verbatim, so it must not be somewhere markup can run.

// `wrap` is for the one block that is prose rather than JSON: pretty-printed
// JSON is already broken into short lines and wants its indentation left alone,
// but the system prompt is paragraphs, and a paragraph you have to scroll
// sideways to read is not one anybody reads.
function wireBlock(summary, body, open, wrap) {
  const details = document.createElement("details");
  details.open = !!open;
  const head = document.createElement("summary");
  head.textContent = summary;
  const pre = document.createElement("pre");
  pre.className = `wire-pre${wrap ? " wire-wrap" : ""}`;
  pre.textContent = body;
  details.append(head, pre);
  return details;
}

// The one-line reason a round happened, and what it cost.
function wireRoundSummary(round, n) {
  const res = round.response;
  const use = res.usage || {};
  return (
    `round ${n} · ${round.messages.length} message${round.messages.length === 1 ? "" : "s"}` +
    ` · ${res.stop_reason} · ${use.input_tokens ?? "?"} in / ${use.output_tokens ?? "?"} out` +
    ` · ${round.ms}ms`
  );
}

function openWire() {
  const body = $("wire-body");
  body.replaceChildren();

  state.agent.wire.forEach((turn, i) => {
    // The newest turn is what you opened this to see, so its rounds start
    // expanded; the system prompt and the schemas never do — they are long, and
    // the same every turn.
    const newest = i === state.agent.wire.length - 1;
    const section = document.createElement("div");
    section.className = "wire-turn";

    const prompt = document.createElement("div");
    prompt.className = "wire-prompt";
    prompt.textContent = turn.prompt;
    const meta = document.createElement("div");
    meta.className = "wire-meta";
    meta.textContent = `${turn.call.model} · max_tokens ${turn.call.max_tokens}`;
    section.append(prompt, meta);

    section.append(
      wireBlock(
        `system prompt · ${turn.call.system.length.toLocaleString()} chars`,
        turn.call.system, // raw text: it is prose, and JSON would only escape it
        false,
        true
      ),
      wireBlock(
        `tools · ${turn.call.tools.length} schema${turn.call.tools.length === 1 ? "" : "s"}`,
        JSON.stringify(turn.call.tools, null, 2),
        false
      )
    );

    turn.rounds.forEach((round, r) => {
      const details = document.createElement("details");
      details.open = newest;
      const head = document.createElement("summary");
      head.textContent = wireRoundSummary(round, r + 1);
      details.append(head);
      details.append(
        wireBlock("→ messages", JSON.stringify(round.messages, null, 2), true),
        wireBlock("← response", JSON.stringify(round.response, null, 2), true)
      );
      section.append(details);
    });

    body.append(section);
  });

  $("wire-modal").showModal();
}

// ---------- Init ----------

async function init() {
  state.effects = await api("/api/effects");
  buildEffectButtons();
  // and nothing lit: the app opens showing the image as it is, and the first
  // click on a feature button is what starts previewing an effect on top of it

  initZoom();
  initCropOverlay();
  initClusterPlot();
  initEmbedMap();
  $("apply-btn").onclick = applyEffect;
  // the live-preview checkbox is gone — the feature buttons toggle the preview
  // themselves — so sweep the preference it left in browsers that ran the old
  // build. Nothing reads the key; this only stops it sitting there forever.
  localStorage.removeItem("picky:livePreview");
  // both control containers drive the one debounce; the selection lives in its
  // own section now, so it needs its own listener
  for (const id of ["effect-params", "select-controls"]) {
    $(id).addEventListener("input", () => {
      if (preview.active) schedulePreview();
    });
  }
  $("delete-btn").onclick = deleteImage;
  $("save-preset-btn").onclick = savePreset;

  // The Image map is the intended way to pick an image, so the filmstrip is
  // opt-in — only an explicit "1" turns it on, and the choice outlives the
  // session.
  setFilmstrip(localStorage.getItem("picky:filmstrip") === "1");
  $("film-btn").onclick = () => setFilmstrip(!state.filmstrip);

  $("stats-btn").onclick = openStats;
  $("stats-close-btn").onclick = () => $("stats-modal").close();
  $("presets-btn").onclick = () => $("presets-modal").showModal();
  $("presets-close-btn").onclick = () => $("presets-modal").close();
  $("edit-cancel-btn").onclick = () => closeEdit(true);
  $("edit-save-btn").onclick = saveEdit;
  $("edit-params").addEventListener("input", schedulePreview);
  // Esc closes a <dialog> without going through our buttons, so clean up here too
  $("edit-modal").addEventListener("close", () => {
    if (edit.node) closeEdit(true);
  });
  $("file-input").onchange = (e) => {
    // copied out of the live FileList, which the reset below empties
    const files = [...e.target.files];
    e.target.value = "";
    if (files.length) uploadFiles(files);
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
    // a dropped folder full of photos is a batch like any other; anything that
    // is not a JPEG is dropped here rather than sent for the server to 400
    const files = [...e.dataTransfer.files].filter(
      (f) => f.type === "image/jpeg" || /\.jpe?g$/i.test(f.name)
    );
    if (files.length) uploadFiles(files);
  });

  // The command row exists only if the server has a key to talk to Claude with.
  // Asked once, here, rather than discovered by a failing POST later.
  try {
    const agentInfo = await api("/api/agent");
    if (agentInfo.configured) {
      $("agent-section").hidden = false;
      $("agent-go").onclick = runAgent;
      $("agent-clear").onclick = clearAgent;
      // Both bindings live inside the configured branch: with no key there is
      // no row, so there is nothing that could open the dialog either.
      $("agent-wire").onclick = openWire;
      $("wire-close-btn").onclick = () => $("wire-modal").close();
      $("agent-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault(); // the row lives in a footer, not a form, but be sure
          runAgent();
        }
      });
    }
  } catch {
    // An older server without the endpoint just has no command row.
  }

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
