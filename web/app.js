const state = {
  effects: [],
  images: [],
  imageId: null,
  nodes: [],       // flat node list for the selected image
  nodeId: null,    // selected node
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
  if ($("effect-select").value === "blend") renderEffectControls();
  renderTree();
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

function renderTree() {
  const container = $("tree");
  container.innerHTML = "";
  if (state.imageId === null) {
    container.textContent = "No image selected.";
    return;
  }
  const byParent = new Map();
  for (const n of state.nodes) {
    const key = n.parent_id ?? "root";
    if (!byParent.has(key)) byParent.set(key, []);
    byParent.get(key).push(n);
  }
  const build = (parentKey) => {
    const children = byParent.get(parentKey) || [];
    if (!children.length) return null;
    const ul = document.createElement("ul");
    for (const node of children) {
      const li = document.createElement("li");
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
      if (node.params) {
        const span = document.createElement("span");
        span.className = "params";
        span.textContent = nodeParamsText(node);
        div.appendChild(span);
      }
      if (node.parent_id !== null) {
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
      li.appendChild(div);
      const sub = build(node.id);
      if (sub) li.appendChild(sub);
      ul.appendChild(li);
    }
    return ul;
  };
  container.appendChild(build("root"));
}

// ---------- Effects ----------

function renderEffectControls() {
  const select = $("effect-select");
  const effect = state.effects.find((e) => e.name === select.value);
  const box = $("effect-params");
  box.innerHTML = "";
  for (const p of effect.params) {
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
      sel.value = p.default;
      row.append(label, sel);
    } else {
      const valSpan = document.createElement("span");
      valSpan.textContent = p.default;
      label.appendChild(valSpan);
      const input = document.createElement("input");
      input.type = "range";
      input.min = p.min;
      input.max = p.max;
      if (p.step !== undefined) input.step = p.step;
      input.value = p.default;
      input.dataset.param = p.name;
      input.oninput = () => (valSpan.textContent = input.value);
      row.append(label, input);
    }
    box.appendChild(row);
  }

  let canApply = state.imageId !== null;
  if (effect.name === "blend") {
    const others = state.nodes.filter((n) => n.id !== state.nodeId);
    const row = document.createElement("div");
    row.className = "param-row";
    const label = document.createElement("label");
    const nameSpan = document.createElement("span");
    nameSpan.textContent = "Blend selected node with";
    label.appendChild(nameSpan);
    const sel = document.createElement("select");
    sel.id = "blend-with";
    for (const n of others) {
      const opt = document.createElement("option");
      opt.value = n.id;
      opt.textContent = `#${n.id} ${nodeLabel(n)}${n.params ? " · " + nodeParamsText(n) : ""}`;
      sel.appendChild(opt);
    }
    row.append(label, sel);
    box.appendChild(row);
    canApply = canApply && others.length > 0;

    // the weight param only affects the "average" mode
    const modeSel = box.querySelector('[data-param="mode"]');
    const weightRow = box.querySelector('[data-param="weight"]')?.closest(".param-row");
    if (modeSel && weightRow) {
      const syncWeight = () => (weightRow.hidden = modeSel.value !== "average");
      modeSel.addEventListener("change", syncWeight);
      syncWeight();
    }
  }
  $("apply-btn").disabled = !canApply;
  $("preview-btn").disabled = !canApply;
}

function readEffectForm() {
  const effect = $("effect-select").value;
  const params = {};
  document.querySelectorAll("#effect-params [data-param]").forEach((el) => {
    params[el.dataset.param] = el.type === "range" ? Number(el.value) : el.value;
  });
  let parent2_id = null;
  if (effect === "blend") {
    const withSel = $("blend-with");
    parent2_id = withSel && withSel.value ? Number(withSel.value) : null;
  }
  return { effect, params, parent2_id };
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

const preview = { active: false, url: null, seq: 0, timer: null };

function togglePreview() {
  if (preview.active) {
    exitPreview(true);
  } else {
    preview.active = true;
    $("preview-btn").classList.add("active");
    refreshPreview();
  }
}

function exitPreview(restoreSrc) {
  clearTimeout(preview.timer);
  preview.timer = null;
  preview.seq++; // invalidate in-flight fetches
  preview.active = false;
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
  if (!preview.active || state.nodeId === null) return;
  const { effect, params, parent2_id } = readEffectForm();
  if (effect === "blend" && parent2_id === null) return;
  const body = { effect, params };
  if (parent2_id !== null) body.parent2_id = parent2_id;
  const seq = ++preview.seq;
  const btn = $("preview-btn");
  btn.classList.add("busy");
  try {
    const res = await fetch(`/api/nodes/${state.nodeId}/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
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
    btn.classList.remove("busy");
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
  const hasChildren = state.nodes.some((n) => n.parent_id === node.id);
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
  const select = $("effect-select");
  for (const e of state.effects) {
    const opt = document.createElement("option");
    opt.value = e.name;
    opt.textContent = e.label;
    select.appendChild(opt);
  }
  select.onchange = () => {
    renderEffectControls();
    if (preview.active) schedulePreview();
  };
  renderEffectControls();

  initZoom();
  initClusterPlot();
  $("apply-btn").onclick = applyEffect;
  $("preview-btn").onclick = togglePreview;
  $("effect-params").addEventListener("input", () => {
    if (preview.active) schedulePreview();
  });
  $("delete-btn").onclick = deleteImage;
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
