# Picky

A browser-based app for applying visual effects to JPG images, with a git-like
**work tree** per image: every effect you apply creates a child node of the
currently selected node, so you can return to any earlier state and branch off
in a new direction. Images, trees, and rendered results persist across
restarts.

![Picky screenshot: a posterized orchid photo, with the side panel showing the Select step above the effect picker (Posterize chosen) and its RGB cluster plot, and the work tree drawn along the bottom as a left-to-right commit graph with branches and blend merges](docs/screenshot.png)

## Features

- **Image catalog** — upload JPGs (file picker or drag-and-drop); everything
  you've loaded stays in the gallery until you delete it.
- **Branching work tree** — effects stack as nodes in a tree, not a linear
  undo history, drawn as a commit graph running left to right in a strip under
  the image. Click any node to view it or branch from it; delete a node to
  prune it and everything derived from it.
- **Live preview** — the effect you are setting up renders over the image as
  you tune it, with no button to press. Uncheck **live preview** to see the
  selected node's own pixels again.
- **Effects**
  - *Posterize* — k-means clustering on RGB values, run in PCA-whitened space
    so clusters spread across color rather than bunching along the luminance
    diagonal. Selecting a posterize node shows a rotatable **3D scatter plot**
    of sampled pixels in RGB space, colored by cluster average.
  - *Tone curve* — a Photoshop-style curves editor. Drag control points on a
    grid drawn over the input's luma **histogram**; they are interpolated with
    a monotone cubic spline, so the curve never overshoots between points and
    never inverts local contrast. Behind the same button, *Gamma* offers the
    one-slider version of the same idea.
  - *Pixelate* — average pixels into bins of a chosen size, in a **square
    grid** or a **hexagonal** one. Hex bins are a pointy-top honeycomb: rows
    offset by half a cell, each pixel taking the mean color of the hexagon it
    falls in.
  - *Gaussian blur*, *Sobel edges*, *Floyd–Steinberg dither*
  - *Blend* — combine the selected node with any other node in the tree using
    average, additive, multiplicative, or subtractive blending. A **weight**
    slider sets the second node's share of the result; it applies to *average*
    only, the other three modes being unweighted.
- **Click to segment** — *Limit to object* confines any effect to one thing
  instead of the whole frame. Hit **Select object** and click the subject; a
  [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) model finds it and
  traces its boundary as an outline over the image, and the effect then applies
  inside it and leaves the rest alone. A dropdown chooses how much of the
  subject one click means — *auto (best match)*, or explicitly *whole*,
  *part*, or *subpart* — and **invert** flips the region, so the effect hits
  the background instead. The selection is saved with the node (masked nodes
  are marked in the work tree), so editing the effect's params later
  re-composites against the same region. Model weights (~45 MB) download on
  first use.
- **Saved masks** — **picking an object and applying an effect saves that
  object automatically**, freezing its pixels to disk as a durable mask rather
  than a click that has to be re-segmented. It appears as a silhouette icon
  under the picker, named for you (*Object 1*, *Object 2*, …) and already
  ticked, so the next effect lands on exactly the same region — that is what
  lets you stack several effects on one object without re-selecting it each
  time. The shape is identical every time, and it survives deleting the node it
  was picked on. **Tick several and they combine** — the effect applies to
  their union, so one blur can cover two objects. **Save selection** banks an
  object you are not ready to use yet. A mask belongs to its image, and one
  still in use cannot be deleted out from under the nodes that reference it.
- **Presets** — save the chain that produced a node as a named recipe and
  replay it on any other image. A preset stores its steps with *relative*
  parent references, not node ids, so a branching recipe (say `edges` blended
  back onto the original) reproduces its shape on whatever node you apply it
  to. Tick an object first and the whole recipe is confined to it — every step
  is masked to what you ticked, in place of whatever the recipe captured, so a
  chain built on one photo's subject can be aimed at another photo's. Applying
  one is all-or-nothing: if any step fails, the nodes it already created are
  rolled back.
- **Preview zoom** — scroll to zoom (cursor-anchored), drag to pan,
  double-click to reset.
- **Export** — download any node's render as a JPG named after its effect
  chain (e.g. `photo-posterize-blur.jpg`).
- **Library stats** — image, node, preset, and mask counts plus disk usage. The
  render cache is reported apart from the database, originals, and saved masks:
  it is ~85% of the bytes and regenerates on demand, so a single "on disk"
  figure would badly misstate what you would actually lose.

## Setup

Requires Python 3.13 (see `.tool-versions`).

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```sh
.venv/bin/uvicorn server.main:app --port 8000
```

Then open <http://localhost:8000>. Uploaded originals, the SQLite database,
and cached renders live in `data/` (gitignored).

The segmentation models are fetched to `data/models/` the first time you use
click-to-segment, from a revision-pinned Hugging Face URL. To supply your own
ONNX exports instead, point `PICKY_SAM_ENCODER` and `PICKY_SAM_DECODER` at
them; nothing else in the app requires the download.

## Architecture

```
server/
  main.py       FastAPI routes + static file serving
  db.py         SQLite schema and queries (images, nodes, presets, masks)
  effects.py    effect registry: each effect maps an RGB numpy array -> array
  rendering.py  node render pipeline with per-node JPEG cache
  sam.py        MobileSAM via onnxruntime: click a point, get a mask
web/
  index.html, app.js, style.css   vanilla JS single-page frontend
```

Each work-tree node stores its parent, effect name, and parameters; renders
are materialized at node creation and re-derived lazily if a cache file is
missing. Blend nodes carry a second parent reference (`parent2_id`), and
deleting a node cascades through both parent links.

Effects declare their parameter specs (range, default, choices) in the
registry, and the frontend generates its controls from `GET /api/effects`, so
an effect's *parameters* cost nothing on the client. The effect itself needs
four things: the registry entry in `server/effects.py`, an `EFFECT_BUTTONS`
entry in `web/app.js`, an inline SVG icon painting with `currentColor`, and a
`.fx-<name>` hue in `web/style.css`. The button list is not derived from the
registry — a registry effect missing from it throws when selected.

Segmentation splits into a heavy image encoder and a light prompt decoder.
The encoder's output is cached per node alongside its render, so the cost is
paid once per node no matter how many points you click; each click is then
only the decoder. A node's own selection is stored as the click, not the
pixels, and re-decoded — which keeps it exact when the node's params are
edited. Saving a mask is the deliberate opposite: it writes the decoded region
to `data/masks/<id>.png` as a 1-bit image, so reuse skips the model entirely
and the shape cannot drift. That file sits outside `data/renders/` because
everything there is a cache that invalidation is free to delete. Applying an
effect to a fresh pick saves it that way first, so the object outlives the
click — a click is only meaningful on the node it was made on, and applying an
effect moves you off that node.

A selection is a *union* in either form — a list of click points, or a list of
saved mask ids — OR'd together with one `invert` over the result. Selections
written before unions existed are upgraded when they are read, so nothing on
disk is rewritten. Presets can only carry the click form (a mask is pixels, and
pixels belong to one image), so a multi-mask selection is captured as the union
of the clicks those masks were frozen from and re-segmented on whatever image
the preset is replayed against.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET    | `/api/effects` | effect registry with parameter specs |
| POST   | `/api/images` | upload a JPG (multipart) |
| GET    | `/api/images` | list images |
| GET    | `/api/images/{id}/tree` | all work-tree nodes for an image |
| POST   | `/api/images/{id}/nodes` | apply an effect (`parent_id`, `effect`, `params`, optional `parent2_id` for blend, optional `selection` to mask it) |
| PATCH  | `/api/nodes/{id}` | edit a node's `params`, `parent2_id`, or `selection` in place, re-rendering it and dropping its descendants' caches |
| GET    | `/api/nodes/{id}/render` | rendered JPEG (`?thumb=1` thumbnail, `?download=1` attachment) |
| POST   | `/api/nodes/{id}/preview` | render an effect on top of a node in memory — no node created, nothing cached |
| POST   | `/api/nodes/{id}/mask` | outline PNG for a selection — `points` segmented on the fly, or `masks` (saved ids) read back, plus an optional `invert`; persists nothing but the node's cached embedding |
| GET    | `/api/nodes/{id}/clusters` | k-means scatter data for posterize nodes |
| GET    | `/api/nodes/{id}/histogram` | 256-bin luma histogram of a node's render, for the tone-curve editor's backdrop |
| DELETE | `/api/nodes/{id}` | delete a node and everything derived from it |
| DELETE | `/api/images/{id}` | delete an image, its tree, and its files |
| GET    | `/api/presets` | list saved presets with their steps |
| POST   | `/api/presets` | save a node's ancestor chain as a preset (`name`, `node_id`) |
| POST   | `/api/nodes/{id}/apply-preset` | replay a preset onto a node (`preset_id`, optional `selection` to mask every step of it — it replaces the selections the recipe carries) |
| DELETE | `/api/presets/{id}` | delete a preset |
| GET    | `/api/images/{id}/masks` | list an image's saved masks, each with a `used_by` count |
| POST   | `/api/images/{id}/masks` | freeze a click selection into a mask (`node_id`, `selection`, optional `name` — omit it and the server picks the next unused *Object N*) |
| GET    | `/api/masks/{id}/thumb` | the mask's silhouette as a small grayscale PNG — the icon in the mask grid; computed per request and served `immutable`, so cache-bust on the mask's `created_at` |
| PATCH  | `/api/masks/{id}` | rename a mask (`name`) |
| DELETE | `/api/masks/{id}` | delete a mask and its PNG; 409 while any node still references it |
| GET    | `/api/stats` | library counts and disk usage |
