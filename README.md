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
  - *Blur* — a **Gaussian** kernel for soft focus, or a flat **disk** for
    lens-style defocus, where a highlight blooms into a hard-edged circle
    rather than smearing.
  - *Bokeh* — **depth-of-field, with no selection involved**. A
    [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
    model estimates distance for every pixel, and the blur grows with distance
    from the focal plane the way a lens's circle of confusion does — so a
    background falls away *continuously* instead of switching to blurred along a
    traced silhouette, which is the giveaway of a masked blur. **Amount** sets
    the maximum blur as a percentage of the frame (so it means the same thing on
    an 800 × 600 and a 40 MP photo), and **Focus on a click** aims the focal
    plane by clicking the thing you want sharp rather than by hunting with the
    slider. **Falloff** is the artistic override: 1 is what the optics actually
    do, higher holds more of the scene near-sharp and then drops it away at the
    extremes, lower blurs everything off the focal plane at once.
    **Highlight bloom** restores what an average destroys — a lens spreads a
    pinpoint of sun across the whole aperture, so it comes out *as bright as the
    highlight was* rather than as a faintly lighter smudge.

    **Aperture** chooses which lens the out-of-focus shapes come from. A
    photographic **lens** is round in the middle of the frame, and **Swirl**
    clips it to a progressively thinner cat's eye toward the edges — the
    Helios 44-2 look, where a busy background appears to rotate around the
    subject — darkening the corners as it goes, because a narrowed aperture is
    passing less light. A **mirror** lens is a reflecting telephoto, whose
    secondary mirror shadows the middle of the aperture: **Hole Ø** sets how
    much, and every highlight comes out as the doughnut those lenses are known
    for. The two are alternatives, not a mix, for reasons of optics as much as
    arithmetic — catadioptrics don't swirl.

    **Show** switches the same node between three views: the finished *bokeh*,
    the *depth* map the model saw (useful for aiming Focus), and *kernels* — the
    apertures themselves outlined at true size on a grid over the photo, so you
    can read the blur's size and shape across the frame before committing.
    **Kernel grid** sets how many columns that grid has; wound up, it puts
    several samples either side of a depth edge. Model weights (~99 MB) download
    the first time you press the Bokeh button.
  - *Sobel edges*, *Floyd–Steinberg dither*
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
- **Or name the object** — type what you want into the box beside the picker —
  *the dog*, *the sky*, *the red car* — and press **Find**. A
  [CLIPSeg](https://github.com/timojl/clipseg) model reads the words against the
  picture and works out *where* the thing is; **SAM still traces the boundary**,
  so a named selection has exactly the same crisp edges as a clicked one, and
  the level dropdown still re-ranks it. The size of what it found chooses the
  level for you, which is how *the sky* and *a cloud* come out as different
  selections. The line under the row says what happened — *Found "the hair" —
  3.3% of the frame* — and says **weak match** when nothing in the picture
  really fits, rather than pretending; the outline is drawn either way, so you
  can see for yourself in a glance. Phrasing is not something you have to get
  right: the query is tried under several wordings at once, so *the person* and
  *a person* find the same subject. It works in the edit dialog too, where
  clicking cannot, so you can change which object an existing effect is limited
  to. Model weights (~139 MB) download the first time you press **Find**.
- **Saved masks** — **selecting an object and applying an effect saves that
  object automatically**, whether you clicked it or named it, freezing its
  pixels to disk as a durable mask rather than a pick that has to be
  re-segmented. It appears as a silhouette icon under the picker, named for you
  (*Object 1*, *Object 2*, …) and already
  ticked, so the next effect lands on exactly the same region — that is what
  lets you stack several effects on one object without re-selecting it each
  time. The shape is identical every time, and it survives deleting the node it
  was picked on. **Tick several and they combine** — the effect applies to
  their union, so one blur can cover two objects. **Save selection** banks an
  object you are not ready to use yet. A mask belongs to its image, and one
  still in use cannot be deleted out from under the nodes that reference it.
- **Crop & rotate** — **3 · Frame** opens a framing tool: drag across the image
  to draw a crop frame, and drag its handles or edges to adjust. To straighten,
  click **Straighten** and drag a line along whatever should be horizontal — the
  horizon, a windowsill — and the image turns to level it. Draw another to
  refine. The image is rotated about its center onto an expanded canvas and the
  frame is taken from *that*, so a frame may include the empty corners a rotation
  leaves — they come out black, which is often what you want for a deliberate
  tilt. While framing, the preview shows the rotated but uncropped image with
  everything outside the frame dimmed, so you can see what you are giving up, and
  the panel names the frame's true size in pixels — what you would actually
  export. **Save frame** applies it everywhere at once: preview, gallery
  thumbnails, and Export.

  The crop is *not* a step in the work tree — it belongs to the image and is
  applied after every effect, on the way out. So re-framing costs nothing but
  the crop itself (no effect is recomputed), saved masks stay in the image's
  original coordinates and never need re-picking, and presets carry no framing
  with them.
- **Presets** — save the chain that produced a node as a named recipe and
  replay it on any other image. A preset stores its steps with *relative*
  parent references, not node ids, so a branching recipe (say `edges` blended
  back onto the original) reproduces its shape on whatever node you apply it
  to. Tick an object first and the whole recipe is confined to it — every step
  is masked to what you ticked, in place of whatever the recipe captured, so a
  chain built on one photo's subject can be aimed at another photo's. Applying
  one is all-or-nothing: if any step fails, the nodes it already created are
  rolled back.
- **Image map** — **Image map**, under the gallery, plots the whole library as a
  rotatable 3D point cloud in which photos of similar things sit near each
  other: crocuses beside crocuses, graffiti beside graffiti, and duplicate
  uploads on top of one another. Each image is embedded by the vision half of
  [CLIP](https://openai.com/research/clip), which was trained against captions
  and so groups by *subject*, not by palette — a red car does not land beside a
  red sunset. Drag to rotate, tick **Spin** (or double-click the cloud) to let it
  turn on its own, and click a point for that image's thumbnail and an **Open**
  button, which makes the map a way to navigate a large library by eye. A row of
  chips filters by *edit* rather than by subject — **Edited**, **Untouched**, and
  one per kind of effect the library actually contains, each with its count;
  lighting several widens the filter rather than narrowing it. Below it a second
  row filters by what is *in* the picture, from a pass of
  [YOLOv8](https://docs.ultralytics.com/models/yolov8/) over the library: a chip
  per object it found, commonest first. Its vocabulary is COCO's 80 everyday
  things, so it is precise about a person, a dog or a bicycle and has no word at
  all for a macaque or a temple — an object with no chip is one the detector
  cannot name, not one your library lacks. The two rows compose with the search
  box, which is where they earn their keep: searching *people* finds the people
  and also some macaques and empty landscapes, because CLIP reads the whole
  frame, and lighting **person** removes them because the detector scores a
  region. Type a word that *names* one of the 80 classes — *people*, *dogs*,
  *a cat*, *bike* — and its chip lights automatically, with the status line
  saying so; press it to search without it. A word the detector has no class
  for lights nothing, and a chip you press yourself is never overridden by a
  later search. **PCA** is stable,
  so adding an image nudges the map rather than reshuffling it; **t-SNE** draws
  tighter clumps but its layout is not comparable between runs. Points are
  painted in each photo's average color. Model weights (~335 MB) download on
  first use, and each image is embedded once (~12 ms) and cached. That first run
  reports itself — the status line counts the megabytes down and then the
  images — instead of leaving the modal blank for a minute. The fitted layout is
  cached too, keyed by the vectors that produced it, so reopening the map is
  instant until the library itself changes.
- **Preview zoom** — scroll to zoom (cursor-anchored), drag to pan,
  double-click to reset.
- **Export** — download any node's render as a JPG named after its effect
  chain (e.g. `photo-posterize-blur.jpg`).
- **Library stats** — image, node, preset, and mask counts plus disk usage. The
  render cache and the downloaded model weights are reported apart from the
  database, originals, and saved masks: between them they are the bulk of the
  bytes and both come back on their own (one regenerates, the other
  re-downloads), so a single "on disk" figure would badly misstate what you
  would actually lose.

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

Model weights are fetched to `data/models/` from revision-pinned Hugging Face
URLs, each on the first gesture that needs it and never before: segmentation
(~45 MB) when you click to segment, text-prompted segmentation (~139 MB) when
you press **Find** to name an object, depth (~99 MB) when you press **Bokeh**,
CLIP's vision tower (~335 MB) and YOLOv8n (~12.8 MB) when you open the Image
map, and CLIP's text tower (254 MB) only if you search there. To supply your own
ONNX exports instead, point the matching env var at an absolute path —
`PICKY_SAM_ENCODER`, `PICKY_SAM_DECODER`, `PICKY_CLIPSEG_MODEL`,
`PICKY_DEPTH_MODEL`, `PICKY_CLIP_MODEL`, `PICKY_CLIP_TEXT_MODEL`, or
`PICKY_YOLO_MODEL`. Nothing else in the app requires any of the downloads.

The detector's weights are the one exception to this project's licensing:
YOLOv8n is Ultralytics', under **AGPL-3.0**, where every other model here is
Apache-2.0 or MIT. That is worth reading before distributing Picky; running it
locally is unaffected.

## Architecture

```
server/
  main.py       FastAPI routes + static file serving
  db.py         SQLite schema and queries (images, nodes, presets, masks)
  effects.py    effect registry: each effect maps an RGB numpy array -> array
  rendering.py  node render pipeline with per-node JPEG cache
  sam.py        MobileSAM via onnxruntime: click a point, get a mask
  clipseg.py    CLIPSeg via onnxruntime: a phrase in, the place it means out
  depth.py      Depth Anything V2 via onnxruntime: a photo in, relative depth out
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

**Naming an object produces a click.** CLIPSeg answers with a 352 × 352 heat
map, which is far too coarse to be a mask — so the server does not use it as
one. It smooths the map, takes its peak as a point, measures how much of the
frame the hot region covers, and asks SAM which of its candidate masks at that
point is closest to that size. What comes back over the wire is an `{x, y,
level}`, exactly what clicking produces, and the browser stores it as exactly
that. Nothing downstream — the outline, banking, presets, invert — can tell
which one you did, and no CLIPSeg pixel is ever stored or shown. The one thing
this rests on is that the code choosing the level and the code re-decoding it
later rank SAM's candidates through the same helper; if those two orderings
drifted apart, a saved selection would quietly resolve to a different mask than
the one you were shown.

Two details are worth knowing if you change it. The model is pinned to its
*quantized* export, a quarter the size of the float one, on the argument that
its output only has to locate — the boundary is SAM's work, so quantization
error would have to move the peak off the subject before it cost anything. And
the query is run under four phrasings in one batch, combined by taking the
strongest response at each pixel rather than the average: a wording that simply
misses contributes zeros, and averaging those in was enough to drop a real
subject below the confidence floor over nothing but the choice between *the*
and *a*.

Bokeh's variable blur is a **layered gather**. Depth is split into bands whose
weights sum to 1 at every pixel; each band is filtered at its own radius and the
bands are composited far to near with premultiplied alpha. The layering is not an
optimization — it is what corrects the direction of the operation. A real defocus
*scatters*, each scene point spreading its energy over its own circle of
confusion, whereas a filter *gathers*, each output pixel averaging a
neighbourhood that may straddle two surfaces. Blurring the whole frame at each
radius and interpolating would let the subject average into the background and
smear back out as a halo — the same tell as the masked blur this replaces.
Carrying each band's coverage as alpha is what keeps the sharp subject in front
of a blur that never reaches into it.

The kernels themselves are flat apertures, normalized to average. Summing one tap
by tap would be O(w·h·r²), so each is decomposed into horizontal runs and read
out of a cumulative sum along x — two lookups per run, O(w·h·r) and exact. The
round and annular apertures are the same kernel at every pixel and so are
genuinely convolutions; the cat's eye is not, since its shape depends where in
the frame it sits. That one is a linear *shift-variant* filter, held constant
over 64 px tiles, and it stays affordable because the cumulative sum does not
depend on the kernel at all: one is built for the whole frame and each tile reads
its own offsets out of it, with no halo and no extra data. Everything blurred is
computed on a canvas bounded in both size and radius and recombined with the
full-resolution original wherever the blur already exceeds the detail the
downsample threw away, which is what keeps a 40 MP render to seconds and tens of
megabytes. The depth map is cached in-process against a hash of the pixels rather
than beside the render, because an effect is a plain registry entry that never
learns its node's id.

The crop deliberately sits *outside* the work tree, as an output stage applied
after every effect. Every registry effect maps an array to an array of the same
size, and much of the app depends on that — most of all saved masks, whose frozen
pixels line up with any node of their image precisely because every node shares
the original's dimensions. A crop node would break that and force masks to be
warped forward through it; a crop applied on the way out leaves it intact. The
practical payoffs: re-framing recomputes no effect (only the framing itself is
cached, as `renders/<id>.out.jpg`, and skipped entirely when the crop is the
identity), and clicks on the framed preview are mapped back into node
coordinates by an affine the server publishes with the image — so the browser
places a pick without knowing any of the geometry.

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
| GET    | `/api/images` | list images, each with its `crop` and the set of `effects` its work tree carries |
| GET    | `/api/images/{id}` | one image with its `crop` and the `geometry` that crop implies (output size, and the affine mapping a framed pixel back to the node's own space) |
| PUT    | `/api/images/{id}/crop` | set the image's framing (`crop`: `{angle, rect}`, or `null` to clear); drops only the framed-output cache, never the effect renders |
| POST   | `/api/images/{id}/crop-preview` | the rotated but *unframed* proxy the frame editor drags over (`node_id`, `angle`), rendered small; `X-Canvas-Width`/`-Height` report the full-size canvas it stands for, so the editor can name true output pixels |
| GET    | `/api/images/{id}/tree` | all work-tree nodes for an image |
| POST   | `/api/images/{id}/nodes` | apply an effect (`parent_id`, `effect`, `params`, optional `parent2_id` for blend, optional `selection` to mask it) |
| PATCH  | `/api/nodes/{id}` | edit a node's `params`, `parent2_id`, or `selection` in place, re-rendering it and dropping its descendants' caches |
| GET    | `/api/nodes/{id}/render` | rendered JPEG (`?thumb=1` thumbnail, `?download=1` attachment) |
| POST   | `/api/nodes/{id}/preview` | render an effect on top of a node in memory — no node created, nothing cached |
| POST   | `/api/nodes/{id}/mask` | outline PNG for a selection — `points` segmented on the fly, or `masks` (saved ids) read back, plus an optional `invert`; persists nothing but the node's cached embedding |
| POST   | `/api/nodes/{id}/select-text` | find a named object (`query`) in a node's pixels, answering with the `x`, `y` and `level` a click would have produced, plus the `score`, the `coverage` the resulting mask has, and `confident`. Stores nothing; 409s rather than downloading the model inside the request |
| GET    | `/api/nodes/{id}/clusters` | k-means scatter data for posterize nodes |
| GET    | `/api/nodes/{id}/histogram` | 256-bin luma histogram of a node's render, for the tone-curve editor's backdrop |
| GET    | `/api/nodes/{id}/depth-at` | the depth under one pixel (`x`, `y`) on bokeh's own normalized scale — what click-to-focus reads a `focus` value off; stores nothing |
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
| GET    | `/api/embedding-map` | the whole library as 3D points (`method`: `tsne`\|`pca`). Embeds anything it finds missing, so it answers with or without a `prepare`; the fit itself is cached against the vectors that produced it |
| GET    | `/api/embedding-map/search` | score every image against a phrase (`q`), as a raw CLIP cosine, the probability the phrase beats 850 rival subjects, and a z-score over this query's own spread; plus `tag`, the COCO class the phrase names, if any. 409s rather than downloading the text tower inside a GET |
| POST   | `/api/embedding-map/prepare` | start the background embedding pass (~335 MB of weights on a fresh install, then one forward pass per image); answers `done` synchronously when nothing is missing |
| GET    | `/api/embedding-map/progress` | where that pass got to — `state`, `phase`, `done`/`total` |
| POST   | `/api/detect/prepare` | start the background object-detection pass (12.8 MB of weights on a fresh install, then ~40 ms per image); answers `done` synchronously when every image is already labelled |
| GET    | `/api/detect/progress` | where that pass got to — `state`, `phase`, `done`/`total`, plus `ready` for weights an earlier run fetched and `pending` for images still unlabelled |
| GET    | `/api/images/{id}/detections` | the objects found in one image — `label`, `score`, and a `box` in fractions of the framed image — plus `detected`, which tells "found nothing" apart from "never looked" |
| POST   | `/api/text-model/prepare` | start fetching CLIP's text tower (254 MB), which only searching needs |
| GET    | `/api/text-model/progress` | where that download got to, including `ready` for a tower an earlier run already fetched |
| POST   | `/api/depth-model/prepare` | start fetching the depth model (99 MB), which only Bokeh needs |
| GET    | `/api/depth-model/progress` | where that download got to |
| POST   | `/api/select-model/prepare` | start fetching the text-prompted segmentation model (139 MB), which only naming an object needs |
| GET    | `/api/select-model/progress` | where that download got to, including `ready` for a model an earlier run already fetched |
| GET    | `/api/agent` | whether an `ANTHROPIC_API_KEY` is configured, and which model the command row would talk to; the frontend hides the row when it is not |
| POST   | `/api/agent` | run one turn (`prompt`, the browser's own `history`, and the `image_id`/`node_id` on screen). Answers with the `reply`, the `steps` it took, the updated `history`, a `focus` to navigate to, a `pending` action awaiting confirmation, and the `wire` record of the API calls the turn actually made |
| GET    | `/api/stats` | library counts and disk usage |
