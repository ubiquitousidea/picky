# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
python3 -m venv .venv                      # Python version comes from .tool-versions (asdf)
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn server.main:app --port 8000   # serves API + frontend at http://localhost:8000
```

There is no build step, linter, or test suite. Verification is done by exercising
the API with curl (upload a JPG, create nodes, fetch renders) and comparing
rendered pixels with Pillow/NumPy against expected transforms. The frontend can
be checked with `node --check web/app.js` and screenshotted headlessly:
`"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --screenshot=out.png "http://localhost:8000/?image=N&node=M"`.

Verification gotchas:

- Port 8000 is often occupied by the user's own dev server — run test instances
  on a spare port, and leave the user's process alone.
- Invoke Python as `.venv/bin/python` with an absolute path; the bare `python3`
  asdf shim fails outside the repo (no `.tool-versions`), e.g. in scratch dirs.
- Headless Chrome may snapshot before the large preview JPEG finishes loading —
  add `--virtual-time-budget=10000` (and `--hide-scrollbars` for clean shots).
- `data/` is the user's real library, shared by every server instance. Create
  test images via the API and DELETE them when done; never wipe the directory.

## Architecture

Picky is a JPG effects app: FastAPI backend (`server/`), vanilla-JS single-page
frontend (`web/`, no framework, no build), SQLite + files under `data/`
(gitignored: `picky.db`, `originals/<image_id>.jpg`, `renders/<node_id>.jpg`).

The core model is a **branching work tree per image** (git-like, not an undo
stack). Each `nodes` row has `parent_id` (NULL = root, which *is* the original
JPG), an `effect` name, and JSON `params`. Blend nodes additionally carry
`parent2_id` — the tree is really a DAG with at most two parents.

Key invariants that cut across files:

- **A render always matches its node's *current* params.** `rendering.render_node()`
  materializes a node by applying its effect to the parent's *cached* JPEG,
  recursively, and caches the result; it only recomputes if the cache file is
  missing. File-existence is the entire cache key, so **invalidation means
  deleting files** — that is what `PATCH /api/nodes/{id}` does when it edits
  params in place: `db.descendant_ids()` (read-only twin of `delete_node`'s
  recursive CTE, over *both* parent links) plus `rendering.delete_render_files()`
  drops the node's whole descendant closure, then the edited node alone is
  re-rendered eagerly so a bad param fails on the PATCH instead of on some later
  GET. Descendants come back lazily on their next request. Same for the per-node
  k-means cluster JSON cache (`renders/<id>.clusters.json`).
  Two things this does NOT cover: changing an effect's *implementation* still
  does not retroactively change existing nodes (nothing deletes their files), and
  neither does replaying a preset, which runs today's code to create new nodes.
- **PATCH updates the row, then invalidates — never the reverse.** These are sync
  `def` endpoints, so FastAPI runs them concurrently in a threadpool. After the
  commit, any racing re-render already uses the new params, so the file sweep can
  only be deleting output from the old ones. Sweeping first would let a
  concurrent read re-materialize a stale cache that then looks like a valid hit
  forever. `update_node` also short-circuits when nothing actually changed, so
  re-saving an unedited form doesn't throw away a subtree of renders.
- **Editing a blend's second input needs a cycle guard; creating one does not.**
  `parent2_id` must be `< node_id`. Node ids are topological, so a smaller id
  cannot be downstream — without this, pointing a blend at its own descendant
  sends `render_node` into infinite recursion. Creation is safe for free (a new
  node always has the largest id); an in-place edit is not. The check also
  preserves the "both parents have smaller ids" property `layoutGraph()` relies
  on. The frontend's target picker passes the same bound (`maxId`) so the
  impossible options never appear.
- **Effects are registry entries.** `server/effects.py` `EFFECTS` maps name →
  `apply(np.uint8 RGB array, params) -> array` plus param specs (`int` ranges,
  `float` ranges with optional `step`, or `choice`). The frontend generates its
  controls from `GET /api/effects`, so a new effect is one registry entry and
  nothing else. Blend is the exception: it takes two images, so it lives
  outside `EFFECTS` (`BLEND_SPEC`, `apply_blend`) and is special-cased in
  `main.py` and `rendering.py`. Blend's `weight` param (share of the second
  image) only affects the `average` mode; `rendering.py` reads it with
  `.get("weight", 0.5)` because nodes created before the param existed lack
  the key — keep that default when touching the call site. Pixelate's `shape`
  (`square` | `hexagon`) is read the same way, for the same reason: a param
  added to a shipped effect is always absent from the rows already on disk,
  since nothing migrates node params and `validate_params` only backfills on
  the next write.
- **Previews never persist, and run by default.** `POST /api/nodes/{id}/preview`
  renders an effect against a node's cached pixels entirely in memory
  (`rendering.render_preview`) — no DB row, no file in `data/renders/`. It shares
  the effect-application core (`rendering._apply`) with `render_node` so preview
  and Apply can never disagree. Frontend preview mode lives in the `preview`
  state object in `web/app.js`; every path that changes the selection exits
  preview through the `exitPreview()` call at the top of `renderSelection()` —
  keep that single choke point — and `renderSelection()` re-enters at the *end*
  via `enterApplyPreview()`, which is debounced so clicking through tree nodes
  coalesces into one full-resolution render. There is no Preview button: the
  `#live-preview` checkbox (default on, persisted in `localStorage`) is the only
  control, and unchecking it restores the node's own render — that is the
  "show me the original" gesture. `preview.source` names whoever is currently
  driving it (the Apply panel or the edit modal) and is the *only* thing that
  varies: one debounce, one `seq` stale-response counter, one owner of the blob
  URL. Note the edit modal previews against the edited node's **parent** —
  re-applying its effect to its own input — because the endpoint composes on
  *top* of the node you name. `openEdit()` must call `exitPreview(false)`
  **unconditionally** before installing its own source: live preview leaves a
  debounce timer armed, and only `exitPreview()` clears it, so skipping that lets
  a stale timer re-fire against whatever source it finds. Symmetrically
  `closeEdit(true)` (Cancel and Esc) hands the preview back to the Apply panel;
  the Save path gets there through `selectImage()`.
- **A selection lives in a store, not in the DOM, and in its own panel
  section.** `state.selection` (Apply panel) and `edit.selection` (modal) each
  hold `{value, nodeId, imageId}`; `appendSelectionControls()` takes the store
  as a parameter and `readParams()` reads it back. It used to ride on a hidden
  `<input data-selection>` inside `#effect-params`, which `buildParamControls()`'s
  `container.innerHTML = ""` destroyed on every effect switch — the pick vanished
  while the overlay stayed painted, so the next preview silently covered the
  whole image. The Apply panel's copy now renders into its own
  `#select-controls` section (step 1, *above* the effect picker), which removes
  that hazard at the source and matches the workflow: you pick and save objects
  before you choose what to do to them. `pruneSelection()` at the top of
  `renderSelectControls()` is the one place a selection expires, and the two
  shapes expire differently: **click points** die when `nodeId` changes (their
  coords are in that node's pixel space), **saved masks** only when `imageId`
  does (their pixels are frozen and every node of an image shares dimensions) —
  which is what lets you stack several masked effects on one object.
  Corollaries: `exitPreview()` deliberately does *not* clear the overlay (its
  lifetime tracks the store, so unchecking live preview leaves a pick lit);
  `appendSelectionControls()` opens with `disarmPick()`, because a rebuild is
  exactly when an armed picker's commit closure becomes detached; and
  `closeEdit()` repaints the Apply panel's selection, since the modal borrows
  the single overlay.
- **The saved-mask grid is part of the selection control, not a section of its
  own.** Ticking a mask *is* editing the selection, so the picker, the
  level/invert options, "Save selection", and the saved objects are one
  component with one render path — the panel and the store cannot drift. Its
  `allowManage` flag is off in the edit modal (deleting a mask the node uses
  would 409 anyway); `allowPick` is off there too, since re-picking needs a
  click on a page `showModal()` made inert. The chips are built **once** per
  control and only restyled on a toggle: rebuilding them would detach the very
  `<li>` whose click handler is mid-flight.
  A mask is a shape, so it is drawn as a **picture, not a row of text** — a grid
  of square chips, each an `<img>` of `GET /api/masks/{id}/thumb`, with the
  name, dimensions and used-by count in the `title`. A preset stays a text row
  because a recipe has no picture. The `src` carries `?v=<created_at>` because
  the icon is served with a year-long `max-age` (the grid is rebuilt on every
  selection change, so uncached it would refetch every icon) and mask ids are
  rowids that SQLite hands back out after a delete.
- **Saved masks freeze pixels; click points are a recipe — and both are
  unions.** `selection` has two shapes: `{points: [{x, y, level}, ...], invert}`
  re-segmented by SAM, or `{masks: [id, ...], invert}` loaded from
  `data/masks/<id>.png`. The members are OR'd and `invert` applies once, to the
  result. The *click* shape is a list even though the UI only ever produces one
  point, because that is the only place a multi-mask selection can degrade to
  when a preset captures it (`_portable_selection`) — see the preset bullet.
  `effects.validate_selection` accepts both, stays total, and also normalizes
  the pre-union spellings (`{mask, invert}`, `{x, y, invert, level}`) that node
  rows, mask specs and stored presets on disk still hold. **Nothing migrates
  those on disk; they upgrade on read**, in `db.node_dict` — so exactly one
  shape ever leaves the database, which is also what keeps `update_node`'s
  no-op check from needlessly nuking a subtree when the stored row was written
  in an older spelling. `db` imports `effects` for this; that is the acyclic
  direction, since `effects` imports no DB. `rendering.compute_mask` normalizes
  again on entry, deliberately: every caller already hands it a clean selection
  (via `node_dict` or `main.py`'s request validation), so that call is the belt
  to their braces — it is what keeps the mask-building function total on its
  own rather than by convention.
  **A click selection is frozen automatically the moment it is used.**
  `bankSelection()` runs in `applyEffect()` *before* the node POST, so the store
  crosses over to the mask shape and survives `pruneSelection()` into the node
  it just made — without it, using a pick destroys it, since Apply selects the
  new node and a point's coords only mean anything in the node they were picked
  on. That is also what stops the *next* Apply freezing a second copy: it finds
  `{masks}` and no-ops. This lives in the frontend rather than the `create_node`
  endpoint on purpose — from the server the store would still hold the points,
  so every Apply would duplicate, and preset replay (which builds nodes through
  `db.create_node`) would mint a mask per masked step. A failed bank still
  applies the effect, with the click selection, and says so in one alert
  afterwards: a bookkeeping failure is no reason to refuse the edit, but it must
  not be silent either. The explicit "Save selection" button survives for
  banking an object you are not ready to use, and shares the same function.
  Names are **server-generated** (`_auto_mask_name`: the lowest unused
  `Object N` for that image, retried on `IntegrityError` because these are sync
  `def` endpoints running concurrently in a threadpool) — the grid identifies a
  mask by its picture, so a name only has to satisfy the `masks(image_id, name)`
  index and read sensibly in a tooltip. `MaskCreate.name` is therefore optional;
  an explicitly *empty* one is still a 400, and a caller-supplied one still 409s
  without retrying. `db.list_masks` orders by `created_at, id` — not by name
  (text order puts `Object 10` before `Object 2`) and not by id alone (rowids
  are reused, so a brand-new mask could sort into the middle of the grid).
  Masks live *outside* `renders/` on purpose: that directory is a regenerable
  cache swept by node id (`delete_render_files`), and a saved mask is user data
  keyed by mask id whose whole point is that it is *not* regenerable. Stored
  1-bit and colorized on read (`rendering._overlay_png`), so
  `POST /api/nodes/{id}/mask` serves both shapes and the frontend overlay needs
  no branch. A mask's **icon is not a file** (`rendering.mask_thumb_png`), for
  the same reason the histogram is not: ~15 ms even on a 7728×5152 mask and
  under a kilobyte on the wire, cheaper than the `load_mask` these endpoints
  already run. A `<id>.thumb.png` would be a second file keyed by a rowid
  SQLite reuses, so one missed unlink would serve a deleted mask's silhouette
  as its successor's, and it would need excluding from `storage_stats`, whose
  masks line means "the bytes you cannot regenerate". Derived from the one file
  whose deletion is already correct, it cannot go stale. Two Pillow details are
  load-bearing: a stored mask opens as mode `"1"`, which resizes NEAREST
  whatever filter you pass, so it is converted to `"L"` first; and the resample
  is `BOX`, because on a binary mask that is exactly "what fraction of this
  source cell was selected" while cubic filters ring — bright halos outside the
  object, dark ones inside. The PNG is frozen **post-invert** — what you saw is what you saved —
  so a reference's own `invert` toggles on top, which is why `bankSelection()`
  follows up with `invert: false` and why `_portable_selection` XORs. Masks belong to the
  image they were picked on (`_check_selection` 400s if *any* member is foreign,
  since they are OR'd into one image's dimensions); deleting one that nodes
  reference is a 409 rather than a silent repaint of committed pixels, and
  finding those referrers needs **two arms** — a `json_extract` for the legacy
  scalar and a `json_each` for the array. Both live in one shared `MASK_REFS`
  constant, which `db.nodes_using_mask` filters (it guards the delete) and
  `db.list_masks(with_use_counts=True)` groups (it renders the `used_by` warning
  that *predicts* the delete), for the same reason `descendant_ids` and
  `delete_node` share `DESCENDANT_CTE`: a check and the thing it checks must not
  be able to drift. They did while the SQL was duplicated — the count's join
  compared a typeless subquery column against `masks.id`, whose INTEGER affinity
  coerced a TEXT id that the guard's `= ?` against a bound int did not match, so
  a `{"masks": ["7"]}` row was counted but not refused. Hence the `CAST` inside
  `MASK_REFS`: it puts both on the one interpretation
  `effects.validate_selection` already applies on read (Python's `int()`).
  `MASK_REFS` is `UNION ALL`, so each caller spells its own dedupe — `DISTINCT`
  for the id list, `COUNT(DISTINCT node_id)` for the counts — which is what
  collapses a node holding both spellings. It deliberately cannot be filtered by
  `image_id` (~16x cheaper, and the obvious optimization), since a cross-image
  reference is exactly what must still block a delete.
- **`used_by` is joined into the mask list, not merged in afterwards.** The
  counting is a `with_use_counts` flag on `db.list_masks` rather than a second
  function, so the list and its counts are one query, one connection, one
  snapshot — as two queries the counts were read *first*, so a node created in
  between made an in-use mask report `used_by: 0`, dropping the warning and
  letting the user click into the 409 it exists to prevent. The flag is opt-in
  because the join scans every node in the library and two of the three callers
  (`_check_selection`'s id set, `_auto_mask_name`'s taken-names set) want
  neither the number nor the scan. `mask_dict` omits the key entirely when
  nobody counted, so `used_by: 0` always means *checked and unused* rather than
  *not looked at*.
- **The overlay is an outline, not a fill.** `_overlay_png` emits the mask's
  4-neighbour inner boundary, dilated to a width scaled by image size (the
  overlay is displayed fitted to the panel, so a 1px line on a 4000px photo
  would vanish), as an opaque white core over a dark casing so it reads on any
  photograph. It is transparent everywhere else, which is why `#mask-overlay`
  carries no `opacity` — a translucent white fill used to hide the very pixels
  you were picking.
- **Param controls are built and read in one place.** `buildParamControls()` /
  `appendBlendTarget()` / `appendSelectionControls()` / `readParams()` in
  `web/app.js` are shared by the Apply panel and the edit modal, which is what
  keeps ranges, defaults, and value coercion from drifting between "create a
  node" and "edit a node". The blend target picker is found by the `.blend-with`
  class *scoped to its container*, not a global id, so two of them can coexist
  while the modal is open. The modal cannot re-*pick* a point (`showModal()`
  makes the page inert) but can still choose a saved mask, which needs no click.
- **The effect picker is a row of icon buttons, and its selection lives in
  `state.effect`** — not in the DOM, as it did when it was a `<select>`.
  `EFFECT_BUTTONS` in `web/app.js` maps one button to one or more registry
  effects, `setEffect()` is the single write path (retoggle buttons → rebuild
  params → re-preview), and each icon is inline SVG painting with
  `currentColor` so it takes its hue from the `.fx-<name>` class already on the
  button. Two buttons stand for a pair of effects behind a **Method** dropdown,
  because each pair is one idea with two algorithms: posterize/dither (reduce to
  N colors) and curves/gamma (reshape tone). That grouping is presentation only:
  `EFFECTS` still holds two entries per pair, nodes still store `posterize` or
  `dither`, `curves` or `gamma`, each gets its own `.fx-*` hue in the tree, and
  the cluster plot is still posterize-only (as the curve editor is
  curves-only). The Method row is built by `renderEffectControls()`, *not*
  `buildParamControls()` — the edit modal shares the latter and must never grow
  the control, since `PATCH` edits params, not a node's effect. Its `<select>`
  deliberately carries no `data-param`, which is what keeps `readParams()` from
  posting it as one.
- **The tone curve's LUT is implemented twice, on purpose.**
  `effects._curve_lut` and `curveLut()` in `web/app.js` are the same
  Fritsch–Carlson monotone cubic interpolation, line for line. The editor has to
  draw the exact transfer function the server will apply, and asking the server
  would cost a round trip per `pointermove` — so the duplication buys WYSIWYG.
  Change one, change the other; the interpolation is *monotone* rather than
  natural-cubic because a plain spline overshoots between distant control points
  and a tone curve that dips below its neighbours inverts local contrast.
  `points` is also the first param type that is not a scalar, so it needs an
  explicit branch in **both** `effects.validate_params` (whose `else` is a
  catch-all `int()`) and `buildParamControls()` (whose `else` is a catch-all
  range slider), plus one in `readParams()` — the points ride on a hidden
  `<input data-param>` as JSON precisely so `[data-param]` still finds them and
  a dispatched `input` event still drives the shared preview debounce.
  `_clean_points` is deliberately *total*: it clamps and falls back rather than
  raising, because `validate_params` is called unguarded and an exception there
  would be a 500 rather than a 400.
- **The curve editor's histogram (`GET /api/nodes/{id}/histogram`) is not
  cached**, unlike the posterize cluster JSON: it is one pass over a
  `draft()`-reduced decode, far cheaper than the render it reads. That is what
  keeps it out of `delete_render_files` and `storage_stats` — no new file kind
  under `renders/`, no new invalidation edge. It describes the node the effect
  will be applied *to*, which is why `buildParamControls()` takes a
  `sourceNodeId`: `state.nodeId` from the Apply panel, `node.parent_id` from the
  edit modal — the same node each one previews against.
- **Deletes cascade through both parent links** (`db.delete_node`'s recursive
  CTE), and no deletion order is FK-safe for two-parent graphs, so deletes run
  with `PRAGMA defer_foreign_keys = ON`. File cleanup (render/thumb/clusters)
  happens in `rendering.delete_render_files` from the returned id list.
- **Presets store a sub-DAG with relative parent indices**, never node ids.
  `main.capture_steps()` collects the selected node's ancestor closure over
  *both* parent links (that closure is exactly its pixel-dependency set) in id
  order, and rewrites each parent reference as an index: `0` = whatever node the
  preset is applied to, `1..i-1` = an earlier step. That is what makes an
  `edges` → `blend with the original` chain replay correctly on a different
  image. Absolute ids would be wrong across images and unsafe even within one,
  since SQLite reuses rowids — hence no FK from `presets` to `nodes`. Steps are
  stored as `{"version": n, "steps": [...]}` and params are re-normalized through
  `validate_params` at both save and apply, which also backfills params added
  since the node was made (e.g. blend's `weight`). A **saved-mask selection
  degrades back to the click points it was frozen from** (`_portable_selection`):
  a preset stores params, not pixels, and a mask is both pixels and image-scoped,
  so it must replay as the recipe that produced it. Dropping the selection
  instead would quietly turn a masked blur into a whole-image blur. This is the
  reason the click shape is a union too — a union of masks has nowhere else to
  land. Inverts are where it goes lossy: one mask XORs exactly, but several have
  no single invert that reproduces the union, so a mask whose spec was inverted
  is dropped from the step (the rest still replay — better than discarding the
  whole selection). `_validate_steps` strips any mask ref that reached a stored
  preset anyway, as older or hand-edited data.
- **Applying a preset is all-or-nothing, and cannot be one transaction.**
  `rendering.render_node` reads through its own connection, so a step's row must
  be committed before the next step can reference it — the loop is
  create → commit → render per step, with `_validate_steps` pre-flighting the
  whole recipe first and a compensating `db.delete_nodes` +
  `rendering.delete_render_files` unwinding on failure. The file sweep is not
  optional: `render_node` treats any existing file at the render path as a cache
  hit, so an orphaned render would later be served as a different node's pixels.
  `db.delete_nodes` deletes highest id first — since a node's id always exceeds
  its parents', that removes referrers before referents and needs no deferred-FK
  window (unlike `delete_node`, whose cascade order is arbitrary).
  A preset stores params, not pixels: replaying one runs *today's* effect code,
  so it need not reproduce the original node's pixels if an effect implementation
  changed since the preset was saved.
- **An apply-time selection masks every step, and beats the recipe's own.**
  `PresetApply.selection` limits a whole chain to a ticked object the same way
  the Apply panel's selection limits one effect — what is ticked is what gets
  edited — so `apply_preset` passes it to every `db.create_node` in place of
  `step["selection"]`. Honouring both instead would mean intersecting a union of
  masks with the union of *points* a step carries (`_portable_selection`'s
  degradation, captured against some other image), and there is no intersection
  shape in the model. It is resolved through `validate_selection` +
  `_check_selection` up beside `_validate_steps`, **before** the loop, so a
  foreign mask is a 400 rather than a half-written chain — same pre-flight
  discipline as the steps themselves. The frontend banks the pick first
  (`applyPreset` calls `bankSelection`, as `applyEffect` does) for the reason in
  the banking bullet, and because one banked mask id then masks every step
  rather than minting one mask per masked step. With nothing ticked the recipe's
  own selections replay exactly as before.
- **Posterize clusters in PCA-whitened RGB space** (`_fit_kmeans`), not raw
  RGB — raw k-means bunches centroids along the luminance diagonal. Whitening
  is manual (`sqrt(explained_variance_ + 1e-4)`) rather than sklearn's
  `whiten=True` so grayscale/solid images (zero chroma variance) don't divide
  by zero. Centroids are inverse-transformed back to RGB; all sampling is
  seeded so the effect and its cluster-plot data agree.
- **Crop & rotate is an output stage, not a node.** One framing per image
  (`images.crop`, JSON `{angle, rect}`, NULL = none), applied *after* the whole
  work tree on the way out. Deliberately not an entry in `EFFECTS`: every
  registry effect maps an array to an array of the same size, and the app leans
  on that everywhere — so a crop node would break the "every node of an image
  shares its dimensions" invariant above, and paying that off meant storing masks
  in the original's space and warping them forward through each node's crops.
  Keeping the crop outside the tree makes that invariant *stay true*: the tree
  computes at original dimensions, **no saved mask is ever warped**, and
  `compute_mask`, `_check_selection` and masked apply need no geometric case.
  Presets are unaffected, because a crop is not a node to capture.
  `rendering.render_output()` is the one place it is applied — every viewer
  (preview, thumbnail, Export) goes through it, and it *short-circuits to
  `render_path` when the crop is the identity*, so an uncropped library pays
  nothing in time or bytes. Otherwise it caches `renders/<id>.out.jpg`; that
  suffix has to be excluded by name in `storage_stats` (a `.out.jpg` is also a
  `.jpg`, like `.thumb.jpg`) and swept in `delete_render_files`. Changing a crop
  sweeps **only** `.out.jpg` + `.thumb.jpg` for that image's nodes — the
  expensive effect renders upstream never saw the crop, and not re-running
  k-means to re-frame is the entire point.
- **`effects.crop_geometry()` is the only place PIL's rotate-expand rounding is
  known.** It returns `{crop, source, canvas, box, output, inverse}`, and
  `apply_geometry` takes its box from it rather than recomputing — so the pixels
  a caller gets are always the ones its `inverse` describes. The arithmetic is
  PIL's own, not a formula for it: `round(w·cos + h·sin)` disagrees with PIL by
  1–2 px at most angles (400×300 at −13° → PIL 458×384, the formula 457×382),
  PIL's expand offset is *rotated* rather than added, and PIL short-circuits a
  quarter turn to a transpose whose size differs from the general path's
  ceil/floor on odd dimensions (2×3 at 90° → 3×2, not 4×3). All three are
  reproduced and were verified exact across 120 size/angle combinations.
  `inverse` is `[a,b,c,d,e,f]` mapping an output pixel back to a source pixel;
  the frontend applies it in `toNodeSpace()` so a click on the framed preview
  becomes a coordinate in the node's own space with no trigonometry — and no
  knowledge of any of the above — in the browser.
- **The mask *outline* is framed; the mask itself never is.** `mask_png` runs the
  overlay PNG through the same `apply_geometry` (NEAREST — the outline is already
  antialiased into its casing, and interpolating an RGBA stencil smears its alpha
  into a halo). That is display-only, so the outline registers on the framed
  preview; the mask that gets *composited* in `_apply` is computed and used at
  node dimensions. Do not confuse the two — the second one is the mask warping
  this design exists to avoid.
- **Schema changes migrate in place** in `db.init()` (PRAGMA table_info check +
  ALTER TABLE) — user databases contain real work, never require a wipe.
- **Node ids are topological.** `GET /api/images/{id}/tree` returns nodes
  ordered by id, and both parents of a node always have smaller ids (SQLite
  autoincrement + parents must exist at creation). The work-tree DAG renderer
  (`layoutGraph()` in `web/app.js`) depends on this to assign commit-graph
  lanes in a single forward pass — one step per node, curves forking at branches
  and merging into blend steps. `layoutGraph()` names no axis: it emits
  `{node, lane, continues, passThrough, parentLinks}` per step, and only
  `buildRailCell()` and the DOM know the graph runs **left to right**, as a
  horizontal strip under the preview (`#tree-section`, its own body-grid row)
  rather than down the side panel. Flow is x, lanes are y. Chips are narrow, so
  the effect's params live in the `title` and the `◎` selection badge rides on
  the id line — the label ellipsises, and an ellipsised badge is an invisible
  one. `renderTree()` scrolls the selected column into view, since the strip
  grows rightward as work accumulates.
- **Frontend files are served `Cache-Control: no-cache`** (`NoCacheStaticFiles`
  in `main.py`). Before this, browsers heuristically cached `app.js` and stale
  scripts produced phantom UI bugs (dead buttons, broken sliders) against a
  new backend. Keep the header; there is deliberately no cache-busting in
  filenames.

The side panel is ordered by the workflow it serves — pick an object, save it as
a mask, repeat, then choose an effect, tick the masks it applies to, tune, and
Apply. So: **1 · Select** (`#select-controls`), **2 · Effect**
(`#effects-section`), **3 · Frame** (`#crop-section`), then Presets and the RGB
cluster plot as collapsed `<details>` (reference material, not part of the loop)
and Export/Delete pinned to the bottom with `margin-top: auto`. The work tree
left the panel entirely for its horizontal strip under the preview.

**3 · Frame** is a `<details>` whose open state *is* crop mode: opening it swaps
the preview for a rotated-but-unframed proxy
(`POST /api/images/{id}/crop-preview`, capped at ~1600 px — straightening
re-renders it, and the frame is stored as fractions of the canvas, so the proxy
and the full-size output describe the same crop), hides the mask overlay, and
arms `#crop-overlay`. Crop mode leaves through
`renderSelection()`'s `exitCropMode(false)`, the same choke point `exitPreview()`
uses, so any change of selection ends it exactly once. Two things about the
overlay are load-bearing and were both bugs first: it is an `<svg>`, so
visibility is toggled with `toggleAttribute("hidden")` — `hidden` is an
`HTMLElement` property that `SVGElement` does not implement, and assigning it
sets a silent JS expando while the attribute (and `display: none`) stays put; and
it carries an invisible but *painted* full-size `<rect>` as a hit target, because
SVG hit testing only finds painted geometry and at a full-size frame the dimming
path encloses zero area — without it the opening "drag across the image" gesture
falls straight through the page. For the same reason a full-size frame treats an
interior drag as a *new* frame rather than a move: a frame with no slack cannot
move, so the gesture the hint describes would do nothing.

**Rotation is a drawn line, not a slider, and the readout's pixels come off the
wire.** The angle was a `<input type="range">` whose every step scheduled a fresh
server rotation of the proxy, so the image lurched along behind the thumb; and an
angle is not what anyone knows about a photograph anyway — they know where the
horizon is. The **Straighten** button arms the overlay to draw a line instead of
a frame (`crop.level` / `crop.line`, and `pointerdown` checks it *before*
`cropHit`), and on release `lineAngle()` turns that line into an angle. Three
things about it are load-bearing: it is a **delta** added to `crop.angle`, since
the proxy on screen is already rotated and a line drawn on it is measured in that
rotated canvas — which is exactly what makes a second line refine the first
rather than restate it; it is measured in **displayed pixels**, because the
stored fractions have a different denominator per axis and an angle taken from
them is skewed by the aspect ratio; and the gesture ends in one
`refreshCropProxy()` with no debounce, which is the whole of the fix — one
gesture, one render. A too-short drag is a tap, so it is discarded and the tool
stays armed rather than silently rotating by whatever a twitch measured.
The pixel readout used to multiply the frame fractions by `#preview`'s
`naturalWidth`, which in crop mode is the *proxy* — a 3000×2000 image reported
1600×1067. The proxy response now carries the full-size rotated canvas in
`X-Canvas-Width`/`-Height` (headers, not a second endpoint: the readout changes
on every frame drag but the canvas only on a change of angle), and `frameSizePx()`
mirrors `crop_geometry`'s box block — including `pyRound`, Python's half-to-even,
which one frame in a few hundred needs to agree to the pixel. The *canvas* is
never recomputed in the browser, so PIL's rotate-expand rounding still lives in
exactly one place.

Gallery thumbnails carry the crop as a `?v=` tag (`cropTag()`), because
re-framing changes an image's pixels without changing its node id and the URL is
what the browser caches — the same trick as the mask grid's `?v=<created_at>`.
The tag is absent when there is no crop, so an unframed library's URLs are
unchanged.

Frontend state lives in one `state` object in `web/app.js`; selection flows
through `selectImage()` → `renderSelection()`, which updates preview, export
link, selection controls, effect controls, tree, and the cluster plot, then
re-enters live preview. The effect controls are rebuilt from scratch on *every*
selection change, not just when the effect is blend: the Apply panel describes
an operation on the selected node, and two of its controls read that node
(blend's target picker, the curve editor's histogram). The rebuild seeds `{}`,
so params return to their registry defaults — that is what resets a dragged tone
curve back to y=x when you switch images. The cluster plot is a hand-rolled
canvas 3D scatter, shown only for posterize nodes. Deep links `?image=N&node=M`
override the localStorage last-image.

`state.presets` is fetched once in `init()` and refreshed only after a save or
delete — `renderSelection()` calls the synchronous `updatePresetControls()`, not
a fetch. Clicking a preset row applies it to the *selected* node, limited to the
*selected* object, and then routes through `selectImage()`, which is what
refetches the tree so all of the preset's new nodes appear. That synchronous
call is also what keeps each row's `title` naming the currently ticked object:
the rows are built by `renderPresets()` (only when the list itself changes) but
titled by `updatePresetControls()`, which is why a row carries
`dataset.presetId` — it is how the title is recomposed without a rebuild.

`state.masks` follows the adapted rule: masks are image-scoped, so they are
fetched in `selectImage()` alongside the tree and refreshed only after a
save/rename/delete — and Apply is now one of those, since it banks the
selection, as is applying a preset, which banks it for the same reason. Neither
path needs an extra refresh call: `applyEffect()` and `applyPreset()` both end
in `selectImage()`, which refetches the list anyway. `bankSelection()` still
appends the new row locally first, so the window between banking and that
refetch — which a failed Apply can leave open indefinitely — never shows a
store pointing at a mask the grid does not have. The grid has no render
function of its own — `renderSelectControls()` draws it as part of the selection
control — so `refreshMasks()` calls that. Ticking a mask chip unions it into the
Apply panel's selection (the analogue of a preset row acting on the selected
node) rather than creating anything.

The two modals are native `<dialog>` elements (Esc and focus trapping for free);
`prompt()`/`confirm()`/`alert()` remain the idiom everywhere else. Two gotchas:
the global `* { margin: 0 }` reset defeats a dialog's centering, so `dialog`
re-sets `margin: auto`; and Esc closes a dialog without going through the
buttons, so `#edit-modal` listens for `close` to run the same teardown Cancel
does. `openEdit()` is the one selection change that must *not* be followed by
`renderSelection()` — it selects the node first, then calls `exitPreview(false)`
unconditionally, and only then starts its own preview, because that choke point
would otherwise cancel it immediately (and, with live preview, leave a debounce
timer armed behind it). After a
successful PATCH the node id is unchanged, so `saveEdit()` clears
`cluster.nodeId` by hand; `updateClusterPlot()` refetches only on an id change
and would otherwise keep drawing the old centroids.

`GET /api/stats` (`db.stats()` + `rendering.storage_stats()`) backs the Library
stats modal. It reports the render cache apart from the database, originals, and
saved masks because the cache is ~85% of the bytes and is fully regenerable —
one lumped "on disk" number would badly misrepresent what a user would lose.
