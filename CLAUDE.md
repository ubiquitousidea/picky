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
(gitignored: `picky.db`, `originals/<image_id>.jpg`, `renders/<node_id>.jpg`,
`masks/<mask_id>.png`).

The core model is a **branching work tree per image** (git-like, not an undo
stack): each `nodes` row has `parent_id` (NULL = root, which *is* the original
JPG), an `effect` name, and JSON `params`. Blend nodes also carry `parent2_id`,
so it is really a DAG with at most two parents.

The code is commented densely, and every rule below is explained at the call
site it constrains. This file lists only what **spans files** — where a change in
one place silently breaks another.

### Rendering and caching

- **A render always matches its node's current params.** File existence is the
  entire cache key (`rendering.render_node`), so invalidation means *deleting
  files*: `PATCH /api/nodes/{id}` sweeps the node's descendant closure
  (`db.descendant_ids` + `rendering.delete_render_files` — renders, thumbs,
  `.out.jpg`, cluster JSON) and re-renders the edited node eagerly, so a bad
  param fails on the PATCH rather than a later GET. Not covered: changing an
  effect's *implementation* does not retroactively change existing nodes, and
  neither does replaying a preset.
- **PATCH updates the row, then invalidates — never the reverse.** These are sync
  `def` endpoints running concurrently in a threadpool; sweeping first would let
  a racing read re-materialize stale pixels that then look like a valid cache hit
  forever.
- **Previews never persist.** `POST /api/nodes/{id}/preview` renders in memory —
  no row, no file — and shares `rendering._apply` with `render_node`, so preview
  and Apply cannot disagree.
- **Deletes cascade through both parent links** (`db.delete_node`'s recursive
  CTE, shared with `descendant_ids` as `DESCENDANT_CTE`). No deletion order is
  FK-safe for a two-parent graph, so it runs under `PRAGMA defer_foreign_keys`;
  `db.delete_nodes` (highest id first) is the exception that needs no such
  window. Always pair a delete with `rendering.delete_render_files`.

### Effects

- **Effects are registry entries.** `EFFECTS` in `server/effects.py` maps name →
  `apply(rgb_array, params)` plus param specs; the frontend builds its controls
  from `GET /api/effects`, so a new effect is one entry and nothing else. Blend
  is the exception (two inputs): `BLEND_SPEC` / `apply_blend`, special-cased in
  `main.py` and `rendering.py`.
- **A param added to a shipped effect is absent from every row already on
  disk** — nothing migrates node params — so read it with a default, as blend's
  `weight`, pixelate's `shape` and blur's `kernel` do. Widening a *range* needs
  no such care: `validate_params` clamps.
- **Every registry effect maps an array to an array of the same size**, so every
  node of an image shares the original's dimensions. Much of the app leans on
  this — see crop, below.
- **The tone curve's LUT is implemented twice on purpose** —
  `effects._curve_lut` and `curveLut()` in `web/app.js`, line for line — so the
  editor draws exactly the transfer function the server will apply. Change one,
  change the other.

### Selections and masks

- **A selection is a union, in one of two shapes**: `{points: [...], invert}`
  re-segmented by SAM, or `{masks: [id, ...], invert}` loaded from
  `data/masks/<id>.png`. Members are OR'd and `invert` applies once, to the
  result. Older spellings still on disk **upgrade on read** in `db.node_dict`
  (via `effects.validate_selection`), so exactly one shape ever leaves the
  database; nothing migrates them in place.
- **A click selection is banked into a saved mask by the frontend**
  (`bankSelection()`, called from `applyEffect`/`applyPreset`), not by
  `create_node`. From the server the store would still hold the points, so every
  Apply would duplicate and preset replay would mint a mask per masked step.
- **Masks live outside `renders/`.** That directory is a regenerable cache swept
  by node id; a saved mask is user data whose whole point is that it is *not*
  regenerable. Its PNG is frozen **post-invert**, so a reference's own `invert`
  toggles on top.
- **A mask any node references cannot be deleted (409).** The guard
  (`db.nodes_using_mask`) and the `used_by` count that predicts it share one
  `MASK_REFS` constant, and the count is *joined into* `db.list_masks` rather
  than fetched separately — so the check and the warning can neither drift nor
  race.
- **Masks are image-scoped**: `main._check_selection` 400s if any member is
  foreign, since members are OR'd into one image's dimensions.

### Presets

- **Presets store a sub-DAG with relative parent indices, never node ids**
  (`main.capture_steps`): `0` = the node the preset is applied to, `1..i-1` = an
  earlier step. Absolute ids are wrong across images and unsafe even within one,
  since SQLite reuses rowids — hence no FK from `presets` to `nodes`.
- **A preset stores params, not pixels**, so replay runs *today's* effect code
  and need not reproduce the original pixels. A saved-mask selection degrades
  back to the click points it was frozen from (`_portable_selection`), and is
  dropped when an invert makes that lossy.
- **Applying a preset is all-or-nothing and cannot be one transaction** —
  `render_node` reads through its own connection, so each step must commit before
  the next can reference it. The loop is create → commit → render, pre-flighted
  by `_validate_steps` and unwound by a compensating `db.delete_nodes` +
  `rendering.delete_render_files`. The file sweep is not optional: an orphaned
  render would later be served as a different node's pixels.
- **An apply-time selection replaces the recipe's own** for every step, and is
  validated before the loop starts, so a foreign mask is a 400 rather than a
  half-written chain.

### Crop, embeddings, schema

- **Crop & rotate is an output stage, not a node.** One framing per image
  (`images.crop`), applied after the whole work tree in
  `rendering.render_output()` — the one place it happens, which every viewer goes
  through and which short-circuits when the crop is the identity. Keeping it
  outside the tree is what keeps the same-dimensions invariant true, and means
  **no saved mask is ever warped**. `.out.jpg` must be excluded by name in
  `storage_stats` and swept in `delete_render_files`.
- **`effects.crop_geometry()` is the only place PIL's rotate-expand rounding is
  known** — it reproduces PIL's arithmetic rather than deriving a formula for it.
  Its `inverse` is what lets the frontend map a click on the framed preview back
  into node space with no trigonometry.
- **The mask *outline* is framed for display; the mask composited in `_apply`
  never is.** Do not confuse the two.
- **`images.embedding` is a column, not a file.** Image ids are rowids SQLite
  reuses, so an `<image_id>.npy` would outlive its image and be read back as its
  successor's position in the cloud. It embeds the *thumbnail*, so re-framing
  invalidates it (`db.clear_embedding` in the crop PUT) and nothing else does.
  The projection is library-wide, so it is deliberately never cached.
- **Cluster labels need the *joint* 512-d space, which `embed.py` does not
  promise.** `server/label_vectors.npz` holds CLIP text vectors, so it is only
  comparable to an `image_embeds` (projection) export — swap in a bare vision
  export's 768-d `pooler_output` and image-to-image similarity still works while
  labelling becomes meaningless. `labels.label_clusters` therefore compares
  widths and returns `label: null` rather than nonsense. The npz is generated by
  `tools/build_label_vectors.py` (which owns the vocabulary, the text tower and
  the only tokenizer in the tree) and **never** by the server: nothing under
  `server/` encodes text, and re-running that script is the only way a new word
  reaches the map.
- **`server/embed_job.py` is only ever an optimization.**
  `GET /api/embedding-map` still embeds whatever it finds missing, so the map is
  correct if nobody prepared, if the job died, or if both run at once. Keep it
  that way — a prepare the map *required* would be a second source of truth.
- **Schema changes migrate in place** in `db.init()` (PRAGMA table_info check +
  ALTER TABLE). User databases contain real work; never require a wipe.

### Frontend

- **Node ids are topological** — both parents of a node always have smaller ids —
  which lets `layoutGraph()` assign commit-graph lanes in one forward pass, and
  makes `parent2_id < node_id` a sufficient cycle guard when *editing* a blend
  (creation is safe for free, since a new node has the largest id).
- **State lives in one `state` object, not in the DOM**: the selection
  (`state.selection`), the current effect (`state.effect`), the crop. The effect
  panel is torn down and rebuilt on every selection change, which used to take
  the pick with it.
- **`selectImage()` → `renderSelection()` is the single choke point.** Every path
  that changes the selection leaves preview and crop mode there, and re-enters
  live preview at the end. `openEdit()` is the one deliberate exception, and must
  call `exitPreview(false)` itself.
- **Selections expire by shape** in `pruneSelection()`: click points die when
  `nodeId` changes (their coords are in that node's pixel space), saved masks
  only when `imageId` does — which is what lets you stack masked effects on one
  object.
- **Param controls are built and read in one place** (`buildParamControls()` /
  `readParams()`), shared by the Apply panel and the edit modal, so ranges,
  defaults and coercion cannot drift between create and edit. The Method dropdown
  is presentation only: nodes still store `posterize` vs `dither`, `curves` vs
  `gamma`.
- **A URL whose bytes change without its id changing carries a version tag** —
  the mask grid's `?v=<created_at>`, the gallery's `cropTag()` (which the Image
  map's sprites key on too) — because both are served with long cache lifetimes
  over ids SQLite reuses.
- **Frontend files are served `Cache-Control: no-cache`** (`NoCacheStaticFiles`),
  and there is deliberately no cache-busting in filenames. Keep the header:
  stale `app.js` against a new backend produces phantom UI bugs.
- **Modals are native `<dialog>`.** Esc closes one without going through its
  buttons, so each listens for `close` to run the same teardown (guarded by a
  sentinel) — for the Image map that is what stops a `requestAnimationFrame` loop
  spinning behind a hidden dialog.
- The side panel is ordered by the workflow: **1 · Select**, **2 · Effect**,
  **3 · Frame** (a `<details>` whose open state *is* crop mode), then Presets and
  the RGB cluster plot as collapsed reference, with Export/Delete pinned to the
  bottom. The work tree is a horizontal strip under the preview.
- Deep links `?image=N&node=M` override the localStorage last-image.
