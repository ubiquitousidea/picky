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
- **Bokeh is the one effect with a model behind it**, and it stays a plain
  registry entry: `apply(img, params)` carries no node id, so its depth map is
  cached in-process on a hash of the pixels (`depth.depth_map`) rather than on
  disk beside `<node_id>.embedding.npy`. That is the trade — a cache keyed by
  node would cost the registry's "a new effect is one entry and nothing else".
  Two things live outside the effect because of it: `main._check_effect_ready`
  409s every render path by effect *name* until the weights are on disk, and
  `setEffect()` is what triggers the download.
- **Bokeh blurs at a bounded working resolution and recombines the original at
  full size.** Both caps (`_BOKEH_WORK_PX`, `_BOKEH_WORK_R`) are what keep a
  40 MP render to seconds and megabytes, and they are only safe because the
  pixels taken from that canvas are, by construction, the blurred ones — the
  crossover is placed where the upsample's lost detail is already smaller than
  the blur. Its layers are composited far-to-near with premultiplied alpha, not
  lerped between two globally blurred frames; the cheap version smears the
  subject outward as a halo, which is the same artifact as the hard-edged
  masked blur bokeh exists to replace.
- **`_disk_blur` and `_disk_mean` are deliberately not one function** — uint8
  and exactly-int32 for the shipped Blur, float32 and N channels for bokeh's
  premultiplied layers. Generalizing the first would change what every existing
  disk-blur node re-renders to. Only the geometry is shared, and only as the
  run generators: `_disk_runs` for both of those, `_lens_runs` for `_swirl_mean`
  — which is the third convolution here and, at aspect 1, returns `_disk_runs`
  run for run, so the swirl slider has no step at the bottom.
- **Swirl's kernel varies per pixel, and the prefix sum is what makes that
  affordable.** The run trick looks incompatible with a kernel that changes
  across the frame — the bounds stop being sliceable constants — but the
  cumulative sum does not depend on the kernel at all. `_swirl_mean` builds one
  for the whole frame and each `_SWIRL_TILE` square reads its own offsets out of
  it, so a tile costs no halo and no extra data. `_SWIRL_TILE` is the dial
  between seams and speed; the kernel steps at tile edges, and the step is
  self-limiting because anisotropy grows with distance from the centre exactly
  as fast as the angle between neighbouring tiles shrinks.
- **The cat's eye and the corner darkening are two readings of one aperture**,
  so `_swirl_aspect` is shared by `_swirl_mean` and `_vignette` rather than
  written twice — apart, each still looks right and together they describe the
  wrong lens. The dimming cannot live in the kernel's normalizer, which is the
  obvious place for it: coverage rides through the same kernel, and
  `_bokeh_layers` composites with `1 - alpha` and then un-premultiplies by
  accumulated coverage, so a scaled normalizer would corrupt the occlusion and
  then divide itself back out. `_vignette` therefore runs last, in `bokeh()`, on
  sharp and blurred pixels alike — a lens vignettes both, and dimming only what
  was blurred would put a step along the recombine crossover.
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
- **Bokeh's optical axis is the centre of the node's own pixels**, which the
  crop then moves. Both `_swirl_mean` and `_vignette` measure from there, so
  re-framing slides the swirl's centre and the vignette's centre off-centre in
  what you actually see — the one thing about a node's appearance that changes
  without its params changing. That is what cropping does to a real frame, so it
  is left alone; the thing not to do is "fix" it by reaching for `images.crop`
  from inside an effect, which would put the output stage back inside the tree.
- **`effects.crop_geometry()` is the only place PIL's rotate-expand rounding is
  known** — it reproduces PIL's arithmetic rather than deriving a formula for it.
  Its `inverse` is what lets the frontend map a click on the framed preview back
  into node space with no trigonometry.
- **The mask *outline* is framed for display; the mask composited in `_apply`
  never is.** Do not confuse the two.
- **A filename is an image's identity at import**, so `db.create_image` returns
  `(image, created)` and re-importing a name already in the library yields the
  *existing* row with `created False` (match is `COLLATE NOCASE`; there is no
  UNIQUE index, since databases predate the check). What `main.upload_image`
  must never do on that branch is write the bytes: `rendering.original_path` is
  keyed by image id, so it would replace that image's original while its whole
  render closure stayed cached — and file existence is the entire cache key. The
  skip is a 200 carrying `duplicate: true`, not a 409, which is what keeps
  `uploadFiles()` reporting it in the label instead of the failure alert.
- **`images.embedding` is a column, not a file.** Image ids are rowids SQLite
  reuses, so an `<image_id>.npy` would outlive its image and be read back as its
  successor's position in the cloud. It embeds the *thumbnail*, so re-framing
  invalidates it (`db.clear_embedding` in the crop PUT) and nothing else does.
  The projection is library-wide, so it is deliberately never cached.
- **Cluster labels and search both need the *joint* 512-d space, which
  `embed.py` does not promise.** `server/label_vectors.npz` and
  `text_embed.encode_query` alike are only comparable to an `image_embeds`
  (projection) export — swap in a bare vision export's 768-d `pooler_output` and
  image-to-image similarity still works while both go meaningless.
  `labels.label_clusters` and `main.search_embedding_map` therefore compare
  widths and return nothing rather than nonsense.
- **`server/text_embed.py` owns CLIP's text tower and the only tokenizer in the
  tree; `tools/build_label_vectors.py` owns the vocabulary and the npz.** The
  script imports the tower rather than holding its own, which is what makes a
  typed query that happens to be a vocabulary word land on the exact vector the
  npz stores for it. The npz is still built *only* by that script — nothing
  under `server/` writes it, and re-running it is still the only way a new word
  reaches the cluster labels.
- **The text tower is downloaded only when someone searches.** It is 254 MB and
  most sessions never type in the box, so `text_job.py` is triggered by the
  search box alone — never by opening the map — and `/api/embedding-map/search`
  409s instead of fetching a quarter-gigabyte inside a GET.
- **Three ONNX models, three different normalizations.** SAM's constants are
  applied to 0-255 pixels, CLIP's to 0-1 floats, and `depth.py`'s are plain
  ImageNet over 0-1 floats. Crossing them yields output that still looks like
  output — a plausible depth map, a well-spread vector — and nothing downstream
  can detect it. `sam.py` owns `MODELS_DIR` and `download_model`; `embed.py`
  and `depth.py` both borrow them rather than re-deriving the pinned-revision
  and atomic-write contract.
- **`server/embed_job.py` is only ever an optimization; `server/text_job.py`
  and `server/depth_job.py` are not.** `GET /api/embedding-map` still embeds
  whatever it finds missing, so the map is correct if nobody prepared, if the
  job died, or if both run at once — keep it that way, a prepare the map
  *required* would be a second source of truth. Search and bokeh have no such
  fallback by choice (see above), which is exactly why the three are separate
  modules rather than phases of one: each is triggered by the one gesture that
  needs it — opening the map, typing a query, lighting the Bokeh button — so
  nobody pays for a model they never use.
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
- **The feature buttons are toggles, and `state.effect === null` is the off
  state** — it is also the whole of "live preview is off", which is why there is
  no checkbox for that any more. `setEffect()` remains the single write path and
  is the one place the preview starts (`enterApplyPreview()`) or stops
  (`exitPreview(true)`, which repaints the node's own render). Two things must
  stay in step with it: the lit `.selected` class on the button, and
  `renderEffectControls()`'s early-out — an empty `#effect-params` and a *hidden*
  (not merely disabled) Apply are how "off" looks, so a null effect must never
  reach `buildParamControls()` or Apply. `#apply-btn` therefore ships `hidden` in
  the markup, since that is the state `init()` leaves the panel in.
- **`selectImage()` → `renderSelection()` is the single choke point.** Every path
  that changes the selection leaves preview and crop mode there, and re-enters
  live preview at the end. `openEdit()` is the one deliberate exception, and must
  call `exitPreview(false)` itself.
- **Selections expire by shape** in `pruneSelection()`: click points die when
  `nodeId` changes (their coords are in that node's pixel space), saved masks
  only when `imageId` does — which is what lets you stack masked effects on one
  object.
- **One armed picker, and `selPicker.armed.owner` is the button holding it.**
  Two controls arm it — selecting an object and bokeh's Focus pick — so "press
  the lit button to cancel" has to mean the *lit* one, or taking the picker from
  the other control would cost two presses. Anything that arms it wears
  `.sel-pick`, which is what `disarmPick()` sweeps, so exactly one can be lit.
  A pick armed from inside `#effect-params` must re-query its input when the
  click finally lands (`buildDepthPick`): the panel is rebuilt on every
  selection change, which is precisely the window it was waiting through.
- **A param spec's `pick` is presentation, like the Method dropdown.** It says a
  number can be read off the image instead of dialled in; the param stays a
  float, `validate_params` never sees the key, and the slider stays the only
  thing Apply reads. It is off in the edit modal (`allowPick: false`) for the
  reason the selection's pick is: `showModal()` makes the image inert.
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
  sentinel).
- **Frame is the one dialog opened with `show()`, not `showModal()`** — the crop
  rectangle is dragged on the preview underneath, which a modal would make inert
  (the same inertness `openEdit()` documents as its reason for not re-picking).
  Being non-modal costs it the platform's Esc handling, so `initCropOverlay()`
  wires a `keydown` for it, guarded by `dialog:modal` so a real modal keeps the
  key. Opening the dialog and entering crop mode are one thing: `exitCropMode()`
  closes it, `close` re-enters `exitCropMode()`, and the `crop.active` guard is
  what makes that loop terminate.
- **A `requestAnimationFrame` loop ends itself**; it does not trust callers to
  stop it. The RGB cluster plot's `tick` returns when `#cluster-modal` is closed,
  so "it only spins while you are looking at it" holds however the dialog was
  dismissed — a `close` listener alone would be one forgotten path from a leak.
- **The Image map's search filter is display-only, and `embedMap.scores === null`
  is the off state.** Nothing is re-fitted on the survivors: matches keep the
  coordinates they already had, so the cloud you learned stays put. `p.hidden` is
  read in exactly two places — the draw loop and `embedPointAt` — and the second
  is not optional, since `hw`/`hh` are only written *inside* the draw loop, so a
  point that stops being drawn keeps last frame's extents and stays clickable.
  Scores are keyed by `image_id`, which is what lets a filter survive a
  re-projection; the threshold is applied locally, so dragging Match costs a
  redraw where typing a query costs a request.
- **The page is one column**: a header of dialog-openers, the image at full width,
  and a `#control-bar` of horizontal rows along the bottom — effect + params +
  Apply, then the selection's controls with its saved masks as a strip, then the
  work tree, then the filmstrip. It replaced a three-column desktop grid so the
  app fits a portrait screen; there are no media queries, and `body > *` needs
  `min-width: 0` or a grid item's automatic minimum size widens the single column
  past the viewport and `scrollIntoView()` starts scrolling the whole page.
- **The bar's rows scroll; the two rows that must not hide anything wrap.**
  Effect and selection controls wrap (Apply is the app's primary action), while
  the mask strip, the tree and the filmstrip scroll sideways — a scroller is only
  right where the content is a list you pan through.
- **Bar layout is scoped to `#effect-params` / `#select-controls`, never written
  into `.param-row` or `.sel-mask-list` themselves.** `buildParamControls()` and
  `appendSelectionControls()` build into the edit modal too, where the stacked
  column and the mask *grid* are the right shapes. The scoping is what lets one
  component render as a row in the bar and a column in the dialog.
- **The filmstrip is opt-in** (`picky:filmstrip` in localStorage, off by
  default): the Image map is the intended picker. It is the bar's last row, so
  toggling it never reorders anything above it.
- Deep links `?image=N&node=M` override the localStorage last-image.
