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
  the key — keep that default when touching the call site.
- **Previews never persist.** `POST /api/nodes/{id}/preview` renders an effect
  against a node's cached pixels entirely in memory (`rendering.render_preview`)
  — no DB row, no file in `data/renders/`. It shares the effect-application
  core (`rendering._apply`) with `render_node` so preview and Apply can never
  disagree. Frontend preview mode lives in the `preview` state object in
  `web/app.js`; every path that changes the selection exits preview through the
  `exitPreview()` call at the top of `renderSelection()` — keep that single
  choke point. `preview.source` names whoever is currently driving it (the Apply
  panel or the edit modal) and is the *only* thing that varies: one debounce, one
  `seq` stale-response counter, one owner of the blob URL. Note the edit modal
  previews against the edited node's **parent** — re-applying its effect to its
  own input — because the endpoint composes on *top* of the node you name.
- **Param controls are built and read in one place.** `buildParamControls()` /
  `appendBlendTarget()` / `readParams()` in `web/app.js` are shared by the Apply
  panel and the edit modal, which is what keeps ranges, defaults, and value
  coercion from drifting between "create a node" and "edit a node". The blend
  target picker is found by the `.blend-with` class *scoped to its container*,
  not a global id, so two of them can coexist while the modal is open.
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
  since the node was made (e.g. blend's `weight`).
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
- **Posterize clusters in PCA-whitened RGB space** (`_fit_kmeans`), not raw
  RGB — raw k-means bunches centroids along the luminance diagonal. Whitening
  is manual (`sqrt(explained_variance_ + 1e-4)`) rather than sklearn's
  `whiten=True` so grayscale/solid images (zero chroma variance) don't divide
  by zero. Centroids are inverse-transformed back to RGB; all sampling is
  seeded so the effect and its cluster-plot data agree.
- **Schema changes migrate in place** in `db.init()` (PRAGMA table_info check +
  ALTER TABLE) — user databases contain real work, never require a wipe.
- **Node ids are topological.** `GET /api/images/{id}/tree` returns nodes
  ordered by id, and both parents of a node always have smaller ids (SQLite
  autoincrement + parents must exist at creation). The work-tree DAG renderer
  (`layoutGraph()` in `web/app.js`) depends on this to assign commit-graph
  lanes in a single forward pass — one row per node, per-row SVG gutter,
  curves forking at branches and merging into blend rows.
- **Frontend files are served `Cache-Control: no-cache`** (`NoCacheStaticFiles`
  in `main.py`). Before this, browsers heuristically cached `app.js` and stale
  scripts produced phantom UI bugs (dead buttons, broken sliders) against a
  new backend. Keep the header; there is deliberately no cache-busting in
  filenames.

Frontend state lives in one `state` object in `web/app.js`; selection flows
through `selectImage()` → `renderSelection()`, which updates preview, export
link, tree, effect controls (blend's target picker depends on the tree), and
the cluster plot (a hand-rolled canvas 3D scatter, shown only for posterize
nodes, placed last in the side panel so its appearance doesn't shift the tree).
Deep links `?image=N&node=M` override the localStorage last-image.

`state.presets` is fetched once in `init()` and refreshed only after a save or
delete — `renderSelection()` calls the synchronous `updatePresetControls()`, not
a fetch. Clicking a preset row applies it to the *selected* node and then routes
through `selectImage()`, which is what refetches the tree so all of the preset's
new nodes appear.

The two modals are native `<dialog>` elements (Esc and focus trapping for free);
`prompt()`/`confirm()`/`alert()` remain the idiom everywhere else. Two gotchas:
the global `* { margin: 0 }` reset defeats a dialog's centering, so `dialog`
re-sets `margin: auto`; and Esc closes a dialog without going through the
buttons, so `#edit-modal` listens for `close` to run the same teardown Cancel
does. `openEdit()` is the one selection change that must *not* be followed by
`renderSelection()` — it selects the node first and only then starts its own
preview, because that choke point would otherwise cancel it immediately. After a
successful PATCH the node id is unchanged, so `saveEdit()` clears
`cluster.nodeId` by hand; `updateClusterPlot()` refetches only on an id change
and would otherwise keep drawing the old centroids.

`GET /api/stats` (`db.stats()` + `rendering.storage_stats()`) backs the Library
stats modal. It reports the render cache apart from the database and originals
because the cache is ~85% of the bytes and is fully regenerable — one lumped
"on disk" number would badly misrepresent what a user would lose.
