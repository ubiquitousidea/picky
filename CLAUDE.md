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

- **Renders are immutable snapshots.** `rendering.render_node()` materializes a
  node by applying its effect to the parent's *cached* JPEG, recursively, and
  caches the result; it only recomputes if the cache file is missing. Changing
  an effect's implementation therefore does NOT retroactively change existing
  nodes (this is intentional — children were derived from the old pixels).
  Same for the per-node k-means cluster JSON cache (`renders/<id>.clusters.json`).
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
  choke point.
- **Deletes cascade through both parent links** (`db.delete_node`'s recursive
  CTE), and no deletion order is FK-safe for two-parent graphs, so deletes run
  with `PRAGMA defer_foreign_keys = ON`. File cleanup (render/thumb/clusters)
  happens in `rendering.delete_render_files` from the returned id list.
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
