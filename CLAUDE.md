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
  `apply(np.uint8 RGB array, params) -> array` plus param specs (int ranges or
  `choice`). The frontend generates its controls from `GET /api/effects`, so a
  new effect is one registry entry and nothing else. Blend is the exception:
  it takes two images, so it lives outside `EFFECTS` (`BLEND_SPEC`,
  `apply_blend`) and is special-cased in `main.py` and `rendering.py`.
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

Frontend state lives in one `state` object in `web/app.js`; selection flows
through `selectImage()` → `renderSelection()`, which updates preview, export
link, tree, effect controls (blend's target picker depends on the tree), and
the cluster plot (a hand-rolled canvas 3D scatter, shown only for posterize
nodes). Deep links `?image=N&node=M` override the localStorage last-image.
