"""SQLite persistence for images and their effect work trees."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# effects imports no DB, so this direction is the acyclic one. It is needed for
# one thing only: normalizing selections on the way out of node_dict.
from .effects import validate_selection

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ORIGINALS_DIR = DATA_DIR / "originals"
RENDERS_DIR = DATA_DIR / "renders"
# Saved masks live outside RENDERS_DIR on purpose: everything under renders/ is
# a regenerable cache that delete_render_files sweeps by node id, and a saved
# mask is user data keyed by mask id that must never be swept.
MASKS_DIR = DATA_DIR / "masks"
DB_PATH = DATA_DIR / "picky.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL,
  crop TEXT,             -- JSON {angle, rect}; NULL = no framing at all
  embedding BLOB,        -- raw float32 CLIP vector; NULL = not embedded yet
  detected_at TEXT       -- when YOLO last ran; NULL = never, which is not
                         -- the same as "ran and found nothing"
);
CREATE TABLE IF NOT EXISTS nodes (
  id INTEGER PRIMARY KEY,
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  parent_id INTEGER REFERENCES nodes(id),
  parent2_id INTEGER REFERENCES nodes(id),
  effect TEXT,
  params TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS presets (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  steps TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS masks (
  id INTEGER PRIMARY KEY,
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  node_id INTEGER,        -- provenance only: no FK, so the mask outlives the node
  spec TEXT,              -- the click spec it was frozen from, for preset replay
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS masks_image_name ON masks(image_id, name);
-- One row per object YOLO found, in `images.detected_at`'s pass. Boxes are
-- fractions of the *framed* image, like the CLIP vector is of the framed
-- thumbnail, so re-cropping stales both together (see `clear_detections`).
-- No score index: every query here is by image or by label, and the score is
-- only ever read off a row already found.
CREATE TABLE IF NOT EXISTS detections (
  id INTEGER PRIMARY KEY,
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  label TEXT NOT NULL,
  score REAL NOT NULL,
  x0 REAL NOT NULL, y0 REAL NOT NULL, x1 REAL NOT NULL, y1 REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS detections_image ON detections(image_id);
CREATE INDEX IF NOT EXISTS detections_label ON detections(label);
CREATE TABLE IF NOT EXISTS projections (
  method TEXT PRIMARY KEY,   -- one row per method: a new fit replaces the old
  fingerprint TEXT NOT NULL, -- hash of the vectors the fit was over
  coords BLOB NOT NULL,      -- float32 (n, 3), C-order, in list_images() order
  created_at TEXT NOT NULL
);
-- Every per-image question about the work tree is a scan without this: both of
-- `list_images`' correlated subqueries, `image_node_ids`, `get_tree`. At 1426
-- images over 1891 nodes it takes that list from 90 ms to 2 ms.
CREATE INDEX IF NOT EXISTS nodes_image ON nodes(image_id);
"""


def init() -> None:
    for d in (DATA_DIR, ORIGINALS_DIR, RENDERS_DIR, MASKS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(nodes)")]
        if "parent2_id" not in cols:
            conn.execute(
                "ALTER TABLE nodes ADD COLUMN parent2_id INTEGER REFERENCES nodes(id)"
            )
        if "selection" not in cols:
            conn.execute("ALTER TABLE nodes ADD COLUMN selection TEXT")
        image_cols = [r["name"] for r in conn.execute("PRAGMA table_info(images)")]
        if "crop" not in image_cols:
            conn.execute("ALTER TABLE images ADD COLUMN crop TEXT")
        if "embedding" not in image_cols:
            conn.execute("ALTER TABLE images ADD COLUMN embedding BLOB")
        if "detected_at" not in image_cols:
            conn.execute("ALTER TABLE images ADD COLUMN detected_at TEXT")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def node_dict(row: sqlite3.Row) -> dict:
    """A node row as a dict, with its selection normalized to the union shapes.

    Normalizing here rather than at each reader means exactly one selection
    shape ever leaves the database, so neither the API's clients nor
    `update_node`'s no-op check have to know that rows written before unions
    existed spell it `{"mask": n}` or `{"x": .., "y": ..}`. Nothing is rewritten
    on disk — this is a read-time upgrade, and it covers the render path too,
    since `rendering.render_node` reads its node through `get_node`.
    """
    selection = json.loads(row["selection"]) if row["selection"] else None
    return {
        "id": row["id"],
        "image_id": row["image_id"],
        "parent_id": row["parent_id"],
        "parent2_id": row["parent2_id"],
        "effect": row["effect"],
        "params": json.loads(row["params"]) if row["params"] else None,
        "selection": validate_selection(selection),
        "created_at": row["created_at"],
    }


def image_dict(row: sqlite3.Row) -> dict:
    """An image row as a dict, with its crop parsed. NULL stays None — "never
    framed" is worth telling apart from "framed back to the identity", since only
    the former is guaranteed to have no `.out.jpg` files behind it.

    Deliberately *not* normalized through `validate_params` the way `node_dict`
    normalizes selections: a crop has only ever had one shape, so there is
    nothing to upgrade, and `crop_geometry` is total anyway.
    """
    d = dict(row)
    d["crop"] = json.loads(d["crop"]) if d.get("crop") else None
    # Only when the query actually asked for it. An absent key and an empty list
    # are different answers — "nobody looked" against "this image is untouched" —
    # and defaulting to the second would let `get_image`, which does not select
    # the column, claim every image has no edits.
    if "effects" in d:
        d["effects"] = d["effects"].split(",") if d["effects"] else []
    # Same rule, same reason: absent means the query did not ask. Note that an
    # empty list here is genuinely ambiguous in a way `effects` is not — it
    # covers both "detected nothing" and "never detected" — which is what
    # `detected_at` is for, and why it is selected alongside.
    if "labels" in d:
        d["labels"] = sorted(d["labels"].split(",")) if d["labels"] else []
    return d


def create_image(name: str) -> tuple[dict, bool]:
    """Import an image under `name`, or hand back the one already using it.

    Returns `(image, created)`. A filename already in the library is the whole
    duplicate test — there is no pixel hashing — so `created is False` means the
    caller must leave the returned image's original bytes alone (see
    `main.upload_image`).

    The lookup is inside the transaction rather than a separate helper the caller
    calls first, so two uploads of one name cannot both miss it. There is
    deliberately no UNIQUE index backing it: user databases predate this check
    and may already hold duplicate names, which such an index could not be built
    over. `ORDER BY i.id` picks the oldest of them, deterministically.

    Matching is `COLLATE NOCASE` because the filesystems these files are dragged
    from are, so `Sunset.JPG` re-importing beside `sunset.jpg` would read as a
    bug rather than a distinction.
    """
    with connect() as conn:
        # the root-node subselect is `list_images`', not `get_image`': callers
        # navigate to what comes back, and `get_image` omits root_node_id.
        row = conn.execute(
            """SELECT i.id, i.name, i.created_at, i.crop,
                      (SELECT id FROM nodes n WHERE n.image_id = i.id
                       AND n.parent_id IS NULL) AS root_node_id
               FROM images i WHERE i.name = ? COLLATE NOCASE
               ORDER BY i.id LIMIT 1""",
            (name,),
        ).fetchone()
        if row:
            return image_dict(row), False
        now = _now()
        cur = conn.execute(
            "INSERT INTO images (name, created_at) VALUES (?, ?)", (name, now)
        )
        image_id = cur.lastrowid
        cur = conn.execute(
            "INSERT INTO nodes (image_id, parent_id, effect, params, created_at)"
            " VALUES (?, NULL, NULL, NULL, ?)",
            (image_id, now),
        )
        root_id = cur.lastrowid
    return {
        "id": image_id,
        "name": name,
        "created_at": now,
        "crop": None,
        "root_node_id": root_id,
    }, True


def list_images() -> list[dict]:
    """Every image, with its root node and the kinds of edit it carries.

    `effects` is the *set* of effect names anywhere in the image's work tree, in
    no particular order — what the Image map's edit filter asks its question of,
    and cheap enough here (one indexed subquery) that it need not be a second
    endpoint the map would have to keep in step with this one. Every image's root
    node has `effect IS NULL` (see `create_image`), so the list is empty exactly
    when nothing has been done to the image.

    `labels` is the same arrangement one table over: the set of COCO classes
    `detect.py` found, feeding the map's tag filter. It rides here rather than in
    its own endpoint for the reason `effects` does — the map already fetches this
    list, and a second source for "what is in the library" is a second thing to
    keep in step. `detected_at` comes with it because the two answer different
    questions: an empty `labels` with a timestamp means the detector ran and
    found none of its eighty nouns, and without one means it has never run.
    """
    with connect() as conn:
        rows = conn.execute(
            """SELECT i.id, i.name, i.created_at, i.crop, i.detected_at,
                      (SELECT id FROM nodes n WHERE n.image_id = i.id
                       AND n.parent_id IS NULL) AS root_node_id,
                      (SELECT group_concat(DISTINCT n.effect) FROM nodes n
                       WHERE n.image_id = i.id AND n.effect IS NOT NULL) AS effects,
                      (SELECT group_concat(DISTINCT d.label) FROM detections d
                       WHERE d.image_id = i.id) AS labels
               FROM images i ORDER BY i.created_at DESC"""
        ).fetchall()
    return [image_dict(r) for r in rows]


def get_image(image_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT id, name, created_at, crop, detected_at FROM images WHERE id = ?",
            (image_id,),
        ).fetchone()
    return image_dict(row) if row else None


def set_image_crop(image_id: int, crop: dict | None) -> dict:
    """Store an image's framing. `None` clears it back to the whole frame."""
    with connect() as conn:
        conn.execute(
            "UPDATE images SET crop = ? WHERE id = ?",
            (json.dumps(crop) if crop else None, image_id),
        )
    return get_image(image_id)


# ---------- CLIP embeddings (the Image map's coordinates) ----------
#
# Stored as a BLOB on the image row rather than as a file under `renders/`,
# which is where every other derived artifact lives. Two reasons, and the first
# is a correctness one:
#
# - Image ids are rowids, and SQLite hands them back out after a delete. A
#   file named `<image_id>.npy` would therefore outlive its image and be read
#   back as its successor's vector — the same hazard that keeps mask thumbnails
#   from being files. A BLOB in the row dies with the row, through the FK
#   cascade that already exists, with no sweep to remember.
# - `renders/` is swept *by node id* (`delete_render_files`); an image-keyed
#   file there has nothing that would ever clean it up.
#
# The cost is that ~2 KB per image lands in the "database" line of the storage
# stats, which is otherwise reserved for things you cannot regenerate. At 512
# float32 per image that is noise, and worth it for an invalidation edge that
# cannot be forgotten.


def get_embeddings() -> dict[int, bytes]:
    """Every stored vector, keyed by image id — one query, one snapshot."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, embedding FROM images WHERE embedding IS NOT NULL"
        ).fetchall()
    return {r["id"]: r["embedding"] for r in rows}


def set_embedding(image_id: int, blob: bytes) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE images SET embedding = ? WHERE id = ?", (blob, image_id)
        )


def clear_embedding(image_id: int) -> None:
    """Forget an image's vector, so the next map re-embeds it. The one caller is
    the crop endpoint: re-framing to a detail changes what the photo is *of*, so
    its point has to move."""
    with connect() as conn:
        conn.execute(
            "UPDATE images SET embedding = NULL WHERE id = ?", (image_id,)
        )


# ---------- Detected objects (see detect.py) ----------


def set_detections(image_id: int, found: list[dict]) -> None:
    """Replace an image's detections with `found` and stamp it as detected.

    One transaction, and the delete is not conditional on `found` being
    non-empty: a re-detection that finds nothing must leave nothing behind, or
    the previous pass's boxes would outlive the pixels they were measured on.

    The stamp is written even for an empty list — that is the whole point of the
    column. `detect_job` selects on it, so an image the model has no nouns for
    must not be retried on every pass.
    """
    with connect() as conn:
        conn.execute("DELETE FROM detections WHERE image_id = ?", (image_id,))
        conn.executemany(
            """INSERT INTO detections (image_id, label, score, x0, y0, x1, y1)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (image_id, d["label"], d["score"], *d["box"])
                for d in found
            ],
        )
        conn.execute(
            "UPDATE images SET detected_at = ? WHERE id = ?", (_now(), image_id)
        )


def clear_detections(image_id: int) -> None:
    """Forget an image's boxes *and* its stamp, so the next pass re-detects it.

    `clear_embedding`'s twin, called from the same place and for the same
    reason: boxes are fractions of the framed image, so re-framing moves every
    one of them and can bring objects into the frame or take them out. Dropping
    the stamp as well as the rows is what makes the re-detection happen — rows
    alone would read as "detected, found nothing".
    """
    with connect() as conn:
        conn.execute("DELETE FROM detections WHERE image_id = ?", (image_id,))
        conn.execute(
            "UPDATE images SET detected_at = NULL WHERE id = ?", (image_id,)
        )


def get_detections(image_id: int) -> list[dict]:
    """One image's boxes, strongest first."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT label, score, x0, y0, x1, y1 FROM detections
               WHERE image_id = ? ORDER BY score DESC""",
            (image_id,),
        ).fetchall()
    return [
        {
            "label": r["label"],
            "score": r["score"],
            "box": [r["x0"], r["y0"], r["x1"], r["y1"]],
        }
        for r in rows
    ]


def undetected_count() -> int:
    """How many images the detector has never run over.

    A count rather than `len(detected_image_ids())` because `detect_job`'s
    progress is polled: this is one aggregate over an integer column, where the
    set costs a row per image to build and throw away.
    """
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM images WHERE detected_at IS NULL"
        ).fetchone()
    return int(row["n"])


def detected_image_ids() -> set[int]:
    """Every image the detector has run over — one query, not one per image.

    `get_embeddings`' shape, and `detect_job._pending` uses it the same way.
    Reads the stamp rather than the rows, so an image with no objects in it
    counts as done.
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT id FROM images WHERE detected_at IS NOT NULL"
        ).fetchall()
    return {r["id"] for r in rows}


# ---------- The fitted projection (see main._projected) ----------
#
# Unlike an embedding, a point's coordinates are not a property of its own image
# — the fit is over the whole library — so this cannot be a column on the image
# row. It is one row per method holding the whole `(n, 3)` result, with a
# fingerprint of the vectors that produced it standing in for a cache key. That
# is the entire invalidation story: the caller compares fingerprints and refits
# on a mismatch, so nothing here has to know what stales a projection.


def get_projection(method: str) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute(
            "SELECT fingerprint, coords FROM projections WHERE method = ?", (method,)
        ).fetchone()


def set_projection(method: str, fingerprint: str, coords: bytes) -> None:
    """Replace this method's cached fit. REPLACE rather than INSERT because the
    table is a cache of fixed size: two methods, two rows, whatever happens to
    the library."""
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO projections (method, fingerprint, coords, created_at)"
            " VALUES (?, ?, ?, ?)",
            (method, fingerprint, coords, _now()),
        )


def image_node_ids(image_id: int) -> list[int]:
    """Every node of an image — the set whose output cache a crop change
    invalidates. Not `descendant_ids`: a crop applies after the *whole* tree, so
    it stales every branch at once, not a subtree."""
    with connect() as conn:
        return [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM nodes WHERE image_id = ?", (image_id,)
            )
        ]


def get_tree(image_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM nodes WHERE image_id = ? ORDER BY id", (image_id,)
        ).fetchall()
    return [node_dict(r) for r in rows]


def get_node(node_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
    return node_dict(row) if row else None


def create_node(
    image_id: int,
    parent_id: int,
    effect: str,
    params: dict,
    parent2_id: int | None = None,
    selection: dict | None = None,
) -> dict:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO nodes (image_id, parent_id, parent2_id, effect, params, selection, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                image_id,
                parent_id,
                parent2_id,
                effect,
                json.dumps(params),
                json.dumps(selection) if selection else None,
                _now(),
            ),
        )
        node_id = cur.lastrowid
    return get_node(node_id)


def update_node_params(
    node_id: int, params: dict, parent2_id: int | None, selection: dict | None = None
) -> dict:
    """Change an existing node's settings in place. The node keeps its id and its
    place in the tree, so everything derived from it is now stale — the caller
    must invalidate the cached renders of its whole descendant closure
    (see descendant_ids)."""
    with connect() as conn:
        conn.execute(
            "UPDATE nodes SET params = ?, parent2_id = ?, selection = ? WHERE id = ?",
            (
                json.dumps(params),
                parent2_id,
                json.dumps(selection) if selection else None,
                node_id,
            ),
        )
    return get_node(node_id)


DESCENDANT_CTE = """WITH RECURSIVE sub(id) AS (
                     SELECT id FROM nodes WHERE id = ?
                     UNION
                     SELECT n.id FROM nodes n
                     JOIN sub s ON n.parent_id = s.id OR n.parent2_id = s.id
                   ) SELECT id FROM sub"""


def descendant_ids(node_id: int) -> list[int]:
    """The node itself plus everything derived from it, through either parent
    link. Read-only twin of delete_node's traversal — UNION (not UNION ALL) so a
    diamond, where two branches meet at a blend, is visited once."""
    with connect() as conn:
        return [r["id"] for r in conn.execute(DESCENDANT_CTE, (node_id,)).fetchall()]


def delete_node(node_id: int) -> list[int]:
    """Delete a node and everything derived from it — children via either
    parent link, so blends that used a deleted node go too. Returns deleted ids."""
    with connect() as conn:
        ids = [r["id"] for r in conn.execute(DESCENDANT_CTE, (node_id,)).fetchall()]
        # a blend's second parent may appear after the blend in traversal order,
        # so no deletion order is always safe — defer FK checks to commit instead
        conn.execute("PRAGMA defer_foreign_keys = ON")
        conn.executemany("DELETE FROM nodes WHERE id = ?", [(i,) for i in ids])
    return ids


def delete_nodes(node_ids: list[int]) -> None:
    """Delete exactly these nodes and nothing else — used to unwind a partially
    applied preset. Unlike delete_node this does not cascade: the caller already
    knows the full set, and a preset's side branches hang off the base node, so a
    cascade from the first created node would miss them.

    Deleting highest id first is what keeps this FK-safe: a node's id always
    exceeds both its parents', so this removes every referrer before its referent
    and needs no deferred-FK window."""
    if not node_ids:
        return
    with connect() as conn:
        conn.executemany(
            "DELETE FROM nodes WHERE id = ?",
            [(i,) for i in sorted(node_ids, reverse=True)],
        )


def delete_image(image_id: int) -> list[int]:
    """Delete an image and its nodes; returns the deleted node ids."""
    with connect() as conn:
        node_ids = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM nodes WHERE image_id = ?", (image_id,)
            ).fetchall()
        ]
        conn.execute("PRAGMA defer_foreign_keys = ON")
        conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
    return node_ids


def preset_dict(row: sqlite3.Row) -> dict:
    # stored as {"version": n, "steps": [...]} so the format can evolve; callers
    # only ever see the unwrapped list
    doc = json.loads(row["steps"])
    return {
        "id": row["id"],
        "name": row["name"],
        "version": doc.get("version", 1),
        "steps": doc.get("steps", []),
        "created_at": row["created_at"],
    }


def list_presets() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM presets ORDER BY name").fetchall()
    return [preset_dict(r) for r in rows]


def get_preset(preset_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM presets WHERE id = ?", (preset_id,)
        ).fetchone()
    return preset_dict(row) if row else None


def create_preset(name: str, steps: list[dict]) -> dict:
    """Raises sqlite3.IntegrityError if the name is already taken."""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO presets (name, steps, created_at) VALUES (?, ?, ?)",
            (name, json.dumps({"version": 1, "steps": steps}), _now()),
        )
        preset_id = cur.lastrowid
    return get_preset(preset_id)


def delete_preset(preset_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM presets WHERE id = ?", (preset_id,))


# Every (mask, referring node) pair in the library. `nodes_using_mask` filters it
# to guard a delete; `list_masks(with_use_counts=True)` groups it to render the
# warning that *predicts* that delete. They share the text for the same reason
# `descendant_ids` and `delete_node` share `DESCENDANT_CTE`: a check and the thing
# it checks must not be able to drift. They did while this SQL was duplicated —
# the count's join compared a typeless subquery column against `masks.id`, whose
# INTEGER affinity silently coerced a TEXT id that the guard's `= ?` against a
# bound int did not match, so a `{"masks": ["7"]}` row was counted but not refused.
#
# Hence the CAST, which puts both on the one interpretation the rest of the app
# already uses: `effects.validate_selection` runs every id through Python's
# `int()` on read. Casting the two spellings rather than the bound parameter is
# what makes that normalization happen once, here, for every caller.
#
# Two arms because a selection holds a *union* of mask ids, and nothing migrates
# the pre-union `$.mask` spelling out of rows written before it: `json_extract`
# reads the scalar, `json_each` walks the array (it yields no rows when
# `selection` is NULL or has no `masks` key, so it needs no guard).
#
# UNION ALL, not UNION: deduping here would sort every pair in the database on
# every call to collapse only the node that holds one mask in *both* spellings.
# Each caller spells that dedupe itself, over the far smaller set it selects.
#
# Deliberately library-wide rather than scoped to one image's nodes. A reference
# from another image should never exist, but if one did it is precisely what must
# keep a delete from going through — so it has to be visible to both callers.
# Scoping the arms by image_id measures ~16x faster and is the obvious
# optimization; it is wrong for exactly that reason.
MASK_REFS = """SELECT CAST(json_extract(selection, '$.mask') AS INTEGER) AS mask_id,
                        id AS node_id
                   FROM nodes WHERE json_extract(selection, '$.mask') IS NOT NULL
                 UNION ALL
                 SELECT CAST(j.value AS INTEGER), n.id
                   FROM nodes n, json_each(n.selection, '$.masks') j"""


def mask_dict(row: sqlite3.Row) -> dict:
    d = {
        "id": row["id"],
        "image_id": row["image_id"],
        "name": row["name"],
        # provenance only — node_id carries no FK so the mask survives its
        # node's deletion, which means it may dangle or, since SQLite reuses
        # rowids, later name an unrelated node. Never present it as a link.
        "node_id": row["node_id"],
        "spec": json.loads(row["spec"]) if row["spec"] else None,
        "width": row["width"],
        "height": row["height"],
        "created_at": row["created_at"],
    }
    # present only when the caller asked list_masks to join the counts in, so
    # the key is absent rather than 0 when nobody counted — a mask that reports
    # `used_by: 0` has been checked, and the UI may trust it enough to skip a
    # delete warning
    if "used_by" in row.keys():
        d["used_by"] = row["used_by"]
    return d


def list_masks(image_id: int, *, with_use_counts: bool = False) -> list[dict]:
    """Oldest first, so a newly banked mask appends to the grid instead of
    shifting the tiles under the user's cursor.

    Not `ORDER BY name`: names are auto-generated ordinals now, and text order
    puts `Object 10` ahead of `Object 2`. Not `ORDER BY id` either — rowids are
    reused, so a brand-new mask can take a low id and sort into the middle.
    `created_at` is an ISO string with a fixed-width `+00:00` offset, so
    lexicographic order *is* chronological; `id` only breaks ties.

    `with_use_counts` adds each mask's referring-node count (see `MASK_REFS`) as
    `used_by`. It is opt-in because that join scans every node in the library,
    and two of the three callers — the mask-id set `_check_selection` validates
    against, and the taken-names set `_auto_mask_name` scans — want neither the
    number nor the scan. Only the list endpoint renders it.

    Joining it here rather than merging a second query's dict in the endpoint is
    what makes the list and its counts one snapshot. As two queries on two
    connections they were not: the counts were read first, so a node created in
    between made an in-use mask report `used_by: 0` — the grid would then omit
    the warning and let the user click into a 409, which is the exact surprise
    `used_by` exists to prevent."""
    join = count = ""
    if with_use_counts:
        count = ", COALESCE(u.used_by, 0) AS used_by"
        join = (
            " LEFT JOIN ("
            f"   SELECT mask_id, COUNT(DISTINCT node_id) AS used_by FROM ({MASK_REFS})"
            "    GROUP BY mask_id"
            " ) u ON u.mask_id = m.id"
        )
    with connect() as conn:
        rows = conn.execute(
            f"SELECT m.*{count} FROM masks m{join}"
            " WHERE m.image_id = ? ORDER BY m.created_at, m.id",
            (image_id,),
        ).fetchall()
    return [mask_dict(r) for r in rows]


def get_mask(mask_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM masks WHERE id = ?", (mask_id,)).fetchone()
    return mask_dict(row) if row else None


def create_mask(
    image_id: int, name: str, node_id: int, spec: dict, width: int, height: int
) -> dict:
    """Raises sqlite3.IntegrityError if the image already has a mask by this name."""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO masks (image_id, name, node_id, spec, width, height, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, node_id, json.dumps(spec), width, height, _now()),
        )
        mask_id = cur.lastrowid
    return get_mask(mask_id)


def rename_mask(mask_id: int, name: str) -> dict:
    """Raises sqlite3.IntegrityError if the new name collides within the image."""
    with connect() as conn:
        conn.execute("UPDATE masks SET name = ? WHERE id = ?", (name, mask_id))
    return get_mask(mask_id)


def delete_mask(mask_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM masks WHERE id = ?", (mask_id,))


def nodes_using_mask(mask_id: int) -> list[int]:
    """Nodes whose selection references this saved mask. Deleting the mask out
    from under them would change their committed pixels, so the API refuses.

    Reads `MASK_REFS`, whose comment carries the shared reasoning — including why
    this query and the `used_by` count it must agree with are one piece of SQL.

    `DISTINCT` is this caller's half of the dedupe `MASK_REFS` leaves undone: it
    collapses the node holding the same mask in both spellings, which would
    otherwise be named twice in the 409. `ORDER BY` restores the ascending ids the
    old `UNION` used to produce as a side effect of sorting — the 409 message
    lists them, so the order is user-visible.
    """
    with connect() as conn:
        return [
            r["node_id"]
            for r in conn.execute(
                f"SELECT DISTINCT node_id FROM ({MASK_REFS})"
                " WHERE mask_id = ? ORDER BY node_id",
                (mask_id,),
            )
        ]


def stats() -> dict:
    """Row counts for the library stats screen."""
    with connect() as conn:
        def one(sql: str) -> int:
            return conn.execute(sql).fetchone()[0]

        return {
            "images": one("SELECT COUNT(*) FROM images"),
            "nodes": one("SELECT COUNT(*) FROM nodes"),
            "edits": one("SELECT COUNT(*) FROM nodes WHERE effect IS NOT NULL"),
            "presets": one("SELECT COUNT(*) FROM presets"),
            "masks": one("SELECT COUNT(*) FROM masks"),
            "by_effect": [
                dict(r)
                for r in conn.execute(
                    """SELECT effect, COUNT(*) AS count FROM nodes
                       WHERE effect IS NOT NULL
                       GROUP BY effect ORDER BY count DESC, effect"""
                )
            ],
        }
