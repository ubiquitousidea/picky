"""SQLite persistence for images and their effect work trees."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ORIGINALS_DIR = DATA_DIR / "originals"
RENDERS_DIR = DATA_DIR / "renders"
DB_PATH = DATA_DIR / "picky.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL
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
"""


def init() -> None:
    for d in (DATA_DIR, ORIGINALS_DIR, RENDERS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(nodes)")]
        if "parent2_id" not in cols:
            conn.execute(
                "ALTER TABLE nodes ADD COLUMN parent2_id INTEGER REFERENCES nodes(id)"
            )


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def node_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "image_id": row["image_id"],
        "parent_id": row["parent_id"],
        "parent2_id": row["parent2_id"],
        "effect": row["effect"],
        "params": json.loads(row["params"]) if row["params"] else None,
        "created_at": row["created_at"],
    }


def create_image(name: str) -> dict:
    with connect() as conn:
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
    return {"id": image_id, "name": name, "created_at": now, "root_node_id": root_id}


def list_images() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """SELECT i.id, i.name, i.created_at,
                      (SELECT id FROM nodes n WHERE n.image_id = i.id
                       AND n.parent_id IS NULL) AS root_node_id
               FROM images i ORDER BY i.created_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def get_image(image_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT id, name, created_at FROM images WHERE id = ?", (image_id,)
        ).fetchone()
    return dict(row) if row else None


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
) -> dict:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO nodes (image_id, parent_id, parent2_id, effect, params, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (image_id, parent_id, parent2_id, effect, json.dumps(params), _now()),
        )
        node_id = cur.lastrowid
    return get_node(node_id)


def delete_node(node_id: int) -> list[int]:
    """Delete a node and everything derived from it — children via either
    parent link, so blends that used a deleted node go too. Returns deleted ids."""
    with connect() as conn:
        ids = [
            r["id"]
            for r in conn.execute(
                """WITH RECURSIVE sub(id) AS (
                     SELECT id FROM nodes WHERE id = ?
                     UNION
                     SELECT n.id FROM nodes n
                     JOIN sub s ON n.parent_id = s.id OR n.parent2_id = s.id
                   ) SELECT id FROM sub""",
                (node_id,),
            ).fetchall()
        ]
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
