"""Picky — browser-based JPG effects app."""

import io
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from . import db, rendering
from .effects import EFFECTS, effect_specs, validate_params

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Picky")
db.init()


class NodeCreate(BaseModel):
    parent_id: int
    parent2_id: int | None = None
    effect: str
    params: dict = {}


class NodeUpdate(BaseModel):
    params: dict = {}
    parent2_id: int | None = None


class PreviewRequest(BaseModel):
    effect: str
    params: dict = {}
    parent2_id: int | None = None


class PresetCreate(BaseModel):
    name: str
    node_id: int


class PresetApply(BaseModel):
    preset_id: int


@app.get("/api/effects")
def get_effects():
    return effect_specs()


@app.post("/api/images")
async def upload_image(file: UploadFile):
    data = await file.read()
    try:
        im = Image.open(io.BytesIO(data))
        im.verify()
    except Exception:
        raise HTTPException(400, "not a valid image file")
    if im.format != "JPEG":
        raise HTTPException(400, f"only JPEG images are supported (got {im.format})")

    image = db.create_image(file.filename or "untitled.jpg")
    rendering.original_path(image["id"]).write_bytes(data)
    return image


@app.get("/api/images")
def list_images():
    return db.list_images()


@app.get("/api/images/{image_id}/tree")
def get_tree(image_id: int):
    if db.get_image(image_id) is None:
        raise HTTPException(404, "image not found")
    return db.get_tree(image_id)


@app.post("/api/images/{image_id}/nodes")
def create_node(image_id: int, body: NodeCreate):
    if db.get_image(image_id) is None:
        raise HTTPException(404, "image not found")
    parent = db.get_node(body.parent_id)
    if parent is None or parent["image_id"] != image_id:
        raise HTTPException(400, "parent node does not belong to this image")
    if body.effect not in EFFECTS and body.effect != "blend":
        raise HTTPException(400, f"unknown effect '{body.effect}'")
    params = validate_params(body.effect, body.params)

    parent2_id = None
    if body.effect == "blend":
        if body.parent2_id is None:
            raise HTTPException(400, "blend requires a second node (parent2_id)")
        other = db.get_node(body.parent2_id)
        if other is None or other["image_id"] != image_id:
            raise HTTPException(400, "blend node does not belong to this image")
        parent2_id = body.parent2_id

    node = db.create_node(image_id, body.parent_id, body.effect, params, parent2_id)
    try:
        rendering.render_node(node["id"])
    except Exception as exc:
        raise HTTPException(500, f"render failed: {exc}")
    return node


@app.patch("/api/nodes/{node_id}")
def update_node(node_id: int, body: NodeUpdate):
    """Change an existing node's settings in place. Unlike Apply, this does not
    add a node — the node keeps its id and its children, which are re-rendered
    from the new pixels."""
    node = db.get_node(node_id)
    if node is None:
        raise HTTPException(404, "node not found")
    if node["parent_id"] is None:
        raise HTTPException(400, "the original has no parameters to edit")
    effect = node["effect"]
    if effect not in EFFECTS and effect != "blend":
        raise HTTPException(400, f"effect '{effect}' no longer exists")
    params = validate_params(effect, body.params)

    parent2_id = None
    if effect == "blend":
        if body.parent2_id is None:
            raise HTTPException(400, "blend requires a second node (parent2_id)")
        other = db.get_node(body.parent2_id)
        if other is None or other["image_id"] != node["image_id"]:
            raise HTTPException(400, "blend node does not belong to this image")
        # Ids are topological, so anything with a smaller id cannot be downstream
        # of this node. Creation gets that for free (a new node always has the
        # largest id); an edit does not, and pointing a blend at its own
        # descendant would send render_node into infinite recursion.
        if body.parent2_id >= node_id:
            raise HTTPException(400, "a blend cannot use a node derived from itself")
        parent2_id = body.parent2_id

    if params == node["params"] and parent2_id == node["parent2_id"]:
        return {"node": node, "invalidated": []}

    # Update first, then invalidate: these endpoints run concurrently in a
    # threadpool, and after the commit any re-render already uses the new params,
    # so the sweep below can only be removing files written from the old ones.
    # The reverse order would let a concurrent read repopulate a stale cache.
    updated = db.update_node_params(node_id, params, parent2_id)
    invalidated = db.descendant_ids(node_id)
    rendering.delete_render_files(invalidated)
    try:
        rendering.render_node(node_id)  # fail loudly here rather than on next GET
    except Exception as exc:
        raise HTTPException(500, f"render failed: {exc}")
    return {"node": updated, "invalidated": invalidated}


def _effect_chain(node: dict) -> list[str]:
    parts = []
    cur = node
    while cur and cur["effect"]:
        parts.append(cur["effect"])
        cur = db.get_node(cur["parent_id"])
    return list(reversed(parts))


@app.get("/api/nodes/{node_id}/render")
def get_render(node_id: int, thumb: bool = False, download: bool = False):
    node = db.get_node(node_id)
    if node is None:
        raise HTTPException(404, "node not found")
    path = rendering.render_thumb(node_id) if thumb else rendering.render_node(node_id)
    if download:
        image = db.get_image(node["image_id"])
        stem = Path(image["name"]).stem or "image"
        suffix = "-".join(_effect_chain(node)) or "original"
        return FileResponse(
            path, media_type="image/jpeg", filename=f"{stem}-{suffix}.jpg"
        )
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/nodes/{node_id}/preview")
def preview_node(node_id: int, body: PreviewRequest):
    node = db.get_node(node_id)
    if node is None:
        raise HTTPException(404, "node not found")
    if body.effect not in EFFECTS and body.effect != "blend":
        raise HTTPException(400, f"unknown effect '{body.effect}'")
    params = validate_params(body.effect, body.params)

    parent2_id = None
    if body.effect == "blend":
        if body.parent2_id is None:
            raise HTTPException(400, "blend requires a second node (parent2_id)")
        other = db.get_node(body.parent2_id)
        if other is None or other["image_id"] != node["image_id"]:
            raise HTTPException(400, "blend node does not belong to this image")
        parent2_id = body.parent2_id

    try:
        data = rendering.render_preview(node_id, body.effect, params, parent2_id)
    except Exception as exc:
        raise HTTPException(500, f"preview failed: {exc}")
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/nodes/{node_id}/clusters")
def get_clusters(node_id: int):
    node = db.get_node(node_id)
    if node is None:
        raise HTTPException(404, "node not found")
    if node["effect"] != "posterize":
        raise HTTPException(400, "cluster data only exists for posterize nodes")
    return FileResponse(rendering.cluster_data(node_id), media_type="application/json")


@app.get("/api/nodes/{node_id}/histogram")
def get_histogram(node_id: int):
    if db.get_node(node_id) is None:
        raise HTTPException(404, "node not found")
    return rendering.histogram(node_id)


@app.delete("/api/nodes/{node_id}")
def delete_node(node_id: int):
    node = db.get_node(node_id)
    if node is None:
        raise HTTPException(404, "node not found")
    if node["parent_id"] is None:
        raise HTTPException(400, "cannot delete the original; delete the image instead")
    node_ids = db.delete_node(node_id)
    rendering.delete_render_files(node_ids)
    return {"deleted": node_ids, "parent_id": node["parent_id"]}


@app.delete("/api/images/{image_id}")
def delete_image(image_id: int):
    if db.get_image(image_id) is None:
        raise HTTPException(404, "image not found")
    node_ids = db.delete_image(image_id)
    rendering.delete_render_files(node_ids)
    rendering.original_path(image_id).unlink(missing_ok=True)
    return {"deleted": image_id}


MAX_PRESET_STEPS = 100


def capture_steps(node_id: int) -> list[dict]:
    """Capture the edits that produced a node as a portable, image-independent
    recipe: every ancestor reachable through *either* parent link (that closure is
    exactly the node's pixel dependency set), in id order, with parent references
    rewritten as indices — 0 meaning "whatever node this gets applied to" and
    1..n-1 meaning an earlier step. Absolute node ids would be unusable on another
    image, and unsafe even on this one since SQLite reuses rowids after deletes."""
    closure: dict[int, dict] = {}
    stack = [node_id]
    while stack:
        current = stack.pop()
        if current in closure:
            continue
        node = db.get_node(current)
        if node is None:
            raise HTTPException(404, f"node {current} not found")
        closure[current] = node
        for parent in (node["parent_id"], node["parent2_id"]):
            if parent is not None:
                stack.append(parent)

    # ids are topological (a parent always has a smaller id), so sorting gives a
    # valid replay order and the root — the only effect-less node — comes first
    ordered = [closure[i] for i in sorted(closure)]
    index = {node["id"]: i for i, node in enumerate(ordered)}

    steps = []
    for node in ordered:
        if node["effect"] is None:
            continue  # the root stands in for the base node, index 0
        if node["effect"] not in EFFECTS and node["effect"] != "blend":
            raise HTTPException(400, f"effect '{node['effect']}' no longer exists")
        # normalizing here also backfills params added since the node was made,
        # e.g. blend's weight, which rendering defaults to 0.5 for older nodes
        params = validate_params(node["effect"], node["params"] or {})
        step = {
            "effect": node["effect"],
            "params": params,
            "parent": index[node["parent_id"]],
            "parent2": index[node["parent2_id"]] if node["effect"] == "blend" else None,
        }
        steps.append(step)
    return steps


def _step_summary(steps: list[dict]) -> str:
    return " → ".join(s.get("effect", "?") for s in steps)


@app.get("/api/presets")
def list_presets():
    return [
        {**preset, "summary": _step_summary(preset["steps"])}
        for preset in db.list_presets()
    ]


@app.post("/api/presets")
def create_preset(body: PresetCreate):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "preset name cannot be empty")
    if db.get_node(body.node_id) is None:
        raise HTTPException(404, "node not found")
    steps = capture_steps(body.node_id)
    if not steps:
        raise HTTPException(400, "select a node with at least one effect")
    if len(steps) > MAX_PRESET_STEPS:
        raise HTTPException(400, f"presets are limited to {MAX_PRESET_STEPS} steps")
    try:
        preset = db.create_preset(name, steps)
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"a preset named '{name}' already exists")
    return {**preset, "summary": _step_summary(preset["steps"])}


@app.delete("/api/presets/{preset_id}")
def delete_preset(preset_id: int):
    if db.get_preset(preset_id) is None:
        raise HTTPException(404, "preset not found")
    db.delete_preset(preset_id)
    return {"deleted": preset_id}


def _validate_steps(steps: list[dict]) -> list[dict]:
    """Check a stored recipe before anything is written, so a malformed preset
    can never leave half a chain behind. Returns the steps with params
    re-normalized against today's effect registry."""
    if not isinstance(steps, list) or not steps:
        raise HTTPException(400, "preset has no steps")
    checked = []
    for i, step in enumerate(steps, start=1):
        effect = step.get("effect") if isinstance(step, dict) else None
        if effect not in EFFECTS and effect != "blend":
            raise HTTPException(400, f"step {i}: unknown effect '{effect}'")

        parent = step.get("parent")
        # bool is an int subclass, and a negative index would silently resolve
        # against the created-node list, so check the type and range explicitly
        if not isinstance(parent, int) or isinstance(parent, bool) or not 0 <= parent < i:
            raise HTTPException(400, f"step {i}: invalid parent index {parent!r}")

        parent2 = step.get("parent2")
        if effect == "blend":
            if parent2 is None:
                raise HTTPException(400, f"step {i}: blend requires a second input")
            if (
                not isinstance(parent2, int)
                or isinstance(parent2, bool)
                or not 0 <= parent2 < i
            ):
                raise HTTPException(400, f"step {i}: invalid parent2 index {parent2!r}")
        else:
            parent2 = None  # never persist a phantom second parent on a 1-input effect

        try:
            params = validate_params(effect, step.get("params") or {})
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, f"step {i}: bad params ({exc})")
        checked.append(
            {"effect": effect, "params": params, "parent": parent, "parent2": parent2}
        )
    return checked


@app.post("/api/nodes/{node_id}/apply-preset")
def apply_preset(node_id: int, body: PresetApply):
    base = db.get_node(node_id)
    if base is None:
        raise HTTPException(404, "node not found")
    preset = db.get_preset(body.preset_id)
    if preset is None:
        raise HTTPException(404, "preset not found")
    steps = _validate_steps(preset["steps"])

    # Each step must be committed and rendered before the next one references it:
    # render_node reads through its own connection, so an open transaction would
    # hide the rows it needs. Unwind by hand if a step fails.
    image_id = base["image_id"]
    created: list[int] = []
    node_at = {0: node_id}
    try:
        for i, step in enumerate(steps, start=1):
            parent2_id = None if step["parent2"] is None else node_at[step["parent2"]]
            node = db.create_node(
                image_id, node_at[step["parent"]], step["effect"], step["params"], parent2_id
            )
            created.append(node["id"])
            node_at[i] = node["id"]
            rendering.render_node(node["id"])
    except Exception as exc:
        try:
            db.delete_nodes(created)
            rendering.delete_render_files(created)
        except Exception:  # a failed cleanup must not mask the real error
            pass
        raise HTTPException(500, f"applying preset failed: {exc}")

    return {"created": created, "terminal_node_id": created[-1]}


@app.get("/api/stats")
def get_stats():
    return {**db.stats(), "storage": rendering.storage_stats()}


class NoCacheStaticFiles(StaticFiles):
    """Make browsers revalidate frontend files on every load (cheap 304s via
    ETag) so app.js/style.css edits are picked up without a hard refresh."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", NoCacheStaticFiles(directory=WEB_DIR, html=True), name="web")
