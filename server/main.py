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
from .effects import EFFECTS, effect_specs, validate_params, validate_selection

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Picky")
db.init()


class NodeCreate(BaseModel):
    parent_id: int
    parent2_id: int | None = None
    effect: str
    params: dict = {}
    selection: dict | None = None


class NodeUpdate(BaseModel):
    params: dict = {}
    parent2_id: int | None = None
    selection: dict | None = None


class PreviewRequest(BaseModel):
    effect: str
    params: dict = {}
    parent2_id: int | None = None
    selection: dict | None = None


class MaskRequest(BaseModel):
    """Either shape of selection — a union of click specs or a union of saved
    mask ids. The fields are all optional because the total `validate_selection`
    is what actually decides which shape this is; the model is documentation.
    `x`/`y`/`level`/`mask` are the pre-union spellings, still accepted."""

    points: list[dict] | None = None
    masks: list[int] | None = None
    x: int | None = None
    y: int | None = None
    mask: int | None = None
    invert: bool = False
    level: str = "auto"


class MaskCreate(BaseModel):
    # omit `name` to have the server pick one — masks are identified by their
    # thumbnail now, so the name is only a tooltip and a uniqueness key
    name: str | None = None
    node_id: int
    selection: dict


class MaskUpdate(BaseModel):
    name: str


class PresetCreate(BaseModel):
    name: str
    node_id: int


class PresetApply(BaseModel):
    # the selection to run the whole recipe through, if any — see apply_preset
    preset_id: int
    selection: dict | None = None


def _check_selection(selection: dict | None, image_id: int) -> dict | None:
    """Resolve saved-mask references. `validate_selection` cannot do this —
    effects.py imports no DB — so ownership becomes a 400 here, mirroring the
    blend target check. A mask is scoped to the image it was picked on, which is
    what guarantees its frozen pixels match every node it can be used on. Every
    member of the union is checked: one foreign mask poisons the whole union,
    since they are OR'd into a single mask of one image's dimensions."""
    if selection is not None and "masks" in selection:
        for mask_id in selection["masks"]:
            mask = db.get_mask(mask_id)
            if mask is None or mask["image_id"] != image_id:
                raise HTTPException(400, "mask does not belong to this image")
    return selection


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
    selection = _check_selection(validate_selection(body.selection), image_id)

    parent2_id = None
    if body.effect == "blend":
        if body.parent2_id is None:
            raise HTTPException(400, "blend requires a second node (parent2_id)")
        other = db.get_node(body.parent2_id)
        if other is None or other["image_id"] != image_id:
            raise HTTPException(400, "blend node does not belong to this image")
        parent2_id = body.parent2_id

    node = db.create_node(
        image_id, body.parent_id, body.effect, params, parent2_id, selection
    )
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
    selection = _check_selection(validate_selection(body.selection), node["image_id"])

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

    if (
        params == node["params"]
        and parent2_id == node["parent2_id"]
        and selection == node["selection"]
    ):
        return {"node": node, "invalidated": []}

    # Update first, then invalidate: these endpoints run concurrently in a
    # threadpool, and after the commit any re-render already uses the new params,
    # so the sweep below can only be removing files written from the old ones.
    # The reverse order would let a concurrent read repopulate a stale cache.
    updated = db.update_node_params(node_id, params, parent2_id, selection)
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
    selection = _check_selection(validate_selection(body.selection), node["image_id"])

    parent2_id = None
    if body.effect == "blend":
        if body.parent2_id is None:
            raise HTTPException(400, "blend requires a second node (parent2_id)")
        other = db.get_node(body.parent2_id)
        if other is None or other["image_id"] != node["image_id"]:
            raise HTTPException(400, "blend node does not belong to this image")
        parent2_id = body.parent2_id

    try:
        data = rendering.render_preview(node_id, body.effect, params, parent2_id, selection)
    except Exception as exc:
        raise HTTPException(500, f"preview failed: {exc}")
    return Response(content=data, media_type="image/jpeg")


@app.post("/api/nodes/{node_id}/mask")
def get_mask(node_id: int, body: MaskRequest):
    """The overlay PNG for a selection on a node's pixels — a fresh click, which
    persists nothing but the node's cached embedding, or a saved mask, which is
    just read back. One endpoint for both is what keeps the frontend's overlay
    code free of branches."""
    node = db.get_node(node_id)
    if node is None:
        raise HTTPException(404, "node not found")
    selection = _check_selection(validate_selection(body.model_dump()), node["image_id"])
    if selection is None:
        raise HTTPException(400, "a selection needs either a click point or a mask id")
    try:
        data = rendering.mask_png(node_id, selection)
    except Exception as exc:
        raise HTTPException(500, f"segmentation failed: {exc}")
    return Response(content=data, media_type="image/png")


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
    # collect the mask ids before the cascade removes their rows; only the files
    # need a hand, the rows go with ON DELETE CASCADE
    mask_ids = [m["id"] for m in db.list_masks(image_id)]
    node_ids = db.delete_image(image_id)
    rendering.delete_render_files(node_ids)
    rendering.original_path(image_id).unlink(missing_ok=True)
    for mask_id in mask_ids:
        rendering.mask_path(mask_id).unlink(missing_ok=True)
    return {"deleted": image_id}


# ---------- Saved masks ----------
#
# A mask freezes a segmentation to a PNG so reuse never re-runs SAM: the shape is
# byte-identical everywhere it is used, and it outlives the node it was picked
# on. It belongs to the image it was picked on and is offered on every node of
# that image's tree — every node of an image shares its dimensions, so the frozen
# pixels always line up.


@app.get("/api/images/{image_id}/masks")
def list_masks(image_id: int):
    if db.get_image(image_id) is None:
        raise HTTPException(404, "image not found")
    # used_by rides along so the UI can warn before a delete is refused. One
    # joined query, not one per mask: banking a selection on every Apply means
    # this list grows with use, and it is refetched on every image switch.
    return db.list_masks(image_id, with_use_counts=True)


def _auto_mask_name(image_id: int) -> str:
    """`Object 1`, `Object 2`, … — the lowest ordinal this image is not using.

    Masks name themselves because the user never has to think about the name:
    the grid identifies them by thumbnail, so a name only has to satisfy the
    masks(image_id, name) index and read sensibly in a tooltip. Reusing a freed
    ordinal (delete `Object 2`, next save takes it back) is deliberate — the
    numbers stay small and they are not identifiers, the ids are.
    """
    taken = {m["name"] for m in db.list_masks(image_id)}
    n = 1
    while f"Object {n}" in taken:
        n += 1
    return f"Object {n}"


# an auto-named save races another one only if two Applies land in the same
# instant; these endpoints are sync `def`, so FastAPI runs them concurrently in
# a threadpool and that is a real, if rare, window
AUTO_NAME_TRIES = 5


@app.post("/api/images/{image_id}/masks")
def create_mask(image_id: int, body: MaskCreate):
    if db.get_image(image_id) is None:
        raise HTTPException(404, "image not found")
    name = None if body.name is None else body.name.strip()
    if name == "":
        # an explicitly empty name is a client bug, not a request to auto-name
        raise HTTPException(400, "mask name cannot be empty")
    node = db.get_node(body.node_id)
    if node is None or node["image_id"] != image_id:
        raise HTTPException(400, "node does not belong to this image")
    spec = validate_selection(body.selection)
    if spec is None or "masks" in spec:
        raise HTTPException(400, "a mask is saved from a click selection")

    try:
        # compute_mask already applies spec's invert, so what gets frozen is
        # exactly what the user was looking at. A reference's own invert then
        # toggles on top of that — see the XOR in capture_steps.
        mask = rendering.compute_mask(body.node_id, spec)
    except Exception as exc:
        raise HTTPException(500, f"segmentation failed: {exc}")

    height, width = mask.shape
    # a caller-supplied name is taken literally and collides once; an auto-name
    # re-derives and retries, since losing the race just means picking again
    for attempt in range(AUTO_NAME_TRIES if name is None else 1):
        chosen = name if name is not None else _auto_mask_name(image_id)
        try:
            saved = db.create_mask(image_id, chosen, body.node_id, spec, width, height)
            break
        except sqlite3.IntegrityError:
            if name is not None or attempt == AUTO_NAME_TRIES - 1:
                raise HTTPException(
                    409, f"a mask named '{chosen}' already exists for this image"
                )
    # row first so the file is named by the id; unwind it if the write fails,
    # so a row can never point at a missing PNG
    try:
        rendering.save_mask(saved["id"], mask)
    except Exception as exc:
        db.delete_mask(saved["id"])
        raise HTTPException(500, f"saving the mask failed: {exc}")
    return {**saved, "used_by": 0}


@app.patch("/api/masks/{mask_id}")
def update_mask(mask_id: int, body: MaskUpdate):
    """Rename. Presets have no equivalent because re-saving one is free; redoing
    a mask means re-picking the object, which is the cost this feature exists to
    remove."""
    mask = db.get_mask(mask_id)
    if mask is None:
        raise HTTPException(404, "mask not found")
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "mask name cannot be empty")
    try:
        return db.rename_mask(mask_id, name)
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"a mask named '{name}' already exists for this image")


@app.delete("/api/masks/{mask_id}")
def delete_mask(mask_id: int):
    """Refuse while nodes still reference it. Clearing their selections instead
    would silently change committed pixels and force a re-render of every one of
    their descendants — a large invisible side effect behind an '×'."""
    if db.get_mask(mask_id) is None:
        raise HTTPException(404, "mask not found")
    users = db.nodes_using_mask(mask_id)
    if users:
        raise HTTPException(
            409,
            f"{len(users)} node{'s' if len(users) > 1 else ''}"
            f" still use{'s' if len(users) == 1 else ''} this mask"
            f" (#{', #'.join(str(i) for i in users)}) — delete them first",
        )
    db.delete_mask(mask_id)
    rendering.mask_path(mask_id).unlink(missing_ok=True)
    return {"deleted": mask_id}


@app.get("/api/masks/{mask_id}/thumb")
def get_mask_thumb(mask_id: int):
    """A mask's silhouette as a small PNG — the icon in the selection grid.

    Computed per request rather than cached (see `mask_thumb_png`). The one
    cache that *is* worth having is the browser's, and that is also where the
    rowid-reuse hazard is unfixable — nothing can unlink a cache entry — so the
    frontend versions the URL with the mask's `created_at` and this is
    `immutable`. That combination is what keeps the grid rebuild that
    `renderSelectControls` does on every selection change from refetching
    every icon.
    """
    if db.get_mask(mask_id) is None:
        raise HTTPException(404, "mask not found")
    try:
        data = rendering.mask_thumb_png(mask_id)
    except Exception as exc:
        raise HTTPException(500, f"thumbnail failed: {exc}")
    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


MAX_PRESET_STEPS = 100


def _portable_selection(selection: dict | None) -> dict | None:
    """A selection as a preset can carry it: click specs, never saved masks.

    A preset stores params, not pixels — replaying one runs today's effect code
    against another image. A saved mask *is* pixels, and image-scoped ones at
    that, so it degrades back to the click specs it was frozen from and gets
    re-segmented on the target. Dropping the selection instead would silently
    turn a masked blur into a whole-image blur. This is the reason the click
    shape is a union too: a union of masks has nowhere else to land.

    Inverts are the lossy part. With a single mask they XOR exactly: the PNG was
    frozen post-invert, so the spec's own invert already produced the saved shape
    and the reference's invert toggles it again. With several, there is no single
    invert that reproduces the union — an inverted region cannot join a union of
    points — so a mask whose spec was inverted is dropped from the step. The
    remaining masks still replay; the alternative is discarding the whole
    selection, which is the failure mode this function exists to avoid.

    A mask deleted since (or one with no stored spec) is skipped for the same
    reason, and a step left with nothing replays unmasked rather than failing
    the save, matching validate_selection's totality.
    """
    if selection is None or "masks" not in selection:
        return selection
    specs = []
    for mask_id in selection["masks"]:
        mask = db.get_mask(mask_id)
        spec = validate_selection(mask["spec"]) if mask else None
        if spec is not None and "points" in spec:
            specs.append(spec)
    if not specs:
        return None
    if len(specs) == 1:
        return {**specs[0], "invert": specs[0]["invert"] != selection["invert"]}
    points = [p for spec in specs if not spec["invert"] for p in spec["points"]]
    return {"points": points, "invert": selection["invert"]} if points else None


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
            # click coords ride along; they are in the parent's pixel space and
            # clamp to the target image's bounds at mask time. Saved masks
            # degrade to the click specs behind them — see _portable_selection.
            "selection": _portable_selection(validate_selection(node["selection"])),
        }
        steps.append(step)
    return steps


def _drop_mask_ref(selection: dict | None) -> dict | None:
    if selection is not None and "masks" in selection:
        return None  # image-scoped; a preset is not
    return selection


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
            {
                "effect": effect,
                "params": params,
                "parent": parent,
                "parent2": parent2,
                # total: presets saved before selections existed, or with
                # malformed data, replay unmasked rather than failing. A mask
                # reference should never reach a stored preset (capture_steps
                # degrades it), so one that did is hand-edited or older data —
                # strip it rather than resolve it against the wrong image.
                "selection": _drop_mask_ref(validate_selection(step.get("selection"))),
            }
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

    image_id = base["image_id"]
    # An apply-time selection masks *every* step, and beats whatever the recipe
    # stored for that step: a preset is a chain of edits, and limiting it to an
    # object is the same gesture as limiting one effect to it — what is ticked is
    # what gets edited. The step's own selection is a click spec captured from
    # some other image, so honouring both would mean intersecting a union of
    # points with a union of masks, a shape the model does not have. Resolved
    # here, next to _validate_steps, so a foreign mask 400s before the first row
    # is written rather than half way through the chain.
    override = _check_selection(validate_selection(body.selection), image_id)

    # Each step must be committed and rendered before the next one references it:
    # render_node reads through its own connection, so an open transaction would
    # hide the rows it needs. Unwind by hand if a step fails.
    created: list[int] = []
    node_at = {0: node_id}
    try:
        for i, step in enumerate(steps, start=1):
            parent2_id = None if step["parent2"] is None else node_at[step["parent2"]]
            node = db.create_node(
                image_id,
                node_at[step["parent"]],
                step["effect"],
                step["params"],
                parent2_id,
                step["selection"] if override is None else override,
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
