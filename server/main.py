"""Picky — browser-based JPG effects app."""

import io
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


class PreviewRequest(BaseModel):
    effect: str
    params: dict = {}
    parent2_id: int | None = None


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


class NoCacheStaticFiles(StaticFiles):
    """Make browsers revalidate frontend files on every load (cheap 304s via
    ETag) so app.js/style.css edits are picked up without a hard refresh."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", NoCacheStaticFiles(directory=WEB_DIR, html=True), name="web")
