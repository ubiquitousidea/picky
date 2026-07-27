"""Render pipeline: materialize a node's image by applying its effect chain."""

from pathlib import Path

import numpy as np
from PIL import Image

import json

from . import db
from .effects import EFFECTS, apply_blend, kmeans_cluster_data

THUMB_SIZE = 320
JPEG_QUALITY = 92


def original_path(image_id: int) -> Path:
    return db.ORIGINALS_DIR / f"{image_id}.jpg"


def render_path(node_id: int) -> Path:
    return db.RENDERS_DIR / f"{node_id}.jpg"


def thumb_path(node_id: int) -> Path:
    return db.RENDERS_DIR / f"{node_id}.thumb.jpg"


def render_node(node_id: int) -> Path:
    """Return the path of the node's rendered JPEG, rendering (recursively) if
    the cache file is missing."""
    node = db.get_node(node_id)
    if node is None:
        raise KeyError(f"node {node_id} not found")
    if node["parent_id"] is None:
        return original_path(node["image_id"])

    out = render_path(node_id)
    if out.exists():
        return out

    parent_file = render_node(node["parent_id"])
    img = np.asarray(Image.open(parent_file).convert("RGB"))
    if node["effect"] == "blend":
        other = Image.open(render_node(node["parent2_id"])).convert("RGB")
        if other.size != (img.shape[1], img.shape[0]):
            other = other.resize((img.shape[1], img.shape[0]))
        result = apply_blend(img, np.asarray(other), node["params"]["mode"])
    else:
        result = EFFECTS[node["effect"]]["apply"](img, node["params"])
    Image.fromarray(result).save(out, quality=JPEG_QUALITY)
    return out


def render_thumb(node_id: int) -> Path:
    out = thumb_path(node_id)
    if out.exists():
        return out
    im = Image.open(render_node(node_id)).convert("RGB")
    im.thumbnail((THUMB_SIZE, THUMB_SIZE))
    im.save(out, quality=85)
    return out


def clusters_path(node_id: int) -> Path:
    return db.RENDERS_DIR / f"{node_id}.clusters.json"


def cluster_data(node_id: int) -> Path:
    """Cached k-means cluster data for a posterize node's 3D scatter plot,
    computed from the same source image the posterization saw (its parent)."""
    out = clusters_path(node_id)
    if out.exists():
        return out
    node = db.get_node(node_id)
    img = np.asarray(Image.open(render_node(node["parent_id"])).convert("RGB"))
    data = kmeans_cluster_data(img, int(node["params"]["k"]))
    out.write_text(json.dumps(data))
    return out


def delete_render_files(node_ids: list[int]) -> None:
    for node_id in node_ids:
        render_path(node_id).unlink(missing_ok=True)
        thumb_path(node_id).unlink(missing_ok=True)
        clusters_path(node_id).unlink(missing_ok=True)
