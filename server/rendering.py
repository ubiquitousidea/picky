"""Render pipeline: materialize a node's image by applying its effect chain."""

import io
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import json

from . import db, sam
from .effects import EFFECTS, apply_blend, kmeans_cluster_data

THUMB_SIZE = 320
JPEG_QUALITY = 92


def original_path(image_id: int) -> Path:
    return db.ORIGINALS_DIR / f"{image_id}.jpg"


def render_path(node_id: int) -> Path:
    return db.RENDERS_DIR / f"{node_id}.jpg"


def thumb_path(node_id: int) -> Path:
    return db.RENDERS_DIR / f"{node_id}.thumb.jpg"


def _apply(
    source_file: Path,
    effect: str,
    params: dict,
    parent2_id: int | None,
    selection: dict | None = None,
    source_node_id: int | None = None,
) -> np.ndarray:
    img = np.asarray(Image.open(source_file).convert("RGB"))
    if effect == "blend":
        other = Image.open(render_node(parent2_id)).convert("RGB")
        if other.size != (img.shape[1], img.shape[0]):
            other = other.resize((img.shape[1], img.shape[0]))
        out = apply_blend(
            img, np.asarray(other), params["mode"], float(params.get("weight", 0.5))
        )
    else:
        out = EFFECTS[effect]["apply"](img, params)
    if selection is not None:
        # masked apply: effect pixels inside the selection, source outside.
        # The mask comes from the same node whose pixels `img` holds, so the
        # shapes always agree.
        mask = compute_mask(source_node_id, selection)
        out = np.where(mask[:, :, None], out, img)
    return out


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
    result = _apply(
        parent_file,
        node["effect"],
        node["params"],
        node["parent2_id"],
        node["selection"],
        node["parent_id"],
    )
    Image.fromarray(result).save(out, quality=JPEG_QUALITY)
    return out


def render_preview(
    node_id: int,
    effect: str,
    params: dict,
    parent2_id: int | None = None,
    selection: dict | None = None,
) -> bytes:
    """Render an effect applied to a node's image in memory — no node row, no
    cache file. Shows exactly what a child created by Apply would look like."""
    result = _apply(render_node(node_id), effect, params, parent2_id, selection, node_id)
    buf = io.BytesIO()
    Image.fromarray(result).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


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


# ---------- Click-to-segment (SAM) ----------


def embedding_path(node_id: int) -> Path:
    return db.RENDERS_DIR / f"{node_id}.embedding.npy"


def node_embedding(node_id: int) -> np.ndarray:
    """The SAM image-encoder embedding of a node's rendered pixels, cached on
    disk like renders and cluster data (file existence is the cache key).

    Written atomically, unlike the JPEG renders: np.load of a half-written
    .npy raises, whereas a truncated JPEG merely looks bad — and two racing
    mask requests may both compute, so the tmp + os.replace also keeps them
    from interleaving writes.
    """
    out = embedding_path(node_id)
    if out.exists():
        return np.load(out)
    img = np.asarray(Image.open(render_node(node_id)).convert("RGB"))
    embedding = sam.compute_embedding(img)
    fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, embedding)
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return embedding


def compute_mask(node_id: int, selection: dict) -> np.ndarray:
    """Boolean mask for a click on a node's rendered pixels."""
    with Image.open(render_node(node_id)) as im:
        w, h = im.size  # header-only read; no full decode
    mask = sam.decode_mask(
        node_embedding(node_id), h, w, selection["x"], selection["y"], selection["level"]
    )
    if selection.get("invert"):
        mask = ~mask
    return mask


def mask_png(node_id: int, selection: dict) -> bytes:
    """The mask as a white-where-selected RGBA PNG for the frontend overlay
    (translucency is the frontend's job — CSS opacity on the whole image)."""
    mask = compute_mask(node_id, selection)
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = 255
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return buf.getvalue()


HIST_SAMPLE = 512


def histogram(node_id: int) -> dict:
    """Luma histogram of a node's rendered pixels, drawn behind the tone-curve
    editor's grid.

    Deliberately not cached to disk the way `cluster_data` is: k-means is
    expensive, this is one pass over a reduced decode (`draft()` lets libjpeg
    scale during the DCT), so it costs far less than the render it reads.
    Staying uncached also keeps it out of `delete_render_files` and
    `storage_stats` — no new file kind under `renders/`, no new invalidation edge.
    """
    im = Image.open(render_node(node_id))
    im.draft("RGB", (HIST_SAMPLE, HIST_SAMPLE))
    arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    luma = (arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.uint8)
    return {"luma": np.bincount(luma.ravel(), minlength=256).tolist()}


def delete_render_files(node_ids: list[int]) -> None:
    for node_id in node_ids:
        render_path(node_id).unlink(missing_ok=True)
        thumb_path(node_id).unlink(missing_ok=True)
        clusters_path(node_id).unlink(missing_ok=True)
        embedding_path(node_id).unlink(missing_ok=True)


def _totals(paths) -> dict:
    files = [p for p in paths if p.is_file()]
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def storage_stats() -> dict:
    """Bytes on disk, split by kind. Everything under `renders` is a cache that
    rebuilds from the originals plus the node rows, so it is reported apart from
    the data that cannot be regenerated."""
    renders = list(db.RENDERS_DIR.iterdir())
    thumbs = [p for p in renders if p.name.endswith(".thumb.jpg")]
    clusters = [p for p in renders if p.name.endswith(".clusters.json")]
    embeddings = [p for p in renders if p.name.endswith(".embedding.npy")]
    return {
        # glob, not DB_PATH.stat(), so a future WAL journal mode still adds up
        "database": _totals(db.DATA_DIR.glob("picky.db*")),
        "originals": _totals(db.ORIGINALS_DIR.iterdir()),
        # a thumb is also a .jpg, so exclude it rather than matching on suffix
        "renders": _totals(
            p for p in renders if p.suffix == ".jpg" and not p.name.endswith(".thumb.jpg")
        ),
        "thumbs": _totals(thumbs),
        "clusters": _totals(clusters),
        # SAM image embeddings, ~4 MB apiece — big enough to deserve a line
        "embeddings": _totals(embeddings),
    }
