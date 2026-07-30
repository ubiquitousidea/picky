"""Render pipeline: materialize a node's image by applying its effect chain."""

import io
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import json

from . import db, sam
from .effects import EFFECTS, apply_blend, kmeans_cluster_data, validate_selection

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
        # For a click spec the mask is segmented from the same node whose pixels
        # `img` holds, so the shapes always agree. A saved mask ignores
        # source_node_id and still agrees, one step weaker: every EFFECTS entry
        # is a pixel-for-pixel transform and blend resizes its second input to
        # the first, so every node of an image shares its dimensions.
        mask = compute_mask(source_node_id, selection)
        if mask.shape != img.shape[:2]:
            raise ValueError(
                f"mask is {mask.shape[1]}×{mask.shape[0]}, "
                f"image is {img.shape[1]}×{img.shape[0]}"
            )
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


def compute_mask(node_id: int | None, selection: dict) -> np.ndarray:
    """Boolean mask for a selection: the union of some saved masks' frozen
    pixels, or the union of some clicks re-segmented against a node's rendered
    pixels.

    Normalizes through `validate_selection` on the way in, so this is total for
    any shape a selection has ever had on disk and callers need not pre-clean.
    In practice they all do — `db.node_dict` normalizes everything read from the
    database (which is how `render_node`'s `node["selection"]` arrives here) and
    `main.py` normalizes every request body — so this is the belt to their
    braces, and the one place that stays correct if a future caller forgets.

    `invert` is applied once at the end to the whole union, so toggling it on
    saved masks behaves exactly as it does on clicks — and costs nothing, since
    a saved mask's PNG was already frozen with its own invert baked in."""
    selection = validate_selection(selection)
    if selection is None:
        raise ValueError("selection is empty")
    if "masks" in selection:
        # node_id is irrelevant: frozen pixels
        masks = [load_mask(mask_id) for mask_id in selection["masks"]]
    else:
        with Image.open(render_node(node_id)) as im:
            w, h = im.size  # header-only read; no full decode
        # one encoder pass, cached; each extra point costs only a decoder run
        embedding = node_embedding(node_id)
        masks = [
            sam.decode_mask(embedding, h, w, p["x"], p["y"], p["level"])
            for p in selection["points"]
        ]
    mask = masks[0]
    for other in masks[1:]:
        if other.shape != mask.shape:
            raise ValueError(
                f"masks disagree: {mask.shape[1]}×{mask.shape[0]} "
                f"vs {other.shape[1]}×{other.shape[0]}"
            )
        mask = mask | other
    if selection["invert"]:
        mask = ~mask
    return mask


# ---------- Saved masks (frozen pixels) ----------


def mask_path(mask_id: int) -> Path:
    return db.MASKS_DIR / f"{mask_id}.png"


def save_mask(mask_id: int, mask: np.ndarray) -> None:
    """Freeze a boolean mask to disk as a 1-bit PNG — a few tens of KB even for
    a pathological mask, and the point of the whole feature: reuse loads these
    pixels back rather than re-running SAM, so the shape never drifts."""
    Image.fromarray(mask).save(mask_path(mask_id), format="PNG", optimize=True)


def load_mask(mask_id: int) -> np.ndarray:
    with Image.open(mask_path(mask_id)) as im:
        return np.asarray(im.convert("1")).astype(bool)


MASK_THUMB_PX = 160


def mask_thumb_png(mask_id: int) -> bytes:
    """A saved mask shrunk to an icon: white where selected, black where not.

    A silhouette rather than the photo behind it, because this is a picture of
    the *mask* — a shape against a flat field is legible at 50-odd display
    pixels in a way a downsampled photograph is not, and an inverted pick reads
    at a glance for free.

    Deliberately **not** cached to a file, for the same reason `histogram` is
    not: ~15 ms even on a 7728×5152 mask and well under a kilobyte on the wire,
    which is cheaper than the `load_mask` these endpoints already run. A
    `<id>.thumb.png` would be a second file keyed by a rowid SQLite reuses, so
    one missed unlink would serve a deleted mask's silhouette as its
    successor's — and it would need excluding from `storage_stats`, whose masks
    line means "the bytes you cannot regenerate". Derived instead from the one
    file whose deletion is already correct, it cannot go stale.
    """
    with Image.open(mask_path(mask_id)) as im:
        # a stored mask opens as mode "1", which resizes NEAREST whatever filter
        # is asked for — convert first or the silhouette comes back jagged
        thumb = im.convert("L")
    # BOX, not thumbnail()'s default: on a binary mask a box filter is exactly
    # "what fraction of this source cell was selected". Cubic filters ring, and
    # ringing puts bright halos outside the object and dark ones inside it.
    thumb.thumbnail((MASK_THUMB_PX, MASK_THUMB_PX), Image.BOX)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _shift(mask: np.ndarray, step: int, axis: int) -> np.ndarray:
    """`mask` translated by `step` along `axis`, shifting False in at the edge."""
    out = np.zeros_like(mask)
    dst, src = [slice(None), slice(None)], [slice(None), slice(None)]
    if step > 0:
        dst[axis], src[axis] = slice(step, None), slice(None, -step)
    else:
        dst[axis], src[axis] = slice(None, step), slice(-step, None)
    out[tuple(dst)] = mask[tuple(src)]
    return out


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a boolean mask by a (2*radius+1) square.

    Separable and logarithmic: a square structuring element is a horizontal line
    composed with a vertical one, and dilating by `a` then by `b` dilates by
    `a + b` — so a line of length `radius` is reached in doubling steps
    (1, 2, 4, …). That is O(log radius) whole-array passes per axis.

    The obvious `ImageFilter.MaxFilter(2 * radius + 1)` is O(w·h·radius²) and
    took **30 seconds** on a 7728×5152 mask, since the stroke width scales with
    the image; this runs in well under a second on the same input.
    """
    if radius <= 0:
        return mask
    for axis in (0, 1):
        step, remaining = 1, radius
        while remaining > 0:
            k = min(step, remaining)
            mask = mask | _shift(mask, k, axis) | _shift(mask, -k, axis)
            remaining -= k
            step *= 2
    return mask


def _overlay_png(mask: np.ndarray) -> bytes:
    """A boolean mask as an RGBA PNG tracing its *outline* for the frontend
    overlay. Saved masks are *stored* 1-bit and colorized here on read, so
    there is one file and one truth.

    An outline rather than a translucent fill because the fill hid the very
    pixels you are picking. It is drawn fully opaque and composited with a dark
    casing under a white core, so it stays legible over any photograph — the
    frontend just lays the PNG over the render at opacity 1.

    Stroke width scales with the image because the overlay is displayed fitted
    to the preview panel: a 1px line on a 4000px photo would vanish.
    """
    h, w = mask.shape
    # the stroke is 2*radius+1 wide; ~1/250 of the short side puts it at 3-4
    # display pixels once the overlay is fitted to the panel, at any image size
    radius = max(1, round(min(h, w) / 500))
    # inner boundary: selected pixels missing at least one 4-neighbour
    pad = np.pad(mask, 1)
    edge = mask & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    core = _dilate(edge, radius)
    casing = _dilate(edge, radius + 2)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[casing] = (16, 18, 22, 255)
    rgba[core] = (255, 255, 255, 255)
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return buf.getvalue()


def mask_png(node_id: int, selection: dict) -> bytes:
    """The overlay PNG for a selection on a node — either shape."""
    return _overlay_png(compute_mask(node_id, selection))


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
    the data that cannot be regenerated — which now includes saved masks, whose
    frozen pixels are exactly what re-running SAM would not reproduce."""
    renders = list(db.RENDERS_DIR.iterdir())
    thumbs = [p for p in renders if p.name.endswith(".thumb.jpg")]
    clusters = [p for p in renders if p.name.endswith(".clusters.json")]
    embeddings = [p for p in renders if p.name.endswith(".embedding.npy")]
    return {
        # glob, not DB_PATH.stat(), so a future WAL journal mode still adds up
        "database": _totals(db.DATA_DIR.glob("picky.db*")),
        "originals": _totals(db.ORIGINALS_DIR.iterdir()),
        "masks": _totals(db.MASKS_DIR.iterdir()),
        # a thumb is also a .jpg, so exclude it rather than matching on suffix
        "renders": _totals(
            p for p in renders if p.suffix == ".jpg" and not p.name.endswith(".thumb.jpg")
        ),
        "thumbs": _totals(thumbs),
        "clusters": _totals(clusters),
        # SAM image embeddings, ~4 MB apiece — big enough to deserve a line
        "embeddings": _totals(embeddings),
    }
