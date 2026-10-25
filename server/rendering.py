"""Render pipeline: materialize a node's image by applying its effect chain."""

import io
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import json

from . import db, detect, embed, sam
from .effects import (
    EFFECTS,
    apply_blend,
    apply_geometry,
    crop_geometry,
    is_identity_crop,
    kmeans_cluster_data,
    validate_selection,
)

THUMB_SIZE = 320
JPEG_QUALITY = 92
# The rotated-but-uncropped proxy the frame editor drags over. Capped well below
# full size because straightening re-renders it, and the frame is stored as
# *fractions* of the canvas — so a proxy and the full-size render describe the
# same crop, and nothing is lost by dragging against the small one. What *is*
# lost is the true pixel count, which is why `crop_preview` also returns the
# full-size canvas.
CROP_PREVIEW_PX = 1600


def original_path(image_id: int) -> Path:
    return db.ORIGINALS_DIR / f"{image_id}.jpg"


def render_path(node_id: int) -> Path:
    return db.RENDERS_DIR / f"{node_id}.jpg"


def thumb_path(node_id: int) -> Path:
    return db.RENDERS_DIR / f"{node_id}.thumb.jpg"


def output_path(node_id: int) -> Path:
    return db.RENDERS_DIR / f"{node_id}.out.jpg"


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


# ---------- The output stage: the image's one framing, applied last ----------
#
# A crop is not a node. The work tree computes every node at the original's
# dimensions — which is what keeps saved masks, `compute_mask` and blend's resize
# all working on one shared size — and the image's framing is applied once, here,
# to whatever the tree produced. Everything a user *sees* comes through this
# stage; everything the tree *computes* is upstream of it and never sees a crop.


def image_crop(image_id: int) -> dict | None:
    image = db.get_image(image_id)
    return image["crop"] if image else None


def render_output(node_id: int) -> Path:
    """The node's rendered JPEG with its image's framing applied — what every
    viewer of this node gets: the preview, the tree thumbnail, and Export.

    Falls straight through to `render_path` when the crop is the identity, so an
    uncropped library pays nothing: no second decode, no second file on disk.
    That short-circuit is why `output_path` may simply not exist, and why
    `is_identity_crop` has to agree with what this writes.
    """
    node = db.get_node(node_id)
    if node is None:
        raise KeyError(f"node {node_id} not found")
    rendered = render_node(node_id)
    crop = image_crop(node["image_id"])
    if is_identity_crop(crop):
        return rendered

    out = output_path(node_id)
    if out.exists():
        return out
    im = Image.open(rendered).convert("RGB")
    apply_geometry(im, crop, Image.Resampling.BICUBIC).save(out, quality=JPEG_QUALITY)
    return out


def crop_preview(node_id: int, crop: dict | None) -> tuple[bytes, list[int]]:
    """The node's pixels rotated onto the expanded canvas but *not* framed —
    the backdrop the frame editor drags over, so the user can see what falls
    outside the frame and what the black corners will be.

    Downscaled first: the frame is stored as fractions of the canvas, so the
    proxy and the full-size output describe the same crop.

    Returns the JPEG *and* the canvas the full-size render would rotate onto,
    which is the one number the editor cannot work out for itself: the frame
    readout has to report the pixels that will actually export, and deriving
    them from the proxy's own size is how it came to lie about them. Measured
    from `im.size` before the downscale — the render this proxy stands for —
    rather than from the original, so nothing here assumes the two match.
    """
    im = Image.open(render_node(node_id)).convert("RGB")
    # rect ignored on purpose — the SVG overlay draws the frame, this is the canvas
    angle = float((crop or {}).get("angle") or 0.0)
    canvas = crop_geometry(im.size, {"angle": angle})["canvas"]
    im.thumbnail((CROP_PREVIEW_PX, CROP_PREVIEW_PX), Image.Resampling.BICUBIC)
    if angle:
        im = im.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=0)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), canvas


def delete_output_files(node_ids: list[int]) -> None:
    """Drop just the output stage's cache. Changing an image's framing calls
    this and nothing else: the effect renders upstream are unaffected by a crop,
    and they are the expensive ones — not re-running k-means to re-crop is the
    whole point of the crop being an output stage."""
    for node_id in node_ids:
        output_path(node_id).unlink(missing_ok=True)
        thumb_path(node_id).unlink(missing_ok=True)


def render_preview(
    node_id: int,
    effect: str,
    params: dict,
    parent2_id: int | None = None,
    selection: dict | None = None,
) -> bytes:
    """Render an effect applied to a node's image in memory — no node row, no
    cache file. Shows exactly what a child created by Apply would look like.

    Framed on the way out like every other viewer, so toggling live preview off
    and on never changes the framing under the user.
    """
    result = _apply(render_node(node_id), effect, params, parent2_id, selection, node_id)
    im = Image.fromarray(result)
    node = db.get_node(node_id)
    crop = image_crop(node["image_id"]) if node else None
    if not is_identity_crop(crop):
        im = apply_geometry(im, crop, Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def render_thumb(node_id: int) -> Path:
    out = thumb_path(node_id)
    if out.exists():
        return out
    # from the framed output, so a tree chip shows what the node exports
    im = Image.open(render_output(node_id)).convert("RGB")
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


# ---------- CLIP embeddings (the Image map) ----------


def embed_image(image_id: int, root_node_id: int) -> np.ndarray:
    """Compute and store an image's CLIP vector, returning it.

    Reads the *thumbnail*, not the original. That is the whole reason embedding
    a library is quick: `render_thumb` is already on disk for every image (the
    filmstrip drew it), it is already framed through `render_output`, and CLIP
    only looks at 224px anyway — so this costs one small decode plus one forward
    pass instead of decoding a median 10.7 MP original. The map then describes
    exactly what the filmstrip shows, framing included.

    The 320px thumb is upscaled ~5% on its short axis to reach CLIP's 224px
    crop, which is immaterial to a model whose patches are 32px.
    """
    img = np.asarray(Image.open(render_thumb(root_node_id)).convert("RGB"))
    vec = embed.compute_embedding(img)
    db.set_embedding(image_id, vec.tobytes())
    return vec


# ---------- Detected objects (the Image map's tag filter) ----------


def detect_image(image_id: int, root_node_id: int) -> list[dict]:
    """Detect an image's objects and store them, returning them.

    Reads the framed *output* rather than the thumbnail `embed_image` uses, and
    the difference is the whole reason detection is the slower of the two. CLIP
    looks at 224px and a 320px thumb already exceeds that; YOLO looks at 640 and
    a thumb would be an upscale, which costs exactly the small distant objects a
    detector is worth having for. So this decodes the original — but only down
    to `detect.DECODE_HINT`, since libjpeg can scale during the DCT and the
    frame is bound for a 640 square regardless.

    Framed, though, like the thumbnail is: the boxes have to describe the photo
    as it is shown, or a viewer would draw them over a crop they do not match.
    That is what `db.clear_detections` in the crop endpoint is for.
    """
    im = Image.open(render_output(root_node_id))
    # Before load(), which is when the hint takes effect — this is `mean_color`'s
    # trick, and here it is the difference between a 4-minute pass over the
    # library and a 25-minute one.
    im.draft("RGB", (detect.DECODE_HINT, detect.DECODE_HINT))
    found = detect.detect(np.asarray(im.convert("RGB")))
    db.set_detections(image_id, found)
    return found


# The point cloud paints each image in its own average color, which is what
# makes the cloud read as the library rather than as abstract dots.
MEAN_COLOR_SAMPLE = 32


def mean_color(root_node_id: int) -> list[int]:
    """An image's average RGB, from its thumbnail.

    Not cached anywhere, for `histogram`'s reason: `draft()` lets libjpeg scale
    during the DCT, so a 320px thumb decodes at ~40px and this costs less than
    the stat call a cache lookup would need. No new file kind, no new
    invalidation edge — and it re-derives from the thumb, whose own
    invalidation is already correct.
    """
    im = Image.open(render_thumb(root_node_id))
    im.draft("RGB", (MEAN_COLOR_SAMPLE, MEAN_COLOR_SAMPLE))
    arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    return [int(round(c)) for c in arr.reshape(-1, 3).mean(axis=0)]


# ---------- Click-to-segment (SAM) ----------

# How much of a node's render `locate_object` decodes. Generous rather than 352:
# libjpeg's DCT scaling only halves, so this is the smallest draft that leaves
# any frame at least CLIPSeg's own input size in both directions.
LOCATE_SAMPLE = 1024


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


# How many nodes' SAM 3 features to hold. Small on purpose — see below.
_SAM3_CACHE = 3
_sam3_features: "OrderedDict[int, dict]" = OrderedDict()


def sam3_features(node_id: int) -> dict[str, np.ndarray]:
    """SAM 3's image-encoder output for a node, cached **in process**.

    The encoder is the expensive third of `sam3.segment` (~2.6s of the ~4.5s a
    cold query costs) and depends only on the pixels, so holding its output
    makes a second question about the same photo ~0.4s instead. That much is
    `node_embedding`'s trade exactly.

    What is *not* like `node_embedding` is where it lives. MobileSAM's embedding
    is 4 MB and goes on disk beside the render; SAM 3's four tensors are **~117
    MB per node** — `fpn_0` alone is 256x288x288 float32 — so the same treatment
    would put tens of gigabytes under `data/renders/` for a library anyone
    actually used this on. That is not a cache, it is a leak with a sweep
    attached.

    So this follows `depth.depth_map` instead, which caches in process for the
    same reason and which CLAUDE.md already records as the precedent. The
    consequences are all things not to do elsewhere: **no file kind under
    `renders/`**, so `delete_render_files` and `storage_stats` need no clause,
    and nothing can be left stale on disk by a missed sweep. It is keyed by node
    id and a node's pixels never change — a PATCH creates a new render but
    `delete_render_files` is what invalidates it, so drop the entry there too
    rather than trusting the id to stay meaningful.

    Bounded at three because the bound is memory, not disk: three nodes is
    ~350 MB on top of the 3.4 GB of sessions, and the working set for "ask
    several things about the photo in front of me" is one.
    """
    from . import sam3  # imported here for `locate_object`'s reason, one size up

    cached = _sam3_features.get(node_id)
    if cached is not None:
        _sam3_features.move_to_end(node_id)
        return cached
    img = np.asarray(Image.open(render_node(node_id)).convert("RGB"))
    features = sam3.encode_image(img)
    _sam3_features[node_id] = features
    while len(_sam3_features) > _SAM3_CACHE:
        _sam3_features.popitem(last=False)
    return features


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


def locate_object(node_id: int, query: str) -> dict:
    """Turn a phrase into the click a person would have made on this node.

    The whole of "select by name", and deliberately nothing more: it returns a
    point and a level, which is a `{points: [...]}` selection's entire content.
    Everything after this — the overlay, banking into a saved mask, a preset
    replaying it — cannot tell that a sentence rather than a finger produced it.

    The decode is drafted down: `clipseg` squashes to 352 square whatever it is
    given, so a full 40 MP decode would be thrown away, and `histogram` reads
    its pixels the same way. `im.size` is read *before* the draft because the
    point has to come back in the node's own pixel space — the space the
    selection stores, and the space `sam` re-decodes it in.
    """
    from . import clipseg  # imported here so a broken/absent export costs only this

    with Image.open(render_node(node_id)) as im:
        w, h = im.size
        im.draft("RGB", (LOCATE_SAMPLE, LOCATE_SAMPLE))
        small = np.asarray(im.convert("RGB"))

    found = clipseg.locate(small, query)
    # draft() only ever *reduces*, and by a power of two, so this is a scale-up
    # by a small integer factor and never a stretch.
    sh, sw = small.shape[:2]
    x = min(w - 1, round(found["x"] * w / sw))
    y = min(h - 1, round(found["y"] * h / sh))

    # The blob was measured on the drafted pixels; the candidates are the node's.
    area = found["area"] * (w * h) / (sw * sh)
    level, pixels = sam.level_for_area(node_embedding(node_id), h, w, x, y, area)
    return {
        "x": x,
        "y": y,
        "level": level,
        "score": round(found["score"], 4),
        # what the selection will actually cover, not what CLIPSeg guessed
        "coverage": round(pixels / (w * h), 4),
        "confident": found["confident"],
    }


def locate_object_sam3(node_id: int, query: str) -> dict:
    """`locate_object`'s answer to the same question, from one model instead of
    two — and as *pixels* rather than as a click.

    That difference is the whole reason this is a second function and not a
    branch inside the first. CLIPSeg finds a place and MobileSAM draws the
    boundary there, so the answer compresses to `{x, y, level}` and re-decodes
    later; SAM 3 draws the boundary itself, and there is no point-and-level that
    would reproduce it. Handing this mask back through the `{points: [...]}`
    shape would re-derive a MobileSAM mask and throw away the one improvement
    the model is here for.

    So this returns the mask, and `main.select_text` freezes it into a saved
    mask — the shape that stores pixels. Nothing new is added to
    `effects.validate_selection`: there are still exactly two selection shapes,
    and a SAM 3 answer is the same `{masks: [id]}` a hand-saved selection is.

    Unlike `locate_object` this does **not** draft the decode down. CLIPSeg
    squashes to 352 and MobileSAM to 1024, so drafting there throws away nothing;
    here the mask comes back at the encoder's own resolution and every pixel of
    input resolution is a pixel of boundary accuracy.

    Returns `locate_object`'s keys plus `mask` and `instances`, so callers that
    only want the metadata can share code with the CLIPSeg path — `x` and `y`
    are the best instance's centroid, which is what lets `agent.measure_focus`
    take a depth reading from an answer that is otherwise mask-shaped.
    """
    from . import sam3  # imported here for `locate_object`'s reason, one size up

    with Image.open(render_node(node_id)) as im:
        w, h = im.size
        img = np.asarray(im.convert("RGB"))

    found = sam3.segment(sam3_features(node_id), sam3.encode_text(query), h, w)
    mask = found["mask"]
    if mask.shape != (h, w):
        # The encoder works at its own square resolution and the masks are
        # resized back out of it, so a mismatch here means the resize used the
        # drafted or the padded dimensions rather than the node's. Refuse it:
        # `compute_mask` unions by `|`, which would broadcast a wrong-shaped
        # mask into a plausible-looking one rather than failing.
        raise ValueError(
            f"SAM 3 mask is {mask.shape[1]}×{mask.shape[0]}, node is {w}×{h}"
        )
    # Coerced at the boundary, not trusted: these come off numpy arrays, and a
    # numpy int64 travels fine through every expression here and then fails to
    # serialize in the response — a 500 from the JSON encoder, a long way from
    # the line that produced it.
    return {
        "x": int(found["x"]),
        "y": int(found["y"]),
        # There is no level: the pixels are the answer. The key is absent rather
        # than None so a caller that reads it is a KeyError here instead of a
        # silently unlevelled click somewhere downstream.
        "score": round(float(found["score"]), 4),
        "coverage": round(int(mask.sum()) / (w * h), 4),
        "confident": bool(found["score"] >= sam3.MATCH_FLOOR),
        "instances": len(found["instances"]),
        "mask": mask,
    }


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
    """The overlay PNG for a selection on a node — either shape, framed to match
    what the preview is showing.

    The framing here is **display only**. The mask that actually gets composited
    in `_apply` is computed and used at node dimensions and is never warped —
    that is the point of keeping the crop out of the work tree. This one is
    warped because it is laid over the *framed* preview, and an unframed outline
    would sit off the object once the image is rotated.

    NEAREST, not bicubic: the outline is already antialiased into its casing, and
    interpolating an RGBA stencil smears its alpha into a halo.
    """
    png = _overlay_png(compute_mask(node_id, selection))
    node = db.get_node(node_id)
    crop = image_crop(node["image_id"]) if node else None
    if is_identity_crop(crop):
        return png
    with Image.open(io.BytesIO(png)) as im:
        framed = apply_geometry(im.convert("RGBA"), crop, Image.Resampling.NEAREST)
    buf = io.BytesIO()
    framed.save(buf, format="PNG")
    return buf.getvalue()


def cutout_image(node_id: int, selection: dict) -> Image.Image:
    """The selection cut out of the node's render: RGBA, transparent outside the
    selection and trimmed to its bounding box — a layer to drop into another
    editor, or into a collage, which freezes exactly this.

    A viewer, so it is framed like Export (`render_output`) and like the overlay
    — the selection you see outlined is exactly what comes out. The mask is
    framed the way `mask_png` frames it, NEAREST so the alpha stays the binary
    mask `_apply` composites with rather than a bicubic halo; a rotation's
    corners fill with 0 and so come out transparent.

    Persists nothing, so there is no file kind for `delete_render_files` or
    `storage_stats` to know about.
    """
    rgb = Image.open(render_output(node_id)).convert("RGB")
    alpha = Image.fromarray(compute_mask(node_id, selection).astype(np.uint8) * 255, "L")
    node = db.get_node(node_id)
    crop = image_crop(node["image_id"]) if node else None
    if not is_identity_crop(crop):
        alpha = apply_geometry(alpha, crop, Image.Resampling.NEAREST)
    if alpha.size != rgb.size:
        raise ValueError(
            f"mask is {alpha.size[0]}×{alpha.size[1]}, "
            f"image is {rgb.size[0]}×{rgb.size[1]}"
        )
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("selection is empty after framing")
    rgb.putalpha(alpha)
    return rgb.crop(bbox)


def cutout_png(node_id: int, selection: dict) -> bytes:
    """`cutout_image` as PNG bytes — the Export cut-out download."""
    buf = io.BytesIO()
    cutout_image(node_id, selection).save(buf, format="PNG")
    return buf.getvalue()


def cutout_path(piece_id: int) -> Path:
    return db.CUTOUTS_DIR / f"{piece_id}.png"


def save_cutout(piece_id: int, image: Image.Image) -> None:
    """Freeze a collage piece's pixels. Written beside and renamed into place, so
    a crash mid-write cannot leave a truncated PNG that the row then vouches for."""
    path = cutout_path(piece_id)
    tmp = path.with_suffix(".png.tmp")
    image.save(tmp, format="PNG")
    tmp.replace(path)


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
        output_path(node_id).unlink(missing_ok=True)
        thumb_path(node_id).unlink(missing_ok=True)
        clusters_path(node_id).unlink(missing_ok=True)
        embedding_path(node_id).unlink(missing_ok=True)
        # Not a file: SAM 3's features are held in memory (see `sam3_features`),
        # but they are keyed by node id just the same, so an edited node has to
        # forget them here or the next query answers from the old pixels.
        _sam3_features.pop(node_id, None)


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
    outputs = [p for p in renders if p.name.endswith(".out.jpg")]
    clusters = [p for p in renders if p.name.endswith(".clusters.json")]
    embeddings = [p for p in renders if p.name.endswith(".embedding.npy")]
    return {
        # glob, not DB_PATH.stat(), so a future WAL journal mode still adds up
        "database": _totals(db.DATA_DIR.glob("picky.db*")),
        "originals": _totals(db.ORIGINALS_DIR.iterdir()),
        "masks": _totals(db.MASKS_DIR.iterdir()),
        # collage pieces: frozen pixels, user data for a mask's reason
        "cutouts": _totals(db.CUTOUTS_DIR.iterdir()),
        # ONNX weights (MobileSAM, CLIP, depth, CLIPSeg) — neither user data nor
        # a cache: they re-download rather than rebuild. Reported because at the
        # better part of a gigabyte, once every feature that has a model behind
        # it has been used, they are among the largest things on disk and were
        # previously invisible here. Whatever is in the directory counts, so a
        # fifth model needs no line of its own.
        "models": _totals(
            sam.MODELS_DIR.iterdir() if sam.MODELS_DIR.is_dir() else []
        ),
        # thumbs and framed outputs are also .jpg, so exclude them by name rather
        # than matching on suffix — otherwise their bytes would be counted twice
        "renders": _totals(
            p
            for p in renders
            if p.suffix == ".jpg"
            and not p.name.endswith((".thumb.jpg", ".out.jpg"))
        ),
        "thumbs": _totals(thumbs),
        # the framed output stage — regenerable from the renders plus the crop
        "outputs": _totals(outputs),
        "clusters": _totals(clusters),
        # SAM image embeddings, ~4 MB apiece — big enough to deserve a line.
        # SAM 3's are not here on purpose: at ~117 MB a node they are held in
        # memory rather than written down (see `sam3_features`), so there is no
        # file kind to count.
        "embeddings": _totals(embeddings),
    }
