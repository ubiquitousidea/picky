"""Monocular depth via onnxruntime: one photo in, a relative depth map out.

Backs the `bokeh` effect, whose whole point is that it needs no selection. A
segmented subject blurred against a sharp silhouette has a hard edge no lens
produces; a depth map gives a *continuous* distance per pixel, so the blur can
grow with distance the way a real circle of confusion does.

Weights are Depth Anything V2 Small as a single-input ONNX export. Sources and
their precedence are `sam.py`'s exactly, and the download itself is literally
`sam.download_model` — see that module's docstring for why the revision is
pinned to a commit hash and why the write is atomic. This is the third ONNX
model in the tree and, more to the point, **the third distinct preprocessing**:
SAM normalizes 0-255 pixels with ImageNet-ish constants, CLIP normalizes 0-1
floats with CLIP's own, and this normalizes 0-1 floats with plain ImageNet's.
Feeding any of the three the wrong constants yields a depth map that still
looks like a depth map — smooth, plausible, wrong where it matters.

Two properties of the output that everything downstream depends on:

- **It is relative *inverse* depth (disparity): larger means nearer.** Scale and
  shift are arbitrary and vary per image, so it is meaningless until normalized;
  `depth_map` does that here rather than leaving it to callers, since a caller
  that forgot would silently get a plausible-looking wrong answer.
- **Disparity is the right space to stay in.** A lens's circle of confusion is
  proportional to |1/z_focus - 1/z|, i.e. it is *linear in disparity*. So a
  straight ramp on this model's own output is what an actual lens does, and
  bokeh's `falloff` param is an artistic override rather than a correction for
  working in the wrong units. Converting to metric depth would be both
  impossible (no scale) and wrong.

The map comes back at the model's own resolution, not the image's. Depth is
inherently low-frequency and the caller knows what size it wants; returning the
small array is also what makes the in-process cache cheap enough to hold.
"""

import functools
import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

# MODELS_DIR and the downloader are sam.py's, for embed.py's reason: that
# module owns the conventions for on-disk model weights.
from .sam import MODELS_DIR, download_model

# onnx-community/depth-anything-v2-small at revision
# 4472b7362082ad9968fee890ca0f1e5aca36b93d
_HF_URL = (
    "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/"
    "4472b7362082ad9968fee890ca0f1e5aca36b93d/onnx/model.onnx"
)
_ENV_VAR = "PICKY_DEPTH_MODEL"
_FILENAME = "depth_anything_v2_small.onnx"

# DPT's preprocessing, from the export's own preprocessor_config.json: resize so
# the *shortest* side is 518 with the aspect kept, both sides rounded to a
# multiple of 14 (the ViT patch size — the graph's H and W are dynamic, but only
# over multiples of the patch grid).
INPUT_SIZE = 518
PATCH = 14
# Plain ImageNet, over 0-1 floats. NOT sam._PIXEL_MEAN/_STD (0-255) and NOT
# embed._PIXEL_MEAN/_STD (CLIP's own).
_PIXEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_PIXEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Percentiles, not min/max: one blown-out specular highlight or one sensor-noise
# outlier at the far end would otherwise compress the entire scene into a
# fraction of the 0-1 range, and with it the entire blur ramp.
_CLIP_LO, _CLIP_HI = 1.0, 99.0


def model_path() -> Path:
    """Where the weights live — env override, else `data/models/`.

    The file need not exist yet: `.is_file()` on this is exactly the question
    "would applying bokeh download 99 MB?", which `depth_job.start()` and
    `main._check_effect_ready` both ask without touching the network.
    """
    override = os.environ.get(_ENV_VAR)
    return Path(override) if override else MODELS_DIR / _FILENAME


def ready() -> bool:
    """Is a bokeh render free right now, or does it cost a 99 MB download?"""
    return model_path().is_file()


def ensure_model(on_progress=None) -> Path:
    """The weights, downloading them on first use.

    Public and separate from `_session` for `embed.ensure_model`'s reason: the
    prepare job does the slow part up front and reports it. Unlike the Image
    map, though, nothing here falls back to doing it inline — `depth_job` is the
    only path to readiness, because a 99 MB fetch inside a render request is
    exactly the minute-of-nothing this arrangement exists to avoid.
    """
    path = model_path()
    if path.is_file():
        return path
    if os.environ.get(_ENV_VAR):
        # An explicit override that points nowhere is a misconfiguration, not an
        # invitation to fetch our own copy over the top of it.
        raise FileNotFoundError(f"{_ENV_VAR} points at missing file {path}")
    download_model(_HF_URL, path, on_progress=on_progress)
    return path


@functools.lru_cache(maxsize=None)
def _session():
    # onnxruntime imported lazily, like sam._session, embed._session and sklearn
    # in effects._fit_kmeans: everything unrelated to bokeh must keep working if
    # the wheel is missing or broken on this platform.
    import onnxruntime

    return onnxruntime.InferenceSession(
        str(ensure_model()), providers=["CPUExecutionProvider"]
    )


def _target_size(w: int, h: int) -> tuple[int, int]:
    """DPT's `keep_aspect_ratio` resize, reproduced.

    It computes a scale for each axis against 518, keeps whichever is *closer
    to 1*, and applies that one to both — which for any downscale is the larger
    scale, so the shortest side lands on 518 and the longest overshoots it.
    Both results are then snapped to the nearest multiple of 14.

    Reproduced rather than approximated for `effects.crop_geometry`'s reason:
    the model was trained through this exact function, and "resize the short
    side to 518" is only the same thing until the rounding disagrees.
    """
    scale = max(INPUT_SIZE / w, INPUT_SIZE / h)
    snap = lambda v: max(PATCH, int(round(v * scale / PATCH)) * PATCH)
    return snap(w), snap(h)


def _preprocess(img: np.ndarray) -> np.ndarray:
    im = Image.fromarray(img)
    tw, th = _target_size(*im.size)
    # A straight BICUBIC from 40 MP to ~0.4 MP resamples every source pixel
    # through the filter; reducing by an integer factor first is a box average
    # that costs a fraction of it and leaves BICUBIC a nearly-right-sized image
    # to finish on. The factor is kept conservative (>= 2x headroom) so the
    # box step never does the shaping.
    factor = min(im.size[0] // (tw * 2), im.size[1] // (th * 2))
    if factor > 1:
        im = im.reduce(factor)
    im = im.resize((tw, th), Image.Resampling.BICUBIC)
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - _PIXEL_MEAN) / _PIXEL_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def _infer(img: np.ndarray) -> np.ndarray:
    """Run the model and normalize its output to 0-1, 1 = nearest."""
    session = _session()
    x = _preprocess(img)
    out = session.run(None, {session.get_inputs()[0].name: x})[0]
    # (1, h, w) — the head predicts at the input resolution, so there is no
    # upsampling to undo here.
    depth = np.asarray(out, dtype=np.float32).reshape(out.shape[-2], out.shape[-1])
    lo, hi = np.percentile(depth, [_CLIP_LO, _CLIP_HI])
    if hi <= lo:
        # A perfectly flat prediction (a blank frame, a solid colour) has no
        # near and no far. Everything at the focal plane is the honest answer,
        # and it keeps the division below from producing inf.
        return np.ones_like(depth)
    return np.clip((depth - lo) / (hi - lo), 0.0, 1.0)


# Hand-rolled rather than `functools.lru_cache`d, because the thing to key on is
# a numpy array, which is unhashable — the digest is computed here and the cache
# keyed on that. Four maps of ~0.4 MP is well under a megabyte however large the
# images are. No lock: these are sync endpoints sharing a threadpool, dict
# operations are atomic under the GIL, and the worst a race can do is compute
# the same deterministic map twice.
_CACHE_MAX = 4
_cache: "OrderedDict[bytes, np.ndarray]" = OrderedDict()


def depth_map(img: np.ndarray) -> np.ndarray:
    """Normalized relative depth for an RGB uint8 array — float32 0-1, 1 = near.

    Cached in-process on the *content* of the pixels, which is not the obvious
    choice and is deliberate. Every other derived-per-node artifact in this app
    (`renders/<id>.embedding.npy`, cluster JSON) is cached on disk under a node
    id — but a registry effect's `apply(img, params)` receives no node id, and
    growing one so this could have a file would trade "a new effect is one entry
    and nothing else" for a cache. A content hash buys the thing that actually
    matters anyway: dragging Focus or Amount re-previews the same source pixels,
    so the model runs once for a whole editing session on one node.
    """
    key = hashlib.blake2b(img.tobytes()).digest()
    hit = _cache.get(key)
    if hit is not None:
        _cache.move_to_end(key)
        return hit
    depth = _infer(img)
    _cache[key] = depth
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    return depth


# A click is aimed at an object, not at a pixel, and the pixel it actually lands
# on may be an edge — where a depth model interpolates between the near thing
# and the far thing behind it, and reports a plane nothing is on. A median over
# a small window answers the question that was asked.
_SAMPLE_FRACTION = 0.01


def depth_at(img: np.ndarray, x: int, y: int) -> float:
    """Normalized depth under one pixel of `img`, on `depth_map`'s scale.

    The scale is the whole point: this exists to fill in bokeh's `focus`, so it
    has to be the same normalization the render will use, which is why it goes
    through `depth_map` rather than inferring anything smaller. Coordinates are
    in the image's own pixel space and are clamped, so a click on the last row
    is a click on the last row rather than an error.
    """
    d = depth_map(img)
    dh, dw = d.shape
    h, w = img.shape[:2]
    cx = min(dw - 1, max(0, round(x * dw / w)))
    cy = min(dh - 1, max(0, round(y * dh / h)))
    r = max(1, round(_SAMPLE_FRACTION * max(dh, dw)))
    patch = d[max(0, cy - r) : cy + r + 1, max(0, cx - r) : cx + r + 1]
    return round(float(np.median(patch)), 4)
