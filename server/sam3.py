"""SAM 3 (Segment Anything with Concepts) via onnxruntime: name a thing, get
every instance of it.

The other half of "select by name". `clipseg.py` + `sam.py` answer a phrase with
one *point*, which MobileSAM then turns into one boundary; this answers with
masks directly, and with **all** the instances rather than the best peak. Since a
picky selection is already a union, those OR together for free — "bird" on a
flock is one mask over 76 of them, which the CLIPSeg path cannot express at any
setting.

Three graphs, and which of them runs when is the whole performance story:

    vision encoder   454.0M   depends only on the pixels   1856 MB   ~2.6s
    language encoder 353.5M   depends only on the words    1413 MB   ~0.2s
    decoder           ~32M    depends on both                97 MB   ~0.4s

So the first question about a node costs ~4.5s and the second ~0.4s. Both
encoders are therefore cached — the vision one in `rendering.sam3_features`, the
text one here on the query string.

**This model cannot download itself, and it is the only one here that cannot.**
Meta publishes PyTorch weights, no ONNX, behind a gate a human approves by hand;
the graphs come from `tools/export_sam3_onnx.py`, run once. `ensure_model`
raises with that instruction rather than fetching, and `sam3_job.py` reports it
rather than showing a progress bar that can never move.

Licensing is not the usual answer either: these weights are Meta's **SAM
License**, not Apache or MIT. That makes two encumbered models in the tree
alongside `detect.py`'s AGPL-3.0 YOLO — a licence to read before this app is
distributed, not merely run.

**Everything below was measured against the export, not inferred from it.** The
constants, the padding token and the resize filter were each checked by running
the real model both ways and comparing; `self_check()` is that comparison kept
around, and the numbers it defends are in the comments at each one. This module
is the tree's most exposed to silent preprocessing error — a wrong constant
yields a mask that still looks like a mask — so nothing here is taken on faith.
"""

import functools
import os
from pathlib import Path

import numpy as np
from PIL import Image

from . import text_embed
from .sam import MODELS_DIR

# All read off `facebook/sam3`'s own `processor_config.json` / `config.json`.
#
# SAM 3 squashes to 1008 square, where SAM 1 pads to 1024 and CLIPSeg squashes
# to 352. Aspect is not preserved and that is upstream's choice, not ours.
INPUT_SIZE = 1008

# **The fifth normalization in this tree, and it is none of the other four.**
# Symmetric 0.5/0.5 over 0-1 floats — the SigLIP convention — where `depth.py`
# and `clipseg.py` use ImageNet's asymmetric constants over the same 0-1 range,
# `sam.py` uses ImageNet-scaled ones over 0-255, and `detect.py` uses no
# normalization beyond the rescale. Written out here rather than imported for
# CLAUDE.md's standing reason: two checkpoints agreeing on a constant is a
# coincidence, and sharing the array would make re-pinning either a silent
# change to the other.
_PIXEL_MEAN = np.float32(0.5)
_PIXEL_STD = np.float32(0.5)

# SAM 3's text tower is a `clip_text_model` over CLIP's own 49408-token
# vocabulary (bos 49406, eot 49407) — the exact vocabulary `text_embed` already
# owns, which is what makes borrowing its tokenizer correct rather than merely
# convenient. Verified: our ids are byte-identical to `Sam3Processor`'s. What
# differs from CLIP is the window — 32 positions, not 77.
CONTEXT_LENGTH = 32

# Padding is the **eot** id, not zero. `Sam3Processor` pads with `<|endoftext|>`
# and the attention mask covers only the real tokens, so a zero-padded stream is
# masked-out garbage rather than an error — it changed nothing measurable here,
# which is exactly why it is pinned rather than left to luck.
_PAD_ID = 49407

# Meta's own `score_threshold_detection`, not a number invented here. It does
# double duty: instances below it are dropped from the union, and a query whose
# best instance is below it is not confident at all. One constant rather than
# two because the measured separation is stark — a present subject's best
# instance scores 0.93-0.96, and an absent one yields *no* instances whatsoever
# ("bird" on a photo of people returned zero, not a weak guess). That is a
# sharper signal than `clipseg.MATCH_FLOOR` gets to work with, and it is why
# there is no "weak match" band here worth naming.
MATCH_FLOOR = 0.5

# The decoder emits a fixed 200 candidate slots at 288 square
# (`low_res_mask_size`), thresholded and upsampled to the frame afterwards.
_NUM_QUERIES = 200
LOW_RES_MASK = 288

_MODELS = {
    # Each graph and its env override. No `external` flag, deliberately: the
    # weights *may* live in an `.onnx.data` sidecar and may not, and which one
    # happens is not ours to declare. Protobuf caps a message at 2 GB, and these
    # land at 1.7 GB and 1.3 GB — under it, so torch inlines them — but a
    # re-export against a different checkpoint or opset could cross the line and
    # start emitting sidecars. `ready()` therefore asks the filesystem what the
    # layout is rather than asserting one, because asserting the wrong one fails
    # in the direction that looks like "not exported" for files that are there.
    "image": {
        "env": "PICKY_SAM3_IMAGE_ENCODER",
        "filename": "sam3_image_encoder.onnx",
    },
    "language": {
        "env": "PICKY_SAM3_LANGUAGE_ENCODER",
        "filename": "sam3_language_encoder.onnx",
    },
    "decoder": {
        "env": "PICKY_SAM3_DECODER",
        "filename": "sam3_decoder.onnx",
    },
}

EXPORT_HINT = (
    "SAM 3's ONNX graphs are not in data/models/. Meta publishes no ONNX export, "
    "so they are built once by hand: request access at "
    "https://huggingface.co/facebook/sam3, put the token in PICKY_HF_TOKEN, "
    "then run `.venv/bin/python tools/export_sam3_onnx.py`."
)


def model_path(kind: str) -> Path:
    """Where one graph is, whether or not it exists.

    `sam._model_path` downloads on a miss; this cannot, so it is the shape of
    `clipseg.model_path` instead — a location, and `ready()` asks the question.
    """
    spec = _MODELS[kind]
    override = os.environ.get(spec["env"])
    return Path(override) if override else MODELS_DIR / spec["filename"]


def model_paths() -> dict[str, Path]:
    return {kind: model_path(kind) for kind in _MODELS}


def fixture_path() -> Path:
    return MODELS_DIR / "sam3_fixture.npz"


def ready() -> bool:
    """All three graphs on disk, with their weights, whichever layout they use.

    A sidecar is required only if one is *there*: an `.onnx` whose `.onnx.data`
    has gone missing loads happily and fails at the first `run()`, which is the
    shape of bug file-existence caching exists to rule out — but demanding a
    sidecar that was never written reports "not exported" for a working export,
    which is the same bug pointing the other way.

    Unlike `clipseg.ready()` this does not also require the tokenizer: those two
    files are 1.7 MB and `ensure_model` can fetch them, where the graphs are the
    thing that needs a person.
    """
    for kind in _MODELS:
        path = model_path(kind)
        if not path.is_file() or path.stat().st_size == 0:
            return False
        sidecar = path.with_suffix(".onnx.data")
        if sidecar.exists() and sidecar.stat().st_size == 0:
            return False
    return True


def ensure_model(on_progress=None) -> None:
    """Raise unless the graphs are here; fetch only the tokenizer. **Never
    downloads a graph.**

    Every other `ensure_model` in the tree fetches everything it needs. This one
    is the odd module out, and the asymmetry is upstream's, not a choice: there
    is no URL to fetch, gated or otherwise, because Meta ships no ONNX.
    `on_progress` is honoured for the tokenizer and then ignored, so `sam3_job`
    can call this exactly like `clipseg_job` calls its sibling — a job module
    that had to special-case one model would be the convention leaking.
    """
    if not ready():
        raise FileNotFoundError(EXPORT_HINT)
    text_embed.ensure_tokenizer(on_progress=on_progress)


@functools.lru_cache(maxsize=None)
def _session(kind: str):
    # onnxruntime imported lazily, like every other model module here, so the
    # server starts and everything unrelated to segmentation works even if the
    # wheel is missing or broken on this platform.
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    return onnxruntime.InferenceSession(
        str(model_path(kind)), options, providers=["CPUExecutionProvider"]
    )


def _preprocess(img: np.ndarray) -> np.ndarray:
    """RGB uint8 -> NCHW float32 at 1008 square, normalized.

    **Plain PIL bilinear on the uint8 pixels, and that choice is measured.**
    `Sam3Processor` resizes with torchvision's antialiased bilinear, which this
    cannot call — torch is a build-time dependency and must not enter the
    server. Four candidates were run end to end against the processor's own
    tensor and scored on the union mask, which is the thing picky actually uses:

        pil bilinear (uint8)   max px diff 0.008   union IoU 0.994-0.999
        pil bilinear (float)   max px diff 0.008   union IoU 0.986-0.991
        pil bicubic            max px diff 0.204   union IoU 0.963-0.969
        pil lanczos            max px diff 0.322   union IoU 0.953-0.969

    So the cheapest and least clever option is also the most faithful, and the
    two "higher quality" filters are markedly worse — they sharpen, and this
    model is sensitive enough that a max deviation of 1/255 can flip a
    borderline instance. Do not "improve" this to bicubic.
    """
    im = Image.fromarray(img).resize(
        (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR
    )
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - _PIXEL_MEAN) / _PIXEL_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def _tokens(query: str) -> tuple[np.ndarray, np.ndarray]:
    """`query` as CLIP token ids and an attention mask, padded to 32.

    Reuses `text_embed`'s tokenizer — the only one in the tree, and deliberately
    separable from its 254 MB tower for exactly this kind of borrowing (see
    `clipseg.py`, which needs the vocabulary and none of the graph). A second
    tokenizer would be a second copy of CLIP's merge table on disk and a second
    module owning what a phrase means.

    Truncation keeps the eot id last, as `text_embed` does for its own window:
    the text tower reads its pooled vector at that token, so a phrase clipped
    without one is pooled at whatever happened to land in the final slot.
    """
    ids = list(text_embed.tokenizer().encode(query))
    if len(ids) > CONTEXT_LENGTH:
        ids = ids[: CONTEXT_LENGTH - 1] + [_PAD_ID]
    out = np.full((1, CONTEXT_LENGTH), _PAD_ID, dtype=np.int64)
    mask = np.zeros((1, CONTEXT_LENGTH), dtype=np.int64)
    out[0, : len(ids)] = ids
    mask[0, : len(ids)] = 1
    return out, mask


def encode_image(img: np.ndarray) -> dict[str, np.ndarray]:
    """Run the image encoder over an RGB uint8 array. The expensive third.

    Keeps only the tensors the decoder actually consumes. The encoder emits four
    FPN levels and four position encodings, but `Sam3Model.forward` slices
    `[:-1]` off the features and passes only the *last* position encoding on, so
    the exported decoder takes `fpn_0`, `fpn_1`, `fpn_2` and `pos_2` and nothing
    else. Carrying the other four would double the cache for tensors no graph
    has an input for.
    """
    session = _session("image")
    outputs = session.run(None, {"pixel_values": _preprocess(img)})
    named = dict(zip([o.name for o in session.get_outputs()], outputs))
    wanted = [i.name for i in _session("decoder").get_inputs() if i.name in named]
    return {name: named[name] for name in wanted}


@functools.lru_cache(maxsize=32)
def _encode_text_cached(query: str) -> tuple[np.ndarray, np.ndarray]:
    ids, mask = _tokens(query)
    pooled = _session("language").run(
        None, {"input_ids": ids, "attention_mask": mask}
    )[0]
    return pooled, mask


def encode_text(query: str) -> tuple[np.ndarray, np.ndarray]:
    """Run the language encoder over one phrase; returns `(pooled, mask)`.

    Cached on the string because that is the entire input: asking two nodes
    about "the dog" runs this once. Bounded rather than unbounded — the pooled
    tensor is small, but the cache would otherwise grow a slot per phrase for
    the life of the process.
    """
    return _encode_text_cached(query)


def segment(vision: dict, language: tuple, height: int, width: int) -> dict:
    """Every instance of the phrase, and the union of them.

    Returns::

        {"mask": bool (height, width),   # union of instances at/above MATCH_FLOOR
         "instances": [{"score": float, "box": [x0, y0, x1, y1]}, ...],
         "score": float,                 # the best instance's
         "x": int, "y": int}             # centroid of the best instance's mask

    The union is the feature. The `(x, y)` is not decoration either: it is what
    lets `agent.measure_focus` take a depth reading from a mask-shaped answer.

    **The union is built at 288 square and upsampled once, not per instance.**
    Going through `Sam3Processor.post_process_instance_segmentation` at full
    size materialises `(instances, height, width)` int64 — 76 birds on a 24 MP
    frame is about 15 GB and four and a half seconds — to produce a union that
    is one boolean plane. Reducing at the decoder's own resolution costs a
    thousandth of that, and the only thing lost is per-instance boundaries at
    full size, which nothing here wants.

    **The reduction is a max over logits, and the threshold comes after the
    upsample.** Those are one decision, not two. `max` rather than `any` because
    a union of thresholds is exactly a threshold of the maximum
    (`∃i: Lᵢ > 0` iff `maxᵢ Lᵢ > 0`, checked), so this leaves a *float* plane
    where the obvious version leaves a boolean one — and a float plane can be
    interpolated. Upsampling booleans is what a nearest-neighbour resize is for,
    and at the ~21x this needs it turns every edge into a staircase of 21-pixel
    blocks. Interpolating the logits and thresholding at full size instead puts
    the boundary sub-pixel, which is exactly what `sam._upscale` does for
    MobileSAM (`_upscale(...) > 0.0`) and the reason the two backends now agree
    about what an edge looks like.

    It costs about 50 ms and roughly 6% of the covered area: interpolation pulls
    a thin structure's logits toward zero, so the smallest instances — a few
    pixels across at 288, sub-pixel in the frame — erode. That is the right
    trade and deliberately not compensated with a threshold nudge; matching
    MobileSAM's arithmetic is worth more than recovering blobs whose boundary
    was never meaningful at this resolution.
    """
    pooled, mask = language
    session = _session("decoder")
    feed = {name: vision[name] for name in vision}
    feed["text_pooled"] = pooled
    feed["attention_mask"] = mask
    masks, boxes, logits, _presence = session.run(None, feed)

    scores = 1.0 / (1.0 + np.exp(-logits[0].astype(np.float32)))
    keep = np.flatnonzero(scores >= MATCH_FLOOR)
    if keep.size == 0:
        return {
            "mask": np.zeros((height, width), dtype=bool),
            "instances": [],
            "score": float(scores.max()) if scores.size else 0.0,
            "x": width // 2,
            "y": height // 2,
        }

    # Mask logits, so the boundary is where they cross zero — and it is crossed
    # at full size, not at 288 (see above).
    plane = masks[0][keep].max(axis=0).astype(np.float32)
    union = (
        np.asarray(
            Image.fromarray(plane, mode="F").resize(
                (width, height), Image.Resampling.BILINEAR
            )
        )
        > 0.0
    )

    best = int(keep[np.argmax(scores[keep])])
    x, y = _centroid(masks[0][best] > 0.0, width, height)
    order = np.argsort(-scores[keep])
    return {
        "mask": union,
        "instances": [
            {
                "score": round(float(scores[keep[i]]), 4),
                # cx, cy, w, h in 0..1 -> x0, y0, x1, y1, which is `detect.py`'s
                # convention and so the one anything downstream already reads.
                "box": _corners(boxes[0][keep[i]]),
            }
            for i in order
        ],
        "score": float(scores[best]),
        "x": x,
        "y": y,
    }


def _centroid(low: np.ndarray, width: int, height: int) -> tuple[int, int]:
    """A representative point inside one low-res instance mask, in frame pixels.

    The mean of the set pixels, which for a concave subject can fall outside it
    — a point on a horseshoe's axis is in the gap. That is tolerable here
    because the only reader is a depth sample, and depth varies smoothly across
    a neighbourhood; it would not be tolerable if this were seeding a click.
    """
    ys, xs = np.nonzero(low)
    if xs.size == 0:
        return width // 2, height // 2
    rows, cols = low.shape
    x = int(min(width - 1, (xs.mean() + 0.5) * width / cols))
    y = int(min(height - 1, (ys.mean() + 0.5) * height / rows))
    return x, y


def _corners(box: np.ndarray) -> list[float]:
    """One predicted box as `[x0, y0, x1, y1]` fractions — `detect.py`'s
    convention, so anything downstream already knows how to read it.

    **The decoder already emits corners.** DETR's own parameterisation is
    `(cx, cy, w, h)` and this model is a DETR, so converting looked obviously
    right; transformers converts for you before `pred_boxes` leaves the graph,
    and converting a second time yields boxes that are wrong without being
    absurd — plausible fractions, roughly the right part of the frame, many
    pinned to 1.0 by the clamp that was hiding the overflow. Checked against
    each instance's own mask bounding box across 76 of them: read as corners the
    mean max error is 0.008, read as centre-and-size it is 0.310.

    Hence the assertion rather than a comment. It is `detect._boxes`' tripwire
    one model along: a re-export that changed the convention would otherwise
    reintroduce exactly the bug this docstring is about, and no output would
    look wrong enough to notice.
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    if x1 < x0 or y1 < y0:
        raise RuntimeError(
            "SAM 3 decoder emitted a box whose corners are inverted; this "
            "export appears to use centre-and-size, not the corner convention "
            "this decoder expects"
        )
    return [
        round(max(0.0, x0), 5),
        round(max(0.0, y0), 5),
        round(min(1.0, x1), 5),
        round(min(1.0, y1), 5),
    ]


def self_check(tol: float = 0.05) -> dict:
    """Prove the runtime reproduces the export, on the fixture it wrote.

    The one test that catches a wrong constant. Preprocessing errors here do not
    raise and do not look wrong — a mask still comes back with a plausible
    boundary — so the only way to know the resize filter, the normalization and
    the pad token are still right is to run the pixels the exporter saw and
    compare against what PyTorch said about them.

    Checks both halves separately, because they fail for different reasons: our
    *pixels* against the processor's, and our logits against PyTorch's given the
    processor's own pixels. The first drifting means preprocessing changed; the
    second means the graph did.
    """
    if not ready():
        return {"ok": False, "reason": "not exported"}
    if not fixture_path().is_file():
        return {"ok": False, "reason": "no fixture — re-run tools/export_sam3_onnx.py"}

    from . import rendering

    data = np.load(fixture_path(), allow_pickle=False)
    source = rendering.db.ORIGINALS_DIR / str(data["source"])
    query = str(data["query"])
    result: dict = {"source": source.name, "query": query}

    ids, mask = _tokens(query)
    result["tokens_match"] = bool(np.array_equal(ids, data["input_ids"]))

    if source.is_file():
        img = np.asarray(Image.open(source).convert("RGB"))
        result["pixel_max_diff"] = float(
            np.abs(_preprocess(img) - data["pixel_values"]).max()
        )

    session = _session("image")
    outputs = session.run(None, {"pixel_values": data["pixel_values"]})
    named = dict(zip([o.name for o in session.get_outputs()], outputs))
    decoder = _session("decoder")
    feed = {i.name: named[i.name] for i in decoder.get_inputs() if i.name in named}
    pooled = _session("language").run(
        None, {"input_ids": data["input_ids"], "attention_mask": data["attention_mask"]}
    )[0]
    feed["text_pooled"] = pooled
    feed["attention_mask"] = data["attention_mask"]
    _masks, boxes, logits, _presence = decoder.run(None, feed)

    result["logit_max_diff"] = float(np.abs(logits - data["pred_logits"]).max())
    result["box_max_diff"] = float(np.abs(boxes - data["pred_boxes"]).max())
    ours = int((logits[0] >= 0).sum())
    theirs = int((data["pred_logits"][0] >= 0).sum())
    result["instances"] = {"onnx": ours, "torch": theirs}
    result["ok"] = (
        result["tokens_match"]
        and result["logit_max_diff"] < tol
        and ours == theirs
    )
    return result
