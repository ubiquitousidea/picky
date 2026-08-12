"""Text-prompted segmentation via onnxruntime: a phrase in, a heatmap out.

Backs "name an object" — the Select control's text box and the agent's
`select_object`. SAM segments what you *point at*; nothing in the tree could
segment what you *name*, because SAM's decoder here takes a click and nothing
else. CLIPSeg closes that gap: CLIP's joint space already puts a phrase and a
photo in one geometry, and CLIPSeg hangs a small decoder off a frozen CLIP
vision tower that turns the pairing into a per-pixel score.

**What comes out of here is a location, not a selection.** The map is 352x352
and blobby — it would make a soft, low-resolution mask with none of the edges
the app's masks are expected to have. So `locate` reduces it to the one thing
it is reliable about, the *place* the phrase is talking about, and hands that to
SAM as if someone had clicked there. That is what makes a typed selection and a
clicked one the same object downstream: both are `{points: [{x, y, level}]}`,
both re-segment through `sam.decode_mask`, both bank into a saved mask the same
way. Nothing new is stored, and no viewer ever sees a CLIPSeg pixel.

Weights are CLIPSeg rd64-refined as a single ONNX graph. Sources and their
precedence are `sam.py`'s exactly, and the download itself is literally
`sam.download_model`. Two things about it are worth knowing:

- **The quantized export is the one pinned, deliberately.** It is 139 MB against
  the float graph's 545 MB, and quantization error would have to move the
  heatmap's *peak off the object* before it cost anything here — the boundary is
  SAM's work, not this model's. Measured over this library the peak does not
  move: real subjects score 0.36-0.93 and absent ones 0.00-0.21, which is the
  same separation the float graph gives. `PICKY_CLIPSEG_MODEL` takes the float
  one for anybody who disagrees.
- **The normalization is plain ImageNet over 0-1 floats** — numerically the same
  constants `depth.py` uses, and deliberately not imported from it. Two
  checkpoints agreeing on their preprocessing is a coincidence, not a contract;
  sharing the array would make a future re-pin of either one a silent change to
  the other. This is the fourth ONNX model in the tree and still the *third*
  distinct preprocessing — see `depth.py` for the other two, and for what
  crossing them looks like (output that stays plausible and stops being right).

The resize is a **squash to 352x352, not `embed.py`'s shortest-side-then-crop**.
A center crop would throw away the sides of the frame, and an object the person
just named by hand is exactly as likely to be there as anywhere else.
"""

import functools
import os
from pathlib import Path

import numpy as np
from PIL import Image

# MODELS_DIR and the downloader are sam.py's, for embed.py's reason: that
# module owns the conventions for on-disk model weights.
from .sam import MODELS_DIR, download_model
from . import text_embed

# Xenova/clipseg-rd64-refined at revision 924dc94f85f58739f353f94258b33bc47eae4862
_HF_URL = (
    "https://huggingface.co/Xenova/clipseg-rd64-refined/resolve/"
    "924dc94f85f58739f353f94258b33bc47eae4862/onnx/model_quantized.onnx"
)
_ENV_VAR = "PICKY_CLIPSEG_MODEL"
_FILENAME = "clipseg_rd64_refined.onnx"

# The decoder's output grid, and the input it is tied to.
INPUT_SIZE = 352

# Phrasings the query is tried under, all in one batch. CLIPSeg is sensitive to
# wording in a way that has nothing to do with the picture: on one library photo
# "a person" scored 0.30 and "the person" 0.19 against the same subject, while
# "the hair" and "the black shirt" scored over 0.89 in the same frame. Someone
# typing a name for a thing they can see should not have to guess the article.
#
# This is `text_embed.TEMPLATES`' trick and half its list, but reduced
# differently — see `heatmap`, where the difference is the whole point.
TEMPLATES = [
    "{}",
    "a photo of {}",
    "a photo of a {}",
    "a close-up photo of {}",
]
# Plain ImageNet, over 0-1 floats — see the module docstring on why this is not
# `depth._PIXEL_MEAN` even though the numbers match.
_PIXEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_PIXEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Smoothing radius, in cells of the 352-grid, applied before the argmax. A raw
# argmax lands on whichever single cell won by noise, which on a broad subject
# can be a fringe cell straddling the edge — and a point one pixel outside the
# object gives SAM a completely different object. Averaging first moves the
# answer to the middle of the hottest *region*, which is where a person clicks.
_SMOOTH_R = 3

# Where the peak stops meaning "that thing is in this photo". A heatmap always
# has a maximum: ask for a submarine and some cell still wins, so this is the
# same problem `agent.STRONG_MATCH` solves for library search, with the same
# answer. Measured over this library at this export, through `TEMPLATES`:
# subjects that are really there score 0.36-0.99, ones that are not (submarine,
# dog, giraffe, moon, laptop, over photos holding none of them) reach 0.21 at
# the very most. The gap is wide enough that the exact number between them
# barely matters.
#
# A floor on *confidence*, not on results. `locate` returns the point either
# way, because the person can see the outline the moment it lands and "no match"
# is the wrong answer to a subject that is there but oddly worded. What the flag
# changes is that the agent, which cannot see anything, is told not to edit on it.
MATCH_FLOOR = 0.30

# Where the blob ends, as a fraction of its own peak. Only used to *measure* the
# subject, so SAM can pick the candidate mask closest to that size; the mask
# itself never comes from these pixels.
_BLOB_OF_PEAK = 0.5


def model_path() -> Path:
    """Where the weights live — env override, else `data/models/`.

    The file need not exist yet: `.is_file()` on this is exactly the question
    "would naming an object download 139 MB?", which `clipseg_job.start()` and
    `main.select_text` both ask without touching the network.
    """
    override = os.environ.get(_ENV_VAR)
    return Path(override) if override else MODELS_DIR / _FILENAME


def ready() -> bool:
    """Is naming an object free right now, or does it cost a download?

    Two files, since the tokenizer is CLIP's and lives beside the text tower's
    vocabulary — `text_embed.ensure_tokenizer` fetches those without the 254 MB
    graph, so a library that has never searched still only pays for this model.
    """
    return model_path().is_file() and text_embed.tokenizer_ready()


def ensure_model(on_progress=None) -> Path:
    """The weights, downloading them on first use.

    Public and separate from `_session` for `embed.ensure_model`'s reason: the
    prepare job does the slow part up front and reports it. Like `depth.py` and
    unlike the Image map, nothing falls back to doing it inline — `clipseg_job`
    is the only path to readiness, because a 139 MB fetch inside the request
    that a person is watching a text box for is the wait this arrangement exists
    to avoid.

    The tokenizer's two files come first and flash past, so the bar belongs to
    the graph — `text_embed._FILES` is ordered for the same reason.
    """
    text_embed.ensure_tokenizer(on_progress=on_progress)
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
    # onnxruntime imported lazily, like sam._session and embed._session: the
    # server must start, and everything unrelated to naming an object must keep
    # working, if the wheel is missing or broken on this platform.
    import onnxruntime

    options = onnxruntime.SessionOptions()
    # Errors only. This export declares `logits` as (num_queries, 352, 352) and
    # then squeezes the batch axis away for a single prompt, which is legal and
    # which onnxruntime warns about on *every* run — a line of server log per
    # keystroke-and-Enter, describing something `heatmap`'s reshape already
    # absorbs on purpose.
    options.log_severity_level = 3
    return onnxruntime.InferenceSession(
        str(ensure_model()), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _preprocess(img: np.ndarray) -> np.ndarray:
    """RGB uint8 -> NCHW float32 at 352x352, squashed and ImageNet-normalized."""
    im = Image.fromarray(img).resize(
        (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR
    )
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - _PIXEL_MEAN) / _PIXEL_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def heatmap(img: np.ndarray, query: str) -> np.ndarray:
    """Score every part of an RGB uint8 array against a phrase, 0-1.

    One session run for the whole of `TEMPLATES` — the graph takes a batch of
    prompts against a single image, and the four together cost what one costs
    plus a fifth of a second, so there is nothing to save by asking for fewer.

    **The templates are reduced by a pixelwise max, where
    `text_embed.encode_terms` averages.** They are not the same operation on the
    same kind of thing: there, each prompt is a unit vector in one space and
    their mean *is* the concept, phrasing averaged away. Here each prompt is an
    independent per-pixel probability, and a phrasing that simply fails
    contributes zeros — the mean lets it vote the subject down, which is exactly
    the failure this exists to fix (it left "the person" at 0.26, under the
    floor, on a photo full of people). The max asks whether *any* phrasing found
    the thing, and it does not invent matches, because a subject that is not in
    the picture is missed by all four: measured over this library, absent
    subjects reach 0.21 at the very most and present ones start around 0.36.

    Inputs are fed and the output taken by introspection rather than by index,
    for the reason `sam.decode_mask` introspects its decoder and
    `embed._pick_output` its outputs: exports vary, and the names are the only
    thing that says which tensor is which. The sequence axis is dynamic, so the
    ids are padded to CLIP's context length with a matching attention mask —
    exactly `text_embed.encode_terms`' arrangement, and for its reason: the
    pooled text vector is read at the end-of-text token, which is found by an
    argmax over the ids, so padding must be 0 and not the EOT id itself.
    """
    session = _session()
    tok = text_embed.tokenizer()
    prompts = [template.format(query) for template in TEMPLATES]
    ids = np.zeros((len(prompts), text_embed.CONTEXT_LENGTH), dtype=np.int64)
    mask = np.zeros((len(prompts), text_embed.CONTEXT_LENGTH), dtype=np.int64)
    for row, prompt in enumerate(prompts):
        tokens = tok.encode(prompt)
        ids[row, : len(tokens)] = tokens
        mask[row, : len(tokens)] = 1

    feeds = {}
    for info in session.get_inputs():
        if "pixel" in info.name:
            # The image batch axis is the graph's own and separate from the
            # text's, but this export pairs them one for one, so the same
            # picture is handed to it once per prompt.
            feeds[info.name] = np.repeat(_preprocess(img), len(prompts), axis=0)
        elif "attention" in info.name or "mask" in info.name:
            feeds[info.name] = mask
        elif "ids" in info.name or "input" in info.name:
            feeds[info.name] = ids
    outputs = session.run(None, feeds)

    # One 352x352 map per prompt, however the graph shapes them — the reshape
    # absorbs a squeezed batch axis, which is what this export does when there
    # is only one prompt (and what its own declared shape then disagrees with,
    # hence the log level in `_session`).
    logits = next(o for o in outputs if o.size == len(prompts) * INPUT_SIZE**2)
    logits = np.asarray(logits, dtype=np.float32).reshape(-1, INPUT_SIZE, INPUT_SIZE)
    # A sigmoid, so the number that reaches MATCH_FLOOR and the UI is a
    # probability rather than a logit whose scale means nothing to anyone.
    return (1.0 / (1.0 + np.exp(-logits))).max(axis=0)


def _smooth(a: np.ndarray, r: int) -> np.ndarray:
    """Separable box mean, radius `r`, edges clamped — one cumsum per axis.

    The prefix-sum trick `effects._disk_mean` runs on, in its simplest form: a
    window of any size costs two reads once the cumulative sum exists, so the
    radius is free. Edges repeat rather than darkening, since a dimmed border
    would bias the argmax away from a subject that runs off the frame.
    """
    a = a.astype(np.float32)
    for axis in (0, 1):
        a = np.swapaxes(a, 0, axis)
        pad = np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)])
        cs = np.concatenate([np.zeros((1,) + pad.shape[1:], np.float32), pad.cumsum(0)])
        a = np.swapaxes((cs[2 * r + 1 :] - cs[: -(2 * r + 1)]) / (2 * r + 1), 0, axis)
    return a


def locate(img: np.ndarray, query: str) -> dict:
    """Where in an RGB uint8 array a phrase is talking about.

    Returns the point in the array's own pixel coordinates, the smoothed peak as
    `score`, and `area` — the pixel count of the region above half that peak,
    scaled to the array. `area` is not a mask and is never shown; it is the size
    hint that lets `sam.level_for_area` choose between the candidate masks under
    the point, which is the difference between "the sky" selecting the sky and
    selecting one cloud in it.
    """
    h, w = img.shape[:2]
    hm = _smooth(heatmap(img, query), _SMOOTH_R)
    flat = int(np.argmax(hm))
    peak = float(hm.flat[flat])
    cells = int((hm > _BLOB_OF_PEAK * peak).sum()) if peak > 0 else 0
    return {
        # +0.5 lands on the centre of the winning cell rather than its corner,
        # then clamped: the last cell's centre rounds past the last pixel.
        "x": min(w - 1, int((flat % INPUT_SIZE + 0.5) * w / INPUT_SIZE)),
        "y": min(h - 1, int((flat // INPUT_SIZE + 0.5) * h / INPUT_SIZE)),
        "score": peak,
        "area": cells * (h * w) / (INPUT_SIZE * INPUT_SIZE),
        "confident": peak >= MATCH_FLOOR,
    }
