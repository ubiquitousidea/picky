"""CLIP image embeddings via onnxruntime: turn a photo into a semantic vector.

Backs the Image map, the 3D view of the whole library. The point of using CLIP
rather than something hand-rolled out of color histograms is *what* it groups
by: CLIP's image tower was trained against captions, so its neighbourhoods are
about subject and scene — beaches near beaches, faces near faces — where a
palette descriptor would file a red car next to a red sunset.

Weights are the vision tower of OpenAI CLIP ViT-B/32, as a single-input ONNX
export. Sources and their precedence are `sam.py`'s exactly, and the download
itself is literally `sam.download_model` — see that module's docstring for why
the revision is pinned to a commit hash and why the write is atomic.

Two things here differ from `sam.py` and are easy to get wrong:

- **The normalization constants are CLIP's, not SAM's.** They are also applied
  to 0-1 floats, where SAM's are applied to 0-255. Feeding SAM's ImageNet-ish
  numbers here yields vectors that still look plausible — finite, well spread,
  they still draw a pretty cloud — but whose neighbourhoods are meaningless.
  Nothing downstream can detect this; only comparing nearest neighbours against
  what the photos actually depict can.
- **The output is L2-normalized before it leaves.** CLIP is a cosine-similarity
  space, and PCA and t-SNE downstream only understand Euclidean distance. On
  unit vectors the two agree up to a monotone transform, so normalizing here is
  what makes "near in the plot" mean "similar photo".

The output *width* is deliberately not hardcoded: a `CLIPVisionModelWithProjection`
export emits the 512-d joint-space `image_embeds`, a bare `CLIPVisionModel`
emits a 768-d `pooler_output`, and either is a fine basis for image-to-image
similarity since no text tower is involved. Callers must therefore treat the
length as data — `rendering.image_embedding` stores it as raw float32 bytes and
`main.embedding_map` drops any cached vector whose width disagrees with today's
model, so swapping the export re-embeds instead of raising.
"""

import functools
import os
from pathlib import Path

import numpy as np
from PIL import Image

# MODELS_DIR and the downloader are sam.py's: it owns the conventions for
# on-disk model weights, and this is the second module to need them.
from .sam import MODELS_DIR, download_model

# Qdrant/clip-ViT-B-32-vision at revision e0c24ed0fa57fa3e4f97f30de74c51d944036ace
_HF_URL = (
    "https://huggingface.co/Qdrant/clip-ViT-B-32-vision/resolve/"
    "e0c24ed0fa57fa3e4f97f30de74c51d944036ace/model.onnx"
)
_ENV_VAR = "PICKY_CLIP_MODEL"
_FILENAME = "clip_vit_b32_vision.onnx"

INPUT_SIZE = 224
# CLIP's own normalization, over 0-1 floats. NOT sam._PIXEL_MEAN/_STD.
_PIXEL_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_PIXEL_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _model_path() -> Path:
    override = os.environ.get(_ENV_VAR)
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"{_ENV_VAR} points at missing file {path}")
        return path
    path = MODELS_DIR / _FILENAME
    if not path.is_file():
        download_model(_HF_URL, path)
    return path


@functools.lru_cache(maxsize=None)
def _session():
    # onnxruntime imported lazily, like sam._session and sklearn in
    # effects._fit_kmeans: everything unrelated to the Image map must keep
    # working if the wheel is missing or broken on this platform.
    import onnxruntime

    return onnxruntime.InferenceSession(
        str(_model_path()), providers=["CPUExecutionProvider"]
    )


def _preprocess(img: np.ndarray) -> np.ndarray:
    """CLIP's preprocessing: resize the *shortest* side to 224, center crop,
    scale to 0-1, normalize, NCHW.

    Shortest-side-then-crop, not a squash to 224x224 — the model saw centered
    square crops in training, and an aspect-distorted frame lands somewhere
    unrelated.
    """
    im = Image.fromarray(img)
    w, h = im.size
    scale = INPUT_SIZE / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    im = im.resize((new_w, new_h), Image.Resampling.BICUBIC)
    left = (new_w - INPUT_SIZE) // 2
    top = (new_h - INPUT_SIZE) // 2
    im = im.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))

    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - _PIXEL_MEAN) / _PIXEL_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def _pick_output(outputs: list[np.ndarray]) -> np.ndarray:
    """The pooled image vector among the session's outputs.

    Introspected rather than indexed, for the reason sam.py introspects its
    encoder's input rank: exports vary. A projection export returns
    `image_embeds` (1, 512) alongside `last_hidden_state` (1, 50, 768); a bare
    vision export returns `pooler_output` (1, 768) alongside the same. In both
    the pooled vector is the only rank-2 output, so shape identifies it without
    trusting either an order or a name.
    """
    pooled = [o for o in outputs if o.ndim == 2]
    if not pooled:
        raise RuntimeError(
            "CLIP export returned no rank-2 output to use as an image vector; "
            f"got shapes {[o.shape for o in outputs]}"
        )
    # A projection export's 512-d image_embeds is preferred over any wider
    # pooled tensor sharing the graph.
    return min(pooled, key=lambda o: o.shape[1])


def compute_embedding(img: np.ndarray) -> np.ndarray:
    """Embed an RGB uint8 array as a unit-length float32 vector."""
    session = _session()
    x = _preprocess(img)
    outputs = session.run(None, {session.get_inputs()[0].name: x})
    vec = _pick_output(outputs)[0].astype(np.float32)
    norm = float(np.linalg.norm(vec))
    # A zero vector is not reachable from a real photo, but dividing by zero
    # would poison every distance in the projection rather than one point.
    return vec / norm if norm > 0 else vec
