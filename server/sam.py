"""MobileSAM (Segment Anything) via onnxruntime: click a point, get a mask.

Two-stage pipeline, split so the cost lands in the right place:

- `compute_embedding` runs the heavy image encoder (~0.5-2s on CPU). Callers
  cache its output per node (`rendering.node_embedding`), so each node pays
  this once no matter how many clicks follow.
- `decode_mask` runs the prompt decoder (~10-50ms): fixed arithmetic on the
  embedding plus one click, so masks are deterministic given (embedding, click)
  — which is what lets a stored selection re-render byte-identically.

Model weights are ONNX exports of the official MobileSAM checkpoint. Sources,
in order of precedence:

1. `PICKY_SAM_ENCODER` / `PICKY_SAM_DECODER` env vars (absolute paths).
2. Files already present in `data/models/`.
3. Lazy download from Hugging Face, pinned to a specific *revision hash* —
   a git commit id content-addresses the files, so the URL cannot silently
   start serving different bytes (the same integrity property as pinning a
   file hash, resting on HF's content addressing).

Exports vary (I/O names, whether preprocessing is baked into the graph,
low-res vs full-size mask output), so both entry points introspect the
session instead of hardcoding one export's contract. For the encoder the
tell is the input *rank*, and `compute_embedding` handles both families:

- **Rank 3** — `(image_height, image_width, 3)`, dynamic HW, fixed 64x64
  output. Preprocessing is baked into the graph: it wants raw 0-255 pixels
  and normalizes and pads to 1024 itself. This is what `_HF_BASE` serves.
- **Rank 4** — `(1, 3, 1024, 1024)`, a plain `torch.onnx.export` of
  `model.image_encoder`. Caller must normalize and pad.

Do *not* distinguish them by dtype: the rank-3 export declares float, so a
`uint8`-vs-float test silently picks the rank-4 path and double-normalizes.

To produce the files yourself, the MobileSAM repo's
`scripts/export_onnx_model.py` exports the decoder — do NOT pass
`--return-single-mask`; the level picker needs the multi-mask output.
"""

import functools
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from . import db

MODELS_DIR = db.DATA_DIR / "models"

# Acly/MobileSAM at revision 0d3b403339b4674a82493d5e97964dd78089ddc8
_HF_BASE = (
    "https://huggingface.co/Acly/MobileSAM/resolve/"
    "0d3b403339b4674a82493d5e97964dd78089ddc8"
)
_MODELS = {
    "encoder": {
        "env": "PICKY_SAM_ENCODER",
        "filename": "mobile_sam_image_encoder.onnx",
        "url": f"{_HF_BASE}/mobile_sam_image_encoder.onnx",
    },
    "decoder": {
        "env": "PICKY_SAM_DECODER",
        "filename": "sam_mask_decoder_multi.onnx",
        "url": f"{_HF_BASE}/sam_mask_decoder_multi.onnx",
    },
}

# SAM's fixed input resolution and ImageNet-ish normalization constants.
INPUT_SIZE = 1024
_PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def _model_path(kind: str) -> Path:
    spec = _MODELS[kind]
    override = os.environ.get(spec["env"])
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"{spec['env']} points at missing file {path}")
        return path
    path = MODELS_DIR / spec["filename"]
    if not path.is_file():
        download_model(spec["url"], path)
    return path


def download_model(url: str, dest: Path, on_progress=None) -> None:
    """Fetch to a tmp file and os.replace() into place, so a crashed download
    never leaves a truncated model where the existence check would trust it.

    Public because `embed.py` fetches its CLIP weights the same way: the atomic
    write and the pinned-revision contract in this module's docstring must not
    exist as two copies that can drift apart.

    `on_progress(bytes_done, bytes_total)` is called once per chunk, and only
    while bytes are actually moving — so a caller can tell "downloading" from
    "already on disk" without asking a second question. `bytes_total` is None
    when the server sends no Content-Length. This is what lets the Image map's
    prepare job say *which* minute of the first run you are in; nothing else
    passes a callback, and the default keeps these downloads silent.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as resp:
            length = resp.headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress(done, total)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    print(f"picky: downloaded {dest.name} ({dest.stat().st_size} bytes, "
          f"sha256 {digest.hexdigest()})")


@functools.lru_cache(maxsize=None)
def _session(kind: str):
    # onnxruntime imported lazily, like sklearn in effects._fit_kmeans: the
    # server must start (and everything unrelated to segmentation must work)
    # even if the wheel is missing or broken on this platform.
    import onnxruntime

    return onnxruntime.InferenceSession(
        str(_model_path(kind)), providers=["CPUExecutionProvider"]
    )


def _resize_dims(h: int, w: int) -> tuple[int, int]:
    # SAM's ResizeLongestSide rounding, exactly: coords must be transformed
    # with the same scale the encoder resize used or masks drift off-click.
    scale = INPUT_SIZE / max(h, w)
    return int(h * scale + 0.5), int(w * scale + 0.5)


def compute_embedding(img: np.ndarray) -> np.ndarray:
    """Run the SAM image encoder on an RGB uint8 array -> float32 embedding.

    The resize to longest-side 1024 is ours in either case — no export resizes
    internally, and feeding a full-size image instead yields a completely
    unrelated embedding (cosine 0.31 against the correct one).
    """
    h, w = img.shape[:2]
    new_h, new_w = _resize_dims(h, w)
    resized = np.asarray(
        Image.fromarray(img).resize((new_w, new_h), Image.Resampling.BILINEAR)
    )

    session = _session("encoder")
    info = session.get_inputs()[0]
    # symbolic dims come through as strings, so read the rank, not the values:
    # rank 3 is the HWC preprocessing-included family, rank 4 the bare encoder.
    if len(info.shape) == 3:
        # The graph does the mean/std normalize and the pad-to-1024 itself and
        # wants raw 0-255 pixels. Doing either here as well hands the encoder
        # values in ~[-2, 2], which degrades the embedding into one that returns
        # most of the frame for every click — the bug this branch exists to fix.
        # Dtype alone cannot tell the families apart: this export declares
        # float, not uint8.
        x = resized if "uint8" in info.type else resized.astype(np.float32)
    else:
        canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized  # zero-pad bottom/right, like SAM
        x = (
            canvas
            if "uint8" in info.type
            else (canvas.astype(np.float32) - _PIXEL_MEAN) / _PIXEL_STD
        )
        x = x.transpose(2, 0, 1)[None] if info.shape[1] == 3 else x[None]
    (embedding,) = session.run(None, {info.name: np.ascontiguousarray(x)})
    return embedding.astype(np.float32)


def decode_mask(
    embedding: np.ndarray, orig_h: int, orig_w: int, x: int, y: int, level: str
) -> np.ndarray:
    """Decode one click into a boolean (orig_h, orig_w) mask.

    `level` picks among SAM's candidate masks: `auto` trusts the model's IoU
    ranking; `whole`/`part`/`subpart` rank the multi-mask candidates by area
    (SAM does not guarantee an ordering, so index order can't be trusted).
    """
    x = min(max(0, x), orig_w - 1)
    y = min(max(0, y), orig_h - 1)
    new_h, new_w = _resize_dims(orig_h, orig_w)
    # same transform as ResizeLongestSide.apply_coords: floats, not rounded
    coords = np.array(
        [[[x * (new_w / orig_w), y * (new_h / orig_h)], [0.0, 0.0]]],
        dtype=np.float32,
    )
    labels = np.array([[1, -1]], dtype=np.float32)  # click + padding point

    session = _session("decoder")
    available = {}
    for info in session.get_inputs():
        available[info.name] = info
    feeds = {}
    for name in available:
        if "embed" in name:
            feeds[name] = embedding
        elif "coord" in name:
            feeds[name] = coords
        elif "label" in name:
            feeds[name] = labels
        elif name == "mask_input":
            feeds[name] = np.zeros((1, 1, 256, 256), dtype=np.float32)
        elif name == "has_mask_input":
            feeds[name] = np.zeros(1, dtype=np.float32)
        elif "size" in name:
            feeds[name] = np.array([orig_h, orig_w], dtype=np.float32)
    outputs = session.run(None, feeds)

    # masks: the rank-4 output (1, N, H, W); scores: the rank-2 one (1, N)
    masks = next(o for o in outputs if o.ndim == 4)[0]
    scores = next((o for o in outputs if o.ndim == 2), None)
    binary = [_upscale(m, orig_h, orig_w, new_h, new_w) > 0.0 for m in masks]

    # the multi-mask granularities are candidates 1.. when the single-mask
    # token is also present (official multi export returns 4 masks)
    first = 1 if len(binary) > 3 else 0
    if level == "auto" and scores is not None:
        # skip the single-mask token for the same reason the level picker does,
        # and as official SAM does whenever the multi-mask outputs are in play
        pick = first + int(np.argmax(scores[0][first:]))
    else:
        candidates = list(range(first, len(binary)))
        by_area = sorted(candidates, key=lambda i: (int(binary[i].sum()), -i))
        if level == "subpart":
            pick = by_area[0]
        elif level == "part":
            pick = by_area[len(by_area) // 2]
        elif level == "whole":
            pick = by_area[-1]
        else:  # auto without scores
            pick = by_area[-1]
    return binary[pick]


def _upscale(logits: np.ndarray, orig_h: int, orig_w: int, new_h: int, new_w: int) -> np.ndarray:
    """Bring one mask's logits to full resolution.

    Full-size exports (orig_im_size in the graph) come back already at
    (orig_h, orig_w). Low-res exports return 256x256 logits covering the
    padded 1024 square, so crop the valid region before scaling up —
    mode-F PIL images interpolate float logits bilinearly.
    """
    if logits.shape == (orig_h, orig_w):
        return logits
    lh, lw = logits.shape
    valid_h = max(1, round(lh * new_h / INPUT_SIZE))
    valid_w = max(1, round(lw * new_w / INPUT_SIZE))
    im = Image.fromarray(logits[:valid_h, :valid_w].astype(np.float32), mode="F")
    return np.asarray(im.resize((orig_w, orig_h), Image.Resampling.BILINEAR))
