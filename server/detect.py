"""COCO object detection via onnxruntime: what is in a photo, and where.

The complement to `embed.py`. That module answers "what is this photo *like*",
as one vector for the whole frame; this one answers "what objects are in it",
as a list of boxes. The two fail in opposite directions, which is the reason
both exist. CLIP is open-vocabulary and knows about weather, mood and style, but
one vector summarizing a whole frame is dominated by scene gestalt — searching
"people" ranks a macaque portrait alongside the people, since it shares the
face, the framing and the depth of field. A detector has only 80 nouns and
nothing to say about a photo being wistful, but `person` is among the most
heavily trained classes in computer vision, and it scores a *region* rather than
a scene, so the background cannot leak into the answer.

**The 80 classes are the whole vocabulary, and there is no `monkey` among
them.** That is not a bug to work around: run this on a macaque and it answers
`dog` at 0.61, because the nearest of the eighty is all it can say. A caller
must therefore read a detection as "there is a dog-ish animal here" and never as
"there is no monkey here" — absence of a label means the label was not detected,
which for anything outside COCO is guaranteed in advance. `labels.py`'s 857-term
vocabulary is what covers the rest, and the two are meant to be read together.

Weights are YOLOv8n, the smallest of the family, as a 640x640 ONNX export.
Sources and their precedence are `sam.py`'s exactly, and the download itself is
literally `sam.download_model` — see that module for why the revision is pinned
to a commit hash and why the write is atomic. At 12.8 MB it is by far the
smallest model in the tree, and at ~40 ms an image the only one that can afford
to run inside an upload.

Three things here are easy to get wrong:

- **This export emits box coordinates normalized to the letterbox, not in
  pixels.** Ultralytics' own export writes them in 0..640 and most published
  decoding snippets assume that, so the arithmetic looks right and every box
  lands in the top-left corner, a fraction of a pixel wide. The pinned revision
  is what makes it safe to hardcode the convention, and `_boxes` checks it.
- **The preprocessing is a letterbox, not a resize.** Squashing to a square
  moves every box and costs small objects the aspect they were trained at; the
  padding colour is the 114 grey the training pipeline used.
- **Scores are already sigmoid'd in the graph.** Applying another one is
  silently survivable — it is monotone, so the ranking and the boxes are
  unchanged and only the numbers move — which is exactly why it would never be
  noticed. It would quietly push every score toward 0.5 and make
  `CONF_THRESHOLD` mean something else.
"""

import functools
import os
from pathlib import Path

import numpy as np
from PIL import Image

# MODELS_DIR and the downloader are sam.py's: it owns the conventions for
# on-disk model weights, and this is the fifth module to need them.
from .sam import MODELS_DIR, download_model

# SpotLab/YOLOv8Detection at revision 3005c6751fb19cdeb6b10c066185908faf66a097.
# The weights are Ultralytics YOLOv8n trained on COCO, and are AGPL-3.0 —
# unlike every other model here, which is Apache or MIT. That is a licence to
# read before this app is distributed, not merely run.
_HF_URL = (
    "https://huggingface.co/SpotLab/YOLOv8Detection/resolve/"
    "3005c6751fb19cdeb6b10c066185908faf66a097/yolov8n.onnx"
)
_ENV_VAR = "PICKY_YOLO_MODEL"
_FILENAME = "yolov8n.onnx"

# The graph's fixed input. Not negotiable per-call: the export has a static
# 1x3x640x640 input, so a different size is a different model.
INPUT_SIZE = 640
# Ultralytics' letterbox padding, and the value the model saw in training.
_PAD = 114

# How much JPEG to decode before detecting. The frame is going to 640 square
# regardless, so full-resolution decode of a 40 MP original is almost all
# waste — but decoding *at* 640 is not enough either, since PIL's DCT scaling
# only goes in powers of two and the last halving is what small objects live in.
# 1280 is the compromise, and it is what takes a pass over the library from
# ~25 minutes to ~4.
DECODE_HINT = 1280

# Below this a detection is not stored at all. Deliberately Ultralytics' own
# default rather than something stricter: the score rides along on every row, so
# a caller wanting more confidence can filter, but a row never written cannot be
# recovered without re-running the model over the whole library.
CONF_THRESHOLD = 0.25
# Two boxes of one class overlapping by more than this are the same object.
IOU_THRESHOLD = 0.45
# A guard against a pathological frame, not a considered limit — a dense crowd
# legitimately reaches 30-40 people. Nothing in the app pages through these.
MAX_DETECTIONS = 100

# COCO's 80 classes, in the order the detection head emits them. Written here
# rather than read from the graph because this export carries no metadata map
# (Ultralytics' own `names` field does not survive it), and a caller that mixes
# up the order gets plausible labels on the wrong boxes.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def model_path() -> Path:
    """Where the weights live — env override, else `data/models/`.

    The file need not exist yet: `.is_file()` on this is exactly the question
    "would detecting cost a download?", which `ready()` answers without touching
    the network.
    """
    override = os.environ.get(_ENV_VAR)
    return Path(override) if override else MODELS_DIR / _FILENAME


def ready() -> bool:
    """Is detecting free right now, or does it cost a 12.8 MB download?

    `embed.model_path().is_file()`'s question. `main.upload_image` uses it to
    decide whether to detect inline, and `detect_job` to short-circuit.
    """
    return model_path().is_file()


def ensure_model(on_progress=None) -> Path:
    """The weights, downloading them on first use.

    Public and separate from `_session` for `embed.ensure_model`'s reason: the
    slow part has to be doable up front, off the request, with progress.
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
    # onnxruntime imported lazily, like embed._session and sam._session: the
    # server must start, and everything unrelated to detection must keep
    # working, if the wheel is missing or broken on this platform.
    import onnxruntime

    return onnxruntime.InferenceSession(
        str(ensure_model()), providers=["CPUExecutionProvider"]
    )


def _letterbox(img: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Fit `img` into a 640 square on grey padding, preserving aspect.

    Returns the canvas and the mapping back out of it — the scale applied and
    the padding added — since every box comes back in canvas coordinates and
    has to be undone through exactly these numbers.
    """
    im = Image.fromarray(img)
    w, h = im.size
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
    new_w, new_h = round(w * scale), round(h * scale)
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (_PAD, _PAD, _PAD))
    pad_x, pad_y = (INPUT_SIZE - new_w) // 2, (INPUT_SIZE - new_h) // 2
    canvas.paste(im.resize((new_w, new_h), Image.Resampling.BILINEAR), (pad_x, pad_y))
    x = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), scale, pad_x, pad_y


def _boxes(raw: np.ndarray, scale: float, pad_x: int, pad_y: int,
           width: int, height: int) -> np.ndarray:
    """Centre-form boxes out of the head, as corners normalized to the image.

    `raw` is the (n, 4) block of cx, cy, w, h. **This export writes them
    normalized to the letterbox**, so they are scaled up by `INPUT_SIZE` before
    the padding is removed — see the module docstring for why that is worth
    stating twice.
    """
    if len(raw) and raw.max() > 1.5:
        # The pinned revision cannot change under us, so this is unreachable
        # unless PICKY_YOLO_MODEL points at a differently-exported graph. Say so
        # rather than silently returning boxes 640x too small.
        raise RuntimeError(
            "YOLO export emits box coordinates in pixels, not normalized to the "
            "letterbox; this decoder expects the pinned revision's convention"
        )
    cx, cy, bw, bh = (raw[:, i] * INPUT_SIZE for i in range(4))
    corners = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    corners[:, [0, 2]] = (corners[:, [0, 2]] - pad_x) / scale / width
    corners[:, [1, 3]] = (corners[:, [1, 3]] - pad_y) / scale / height
    # A box may legitimately run off the frame — the model extrapolates a
    # partly-visible object — but a stored fraction outside 0..1 would be a
    # trap for every consumer that multiplies by a display size.
    return np.clip(corners, 0.0, 1.0)


def _nms(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
    """Greedy non-maximum suppression, returning kept indices, best first.

    Written out rather than pulled from torchvision because it is fifteen lines
    over at most a few hundred boxes, and the alternative is a dependency the
    rest of the app does not have.
    """
    order = np.argsort(-scores)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while len(order):
        best = order[0]
        keep.append(int(best))
        if len(order) == 1:
            break
        rest = order[1:]
        x0 = np.maximum(boxes[best, 0], boxes[rest, 0])
        y0 = np.maximum(boxes[best, 1], boxes[rest, 1])
        x1 = np.minimum(boxes[best, 2], boxes[rest, 2])
        y1 = np.minimum(boxes[best, 3], boxes[rest, 3])
        overlap = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
        union = area[best] + area[rest] - overlap
        order = rest[overlap / np.where(union > 0, union, 1) <= IOU_THRESHOLD]
    return keep


def detect(img: np.ndarray) -> list[dict]:
    """Objects in an RGB uint8 array, as `{label, score, box}` best first.

    `box` is `[x0, y0, x1, y1]` as fractions of the image, which is what makes a
    row survive being displayed at thumbnail size, at preview size and in the
    map's sprites without anything remembering which one it was measured in. It
    is *not* invariant to re-framing, though — see `db.clear_detections`.

    Suppression is per class, so a person standing in front of a car yields both;
    only two boxes of the *same* class competing for one object are merged.
    """
    session = _session()
    height, width = img.shape[:2]
    x, scale, pad_x, pad_y = _letterbox(img)
    # (1, 84, 8400) -> (8400, 84): 8400 candidate anchors, each 4 box numbers
    # followed by one already-sigmoid'd score per COCO class.
    out = session.run(None, {session.get_inputs()[0].name: x})[0][0].T
    class_scores = out[:, 4:]
    best = class_scores.argmax(axis=1)
    score = class_scores[np.arange(len(out)), best]

    hit = score >= CONF_THRESHOLD
    if not hit.any():
        return []
    boxes = _boxes(out[hit, :4], scale, pad_x, pad_y, width, height)
    score, best = score[hit], best[hit]

    found = []
    for cls in np.unique(best):
        member = best == cls
        member_boxes, member_scores = boxes[member], score[member]
        for index in _nms(member_boxes, member_scores):
            found.append(
                {
                    "label": COCO_CLASSES[int(cls)],
                    "score": float(member_scores[index]),
                    "box": [round(float(v), 5) for v in member_boxes[index]],
                }
            )
    found.sort(key=lambda d: -d["score"])
    return found[:MAX_DETECTIONS]
