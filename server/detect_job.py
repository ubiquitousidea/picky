"""Detecting the library's objects off the request, with progress.

`embed_job.py`'s shape — daemon thread, one module-level progress dict, a pass
that skips what is already done — and every argument in that module's docstring
for why the state is shaped this way applies here unchanged.

**What differs is that this one is not merely a cache warmer.** `GET
/api/embedding-map` embeds whatever it finds missing, so that job is only ever an
optimization; there is no equivalent here and there must not be. Detection is
~40 ms an image, which is nothing on an upload and forty seconds inside a request
that touches a thousand of them, so nothing computes a missing detection on
demand. An image that has never been through this job simply has no labels, the
tag filter does not offer it, and the CLIP search — which needs none of this —
answers exactly as it did before. That is the degradation this feature is
designed around: tags make search sharper, and their absence makes it no worse
than it was.

So there are exactly two ways a row gets written: `main.upload_image` detects
inline when the weights are already on disk, and this job sweeps whatever that
missed — every image imported before the feature existed, everything imported
while the model was still downloading, and everything re-framed since (the crop
endpoint drops the stamp, see `db.clear_detections`).
"""

import threading

from . import db, detect, rendering

# state: idle | running | done | error
# phase: model (resolving weights) | download (fetching them) | detect | None
# done/total: bytes while downloading, images while detecting
_lock = threading.Lock()
_job = {"state": "idle", "phase": None, "done": 0, "total": 0, "error": None}


def snapshot() -> dict:
    # Both extras answer questions the job's own state cannot. `ready` is
    # text_job's: the weights are on disk, which is true after this job fetched
    # them and on every later server start where this job is `idle` and always
    # will be. `pending` is how many images still have no labels, which is what
    # makes a sweep that has never run distinguishable from a library that
    # simply contains none of COCO's eighty nouns.
    #
    # Counted outside the lock: it is a database query, and the worker thread
    # takes this lock on every image.
    pending = db.undetected_count()
    with _lock:
        return dict(_job, ready=detect.ready(), pending=pending)


def _set(**fields) -> None:
    with _lock:
        _job.update(fields)


def _pending() -> list[dict]:
    """Images the detector has not run over — one query, not one per image."""
    done = db.detected_image_ids()
    return [image for image in db.list_images() if image["id"] not in done]


def start() -> dict:
    """Begin a pass over the library, or rejoin the one already running.

    Restarts after a finished pass rather than latching `done`, for
    `embed_job.start`'s reason: a completed run says nothing about images
    imported or re-framed since, and re-running is nearly free because a
    stamped image is skipped rather than re-detected.
    """
    with _lock:
        if _job["state"] == "running":
            return dict(_job)
        # Claim the job before deciding what it involves, so two concurrent
        # callers (these are sync endpoints sharing a threadpool) cannot both
        # spawn a thread.
        _job.update(state="running", phase="model", done=0, total=0, error=None)

    try:
        # Answer "nothing to do" here rather than making the caller poll for
        # it — the common case once the library has been swept once.
        if not _pending() and detect.ready():
            _set(state="done", phase=None)
            return snapshot()
    except Exception as exc:
        # Never leave the job claimed-but-unstarted: without this a failure in
        # the check above would read as "running" forever.
        _set(state="error", phase=None, error=_describe(exc))
        return snapshot()

    threading.Thread(target=_run, name="picky-detect", daemon=True).start()
    return snapshot()


def _run() -> None:
    try:
        detect.ensure_model(on_progress=_downloading)
        pending = _pending()
        _set(phase="detect", done=0, total=len(pending))
        for n, image in enumerate(pending, 1):
            try:
                rendering.detect_image(image["id"], image["root_node_id"])
            except Exception as exc:
                # One unreadable original must not cost the other 1500 images
                # their labels. The image stays unstamped, so the next pass
                # tries it again — which is the right outcome if the cause was
                # a half-written upload rather than a corrupt file.
                print(f"picky: could not detect in image {image['id']}: {exc}")
            _set(done=n)
        _set(state="done", phase=None)
    except Exception as exc:
        _set(state="error", phase=None, error=_describe(exc))


def _downloading(done: int, total: int | None) -> None:
    # Sets the phase on every chunk rather than once up front, so it flips only
    # when bytes are actually moving — an already-downloaded model never shows a
    # download phase at all.
    _set(phase="download", done=done, total=total or 0)


def _describe(exc: Exception) -> str:
    # The type carries most of the diagnosis for the failures that actually
    # happen here (ModuleNotFoundError: onnxruntime, URLError, FileNotFoundError
    # from a bad PICKY_YOLO_MODEL), and several of them stringify to nothing.
    return f"{type(exc).__name__}: {exc}"
