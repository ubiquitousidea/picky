"""Fetching the text-prompted segmentation model off the request, with progress.

`text_job.py`'s idiom, for `text_job.py`'s reason: 139 MB as one blocking POST
is a wait with no way to tell a slow download from a hung server. The work moves
to a daemon thread and `snapshot()` reports where it got to.

Strict in the same way that module is, and for the same reason: **this is not an
optimization.** `GET /api/embedding-map` embeds whatever it finds missing, so
nobody has to prepare it; `POST /api/nodes/{id}/select-text` cannot do the
equivalent, because downloading the model inside the request is the thing this
exists to avoid. So naming an object *refuses* until `clipseg.ready()`, and this
job is the only way it becomes ready.

It is a fourth job module rather than a phase of one of the others because each
is triggered by the one gesture that needs it — opening the map, typing a query,
lighting the Bokeh button, naming an object — and somebody who never names an
object must never pay for this model. Note that it *does* overlap `text_job` by
one thing: CLIP's tokenizer files, which `clipseg.ensure_model` asks for and
which search downloads too. They are 1.7 MB and `download_model` skips what is
already there, so whichever gesture comes first pays for them.
"""

import threading

from . import clipseg

# state: idle | running | done | error
# phase: model (resolving paths) | download (bytes moving) | None
# done/total: bytes, always
_lock = threading.Lock()
_job = {"state": "idle", "phase": None, "done": 0, "total": 0, "error": None}


def snapshot() -> dict:
    # `ready` rides along because it is the question the frontend actually has,
    # and it is true in two different states: after a download this job ran, and
    # on any later server start, where this job is `idle` and always will be.
    with _lock:
        return dict(_job, ready=clipseg.ready())


def _set(**fields) -> None:
    with _lock:
        _job.update(fields)


def start() -> dict:
    """Fetch the model, or rejoin the fetch already running.

    Latches `done` like `text_job.start()` and unlike `embed_job.start()`: that
    job re-runs because images get added after a pass finishes, and this one has
    nothing that can go stale — the revision is pinned, so the files on disk are
    the files there will ever be.
    """
    with _lock:
        if _job["state"] == "running":
            return dict(_job, ready=clipseg.ready())
        # Claim the job before deciding what it involves, so two concurrent
        # prepare calls (these are sync endpoints, so they share a threadpool)
        # cannot both spawn a thread.
        _job.update(state="running", phase="model", done=0, total=0, error=None)

    if clipseg.ready():
        # The common case on every run after the first: answer it here rather
        # than making the caller poll for it.
        _set(state="done", phase=None)
        return snapshot()

    threading.Thread(target=_run, name="picky-clipseg-model", daemon=True).start()
    return snapshot()


def _run() -> None:
    try:
        clipseg.ensure_model(on_progress=_downloading)
        _set(state="done", phase=None)
    except Exception as exc:
        _set(state="error", phase=None, error=_describe(exc))


def _downloading(done: int, total: int | None) -> None:
    # Sets the phase on every chunk rather than once up front, so it flips only
    # when bytes are actually moving. The counters restart per file — the
    # tokenizer's two come first — but together they are under two megabytes, so
    # what a viewer sees is one bar for the graph.
    _set(phase="download", done=done, total=total or 0)


def _describe(exc: Exception) -> str:
    # The type carries most of the diagnosis for the failures that actually
    # happen here (URLError, FileNotFoundError from a bad PICKY_CLIPSEG_MODEL),
    # and several of them stringify to nothing.
    return f"{type(exc).__name__}: {exc}"
