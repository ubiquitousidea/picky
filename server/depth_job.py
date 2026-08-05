"""Fetching the depth model off the request, with progress.

`text_job.py`'s idiom, and `text_job.py`'s kind — the distinction that module
draws between itself and `embed_job` is the one that matters here:

**`embed_job` is only ever an optimization and this is not.**
`GET /api/embedding-map` embeds whatever it finds missing, so nobody has to
prepare. A bokeh render cannot do the equivalent, for the reason search cannot:
fetching 99 MB inside a request is a minute of nothing, with no way to tell a
slow download from a hung server. So `main._check_effect_ready` *refuses* a
bokeh preview or apply until `depth.ready()`, and this job is the only way it
becomes ready.

It is also why this is a separate module rather than another phase of an
existing job: someone who never touches the Bokeh button must never pay for the
depth model, exactly as someone who never types in the search box must never
pay for the text tower.

One file, one phase, `done`/`total` always bytes.
"""

import threading

from . import depth

# state: idle | running | done | error
# phase: model (resolving paths) | download (bytes moving) | None
_lock = threading.Lock()
_job = {"state": "idle", "phase": None, "done": 0, "total": 0, "error": None}


def snapshot() -> dict:
    # `ready` rides along for `text_job.snapshot`'s reason: it is the question
    # the frontend actually has, and it is true in two different states — after
    # a download this job ran, and on any later server start, where this job is
    # `idle` and always will be.
    with _lock:
        return dict(_job, ready=depth.ready())


def _set(**fields) -> None:
    with _lock:
        _job.update(fields)


def start() -> dict:
    """Fetch the depth model, or rejoin the fetch already running.

    Latches `done` where `embed_job.start()` deliberately does not: that job
    re-runs because images get added after a pass finishes, and this one has
    nothing that can go stale — the revision is pinned, so the file on disk is
    the file there will ever be.
    """
    with _lock:
        if _job["state"] == "running":
            return dict(_job, ready=depth.ready())
        # Claim the job before deciding what it involves, so two concurrent
        # prepare calls (these are sync endpoints, so they share a threadpool)
        # cannot both spawn a thread.
        _job.update(state="running", phase="model", done=0, total=0, error=None)

    if depth.ready():
        # The common case on every run after the first: answer it here rather
        # than making the caller poll for it.
        _set(state="done", phase=None)
        return snapshot()

    threading.Thread(target=_run, name="picky-depth-model", daemon=True).start()
    return snapshot()


def _run() -> None:
    try:
        depth.ensure_model(on_progress=_downloading)
        _set(state="done", phase=None)
    except Exception as exc:
        _set(state="error", phase=None, error=_describe(exc))


def _downloading(done: int, total: int | None) -> None:
    # Sets the phase on every chunk rather than once up front, so it flips only
    # when bytes are actually moving — which is how a caller tells "downloading"
    # from "already on disk" without asking a second question.
    _set(phase="download", done=done, total=total or 0)


def _describe(exc: Exception) -> str:
    # The type carries most of the diagnosis for the failures that actually
    # happen here (URLError, FileNotFoundError from a bad PICKY_DEPTH_MODEL),
    # and several of them stringify to nothing.
    return f"{type(exc).__name__}: {exc}"
