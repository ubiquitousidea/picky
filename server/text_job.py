"""Fetching CLIP's text tower off the request, with progress.

`embed_job.py`'s idiom, for `embed_job.py`'s reason: 254 MB as one blocking GET
is a minute of nothing, with no way to tell a slow download from a hung server.
The work moves to a daemon thread and `snapshot()` reports where it got to.

Simpler than that module in one way and stricter in another.

Simpler: there is no per-library pass here, only the download, so `done`/`total`
are always bytes and there is one phase.

Stricter: **`embed_job` is only ever an optimization and this is not.**
`GET /api/embedding-map` embeds whatever it finds missing, so nobody has to
prepare; `GET /api/embedding-map/search` cannot do the equivalent, because
downloading a quarter of a gigabyte inside a GET is the thing this exists to
avoid. So search *refuses* until `text_embed.ready()`, and this job is the only
way it becomes ready.

That is also why this is a separate job rather than another phase of
`embed_job._run`: that one runs on every open of the Image map, and someone who
never types in the search box must never pay for the text tower.
"""

import threading

from . import text_embed

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
        return dict(_job, ready=text_embed.ready())


def _set(**fields) -> None:
    with _lock:
        _job.update(fields)


def start() -> dict:
    """Fetch the text tower, or rejoin the fetch already running.

    Latches `done` where `embed_job.start()` deliberately does not: that job
    re-runs because images get added after a pass finishes, and this one has
    nothing that can go stale — the revision is pinned, so the files on disk are
    the files there will ever be.
    """
    with _lock:
        if _job["state"] == "running":
            return dict(_job, ready=text_embed.ready())
        # Claim the job before deciding what it involves, so two concurrent
        # prepare calls (these are sync endpoints, so they share a threadpool)
        # cannot both spawn a thread.
        _job.update(state="running", phase="model", done=0, total=0, error=None)

    if text_embed.ready():
        # The common case on every run after the first: answer it here rather
        # than making the caller poll for it.
        _set(state="done", phase=None)
        return snapshot()

    threading.Thread(target=_run, name="picky-text-model", daemon=True).start()
    return snapshot()


def _run() -> None:
    try:
        text_embed.ensure_model(on_progress=_downloading)
        _set(state="done", phase=None)
    except Exception as exc:
        _set(state="error", phase=None, error=_describe(exc))


def _downloading(done: int, total: int | None) -> None:
    # Sets the phase on every chunk rather than once up front, so it flips only
    # when bytes are actually moving. The counters restart per file — three
    # files are fetched — but the two small ones are under a megabyte together,
    # so what a viewer sees is one bar for the graph.
    _set(phase="download", done=done, total=total or 0)


def _describe(exc: Exception) -> str:
    # The type carries most of the diagnosis for the failures that actually
    # happen here (URLError, FileNotFoundError from a bad PICKY_CLIP_TEXT_MODEL),
    # and several of them stringify to nothing.
    return f"{type(exc).__name__}: {exc}"
