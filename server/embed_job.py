"""Building the library's CLIP vectors off the request, with progress.

The Image map needs every image embedded before it can be projected, and on a
fresh install that means downloading ~335 MB of weights and then running the
model over the whole library. As one blocking GET that was tens of seconds of
nothing, with no way to tell a slow download from a hung server — so the work
moves to a daemon thread and `snapshot()` reports where it got to, which is all
the modal's status line needs.

This is a **cache warmer, not the source of truth**. `GET /api/embedding-map`
still embeds anything it finds missing, so the map is correct whether or not
anyone called prepare, and a job that dies takes nothing with it. The two may
even overlap: both write the same vector for the same image through
`db.set_embedding`, so the worst case is computing one twice — never a wrong
answer, which is what makes it safe to leave the blocking path alone rather than
gate it behind the job.

Progress is a single module-level dict rather than a per-job record because
there is nothing to disambiguate: the work is idempotent and library-wide, so
two callers wanting it done want the *same* pass, and `start()` hands the second
one the first one's progress. The thread is a daemon since every image commits
its own vector as it goes — killing the server mid-pass loses at most the one in
flight, and the next `start()` picks up the rest.
"""

import threading

from . import db, embed, rendering

# state: idle | running | done | error
# phase: model (resolving weights) | download (fetching them) | embed | None
# done/total: bytes while downloading, images while embedding
_lock = threading.Lock()
_job = {"state": "idle", "phase": None, "done": 0, "total": 0, "error": None}


def snapshot() -> dict:
    with _lock:
        return dict(_job)


def _set(**fields) -> None:
    with _lock:
        _job.update(fields)


def _pending() -> list[dict]:
    """Images with no cached vector — one query, not one per image."""
    cached = db.get_embeddings()
    return [image for image in db.list_images() if image["id"] not in cached]


def start() -> dict:
    """Begin a pass over the library, or rejoin the one already running.

    Restarts after a finished pass rather than latching `done`: a completed run
    says nothing about images uploaded since, and re-running is nearly free
    because a cached vector is skipped, not recomputed.
    """
    with _lock:
        if _job["state"] == "running":
            return dict(_job)
        # Claim the job before deciding what it involves, so two concurrent
        # prepare calls (these are sync endpoints, so they share a threadpool)
        # cannot both spawn a thread.
        _job.update(state="running", phase="model", done=0, total=0, error=None)

    try:
        # Answer "nothing to do" here instead of making the caller poll for it —
        # that is the common case, every open after the first.
        if not _pending() and embed.model_path().is_file():
            _set(state="done", phase=None)
            return snapshot()
    except Exception as exc:
        # Never leave the job claimed-but-unstarted: without this a failure in
        # the check above would read as "running" forever.
        _set(state="error", phase=None, error=_describe(exc))
        return snapshot()

    threading.Thread(target=_run, name="picky-embed", daemon=True).start()
    return snapshot()


def _run() -> None:
    try:
        embed.ensure_model(on_progress=_downloading)
        pending = _pending()
        _set(phase="embed", done=0, total=len(pending))
        for n, image in enumerate(pending, 1):
            rendering.embed_image(image["id"], image["root_node_id"])
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
    # from a bad PICKY_CLIP_MODEL), and several of them stringify to nothing.
    return f"{type(exc).__name__}: {exc}"
