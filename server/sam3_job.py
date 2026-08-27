"""Reporting whether SAM 3's graphs are on disk, in `clipseg_job`'s shape.

A fifth job module, because the tree's convention is one per gesture and picking
SAM 3 in the selection row is a gesture. It answers the same
`{state, phase, done, total, error, ready}` the other four do, so the frontend's
prepare-then-poll loop needs no new shape to understand.

What it does *not* do is download. Meta ships no ONNX export and gates the
weights behind manual approval, so the graphs arrive by way of
`tools/export_sam3_onnx.py` and never by way of a request — see `sam3.py`. That
makes this the one job with no `download` phase and no thread: the question
"are they there" is answered synchronously, and the only two answers are `done`
and `error`.

Keeping it a job module anyway is deliberate. The alternative is the endpoint
special-casing this one model, and then the browser special-casing it too, which
would put "SAM 3 is the odd one" in three files instead of one. Here the oddness
is a missing branch in `start()` and nothing else.
"""

import threading

from . import sam3

# state: idle | running | done | error
# phase: always None here — there is no download to be in the middle of
# done/total: always 0, for the same reason. Both ride along so the snapshot is
# the shape the other four return and the frontend's poll loop stays one loop.
_lock = threading.Lock()
_job = {"state": "idle", "phase": None, "done": 0, "total": 0, "error": None}


def snapshot() -> dict:
    # `ready` is the question the frontend actually has, and — as in every
    # sibling — it is true in more than one state: after an export, and on every
    # later server start, where this job is `idle` and always will be.
    with _lock:
        return dict(_job, ready=sam3.ready())


def _set(**fields) -> None:
    with _lock:
        _job.update(fields)


def start() -> dict:
    """Answer whether SAM 3 can run, and say what to do if it cannot.

    No lock dance around claiming the job, and no thread: `clipseg_job.start()`
    claims first so two concurrent prepare calls cannot both spawn a download,
    and there is nothing here for two callers to duplicate. `sam3.ready()` is a
    handful of stat calls.

    The error is not a failure of this job — it is the normal state of a machine
    where nobody has run the export yet, and it carries the command rather than a
    diagnosis so the browser and the agent can both just repeat it.
    """
    if sam3.ready():
        _set(state="done", phase=None, error=None)
    else:
        _set(state="error", phase=None, error=sam3.EXPORT_HINT)
    return snapshot()
