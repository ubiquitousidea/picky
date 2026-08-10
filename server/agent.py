"""Driving the app from a sentence — Claude Haiku with the app's own endpoints as tools.

This is a **second front end over the same API**, not a second implementation of
the app. Every write below goes through the function the Apply button goes
through (`main.create_node`, `main.update_node`, `main.apply_preset`), so
parent-ownership checks, `validate_params`' clamping, `_check_effect_ready`'s
409, the descendant sweep and the eager re-render all hold here for free. The
alternative — reaching into `db` and `rendering` directly — would be a second
place that has to know the rules, and the first one to drift would be the one
nobody was testing.

The same argument shapes what the model is *told*: the effect vocabulary is
rendered out of `EFFECTS` (`_effects_doc`), so a new effect is still one registry
entry and nothing else. Hand-writing a tool schema per effect would have made
this file the second place an effect has to be declared.

Two things are deliberately narrower than the UI:

- **No selections.** A click selection is a pixel coordinate the model cannot
  see, and banking one into a saved mask is the frontend's job by design (see
  `bankSelection` in CLAUDE.md) — so "blur the background" becomes a whole-frame
  blur, and the system prompt makes the model say so rather than pretend.
- **Deleting proposes, it does not act.** `delete_node`/`delete_photo` describe
  what would go and hand a `pending` action back to the browser, which runs the
  ordinary DELETE after the user presses a button. A misread sentence should not
  be able to destroy work, and the manual path already asks (`deleteNode()`'s
  `confirm()`); this is that gate, kept where the user can see it.
"""

import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import HTTPException

from . import db, depth_job, text_job
from .effects import effect_specs

# Haiku by choice: these turns are a handful of tool calls and a sentence, and
# the whole point of the row is that it answers while you are still looking at
# it. Note what is *absent* from the request in `run_turn` — Haiku 4.5 takes
# neither `output_config.effort` (it errors) nor adaptive thinking, so an
# Opus-shaped call copied in here would 400.
MODEL = "claude-haiku-4-5"
MAX_TOKENS = 2048

# The loop is bounded because an unbounded one bills you for its confusion. Ten
# rounds is far past any request this tool surface can honestly need — the
# longest real chain is find → show → apply → apply — so hitting it means the
# model is stuck, and saying so beats spinning.
MAX_ROUNDS = 10

# Kept on the client and posted back, so the server holds no session. Trimmed
# because "make it stronger" only ever needs the last few exchanges, and an
# unbounded transcript is an unbounded bill.
MAX_HISTORY = 20

# Where a raw CLIP image-text cosine stops meaning "that subject is in this
# photo". Measured over this library: real matches land at 0.28–0.32, a query
# for something absent still lands at 0.24–0.27, because search ranks the
# library rather than judging it — the top hit for gibberish scores as highly as
# the top hit for a photo you actually own.
#
# This is a floor on *confidence*, not on results. Below it the photos still come
# back, because "nothing matched" is the wrong answer when the subject is there
# and merely oddly worded; what changes is that `_find_photos` tells the model,
# in the tool result rather than the system prompt, to ask before editing. Said
# only in the prompt, this lost to the model's enthusiasm every time.
STRONG_MATCH = 0.275

# The repo root's .env, addressed rather than searched: uvicorn can be started
# from anywhere, and `load_dotenv()`'s default walk up from the cwd would find
# nothing (or worse, someone else's file) when it is started from outside.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def configured() -> bool:
    """Whether there is a key to talk to Claude with.

    Read live rather than latched at import so exporting the key and restarting
    is the whole fix. The frontend asks first and hides the row when this is
    false — a control that can only fail is worse than no control.
    """
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    """One client, built on first use.

    Not at import: `anthropic.Anthropic()` raises without a key, and this module
    is imported by `main` unconditionally. A server with no key must still start
    and serve every other endpoint.
    """
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY itself
    return _client


class _ToolError(Exception):
    """A tool result the model should read and recover from, not a 500.

    Everything a tool can refuse arrives this way — a 400 from `create_node`, an
    unknown preset name, a model still downloading — and comes back as a
    `tool_result` with `is_error`. The turn continues; the model apologises or
    tries something else. Raising past the loop would end the turn on a stack
    trace, which is the wrong answer to "you asked for radius 500".
    """


# ---------- What the model is told ----------


def _param_line(p: dict) -> str:
    """One param spec as a line the model can dial from."""
    kind = p["type"]
    if kind == "choice":
        detail = "one of " + ", ".join(p["options"])
    elif kind == "points":
        detail = f"list of [x, y] pairs, {p['min']}..{p['max']}, max {p['max_points']}"
    else:
        detail = f"{kind} {p['min']}..{p['max']}"
    return f"    {p['name']}: {detail} (default {json.dumps(p['default'])})"


def _effects_doc() -> str:
    """The effect registry, rendered for a prompt.

    Derived from `effect_specs()` — the same source `GET /api/effects` builds the
    frontend's controls from — so the model's vocabulary and the app's cannot
    disagree, and adding an effect to `EFFECTS` is still the only step.
    """
    blocks = []
    for spec in effect_specs():
        lines = [f"  {spec['name']} — {spec['label']}"]
        lines += [_param_line(p) for p in spec["params"]]
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


SYSTEM = """\
You are the assistant inside Picky, a photo effects app. The person is looking \
at one photo and talking to you about it. Use the tools to actually change what \
they see — do not describe what they could do, do it.

Effects you can apply, with their parameters:

{effects}

Notes that matter:

- Parameters are clamped to their range by the server, and it does not complain \
when it clamps. `apply_effect` and `edit_node` return the values that were \
actually stored — report those, not the ones you asked for.
- Omitted parameters take their default. When editing a node, name only the \
parameter you are changing; the rest are kept.
- `blend` mixes two nodes of the same photo and needs `second_node_id`.
- Effects apply to the whole frame. You cannot select an object, a background or \
a region — that needs a click on the picture, which only the person can do. If \
they ask for "blur the background", apply the effect to the whole frame and tell \
them plainly that you blurred everything, and that they can use the Select \
control to limit it to an object.
- Deleting is not something you do. The delete tools only describe what would \
go; the person confirms with a button. Stop and say what you queued.

{context}

Work in the fewest steps that does the job, then reply in one or two plain \
sentences saying what you changed. No preamble, no lists of what you could do \
next."""


def _context_doc(context: dict) -> str:
    """The photo the person is looking at, as the model's starting point.

    Without this every request would open with "which photo?", when the answer is
    on screen. `image_id`/`node_id` come from `state.imageId`/`state.nodeId`, so
    "blur this" means the node whose render is on the preview right now.
    """
    image_id, node_id = context.get("image_id"), context.get("node_id")
    if not image_id or not node_id:
        return "No photo is open. Use find_photos to pick one before doing anything else."
    image = db.get_image(image_id)
    node = db.get_node(node_id)
    if image is None or node is None:
        return "No photo is open. Use find_photos to pick one before doing anything else."

    from . import main  # see `_dispatch`

    chain = main._effect_chain(node) or ["the original, unedited"]
    return (
        f"Open right now: photo '{image['name']}' (image_id {image_id}), viewing "
        f"node_id {node_id} — {' → '.join(chain)}. "
        "'this', 'it' and 'the photo' mean that node unless they name another photo."
    )


# ---------- Tools ----------

TOOLS = [
    {
        "name": "find_photos",
        "description": (
            "Search the whole library for photos matching a description of what "
            "they show. Returns the closest matches, closest first.\n\n"
            "This always returns something — it ranks the library against your "
            "words, it does not decide whether the photo you want is in there. "
            "`score` is the raw similarity and is the better of the two numbers "
            "for that: around 0.30 and up usually means the subject really is in "
            "the photo, around 0.25 usually means it is not and these are just "
            "the least-bad guesses. `z` is only rank within this library, so the "
            "top hit scores high even for a query nothing matches. When the "
            "scores are low or the request was vague, name what you found and "
            "let the person confirm instead of editing it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "What the photo shows, in plain words — 'a red boat on "
                        "water', 'snow on mountains'. Filenames are matched too."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "How many matches to return, 1 to 10. Default 5.",
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "show_photo",
        "description": (
            "Put a photo on screen. Call this when the person asked to see "
            "something rather than change it, and after finding a photo you are "
            "about to work on."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {"type": "integer"},
                "node_id": {
                    "type": "integer",
                    "description": "Which version to show. Defaults to the unedited original.",
                },
            },
            "required": ["image_id"],
        },
    },
    {
        "name": "list_photo_nodes",
        "description": (
            "List every version of a photo — the work tree. Each node is one "
            "effect applied to its parent; the root is the untouched original. "
            "Use this to find a node to build on, re-tune, or remove."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"image_id": {"type": "integer"}},
            "required": ["image_id"],
        },
    },
    {
        "name": "apply_effect",
        "description": (
            "Apply an effect to a version of a photo, creating a new version "
            "below it and putting it on screen. This is how you change what the "
            "person sees. Chain effects by applying the next one to the node_id "
            "this returns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {"type": "integer"},
                "node_id": {
                    "type": "integer",
                    "description": "The version to apply the effect to.",
                },
                "effect": {
                    "type": "string",
                    "description": "An effect name from the list in your instructions.",
                },
                "params": {
                    "type": "object",
                    "description": "Parameters for the effect. Omitted ones take their default.",
                },
                "second_node_id": {
                    "type": "integer",
                    "description": "For blend only: the other version to mix in.",
                },
            },
            "required": ["image_id", "node_id", "effect"],
        },
    },
    {
        "name": "edit_node",
        "description": (
            "Re-tune an existing version in place, keeping its id and anything "
            "built on top of it. Use this for 'make that stronger' or 'a bit "
            "less' — it changes the version you already made instead of stacking "
            "another one on it. Name only the parameters you are changing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "integer"},
                "params": {"type": "object"},
            },
            "required": ["node_id", "params"],
        },
    },
    {
        "name": "list_presets",
        "description": "List the saved effect chains the person can replay by name.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "apply_preset",
        "description": (
            "Replay a saved effect chain onto a version of a photo, creating one "
            "new version per step and showing the last one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "integer"},
                "preset_name": {"type": "string"},
            },
            "required": ["node_id", "preset_name"],
        },
    },
    {
        "name": "delete_node",
        "description": (
            "Queue the removal of one version and everything built on it. This "
            "does NOT delete — it asks the person to confirm with a button. Stop "
            "after calling it and say what you queued."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"node_id": {"type": "integer"}},
            "required": ["node_id"],
        },
    },
    {
        "name": "delete_photo",
        "description": (
            "Queue the removal of a whole photo, its original file and every "
            "version of it. This does NOT delete — it asks the person to confirm "
            "with a button. Stop after calling it and say what you queued."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"image_id": {"type": "integer"}},
            "required": ["image_id"],
        },
    },
]


class _Turn:
    """What one turn accumulated besides words.

    `focus` is where the browser should navigate when the turn ends — set by
    every tool that puts something on screen, last write wins, so it is wherever
    the model finished. `pending` is a destructive action waiting on a button.
    `steps` is the transcript the row prints, so the person can see what was
    done to their library rather than trusting a summary of it.
    """

    def __init__(self, context: dict):
        self.context = context
        self.focus: dict | None = None
        self.pending: dict | None = None
        self.steps: list[dict] = []


def _node_of(image_id: int, node_id: int | None) -> dict:
    """Resolve a node, defaulting to the image's original, with the ownership
    check every write endpoint makes — done here too so the model gets a
    sentence back instead of a 400 it has to interpret."""
    image = db.get_image(image_id)
    if image is None:
        raise _ToolError(f"there is no photo with image_id {image_id}")
    if node_id is None:
        tree = db.get_tree(image_id)
        root = next((n for n in tree if n["parent_id"] is None), None)
        if root is None:
            raise _ToolError(f"photo {image_id} has no versions")
        return root
    node = db.get_node(node_id)
    if node is None or node["image_id"] != image_id:
        raise _ToolError(f"node_id {node_id} is not a version of photo {image_id}")
    return node


def _chain(node: dict) -> str:
    from . import main  # see `_dispatch`

    return " → ".join(main._effect_chain(node)) or "original"


def _find_photos(args: dict, turn: _Turn) -> tuple[str, str]:
    from . import main

    description = str(args.get("description") or "").strip()
    if not description:
        raise _ToolError("description must not be empty")
    limit = max(1, min(10, int(args.get("limit") or 5)))

    images = db.list_images()
    if not images:
        raise _ToolError("the library is empty — there are no photos yet")
    by_id = {img["id"]: img for img in images}

    # Filenames first, and unconditionally: it costs nothing, needs no model, and
    # "the sunset one" is very often literally the filename. It is also the whole
    # answer while the text tower is still coming down.
    words = [w for w in description.lower().split() if len(w) > 2]
    named = [
        img for img in images if any(w in img["name"].lower() for w in words)
    ]

    scores: dict[str, dict] = {}
    note = ""
    try:
        # The search endpoint itself, not a reimplementation of it: this is where
        # the query encoding, the library's cached vectors and the z-scoring
        # live, and search and the Image map must agree about all three.
        scores = main.search_embedding_map(q=description)["scores"]
    except HTTPException as exc:
        if exc.status_code != 409:
            raise
        # The 254 MB text tower is fetched only when someone searches — and this
        # is someone searching. Start it and answer with what needs no model,
        # rather than holding a request open for a quarter-gigabyte download.
        job = text_job.start()
        pct = int(100 * job["done"] / job["total"]) if job.get("total") else 0
        note = (
            f" (searching by what photos look like needs a model that is still "
            f"downloading, {pct}% done — these are filename matches only; ask "
            f"them to try again shortly for a proper search)"
        )
        if not named:
            raise _ToolError(
                f"the photo search model is still downloading ({pct}% done) and no "
                f"filename matches '{description}'. Tell them to try again shortly."
            )

    ranked = sorted(
        ((int(k), v) for k, v in scores.items()),
        key=lambda kv: -kv[1]["z"],
    )
    hits, seen = [], set()
    for image_id, score in ranked[:limit]:
        image = by_id.get(image_id)
        if image is None:
            continue
        seen.add(image_id)
        hits.append(
            {
                "image_id": image_id,
                "name": image["name"],
                "original_node_id": image["root_node_id"],
                # Both numbers, because they answer different questions and only
                # one of them answers "is it actually in there" — see the tool's
                # description.
                "score": score["score"],
                "z": score["z"],
                "confidence": "strong" if score["score"] >= STRONG_MATCH else "weak",
                "already_applied": image["effects"],
            }
        )
    for image in named:
        if image["id"] in seen or len(hits) >= limit:
            continue
        hits.append(
            {
                "image_id": image["id"],
                "name": image["name"],
                "original_node_id": image["root_node_id"],
                "matched": "filename",
                "already_applied": image["effects"],
            }
        )

    if not hits:
        raise _ToolError(f"nothing in the library matches '{description}'")

    # The directive rides in the tool result, next to the numbers it is about,
    # because that is where the model is actually reading.
    if not any(h.get("confidence") == "strong" for h in hits):
        note = (
            " — NOTE: none of these is a confident match, so the library "
            "probably does not contain what was asked for. Do NOT edit any of "
            "them. Say what the closest photos are and ask which one they meant."
            + note
        )
    return json.dumps(hits) + note, f"searched for “{description}” · {len(hits)} match(es)"


def _show_photo(args: dict, turn: _Turn) -> tuple[str, str]:
    node = _node_of(int(args["image_id"]), args.get("node_id"))
    image = db.get_image(node["image_id"])
    turn.focus = {"image_id": node["image_id"], "node_id": node["id"]}
    return (
        json.dumps(
            {
                "showing": image["name"],
                "image_id": node["image_id"],
                "node_id": node["id"],
                "effects_applied": _chain(node),
            }
        ),
        f"showed {image['name']} · {_chain(node)}",
    )


def _list_photo_nodes(args: dict, turn: _Turn) -> tuple[str, str]:
    image_id = int(args["image_id"])
    if db.get_image(image_id) is None:
        raise _ToolError(f"there is no photo with image_id {image_id}")
    nodes = [
        {
            "node_id": n["id"],
            "built_on": n["parent_id"],
            "effect": n["effect"] or "the original",
            "params": n["params"],
        }
        for n in db.get_tree(image_id)
    ]
    return json.dumps(nodes), f"listed {len(nodes)} version(s)"


def _apply_effect(args: dict, turn: _Turn) -> tuple[str, str]:
    from . import main

    image_id = int(args["image_id"])
    effect = str(args.get("effect") or "")
    parent = _node_of(image_id, args.get("node_id"))
    params = args.get("params") or {}
    if not isinstance(params, dict):
        raise _ToolError("params must be an object")

    try:
        node = main.create_node(
            image_id,
            main.NodeCreate(
                parent_id=parent["id"],
                parent2_id=args.get("second_node_id"),
                effect=effect,
                params=params,
            ),
        )
    except HTTPException as exc:
        # Bokeh is unavailable rather than slow until its depth model is on disk,
        # and lighting the Bokeh button is what starts that download in the UI.
        # Asking for bokeh here is the same gesture, so start it the same way.
        if exc.status_code == 409 and effect == "bokeh":
            job = depth_job.start()
            pct = int(100 * job["done"] / job["total"]) if job.get("total") else 0
            raise _ToolError(
                f"bokeh needs a depth model that is still downloading ({pct}% done). "
                "Tell them to try again shortly."
            )
        raise _ToolError(str(exc.detail))

    turn.focus = {"image_id": image_id, "node_id": node["id"]}
    return (
        json.dumps(
            {
                "node_id": node["id"],
                "effect": effect,
                # The stored params, which is the point: `validate_params` clamps
                # silently, so this is the only place the difference between what
                # was asked for and what happened is visible.
                "params_stored": node["params"],
            }
        ),
        f"applied {effect} {_terse(node['params'])}",
    )


def _edit_node(args: dict, turn: _Turn) -> tuple[str, str]:
    from . import main

    node_id = int(args["node_id"])
    node = db.get_node(node_id)
    if node is None:
        raise _ToolError(f"there is no version with node_id {node_id}")
    if node["parent_id"] is None:
        raise _ToolError("the original has no settings to change")
    changes = args.get("params") or {}
    if not isinstance(changes, dict) or not changes:
        raise _ToolError("params must be a non-empty object")

    try:
        result = main.update_node(
            node_id,
            main.NodeUpdate(
                # Merged over the node's own params, not sent bare:
                # `validate_params` fills every missing key with its default, so
                # a bare {"radius": 12} would quietly reset the kernel too. The
                # tool promises "name only what you are changing", and this is
                # what makes that true.
                params={**(node["params"] or {}), **changes},
                # Carried, not defaulted: NodeUpdate's own defaults are None,
                # which would clear the node's selection and break a blend.
                parent2_id=node["parent2_id"],
                selection=node["selection"],
            ),
        )
    except HTTPException as exc:
        raise _ToolError(str(exc.detail))

    updated = result["node"]
    turn.focus = {"image_id": node["image_id"], "node_id": node_id}
    return (
        json.dumps(
            {
                "node_id": node_id,
                "effect": node["effect"],
                "params_stored": updated["params"],
                # `invalidated` counts the node itself; what the model wants to
                # know is how much downstream work it disturbed.
                "rerendered_below": max(0, len(result["invalidated"]) - 1),
            }
        ),
        f"re-tuned {node['effect']} {_terse(updated['params'])}",
    )


def _list_presets(args: dict, turn: _Turn) -> tuple[str, str]:
    from . import main

    presets = [
        {"name": p["name"], "steps": p["summary"]} for p in main.list_presets()
    ]
    if not presets:
        raise _ToolError("no presets have been saved yet")
    return json.dumps(presets), f"listed {len(presets)} preset(s)"


def _apply_preset(args: dict, turn: _Turn) -> tuple[str, str]:
    from . import main

    wanted = str(args.get("preset_name") or "").strip().lower()
    preset = next((p for p in db.list_presets() if p["name"].lower() == wanted), None)
    if preset is None:
        known = ", ".join(p["name"] for p in db.list_presets()) or "none"
        raise _ToolError(f"there is no preset named '{wanted}'. Saved presets: {known}")

    node_id = int(args["node_id"])
    node = db.get_node(node_id)
    if node is None:
        raise _ToolError(f"there is no version with node_id {node_id}")
    try:
        result = main.apply_preset(node_id, main.PresetApply(preset_id=preset["id"]))
    except HTTPException as exc:
        raise _ToolError(str(exc.detail))

    turn.focus = {
        "image_id": node["image_id"],
        "node_id": result["terminal_node_id"],
    }
    return (
        json.dumps(
            {
                "preset": preset["name"],
                "node_id": result["terminal_node_id"],
                "versions_added": len(result["created"]),
            }
        ),
        f"applied preset “{preset['name']}” · {len(result['created'])} step(s)",
    )


def _queue(turn: _Turn, action: str, target_id: int, description: str) -> tuple[str, str]:
    if turn.pending is not None:
        raise _ToolError("something is already waiting to be confirmed — one at a time")
    turn.pending = {"action": action, "id": target_id, "description": description}
    return (
        f"Queued, not done. {description}. The person must press a button to "
        "confirm. Stop here and tell them what is waiting.",
        f"queued: {description}",
    )


def _delete_node(args: dict, turn: _Turn) -> tuple[str, str]:
    node_id = int(args["node_id"])
    node = db.get_node(node_id)
    if node is None:
        raise _ToolError(f"there is no version with node_id {node_id}")
    if node["parent_id"] is None:
        raise _ToolError(
            "that is the original — removing it means removing the whole photo, "
            "which is delete_photo"
        )
    below = len(db.descendant_ids(node_id)) - 1
    tail = f" and the {below} version(s) built on it" if below > 0 else ""
    return _queue(
        turn, "delete_node", node_id, f"Remove version #{node_id} ({_chain(node)}){tail}"
    )


def _delete_photo(args: dict, turn: _Turn) -> tuple[str, str]:
    image_id = int(args["image_id"])
    image = db.get_image(image_id)
    if image is None:
        raise _ToolError(f"there is no photo with image_id {image_id}")
    count = len(db.get_tree(image_id))
    return _queue(
        turn,
        "delete_photo",
        image_id,
        f"Delete the photo '{image['name']}' and all {count} of its versions",
    )


HANDLERS = {
    "find_photos": _find_photos,
    "show_photo": _show_photo,
    "list_photo_nodes": _list_photo_nodes,
    "apply_effect": _apply_effect,
    "edit_node": _edit_node,
    "list_presets": _list_presets,
    "apply_preset": _apply_preset,
    "delete_node": _delete_node,
    "delete_photo": _delete_photo,
}


def _terse(params: dict | None) -> str:
    """Params for the one-line transcript — the row is a strip, not a report."""
    if not params:
        return ""
    return " ".join(
        f"{k}={v}" for k, v in params.items() if not isinstance(v, (list, dict))
    )


def _dispatch(name: str, args: dict, turn: _Turn) -> tuple[str, bool]:
    """Run one tool call, and never raise past here.

    `main` is imported inside the handlers rather than at module scope because
    `main` imports *this* module to mount the endpoints — a top-level import
    either way is a cycle. By the time any handler runs, `main` is fully loaded.
    """
    handler = HANDLERS.get(name)
    if handler is None:
        turn.steps.append({"tool": name, "summary": "unknown tool", "ok": False})
        return f"there is no tool called {name}", True
    try:
        text, summary = handler(args, turn)
    except _ToolError as exc:
        turn.steps.append({"tool": name, "summary": str(exc), "ok": False})
        return str(exc), True
    except Exception as exc:  # a bug here is still the model's to work around
        turn.steps.append({"tool": name, "summary": f"failed: {exc}", "ok": False})
        return f"that failed: {exc}", True
    turn.steps.append({"tool": name, "summary": summary, "ok": True})
    return text, False


# ---------- The loop ----------


def _trim(history: list) -> list:
    """The last few exchanges, cut on a boundary the API will accept.

    A user message carrying `tool_result` blocks is only valid after the
    assistant message that asked for them, so a blind tail slice can produce a
    400. Walk forward to the first plain user turn instead: worst case the whole
    transcript goes, which costs the model its memory of the conversation and
    nothing else.
    """
    tail = history[-MAX_HISTORY:]
    for i, message in enumerate(tail):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return tail[i:]
        if isinstance(content, list) and not any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            return tail[i:]
    return []


def run_turn(prompt: str, history: list, context: dict) -> dict:
    """One request from the box, played out to the end.

    Returns the reply, the transcript, the updated history for the client to
    hold, where to navigate, and any destructive action awaiting a button.
    """
    turn = _Turn(context)
    messages = _trim([dict(m) for m in history]) + [{"role": "user", "content": prompt}]
    system = SYSTEM.format(effects=_effects_doc(), context=_context_doc(context))
    client = _get_client()

    reply = ""
    for _ in range(MAX_ROUNDS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=TOOLS,
            messages=messages,
        )
        # Serialized to plain dicts because this list makes a round trip through
        # the browser as JSON; `exclude_none` drops the optional block fields the
        # API does not want echoed back.
        messages.append(
            {
                "role": "assistant",
                "content": [b.model_dump(exclude_none=True) for b in response.content],
            }
        )
        reply = "\n".join(b.text for b in response.content if b.type == "text").strip()

        if response.stop_reason != "tool_use":
            break

        # Every result in one user message. Splitting them across several is how
        # you teach the model to stop making parallel calls.
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            text, failed = _dispatch(block.name, block.input or {}, turn)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": text,
                    "is_error": failed,
                }
            )
        messages.append({"role": "user", "content": results})
    else:
        reply = (
            reply
            or "I went round in circles on that one — could you say it a different way?"
        )

    return {
        "reply": reply,
        "steps": turn.steps,
        "history": messages,
        "focus": turn.focus,
        "pending": turn.pending,
    }
