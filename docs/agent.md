# `server/agent.py` — the command row

A sentence typed into the bar's first row becomes calls to the app's own
endpoints. This document explains what is in the file, why it is shaped that
way, and how to give the assistant a new skill.

The one-line summary: **`agent.py` is a second front end over the API, not a
second implementation of the app.** Everything it writes re-enters `main.py`
through the very functions the browser posts to.

---

## 1. Where it sits

```
  #agent-input (web/app.js: runAgent)
        │  POST /api/agent  {prompt, history, image_id, node_id}
        ▼
  main.agent_turn ──────────► agent.run_turn(prompt, history, context)
                                    │
                                    │  client.messages.create(...)   ← Claude Haiku 4.5
                                    │  ◄── tool_use blocks
                                    ▼
                              agent._dispatch → HANDLERS[name]
                                    │
                                    ▼
                              main.create_node / update_node / apply_preset
                              main.select_text / create_mask / get_depth_at
                              main.search_embedding_map / list_presets
                                    │
                                    ▼
                              db · rendering · effects  (the ordinary path)
```

`main` imports `agent` to mount the two endpoints, so every handler imports
`main` **inside the function body**. The cycle is real; the deferred import is
the fix. By the time a handler runs, `main` is fully loaded.

The two endpoints are in `main.py`, not here:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/agent` | `{configured, model}`. `init()` asks once and leaves `#agent-section` hidden when there is no key — a box that can only fail is worse than no box. |
| `POST /api/agent` | Plays one turn. 409 with no key, 502 for an API error, surfacing the API's own complaint (a bad key, a rate limit and an overload each need something different from the user). |

---

## 2. Module map

### Configuration

| Name | What it is |
| --- | --- |
| `MODEL = "claude-haiku-4-5"` | Chosen for latency: a turn is a handful of tool calls and a sentence, and the row should answer while you are still looking at it. Note what is *absent* from the request — Haiku 4.5 takes neither `output_config.effort` nor adaptive thinking, so an Opus-shaped call copied in here would 400. |
| `MAX_TOKENS = 2048` | Per response. |
| `MAX_ROUNDS = 10` | Tool-loop bound. The longest honest chain is find → show → apply → apply, so hitting ten means the model is stuck, and saying so beats spinning. |
| `MAX_HISTORY = 20` | Messages kept from the posted-back transcript. |
| `STRONG_MATCH = 0.275` | Where a raw CLIP image-text cosine stops meaning "that subject is in this photo". Measured on this library: real matches land at 0.28–0.32, a query for something absent still lands at 0.24–0.27. |
| `load_dotenv(<repo>/.env)` | Addressed, not searched — uvicorn can be started from anywhere, and the default walk up from the cwd would find nothing, or someone else's file. |
| `configured()` | Read live rather than latched at import, so exporting the key and restarting is the whole fix. |
| `_get_client()` | Built on first use. `anthropic.Anthropic()` raises without a key, and this module is imported unconditionally — a server with no key must still serve every other endpoint. |

### What the model is told

`SYSTEM` is a template with two holes:

- `{effects}` ← `_effects_doc()` ← `effects.effect_specs()`, the same source
  `GET /api/effects` builds the frontend's controls from. **There is
  deliberately no tool schema per effect**: `apply_effect` takes a name and a
  free-form `params`, and `validate_params` remains the only thing that judges
  them. So a new effect is still one registry entry and nothing else.
  `_param_line()` renders one param, carrying its `label` — a range alone does
  not say which end is which; `focus: float 0.0..1.0` is silent on the thing
  "Focus (1 = nearest)" says.
- `{context}` ← `_context_doc(context)`, the photo on screen right now
  (`state.imageId` / `state.nodeId`), so "blur this" resolves without a round
  trip asking which photo.

The prose in `SYSTEM` covers the things that are policy rather than schema:
report the *stored* params because clamping is silent, `select_object` for
masks but `measure_focus` for bokeh's focus, several photos are a map and not a
list, and deleting is not something the model does.

### `_Turn`

One turn's side effects, beyond the words:

| Field | Meaning |
| --- | --- |
| `focus` | Where the browser should navigate when the turn ends — `{image_id, node_id}`. Every tool that puts something on screen writes it; last write wins, so it is wherever the model finished. |
| `map` | `{query, image_ids}` — a set of photos to open the Image map pinned to. Ids and not the query, so the map shows exactly what the transcript above it just named; the words only label the button that clears the filter. |
| `pending` | A destructive action awaiting a button. |
| `steps` | The one-line transcript the row prints, so the person sees what was done to their library rather than a summary of it. |

### The tools

`TOOLS` is the schema list sent to the API; `HANDLERS` maps each name to a
Python function. Both must list the same twelve names.

| Tool | Handler | Writes | Re-enters `main` via |
| --- | --- | --- | --- |
| `find_photos` | `_find_photos` | — | `search_embedding_map` (+ filename matching) |
| `show_photo` | `_show_photo` | `focus` | — (`db` reads) |
| `show_photos` | `_show_photos` | `map`, `focus` | — (`db` reads) |
| `list_photo_nodes` | `_list_photo_nodes` | — | — (`db.get_tree`) |
| `select_object` | `_select_object` | `focus`, a saved mask | `select_text` (SAM 3 saves the mask there); `create_mask` under CLIPSeg |
| `measure_focus` | `_measure_focus` | nothing at all | `select_text` with `method="clipseg"` → `get_depth_at` |
| `apply_effect` | `_apply_effect` | `focus`, a node | `create_node` |
| `edit_node` | `_edit_node` | `focus`, a node in place | `update_node` |
| `list_presets` | `_list_presets` | — | `list_presets` |
| `apply_preset` | `_apply_preset` | `focus`, a sub-DAG | `apply_preset` |
| `delete_node` | `_delete_node` | `pending` only | — |
| `delete_photo` | `_delete_photo` | `pending` only | — |

Shared helpers: `_node_of` (resolve a node, defaulting to the image's original,
with the ownership check every write endpoint makes), `_chain` (an effect chain
as prose), `_locate` (the words → `main.select_text` half that both naming
tools start with, so a 409 from either backend is handled in one place; its
`method` is the caller's, never the model's), `_mask_selection`
(`mask_id`/`invert_mask` → the selection dict the browser builds),
`_mask_where` (a stored selection as the phrase the transcript prints),
`_queue` (one pending action at a time), `_terse` (params for a one-line row).

### The loop

`run_turn()` builds **one** `request` dict and calls
`client.messages.create(**request)`, so the Wire dialog cannot show a request
that was not sent. `messages` is grown in place across rounds; each round's
copy is deep-copied into `wire["rounds"]` because a reference would show every
round holding the final transcript, which is precisely what that view exists to
tell apart.

Each round: send → record → append the assistant turn → stop if
`stop_reason != "tool_use"` → otherwise dispatch every `tool_use` block and
append **all** results in one user message (splitting them across several
messages is how you teach a model to stop making parallel calls).

`_trim()` cuts the posted-back history on a boundary the API accepts: a user
message carrying `tool_result` blocks is only valid after the assistant message
that asked for them, so it walks forward to the first plain user turn rather
than taking a blind tail slice. Worst case the whole transcript goes, which
costs the model its memory of the conversation and nothing else.

`run_turn` returns:

```json
{"reply": "...", "steps": [...], "history": [...],
 "focus": {...} | null, "map": {...} | null, "pending": {...} | null,
 "wire": {"call": {...}, "rounds": [...]}}
```

The browser (`runAgent`) prints `steps`, then `reply`, then navigates through
`selectImage()` — the single choke point every path that changes what you are
looking at goes through — then renders `pending`, then opens the map last,
because the map is a modal and a confirm button appended under a backdrop that
has already taken the screen is a button nobody sees.

---

## 3. The four invariants

Everything in the file follows from these. Break one and the file starts being
a second implementation.

**1. Writes go through `main`, never `db`/`rendering`.** Parent-ownership
checks, `validate_params`' clamping, `_check_effect_ready`'s 409, the
descendant sweep and the eager re-render all hold for free. Reaching past
`main` would be a second place that has to know the rules, and the first one to
drift would be the one nobody was testing.

**2. Refusal lives in the tool result, not the system prompt.** A CLIP search
ranks the library rather than judging it, so the top hit for gibberish scores
about as well as the top hit for a photo you own. Below `STRONG_MATCH`,
`_find_photos` still returns the photos but prepends a do-not-edit directive
*to the result*, next to the numbers it is about. Said only in the system
prompt, this lost to the model's enthusiasm every time and the model edited a
real library image. `_select_object` and `_measure_focus` refuse the same way,
in the result, when `select_text` reports a match that is not `confident`,
whichever backend found it.

**3. Coordinates are out of reach; names are not.** A click is a pixel the
model cannot see. `select_object` is the whole of its access to selections: it
runs the same text search the Select control's box runs and ends with a saved
mask, so the model passes a `mask_id` and never a position — and
`_check_selection` refuses an id belonging to another photo, so it cannot
invent one either. *Where* the mask is saved depends on the backend
`select_text` ran. SAM 3, the default, answers with pixels no click would
reproduce, so `select_text` saves them itself and returns a `mask_id`, which
`_select_object` reads off the result. CLIPSeg answers with a click-shaped
point, which `_select_object` freezes through `main.create_mask`. "Blur the background" is that mask plus
`invert_mask`. An object that cannot be named is still a whole-frame edit the
model is told to own up to.

Two asymmetries with the browser are deliberate: the mask is banked *here*
rather than at Apply (`bankSelection` exists because a live click selection
would be re-banked on every Apply; a tool call happens once and its result must
be a durable handle), and it is named after the words, because a `mask_id` is
all the model gets while the person has a thumbnail.

**4. Destructive tools propose; they never act.** `delete_node` /
`delete_photo` return a description and set `turn.pending`; the browser renders
the button and calls the ordinary `DELETE` itself (`renderAgentPending`). The
agent never holds that trigger — the manual path already asks
(`deleteNode`'s `confirm()`), and a misread sentence must not be able to
destroy work.

### Two supporting rules

**`_ToolError` is the only way to refuse.** It becomes a `tool_result` with
`is_error: true` and the turn continues; the model apologises or tries
something else. Raising past `_dispatch` would end the turn on a stack trace,
which is the wrong answer to "you asked for radius 500". `_dispatch` also
catches bare `Exception` — a bug here is still the model's to work around.

**Model gates start the download and say so.** Four tools ride on lazily
fetched weights, and each starts the job its own gesture would start in the UI:
`find_photos` → `text_job` (the 254 MB text tower is fetched only when someone
searches, and an agent search *is* someone searching), `apply_effect` and
`measure_focus` → `depth_job`, and the naming tools → `clipseg_job` when CLIPSeg
was the backend asked for. Each returns "still downloading, N% done" rather than
holding a request open for a quarter-gigabyte. SAM 3 is the exception: its
graphs come from an export someone runs by hand, so there is no job to start.
The tool passes the endpoint's own message, which carries the export command,
on to the model instead of a percentage.

**`measure_focus` always uses CLIPSeg.** Under SAM 3, `select_text` saves a
mask as a side effect, and a tool documented to write nothing must not leave
one behind just to read a depth. So `_measure_focus` calls
`_locate(args, method="clipseg")`, and its 409 is two different models:
CLIPSeg to find the thing and the depth net to measure it.

---

## 4. Adding a skill

"Skill" here means a new tool in the loop. Three cases, in increasing order of
work.

### Case 0 — a new *effect* needs no change here at all

Add the entry to `EFFECTS` in `server/effects.py`. `_effects_doc()` renders it
into the system prompt from `effect_specs()`, and `apply_effect` already takes
a name and a free-form `params`. That is the whole point of there being no tool
schema per effect: do not add one.

The same is true of a new preset — `list_presets` reads the table.

### Case 1 — a new tool over an endpoint that already exists

This is the normal case, and it is four edits in this file.

**a. Write the handler.** The signature is fixed:

```python
def _my_tool(args: dict, turn: _Turn) -> tuple[str, str]:
    """One line on what it does, and why it is not the browser's version of it."""
    from . import main  # deferred: main imports this module — see `_dispatch`

    image_id = int(args["image_id"])
    node = _node_of(image_id, args.get("node_id"))   # existence + ownership

    try:
        result = main.some_endpoint(node["id"], main.SomeRequest(...))
    except HTTPException as exc:
        raise _ToolError(str(exc.detail))            # a sentence, not a 500

    turn.focus = {"image_id": image_id, "node_id": node["id"]}   # if it changed the view
    return (
        json.dumps({"node_id": node["id"], "stored": result["params"]}),  # for the model
        f"did the thing · {node['id']}",                                  # for the row
    )
```

The two return values are for two different readers. The first is the
`tool_result` the model reads — JSON, and reporting **what was stored** rather
than what was asked for, since clamping is silent. The second is the one-line
summary the person reads in `#agent-log`; keep it short, it is a strip and not
a report.

**b. Add the schema to `TOOLS`.** Write the description for a model that
cannot see the picture. Say what the tool is *for* rather than what it does,
say when *not* to call it, and put any judgement the model has to make into
the description of the field it is about — the `find_photos` and
`select_object` entries are the worked examples. Prefer plain nouns the person
would use ("version" for a node, "photo" for an image).

**c. Register it in `HANDLERS`** under the same name.

**d. Decide what the turn carries back.** Set `turn.focus` if the tool changes
what is on screen; set `turn.map` for a set of photos; call `_queue` if it is
destructive. Set none of them if it only reads — `measure_focus` deliberately
writes nothing, not even `focus`, because nothing it does changes the view.

Then check the four boxes:

- [ ] Does it write through `main`, not `db`/`rendering`?
- [ ] Does every refusal raise `_ToolError` with a sentence the model can act on?
- [ ] Does it take names rather than coordinates?
- [ ] If it destroys anything, does it `_queue` instead of doing it?

And two more if they apply:

- [ ] Does it need a model that downloads lazily? Catch the 409, start the job,
      report the percentage.
- [ ] Is the judgement call in the *result* rather than only in `SYSTEM`?

### Case 2 — a new tool with no endpoint behind it

Add the endpoint to `main.py` first, and let the browser use it too if the
gesture exists there. A tool that reaches into `db`/`rendering` directly
because "there is no endpoint for it yet" is exactly the second implementation
this file is arranged to avoid — and it silently skips whatever `main` was
doing around that call, which is usually the ownership check and the cache
sweep.

If the capability genuinely has no UI (say, "how much disk is this library
using"), a read-only handler over `db` is fine — `_list_photo_nodes` is one.
The rule is about *writes*.

### Case 3 — a tool the browser has to finish

`show_photos` and the two delete tools are this shape: the handler sets a field
on `_Turn`, and `runAgent` in `web/app.js` does the rest. Adding one means

1. a new field on `_Turn` (documented in its docstring — that class is the
   contract),
2. including it in `run_turn`'s return dict,
3. handling it in `runAgent`, in the right order relative to `focus`,
   `pending` and `map` (see the comments there — modals go last),
4. and, if it needs UI, a renderer beside `renderAgentPending`.

Use this shape when the browser holds something the server cannot: the trigger
for a destructive action, or a view (the Image map) that only exists on the
client.

### What not to do

- **Do not add a tool schema per effect.** `_effects_doc()` exists so the
  registry stays the only declaration.
- **Do not give the model coordinates.** If a new tool wants a point, it wants
  a name plus `_locate`.
- **Do not put restraint in `SYSTEM` alone.** It loses.
- **Do not let a tool call `client.messages.create` itself.** One request dict,
  one place, or the Wire log stops being the truth.
- **Do not have the agent perform a delete**, however well-worded the
  confirmation in the reply is.

### Verifying

There is no test suite; exercise the endpoint:

```sh
curl -s localhost:8010/api/agent | python3 -m json.tool          # is a key configured
curl -s -X POST localhost:8010/api/agent \
  -H 'content-type: application/json' \
  -d '{"prompt":"blur this a little","image_id":1,"node_id":1}' \
  | python3 -m json.tool
```

Read `steps` for what ran and `wire.rounds[*].response` for what the model
actually asked for — the Wire dialog in the browser (`#agent-wire`) is the same
data, formatted. Run test instances on a spare port; port 8000 is often the
user's own server, and `data/` is a real library.
