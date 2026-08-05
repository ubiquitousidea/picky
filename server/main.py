"""Picky — browser-based JPG effects app."""

import io
import sqlite3
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from . import db, embed_job, labels, rendering, text_embed, text_job
from .effects import (
    EFFECTS,
    crop_geometry,
    effect_specs,
    validate_params,
    validate_selection,
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Picky")
db.init()


class NodeCreate(BaseModel):
    parent_id: int
    parent2_id: int | None = None
    effect: str
    params: dict = {}
    selection: dict | None = None


class NodeUpdate(BaseModel):
    params: dict = {}
    parent2_id: int | None = None
    selection: dict | None = None


class PreviewRequest(BaseModel):
    effect: str
    params: dict = {}
    parent2_id: int | None = None
    selection: dict | None = None


class MaskRequest(BaseModel):
    """Either shape of selection — a union of click specs or a union of saved
    mask ids. The fields are all optional because the total `validate_selection`
    is what actually decides which shape this is; the model is documentation.
    `x`/`y`/`level`/`mask` are the pre-union spellings, still accepted."""

    points: list[dict] | None = None
    masks: list[int] | None = None
    x: int | None = None
    y: int | None = None
    mask: int | None = None
    invert: bool = False
    level: str = "auto"


class MaskCreate(BaseModel):
    # omit `name` to have the server pick one — masks are identified by their
    # thumbnail now, so the name is only a tooltip and a uniqueness key
    name: str | None = None
    node_id: int
    selection: dict


class MaskUpdate(BaseModel):
    name: str


class PresetCreate(BaseModel):
    name: str
    node_id: int


class PresetApply(BaseModel):
    # the selection to run the whole recipe through, if any — see apply_preset
    preset_id: int
    selection: dict | None = None


class CropUpdate(BaseModel):
    # null clears the framing back to the whole image
    crop: dict | None = None


class CropPreviewRequest(BaseModel):
    node_id: int
    angle: float = 0.0


def _image_response(image: dict) -> dict:
    """An image plus the geometry its crop implies.

    The source size is the *original's*, which is also every node's: the crop is
    an output stage outside the work tree, so nothing in the tree changes
    dimensions. That is what lets one geometry describe every node of the image.

    Sent with the image rather than fetched separately because the frontend needs
    the `inverse` affine to place a click, and a second round trip for six
    numbers it will need on the very next interaction is a round trip wasted.
    """
    with Image.open(rendering.original_path(image["id"])) as im:
        size = im.size  # header-only; no decode
    return {**image, "geometry": crop_geometry(size, image["crop"])}


def _check_selection(selection: dict | None, image_id: int) -> dict | None:
    """Resolve saved-mask references. `validate_selection` cannot do this —
    effects.py imports no DB — so ownership becomes a 400 here, mirroring the
    blend target check. A mask is scoped to the image it was picked on, which is
    what guarantees its frozen pixels match every node it can be used on. Every
    member of the union is checked: one foreign mask poisons the whole union,
    since they are OR'd into a single mask of one image's dimensions."""
    if selection is not None and "masks" in selection:
        for mask_id in selection["masks"]:
            mask = db.get_mask(mask_id)
            if mask is None or mask["image_id"] != image_id:
                raise HTTPException(400, "mask does not belong to this image")
    return selection


@app.get("/api/effects")
def get_effects():
    return effect_specs()


@app.post("/api/images")
async def upload_image(file: UploadFile):
    data = await file.read()
    try:
        im = Image.open(io.BytesIO(data))
        im.verify()
    except Exception:
        raise HTTPException(400, "not a valid image file")
    if im.format != "JPEG":
        raise HTTPException(400, f"only JPEG images are supported (got {im.format})")

    # A filename already in the library is a re-import: `create_image` hands back
    # the existing row and nothing is written. Skipping the write is the point,
    # not an optimization — `original_path` is keyed by image id, so writing here
    # would replace *that* image's original while its renders, thumbs and
    # `.out.jpg` stayed on disk, and file existence is the entire render cache
    # key. Those stale pixels would then be served as a valid cache hit forever.
    # The 200 (rather than a 409) is what keeps a skip out of the caller's
    # failure path: it is the feature working, and the image it names is real.
    image, created = db.create_image(file.filename or "untitled.jpg")
    if created:
        rendering.original_path(image["id"]).write_bytes(data)
    return {**image, "duplicate": not created}


@app.get("/api/images")
def list_images():
    return db.list_images()


EMBED_METHODS = ("pca", "tsne")


def _project(vectors, method: str):
    """Project unit-length CLIP vectors down to 3 dimensions for display.

    PCA is the default because it is *stable*: adding an image nudges the map
    instead of reshuffling it, and the axes stay real directions of variation.
    t-SNE draws tighter, more obvious clumps, but its layout is not comparable
    across library changes and between-cluster distances mean nothing.

    Both are guarded for tiny libraries, where the defaults raise: PCA cannot
    ask for more components than it has samples, and t-SNE's perplexity must be
    under the sample count. Below four images there is no neighbourhood
    structure to find at all, so t-SNE degrades to PCA rather than erroring.
    """
    from sklearn.decomposition import PCA

    n = len(vectors)
    if n == 1:
        return np.zeros((1, 3), dtype=np.float32)

    components = min(3, n, vectors.shape[1])
    coords = PCA(n_components=components).fit_transform(vectors)

    if method == "tsne" and n >= 4:
        from sklearn.manifold import TSNE

        # PCA first: t-SNE on 512 raw dims is slow and noisier than on the
        # handful of components that carry the variance.
        reduced = PCA(n_components=min(50, n, vectors.shape[1])).fit_transform(vectors)
        coords = TSNE(
            n_components=3,
            # seeded, so reopening the map does not reshuffle it
            random_state=0,
            init="pca",
            # sklearn requires 0 < perplexity < n_samples, so the usual 30 has
            # to scale down for a small library — (n-1)/3 stays under n for
            # every n >= 2, where a constant floor would not (a floor of 5 is
            # already illegal at n = 4).
            perplexity=min(30.0, max(1.0, (n - 1) / 3)),
        ).fit_transform(reduced)

    # zero-pad when PCA had fewer than 3 components to give
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))

    # Normalize into the unit cube the frontend frames, so the view never has to
    # guess a scale. A single point (or identical images) collapses to the
    # origin, hence the guard.
    extent = float(np.abs(coords).max())
    return coords / extent if extent > 0 else coords


MAX_CLUSTERS = 20


def _auto_k(n: int) -> int:
    """How many clusters to label a library of `n` images with.

    Grows like sqrt(n) rather than linearly because the labels are read as a
    legend over the cloud, not as an index of it: past a dozen they overlap, and
    a cluster too small to see is a label pointing at nothing. ~8 at a hundred
    images, still 12 at a thousand.
    """
    return max(2, min(12, round((n / 2) ** 0.5), n))


def _cluster(vectors, k: int):
    """Group the library in CLIP space, and name each group.

    Clustering happens in the full 512-d embedding, *not* in the 3 dimensions
    `_project` returns: the projection is a lossy shadow (PCA over CLIP keeps a
    fraction of the variance, and t-SNE's between-cluster distances mean
    nothing), so grouping there would name a group that only looks like one from
    the current camera angle. Grouping here means the labels answer "what do
    these photos have in common", which is the question.

    Seeded, for exactly the reason `_project`'s t-SNE is: the map is looked at
    repeatedly, and a library that reshuffles its labels between two identical
    requests reads as broken.
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(vectors)
    # A mean of unit vectors is not one, and `label_clusters` needs the dot
    # product against the labels to be a cosine.
    centroids = km.cluster_centers_.astype(np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = np.divide(centroids, norms, out=np.zeros_like(centroids), where=norms > 0)
    return km.labels_, labels.label_clusters(centroids)


def _collect_embeddings(images: list[dict]):
    """Each image's vector (cached on its row, or computed now) and its point."""
    cached = db.get_embeddings()
    vectors, points = [], []
    for image in images:
        blob = cached.get(image["id"])
        if blob:
            vec = np.frombuffer(blob, dtype=np.float32)
        else:
            try:
                vec = rendering.embed_image(image["id"], image["root_node_id"])
            except Exception as exc:
                raise HTTPException(500, f"could not embed {image['name']}: {exc}")
        vectors.append(vec)
        points.append(
            {
                "image_id": image["id"],
                "node_id": image["root_node_id"],
                "name": image["name"],
                "color": rendering.mean_color(image["root_node_id"]),
            }
        )
    return vectors, points


def _library_vectors(images: list[dict]):
    """`_collect_embeddings`, guaranteed rectangular.

    Every vector must be the model's output width for the stack to be a matrix.
    A mixed cache means the ONNX export changed under it (a 512-d projection
    head vs a 768-d pooled output), so re-embed the library once rather than 500
    — the map is a view, and a stale cache is not a reason to refuse to draw it.

    Shared by the map and by search so the two cannot disagree about the
    library: search reaches this with everything cached, but if it ever did not,
    silently recovering in one place beats a `np.vstack` ValueError in the other.
    """
    vectors, points = _collect_embeddings(images)
    if len({v.shape[0] for v in vectors}) > 1:
        for image in images:
            db.clear_embedding(image["id"])
        vectors, points = _collect_embeddings(images)
    return np.vstack(vectors).astype(np.float32), points


@app.get("/api/embedding-map")
def embedding_map(method: str = "pca", clusters: int = 0):
    """Every image as a point in 3D — the Image map's data.

    Library-scoped, unlike every other endpoint here, and necessarily so: the
    projection is fitted across the whole library, so no point's coordinates are
    a property of its own image. That is why only the 512-d vectors are cached
    (on the image row) and the projection is redone per request — it is a
    sub-second fit over ~100 points, against embeddings that are the expensive
    part and are computed once.

    Any vector this finds missing it computes on the spot, so the map is correct
    regardless of whether anyone called `prepare` first. That is deliberately
    left in place: it makes the endpoint answerable on its own, and it means the
    prepare job is only ever an optimization that can be skipped or fail.

    `clusters` is how many groups to label, or 0 to let `_auto_k` decide from the
    library size — which is what the frontend sends until you touch the slider,
    so the control can be initialized from the answer rather than duplicating the
    formula.

    Clustering and labelling are recomputed per request alongside the projection
    and deliberately not cached. Both are milliseconds at this scale (k-means
    over ~100x512, then one ~850x512 matrix product), and a cache keyed on
    library state would be a second thing that can disagree with the map.
    """
    if method not in EMBED_METHODS:
        raise HTTPException(400, f"unknown method {method!r}")
    if clusters and not 2 <= clusters <= MAX_CLUSTERS:
        raise HTTPException(400, f"clusters must be 0 or 2..{MAX_CLUSTERS}")

    images = db.list_images()
    if not images:
        return {"method": method, "points": [], "clusters": []}

    stacked, points = _library_vectors(images)
    coords = _project(stacked, method)
    for point, (x, y, z) in zip(points, coords):
        point["x"], point["y"], point["z"] = float(x), float(y), float(z)

    k = min(clusters or _auto_k(len(points)), len(points))
    if k < 2:
        # One image is not a group, and has its own thumbnail to look at.
        return {"method": method, "points": points, "clusters": []}
    assignments, named = _cluster(stacked, k)

    groups = []
    for index, name in enumerate(named):
        members = coords[assignments == index]
        # The label's position is the mean of its members' *projected* coords,
        # not the projection of its 512-d centroid. For PCA the two are the same
        # point (projection is linear); for t-SNE the second does not exist at
        # all, since t-SNE has no out-of-sample transform. Averaging afterwards
        # is the only definition that works for both, and it is the one that
        # puts the label where the pictures are.
        centre = members.mean(axis=0)
        groups.append(
            {
                "id": index,
                "label": name["label"],
                "score": name["score"],
                "count": int(len(members)),
                "x": float(centre[0]),
                "y": float(centre[1]),
                "z": float(centre[2]),
            }
        )
    for point, index in zip(points, assignments):
        point["cluster"] = int(index)
    return {"method": method, "points": points, "clusters": groups}


@app.get("/api/embedding-map/search")
def search_embedding_map(q: str):
    """Score every image against a typed phrase — the Image map's search box.

    The same trick `labels.py` names clusters with, run the other way. CLIP put
    captions and the photos they describe in one 512-d space, so one dot product
    between a query vector and the library's cached image vectors is the whole
    search: no index, no per-image model run, sub-millisecond over a few hundred
    images.

    **Scores come back as z-scores, and that is the point.** A raw CLIP
    image-text cosine sits in a narrow band around 0.2 whose *position* moves
    with how the query was phrased, so "keep everything above 0.28" means
    something different for every query and nothing at all to a user. Measured
    in standard deviations above this query's own mean over this library, one
    slider position means roughly one strictness whatever you type. The raw
    cosine rides along anyway, because it is what makes the endpoint legible
    from curl.

    Unlike `embedding_map`, this refuses rather than doing the slow thing
    itself: that endpoint embeds whatever it finds missing, but the equivalent
    here would be downloading 254 MB inside a GET. So a library that has never
    searched gets a 409 and the frontend runs `text_job` — see that module.
    """
    query = q.strip()
    if not query:
        raise HTTPException(400, "q must not be empty")
    if not text_embed.ready():
        # 409, not 503: the state is wrong rather than the server unwell, and it
        # is the caller's `POST /api/text-model/prepare` that fixes it.
        raise HTTPException(409, "the text model is not downloaded yet")

    images = db.list_images()
    if not images:
        return {"query": query, "mean": 0.0, "std": 0.0, "scores": {}}

    # `_library_vectors` rather than a query of its own: it is all cache hits in
    # practice, since search is only reachable from an already-open map, but
    # reusing it is what stops search and the map disagreeing about which images
    # exist or what their vectors are.
    stacked, points = _library_vectors(images)

    try:
        vec = text_embed.encode_query(query)
    except Exception as exc:
        raise HTTPException(500, f"could not encode the query: {exc}")

    if stacked.shape[1] != vec.shape[0]:
        # Not the joint space — `labels.py`'s docstring explains the whole trap,
        # and this is the same answer: say nothing rather than something wrong.
        print(
            f"picky: query is {vec.shape[0]}-d but embeddings are "
            f"{stacked.shape[1]}-d; skipping search"
        )
        return {"query": query, "mean": 0.0, "std": 0.0, "scores": {}}

    # Both sides are unit length, so the dot product is the cosine.
    scores = stacked @ vec
    mean = float(scores.mean())
    std = float(scores.std())
    # A library of one image, or of identical ones, has no spread to measure
    # against; every z of 0 filters to "everything", which is the honest answer.
    zs = (scores - mean) / std if std > 0 else np.zeros_like(scores)
    return {
        "query": query,
        "mean": mean,
        "std": std,
        "scores": {
            str(point["image_id"]): {"score": round(float(s), 4), "z": round(float(z), 3)}
            for point, s, z in zip(points, scores, zs)
        },
    }


@app.post("/api/text-model/prepare")
def prepare_text_model():
    """Start fetching CLIP's text tower, and report where it already is.

    Search's counterpart to `/api/embedding-map/prepare`, and called at a
    deliberately different moment: the map's prepare runs on every open, this
    one only when the search box is first used. The tower is 254 MB and most
    sessions never search, so nothing but a keystroke may trigger it.
    """
    return text_job.start()


@app.get("/api/text-model/progress")
def text_model_progress():
    """Where the fetch from `prepare` got to. Safe to call at any time; a server
    nobody has searched on this run reports `idle` with `ready` already true if
    an earlier run downloaded the files."""
    return text_job.snapshot()


@app.post("/api/embedding-map/prepare")
def prepare_embedding_map():
    """Start the background embedding pass, and report where it already is.

    The point is that the first open of the Image map is not an unexplained
    wait: the ~335 MB model download and the pass over the library happen off
    the request, and the modal polls `/progress` for a status line rather than
    holding one connection open for a minute with nothing to show.

    The frontend calls this unconditionally instead of guessing whether it is
    needed — when everything is cached this answers `done` synchronously, so
    the usual case costs one round trip and no poll.
    """
    return embed_job.start()


@app.get("/api/embedding-map/progress")
def embedding_map_progress():
    """Where the pass from `prepare` got to. Safe to call at any time; a library
    nobody has prepared this run reports `idle`."""
    return embed_job.snapshot()


@app.get("/api/images/{image_id}")
def get_image(image_id: int):
    image = db.get_image(image_id)
    if image is None:
        raise HTTPException(404, "image not found")
    return _image_response(image)


@app.put("/api/images/{image_id}/crop")
def set_crop(image_id: int, body: CropUpdate):
    """Set the image's one framing — applied after every node, on the way out.

    Invalidation is deliberately narrow: only the output stage's cache goes
    (`.out.jpg` and `.thumb.jpg`). The effect renders upstream never saw the
    crop, so re-framing must not throw away a subtree of k-means work — that is
    the whole reason the crop is not a node.
    """
    image = db.get_image(image_id)
    if image is None:
        raise HTTPException(404, "image not found")
    crop = validate_params("crop", body.crop) if body.crop else None
    image = db.set_image_crop(image_id, crop)
    rendering.delete_output_files(db.image_node_ids(image_id))
    # The Image map embeds the *framed* thumbnail, so re-framing moves the point:
    # crop to one detail and the photo is semantically a different photo. This is
    # the only edge that stales an embedding — a delete takes it with the row.
    db.clear_embedding(image_id)
    return _image_response(image)


@app.post("/api/images/{image_id}/crop-preview")
def crop_preview(image_id: int, body: CropPreviewRequest):
    """The rotated-but-unframed backdrop the frame editor drags over.

    The full-size rotated canvas rides along in headers, because the proxy is
    downscaled and the editor's pixel readout must report what will actually
    export. Headers rather than a second endpoint: the readout changes on every
    *frame* drag, but the canvas only on a change of *angle* — so one number
    pair per proxy covers every repaint with no extra round trip.
    """
    node = db.get_node(body.node_id)
    if node is None or node["image_id"] != image_id:
        raise HTTPException(400, "node does not belong to this image")
    angle = validate_params("crop", {"angle": body.angle})["angle"]
    try:
        data, canvas = rendering.crop_preview(body.node_id, {"angle": angle})
    except Exception as exc:
        raise HTTPException(500, f"crop preview failed: {exc}")
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"X-Canvas-Width": str(canvas[0]), "X-Canvas-Height": str(canvas[1])},
    )


@app.get("/api/images/{image_id}/tree")
def get_tree(image_id: int):
    if db.get_image(image_id) is None:
        raise HTTPException(404, "image not found")
    return db.get_tree(image_id)


@app.post("/api/images/{image_id}/nodes")
def create_node(image_id: int, body: NodeCreate):
    if db.get_image(image_id) is None:
        raise HTTPException(404, "image not found")
    parent = db.get_node(body.parent_id)
    if parent is None or parent["image_id"] != image_id:
        raise HTTPException(400, "parent node does not belong to this image")
    if body.effect not in EFFECTS and body.effect != "blend":
        raise HTTPException(400, f"unknown effect '{body.effect}'")
    params = validate_params(body.effect, body.params)
    selection = _check_selection(validate_selection(body.selection), image_id)

    parent2_id = None
    if body.effect == "blend":
        if body.parent2_id is None:
            raise HTTPException(400, "blend requires a second node (parent2_id)")
        other = db.get_node(body.parent2_id)
        if other is None or other["image_id"] != image_id:
            raise HTTPException(400, "blend node does not belong to this image")
        parent2_id = body.parent2_id

    node = db.create_node(
        image_id, body.parent_id, body.effect, params, parent2_id, selection
    )
    try:
        rendering.render_node(node["id"])
    except Exception as exc:
        raise HTTPException(500, f"render failed: {exc}")
    return node


@app.patch("/api/nodes/{node_id}")
def update_node(node_id: int, body: NodeUpdate):
    """Change an existing node's settings in place. Unlike Apply, this does not
    add a node — the node keeps its id and its children, which are re-rendered
    from the new pixels."""
    node = db.get_node(node_id)
    if node is None:
        raise HTTPException(404, "node not found")
    if node["parent_id"] is None:
        raise HTTPException(400, "the original has no parameters to edit")
    effect = node["effect"]
    if effect not in EFFECTS and effect != "blend":
        raise HTTPException(400, f"effect '{effect}' no longer exists")
    params = validate_params(effect, body.params)
    selection = _check_selection(validate_selection(body.selection), node["image_id"])

    parent2_id = None
    if effect == "blend":
        if body.parent2_id is None:
            raise HTTPException(400, "blend requires a second node (parent2_id)")
        other = db.get_node(body.parent2_id)
        if other is None or other["image_id"] != node["image_id"]:
            raise HTTPException(400, "blend node does not belong to this image")
        # Ids are topological, so anything with a smaller id cannot be downstream
        # of this node. Creation gets that for free (a new node always has the
        # largest id); an edit does not, and pointing a blend at its own
        # descendant would send render_node into infinite recursion.
        if body.parent2_id >= node_id:
            raise HTTPException(400, "a blend cannot use a node derived from itself")
        parent2_id = body.parent2_id

    if (
        params == node["params"]
        and parent2_id == node["parent2_id"]
        and selection == node["selection"]
    ):
        return {"node": node, "invalidated": []}

    # Update first, then invalidate: these endpoints run concurrently in a
    # threadpool, and after the commit any re-render already uses the new params,
    # so the sweep below can only be removing files written from the old ones.
    # The reverse order would let a concurrent read repopulate a stale cache.
    updated = db.update_node_params(node_id, params, parent2_id, selection)
    invalidated = db.descendant_ids(node_id)
    rendering.delete_render_files(invalidated)
    try:
        rendering.render_node(node_id)  # fail loudly here rather than on next GET
    except Exception as exc:
        raise HTTPException(500, f"render failed: {exc}")
    return {"node": updated, "invalidated": invalidated}


def _effect_chain(node: dict) -> list[str]:
    parts = []
    cur = node
    while cur and cur["effect"]:
        parts.append(cur["effect"])
        cur = db.get_node(cur["parent_id"])
    return list(reversed(parts))


@app.get("/api/nodes/{node_id}/render")
def get_render(node_id: int, thumb: bool = False, download: bool = False):
    node = db.get_node(node_id)
    if node is None:
        raise HTTPException(404, "node not found")
    # render_output, not render_node: what a viewer gets is the framed result,
    # which is what makes Export framed for free. render_thumb frames too.
    path = rendering.render_thumb(node_id) if thumb else rendering.render_output(node_id)
    if download:
        image = db.get_image(node["image_id"])
        stem = Path(image["name"]).stem or "image"
        suffix = "-".join(_effect_chain(node)) or "original"
        return FileResponse(
            path, media_type="image/jpeg", filename=f"{stem}-{suffix}.jpg"
        )
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/nodes/{node_id}/preview")
def preview_node(node_id: int, body: PreviewRequest):
    node = db.get_node(node_id)
    if node is None:
        raise HTTPException(404, "node not found")
    if body.effect not in EFFECTS and body.effect != "blend":
        raise HTTPException(400, f"unknown effect '{body.effect}'")
    params = validate_params(body.effect, body.params)
    selection = _check_selection(validate_selection(body.selection), node["image_id"])

    parent2_id = None
    if body.effect == "blend":
        if body.parent2_id is None:
            raise HTTPException(400, "blend requires a second node (parent2_id)")
        other = db.get_node(body.parent2_id)
        if other is None or other["image_id"] != node["image_id"]:
            raise HTTPException(400, "blend node does not belong to this image")
        parent2_id = body.parent2_id

    try:
        data = rendering.render_preview(node_id, body.effect, params, parent2_id, selection)
    except Exception as exc:
        raise HTTPException(500, f"preview failed: {exc}")
    return Response(content=data, media_type="image/jpeg")


@app.post("/api/nodes/{node_id}/mask")
def get_mask(node_id: int, body: MaskRequest):
    """The overlay PNG for a selection on a node's pixels — a fresh click, which
    persists nothing but the node's cached embedding, or a saved mask, which is
    just read back. One endpoint for both is what keeps the frontend's overlay
    code free of branches."""
    node = db.get_node(node_id)
    if node is None:
        raise HTTPException(404, "node not found")
    selection = _check_selection(validate_selection(body.model_dump()), node["image_id"])
    if selection is None:
        raise HTTPException(400, "a selection needs either a click point or a mask id")
    try:
        data = rendering.mask_png(node_id, selection)
    except Exception as exc:
        raise HTTPException(500, f"segmentation failed: {exc}")
    return Response(content=data, media_type="image/png")


@app.get("/api/nodes/{node_id}/clusters")
def get_clusters(node_id: int):
    node = db.get_node(node_id)
    if node is None:
        raise HTTPException(404, "node not found")
    if node["effect"] != "posterize":
        raise HTTPException(400, "cluster data only exists for posterize nodes")
    return FileResponse(rendering.cluster_data(node_id), media_type="application/json")


@app.get("/api/nodes/{node_id}/histogram")
def get_histogram(node_id: int):
    if db.get_node(node_id) is None:
        raise HTTPException(404, "node not found")
    return rendering.histogram(node_id)


@app.delete("/api/nodes/{node_id}")
def delete_node(node_id: int):
    node = db.get_node(node_id)
    if node is None:
        raise HTTPException(404, "node not found")
    if node["parent_id"] is None:
        raise HTTPException(400, "cannot delete the original; delete the image instead")
    node_ids = db.delete_node(node_id)
    rendering.delete_render_files(node_ids)
    return {"deleted": node_ids, "parent_id": node["parent_id"]}


@app.delete("/api/images/{image_id}")
def delete_image(image_id: int):
    if db.get_image(image_id) is None:
        raise HTTPException(404, "image not found")
    # collect the mask ids before the cascade removes their rows; only the files
    # need a hand, the rows go with ON DELETE CASCADE
    mask_ids = [m["id"] for m in db.list_masks(image_id)]
    node_ids = db.delete_image(image_id)
    rendering.delete_render_files(node_ids)
    rendering.original_path(image_id).unlink(missing_ok=True)
    for mask_id in mask_ids:
        rendering.mask_path(mask_id).unlink(missing_ok=True)
    return {"deleted": image_id}


# ---------- Saved masks ----------
#
# A mask freezes a segmentation to a PNG so reuse never re-runs SAM: the shape is
# byte-identical everywhere it is used, and it outlives the node it was picked
# on. It belongs to the image it was picked on and is offered on every node of
# that image's tree — every node of an image shares its dimensions, so the frozen
# pixels always line up.


@app.get("/api/images/{image_id}/masks")
def list_masks(image_id: int):
    if db.get_image(image_id) is None:
        raise HTTPException(404, "image not found")
    # used_by rides along so the UI can warn before a delete is refused. One
    # joined query, not one per mask: banking a selection on every Apply means
    # this list grows with use, and it is refetched on every image switch.
    return db.list_masks(image_id, with_use_counts=True)


def _auto_mask_name(image_id: int) -> str:
    """`Object 1`, `Object 2`, … — the lowest ordinal this image is not using.

    Masks name themselves because the user never has to think about the name:
    the grid identifies them by thumbnail, so a name only has to satisfy the
    masks(image_id, name) index and read sensibly in a tooltip. Reusing a freed
    ordinal (delete `Object 2`, next save takes it back) is deliberate — the
    numbers stay small and they are not identifiers, the ids are.
    """
    taken = {m["name"] for m in db.list_masks(image_id)}
    n = 1
    while f"Object {n}" in taken:
        n += 1
    return f"Object {n}"


# an auto-named save races another one only if two Applies land in the same
# instant; these endpoints are sync `def`, so FastAPI runs them concurrently in
# a threadpool and that is a real, if rare, window
AUTO_NAME_TRIES = 5


@app.post("/api/images/{image_id}/masks")
def create_mask(image_id: int, body: MaskCreate):
    if db.get_image(image_id) is None:
        raise HTTPException(404, "image not found")
    name = None if body.name is None else body.name.strip()
    if name == "":
        # an explicitly empty name is a client bug, not a request to auto-name
        raise HTTPException(400, "mask name cannot be empty")
    node = db.get_node(body.node_id)
    if node is None or node["image_id"] != image_id:
        raise HTTPException(400, "node does not belong to this image")
    spec = validate_selection(body.selection)
    if spec is None or "masks" in spec:
        raise HTTPException(400, "a mask is saved from a click selection")

    try:
        # compute_mask already applies spec's invert, so what gets frozen is
        # exactly what the user was looking at. A reference's own invert then
        # toggles on top of that — see the XOR in capture_steps.
        mask = rendering.compute_mask(body.node_id, spec)
    except Exception as exc:
        raise HTTPException(500, f"segmentation failed: {exc}")

    height, width = mask.shape
    # a caller-supplied name is taken literally and collides once; an auto-name
    # re-derives and retries, since losing the race just means picking again
    for attempt in range(AUTO_NAME_TRIES if name is None else 1):
        chosen = name if name is not None else _auto_mask_name(image_id)
        try:
            saved = db.create_mask(image_id, chosen, body.node_id, spec, width, height)
            break
        except sqlite3.IntegrityError:
            if name is not None or attempt == AUTO_NAME_TRIES - 1:
                raise HTTPException(
                    409, f"a mask named '{chosen}' already exists for this image"
                )
    # row first so the file is named by the id; unwind it if the write fails,
    # so a row can never point at a missing PNG
    try:
        rendering.save_mask(saved["id"], mask)
    except Exception as exc:
        db.delete_mask(saved["id"])
        raise HTTPException(500, f"saving the mask failed: {exc}")
    return {**saved, "used_by": 0}


@app.patch("/api/masks/{mask_id}")
def update_mask(mask_id: int, body: MaskUpdate):
    """Rename. Presets have no equivalent because re-saving one is free; redoing
    a mask means re-picking the object, which is the cost this feature exists to
    remove."""
    mask = db.get_mask(mask_id)
    if mask is None:
        raise HTTPException(404, "mask not found")
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "mask name cannot be empty")
    try:
        return db.rename_mask(mask_id, name)
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"a mask named '{name}' already exists for this image")


@app.delete("/api/masks/{mask_id}")
def delete_mask(mask_id: int):
    """Refuse while nodes still reference it. Clearing their selections instead
    would silently change committed pixels and force a re-render of every one of
    their descendants — a large invisible side effect behind an '×'."""
    if db.get_mask(mask_id) is None:
        raise HTTPException(404, "mask not found")
    users = db.nodes_using_mask(mask_id)
    if users:
        raise HTTPException(
            409,
            f"{len(users)} node{'s' if len(users) > 1 else ''}"
            f" still use{'s' if len(users) == 1 else ''} this mask"
            f" (#{', #'.join(str(i) for i in users)}) — delete them first",
        )
    db.delete_mask(mask_id)
    rendering.mask_path(mask_id).unlink(missing_ok=True)
    return {"deleted": mask_id}


@app.get("/api/masks/{mask_id}/thumb")
def get_mask_thumb(mask_id: int):
    """A mask's silhouette as a small PNG — the icon in the selection grid.

    Computed per request rather than cached (see `mask_thumb_png`). The one
    cache that *is* worth having is the browser's, and that is also where the
    rowid-reuse hazard is unfixable — nothing can unlink a cache entry — so the
    frontend versions the URL with the mask's `created_at` and this is
    `immutable`. That combination is what keeps the grid rebuild that
    `renderSelectControls` does on every selection change from refetching
    every icon.
    """
    if db.get_mask(mask_id) is None:
        raise HTTPException(404, "mask not found")
    try:
        data = rendering.mask_thumb_png(mask_id)
    except Exception as exc:
        raise HTTPException(500, f"thumbnail failed: {exc}")
    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


MAX_PRESET_STEPS = 100


def _portable_selection(selection: dict | None) -> dict | None:
    """A selection as a preset can carry it: click specs, never saved masks.

    A preset stores params, not pixels — replaying one runs today's effect code
    against another image. A saved mask *is* pixels, and image-scoped ones at
    that, so it degrades back to the click specs it was frozen from and gets
    re-segmented on the target. Dropping the selection instead would silently
    turn a masked blur into a whole-image blur. This is the reason the click
    shape is a union too: a union of masks has nowhere else to land.

    Inverts are the lossy part. With a single mask they XOR exactly: the PNG was
    frozen post-invert, so the spec's own invert already produced the saved shape
    and the reference's invert toggles it again. With several, there is no single
    invert that reproduces the union — an inverted region cannot join a union of
    points — so a mask whose spec was inverted is dropped from the step. The
    remaining masks still replay; the alternative is discarding the whole
    selection, which is the failure mode this function exists to avoid.

    A mask deleted since (or one with no stored spec) is skipped for the same
    reason, and a step left with nothing replays unmasked rather than failing
    the save, matching validate_selection's totality.
    """
    if selection is None or "masks" not in selection:
        return selection
    specs = []
    for mask_id in selection["masks"]:
        mask = db.get_mask(mask_id)
        spec = validate_selection(mask["spec"]) if mask else None
        if spec is not None and "points" in spec:
            specs.append(spec)
    if not specs:
        return None
    if len(specs) == 1:
        return {**specs[0], "invert": specs[0]["invert"] != selection["invert"]}
    points = [p for spec in specs if not spec["invert"] for p in spec["points"]]
    return {"points": points, "invert": selection["invert"]} if points else None


def capture_steps(node_id: int) -> list[dict]:
    """Capture the edits that produced a node as a portable, image-independent
    recipe: every ancestor reachable through *either* parent link (that closure is
    exactly the node's pixel dependency set), in id order, with parent references
    rewritten as indices — 0 meaning "whatever node this gets applied to" and
    1..n-1 meaning an earlier step. Absolute node ids would be unusable on another
    image, and unsafe even on this one since SQLite reuses rowids after deletes."""
    closure: dict[int, dict] = {}
    stack = [node_id]
    while stack:
        current = stack.pop()
        if current in closure:
            continue
        node = db.get_node(current)
        if node is None:
            raise HTTPException(404, f"node {current} not found")
        closure[current] = node
        for parent in (node["parent_id"], node["parent2_id"]):
            if parent is not None:
                stack.append(parent)

    # ids are topological (a parent always has a smaller id), so sorting gives a
    # valid replay order and the root — the only effect-less node — comes first
    ordered = [closure[i] for i in sorted(closure)]
    index = {node["id"]: i for i, node in enumerate(ordered)}

    steps = []
    for node in ordered:
        if node["effect"] is None:
            continue  # the root stands in for the base node, index 0
        if node["effect"] not in EFFECTS and node["effect"] != "blend":
            raise HTTPException(400, f"effect '{node['effect']}' no longer exists")
        # normalizing here also backfills params added since the node was made,
        # e.g. blend's weight, which rendering defaults to 0.5 for older nodes
        params = validate_params(node["effect"], node["params"] or {})
        step = {
            "effect": node["effect"],
            "params": params,
            "parent": index[node["parent_id"]],
            "parent2": index[node["parent2_id"]] if node["effect"] == "blend" else None,
            # click coords ride along; they are in the parent's pixel space and
            # clamp to the target image's bounds at mask time. Saved masks
            # degrade to the click specs behind them — see _portable_selection.
            "selection": _portable_selection(validate_selection(node["selection"])),
        }
        steps.append(step)
    return steps


def _drop_mask_ref(selection: dict | None) -> dict | None:
    if selection is not None and "masks" in selection:
        return None  # image-scoped; a preset is not
    return selection


def _step_summary(steps: list[dict]) -> str:
    return " → ".join(s.get("effect", "?") for s in steps)


@app.get("/api/presets")
def list_presets():
    return [
        {**preset, "summary": _step_summary(preset["steps"])}
        for preset in db.list_presets()
    ]


@app.post("/api/presets")
def create_preset(body: PresetCreate):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "preset name cannot be empty")
    if db.get_node(body.node_id) is None:
        raise HTTPException(404, "node not found")
    steps = capture_steps(body.node_id)
    if not steps:
        raise HTTPException(400, "select a node with at least one effect")
    if len(steps) > MAX_PRESET_STEPS:
        raise HTTPException(400, f"presets are limited to {MAX_PRESET_STEPS} steps")
    try:
        preset = db.create_preset(name, steps)
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"a preset named '{name}' already exists")
    return {**preset, "summary": _step_summary(preset["steps"])}


@app.delete("/api/presets/{preset_id}")
def delete_preset(preset_id: int):
    if db.get_preset(preset_id) is None:
        raise HTTPException(404, "preset not found")
    db.delete_preset(preset_id)
    return {"deleted": preset_id}


def _validate_steps(steps: list[dict]) -> list[dict]:
    """Check a stored recipe before anything is written, so a malformed preset
    can never leave half a chain behind. Returns the steps with params
    re-normalized against today's effect registry."""
    if not isinstance(steps, list) or not steps:
        raise HTTPException(400, "preset has no steps")
    checked = []
    for i, step in enumerate(steps, start=1):
        effect = step.get("effect") if isinstance(step, dict) else None
        if effect not in EFFECTS and effect != "blend":
            raise HTTPException(400, f"step {i}: unknown effect '{effect}'")

        parent = step.get("parent")
        # bool is an int subclass, and a negative index would silently resolve
        # against the created-node list, so check the type and range explicitly
        if not isinstance(parent, int) or isinstance(parent, bool) or not 0 <= parent < i:
            raise HTTPException(400, f"step {i}: invalid parent index {parent!r}")

        parent2 = step.get("parent2")
        if effect == "blend":
            if parent2 is None:
                raise HTTPException(400, f"step {i}: blend requires a second input")
            if (
                not isinstance(parent2, int)
                or isinstance(parent2, bool)
                or not 0 <= parent2 < i
            ):
                raise HTTPException(400, f"step {i}: invalid parent2 index {parent2!r}")
        else:
            parent2 = None  # never persist a phantom second parent on a 1-input effect

        try:
            params = validate_params(effect, step.get("params") or {})
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, f"step {i}: bad params ({exc})")
        checked.append(
            {
                "effect": effect,
                "params": params,
                "parent": parent,
                "parent2": parent2,
                # total: presets saved before selections existed, or with
                # malformed data, replay unmasked rather than failing. A mask
                # reference should never reach a stored preset (capture_steps
                # degrades it), so one that did is hand-edited or older data —
                # strip it rather than resolve it against the wrong image.
                "selection": _drop_mask_ref(validate_selection(step.get("selection"))),
            }
        )
    return checked


@app.post("/api/nodes/{node_id}/apply-preset")
def apply_preset(node_id: int, body: PresetApply):
    base = db.get_node(node_id)
    if base is None:
        raise HTTPException(404, "node not found")
    preset = db.get_preset(body.preset_id)
    if preset is None:
        raise HTTPException(404, "preset not found")
    steps = _validate_steps(preset["steps"])

    image_id = base["image_id"]
    # An apply-time selection masks *every* step, and beats whatever the recipe
    # stored for that step: a preset is a chain of edits, and limiting it to an
    # object is the same gesture as limiting one effect to it — what is ticked is
    # what gets edited. The step's own selection is a click spec captured from
    # some other image, so honouring both would mean intersecting a union of
    # points with a union of masks, a shape the model does not have. Resolved
    # here, next to _validate_steps, so a foreign mask 400s before the first row
    # is written rather than half way through the chain.
    override = _check_selection(validate_selection(body.selection), image_id)

    # Each step must be committed and rendered before the next one references it:
    # render_node reads through its own connection, so an open transaction would
    # hide the rows it needs. Unwind by hand if a step fails.
    created: list[int] = []
    node_at = {0: node_id}
    try:
        for i, step in enumerate(steps, start=1):
            parent2_id = None if step["parent2"] is None else node_at[step["parent2"]]
            node = db.create_node(
                image_id,
                node_at[step["parent"]],
                step["effect"],
                step["params"],
                parent2_id,
                step["selection"] if override is None else override,
            )
            created.append(node["id"])
            node_at[i] = node["id"]
            rendering.render_node(node["id"])
    except Exception as exc:
        try:
            db.delete_nodes(created)
            rendering.delete_render_files(created)
        except Exception:  # a failed cleanup must not mask the real error
            pass
        raise HTTPException(500, f"applying preset failed: {exc}")

    return {"created": created, "terminal_node_id": created[-1]}


@app.get("/api/stats")
def get_stats():
    return {**db.stats(), "storage": rendering.storage_stats()}


class NoCacheStaticFiles(StaticFiles):
    """Make browsers revalidate frontend files on every load (cheap 304s via
    ETag) so app.js/style.css edits are picked up without a hard refresh."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", NoCacheStaticFiles(directory=WEB_DIR, html=True), name="web")
