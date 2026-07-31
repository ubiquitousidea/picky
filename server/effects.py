"""Effect implementations. Each operates on an RGB uint8 numpy array."""

import numpy as np
from PIL import Image, ImageFilter


def _fit_kmeans(img: np.ndarray, k: int):
    """K-means in PCA-whitened RGB space. Raw RGB variance is dominated by the
    luminance diagonal, so clusters bunch along <1,1,1>; whitening equalizes
    variance across principal axes and spreads clusters over color, not just
    brightness. Centroid colors are mapped back to RGB (= per-cluster mean)."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    pixels = img.reshape(-1, 3).astype(np.float32)
    sample = pixels
    if len(sample) > 50_000:
        rng = np.random.default_rng(0)
        sample = sample[rng.choice(len(sample), 50_000, replace=False)]
    pca = PCA(n_components=3, random_state=0).fit(sample)
    # manual whitening with an epsilon so a flat axis (grayscale or solid
    # images have ~zero chroma variance) doesn't divide by zero
    scale = np.sqrt(pca.explained_variance_ + 1e-4)

    def whiten(x: np.ndarray) -> np.ndarray:
        return pca.transform(x) / scale

    km = MiniBatchKMeans(n_clusters=k, n_init="auto", random_state=0).fit(
        whiten(sample)
    )
    centroids = pca.inverse_transform(km.cluster_centers_ * scale).clip(0, 255)
    return km, whiten, centroids, pixels


def posterize_kmeans(img: np.ndarray, params: dict) -> np.ndarray:
    km, whiten, centroids, pixels = _fit_kmeans(img, int(params["k"]))
    labels = km.predict(whiten(pixels))
    return centroids[labels].reshape(img.shape).astype(np.uint8)


def kmeans_cluster_data(img: np.ndarray, k: int, n_points: int = 3000) -> dict:
    """Sampled RGB points with cluster assignments, for the 3D scatter plot."""
    km, whiten, centroids, pixels = _fit_kmeans(img, k)
    rng = np.random.default_rng(1)
    idx = rng.choice(len(pixels), min(n_points, len(pixels)), replace=False)
    points = pixels[idx]
    labels = km.predict(whiten(points))
    return {
        "points": np.column_stack([points.astype(int), labels]).tolist(),
        "centroids": centroids.astype(int).tolist(),
    }


def gaussian_blur(img: np.ndarray, params: dict) -> np.ndarray:
    radius = float(params["radius"])
    out = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(out)


def sobel_edges(img: np.ndarray, params: dict) -> np.ndarray:
    threshold = int(params["threshold"])
    gray = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    padded = np.pad(gray, 1, mode="edge")
    # 3x3 neighborhoods via shifted views of the padded image
    tl, tc, tr = padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:]
    ml, mr = padded[1:-1, :-2], padded[1:-1, 2:]
    bl, bc, br = padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:]
    gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl)
    gy = (bl + 2 * bc + br) - (tl + 2 * tc + tr)
    mag = np.hypot(gx, gy)
    peak = mag.max()
    if peak > 0:
        mag = mag / peak * 255
    if threshold > 0:
        mag = np.where(mag >= threshold, 255.0, 0.0)
    return np.repeat(mag.astype(np.uint8)[:, :, None], 3, axis=2)


def fs_dither(img: np.ndarray, params: dict) -> np.ndarray:
    colors = int(params["colors"])
    out = (
        Image.fromarray(img)
        .quantize(colors=colors, dither=Image.Dither.FLOYDSTEINBERG)
        .convert("RGB")
    )
    return np.asarray(out)


def _curve_lut(points) -> np.ndarray:
    """A 256-entry transfer function through `points`, interpolated with
    Fritsch–Carlson monotone cubic Hermite splines.

    Monotone rather than natural-cubic on purpose: an ordinary spline overshoots
    between widely spaced control points, and a tone curve that dips below its
    neighbours inverts local contrast — visible as banding in flat sky. The
    clamp below is what rules that out.

    `web/app.js` has a line-for-line mirror of this (`curveLut`) so the editor
    can draw the exact function the server will apply without a round trip per
    drag frame. Change one, change the other.
    """
    pts = sorted((int(x), int(y)) for x, y in points)
    x = np.array([p[0] for p in pts], dtype=np.float64)
    y = np.array([p[1] for p in pts], dtype=np.float64)

    h = np.diff(x)
    delta = np.diff(y) / h
    # tangents: the average of the two adjacent secants, one-sided at the ends
    m = np.empty_like(x)
    m[1:-1] = (delta[:-1] + delta[1:]) / 2
    m[0], m[-1] = delta[0], delta[-1]
    for i in range(len(delta)):
        if delta[i] == 0:
            # a flat segment must stay flat, or the cubic bulges out of it
            m[i] = m[i + 1] = 0.0
        else:
            a, b = m[i] / delta[i], m[i + 1] / delta[i]
            s = a * a + b * b
            if s > 9:
                t = 3.0 / np.sqrt(s)
                m[i], m[i + 1] = t * a * delta[i], t * b * delta[i]

    levels = np.arange(256, dtype=np.float64)
    # searchsorted gives each level its segment; the ends are pinned to 0..255
    # by validation, so no level falls outside [x[0], x[-1]]
    seg = np.clip(np.searchsorted(x, levels, side="right") - 1, 0, len(x) - 2)
    t = (levels - x[seg]) / h[seg]
    t2, t3 = t * t, t * t * t
    out = (
        (2 * t3 - 3 * t2 + 1) * y[seg]
        + (t3 - 2 * t2 + t) * h[seg] * m[seg]
        + (-2 * t3 + 3 * t2) * y[seg + 1]
        + (t3 - t2) * h[seg] * m[seg + 1]
    )
    return np.rint(out).clip(0, 255).astype(np.uint8)


def apply_curves(img: np.ndarray, params: dict) -> np.ndarray:
    return _curve_lut(params["points"])[img]


def apply_gamma(img: np.ndarray, params: dict) -> np.ndarray:
    # 1/gamma, not gamma: users read "gamma > 1" as brighter, and the label says so
    gamma = float(params["gamma"])
    lut = np.rint(255.0 * (np.arange(256) / 255.0) ** (1.0 / gamma))
    return lut.clip(0, 255).astype(np.uint8)[img]


PIXEL_SHAPES = ["square", "hexagon"]

# `_hex_pixelate` assigns pixels a band of rows at a time, so its peak memory is
# set by the band rather than by the image — originals run to 40 MP, where a
# single full-frame float temporary is ~160 MB.
_HEX_BAND_ROWS = 1024


def _hex_pixelate(img: np.ndarray, block: int) -> np.ndarray:
    """Pointy-top hexagonal bins: every pixel takes the mean color of the hex
    it falls in. `block` is the hex width (center-to-center within a row), so it
    means the same thing it does for square blocks.

    Hex centers are a triangular lattice — rows `block*sqrt(3)/2` apart with
    alternate rows offset half a cell — whose Voronoi cells are regular
    hexagons. Splitting it into its even and odd rows leaves two plain
    rectangular lattices, and the nearest point of a rectangular lattice is
    coordinate-wise rounding; so a pixel is assigned by rounding twice and
    comparing, with no search. Those offsets factor into an x part and a y part,
    which is why the per-pixel work below is broadcast from 1-D arrays.
    """
    h, w = img.shape[:2]
    sx = float(block)
    sy = block * np.sqrt(3.0)  # one sublattice skips a row, so its period is 2h

    # nearest center in each sublattice, per column and per row. Computed in
    # float64 — `rows - j*sy` cancels two large numbers, and these are 1-D.
    cols = np.arange(w, dtype=np.float64)
    rows = np.arange(h, dtype=np.float64)
    ia, ib = np.rint(cols / sx), np.rint(cols / sx - 0.5)
    ja, jb = np.rint(rows / sy), np.rint(rows / sy - 0.5)
    dxa = ((cols - ia * sx) ** 2).astype(np.float32)[None, :]
    dxb = ((cols - (ib + 0.5) * sx) ** 2).astype(np.float32)[None, :]
    dya = ((rows - ja * sy) ** 2).astype(np.float32)
    dyb = ((rows - (jb + 0.5) * sy) ** 2).astype(np.float32)

    # x, y >= 0 puts every index at 0 or above (np.rint(-0.5) is -0.0), and the
    # offset lattice's indices never exceed the aligned one's
    nx, ny = int(ia.max()) + 1, int(ja.max()) + 1
    n_bins = 2 * nx * ny
    # row and column halves of each sublattice's bin id, kept 1-D and broadcast
    # inside the loop — combining them up here would cost two full-frame arrays
    row_a, col_a = (ja * nx).astype(np.int32), ia.astype(np.int32)
    row_b = (jb * nx).astype(np.int32) + nx * ny
    col_b = ib.astype(np.int32)

    labels = np.empty((h, w), dtype=np.int32)
    sums = np.zeros((n_bins, 3), dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.int64)
    for y0 in range(0, h, _HEX_BAND_ROWS):
        y1 = min(y0 + _HEX_BAND_ROWS, h)
        band = np.where(
            dya[y0:y1, None] + dxa <= dyb[y0:y1, None] + dxb,
            row_a[y0:y1, None] + col_a[None, :],
            row_b[y0:y1, None] + col_b[None, :],
        )
        labels[y0:y1] = band
        flat = band.ravel()
        counts += np.bincount(flat, minlength=n_bins)
        for c in range(3):
            sums[:, c] += np.bincount(
                flat, weights=img[y0:y1, :, c].ravel(), minlength=n_bins
            )

    # bins outside the image get no pixels and are never indexed back out; the
    # clamp is only there to keep the division defined
    means = np.rint(sums / np.maximum(counts, 1)[:, None]).astype(np.uint8)
    return means[labels]


def pixelate(img: np.ndarray, params: dict) -> np.ndarray:
    block = int(params["block"])
    # nodes made before the param existed carry no "shape", the same reason
    # rendering.py reads blend's weight with a default
    if params.get("shape", "square") == "hexagon":
        return _hex_pixelate(img, block)
    im = Image.fromarray(img)
    w, h = im.size
    small = im.resize(
        (max(1, w // block), max(1, h // block)), Image.Resampling.BOX
    )
    return np.asarray(small.resize((w, h), Image.Resampling.NEAREST))


BLEND_MODES = ["average", "additive", "multiplicative", "subtractive"]


def apply_blend(a: np.ndarray, b: np.ndarray, mode: str, weight: float = 0.5) -> np.ndarray:
    ai = a.astype(np.int32)
    bi = b.astype(np.int32)
    if mode == "average":
        # weight is the share of b (the "blend with" target)
        out = np.rint(a.astype(np.float32) * (1.0 - weight) + b.astype(np.float32) * weight)
    elif mode == "additive":
        out = ai + bi
    elif mode == "multiplicative":
        out = ai * bi // 255
    elif mode == "subtractive":
        out = ai - bi
    else:
        raise ValueError(f"unknown blend mode '{mode}'")
    return out.clip(0, 255).astype(np.uint8)


EFFECTS = {
    "posterize": {
        "label": "Posterize (k-means)",
        "apply": posterize_kmeans,
        "params": [
            {"name": "k", "label": "Colors (k)", "type": "int", "min": 2, "max": 32, "default": 8},
        ],
    },
    "blur": {
        "label": "Gaussian blur",
        "apply": gaussian_blur,
        "params": [
            {"name": "radius", "label": "Radius", "type": "int", "min": 1, "max": 30, "default": 4},
        ],
    },
    "edges": {
        "label": "Sobel edges",
        "apply": sobel_edges,
        "params": [
            {"name": "threshold", "label": "Threshold (0 = off)", "type": "int", "min": 0, "max": 255, "default": 0},
        ],
    },
    "dither": {
        "label": "Floyd–Steinberg dither",
        "apply": fs_dither,
        "params": [
            {"name": "colors", "label": "Colors", "type": "int", "min": 2, "max": 64, "default": 8},
        ],
    },
    "curves": {
        "label": "Tone curve",
        "apply": apply_curves,
        "params": [
            {
                "name": "points",
                "label": "Curve",
                "type": "points",
                "min": 0,
                "max": 255,
                "max_points": 16,
                "default": [[0, 0], [255, 255]],
            },
        ],
    },
    "gamma": {
        "label": "Gamma",
        "apply": apply_gamma,
        "params": [
            {"name": "gamma", "label": "Gamma (>1 brightens)", "type": "float", "min": 0.1, "max": 4.0, "step": 0.01, "default": 1.0},
        ],
    },
    "pixelate": {
        "label": "Pixelate",
        "apply": pixelate,
        "params": [
            {"name": "block", "label": "Block size", "type": "int", "min": 2, "max": 64, "default": 12},
            {"name": "shape", "label": "Bin shape", "type": "choice", "options": PIXEL_SHAPES, "default": "square"},
        ],
    },
}


BLEND_SPEC = {
    "label": "Blend with…",
    "params": [
        {
            "name": "mode",
            "label": "Mode",
            "type": "choice",
            "options": BLEND_MODES,
            "default": "average",
        },
        {
            "name": "weight",
            "label": "Blend weight",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "default": 0.5,
        },
    ],
}


def effect_specs() -> list[dict]:
    """Registry as JSON-safe specs for the frontend."""
    specs = [
        {"name": name, "label": spec["label"], "params": spec["params"]}
        for name, spec in EFFECTS.items()
    ]
    specs.append({"name": "blend", "label": BLEND_SPEC["label"], "params": BLEND_SPEC["params"]})
    return specs


def _clean_points(value, p: dict) -> list[list[int]]:
    """Coerce a control-point list into something `_curve_lut` can always eat.

    Total by design — it clamps and falls back rather than raising, like the
    `choice` branch, because `validate_params` is called unguarded from the node,
    preview and preset endpoints and an exception there would be a 500.
    """
    pts = []
    for item in value if isinstance(value, (list, tuple)) else []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            x, y = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            continue
        pts.append([max(p["min"], min(p["max"], x)), max(p["min"], min(p["max"], y))])
    # one point per x: the editor cannot make a duplicate, a preset or a
    # hand-written request can, and two ys at one x has no meaning
    pts = [[x, y] for x, y in sorted({x: y for x, y in pts}.items())]
    if len(pts) < 2:
        return [list(pt) for pt in p["default"]]
    # pin the domain so the LUT covers every input level and never extrapolates
    if pts[0][0] != p["min"]:
        pts.insert(0, [p["min"], pts[0][1]])
    if pts[-1][0] != p["max"]:
        pts.append([p["max"], pts[-1][1]])
    # trim from the middle, after pinning, so the domain survives the cap
    if len(pts) > p["max_points"]:
        pts = pts[: p["max_points"] - 1] + pts[-1:]
    return pts


# How decode_mask picks among SAM's candidate masks for a click: the model's
# own confidence ranking, or one of the three granularities it proposes.
SELECTION_LEVELS = ["auto", "whole", "part", "subpart"]


def _clean_point(value) -> dict | None:
    """One click spec, or None. Only the lower bound of x/y is clamped: the
    upper bound depends on image dimensions, which presets don't know (a recipe
    replays onto differently-sized images), so it clamps at mask time."""
    if not isinstance(value, dict):
        return None
    try:
        x, y = int(value["x"]), int(value["y"])
    except (KeyError, TypeError, ValueError):
        return None
    level = value.get("level")
    return {
        "x": max(0, x),
        "y": max(0, y),
        "level": level if level in SELECTION_LEVELS else "auto",
    }


def validate_selection(value) -> dict | None:
    """Coerce a node's selection into one of two shapes, or None:

    - `{"masks": [id, ...], "invert"}` — saved masks, whose pixels were frozen
      to PNGs when they were created and are simply loaded back.
    - `{"points": [{"x", "y", "level"}, ...], "invert"}` — click specs,
      re-segmented by SAM against the pixels they are applied to.

    Both shapes are *unions*: the members are OR'd together and `invert` applies
    once, to the result. A union of clicks is what lets a multi-mask selection
    degrade into a preset step (`main._portable_selection`), which is why the
    click shape is a list even though the UI only ever produces one point.

    The pre-union shapes (`{"mask", "invert"}` and `{"x", "y", "invert",
    "level"}`) are still accepted and normalized, because node rows, mask specs
    and stored presets on disk hold them and nothing migrates them in place.
    `db.node_dict` is the choke point that upgrades everything read from the
    database; request bodies come through the callers in `main.py`.

    Total like `_clean_points` — anything malformed degrades to None (no
    selection) instead of raising, because this runs unguarded in the node,
    preview and preset endpoints.

    Whether a mask id exists, and whether it belongs to the right image, is not
    checked here — this module deliberately imports no DB. The callers in
    `main.py` turn a bad reference into a 400 (`_check_selection`).
    """
    if not isinstance(value, dict):
        return None
    invert = bool(value.get("invert"))
    # `is not None` rather than `in`: the request models carry every field as
    # `| None`, so unused keys are present-and-null on every request
    raw_masks = value.get("masks")
    if raw_masks is None and value.get("mask") is not None:
        raw_masks = [value["mask"]]
    if raw_masks is not None:
        if not isinstance(raw_masks, (list, tuple)):
            return None
        masks = []
        for m in raw_masks:
            try:
                m = int(m)
            except (TypeError, ValueError):
                continue
            if m not in masks:  # order-preserving dedupe; a union repeats for nothing
                masks.append(m)
        return {"masks": masks, "invert": invert} if masks else None

    raw_points = value.get("points")
    if raw_points is None:
        raw_points = [value]  # the pre-union shape: x/y/level on the selection itself
    if not isinstance(raw_points, (list, tuple)):
        return None
    points = [p for p in (_clean_point(p) for p in raw_points) if p is not None]
    return {"points": points, "invert": invert} if points else None


def validate_params(effect: str, params: dict) -> dict:
    """Clamp and coerce params against the effect's spec; raises on unknown effect."""
    spec = BLEND_SPEC if effect == "blend" else EFFECTS[effect]
    clean = {}
    for p in spec["params"]:
        value = params.get(p["name"], p["default"])
        if p["type"] == "choice":
            clean[p["name"]] = value if value in p["options"] else p["default"]
        elif p["type"] == "points":
            clean[p["name"]] = _clean_points(value, p)
        elif p["type"] == "float":
            value = float(value)
            clean[p["name"]] = max(p["min"], min(p["max"], value))
        else:
            value = int(value)
            clean[p["name"]] = max(p["min"], min(p["max"], value))
    return clean
