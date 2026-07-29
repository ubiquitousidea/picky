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


def pixelate(img: np.ndarray, params: dict) -> np.ndarray:
    block = int(params["block"])
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
