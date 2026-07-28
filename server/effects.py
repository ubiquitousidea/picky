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


def validate_params(effect: str, params: dict) -> dict:
    """Clamp and coerce params against the effect's spec; raises on unknown effect."""
    spec = BLEND_SPEC if effect == "blend" else EFFECTS[effect]
    clean = {}
    for p in spec["params"]:
        value = params.get(p["name"], p["default"])
        if p["type"] == "choice":
            clean[p["name"]] = value if value in p["options"] else p["default"]
        elif p["type"] == "float":
            value = float(value)
            clean[p["name"]] = max(p["min"], min(p["max"], value))
        else:
            value = int(value)
            clean[p["name"]] = max(p["min"], min(p["max"], value))
    return clean
