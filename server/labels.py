"""Name a cluster of images: the text half of the Image map.

CLIP was trained to put a caption and the photo it describes in the *same* 512-d
space, so the nearest label to a group of photos is a description of what they
have in common. That is the whole trick — no captioning model, one dot product.

The text vectors are precomputed. `tools/build_label_vectors.py` encodes a fixed
vocabulary of ~850 scenes and subjects offline and writes `label_vectors.npz`
next to this file; the server never encodes text and never loads a text model.
Adding a word means editing `tools/label_words.txt` and re-running that script —
nothing here reads it.

**The joint space is a precondition, not a given.** `embed.py` deliberately
accepts either CLIP export: a projection export's 512-d `image_embeds`, which is
the shared image/text space, or a bare vision export's 768-d `pooler_output`,
which is not in *any* text space at all. Image-to-image similarity is fine under
both, which is why `embed.py` treats the width as data — but labelling is only
meaningful under the first. A 768-d cache compared against 512-d labels would
not raise; it would be a shape error if we were lucky and silent nonsense if the
widths ever coincided. So the width is checked, and a mismatch yields no labels
rather than wrong ones: the map still draws, it just says nothing.
"""

import functools
from pathlib import Path

import numpy as np

VECTORS_PATH = Path(__file__).resolve().parent / "label_vectors.npz"


@functools.lru_cache(maxsize=None)
def vocabulary():
    """`(names, unit float32 (L, D))`, or None if there is no usable file.

    Cached for the process: it is ~800 KB on disk and immutable, and the map
    re-reads it on every request and every drag of the cluster slider.

    Missing or unreadable is not an error. The file is generated, not authored,
    so a checkout that has never run the build script still serves a working
    Image map — one without labels.
    """
    if not VECTORS_PATH.is_file():
        return None
    try:
        with np.load(VECTORS_PATH, allow_pickle=False) as data:
            # Stored float16 to halve the file; every consumer wants float32,
            # and the rounding is far below the margin between two labels.
            return list(data["names"]), data["vectors"].astype(np.float32)
    except Exception as exc:
        print(f"picky: ignoring unreadable {VECTORS_PATH.name}: {exc}")
        return None


def label_clusters(centroids: np.ndarray) -> list[dict]:
    """The best distinct label for each cluster centroid, in row order.

    Centroids must be unit length and in the same space as the labels; both
    conditions are the caller's (`main._cluster`) to meet.

    Assignment is greedy and *exclusive*: the strongest cluster/label pair in
    the whole matrix is taken first, then the next over the clusters and labels
    still free. Independent argmaxes would be simpler and worse — neighbouring
    clusters of one library tend to share a nearest label, so a photographer
    with two kinds of landscape gets "mountain" twice and learns nothing about
    either. Forcing distinct labels makes the second cluster name its own runner
    up, which is exactly the distinction being asked about.
    """
    vocab = vocabulary()
    blank = [{"label": None, "score": 0.0} for _ in centroids]
    if vocab is None or len(centroids) == 0:
        return blank
    names, vectors = vocab
    if vectors.shape[1] != centroids.shape[1]:
        # Not the joint space — see this module's docstring.
        print(
            f"picky: labels are {vectors.shape[1]}-d but embeddings are "
            f"{centroids.shape[1]}-d; skipping cluster labels"
        )
        return blank

    # Both sides are unit length, so the dot product is the cosine.
    scores = centroids @ vectors.T
    remaining = scores.copy()
    out = blank
    for _ in range(min(len(centroids), len(names))):
        flat = int(np.argmax(remaining))
        cluster, label = divmod(flat, remaining.shape[1])
        out[cluster] = {
            "label": names[label],
            "score": round(float(scores[cluster, label]), 4),
        }
        # Strike the row and the column: this cluster is named, and no other
        # cluster may take this label.
        remaining[cluster, :] = -np.inf
        remaining[:, label] = -np.inf
    return out
