"""The label vocabulary: the field of alternatives a typed query is judged
against.

CLIP was trained to put a caption and the photo it describes in the *same* 512-d
space, so a dot product between a phrase and a photo is a statement about
whether one describes the other. A single such dot product ranks the library but
never asks whether some *other* subject explains a photo better, which is how a
macaque came to score about as well against "people" as a person did.
`zero_shot_logp` asks that question by running the query against ~850 stored
phrases as rivals — the same matrix product, read as a classification rather
than a lookup, which is CLIP's strongest mode where the bare cosine is its
weakest.

**This module used to name the Image map's clusters, and that is gone.** The
k-means groups were real, but a nearest-label lookup answers with *something*
for every centroid whether or not it fits, and the assignment was forced to be
distinct, so a library with two kinds of landscape named the second after its
own runner-up. The vocabulary survives because the harder question — not "what
is this group" but "does this phrase beat every other phrase" — is the one it
turned out to be good at.

The text vectors are precomputed, and this module still never encodes anything.
`tools/build_label_vectors.py` encodes a fixed vocabulary of ~850 scenes and
subjects offline and writes `label_vectors.npz` next to this file; adding a word
means editing `tools/label_words.txt` and re-running that script — nothing here
reads it, and nothing here loads a model.

The tower that script runs is `text_embed.py`, which the search box made a
runtime concern: a typed query is not drawn from any fixed vocabulary, so it has
to be encoded on the spot. The rivals it is scored against are not, which is why
they stay precomputed — 850 phrases is a matrix product against a file, not 850
forward passes on every keystroke.

**The joint space is a precondition, not a given.** `embed.py` deliberately
accepts either CLIP export: a projection export's 512-d `image_embeds`, which is
the shared image/text space, or a bare vision export's 768-d `pooler_output`,
which is not in *any* text space at all. Image-to-image similarity is fine under
both, which is why `embed.py` treats the width as data — but comparing a photo
to a *phrase* is only meaningful under the first. A 768-d cache scored against
512-d rivals would not raise; it would be a shape error if we were lucky and
silent nonsense if the widths ever coincided. So the width is checked, and a
mismatch yields no rivals rather than wrong ones: search falls back to the bare
cosine it used before any of this existed.
"""

import functools
from pathlib import Path

import numpy as np

VECTORS_PATH = Path(__file__).resolve().parent / "label_vectors.npz"

# How close a vocabulary term may sit to the query before `zero_shot_logp`
# treats it as a restatement of the query rather than an alternative to it.
#
# Measured *after* the cone is projected out (`_cone_axis`), and only meaningful
# there: raw text-text cosines across this vocabulary run 0.82 +- 0.05, so
# "people" is 0.95 from "group of people" and still 0.82 from "airplane", and no
# threshold on them separates the two. With the cone removed the same vocabulary
# spreads over about -0.25..0.7 and this one lands where a person would put it —
# "people" excludes "group of people" and "crowd of people", "sunset" excludes
# "sunrise", "dusk" and "dawn", and a query with no near-synonym in the
# vocabulary excludes nothing at all.
SYNONYM_CEILING = 0.5

# CLIP's own logit scale, exp(4.6052) — the temperature its contrastive
# objective was trained at, and the one its zero-shot accuracy is quoted at.
# It is what makes the softmax a decision rather than a shrug: image-text
# cosines live in a band about 0.05 wide, so at T=1 the distribution over 850
# terms is flat to four decimal places.
LOGIT_SCALE = 100.0


@functools.lru_cache(maxsize=None)
def vocabulary():
    """`(names, unit float32 (L, D))`, or None if there is no usable file.

    Cached for the process: it is ~800 KB on disk and immutable, and every
    keystroke in the search box reads it.

    Missing or unreadable is not an error. The file is generated, not authored,
    so a checkout that has never run the build script still searches — just on
    the bare cosine, with nothing for a query to outscore.
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


@functools.lru_cache(maxsize=None)
def _cone_axis() -> np.ndarray | None:
    """The direction CLIP's text embeddings all lean in, unit length.

    Every text vector this model produces has a large component along one shared
    axis — the *cone effect* — which contributes the same amount to every cosine
    between two of them and so carries no information about either. Averaging the
    vocabulary estimates it, and `_decone` projects it out.

    Only the synonym test needs this, and nothing else should reach for it.
    Image-text cosines are compared *across images* against one fixed query,
    where a constant offset cancels and deconing would change every score
    without changing any ranking; it is only text-to-text, one query against 850
    different terms, that the cone dominates.
    """
    vocab = vocabulary()
    if vocab is None:
        return None
    axis = vocab[1].mean(axis=0)
    norm = float(np.linalg.norm(axis))
    return axis / norm if norm > 0 else None


def _decone(vectors: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """`vectors` with the shared direction removed, renormalized to unit length."""
    out = vectors - np.asarray(vectors @ axis)[..., None] * axis
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.where(norms > 0, norms, 1)


def zero_shot_logp(vectors: np.ndarray, query: np.ndarray) -> np.ndarray | None:
    """How well `query` explains each row of `vectors`, as a log-probability.

    `log P(query | image)` in a softmax over the query and every vocabulary term
    that is not a restatement of it — i.e. CLIP zero-shot classification, with
    the label set being "the thing you asked for" against "any of 850 other
    things this photo could be". Returns one value per row, or None when there is
    no usable vocabulary to compete against (see `vocabulary`), which callers
    must read as "score it the old way" rather than as an error.

    **This is the difference between CLIP's strongest mode and its weakest.** A
    bare cosine asks how well the query describes a photo, and answers on a scale
    whose zero nobody knows: image-text cosines sit in a narrow band whose
    position moves with the phrasing, and one vector summarizing a whole frame is
    dominated by scene gestalt, so a macaque portrait scores against "people"
    about what a person does — same primate face, same shallow depth of field,
    same framing. Asking instead which of 850 subjects fits *best* is the task
    CLIP's contrastive objective actually trained: the macaque still scores 0.24
    against "people", but it scores 0.31 against "monkey", and the comparison it
    loses is the one that was never made before.

    The exclusion is what keeps that from backfiring. Left in, the vocabulary's
    own near-synonyms are the competition — a photo of a crowd would be beaten by
    "crowd of people" and lose to the very word that found it — so terms closer
    to the query than `SYNONYM_CEILING` are struck. The query itself is always in
    the field, and it is column 0 of the logits below.
    """
    vocab = vocabulary()
    axis = _cone_axis()
    if vocab is None or axis is None or len(vectors) == 0:
        return None
    _, terms = vocab
    if terms.shape[1] != vectors.shape[1] or terms.shape[1] != query.shape[0]:
        # Not the joint space — see the note on it below. Say nothing rather
        # than something wrong, and let the caller fall back to a plain cosine.
        print(
            f"picky: query is {query.shape[0]}-d and labels are {terms.shape[1]}-d "
            f"but embeddings are {vectors.shape[1]}-d; skipping distractor scoring"
        )
        return None

    rivals = terms[_decone(terms, axis) @ _decone(query, axis) < SYNONYM_CEILING]
    if len(rivals) == 0:
        # A vocabulary entirely of synonyms leaves nothing to compare against,
        # and a softmax over one term is 1.0 for every image.
        return None

    # Column 0 is the query; the rest are what it has to beat. Both sides are
    # unit length everywhere here, so every dot product is a cosine.
    logits = LOGIT_SCALE * np.hstack([(vectors @ query)[:, None], vectors @ rivals.T])
    return logits[:, 0] - np.logaddexp.reduce(logits, axis=1)
