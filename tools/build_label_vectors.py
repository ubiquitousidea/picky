"""Encode the label vocabulary into `server/label_vectors.npz`. Run by hand.

This is a build tool, and what it owns is the *vocabulary*: the word list, the
Places365 scene names, and the npz. The text tower it runs them through is
`server/text_embed.py` — the server's, not this script's — so there is exactly
one CLIP tokenizer in the tree and one definition of how a phrase becomes a
vector.

That split moved. It used to be that the server had no text model at all and
this script owned the tower outright, because the vocabulary is fixed and
encoding it once offline beat shipping 254 MB to every machine. The Image map's
search box ended that: a typed query is not drawn from any fixed vocabulary. So
the tower lives in `server/`, is downloaded lazily and only when someone
searches, and this script is one of its two callers.

Sharing it is not incidental. `text_embed.encode_terms` applies the same seven
prompt templates either way, so a query that happens to *be* a vocabulary word
lands on the exact vector this script wrote for it — a phrase cannot mean one
thing when typed and another when stored.

What the npz is *for* is `labels.zero_shot_logp`: the rival subjects a typed
query has to beat before search will rank a photo highly, which is what stops
"people" returning the macaques. It used to name the Image map's clusters as
well, and that feature is gone — a nearest-label lookup answers with something
for every group whether or not it fits. Adding a word here still changes what
search considers, so this list is not vestigial.

What makes any of it possible is that the vision export in `data/models/` is a
*projection* export: its output is `image_embeds (b, 512)`, CLIP's joint
image/text space, which `rendering.embed_image` stores unit-length. The text
tower's `text_embeds` lands in that same 512-d space, so a stored image vector
and a vocabulary vector are comparable by plain dot product. Swap the vision
model for a bare `CLIPVisionModel` (768-d `pooler_output`) and that stops being
true — see `server/labels.py`, which detects it by width and declines to score.

Run:
    .venv/bin/python tools/build_label_vectors.py

Idempotent, and cheap after the first run: the model and tokenizer files are
cached in `data/models/`. They are no longer safe to delete afterwards — search
loads them at runtime — but deleting them only costs the next search a
re-download, never any data.

The one thing here that fails *silently* is the tokenizer. A wrong token stream
still produces finite, well-spread, unit-length vectors — the labels just stop
meaning anything, and no amount of staring at the output reveals it. That is the
same trap `server/embed.py` documents for its normalization constants, and it
gets the same answer: `--check` scores real images from the library against the
vocabulary, where a broken tokenizer is instantly obvious. Now that the tower is
shared, that check covers the search box too, and is the only thing that does.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import text_embed  # noqa: E402
from server.sam import MODELS_DIR  # noqa: E402

_PLACES365_URL = (
    "https://raw.githubusercontent.com/CSAILVision/places365/master/"
    "categories_places365.txt"
)
_PLACES_CACHE = MODELS_DIR / "categories_places365.txt"

WORDS_FILE = Path(__file__).parent / "label_words.txt"
OUT_FILE = ROOT / "server" / "label_vectors.npz"


# ------------------------------------------------------------------ sources

# Places365's category names are directory names, and it shows: two are
# misspelled upstream, and every possessive lost its apostrophe to the
# underscore convention. Fixed *before* encoding, not on the way out — the
# vector and the string on screen have to be the same phrase, and CLIP saw the
# spelled-correctly version far more often in training.
_PLACES_FIXES = {
    "archaelogical excavation": "archaeological excavation",
    "kindergarden classroom": "kindergarten classroom",
    "childs room": "child's room",
    "butchers shop": "butcher's shop",
    "veterinarians office": "veterinarian's office",
    "artists loft": "artist's loft",
    "fastfood restaurant": "fast food restaurant",
    "barndoor": "barn door",
}


def places365_terms() -> list[str]:
    """The 365 Places365 scene categories, as English phrases.

    Scenes rather than ImageNet's 1000 classes on purpose: ImageNet is 120 dog
    breeds and a long tail of laboratory objects, which for a personal photo
    library is mostly noise, where "where was this taken" is most of what a
    clump of photos has in common.
    """
    if not _PLACES_CACHE.is_file():
        print("downloading categories_places365.txt ...")
        _PLACES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(_PLACES365_URL, timeout=60) as resp:
            _PLACES_CACHE.write_bytes(resp.read())

    terms = []
    for line in _PLACES_CACHE.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        # "/f/field/cultivated 152" -> parts ["field", "cultivated"]
        path = line.split()[0]
        parts = path.strip("/").split("/")[1:]  # drop the a-z initial directory
        # Qualifiers trail the base name on disk and lead it in English, so the
        # segments reverse: field/cultivated -> "cultivated field".
        term = " ".join(reversed(parts)).replace("_", " ")
        terms.append(_PLACES_FIXES.get(term, term))
    return terms


def vocabulary() -> list[str]:
    words = [
        line.strip()
        for line in WORDS_FILE.read_text("utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return sorted(set(words) | set(places365_terms()))


# ------------------------------------------------------------------- checks

def self_check(tokenizer) -> None:
    """Cheap structural checks on the tokenizer, before the slow part."""
    assert len(tokenizer.encoder) == 49408, len(tokenizer.encoder)
    assert (tokenizer.sot, tokenizer.eot) == (49406, 49407)
    # "a photo of a dog" is five whole words in CLIP's vocabulary, so the ids
    # are bracket + 5 + bracket. A tokenizer that has fallen back to characters
    # (the usual symptom of a mis-parsed merge table) gives 17.
    ids = tokenizer.encode("a photo of a dog")
    assert len(ids) == 7, ids
    assert ids[0] == tokenizer.sot and ids[-1] == tokenizer.eot
    assert ids[1] == ids[4], ids  # both "a"
    # Round-trip through the vocabulary: ids must name the words we started with
    decoder = {v: k for k, v in tokenizer.encoder.items()}
    words = "".join(decoder[i] for i in ids[1:-1]).replace("</w>", " ").strip()
    assert words == "a photo of a dog", words
    print(f"tokenizer ok: 'a photo of a dog' -> {ids}")


def check_against_library(names: list[str], vectors: np.ndarray, limit: int) -> None:
    """Score real images from `data/` against the vocabulary and print the top 5.

    The only check that catches a *plausible* tokenizer bug. If photos come back
    with labels that have nothing to do with them, the two towers are not in the
    same space and nothing else here will tell you.
    """
    from server import db

    db.init()
    cached = db.get_embeddings()
    images = [im for im in db.list_images() if im["id"] in cached][:limit]
    if not images:
        print("no cached embeddings in data/picky.db — open the Image map first")
        return
    for image in images:
        vec = np.frombuffer(cached[image["id"]], dtype=np.float32)
        if vec.shape[0] != vectors.shape[1]:
            print(f"width mismatch: images are {vec.shape[0]}-d, labels {vectors.shape[1]}-d")
            return
        scores = vectors @ vec
        top = np.argsort(-scores)[:5]
        listing = ", ".join(f"{names[i]} ({scores[i]:.3f})" for i in top)
        print(f"  {image['name']:<28} {listing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        type=int,
        default=0,
        metavar="N",
        help="after writing, label N images from data/picky.db and print the top 5",
    )
    args = parser.parse_args()

    # Downloads the tower if this machine has never searched or built before.
    # Bytes are reported by `download_model` itself; this only says which file.
    text_embed.ensure_model(on_progress=None)
    self_check(text_embed.tokenizer())

    terms = vocabulary()
    print(f"{len(terms)} terms x {len(text_embed.TEMPLATES)} templates")
    vectors = text_embed.encode_terms(
        terms, on_batch=lambda done, total: print(f"  encoded {done}/{total} prompts")
    )

    # float16 halves the file for free: these are unit vectors, so the rounding
    # is ~1e-3 per component against label margins two orders of magnitude
    # wider. server/labels.py casts back to float32 on load.
    np.savez_compressed(
        OUT_FILE, names=np.array(terms), vectors=vectors.astype(np.float16)
    )
    size = OUT_FILE.stat().st_size
    print(f"wrote {OUT_FILE.relative_to(ROOT)} — {len(terms)} labels, {size / 1024:.0f} KB")

    if args.check:
        check_against_library(terms, vectors, args.check)


if __name__ == "__main__":
    main()
