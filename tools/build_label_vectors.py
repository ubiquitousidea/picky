"""Encode the label vocabulary into `server/label_vectors.npz`. Run by hand.

This is a build tool, not part of the server: the *server* never encodes text,
never downloads a text model, and has no tokenizer. It only reads the small
matrix this writes. That split is the whole point — CLIP's text tower is 254 MB
and the vocabulary is fixed, so paying for it once here beats paying for it on
every machine that runs Picky.

What makes labelling possible at all is that the vision export in
`data/models/` is a *projection* export: its output is `image_embeds (b, 512)`,
CLIP's joint image/text space, which `rendering.embed_image` stores unit-length.
The text tower's `text_embeds` lands in that same 512-d space, so a stored image
vector and a label vector are comparable by plain dot product. Swap the vision
model for a bare `CLIPVisionModel` (768-d `pooler_output`) and that stops being
true — see `server/labels.py`, which detects it by width and declines to label.

Run:
    .venv/bin/python tools/build_label_vectors.py

Idempotent, and cheap after the first run: the model and tokenizer files are
cached in `data/models/` and nothing at runtime reads them, so they can be
deleted afterwards to reclaim 255 MB.

The one thing here that fails *silently* is the tokenizer. A wrong token stream
still produces finite, well-spread, unit-length vectors — the labels just stop
meaning anything, and no amount of staring at the output reveals it. That is the
same trap `server/embed.py` documents for its normalization constants, and it
gets the same answer: `--check` scores real images from the library against the
vocabulary, where a broken tokenizer is instantly obvious.
"""

import argparse
import functools
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The downloader is the server's: it owns the atomic tmp-file-then-replace write
# and the pinned-revision contract, and this is the third module to want them.
from server.sam import MODELS_DIR, download_model  # noqa: E402

# Qdrant/clip-ViT-B-32-text at revision 48ca1db27cb4063eb311ec2aa7f087a808112876,
# the text half of the vision tower `server/embed.py` pins. Both must come from
# the same CLIP checkpoint or the two towers are not in one space.
_TEXT_REV = "48ca1db27cb4063eb311ec2aa7f087a808112876"
_TEXT_BASE = f"https://huggingface.co/Qdrant/clip-ViT-B-32-text/resolve/{_TEXT_REV}"
_FILES = {
    "clip_vit_b32_text.onnx": "model.onnx",
    "clip_vocab.json": "vocab.json",
    "clip_merges.txt": "merges.txt",
}

_PLACES365_URL = (
    "https://raw.githubusercontent.com/CSAILVision/places365/master/"
    "categories_places365.txt"
)
_PLACES_CACHE = MODELS_DIR / "categories_places365.txt"

WORDS_FILE = Path(__file__).parent / "label_words.txt"
OUT_FILE = ROOT / "server" / "label_vectors.npz"

# CLIP's context length. The graph's sequence axis is dynamic, but the position
# embeddings only go this far, and `text_embeds` is pooled at the end-of-text
# token — which the export finds with an argmax over the ids, since 49407 is the
# largest id in the vocabulary. That is why padding is 0 and not the tokenizer
# config's `pad_token` (itself 49407): pad with the EOT id and the argmax has a
# tie to break, and it breaks it at the *first* occurrence for the wrong reason.
CONTEXT_LENGTH = 77

# Prompt templates, a subset of OpenAI's ImageNet set. A term is encoded under
# every one and the results averaged, which is worth a few points over a bare
# word: it averages away the phrasing and leaves the concept. The bare "{}."
# earns its place on terms that already name a medium — "a photo of a black and
# white photograph." is not a sentence anyone wrote a caption with.
TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
    "a cropped photo of a {}.",
    "a bright photo of a {}.",
    "a photo of one {}.",
    "{}.",
]


# ---------------------------------------------------------------- tokenizer

@functools.lru_cache(maxsize=None)
def _byte_encoder() -> dict[int, str]:
    """GPT-2's byte→printable-character map, which CLIP inherits.

    BPE runs over characters, but text is bytes, and a merge table indexed by
    raw bytes would collide with real characters and break on the C0 controls.
    So every byte gets a distinct printable codepoint first: the 188 already-
    printable ones keep theirs, and the rest are shifted up past U+0100.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    table = {b: chr(b) for b in printable}
    spare = 0
    for b in range(256):
        if b not in table:
            table[b] = chr(256 + spare)
            spare += 1
    return table


# CLIP's own splitting pattern, with the two unicode classes it writes as \p{L}
# and \p{N} spelled for the stdlib `re`: [^\W\d_] is "word character that is
# neither digit nor underscore", i.e. a letter. Digits split one at a time.
_PATTERN = re.compile(
    r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|"""
    r"""[^\W\d_]+|\d|[^\s\w]+""",
    re.IGNORECASE,
)


class ClipTokenizer:
    """CLIP's byte-level BPE, from the HF repo's `vocab.json` + `merges.txt`.

    Small enough to inline, and inlining it is what keeps `tokenizers` /
    `transformers` out of the picture entirely — this script's only third-party
    imports are numpy and onnxruntime, both of which the app already needs.
    """

    def __init__(self, vocab_path: Path, merges_path: Path):
        self.encoder: dict[str, int] = json.loads(vocab_path.read_text("utf-8"))
        lines = merges_path.read_text("utf-8").split("\n")
        # line 0 is a "#version:" header; the table ends at the first blank
        merges = [tuple(line.split()) for line in lines[1:] if line.strip()]
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        self.byte_encoder = _byte_encoder()
        self.sot = self.encoder["<|startoftext|>"]
        self.eot = self.encoder["<|endoftext|>"]
        self._cache: dict[str, str] = {}

    def _bpe(self, token: str) -> str:
        """Merge a byte-encoded word down, greediest (lowest-rank) pair first.

        The trailing `</w>` marker is what makes "dog" at the end of a word a
        different symbol from "dog" inside one — without it the merge table's
        word-final entries never match and every word shatters into characters.
        """
        if token in self._cache:
            return self._cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        while len(word) > 1:
            pairs = set(zip(word, word[1:]))
            bigram = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if bigram not in self.ranks:
                break
            first, second = bigram
            merged, i = [], 0
            while i < len(word):
                if (
                    word[i] == first
                    and i < len(word) - 1
                    and word[i + 1] == second
                ):
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = tuple(merged)
        out = " ".join(word)
        self._cache[token] = out
        return out

    def encode(self, text: str) -> list[int]:
        """Token ids for one string, bracketed with start- and end-of-text."""
        # ftfy's mojibake repair, which upstream runs here, is irrelevant to a
        # vocabulary we wrote ourselves; the entity unescape and the whitespace
        # collapse are not, since both change the token stream.
        text = html.unescape(html.unescape(text))
        text = re.sub(r"\s+", " ", text).strip().lower()
        ids = [self.sot]
        for chunk in _PATTERN.findall(text):
            encoded = "".join(self.byte_encoder[b] for b in chunk.encode("utf-8"))
            ids.extend(self.encoder[piece] for piece in self._bpe(encoded).split(" "))
        ids.append(self.eot)
        if len(ids) > CONTEXT_LENGTH:
            # Truncate but keep the EOT, which is where the pooled vector is read
            ids = ids[: CONTEXT_LENGTH - 1] + [self.eot]
        return ids


# ------------------------------------------------------------------ sources

def ensure_files() -> dict[str, Path]:
    paths = {}
    for name, remote in _FILES.items():
        path = MODELS_DIR / name
        if not path.is_file():
            print(f"downloading {name} ...")
            download_model(f"{_TEXT_BASE}/{remote}", path)
        paths[name] = path
    return paths


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


# ------------------------------------------------------------------ encoding

def encode_terms(terms: list[str], model_path: Path, tokenizer: ClipTokenizer):
    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    # Introspected rather than indexed, for the reason embed._pick_output is:
    # exports vary, and `text_embeds` is the only rank-2 output either way.
    out_index = [o.name for o in session.get_outputs()].index("text_embeds")

    prompts = [t.format(term) for term in terms for t in TEMPLATES]
    embeds = []
    for start in range(0, len(prompts), 256):
        batch = prompts[start : start + 256]
        ids = np.zeros((len(batch), CONTEXT_LENGTH), dtype=np.int64)
        mask = np.zeros((len(batch), CONTEXT_LENGTH), dtype=np.int64)
        for row, prompt in enumerate(batch):
            tokens = tokenizer.encode(prompt)
            ids[row, : len(tokens)] = tokens
            mask[row, : len(tokens)] = 1
        out = session.run(None, {"input_ids": ids, "attention_mask": mask})
        embeds.append(out[out_index].astype(np.float32))
        print(f"  encoded {min(start + 256, len(prompts))}/{len(prompts)} prompts")

    # (n_terms, n_templates, 512): unit-normalize each prompt, average the
    # templates, then renormalize — the mean of unit vectors is not one, and
    # every consumer downstream assumes a plain dot product is a cosine.
    stacked = np.vstack(embeds).reshape(len(terms), len(TEMPLATES), -1)
    stacked /= np.linalg.norm(stacked, axis=-1, keepdims=True)
    vectors = stacked.mean(axis=1)
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors.astype(np.float32)


# ------------------------------------------------------------------- checks

def self_check(tokenizer: ClipTokenizer) -> None:
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

    paths = ensure_files()
    tokenizer = ClipTokenizer(paths["clip_vocab.json"], paths["clip_merges.txt"])
    self_check(tokenizer)

    terms = vocabulary()
    print(f"{len(terms)} terms x {len(TEMPLATES)} templates")
    vectors = encode_terms(terms, paths["clip_vit_b32_text.onnx"], tokenizer)

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
