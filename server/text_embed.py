"""CLIP text embeddings via onnxruntime: turn a phrase into a semantic vector.

The mirror of `embed.py`. That module runs CLIP's *vision* tower over a photo;
this one runs the *text* tower over a string, and because both halves come from
one checkpoint their outputs land in the same 512-d space. A dot product between
them is a cosine, and that single fact is what the Image map's search box and
`labels.py`'s cluster names are both built on.

**This used to live only in `tools/build_label_vectors.py`, and the server
deliberately had no text model at all.** The vocabulary behind cluster labels is
fixed and small, so encoding it once offline into `label_vectors.npz` beat
shipping a 254 MB tower to every machine. A search box breaks that bargain: a
typed query is not drawn from any fixed vocabulary, and resolving one against 857
known words fails opaquely the moment you type the 858th. So the tower moved
here, the build script now imports it, and the cost is paid *lazily* — nothing
downloads until someone actually searches (see `text_job.py`).

Everything in this module's contract with the rest of the tree is `embed.py`'s,
for the same reasons that module gives: the download is `sam.download_model` (so
the atomic write and pinned-revision contract exist once), `onnxruntime` is
imported lazily (so a broken wheel breaks only search), the session is cached for
the process, and the returned vector is L2-normalized (so callers may assume a
dot product is a cosine).

Two things here have no analogue in `embed.py` and are the ones to get right:

- **The tokenizer fails silently.** A wrong token stream still yields finite,
  well-spread, unit-length vectors — the searches just stop meaning anything, and
  no amount of staring at the output reveals it. That is why
  `tools/build_label_vectors.py` keeps `--check`, which scores real library
  images against the vocabulary: it is the only test that catches a *plausible*
  tokenizer bug, and it now covers this module too.
- **The revision is pinned to the text half of the vision tower's checkpoint.**
  Two towers from different checkpoints are not in one space, and nothing
  downstream can detect it.

This module also owns **the only tokenizer in the tree**, which is why
`ensure_tokenizer()` is separable from `ensure_model()`: `clipseg.py` needs
CLIP's byte-level BPE and none of this tower, and the alternative — letting it
fetch a second copy of the vocabulary from its own repo — would put CLIP's
merge table on disk twice and make "the tokenizer" a thing two modules own.
The two files are 1.7 MB against the graph's 254 MB, so the split costs the
search path nothing.
"""

import functools
import html
import json
import os
import re
from pathlib import Path

import numpy as np

# MODELS_DIR and the downloader are sam.py's, reached the same way embed.py
# reaches them: that module owns the conventions for on-disk model weights.
from .sam import MODELS_DIR, download_model

# Qdrant/clip-ViT-B-32-text at revision 48ca1db27cb4063eb311ec2aa7f087a808112876,
# the text half of the vision tower `embed.py` pins. Both must come from the same
# CLIP checkpoint or the two towers are not in one space.
_TEXT_REV = "48ca1db27cb4063eb311ec2aa7f087a808112876"
_TEXT_BASE = f"https://huggingface.co/Qdrant/clip-ViT-B-32-text/resolve/{_TEXT_REV}"
# Smallest first, deliberately: `download_model` reports progress per file, so
# the two small ones flash past and the 254 MB one owns the bar for the minutes
# that actually need explaining.
_FILES = {
    "clip_merges.txt": "merges.txt",
    "clip_vocab.json": "vocab.json",
    "clip_vit_b32_text.onnx": "model.onnx",
}
_ENV_VAR = "PICKY_CLIP_TEXT_MODEL"
_MODEL_FILE = "clip_vit_b32_text.onnx"

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
    `transformers` out of the picture entirely — the whole app's third-party
    dependencies stay numpy, Pillow, sklearn and onnxruntime.
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


# ------------------------------------------------------------------- weights

def model_paths() -> dict[str, Path]:
    """Where the three files live — env override for the graph, else `data/models/`.

    The vocabulary and merge table are not overridable, and should not be: they
    are the checkpoint's, not a deployment choice, and a graph paired with
    someone else's merge table is the silent failure this module's docstring
    warns about.
    """
    paths = {name: MODELS_DIR / name for name in _FILES}
    override = os.environ.get(_ENV_VAR)
    if override:
        paths[_MODEL_FILE] = Path(override)
    return paths


def ready() -> bool:
    """Is searching free right now, or does it cost a 254 MB download?

    Answered without touching the network, so it is safe on a request — this is
    `embed.model_path().is_file()`'s question, asked of three files instead of
    one. `text_job` uses it to short-circuit, and `main.text_search` to refuse
    rather than block a GET on a download.
    """
    return all(path.is_file() for path in model_paths().values())


def tokenizer_ready() -> bool:
    """Is `tokenizer()` free right now? `ready()` asked of the small files only.

    `clipseg.ready()`'s other half: that module needs CLIP's BPE and not this
    graph, so it must be able to ask about the two without the answer depending
    on the 254 MB one.
    """
    paths = model_paths()
    return all(paths[name].is_file() for name in _FILES if name != _MODEL_FILE)


def _fetch(names, on_progress=None) -> dict[str, Path]:
    """Download whichever of `names` are missing, in `_FILES` order."""
    paths = model_paths()
    for name in names:
        path = paths[name]
        if path.is_file():
            continue
        if name == _MODEL_FILE and os.environ.get(_ENV_VAR):
            # An explicit override that points nowhere is a misconfiguration, not
            # an invitation to fetch our own copy over the top of it.
            raise FileNotFoundError(f"{_ENV_VAR} points at missing file {path}")
        download_model(f"{_TEXT_BASE}/{_FILES[name]}", path, on_progress=on_progress)
    return paths


def ensure_tokenizer(on_progress=None) -> dict[str, Path]:
    """Just the vocabulary and the merge table — 1.7 MB, no tower.

    Split out of `ensure_model` for `clipseg.py`, which tokenizes with CLIP's
    BPE and never runs this graph: without the split, asking this module for a
    tokenizer would download 254 MB of text tower that the caller has no use
    for. Search still goes through `ensure_model`, which calls this first, so
    nothing about that path changed.
    """
    return _fetch([name for name in _FILES if name != _MODEL_FILE], on_progress)


def ensure_model(on_progress=None) -> dict[str, Path]:
    """The three files, downloading whichever are missing.

    Public and separate from `_session` for `embed.ensure_model`'s reason: the
    slow part has to be doable up front, off the request, with progress. Since
    `download_model` only calls `on_progress` while bytes move, a caller learns
    for free whether a download happened at all.
    """
    return _fetch(list(_FILES), on_progress)


@functools.lru_cache(maxsize=None)
def tokenizer() -> ClipTokenizer:
    # `ensure_tokenizer`, not `ensure_model`: the two small files are all this
    # needs, and `clipseg.py` asks for a tokenizer without ever wanting a tower.
    paths = ensure_tokenizer()
    return ClipTokenizer(paths["clip_vocab.json"], paths["clip_merges.txt"])


@functools.lru_cache(maxsize=None)
def _session():
    # onnxruntime imported lazily, like embed._session and sam._session: the
    # server must start, and everything unrelated to text must keep working, if
    # the wheel is missing or broken on this platform.
    import onnxruntime

    return onnxruntime.InferenceSession(
        str(ensure_model()[_MODEL_FILE]), providers=["CPUExecutionProvider"]
    )


# ------------------------------------------------------------------ encoding

def encode_terms(terms: list[str], on_batch=None) -> np.ndarray:
    """Encode terms as unit-length float32 rows, ensembled over `TEMPLATES`.

    Used both to build the label vocabulary offline (857 terms x 7 prompts) and
    to encode one typed query at runtime. Deliberately the same function for
    both: a query that happens to *be* a vocabulary word must land on the exact
    vector `label_vectors.npz` holds for it, or the search box and the cluster
    labels would quietly disagree about what the word means.

    `on_batch(done, total)` counts *prompts*, and exists for the build script:
    encoding the vocabulary is a couple of minutes on CPU where a query is
    milliseconds, so only the offline caller has anything to report.
    """
    session = _session()
    tok = tokenizer()
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
            tokens = tok.encode(prompt)
            ids[row, : len(tokens)] = tokens
            mask[row, : len(tokens)] = 1
        out = session.run(None, {"input_ids": ids, "attention_mask": mask})
        embeds.append(out[out_index].astype(np.float32))
        if on_batch:
            on_batch(min(start + 256, len(prompts)), len(prompts))

    # (n_terms, n_templates, D): unit-normalize each prompt, average the
    # templates, then renormalize — the mean of unit vectors is not one, and
    # every consumer downstream assumes a plain dot product is a cosine.
    stacked = np.vstack(embeds).reshape(len(terms), len(TEMPLATES), -1)
    stacked /= np.linalg.norm(stacked, axis=-1, keepdims=True)
    vectors = stacked.mean(axis=1)
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors.astype(np.float32)


@functools.lru_cache(maxsize=256)
def encode_query(text: str) -> np.ndarray:
    """One typed phrase as a unit-length float32 vector.

    Cached because the search box re-encodes on a debounce: backspacing over a
    word and retyping it is a repeat query, and seven transformer passes is by
    far the most expensive part of answering one. The bound is generous — these
    are 2 KB each — and the cache is per-process, so nothing outlives a restart.

    Frozen before it is handed out, because a cached mutable is a trap: one
    caller normalizing in place would silently change what every later search
    for the same phrase means.
    """
    vec = encode_terms([text])[0]
    vec.setflags(write=False)
    return vec
