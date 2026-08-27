"""Export SAM 3's three ONNX graphs into `data/models/`. Run by hand, once.

This script owns **the export**: the split into three graphs, the pinned
upstream revision, and the fixture that proves the result matches PyTorch. What
it does not own is how those graphs are used — that is `server/sam3.py`, which
imports nothing from here and never runs torch.

Why a script at all, when every other model in this tree downloads itself:

- **Meta publishes no ONNX.** `facebook/sam3` ships PyTorch safetensors and a
  native checkpoint, nothing else. There is no URL for `sam.download_model` to
  fetch, so `sam3.ensure_model` raises with the command below rather than
  pretending it can get them.
- **The weights are gated**, and the gate is approved by a human. `PICKY_HF_TOKEN`
  gets you through it once someone at Meta says yes; nothing automatic can.
- **`torch` is a build-time dependency and must never enter
  `requirements.txt`.** It is ~650 MB with `torchvision` and `transformers`, the
  server imports none of them, and putting them in the install path would cost
  every user who never names an object. Install them for this run and remove
  them afterwards; the graphs are what persist.

Run:
    .venv/bin/pip install torch torchvision transformers onnx
    .venv/bin/python tools/export_sam3_onnx.py
    .venv/bin/pip uninstall -y torch torchvision transformers onnx

## The split, and why it is these three

`Sam3Model.forward` already accepts pre-computed `vision_embeds` and
`text_embeds`, so this is upstream's own seam rather than one invented here.
It matters because of where the parameters sit:

    vision_encoder   454.0M   depends only on the pixels   -> cache per node
    text_encoder     353.5M   depends only on the words    -> cache per query
    everything else   ~32M    depends on both              -> runs every time

That is a 12x saving on a repeat question about one photo (4.5s -> 0.38s
measured), and it is the whole reason to pay for three graphs instead of one.

The decoder wrapper calls `Sam3Model.forward` through that pre-computed path
rather than re-implementing the DETR plumbing, so the split cannot drift from
what transformers does — a re-export against a newer transformers picks up their
changes instead of silently disagreeing with them.

**The decoder takes four of the encoder's eight outputs**, not all of them:
`forward` slices `fpn_hidden_states[:-1]` and passes only
`fpn_position_encoding[-1]` onward, so `fpn_3`, `pos_0`, `pos_1` and `pos_3` are
dead and the ONNX exporter prunes them. That is load-bearing for the cache:
`sam3.encode_image` keeps only the four that survive.

## The fixture

`sam3_fixture.npz` is written beside the graphs and is the point of this script
as much as the graphs are. It holds one real image's preprocessed pixels, its
token ids, and the logits/boxes PyTorch produced from them — so
`sam3.self_check()` can prove the runtime reproduces the export rather than
merely running. Wrong preprocessing here yields a mask that still looks like a
mask, which is the failure CLAUDE.md's normalization rule exists to prevent, and
no amount of eyeballing an outline would catch it.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Pinned for `server/sam.py`'s reason: a revision hash content-addresses the
# files, so the export cannot silently start from different weights. sam3.1's
# checkpoints are newer but ship only a native `.pt` — no safetensors — so the
# transformers path this script uses needs sam3.
HF_REPO = "facebook/sam3"
HF_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"

# SAM 3's own, read off the checkpoint's processor_config.json. Restated in
# `server/sam3.py`, which must not import this file — the duplication is the
# point: the runtime states the contract it relies on, and `self_check()` is
# what proves the two still agree.
INPUT_SIZE = 1008
CONTEXT_LENGTH = 32
OPSET = 21

GATE_URL = "https://huggingface.co/facebook/sam3"


def _token() -> str:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    token = os.environ.get("PICKY_HF_TOKEN")
    if not token:
        raise SystemExit(
            f"PICKY_HF_TOKEN is not set. Accept the licence at {GATE_URL}, wait "
            f"for Meta to approve it (a human does), then put the token in .env."
        )
    # huggingface_hub reads its own variable; a bare 401 from a gated repo is
    # indistinguishable from a wrong URL, so fail with the gate named instead.
    os.environ.setdefault("HF_TOKEN", token)
    return token


def _export(module, args, dest: Path, input_names, output_names) -> None:
    """Trace one graph to a tmp path and move it into place.

    Atomic for `sam.download_model`'s reason, with one wrinkle of its own: the
    two big graphs exceed protobuf's 2 GB message limit, so the exporter writes
    an `.onnx.data` sidecar beside the graph and the pair has to move together.
    A graph whose weights are missing loads fine and fails at the first run —
    exactly the shape of bug that file-existence caching is meant to rule out.
    """
    import torch

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        staged = Path(tmp) / dest.name
        t0 = time.time()
        torch.onnx.export(
            module,
            args,
            str(staged),
            input_names=input_names,
            output_names=output_names,
            opset_version=OPSET,
            dynamo=False,
        )
        produced = sorted(Path(tmp).glob(f"{dest.stem}*"))
        size = sum(p.stat().st_size for p in produced)
        for path in produced:
            os.replace(path, dest.parent / path.name)
        print(
            f"  {dest.name}: {size / 1e6:.0f} MB in {time.time() - t0:.0f}s"
            f"{' (+ external data)' if len(produced) > 1 else ''}",
            flush=True,
        )


def main() -> None:
    _token()
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor
    from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

    from server import db

    out_dir = db.DATA_DIR / "models"
    torch.set_grad_enabled(False)

    print(f"loading {HF_REPO}@{HF_REVISION[:8]} …", flush=True)
    processor = Sam3Processor.from_pretrained(HF_REPO, revision=HF_REVISION)
    model = Sam3Model.from_pretrained(
        HF_REPO, revision=HF_REVISION, dtype=torch.float32
    ).eval()

    # ---------------------------------------------------------------- vision
    class VisionWrap(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, pixel_values):
            out = self.encoder(pixel_values)
            return (*out.fpn_hidden_states, *out.fpn_position_encoding)

    print("exporting …", flush=True)
    pixel_values = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    levels = len(VisionWrap(model.vision_encoder).eval()(pixel_values)) // 2
    _export(
        VisionWrap(model.vision_encoder).eval(),
        (pixel_values,),
        out_dir / "sam3_image_encoder.onnx",
        ["pixel_values"],
        [f"fpn_{i}" for i in range(levels)] + [f"pos_{i}" for i in range(levels)],
    )

    # -------------------------------------------------------------- language
    class TextWrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).pooler_output

    ids = torch.zeros(1, CONTEXT_LENGTH, dtype=torch.long)
    mask = torch.zeros(1, CONTEXT_LENGTH, dtype=torch.long)
    _export(
        TextWrap(model).eval(),
        (ids, mask),
        out_dir / "sam3_language_encoder.onnx",
        ["input_ids", "attention_mask"],
        ["text_pooled"],
    )

    # --------------------------------------------------------------- decoder
    class _TextEmbeds:
        """Just enough of the text output for `forward` to read `.pooler_output`.

        `forward` wants an object, the graph wants a tensor; this is the adapter
        and it exists only during the trace.
        """

        def __init__(self, pooled):
            self.pooler_output = pooled

    class DecoderWrap(nn.Module):
        def __init__(self, m, levels):
            super().__init__()
            self.m = m
            self.levels = levels

        def forward(self, *args):
            feats = args[: self.levels]
            pos = args[self.levels : 2 * self.levels]
            pooled, attention_mask = args[2 * self.levels :]
            out = self.m(
                vision_embeds=Sam3VisionEncoderOutput(
                    fpn_hidden_states=tuple(feats), fpn_position_encoding=tuple(pos)
                ),
                text_embeds=_TextEmbeds(pooled),
                attention_mask=attention_mask,
            )
            return out.pred_masks, out.pred_boxes, out.pred_logits, out.presence_logits

    with torch.no_grad():
        vision_out = VisionWrap(model.vision_encoder).eval()(pixel_values)
        pooled = TextWrap(model).eval()(ids, mask)
    _export(
        DecoderWrap(model, levels).eval(),
        (*vision_out, pooled, mask),
        out_dir / "sam3_decoder.onnx",
        [f"fpn_{i}" for i in range(levels)]
        + [f"pos_{i}" for i in range(levels)]
        + ["text_pooled", "attention_mask"],
        ["pred_masks", "pred_boxes", "pred_logits", "presence_logits"],
    )

    # --------------------------------------------------------------- fixture
    # A real photo, not noise: the thing being pinned down is preprocessing, and
    # a constant image is invariant to every resize this could get wrong.
    sample = next(
        (p for p in sorted(db.ORIGINALS_DIR.glob("*.jpg")) if p.stat().st_size > 5e5),
        None,
    )
    if sample is None:
        print("no original to build a fixture from — skipping", flush=True)
        return
    image = Image.open(sample).convert("RGB")
    query = "a photo"
    batch = processor(images=image, text=query, return_tensors="pt")
    with torch.no_grad():
        reference = model(**batch)
    np.savez_compressed(
        out_dir / "sam3_fixture.npz",
        pixel_values=batch["pixel_values"].numpy().astype(np.float32),
        input_ids=batch["input_ids"].numpy().astype(np.int64),
        attention_mask=batch["attention_mask"].numpy().astype(np.int64),
        pred_logits=reference.pred_logits.numpy().astype(np.float32),
        pred_boxes=reference.pred_boxes.numpy().astype(np.float32),
        query=np.array(query),
        source=np.array(sample.name),
    )
    print(f"  sam3_fixture.npz: from {sample.name}, query {query!r}", flush=True)
    print("\ndone. Verify with:  .venv/bin/python -c "
          "'from server import sam3; print(sam3.self_check())'", flush=True)


if __name__ == "__main__":
    main()
