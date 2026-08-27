# Why SAM 3 does not replace YOLOv8n for tagging

Decision: **keep `server/detect.py` as it is.** SAM 3 joins the tree as a
select-by-name backend only. This note records why, so the question does not get
re-opened from scratch.

The two models look interchangeable from a distance — both find objects, and
SAM 3's vocabulary is open where YOLO's is eighty COCO nouns. They are not.
Tagging and selecting are different jobs with opposite cost profiles, and the
tag chips are shaped around the one YOLO does.

## What the tagging pass actually is

`detect_job.py` sweeps the whole library, one image per `session.run`, batch 1,
single thread, `CPUExecutionProvider`. Every image is visited exactly once and
stamped in `images.detected_at`; nothing detects on demand, because a GET that
touched a thousand images would hold a request open for the length of the sweep.

The library today, measured (`data/picky.db`):

| | |
|---|---|
| images | 1683 |
| swept (`detected_at` set) | 1683 (100%) |
| with ≥1 detection | 1098 (65.2%) |
| swept, found nothing | **585 (34.8%)** |
| detection rows | 3522 |
| distinct labels seen | 70 of 80 |

YOLOv8n is 12.8 MB and runs ~40 ms per image. With JPEG decode the full pass is
about four minutes — and that is already the optimised number;
`detect.DECODE_HINT = 1280` is what took it from ~25 minutes, per the comment at
its definition.

## Speed: the disqualifying number

SAM 3 is 848M parameters. Its ONNX export is 3.58 GB — the image encoder alone
(1.83 GB) is five times picky's current largest model. Published latency is
~1.1 s per image on an RTX 4090 with the TensorRT execution provider.

Picky has no GPU path. On `CPUExecutionProvider` a ViT-scale encoder typically
runs 20-100× slower than a tuned GPU one, which puts a single image somewhere in
**20-60 seconds**. That range is extrapolated, not measured — but every value in
it is disqualifying:

| | per image | full sweep (1683) |
|---|---|---|
| YOLOv8n | ~40 ms | ~4 minutes |
| SAM 3 (CPU, est.) | 20-60 s | **9-28 hours** |

There is no batching left to recover. The sweep is already the simplest possible
loop; the cost is the model, not the plumbing.

## Accuracy: the open vocabulary is not free

Published head-to-head benchmarks put specialist YOLO **ahead** of SAM 3 by
20-47% AP across detection and segmentation tasks — wheat heads, weapons in CCTV,
concrete cracks, blood cells — with the gap widest on *exhaustive* mapping, which
is exactly what tagging is. SAM 3's strength is answering an arbitrary phrase on
one image, not enumerating everything in a library.

Note what this does and does not say. Those benchmarks are against YOLO
*fine-tuned* on each domain, where picky runs stock COCO weights, so the margin
would be narrower here. The speed gap would not be.

## Shape: open vocabulary breaks the tag chips

This is the argument that would still hold even if SAM 3 were free and instant.

`renderTagChips()` counts a **closed** set of labels across the library and
orders the chips by how many photos carry each. `detect.class_for_query()` exists
to answer "does this phrase *name* a class", which is what lets `autoLightTag`
light a chip for a typed query. Both questions dissolve under an open vocabulary:
there is no fixed set to count, and every phrase names something.

Recovering the chips means choosing a prompt list and running it per image —
which is a closed vocabulary again, just picky's instead of COCO's. The list
already exists (`tools/label_words.txt`, ~850 terms). Running 850 prompts against
each of 1683 images multiplies a cost that was already prohibitive.

The chips also lean on a distinction worth preserving: a tag's absence is a fact
about a *detector*, not about the photo. That is why the tag row has no `none`
token where the edit row does — "carries no effects" is checkable, "contains no
objects" would be a claim no detector here can make.

## The prize, and why it does not pay

The real upside is the 585 images (35%) that were swept and found nothing —
macaques, temples, mountains, the things COCO has no word for. That is a genuine
gap and it is measurable.

It still does not pay for a 9-28 hour sweep that also makes the chip row
incoherent. If it becomes worth chasing, the bounded version is a **manual,
per-image re-tag** over just those 585 — never a sweep, and never on the path of
a GET. `db.detected_image_ids()` and `db.undetected_count()` already isolate the
set.

## One thing keeping YOLO costs us

`detect.py`'s weights are Ultralytics YOLOv8n, **AGPL-3.0** — flagged at the
`_HF_URL` definition as "a licence to read before this app is distributed, not
merely run", and the only copyleft dependency in the tree.

Replacing it with SAM 3 would not fix that. SAM 3's weights are under Meta's
custom **SAM License**, gated behind manual approval, and carry their own
redistribution questions. So this change adds a licence to read rather than
removing one, and that fact belongs in the same breath as the AGPL note.
