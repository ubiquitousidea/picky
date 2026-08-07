"""Effect implementations. Each operates on an RGB uint8 numpy array."""

import functools
import math

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


BLUR_KERNELS = ["gaussian", "disk"]

# `_disk_blur` works a band of rows at a time, for the reason `_hex_pixelate`
# does: a full-frame int32 running sum of a 40 MP original is ~480 MB, a band
# ~116 MB at the largest radius.
_DISK_BAND_ROWS = 1024


def _disk_runs(r: int) -> tuple[np.ndarray, np.ndarray, int]:
    """A flat disk of radius `r` as horizontal runs: the row offsets, each row's
    half-width, and the total pixel area.

    Extracted because `_disk_blur` and `_disk_mean` both need exactly this and
    a disk that disagreed with itself between them would be very hard to see —
    two blurs that are each internally consistent and slightly different sizes.
    """
    dys = np.arange(-r, r + 1)
    half = np.floor(np.sqrt(r * r - dys * dys)).astype(np.int64)
    return dys, half, int((2 * half + 1).sum())


def _hole_radius(r: int, hole: float) -> int:
    """Radius of the shadow a mirror lens's secondary casts in an aperture of
    radius `r`, for an obstruction of `hole` times the diameter.

    One line, and shared by `_disk_mean` and `_aperture_stamp` for exactly the
    reason `_disk_runs` is shared: the rounding is the whole content, and two
    copies of it that disagreed by a pixel would put a ring in the kernel chart
    that is not the ring the blur used — which is the one failure the chart
    exists to rule out.

    Clamped to `r - 1`, so the ring is never narrower than the pixel it is drawn
    in and the normalizer can never reach 0. That also makes the donut collapse
    to a plain disk as the radius approaches the focal plane, which is what a
    real one does: the hole is a fixed fraction of a circle of confusion that is
    itself going to zero, and there is nothing left to see through.
    """
    if hole <= 0:
        return 0
    return min(r - 1, int(round(r * min(hole, 1.0))))


def _disk_blur(img: np.ndarray, r: int) -> np.ndarray:
    """Convolution with a flat disk of radius `r` — defocus, not soft focus.

    A Gaussian is separable, so Pillow blurs in O(w*h) whatever the radius. A
    disk is not, and the textbook 2-D convolution is O(w*h*r^2): 31k taps per
    pixel at r=100, ~106 s for a 40 MP frame. So this doesn't convolve.

    A disk is a stack of horizontal runs — row `dy` spans |dx| <= sqrt(r^2-dy^2)
    — and a cumulative sum along x reduces a run of *any* length to two lookups.
    That makes it 2(2r+1) reads per pixel, O(w*h*r), and exact: 8.5 s at 40 MP
    and r=100, ~19x a Gaussian there and ~6x at r=30.

    int32 holds every intermediate exactly, so there is no float rounding to
    reason about: a row's running sum tops out at width*255 (~2.0e6) and the
    accumulator at pi*r^2*255 (~8.0e6), both far inside 2^31.
    """
    h, w = img.shape[:2]
    dys, half, area = _disk_runs(r)

    out = np.empty_like(img)
    for y0 in range(0, h, _DISK_BAND_ROWS):
        y1 = min(y0 + _DISK_BAND_ROWS, h)
        top, bot = y0 - r, y1 + r
        # edge padding, matching how Pillow's Gaussian treats the border
        src = np.pad(
            img[max(top, 0) : min(bot, h)],
            ((max(0, -top), max(0, bot - h)), (r, r), (0, 0)),
            mode="edge",
        )
        # leading zero column, so a run sum is C[hi+1] - C[lo] with no case at x=0
        sums = np.zeros((src.shape[0], src.shape[1] + 1, 3), dtype=np.int32)
        np.cumsum(src, axis=1, dtype=np.int32, out=sums[:, 1:])

        n = y1 - y0
        acc = np.zeros((n, w, 3), dtype=np.int32)
        for dy, hw in zip(dys, half):
            k = int(dy) + r  # band row holding source row (y + dy)
            lo, hi = r - int(hw), r + int(hw) + 1
            acc += sums[k : k + n, hi : hi + w] - sums[k : k + n, lo : lo + w]
        # round half up on non-negative integers, without a full-frame float
        out[y0:y1] = ((acc + area // 2) // area).astype(np.uint8)
    return out


def gaussian_blur(img: np.ndarray, params: dict) -> np.ndarray:
    radius = float(params["radius"])
    # nodes made before the param existed carry no "kernel", the same reason
    # pixelate reads "shape" with a default
    if params.get("kernel", "gaussian") == "disk":
        return _disk_blur(img, max(1, int(round(radius))))
    out = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(out)


# Rec. 601 luma. Shared by bokeh's bloom and `sobel_edges`, which need to agree
# on what "bright" means no more than they need to disagree.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


# ---------- Bokeh: defocus that follows depth ----------
#
# The alternative this replaces is: segment the subject, invert, disk-blur. That
# blurs every background pixel by the same radius right up against a
# pixel-perfect silhouette, and the hard edge is what gives it away. A lens
# instead grows its circle of confusion with distance from the focal plane, so
# this drives the radius from a monocular depth estimate and no selection is
# involved at all.

SHOW_MODES = ["bokeh", "depth", "kernels"]

# The two pupils this effect can have: a photographic lens's, round on the axis
# and clipped to a cat's eye off it by `swirl`, and a reflecting telephoto's,
# which is an annulus because the secondary mirror sits on the axis and shadows
# the middle of it — the source of the ring-shaped highlights those lenses are
# known for.
#
# A mode rather than two sliders that mix, and the exclusivity is the whole
# design. Physically the two clips are independent: a barrel vignettes the
# pupil from the sides while the secondary keeps its size, so an off-axis
# mirror kernel is a cat's eye with a round bite out of the middle, pinched
# into two disconnected lobes wherever the aperture gets narrower than the
# obstruction. That shape is drawable, but it costs `_vignette` the closed form
# it runs on: the light a pinched aperture passes is a lens-minus-disc area
# with no elementary solution, and approximating it would put a wrong ramp
# across every sky. Catadioptrics do not swirl and double-Gausses have no
# secondary, so the combination was never a lens anyway. `bokeh` resolves this
# to `swirl` and `hole` once, and nothing downstream sees the mode.
APERTURE_MODES = ["lens", "mirror"]

# Where the kernel chart's `density` slider starts, in columns across the long
# side; the short side gets however many rows keep the cells square. Enough to
# read the shape's drift from centre to corner without the lattice itself
# becoming the thing you look at — the density that answers "what is this blur
# doing", where the top of the range answers "what does it do *there*".
_KERNEL_COLS = 9

# Everything blurred is computed on a canvas no larger than this on the long
# side. Blurred pixels have no high-frequency content by definition, so nothing
# visible survives the round trip — and the pixels that *are* sharp never come
# from this canvas at all (see the recombine at the end). It is what keeps the
# whole effect inside tens of megabytes: a 40 MP frame as one 4-channel float32
# layer would be 640 MB, and there are two of them live at once.
_BOKEH_WORK_PX = 2048
# A second, independent cap on the same scale: past this radius a bigger canvas
# only makes the blur slower, never better. Together the two make an extreme
# `amount` cost about what a moderate one does.
_BOKEH_WORK_R = 96
# Layer count from the radius — a 3 px blur has no use for 12 distinct radii.
_BOKEH_MIN_LAYERS, _BOKEH_MAX_LAYERS = 4, 12
# Side of the square the swirl kernel is held constant over. See `_swirl_mean`
# for why this is a seam/speed dial and why 64 is where it sits.
_SWIRL_TILE = 64


def _disk_mean(arr: np.ndarray, r: int, hole: float = 0.0) -> np.ndarray:
    """Mean over a flat disk of radius `r`, for float32 arrays of any depth —
    or over the annulus a mirror lens's secondary leaves of it, at `hole`.

    `_disk_blur`'s algorithm — see that docstring for why a disk is summed as
    horizontal runs rather than convolved — over float32 and C channels instead
    of uint8 and exactly 3. The two are deliberately not one function: the
    uint8 path's promise that "int32 holds every intermediate exactly" is
    load-bearing for a shipped effect, and generalizing it would quietly change
    what every existing disk-blur node re-renders to. Only the disk geometry is
    shared, via `_disk_runs`.

    The annulus is the disk minus the disk the secondary casts, summed straight
    out of the prefix sum already built rather than through a run generator of
    its own. A generator would have to hand back two runs for every row that
    crosses the hole, where subtraction reuses `_disk_runs` unchanged and gets
    the outer boundary *identical* to the plain kernel's by construction — the
    same property sharing `_disk_runs` with `_disk_blur` buys. The inner rows
    and columns are inside the `r`-padding by construction, so they index the
    same `sums`, and the cost is one extra accumulate per inner row: about 1.4x
    a plain disk at a typical obstruction, with no extra arrays.

    `hole = 0` skips the second pass entirely, so the plain path is not merely
    equivalent but bit-identical to what it was before mirror apertures existed.

    No banding, unlike its sibling: callers work at `_BOKEH_WORK_PX`, where a
    full-frame float32 running sum is tens of megabytes rather than hundreds.
    """
    if r < 1:
        return arr
    h, w, c = arr.shape
    dys, half, area = _disk_runs(r)
    src = np.pad(arr, ((r, r), (r, r), (0, 0)), mode="edge")
    # leading zero column, so a run sum is C[hi] - C[lo] with no case at x=0
    sums = np.zeros((src.shape[0], src.shape[1] + 1, c), dtype=np.float32)
    np.cumsum(src, axis=1, dtype=np.float32, out=sums[:, 1:])

    acc = np.zeros((h, w, c), dtype=np.float32)
    for dy, hw in zip(dys, half):
        k = int(dy) + r  # padded row holding source row (y + dy)
        lo, hi = r - int(hw), r + int(hw) + 1
        acc += sums[k : k + h, hi : hi + w] - sums[k : k + h, lo : lo + w]

    r_hole = _hole_radius(r, hole)
    if r_hole:
        dys_h, half_h, area_h = _disk_runs(r_hole)
        for dy, hw in zip(dys_h, half_h):
            k = int(dy) + r  # still offset by the *outer* pad
            lo, hi = r - int(hw), r + int(hw) + 1
            acc -= sums[k : k + h, hi : hi + w] - sums[k : k + h, lo : lo + w]
        area -= area_h
    return acc / area


def _lens_runs(r: int, t: float, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """A cat's-eye aperture as horizontal runs: row offsets, and each row's first
    and last column, plus the total area.

    `_disk_runs` generalized. The shape is the intersection of two equal circles
    whose centres lie on the line at angle `theta` — two arcs meeting at two
    points — which is what a lens with mechanical vignetting actually has for a
    pupil off-axis: the barrel clips the aperture from both sides.

    Parametrized by the aspect ratio `t` (short axis over long) rather than by
    how far apart the circles are, because the obvious construction is wrong in
    a way that is easy to miss. Pull two circles of radius `r` apart and the
    shape narrows — but it also gets *shorter*, so the corners of the frame come
    out less blurred rather than differently blurred. Holding the long axis at
    `r` and solving for the circles instead:

        d  = r (1/t - t)          centre separation
        rc = r (1/t + t) / 2      radius of each circle

    spans exactly +-r across `theta` and +-t*r along it. `t = 1` gives d = 0 and
    rc = r, and this returns the disk — literally, run for run.

    The shape is convex, so every row of it is still a single contiguous
    interval, which is the whole reason `_disk_mean` is fast (see `_disk_blur`).
    Rows are scanned over +-r because the far point of the shape is never
    farther than `r` from the centre, and rows whose two intervals do not
    overlap drop out.
    """
    t = min(max(t, 0.02), 1.0)
    d = r * (1.0 / t - t)
    rc = r * (1.0 / t + t) / 2.0
    a, b = (d / 2) * np.cos(theta), (d / 2) * np.sin(theta)
    dys = np.arange(-r, r + 1)
    h1 = np.sqrt(np.maximum(rc * rc - (dys - b) ** 2, 0.0))
    h2 = np.sqrt(np.maximum(rc * rc - (dys + b) ** 2, 0.0))
    lo = np.ceil(np.maximum(a - h1, -a - h2)).astype(np.int64)
    hi = np.floor(np.minimum(a + h1, -a + h2)).astype(np.int64)
    keep = hi >= lo
    dys, lo, hi = dys[keep], lo[keep], hi[keep]
    return dys, lo, hi, int((hi - lo + 1).sum())


@functools.lru_cache(maxsize=48)
def _aperture_stamp(r: int, t: float, theta: float, thickness: int, hole: float) -> np.ndarray:
    """The aperture's outline as a `(2r+1, 2r+1)` bool tile, centred.

    The kernel chart's mark: the aperture `_lens_runs` gives, rasterized solid
    and then hollowed out by its own erosion. It is not a circle this function
    derives for itself, so the outline is by construction the exact footprint
    the blur gets — the property the chart exists to have.

    A mirror aperture's hole is cleared out of the solid shape *before* the
    erosion, so the ring around it costs nothing extra: every boundary pixel of
    a set survives `filled & ~erode(filled)`, and the hole's rim is boundary.
    It is cleared with `_lens_runs` rather than `_disk_runs` only to keep one
    code path — the chart passes `t = 1.0` in mirror mode, where the two return
    the same runs — and its radius comes from the shared `_hole_radius`, which
    is what stops the drawn ring and the summed one landing a pixel apart.

    Erosion rather than the obvious "draw it again a few pixels smaller and
    subtract", which is wrong in a way that only shows up on the elongated
    apertures a high `swirl` produces. Two independently rasterized cat's eyes
    differ by about one column along a steep flank however far apart their
    radii are, so the difference there is a one-pixel run per row while the
    boundary itself is climbing two or three columns per row — a dashed line,
    right at the pointed tips that are the whole reason for looking. `filled &
    ~erode(filled)` cannot do that: every boundary pixel of a set is in it, so
    the outline is closed at any thickness, and `thickness` becomes an honest
    perpendicular width instead of a horizontal one.

    Cached because the cache is what makes a dense grid affordable — but only
    where it can be exact. At `swirl = 0` the aspect is 1.0, `_lens_runs` puts
    its two circles at d = 0, and the shape stops depending on `theta` at all;
    the caller passes 0.0 for it there, so every cell sharing a radius shares
    one stamp and the whole chart costs a handful of rasterizations. A mirror
    chart is that case always — the modes are exclusive, so its aspect is 1.0
    everywhere — and caches just as well. With swirl on, each cell has its own
    angle and mostly misses — which is the right way round. Rounding `theta` into buckets would buy that back by drawing a
    kernel rotated a few degrees from the one the render uses, and a chart that
    quietly disagrees with the render is the failure this whole view is built
    to avoid.

    `maxsize` is small because a tile is O(r^2) and `r` here is the
    *full-resolution* radius: at the top of `amount` on a 40 MP frame one stamp
    is most of a megabyte.

    Returned tiles are shared, so callers must treat them as read-only.
    """
    size = 2 * r + 1
    filled = np.zeros((size, size), dtype=bool)
    dys, lo, hi, _ = _lens_runs(r, t, theta)
    for dy, left, right in zip(dys, lo, hi):
        filled[int(dy) + r, int(left) + r : int(right) + r + 1] = True
    r_hole = _hole_radius(r, hole)
    if r_hole:
        dys_h, lo_h, hi_h, _ = _lens_runs(r_hole, t, theta)
        for dy, left, right in zip(dys_h, lo_h, hi_h):
            filled[int(dy) + r, int(left) + r : int(right) + r + 1] = False
    if r <= thickness:
        # A ring as thick as the kernel is wide is not a ring. Below that the
        # mark is the aperture solid, which is what the chart drew at every size
        # before it had a density to worry about.
        return filled

    inner = filled
    for _ in range(thickness):
        # One 4-neighbour erosion step. The tile is exactly the shape's bounding
        # box, so the shape touches all four borders and the world outside them
        # is empty — hence the border rows and columns erode away rather than
        # being left as they were by the shifts.
        eroded = inner.copy()
        eroded[1:, :] &= inner[:-1, :]
        eroded[:-1, :] &= inner[1:, :]
        eroded[:, 1:] &= inner[:, :-1]
        eroded[:, :-1] &= inner[:, 1:]
        eroded[0, :] = eroded[-1, :] = False
        eroded[:, 0] = eroded[:, -1] = False
        inner = eroded
    return filled & ~inner


def _swirl_aspect(rho: np.ndarray | float, swirl: float) -> np.ndarray | float:
    """The aperture's aspect ratio (short axis over long) at normalized radius
    `rho` — distance from the optical axis over half the frame diagonal, so 1 is
    exactly the corner.

    Quadratic rather than linear because that is where the mechanical vignetting
    is: a Helios is very nearly round across the middle third and clips hard in
    the last, so a straight ramp spends the slider's range deforming the part of
    the frame a real lens leaves alone. At swirl 0.85 this is still 0.91 a third
    of the way out where a linear ramp would already be at 0.72.

    Shared by `_swirl_mean` and `_vignette` — the shape and the dimming are two
    readings of one aperture, and they would be very hard to catch drifting
    apart: each looks right on its own, and together they just say the wrong
    lens.
    """
    return 1.0 - swirl * rho * rho


def _lens_area_ratio(t: np.ndarray | float) -> np.ndarray | float:
    """Area of the cat's eye over the area of the disk it was cut from — the
    fraction of the light that gets through, at aspect `t`.

    The two-circle construction of `_lens_runs`, integrated instead of scanned:
    two circular segments, in units of the disk's radius. It depends only on the
    aspect and not on `r`, which is what lets one frame-wide map serve every
    band's radius. Checked against `_lens_runs`' own pixel count to three
    decimals over the whole range the slider can reach.
    """
    t = np.clip(t, 0.02, 1.0)
    d, rc = 1.0 / t - t, (1.0 / t + t) / 2.0
    lens = 2 * rc * rc * np.arccos(d / (2 * rc)) - (d / 2) * np.sqrt(4 * rc * rc - d * d)
    return lens / np.pi


def _vignette(img: np.ndarray, swirl: float) -> np.ndarray:
    """Darken toward the corners by exactly as much light as the aperture that
    `_swirl_mean` narrowed is no longer passing.

    **The swirl does not already do this, though the optics say it should.** A
    real clipped pupil vignettes *because* it is clipped — one aperture produces
    the cat's eye and the corner darkening both, which is why one slider drives
    them. But `_swirl_mean` divides by the cat's eye's own area, so it is a mean
    and not an integral: narrowing the aperture changes which pixels get averaged
    and never their level. A flat frame at the top of the slider comes back flat,
    in the corners where the aperture passes 12.8% of the light exactly as much
    as on the axis. The shape of the light loss is free; the magnitude has to be
    put back by hand, and this is the hand — which is also why the param is
    labelled "Swirl (shape + vignette)" rather than for the swirl alone.

    It runs here, on the finished full-resolution frame, rather than inside the
    blur, for three reasons.

    The kernel cannot carry it. Normalizing by the *disk's* area instead of the
    cat's eye's is the obvious way to have the dimming for free, and it fails
    twice over: coverage rides through that same kernel as alpha, so a scaled
    normalizer would report a fully covering band as the area ratio instead of 1
    and let far bands bleed through near ones — and `_bokeh_layers` then
    un-premultiplies by the coverage that accumulated, which divides the very
    gain back out. Corrupted occlusion, and still no vignette. See `_swirl_mean`.

    A lens vignettes sharp light as well as defocused light. `bokeh` recombines
    the original full-size pixels with the blurred working canvas, so dimming
    only what was blurred would put a brightness step exactly along the
    recombine crossover, which is the one seam this effect works hardest to hide.

    And below some radius no kernel runs at all: `bokeh` takes its sub-pixel
    early-out before `_bokeh_layers` is ever reached, and vignettes there too.
    The aperture is the aperture whether or not it blurred anything, and corner
    brightness must not jump the instant `amount` crosses that threshold.

    Not a stylistic vignette with its own falloff, in any of those places: the
    gain *is* `_lens_area_ratio(_swirl_aspect(rho))`, the same aspect the kernel
    reads, integrated into an area fraction — so one slider describes one
    physical pupil. About -1.5 stops in the corners at swirl 0.6, which is
    roughly a fast vintage double-Gauss wide open, and -3 at the top of the
    slider, which is what the top of the slider is for.

    Banded like `_disk_blur` for the same reason: at 40 MP a full-frame float
    gain map is 480 MB. The gain is computed in float and applied per band
    rather than quantized into an 8-bit map and multiplied through, because a
    vignette is a smooth ramp across large flat areas — sky, exactly where 8-bit
    steps in the *gain* would band visibly.
    """
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    half_diagonal = np.hypot(cy, cx)
    dx2 = (np.arange(w, dtype=np.float32) - cx) ** 2

    out = np.empty_like(img)
    for y0 in range(0, h, _DISK_BAND_ROWS):
        y1 = min(y0 + _DISK_BAND_ROWS, h)
        dy = np.arange(y0, y1, dtype=np.float32) - cy
        rho = np.sqrt(dy[:, None] ** 2 + dx2[None, :]) / half_diagonal
        gain = _lens_area_ratio(_swirl_aspect(rho, swirl)).astype(np.float32)
        band = img[y0:y1] * gain[..., None]
        out[y0:y1] = (band + 0.5).astype(np.uint8)  # non-negative: round half up
    return out


def _swirl_mean(arr: np.ndarray, r: int, swirl: float) -> np.ndarray:
    """`_disk_mean`, but with the kernel deforming toward the edges of the frame.

    Round on the optical axis and a progressively thinner cat's eye away from
    it, long axis tangential — the Helios 44-2 look, where the deformation is
    what makes a busy background appear to swirl.

    A kernel that varies per pixel looks like it must destroy the run trick: the
    run bounds stop being constants that can be sliced and become arrays that
    have to be gathered, which measures far too slow to use. The way out is that
    **the prefix sum does not depend on the kernel**. So it is built once for the
    whole frame, exactly as `_disk_mean` builds it, and then each tile reads its
    own offsets out of that one shared array. A tile therefore costs no halo and
    no extra data — the element count is identical to a single full-frame pass,
    and all that is added is one numpy call per row per tile. Measured against
    `_disk_mean` on a 1365x2048x5 working frame: 1.6x at r=16, 1.8x at r=31,
    1.9x at r=60.

    `_SWIRL_TILE` is the dial between seams and speed. The kernel steps at tile
    boundaries, but the step is self-limiting: the shape's anisotropy grows with
    the distance from the centre while the angle between neighbouring tiles
    shrinks as tile/distance, so their product is about tile/half-diagonal
    wherever you stand — and it vanishes at the centre, where the kernel is
    round and its orientation means nothing. In smooth regions the difference is
    second order anyway, since both kernels are normalized and symmetric about
    the same point. It holds up under the worst case there is — a field of blown
    specular points defocused into hard-edged cat's eyes, where a stepped kernel
    would show as blobs assembled from mismatched halves — so 64 stays. Halving
    it halves the mismatch and costs ~1.5x if that is ever wanted.

    Each kernel is normalized by its *own* area, so this pass changes only the
    shape. The corner darkening a clipped aperture also produces is `_vignette`,
    applied once to the finished frame — it cannot be had here by dividing by
    the disk's area instead. Coverage rides through this same kernel, and a
    normalizer that scaled it would not darken the composite but corrupt it:
    `_bokeh_layers` composites with `1 - alpha` and then un-premultiplies by the
    coverage that accumulated, so under-covered bands would let farther bands
    bleed through and the gain would be divided back out at the end.
    """
    if r < 1:
        return arr
    h, w, c = arr.shape
    src = np.pad(arr, ((r, r), (r, r), (0, 0)), mode="edge")
    sums = np.zeros((src.shape[0], src.shape[1] + 1, c), dtype=np.float32)
    np.cumsum(src, axis=1, dtype=np.float32, out=sums[:, 1:])

    # The optical axis is the centre of the node's own pixels. The crop is an
    # output stage applied after the whole tree, so framing off-centre later
    # moves the swirl's centre off-centre in the result — which is what cropping
    # does to a real frame too.
    cy, cx = h / 2.0, w / 2.0
    half_diagonal = np.hypot(cy, cx)

    out = np.empty((h, w, c), dtype=np.float32)
    for y0 in range(0, h, _SWIRL_TILE):
        y1 = min(y0 + _SWIRL_TILE, h)
        for x0 in range(0, w, _SWIRL_TILE):
            x1 = min(x0 + _SWIRL_TILE, w)
            my, mx = (y0 + y1) / 2.0 - cy, (x0 + x1) / 2.0 - cx
            aspect = _swirl_aspect(np.hypot(my, mx) / half_diagonal, swirl)
            dys, lo, hi, area = _lens_runs(r, aspect, np.arctan2(my, mx))
            th, tw = y1 - y0, x1 - x0
            acc = np.zeros((th, tw, c), dtype=np.float32)
            for dy, left, right in zip(dys, lo, hi):
                k = y0 + int(dy) + r  # padded row holding source row (y + dy)
                a = x0 + r + int(right) + 1
                b = x0 + r + int(left)
                acc += sums[k : k + th, a : a + tw] - sums[k : k + th, b : b + tw]
            out[y0:y1, x0:x1] = acc / area
    return out


def _kernel_chart(
    img: np.ndarray,
    d_small: np.ndarray,
    focus: float,
    falloff: float,
    span: float,
    r_full: float,
    swirl: float,
    hole: float,
    density: int,
) -> np.ndarray:
    """The apertures themselves, outlined at true size on a grid over the frame.

    A diagram of what `show: "bokeh"` is doing, for aiming the other sliders:
    every ring is the actual footprint `_lens_runs` hands the blur, at the
    actual radius the depth under it earns, so its size reads `amount`, `focus`
    and `falloff` and its shape reads the aperture — the cat's eye `swirl`
    clips, or the annulus a mirror's `hole` leaves. Nothing here recomputes
    that geometry — a chart that drew its own idea of the kernel could agree
    with the render today and drift from it silently, which is the one way a
    diagnostic view is worse than none.

    `density` is what makes the chart answer a second question. At the default
    9 columns the grid reads the blur across the whole frame; wound up, it puts
    several samples on each side of a depth edge, which is the only way to see
    whether the kernel steps across that edge or ramps over it and how wide the
    ramp is. The mark had to stop being a filled disc for that to be possible:
    a dense grid and a solid mark cannot coexist, since neighbours overlap
    everywhere at a large `amount` and the frame goes to a white sheet. An
    outline overlaps legibly at any density, still reads size and cat's-eye
    shape at a glance, and is what makes the maximum composite below do what it
    always claimed to.

    Drawn at the *full-resolution* radius rather than the working canvas's,
    because that is the blur you see in the output; `_BOKEH_WORK_R` is an
    internal economy, not something to explain to someone aiming a slider.

    One deliberate omission: the rings are not dimmed by `_vignette`, though
    the render's kernels are. The corner ones are exactly what is worth looking
    at, and dimming them by the three stops the aperture actually costs would
    hide the shape this view exists to show — their area already tells that
    story.
    """
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    half_diagonal = np.hypot(cy, cx)
    dh, dw = d_small.shape[:2]

    cols = density if w >= h else max(3, round(density * w / h))
    rows = max(3, round(cols * h / w)) if w >= h else density
    step_y, step_x = h / rows, w / cols

    # Fixed to the frame, not to the kernel: a stroke proportional to `r` would
    # weight the far ring heavier than the near one, which is the comparison
    # being made. Scaled by the frame so it survives the preview's downscale.
    thickness = max(1, round(max(h, w) / 800))

    # The backdrop is the frame itself, dark enough that a white ring reads
    # against a blown highlight but still legible as the photo — the kernels
    # mean much more when you can see which of them landed on the subject.
    out = (img * 0.3).astype(np.uint8)
    ink = np.zeros((h, w), dtype=bool)

    for row in range(rows):
        for col in range(cols):
            gy, gx = (row + 0.5) * step_y, (col + 0.5) * step_x
            # nearest depth sample; the map is ~0.4 MP and this wants one
            # number. Deliberately not `depth.depth_at`'s median window: that
            # exists to stop a *picked* focus landing on an edge value, and
            # here the edge is the thing being looked at.
            dy_i = min(dh - 1, max(0, int(gy / h * dh)))
            dx_i = min(dw - 1, max(0, int(gx / w * dw)))
            pixel_t = abs(float(d_small[dy_i, dx_i]) - focus) / span
            r = int(round(r_full * pixel_t**falloff))
            gy_i, gx_i = int(gy), int(gx)

            if r < 1:
                # In focus: no aperture to draw, and an empty cell would read as
                # a bug rather than as the focal plane. A dot marks the sample,
                # sized off the same stroke as the rings so the focal plane does
                # not simply vanish on a large frame.
                k = thickness
                ink[max(0, gy_i - k) : gy_i + k + 1, max(0, gx_i - k) : gx_i + k + 1] = True
                continue

            rho = np.hypot(gy - cy, gx - cx) / half_diagonal
            aspect = _swirl_aspect(rho, swirl)
            # Exactly round: `_lens_runs` puts both circles at one point, so the
            # angle cannot matter. Passing 0.0 rather than the cell's own angle
            # is what collapses a whole swirl-free chart onto one stamp per
            # radius — a cache hit, not an approximation.
            theta = 0.0 if aspect >= 1.0 else float(np.arctan2(gy - cy, gx - cx))
            stamp = _aperture_stamp(r, float(aspect), theta, min(thickness, max(1, r // 3)), hole)

            top, left = gy_i - r, gx_i - r
            y0, y1 = max(0, top), min(h, top + stamp.shape[0])
            x0, x1 = max(0, left), min(w, left + stamp.shape[1])
            if y1 > y0 and x1 > x0:
                ink[y0:y1, x0:x1] |= stamp[y0 - top : y1 - top, x0 - left : x1 - left]

    out[ink] = (out[ink] * 0.25 + 0.75 * 255).astype(np.uint8)
    return out


def _bloom_luminance(
    rgb: np.ndarray, alpha: np.ndarray, bright: np.ndarray, bloom: float
) -> np.ndarray:
    """Replace a blurred band's luminance with the *soft maximum* over the disk,
    keeping the colour the plain blur produced.

    A flat average is what makes a rendered defocus look computed rather than
    photographed. A lens spreads a highlight's energy across the whole disk, so
    a pinpoint of sun in a dark hedge becomes a *disk as bright as the sun was*;
    an average divides that same energy by the disk's area and returns something
    barely lighter than the hedge.

    The fix is the standard three steps, in luminance only:

        exponentiate -> convolve -> log        L' = 1 + ln(mean(e^(k(L-1)))) / k

    which is a soft maximum: at k -> 0 it *is* the mean (expand the exponential
    and the k cancels), at large k it approaches max(L), and in between one
    bright pixel pulls the whole disk up without erasing what else is there.
    Shifting by 1 inside the exponential is what keeps every intermediate in
    (0, 1] rather than overflowing at k=12; it cancels out in the log.

    Only luminance goes through it. Running each channel separately would drag
    hues toward whichever primary happened to be brightest, so the chroma stays
    the linear blur's and this rescales it: output luminance is exactly L', and
    the ratios between channels are untouched. That also means k -> 0 returns
    the plain blur *continuously*, so the slider has no step at the bottom.

    `rgb` is premultiplied and stays that way — scaling a premultiplied colour
    by a gain is scaling the colour, so there is nothing to undo and redo.
    """
    safe = np.maximum(alpha, 1e-6)  # a band contributes nothing where it is 0
    soft = 1.0 + np.log(np.maximum(bright / safe, 1e-30)) / bloom
    soft = np.clip(soft, 0.0, 1.0)  # guaranteed by the algebra; float32 is not
    linear = (rgb @ _LUMA)[..., None] / (255.0 * safe)
    # The floor caps the gain where there is no luminance to scale up. Black
    # stays black without it — soft vanishes with linear — but not divided by 0.
    return rgb * (soft / np.maximum(linear, 1e-4))


def _bokeh_layers(
    src: np.ndarray, d: np.ndarray, radii: np.ndarray, bloom: float, swirl: float, hole: float
) -> np.ndarray:
    """Composite `src` as depth-ordered layers, each blurred by its own radius.

    `d` is normalized depth (1 = nearest) and `radii[k]` is the blur radius for
    band k, whose centre is at depth k/(L-1). Each pixel is split between its
    two neighbouring bands by linear interpolation, so the partition is smooth
    and there is no banding where a band boundary crosses a surface.

    The bands are then composited far to near with premultiplied alpha. This is
    the step that earns the design, and the tempting simplification — blur the
    whole frame at each radius and lerp between the two nearest results — is
    what it exists to avoid: a globally blurred layer has averaged the subject
    into itself, so the subject's colour smears outward as a halo. That is the
    same artifact as the hard-edged mask blur this effect replaces, wearing a
    softer hat. Carrying coverage as alpha instead gets the occlusion right:
    the sharp subject covers the blurred background, and the background's blur
    never reaches into the subject.
    """
    layers = len(radii)
    pos = d * (layers - 1)
    lower = np.floor(pos).clip(0, layers - 2)
    frac = (pos - lower).astype(np.float32)
    lower = lower.astype(np.int32)

    # The bloom weight is built from the working-resolution pixels, like
    # everything else here. It is the one place that costs something real: a
    # highlight smaller than a working pixel has already been averaged down by
    # the downsample before its exponential is taken, so a pinpoint glint on a
    # 40 MP frame blooms less than it should. Raising `bloom` compensates, and
    # anything a few pixels across on the original arrives intact.
    bright = None
    if bloom > 0:
        lum = (src @ _LUMA) / 255.0
        bright = np.exp(bloom * (lum - 1.0))

    out = np.zeros(src.shape, dtype=np.float32)
    coverage = np.zeros(src.shape[:2] + (1,), dtype=np.float32)
    layer = np.empty(src.shape[:2] + (4 if bright is None else 5,), dtype=np.float32)
    for k in range(layers):
        # a pixel lands in band k as the lower of its pair (weight 1-frac) or as
        # the upper (weight frac), never both
        weight = np.where(lower == k, 1.0 - frac, 0.0)
        weight += np.where(lower + 1 == k, frac, 0.0)
        if not weight.any():
            continue  # nothing at this depth: real scenes leave bands empty
        layer[..., :3] = src * weight[..., None]  # premultiplied
        layer[..., 3] = weight
        if bright is not None:
            layer[..., 4] = weight * bright
        radius = int(round(radii[k]))
        # All three kernels are the same convolution over a different aperture,
        # so everything downstream is indifferent to which ran — including the
        # two channels riding along. Coverage goes through it, so occlusion
        # stays consistent; so does the exponential, which is what makes a
        # blooming highlight come out as a cat's eye or a ring rather than a
        # disk, and is most of what either aperture is for.
        #
        # Two arms, not three, because `bokeh` has already resolved the mode:
        # `swirl` and `hole` are never both non-zero, so a mirror aperture is
        # the untiled path with a hole in its kernel — which is right on its own
        # terms, since a mirror's pupil is the same at every point of the frame
        # and has nothing for `_swirl_mean`'s per-tile geometry to do.
        if swirl > 0:
            blurred = _swirl_mean(layer, radius, swirl)
        else:
            blurred = _disk_mean(layer, radius, hole)
        rgb, alpha = blurred[..., :3], blurred[..., 3:4]
        # One pass carries the exponential through the same aperture as the
        # colour, which is the point: the soft maximum has to be taken over
        # exactly the pixels the blur averaged, or it describes a different
        # neighbourhood. A radius under a pixel leaves both untouched, and the
        # algebra then returns L unchanged — sharp bands do not bloom.
        if bright is not None:
            rgb = _bloom_luminance(rgb, alpha, blurred[..., 4:5], bloom)
        # `over`, with the new band in front of everything accumulated so far —
        # which holds because k ascends with depth toward the viewer
        behind = 1.0 - alpha
        out = rgb + behind * out
        coverage = alpha + behind * coverage
    # Coverage falls short of 1 wherever the blurs spread a band's weight
    # outward faster than its neighbours filled in behind it; un-premultiplying
    # by what actually arrived is what keeps those places from darkening.
    return out / np.maximum(coverage, 1e-6)


def bokeh(img: np.ndarray, params: dict) -> np.ndarray:
    # Imported here rather than at module scope for two reasons, the first
    # fatal: db.py imports this module, and depth -> sam -> db closes the
    # circle. The second is _fit_kmeans's — nothing unrelated to bokeh should
    # stop working because an onnxruntime wheel is broken on this platform.
    from . import depth

    h, w = img.shape[:2]
    d_small = depth.depth_map(img)  # float32 0-1 at the model's resolution

    if params["show"] == "depth":
        # What the model saw, so `focus` can be aimed at something. Bilinear up
        # from ~0.4 MP: this is a view of the depth map, not a smoothing of it.
        grey = Image.fromarray((d_small * 255).astype(np.uint8))
        return np.asarray(grey.resize((w, h), Image.Resampling.BILINEAR).convert("RGB"))

    focus = float(params["focus"])
    falloff = float(params["falloff"])
    # nodes made before bloom and the apertures existed carry none of them, the
    # same reason blur reads "kernel" with a default — and "lens" at swirl 0 is
    # what they rendered with
    bloom = float(params.get("bloom", 0.0))
    # The aperture mode is resolved here and nowhere else: past this point there
    # is no mode, only a `swirl` and a `hole` of which at most one is non-zero.
    # That is what keeps the exclusivity from having to be re-stated by every
    # kernel, and what leaves `_vignette` untouched — see `APERTURE_MODES`. A
    # mirror aperture therefore does not vignette at all, which is not an
    # omission: the corner darkening belongs to the barrel that clips a
    # photographic pupil, and a catadioptric's is famously flat by comparison.
    mirror = params.get("aperture", "lens") == "mirror"
    swirl = 0.0 if mirror else float(params.get("swirl", 0.0))
    hole = float(params.get("obstruction", 0.0)) if mirror else 0.0
    # `amount` is a percentage of the long side, where blur's `radius` is in
    # pixels: a depth-of-field setting has to mean the same thing on the 800x600
    # and the 40 MP frames in one library, and the blur that reads as "portrait
    # lens" is a fraction of the frame, not a count of pixels.
    r_full = float(params["amount"]) / 100.0 * max(h, w)
    # Distance from the focal plane, normalized so the far end of the scene
    # reaches exactly `amount`.
    span = max(focus, 1.0 - focus)

    if params["show"] == "kernels":
        # Before the working canvas is sized: the chart is about the blur you
        # get, and `scale` is only ever about how cheaply it is computed.
        density = int(params.get("density", _KERNEL_COLS))
        return _kernel_chart(img, d_small, focus, falloff, span, r_full, swirl, hole, density)

    scale = min(1.0, _BOKEH_WORK_PX / max(h, w), _BOKEH_WORK_R / max(r_full, 1e-6))
    ww, wh = max(1, round(w * scale)), max(1, round(h * scale))
    r_work = r_full * scale
    if r_work < 1.0:
        # sub-pixel everywhere; there is no blur to apply. The aperture is still
        # the aperture, though — skipping the vignette here would make corner
        # brightness jump the instant `amount` crossed this threshold. A mirror
        # aperture has nothing to preserve: it does not vignette, and a hole in
        # a sub-pixel kernel is a hole in nothing.
        return _vignette(img, swirl) if swirl > 0 else img.copy()

    # BOX going down is an area average — a subsample would alias the very
    # detail the blur is meant to dissolve.
    src_im = Image.fromarray(img).resize((ww, wh), Image.Resampling.BOX)
    src = np.asarray(src_im, dtype=np.float32)
    d = np.asarray(
        Image.fromarray(d_small).resize((ww, wh), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )

    layers = int(np.clip(round(r_work / 3), _BOKEH_MIN_LAYERS, _BOKEH_MAX_LAYERS))
    # Linear (falloff 1) is not a stand-in for something better: a real circle
    # of confusion is proportional to |1/z_focus - 1/z|, and the model's output
    # *is* inverse depth, so a straight ramp on it is what the optics do. See
    # server/depth.py.
    band_t = np.abs(np.linspace(0.0, 1.0, layers) - focus) / span
    work = _bokeh_layers(src, d, r_work * band_t**falloff, bloom, swirl, hole)

    # Recombine at full resolution. Everything sharp comes from the original —
    # the whole point of the working canvas is that it is only ever asked for
    # pixels that are blurred — and the crossover is placed where the detail the
    # upsample threw away (about 1/scale pixels) is already smaller than the
    # blur being applied, so there is nothing to see at either end of it.
    soft = 1.0 / scale
    lo, hi = max(2.0, 1.5 * soft), max(6.0, 4.0 * soft)
    pixel_t = np.abs(d - focus) / span
    mix = np.clip((r_full * pixel_t**falloff - lo) / (hi - lo), 0.0, 1.0)
    mix = mix * mix * (3.0 - 2.0 * mix)  # smoothstep: no seam at either end

    blurred = Image.fromarray(work.clip(0, 255).astype(np.uint8))
    mask = Image.fromarray((mix * 255).astype(np.uint8), mode="L")
    # Image.composite rather than a numpy lerp: at 40 MP the float temporaries
    # would be half a gigabyte, and this is the same arithmetic in C.
    out = np.asarray(
        Image.composite(
            blurred.resize((w, h), Image.Resampling.BICUBIC),
            Image.fromarray(img),
            mask.resize((w, h), Image.Resampling.BILINEAR),
        )
    )
    # Last, and over sharp and blurred pixels alike — see `_vignette`.
    return _vignette(out, swirl) if swirl > 0 else out


def sobel_edges(img: np.ndarray, params: dict) -> np.ndarray:
    threshold = int(params["threshold"])
    gray = img.astype(np.float32) @ _LUMA
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
        "label": "Blur",
        "apply": gaussian_blur,
        "params": [
            {"name": "radius", "label": "Radius", "type": "int", "min": 1, "max": 100, "default": 4},
            {"name": "kernel", "label": "Kernel", "type": "choice", "options": BLUR_KERNELS, "default": "gaussian"},
        ],
    },
    "bokeh": {
        "label": "Bokeh",
        "apply": bokeh,
        "params": [
            {"name": "amount", "label": "Amount (% of frame)", "type": "float", "min": 0.1, "max": 6.0, "step": 0.1, "default": 1.5},
            # `pick` says this float can be read off the image instead of dialled
            # in — the frontend grows a button that arms the same picker
            # click-to-segment uses and fills the slider from the depth under
            # the click. Presentation only: the param is still a float, and
            # `validate_params` never sees the key.
            {"name": "focus", "label": "Focus (1 = nearest)", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "default": 1.0, "pick": "depth"},
            {"name": "falloff", "label": "Falloff (1 = optical)", "type": "float", "min": 0.3, "max": 3.0, "step": 0.1, "default": 1.0},
            {"name": "bloom", "label": "Highlight bloom", "type": "float", "min": 0.0, "max": 12.0, "step": 0.5, "default": 0.0},
            # The dropdown sits immediately above the two sliders it switches
            # between, since only one of them applies at a time and nothing in
            # the panel can hide the other — `density`'s precedent. "lens" is
            # the default because it is what every node already on disk
            # rendered as, no `aperture` key having existed when it was saved.
            {"name": "aperture", "label": "Aperture", "type": "choice", "options": APERTURE_MODES, "default": "lens"},
            {"name": "swirl", "label": "Swirl (lens: shape + vignette)", "type": "float", "min": 0.0, "max": 0.85, "step": 0.05, "default": 0.0},
            # Hole diameter over aperture diameter. Defaults to a real
            # catadioptric's ~0.4 rather than to 0, unlike every other param
            # here, because it is gated behind the dropdown: a 0 default would
            # make choosing "mirror" do nothing at all and read as broken.
            {"name": "obstruction", "label": "Hole Ø (mirror)", "type": "float", "min": 0.0, "max": 0.85, "step": 0.05, "default": 0.4},
            # Read by the "kernels" view alone, and sitting next to `show` for
            # that reason. The panel has no way to show a param conditionally,
            # and a diagnostic knob is not worth growing a mode system in
            # `buildParamControls` for.
            {"name": "density", "label": "Kernel grid (columns)", "type": "int", "min": 3, "max": 32, "step": 1, "default": _KERNEL_COLS},
            {"name": "show", "label": "Show", "type": "choice", "options": SHOW_MODES, "default": "bokeh"},
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


# ---------- Crop & rotate: an output stage, not an effect ----------
#
# A crop is one framing per image, applied after every node on the way out, so
# it is deliberately *not* an entry in EFFECTS: every registry effect maps an
# array to an array of the same size, and the app leans on that everywhere
# (masked apply, blend's resize, `rendering.compute_mask`). Keeping the crop
# outside the work tree is what lets that invariant stay true — every node still
# renders at the original's dimensions, so no saved mask is ever warped.
#
# It lives beside BLEND_SPEC for the same reason blend does: a spec the frontend
# needs and `validate_params` must accept, that is not a registry entry.

# The smallest fraction of the canvas a frame may shrink to. A zero-width frame
# has no pixels, and PIL's crop of an empty box returns an image JPEG cannot
# encode.
MIN_RECT = 0.01

CROP_SPEC = {
    "label": "Frame",
    "params": [
        # positive is counter-clockwise, which is PIL's direction — the label
        # says so because a bare number gives no other clue
        {"name": "angle", "label": "Rotate (° ccw)", "type": "float", "min": -90.0, "max": 90.0, "step": 0.1, "default": 0.0},
        {"name": "rect", "label": "Frame", "type": "rect", "min": MIN_RECT, "default": [0.0, 0.0, 1.0, 1.0]},
    ],
}

IDENTITY_CROP = {"angle": 0.0, "rect": [0.0, 0.0, 1.0, 1.0]}

# Named specs `validate_params` accepts that are not entries in EFFECTS.
NON_EFFECT_SPECS = {"blend": BLEND_SPEC, "crop": CROP_SPEC}


def is_identity_crop(crop: dict | None) -> bool:
    """True when a crop would leave the image untouched, so callers can skip the
    whole output stage — an uncropped library must pay nothing, in time or in
    bytes on disk."""
    if not crop:
        return True
    clean = validate_params("crop", crop)
    return clean["angle"] == 0.0 and clean["rect"] == IDENTITY_CROP["rect"]


def _rotate_transform(size: tuple[int, int], angle: float) -> tuple[tuple[int, int], list[float]]:
    """The expanded canvas and inverse affine of `Image.rotate(angle, expand=True)`.

    This is PIL's own arithmetic, not a formula for it. The obvious
    `round(w·cos + h·sin)` disagrees with PIL by 1-2 px at most angles (400×300
    at −13°: PIL 458×384, the formula 457×382), and we need the matrix regardless
    — `Image.transform`'s AFFINE matrix *is* the output→source map the frontend
    uses to place a click. Deriving the size and the matrix from two different
    places is the drift this avoids, so they come out of one computation.

    The 15-place rounding is PIL's: it makes cos(±90°) exactly 0, which is what
    makes a quarter turn land on integer pixels instead of 6e-17 off them.
    """
    w, h = size
    cx, cy = w / 2.0, h / 2.0
    a = -math.radians(angle)
    m = [
        round(math.cos(a), 15), round(math.sin(a), 15), 0.0,
        round(-math.sin(a), 15), round(math.cos(a), 15), 0.0,
    ]

    def xform(x: float, y: float) -> tuple[float, float]:
        return m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]

    m[2], m[5] = xform(-cx, -cy)
    m[2] += cx
    m[5] += cy
    if angle % 360.0 in (90.0, 270.0):
        # PIL short-circuits a quarter turn to a transpose, which is exactly
        # (h, w). The general path's ceil/floor can come out a pixel larger on
        # odd dimensions (2×3 at 90° → 4×3, not 3×2), so follow the fast path.
        nw, nh = h, w
    else:
        xs, ys = zip(*(xform(x, y) for x, y in ((0, 0), (w, 0), (w, h), (0, h))))
        nw = math.ceil(max(xs)) - math.floor(min(xs))
        nh = math.ceil(max(ys)) - math.floor(min(ys))
    # the expand offset is *rotated*, not added: PIL multiplies a translation
    # matrix in from the right, so it arrives through the same transform
    m[2], m[5] = xform(-(nw - w) / 2.0, -(nh - h) / 2.0)
    return (nw, nh), m


def crop_geometry(size: tuple[int, int], crop: dict | None) -> dict:
    """How an image's output pixels relate to its source pixels — the one place
    the crop's geometry is worked out.

    - `canvas` is the rotation's expanded size, `box` the frame taken from it in
      whole pixels, `output` the result's size.
    - `inverse` is `[a, b, c, d, e, f]` mapping an *output* pixel back to a
      *source* pixel: `sx = a·ox + b·oy + c`, `sy = d·ox + e·oy + f`. The
      frontend uses it to turn a click on the framed preview into a coordinate in
      the node's own space, which is what keeps trigonometry — and PIL's expand
      rounding — out of the browser entirely.

    `rect` is fractions of the *rotated* canvas, so a frame survives a change of
    angle. Corners the rotation leaves empty are inside that canvas and render
    black, which needs no code: `rotate` fills them and `crop` pads with zeros.
    """
    crop = validate_params("crop", crop or {})
    angle = crop["angle"]
    if angle:
        canvas, m = _rotate_transform(size, angle)
    else:
        canvas, m = (size[0], size[1]), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    cw, ch = canvas
    x, y, rw, rh = crop["rect"]
    left, top = round(x * cw), round(y * ch)
    # at least one pixel each way, whatever the rounding did to a tiny frame
    right = max(left + 1, round((x + rw) * cw))
    bottom = max(top + 1, round((y + rh) * ch))
    return {
        "crop": crop,
        "source": [size[0], size[1]],
        "canvas": [cw, ch],
        "box": [left, top, right, bottom],
        "output": [right - left, bottom - top],
        # composing the box offset into the translation is the whole of "then
        # crop": a crop is a translation of the output origin
        "inverse": [
            m[0], m[1], m[0] * left + m[1] * top + m[2],
            m[3], m[4], m[3] * left + m[4] * top + m[5],
        ],
    }


def apply_geometry(im: Image.Image, crop: dict | None, resample) -> Image.Image:
    """Rotate about the center onto an expanded canvas, then take the frame from
    that canvas (Lightroom's straighten model).

    Takes its box from `crop_geometry` rather than recomputing it, so the pixels
    a caller gets are always the ones its `inverse` describes. Shared by the
    photo path (bicubic) and the mask-outline path (nearest) for the same reason
    `_curve_lut` has exactly one implementation per side of the wire.
    """
    geom = crop_geometry(im.size, crop)
    if geom["crop"]["angle"]:
        im = im.rotate(
            geom["crop"]["angle"], resample=resample, expand=True, fillcolor=0
        )
    return im.crop(tuple(geom["box"]))


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


def _clean_rect(value, p: dict) -> list[float]:
    """Coerce a crop frame into `[x, y, w, h]` fractions `crop_geometry` can
    always eat. Total for the same reason as `_clean_points`: `validate_params`
    is called unguarded, so a raise here is a 500 rather than a 400.

    The frame is clamped inside the rotated canvas rather than allowed to hang
    off it. Black corners are still reachable — they are *inside* the canvas the
    rotation expanded to — so nothing is lost by keeping the numbers in 0..1.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return list(p["default"])
    try:
        x, y, w, h = (float(v) for v in value)
    except (TypeError, ValueError):
        return list(p["default"])
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return list(p["default"])
    x = min(max(x, 0.0), 1.0 - p["min"])
    y = min(max(y, 0.0), 1.0 - p["min"])
    # clamp the size against the origin, so a frame never extends past the edge
    w = min(max(w, p["min"]), 1.0 - x)
    h = min(max(h, p["min"]), 1.0 - y)
    return [x, y, w, h]


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
    # blend and crop are the two specs that are not registry entries: one takes
    # two images, the other is not an effect at all
    spec = NON_EFFECT_SPECS.get(effect) or EFFECTS[effect]
    clean = {}
    for p in spec["params"]:
        value = params.get(p["name"], p["default"])
        if p["type"] == "choice":
            clean[p["name"]] = value if value in p["options"] else p["default"]
        elif p["type"] == "points":
            clean[p["name"]] = _clean_points(value, p)
        elif p["type"] == "rect":
            clean[p["name"]] = _clean_rect(value, p)
        elif p["type"] == "float":
            value = float(value)
            clean[p["name"]] = max(p["min"], min(p["max"], value))
        else:
            value = int(value)
            clean[p["name"]] = max(p["min"], min(p["max"], value))
    return clean
