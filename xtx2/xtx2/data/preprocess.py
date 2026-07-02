"""Centre-focused 3-level pyramid preprocessing (spec section 4).

This module is the **single** implementation used by both training and inference,
guaranteeing bit-for-bit identical geometry. The pyramid is defined on a virtual
**1536x1536 canvas** (``1536 = 4 x 384``):

1. The whole input image is **letterboxed** onto the canvas: aspect-preserving
   downscale so the long side fits 1536 (never upscaled), pasted centred with
   black padding. Nothing is ever cropped away at level_1.
2. Three ``384 x 384`` levels are cut from the (virtual) canvas:
   * **level_1** = whole 1536 canvas -> 384 (exact 4x reduction).
   * **level_2** = centred 768 region -> 384 (exact 2x reduction).
   * **level_3** = centred 384 region, used as-is (native slice).

**One-shot rendering (section 4.3):** the full canvas is never materialised in
the production path. Each level is produced with at most ONE interpolation
directly from the original image, by composing the letterbox affine with the
level affine and resizing the corresponding source ROI straight into a black
384 canvas (``INTER_AREA`` for downscales). In the small-image branch
(``s = 1``) level_3 is a pure pixel copy with zero interpolation.

Implementation note: paste offsets are aligned down to a multiple of 4 (all
level scales 4/2/1 divide 4), so the composed ``original -> level_k`` transform
always has an integral translation in level space. The ROI-resize + paste
rendering therefore realises the recorded affine exactly; the letterbox stays
centred to within 3 px.

Each level carries an affine ``T_k`` mapping **network pixel (384 space) ->
original image pixel** (section 4.4), so detections map back with one matrix
multiply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from ..utils import geometry
from ..utils.geometry import AffineMatrix
from ..utils.logging import get_logger

logger = get_logger(__name__)

CANVAS_SIZE: Final[int] = 1536
NETWORK_SIZE: Final[int] = 384

# Canvas-space region side per level: level_1 sees the whole canvas, level_2 the
# centred 768 square, level_3 the centred 384 square.
LEVEL_REGIONS: Final[dict[int, int]] = {1: 1536, 2: 768, 3: 384}
LEVEL_KEYS: Final[tuple[int, int, int]] = (1, 2, 3)

# Paste offsets are aligned down to a multiple of this so all three level
# transforms have integral translations (4 = max canvas/level scale ratio).
_OFFSET_ALIGN: Final[int] = 4


def coerce_bgr(image: np.ndarray) -> np.ndarray:
    """Coerce arbitrary input to a contiguous ``HxWx3`` uint8 BGR array.

    Grayscale is promoted to 3 channels; 4-channel input has alpha dropped
    (section 2.1 / 16).
    """

    if image is None:
        raise ValueError("image is None")
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Unsupported image shape {arr.shape}; expected HxW, HxWx1/3/4")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


@dataclass(slots=True)
class CanvasPlacement:
    """Letterbox placement of the original image on the virtual 1536 canvas.

    Attributes:
        orig_w / orig_h: original image size.
        scale_x / scale_y: effective original->canvas scale per axis. Both equal
            the section 4.1 scale ``s`` up to the +-0.5 px resize rounding, and
            are exactly 1.0 in the small-image branch.
        paste_x / paste_y: integer paste offset ``(px, py)`` on the canvas.
        resized_w / resized_h: integer size of the pasted image on the canvas.
    """

    orig_w: int
    orig_h: int
    scale_x: float
    scale_y: float
    paste_x: int
    paste_y: int
    resized_w: int
    resized_h: int


def compute_placement(
    *,
    width: int,
    height: int,
    scale_factor: float = 1.0,
    offset_x: int = 0,
    offset_y: int = 0,
) -> CanvasPlacement:
    """Compute the section 4.1 letterbox placement for a ``width x height`` image.

    ``scale_factor``/``offset_x``/``offset_y`` perturb the canonical placement
    (used by the training augmentation to jitter the canvas placement, section
    8.5); the defaults produce the exact spec geometry:

    * large image (``max(W,H) > 1536``): scale ``s = 1536 / max(W,H)``, centred.
    * small image: pasted centred, unscaled (``s = 1``, no upscaling).
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size {width}x{height}")

    base_scale = min(1.0, CANVAS_SIZE / float(max(width, height)))
    s = base_scale * float(scale_factor)
    resized_w = max(1, int(round(width * s)))
    resized_h = max(1, int(round(height * s)))

    paste_x = (CANVAS_SIZE - resized_w) // 2 + int(offset_x)
    paste_y = (CANVAS_SIZE - resized_h) // 2 + int(offset_y)
    # Align down to a multiple of 4 so every level transform is integral.
    paste_x = (paste_x // _OFFSET_ALIGN) * _OFFSET_ALIGN
    paste_y = (paste_y // _OFFSET_ALIGN) * _OFFSET_ALIGN

    return CanvasPlacement(
        orig_w=width,
        orig_h=height,
        scale_x=resized_w / float(width),
        scale_y=resized_h / float(height),
        paste_x=paste_x,
        paste_y=paste_y,
        resized_w=resized_w,
        resized_h=resized_h,
    )


def _forward_affine(placement: CanvasPlacement, level: int) -> AffineMatrix:
    """Affine mapping **original image pixel -> level ``level`` 384 pixel**."""

    region = LEVEL_REGIONS[level]
    r = region / float(NETWORK_SIZE)  # canvas px per level px: 4, 2, 1
    origin = (CANVAS_SIZE - region) // 2  # canvas-space region origin: 0, 384, 576
    return geometry.make_affine(
        scale_x=placement.scale_x / r,
        scale_y=placement.scale_y / r,
        tx=(placement.paste_x - origin) / r,
        ty=(placement.paste_y - origin) / r,
    )


def transforms_from_placement(placement: CanvasPlacement) -> dict[int, AffineMatrix]:
    """Return ``{level: T_k}`` where ``T_k`` maps network (384) px -> original px."""

    return {
        level: geometry.invert_affine(_forward_affine(placement, level))
        for level in LEVEL_KEYS
    }


def compute_transforms(*, width: int, height: int) -> dict[int, AffineMatrix]:
    """Geometry-only path: the three canonical ``T_k`` for an unaugmented image."""

    return transforms_from_placement(compute_placement(width=width, height=height))


def _render_level(image: np.ndarray, placement: CanvasPlacement, level: int) -> np.ndarray:
    """Render one 384x384 level with at most one interpolation (section 4.3).

    The composed ``original -> level`` transform is a pure scale + translation;
    we resize the source ROI covered by the level's field of view straight into
    a black 384 canvas. When the composed scale is exactly 1 with integral
    offsets (level_3 in the small-image branch) this is a pure pixel copy.
    """

    forward = _forward_affine(placement, level)
    ax, bx = forward[0, 0], forward[0, 2]
    ay, by = forward[1, 1], forward[1, 2]
    height, width = image.shape[:2]

    out = np.zeros((NETWORK_SIZE, NETWORK_SIZE, 3), dtype=np.uint8)

    # Destination rect (level space) covered by the original image, clipped.
    dx0 = max(0, int(round(bx)))
    dy0 = max(0, int(round(by)))
    dx1 = min(NETWORK_SIZE, int(round(ax * width + bx)))
    dy1 = min(NETWORK_SIZE, int(round(ay * height + by)))
    if dx1 <= dx0 or dy1 <= dy0:
        return out  # image entirely outside this level's field of view

    # Source ROI (original-image space) mapped from the destination rect.
    sx0 = max(0, int(round((dx0 - bx) / ax)))
    sy0 = max(0, int(round((dy0 - by) / ay)))
    sx1 = min(width, int(round((dx1 - bx) / ax)))
    sy1 = min(height, int(round((dy1 - by) / ay)))
    if sx1 <= sx0 or sy1 <= sy0:
        return out

    src = image[sy0:sy1, sx0:sx1]
    dst_w, dst_h = dx1 - dx0, dy1 - dy0
    if src.shape[1] == dst_w and src.shape[0] == dst_h:
        out[dy0:dy1, dx0:dx1] = src  # pure copy, zero interpolation
    else:
        interp = cv2.INTER_AREA if (dst_w < src.shape[1] or dst_h < src.shape[0]) else cv2.INTER_LINEAR
        out[dy0:dy1, dx0:dx1] = cv2.resize(src, (dst_w, dst_h), interpolation=interp)
    return out


def build_levels(image: np.ndarray, placement: CanvasPlacement) -> dict[int, np.ndarray]:
    """Build the three 384x384 level images one-shot from the original image."""

    return {level: _render_level(image, placement, level) for level in LEVEL_KEYS}


def build_canvas(image: np.ndarray, placement: CanvasPlacement) -> np.ndarray:
    """Materialise the full 1536x1536 letterbox canvas.

    Reference/debug path only (used by tests to validate the one-shot rendering);
    the production pipeline never calls this.
    """

    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
    if (placement.resized_w, placement.resized_h) == (placement.orig_w, placement.orig_h):
        resized = image
    else:
        interp = cv2.INTER_AREA if placement.resized_w < placement.orig_w else cv2.INTER_LINEAR
        resized = cv2.resize(image, (placement.resized_w, placement.resized_h), interpolation=interp)
    x0 = max(0, placement.paste_x)
    y0 = max(0, placement.paste_y)
    x1 = min(CANVAS_SIZE, placement.paste_x + placement.resized_w)
    y1 = min(CANVAS_SIZE, placement.paste_y + placement.resized_h)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = resized[
            y0 - placement.paste_y : y1 - placement.paste_y,
            x0 - placement.paste_x : x1 - placement.paste_x,
        ]
    return canvas


@dataclass(slots=True)
class PreprocessResult:
    """Output of :func:`preprocess_image`.

    Attributes:
        levels: mapping ``{1|2|3: uint8 (384,384,3)}``.
        transforms: mapping ``{1|2|3: (2,3) float64 affine}`` (network 384 -> original px).
        orig_size: ``(W, H)`` of the original input image.
        placement: the letterbox placement used (for debugging / augmentation).
    """

    levels: dict[int, np.ndarray]
    transforms: dict[int, AffineMatrix]
    orig_size: tuple[int, int]
    placement: CanvasPlacement


def preprocess_image(
    image: np.ndarray, *, placement: CanvasPlacement | None = None
) -> PreprocessResult:
    """Full preprocessing: coerce -> letterbox placement -> one-shot levels + T_k.

    ``placement`` may be supplied to render an augmented (jittered) placement;
    by default the canonical section 4.1 placement is used.
    """

    bgr = coerce_bgr(image)
    height, width = bgr.shape[:2]
    if placement is None:
        placement = compute_placement(width=width, height=height)
    levels = build_levels(bgr, placement)
    transforms = transforms_from_placement(placement)
    return PreprocessResult(
        levels=levels,
        transforms=transforms,
        orig_size=(width, height),
        placement=placement,
    )
