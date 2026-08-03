"""Whole-frame and centre-focused pyramid preprocessing.

This module is the **single** implementation used by both training and
inference, guaranteeing bit-for-bit identical geometry. Two view modes exist:

**Whole-frame mode** (``mode="whole"``, the XTX3-u default): the image is
letterboxed straight into one ``S x S`` network view (aspect-preserving, never
upscaled, centred with black padding). One network pass per frame.

**Pyramid mode** (``mode="pyramid"``, evaluated against whole-frame per the
spec): three ``S x S`` views cut from a virtual ``4S x 4S`` canvas (for
``S = 384`` this is the exact XTX2 geometry, enabling a like-for-like
comparison):

1. The input is letterboxed onto the canvas (downscale so the long side fits
   ``4S``, never upscaled, centred with black padding).
2. Three views are cut from the (virtual) canvas:
   * **view 1** = whole ``4S`` canvas -> ``S`` (exact 4x reduction).
   * **view 2** = centred ``2S`` region -> ``S`` (exact 2x reduction).
   * **view 3** = centred ``S`` region, used as-is (native slice).

**One-shot rendering:** the full canvas is never materialised in the
production path. Each view is produced with at most ONE interpolation directly
from the original image, by composing the letterbox affine with the view
affine and resizing the corresponding source ROI straight into a black ``S``
canvas (``INTER_AREA`` for downscales). Paste offsets are aligned down to a
multiple of 4 (all view scales 4/2/1 divide 4) so the composed
``original -> view`` transform always has an integral translation in view
space; the ROI-resize + paste rendering therefore realises the recorded affine
exactly.

Each view carries an affine ``T_k`` mapping **network pixel -> original image
pixel**, so detections map back with one matrix multiply. The whole-frame view
uses key ``0``; pyramid views use keys ``1 | 2 | 3``.
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

DEFAULT_NETWORK_SIZE: Final[int] = 384

WHOLE_VIEW_KEY: Final[int] = 0
PYRAMID_VIEW_KEYS: Final[tuple[int, int, int]] = (1, 2, 3)

VIEW_MODES: Final[tuple[str, str]] = ("whole", "pyramid")

# Paste offsets are aligned down to a multiple of this so all view transforms
# have integral translations (4 = max canvas/view scale ratio).
_OFFSET_ALIGN: Final[int] = 4


def view_keys_for_mode(mode: str) -> tuple[int, ...]:
    """Return the ordered view keys for a mode (order == merge priority)."""

    if mode == "whole":
        return (WHOLE_VIEW_KEY,)
    if mode == "pyramid":
        return PYRAMID_VIEW_KEYS
    raise ValueError(f"Unknown view mode {mode!r}; expected one of {VIEW_MODES}")


def view_regions(mode: str, network_size: int) -> dict[int, int]:
    """Canvas-space square side per view key for the given mode and size."""

    if mode == "whole":
        return {WHOLE_VIEW_KEY: network_size}
    if mode == "pyramid":
        return {1: 4 * network_size, 2: 2 * network_size, 3: network_size}
    raise ValueError(f"Unknown view mode {mode!r}; expected one of {VIEW_MODES}")


def canvas_size_for_mode(mode: str, network_size: int) -> int:
    """Virtual canvas side: ``S`` for whole-frame, ``4S`` for the pyramid."""

    return network_size if mode == "whole" else 4 * network_size


def coerce_bgr(image: np.ndarray) -> np.ndarray:
    """Coerce arbitrary input to a contiguous ``HxWx3`` uint8 BGR array.

    Grayscale is promoted to 3 channels; 4-channel input has alpha dropped.
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
    """Letterbox placement of the original image on a virtual square canvas.

    Attributes:
        canvas_size: side of the virtual canvas (``S`` whole, ``4S`` pyramid).
        orig_w / orig_h: original image size.
        scale_x / scale_y: effective original->canvas scale per axis (equal to
            the letterbox scale up to +-0.5 px resize rounding; exactly 1.0 in
            the small-image branch).
        paste_x / paste_y: integer paste offset ``(px, py)`` on the canvas.
        resized_w / resized_h: integer size of the pasted image on the canvas.
    """

    canvas_size: int
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
    canvas_size: int,
    scale_factor: float = 1.0,
    offset_x: int = 0,
    offset_y: int = 0,
) -> CanvasPlacement:
    """Compute the letterbox placement of a ``width x height`` image on a canvas.

    ``scale_factor``/``offset_x``/``offset_y`` perturb the canonical placement
    (used by the training augmentation to jitter the placement); the defaults
    produce the canonical geometry:

    * large image (``max(W,H) > canvas``): scale ``s = canvas / max(W,H)``, centred.
    * small image: pasted centred, unscaled (``s = 1``, no upscaling).
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size {width}x{height}")
    if canvas_size <= 0:
        raise ValueError(f"Invalid canvas size {canvas_size}")

    base_scale = min(1.0, canvas_size / float(max(width, height)))
    s = base_scale * float(scale_factor)
    resized_w = max(1, int(round(width * s)))
    resized_h = max(1, int(round(height * s)))

    paste_x = (canvas_size - resized_w) // 2 + int(offset_x)
    paste_y = (canvas_size - resized_h) // 2 + int(offset_y)
    # Align down to a multiple of 4 so every view transform is integral.
    paste_x = (paste_x // _OFFSET_ALIGN) * _OFFSET_ALIGN
    paste_y = (paste_y // _OFFSET_ALIGN) * _OFFSET_ALIGN

    return CanvasPlacement(
        canvas_size=canvas_size,
        orig_w=width,
        orig_h=height,
        scale_x=resized_w / float(width),
        scale_y=resized_h / float(height),
        paste_x=paste_x,
        paste_y=paste_y,
        resized_w=resized_w,
        resized_h=resized_h,
    )


def _forward_affine(placement: CanvasPlacement, region: int, network_size: int) -> AffineMatrix:
    """Affine mapping **original image pixel -> view network pixel**.

    ``region`` is the canvas-space square side covered by the view (centred).
    """

    r = region / float(network_size)  # canvas px per view px: 4, 2, or 1
    origin = (placement.canvas_size - region) // 2  # canvas-space region origin
    return geometry.make_affine(
        scale_x=placement.scale_x / r,
        scale_y=placement.scale_y / r,
        tx=(placement.paste_x - origin) / r,
        ty=(placement.paste_y - origin) / r,
    )


def transforms_from_placement(
    placement: CanvasPlacement, *, mode: str, network_size: int
) -> dict[int, AffineMatrix]:
    """Return ``{view_key: T_k}`` where ``T_k`` maps network px -> original px."""

    regions = view_regions(mode, network_size)
    return {
        key: geometry.invert_affine(_forward_affine(placement, region, network_size))
        for key, region in regions.items()
    }


def compute_transforms(
    *, width: int, height: int, mode: str = "pyramid", network_size: int = DEFAULT_NETWORK_SIZE
) -> dict[int, AffineMatrix]:
    """Geometry-only path: canonical ``T_k`` for an unaugmented image."""

    placement = compute_placement(
        width=width, height=height, canvas_size=canvas_size_for_mode(mode, network_size)
    )
    return transforms_from_placement(placement, mode=mode, network_size=network_size)


def _render_view(
    image: np.ndarray, placement: CanvasPlacement, region: int, network_size: int
) -> np.ndarray:
    """Render one ``S x S`` view with at most one interpolation.

    The composed ``original -> view`` transform is a pure scale + translation;
    we resize the source ROI covered by the view's field of view straight into
    a black canvas. When the composed scale is exactly 1 with integral offsets
    this is a pure pixel copy.
    """

    forward = _forward_affine(placement, region, network_size)
    ax, bx = forward[0, 0], forward[0, 2]
    ay, by = forward[1, 1], forward[1, 2]
    height, width = image.shape[:2]

    out = np.zeros((network_size, network_size, 3), dtype=np.uint8)

    # Destination rect (view space) covered by the original image, clipped.
    dx0 = max(0, int(round(bx)))
    dy0 = max(0, int(round(by)))
    dx1 = min(network_size, int(round(ax * width + bx)))
    dy1 = min(network_size, int(round(ay * height + by)))
    if dx1 <= dx0 or dy1 <= dy0:
        return out  # image entirely outside this view's field of view

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


def render_view(
    image: np.ndarray, placement: CanvasPlacement, *, view_key: int, mode: str, network_size: int
) -> np.ndarray:
    """Render a single view identified by its key (training path)."""

    region = view_regions(mode, network_size)[view_key]
    return _render_view(image, placement, region, network_size)


def build_views(
    image: np.ndarray, placement: CanvasPlacement, *, mode: str, network_size: int
) -> dict[int, np.ndarray]:
    """Build all ``S x S`` views one-shot from the original image."""

    regions = view_regions(mode, network_size)
    return {
        key: _render_view(image, placement, region, network_size)
        for key, region in regions.items()
    }


def build_canvas(image: np.ndarray, placement: CanvasPlacement) -> np.ndarray:
    """Materialise the full letterbox canvas.

    Reference/debug path only (used by tests to validate the one-shot rendering);
    the production pipeline never calls this.
    """

    size = placement.canvas_size
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if (placement.resized_w, placement.resized_h) == (placement.orig_w, placement.orig_h):
        resized = image
    else:
        interp = cv2.INTER_AREA if placement.resized_w < placement.orig_w else cv2.INTER_LINEAR
        resized = cv2.resize(image, (placement.resized_w, placement.resized_h), interpolation=interp)
    x0 = max(0, placement.paste_x)
    y0 = max(0, placement.paste_y)
    x1 = min(size, placement.paste_x + placement.resized_w)
    y1 = min(size, placement.paste_y + placement.resized_h)
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
        mode: ``"whole"`` or ``"pyramid"``.
        network_size: view side length ``S``.
        views: mapping ``{view_key: uint8 (S,S,3)}`` (key 0 whole, 1|2|3 pyramid).
        transforms: mapping ``{view_key: (2,3) float64 affine}`` (network -> original px).
        orig_size: ``(W, H)`` of the original input image.
        placement: the letterbox placement used (for debugging / augmentation).
    """

    mode: str
    network_size: int
    views: dict[int, np.ndarray]
    transforms: dict[int, AffineMatrix]
    orig_size: tuple[int, int]
    placement: CanvasPlacement


def preprocess_image(
    image: np.ndarray,
    *,
    mode: str = "pyramid",
    network_size: int = DEFAULT_NETWORK_SIZE,
    placement: CanvasPlacement | None = None,
) -> PreprocessResult:
    """Full preprocessing: coerce -> letterbox placement -> one-shot views + T_k.

    ``placement`` may be supplied to render an augmented (jittered) placement;
    by default the canonical placement is used.
    """

    if mode not in VIEW_MODES:
        raise ValueError(f"Unknown view mode {mode!r}; expected one of {VIEW_MODES}")
    bgr = coerce_bgr(image)
    height, width = bgr.shape[:2]
    if placement is None:
        placement = compute_placement(
            width=width, height=height, canvas_size=canvas_size_for_mode(mode, network_size)
        )
    views = build_views(bgr, placement, mode=mode, network_size=network_size)
    transforms = transforms_from_placement(placement, mode=mode, network_size=network_size)
    return PreprocessResult(
        mode=mode,
        network_size=network_size,
        views=views,
        transforms=transforms,
        orig_size=(width, height),
        placement=placement,
    )
