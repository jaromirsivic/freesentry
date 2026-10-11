"""Torch-free letterbox for the runtime (NumPy/OpenCV port of ``xtx4.data.preprocess``).

Must stay **bit-for-bit identical** to the training/evaluation geometry
(``xtx4.data.preprocess.preprocess_image``); ``tests/test_runtime.py`` checks the
rendered views and the affines on random images. It is a separate copy (rather
than an import) so the deployment runtime has no dependency on the training
package layout -- only ``numpy`` and ``cv2``.

* **whole** (default, XTX4-u): the frame is letterboxed into one ``S x S`` view:
  aspect preserved, never upscaled, centred, black padding, paste offset aligned
  down to a multiple of 4, ``INTER_AREA`` when downscaling.
* **pyramid** (kept for the n/m/l ablation): three ``S x S`` views of a virtual
  ``4S x 4S`` canvas -- view 1 = whole canvas (4x), view 2 = centred ``2S`` (2x),
  view 3 = centred ``S`` (1x).

Every view carries a 2x3 float64 affine ``T_k`` mapping **network px -> frame px**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

WHOLE_VIEW_KEY: Final[int] = 0
PYRAMID_VIEW_KEYS: Final[tuple[int, int, int]] = (1, 2, 3)
VIEW_MODES: Final[tuple[str, str]] = ("whole", "pyramid")
_OFFSET_ALIGN: Final[int] = 4


def view_keys_for_mode(mode: str) -> tuple[int, ...]:
    if mode == "whole":
        return (WHOLE_VIEW_KEY,)
    if mode == "pyramid":
        return PYRAMID_VIEW_KEYS
    raise ValueError(f"Unknown view mode {mode!r}; expected one of {VIEW_MODES}")


def view_regions(mode: str, network_size: int) -> dict[int, int]:
    """Canvas-space square side covered by each view."""

    if mode == "whole":
        return {WHOLE_VIEW_KEY: network_size}
    if mode == "pyramid":
        return {1: 4 * network_size, 2: 2 * network_size, 3: network_size}
    raise ValueError(f"Unknown view mode {mode!r}; expected one of {VIEW_MODES}")


def canvas_size_for_mode(mode: str, network_size: int) -> int:
    return network_size if mode == "whole" else 4 * network_size


def coerce_bgr(image: np.ndarray) -> np.ndarray:
    """Contiguous ``HxWx3`` uint8 BGR (gray promoted, alpha dropped)."""

    if image is None:
        raise ValueError("image is None")
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    elif not (arr.ndim == 3 and arr.shape[2] == 3):
        raise ValueError(f"Unsupported image shape {arr.shape}; expected HxW, HxWx1/3/4")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


@dataclass(slots=True)
class Placement:
    """Canonical letterbox placement of a frame on the (virtual) square canvas."""

    canvas_size: int
    orig_w: int
    orig_h: int
    scale_x: float
    scale_y: float
    paste_x: int
    paste_y: int
    resized_w: int
    resized_h: int


def compute_placement(*, width: int, height: int, canvas_size: int) -> Placement:
    """Downscale-only, centred placement with offsets aligned down to 4."""

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size {width}x{height}")
    scale = min(1.0, canvas_size / float(max(width, height)))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    paste_x = ((canvas_size - resized_w) // 2 // _OFFSET_ALIGN) * _OFFSET_ALIGN
    paste_y = ((canvas_size - resized_h) // 2 // _OFFSET_ALIGN) * _OFFSET_ALIGN
    return Placement(
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


def _forward_affine(placement: Placement, region: int, network_size: int) -> np.ndarray:
    """Affine **frame px -> view network px** for the centred canvas region ``region``."""

    r = region / float(network_size)
    origin = (placement.canvas_size - region) // 2
    return np.array(
        [
            [placement.scale_x / r, 0.0, (placement.paste_x - origin) / r],
            [0.0, placement.scale_y / r, (placement.paste_y - origin) / r],
        ],
        dtype=np.float64,
    )


def invert_affine(matrix: np.ndarray) -> np.ndarray:
    full = np.eye(3, dtype=np.float64)
    full[:2, :] = matrix
    return np.linalg.inv(full)[:2, :]


def _render_view(image: np.ndarray, forward: np.ndarray, network_size: int) -> np.ndarray:
    """Render one ``S x S`` view with at most one interpolation (ROI resize + paste)."""

    ax, bx = forward[0, 0], forward[0, 2]
    ay, by = forward[1, 1], forward[1, 2]
    height, width = image.shape[:2]
    out = np.zeros((network_size, network_size, 3), dtype=np.uint8)

    dx0 = max(0, int(round(bx)))
    dy0 = max(0, int(round(by)))
    dx1 = min(network_size, int(round(ax * width + bx)))
    dy1 = min(network_size, int(round(ay * height + by)))
    if dx1 <= dx0 or dy1 <= dy0:
        return out

    sx0 = max(0, int(round((dx0 - bx) / ax)))
    sy0 = max(0, int(round((dy0 - by) / ay)))
    sx1 = min(width, int(round((dx1 - bx) / ax)))
    sy1 = min(height, int(round((dy1 - by) / ay)))
    if sx1 <= sx0 or sy1 <= sy0:
        return out

    src = image[sy0:sy1, sx0:sx1]
    dst_w, dst_h = dx1 - dx0, dy1 - dy0
    if src.shape[1] == dst_w and src.shape[0] == dst_h:
        out[dy0:dy1, dx0:dx1] = src
    else:
        downscale = dst_w < src.shape[1] or dst_h < src.shape[0]
        interp = cv2.INTER_AREA if downscale else cv2.INTER_LINEAR
        out[dy0:dy1, dx0:dx1] = cv2.resize(src, (dst_w, dst_h), interpolation=interp)
    return out


@dataclass(slots=True)
class PreparedViews:
    """Network views of one frame.

    ``views``: key -> uint8 ``(S, S, 3)`` BGR; ``transforms``: key -> ``(2, 3)``
    network px -> frame px; ``orig_size``: ``(W, H)``.
    """

    mode: str
    network_size: int
    views: dict[int, np.ndarray]
    transforms: dict[int, np.ndarray]
    orig_size: tuple[int, int]
    placement: Placement


def prepare_views(image: np.ndarray, *, network_size: int, mode: str = "whole") -> PreparedViews:
    """Letterbox ``image`` (BGR) into the network views of ``mode`` (default whole-frame)."""

    if mode not in VIEW_MODES:
        raise ValueError(f"Unknown view mode {mode!r}; expected one of {VIEW_MODES}")
    bgr = coerce_bgr(image)
    height, width = bgr.shape[:2]
    placement = compute_placement(width=width, height=height, canvas_size=canvas_size_for_mode(mode, network_size))
    views: dict[int, np.ndarray] = {}
    transforms: dict[int, np.ndarray] = {}
    for key, region in view_regions(mode, network_size).items():
        forward = _forward_affine(placement, region, network_size)
        views[key] = _render_view(bgr, forward, network_size)
        transforms[key] = invert_affine(forward)
    return PreparedViews(
        mode=mode, network_size=network_size, views=views, transforms=transforms,
        orig_size=(width, height), placement=placement,
    )


def letterbox(image: np.ndarray, network_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Whole-frame shortcut: ``(view uint8 (S, S, 3), T (2, 3) network px -> frame px)``."""

    prepared = prepare_views(image, network_size=network_size, mode="whole")
    return prepared.views[WHOLE_VIEW_KEY], prepared.transforms[WHOLE_VIEW_KEY]


def to_blob(view_bgr: np.ndarray) -> np.ndarray:
    """``(S, S, 3)`` uint8 BGR -> ``(1, 3, S, S)`` float32 ``/255`` (the graph input contract)."""

    return np.ascontiguousarray(view_bgr.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255.0)


def apply_affine_points(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine to ``(..., 2)`` points."""

    return xy @ matrix[:, :2].T + matrix[:, 2]
