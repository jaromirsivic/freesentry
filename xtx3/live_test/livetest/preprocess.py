"""Whole-frame / pyramid letterbox geometry (numpy port of ``xtx3.data.preprocess``).

Must stay bit-for-bit identical to the training implementation; verified by
``tests/test_preprocess_parity.py``.

* **whole**: the frame is letterboxed into one ``S x S`` view (aspect preserved,
  never upscaled, centred, black padding, paste offset aligned down to 4).
* **pyramid**: three ``S x S`` views of a virtual ``4S x 4S`` canvas -- view 1 =
  whole canvas (4x reduction), view 2 = centred ``2S`` (2x), view 3 = centred ``S``.

Every view carries a 2x3 affine ``T_k`` mapping network pixels -> frame pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WHOLE_VIEW_KEY = 0
PYRAMID_VIEW_KEYS = (1, 2, 3)
_OFFSET_ALIGN = 4


def view_keys_for_mode(mode: str) -> tuple[int, ...]:
    if mode == "whole":
        return (WHOLE_VIEW_KEY,)
    if mode == "pyramid":
        return PYRAMID_VIEW_KEYS
    raise ValueError(f"Unknown view mode {mode!r}")


def view_regions(mode: str, network_size: int) -> dict[int, int]:
    """Canvas-space square side covered by each view."""

    if mode == "whole":
        return {WHOLE_VIEW_KEY: network_size}
    if mode == "pyramid":
        return {1: 4 * network_size, 2: 2 * network_size, 3: network_size}
    raise ValueError(f"Unknown view mode {mode!r}")


def canvas_size_for_mode(mode: str, network_size: int) -> int:
    return network_size if mode == "whole" else 4 * network_size


@dataclass(slots=True)
class Placement:
    canvas_size: int
    orig_w: int
    orig_h: int
    scale_x: float
    scale_y: float
    paste_x: int
    paste_y: int


def compute_placement(*, width: int, height: int, canvas_size: int) -> Placement:
    """Canonical letterbox placement (downscale only, centred, offsets aligned to 4)."""

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
    )


def _forward_affine(placement: Placement, region: int, network_size: int) -> np.ndarray:
    """Affine **frame pixel -> view network pixel**."""

    r = region / float(network_size)
    origin = (placement.canvas_size - region) // 2
    return np.array(
        [
            [placement.scale_x / r, 0.0, (placement.paste_x - origin) / r],
            [0.0, placement.scale_y / r, (placement.paste_y - origin) / r],
        ],
        dtype=np.float64,
    )


def _invert_affine(matrix: np.ndarray) -> np.ndarray:
    full = np.eye(3, dtype=np.float64)
    full[:2, :] = matrix
    return np.linalg.inv(full)[:2, :]


def _render_view(image: np.ndarray, forward: np.ndarray, network_size: int) -> np.ndarray:
    """Render one ``S x S`` view with at most one interpolation."""

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
    mode: str
    network_size: int
    views: dict[int, np.ndarray]  # key -> uint8 (S, S, 3) BGR
    transforms: dict[int, np.ndarray]  # key -> (2, 3) network px -> frame px
    orig_size: tuple[int, int]  # (W, H)


def prepare_views(image: np.ndarray, *, mode: str, network_size: int) -> PreparedViews:
    """Letterbox ``image`` (BGR uint8) into the network views of ``mode``."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 BGR frame, got shape {image.shape}")
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width = image.shape[:2]
    placement = compute_placement(
        width=width, height=height, canvas_size=canvas_size_for_mode(mode, network_size)
    )
    views: dict[int, np.ndarray] = {}
    transforms: dict[int, np.ndarray] = {}
    for key, region in view_regions(mode, network_size).items():
        forward = _forward_affine(placement, region, network_size)
        views[key] = _render_view(image, forward, network_size)
        transforms[key] = _invert_affine(forward)
    return PreparedViews(
        mode=mode,
        network_size=network_size,
        views=views,
        transforms=transforms,
        orig_size=(width, height),
    )


def apply_affine_points(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine to ``(..., 2)`` points."""

    return xy @ matrix[:, :2].T + matrix[:, 2]
