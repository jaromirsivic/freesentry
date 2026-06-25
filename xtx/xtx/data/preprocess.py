"""A/B/C crop cascade preprocessing (spec section 4).

This module is the **single** implementation used by both training and inference,
guaranteeing bit-for-bit identical geometry. Given an OpenCV BGR image it:

1. Normalises size so the working image has ``min(W, H) >= 1080``
   (black-pad small images centred, or aspect-preserving resize of large ones).
2. Takes the centred square ``S x S`` and resizes it to exactly ``1080 x 1080``
   (the spec's recommended simpler path: section 4.3).
3. Builds three ``384 x 384`` crops:
   * **A** = whole 1080 square resized to 384 (full field of view).
   * **B** = central 768 region resized to 384 (exact 2x downscale, INTER_AREA).
   * **C** = central 384 region used natively (pure slice, no interpolation).

Each crop carries an affine ``T_k`` mapping **network-pixel (384 space) ->
original-image pixel**, so detections map straight back with one matrix multiply.
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

SQUARE_SIZE: Final[int] = 1080
NETWORK_SIZE: Final[int] = 384
REGION_B: Final[int] = 768
REGION_C: Final[int] = 384

CROP_KEYS: Final[tuple[str, str, str]] = ("A", "B", "C")


def coerce_bgr(image: np.ndarray) -> np.ndarray:
    """Coerce arbitrary input to a contiguous ``HxWx3`` uint8 BGR array."""

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
class PreprocessResult:
    """Output of :func:`preprocess_image`.

    Attributes:
        crops: mapping ``{"A"|"B"|"C": uint8 (384,384,3)}``.
        transforms: mapping ``{"A"|"B"|"C": (2,3) affine}`` (network 384 -> original px).
        orig_size: ``(W, H)`` of the original input image.
        square_bgr: the normalised 1080x1080 square (kept for debugging/augmentation).
    """

    crops: dict[str, np.ndarray]
    transforms: dict[str, AffineMatrix]
    orig_size: tuple[int, int]
    square_bgr: np.ndarray


def _size_normalisation_affine(*, width: int, height: int) -> tuple[np.ndarray, AffineMatrix]:
    """Return ``(working_image_plan, T_working_to_original)``.

    Instead of materialising here, return a small plan describing how to build the
    working image plus the affine mapping working-pixel -> original-pixel.
    The plan is ``(mode, params)``.
    """

    min_side = min(width, height)
    if min_side < SQUARE_SIZE:
        canvas_w = max(SQUARE_SIZE, width)
        canvas_h = max(SQUARE_SIZE, height)
        paste_x = (canvas_w - width) // 2
        paste_y = (canvas_h - height) // 2
        plan = np.array(
            [0, canvas_w, canvas_h, paste_x, paste_y], dtype=np.int64
        )  # mode 0 = pad
        t_working_to_orig = geometry.make_affine(
            scale_x=1.0, scale_y=1.0, tx=float(-paste_x), ty=float(-paste_y)
        )
        return plan, t_working_to_orig

    # min_side >= 1080: resize so the shorter side becomes exactly 1080 (no-op if ==1080).
    resize_scale = SQUARE_SIZE / float(min_side)
    new_w = int(round(width * resize_scale))
    new_h = int(round(height * resize_scale))
    plan = np.array([1, new_w, new_h, 0, 0], dtype=np.int64)  # mode 1 = resize
    t_working_to_orig = geometry.make_affine(
        scale_x=1.0 / resize_scale, scale_y=1.0 / resize_scale, tx=0.0, ty=0.0
    )
    return plan, t_working_to_orig


def _build_working_image(image: np.ndarray, plan: np.ndarray) -> np.ndarray:
    mode = int(plan[0])
    if mode == 0:
        canvas_w, canvas_h, paste_x, paste_y = (int(plan[1]), int(plan[2]), int(plan[3]), int(plan[4]))
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        h, w = image.shape[:2]
        canvas[paste_y : paste_y + h, paste_x : paste_x + w] = image
        return canvas
    new_w, new_h = int(plan[1]), int(plan[2])
    interp = cv2.INTER_AREA if (new_w < image.shape[1]) else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def compute_transforms(*, width: int, height: int) -> dict[str, AffineMatrix]:
    """Compute the three ``T_k`` affines for an image of size ``width x height``.

    This is the geometry-only path (no pixel work) used by the target transform in
    the dataset to map GT annotations into 384 network space via ``invert(T_k)``.
    """

    plan, t_working_to_orig = _size_normalisation_affine(width=width, height=height)
    if int(plan[0]) == 0:
        working_w, working_h = int(plan[1]), int(plan[2])
    else:
        working_w, working_h = int(plan[1]), int(plan[2])

    square_side = min(working_w, working_h)
    ox = (working_w - square_side) // 2
    oy = (working_h - square_side) // 2

    # square (1080) pixel -> working pixel: scale by square_side/1080, then offset.
    scale_sq_to_working = square_side / float(SQUARE_SIZE)
    t_square_to_working = geometry.make_affine(
        scale_x=scale_sq_to_working, scale_y=scale_sq_to_working, tx=float(ox), ty=float(oy)
    )
    t_square_to_orig = geometry.compose(t_working_to_orig, t_square_to_working)

    offset_b = (SQUARE_SIZE - REGION_B) // 2
    offset_c = (SQUARE_SIZE - REGION_C) // 2

    # net (384) pixel -> 1080-square pixel, per crop.
    t_a_crop = geometry.make_affine(
        scale_x=SQUARE_SIZE / NETWORK_SIZE, scale_y=SQUARE_SIZE / NETWORK_SIZE, tx=0.0, ty=0.0
    )
    t_b_crop = geometry.make_affine(
        scale_x=REGION_B / NETWORK_SIZE, scale_y=REGION_B / NETWORK_SIZE,
        tx=float(offset_b), ty=float(offset_b),
    )
    t_c_crop = geometry.make_affine(
        scale_x=1.0, scale_y=1.0, tx=float(offset_c), ty=float(offset_c)
    )

    return {
        "A": geometry.compose(t_square_to_orig, t_a_crop),
        "B": geometry.compose(t_square_to_orig, t_b_crop),
        "C": geometry.compose(t_square_to_orig, t_c_crop),
    }


def build_abc_crops(square_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Build the A/B/C 384x384 crops from a 1080x1080 square."""

    if square_bgr.shape[:2] != (SQUARE_SIZE, SQUARE_SIZE):
        raise ValueError(f"square must be {SQUARE_SIZE}x{SQUARE_SIZE}, got {square_bgr.shape[:2]}")

    crop_a = cv2.resize(square_bgr, (NETWORK_SIZE, NETWORK_SIZE), interpolation=cv2.INTER_AREA)

    offset_b = (SQUARE_SIZE - REGION_B) // 2
    region_b = square_bgr[offset_b : offset_b + REGION_B, offset_b : offset_b + REGION_B]
    crop_b = cv2.resize(region_b, (NETWORK_SIZE, NETWORK_SIZE), interpolation=cv2.INTER_AREA)

    offset_c = (SQUARE_SIZE - REGION_C) // 2
    crop_c = np.ascontiguousarray(
        square_bgr[offset_c : offset_c + REGION_C, offset_c : offset_c + REGION_C]
    )

    return {"A": crop_a, "B": crop_b, "C": crop_c}


def preprocess_image(image: np.ndarray) -> PreprocessResult:
    """Full preprocessing: coerce -> size-norm -> square -> 1080 -> A/B/C + transforms."""

    bgr = coerce_bgr(image)
    height, width = bgr.shape[:2]

    plan, _ = _size_normalisation_affine(width=width, height=height)
    working = _build_working_image(bgr, plan)
    working_h, working_w = working.shape[:2]

    square_side = min(working_w, working_h)
    ox = (working_w - square_side) // 2
    oy = (working_h - square_side) // 2
    square = working[oy : oy + square_side, ox : ox + square_side]
    if square_side != SQUARE_SIZE:
        interp = cv2.INTER_AREA if square_side > SQUARE_SIZE else cv2.INTER_LINEAR
        square = cv2.resize(square, (SQUARE_SIZE, SQUARE_SIZE), interpolation=interp)
    square = np.ascontiguousarray(square)

    crops = build_abc_crops(square)
    transforms = compute_transforms(width=width, height=height)

    return PreprocessResult(
        crops=crops,
        transforms=transforms,
        orig_size=(width, height),
        square_bgr=square,
    )
