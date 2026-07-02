"""Geometric + photometric augmentation (spec section 8.5).

Augmentation is applied at the **canvas stage** so geometry stays consistent
across the three levels:

* Horizontal flip (with left/right keypoint index swap) and photometric jitter
  operate on the original image / pixel-space targets before letterboxing.
* Random scale jitter and translation perturb the **canvas placement** itself
  (``CanvasPlacement`` scale/offset), which naturally varies which people land
  in the level_2/level_3 centre regions -- key for the centre-recovery objective.
* Optional 4-image mosaic composes source images before letterboxing and is
  disabled for the last ``close_mosaic_epochs`` epochs by the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..utils.keypoints import build_flip_index
from .preprocess import CanvasPlacement, compute_placement


@dataclass(slots=True)
class AugmentConfig:
    fliplr: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    scale_jitter: float = 0.5
    translate: float = 0.1
    mosaic: float = 0.5

    @classmethod
    def from_dict(cls, data: dict | None) -> "AugmentConfig":
        data = data or {}
        return cls(
            fliplr=float(data.get("fliplr", 0.5)),
            hsv_h=float(data.get("hsv_h", 0.015)),
            hsv_s=float(data.get("hsv_s", 0.7)),
            hsv_v=float(data.get("hsv_v", 0.4)),
            scale_jitter=float(data.get("scale_jitter", 0.5)),
            translate=float(data.get("translate", 0.1)),
            mosaic=float(data.get("mosaic", 0.5)),
        )


def horizontal_flip(
    image: np.ndarray, boxes: np.ndarray, kpts: np.ndarray, *, flip_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flip image + targets horizontally, swapping left/right keypoint indices."""

    width = image.shape[1]
    flipped = np.ascontiguousarray(image[:, ::-1])
    new_boxes = boxes.copy()
    if boxes.shape[0] > 0:
        new_boxes[:, 0] = width - boxes[:, 2]
        new_boxes[:, 2] = width - boxes[:, 0]
    new_kpts = kpts.copy()
    if kpts.shape[0] > 0:
        new_kpts = new_kpts[:, flip_index, :]
        present = new_kpts[:, :, 2] > 0
        new_kpts[:, :, 0] = np.where(present, width - new_kpts[:, :, 0], new_kpts[:, :, 0])
    return flipped, new_boxes, new_kpts


def hsv_jitter(image: np.ndarray, *, h_gain: float, s_gain: float, v_gain: float) -> np.ndarray:
    """Random HSV jitter (in-gamut), OpenCV BGR in/out."""

    if h_gain <= 0 and s_gain <= 0 and v_gain <= 0:
        return image
    gains = np.random.uniform(-1, 1, 3) * np.array([h_gain, s_gain, v_gain]) + 1
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] * gains[0]) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * gains[1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * gains[2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def jittered_placement(
    *,
    width: int,
    height: int,
    scale_jitter: float,
    translate: float,
) -> CanvasPlacement:
    """Random scale + translation of the canvas placement (section 8.5).

    Scale multiplies the canonical letterbox scale by ``U(1-j, 1+j)`` (clamped so
    the pasted image never exceeds the canvas by more than 2x) and the paste
    offset is shifted by up to ``translate * 1536`` in each axis.
    """

    factor = 1.0
    if scale_jitter > 0:
        factor = float(np.random.uniform(max(0.1, 1.0 - scale_jitter), 1.0 + scale_jitter))
    offset_x = offset_y = 0
    if translate > 0:
        limit = translate * 1536
        offset_x = int(np.random.uniform(-limit, limit))
        offset_y = int(np.random.uniform(-limit, limit))
    return compute_placement(
        width=width, height=height, scale_factor=factor, offset_x=offset_x, offset_y=offset_y
    )


class Augmentor:
    """Applies the pixel-space augmentation pipeline to (image, boxes, kpts)."""

    def __init__(self, *, config: AugmentConfig, enabled: bool = True) -> None:
        self.config = config
        self.enabled = enabled
        self._flip_index = build_flip_index()

    def __call__(
        self, image: np.ndarray, boxes: np.ndarray, kpts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.enabled:
            return image, boxes, kpts
        cfg = self.config
        if np.random.random() < cfg.fliplr:
            image, boxes, kpts = horizontal_flip(image, boxes, kpts, flip_index=self._flip_index)
        image = hsv_jitter(image, h_gain=cfg.hsv_h, s_gain=cfg.hsv_s, v_gain=cfg.hsv_v)
        return image, boxes, kpts

    def placement(self, *, width: int, height: int) -> CanvasPlacement:
        """Return the (possibly jittered) canvas placement for this sample."""

        if not self.enabled:
            return compute_placement(width=width, height=height)
        return jittered_placement(
            width=width,
            height=height,
            scale_jitter=self.config.scale_jitter,
            translate=self.config.translate,
        )


def mosaic4(
    images: list[np.ndarray],
    boxes_list: list[np.ndarray],
    kpts_list: list[np.ndarray],
    *,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose four images into a single ``output_size x output_size`` mosaic."""

    if len(images) != 4:
        raise ValueError("mosaic4 requires exactly 4 images")
    half = output_size // 2
    canvas = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    quadrants = [(0, 0), (half, 0), (0, half), (half, half)]
    out_boxes: list[np.ndarray] = []
    out_kpts: list[np.ndarray] = []
    for (ox, oy), img, boxes, kpts in zip(quadrants, images, boxes_list, kpts_list):
        resized = cv2.resize(img, (half, half), interpolation=cv2.INTER_LINEAR)
        canvas[oy : oy + half, ox : ox + half] = resized
        sx = half / img.shape[1]
        sy = half / img.shape[0]
        if boxes.shape[0] > 0:
            b = boxes.copy()
            b[:, [0, 2]] = b[:, [0, 2]] * sx + ox
            b[:, [1, 3]] = b[:, [1, 3]] * sy + oy
            out_boxes.append(b)
        if kpts.shape[0] > 0:
            k = kpts.copy()
            k[:, :, 0] = k[:, :, 0] * sx + ox
            k[:, :, 1] = k[:, :, 1] * sy + oy
            out_kpts.append(k)
    boxes_arr = np.concatenate(out_boxes, axis=0) if out_boxes else np.zeros((0, 4), np.float32)
    kpts_arr = np.concatenate(out_kpts, axis=0) if out_kpts else np.zeros((0, 17, 3), np.float32)
    return canvas, boxes_arr, kpts_arr
