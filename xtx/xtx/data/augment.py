"""Geometric + photometric augmentation (spec section 8.5).

Augmentation runs on the original image (and pixel-space targets) *before* the
A/B/C crop generation, so the cascade geometry stays consistent. Random scale and
translation of the image naturally augment which people land in the B/C centre
regions -- key for the centre-recovery objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..utils.keypoints import build_flip_index


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

    height, width = image.shape[:2]
    flipped = np.ascontiguousarray(image[:, ::-1])
    new_boxes = boxes.copy()
    if boxes.shape[0] > 0:
        new_boxes[:, 0] = width - boxes[:, 2]
        new_boxes[:, 2] = width - boxes[:, 0]
    new_kpts = kpts.copy()
    if kpts.shape[0] > 0:
        new_kpts = new_kpts[:, flip_index, :]
        visible = new_kpts[:, :, 2] > 0
        new_kpts[:, :, 0] = np.where(visible, width - new_kpts[:, :, 0], new_kpts[:, :, 0])
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


def random_affine(
    image: np.ndarray,
    boxes: np.ndarray,
    kpts: np.ndarray,
    *,
    scale_jitter: float,
    translate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random scale + translation about the image centre (same output size)."""

    height, width = image.shape[:2]
    scale = float(np.random.uniform(1.0 - scale_jitter, 1.0 + scale_jitter))
    scale = max(scale, 0.1)
    max_tx = translate * width
    max_ty = translate * height
    tx = float(np.random.uniform(-max_tx, max_tx))
    ty = float(np.random.uniform(-max_ty, max_ty))
    cx, cy = width / 2.0, height / 2.0

    matrix = np.array(
        [
            [scale, 0.0, cx - scale * cx + tx],
            [0.0, scale, cy - scale * cy + ty],
        ],
        dtype=np.float64,
    )
    warped = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    new_boxes = boxes.copy()
    if boxes.shape[0] > 0:
        corners = np.concatenate(
            [boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]], axis=0
        )
        mapped = (np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1) @ matrix.T)
        mapped = mapped.reshape(4, boxes.shape[0], 2)
        x_coords = mapped[:, :, 0]
        y_coords = mapped[:, :, 1]
        new_boxes = np.stack(
            [x_coords.min(0), y_coords.min(0), x_coords.max(0), y_coords.max(0)], axis=1
        ).astype(np.float32)

    new_kpts = kpts.copy()
    if kpts.shape[0] > 0:
        flat = kpts[:, :, :2].reshape(-1, 2)
        mapped = (np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1) @ matrix.T)
        new_kpts[:, :, :2] = mapped.reshape(kpts.shape[0], kpts.shape[1], 2).astype(np.float32)
    return warped, new_boxes, new_kpts


class Augmentor:
    """Applies the configured augmentation pipeline to (image, boxes, kpts)."""

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
        if cfg.scale_jitter > 0 or cfg.translate > 0:
            image, boxes, kpts = random_affine(
                image, boxes, kpts, scale_jitter=cfg.scale_jitter, translate=cfg.translate
            )
        if np.random.random() < cfg.fliplr:
            image, boxes, kpts = horizontal_flip(image, boxes, kpts, flip_index=self._flip_index)
        image = hsv_jitter(image, h_gain=cfg.hsv_h, s_gain=cfg.hsv_s, v_gain=cfg.hsv_v)
        return image, boxes, kpts


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
