"""Geometric and photometric augmentation for the nine-landmark (head + torso) task.

Design rules (implementation_plan_fable.md §7.3):

* Every geometric op works on the source image in **source pixels** and returns,
  besides the new image and targets, the affine ``new px -> old px`` so the
  dataset can compose an exact ``network -> original image`` transform for
  non-mosaic samples.
* Points that leave the image get ``v = 0``; unknown points (``v = -1``) stay
  unknown whatever happens to the geometry.
* Scale / translation jitter perturbs the **letterbox placement** (no extra
  resampling): the dataset renders the augmented source straight into the
  ``S x S`` view with :func:`xtx4.data.preprocess.render_view`.
* ``mosaic4`` letterboxes each source into its quadrant (aspect preserved,
  black padding) instead of XTX3's stretching to a square.
* ``upper_body_crop`` simulates close-up camera framing by cutting the image
  below the hips or the shoulders of one person.
* Camera photometrics (motion blur k <= 5, gaussian noise sigma <= 8/255, JPEG 40-95)
  run on the source before letterboxing, so the black padding stays exactly
  black like at inference; HSV and gamma are applied by the dataset to the
  rendered view (cheaper; black stays black under both).

All randomness comes from an explicit ``numpy.random.Generator``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable, Final

import cv2
import numpy as np

from ..utils.geometry import AffineMatrix, make_affine
from ..utils.keypoints import FACE_INDICES, NUM_KEYPOINTS, build_flip_index
from ..utils.logging import get_logger
from .preprocess import CanvasPlacement, compute_placement

logger = get_logger(__name__)

_FLIP_INDEX: Final[np.ndarray] = build_flip_index()
_SHOULDERS: Final[tuple[int, int]] = (5, 6)
_HIPS: Final[tuple[int, int]] = (7, 8)


# --------------------------------------------------------------------------- config


@dataclass(slots=True)
class AugmentConfig:
    """Augmentation probabilities / ranges (``configs/u.json`` -> ``data.augment``)."""

    fliplr: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    scale_min: float = 0.5
    scale_max: float = 1.5
    translate: float = 0.1
    mosaic: float = 0.5
    upper_body_crop: float = 0.3
    motion_blur: float = 0.2
    noise: float = 0.2
    jpeg: float = 0.3
    gamma: float = 0.3
    # Ranges (not in u.json, rarely changed).
    motion_blur_max_k: int = 5
    noise_sigma_max: float = 8.0 / 255.0
    jpeg_quality_min: int = 40
    jpeg_quality_max: int = 95
    gamma_min: float = 0.7
    gamma_max: float = 1.4
    crop_min_area: float = 0.25

    @classmethod
    def from_dict(cls, data: dict | None) -> "AugmentConfig":
        data = dict(data or {})
        if "scale_jitter" in data:  # XTX3 key: symmetric jitter around 1
            jitter = float(data.pop("scale_jitter"))
            data.setdefault("scale_min", max(0.1, 1.0 - jitter))
            data.setdefault("scale_max", 1.0 + jitter)
        known = {f.name: f for f in fields(cls)}
        kwargs = {}
        for key, value in data.items():
            if key not in known:
                logger.warning("Unknown augment key ignored: %s", key)
                continue
            kwargs[key] = type(known[key].default)(value)
        cfg = cls(**kwargs)
        if cfg.scale_min <= 0 or cfg.scale_max < cfg.scale_min:
            raise ValueError(f"Invalid scale range {cfg.scale_min}..{cfg.scale_max}")
        return cfg

    @classmethod
    def disabled(cls) -> "AugmentConfig":
        """All probabilities zero, identity placement (useful for debugging)."""

        return cls(fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0, scale_min=1, scale_max=1, translate=0,
                   mosaic=0, upper_body_crop=0, motion_blur=0, noise=0, jpeg=0, gamma=0)


# --------------------------------------------------------------------------- targets


@dataclass(slots=True)
class SourceTargets:
    """Per-person targets in **source pixels** while the image is being augmented.

    ``keypoints (M,K,3)`` float64 ``[x, y, v]``; ``boxes (M,4)`` xyxy (NaN rows
    when unknown); ``area (M,)`` px² (NaN when unknown); ``iscrowd (M,)`` bool.
    """

    keypoints: np.ndarray
    boxes: np.ndarray
    area: np.ndarray
    iscrowd: np.ndarray

    @classmethod
    def empty(cls) -> "SourceTargets":
        return cls(
            keypoints=np.zeros((0, NUM_KEYPOINTS, 3), np.float64),
            boxes=np.zeros((0, 4), np.float64),
            area=np.zeros((0,), np.float64),
            iscrowd=np.zeros((0,), bool),
        )

    def __len__(self) -> int:
        return int(self.keypoints.shape[0])

    def copy(self) -> "SourceTargets":
        return SourceTargets(self.keypoints.copy(), self.boxes.copy(), self.area.copy(), self.iscrowd.copy())

    def scaled_translated(self, sx: float, sy: float, tx: float, ty: float) -> "SourceTargets":
        """Apply ``x' = sx·x + tx``, ``y' = sy·y + ty`` (sx, sy > 0) to labeled points and boxes."""

        out = self.copy()
        lab = out.keypoints[:, :, 2] > 0
        out.keypoints[:, :, 0] = np.where(lab, out.keypoints[:, :, 0] * sx + tx, out.keypoints[:, :, 0])
        out.keypoints[:, :, 1] = np.where(lab, out.keypoints[:, :, 1] * sy + ty, out.keypoints[:, :, 1])
        out.boxes[:, [0, 2]] = out.boxes[:, [0, 2]] * sx + tx
        out.boxes[:, [1, 3]] = out.boxes[:, [1, 3]] * sy + ty
        out.area = out.area * (sx * sy)
        return out

    def clip_to(self, width: int, height: int) -> "SourceTargets":
        """Labeled points outside ``[0,W) x [0,H)`` -> ``v = 0``; boxes clipped."""

        kp = self.keypoints
        lab = kp[:, :, 2] > 0
        inside = (kp[:, :, 0] >= 0) & (kp[:, :, 0] < width) & (kp[:, :, 1] >= 0) & (kp[:, :, 1] < height)
        kp[:, :, 2] = np.where(lab & ~inside, 0.0, kp[:, :, 2])
        self.boxes[:, [0, 2]] = np.clip(self.boxes[:, [0, 2]], 0, width)
        self.boxes[:, [1, 3]] = np.clip(self.boxes[:, [1, 3]], 0, height)
        return self

    @staticmethod
    def concatenate(items: list["SourceTargets"]) -> "SourceTargets":
        if not items:
            return SourceTargets.empty()
        return SourceTargets(
            keypoints=np.concatenate([t.keypoints for t in items], axis=0),
            boxes=np.concatenate([t.boxes for t in items], axis=0),
            area=np.concatenate([t.area for t in items], axis=0),
            iscrowd=np.concatenate([t.iscrowd for t in items], axis=0),
        )


# --------------------------------------------------------------------------- geometric ops


def horizontal_flip(image: np.ndarray, targets: SourceTargets) -> tuple[np.ndarray, SourceTargets, AffineMatrix]:
    """Mirror image and targets; swaps left/right landmarks and keeps ``v`` per landmark."""

    width = image.shape[1]
    flipped = np.ascontiguousarray(image[:, ::-1])
    out = targets.copy()
    if len(out):
        out.keypoints = out.keypoints[:, _FLIP_INDEX, :]
        lab = out.keypoints[:, :, 2] > 0
        out.keypoints[:, :, 0] = np.where(lab, width - out.keypoints[:, :, 0], out.keypoints[:, :, 0])
        x1 = width - targets.boxes[:, 2]
        x2 = width - targets.boxes[:, 0]
        out.boxes[:, 0], out.boxes[:, 2] = x1, x2
    # new px -> old px: x_old = W - x_new
    return flipped, out, np.array([[-1.0, 0.0, float(width)], [0.0, 1.0, 0.0]])


def _person_scale(kp: np.ndarray, height: int) -> float:
    """Rough torso length in px from labeled shoulders / hips / face points."""

    lab = kp[:, 2] > 0
    sho = [i for i in _SHOULDERS if lab[i]]
    hip = [i for i in _HIPS if lab[i]]
    if sho and hip:
        return max(4.0, abs(float(kp[hip, 1].mean() - kp[sho, 1].mean())))
    if len(sho) == 2:
        return max(4.0, 1.2 * float(np.hypot(*(kp[5, :2] - kp[6, :2]))))
    face = [i for i in FACE_INDICES if lab[i]]
    if len(face) >= 2:
        pts = kp[face, :2]
        return max(4.0, 2.5 * float(np.ptp(pts, axis=0).max()))
    return max(4.0, 0.1 * height)


def upper_body_crop(
    image: np.ndarray,
    targets: SourceTargets,
    rng: np.random.Generator,
    *,
    min_area_frac: float = 0.25,
) -> tuple[np.ndarray, SourceTargets, AffineMatrix] | None:
    """Crop so the image ends below the hips (or shoulders) of a random person.

    The chosen person keeps all of its labeled points above the cut line; the
    left/right/top borders are trimmed randomly but never across that person.
    The crop keeps at least ``min_area_frac`` of the image area (the bottom is
    lowered if needed, which keeps the "ends below" property). Returns ``None``
    when no person has a labeled point.
    """

    height, width = image.shape[:2]
    kp_all = targets.keypoints
    candidates = [i for i in range(len(targets)) if not targets.iscrowd[i] and (kp_all[i, :, 2] > 0).any()]
    if not candidates:
        return None
    kp = kp_all[int(rng.choice(candidates))]
    lab = kp[:, 2] > 0
    torso = _person_scale(kp, height)

    use_hips = bool(rng.random() < 0.5)
    hips = [i for i in _HIPS if lab[i]]
    shoulders = [i for i in _SHOULDERS if lab[i]]
    if use_hips and hips:
        ref_y = float(kp[hips, 1].max())
        margin = rng.uniform(0.05, 0.4) * torso
    elif shoulders:
        ref_y = float(kp[shoulders, 1].max())
        margin = rng.uniform(0.1, 0.6) * torso
    elif hips:
        ref_y = float(kp[hips, 1].max())
        margin = rng.uniform(0.05, 0.4) * torso
    else:
        ref_y = float(kp[lab, 1].max())
        margin = rng.uniform(0.2, 0.8) * torso

    xs, ys = kp[lab, 0], kp[lab, 1]
    pad = 0.5 * torso
    x0 = int(np.floor(rng.uniform(0.0, 0.5) * max(0.0, float(xs.min()) - pad)))
    x1 = int(np.ceil(width - rng.uniform(0.0, 0.5) * max(0.0, width - (float(xs.max()) + pad))))
    y0 = int(np.floor(rng.uniform(0.0, 0.5) * max(0.0, float(ys.min()) - pad)))
    y1 = int(min(height, np.ceil(ref_y + margin + 1.0)))
    min_area = min_area_frac * width * height
    if (x1 - x0) * (y1 - y0) < min_area:
        x0, x1, y0 = 0, width, 0
    if (x1 - x0) * (y1 - y0) < min_area:
        y1 = int(min(height, np.ceil(min_area / max(1, x1 - x0))))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, max(x1, x0 + 1)), min(height, max(y1, y0 + 1))

    cropped = np.ascontiguousarray(image[y0:y1, x0:x1])
    out = targets.scaled_translated(1.0, 1.0, -float(x0), -float(y0)).clip_to(x1 - x0, y1 - y0)
    return cropped, out, make_affine(scale_x=1.0, scale_y=1.0, tx=float(x0), ty=float(y0))


@dataclass(slots=True)
class MosaicTile:
    """Where a source landed in the mosaic: ``x' = scale_x·x + offset_x`` (same for y)."""

    scale_x: float
    scale_y: float
    offset_x: int
    offset_y: int
    width: int
    height: int


def mosaic4(
    images: list[np.ndarray],
    targets: list[SourceTargets],
    *,
    output_size: int,
    tile_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, SourceTargets, list[MosaicTile]]:
    """Compose four sources into an ``output_size²`` mosaic, one letterboxed source per quadrant.

    Each source keeps its aspect ratio (scale ``min(1, half/W, half/H)``, never
    upscaled) and is centred in its quadrant on black. Order: top-left,
    top-right, bottom-left, bottom-right. ``tile_fn`` (size-preserving, e.g.
    camera photometrics) is applied to each resized tile before pasting, so
    the padding stays black and the op runs on fewer pixels.
    """

    if len(images) != 4 or len(targets) != 4:
        raise ValueError("mosaic4 requires exactly 4 images and 4 target sets")
    half = output_size // 2
    canvas = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    quadrants = ((0, 0), (half, 0), (0, half), (half, half))
    tiles: list[MosaicTile] = []
    parts: list[SourceTargets] = []
    for (qx, qy), img, tgt in zip(quadrants, images, targets):
        h, w = img.shape[:2]
        s = min(1.0, half / float(w), half / float(h))
        rw, rh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        ox, oy = qx + (half - rw) // 2, qy + (half - rh) // 2
        if (rw, rh) == (w, h):
            resized = img
        else:
            interp = cv2.INTER_AREA if rw < w else cv2.INTER_LINEAR
            resized = cv2.resize(img, (rw, rh), interpolation=interp)
        if tile_fn is not None:
            resized = tile_fn(resized)
        canvas[oy : oy + rh, ox : ox + rw] = resized
        sx, sy = rw / float(w), rh / float(h)
        tiles.append(MosaicTile(sx, sy, ox, oy, rw, rh))
        parts.append(tgt.scaled_translated(sx, sy, float(ox), float(oy)))
    return canvas, SourceTargets.concatenate(parts), tiles


def jittered_placement(
    *, width: int, height: int, canvas_size: int, cfg: AugmentConfig, rng: np.random.Generator
) -> CanvasPlacement:
    """Letterbox placement with scale factor ``U(scale_min, scale_max)`` and translation."""

    factor = float(rng.uniform(cfg.scale_min, cfg.scale_max)) if cfg.scale_max > cfg.scale_min else cfg.scale_min
    offset_x = offset_y = 0
    if cfg.translate > 0:
        limit = cfg.translate * canvas_size
        offset_x = int(round(rng.uniform(-limit, limit)))
        offset_y = int(round(rng.uniform(-limit, limit)))
    return compute_placement(
        width=width, height=height, canvas_size=canvas_size,
        scale_factor=factor, offset_x=offset_x, offset_y=offset_y,
    )


# --------------------------------------------------------------------------- photometric ops

_LUT_X: Final[np.ndarray] = np.arange(256, dtype=np.float32)


def hsv_jitter(image: np.ndarray, rng: np.random.Generator, *, h_gain: float, s_gain: float, v_gain: float) -> np.ndarray:
    """YOLO-style HSV gain jitter via lookup tables (BGR in/out, uint8)."""

    if h_gain <= 0 and s_gain <= 0 and v_gain <= 0:
        return image
    r = rng.uniform(-1, 1, 3) * np.array([h_gain, s_gain, v_gain]) + 1.0
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    lut_h = ((_LUT_X * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(_LUT_X * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(_LUT_X * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def gamma_adjust(image: np.ndarray, rng: np.random.Generator, *, gamma_min: float = 0.7, gamma_max: float = 1.4) -> np.ndarray:
    gamma = float(rng.uniform(gamma_min, gamma_max))
    lut = np.clip(np.round(((_LUT_X / 255.0) ** gamma) * 255.0), 0, 255).astype(np.uint8)
    return cv2.LUT(image, lut)


def motion_blur(image: np.ndarray, rng: np.random.Generator, *, max_k: int = 5) -> np.ndarray:
    """Linear motion blur with odd kernel size ``3..max_k`` at a random angle."""

    sizes = [k for k in range(3, max(3, max_k) + 1, 2)]
    k = int(rng.choice(sizes))
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D(((k - 1) / 2.0, (k - 1) / 2.0), float(rng.uniform(0, 180)), 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    total = float(kernel.sum())
    if total <= 0:
        return image
    return cv2.filter2D(image, -1, kernel / total, borderType=cv2.BORDER_REFLECT)


def gaussian_noise(image: np.ndarray, rng: np.random.Generator, *, sigma_max: float = 8.0 / 255.0) -> np.ndarray:
    """Additive gaussian sensor noise, sigma ~ U(0, sigma_max) (in [0,1] units)."""

    sigma = float(rng.uniform(0.0, sigma_max)) * 255.0
    if sigma < 0.25:
        return image
    cv2.setRNGSeed(int(rng.integers(0, 2**31 - 1)))
    noise = np.empty(image.shape, np.int16)
    cv2.randn(noise, 0.0, sigma)
    return cv2.add(image, noise, dtype=cv2.CV_8U)


def jpeg_compress(image: np.ndarray, rng: np.random.Generator, *, q_min: int = 40, q_max: int = 95) -> np.ndarray:
    quality = int(rng.integers(q_min, q_max + 1))
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return image
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return image if decoded is None else decoded


# --------------------------------------------------------------------------- pipeline


class Augmentor:
    """Applies the configured ops with their probabilities."""

    def __init__(self, *, config: AugmentConfig) -> None:
        self.config = config

    def source(
        self, image: np.ndarray, targets: SourceTargets, rng: np.random.Generator
    ) -> tuple[np.ndarray, SourceTargets, AffineMatrix]:
        """Per-source ops: geometric (crop, flip) then camera photometrics."""

        image, targets, affine = self.geometric(image, targets, rng)
        return self.camera(image, rng), targets, affine

    def geometric(
        self, image: np.ndarray, targets: SourceTargets, rng: np.random.Generator
    ) -> tuple[np.ndarray, SourceTargets, AffineMatrix]:
        """Upper-body crop and flip. Returns ``(image, targets, A)``, ``A``: new px -> input px."""

        cfg = self.config
        affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        if cfg.upper_body_crop > 0 and rng.random() < cfg.upper_body_crop:
            res = upper_body_crop(image, targets, rng, min_area_frac=cfg.crop_min_area)
            if res is not None:
                image, targets, a = res
                affine = _compose(affine, a)
        if cfg.fliplr > 0 and rng.random() < cfg.fliplr:
            image, targets, a = horizontal_flip(image, targets)
            affine = _compose(affine, a)
        return image, targets, affine

    def camera(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Motion blur, sensor noise, JPEG re-compression (each with its own probability)."""

        cfg = self.config
        if cfg.motion_blur > 0 and rng.random() < cfg.motion_blur:
            image = motion_blur(image, rng, max_k=cfg.motion_blur_max_k)
        if cfg.noise > 0 and rng.random() < cfg.noise:
            image = gaussian_noise(image, rng, sigma_max=cfg.noise_sigma_max)
        if cfg.jpeg > 0 and rng.random() < cfg.jpeg:
            image = jpeg_compress(image, rng, q_min=cfg.jpeg_quality_min, q_max=cfg.jpeg_quality_max)
        return image

    def view(self, view_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Global colour ops on the rendered ``S x S`` view (padding stays black)."""

        cfg = self.config
        view_bgr = hsv_jitter(view_bgr, rng, h_gain=cfg.hsv_h, s_gain=cfg.hsv_s, v_gain=cfg.hsv_v)
        if cfg.gamma > 0 and rng.random() < cfg.gamma:
            view_bgr = gamma_adjust(view_bgr, rng, gamma_min=cfg.gamma_min, gamma_max=cfg.gamma_max)
        return view_bgr

    def placement(self, *, width: int, height: int, canvas_size: int, rng: np.random.Generator) -> CanvasPlacement:
        return jittered_placement(width=width, height=height, canvas_size=canvas_size, cfg=self.config, rng=rng)


def _compose(outer: AffineMatrix, inner: AffineMatrix) -> AffineMatrix:
    """``outer ∘ inner`` for 2x3 affines (``inner`` applied first)."""

    a = np.vstack([outer, [0.0, 0.0, 1.0]])
    b = np.vstack([inner, [0.0, 0.0, 1.0]])
    return (a @ b)[:2]
