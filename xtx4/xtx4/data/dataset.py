"""Training / evaluation dataset that renders every sample directly at the batch size ``S``.

Fixes relative to XTX3 (implementation_plan_fable.md §2.4):

* **D3** -- the sampler hands ``(idx, S)`` keys and the augmented source is
  letterboxed straight into ``S x S`` with :func:`preprocess.render_view`
  (one ``INTER_AREA`` resample, the deployment geometry). No tensor resizing.
* **D4** -- a person is a target when at least ``min_labeled_keypoints`` of its
  labeled landmarks fall **inside the rendered view**; the full-body box
  centre is never consulted (a head visible at the bottom edge with the body
  cut off is still a person).
* **D5** -- the visibility channel keeps the CONTRACTS §2 semantics: ``-1``
  unknown stays ``-1``; labeled points that leave the view become ``0``.

Targets follow CONTRACTS §5. ``SampleMeta.transform`` maps network px to the
**original image** px (crop / flip / placement composed) for non-mosaic
samples; for mosaic samples it maps to the mosaic canvas (``meta.mosaic``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils import geometry
from ..utils.geometry import AffineMatrix
from ..utils.keypoints import NUM_KEYPOINTS
from ..utils.logging import get_logger
from .annotations import ImageSample
from .augment import AugmentConfig, Augmentor, SourceTargets, mosaic4
from .preprocess import coerce_bgr, compute_placement, render_view, transforms_from_placement

logger = get_logger(__name__)

_MODE = "whole"
_VIEW = 0


@dataclass(slots=True)
class SampleMeta:
    """Per-sample metadata.

    Attributes:
        image_path: source image path (the first tile for mosaic samples).
        size: network size ``S`` of the rendered view.
        transform: ``(2,3)`` float64 affine, network px -> original image px
            (mosaic: -> mosaic canvas px).
        orig_size: ``(W, H)`` of the original image.
        sample_index: dataset index of the (primary) sample.
        mosaic: True when the view was rendered from a mosaic.
    """

    image_path: str
    size: int
    transform: np.ndarray
    orig_size: tuple[int, int]
    sample_index: int = -1
    mosaic: bool = False


# --------------------------------------------------------------------------- helpers


def read_image(path: Path | str) -> np.ndarray | None:
    """Read a BGR uint8 image (unicode-safe on Windows); ``None`` when unreadable."""

    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return None if img is None else coerce_bgr(img)


def denormalise_sample(sample: ImageSample, *, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(boxes_xyxy (M,4), keypoints (M,K,3))`` in pixels of a ``width x height`` image.

    Boxes are NaN rows when the annotation has none; ``v`` is copied unchanged
    (``-1`` unknown, ``0`` unlabeled); unlabeled points have coordinates 0.
    """

    num = len(sample.annotations)
    boxes = np.full((num, 4), np.nan, dtype=np.float32)
    kpts = np.zeros((num, NUM_KEYPOINTS, 3), dtype=np.float32)
    for i, ann in enumerate(sample.annotations):
        if ann.bbox_xywh_norm is not None:
            bx, by, bw, bh = (float(v) for v in ann.bbox_xywh_norm)
            boxes[i] = [bx * width, by * height, (bx + bw) * width, (by + bh) * height]
        kpts[i, :, 0] = ann.keypoints_norm[:, 0] * width
        kpts[i, :, 1] = ann.keypoints_norm[:, 1] * height
        kpts[i, :, 2] = ann.keypoints_norm[:, 2]
    return boxes, kpts


def source_targets(sample: ImageSample, *, width: int, height: int) -> SourceTargets:
    """Annotations of ``sample`` as :class:`SourceTargets` in source pixels."""

    boxes, kpts = denormalise_sample(sample, width=width, height=height)
    area = np.array(
        [np.nan if a.area_norm is None else a.area_norm * width * height for a in sample.annotations],
        dtype=np.float64,
    )
    iscrowd = np.array([bool(a.iscrowd) for a in sample.annotations], dtype=bool)
    return SourceTargets(
        keypoints=kpts.astype(np.float64), boxes=boxes.astype(np.float64), area=area, iscrowd=iscrowd
    )


def build_targets(
    targets: SourceTargets,
    *,
    forward: AffineMatrix,
    size: int,
    min_labeled_keypoints: int = 1,
) -> dict[str, torch.Tensor]:
    """Map source-pixel targets into the ``S x S`` view (CONTRACTS §5).

    ``forward`` maps source px -> network px (scale + translation). Labeled points
    outside ``[0, S)`` get ``v = 0``. Non-crowd persons are kept when at least
    ``min_labeled_keypoints`` labeled points remain; crowd regions are kept when
    their box overlaps the view. Coordinates of points with ``v <= 0`` are zeroed.
    """

    m = len(targets)
    if m == 0:
        return _empty_targets()
    kp = targets.keypoints.copy()
    flat = geometry.apply_to_points(forward, kp[:, :, :2].reshape(-1, 2)).reshape(m, NUM_KEYPOINTS, 2)
    kp[:, :, :2] = flat
    lab = kp[:, :, 2] > 0
    inside = (kp[:, :, 0] >= 0) & (kp[:, :, 0] < size) & (kp[:, :, 1] >= 0) & (kp[:, :, 1] < size)
    kp[:, :, 2] = np.where(lab & ~inside, 0.0, kp[:, :, 2])
    kp[:, :, :2] = np.where((kp[:, :, 2] > 0)[..., None], kp[:, :, :2], 0.0)

    boxes = targets.boxes.copy()
    finite = np.isfinite(boxes).all(axis=1)
    if finite.any():
        boxes[finite] = geometry.apply_to_boxes_xyxy(forward, boxes[finite])
        boxes[finite] = np.clip(boxes[finite], 0.0, float(size))
    sx, sy = abs(float(forward[0, 0])), abs(float(forward[1, 1]))
    area = targets.area * (sx * sy)

    n_lab = (kp[:, :, 2] > 0).sum(axis=1)
    crowd = targets.iscrowd.astype(bool)
    box_overlap = finite & (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
    keep = np.where(crowd, box_overlap, n_lab >= max(1, int(min_labeled_keypoints)))
    if not keep.any():
        return _empty_targets()
    return {
        "keypoints": torch.from_numpy(kp[keep].astype(np.float32)),
        "area": torch.from_numpy(area[keep].astype(np.float32)),
        "boxes": torch.from_numpy(boxes[keep].astype(np.float32)),
        "iscrowd": torch.from_numpy(crowd[keep].copy()),
    }


def _empty_targets() -> dict[str, torch.Tensor]:
    return {
        "keypoints": torch.zeros((0, NUM_KEYPOINTS, 3), dtype=torch.float32),
        "area": torch.zeros((0,), dtype=torch.float32),
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "iscrowd": torch.zeros((0,), dtype=torch.bool),
    }


def view_to_tensor(view_bgr: np.ndarray) -> torch.Tensor:
    """uint8 ``(S,S,3)`` BGR -> float32 ``(3,S,S)`` in ``[0, 1]``."""

    return torch.from_numpy(np.ascontiguousarray(view_bgr.transpose(2, 0, 1))).float().div_(255.0)


# --------------------------------------------------------------------------- dataset


class XTX4PoseDataset(Dataset):
    """Box-free pose dataset keyed by ``(sample_idx, S)``.

    Args:
        samples: scanned :class:`ImageSample` list (see :func:`annotations.scan_split`).
        sizes: allowed network sizes; the first one is the default for an int key.
        augment_cfg: :class:`AugmentConfig` (``None`` -> defaults).
        augment: master switch; False gives the canonical (deployment) letterbox.
        min_labeled_keypoints: inclusion threshold for targets inside the view.
        mosaic: initial mosaic state (only effective when ``augment``); see :meth:`set_mosaic`.
        seed: base seed for the per-process augmentation RNG.
    """

    def __init__(
        self,
        samples: Sequence[ImageSample],
        sizes: Sequence[int] = (320,),
        augment_cfg: AugmentConfig | None = None,
        augment: bool = False,
        min_labeled_keypoints: int = 1,
        mosaic: bool = True,
        *,
        seed: int = 0,
    ) -> None:
        self.samples: list[ImageSample] = list(samples)
        if not self.samples:
            raise ValueError("XTX4PoseDataset needs at least one sample")
        self.sizes = tuple(int(s) for s in sizes)
        if not self.sizes or any(s <= 0 for s in self.sizes):
            raise ValueError(f"sizes must be positive, got {self.sizes}")
        self.config = augment_cfg or AugmentConfig()
        self.augment = bool(augment)
        self.min_labeled_keypoints = int(min_labeled_keypoints)
        self.mosaic_enabled = bool(mosaic)
        self.augmentor = Augmentor(config=self.config)
        self.seed = int(seed)
        self._rng: np.random.Generator | None = None
        self._rng_key: tuple[int, int] | None = None

    def __len__(self) -> int:
        return len(self.samples)

    # -- control -----------------------------------------------------------------

    def set_mosaic(self, enabled: bool) -> None:
        """Enable/disable mosaic (close-mosaic phase). Takes effect in new workers;
        with ``persistent_workers=True`` recreate the DataLoader after calling it."""

        self.mosaic_enabled = bool(enabled)

    def reseed(self, seed: int) -> None:
        """Deterministically reset the augmentation RNG of this process (tests)."""

        self._rng = np.random.default_rng(seed)
        self._rng_key = (os.getpid(), torch.initial_seed())

    def _get_rng(self) -> np.random.Generator:
        key = (os.getpid(), torch.initial_seed())
        if self._rng is None or self._rng_key != key:
            self._rng = np.random.default_rng([self.seed, key[1] % (2**63)])
            self._rng_key = key
        return self._rng

    # -- loading -----------------------------------------------------------------

    def load(self, idx: int) -> tuple[np.ndarray, SourceTargets]:
        """Decoded BGR image and its targets in source pixels."""

        sample = self.samples[idx]
        image = read_image(sample.image_path)
        if image is None:
            logger.warning("Failed to read image, using blank: %s", sample.image_path)
            image = np.zeros((max(1, sample.height or 32), max(1, sample.width or 32), 3), np.uint8)
        height, width = image.shape[:2]
        return image, source_targets(sample, width=width, height=height)

    # -- main --------------------------------------------------------------------

    def __getitem__(self, key: tuple[int, int] | int) -> tuple[torch.Tensor, dict[str, torch.Tensor], SampleMeta]:
        if isinstance(key, (tuple, list)):
            idx, size = int(key[0]), int(key[1])
        else:
            idx, size = int(key), self.sizes[0]
        image, targets = self.load(idx)
        orig_h, orig_w = image.shape[:2]
        to_orig = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        use_mosaic = False

        if self.augment:
            rng = self._get_rng()
            cfg = self.config
            use_mosaic = self.mosaic_enabled and cfg.mosaic > 0 and rng.random() < cfg.mosaic
            if use_mosaic:
                tiles_img, tiles_tgt = [], []
                for j, src_idx in enumerate([idx] + [int(i) for i in rng.integers(0, len(self.samples), 3)]):
                    img_j, tgt_j = (image, targets) if j == 0 else self.load(src_idx)
                    img_j, tgt_j, _ = self.augmentor.geometric(img_j, tgt_j, rng)
                    tiles_img.append(img_j)
                    tiles_tgt.append(tgt_j)
                # Camera ops run per tile after the quadrant downscale (fewer pixels, black gaps stay black).
                image, targets, _ = mosaic4(
                    tiles_img, tiles_tgt, output_size=2 * size,
                    tile_fn=lambda tile: self.augmentor.camera(tile, rng),
                )
            else:
                image, targets, to_orig = self.augmentor.source(image, targets, rng)
            height, width = image.shape[:2]
            placement = self.augmentor.placement(width=width, height=height, canvas_size=size, rng=rng)
        else:
            height, width = image.shape[:2]
            placement = compute_placement(width=width, height=height, canvas_size=size)

        view = render_view(image, placement, view_key=_VIEW, mode=_MODE, network_size=size)
        if self.augment:
            view = self.augmentor.view(view, self._get_rng())
        net_to_src = transforms_from_placement(placement, mode=_MODE, network_size=size)[_VIEW]
        forward = geometry.invert_affine(net_to_src)
        target_dict = build_targets(
            targets, forward=forward, size=size, min_labeled_keypoints=self.min_labeled_keypoints
        )
        transform = net_to_src if use_mosaic else geometry.compose(to_orig, net_to_src)
        meta = SampleMeta(
            image_path=str(self.samples[idx].image_path),
            size=size,
            transform=transform,
            orig_size=(orig_w, orig_h),
            sample_index=idx,
            mosaic=use_mosaic,
        )
        return view_to_tensor(view), target_dict, meta


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor], SampleMeta]],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], list[SampleMeta]]:
    """Stack images ``(B,3,S,S)``; targets and metas stay lists (variable person counts)."""

    sizes = {item[0].shape[-1] for item in batch}
    if len(sizes) != 1:
        raise ValueError(f"A batch must have a single network size, got {sorted(sizes)}")
    images = torch.stack([item[0] for item in batch], dim=0)
    return images, [item[1] for item in batch], [item[2] for item in batch]


def worker_init_fn(worker_id: int) -> None:  # noqa: ARG001 - DataLoader signature
    """DataLoader worker init: single-threaded OpenCV (workers already parallelise)."""

    cv2.setNumThreads(0)


# --------------------------------------------------------------------------- evaluation


def build_eval_ground_truth(
    samples: Sequence[ImageSample],
    max_images: int | None = None,
    *,
    load_images: bool = True,
    area_fallback: str = "bbox",
) -> tuple[list, list[dict[str, np.ndarray]]]:
    """Ground truth in **source pixels** for evaluation (consumed by ``metrics.ImageGroundTruth``).

    Returns ``(images, gts)``: ``images`` are BGR arrays (or the image paths when
    ``load_images=False``); each gt is ``dict(keypoints (G,K,3) [x,y,v] float32,
    areas (G,) float32, iscrowd (G,) bool)``. Unreadable images are skipped.
    Area: COCO segmentation area when known; else ``area_fallback``:
    ``"bbox"`` (box w·h, the XTX3 convention) and, without a box, the
    labeled-point extent ``(1.5·w)·(1.5·h)`` (floor 16 px²).
    """

    if area_fallback not in ("bbox", "extent"):
        raise ValueError(f"area_fallback must be 'bbox' or 'extent', got {area_fallback!r}")
    chosen = list(samples) if max_images is None else list(samples)[: int(max_images)]
    images: list = []
    gts: list[dict[str, np.ndarray]] = []
    for sample in chosen:
        if load_images:
            image = read_image(sample.image_path)
            if image is None:
                logger.warning("Eval GT: unreadable image skipped: %s", sample.image_path)
                continue
            height, width = image.shape[:2]
            images.append(image)
        else:
            width, height = sample.width, sample.height
            if width <= 0 or height <= 0:
                from .annotations import read_image_size

                width, height = read_image_size(sample.image_path)
            images.append(str(sample.image_path))
        boxes, kpts = denormalise_sample(sample, width=width, height=height)
        areas = np.zeros((len(sample.annotations),), np.float32)
        for i, ann in enumerate(sample.annotations):
            if ann.area_norm is not None:
                areas[i] = ann.area_norm * width * height
            elif area_fallback == "bbox" and np.isfinite(boxes[i]).all():
                areas[i] = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            else:
                lab = kpts[i, :, 2] > 0
                if lab.any():
                    ext = np.ptp(kpts[i, lab, :2], axis=0)
                    areas[i] = max(16.0, float(1.5 * ext[0] * 1.5 * ext[1]))
                else:
                    areas[i] = 16.0
        gts.append({
            "keypoints": kpts,
            "areas": areas,
            "iscrowd": np.array([bool(a.iscrowd) for a in sample.annotations], dtype=bool),
        })
    return images, gts
