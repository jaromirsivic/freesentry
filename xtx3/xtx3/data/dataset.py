"""PyTorch dataset producing whole-frame or pyramid training samples.

In **whole-frame mode** each source image yields one sample (view 0). In
**pyramid mode** each source image yields three samples (views 1, 2, 3) and a
:class:`BalancedSampler` controls the view ratio. The target transform maps
normalised-original annotations into network space via ``invert(T_k)`` -- the
exact same geometry as inference -- and applies the inclusion / visibility
rules (people outside a view's field of view produce no target there; keypoints
outside the frame get visibility 0 and are masked from the localisation loss).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..utils import geometry
from ..utils.geometry import AffineMatrix
from ..utils.keypoints import NUM_KEYPOINTS
from ..utils.logging import get_logger
from .annotations import ImageSample, scan_split
from .augment import AugmentConfig, Augmentor, mosaic4
from .preprocess import (
    DEFAULT_NETWORK_SIZE,
    canvas_size_for_mode,
    coerce_bgr,
    compute_transforms,
    render_view,
    transforms_from_placement,
    view_keys_for_mode,
)

logger = get_logger(__name__)


def denormalise_sample(
    sample: ImageSample, *, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(boxes_xyxy_px (M,4), keypoints_px (M,K,3))`` for a sample."""

    num = len(sample.annotations)
    boxes = np.zeros((num, 4), dtype=np.float32)
    kpts = np.zeros((num, NUM_KEYPOINTS, 3), dtype=np.float32)
    for i, ann in enumerate(sample.annotations):
        bx, by, bw, bh = ann.bbox_xywh_norm
        boxes[i] = [bx * width, by * height, (bx + bw) * width, (by + bh) * height]
        kpts[i, :, 0] = ann.keypoints_norm[:, 0] * width
        kpts[i, :, 1] = ann.keypoints_norm[:, 1] * height
        kpts[i, :, 2] = ann.keypoints_norm[:, 2]
    return boxes, kpts


def targets_to_network(
    boxes_px: np.ndarray,
    kpts_px: np.ndarray,
    *,
    transform: AffineMatrix,
    network_size: int = DEFAULT_NETWORK_SIZE,
    min_visible_keypoints: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Map original-pixel targets into a view's network space.

    Returns ``(boxes (M',4), keypoints (M',K,3))`` keeping only people whose bbox
    centre falls inside the view's frame and that retain at least
    ``min_visible_keypoints`` visible keypoints inside it. Keypoints outside the
    frame get ``visibility = 0`` (masked from the localisation loss).
    """

    if boxes_px.shape[0] == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, NUM_KEYPOINTS, 3), dtype=np.float32),
        )

    inv = geometry.invert_affine(transform)  # original px -> network px
    net_boxes = geometry.apply_to_boxes_xyxy(inv, boxes_px).astype(np.float32)
    net_kpts = geometry.apply_to_keypoints(inv, kpts_px).astype(np.float32)

    kept_boxes: list[np.ndarray] = []
    kept_kpts: list[np.ndarray] = []
    for i in range(net_boxes.shape[0]):
        box = net_boxes[i].copy()
        kpt = net_kpts[i].copy()

        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        if not (0 <= cx <= network_size and 0 <= cy <= network_size):
            continue

        inside = (
            (kpt[:, 0] >= 0)
            & (kpt[:, 0] < network_size)
            & (kpt[:, 1] >= 0)
            & (kpt[:, 1] < network_size)
        )
        kpt[~inside, 2] = 0.0  # out-of-frame keypoints become absent
        if int((kpt[:, 2] > 0).sum()) < min_visible_keypoints:
            continue

        box[0::2] = np.clip(box[0::2], 0, network_size)
        box[1::2] = np.clip(box[1::2], 0, network_size)
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            continue
        kept_boxes.append(box)
        kept_kpts.append(kpt)

    if not kept_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, NUM_KEYPOINTS, 3), dtype=np.float32),
        )
    return np.stack(kept_boxes, axis=0), np.stack(kept_kpts, axis=0)


def view_to_tensor(view_bgr: np.ndarray) -> torch.Tensor:
    """Convert a uint8 ``(S,S,3)`` BGR view to a float ``(3,S,S)`` tensor /255."""

    return torch.from_numpy(np.ascontiguousarray(view_bgr)).permute(2, 0, 1).float() / 255.0


@dataclass(slots=True)
class SampleMeta:
    image_path: str
    view: int
    transform: np.ndarray  # T_k, network px -> source-image px
    orig_size: tuple[int, int]


class XTX3PoseDataset(Dataset):
    """Dataset of whole-frame or pyramid view samples with network-space targets."""

    def __init__(
        self,
        *,
        split_dir: Path,
        view_mode: str = "pyramid",
        network_size: int = DEFAULT_NETWORK_SIZE,
        augment_config: AugmentConfig | None = None,
        augment: bool = True,
        min_visible_keypoints: int = 1,
        mosaic_enabled: bool = True,
    ) -> None:
        self.samples: list[ImageSample] = scan_split(Path(split_dir))
        if not self.samples:
            raise ValueError(f"No usable samples found under {split_dir}")
        self.view_mode = view_mode
        self.network_size = network_size
        self.view_keys = view_keys_for_mode(view_mode)
        self.min_visible_keypoints = min_visible_keypoints
        self.augment = augment
        self.mosaic_enabled = augment and mosaic_enabled
        self.augmentor = Augmentor(config=augment_config or AugmentConfig(), enabled=augment)
        # Flattened (sample_idx, view) index: one entry per view per source image.
        self.index: list[tuple[int, int]] = [
            (i, view) for i in range(len(self.samples)) for view in self.view_keys
        ]

    def __len__(self) -> int:
        return len(self.index)

    def set_mosaic(self, *, enabled: bool) -> None:
        self.mosaic_enabled = self.augment and enabled

    def _load(self, sample_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample = self.samples[sample_idx]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Failed to read image, using blank: %s", sample.image_path)
            image = np.zeros(
                (sample.height or self.network_size, sample.width or self.network_size, 3),
                np.uint8,
            )
        image = coerce_bgr(image)
        height, width = image.shape[:2]
        boxes, kpts = denormalise_sample(sample, width=width, height=height)
        return image, boxes, kpts

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], int, SampleMeta]:
        sample_idx, view = self.index[i]
        image, boxes, kpts = self._load(sample_idx)

        use_mosaic = self.mosaic_enabled and random.random() < self.augmentor.config.mosaic
        if use_mosaic:
            others = [random.randrange(len(self.samples)) for _ in range(3)]
            imgs, bxs, kps = [image], [boxes], [kpts]
            for idx in others:
                im2, b2, k2 = self._load(idx)
                imgs.append(im2)
                bxs.append(b2)
                kps.append(k2)
            image, boxes, kpts = mosaic4(imgs, bxs, kps, output_size=max(image.shape[:2]))

        if self.augment:
            image, boxes, kpts = self.augmentor(image, boxes, kpts)

        height, width = image.shape[:2]
        canvas_size = canvas_size_for_mode(self.view_mode, self.network_size)
        placement = self.augmentor.placement(width=width, height=height, canvas_size=canvas_size)
        transforms = transforms_from_placement(
            placement, mode=self.view_mode, network_size=self.network_size
        )
        transform = transforms[view]  # network px -> (augmented) source px

        # Only the requested view's pixels are rendered (one-shot rendering).
        view_bgr = render_view(
            image, placement, view_key=view, mode=self.view_mode, network_size=self.network_size
        )

        net_boxes, net_kpts = targets_to_network(
            boxes,
            kpts,
            transform=transform,
            network_size=self.network_size,
            min_visible_keypoints=self.min_visible_keypoints,
        )

        tensor = view_to_tensor(view_bgr)
        targets = {
            "boxes": torch.from_numpy(net_boxes),
            "keypoints": torch.from_numpy(net_kpts),
        }
        meta = SampleMeta(
            image_path=str(self.samples[sample_idx].image_path),
            view=view,
            transform=transform.astype(np.float32),
            orig_size=(width, height),
        )
        return tensor, targets, view, meta

    def per_view_target_counts(self) -> dict[int, int]:
        """Count valid targets per view over the whole split (dataset stats)."""

        counts = {view: 0 for view in self.view_keys}
        for sample in self.samples:
            width = sample.width or 1
            height = sample.height or 1
            boxes, kpts = denormalise_sample(sample, width=width, height=height)
            transforms = compute_transforms(
                width=width, height=height, mode=self.view_mode, network_size=self.network_size
            )
            for view in self.view_keys:
                net_boxes, _ = targets_to_network(
                    boxes,
                    kpts,
                    transform=transforms[view],
                    network_size=self.network_size,
                    min_visible_keypoints=self.min_visible_keypoints,
                )
                counts[view] += net_boxes.shape[0]
        return counts


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor], int, SampleMeta]],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], list[int], list[SampleMeta]]:
    """Batch variable-target samples: stack images, keep targets as a list."""

    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    views = [item[2] for item in batch]
    metas = [item[3] for item in batch]
    return images, targets, views, metas


class BalancedSampler(Sampler[int]):
    """Samples flattened indices to honour the configured pyramid view ratio."""

    def __init__(
        self,
        *,
        index: list[tuple[int, int]],
        view_ratio: dict[int, float] | None = None,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.index = index
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        views = sorted({view for (_, view) in index})
        ratio_map = {view: 1.0 for view in views}
        if view_ratio:
            ratio_map.update({int(k): float(v) for k, v in view_ratio.items() if int(k) in ratio_map})
        total = sum(ratio_map.values()) or 1.0
        self.weights = torch.tensor(
            [ratio_map[view] / total for (_, view) in index], dtype=torch.double
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        num_samples = len(self.index)
        if not self.shuffle:
            yield from range(num_samples)
            return
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        drawn = torch.multinomial(self.weights, num_samples, replacement=True, generator=generator)
        yield from drawn.tolist()

    def __len__(self) -> int:
        return len(self.index)
