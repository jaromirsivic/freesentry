"""PyTorch dataset producing explicit A/B/C training samples (spec section 8.3-8.4).

Each source image yields three samples (A, B, C); a :class:`BalancedSampler`
controls the A:B:C ratio. The target transform maps normalised-original
annotations into 384 network space via ``invert(T_k)`` and applies the inclusion /
visibility rules (people outside a crop's region produce no target there).
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
from .preprocess import CROP_KEYS, NETWORK_SIZE, coerce_bgr, preprocess_image

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
    network_size: int = NETWORK_SIZE,
    min_visible_keypoints: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Map original-pixel targets into a crop's 384 network space (section 8.3).

    Returns ``(boxes (M',4), keypoints (M',K,3))`` keeping only people whose bbox
    centre falls inside the crop and that retain at least ``min_visible_keypoints``
    visible keypoints inside the frame. Keypoints outside the frame become absent.
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
        kpt[~inside, 2] = 0.0  # mark out-of-frame keypoints absent
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


def crop_to_tensor(crop_bgr: np.ndarray) -> torch.Tensor:
    """Convert a uint8 ``(384,384,3)`` BGR crop to a float ``(3,384,384)`` tensor /255."""

    tensor = torch.from_numpy(np.ascontiguousarray(crop_bgr)).permute(2, 0, 1).float() / 255.0
    return tensor


@dataclass(slots=True)
class SampleMeta:
    image_path: str
    crop: str
    transform: np.ndarray
    orig_size: tuple[int, int]


class XTXPoseDataset(Dataset):
    """Dataset of A/B/C crops with network-space targets."""

    def __init__(
        self,
        *,
        split_dir: Path,
        augment_config: AugmentConfig | None = None,
        augment: bool = True,
        min_visible_keypoints: int = 1,
        mosaic_enabled: bool = True,
        num_keypoints: int = NUM_KEYPOINTS,
    ) -> None:
        self.samples: list[ImageSample] = scan_split(Path(split_dir), num_keypoints=num_keypoints)
        if not self.samples:
            raise ValueError(f"No usable samples found under {split_dir}")
        self.min_visible_keypoints = min_visible_keypoints
        self.augment = augment
        self.mosaic_enabled = augment and mosaic_enabled
        self.augmentor = Augmentor(
            config=augment_config or AugmentConfig(), enabled=augment
        )
        # Flattened (sample_idx, crop) index, three entries per source image.
        self.index: list[tuple[int, str]] = [
            (i, crop) for i in range(len(self.samples)) for crop in CROP_KEYS
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
            image = np.zeros((sample.height or 1080, sample.width or 1080, 3), dtype=np.uint8)
        image = coerce_bgr(image)
        height, width = image.shape[:2]
        boxes, kpts = denormalise_sample(sample, width=width, height=height)
        return image, boxes, kpts

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], str, SampleMeta]:
        sample_idx, crop = self.index[i]
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

        result = preprocess_image(image)
        transform = result.transforms[crop]  # network (384) -> augmented-image px (T_k)
        net_boxes, net_kpts = targets_to_network(
            boxes,
            kpts,
            transform=transform,
            min_visible_keypoints=self.min_visible_keypoints,
        )

        tensor = crop_to_tensor(result.crops[crop])
        targets = {
            "boxes": torch.from_numpy(net_boxes),
            "keypoints": torch.from_numpy(net_kpts),
        }
        meta = SampleMeta(
            image_path=str(self.samples[sample_idx].image_path),
            crop=crop,
            transform=transform.astype(np.float32),
            orig_size=result.orig_size,
        )
        return tensor, targets, crop, meta


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor], str, SampleMeta]],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], list[str], list[SampleMeta]]:
    """Batch variable-target samples: stack images, keep targets as a list."""

    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    crops = [item[2] for item in batch]
    metas = [item[3] for item in batch]
    return images, targets, crops, metas


class BalancedSampler(Sampler[int]):
    """Samples flattened indices to honour the configured A:B:C ratio."""

    def __init__(
        self,
        *,
        index: list[tuple[int, str]],
        abc_ratio: tuple[float, float, float] = (1.0, 1.0, 1.0),
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.index = index
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        ratio_map = {"A": abc_ratio[0], "B": abc_ratio[1], "C": abc_ratio[2]}
        total = sum(ratio_map.values()) or 1.0
        self.weights = torch.tensor(
            [ratio_map[crop] / total for (_, crop) in index], dtype=torch.double
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
