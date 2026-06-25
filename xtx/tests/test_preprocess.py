"""A/B/C geometry + round-trip transform correctness (spec acceptance 2)."""

from __future__ import annotations

import numpy as np
import pytest

from xtx.data.preprocess import NETWORK_SIZE, coerce_bgr, compute_transforms, preprocess_image
from xtx.utils import geometry

IMAGE_SIZES = [(1080, 1080), (1920, 1080), (320, 240), (2000, 4000), (900, 3000), (640, 480)]


@pytest.mark.parametrize("width,height", IMAGE_SIZES)
def test_crop_shapes(width: int, height: int) -> None:
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    result = preprocess_image(image)
    for key in ("A", "B", "C"):
        assert result.crops[key].shape == (NETWORK_SIZE, NETWORK_SIZE, 3)
    assert result.orig_size == (width, height)


@pytest.mark.parametrize("width,height", IMAGE_SIZES)
def test_roundtrip_network_to_original(width: int, height: int) -> None:
    """Network-384 -> original -> network round-trip must be within <=1px."""

    transforms = compute_transforms(width=width, height=height)
    rng = np.random.default_rng(0)
    net_points = rng.uniform(0, NETWORK_SIZE, size=(50, 2))
    for transform in transforms.values():
        inv = geometry.invert_affine(transform)
        original = geometry.apply_to_points(transform, net_points)
        back = geometry.apply_to_points(inv, original)
        assert np.max(np.abs(back - net_points)) <= 1.0


def test_keypoint_roundtrip_within_one_pixel() -> None:
    transforms = compute_transforms(width=1280, height=720)
    rng = np.random.default_rng(1)
    kpts = np.zeros((4, 17, 3), dtype=np.float64)
    kpts[:, :, :2] = rng.uniform(0, NETWORK_SIZE, size=(4, 17, 2))
    kpts[:, :, 2] = 2
    for transform in transforms.values():
        inv = geometry.invert_affine(transform)
        mapped = geometry.apply_to_keypoints(transform, kpts)
        back = geometry.apply_to_keypoints(inv, mapped)
        assert np.max(np.abs(back[:, :, :2] - kpts[:, :, :2])) <= 1.0
        assert np.array_equal(back[:, :, 2], kpts[:, :, 2])  # confidence untouched


def test_center_maps_to_image_center() -> None:
    result = preprocess_image(np.random.randint(0, 255, (1080, 1080, 3), dtype=np.uint8))
    center_net = np.array([[NETWORK_SIZE / 2, NETWORK_SIZE / 2]])
    for transform in result.transforms.values():
        mapped = geometry.apply_to_points(transform, center_net)[0]
        assert np.allclose(mapped, [540, 540], atol=1.0)


def test_coerce_bgr_channels() -> None:
    gray = np.random.randint(0, 255, (100, 120), dtype=np.uint8)
    rgba = np.random.randint(0, 255, (100, 120, 4), dtype=np.uint8)
    assert coerce_bgr(gray).shape == (100, 120, 3)
    assert coerce_bgr(rgba).shape == (100, 120, 3)
    assert coerce_bgr(gray).dtype == np.uint8
