"""Preprocessing tests (spec section 15, acceptance criteria 2 and 3).

Covers: 1536 letterbox canvas geometry, the 4x/2x/1x pyramid, level_3 being
interpolation-free in the small-image branch, keypoint round-trip mapping
within <= 0.5 px for both a 1920x1080 and a 320x240 synthetic case, and the
one-shot rendering agreeing with the recorded affines and (in the aligned
small-image branch) with the full-canvas reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from xtx2.data.preprocess import (
    CANVAS_SIZE,
    LEVEL_KEYS,
    NETWORK_SIZE,
    build_canvas,
    build_levels,
    coerce_bgr,
    compute_placement,
    compute_transforms,
    preprocess_image,
)
from xtx2.utils import geometry


def _synthetic_image(width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(width * 1000 + height)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


class TestPlacement:
    def test_large_image_scale_and_offset(self) -> None:
        # 1920x1080 -> s = 0.8 -> 1536x864 pasted at (0, 336) (spec 4.1 example).
        placement = compute_placement(width=1920, height=1080)
        assert placement.resized_w == 1536
        assert placement.resized_h == 864
        assert placement.paste_x == 0
        assert placement.paste_y == 336
        assert placement.scale_x == pytest.approx(0.8)
        assert placement.scale_y == pytest.approx(0.8)

    def test_small_image_not_upscaled(self) -> None:
        placement = compute_placement(width=320, height=240)
        assert placement.resized_w == 320
        assert placement.resized_h == 240
        assert placement.scale_x == 1.0
        assert placement.scale_y == 1.0
        # Centred to within the 4 px alignment.
        assert abs(placement.paste_x - (CANVAS_SIZE - 320) // 2) < 4
        assert abs(placement.paste_y - (CANVAS_SIZE - 240) // 2) < 4

    def test_offsets_are_aligned(self) -> None:
        for w, h in [(1920, 1080), (320, 240), (4000, 2000), (777, 555)]:
            placement = compute_placement(width=w, height=h)
            assert placement.paste_x % 4 == 0
            assert placement.paste_y % 4 == 0


class TestTransformsWorkedExample:
    """Spec section 4.4 worked example for a 1920x1080 input."""

    def test_level_transforms(self) -> None:
        transforms = compute_transforms(width=1920, height=1080)
        # T_1: x_orig = 4*x_net / 0.8, y_orig = (4*y_net - 336) / 0.8
        pt = geometry.apply_to_points(transforms[1], np.array([[100.0, 100.0]]))[0]
        assert pt[0] == pytest.approx(4 * 100 / 0.8, abs=1e-6)
        assert pt[1] == pytest.approx((4 * 100 - 336) / 0.8, abs=1e-6)
        # T_2: x_orig = (2*x_net + 384) / 0.8, y_orig = (2*y_net + 384 - 336) / 0.8
        pt = geometry.apply_to_points(transforms[2], np.array([[100.0, 100.0]]))[0]
        assert pt[0] == pytest.approx((2 * 100 + 384) / 0.8, abs=1e-6)
        assert pt[1] == pytest.approx((2 * 100 + 384 - 336) / 0.8, abs=1e-6)
        # T_3: x_orig = (x_net + 576) / 0.8, y_orig = (y_net + 576 - 336) / 0.8
        pt = geometry.apply_to_points(transforms[3], np.array([[100.0, 100.0]]))[0]
        assert pt[0] == pytest.approx((100 + 576) / 0.8, abs=1e-6)
        assert pt[1] == pytest.approx((100 + 576 - 336) / 0.8, abs=1e-6)


@pytest.mark.parametrize("size", [(1920, 1080), (320, 240)])
class TestRoundTrip:
    """original -> level-k -> original round trip within <= 0.5 px (criterion 2)."""

    def test_round_trip(self, size: tuple[int, int]) -> None:
        width, height = size
        transforms = compute_transforms(width=width, height=height)
        rng = np.random.default_rng(0)
        points = rng.uniform([0, 0], [width, height], size=(64, 2))
        for level in LEVEL_KEYS:
            forward = geometry.invert_affine(transforms[level])  # original -> level
            net = geometry.apply_to_points(forward, points)
            back = geometry.apply_to_points(transforms[level], net)
            assert np.abs(back - points).max() <= 0.5

    def test_keypoint_round_trip(self, size: tuple[int, int]) -> None:
        width, height = size
        transforms = compute_transforms(width=width, height=height)
        rng = np.random.default_rng(1)
        kpts = np.zeros((3, 17, 3))
        kpts[:, :, 0] = rng.uniform(0, width, size=(3, 17))
        kpts[:, :, 1] = rng.uniform(0, height, size=(3, 17))
        kpts[:, :, 2] = 2
        for level in LEVEL_KEYS:
            inv = geometry.invert_affine(transforms[level])
            net = geometry.apply_to_keypoints(inv, kpts)
            back = geometry.apply_to_keypoints(transforms[level], net)
            assert np.abs(back[:, :, :2] - kpts[:, :, :2]).max() <= 0.5
            assert np.array_equal(back[:, :, 2], kpts[:, :, 2])


class TestLevels:
    def test_shapes_and_dtype(self) -> None:
        result = preprocess_image(_synthetic_image(1920, 1080))
        assert set(result.levels.keys()) == {1, 2, 3}
        for level in LEVEL_KEYS:
            assert result.levels[level].shape == (NETWORK_SIZE, NETWORK_SIZE, 3)
            assert result.levels[level].dtype == np.uint8

    def test_level3_pure_copy_small_branch(self) -> None:
        """s = 1: level_3 must be a pure pixel copy (zero interpolation)."""

        image = _synthetic_image(1200, 900)  # small branch: s = 1, covers level_3
        placement = compute_placement(width=1200, height=900)
        levels = build_levels(image, placement)
        canvas = build_canvas(image, placement)
        expected = canvas[576:960, 576:960]
        assert np.array_equal(levels[3], expected)

    def test_small_branch_matches_canvas_reference(self) -> None:
        """With aligned offsets and s = 1, one-shot levels match the full-canvas
        reference bit-for-bit (level 3) and near-exactly (levels 1, 2)."""

        import cv2

        image = _synthetic_image(320, 240)
        placement = compute_placement(width=320, height=240)
        levels = build_levels(image, placement)
        canvas = build_canvas(image, placement)

        ref1 = cv2.resize(canvas, (384, 384), interpolation=cv2.INTER_AREA)
        ref2 = cv2.resize(canvas[384:1152, 384:1152], (384, 384), interpolation=cv2.INTER_AREA)
        ref3 = canvas[576:960, 576:960]
        assert np.array_equal(levels[3], ref3)
        # Integer 4x / 2x box filters over aligned offsets: tiny rounding only.
        assert np.abs(levels[1].astype(int) - ref1.astype(int)).max() <= 1
        assert np.abs(levels[2].astype(int) - ref2.astype(int)).max() <= 1

    def test_rendering_agrees_with_affine(self) -> None:
        """A bright marker placed in the original must land where T_k^-1 says."""

        for width, height in [(1920, 1080), (320, 240)]:
            image = np.zeros((height, width, 3), dtype=np.uint8)
            mx, my = width // 2 + 15, height // 2 - 10
            image[max(0, my - 4) : my + 4, max(0, mx - 4) : mx + 4] = 255

            result = preprocess_image(image)
            for level in LEVEL_KEYS:
                inv = geometry.invert_affine(result.transforms[level])
                expected = geometry.apply_to_points(inv, np.array([[mx, my]], float))[0]
                if not (8 <= expected[0] < NETWORK_SIZE - 8 and 8 <= expected[1] < NETWORK_SIZE - 8):
                    continue  # marker outside this level's field of view
                gray = result.levels[level].sum(axis=2).astype(np.float64)
                assert gray.max() > 0, f"marker missing in level {level} ({width}x{height})"
                ys, xs = np.nonzero(gray > gray.max() * 0.5)
                cx = float((xs * gray[ys, xs]).sum() / gray[ys, xs].sum())
                cy = float((ys * gray[ys, xs]).sum() / gray[ys, xs].sum())
                err = np.hypot(cx - expected[0], cy - expected[1])
                assert err <= 1.5, f"level {level}: marker error {err:.2f}px"

    def test_extreme_aspect_ratio_nothing_cropped(self) -> None:
        """4000x2000: the full frame must be visible in level_1 (edge case 16)."""

        image = np.zeros((2000, 4000, 3), dtype=np.uint8)
        image[:, :8] = 255  # left edge
        image[:, -8:] = 255  # right edge
        result = preprocess_image(image)
        transforms = result.transforms
        inv = geometry.invert_affine(transforms[1])
        left = geometry.apply_to_points(inv, np.array([[0.0, 1000.0]]))[0]
        right = geometry.apply_to_points(inv, np.array([[4000.0, 1000.0]]))[0]
        assert 0 <= left[0] <= NETWORK_SIZE
        assert 0 <= right[0] <= NETWORK_SIZE
        # And the pixels are actually present in the rendered level_1.
        level1 = result.levels[1]
        assert level1[:, : NETWORK_SIZE // 2].max() > 0
        assert level1[:, NETWORK_SIZE // 2 :].max() > 0


class TestCoerce:
    def test_grayscale_promoted(self) -> None:
        gray = np.zeros((10, 12), dtype=np.uint8)
        assert coerce_bgr(gray).shape == (10, 12, 3)

    def test_bgra_alpha_dropped(self) -> None:
        bgra = np.zeros((10, 12, 4), dtype=np.uint8)
        assert coerce_bgr(bgra).shape == (10, 12, 3)

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            coerce_bgr(np.zeros((10, 12, 2), dtype=np.uint8))
